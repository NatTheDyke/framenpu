"""Device run of the resident teacher tail through the linked operator image
(tools/build_teacher_link.py) on the qualified corpus crops.

Phase A: per-layer qualification with the ordinary cached-entry kernels, as in
tests/hardware_teacher_resident.py (FP16 bounds, ncnn blob drift, heads).

Seal: the image is written to the owned reservation 0x1ff0000..0x1ff4000 with
a zero seal vector, read back, then the slot loader's seal command discards
the cached lines over the reservation and sets the seal.

Phase B: per crop, alternating cached-entry frames and linked frames (same
7-job resident plan, same kernels' source). Every linked frame's final tensor,
and in its first round every stored tensor, must be bit-identical to phase A.
Before and after every linked request the image and seal are compared with
what was sealed.

Controls (first crop): a guarded fault inside a linked resident job, a native
descriptor rejection, a second seal refused while sealed, then a clean linked
frame. After the frames: retire, and a resident request refused natively by
the loader because the image is no longer sealed.

Protections as tests/hardware_teacher_resident.py (lease, refusal record,
audited direct dispatch, no retry/recovery/restart). The window and, only
after a confirmed retire, the reservation are restored and the whole arena
is compared with its starting hash. Only statistics are written.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools")]
from framenpu import Device, Job, JobFailed
from framenpu import RecoveryRequired
from framenpu.resident_gemm import _lease, RECORD
from framenpu.resident_landmark import CONTROL, ARENA_START, ARENA_END
from hardware_hmx import PowerSnapshot, audited_device
from framenpu.resident_gemm import SSR
from hardware_resident import settled_health
from hardware_landmark import digest, service
from hmx_multi_run import failure_snapshot
import hardware_teacher_resident as HR
import teacher_tail_layout as T
import teacher_resident_layout as RL

CODE, CODE_BYTES, SEAL_AT, SEAL = 0x1FF0000, 0x4000, 0x1FF3F80, 0x544C4E4B
ADMIN_OPCODE, ADMIN_MAGIC = 112, 0x544C4144
IMAGE_CACHED = 0x87000000 + CODE
SLOT_UNCACHED = 0xF0A92000


def load_link(root):
    record = json.loads((root / "link.json").read_text())
    loader = Job.load(root / "teacher_link_loader.fjob")
    image = (root / "teacher_link.bin").read_bytes()
    if (loader.name != record["loader"]["name"] or hashlib.sha256(loader.code).hexdigest() != record["loader"]["sha256"]
            or hashlib.sha256(image).hexdigest() != record["linked"]["sha256"]
            or not 0 < len(image) <= SEAL_AT - CODE or not loader.requires_hmx or not loader.supports_direct):
        raise ValueError("linked build record differs")
    for name, sha in record["sources"].items():
        path = ROOT / name
        if path.exists() and digest(path) != sha:
            raise ValueError(f"linked source pin differs: {name}")
    return loader, image, record


def linked_completion(result, image_len, *, fault=False, reject=False, output_bytes, admin=False):
    """Completion integrity for the slot loader: no publication or cleanup markers."""
    try:
        if len(result.workers) != 3 or len(result.output) != output_bytes:
            raise ValueError("incomplete result")
        for i, row in enumerate(result.workers):
            if (len(row) != 16 or not row[0] or row[0] != result.workers[0][0] or row[1] != i
                    or row[7] != SSR[i] or tuple(row[14:16]) != (0, 0)):
                raise ValueError("worker identity/SSR/markers differ")
            if i and (row[2] or any(row[3:6])):
                raise ValueError("unexpected idle-worker fault")
        row = result.workers[0]
        if row[8] != 0x87000000 or result.kernel_reference_ticks is None or not 0 < result.kernel_reference_ticks < 768000:
            raise ValueError("EVB/timing differs")
        if row[2] == 2:
            if admin or not fault or not IMAGE_CACHED <= row[4] < IMAGE_CACHED + image_len:
                raise ValueError("native fault outside the linked image")
            if row[3] & 255 != 37 or struct.unpack_from("<I", result.output)[0] != 0xDECAFBAD:
                raise ValueError("guarded trap/sentinel differs")
        elif row[2] != (1 if reject else 0) or any(row[3:6]):
            raise ValueError("native status differs")
    except Exception as exc:
        raise RecoveryRequired("linked completion integrity invalid; retain resources") from exc


class LinkRunner(HR.Runner):
    def __init__(self, dev, mem, jobs, graph, specs, loader, image):
        super().__init__(dev, mem, jobs, graph, specs)
        self.loader, self.image = loader, image
        self.reservation = image + bytes(CODE_BYTES - len(image))  # zero seal vector
        self.sealed = False

    def _check_reservation(self, sealed):
        mem = self.mem
        if bytes(mem[CODE : CODE + len(self.image)]) != self.image:
            raise RecoveryRequired("linked image changed")
        seal = struct.unpack_from("<I", mem, SEAL_AT)[0]
        rest = bytes(mem[CODE + len(self.image) : CODE + CODE_BYTES])
        if seal != (SEAL if sealed else 0) or rest[: SEAL_AT - CODE - len(self.image)] != bytes(SEAL_AT - CODE - len(self.image)) or any(rest[SEAL_AT - CODE - len(self.image) + 4 :]):
            raise RecoveryRequired("linked reservation padding/seal differs")

    def admin(self, command, *, reject=False):
        seen = []

        def validate(result):
            linked_completion(result, len(self.image), reject=reject, output_bytes=256, admin=True)
            seen.append(result)

        before = self.sealed
        data = struct.pack("<4I", ADMIN_MAGIC, command, CODE, CODE_BYTES).ljust(128, b"\0")
        try:
            res = self.dev._submit(self.loader, data, opcode=ADMIN_OPCODE, params=(0,), output_size=256,
                                   use_direct=True, _completion_validator=validate,
                                   _prepare=lambda t: self._check_reservation(before))
        except JobFailed:
            if not reject or not seen or seen[0].workers[0][2] != 1:
                raise
            self._check_reservation(before)
            return None
        if reject:
            raise ValueError("expected admin rejection missing")
        m = struct.unpack_from("<64I", res.output)
        after = command == 7
        if (m[0] != ADMIN_MAGIC or not SLOT_UNCACHED <= m[1] < SLOT_UNCACHED + len(self.loader.code)
                or m[2] != (SEAL if after else 0) or any(m[3:])):
            raise RecoveryRequired(f"admin metadata differs: {[hex(w) for w in m[:4]]}")
        self.sealed = after
        self._check_reservation(after)
        return res.kernel_reference_ticks / 19200

    def seal(self):
        self.mem[CODE : CODE + CODE_BYTES] = self.reservation
        if bytes(self.mem[CODE : CODE + CODE_BYTES]) != self.reservation:
            raise RecoveryRequired("linked image readback differs")
        return self.admin(7)

    def submit(self, plan, job, *, fault=False, reject=False, linked=False, refused=False):
        if not linked:
            return super().submit(plan, job, fault=fault, reject=reject)
        for off, data in job.staged.items():
            self.put(off, data)
        if job.kind == "rs":
            data, opcode, params = RL.payload(plan), RL.RS_OPCODE, (1, job.first, job.end, 0, 0, int(fault))
        else:
            data, opcode, params = job.desc, T.DW_OPCODE, (1, 0, 0, 0, 0, int(fault))
        seen = []
        sealed = self.sealed

        def validate(result):
            # refused: an unsealed loader's own frame_job rejects the request
            linked_completion(result, len(self.image), fault=fault, reject=reject or refused,
                              output_bytes=T.META_BYTES)
            self._check_reservation(sealed)
            seen.append(result)

        t0 = time.perf_counter()
        try:
            res = self.dev._submit(self.loader, data, opcode=opcode, params=params, output_size=T.META_BYTES,
                                   use_direct=True, _completion_validator=validate,
                                   _prepare=lambda t: self._check_reservation(sealed))
        except JobFailed:
            if not seen or not (fault or reject or refused) or seen[0].workers[0][2] != (2 if fault else 1):
                raise
            return None
        wall = time.perf_counter() - t0
        if fault or reject or refused:
            raise ValueError("expected fault/rejection missing")
        m = struct.unpack_from("<256I", res.output)
        ref = res.kernel_reference_ticks
        if not IMAGE_CACHED <= m[1] < IMAGE_CACHED + len(self.image) or any(m[64:96]):
            raise ValueError(f"{job.ops}: linked pc/finite flags differ: {[hex(w) for w in m[:8]]}")
        if job.kind == "rs":
            n = job.end - job.first
            ticks = list(m[96 : 96 + n])
            if (m[0] != RL.RS_MAGIC or m[2] != n or any(m[3:64]) or any(m[96 + n :])
                    or any(b < a for a, b in zip(ticks, ticks[1:])) or not 0 < ticks[-1] <= ref):
                raise ValueError(f"{job.ops}: resident metadata/budget differs")
            per = [b - a for a, b in zip([0] + ticks, ticks)]
            return {"kind": "rs", "wall_ms": wall * 1e3, "wrapper_ms": ref / 19200, "body_ms": ticks[-1] / 19200,
                    "op_us": [p / HR.TICKS_PER_US for p in per]}
        gin = T.grid_of(self.sh[job.op.src])
        if m[0] != T.DW_MAGIC or m[2] != gin.nj4 or any(m[7:32]) or not 0 < m[4] + m[5] + m[6] <= ref:
            raise ValueError(f"{job.ops}: stride-2 metadata/budget differs")
        return {"kind": "s2", "wall_ms": wall * 1e3, "wrapper_ms": ref / 19200,
                "body_ms": (m[4] + m[5] + m[6]) / 19200, "op_us": [(m[4] + m[5] + m[6]) / HR.TICKS_PER_US]}

    def frame(self, blob, *, observe, linked=False):
        plan = self.plans["resident"]
        t0 = time.perf_counter()
        self.inputs(plan, blob)
        rows, seen = [], {}
        for job in plan.jobs:
            rows.append(self.submit(plan, job, linked=linked))
            if observe:
                for b in job.stores:
                    seen[b] = self.raw(plan, b)
        final = self.raw(plan, RL.FINAL)
        return rows, seen, final, (time.perf_counter() - t0) * 1e3


def run(runner, corpus, manifest, rounds):
    import numpy as np

    graph, specs, sh = runner.graph, runner.specs, runner.sh
    shape_of = sh | {"in0": (3, 224, 224)}
    crops, ref_raw = {}, {}

    def loader(crop):
        def blob(name):
            raw = (corpus / crop["name"] / f"blob_{name}.f32").read_bytes()
            if hashlib.sha256(raw).hexdigest() != crop["files"][name]["sha256"]:
                raise ValueError(f"corpus blob hash differs: {crop['name']}/{name}")
            return np.frombuffer(raw, "<f4").reshape(shape_of[name]).astype(np.float64)
        return blob

    for crop in manifest["crops"]:  # phase A, cached-entry kernels
        blob = loader(crop)
        t, ratios, raw = runner.qualify(blob)
        ref_raw[crop["name"]] = raw
        crops[crop["name"]] = {"split": crop["split"],
                               "qualify": {"max_bound_ratio": max(ratios.values())} | HR.heads_row(graph, specs, t[RL.FINAL], blob)}
        print(json.dumps({"phase": "qualify", "crop": crop["name"], "max_bound_ratio": max(ratios.values())}), flush=True)

    plan = runner.plans["resident"]
    runner.install(plan)
    seal_ms = runner.seal()
    for i, crop in enumerate(manifest["crops"]):
        blob = loader(crop)
        name = crop["name"]
        frames = {"cached": [], "linked": []}
        compared = set()
        for r in range(rounds):
            for mode in ("cached", "linked"):
                observe = r == 0 and mode == "linked"
                rows, seen, final, wall = runner.frame(blob, observe=observe, linked=mode == "linked")
                if final != ref_raw[name][RL.FINAL]:
                    raise ValueError(f"{name} {mode} round {r}: final tensor differs from qualification")
                for b, data in seen.items():
                    if data != ref_raw[name][b]:
                        raise ValueError(f"{name}: linked stored tensor {b} differs from qualification")
                    compared.add(b)
                frames[mode].append({"jobs": rows, "host_wall_ms": wall})
                print(json.dumps({"phase": mode, "crop": name, "round": r,
                                  "wrapper_ms": sum(j["wrapper_ms"] for j in rows),
                                  "body_ms": sum(j["body_ms"] for j in rows), "host_wall_ms": wall}), flush=True)
        entry = {"frames": frames, "bit_identical_tensors_linked_round0": sorted(compared | {RL.FINAL})}
        if i == 0:
            rs = next(j for j in plan.jobs if j.kind == "rs")
            runner.submit(plan, rs, fault=True, linked=True)
            table = bytearray(plan.table)
            k = next(n for n in range(rs.first, rs.end) if struct.unpack_from("<I", table, 128 * n)[0] == 1)
            struct.pack_into("<I", table, 128 * k + 36, T.WIN_HI)
            runner.put(plan.table_at, bytes(table))
            runner.submit(plan, rs, reject=True, linked=True)
            runner.install(plan)
            runner.admin(7, reject=True)  # already sealed
            rows, _, final, _ = runner.frame(blob, observe=False, linked=True)
            if final != ref_raw[name][RL.FINAL]:
                raise ValueError("clean linked frame after controls differs")
            entry["controls"] = {"guarded_fault_linked": 1, "descriptor_rejection_linked": 1,
                                 "second_seal_refused": 1, "clean_after_controls": True}
        crops[name]["frames"] = entry
    return crops, seal_ms


def summarize(crops, plan):
    import numpy as np

    med = lambda xs: float(np.median(xs))  # noqa: E731
    out = {"crops": len(crops), "jobs_per_frame": len(plan.jobs),
           "max_layer_bound_ratio": max(c["qualify"]["max_bound_ratio"] for c in crops.values()),
           "max_landmark_px": max(c["qualify"]["landmarks"]["max_px"] for c in crops.values()),
           "linked_bit_identical_to_qualification": True}
    for mode in ("cached", "linked"):
        frames = [f for c in crops.values() for f in c["frames"]["frames"][mode]]
        out[mode] = {
            "frames": len(frames),
            "wrapper_total_ms": med([sum(j["wrapper_ms"] for j in f["jobs"]) for f in frames]),
            "body_total_ms": med([sum(j["body_ms"] for j in f["jobs"]) for f in frames]),
            "host_wall_ms": med([f["host_wall_ms"] for f in frames]),
            "per_job": [{"kind": j.kind, "wrapper_ms": med([f["jobs"][n]["wrapper_ms"] for f in frames]),
                         "body_ms": med([f["jobs"][n]["body_ms"] for f in frames]),
                         "wall_ms": med([f["jobs"][n]["wall_ms"] for f in frames])}
                        for n, j in enumerate(plan.jobs)],
        }
    return out


def main():
    from framenpu.convolution import load_landmark_graph
    from framenpu.landmark import specifications

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-on-device", action="store_true")
    for name in ("jobs", "model", "corpus", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    if not args.run_on_device or os.geteuid() != 0 or not 2 <= args.rounds <= 20:
        p.error("Frame root, --run-on-device and rounds in 2..20 required")
    jobs = HR.load_jobs(args.jobs)
    loader_job, image, link_record = load_link(args.jobs)
    graph = load_landmark_graph(args.model / "hand.ncnn.param", args.model / "hand.ncnn.bin")
    specs = specifications(graph)
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    if manifest["param_sha256"] != graph.param_sha256 or manifest["weights_sha256"] != graph.weights_sha256:
        p.error("corpus/model mismatch")
    report = {
        "format": "framenpu-teacher-link-run",
        "builds": {k: v[1]["sha256"] for k, v in jobs.items()} | {"loader": link_record["loader"], "linked": link_record["linked"]},
        "source_sha256": {n: digest(ROOT / n) for n in ("tests/hardware_teacher_link.py", "tests/hardware_teacher_resident.py",
                                                       "tests/teacher_resident_layout.py", "tests/teacher_tail_layout.py")},
        "manifest_sha256": digest(args.corpus / "manifest.json"),
        "window": [f"{T.WIN_LO:#x}", f"{T.WIN_HI:#x}"],
        "reservation": [f"{CODE:#x}", f"{CODE + CODE_BYTES:#x}"],
        "audits": {"restored_jobs": 0, "restored_hmx_jobs": 0},
        "service_before": service(),
    }
    failed = False
    lease = _lease()
    try:
        with PowerSnapshot() as power:
            baseline = power.read()
            report["initial_health"] = settled_health(power, baseline)
            try:
                with Device() as observer:
                    mem = observer._transport.memory
                    if bytes(mem[CONTROL : CONTROL + 8]) == RECORD:
                        raise ValueError("resident refusal record present")
                    before = bytes(mem[ARENA_START:ARENA_END])
                report["arena_before_sha256"] = hashlib.sha256(before).hexdigest()
                window = before[T.WIN_LO - ARENA_START : T.WIN_HI - ARENA_START]
                reservation = before[CODE - ARENA_START : CODE + CODE_BYTES - ARENA_START]
                retired = True
                try:
                    with audited_device(power, report["audits"])(dispatch="direct") as dev:
                        runner = LinkRunner(dev, dev._transport.memory, jobs, graph, specs, loader_job, image)
                        try:
                            report["crops"], report["seal_wrapper_ms"] = run(runner, args.corpus, manifest, args.rounds)
                        finally:
                            if runner.sealed:
                                retired = False
                                if not dev._transport.poisoned:
                                    report["retire_wrapper_ms"] = runner.admin(8)
                                    retired = True
                        plan = runner.plans["resident"]
                        runner.submit(plan, plan.jobs[0], linked=True, refused=True)  # unsealed: loader refuses
                        report["unsealed_request_refused"] = True
                        report["summary"] = summarize(report["crops"], plan)
                finally:
                    with Device() as observer:
                        mem = observer._transport.memory
                        mem[T.WIN_LO : T.WIN_HI] = window
                        if retired:
                            mem[CODE : CODE + CODE_BYTES] = reservation
                        after = bytes(mem[ARENA_START:ARENA_END])
                    report["arena_after_sha256"] = hashlib.sha256(after).hexdigest()
                    report["reservation_restored"] = retired
                if after != before:
                    raise ValueError("arena differs after window/reservation restoration")
                report["final_health"] = settled_health(power, baseline)
                report["service_after"] = service()
                if report["service_before"] != report["service_after"]:
                    raise ValueError("DSP service changed")
                report["all_checks_passed"] = True
            except BaseException as exc:
                failed = True
                report["error"] = f"{type(exc).__name__}: {exc}"
                report["after_failure"] = failure_snapshot(power)
                report["interpretation"] = "stopped without retry/recovery/restart"
    finally:
        lease.close()
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"all_checks_passed": report.get("all_checks_passed", False), "error": report.get("error"),
                      "summary": report.get("summary")}, indent=1), flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
