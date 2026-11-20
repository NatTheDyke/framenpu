"""Exclusive Mercury replay session for the qualified Steam Frame image.

Accepts already-normalized ONNX feeds for 1..3 crops. Convolutions run on the
DSP; the hint and heads use the qualified host contract. No crop queue,
background batching, retry, firmware recovery or service restart is used.
"""

import hashlib
import json
from pathlib import Path
import secrets
import struct
import threading

from . import mercury
from . import _mercury_common as C
from . import _mercury_layout as L
from . import _native
from ._mercury_host import NativeRequest
from .device import Device
from .errors import InvalidModel, RecoveryRequired
from .job import Job
from .resident_gemm import _lease, RECORD, SSR
from .resident_landmark import CONTROL

IMAGE_SHA256 = "d7a78a81469c10acaf193b0d99cc0415f92d1de845a7a9aa292a7a1ebe691819"
# The same build with --mheads: the heatmap head (step 2c) runs on the DSP too.
HEADS_IMAGE_SHA256 = "582d79ce821304ab90b48d18c5e9d7c5e4396614b9c6eb70bf2e7b6febf114d3"
LOADER_SHA256 = "72280ad93eb4ffcf49e7de26908fa2d8f9056e0d9e1e8618f75eba867a7ba3f0"
CODE, CODE_BYTES, SEAL_AT, SEAL = 0x1FF0000, 0x4000, 0x1FF3F80, 0x544C4E4B
IMAGE_CACHED, SLOT_UNCACHED = 0x88FF0000, 0xF0A92000
ADMIN_MAGIC, ADMIN_OPCODE = 0x544C4144, 112


def load_jobs(folder):
    """Accept the qualified replay build only (with or without the DSP heatmap
    head), including its existing loader. Returns (loader, image, heads)."""
    folder = Path(folder)
    record = json.loads((folder / "link.json").read_text())
    loader = Job.load(folder / "teacher_link_loader.fjob")
    image = (folder / "teacher_link.bin").read_bytes()
    heads = bool(record.get("mheads"))
    want = HEADS_IMAGE_SHA256 if heads else IMAGE_SHA256
    if (
        hashlib.sha256(image).hexdigest() != want
        or hashlib.sha256(loader.code).hexdigest() != LOADER_SHA256
        or record["linked"]["sha256"] != want
        or record["loader"]["sha256"] != LOADER_SHA256
        or loader.name != "teacher_link_loader_v1"
        or not loader.requires_hmx
        or not loader.supports_direct
        or record.get("pw_group") != 8
        or any(
            not record.get(k)
            for k in ("stride2", "s2_even", "s2_direct", "pw_fast", "dw_fast", "replay")
        )
        or any(record.get(k) for k in ("heads", "stem", "pw_profile"))
    ):
        raise InvalidModel("Mercury requires the qualified replay image and loader")
    return loader, image, heads


def _completion(result, output_bytes):
    """Check worker identity, code/EVB, status and fault fields under the gate."""
    if len(result.workers) != 3 or len(result.output) != output_bytes:
        raise RecoveryRequired("Mercury completion is incomplete")
    for i, row in enumerate(result.workers):
        if (
            len(row) != 16
            or not row[0]
            or row[0] != result.workers[0][0]
            or row[1] != i
            or row[7] != SSR[i]
            or any(row[2:6])
            or any(row[14:16])
        ):
            raise RecoveryRequired("Mercury worker completion differs")
    if (
        result.workers[0][8] != 0x87000000
        or result.kernel_reference_ticks is None
        or not 0 < result.kernel_reference_ticks < 768000
    ):
        raise RecoveryRequired("Mercury EVB or wrapper time differs")


def _completion_guard(check):
    """Any interrupted or failed integrity check retains the DSP resources."""

    def validate(result):
        try:
            return check(result)
        except RecoveryRequired:
            raise
        except BaseException as exc:
            raise RecoveryRequired("Mercury completion validation failed") from exc

    return validate


def replay_metadata(result, count, descriptors, image_bytes):
    _completion(result, C.META_BYTES)
    m = struct.unpack("<256I", result.output)
    ticks = list(m[96 : 96 + descriptors])
    ends = list(m[48 : 48 + count]) if count > 1 else [ticks[-1]]
    base = ends[-2] if count > 1 else 0
    if (
        m[0] != C.RS_MAGIC
        or m[2] != descriptors * count
        or not IMAGE_CACHED <= m[1] < IMAGE_CACHED + image_bytes
        or any(m[3:48])
        or any(m[48 + count : 96] if count > 1 else m[48:96])
        or any(m[96 + descriptors :])
        or any(b <= a for a, b in zip([0] + ends, ends))
        or any(b < a for a, b in zip(ticks, ticks[1:]))
        or not base
        < ticks[0]
        <= ticks[-1]
        == ends[-1]
        <= min(384000, result.kernel_reference_ticks)
    ):
        raise RecoveryRequired("Mercury replay metadata or budget differs")
    return {
        "crops": count,
        "crop_end_ticks": ends,
        "crop_body_ms": [(b - a) / 19200 for a, b in zip([0] + ends, ends)],
        "body_ms": ends[-1] / 19200,
    }


class MercurySession:
    """One process owns the shared window until close; explicit idle releases HMX.

    A session opens lazily on the first infer(), holds power between calls, and
    releases it at idle(). A poisoned session retains its refusal record and
    device bytes and cannot reopen. The caller may use an independent CPU path.
    """

    def __init__(
        self,
        model_path,
        jobs,
        *,
        require_pin=True,
        native=None,
        _device_factory=Device,
        _lease_factory=_lease,
    ):
        """native: None uses the bundled host kernels when this platform and image allow
        (else NumPy), False always uses NumPy, True requires the kernels, and a
        _native.Host instance uses that library (tests)."""
        # require_pin=False accepts retrained weights for the same graph: lower() still validates
        # the structure, shapes and operators, and every plan row is bounds-checked below.
        self.model = mercury.load_keypoint(model_path, require_pin=require_pin)
        self.loader, self.image, self.heads = load_jobs(jobs)
        self.image_sha256 = HEADS_IMAGE_SHA256 if self.heads else IMAGE_SHA256
        self.plan = L.build_plan(
            self.model, prefix=True, batch=3, replay=True, pw_group=8, heads=self.heads
        )
        for k in range(3):
            for row in struct.iter_unpack("<32I", self.plan.plan.table):
                if row[0] == L.KIND_RG and self.heads:
                    ok = C.regrid_ok(row)
                else:
                    ok = C.native_ok(
                        C.replay_row(row, k, self.plan.plan.crop_stride),
                        s2_direct=0,
                        pw_group=8,
                        mheads=self.heads,
                    )
                if not ok:
                    raise InvalidModel("Mercury plan exceeds native bounds")
        self.native = None
        if native is not False:
            host = native if isinstance(native, _native.Host) else _native.bundled()
            if host is None:
                if native is True:
                    raise InvalidModel("the native host kernels are not available here")
            else:
                try:
                    self.native = NativeRequest(host, self.model, self.plan, self.heads)
                except ValueError as exc:
                    if native is not None:
                        raise InvalidModel(f"native host path: {exc}") from exc
        self._device_factory, self._lease_factory = _device_factory, _lease_factory
        self.device = self.lease = self.power = None
        self.closed = self.owned = self.sealed = False
        self.marker = RECORD + secrets.token_bytes(16) + bytes.fromhex(self.image_sha256)
        self.saved = []
        self.lock = threading.RLock()
        self.last_timing = None
        self.crop_ms = 0.0  # last measured DSP body per crop: the next request sleeps most of it

    def _owner(self):
        if (
            bytes(self.device._transport.memory[CONTROL : CONTROL + len(self.marker)])
            != self.marker
        ):
            raise RecoveryRequired("Mercury arena ownership changed")

    def _reservation(self, sealed):
        mem = self.device._transport.memory
        expect = bytearray(self.image.ljust(CODE_BYTES, b"\0"))
        struct.pack_into("<I", expect, SEAL_AT - CODE, SEAL if sealed else 0)
        if bytes(mem[CODE : CODE + CODE_BYTES]) != expect:
            raise RecoveryRequired("Mercury image or seal changed")
        self._owner()

    def _gate(self, action):
        t = self.device._transport
        t.claim(self.device.claim_timeout)
        try:
            action(t.memory)
        except RecoveryRequired:
            t.poisoned = True
            raise
        finally:
            if not t.poisoned:
                t.release_unsubmitted()

    def _admin(self, command):
        before, after = self.sealed, command == 7

        def validate(result):
            _completion(result, 256)
            m = struct.unpack("<64I", result.output)
            if (
                m[0] != ADMIN_MAGIC
                or not SLOT_UNCACHED <= m[1] < SLOT_UNCACHED + len(self.loader.code)
                or m[2] != (SEAL if after else 0)
                or any(m[3:])
            ):
                raise RecoveryRequired("Mercury seal/retire completion differs")
            self._reservation(after)
            self.sealed = after

        self.device._submit(
            self.loader,
            struct.pack("<4I", ADMIN_MAGIC, command, CODE, CODE_BYTES).ljust(128, b"\0"),
            opcode=ADMIN_OPCODE,
            params=(0,),
            output_size=256,
            use_direct=True,
            _prepare=lambda t: self._reservation(before),
            _completion_validator=_completion_guard(validate),
            _held_power=self.power,
        )

    def _open(self):
        if self.closed:
            raise RuntimeError("Mercury session is closed")
        if self.device is not None:
            return
        self.lease = self._lease_factory()
        try:
            self.device = self._device_factory(dispatch="direct")
            self.device.__enter__()

            def install(mem):
                if bytes(mem[CONTROL : CONTROL + len(RECORD)]) == RECORD:
                    raise RecoveryRequired("resident refusal record present")
                self.saved = [
                    (at, bytes(mem[at : at + size]))
                    for at, size in (
                        (C.WIN_LO, C.WIN_HI - C.WIN_LO),
                        (CODE, CODE_BYTES),
                        (CONTROL, 128),
                    )
                ]
                self.owned = True
                mem[CONTROL : CONTROL + len(self.marker)] = self.marker
                image = self.image.ljust(CODE_BYTES, b"\0")
                mem[CODE : CODE + CODE_BYTES] = image
                for off, data in C.setup_writes(self.plan.plan).items():
                    mem[off : off + len(data)] = data
                    if bytes(mem[off : off + len(data)]) != data:
                        raise RecoveryRequired("Mercury parameter upload differs")
                self._reservation(False)

            self._gate(install)
            self._admin(7)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        with self.lock:
            self._open()
        return self

    def __exit__(self, *exc):
        self.close()

    def infer(self, feeds):
        """Run the available 1..3 normalized crops now, preserving their order."""
        import numpy as np

        if not 1 <= len(feeds) <= 3:
            raise ValueError("Mercury requests carry one to three crops")
        prepared = []
        for feed in feeds if self.native is None else ():
            ts = {}
            for name, shape in mercury.INPUTS.items():
                v = np.asarray(feed[name], np.float32)
                if v.shape != shape or not np.isfinite(v).all():
                    raise ValueError(f"{name}: finite values of shape {shape} required")
                ts[name] = v.reshape(shape[1:] if len(shape) > 1 else shape)
            if not np.isfinite(ts["inputImg"].astype(np.float16)).all():
                raise ValueError("image exceeds FP16 storage")
            for name in L.HOST_HINT_OPS[:3]:
                op = self.model.op(name)
                ts[op.output] = mercury._step(op, ts, "fp16", "fp16")
            if not np.isfinite(ts[L.HINT]).all():
                raise ValueError("hint exceeds FP16 storage")
            prepared.append({L.STEM_COLS: L.stem_columns(ts["inputImg"]), L.HINT: ts[L.HINT]})
        with self.lock:
            if self.native is not None:  # under the lock: the workspace is shared
                prepared = self.native.prepare(feeds)
            self._open()
            if self.power is None:
                self.power = self.device._hold_power()
            count, result_maps = len(feeds), []

            def stage(mem):
                self._reservation(True)
                if self.native is not None:
                    self.native.stage(mem, prepared)
                    return
                for k, ts in enumerate(prepared):
                    L.stage_inputs(
                        self.plan.crop(k),
                        lambda at, data: mem.__setitem__(slice(at, at + len(data)), data),
                        ts,
                    )

            self._gate(stage)
            job = self.plan.plan.jobs[0]

            def validate(result):
                self._reservation(True)
                self.last_timing = replay_metadata(
                    result, count, job.end - job.first, len(self.image)
                )
                get = lambda at, n: bytes(self.device._transport.memory[at : at + n])  # noqa: E731
                for k in range(count):
                    if self.native is not None:
                        if not self.native.read(prepared, get, k):
                            raise RecoveryRequired("Mercury output has invalid padding or values")
                        continue
                    maps = {}
                    for name in (L.OUTPUT, L.HEATMAP) if self.heads else (L.OUTPUT,):
                        out, stray = L.read_tensor(self.plan.crop(k), get, name)
                        if stray or not np.isfinite(out).all():
                            raise RecoveryRequired("Mercury output has invalid padding or values")
                        maps[name] = out.astype(np.float32)
                    result_maps.append(maps)

            self.device._submit(
                self.loader,
                C.payload(self.plan.plan, count),
                opcode=C.RS_OPCODE,
                params=(1, job.first, job.end, 0, 0, 0),
                output_size=C.META_BYTES,
                use_direct=True,
                _prepare=lambda t: self._reservation(True),
                _completion_validator=_completion_guard(validate),
                _held_power=self.power,
                _expect_s=0.8e-3 * self.crop_ms * count,
            )
            self.crop_ms = self.last_timing["body_ms"] / count
            if self.native is not None:
                return self.native.finish(prepared)
            if not self.heads:
                return [mercury.run_tail(self.model, m, L.LAST) for m in result_maps]
            # the heatmap head ran on the DSP: only the pooled vector heads remain
            return [
                mercury.run_tail(
                    self.model,
                    {L.OUTPUT: m[L.OUTPUT], "heatmap_xy": m[L.HEATMAP][:21]},
                    L.LAST,
                    skip=L.HEAD_OPS,
                )
                for m in result_maps
            ]

    def idle(self):
        """Release held clocks when tracking has no observations; keep the plan."""
        with self.lock:
            if self.power is not None:
                power, self.power = self.power, None
                if self.device._transport.poisoned:
                    power.abandon()
                else:
                    self.device._release_power(power)

    def close(self):
        with self.lock:
            if self.closed:
                return
            try:
                self.idle()
                if self.device is not None and self.owned and not self.device._transport.poisoned:
                    if self.sealed:
                        self._admin(8)

                    def restore(mem):
                        self._reservation(False)
                        for off, data in self.saved:
                            mem[off : off + len(data)] = data
                            if bytes(mem[off : off + len(data)]) != data:
                                raise RecoveryRequired("Mercury arena restoration differs")
                        self.owned = False

                    self._gate(restore)
            except BaseException:
                if self.device is not None and self.owned:
                    self.device._transport.poisoned = True
                raise
            finally:
                try:
                    if self.device is not None:
                        self.device.close()
                finally:
                    if self.lease is not None:
                        self.lease.close()
                    self.closed = True
