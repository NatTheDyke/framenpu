"""Exclusive Mercury replay session for the qualified Steam Frame image.

Accepts already-normalized ONNX feeds for 1..3 crops. Convolutions run on the
DSP; the hint and heads use the qualified host contract. No crop queue,
background batching, retry, firmware recovery or service restart is used.
"""

import secrets
import struct
import threading
import time

from . import mercury
from . import _mercury_common as C
from . import _mercury_layout as L
from . import _native
from ._mercury_host import NativeDetector, NativeRequest
from .device import Device
from .errors import InvalidModel, RecoveryRequired
from framenpu_cdsp import _layout_native
from framenpu_cdsp.resident import arena_lease as _lease
from framenpu_cdsp.mercury_device import (  # noqa: F401  (re-exported: the device half's names)
    ADMIN_MAGIC,
    ADMIN_OPCODE,
    CODE,
    CODE_BYTES,
    CROPS4_IMAGE_SHA256,
    DET_IMAGE_SHA256,
    HEADS_IMAGE_SHA256,
    IMAGE_CACHED,
    IMAGE_SHA256,
    LOADER_SHA256,
    MULTI_IMAGE_SHA256,
    MULTI_SEGMENTS,
    RES_SLOTS,
    RESIDENT_IMAGE_SHA256,
    SEAL,
    SEAL_AT,
    SLOT_UNCACHED,
    SLOTS_IMAGE_SHA256,
    STREAM_IMAGE_SHA256,
    STREAM_LATE,
    MercuryLease,
    _completion,
    _completion_guard,
    _segment_metadata,
    frame_metadata,
    image_features,
    image_kind,
    load_jobs,
    open_image,
    replay_metadata,
    reservation,
)


class _Model:
    """One model a session serves: its lowered graph, plan, residency tag and tag slot."""

    def __init__(self, model, plan, slot, outputs):
        self.model, self.plan, self.slot, self.outputs = model, plan, slot, outputs
        self.native = None
        self.crop_ms = 0.0  # last measured DSP body per crop: the next request sleeps most of it
        self.tag = (0, 0, 0, 0)  # the kernel copies the images unless the slot holds it
        while not self.tag[0]:
            self.tag = struct.unpack("<4I", secrets.token_bytes(16))


class _Pending:
    """One request on its way through host prepare -> device run -> host finish."""

    def __init__(self, sl, count, timed):
        self.sl, self.count = sl, count
        self.prepared, self.extra, self.maps, self.timing = [], [], [], None
        self.marks = {} if timed else None
        self.restart()

    def restart(self):
        """start timing on the calling thread (a phase runs on one thread)"""
        if self.marks is not None:
            self._t, self._c = time.perf_counter(), time.thread_time()

    def mark(self, name):
        """wall ms of the phase since the last mark, plus its thread CPU ms and core (profiling)"""
        if self.marks is not None:
            now, cpu = time.perf_counter(), time.thread_time()
            self.marks[name] = (now - self._t) * 1e3
            self.marks[name + ".cpu"] = (cpu - self._c) * 1e3
            self.marks[name + ".core"] = _core()
            self._t, self._c = now, cpu


def _core():
    """the core the calling thread runs on (profiling; -1 if unknown)"""
    try:
        with open("/proc/thread-self/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[36])
    except (OSError, ValueError, IndexError):
        return -1


def _outputs(model):
    """per-crop outputs: all of Mercury's; for the stereo student only what its 3D hand
    needs (joints3d, is-hand logit)"""
    if model.detector:
        return tuple(model.outputs)
    return tuple(model.outputs) if model.views == 1 else ("joints3d", "scalar_extras")


class MercurySession:
    """One process owns the shared window until close; explicit idle releases HMX.

    A session opens lazily on the first infer(), holds power between calls, and
    releases it at idle(). A poisoned session retains its refusal record and
    device bytes and cannot reopen. The caller may use an independent CPU path.

    extra_models (a --res-slots image): further (path, crops per request) models served
    by the same session, infer(feeds, model=k) with k = 1, 2, ...; each gets its own part
    of the window and of the blob region and its own residency tag slot, so switching
    models copies no parameters. An extra model may be Mercury's hand detector
    (inputImg 160x160 -> hand_exists, cx, cy, size); it needs the packed-bias wide-window
    image (DET_IMAGE_SHA256) to fit beside a keypoint model.
    """

    def __init__(
        self,
        model_path,
        jobs,
        *,
        require_pin=True,
        native=None,
        resident=None,
        extra_models=(),
        _device_factory=Device,
        _lease_factory=_lease,
    ):
        """native: None uses the bundled host kernels when this platform and image allow
        (else NumPy), False always uses NumPy, True requires the kernels, and a
        (_native.Host, _layout_native.LayoutHost) pair uses those libraries (tests; a lone
        Host takes the bundled layout library). resident: None keeps the
        parameter images in the FeatureNN blob region when the image supports it, True
        requires that, False copies them for every crop (legacy plan, same image)."""
        # require_pin=False accepts retrained weights for the same graph: lower() still validates
        # the structure, shapes and operators, and every plan row is bounds-checked below.
        model = mercury.load_keypoint(model_path, require_pin=require_pin)
        # the image's features come from its link record, which load_jobs checked against the
        # pinned hash (framenpu_cdsp.mercury_device)
        self.loader, self.image, f = open_image(jobs)
        self.features = f
        self.heads, self.image_sha256 = f.heads, f.sha256
        self.res_slots, self.max_crops = f.res_slots, f.max_crops
        self.stream, self.multi = f.stream, f.multi
        self.resident_image = f.resident
        self.cbias, self.win_hi = f.cbias, f.win_hi  # RS_CBIAS + RS_WIN_WIDE
        if extra_models and not self.res_slots:
            raise InvalidModel("several models need the --res-slots image")
        if len(extra_models) >= RES_SLOTS:
            raise InvalidModel(f"at most {RES_SLOTS} models")
        if resident and not self.resident_image:
            raise InvalidModel("this Mercury image has no resident parameter support")
        self.resident = self.resident_image if resident is None else bool(resident)
        if self.cbias and not self.resident:
            raise InvalidModel("the packed-bias image runs resident requests only")
        # an RS_TILES build keeps 1x1-only tensors in B-tile layout (no gather). A record
        # that misstates it is harmless: a build without RS_TILES ignores the layout word
        # (every tensor stays planar), and one with it runs flag-free rows planar too.
        self.tiles = self.resident and f.tiles
        # 8-block 1x1 groups (--pw-deep --pw-lg 3): flagged ops get a larger scratch. A build
        # without them refuses flag 8; one with them runs unflagged ops 4-block.
        self.deep8 = self.tiles and f.deep8
        self.code_bytes, self.seal_at = f.code_bytes, f.seal_at
        self.models = []
        res_base = win_base = None
        for slot, (m, batch) in enumerate(
            [(model, self.max_crops)]
            + [(mercury.load_model(path, require_pin=False), batch) for path, batch in extra_models]
        ):
            if not 1 <= batch <= (self.max_crops if slot == 0 else 3):
                raise InvalidModel(
                    f"model {slot}'s requests carry one to {self.max_crops if slot == 0 else 3} crops"
                )
            try:
                plan = L.build_plan(
                    m,
                    prefix=not m.detector,
                    batch=batch,
                    replay=True,
                    pw_group=8,
                    heads=self.heads and not m.detector,
                    resident=self.resident,
                    tiles=self.tiles,
                    deep8=self.deep8,
                    res_base=res_base,
                    win_base=win_base,
                    cbias=self.cbias,
                    win_hi=self.win_hi,
                    max_crops=self.max_crops,
                )
            except ValueError as exc:
                raise InvalidModel(f"model {slot}: {exc}") from exc
            if extra_models and not self.resident:
                raise InvalidModel("several models need resident parameters")
            res_base, win_base = plan.res_end, -(-plan.win_end // 4096) * 4096
            for k in range(batch):
                for row in struct.iter_unpack("<32I", plan.plan.table):
                    if row[0] == L.KIND_RG and self.heads:
                        ok = C.regrid_ok(row)
                    else:
                        ok = C.native_ok(
                            C.replay_row(row, k, plan.plan.crop_stride, self.win_hi),
                            s2_direct=0,
                            pw_group=8,
                            mheads=self.heads,
                            resident=self.resident_image,
                            tiles=self.tiles,
                            deep8=self.deep8,
                            cbias=self.cbias,
                            win_hi=self.win_hi,
                        )
                    if not ok:
                        raise InvalidModel("Mercury plan exceeds native bounds")
            self.models.append(_Model(m, plan, slot, _outputs(m)))
        if native is not False:
            # the model kernels (framenpu) and the layout kernels (the device backend's)
            if isinstance(native, tuple):
                host, layout = native
            else:
                host = native if isinstance(native, _native.Host) else _native.bundled()
                layout = _layout_native.bundled()
            if host is None or layout is None:
                if native is True:
                    raise InvalidModel("the native host kernels are not available here")
            else:
                for sl in self.models:
                    try:
                        sl.native = (
                            NativeDetector(host, layout, sl.model, sl.plan, sl.outputs)
                            if sl.model.detector
                            else NativeRequest(
                                host, layout, sl.model, sl.plan, self.heads, sl.outputs
                            )
                        )
                    except ValueError as exc:
                        # an explicit request covers the primary model; extra models (other
                        # graphs) fall back to NumPy where the kernels do not cover them
                        if native is not None and sl.slot == 0:
                            raise InvalidModel(f"native host path: {exc}") from exc
        self._dev = MercuryLease(
            self.loader,
            self.image,
            f,
            self.models,
            self._stage,
            self._read_outputs,
            self.resident,
            device_factory=_device_factory,
            lease_factory=_lease_factory,
        )
        self.lock = threading.RLock()
        self.last_timing = None
        self.phase_times = False  # True: last_timing["phases_ms"] times each host step (profiling)
        self._host_cost = {}  # (prepare|finish, model, crops) -> seconds, a running estimate

    # the primary model (model 0), as single-model callers use it
    model = property(lambda self: self.models[0].model)
    plan = property(lambda self: self.models[0].plan)
    tag = property(lambda self: self.models[0].tag)
    native = property(lambda self: self.models[0].native)
    outputs = property(lambda self: self.models[0].outputs)
    crop_ms = property(lambda self: self.models[0].crop_ms)

    # the device half's state, as callers and tests read it
    device = property(lambda self: self._dev.device)
    power = property(lambda self: self._dev.power)
    sealed = property(lambda self: self._dev.sealed)
    owned = property(lambda self: self._dev.owned)
    saved = property(lambda self: self._dev.saved)
    marker = property(lambda self: self._dev.marker)
    closed = property(lambda self: self._dev.closed)

    def _open(self):
        self._dev.open()

    def _admin(self, command):
        self._dev._admin(command)

    def _payload(self, count, sl=None):
        return self._dev._payload(count, sl)

    def _device_run(self, p, during=None):
        """Stage, run and read back one prepared request (caller holds the lock). during:
        host-only work to run while the DSP computes (Device._submit's wait hook)."""
        self._dev.run_frame([p], during)

    def _device_run_frame(self, ps, during=None):
        self._dev.run_frame(ps, during)

    def _device_run_streamed(self, p0, sl1, feeds1):
        return self._dev.run_streamed(p0, sl1, feeds1, self._host_prepare, self._host_finish)

    def idle(self):
        """Release held clocks when tracking has no observations; keep the plan."""
        with self.lock:
            self._dev.idle()

    def close(self):
        with self.lock:
            self._dev.close()

    def __enter__(self):
        with self.lock:
            self._open()
        return self

    def __exit__(self, *exc):
        self.close()

    def infer(self, feeds, model=0):
        """Run the available 1..3 normalized crops (of extra model k: 1..its crops per
        request) now, preserving their order."""
        sl = self._model_for(model, feeds)
        with self.lock:
            p = self._host_prepare(sl, feeds)
            self._device_run(p)
            outs = self._host_finish(p)
            self.last_timing = p.timing
            return outs

    def infer_many(self, requests):
        """Run several requests [(model, feeds), ...] back to back, in order, overlapping the
        host side with the DSP: while request i runs on the DSP, the waiting thread prepares
        request i+1 and finishes request i-1 in the time it would otherwise sleep (only work
        whose measured cost fits that sleep runs there, so completion is never noticed later
        than by infer; the rest runs right after). One thread: no hand-off or lock contention.
        Each request is its own DSP job (the gate is held per job, as with infer). Returns each
        request's outputs; last_timing holds the per-request timings under "requests". Any
        failure stops the sequence (no retries)."""
        reqs = [(self._model_for(m, f), f) for m, f in requests]
        if not reqs:
            raise ValueError("infer_many needs at least one request")
        lanes, seen = [], {}
        for sl, f in reqs:  # three requests in flight never share a workspace
            key = (sl.slot, len(f))
            lanes.append(seen.get(key, 0) % 3)
            seen[key] = seen.get(key, 0) + 1
        n = len(reqs)
        pend, outs, overlapped = [None] * n, [None] * n, [0]

        def step(kind, j):
            key = (kind, reqs[j][0].slot, len(reqs[j][1]))
            t = time.perf_counter()
            if kind == "prepare":
                pend[j] = self._host_prepare(reqs[j][0], reqs[j][1], lanes[j])
            else:
                outs[j] = self._host_finish(pend[j])
            dt = time.perf_counter() - t
            old = self._host_cost.get(key)
            self._host_cost[key] = dt if old is None else 0.8 * old + 0.2 * dt

        with self.lock:
            step("prepare", 0)
            for i in range(n):
                todo = ([("prepare", i + 1)] if i + 1 < n else []) + (
                    [("finish", i - 1)] if i >= 1 else []
                )
                sl, count = pend[i].sl, pend[i].count
                budget, inside = 0.8e-3 * sl.crop_ms * count, []  # infer's expected-time sleep
                for kind, j in todo:
                    est = self._host_cost.get((kind, reqs[j][0].slot, len(reqs[j][1])))
                    if est is None or sum(e for _, _, e in inside) + est * 1.25 + 1e-4 > budget:
                        break
                    inside.append((kind, j, est))
                during = (lambda: [step(k, j) for k, j, _ in inside]) if inside else None
                overlapped[0] += len(inside)
                self._device_run(pend[i], during)
                for kind, j in todo[len(inside) :]:
                    step(kind, j)
            step("finish", n - 1)
            self.last_timing = {
                "requests": [p.timing for p in pend],
                "body_ms": sum(p.timing["body_ms"] for p in pend),
                "overlapped": overlapped[0],
                "host_steps": 2 * n,
            }
            return outs

    def infer_frame(self, requests):
        """Run one frame's requests [(model, feeds), ...] (distinct models, each 1..its crops
        per request), in order, as ONE DSP job on a --multi image: each request is a segment of
        the same submission (one gate hold, one wake and completion, one 20 ms budget), so a
        frame that runs the detector and ft7 pays one round trip. Up to MULTI_SEGMENTS requests.
        Without the --multi image the requests run as infer_many. Returns each request's
        outputs; last_timing holds the per-request timings under "requests". Any failure fails
        the whole frame (no retries)."""
        reqs = [(self._model_for(m, f), f) for m, f in requests]
        if not reqs:
            raise ValueError("infer_frame needs at least one request")
        if not self.multi:
            return self.infer_many(requests)
        if len(reqs) > MULTI_SEGMENTS:
            raise ValueError(f"a frame carries at most {MULTI_SEGMENTS} requests")
        if len({sl.slot for sl, _ in reqs}) != len(reqs):
            raise ValueError("a frame's requests are of distinct models")  # one workspace each
        with self.lock:
            if self.stream and len(reqs) == 2:
                p0 = self._host_prepare(*reqs[0])
                p1, early = self._device_run_streamed(p0, *reqs[1])
                pend = [p0, p1]
                outs = [
                    early if early is not None else self._host_finish(p0),
                    self._host_finish(p1),
                ]
            else:
                early = None
                pend = [self._host_prepare(sl, f) for sl, f in reqs]
                self._device_run_frame(pend)
                outs = [self._host_finish(p) for p in pend]
            self.last_timing = {
                "requests": [p.timing for p in pend],
                "body_ms": sum(p.timing["body_ms"] for p in pend),
                "segments": len(pend),
                "streamed": self.stream and len(reqs) == 2,
                "early_finish": early is not None,
            }
            return outs

    def _model_for(self, model, feeds):
        sl = self.models[model]
        if not 1 <= len(feeds) <= sl.plan.batch:
            raise ValueError(f"requests of this model carry one to {sl.plan.batch} crops")
        return sl

    def _host_prepare(self, sl, feeds, lane=0):
        """Check and convert one request's inputs (no device access)."""
        import numpy as np

        p = _Pending(sl, len(feeds), self.phase_times)
        if sl.native is not None:
            p.prepared = sl.native.prepare(feeds, lane)
            p.mark("host_prepare_ms")
            return p
        for feed in feeds:
            ts = {}
            for name, shape in sl.model.inputs.items():
                v = np.asarray(feed[name], np.float32)
                if v.shape != shape or not np.isfinite(v).all():
                    raise ValueError(f"{name}: finite values of shape {shape} required")
                ts[name] = v.reshape(shape[1:] if len(shape) > 1 else shape)
            if not np.isfinite(ts["inputImg"].astype(np.float16)).all():
                raise ValueError("image exceeds FP16 storage")
            if sl.model.detector:
                p.extra.append({})
                p.prepared.append({L.STEM_COLS: L.stem_columns(ts["inputImg"], mercury.DET_SIZE)})
                continue
            p.extra.append({k: ts[k] for k in ts.keys() - mercury.INPUTS.keys()})
            for name in L.HOST_HINT_OPS[:3]:
                op = sl.model.op(name)
                ts[op.output] = mercury._step(op, ts, "fp16", "fp16")
            if not np.isfinite(ts[L.HINT]).all():
                raise ValueError("hint exceeds FP16 storage")
            p.prepared.append({L.STEM_COLS: L.stem_columns(ts["inputImg"]), L.HINT: ts[L.HINT]})
        p.mark("host_prepare_ms")
        return p

    def _device_run(self, p, during=None):
        """Stage, run and read back one prepared request (caller holds the lock). during:
        host-only work to run while the DSP computes (Device._submit's wait hook)."""
        self._device_run_frame([p], during)

    def _stage(self, mem, q):
        """Write one prepared request's inputs into its window part."""
        if q.sl.native is not None:
            q.sl.native.stage(mem, q.prepared)
            return
        for k, ts in enumerate(q.prepared):
            L.stage_inputs(
                q.sl.plan.crop(k),
                lambda at, data: mem.__setitem__(slice(at, at + len(data)), data),
                ts,
            )

    def _read_outputs(self, p, get):
        """Read and check one request's outputs from the window (into its workspace or p.maps);
        get(at, n) reads the window."""
        import numpy as np

        sl, count, prepared = p.sl, p.count, p.prepared
        p.maps = []
        for k in range(count):
            if sl.native is not None:
                if not sl.native.read(prepared, get, k):
                    raise RecoveryRequired("Mercury output has invalid padding or values")
                continue
            maps = {}
            names = (
                (sl.plan.final,)
                if sl.model.detector
                else (L.OUTPUT, L.HEATMAP)
                if self.heads
                else (L.OUTPUT,)
            )
            for name in names:
                out, stray = L.read_tensor(sl.plan.crop(k), get, name)
                if stray or not np.isfinite(out).all():
                    raise RecoveryRequired("Mercury output has invalid padding or values")
                maps[name] = out.astype(np.float32)
            p.maps.append(maps)

    def _host_finish(self, p):
        """The request's outputs from its read-back maps (no device access)."""
        sl = p.sl
        p.restart()
        if sl.native is not None:
            outs = sl.native.finish(p.prepared)
        elif sl.model.detector:
            # the device ran the convolutions; the dense tail is 8 outputs per crop
            outs = [
                mercury.run_tail(sl.model, m, sl.plan.ops[-1].name, outputs=sl.outputs)
                for m in p.maps
            ]
        elif not self.heads:
            outs = [
                mercury.run_tail(sl.model, m | x, L.LAST, outputs=sl.outputs)
                for m, x in zip(p.maps, p.extra)
            ]
        else:
            # the heatmap head ran on the DSP: only the pooled vector heads remain
            names = sl.outputs if sl.model.views == 1 else sl.outputs + ("heatmap_xy",)
            outs = [
                mercury.run_tail(
                    sl.model,
                    {L.OUTPUT: m[L.OUTPUT], "heatmap_xy": m[L.HEATMAP][:21]} | x,
                    L.LAST,
                    skip=L.HEAD_OPS,
                    outputs=names,
                )
                for m, x in zip(p.maps, p.extra)
            ]
        p.mark("host_finish_ms")
        if (
            p.marks is not None and p.timing is not None
        ):  # a streamed early finish: set after the job
            p.timing["phases_ms"] = p.marks
        return outs
