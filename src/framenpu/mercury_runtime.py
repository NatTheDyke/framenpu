"""Exclusive Mercury replay session for the qualified Steam Frame image.

Accepts already-normalized ONNX feeds for 1..3 crops. Convolutions run on the
DSP; the hint and heads use the qualified host contract. No crop queue,
background batching, retry, firmware recovery or service restart is used.
"""

import threading
import time

from . import mercury
from . import _native
from ._mercury_host import NativeDetector, NativeRequest
from .contracts import Refusal
from .errors import InvalidModel


class _Model:
    """One model a session serves: its lowered graph and the device plugin's compiled plan (its
    split between host and device, residency slot and tag, measured crop time)."""

    def __init__(self, model, compiled, outputs):
        self.model, self.compiled, self.outputs = model, compiled, outputs
        self.split = compiled.split
        self.native = None
        self.numpy = compiled.numpy_staging()  # the layout half of the NumPy host path

    # what the lease reads of a request's model: the compiled plan's
    plan = property(lambda self: self.compiled.plan)
    slot = property(lambda self: self.compiled.slot)
    tag = property(lambda self: self.compiled.tag)
    crop_ms = property(
        lambda self: self.compiled.crop_ms,
        lambda self, v: setattr(self.compiled, "crop_ms", v),
    )


class _Pending:
    """One request on its way through host prepare -> device run -> host finish."""

    def __init__(self, sl, count, timed):
        self.sl, self.count = sl, count
        self.prepared, self.extra, self.staged, self.timing = [], [], None, None
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
        jobs=None,
        *,
        plugin=None,
        require_pin=True,
        native=None,
        resident=None,
        extra_models=(),
        _device_factory=None,
        _lease_factory=None,
    ):
        """native: None uses the bundled host kernels when this platform and image allow
        (else NumPy), False always uses NumPy, True requires the kernels, and a
        (_native.Host, _layout_native.LayoutHost) pair uses those libraries (tests; a lone
        Host takes the bundled layout library). resident: None keeps the
        parameter images in the FeatureNN blob region when the image supports it, True
        requires that, False copies them for every crop (legacy plan, same image).
        plugin: the device plugin (framenpu.devices); None loads "cdsp" on jobs (None: the image
        the backend ships)."""
        # require_pin=False accepts retrained weights for the same graph: lower() still validates
        # the structure, shapes and operators, and every plan row is bounds-checked below.
        model = mercury.load_keypoint(model_path, require_pin=require_pin)
        # the plugin opened and validated the image against its pinned hashes; the features come
        # from the image's link record
        if plugin is None:
            from . import devices

            options = {"jobs": jobs, "resident": resident}
            if _device_factory is not None:
                options["_device_factory"] = _device_factory
            if _lease_factory is not None:
                options["_lease_factory"] = _lease_factory
            plugin = devices.load("cdsp", **options)
        self.plugin = plugin
        self.loader, self.image, self.features = plugin.loader, plugin.image, plugin.features
        self.resident, self.tiles, self.deep8 = plugin.resident, plugin.tiles, plugin.deep8
        self.win_hi = plugin.features.win_hi
        self.code_bytes, self.seal_at = plugin.features.code_bytes, plugin.features.seal_at
        caps = plugin.capabilities()
        # what the image's record says it supports (the plugin checked it against its pins)
        self.image_sha256, self.max_crops = caps.image_sha256, caps.max_crops
        self.heads = "heads" in caps.features
        self.res_slots = "res_slots" in caps.features
        self.stream, self.multi, self.segments = caps.stream, caps.segments > 1, caps.segments
        self.resident_image = "resident" in caps.features
        self.cbias = "cbias" in caps.features
        if extra_models and caps.resident_slots < 2:
            raise InvalidModel("several models need the --res-slots image")
        if extra_models and 1 + len(extra_models) > caps.resident_slots:
            raise InvalidModel(f"at most {caps.resident_slots} models")
        self.models = []
        for slot, (m, batch) in enumerate(
            [(model, self.max_crops)]
            + [(mercury.load_model(path, require_pin=False), batch) for path, batch in extra_models]
        ):
            compiled = plugin.compile(m, batch)
            if isinstance(compiled, Refusal):
                raise InvalidModel(compiled.reason)
            self.models.append(_Model(m, compiled, _outputs(m)))
        if native is not False:
            # the model kernels (framenpu) and the layout kernels (the device backend's)
            if isinstance(native, tuple):
                host, layout = native
            else:
                host = native if isinstance(native, _native.Host) else _native.bundled()
                layout = plugin.layout()
            if host is None or layout is None:
                if native is True:
                    raise InvalidModel("the native host kernels are not available here")
            else:
                for sl in self.models:
                    try:
                        kind = NativeDetector if sl.model.detector else NativeRequest
                        sl.native = kind(
                            host,
                            sl.compiled.native_staging(layout),
                            sl.model,
                            sl.split,
                            sl.outputs,
                        )
                    except ValueError as exc:
                        # an explicit request covers the primary model; extra models (other
                        # graphs) fall back to NumPy where the kernels do not cover them
                        if native is not None and sl.slot == 0:
                            raise InvalidModel(f"native host path: {exc}") from exc
        self._dev = plugin.open()
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
        frame that runs the detector and ft7 pays one round trip. Up to the image's segments (capabilities) requests.
        Without the --multi image the requests run as infer_many. Returns each request's
        outputs; last_timing holds the per-request timings under "requests". Any failure fails
        the whole frame (no retries)."""
        reqs = [(self._model_for(m, f), f) for m, f in requests]
        if not reqs:
            raise ValueError("infer_frame needs at least one request")
        if not self.multi:
            return self.infer_many(requests)
        if len(reqs) > self.segments:
            raise ValueError(f"a frame carries at most {self.segments} requests")
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
        most = sl.compiled.max_crops
        if not 1 <= len(feeds) <= most:
            raise ValueError(f"requests of this model carry one to {most} crops")
        return sl

    def _host_prepare(self, sl, feeds, lane=0):
        """Check and convert one request's inputs (no device access)."""
        import numpy as np

        p = _Pending(sl, len(feeds), self.phase_times)
        if sl.native is not None:
            p.prepared = sl.native.prepare(feeds, lane)
            p.staged = p.prepared.sw
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
                p.prepared.append({"inputImg": ts["inputImg"]})
                continue
            p.extra.append({k: ts[k] for k in ts.keys() - mercury.INPUTS.keys()})
            for name in sl.split.prefix:
                op = sl.model.op(name)
                ts[op.output] = mercury._step(op, ts, "fp16", "fp16")
            hint = sl.split.staged[1]
            if not np.isfinite(ts[hint]).all():
                raise ValueError("hint exceeds FP16 storage")
            p.prepared.append({"inputImg": ts["inputImg"], hint: ts[hint]})
        # the layout half converts to the kernel's layout now, outside the gate
        p.staged = sl.numpy.request(p.prepared)
        p.mark("host_prepare_ms")
        return p

    def _device_run(self, p, during=None):
        """Stage, run and read back one prepared request (caller holds the lock). during:
        host-only work to run while the DSP computes (Device._submit's wait hook)."""
        self._device_run_frame([p], during)

    def _host_finish(self, p):
        """The request's outputs from its read-back maps (no device access)."""
        sl = p.sl
        p.restart()
        if sl.native is not None:
            outs = sl.native.finish(p.prepared)
        elif sl.model.detector:
            # the device ran the convolutions; the dense tail is 8 outputs per crop
            outs = [
                mercury.run_tail(sl.model, m, sl.split.last, outputs=sl.outputs)
                for m in p.staged.maps
            ]
        elif not sl.split.skip:
            outs = [
                mercury.run_tail(sl.model, m | x, sl.split.last, outputs=sl.outputs)
                for m, x in zip(p.staged.maps, p.extra)
            ]
        else:
            # the heatmap head ran on the DSP: only the pooled vector heads remain
            names = sl.outputs if sl.model.views == 1 else sl.outputs + ("heatmap_xy",)
            outs = [
                mercury.run_tail(sl.model, m | x, sl.split.last, skip=sl.split.skip, outputs=names)
                for m, x in zip(p.staged.maps, p.extra)
            ]
        p.mark("host_finish_ms")
        if (
            p.marks is not None and p.timing is not None
        ):  # a streamed early finish: set after the job
            p.timing["phases_ms"] = p.marks
        return outs
