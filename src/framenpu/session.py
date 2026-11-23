"""framenpu.open: models by manifest name, on one backend (docs/CONTRACTS.md, Session API).

    s = framenpu.open(["models/mercury-ft7", "models/mercury-detector-160"], backend="service")
    outs = s.infer("mercury-ft7", crops)            # 1..s.max_crops("mercury-ft7") crops
    kp, det = s.infer_frame([("mercury-ft7", crops), ("mercury-detector-160", det_crops)])

Backends: "cpu" (onnxruntime), "device" (the DSP in process; root), "service" (the root NPU
service, framenpu.service) and "auto" (the service when its socket exists, else the CPU). Every
backend but "cpu" runs behind a CPU fallback that latches on the first failure, as the
tracker's FallbackBackend does: RecoveryRequired is never retried, and a request is replayed on
the CPU whole. Requests carry the crops available now; nothing is padded or queued.

Each manifest names its weights file beside it; its SHA-256 is checked when the CPU or the
device loads it. The service runs its own root-owned models; a session only checks that it
serves a model of each manifest's kind. The service's model numbers (0 keypoint, 1 stereo3d,
2 detector) stay wire detail.
"""

import hashlib
import logging
import os
from pathlib import Path

from .contracts import load_manifest

LOG = logging.getLogger(__name__)
BACKENDS = ("auto", "cpu", "device", "service")
SERVICE_MODEL = {"keypoint": 0, "stereo3d": 1, "detector": 2}
CPU_CROPS = 4  # one ONNX run per crop, so any group; 4 keeps the tracker's planning unchanged


class Entry:
    """One opened model: its manifest and its weights file."""

    def __init__(self, where):
        p = Path(where)
        path = p / "model.json" if p.is_dir() else p
        self.manifest = load_manifest(path)
        self.weights = path.parent / self.manifest.weights_file
        self.name, self.kind = self.manifest.name, self.manifest.kind
        self.outputs = tuple(t.name for t in self.manifest.outputs)
        self.shapes = {t.name: tuple(t.shape) for t in self.manifest.outputs}

    def verified_weights(self):
        data = self.weights.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.manifest.weights_sha256:
            raise ValueError(f"{self.weights.name} does not match the {self.name} manifest")
        return self.weights


class _Base:
    name = "?"
    last_timing = None

    def __init__(self, entries):
        self.entries = {e.name: e for e in entries}
        self.models = tuple(self.entries)

    def _entry(self, model):
        try:
            return self.entries[model]
        except KeyError:
            raise ValueError(f"model {model!r} is not open in this session") from None

    def infer_frame(self, requests):
        """Back to back where the backend has no frame submission."""
        _distinct(requests)
        return [self.infer(m, crops) for m, crops in requests]

    def idle(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _distinct(requests):
    if not requests or len({m for m, _ in requests}) != len(requests):
        raise ValueError("a frame carries one or more requests of distinct models")


class CPUSession(_Base):
    name = "cpu"

    def __init__(self, entries, threads=1):
        import onnxruntime as ort

        super().__init__(entries)
        self.sess = {}
        for e in entries:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess[e.name] = ort.InferenceSession(
                str(e.verified_weights()), so, providers=["CPUExecutionProvider"]
            )

    def max_crops(self, model):
        self._entry(model)
        return CPU_CROPS

    def infer(self, model, crops):
        e = self._entry(model)
        s = self.sess[e.name]
        return [dict(zip(e.outputs, s.run(list(e.outputs), f))) for f in crops]


class ServiceSession(_Base):
    name = "framenpu"

    def __init__(self, entries, socket_path=None):
        from .service import SOCKET, Client

        super().__init__(entries)
        kinds = [e.kind for e in entries]
        if len(set(kinds)) != len(kinds):
            raise ValueError("the NPU service serves one model of each kind")
        self.client = Client(socket_path or SOCKET)
        self.number = {}
        for e in entries:
            n = SERVICE_MODEL[e.kind]
            if n not in self.client.models:
                self.client.close()
                raise RuntimeError(f"the NPU service does not serve a {e.kind} model ({e.name})")
            self.number[e.name] = n

    @property
    def last_timing(self):
        return self.client.last_timing

    def max_crops(self, model):
        return self.client._most(self.number[self._entry(model).name])

    def infer(self, model, crops):
        return self.client.infer(crops, model=self.number[self._entry(model).name])

    def infer_frame(self, requests):
        _distinct(requests)
        if len(requests) == 1 or len(requests) > self.client.max_frame:
            return super().infer_frame(requests)
        return self.client.infer_frame([(self.number[self._entry(m).name], c) for m, c in requests])

    def idle(self):
        self.client.idle()

    def close(self):
        self.client.close()


class DeviceSession(_Base):
    name = "framenpu"
    FRAME = 2  # requests per frame submission (framenpu_cdsp MULTI_SEGMENTS)
    EXTRA_CROPS = {"stereo3d": 2, "detector": 2}

    def __init__(self, entries, jobs):
        from .mercury_runtime import MercurySession

        super().__init__(entries)
        if not entries or entries[0].kind != "keypoint":
            raise ValueError("the device session's first model is a keypoint model")
        extra = [(str(e.verified_weights()), self.EXTRA_CROPS[e.kind]) for e in entries[1:]]
        # the weights matched their manifest above; MercurySession still validates the graph
        self.session = MercurySession(
            str(entries[0].verified_weights()), jobs, require_pin=False, extra_models=extra
        )
        self.index = {e.name: k for k, e in enumerate(entries)}

    @property
    def last_timing(self):
        return self.session.last_timing

    def max_crops(self, model):
        k = self.index[self._entry(model).name]
        return self.session.max_crops if k == 0 else self.EXTRA_CROPS[self.entries[model].kind]

    def infer(self, model, crops):
        k = self.index[self._entry(model).name]
        return self.session.infer(crops, model=k) if k else self.session.infer(crops)

    def infer_frame(self, requests):
        _distinct(requests)
        if len(requests) == 1 or len(requests) > self.FRAME:
            return super().infer_frame(requests)
        return self.session.infer_frame([(self.index[self._entry(m).name], c) for m, c in requests])

    def idle(self):
        self.session.idle()

    def close(self):
        self.session.close()


class FallbackSession(_Base):
    """An accelerator session that latches to the CPU session for the rest of its lifetime on
    its first failure (opening included). The CPU session is always built first."""

    def __init__(self, cpu, accelerator_factory):
        super().__init__(list(cpu.entries.values()))
        self.cpu, self.accelerator = cpu, None
        try:
            self.accelerator = accelerator_factory()
        except Exception as exc:
            LOG.warning("FrameNPU unavailable; using CPU: %s", exc)

    @property
    def name(self):
        return self.accelerator.name if self.accelerator is not None else "cpu"

    @property
    def last_timing(self):
        return getattr(self.accelerator, "last_timing", None)

    def max_crops(self, model):
        a = self.accelerator if self.accelerator is not None else self.cpu
        return a.max_crops(model)

    def _disable(self, exc):
        backend, self.accelerator = self.accelerator, None
        LOG.warning("FrameNPU disabled for this session; using CPU: %s", exc)
        try:
            backend.close()
        except Exception as cleanup:
            LOG.warning("FrameNPU cleanup retained device state: %s", cleanup)

    def _check(self, model, out, n):
        import numpy as np

        if len(out) != n:
            raise ValueError("FrameNPU returned the wrong crop count")
        for crop in out:
            for name, shape in self.entries[model].shapes.items():
                v = np.asarray(crop[name])
                if v.shape != shape or not np.isfinite(v).all():
                    raise ValueError(f"FrameNPU returned invalid {name}")
        return out

    def _bounds(self, model, crops):
        most = self.max_crops(self._entry(model).name)
        if not 1 <= len(crops) <= most:
            raise ValueError(f"{model} requests carry one to {most} crops")

    def infer(self, model, crops):
        self._bounds(model, crops)
        if self.accelerator is not None:
            try:
                return self._check(model, self.accelerator.infer(model, crops), len(crops))
            except Exception as exc:
                self._disable(exc)
        return self.cpu.infer(model, crops)

    def infer_frame(self, requests):
        _distinct(requests)
        for m, crops in requests:
            self._bounds(m, crops)
        if self.accelerator is not None:
            try:
                outs = self.accelerator.infer_frame(requests)
                if len(outs) != len(requests):
                    raise ValueError("FrameNPU returned the wrong request count")
                return [self._check(m, o, len(c)) for (m, c), o in zip(requests, outs)]
            except Exception as exc:
                self._disable(exc)
        return self.cpu.infer_frame(requests)

    def idle(self):
        if self.accelerator is not None:
            try:
                self.accelerator.idle()
            except Exception as exc:
                self._disable(exc)

    def close(self):
        backend, self.accelerator = self.accelerator, None
        try:
            if backend is not None:
                backend.close()
        except Exception as exc:
            LOG.warning("FrameNPU cleanup retained device state: %s", exc)
        finally:
            self.cpu.close()


def open(manifests, backend="auto", *, jobs=None, socket=None, threads=1, _cpu=CPUSession):
    """Open models (manifest files or folders holding model.json) on one backend; see the module
    docstring. jobs: the image folder for "device"; socket: the service's socket for "service"
    and "auto"; threads: onnxruntime threads per CPU model."""
    if backend not in BACKENDS:
        raise ValueError(f"backend: one of {BACKENDS}")
    entries = [Entry(m) for m in manifests]
    if not entries or len({e.name for e in entries}) != len(entries):
        raise ValueError("open one or more models of distinct names")
    if backend == "auto":
        from .service import SOCKET

        backend = "service" if os.path.exists(socket or SOCKET) else "cpu"
    cpu = _cpu(entries, threads)
    if backend == "cpu":
        return cpu
    if backend == "service":
        return FallbackSession(cpu, lambda: ServiceSession(entries, socket))
    if jobs is None:
        raise ValueError("the device backend needs the image folder (jobs)")
    return FallbackSession(cpu, lambda: DeviceSession(entries, jobs))
