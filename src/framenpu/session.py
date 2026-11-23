"""framenpu.open: models by manifest name, on one backend (docs/CONTRACTS.md, Session API).

    s = framenpu.open(["student_ft7.onnx", "grayscale_detection_160x160.onnx"], backend="service")
    outs = s.infer("mercury-ft7", crops)            # 1..s.max_crops("mercury-ft7") crops
    kp, det = s.infer_frame([("mercury-ft7", crops), ("mercury-detector-160", det_crops)])

Backends: "cpu" (onnxruntime), "device" (the DSP in process; root), "service" (the root NPU
service, framenpu.service) and "auto" (the service when its socket exists, else the CPU). Every
backend but "cpu" runs behind a CPU fallback that latches on the first failure, as the
tracker's FallbackBackend does: RecoveryRequired is never retried, and a request is replayed on
the CPU whole. Requests carry the crops available now; nothing is padded or queued.

A model is given as its manifest (a model.json file, or the folder holding it, with the
weights file beside it) or as its weights file (.onnx): then the model.json beside it, else the
manifest bundled with framenpu (framenpu/manifests) whose weights hash matches the file. The
weights' SHA-256 is checked when the CPU or the device loads them. The service runs its own
root-owned models: a model it does not serve runs on the CPU (without latching the others),
and backend_of(model) says where each model runs now. The service's model numbers (0 keypoint,
1 stereo3d, 2 detector) stay wire detail.
"""

import dataclasses
import hashlib
import logging
import os
from pathlib import Path

from .contracts import load_manifest

LOG = logging.getLogger(__name__)
BUNDLED = Path(__file__).with_name("manifests")
BACKENDS = ("auto", "cpu", "device", "service")
SERVICE_MODEL = {"keypoint": 0, "stereo3d": 1, "detector": 2}
CPU_CROPS = 4  # one ONNX run per crop, so any group; 4 keeps the tracker's planning unchanged


def bundled_manifests():
    """name -> Manifest of the manifests shipped with framenpu"""
    return {m.name: m for m in (load_manifest(p) for p in sorted(BUNDLED.glob("*.json")))}


class Entry:
    """One opened model: its manifest and its weights file."""

    def __init__(self, where, like=None):
        """where: a manifest, its folder or a weights file. like: for a weights file with no
        manifest (a retrained model of the same graph), the bundled manifest whose name,
        inputs and outputs it takes, with this file's hash."""
        if isinstance(where, tuple):
            where, like = where
        p = Path(where)
        if p.is_dir() or p.suffix == ".json":
            path = p / "model.json" if p.is_dir() else p
            self.manifest = load_manifest(path)
            self.weights = path.parent / self.manifest.weights_file
        else:
            beside = p.with_name("model.json")
            m = load_manifest(beside) if beside.is_file() else None
            if m is None or m.weights_file != p.name:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                bundled = bundled_manifests()
                found = [b for b in bundled.values() if b.weights_sha256 == digest]
                if found:
                    m = found[0]
                elif like in bundled:
                    m = dataclasses.replace(
                        bundled[like], weights_file=p.name, weights_sha256=digest
                    )
                else:
                    raise ValueError(f"{p.name}: no manifest (put a model.json beside it)")
            self.manifest, self.weights = m, p
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

    def infer_many(self, requests):
        """Several requests (models may repeat), in order; back to back here."""
        if not requests:
            raise ValueError("infer_many carries one or more requests")
        return [self.infer(m, crops) for m, crops in requests]

    def served(self, model):
        """True when this backend runs the model (a service may not serve every model)."""
        return self._entry(model).name in self.entries

    def backend_of(self, model):
        return self.name

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
    """onnxruntime, one run per crop. The first model loads when the session opens, the others
    on their first request (a fallback CPU session then costs only the model it runs)."""

    name = "cpu"

    def __init__(self, entries, threads=1):
        import onnxruntime as ort

        super().__init__(entries)
        self._ort, self.threads, self.sess = ort, threads, {}
        if entries:
            self._load(entries[0])

    def _load(self, e):
        ort = self._ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(str(e.verified_weights()), so, providers=["CPUExecutionProvider"])
        self.sess[e.name] = s
        return s

    def max_crops(self, model):
        self._entry(model)
        return CPU_CROPS

    def infer(self, model, crops):
        e = self._entry(model)
        s = self.sess.get(e.name) or self._load(e)
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
        # models the service does not serve run on the CPU (FallbackSession routes them)
        self.number = {
            e.name: SERVICE_MODEL[e.kind]
            for e in entries
            if SERVICE_MODEL[e.kind] in self.client.models
        }
        if not self.number:
            self.client.close()
            raise RuntimeError("the NPU service serves none of these models")

    def served(self, model):
        return self._entry(model).name in self.number

    def _number(self, model):
        name = self._entry(model).name
        if name not in self.number:
            raise ValueError(f"the NPU service does not serve {name}")
        return self.number[name]

    @property
    def last_timing(self):
        return self.client.last_timing

    def max_crops(self, model):
        return self.client._most(self._number(model))

    def infer(self, model, crops):
        return self.client.infer(crops, model=self._number(model))

    def infer_frame(self, requests):
        _distinct(requests)
        if len(requests) == 1 or len(requests) > self.client.max_frame:
            return super().infer_frame(requests)
        return self.client.infer_frame([(self._number(m), c) for m, c in requests])

    def infer_many(self, requests):
        """One round trip per max_seq requests (the service overlaps their host work)."""
        if not requests:
            raise ValueError("infer_many carries one or more requests")
        most = self.client.max_seq
        if most < 2 or len(requests) == 1:
            return super().infer_many(requests)
        reqs = [(self._number(m), c) for m, c in requests]
        out, timings = [], []
        for k in range(0, len(reqs), most):
            out += self.client.infer_many(reqs[k : k + most])
            timings += (self.client.last_timing or {}).get("requests", [])
        if len(reqs) > most:  # several round trips: their requests' timings together
            self.client.last_timing = {"requests": timings}
        return out

    def idle(self):
        self.client.idle()

    def close(self):
        self.client.close()


class DeviceSession(_Base):
    name = "framenpu"
    FRAME = 2  # requests per frame submission (the cdsp plugin's Capabilities.segments)
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

    def infer_many(self, requests):
        if not requests:
            raise ValueError("infer_many carries one or more requests")
        return self.session.infer_many([(self.index[self._entry(m).name], c) for m, c in requests])

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

    def _acc(self, model):
        """the accelerator when it runs this model now, else None (the CPU runs it)"""
        a = self.accelerator
        return a if a is not None and a.served(model) else None

    def served(self, model):
        return self._acc(self._entry(model).name) is not None

    def backend_of(self, model):
        return self.accelerator.name if self.served(model) else "cpu"

    @property
    def last_timing(self):
        return getattr(self.accelerator, "last_timing", None)

    def max_crops(self, model):
        a = self._acc(self._entry(model).name)
        return (a or self.cpu).max_crops(model)

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
        a = self._acc(model)
        if a is not None:
            try:
                return self._check(model, a.infer(model, crops), len(crops))
            except Exception as exc:
                self._disable(exc)
        return self.cpu.infer(model, crops)

    def _routed(self, requests, run):
        """The accelerator's share of the requests through run(accelerator, share), the rest on
        the CPU; results in request order. A failure latches to the CPU, which then runs all."""
        for m, crops in requests:
            self._bounds(m, crops)
        acc = [k for k, (m, _) in enumerate(requests) if self._acc(m) is not None]
        out = [None] * len(requests)
        if acc:
            share = [requests[k] for k in acc]
            try:
                got = run(self.accelerator, share)
                if len(got) != len(share):
                    raise ValueError("FrameNPU returned the wrong request count")
                for k, (m, c), o in zip(acc, share, got):
                    out[k] = self._check(m, o, len(c))
            except Exception as exc:
                self._disable(exc)
                out = [None] * len(requests)
        for k, (m, c) in enumerate(requests):
            if out[k] is None:
                out[k] = self.cpu.infer(m, c)
        return out

    def infer_frame(self, requests):
        _distinct(requests)
        return self._routed(requests, lambda a, share: a.infer_frame(share))

    def infer_many(self, requests):
        if not requests:
            raise ValueError("infer_many carries one or more requests")
        return self._routed(requests, lambda a, share: a.infer_many(share))

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
    """Open models on one backend; see the module docstring. Each model is a manifest, its
    folder, a weights file, or (weights file, bundled manifest name) for a retrained model of
    that graph with no manifest of its own. jobs: the image folder for "device" (default: the device backend's bundled image); socket: the service's socket for "service"
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
    return FallbackSession(cpu, lambda: DeviceSession(entries, jobs))
