"""The interfaces between framehands, framenpu and the framenpu-cdsp device backend.

docs/CONTRACTS.md is the specification; this module is its checkable form. It imports only the
standard library, so every layer (and a PC or CI install without the device backend) can use it.

Three of the four contracts live here:

- Session API (framehands -> framenpu): ``Session``.
- Model manifest (travels with each ONNX file): ``validate_manifest`` / ``load_manifest``.
- Device plugin ABI (framenpu -> device backend): ``DevicePlugin``, ``DeviceLease``,
  ``Refusal``, ``Capabilities``.

The fourth, the service wire protocol, stays with its server and client; its version is
``PROTOCOL_VERSION`` here so both ends and the docs name one number.

Phase 1 of the library split writes these down; nothing implements them yet under these names.
MercurySession and the tracker's backends are what the Session API is distilled from.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

MANIFEST_VERSION = 1
PLUGIN_ABI = 1
PROTOCOL_VERSION = 4
DEVICE_ENTRY_POINTS = "framenpu.devices"

# Inputs and outputs are numpy arrays; typed as Any so this module does not import numpy.
Feed = Mapping[str, Any]  # one crop: input name -> array of the manifest's shape
Outputs = dict  # one crop: output name -> array of the manifest's shape


# --- Session API ---------------------------------------------------------------------------


@runtime_checkable
class Session(Protocol):
    """Models opened together, run on one backend (CPU, in-process device, or the service).

    Models are addressed by manifest name. Requests carry 1..max_crops(model) crops, run in
    order, and return one Outputs per crop in that order; a request is never padded.
    """

    models: Sequence[str]
    last_timing: Mapping[str, Any] | None

    def max_crops(self, model: str) -> int:
        """Crops one request of this model may carry on this backend now."""

    def infer(self, model: str, crops: Sequence[Feed]) -> list[Outputs]:
        """Run one request now."""

    def infer_frame(self, requests: Sequence[tuple[str, Sequence[Feed]]]) -> list[list[Outputs]]:
        """Run one frame's requests (distinct models) as one submission where the backend can,
        else back to back. Returns each request's outputs."""

    def idle(self) -> None:
        """Release held clocks; keep plans and resident parameters."""

    def close(self) -> None:
        """Release everything the session owns. A session that needs recovery stays refused."""


# --- Model manifest ------------------------------------------------------------------------

KINDS = ("keypoint", "stereo3d", "detector")
DTYPES = ("float32",)
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """A manifest that does not meet the contract; the message names the field."""


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class Manifest:
    name: str
    kind: str
    weights_file: str
    weights_sha256: str
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    preferred_crops: int
    decode: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)

    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        return {t.name: t.shape for t in self.inputs}

    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {t.name: t.shape for t in self.outputs}


def _need(d, key, kind, where):
    if key not in d:
        raise ManifestError(f"{where}{key}: missing")
    v = d[key]
    if not isinstance(v, kind) or isinstance(v, bool):
        raise ManifestError(f"{where}{key}: expected {getattr(kind, '__name__', kind)}")
    return v


def _tensors(items, where):
    if not isinstance(items, list) or not items:
        raise ManifestError(f"{where}: expected a non-empty list")
    out, seen = [], set()
    for i, t in enumerate(items):
        at = f"{where}[{i}]."
        if not isinstance(t, dict):
            raise ManifestError(f"{where}[{i}]: expected an object")
        name = _need(t, "name", str, at)
        if not name or name in seen:
            raise ManifestError(f"{at}name: empty or repeated")
        seen.add(name)
        shape = _need(t, "shape", list, at)
        if not shape or not all(
            isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape
        ):
            raise ManifestError(f"{at}shape: expected positive integers")
        if shape[0] != 1:
            raise ManifestError(f"{at}shape: the leading dimension is one crop (1)")
        dtype = t.get("dtype", "float32")
        if dtype not in DTYPES:
            raise ManifestError(f"{at}dtype: one of {DTYPES}")
        out.append(Tensor(name, tuple(shape), dtype))
    return tuple(out)


def validate_manifest(doc: Mapping[str, Any]) -> Manifest:
    """A parsed manifest (JSON object) -> Manifest, or ManifestError naming the first problem."""
    if not isinstance(doc, Mapping):
        raise ManifestError("manifest: expected an object")
    version = _need(doc, "manifest_version", int, "")
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"manifest_version: {version} is not supported (this is {MANIFEST_VERSION})"
        )
    name = _need(doc, "name", str, "")
    if not _NAME.match(name):
        raise ManifestError("name: lowercase letters, digits, '.', '_' or '-', at most 64")
    kind = _need(doc, "kind", str, "")
    if kind not in KINDS:
        raise ManifestError(f"kind: one of {KINDS}")
    weights = _need(doc, "weights", dict, "")
    wfile = _need(weights, "file", str, "weights.")
    if not wfile or os.path.basename(wfile) != wfile:
        raise ManifestError("weights.file: a bare file name beside the manifest")
    sha = _need(weights, "sha256", str, "weights.")
    if not _SHA256.match(sha):
        raise ManifestError("weights.sha256: 64 lowercase hex digits")
    inputs = _tensors(doc.get("inputs"), "inputs")
    outputs = _tensors(doc.get("outputs"), "outputs")
    batch = _need(doc, "batch", dict, "")
    preferred = _need(batch, "preferred", int, "batch.")
    if not 1 <= preferred <= 8:
        raise ManifestError("batch.preferred: 1..8")
    decode = doc.get("decode", {})
    if not isinstance(decode, dict):
        raise ManifestError("decode: expected an object")
    if kind == "keypoint":
        off = decode.get("heatmap_cell_offset")
        if (
            not isinstance(off, (int, float))
            or isinstance(off, bool)
            or not math.isfinite(off)
            or off not in (0, 0.5)
        ):
            raise ManifestError("decode.heatmap_cell_offset: 0 or 0.5 (keypoint models must say)")
    source = doc.get("source", {})
    if not isinstance(source, dict):
        raise ManifestError("source: expected an object")
    if source:
        _need(source, "repo", str, "source.")
        rev = _need(source, "revision", str, "source.")
        if not re.match(r"^[0-9a-f]{40}$", rev):
            raise ManifestError("source.revision: a full 40-digit commit hash, not a branch")
    known = {
        "manifest_version",
        "name",
        "kind",
        "weights",
        "inputs",
        "outputs",
        "batch",
        "decode",
        "source",
        "description",
    }
    extra = sorted(set(doc) - known)
    if extra:
        raise ManifestError(f"{extra[0]}: unknown field")
    return Manifest(name, kind, wfile, sha, inputs, outputs, preferred, dict(decode), dict(source))


def load_manifest(path: str | os.PathLike) -> Manifest:
    """Read and validate a manifest file (model.json beside its weights)."""
    with open(path, encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest: not JSON ({exc.msg})") from None
    return validate_manifest(doc)


# --- Device plugin ABI ---------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    """What a device backend's loaded image supports, read from its image record."""

    abi: int
    image_sha256: str
    max_crops: int
    segments: int = 1  # requests per submission (infer_frame); 1 = no frames
    stream: bool = False  # segment 2 staged while segment 1 runs
    resident_slots: int = 0  # models whose parameters stay resident together
    features: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Refusal:
    """compile()'s answer for a graph or part of one the device will not run; framenpu runs
    that part on the host instead. ``reason`` is for logs and is never parsed."""

    reason: str


class Plan(Protocol):
    """An opaque compiled model for one backend; only that backend reads its contents."""

    model: str
    max_crops: int


class DeviceLease(Protocol):
    """Exclusive use of the device between open() and close()."""

    def run(self, requests: Sequence[tuple[Plan, Sequence[Any]]], during=None) -> list[list[Any]]:
        """Run staged requests [(plan, staged inputs per crop), ...] as one submission.
        ``during()``, if given, runs on the calling thread while the device works. Raises the
        backend's RecoveryRequired when the device state needs recovery; nothing is retried."""

    def idle(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class DevicePlugin(Protocol):
    """Registered under the ``framenpu.devices`` entry-point group."""

    abi: int
    name: str

    def capabilities(self) -> Capabilities: ...

    def compile(self, graph: Any, batch: int) -> Plan | Refusal: ...

    def open(self) -> DeviceLease: ...


def abi_compatible(abi: int, supported: range = range(PLUGIN_ABI, PLUGIN_ABI + 1)) -> bool:
    """Whether framenpu can load a plugin of this ABI."""
    return isinstance(abi, int) and not isinstance(abi, bool) and abi in supported
