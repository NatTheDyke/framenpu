# Library contracts

The hand-tracking code splits into three libraries that depend in one direction:
**framehands** (tracking) uses **framenpu** (models, sessions, CPU fallback, the NPU service),
which loads **framenpu-cdsp** (the CDSP device backend) as a plugin. Nothing lower imports
anything above it. Four interfaces carry all the coupling; each is versioned on its own.

| Contract | Between | Version | Checkable form |
| --- | --- | --- | --- |
| Session API | framehands -> framenpu | framenpu semver | `framenpu.contracts.Session` |
| Model manifest | each model -> framenpu | `manifest_version` = 1 | `framenpu.contracts.validate_manifest` |
| Device plugin ABI | framenpu -> framenpu-cdsp | `abi` = 1 | `framenpu.contracts.DevicePlugin` |
| Service wire protocol | service client <-> `framenpu serve` | `version` = 4 | the service module |

`tests/test_boundaries.py` enforces the import direction. Its known-violation list is today's
coupling; it may only shrink.

## Session API

A session holds models opened together on one backend: CPU (onnxruntime), the device in
process, or the root NPU service. framehands sees named tensors in and named tensors out.
Crop geometry, normalisation, hint planes and heatmap decode stay in framehands; batching
rules, lanes and DSP layout stay below.

| Member | Meaning |
| --- | --- |
| `models` | Manifest names of the opened models |
| `max_crops(model)` | Crops one request may carry on this backend now (it can change after a fallback) |
| `infer(model, crops)` | One request of 1..`max_crops` crops, run now; outputs per crop, in order |
| `infer_frame([(model, crops), ...])` | A frame's requests of distinct models: one submission where the backend can, else back to back |
| `infer_many([(model, crops), ...])` | Several requests, models may repeat, in order: one round trip where the backend can (the service overlaps their host work) |
| `backend_of(model)` | Where the model runs now: the backend's name, or `"cpu"` (a service may not serve every model; a fallback moves all) |
| `last_timing` | The last call's timing record, or None |
| `idle()` | Release held clocks; keep plans and resident parameters |
| `close()` | Release everything; a session that needs recovery stays refused |

Rules carried over from the tracker today:

- A request carries the crops available now. It is never padded, and there is no batching timer.
- Any backend failure latches that session to the CPU for its lifetime; `RecoveryRequired` is
  never retried.
- Models are addressed by manifest name, not by number (the service's 0/1/2 is wire detail).
- A model the service does not serve runs on the CPU without moving the others.

Entry point: `framenpu.open(manifests, backend="auto"|"cpu"|"device"|"service", jobs=None,
socket=None, threads=1)` (`framenpu.session`). `manifests` are `model.json` files or the folders
holding them; the CPU and device backends check each weights file against its manifest. "auto"
is the service when its socket exists, else the CPU.

## Model manifest

`model.json` beside each ONNX file, in the model's Hugging Face repo. It replaces the copies of
the input and output specs now in `npuservice.py` (`SPECS`), `inference.py`
(`OUTPUTS`/`SHAPES`) and `mercury.py` (`INPUTS`/`OUTPUTS`/`SIGNATURES`).

```json
{
  "manifest_version": 1,
  "name": "mercury-ft7",
  "kind": "keypoint",
  "description": "Mercury keypoint student, grayscale, 128x128",
  "weights": {"file": "model.onnx", "sha256": "<64 hex>"},
  "inputs": [
    {"name": "inputImg", "shape": [1, 1, 128, 128]},
    {"name": "lastKeypoints", "shape": [1, 42]},
    {"name": "useLastKeypoints", "shape": [1]}
  ],
  "outputs": [
    {"name": "heatmap_xy", "shape": [1, 21, 22, 22]},
    {"name": "heatmap_depth", "shape": [1, 21, 22]},
    {"name": "scalar_extras", "shape": [1, 8]},
    {"name": "curls", "shape": [1, 10]}
  ],
  "batch": {"preferred": 3},
  "decode": {"heatmap_cell_offset": 0},
  "source": {"repo": "<hf user>/<repo>", "revision": "<40 hex commit>"}
}
```

| Field | Rule |
| --- | --- |
| `manifest_version` | 1; readers refuse other versions |
| `name` | Lowercase, `[a-z0-9._-]`, at most 64; how sessions address the model |
| `kind` | `keypoint`, `stereo3d` or `detector` |
| `weights` | A bare file name beside the manifest, and its SHA-256 |
| `inputs`, `outputs` | Name, shape (leading dimension 1: one crop), dtype `float32` |
| `batch.preferred` | Crops per request the model was tuned for; the backend's `max_crops` still caps it |
| `decode.heatmap_cell_offset` | Keypoint models must state 0 or 0.5 (the Monado j+0.5 vs student j+0 mismatch becomes a field) |
| `source` | Optional; Hugging Face repo and the full commit hash it was fetched at, never a branch |

Unknown fields are refused, so a typo cannot silently mean "default".

## Device plugin ABI

framenpu finds backends through the `framenpu.devices` entry-point group, so a PC or CI install
runs with the CPU backend alone.

| Member | Meaning |
| --- | --- |
| `abi`, `name` | Plugin ABI version (1) and backend name; framenpu loads only ABIs it lists |
| `capabilities()` | From the loaded image's record: `max_crops`, `segments`, `stream`, `resident_slots`, `features`, `image_sha256` |
| `compile(graph, batch)` | A `Plan`, or a `Refusal(reason)`; framenpu runs a refused part on the host (as the hint prefix and pooled heads run today) |
| `open()` | A lease with `run(requests, during=)`, `idle()`, `close()` |

The backend keeps a pinned table of image hashes it accepts, as now, but code branches on
capability fields, not on which hash matched. That retires the `*_IMAGE_SHA256` membership
tests in `mercury_runtime.py`. The backend owns its error types, including `RecoveryRequired`;
framenpu re-exports them.

## Service wire protocol

Unchanged from version 4 (documented at the top of the service module): a u32 header length,
a JSON header, raw float32 payloads of validated shapes; no pickle. It moves into framenpu as
`framenpu serve` with its client, so both ends ship in one package. From version 5, `ping`
reports each served model's manifest and crop limit, and client and server accept one version
of skew in either direction, because the root service and the user tracker install separately.
