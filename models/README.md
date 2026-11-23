# Model manifests

One folder per model, holding its `model.json` ([manifest v1](../docs/CONTRACTS.md#model-manifest)).
The weights are not in this repository. Put the ONNX file named in `weights.file` beside its
manifest; its SHA-256 must match `weights.sha256`.

| Manifest | Kind | Notes |
| --- | --- | --- |
| `mercury-ft7` | keypoint | The live keypoint student (decode cell offset 0) |
| `mercury-keypoint-jan18` | keypoint | Mercury's keypoint model, the teacher (Monado decode, cell offset 0.5) |
| `mercury-stereo3d-tpc1` | stereo3d | The two-view 3D joint student; the service returns `joints3d` and `scalar_extras` |
| `mercury-detector-160` | detector | Mercury's hand detector (left, right per output) |

Each model will get its own Hugging Face repo (ONNX, manifest, model card). The manifest's
`source` then records that repo and the commit it was fetched at.
