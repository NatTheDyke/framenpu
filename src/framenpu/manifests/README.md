# Model manifests

One manifest per model ([manifest v1](../../../docs/CONTRACTS.md#model-manifest)), shipped with framenpu.
The weights are not in this repository. `framenpu.open` takes a weights file and finds its
manifest here by the file's SHA-256 (or uses a `model.json` beside the file).

| Manifest | Kind | Notes |
| --- | --- | --- |
| `mercury-ft7` | keypoint | The live keypoint student (decode cell offset 0) |
| `mercury-keypoint-jan18` | keypoint | Mercury's keypoint model, the teacher (Monado decode, cell offset 0.5) |
| `mercury-stereo3d-tpc1` | stereo3d | The two-view 3D joint student; the service returns `joints3d` and `scalar_extras` |
| `mercury-detector-160` | detector | Mercury's hand detector (left, right per output) |

Where the weights come from (`python3 tools/fetch_models.py DIR` fetches them all by hash):

| Manifest | Weights |
| --- | --- |
| `mercury-ft7` | [natthedyke1/mercury-ft7](https://huggingface.co/natthedyke1/mercury-ft7), CC BY-NC-SA 4.0 (`source` records the commit) |
| `mercury-stereo3d-tpc1` | [natthedyke1/mercury-stereo3d-tpc1](https://huggingface.co/natthedyke1/mercury-stereo3d-tpc1), CC BY-NC-SA 4.0 |
| `mercury-keypoint-jan18`, `mercury-detector-160` | Monado's [hand-tracking-models](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models) at `37a8a81b`; no published license, so not re-hosted |

The `model.json` in each Hugging Face repo is this manifest without `source` (a repo cannot hold
its own commit hash).
