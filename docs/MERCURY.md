# Mercury keypoint model: lowering, FP16 contract and forecast

This is step 1 of moving from the FrameTop teacher to Monado's Mercury
keypoint model, which is the per-hand model used by
[steam-frame-hand-tracking](https://github.com/spyjacob/steam-frame-hand-tracking).
That stack already runs Mercury on the Frame CPU through ONNX Runtime. Its
report names CPU heat as the binding limit and asks for 4–8x more Mercury
throughput.

Everything here is **offline**. There was no DSP access, no device run and no
kernel change. Two results follow:

- The model lowers onto the operator family framenpu already runs. FP16 on
  HMX and HVX is numerically adequate.
- A forecast based on the teacher's measured per-op costs says a first port
  costs about **19 ms of DSP time per crop**, with the heads on the host. That
  is slower per crop than the CPU (11.3 ms hot), so the early benefit is CPU
  offload, not lower latency.

Evidence: [reference](../evidence/mercury-keypoint-reference.json) and
[forecast](../evidence/mercury-keypoint-forecast.json). Both contain only
statistics.

## Model and license

| | |
|---|---|
| File | `grayscale_keypoint_jan18.onnx`, SHA256 `40c0daa5…d1d5` |
| Source | gitlab.freedesktop.org/monado/utilities/hand-tracking-models @ `37a8a81b` |
| License | **None published.** The model repository has no license file (checked 2026-11-17). The weights are fetched by hash, never bundled or committed, and evidence holds statistics only. |
| Inputs | `inputImg` 1x1x128x128, `lastKeypoints` 1x42 (temporal hint), `useLastKeypoints` 1 |
| Outputs | `heatmap_xy` 21x22x22, `heatmap_depth` 21x22, `scalar_extras` 8, `curls` 10 |
| Size | 986,168 parameters, 122.9 M conv MACs (teacher: 145.6 M at 224²) |

Mercury's tracker code is BSL-1.0 (Monado). The decoder in
`framenpu.mercury.decode` follows the conventions of `hg_model.cpp`, as
ported in that stack's `mercury_kp.py`.

## Lowering

`framenpu.onnx_graph` reads ONNX protobuf without dependencies. On both
Mercury models it matches the `onnx` library exactly on every node, attribute
and initializer.

`framenpu.mercury.lower` pattern-matches the graph into 60 operations and
refuses anything else:

| Lowered op | Count | Unit | Notes |
|---|---:|---|---|
| Pointwise 1x1 | 23 | HMX | 12 residual adds fused into the projection |
| Depthwise 3x3 | 18 | HVX | 4 at stride 2 |
| Dense 3x3 | 3 | HMX (im2col) | 1-channel stem with K=9; two heatmap heads with padding 2 |
| ConvTranspose 2x2 s2 | 1 | HMX | Lowered as pointwise to 4·cout channels, then a pixel shuffle |
| Gemm (+ folded BN) | 5 | FP32 | 42→8192 hint branch; curl and extras heads |
| 1x1 conv on pooled map | 1 | FP32 | 160→462 depth head |
| AveragePool | 3 | FP32 | See the quirk below |
| Layout (reshape, scale, concat) | 6 | — | Hint plane scaled by `useLastKeypoints`, then concatenated |

Two semantics differ from the teacher and are covered by tests:

- **Stride-2 depthwise uses symmetric PyTorch padding (1,1).** It keeps the
  *even* output phase of the stride-1 result. The teacher's TF-SAME padding
  kept the odd phase, so the decimation phase becomes a kernel parameter.
- **The heads pool an 8x8 map with kernel 6/stride 6 and kernel 3/stride 3,
  in floor mode.** Both read only rows and columns 0–5; the last two are
  ignored. A device implementation has to reproduce this exactly.

The FP32 evaluator, with FP64 accumulation, matches ONNX Runtime 1.32 FP32
to at most 7.6e-6 absolute on every output. That covers both hint modes on
64 real crops and on synthetic inputs.

## FP16 contract

The contract follows the qualified teacher tail:

- **HMX:** FP16 inputs and weights, FP16-rounded accumulator, then FP16 bias,
  clip and residual.
- **HVX depthwise:** FP32 tap accumulation, then FP16 rounding.
- **FP32 tail:** pooling and heads run in FP32 on FP16 activations.

**Corpus:** 4 recorded fisheye frames from the existing private corpus. Each
is cropped to 128² in 8 orientations and normalized the way Mercury does. Each
crop runs with no hint and then with the oracle's own keypoints as the hint,
giving 64 evaluations; 63 have oracle `is_hand` ≥ 0.97. These are real hands,
but they are not Mercury's stereographic tracking crops. The heatmaps are
diffuse (median peak 0.34), so this measures numerical drift, not tracking
accuracy.

| FP16 contract vs ONNX Runtime FP32, accepted views | Median | p95 | Max |
|---|---:|---:|---:|
| Keypoint error, worst joint (128 px crop) | 0.159 px | 0.457 px | 1.384 px |
| Keypoint error, mean over joints | 0.059 px | 0.136 px | 0.202 px |
| Worst joint among joints with heatmap peak ≥ 0.3 | 0.100 px | 0.225 px | 1.384 px |
| Relative depth (hand-size units) | 0.0030 | 0.0059 | 0.0101 |
| `is_hand` | 6e-6 | 8e-5 | 2e-4 |
| Curls | 0.0095 | 0.0162 | 0.0174 |
| Acceptance flips at 0.97 | | | 0 |

The control moves the crop by one source pixel, which is under half a crop
pixel. In FP32 that changes keypoints by 4.5 px mean (median). FP16 drift is
about 75x smaller than the model's own sensitivity. The largest errors occur
on joints with ambiguous heatmaps, where the argmax can move between peaks.

The worst intermediate tensors relative to their range are:

- the FP32 curl and extras Gemm input chain, at 3.4% (inherited from FP16
  pooling input);
- the final heatmap convolution, at 1.8%;
- the 64² expansion, at 1.6%.

## Forecast (not measured)

The forecast fits cost models to the teacher tail's measured per-op compute
(28 pointwise and 14 depthwise jobs) using non-negative least squares:

- **Pointwise** cost tracks output values, not MACs, because the per-row
  post-processing dominates.
- **Depthwise** cost tracks taps x values. A stride-2 layer pays for a full
  stride-1 pass plus a gather.

Both fits reproduce the teacher totals within 1.3%, but individual small ops
can be off by up to about 117%. Treat only the totals as meaningful.

The compute sum is scaled by the teacher's request/compute ratio (1.53). FP32
work is costed at the teacher's measured FP32-head rate of 14 ns per weight
value, which is mostly parameter copy. Host overhead is 9.4 ms per request:
submission time beyond DSP plus staging, from the single-request teacher
path.

| Layout | DSP ms/crop | Host wall ms/crop, 1 crop/request | 2 crops/request | DSP-bound crops/s |
|---|---:|---:|---:|---:|
| Everything on DSP, hint on host | 24.3 (over the 20 ms budget) | 33.7 | 29.0 | 41 |
| Convolutions on DSP; heads and hint on host | **19.0** | 28.4 | 23.7 | 53 |
| Same, plus a direct stride-2 depthwise (optimistic) | 13.6 | 23.0 | 18.3 | 73 |

For comparison, the stack's CPU ONNX Runtime path takes 11.3 ms per crop
hot. At its normal low CPU priority it takes 16–30 ms. It currently handles
about 24 crops/s in total.

What the forecast says to build:

1. **Run the 42→8192 hint Gemm and the FP32 heads on the host.** On the DSP
   in FP32 they would cost 5.1 ms and 3.4 ms, dominated by moving 1.7 MB of
   FP32 weights per crop. On the host they total 0.58 M MACs, which is small
   next to the 11–30 ms the stack spends per crop today. The DSP returns the
   heatmap and the pooled 160-channel features.
2. **Fuse the 64² expansion with the stride-2 depthwise, in bands.**
   `Conv_10` writes 192x64x64 in FP16 (1.5 MiB, 12,288 vectors); together with
   the following depthwise output this exceeds the 15,296-vector VTCM span.
   `Conv_14` (stride 2) is also the single most expensive op at 2.9 ms. This
   is where the teacher's banded-prefix machinery applies.
3. **Batch two crops per request,** for both hands or both cameras. This
   halves the per-request host overhead and the weight streaming. It is also
   the stack's own top request: true stereo on every update.

The ceiling is real. A single HMX owner at about 14–19 ms per crop gives
roughly 50–70 crops/s. That would put both hands at 25–35 Hz with Mercury
fully off the CPU. Going higher needs per-op efficiency the teacher work has
not yet shown.

## Next steps

- **Step 2: Mercury on the DSP.** Build banded 64² and resident 32²→8²
  stages plus the heatmap-head convolutions, using the qualified HMX/HVX
  kernels. Qualify on the decoded keypoints against this reference, then time
  1 and 2 crops per request.
- **Corpus.** Collect real Mercury tracking crops on the headset, using the
  stack's local-only `validate_mercury.py`. Do this before freezing accuracy
  ceilings; the affine crops here are only a numerical stand-in.
- **Detector.** `grayscale_detection_160x160.onnx` (~59 M MACs, SHA256
  `1f1a039a…ed36`) uses the same op family. It is the candidate to replace
  the MediaPipe discovery stage, the stack's most expensive CPU stage.
- **Kinematic fit (later track).** Monado's Levenberg–Marquardt hand fit is
  small FP32 linear algebra (on the order of 26 DOF), with branchy scalar
  control. That suits HVX/scalar code rather than HMX. Measure its CPU cost on
  the Frame first, so that a port is only made if it pays off.

## Reproduce

~~~sh
# local, outside the repo; weights are not redistributed
curl -sSLo /tmp/mercury/grayscale_keypoint_jan18.onnx \
  https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models/-/raw/37a8a81b/grayscale_keypoint_jan18.onnx
FNPU_MERCURY_MODEL=/tmp/mercury/grayscale_keypoint_jan18.onnx python -m unittest tests.test_mercury
# needs numpy + onnxruntime and the private convolution corpus
python tools/mercury_reference.py --model /tmp/mercury/grayscale_keypoint_jan18.onnx \
  --corpus /tmp/framenpu-convolution/corpus --output evidence/mercury-keypoint-reference.json
python tools/mercury_forecast.py --model /tmp/mercury/grayscale_keypoint_jan18.onnx \
  --teacher-model /tmp/framenpu-convolution/model --output evidence/mercury-keypoint-forecast.json
~~~
