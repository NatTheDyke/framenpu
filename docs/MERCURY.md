# Mercury keypoint model: lowering, FP16 contract and forecast

This is step 1 of moving from the FrameTop teacher to Monado's Mercury
keypoint model, which is the per-hand model used by
[steam-frame-hand-tracking](https://github.com/spyjacob/steam-frame-hand-tracking).
That stack already runs Mercury on the Frame CPU through ONNX Runtime. Its
report names CPU heat as the binding limit and asks for 4–8x more Mercury
throughput.

Step 1 (lowering, contract, forecast) is **offline**: no DSP access, no device
run, no kernel change. Step 2a, [below](#step-2a-backbone-on-the-dsp), runs the
backbone on the DSP. Two step-1 results:

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
- **HVX depthwise:** FP32 tap accumulation, then FP16 rounding. (The resident
  kernel actually multiplies and accumulates in HVX qf16; see step 2b. The
  emulator keeps FP32 accumulation as its reference model.)
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

## Step 2a: backbone on the DSP

Step 2a runs `Conv_19 .. Conv_139` on the DSP: 36 operations, 20 pointwise on
HMX and 16 depthwise on HVX (2 at stride 2), from 32x32x32 down to 160x8x8. They run as **one
resident request** through the sealed linked image. The 64x64 prefix
(`Conv_0 .. Conv_18`) runs in FP32 on the host, and so do the hint branch,
pooling and heads. Evidence: [device run](../evidence/mercury-backbone-device.json),
[early-layer pilot](../evidence/mercury-backbone-pilot.json).

Kernel and plan:

- **Even-phase stride 2.** `tools/build_teacher_link.py --stride2 --s2-even`
  builds a new image (`d0427fb0…`, 5,488 B) whose stride-2 gather starts at
  stride-1 word (0, 0) instead of (1, 1). The default build and every
  qualified teacher image are byte-identical to before (stride-2 image
  `ef5721f8…`, loader `72280ad9`).
- **Free concat.** `Conv_56` writes its 64 channels into the first 32 channel
  pairs of the 96-channel tensor. The host-computed hint plane (Gemm, reshape,
  scale by `useLastKeypoints`, FP16) is staged right after it in the window
  and loaded into the last 16 pairs, so nothing is copied.
- **Plan** (`tests/mercury_layout.py`): 39 descriptors (input load, 36 ops,
  hint load, final store), peak VTCM 10,911 of 15,296 vectors, window
  `0x1a40000..0x1b65800`. Qualification splits into two per-op segments at
  the concat, and the device's own bytes are carried across the split.

Offline, the lane-level simulator of the kernel addressing (now with a
decimation-phase parameter) reproduces the FP16 contract bit-exactly on all
24 checked crops. With the teacher's odd phase it does not. The harness itself
is tested end to end against that simulator (`tests/test_mercury_layout.py`).

Device run (`tests/hardware_mercury.py`): 16 private cases, which are the 4
corpus frames x 2 orientations x {no hint, self-hint}. Each case is qualified
per op, then gets 5 merged frames. This was preceded by a pilot through
`Conv_33`.

| Result | Value |
|---|---:|
| Worst per-op ratio to the FP16 bound (from the device's own inputs) | 0.73 (`Conv_42`) |
| Merged frames bit-identical to per-op qualification | 16 / 16 cases |
| DSP request body, median | **6.58 ms** (forecast 8.59 ms) |
| of which pointwise / depthwise / stride-2 depthwise / load-store | 3.96 / 1.36 / 1.09 / 0.16 ms |
| Host wall per request (stage, submit, read back), median | 13.9 ms |
| Host FP32 pooling and heads (NumPy, on the Frame), median | 2.8 ms |
| Keypoints vs FP32, worst joint: median / p95 / max | 0.147 / 0.486 / 0.491 px |
| `is_hand` acceptance flips at 0.97; curls max drift | 0; 0.0159 |
| Controls: guarded fault, descriptor rejection, 12 stride-2 field rejections, second seal, unsealed request | all refused natively; clean frame after |
| Restoration: window, reservation, whole-arena hash, service | unchanged (677 audits) |

The device is **not bit-identical to the host FP16 emulator** (0 of 16). The
lane simulator is, because it uses the same arithmetic model. The device and
emulator keypoints differ by at most 0.35 px. Both drift the same amount from
FP32: a median worst joint of 0.147 px on the device against 0.159 px for the
emulator. So the hardware rounding differs from the contract model in detail
but stays well inside the bound. The acceptance criterion is the per-op bound
check, as it was for the teacher. Step 2b found the cause: the depthwise
kernel accumulates in qf16, not FP32.

The 2a report pins the harness and planner as they were before step 2b. The
current planner emits a byte-identical backbone plan (descriptors, parameter
images and window layout, checked), and only the depthwise bound has
changed since.

The most expensive ops are the 32x32 stage: `Conv_33` (stride 2, 0.94 ms) and
the two 32→192 expansions `Conv_19` and `Conv_29` (0.75 ms each). They make up
37% of the request.

## Step 2b: the whole convolution network on the DSP

Step 2b moves the 64x64 prefix onto the DSP, so `Conv_0 … Conv_139` (52
resident ops) runs as **one request** of the same qualified image
(`d0427fb0…`). There is no kernel change. The host now only im2cols the FP16
image for the stem, computes the hint plane, and runs pooling and the heads.
Evidence: [device run](../evidence/mercury-network-device.json),
[pilot](../evidence/mercury-network-pilot.json),
[first pilot, stopped](../evidence/mercury-network-pilot-flush-failure.json).

The prefix needs three rewrites (`tests/mercury_layout.py`, `prefix=True`):

- **Stem as a pointwise.** The 1-channel 3x3 stride-2 stem becomes a K=10 HMX
  pointwise over host im2col columns (9 taps plus a zero plane, 64x64). The
  host does data movement only. The columns are 93 KB per crop against 32 KB
  for the image. Its arithmetic is identical to the dense stem (tested).
- **Banding by channels, not rows.** The 192x64x64 expansion (`Conv_10`)
  would need 13,728 vectors of the 15,296-vector VTCM span. A depthwise
  layer is per-channel, so `Conv_10` and the stride-2 depthwise `Conv_14`
  run as six 32-channel groups. Each group's output is a view into its slice
  of the 192x32x32 `Conv_14` tensor, the same free-composite mechanism as
  the hint concat. There are no halos and no recompute.
- **Plan:** 55 descriptors, peak VTCM 11,289 vectors, window up to
  `0x1bb8800`. Qualification runs in 4 segments, split at the two
  composites.

**The first pilot stopped on a bound failure.** `Conv_14.g1` reached 1.26x
the allowance, at 1 value out of 32,768 (evidence above). A private dump of
that op showed the device returned exactly the FP16 bias where the exact
result added one tap product of 2.07e-5. That product is below FP16's normal
range, and the kernel's depthwise multiplies and accumulates in HVX **qf16**,
which flushes such values to zero. The FP32-accumulation model behind the
bound didn't cover this.

The depthwise bound now adds an absolute 18·2^-14 ≈ 1.1e-3 (9 products plus
9 partial sums, each losing at most 2^-14). That is about one FP16 ulp at
1.0, against activations in [0, 6]; the relative terms are unchanged. A
regression test rebuilds the observed case and fails with the old floor.
This also explains why the device was never bit-identical to the emulator
in 2a. After the fix, the same pilot and then the full run passed.

Device run: the same 16 private cases, compared against the **full** FP16
contract with no FP32 host prefix.

| Result | Value |
|---|---:|
| Worst per-op ratio to the FP16 bound | 0.39 (`Conv_52`) |
| Merged frames bit-identical to per-op qualification | 16 / 16 cases |
| DSP request body, median | **15.14 ms** (20 ms native budget) |
| of which `Conv_10` groups / `Conv_14` groups / rest of prefix / backbone / load-store | 3.48 / 3.59 / 1.50 / 6.41 / 0.17 ms |
| Host wall per request (columns, hint, submit, read back), median | 22.7 ms |
| Host FP32 pooling and heads (NumPy, on the Frame), median | 2.8 ms |
| Keypoints vs FP32, worst joint: median / p95 / max | 0.185 / 0.396 / 0.669 px |
| `is_hand` flips at 0.97; curls max drift | 0; 0.0156 |
| Controls: guarded fault, descriptor rejection, 48 stride-2 field rejections, second seal, unsealed request | all refused natively; clean frame after |
| Restoration: window, reservation, whole-arena hash, service | unchanged (969 audits) |

All the convolutions of the model now run on the DSP at about **66 crops/s**
if DSP-bound. That is slower per crop than the stack's hot CPU path
(11.3 ms), but takes the convolutions off the CPU. The expansion and
stride-2 pair at 64² is 47% of the request. The kernel's stride-2 path
computes the full stride-1 result and then decimates, so a direct stride-2
depthwise would remove about three quarters of `Conv_14`. The 7.5 ms of host
overhead per request is the other large item.

## Direct stride-2 depthwise

The old stride-2 path computed the full stride-1 depthwise into a temporary
grid, then copied every kept word one at a time with scalar code. A new
build option, `tools/build_teacher_link.py --stride2 --s2-even --s2-direct`
(image `58c5ec58…`, 5,904 B, define `RS_S2_DIRECT`), computes only the kept
outputs instead. For each output row and 32-column block:

1. It deals the three input rows into even and odd columns with `vdeal`
   (two unaligned load pairs per row).
2. It accumulates the same 9 qf16 products in the same (dy, dx) order as the
   stride-1 kernel, then applies the same bias add and clip.
3. It stores the vector in one unaligned write, with lanes past the row
   zeroed.

Rows are written in increasing order, so the zero spill only lands on pad
words or on words a later row rewrites. Two extra native checks bound the
reads and the spill. The descriptor format is unchanged, so the same plans
run on both images. One compiler hazard came up: LLVM fused `hf(acc)` with
the following hf+hf bias add into a single qf16+hf add, which skips a
rounding step. The kernel forces the FP16 rounding through a volatile VTCM
slot, as the stride-1 path effectively does, and the disassembly confirms
the ops are separate. Every existing image is byte-identical (`d0427fb0…`,
`ef5721f8…`).

Offline, a word-level model of the direct kernel's addressing is added to
the lane simulator (`simulate(..., s2_direct=True)`). It reproduces the
stride-1-plus-decimation result bit-exactly on every Mercury geometry
(64→32, 32→16, 16→8), and it raises on any access outside the native bounds.
A test checks that the direct build rejects an output plane too short for
the spill.

Device ([direct](../evidence/mercury-s2-direct-device.json),
[baseline rerun](../evidence/mercury-s2-direct-baseline.json),
[first-op pilot](../evidence/mercury-s2-direct-pilot.json)): the whole
network (`--prefix`) ran with the same 16 cases × 5 rounds on both images.

| | Stride-1 + decimate (`d0427fb0`) | Direct (`58c5ec58`) |
|---|---:|---:|
| `Conv_14`, per 32-channel group (64→32) | 598 µs | **72 µs** (8.3x) |
| `Conv_33` (32→16, 192 ch) | 936 µs | 189 µs (5.0x) |
| `Conv_118` (16→8, 576 ch) | 150 µs | 60 µs (2.5x) |
| All 8 stride-2 ops | 4.67 ms | 0.68 ms |
| All other ops | 10.48 ms | 10.48 ms |
| **DSP request body, median** | 15.14 ms | **11.15 ms** |
| Host wall per request, median | 22.5 ms | 18.5 ms |
| Final tensor vs the other image | | **bit-identical, 16/16 cases** |

Everything else is unchanged on the direct image:

- every op within the FP16 bound (worst 0.39), and merged frames identical to
  per-op qualification;
- the same keypoint drift (0.185 px median / 0.669 px max vs FP32);
- all controls refused natively (48 stride-2 field rejections);
- 969 restoration audits, whole arena unchanged.

DSP-bound throughput rises from 66 to about 90 crops/s.

The largest remaining items are the HMX pointwise layers: the six `Conv_10`
groups alone are 3.48 ms. Next is the host overhead of about 7.4 ms per
request.

## Next steps

- **Pointwise cost.** HMX 1x1 layers are now 8.77 of the 11.15 ms (depthwise
  1.54, stride 2 0.68, load/store 0.17). The
  per-row post-processing (bias, clip, mask, store per output vector)
  dominates, not the MACs (see the forecast fit).
- **Step 2c: heatmap heads on the DSP** (the two dense 3x3 convolutions and
  the 2x2 transposed convolution). Then batch 2 crops per request and time
  the stack's seam (`MercuryKeypoint.run`) end to end.
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
# step 2a: offline plan/simulator/harness tests, then the device run (root on the Frame)
FNPU_MERCURY_MODEL=/tmp/mercury/grayscale_keypoint_jan18.onnx python -m unittest tests.test_mercury_layout
python tools/build_teacher_link.py --stride2 --s2-even --output /tmp/mercury/jobs
python tools/mercury_device_cases.py --model /tmp/mercury/grayscale_keypoint_jan18.onnx \
  --corpus /tmp/framenpu-convolution/corpus --output /tmp/mercury/cases   # private, never commit
sudo python tests/hardware_mercury.py --run-on-device --jobs /tmp/mercury/jobs \
  --model /tmp/mercury/grayscale_keypoint_jan18.onnx --cases /tmp/mercury/cases \
  --output result.json --rounds 5 [--stop-after Conv_33 --max-cases 1]
# step 2b: the same, plus --prefix (pilot: --prefix --stop-after Conv_18 --max-cases 1)
# direct stride 2: build with --stride2 --s2-even --s2-direct; compare final_sha256 per case
~~~
