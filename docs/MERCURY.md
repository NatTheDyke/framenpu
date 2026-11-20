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

## Fast 1x1 post

After the direct stride-2 kernel, the HMX 1x1 layers were 8.77 of the 11.15 ms.
A timing-only build, `--pw-profile`, adds tick totals for each phase of the
1x1 loop: parameter copy, output zeroing, B gather, HMX, post, and barriers
inserted at chosen points. Arithmetic and outputs are unchanged. The first
profile showed that the MACs were not the problem:

- With the original post loop
  ([profile](../evidence/mercury-pw-profile-before.json)), HMX work is about
  0.2 µs per 16-row tile. A barrier placed right after a tile's post loop
  waits about 2.8 µs, though (5.70 ms over the request). That wait is roughly
  fixed per barrier, not per store.
- A barrier right after the HMX store costs nothing. Without the extra
  barriers, the same wait lands on the next barrier or the next HMX op
  instead. The gather's stores drain at once.
- The post loop does a read-modify-write of the overflow-flag slot
  `RS_FLAGS` (`LV(62)`) for every output vector. That slot is in the same
  8 KiB scratch block as the HMX output tile `LV(0)`. Why these stores drain
  so slowly is not established; the measured fact is that removing the
  per-vector update removes most of the wait.

The new build option `--pw-fast` (`RS_PW_FAST`, image `e7b06441…`, 6,128 B)
replaces the 1x1 post with `rs_pw_post`, which handles one tile per call. It
runs the same operations as `rs_post` in the same order, hoists the descriptor
fields, and keeps the flags in a register, so `RS_FLAGS` is updated once per
tile. The disassembly confirms that both FP16 roundings, after the bias and
after the residual, stay separate ops. A per-op flag update would make the
compiler spill the vector to the stack, which the link check rejects. All
earlier images are byte-identical and frozen in `tests/test_teacher_link.py`.
The depthwise and stem paths still use `rs_post`.

With the fast post ([profile](../evidence/mercury-pw-profile-after.json)), the
post-loop wait falls from 5.70 to 1.36 ms, and the post itself from 1.32 to
0.90 ms. The remaining wait (0.67 µs per tile) is still the largest single
item. Whether the one per-tile `RS_FLAGS` update causes it is untested.

Device ([fast post](../evidence/mercury-pw-fast-device.json), after a
[pilot through `Conv_18`](../evidence/mercury-pw-fast-pilot.json)): the same
16 cases × 5 rounds as the direct stride-2 run.

| | Direct stride 2 (`58c5ec58`) | + fast 1x1 post (`e7b06441`) |
|---|---:|---:|
| `Conv_0` / `Conv_10` group (64x64, 16→16) | 579 µs | 226 µs |
| `Conv_19` / `Conv_29` (32x32, 16→96) | 748 µs | 251 µs |
| All 1x1 ops | 8.77 ms | 3.64 ms |
| Depthwise / stride 2 / load+store | 1.54 / 0.68 / 0.17 ms | unchanged |
| **DSP request body, median** | 11.15 ms | **6.03 ms** |
| Host wall per request, median | 18.5 ms | 13.5 ms |
| Final tensor vs the other image | | **bit-identical, 16/16 cases** |

Bounds (worst 0.394), keypoint drift, controls, the 969 restoration audits and
the arena hash are all unchanged. DSP-bound throughput is about 165 crops/s.

## Host time: held HMX power and 2-crop batches

With the DSP at 6.03 ms, host overhead dominated a request: 13.5 ms host
wall per crop. Each merged frame now records its host phases: staging,
readback, and the `device.py` submit marks. Every request used to power HMX
on and off around itself. `power.start()` rehashes the Qualcomm image,
uploads the sequencer program, locks the PLL and runs the power-on sequence;
`close()` runs the power-off sequence, with a 1 ms sleep, and verifies the
restored registers.

Two harness options address this:

- `--held-power` uses the existing session-held owner
  (`Device._hold_power`, as the FrameTop runtime does). HMX is powered once
  after the seal. Each request proves the held configuration read-only under
  the gate, and the session releases it under the gate before the retire.
  After a `RecoveryRequired`, it only unmaps and never gates clocks.
- `--batch 2` runs two crops in one request
  (`build_plan(..., batch=2)`): the merged sequence twice, 110 of 160
  descriptors. The parameter images are shared, and each crop has its own
  input and final-tensor window slots (about 270 KB more window). VTCM is
  reused from crop to crop; the peak is unchanged, at 11,289 vectors. A
  simulator test runs two different inputs and checks that each crop equals
  its own batch-1 result. With different inputs, crop 1 can't pass by reading
  crop 0's leftover VTCM. On the device, batch frame *i* runs cases *i* and
  *i*+1, and each crop's final must equal its own case's qualify result.

Host timing on the Frame is noisy: SteamVR is running and the governor is
schedutil. Single frames vary by up to 5 ms, and one full run had every host
phase 5–10x slower. So the timing comparison comes from short alternating
runs (4 cases × 20 rounds, one with per-request power, then three pairs of
batch 1 / batch 2 with held power,
[percall](../evidence/mercury-host-ab-percall.json),
[b1](../evidence/mercury-host-ab-b1-r1.json)/[b2](../evidence/mercury-host-ab-b2-r1.json)
r1–r3):

| Per crop | Per-request power | Held, batch 1 | Held, batch 2 |
|---|---:|---:|---:|
| **Host wall, median** | 14.06 ms | 8.87 ms | **7.80 ms** |
| Host wall, min / p90 | 13.10 / 18.30 | 7.90 / 13.42 | 7.12 / 9.78 |
| DSP body | 6.03 | 6.03 | 6.03 |
| HMX power on + off (verify when held) | 4.39 + 1.49 | 0.16 | 0.08 |
| Dispatch: publish-to-completion minus DSP wrapper time | 0.9 | 0.9 | 0.46 |
| Gate claim + acknowledge (incl. harness audit) | 0.65 | 0.78 | 0.37 |
| Staging (im2col + grid packing) | 0.31 | 0.59 | 0.54 |

The full runs carry the correctness evidence
([held, batch 1](../evidence/mercury-held-power-device.json),
[batch 2](../evidence/mercury-batch2-device.json), after a
[batch-2 pilot](../evidence/mercury-batch-pilot.json)). They use the same 16
cases × 5 rounds on image `e7b06441`. Final tensors are bit-identical on
16/16 cases to the per-request-power run, with the same bounds, keypoint
drift and controls (96 stride-2 rejections at batch 2). Restoration audits
pass (969 and 1,017), the arena hash is unchanged, and the DSP service is
unchanged.

Held power keeps the HMX clocks on for the life of the session, so a
runtime must release them when tracking goes idle. A 2-crop request takes
12.1 ms on the DSP and about 15.6 ms of host wall for both hands; two
batch-1 requests take about 17.7 ms.

## Fast depthwise

A batch of 4 crops needs at most 5 ms per crop to fit the 20 ms native
budget (and more than the 160-descriptor cap allows: 4 × 55 = 220). So
this round went after DSP time.

`--dw-fast` (`RS_DW_FAST`, image `89511c18…`, 6,416 B) rewrites both
depthwise kernels with the same products, accumulation order and post:

- the 3x3 taps are fully unrolled; the generic loop stays for other kernel
  sizes;
- the stride-1 post (bias, clip, flags, mask) is inlined instead of calling
  `rs_post` per vector;
- the overflow flags stay in a register for the whole op, so `RS_FLAGS` is
  updated once per op instead of read-modified-written per vector;
- the FP16 rounding before the bias stays its own step through an empty
  inline-asm register fence (`RS_FENCE`). This replaces the volatile VTCM
  round trip in the direct stride-2 kernel. The disassembly has no fused
  `qf16 + hf` add and no vector stack access.

Device ([full run](../evidence/mercury-dw-fast-device.json), after a
[pilot through `Conv_18`](../evidence/mercury-dw-fast-pilot.json); held
power, 16 cases × 5 rounds):

| | Fast 1x1 post (`e7b06441`) | + fast depthwise (`89511c18`) |
|---|---:|---:|
| Depthwise, 15 ops (largest: `Conv_23` 296 → 164 µs) | 1.54 ms | 0.98 ms |
| Stride 2, 8 ops | 0.68 ms | 0.59 ms |
| 1x1, 29 ops | 3.63 ms | 3.55 ms |
| **DSP request body, median** | 6.02 ms | **5.30 ms** |
| Final tensor vs the other image | | **bit-identical, 16/16 cases** |

Bounds (worst 0.394), keypoint drift, controls, the 969 restoration audits and
the arena hash are all unchanged.

Negative results from the same round. These were instrument runs; the code
was reverted, and their reports are not kept as evidence.

- **Copies are a floor.** Issuing 8 DDR loads before their stores in
  `rs_copy` made the 1x1 parameter copy slower (723 → 851 µs). DDR vector
  loads serialize at about 0.1 µs each, so the 1.16 ms per crop of parameter
  and activation copies can't be overlapped this way. Sharing parameters
  across a batch only fits VTCM at the 8x8 stage (about 0.13 ms per crop).
- **The 1x1 post-loop wait is not the flags slot.** Moving `RS_FLAGS`
  outside the `LV` block left the wait unchanged (1.36 → 1.32 ms). On the
  first attempt, the kernel rejected a legal merged descriptor and the
  harness stopped with `RecoveryRequired`, which needed a reboot. The plans
  top-allocate tensors to the very end of the scratch span (vector 15,296),
  so shrinking the span by one vector broke them, and the harness's
  pre-check used the planner's span rather than the build's. Any future
  reserved slot must shrink the planner's span too.
- **The wait is not plain store drain or HMX tail either.** A microbenchmark
  of 16 stores at the post's stride plus a barrier drains in 0.10 µs
  (contiguous: 0.06 µs). A 1 µs ALU spin after the HMX store made the post
  itself faster (909 → 545 µs: its loads of the HMX output tile wait for
  HMX). But the barrier after the post got slower (1.32 → 1.74 ms). Its
  cause is open.

## Grouped 1x1

The 1x1 profile showed a wait of about 0.65 µs per HMX tile after the post
loop (about 1.3 ms per crop), which the [fast depthwise](#fast-depthwise)
round could not explain. `--pw-group N` (`RS_PW_GROUP`, with `--pw-fast`)
runs a 1x1 op over groups of N column blocks. Each group:

1. gathers every block's B tile;
2. takes one barrier;
3. runs every HMX tile of the group into its own output buffer;
4. then posts them all.

Each tile gets the same HMX loads and the same `rs_pw_post` as before, so
outputs are bit-identical. The 1x1 scratch grows to N × (nk + no) × 16
vectors (at most 2,384 for N = 8). The native check and its Python mirror
(`native_ok(pw_group=)`) bound it, and the planner sizes it
(`build_plan(pw_group=)`, read from `link.json` by the harness). Plans and
descriptor tables are unchanged.

**What the wait was.** With the posts moved out of the way, the profile's
HMX phase alone takes 1.48 ms. Before, HMX plus its barrier plus the
post-loop wait came to about 1.6 ms. So the wait was HMX still working:
`hmx_store` returns before the tile is written, and the next barrier or the
post's loads pay for it. At about 0.21 µs per 32×32 HMX load (6,900 loads
per crop) this is the HMX cost on this thread, and issuing it stalls the
thread. Grouping removes the per-tile post → HMX transitions, not the HMX
work. A flag-free post (an instrument) changed nothing, which also rules out
the flags update.

Device (held power, 16 cases × 5 rounds, after a pilot through `Conv_18`):

| | Fast depthwise (`89511c18`) | Group 4 (`36083a22`) | **Group 8** (`a2377f9c`) |
|---|---:|---:|---:|
| 1x1, 29 ops | 3.55 ms | | 3.18 ms |
| **DSP request body, median** | 5.30 ms | 5.06 ms | **4.92 ms** |
| Final tensor vs fast depthwise | | 16/16 identical | **16/16 identical** |

Group 8 is the qualified image ([full run](../evidence/mercury-pw-group-device.json),
[pilot](../evidence/mercury-pw-group-pilot.json)). The group-4 run was a
step on the way, and its report isn't kept. Bounds (worst 0.394), keypoint
drift, the 969 restoration audits and the arena hash are unchanged.

## One to three crops per request

`--replay` on `build_teacher_link.py` adds crop replay to the grouped-1x1
image (`d7a78a81…`, 6,960 B; `RS_REPLAY`). Existing images and the loader
remain byte-identical, with hashes frozen in `tests/test_teacher_link.py`.
The whole network still uses **55 descriptors**, replayed once for each crop.
This avoids the 160-descriptor table limit that an expanded 3-crop table
would exceed (165 descriptors). The 20 ms native budget covers the entire
request, including all crops.

**Three is a maximum.** The Mercury harness reads `replay` from `link.json`;
`--batch 3` reserves slots for up to three crops but sends 1, 2 or 3 as
available, with no padding. Its test schedule cycles the count as
`1 + (round + case_index) % batch`, so smaller requests also follow larger
ones. Each crop gets the next case cyclically, starting with the current
case. Every crop of every request must match its own per-op qualification
result. The lane simulator also checks different inputs with VTCM carried
from one crop to the next.

`mercury_layout.build_plan(..., batch=3, replay=True, pw_group=8)` emits all
three crop sequences first and verifies that their VTCM layouts and compute
descriptors are identical, while every load/store window address differs by
one fixed crop stride. It then retains only crop 0's table. Parameter images
are shared. The full-network stride is `0x43000` bytes; window end is
`0x1c3e800`; peak VTCM remains 11,289 vectors.

The request and completion fields are:

| Field (32-bit words) | Replay meaning |
|---|---|
| Input 0..2 | Existing magic, table address, descriptor count |
| Input 3 | Actual crop count, 1..3 (0 retains the one-crop convention) |
| Input 4 | Crop stride in bytes, a 4 KiB multiple; zero for one crop |
| Metadata 2 | Total descriptors executed: crop count × table range length |
| Metadata 48..50 | Cumulative end ticks per crop when count > 1; otherwise zero |
| Metadata 96 onward | Cumulative descriptor end ticks for the **last** crop |

Before validating each load/store descriptor, the kernel adds
`crop_index * stride` to its window address. It bounds the stride before
multiplication and checks every adjusted descriptor against the existing
window. Compute descriptors and parameter addresses stay shared. The
harness checks all crop ends are increasing, the last end equals the last
descriptor tick, the last crop's descriptors follow the preceding crop end,
and the total stays within the wrapper time and native budget. Per-crop
body durations come from consecutive crop-end differences. The report keeps
these as `crop_end_ticks` and `crop_body_ms`; `op_us` describes the last
crop, identified by `op_us_crop_index`.

Device evidence: [pilot through Conv_18](../evidence/mercury-replay-pilot.json)
(3 cases × 4 rounds), then the
[full network](../evidence/mercury-replay-device.json) (16 cases × 5 rounds).
The full run sent 80 requests carrying 159 crops. Timing excludes each
case's first request (64 requests, 128 crops remain):

| Crops in request | Timed requests | DSP body, median | DSP per crop | Host wall per request | Host wall per crop |
|---|---:|---:|---:|---:|---:|
| 1 | 21 | 4.98 ms | 4.98 ms | 9.38 ms | 9.38 ms |
| 2 | 22 | 9.95 ms | 4.97 ms | 15.04 ms | 7.52 ms |
| 3 | 21 | 14.92 ms | 4.97 ms | 21.44 ms | 7.15 ms |

The longest request body was 14.94 ms. Replay keeps DSP time per crop near
the previous 4.92 ms; batching amortizes host costs. Host wall includes
staging, submit and final-tensor readback, and excludes the host heads
(2.09 ms median per case here). As with the earlier held-power runs, these
host times are noisy observations on the running Frame, not an end-to-end
Mercury tracking benchmark.

All crops matched qualification bit for bit; all 16 case final hashes match
the previous grouped-1x1 evidence. Worst bound ratio remains 0.394, worst
joint drift versus FP32 remains 0.185 px median / 0.669 px max, with no
`is_hand` flips. New controls natively reject **4 crops**, a **misaligned
stride**, and an **aligned stride that moves a later crop outside the
window**. The last passes the stride header check and exercises adjusted
descriptor validation. A clean 3-crop request passes after the controls.
The guarded fault, descriptor and 48 stride-2 field rejections, second-seal
and unsealed-request controls also pass. All 972 restoration audits pass;
window, reservation, whole-arena hash, firmware/power state and DSP service
are unchanged.

## Step 2c: heatmap head on the DSP

`--mheads` on `build_teacher_link.py` (`RS_MHEADS`) adds two descriptor kinds
to the replay image (`582d79ce…`, 8,016 B). Every other image, including
`d7a78a81…`, stays byte-identical.

- **Kind 7, regrid:** an exact word copy between grids, with per-phase source
  and destination bases and destination row/column steps. The destination is
  zeroed first.
- **Kind 8, 3x3 on HMX:** kind 1 with K = 9 taps × `cpi` pairs. B row
  `t·cpi + p` is pair `p` read unaligned at word offset
  `(t/3 − 1)·ws + t%3 − 1`, and the reads are bounded like the depthwise reach.
  The post, parameters and scratch are those of 1x1.

`build_plan(heads=True)` appends five ops after `Conv_139`:

1. Regrid the 8x8 map into a 10x10 grid with a zero ring. A padding-2 3x3
   then becomes padding 1 on the same grid.
2. Run `Conv_159` as kind 8 (63 → 64 channels).
3. Run `ConvTranspose_163` as a 1x1 to 168 channels, phase-major
   (`(2a + b)·42 + c`).
4. Regrid with 4 phases at column/row step 2 to pixel-shuffle the result
   straight into the ring of a 22x22 grid.
5. Run `Conv_164` as kind 8 (21 → 22 channels).

The plan stores `input.336` as well, because the pooled vector heads (depth,
extras, curls) stay on the host (`run_tail(..., skip=HEAD_OPS)`). It has 61
descriptors, and VTCM peaks at 11.3K vectors.

The FP16 contract is unchanged, since the contract already ran these ops as
HMX. The simulator matches the host emulator exactly.

On the device, against the current image on 96 real tracking crops
([evidence](../evidence/mercury-heads-device.json)):

- **Heatmap:** differs by at most 0.001.
- **Keypoints:** the worst joint moves 0.005 px median and 0.19 px max.
- **Other outputs:** identical.

| Per crop | Current | Heads on DSP |
|---|---:|---:|
| DSP body | 4.93 ms | 4.64 ms |
| Session wall, 1 crop | 10.6 ms | 8.0 ms |
| Host CPU, 1 crop | 4.5 ms | 2.3 ms |
| Host CPU, 2 crops | 4.0 ms | 2.0 ms |

## Native host path

With the heads image, the CPU side of a request runs in a small C library
(`src/framenpu/host/fnpu_host.c`, through ctypes). It covers input checks,
FP16 conversion, the stem columns, the hint layer (`Gemm_58` → scale →
FP16), grid staging and readback with the padding checks, and the pooled
vector heads.

- **Bundled build:** `data/host_aarch64.so`, built by
  `tools/build_host_native.py` and recorded in `data/host_aarch64.json`
  (bytes, SHA-256, source hash, flags, toolchain). It is freestanding (no
  libc, no Python API) and rebuilds byte-identical.
- **Loading:** only on aarch64, and only after its hash matches the
  manifest. `MercurySession(native=...)` picks the path: `None` (default)
  uses the library when available, `False` always uses NumPy, and `True`
  requires the library.
- **Numerics:**
  - FP32 → FP16 rounding matches NumPy on all 2³² inputs.
  - Staged bytes are identical to the NumPy path.
  - The dense heads are bit-identical (FP64 sums of exact FP32 products, in
    the same order).
  - `Conv_143`, which the reference runs as an FP32 BLAS matmul, is summed in
    FP64. It differs by at most 2.4e-7 on 96 real crops.

On the device, session CPU per crop fell from 1.76 to 1.56 ms (1 crop), 1.49
to 1.26 ms (2 crops) and 1.40 to 1.17 ms (3 crops)
([evidence](../evidence/mercury-native-host-device.json)). The kernels run
about twice as slowly right after the ~5 ms DSP wait as back to back, even
with no other load, so host work after the wait costs about double its
hot-cache time.

## Next steps

- **Remaining DSP time** (4.92 ms per crop): 1x1 3.18 ms, of which HMX
  itself is about 1.45 and parameter copies about 0.72; depthwise 0.98;
  stride 2 0.59; activation load/store 0.17. Hiding the posts and gathers
  behind HMX would need a second hardware thread for the HVX work.
- **Tracker integration.** The [pluggable backend](MERCURY_INTEGRATION.md)
  submits actual groups of 1..3 crops immediately and keeps CPU as the default
  and fallback. A [live comparison](MERCURY_LIVE.md) confirms worker CPU savings
  and a modest two-crop gain; the remaining target is consistent end-to-end
  improvement and sustained thermal/coexistence qualification.
- **Host overhead** is now about 1.8 ms per crop at batch 2 with held power.
  The remaining pieces are the dispatch wake and completion poll (about 0.9 ms
  per request), the staging in NumPy (about 0.55 ms per crop) and the gate
  claim and acknowledge. Part of the last is the harness audit, which a
  runtime would not repeat on every request.
- **Host CPU after the native path** (about 1.6 ms per 1-crop request): about
  0.75 ms is Python bookkeeping around the DSP call (backup and restore,
  gate, reservation and power checks, polling). The host kernels take about
  1.1 ms in-session, half of it the slowdown after the DSP wait.
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
# fast 1x1 post: add --pw-fast (timing breakdown: also --pw-profile, 1 case x 3 rounds)
# host time: --held-power [--batch 2]; compare per-crop wall over alternating short runs
# fast depthwise: add --dw-fast to the build
# grouped 1x1: add --pw-group 8 to the build
# replay: build the current image, then deploy the sources and jobs to the Frame
python tools/build_teacher_link.py --stride2 --s2-even --s2-direct --pw-fast \
  --dw-fast --pw-group 8 --replay --output /tmp/mercury/jobs-replay
# Frame root: pilot first (3 cases x 4 rounds), then full (omit --stop-after/--max-cases)
sudo python tests/hardware_mercury.py --run-on-device --jobs /tmp/mercury/jobs-replay \
  --model /tmp/mercury/grayscale_keypoint_jan18.onnx --cases /tmp/mercury/cases \
  --prefix --held-power --batch 3 --stop-after Conv_18 --max-cases 3 --rounds 4 \
  --output replay-pilot.json
sudo python tests/hardware_mercury.py --run-on-device --jobs /tmp/mercury/jobs-replay \
  --model /tmp/mercury/grayscale_keypoint_jan18.onnx --cases /tmp/mercury/cases \
  --prefix --held-power --batch 3 --rounds 5 --output replay-device.json
~~~
