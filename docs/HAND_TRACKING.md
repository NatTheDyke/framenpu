# Custom hand tracking on Frame: next milestones

Assessment: 2026-11-14, against FrameNPU 0.1.0 and the pinned FrameTop v20a
landmark graph described below. Bounded [two-depth-tile execution](MULTITILE.md)
now passes numerical, timing and restoration checks, as do
[HVX packing for full aligned tiles](HVX.md) and [HVX output conversion](READBACK.md).
The next priority is the single-owner HMX throughput investigation, followed by
owned model memory and a real convolution checked against ncnn. A persistent
DDR region is verified, but its allocator and executor remain unimplemented.
Full landmark inference and its sustained frame rate
remain unverified. An opt-in [direct dispatcher](DISPATCH.md) now passes CPU/DSP
and HMX device tests with a worker join and reversible installation. Median
warm empty submission fell from 4.29 ms to 0.73 ms; complete 32x32 HMX GEMM
submission fell from 29.34 ms to 25.36 ms. Tracking coexistence checks remain
pending. The cost examples below describe the default benchmark backend
and retain the earlier roughly 13 ms research estimate; it is not a fixed cost
across device conditions.

## What already works

The [device baseline](VALIDATION.md) establishes custom C jobs, padded HMX
matrix products, DSP accumulation/activations and a serialized XOR model on
SteamOS 0.5.8 / CDSP ABI 9 / Hexagon v75. Firmware identity and completion checks
are implemented. These are the foundations for a backend; the current model
format executes only sequential Dense, ReLU and Clip layers.

The default benchmark path has three costs that dominate a convolutional workload:

- Every submission invokes the stock FLOPS benchmark, costing roughly 13 ms.
  [Device.matmul](../src/framenpu/device.py) submits each depth/output tile
  separately, reads partial results into Python and submits DSP additions.
- Each job uploads code and operands, backs up/restores borrowed memory, and
  scopes HMX power to that submission. Weights and intermediate tensors have
  no persistent ownership.
- The stock dispatcher releases its completion/ownership window after roughly
  50 ms. The existing host limit is 40 ms; increasing it does not extend
  ownership. See the [failure contract](ABI.md#completion-window-and-failures).

`JobResult.elapsed_ms` starts after upload and HMX power setup and ends before
power teardown, restoration and acknowledgment. Measure around the entire
inference call when comparing end-to-end latency.

## Workload and capacity audit

This audit uses the locally retained graph from model revision
`03d7695fecb80f6cf8f3aa9321de130b5efd08b3`, with FrameTop source revision
`edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729`. It does not assume future model
revisions have the same topology. The landmark param SHA256 is
`54e08f284ff6f121ff00f4a9c65afc38c0dbb22a32f4af44fc4e0fde6c4a1bbd`.
The [model graph](https://huggingface.co/DeeJanuz/frametop-hands-models/blob/03d7695fecb80f6cf8f3aa9321de130b5efd08b3/v20a/hand.ncnn.param)
has 77 layers / 90 blobs: 31 Convolution, 16 ConvolutionDepthWise,
4 InnerProduct, 10 BinaryOp, 2 Pooling, 11 Split, Input, Reshape and Squeeze.
Fused activations include clip and sigmoid.

For ordinary convolution, MACs are
`output_height * output_width * weight_count`; depthwise uses the same formula
with its grouped weight count. Dense MACs equal weight count. Applying these
formulas and the graph's explicit strides/padding gives:

| Computation | MACs per crop | Share | Initial implementation |
|---|---:|---:|---|
| Thirty 1x1 convolutions | 115,781,120 | 79.52% | HMX with prepacked weights |
| Sixteen depthwise convolutions | 21,610,176 | 14.84% | HVX 3x3 / 5x5 kernels |
| Initial 3x3 convolution | 8,128,512 | 5.58% | Tiled input packing + HMX |
| Four dense heads | 86,016 | 0.06% | Compare HVX and HMX for small shapes |
| Total | 145,605,824 | 100% | Excludes preprocessing and non-MAC operators |

The first convolution alone is GEMM `(M,K,N) = (12544,27,24)` after lowering
the 224x224x3 input to patches; its output is 112x112x24. With the current
32x32x32 tile schedule this requires `ceil(12544/32) = 392` HMX submissions.
At roughly 13 ms of stock benchmark per submission, that is approximately
5.10 seconds of benchmark overhead alone. This is a hypothetical mapping,
not a measured convolution; the current public matrix API also caps each
dimension at 256.

Memory imposes an independent limit:

| Item | Logical storage before packing/alignment |
|---|---:|
| 224x224x3 input, FP16 | 294 KiB |
| Largest activation, 112x112x64 FP16 | 1.53125 MiB |
| 1,000,160 FP16 weights + 11,552 FP32 biases | 2,046,528 bytes, about 1.95 MiB |
| Current combined input/output job arena | 128 KiB |
| Current HMX scratch span used with save/restore | 8 KiB |

The largest activation is not the peak live tensor set: residual inputs and
operator outputs can coexist. Later layers also have 672 channels, exceeding
the current matrix dimension limit. Tiling must live inside the executor.

Research identified 8 MiB total VTCM and a separate gaze reservation. This is
not a free-capacity measurement. Neither the remaining VTCM nor the 32 MiB
shared DDR mapping can be treated as available model storage. The existing
job spans are borrowed under the operation gate; they are not allocations
that can survive its release.

At two hands, two views and 30 Hz, landmarks require
`145605824 * 4 * 30 = 17.4727 GMAC/s`. Four crops have a total 33.33 ms frame
budget, including transfers, preprocessing and tracking. Equal serial crops
would have less than 8.33 ms each after reserving time for that other work.
These are workload targets, not demonstrated throughput.

## 1. Establish dispatcher ownership and measure overhead

Start with a minimal direct job path that bypasses the FLOPS benchmark and
cooperates with existing firmware clients. It needs:

- A mapped dispatch/wakeup/return path, with explicit ownership of workers,
  executable storage, shared DDR, VTCM and HMX power. Identify a valid extension
  point before assigning an operation selector or choosing memory addresses.
- A versioned command/completion protocol with sequence and session generation,
  descriptor bounds, and explicit acknowledgment. A late completion must not
  be mistaken for a new job after reconnect/reset.
- Bounded work segments and cancellation at a completed tile boundary. Host
  timeout or disconnect cannot authorize reclaiming memory or powering off an
  engine that may still be running. All participating workers must return
  before releasing their resources.
- Required cache maintenance and publication barriers for code, descriptors
  and tensors; an uncached host mapping alone does not prove every DSP access
  is coherent.

Instrument host claim/upload/power/notification/wait/download/cleanup and
DSP queue/packing/transfer/HMX/HVX execution separately. Compare an empty job,
a copy and a repeated resident GEMM at the same verified clock setting.
Report p50/p95/p99, not just the best iteration. Do not infer performance
from advertised TOPS or extrapolate to unvalidated clock rates.

**Exit criterion:** repeated jobs bypass the benchmark, produce correct
results and restore idle state, while delayed acknowledgment, cancellation,
stale completions and another participating client cannot cause premature
reuse. The new path must establish its own completion contract before any
relaxation of the old path's 40 ms limit. Keep the existing backend for
regression comparisons.

## 2. Add resident tensors and a graph execution plan

Introduce tensor descriptors with shape, dtype, layout, strides, arena offset
and lifetime. Load/prepack weights once and keep intermediate activations on
the DSP. Python can remain the research control interface; the steady-state
loop should submit a crop or bounded graph segment, rather than individual
matrix tiles or layers.

Build a DAG plan: Split normally aliases storage, residual inputs stay live
until their last consumer, and shape operators require verified alias/layout
rules. Compute peak live bytes and allocator size including alignment,
packed weights, accumulators, packing buffers and double buffering. Make
DDR the backing store and use only explicitly owned VTCM as a tile cache.
Preserve gaze and FeatureNN reservations through the complete lifetime of
every allocation.

The 4 KiB executable reservation also needs a plan: either verify a larger
owned code region or load bounded kernel modules only after prior execution
has stopped, with the required instruction synchronization. Adding every
operator to the current runtime cannot assume unlimited code space.

**Exit criterion:** a planned residual block has a documented allocation map,
fits the verified reservations, reuses weights across runs and performs no
host round trips for intermediate tensors. No addresses are assigned solely
because a memory span currently contains zeros.

## 3. Validate real convolutions and numerical behavior

Pin the ncnn reference build, param/bin hashes and a deterministic crop corpus.
The current [ncnn tool](../src/framenpu/ncnn.py) only inventories operator
names. Extend preflight to check graph connectivity, inferred shapes,
attributes, binary lengths and supported semantics before device writes.
Decode tagged FP16/FP32 weights and FP32 biases, then transpose/prepack for
the chosen kernel layout. ncnn's
[binary reader](https://github.com/Tencent/ncnn/blob/master/src/modelbin.cpp)
and [convolution loader](https://github.com/Tencent/ncnn/blob/master/src/layer/convolution.cpp)
define tag/alignment, asymmetric padding and activation parameters; pin the
reference revision when implementing conversion.

Use `convclip_0` as the first real test: 3x3, stride 2, 3 input / 24 output
channels, left/top padding 0, right/bottom padding 1 and clip [0,6]. Compare
intermediate outputs against ncnn. Then validate a 1x1 layer and a layer with
reduction depth greater than 32 to exercise accumulation. Generate only the
input patches needed for each tile, rather than materializing a full im2col
tensor. Optimize the 1x1 path first because it accounts for about 79.5% of MACs.

The existing HMX path rounds each depth-32 product to FP16, then sums those
partials in FP32 on the DSP. Exact agreement with its tile-precision CPU
reference does not establish agreement with ncnn or hand tracking accuracy.
Measure real-weight error and saturation before choosing a production
accumulation scheme. Investigate retaining the HMX accumulator across depth
tiles, but validate its precision/range before relying on it. Record maximum
and RMS layer error and the effects on landmark positions and confidence.

**Exit criterion:** actual model weights and crops pass documented numerical
tolerances for the stem, pointwise and multi-depth cases, with separate kernel,
packing and end-to-end timings. Tolerances must be chosen from reference and
tracking accuracy results; the XOR test cannot supply them.

## 4. Complete one landmark graph

Add direct HVX depthwise kernels for 3x3 and 5x5, including stride 2 and
asymmetric padding. Add max/global-average pooling, residual add, fused clip,
sigmoid and validated Split/Reshape/Squeeze handling. Use a stable layout
across operators and fuse bias/clip/add where lifetimes and numerical
semantics permit. HVX should handle packing/conversion and vector operations;
HMX should handle suitable dense matrix work. Benchmark small tails/heads
before choosing their engine.

Validate each layer before comparing all four outputs for a complete crop:
`out0` screen landmarks, `out1` presence, `out2` handedness and `out3` world
landmarks. These names and the replicated grayscale RGB input normalized to
[0,1] come from the pinned
[FrameTop integration](https://github.com/Frametop/frametop/blob/edd7e61fdaa76bfa71f532cf8cb8e0a6bf60d729/hands/track/nets.cpp).
Preserve its crop transform, contrast settings and output interpretation.

**Exit criterion:** one full crop produces all four validated outputs with
resident intermediates. Then benchmark four crops sharing weights; batch them
only when measurements show a throughput benefit within the latency budget.
Initial support can target this pinned DAG while keeping custom weights
replaceable. General ONNX import and INT8 quantization follow a correct FP16
baseline and require their own operator/accuracy validation.

## 5. Integrate and qualify sustained tracking

Expose the executor through a native interface usable from FrameTop's C++
tracking loop. Initially retain palm detection, ROI construction and skeleton
reconstruction on CPU and replace landmark inference. Provide explicit backend
selection and CPU fallback for clean preflight/availability failures. A
poisoned DSP session must retain the existing recovery contract.

Measure one and four crops, queue age and dropped frames, full tracking
p50/p95/p99 latency, tracking accuracy/jitter, gaze timing/missed updates,
CPU time, power and thermals. Compare identical recorded inputs and a worn
headset baseline with and without sustained custom inference. A separate gaze
thread and a live service process do not establish unaffected gaze quality or
available compute bandwidth.

**Exit criterion:** sustained four-crop tracking meets the selected frame
deadline and accuracy target without unacceptable gaze regression or thermal
throttling. Only then assess palm offload and the additional reacquisition
workload. Real-time acceptance requires measurements on the intended firmware
and power setting.

## Next implementation deliverable

The persistent DDR region is now mapped and canary-checked; see
[MEMMAP.md](MEMMAP.md). The [HMX accumulator test](HMX_ACCUMULATOR.md) shows that
the recovered primitives share state across workers: full serialized
transactions pass, while deferred stores and concurrent transactions lose
isolation. Use one HMX owner for complete clear/load/store transactions, with
HVX workers for packing, depthwise and elementwise work. The earlier unchecked
aggregate issue estimates are not independent correct-GEMM throughput.

The [correct single-owner baseline](HMX_BASELINE.md) measures a resident load
slope of 7.41–7.44 GMAC/s, below the approximately 13.9 GMAC/s pointwise target.
Compare the firmware's HMX loop and verified clock configuration before
committing to the complete executor. Prepared cached-code and two-block-load
diagnostics are described in [HMX_THROUGHPUT.md](HMX_THROUGHPUT.md); they have no
checked-in hardware results. Packing, conversion and session overhead
are separate measured costs and cannot alone explain the resident HMX gap.

The benchmark-free dispatcher, two-depth-tile GEMM jobs, full-tile HVX packing
and HVX output unpacking/conversion now have numerical and restoration validation.
FP32 accumulation, initialization and bias/ReLU remain scalar. Establish the allocation/ownership map
before keeping weights across separate jobs. Offline
companion work is a pinned ncnn reference corpus and shape/liveness preflight.
The next model milestone is one real convolution through that path.

Qualcomm's
[XNNPACK/HVX work](https://www.qualcomm.com/developer/blog/2026/03/bringing-xnnpack-hexagon-npu)
is a useful source of microkernel techniques. Its published kernel results do
not establish a compatible loader, scheduler or performance result for this
Frame firmware. The same runtime compatibility must be established separately
before treating a QNN/ONNX package as an alternative backend.
