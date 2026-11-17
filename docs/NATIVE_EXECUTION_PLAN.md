# Native execution toward a tracking budget

The [HMX model feasibility study](HMX_MODEL_FEASIBILITY.md) now quantifies the
next architectural choice using public weights and distinct synthetic inputs.
Packed operand streaming dominates the measured HMX pipeline. A half-width
fused 3x3 student has a conditional 47 ms convolution forecast, still requiring
2.4–3.9x backend improvement plus <=5 ms of other work for the 17–25 ms target.
The next experiment is a bounded packed-FP16 spatial/projection block with real
gathering and coefficient/activation reuse, before training a replacement.
No new graph speedup or accuracy result is claimed.

The earlier [step-1 arithmetic experiments](NATIVE_ARITHMETIC.md) found that
IEEE vector arithmetic returns all-zero results in the tested configuration;
supported persistent QFloat32 reductions work, including with FP16 inputs.
The plan below retains the architectural rationale and completed experiments.

The measured fused graph takes about 498 ms of native time per crop. A 20–30x
reduction requires approximately **17–25 ms per crop**. This is a target, not
a demonstrated consequence of changing precision. Multiple hands/views,
preprocessing and host submission need their own end-to-end budget.

The next priority is a native executor whose numerical contract permits
different rounding and accumulation order. Host replay optimization remains
useful, but does not reduce the 498 ms native cost.

## Change the numerical contract

Retain the current qualified graph as a reference. Its output bits serve as
an error measurement, rather than a requirement for the experimental path.
The existing frozen provisional layer limits remain the historical baseline;
a new path gets its own explicit policy, without widening historical results.

Measure final landmark displacement in the outputs' decoded coordinate units,
confidence/threshold changes and temporal jitter over recorded sequences.
Reference agreement measures regression against the reference model; tracking
accuracy needs representative recordings and independent labels or another
validated task-level reference. The current four crops from one recording
cannot establish general tracking accuracy. Set acceptance limits before
using a heldout sequence to select an implementation.

This permits mixed precision, fused multiply-accumulate, reassociation and
different tensor layouts. Memory bounds, one HMX owner, fault invalidation,
the 20 ms request budget, 40 ms timeout and clean teardown remain requirements.
Finite/range checks may be amortized only where equivalent detection or
proven arithmetic bounds justify it. Clipping must not conceal overflow.

## Why depthwise alone is insufficient

The current native attribution is approximately:

| Work | Native ms/crop |
|---|---:|
| Depthwise | 291 |
| Stem, pointwise convolutions and heads | 182 |
| Pooling, residual and remaining work | 25 |
| Total | 498 |

Even eliminating depthwise entirely leaves about 207 ms, only a 2.4x overall
speedup. All three categories must change. For scale, an illustrative 25 ms
budget of 8 ms depthwise, 12 ms convolution and 5 ms other work would require
roughly 36x, 15x and 5x reductions respectively. These are design allocations,
not measured forecasts.

## Arithmetic and dataflow to investigate

1. **Validate IEEE HVX arithmetic on the device.** Current kernels use QFloat
   multiply/add, conversion back to FP32 and special zero handling. LLVM's
   [Hexagon flag](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-mhvx-ieee-fp)
   enables IEEE vector operations; its
   [intrinsic definitions](https://clang.llvm.org/doxygen/hvx__hexagon__protos_8h_source.html)
   expose FP32 multiply/add and FP16 products accumulated into FP32. Local
   LLVM 23.1.4 cross-compilation and object disassembly for v75 confirm those
   instruction encodings are emitted. This is not device validation or a
   throughput measurement. Probe finite mixed-sign/cancellation cases, zeros,
   subnormals, range edges and fault/cleanup behavior before integrating it.
2. **Use FP16 activation storage with FP32 accumulation as the first mixed
   precision candidate.** Weights are already FP16. Persistent FP32
   activations, software half conversion and CHW gather work remain costly.
   Compare native IEEE FP32 on the existing layout with FP16 inputs/FP32
   accumulation and, separately, FP16 accumulation. Measure conversion and
   staging along with arithmetic. Lower precision does not guarantee a gain.
3. **Design the layout around both depthwise and HMX.** Prototype channel
   blocks/spatial tiles, aligned transfers and VTCM reuse, reducing small-map
   lane waste and per-tap gathers/broadcasts. Keep compatible intermediate
   tiles resident across producer/consumer operations where lifetimes and
   halos allow it. Fuse bias, clip and residual work when legal. Measure
   end-to-end subgraphs including repacking; a fast isolated instruction is
   not a fast graph.
4. **Replace tiny native work units with bounded tiles in a versioned
   experimental protocol.** The current graph performs 145,458 units per
   frame, including a 128-byte DDR state synchronization at every unit and
   frequent budget checks. Investigate publishing progress at coherent tile
   boundaries with bounded worst-case execution and defined fault state.
   Preserve independent host validation and the existing resource limits;
   do not simply delete state synchronization or timeout checks.
5. **Evaluate INT8 only with representative calibration and error evidence.**
   If the mixed precision/dataflow path still misses its budget, compare a
   quantized candidate. Its calibration requirements and available kernels
   must be established; the four-crop corpus is insufficient to assume that
   post-training quantization preserves tracking quality.

The [cached HMX result](HMX_CACHE.md) already shows much higher arithmetic
throughput for a synthetic resident workload than the graph delivers. That
supports investigating staging, packing, transfers and utilization, but does
not predict full-model speed or remove the need for distinct-operand tests.

## First experiment

The completed first experiment used an isolated IEEE arithmetic smoke test,
then supported QFloat alternatives on real-weight depthwise pilots covering
large spatial maps, downsampling and the 14/7-wide maps. It measured error
against ncnn and local precision references, with staging, combined
gather/arithmetic, stores and conversion preparation reported separately.
The modest gains support a channel-blocked VTCM dataflow experiment before
rewriting the graph or adopting a new precision policy.

The compile-only check preceding the hardware experiments used:

```c
Q6_Vsf_vadd_VsfVsf(sum, Q6_Vsf_vmpy_VsfVsf(input, weight));
Q6_Wsf_vmpyacc_WsfVhfVhf(sum_pair, input_half, weight_half);
Q6_Vhf_vmpyacc_VhfVhfVhf(sum_half, input_half, weight_half);
```

with `--target=hexagon -mcpu=hexagonv75 -mhvx -mhvx-length=128b
-mhvx-ieee-fp -Os -ffreestanding -ffp-contract=off`. All three compile and
disassemble to native vector arithmetic. Hardware testing subsequently showed
that the IEEE variants return zeros, so those encodings cannot be used as a
working arithmetic path in the tested configuration. See the experiment
report for the supported alternatives and their measured limits.
