# Native arithmetic experiments

The IEEE HVX route is **operationally unavailable in the tested Frame
configuration**. Distinct finite inputs produce all-zero results from IEEE
FP32 multiply/add, FP16-input FP32 multiply-accumulate and FP16 accumulation.
FP32 and FP16 input echoes pass, and the same job's QFloat control computes
the expected values. The compiler emitting an instruction does not establish
that the device implements the optional extension.

The supported alternatives work, but their isolated depthwise kernel gains
are modest. FP16 storage reduces kernel body time by about 16–18% on the large
maps and 2.5–2.7% on the small maps. These results do not demonstrate the
20–30x reduction needed by the [native execution plan](NATIVE_EXECUTION_PLAN.md).

## IEEE capability and arithmetic controls

The cached 4 KiB-slot probe uses distinct dyadic operands across 64 lanes.
Its input and output stay in the borrowed job arena, with the existing entry
and saved/restored 8 KiB VTCM. It compares:

- QFloat multiply, FP32 conversion and add with exact-zero handling.
- Native IEEE FP32 multiply/add, FP16-input FP32 accumulation and FP16
  accumulation: all three zero their results on this device.
- FP32 inputs with persistent QFloat32 accumulation and one final conversion.
- FP16 inputs with QFloat32 pair products/accumulation and final conversion.

The all-zero IEEE behavior is consistent with the optional extension being
unavailable. A Qualcomm-authored
[QEMU patch series](https://www.mail-archive.com/qemu-devel%40nongnu.org/msg1181724.html)
describes destination zeroing when that extension is disabled. This supports
the interpretation; it does not distinguish missing hardware from platform
disablement. No configuration bits were changed to try to enable it.

The supported arithmetic controls run 1, 32, 256 and 4,096 iterations, three
samples each, against conservative operation-count/absolute-term rounding
bounds. A separate range fixture characterizes zeros, subnormals and overflow
without assuming QFloat has IEEE exceptional-value semantics. Three guarded
faults, three rejection controls and subsequent same-session arithmetic pass.
The half-to-QFloat32 vector-pair layout is independently identified as even
and odd lanes in separate vectors.

Register-only slopes between 256 and 4,096 iterations are about 13.1 ns per
64-lane iteration for the original control, 8.2 ns for persistent QFloat32
with FP32 inputs and 10.6 ns with FP16 inputs. These are small synthetic
samples of the compiled probe, including its loop/fence schedule. They are
not full-kernel or hardware-ceiling claims. No zero-result IEEE timing is
presented as usable arithmetic throughput.

## Real-weight depthwise pilots

All channels of four layers run on each of the four recorded crops. Inputs
are the pinned ncnn layer tensors, rather than propagated outputs of a new
graph. The three arithmetic variants share the same isolated CHW staging
design. Geometry-specific images reduce register pressure; none has vector
accesses on the job stack. Scalar stack accesses and prototype overhead remain.

Mean native kernel body time, milliseconds for an entire layer:

| Layer | Original QFloat control | Persistent QFloat32, FP32 input | Persistent QFloat32, FP16 input |
|---|---:|---:|---:|
| `convdwclip_0`, 112-wide | 22.66 | 22.25 | 18.55 |
| `convdwclip_1`, 112→56 | 29.27 | 29.06 | 24.66 |
| `convdwclip_8`, 14-wide | 46.11 | 45.80 | 44.94 |
| `convdwclip_12`, 7-wide | 63.12 | 62.79 | 61.43 |

These ratios compare variants of **the same isolated probe**, not speedups
against the existing optimized resident row-tiling executor. In particular,
the small-map probe computes one channel's row at a time and leaves most
FP16 vector lanes unused. Its absolute times must not replace the current
graph's timing attribution.

Row-level native brackets measure staging, combined gather/arithmetic and
store/check time separately. Gather/arithmetic falls from 3.15 to 1.83 ms on
the large stride-one layer, but staging alone still takes 11.14 ms with FP16
inputs. Other body work includes coefficient preparation, loop/checkpoint
work and counter updates. Fixed row pitches, extra staging and scalar stack
accesses contribute to this prototype's costs. The timers do not establish
unavoidable arithmetic or memory floors.

Host padding/quantization, full native wrapper and audited host submission are
reported separately in the evidence. The body table excludes job publication
and host preparation. Half panels can fit more channels per job; that reduces
submission count on the stride-two case and must not be mistaken for an
arithmetic speedup.

## Numerical results and scope

Worst absolute deviation over the four crops:

| Layer | FP32 persistent reduction vs FP32 local reference | FP16 storage vs FP32/ncnn | FP16 storage vs precision-matched local reference |
|---|---:|---:|---:|
| `convdwclip_0` | 0.0000200 | 0.02898 | 0.0000143 |
| `convdwclip_1` | 0.0000100 | 0.01119 | 0.00000453 |
| `convdwclip_8` | 0.00000334 | 0.002577 | 0.00000310 |
| `convdwclip_12` | 0.00000191 | 0.001994 | 0.00000238 |

The much smaller precision-matched error attributes most FP16 deviation to
input quantization. No tracking-quality threshold was selected, and the old
frozen policy was not widened. The term-magnitude bound used here is a gross
arithmetic-corruption check, not acceptance of a new graph's accuracy.
Landmark/confidence drift and temporal behavior remain unmeasured.

The final matrix compares 7,093,632 values in 48 saved output files. It passes
15 guarded faults, 15 rejection controls, ordinary SDK regression and 2,487
restoration audits. All sessions restore the full resident arena, firmware
prefix, power, gate and callback; the DSP service is unchanged. Layer-job
wrappers remain below 2.99 ms. No recovery, reboot or service restart occurs.

The first IEEE attempt stopped on a numerical-oracle mismatch after a clean
audited transaction. Independent arena/health checks preceded the diagnostic
captures that established the zero-result behavior. Early generic-kernel
timings are superseded by the final matrix. Records and source/image pins are
in [native-arithmetic-validation.json](../evidence/native-arithmetic-validation.json).

One pass per crop/mode and one boot provide pilot evidence, not randomized
timing, sustained thermal qualification or broad task accuracy. The crops
share one recording. No new arithmetic was integrated into the resident graph.

## Next experiment

Build a channel-blocked depthwise tile that uses the vector lanes and reuses
input rows and coefficient tiles in VTCM. Compare FP32 and FP16 storage under
equal tiling while retaining supported QFloat32 reductions. This will test
the dataflow change directly, before choosing the precision tradeoff. The
convolution path also needs substantial staging/packing/fusion improvements
to approach 17–25 ms native per crop.

```sh
python tools/build_ieee_probe.py --output /private/arithmetic
python tools/build_ieee_probe.py --depthwise --output /private/depthwise
sudo .venv/bin/python tests/hardware_ieee_probe.py --run-on-device --probe /private/arithmetic/probe.fjob --output /private/smoke.json
sudo .venv/bin/python tests/hardware_native_depthwise.py --run-on-device --jobs /private/depthwise --model /private/model --corpus /private/corpus --baseline-report /private/qualified-calibration.json --output /private/pilot.json
```
