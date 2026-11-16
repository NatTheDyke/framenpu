# Resident depthwise gather optimization

The resident depthwise and max-pool modules now mask leading padding in HVX
instead of entering a 32-element scalar loop for every left-edge tap. A load
can span rows or channels while remaining entirely inside the owned input
tensor. Leading, trailing and inactive lanes become positive zero before any
arithmetic. At the beginning or end of the tensor, a bounded fallback clears
one vector and copies only the valid elements.

This changes only [resident_graph_gather.h](../tests/resident_graph_gather.h).
Weights, tensor layouts, accumulation order, numerical policy, session state
machine and ownership remain unchanged. Code and data still use uncached
aliases. The 4 KiB code slot, 8 KiB saved VTCM, 128 KiB borrowed arena, 32-unit
work cap, five-millisecond between-unit budget and 40 ms timeout are preserved.

## Bounds and dataflow

For a tap starting at column `ix` with stride `s` (one or two), active lanes
start at `ceil(-ix/s)` when `ix` is negative and end at
`min(count, ceil((width-ix)/s))`. The contiguous load begins at
`channel*height*width + iy*width + ix`. It is allowed only when its beginning
is nonnegative and all `32*s` words fit inside the input tensor. Stride two
uses two vector loads and the existing word deinterleave. Both edge masks are
applied before multiplication or max pooling. A full trailing mask skips
`vsetq(128)`, preserving the previously qualified full-vector fix.

The boundary fallback uses the same active-lane interval. It initializes the
VTCM temporary with one vector store and copies that interval with scalar DSP
loads/stores. For a seven-wide row this avoids testing and writing 32 elements
at every tap. Extra vector reads within the tensor contribute only masked
lanes, so the existing in-place downsample future-read proof still applies to
all values used by arithmetic. Forward compaction is unchanged.

| Module | Qualified baseline bytes | Optimized bytes |
|---|---:|---:|
| Depthwise | 4,092 | 4,060 |
| Max pooling | 4,028 | 4,004 |
| Other nine modules | unchanged | byte-identical |

## Primitive qualification and timing

The comparison builder compiles the same bounded probe twice with the same
SDK and flags. One uses the gather helper from commit
`95960a2049e4b438a24893fc99639191ffdbd443`; the other uses the current helper.
It checks that the other shared headers are identical and records source,
code and toolchain hashes. Every call stays in the ordinary borrowed job arena.

An independent integer-word oracle checks 40,500 cases per helper across twelve
geometries, including single-element tensors, first/last channels, widths
7/14/28/56/70/112, all 25 tap offsets, padding zero through two, both strides,
full vectors and partial tails. These are 2,592,000 checked word values across
the two helpers. Signed zeros, subnormals, infinities and NaN payloads must be
preserved bit for bit; padding must be positive zero. Graph execution retains
its existing finite-value checks. Ten invalid-descriptor controls must return
before writing any output, and every completion is checked before reclamation.

Paired timing uses one and sixteen gathers, with two checked warmups and ten
measured samples for each count and helper. Helper order alternates by sample.
The difference in median body ticks divided by fifteen estimates the extra
cost per gather without the fixed entry cost. Timings include the probe loop
and output store. These repeated copy fixtures establish microkernel behavior;
full-model latency is measured separately with actual weights and crops.

The paired probe passes 3,038 restoration audits. Representative slope speedups
are 7.09× for a seven-wide left edge in an interior channel, 4.95× when that
channel requires the boundary fallback, 7.74× for a fourteen-wide left edge and
9.81× for a full 112-wide left-edge vector. The 112-wide interior case improves
1.04×; the seven-wide right edge in an interior channel improves 1.17×. First
and last tensor boundaries improve 4.92×/4.51×. Stride-two left padding improves
7.50×. Complete profiles and samples are preserved in the evidence.

## Full graph results

All four crops pass the same 308 layer checks, 360 blob checks and 20,351,360
layer-output values under the unchanged policy. All 512 head values independently
match their saved hashes and are bit-identical to the resident baseline.
Two sessions each reuse one weight upload across two different inputs. All
57,800 main restoration audits and independent full-arena restorations pass.

| Measured quantity, four crops | Resident baseline | Optimized |
|---|---:|---:|
| Audited frame body range | 325.8–326.6 s | 183.6–187.1 s |
| Total frame body time | 1,304.9 s | 743.3 s |
| Depthwise native wrapper time | 397.1 s | 167.2 s |
| Depthwise requests | 66,030 | 30,074 |
| Requests per frame | 23,209–23,215 | 14,217–14,233 |

Depthwise native time improves **2.38×**, with **54.5% fewer depthwise requests**.
Complete audited frames improve **1.76×**, a **43.0% time reduction**. Full-graph
baseline timings come from the earlier qualified run on the same firmware;
the microkernel measurements above alternate helpers within the current run.
Maximum optimized wrapper/publication intervals are 8.164/9.081 ms. Initial
load, input/weight transfers, qualification work and close remain separately
recorded.

These audited graph times include all layer checks. They remain slower than
the 71–79-second streamed proof and do not establish useful tracking latency.
Per-request power work, uncached arithmetic and small vector utilization remain
material costs. No broader accuracy, sustained coexistence or thermal claim is
made. Sanitized results are in
[depthwise-gather-validation.json](../evidence/depthwise-gather-validation.json).

## Reproduction

A full Git checkout containing the baseline commit is required for the paired
build. With the recorded LLVM 23.1.4 toolchain on the workstation:

```sh
python tools/build_gather_comparison.py --output /path/to/gather-probes
python tools/build_resident_landmark.py --output /path/to/optimized-jobs
```

Copy the probes and current source to the supported Frame, then run the
explicitly opt-in qualifier as root:

```sh
sudo .venv/bin/python tests/hardware_gather_optimization.py --run-on-device --probes /path/to/gather-probes --output /path/to/gather-comparison.json
```

Prepare the private model/corpus as in [LANDMARK.md](LANDMARK.md). Run the same
resident graph qualifier using the optimized job directory, first calibration
and then heldout with the completed calibration report:

```sh
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --jobs /path/to/optimized-jobs --model /private/model --corpus /private/corpus --controls --output /path/to/calibration.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --jobs /path/to/optimized-jobs --model /private/model --corpus /private/corpus --split heldout --calibration-report /path/to/calibration.json --output /path/to/heldout.json
```

As before, an uncertain completion or failed checker stops execution and retains
resources; no automatic retry, gate reset, power recovery or service restart is
introduced. Model weights, recordings and raw device dumps remain private.
