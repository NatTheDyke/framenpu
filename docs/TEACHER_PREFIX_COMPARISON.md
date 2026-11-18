# CPU prefix versus banded DSP prefix

The **CPU FP32 prefix + DSP tail/heads** is currently faster: **29.025 ms**
median host wall versus **47.508 ms** for the two-request DSP path. The median
paired saving is **18.476 ms**, across four crops and six rounds each. Both
paths pass the existing head-accuracy ceilings on all four crops.

This compares the same normalized RGB input through all four heads. It
replaces the earlier comparison between full-model DSP timing and a tail-only
number. [Evidence](../evidence/teacher-prefix-comparison.json) contains timing,
accuracy, build hashes, controls and restoration statistics only.

## Timing breakdown

| Median per frame, 24 frames each | CPU FP32 prefix + DSP tail | DSP prefix + DSP tail |
|---|---:|---:|
| CPU prefix, including native input/output copies | 2.708 ms | — |
| DSP prefix, including native wrapper | — | 15.670 ms |
| DSP tail and heads, including wrapper | 16.740 ms | 16.733 ms |
| Total DSP time | 16.740 ms | 32.411 ms |
| Host input conversion/packing/staging | 1.293 ms | 0.096 ms |
| Host time inside submissions beyond recorded DSP time | 8.105 ms | 14.858 ms |
| Complete host wall | **29.025 ms** | **47.508 ms** |
| Host wall range | 27.044–32.212 ms | 46.470–48.453 ms |
| DSP requests | 1 | 2 |

Each table entry is its own median; the columns need not add exactly. Paired
savings are computed from each crop/round's two frame times before taking the
median, rather than subtracting independent medians.

The DSP prefix takes about 13 ms more than the optimized native FP32 CPU
prefix. The second submission also adds host work: snapshots, readback and
restoration audits, image/seal verification, transport, polling and scheduling
are inside the host timer. The measured submission residual grows by about
6.8 ms, partly offset by about 1.2 ms less input packing/staging. The residual
is not an isolated measurement of audit cost alone.

The tail itself does not regress. Both modes use the same linked image,
weights, tail descriptors, arithmetic and 512-byte result. The native image
remains the qualified 10,800-byte banded build; no kernel was modified for
this comparison. The prefix still has halo recomputation, unused lanes in
retained DW1 blocks and wider depthwise/bias arithmetic. Per-operator DSP
timing is recorded in the subsequent [DSP prefix profile](TEACHER_STEM_PROFILE.md),
which identifies scalar pooling/stitching and decimation as a 5.21 ms target.

The original [banded-prefix result](TEACHER_STEM.md) measured 56.754 ms host
wall with unpinned host scheduling and the reference loader's FP64 array.
This comparison pins the complete harness to CPU2 and supplies FP32 arrays
to both paths. The lower 47.508 ms DSP-path host result is a different host
measurement condition, with unchanged kernel code and approximately unchanged
DSP execution time. It is not evidence of a new DSP optimization.

## CPU baseline and timing boundary

The CPU prefix uses the optimized ncnn build pinned in the
[CPU baseline evidence](../evidence/ncnn-cpu-baseline.json), one thread on
CPU2, a Cortex-A720. FP32 mode disables FP16 and BF16 options while retaining
the library's other normal optimized defaults. A small shared-library bridge
runs in the same Python process. There is no subprocess or file I/O per frame.

Each timed CPU frame creates an extractor, copies the loaded RGB input,
extracts blobs 8 and 10, unpacks/converts ncnn storage to planar FP32, and
copies those outputs into reusable host arrays. Python then converts and
packs those actual arrays into the DSP's FP16 grids, stages both, submits
one tail/head request and reads the 512-byte result. No recorded ncnn feature
blobs stand in for CPU computation.

Each timed DSP frame converts/stages the same RGB array as FP16, submits the
banded prefix, then submits the same tail/head job. The feature handoff uses
DDR with no host readback/repacking. Both modes include all submission audits.
Model/library loading, fixed weight/table installation, corpus file reads,
twenty CPU warmups, validation and diagnostic feature readback are outside
frame timing. Camera crop and normalization are outside both modes.

Mode order rotates across a full cycle and reverses on the next cycle. Six
rounds balance each position for both the two- and three-mode schedules.
Host affinity is restored at exit. Power and DSP clocks/configuration are
not changed for this comparison.

## Accuracy and controls

The CPU FP32 hybrid's worst decoded landmark displacement is **0.429077 crop
pixels**; the DSP-prefix path remains **0.374492 pixels**. Both pass all
existing screen max/RMS, world, presence, handedness and incremental head caps.

The CPU FP16 hybrid passes three crops but fails `calibration_controller`:
screen-head RMS is **0.341487**, exceeding the unchanged **0.25** ceiling.
That crop is excluded from FP16 timing. Its 18 frames on the other three
crops measure 27.526 ms host wall and 1.518 ms CPU prefix, but they do not
establish an eligible four-crop alternative and are not the headline baseline.
This describes the CPU-prefix/DSP-tail combination, not all-CPU ncnn accuracy.

Each CPU mode's actual outputs first feed the independent per-layer DSP
qualification kernels, followed by the observed merged tail/head plan. One
intermediate in the FP32 `calibration_open` case differs: one value in blob
28 changes by `2.384185791e-7`, about 0.11246 of the pre-existing layer bound.
Its magnitude is approximately `1.70827e-4`, so the value is **not subnormal**.
The original bit-exact intermediate check was too strong for this additional
input set. Intermediate differences are reported and checked against the
existing numerical bounds; no bound or head ceiling was widened. The final
feature map remains byte-exact against qualification on every CPU case,
including this one. Pooled/head incremental checks also pass.

Every timed result matches its mode's independently checked result byte for
byte. The complete existing DSP-prefix qualification and guarded fault,
descriptor-rejection and seal controls also run before comparison. All
**849 restoration audits (847 HMX)** pass. Window, reservation and whole arena
restore; gate, callback, power and firmware checks pass. DSP service PID,
restart count and boot ID remain unchanged. No recovery/restart/gate reset ran.

Two initial runs stopped on the bit-exact checkpoint difference and restored
normally. A subsequent diagnostic incorrectly classified the small difference
as evidence of subnormal values; the magnitude check rejected that assumption.
The completed report records the difference explicitly under the unchanged
numerical contract. No uncertain completion occurred.

## Reproducing

On the device, build the bridge against the exact previously qualified ncnn
library. The directory contains `ncnn/` sources and `build/src/libncnn.a`;
the builder refuses a library hash mismatch.

```sh
python tools/build_ncnn_prefix.py --ncnn /path/to/qualified-ncnn-build --output cpu
sudo .venv/bin/python tests/hardware_teacher_link.py --run-on-device --stride2 --heads --stem --cpu-prefix-library cpu/libncnn_prefix.so --cpu-core 2 --jobs /path/to/banded-jobs --model /path/to/model --corpus /path/to/corpus --output comparison.json --rounds 6
```

The paired comparison needs no new DSP image. Native CPU library and source
hashes are recorded with the device report. Offline tests check that CPU
computation and staging occur inside every timed frame, request counts and
timing boundaries, balanced ordering, numerical/final-map checks and exclusion
of ineligible precision modes. The complete offline suite passes 281 tests.

CPU FP32 prefix + DSP tail is the current latency choice between these two
experimental paths. This remains a diagnostic Python-host measurement, not
a production native-host benchmark, power/thermal soak or tracking validation.
