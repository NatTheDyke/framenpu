# Depthwise arithmetic attribution

The earlier 246.71 ms coefficient/arithmetic window is mostly surrounding work
in these measurements. Keep the current QFloat/FP32 rounding schedule for the
first optimization experiments: reuse coefficients and improve mask/check
handling before changing numeric format.

## Completed full-crop matrix

The corrected timing checker now accepts legitimate zero-tick intervals.
After a user reboot, all six new early-layer runs and all ten full-crop runs
complete successfully. Each full run checks all 77 layers and then times three
frames. All saved heads independently match the held-power baseline bit for
bit. The matrix passes 9,879 restoration audits, full-arena restoration and
unchanged firmware/power/service state. Baseline and combined-window controls
run before and after the component builds.

| Timed window | Raw median ms | Empty-bracket-adjusted ms |
|---|---:|---:|
| Coefficient load, conversion and broadcast | 103.45 | 101.49 |
| QFloat multiply and FP32 conversion | 4.16 | 2.20 |
| Product nonfinite checks | 49.52 | 47.56 |
| QFloat add and FP32 conversion | 36.18 | 34.21 |
| Exact zero-identity preservation | 89.93 | 87.97 |
| Empty timestamp bracket | 1.96 | — |
| Combined coefficient/arithmetic control, before | 247.00 | 245.04 |
| Combined coefficient/arithmetic control, after | 247.14 | 245.18 |

The core multiply/add windows total about 36.4 ms after the empty-bracket
adjustment, versus about 237.0 ms across coefficient/check/identity windows.
These are different compiled schedules: the component estimates **do not add
up to the combined window**. Dependencies and scheduling shift across timestamp
boundaries. The result supports prioritizing surrounding work; it does not
establish a hardware execution floor or a precise removable percentage.

Disassembly places exactly two core instructions in each multiply/add window:
QFloat operation followed by FP32 conversion. There are no vector stack spills
in these depthwise builds. Profiling adds approximately 7% to complete native
wrapper time. The matching empty bracket estimates timestamp cost, while
counter-update and scheduling effects remain. This is one recorded crop on one
device and firmware, not a sustained tracking or broad accuracy claim.

## Next optimization experiment

Coefficient values are reused over spatial positions, but load/conversion and
broadcast currently recur for every output vector and tap. Test coefficient
preconversion/reuse first, then consider multiple output vectors per tap to
amortize broadcasts. Measure the actual gain: preconversion alone still leaves
loads and broadcasts.

The zero-identity and finite checks preserve existing numerical behavior and
failure detection. Investigate whether their masks or control decisions can be
reused; do not drop them without an equivalent proof and numerical validation.
Any replacement must retain tap accumulation order and the existing bounded
ownership, work-budget and timeout rules. These measurements qualify profiling,
not an optimized operator or a new precision format.

## Batched control: incomplete

A separate register-only probe repeats dependent operations 1,024 and 16,384
times and estimates slopes, avoiding the resolution problem of per-tap timers.
Its initial numeric oracle incorrectly demanded exact mathematical FP32
identities. At the 63-iteration multiply warmup, QFloat multiply-by-one returned
one ULP above the mathematical value (for example .5000000596046448 instead of
.5). No native fault occurred; the checker rejected the result before cleanup,
and the retained-state policy stopped the run. Callback and power were retained
until a subsequent user reboot. The later [reuse qualification](DEPTHWISE_REUSE.md)
completed after that reboot; the batched control itself remains unfinished.

The oracle has been corrected offline to compare these finite dyadic fixtures
against conservative operation-count rounding bounds using the graph's 2^-22
arithmetic scale. This changes no frozen model tolerance. A regression covers
the observed identity-rounding case, corruption and nonfinite rejection. All
183 offline tests pass. The new oracle and batched slopes have not completed
hardware qualification, so no core latency/throughput claim is made.

## Reproduction and records

Use the dedicated harness described in [DEPTHWISE_PROFILE.md](DEPTHWISE_PROFILE.md).
Additional build/harness modes are:

| Mode | Window |
|---|---|
| 7 | Coefficient load, conversion and broadcast |
| 8 | QFloat multiply and FP32 conversion |
| 9 | Product nonfinite checks |
| 10 | QFloat add and FP32 conversion |
| 11 | Exact zero-identity preservation |
| 12 | Empty timestamp bracket |

Full-crop samples, per-layer rows, code/source hashes, controls and the separate
batched-probe failure classification are preserved in
[depthwise-arithmetic-profile.json](../evidence/depthwise-arithmetic-profile.json).
Raw model data and device dumps remain private.
