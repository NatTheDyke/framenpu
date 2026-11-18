# Banded DSP prefix profile

This profiles the scalar control. The subsequent
[vector pooling/stitching implementation](TEACHER_STEM_VECTOR.md) reduces
the 5.215 ms interval below to 0.056 ms and the full prefix to about 10.6 ms,
with byte-identical outputs.

The largest readily identifiable cost is **scalar max-pool stitching plus
stride-2 decimation: 5.215 ms**. The two depthwise operators together take
**2.058 ms**. The three HMX operators take **7.045 ms**, including packing,
memory fences, matrix transactions and post-processing.

The qualified prefix is reproduced at **15.748 ms** including its native
wrapper. Coarse profiling measures **15.747 ms**, and the profiler image
with counters disabled measures **15.693 ms**. These are 24 prefix requests
per mode, four crops x six rounds, on the same boot and normal power settings,
with the host pinned to CPU2. The original image remains byte-identical to
the 15.67 ms comparison build. [Evidence](../evidence/teacher-stem-profile.json)
contains all timing samples and source/build hashes, with no activations.

## Per-operator coarse brackets

These times are measured across all seven bands. Operator time excludes its
parameter copy and band-buffer clear. Each entry is an independent median,
so adding the displayed values need not exactly reproduce the full median.

| Operator | Compute/processing | Parameter copy | Band clear |
|---|---:|---:|---:|
| `convclip_0`, RGB stem | 1.900 ms | 0.079 ms | 0.024 ms |
| `convdwclip_0` | 0.812 ms | 0.068 ms | 0.023 ms |
| `conv_10`, projection | 1.336 ms | 0.088 ms | 0.020 ms |
| `convclip_1`, expansion | 3.809 ms | 0.124 ms | 0.038 ms |
| `convdwclip_1`, stride 2 | 1.246 ms | 0.102 ms | 0.038 ms |
| **Convolution subtotal** | **9.103 ms** | **0.461 ms** | **0.142 ms** |

Other directly bracketed work:

| Work | Time |
|---|---:|
| RGB DDR -> VTCM, 301,056 bytes | 0.249 ms |
| Zero both 56x56 output grids | 0.007 ms |
| 2x2 max-pool and stitch blob 10 into VTCM | 2.760 ms |
| Gather stride-2 words and stitch blob 8 into VTCM | 2.455 ms |
| Blob 8 VTCM -> DDR, 462,848 bytes | 0.384 ms |
| Blob 10 VTCM -> DDR, 115,712 bytes | 0.097 ms |
| Profiler log VTCM -> DDR, 16,384 bytes | 0.014 ms |

The profiler log movement is instrumentation. Other small differences are
descriptor setup, loop/control work, timestamps and wrapper overhead.

The max-pool interval includes its four input loads, finite-half comparisons
and output writes. It is **not pure memory-copy time**. Decimation performs
scalar VTCM loads/stores. Their combined 5.215 ms is separate from the final
0.480 ms DDR handoff. This is the strongest immediate reason to vectorize
pooling and output stitching before changing the depthwise arithmetic.

## HMX execution and memory wait

Fine probes use separate single-band requests. One variant samples one eighth
of blocks, with eight phases covering every block once. Another brackets every
block as a cross-check. The actual arithmetic still executes for every block
in both variants, and every result must match the original image exactly.

| Reconstructed interval across the prefix | Sampled | Fully bracketed |
|---|---:|---:|
| RGB/pointwise B-tile packing | 0.369 ms | 0.364 ms |
| Pre-HMX packing-completion fence | 3.422 ms | 3.224 ms |
| HMX clear/load/store transactions | 2.756 ms | 3.005 ms |
| **HMX transactions + preceding fence, jointly aggregated** | **6.177 ms** | **6.227 ms** |
| Depthwise coefficient pre-splat | 0.079 ms | 0.079 ms |
| Depthwise tap reduction, including input/coefficient loads | 1.544 ms | 1.463 ms |
| FP16 conversion, FP32 bias, clip/check/mask and VTCM output store | 1.413 ms | 1.283 ms |

The transaction bracket measures the existing HMX clear/load/store calls; the
store routine ends with its completion barrier. It does not isolate pure MAC
execution from matrix loads, accumulator clearing or result stores. The
preceding explicit barrier is now timed separately. It can drain earlier
VTCM traffic, so its entire wait cannot be attributed to B packing alone.

The **combined HMX-plus-fence interval, about 6.2 ms, is more stable than its
split**. Inserting timestamps and profiler bookkeeping changes scheduling
and where asynchronous memory waits appear. An earlier bracket omitted the
pointwise pre-HMX barrier and left an unexplained gap; recording the fence
explicitly resolves that omission. Counter writes are kept outside the
packing/HMX/post sequence to reduce interference.

These fine-probe intervals are not an exact additive partition of the
uninstrumented 15.7 ms. In particular, the fully bracketed depthwise replay
adds appreciable instrumentation overhead; the complete depthwise operators'
coarse total of 2.058 ms remains the primary measurement. No hidden residual
is forced into HMX arithmetic. Empty adjacent-read calibration averages about
0.027 counter ticks; it is recorded without subtraction and does not account
for all scheduling effects introduced by instrumentation.

## Halo duplication and unused lanes

The first four operators each execute 504 vector blocks: 16,128 packed
position slots. Of those, 15,232 evaluate valid image coordinates, covering
12,544 unique positions. Thus **2,688 valid position evaluations are repeated**,
equivalent to **24 extra 112-wide rows, 21.43% of the unique image area**.
Another 896 slots are guard, tail or outside-image padding. Including that
padding, the executed slot count is 28.57% above the unique interior count.

For those operators, each band contains 56 blocks wholly within its
non-overlapping 16-row core, 14 wholly outside it, and two crossing a core
boundary. Across the seven bands these are 392 core, 98 halo/padding-only,
and 14 mixed blocks per operator. Block timestamps reconstruct:

| Overlapping block subtotal | Sampled | Fully bracketed |
|---|---:|---:|
| Halo/padding-only blocks | 1.632 ms | 1.602 ms |
| Blocks crossing a core boundary | 0.261 ms | See per-operator evidence |

These times overlap the operator, HMX and reduction intervals above. They
include useful halo work, padding and unused band rows; they are **not a
measured saving from a halo-cache implementation**. Some halos are necessary
for correctness, and caching or pruning them needs a separate validated change.

DW1 already skips 36 of 72 blocks per band. Its retained blocks still execute
8,064 positions across the frame for 3,136 required stride-2 positions:
**38.89% useful lanes**, or 61.11% unused lanes. Its required outputs have
no duplicate image coordinates. The complete operator is only 1.246 ms here,
so eliminating all unused lanes would not remove the much larger max-pool,
stitching or HMX/fence costs.

## Implementation and validation

Build the unmodified control and the optional profiler image separately:

```sh
python tools/build_teacher_link.py --stride2 --heads --stem --output baseline
python tools/build_teacher_link.py --stride2 --heads --stem --stem-profile --output profiled
sudo .venv/bin/python tests/hardware_teacher_link.py --run-on-device --stride2 --heads --stem --stem-profile --profile-baseline-jobs baseline --cpu-core 2 --jobs profiled --model /path/to/model --corpus /path/to/corpus --output profile.json --rounds 6
```

The 14,064-byte profiler image fits the existing reservation, leaving 2,192
bytes before the seal. Original control image, loader, qualification kernels
and default arithmetic remain unchanged. The usual relocation, vector-stack,
writable-data and control-flow checks pass.

Descriptor words 18..20 optionally hold the validated DDR log destination,
level and phase. Level 0 is the original path; level 1 records coarse brackets;
level 2 samples phase 0..7; level 3 brackets all blocks. Levels 2/3 require a
single-band range, and level 3 requires phase zero. All other extension words
must be zero. Profile logs occupy VTCM vectors 14016..14144, beyond all live
prefix buffers but within existing ownership. Their DDR span is checked for
bounds and overlap with every input/output/parameter span. No budget or
timeout is extended.

The completed run checks 24 original, 24 disabled and 24 coarse prefix
requests, 224 phase-selected single-band probes and 28 fully bracketed probes.
All prefix outputs are byte-identical to the unmodified image, including every
band/phase. Full RGB-to-head qualification, existing frozen accuracy ceilings,
guarded faults, descriptor rejects and seal controls also pass. All **800
restoration audits (798 HMX)** pass; window, reservation and full arena restore.
DSP gate, callback, power, firmware, boot, service PID and restart count remain
unchanged. No recovery or restart ran.

All 286 offline tests pass. Added checks cover native bounds mutations, exact
halo/slot counts, phase coverage, log accounting, memory liveness and unchanged
baseline machine code. This profile identifies optimization targets; it does
not change the deployed arithmetic or demonstrate a new inference speedup.
