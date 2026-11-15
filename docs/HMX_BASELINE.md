# Correct single-owner GEMM baseline

This is the historical **uncached-code** baseline. The subsequent
[cached-code comparison](HMX_CACHE.md) identifies instruction fetch as the old
loop's dominant cost and measures a much faster, fully checked resident loop
without changing clock configuration. The rates below remain valid for this
specific baseline; they are not the hardware's demonstrated ceiling.

The verified resident load slope is **7.41–7.44 GMAC/s** with worker 0 as the
only HMX owner. All outputs matched over 400 measured jobs and 16 excluded
warmups. This replaces unchecked issue estimates with a numerically checked
baseline, and separates HMX work from packing, copies, conversion and host
session overhead. It still falls short of the approximately 13.9 GMAC/s needed
for the hand model's pointwise layers alone.

The 2026-11-14 measurements, workload/source hashes, latency percentiles and
per-sample HMX timing series are in
[hmx-baseline-validation.json](../evidence/hmx-baseline-validation.json).

## Workload and correctness

[hmx_baseline.c](../tests/hmx_baseline.c) uses the stock worker-0 entry and the
existing saved/restored 8 KiB VTCM span. No public runtime or power-policy code
changes. Each job performs:

1. Copy two row-major 32×32 FP16 input blocks from uncached job DDR to VTCM.
2. Pack both blocks in place with the runtime's full-tile HVX row-pair shuffle.
3. Execute four complete HMX transactions using the resident packed operands.
   Each transaction clears once, issues D load pairs, then stores once. The
   entire transaction remains with worker 0, following the
   [accumulator ownership result](HMX_ACCUMULATOR.md).
4. Return every transaction's complete packed FP16 tile to DDR for verification.
5. After the last transaction, reuse the no-longer-needed 4 KiB input span for
   row-major FP32 conversion of the final output, then copy it to DDR.

D is 1, 32, 128 or 512, giving logical products `(M,K,N)=(32,32D,32)` with the
same blocks repeated along K. The two 2 KiB blocks remain resident across depth
loads and all four transactions **within a job**. This is a synthetic resident
ceiling, not a distinct-weight 16,384-depth layer or cross-job model residency.

Two deterministic workloads are used: identity times a signed matrix, and a
signed dense matrix product. Inputs are eighth-integers. The host computes
integer dot products, scales by D, and rounds once to FP16. Accumulation is exact
in FP32 for these bounds. The host compares every returned FP16 bit and all
1,024 FP32 conversion values. It also verifies transaction identities, completion
sequences, timestamp ordering and unused metadata.

The measured jobs checked **1,638,400 FP16 values and 409,600 FP32 values**, all
exactly. An earlier eight-job pilot and all 16 warmup jobs were checked as well.
Each of the eight case/depth points has 50 measured samples. Point order is
shuffled deterministically each round, and two warmup rounds are excluded from
the statistics. Quantiles use nearest rank; p99 is the maximum of 50 samples,
not a sustained tail-latency qualification.

## Resident HMX results

Each HMX timing window includes clear, D calls to the recovered load helper,
store/barrier, and normal loop/call overhead. It contains no operand packing,
DDR copying, output conversion or numerical checks. Four transaction durations
are summed for the job. The 19.2 MHz reference counter is used directly; its
frequency is not treated as the DSP core frequency. Consecutive timer-read
overhead is recorded and is not subtracted from reported phase durations.

| Case | Depth blocks D | HMX µs/job p50 | p95 | p99 | GMAC/s at p50 |
|---|---:|---:|---:|---:|---:|
| Identity | 1 | 48.229 | 49.427 | 49.896 | 2.718 |
| Identity | 32 | 593.021 | 613.542 | 616.719 | 7.073 |
| Identity | 128 | 2,292.865 | 2,342.708 | 2,351.302 | 7.317 |
| Identity | 512 | 9,047.812 | 9,170.990 | 9,196.615 | 7.417 |
| Signed dense | 1 | 47.917 | 49.635 | 50.365 | 2.735 |
| Signed dense | 32 | 595.417 | 613.854 | 619.531 | 7.044 |
| Signed dense | 128 | 2,289.167 | 2,352.396 | 2,365.625 | 7.329 |
| Signed dense | 512 | 9,080.677 | 9,193.333 | 9,249.115 | 7.390 |

Least-squares fits of median HMX ticks per transaction versus D give:

| Case | Ticks per load pair | Fitted fixed ticks per transaction | Load-slope GMAC/s |
|---|---:|---:|---:|
| Identity | 84.532 | 155.875 | 7.443 |
| Signed dense | 84.854 | 139.128 | 7.414 |

The intercept is an empirical fit that includes clear/store and software costs;
it does not isolate a hardware instruction's latency. At D=1, a complete HMX
transaction takes about 12 µs. Greater depth amortizes transaction overhead but
does not exceed the measured resident load slope.

## Packing, copying and conversion

The signed dense D=1 job has four output tiles. Only the final tile is converted
to FP32; every tile is returned as packed FP16. These are separate, ordered
diagnostic phases, with no compute/transfer overlap:

| DSP phase | Data/work | p50 µs | p95 µs | p99 µs |
|---|---|---:|---:|---:|
| Input copy | 4 KiB DDR → VTCM, once | 36.875 | 38.177 | 38.229 |
| Packing | Two resident tiles, once | 34.844 | 35.417 | 35.573 |
| HMX | Four D=1 transactions | 47.917 | 49.635 | 50.365 |
| Raw output copies | Four tiles, 8 KiB VTCM → DDR | 81.094 | 83.073 | 83.802 |
| Output conversion | One FP16 tile → FP32 in VTCM | 228.125 | 233.750 | 235.938 |
| Converted output copy | 4 KiB VTCM → DDR | 37.500 | 38.906 | 39.063 |
| Remaining frame work | Validation, scales, metadata, timing and loop gaps | 113.802 | 116.042 | 117.083 |

At D=512, median packing remains 34.740 µs and final-tile conversion is
226.615 µs, while the four HMX transactions take 9.081 ms. This distinguishes
the small-product conversion/staging costs from the deep-reduction HMX limit.
Individual phase quantiles are not additive.

Conversion uses the existing exact `read_row` helper, writing a resident FP32
tile before bulk copying it. It does not include the public runtime's scalar
FP32 partial accumulation, bias or activation. This diagnostic staging schedule
does not change or benchmark the complete public `matmul` implementation.

## Host latency and restoration

Every sample uses a fresh direct Device session. The dispatcher is installed and
removed and HMX is powered on and off for each job. Warmup exclusion does not
make the session or power state persistent. Numeric checks and post-session
health audits occur outside the recorded session wall time.

For signed dense work:

| D | SDK publication-to-result ms p50 / p95 / p99 | Full fresh-session ms p50 / p95 / p99 |
|---|---|---|
| 1 | 1.196 / 1.284 / 1.542 | 32.425 / 41.483 / 47.008 |
| 32 | 1.748 / 1.847 / 2.008 | 32.399 / 37.623 / 43.255 |
| 128 | 3.436 / 3.518 / 3.546 | 34.688 / 42.570 / 47.177 |
| 512 | 10.208 / 10.387 / 10.462 | 40.686 / 49.451 / 55.344 |

The SDK elapsed field includes observing completion and extracting the output.
It excludes host power setup/cleanup and dispatcher install/removal. The full
session field includes those operations and connection overhead. Session wall
times over 40 ms do not imply a callback overrun: the maximum measured
`frame_job` span was 9.788 ms and the maximum SDK elapsed interval was 10.548 ms.

The runner traces gate claim, backup/upload, power-on, host readback, power-off,
DDR restoration and acknowledgment without changing their behavior. For signed
dense D=1, median power-on was 5.059 ms and power-off 1.500 ms. Backup/upload was
0.033 ms and readback 0.027 ms. Dispatcher work and other session costs remain
within the broader submit/session fields; they are not assigned to HMX compute.
The readback interval overlaps the SDK elapsed field, so those fields should
not be summed as disjoint phases.

The powered PLL/clock register configuration was identical in all 400 samples
and is recorded in the evidence. No physical core/HMX clock rate is inferred
from those register values. After every job, the gate was `[1,0,1]`, callback
zero, the dispatcher and firmware prefix matched their originals, and HMX
power configuration matched the initial snapshot. The DSP service stayed
active with zero restarts; no recovery was needed.

## Reproduction and next step

~~~sh
python3 tools/build_kernels.py --source tests/hmx_baseline.c \
    --output build/hmx_baseline.fjob --hmx

# On the Frame, from a checkout layout with src/, tests/ and the FJOB:
sudo python3 tests/hmx_baseline_run.py --run-on-device --probe build/hmx_baseline.fjob \
    --output /tmp/hmx-baseline.json
~~~

The payload is 2,848 bytes, built with LLVM 23.1.4, SHA-256
`819929416deb7ac73f7ae802f1b6a737ec67c109aed1de17c571b59ed80af70b`.
The runner defaults to 50 samples per point, two checked warmups, D=1/32/128/512
and four transactions. Use `--samples 2 --warmup 0 --depths 1 32 --transactions 1`
for a small preflight. Depth is capped at 512 and transactions at four; the
kernel checks a 20 ms local budget after each completed transaction. The stock
40 ms timeout remains unchanged. Any failure stops without recovery.

The baseline establishes a correct resident primitive at about 7.4 GMAC/s.
The next investigation was the [cached-code comparison](HMX_CACHE.md), which
resolved the loop's instruction-fetch bottleneck at unchanged clock settings.
Resident code/tensor ownership, distinct operand staging and fresh-session
overhead are now the next implementation priorities.
