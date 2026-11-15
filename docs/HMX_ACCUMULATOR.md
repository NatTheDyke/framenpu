# HMX accumulator isolation

**The recovered HMX primitives share accumulator state across workers 0/1/2.**
Serialize the complete clear/load/store transaction. Serializing just individual
instructions or load pairs does not preserve another worker's live accumulator.
This finding applies to the recovered primitives on the pinned ABI 9 firmware;
it does not rule out an undiscovered hardware mode or explicit accumulator banks.

The 2026-11-14 experiment checks every FP16 element, with exact comparison against
both an analytical reference and each worker's isolated control. The sanitized
measurements, timing intervals and source hashes are in
[hmx-accumulator-validation.json](../evidence/hmx-accumulator-validation.json).

## Test

[hmx_accum.c](../tests/hmx_accum.c) runs through the existing multi-worker entry.
Each selected worker uses a separate 8 KiB VTCM slice, with the same save/restore
ownership as the earlier [throughput experiment](HMX_THROUGHPUT.md). All operands,
output tiles and scale tables are private to that worker.

All 1,024 activation elements are 1. Weight elements are 1, 2 and 4 for workers
0, 1 and 2. A single depth-32 load therefore produces a uniform tile of 32, 64
or 128. Repeating the load 32 times produces 1,024, 2,048 or 4,096. Every
intermediate sum and expected output is exactly representable; no FP16 tolerance
or overflow explanation is needed. Output VTCM starts as a NaN sentinel.

The test covers all worker pairs (masks 3, 5 and 6) and all three workers (mask
7), at 1 and 32 loads. Each experimental case is repeated three times. The
62-submission suite follows a separate six-submission isolated preflight:

| Case | Submissions | Behavior | Numerically correct submissions |
|---|---:|---|---:|
| Isolated controls | 6 | Each worker alone | 6/6 |
| Serialized controls | 8 | Each worker completes clear/load/store before the next starts | 8/8 |
| Deferred stores | 24 | Each worker clears/loads in turn; only then do stores run in turn | 0/24 |
| Concurrent transactions | 24 | Workers start complete transactions after a setup barrier | 0/24 |

The barriers use single-writer monotonically increasing generation words in
uncached job DDR. They signal peers and abort after 24 ms from kernel entry,
leaving time for scratch restoration and the stock 40 ms completion window.
The output arena is 22,784 bytes: 256 control/metadata bytes, three 2 KiB raw
tiles, and two 8 KiB backup slabs for workers 1/2. Worker 0 uses the entry's
existing backup. This test does not allocate persistent memory.

The kernel records setup-ready, clear/load start, load end, store start and
store end reference ticks for each worker. The runner rejects incomplete
rendezvous, malformed completions, unexpected writes by idle workers, HMX work
before every participant is ready, or overlapping phases in the serialized
cases. Numeric mismatches in the experimental cases are findings; control or
protocol failures stop the suite without recovery.

## Observed accumulator behavior

For one load, all deferred-store repetitions returned:

| Workers | Expected independent tiles | Actual tiles in worker order |
|---|---|---|
| 0, 1 | 32, 64 | 64, 0 |
| 0, 2 | 32, 128 | 128, 0 |
| 1, 2 | 64, 128 | 128, 0 |
| 0, 1, 2 | 32, 64, 128 | 128, 0, 0 |

The same pattern held at 32 loads, scaled by 32. Each row describes every
element of every returned tile. The first store sees the **last worker's**
product; subsequent calls to the recovered store helper see zero. Thus the
shared state is overwritten by another worker's clear/load, and the recovered
store sequence consumes or resets it. The experiment does not assign that reset
to a particular raw instruction within the helper.

The clear/load phases and subsequent stores were explicitly ordered and their
recorded intervals did not overlap. This excludes simultaneous instruction issue
as a necessary cause: merely retaining an accumulator across another worker's
transaction already loses isolation.

Concurrent cases also failed all 24 submissions. All workers' transaction timing
windows overlapped in every case, by 51–3,945 reference ticks (2.66–205.47 µs).
For example, one worker pair produced a uniform 96 tile instead of its own 32
or 64, while its peer received zero. The result depends on interleaving, and is
consistent with shared accumulation plus consumption by the store helper.

Across the suite, 135,168 FP16 values were checked. All 24,576 isolated and
serialized control values matched. All 55,296 deferred-store values differed;
54,272 of 55,296 concurrent values differed. An occasional correct worker inside
a concurrent submission does not make that submission correct.

## Device state and implications

Every submission returned all three workers with error zero and completed all
required rendezvous. The longest publish-to-completion interval was 15.658 ms.
After every session, the operation gate was `[1,0,1]`, callback zero, the original
dispatcher and firmware code prefix were verified, and HMX power configuration
matched the initial snapshot. The DSP service remained active with zero restarts;
no manual recovery was needed.

The earlier aggregate issue estimates, including approximately 19.5 GMAC/s with
three workers, cannot be used as independent correct-GEMM throughput. A suitable
next executor design has one HMX owner handling complete transactions, with HVX
workers handling packing, depthwise and elementwise work. A cooperative lock
around the entire HMX transaction is also possible, but does not create three
independent HMX accumulators. Benchmark useful throughput with correct outputs
before making a hand-tracking performance claim.

The follow-up [single-owner baseline](HMX_BASELINE.md) now supplies that numerical
primitive baseline: approximately 7.4 GMAC/s with resident repeated depth blocks.
Its packing, conversion and host costs are measured separately; it is not yet
a full-model or sustained tracking result.

## Reproduction

Build with Hexagon-capable LLVM (23.1.4 for the recorded run):

~~~sh
python3 handoff-tools/build_mw.py tests/entry_mw.S tests/hmx_accum.c \
    build/hmx_accum.fjob build/hmx_accum
~~~

On the Frame, from a checkout layout containing `src/`, `tests/` and the FJOB:

~~~sh
sudo python3 tests/hmx_accum_run.py --run-on-device --probe build/hmx_accum.fjob \
    --phase controls --output /tmp/hmx-accum-controls.json
sudo python3 tests/hmx_accum_run.py --run-on-device --probe build/hmx_accum.fjob \
    --phase suite --output /tmp/hmx-accum-suite.json
~~~

The runner defaults to isolated controls. `--phase suite` includes the controls
before experimental cases. Exit zero means the experiment completed; check
`isolation_passed` and per-phase numerical results for its outcome. A timeout or
control failure exits nonzero and never clears gates, changes power outside the
normal completed-job cleanup, or restarts services. Workers 1/2 retain the earlier
entry's firmware fault vectors, so an instruction trap can still need deliberate
firmware recovery. The payload is 3,144 bytes, and its measured SHA-256 is
`510bfbb29f6baa980db546bc59347d7e2828be47deb405848a0bc93ad7414b51`.
