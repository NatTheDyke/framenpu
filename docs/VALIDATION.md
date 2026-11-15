# Validation

## Real-device baseline — 2026-11-14

FrameNPU 0.1.0 ran on an authorized Steam Frame with SteamOS 0.5.8, CDSP ABI 9,
and Hexagon v75. The sanitized [numerical summary](../evidence/validation.json)
records the baseline, payload hashes and limitations.

The final hardware suite passed:
- A compiled C vector-add job produced [6, 8, 10, 12].
- An unsupported opcode raised JobFailed after cleanup; the session was reusable.
- GEMMs with (M,K,N) shapes (1,1,2), (4,2,1), (31,17,33), (32,32,32) and
  (33,65,35) exactly matched the independent tile-precision CPU reference.
- DSP ReLU/clipping and a serialized, reloaded two-layer XOR model passed.
- Final workers returned baseline SSR values, callback was zero, control was
  (1,0,1), and HMX clock/program configuration returned to its inactive baseline.

These exact inputs do not establish every FP16 rounding/exceptional-value case.
Sustained eye-tracking accuracy, latency and thermals were not measured.

The baseline-tested runtime was 2,972 bytes, SHA256
068cadcad806f1ac76f9f46c72f284ba89759c1e7990bb0e2ba9dda4c928c1df.
The baseline custom C job was 1,844 bytes. The working tree now includes
regenerated dispatch-aware payloads and an opt-in direct dispatcher. They are
built from included source without proprietary SDK/runtime libraries. The CPU
and HMX qualification below covers these new payloads.

## Direct dispatch CPU validation — 2026-11-14

The 3,344-byte runtime, 96-byte dispatcher and 2,232-byte patch helper passed
device tests on the same exact firmware profile. The sanitized
[dispatch summary](../evidence/dispatch-validation.json) records their hashes,
100 warm samples per CPU probe/backend, numerical and ownership checks, and
final firmware/service state. Median empty-job wall time was 4.294 ms with the
benchmark path and 0.734 ms with direct dispatch.

Custom C vector addition, DSP activations/broadcast addition, expected-error
cleanup, metadata compatibility, installer exclusion, legacy-client access
and restoration passed. The DSP service retained its PID with zero restarts;
the first 112 KiB of firmware code matched the installed image afterward.
The host wait now accepts worker returns preceding master completion state 3.
All 42 offline tests pass, including that transition.

## Direct dispatch HMX qualification — 2026-11-14

The same 3,344-byte runtime passed all five padded/tiled GEMMs, activations and
the serialized/reloaded XOR model through both dispatch backends. Every
reported output exactly matched the deterministic tile-precision CPU reference.
This does not establish all FP16 rounding or exceptional-value behavior.

The qualification harness audited 78 correctness submissions, including 39 HMX
jobs. Before acknowledgment, borrowed DDR buffers, callback and HMX configuration
matched the values captured under that job's operation gate. A separate timing
comparison ran 100 warm samples per backend, without the audit overhead.
Complete 32x32 GEMM submission wall-time p50/p99 was 29.340/39.095 ms on the
benchmark path and 25.359/37.309 ms on the direct path. These times include
uploads, power setup/teardown, packing and cleanup.

Final workers returned baseline SSR values; firmware code, dispatch window and
HMX clock/program configuration were restored. Control was `(1,0,1)`, callback
zero, and the DSP service retained its PID with zero restarts. See the sanitized
[HMX summary](../evidence/hmx-dispatch-validation.json) and opt-in
[qualification harness](../tests/hardware_hmx.py).

Sustained gaze coexistence, tracking accuracy and thermals remain unmeasured;
the direct path remains opt-in.

## Two-tile execution qualification — 2026-11-14

The subsequent 3,380-byte runtime adds opt-in two-depth-tile jobs on direct
dispatch. It passed the complete single-tile numerical suites on both backends
and new padded/seeded matrix cases through K=256. Tests include ordered FP32
rounding, an accumulator above FP16 range, final bias/ReLU and six native
argument rejections followed by successful reuse.

The new correctness run audited 34 submissions (30 HMX-marked). Firmware,
borrowed DDR, worker SSR and power configuration restored with zero service
restarts. A separate 100-sample comparison measured 32x64x32 median wall time
of 60.886 ms versus 38.288 ms, with submissions reduced from four to one and
HMX power cycles from two to one. Maximum measured two-tile job publication
to completion was 26.013 ms; the 40 ms timeout is unchanged.

See [the implementation/measurement notes](MULTITILE.md) and
[sanitized summary](../evidence/multitile-validation.json). All 49 offline tests
pass. HVX packing and persistent model memory are outside this change.

## HVX packing qualification — 2026-11-14

The 3,520-byte runtime adds vector packing for aligned full tiles and aligned
bulk copy, retaining scalar padding/tails. Native probes passed 136 packing
cases, including every halfword bit pattern, and 53 copy cases. Six mixed-path
GEMMs and both existing single-/multi-tile numerical suites passed exactly.
Across these correctness runs, 310 submissions passed restoration audits;
firmware, worker state and power configuration restored with zero service
restarts.

An identical-request comparison against the previous 3,380-byte runtime used
100 warm samples on direct dispatch. Publication-to-completion p50 fell from
8.861 to 0.640 ms for a 4 KiB copy, 14.611 to 10.997 ms for 32x32x32 GEMM,
and 25.281 to 15.789 ms for a two-depth-tile 32x64x32 GEMM. These are job
intervals rather than full Python matrix API call times. See
[HVX notes](HVX.md) and the [sanitized summary](../evidence/hvx-validation.json)
for timing boundaries, percentiles, reproduction and limitations. All 49 offline
tests pass. Output readback remains scalar; persistent model memory and
sustained tracking coexistence remain unvalidated.

## HVX output readback qualification — 2026-11-14

The 3,672-byte runtime unpacks HMX output and converts FP16 to FP32 with HVX
integer operations, retaining scalar FP32 accumulation in reduction order.
Every FP16 bit pattern converted exactly; 71 seeded accumulation cases matched
the old scalar implementation, with guarded edge layouts preserved. Existing
HMX, multi-tile and packing/copy suites passed. Across the four correctness
runs, 573 submissions passed restoration audits with zero service restarts.

Against the 3,520-byte input-packing runtime, 100 warm samples per case reduced
publication-to-completion p50 from 10.989 to 5.479 ms for 32x32x32 and 15.818
to 6.988 ms for 32x64x32. Complete submission p50 for the latter fell from
24.570 to 14.182 ms. These exclude Python matrix serialization and do not
establish sustained tracking latency. See [readback notes](READBACK.md) and
the [sanitized summary](../evidence/readback-validation.json). All 49 offline
tests pass; budgets, power policy and memory reservations are unchanged.

## Completion-window finding

An initial monolithic (33,65,35) GEMM completed all 1,155 outputs correctly, but
the host missed the stock master's roughly 50 ms completion lease. This was
not a hardware pipeline stall. A controlled recovery verified completion,
restored HMX configuration, and restarted the DSP service once to reinitialize
abandoned shared buffers.

The backend now schedules bounded depth/output-32 HMX jobs and separate DSP
accumulation with a 40 ms host timeout and expired-lease detection. This does
not provide hard-real-time or cross-client isolation. See [ABI.md](ABI.md).

## HMX research checkpoint — 2026-11-15

The following hardware findings were measured on 2026-11-14. Checkpoint review
and offline reproduction on 2026-11-15 did not run new device jobs or change
the public runtime, dispatch, power policy or memory reservations.

- The [multi-worker probe](HMX_THROUGHPUT.md) returned with one, two and three
  active workers, but concurrent sentinels differ. Its aggregate issue estimates
  do not establish independent correct GEMMs or model throughput. See the
  [sanitized evidence](../evidence/hmx-multi-validation.json).
- The [accumulator isolation suite](HMX_ACCUMULATOR.md) checked complete output
  tiles at M=1 and M=32 across all worker pairs and all three workers. All six
  isolated and eight serialized controls passed; all 24 deferred-store and
  24 concurrent cases failed numerical isolation. The recovered primitives
  require one owner for each complete clear/load/store transaction. All sessions
  restored state without a service restart. See the
  [sanitized evidence](../evidence/hmx-accumulator-validation.json).
- The [correct single-owner baseline](HMX_BASELINE.md) passed 400 measured jobs,
  16 excluded warmups and eight pilot jobs. Every measured FP16 output tile and
  final FP32 conversion matched exactly: 1,638,400 FP16 and 409,600 FP32 values.
  Resident load slopes were 7.41–7.44 GMAC/s. Phase timings separate HMX work
  from packing, transfers, conversion and fresh-session overhead. This reuses
  identical blocks within one job; it does not establish cross-job residency or
  distinct-weight model throughput. See the
  [sanitized evidence](../evidence/hmx-baseline-validation.json).

The reviewed source hashes match all three research evidence files. Rebuilt
multi-worker, accumulator and baseline payloads match their measured code hashes.
The cached-code and batched-load diagnostics were still awaiting hardware
results at that checkpoint; the subsequent qualification is below. Offline
tests cover numerical references, report validation, timing units, opt-in
boundaries and failure behavior; those tests are not device qualification.

## Cached-code and load-batching qualification — 2026-11-15

The [cached-code comparison](HMX_CACHE.md) passes 800 measured jobs and
32 checked/excluded warmups. Every output matches exactly. Powered clock
configuration is identical in both modes. The fitted resident load slope is
1,264–1,269 GMAC/s cached versus 7.076–7.077 GMAC/s uncached. Signed-dense
D=512/R=4 HMX p50 falls from 9,511.146 to 56.250 us, while fresh-session p50
remains 32.921 ms. Cached publication/cleanup costs roughly 1.5 ms per job.
These synthetic resident-block results identify the old loop's instruction
fetch bottleneck; they do not establish distinct-weight model throughput.
See [the sanitized evidence](../evidence/hmx-cache-validation.json).

The experimental entry cleans cached code on normal and guarded-fault return,
and the host checks confirmation before memory/power reclamation. Deliberate
traps exposed a missing FP restoration when abandoning nested C frames. The
experimental entry fixes it; three final traps return exception code 2, pass
DDR/power restoration audits, and are followed by correct GEMMs. Earlier fault
diagnostics are excluded from timing qualification. The public SDK/runtime
remain unchanged and need separate qualification of that unwind fix.

The [load-batching comparison](HMX_BATCHING.md) passes another 800 measured
jobs and 32 warmups using the corrected entry, with uncached code in both
modes. The fitted resident slopes are 4.933–4.939 GMAC/s for the diagnostic
control and 12.461–12.469 GMAC/s for batch2. Maximum-depth HMX p50 improves by
about 2.52x. Activation and weight storage deliberately alias, and the control
includes diagnostic helper overhead; this is separate from the cached-code
fast path. See [the sanitized evidence](../evidence/hmx-batch2-validation.json).

Across both measured suites, 6,553,600 FP16 and 1,638,400 FP32 values match
exactly. All sessions restore state, with zero service restarts. Maximum
worker spans are 10.323 ms and 14.335 ms, respectively, within the unchanged
40 ms host limit. Sustained tracking, thermals and full-model accuracy remain
unqualified.

## Publication audit

The repository provides an MIT license, self-contained examples, SDK/compiler
source provenance, offline validation, packaging checks and CI. Raw captures,
machine paths, credentials and proprietary firmware are excluded from the
public source; the numerical summary is intentionally sanitized.

The initial publication suite passed 29 offline tests; the current suite passes
81. Seven cache-diagnostic tests additionally cover alias identity, every output,
cleanup confirmation before resource reclamation, confirmed-fault restoration,
clock drift and timeout behavior. These host tests do not establish that the
experimental DSP cache cleanup works on hardware.

Local verification uses Python 3.14.9 and LLVM 23.1.4. The GitHub Actions
matrix is configured for Python 3.11–3.14; this local run does not establish
results for every configured version. Offline power tests use an emulator,
not live fault injection.

The release checks verify a clean exported source tree, wheel/sdist metadata,
an isolated wheel installation, CLI/model inspection and SDK resources.
Repository-maintenance changes do not themselves constitute another hardware
validation run. The included payload's byte identity can be checked with
tools/build_kernels.py --check.
