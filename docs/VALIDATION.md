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
were unchanged for that comparison; the subsequent public unwind port is below.

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

## Public SDK fault-unwind qualification — 2026-11-15

The public entry now reconstructs FP from its saved SP and shared 80-byte frame
size before `deallocframe`, fixing return through abandoned nested C frames.
The runtime remains 3,672 bytes with SHA-256
`4ee5d0122fc7cc1df5960e38cbcc58e43f60e7599364dcb70d72ca47e4fa3578`.
The vector-add FJOB, dispatch patch helper and source manifest are regenerated.
The dispatcher, ABI, power policy, memory bounds and timeout are unchanged.

The ordinary SDK-compiled regression probe keeps 1/2/4/7 helper frames live.
Both dispatch backends and CPU/HMX modes pass 16 normal controls, 48 guarded
leaf traps and 48 following same-session jobs. Exception code/cause/PC, C
continuation sentinels, live frame order and normal checksums all match.
Borrowed DDR, callback and power restore before acknowledgment; SSR, EVB,
dispatch and firmware-prefix checks pass. These runs perform 176 audits,
including 56 HMX jobs and direct-dispatch install/removal helpers.

The rebuilt runtime also passes the existing GEMM/activation/custom-job/saved-XOR
suites on both backends (78 audits, 39 HMX) and direct multi-tile correctness,
precision/rejection/reuse checks (34 audits, 30 HMX). All 288 main-suite audits
pass with the service retaining its PID and zero restarts. Timing harnesses
use one smoke sample here; no new performance claim is made.

See [FAULT_UNWIND.md](FAULT_UNWIND.md) and the
[sanitized evidence](../evidence/fault-unwind-validation.json). Guarded software
traps do not qualify arbitrary exception, memory-corruption or stall recovery.
Existing custom FJOBs embed the old entry and must be rebuilt to get the fix.
Earlier measurement reports retain their measured payload/source hashes.

## Repeated resident GEMM — 2026-11-15

The fixed resident proof passes 100 measured inputs and four checked warmups
at each K=32/64/128, plus six checked reuse/replacement jobs: 318 full outputs,
325,632 FP32 values. All A/B depth tiles are distinct, weights upload/prepack
once per session, and complete output bytes match an integer-derived reference.
The arithmetic is a full-depth HMX transaction with a final FP16 result promoted
to FP32; it does not qualify model error tolerances.

Twenty-one native command rejections preserve sequence/output and permit valid
reuse. Three cached traps flush a dirty output sentinel, confirm code/data
cleanup and invalidate mutable session state. Each is followed by full-span
restoration and a successful fresh session with different weights and a different
tagged instruction image. All 372 restoration audits pass (360 HMX, 12 dispatch
helpers), with original gate/callback, code prefix and power snapshots restored.
The service retains its PID and zero restarts. The subsequent ordinary public
GEMM/activation/custom-job/XOR regression adds 78 passing audits, 39 HMX.

Median audited resident command time is 12.4–12.6 ms; maximum native wrapper
span is 4.613 ms and publication-to-completion 5.674 ms. Full-window cache cleanup
and per-request power dominate. The 40 ms timeout, existing executable slot,
single HMX owner and 8 KiB VTCM scope are preserved. Initial claim-only controls
are excluded; the qualified host initializes/restores within real submissions.
See [RESIDENT_GEMM.md](RESIDENT_GEMM.md) and the
[sanitized evidence](../evidence/resident-gemm-validation.json) for hashes,
all phase records, timing definitions, limits and reproduction.

## Real-weight convolution — 2026-11-15

The [convolution proof](CONVOLUTION.md) pins ncnn `20260526`/`e54f7b1`, the v20a
param/bin hashes and four private recorded IR crops. Strict preflight validates
the complete 77-layer/90-blob graph, tagged weights, shapes and exact binary EOF.
An independent FP64 lowering checks the full/FP16-input ncnn references within
`1e-4`. Native tag-2 code adds FP32 bias/clip within a 3,796-byte payload.

The stem K=27/N=24, pointwise K=24/N=16 and deeper K=64/N=16 pass 2,207,744
values across final calibration and held-out runs. Both use the same frozen
arithmetic rules and total-FP32-error limits. Held-out max absolute / worst-crop
RMS errors are 0.0140700/0.000785676, 0.00543642/0.000328457 and
0.00533867/0.000662564, respectively. Input quantization dominates the stem.
Limits depend only on weight/bias precision and calibration CPU estimates,
with 25% total-error margin; held-out errors cannot widen them.

All 3,558 final restoration audits pass (3,546 HMX commands, 12 helpers), with
original gate/callback/firmware-prefix/power state and the same service PID,
zero restarts. Three guarded traps add 24 pilot audits; earlier resident API
compatibility passes 63 audits and public GEMM/activation/vector/XOR passes 78.
These are separate from final counts. Maximum wrapper/publication-to-completion
times are 5.644/7.260 ms, within the existing limit. The one-tile-per-request
prototype still takes several seconds per layer; no tracking-speed claim is made.

See the [evidence](../evidence/convolution-validation.json) for reference/source
hashes, crop metadata, tolerance derivation, errors, timings and raw-report
digests. Models/images/tensors remain private. Inputs to later layers come from
ncnn intermediates; full graph error propagation, production ARM-FP16 equivalence,
landmark accuracy and a broader independent corpus remain unqualified.
The final host revision normalizes all checker exceptions to RecoveryRequired;
a full repeated matrix produces identical output hashes. Earlier raw-report
digests are retained separately from final audit counts.

## Complete streamed landmark graph — 2026-11-15

The pinned v20a graph passes all 77 layers, 90 blobs and four outputs on four
recorded infrared crops (two calibration, two heldout). The final matrix checks
20,351,360 layer-output values and passes 25,455 restoration audits, including
seven guarded traps and seven native rejections. Saved head tensors independently
reproduce hashes/error statistics for all 512 final output values. No tolerance was
widened after device/heldout observation. See [LANDMARK.md](LANDMARK.md) and the
[sanitized evidence](../evidence/landmark-validation.json) for per-layer results,
frozen provisional limits, source/model/reference pins and debugging/reset history.

HVX depthwise/pooling/residual kernels, native sigmoid and 26 tensor aliases are
implemented; HMX handles full-depth convolution/dense work. Intermediate tensors
remain on the host and audited bodies take 71–79 seconds per crop. This establishes
complete graph correctness within the small corpus, not resident execution,
broad tracking accuracy, real-time performance, coexistence or thermals.

The final same-source matrix and following 78-audit ordinary SDK regression
restore firmware, gate/callback and power with the service PID unchanged and
zero restarts. Earlier numerical identity failures stopped and retained resources;
explicitly authorized device reboots preceded further tests. They are excluded
from final matrix counts. The public SDK payload remains 3672 bytes.

## Resident landmark qualification — 2026-11-16

The [pinned resident graph](RESIDENT_LANDMARK.md) keeps weights and FP32
intermediates in the verified 6,152,192-byte arena. Four crops pass 308 layer
checks, 360 blob checks and 20,351,360 layer-output values under the unchanged
policy. All 512 head values independently reproduce their saved hashes and
are bit-identical to the streamed baseline. Two changing inputs per session
reuse one weight upload. Main restoration audits total 93,741; full-arena
readback, firmware/power restoration and service identity all pass.

Supplemental tests cover lossless FP16 conversion, gathers, stale metadata,
owner exclusion, guarded HMX/in-place depthwise faults, invalid-session refusal
and clean teardown. The standard SDK regression passes another 72 audits.
An initial depthwise mask bug stopped and retained resources; an explicitly
authorized reboot preceded the corrected primitive and final graph matrix.
No numerical policy or timeout was widened. Sanitized evidence is in
[resident-landmark-validation.json](../evidence/resident-landmark-validation.json).

Audited resident bodies take 325.8–326.6 seconds, slower than the streamed
71–79-second proof. Uncached depthwise gathering and per-request power overhead
dominate. This establishes resident dataflow and reuse, without useful tracking
latency, broad accuracy, coexistence or thermal qualification.

## Resident gather optimization — 2026-11-16

The [depthwise/max-pool gather change](DEPTHWISE_GATHER.md) passes the same four
crops and frozen policy. All 20,351,360 layer-output values pass; all 512 saved
head values independently match the resident baseline byte for byte. The main
matrix passes 57,800 restoration audits, one weight load per two-input session,
full-arena restoration and unchanged firmware/power/service state.

Depthwise native time falls from 397.1 to 167.2 seconds across four crops
(2.38×), with requests reduced from 66,030 to 30,074 (54.5%). Audited frame bodies
fall from 325.8–326.6 to 183.6–187.1 seconds, a 1.76× overall improvement. Paired
raw-bit/bounds controls compare 40,500 cases per helper, plus ten rejected
descriptors and alternating timing samples; all 3,038 audits pass. The guarded
in-place depthwise fault refuses reuse and restores the arena. Ordinary SDK
regression passes another 72 audits. No reset, widened tolerance, timeout change
or cached alias was needed. The depthwise/pool modules shrink to 4,060/4,004
bytes; the other nine modules and public SDK payloads are byte-identical.
See [sanitized evidence](../evidence/depthwise-gather-validation.json).

## Cached resident instructions — 2026-11-16

The opt-in [resident cache entry](RESIDENT_CACHE.md) passes the unchanged
four-crop policy: 308 layer checks, 360 blobs and 20,351,360 output values.
All 512 saved head values independently match the earlier baselines bit for bit.
Two sessions each reuse one weight upload across two different inputs. All
44,739 main restoration audits pass, including selected-PC/cleanup validation,
independent arena restoration and unchanged firmware/power/service state.

Audited frames take 153.9–154.6 seconds versus the earlier 183.6–187.1-second
gather baseline, a 17.0% reduction. A sequential first-crop control with identical
module bytes takes 196.95 seconds uncached versus 153.88 seconds cached (1.28×),
with matching work totals and head bits. It is a single control, not a statistical
model benchmark. Small image/alias replacement controls include sixteen guarded
traps followed by reuse and three invalid requests, passing 53 audits. Linked
verification checks all eleven vector tables, forwarding targets and cached
tail targets; cached builds disable jump tables and reject absolute C relocations.
Native cached HMX and in-place depthwise fault controls invalidate/refuse reuse
and restore the arena across another 465/4,558 audits. Ordinary SDK regression
passes 72 audits. No failed completion, reboot or recovery was needed.
No reservation, work cap, power lifetime or timeout was enlarged. Data remains
uncached; public SDK payloads remain byte-identical. Results and limitations are
in [sanitized evidence](../evidence/resident-cache-validation.json).

## Publication audit

The repository provides an MIT license, self-contained examples, SDK/compiler
source provenance, offline validation, packaging checks and CI. Raw captures,
machine paths, credentials and proprietary firmware are excluded from the
public source; the numerical summary is intentionally sanitized.

The initial publication suite passed 29 offline tests; the current suite passes
177. Thirteen [linked-operator](RESIDENT_LINK.md) tests cover held-power
ordering and refusal before publication, batch replay against
the single-module state machine, stop-point/fault bounds, seal/retire ordering,
image and alias checks before reclamation, and the linked-build verifier; the
hardware results are in [resident-link-validation.json](../evidence/resident-link-validation.json)
and [resident-link-fast-validation.json](../evidence/resident-link-fast-validation.json). Seven cache-diagnostic tests additionally cover alias identity, every output,
cleanup confirmation before resource reclamation, confirmed-fault restoration,
clock drift and timeout behavior. These host tests do not establish that the
experimental DSP cache cleanup works on hardware.
Seven unwind-regression tests additionally check normal/fault trace validation,
distinct live frames, completion checks before reclamation, explicit opt-in and
stopping after a timeout. Native unwind qualification is recorded separately above.

The [resident planner](RESIDENT.md) adds 21 offline tests covering residual and
alias lifetimes, same-step input/output/scratch overlap, cache-line alignment,
padded strides, arena bounds, invalid schedules, deterministic placement, bounded
JSON and the CLI. Eighty seeded varied schedules additionally check every pair
of live allocations for overlap and independently recompute peak memory. The
synthetic residual example needs 187,392 bytes versus 297,984 without reuse.
These are host storage checks; no resident loader or device execution was added.
That planner step is followed by 12 resident-GEMM lifecycle tests covering
one-time weight loading, changing inputs, cache/numerical/guard validation before
reclamation, stale identities, fault invalidation, retained timeout state, crash
record refusal, no claim-only windows, sequence wrap, opt-in and bounded idle
observation. They simulate host lifecycle; actual native/cache behavior is
qualified by the separate hardware matrix above.
Thirteen convolution tests cover weight/bias tags and alignment, transposition,
asymmetric padding/patch order, malformed DAGs/binaries/attributes, finite values,
lazy input preflight, proof limits, partial spatial/channel tails, opt-in and
the rule that held-out errors cannot widen frozen limits, and checker exceptions
retaining resources before reclamation. NumPy/Pillow/ncnn are
optional reference-preparation tools, not dependencies of the offline suite.
Seven complete-graph tests additionally exercise alias lifetimes, shared storage,
depthwise/residual/pooling orchestration, K704 reduction and 33-output dense-head
tails, all four outputs/sigmoid, host validation before reclamation, input/model
preflight and heldout exclusion from tolerance fitting. They emulate native I/O;
actual operator arithmetic and all-layer results have the separate hardware proof.

Eleven resident-landmark regressions check lossless row-pair weight packing,
independent in-place downsample and forward-compaction overwrite proofs,
residual lifetimes, phase boundaries, guarded-fault state, bounded full-arena
restoration, interruption during close, program corruption and crash-record
refusal, plus explicit opt-in before artifact/device access in all three new
hardware tools. Lifecycle emulation does not establish native arithmetic or cache
behavior; those require the separate resident hardware qualification.
Five gather-qualification regressions additionally cover the independent
word oracle, signed-zero/NaN preservation, leading/trailing zero fill, the last
input row, borrowed-arena/work budgets, boundary-case coverage and explicit
opt-in. Native copy/bounds behavior has the separate paired hardware proof.
Seven resident-cache regressions check entry tags before device access,
mode-bound identity, full arena restoration, retention on missing cleanup or
wrong PC alias, compiler relocation rejection, linked vector/forwarding targets
and hardware opt-in. These tests emulate host lifecycle or check verifier
fixtures; native behavior has its separate cached graph qualification.

Local verification uses Python 3.14.9 and LLVM 23.1.4. The GitHub Actions
matrix is configured for Python 3.11–3.14; this local run does not establish
results for every configured version. Offline power tests use an emulator,
not live fault injection.

The release checks verify a clean exported source tree, wheel/sdist metadata,
an isolated wheel installation, CLI/model inspection and SDK resources.
Repository-maintenance changes do not themselves constitute another hardware
validation run. The included payload's byte identity can be checked with
tools/build_kernels.py --check.
