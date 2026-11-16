# HMX throughput experiments

The multi-worker rerun on 2026-11-14 completed with one, two and three active
workers. This establishes that workers 1/2 can issue the recovered HMX
instructions when their SSR HMX-enable bit is set. It does **not** establish
three independent HMX engines: concurrent output sentinels differ from the
single-worker control, and the hot loops are not synchronized after setup.

The supported public runtime remains worker-0 HMX. These fixtures are research
tools, not an inference backend. The sanitized measurements and source hashes
are in [hmx-multi-validation.json](../evidence/hmx-multi-validation.json).

The subsequent [accumulator isolation test](HMX_ACCUMULATOR.md) resolved the
ownership question for these primitives: accumulator state is shared across
workers, and each complete clear/load/store transaction must be serialized.
The subsequent [correct single-owner baseline](HMX_BASELINE.md) measures a
7.41–7.44 GMAC/s resident load slope with full-output verification and separate
packing, transfer, conversion and host-session timings.
The later [cached-code comparison](HMX_CACHE.md) identifies uncached instruction
fetch as that loop's bottleneck; its fitted resident slope is 1,264–1,269 GMAC/s
at unchanged clock configuration. A separate [load-batching comparison](HMX_BATCHING.md)
qualifies about 2.52x improvement in an uncached diagnostic. Both preserve one
HMX owner and are synthetic within-job results, not model throughput.

## Previous measurements

The incoming research handoff reported 2.26 GMAC/s with the shipped
clear/load/store primitive and 6.55 GMAC/s for clear-once, repeated load pairs,
store-once. Its decomposition was approximately 96 reference ticks for the
load pair, 52 for clear, and 92 for store/barrier. These are historical handoff
figures, not a fresh reproduction or a statistical qualification.

`c31:30` advances at **19.2 MHz**, not the DSP core frequency. Each pair computes
32 × 32 × 32 = 32,768 nominal MACs. A reference-tick slope `t` corresponds to
`32768 * 19200000 / t / 1e9` GMAC/s of issued work.

The separate weight-residency experiment faulted when the recovered weight word
was issued once before an activation-only loop. That rejects that particular
instruction sequence; it does not prove that every possible weight-resident
HMX implementation is impossible. Raw words must keep their operand registers
pinned inside a single assembly block.

## Multi-worker entry corrections

[entry_mw.S](../tests/entry_mw.S) keeps the worker-0 return packet before the
extra worker path. Placing the path between `registers` and `finish` made worker
0 fall through into a second call after disabling HMX. The corrected original
fixture completed 1,000 iterations with `active=1` in 127,568 reference ticks.

Further inspection found that workers 1/2 did not enable HMX at all. Their
revised path sets SSR bit 26 before `frame_job` and restores the saved SSR after
return. Worker 0 retains the SDK's VTCM backup and fault handler. Workers 1/2
save and restore their own 8 KiB VTCM scratch in the job output arena; they still
use the firmware's fault vectors. A trap on either can require firmware recovery.

[hmx_multi.c](../tests/hmx_multi.c) requires direct dispatch and an output arena
of at least 16,448 bytes: 64 result bytes followed by two 8 KiB backup slabs.
Iterations are limited to 1,000. The operation gate owns these borrowed slices
for the call; they are not persistent allocations.

## Rerun results

One sample at each of 200 and 1,000 iterations was taken for each active count,
using a fresh direct session per submission. The whole call is timed, including
scratch initialization and save/restore. Two-point differences remove constant
costs but do not establish p50/p95/p99 or constant-cost behavior under contention.

| Active workers | Worker tick slopes (ticks/iteration) | Envelope slope | Aggregate issue estimate |
|---|---|---:|---:|
| 1 | 88.736 | 88.736 | 7.090 GMAC/s |
| 2 | 91.463, 96.859 | 96.859 | 12.991 GMAC/s |
| 3 | 114.313, 96.805, 96.756 | 96.729 | 19.513 GMAC/s |

The envelope is the latest active worker end minus the earliest active worker
start, using the shared reference counter. It includes skew between worker
starts; using only the longest individual span can omit that skew.

The full 1,000-iteration submissions took 6.404, 10.581 and 15.728 ms from
publication to observed completion, respectively. All six submissions returned
every worker with error zero. After each session, the gate was `[1,0,1]`, the
callback was zero, the original dispatch window and firmware code prefix were
verified, and the HMX power configuration matched its initial snapshot. No
firmware restart or manual power recovery was needed.

At 1,000 iterations, the first FP16 output words were:

| Active workers | Worker 0 | Worker 1 | Worker 2 |
|---|---|---|---|
| 1 | `0x77d0` | idle | idle |
| 2 | `0x7838` | `0x7330` | idle |
| 3 | `0x77d0` | `0x7bce` | `0x5000` |

All workers use identical operands. These changing sentinels are evidence of
interference and cannot qualify independent GEMMs. The experiment does not
check full tiles or isolate the cause of interference. Also, workers 1/2 copy
their scratch before issuing HMX; without a post-setup barrier, the active-count
parameter does not guarantee simultaneous hot loops. Multiplying issued MACs
by the active count is therefore an issue estimate, not useful model throughput.

The firmware's single gaze-thread HMX call site establishes its scheduling
convention. It cannot, by itself, establish hardware exclusivity or rule out
multiple issuing threads. Likewise, 19.513 GMAC/s of unchecked issue work does
not establish the 17.473 GMAC/s hand-tracking target, which requires correct
results, transfers, other operators and sustained coexistence.

## Reproduction

Build on the workstation using Hexagon-capable LLVM (23.1.4 for this evidence):

~~~sh
python3 handoff-tools/build_mw.py tests/entry_mw.S tests/hmx_multi.c \
    build/hmx_multi.fjob build/hmx_multi
~~~

Deploy `src/`, the Python files in `tests/`, and the built FJOB into the same
checkout layout on the device. Run one active count at a time, inspecting the
report before moving to the next count:

~~~sh
sudo python3 tests/hmx_multi_run.py --run-on-device --probe build/hmx_multi.fjob --active 1
sudo python3 tests/hmx_multi_run.py --run-on-device --probe build/hmx_multi.fjob --active 2
sudo python3 tests/hmx_multi_run.py --run-on-device --probe build/hmx_multi.fjob --active 3
~~~

The runner validates worker identities, sequences, result markers, timestamps
and post-session restoration. It exits on failure and attempts no gate clearing,
power recovery or service restart. It defaults to `active=1`, with 200 and 1,000
iterations. Changing the fixture requires rebuilding; an old 64-byte-output
multi-worker FJOB is not compatible with this runner.

The other `hmx_*_run.py` fixtures also cap their ladders at 1,000 iterations.
The streaming runner defaults to the known working paired control (`--style 1`);
the known-faulting styles 3 and 5 require an explicit selection.

## Accumulator follow-up

The bounded, synchronized [accumulator test](HMX_ACCUMULATOR.md) checked every
output element against isolated worker controls. All 14 isolated/serialized
control submissions passed; all 24 deferred-store and all 24 concurrent
submissions failed numerical isolation. Deferred stores saw the last worker's
product first and zero on subsequent stores, even without simultaneous issue.
Use one HMX owner for complete transactions and investigate HVX overlap for
packing, depthwise and elementwise work. The persistent DDR arena is documented
in [MEMMAP.md](MEMMAP.md).

## Instruction-fetch and load-batching qualification

Two additional fixtures use worker 0 and the existing 8 KiB saved/restored VTCM
span, and leave clock/power policy unchanged. Each passes 800 measured jobs and
32 checked, excluded warmups on 2026-11-15. Read their separate measurement
notes and evidence before comparing rates.

- [hmx_cached.c](../tests/hmx_cached.c) and
  [its runner](../tests/hmx_cached_run.py) compare the uncached code alias with
  a cached alias while retaining the baseline's math and full-output checks.
  The wrapper PC records the selected alias; disassembly must also establish
  that hot helpers use PC-relative calls. Build with the experimental
  [entry_cached.S](../tests/entry_cached.S), derived from the stock entry. It
  cleans and invalidates cached code at the uncached recovery label on both
  normal return and guarded synchronous faults. Worker-0 status words 14/15
  record cleanup armed/completed; the host checks them before restoring DDR or
  powering down HMX. Missing confirmation retains resources and requires
  recovery. The former C-only cleanup could be bypassed by a fault and has been
  replaced. Three explicitly selected guarded trap checks
  (`--verify-fault-cleanup`) pass after correcting SP/FP restoration in the
  experimental entry. It does not recover stalls or faults in cleanup itself.
  The public unwind fix is [qualified separately](FAULT_UNWIND.md); cached
  execution remains experimental. The runner defaults to the uncached control.
  See [HMX_CACHE.md](HMX_CACHE.md) and its
  [evidence](../evidence/hmx-cache-validation.json).
- [hmx_batch2.c](../tests/hmx_batch2.c) and
  [its runner](../tests/hmx_batch2_run.py) compare two ordinary load pairs with
  one two-block load packet. To fit the existing scratch span, activation and
  weight bases deliberately alias: each transaction computes
  `groups * (A0 @ A0 + A1 @ A1)`. Both blocks have distinct deterministic values;
  every returned FP16 tile and final FP32 conversion are checked. This is an
  instruction-batching diagnostic; its measured gain still requires validation
  with distinct activation and weight storage under cached execution. It uses
  the corrected experimental entry with cache cleanup skipped. The runner
  defaults to the ordinary load control. See [HMX_BATCHING.md](HMX_BATCHING.md)
  and its [evidence](../evidence/hmx-batch2-validation.json).

Offline builds:

~~~sh
python3 handoff-tools/build_mw.py tests/entry_cached.S tests/hmx_cached.c \
    build/hmx_cached.fjob build/hmx_cached
python3 handoff-tools/build_mw.py tests/entry_cached.S tests/hmx_batch2.c \
    build/hmx_batch2.fjob build/hmx_batch2
~~~

Both runners require root and explicit `--run-on-device`, verify all outputs
and post-session restoration, and stop at the first failure without recovery.
Start with bounded controls, review the results, and then explicitly select the
experimental mode. Hold the clock configuration constant when comparing modes.
Both runners stop if the recorded powered PLL/clock configuration changes.
Cached-code results from the older stock-entry fixture are incompatible with
the revised runner's required cleanup markers.
