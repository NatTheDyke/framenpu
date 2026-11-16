# Cached-code HMX qualification

Uncached instruction fetch dominated the earlier single-owner resident loop.
An unchanged-clock, identical-payload comparison on 2026-11-15 passed all
800 measured jobs and 32 checked, excluded warmups. The cached resident load
slope is 1,264–1,269 GMAC/s, versus 7.076–7.077 GMAC/s through the uncached alias.
These are synthetic within-job results, not full-model throughput.

The [sanitized evidence](../evidence/hmx-cache-validation.json) contains source
and payload hashes, latency quantiles, per-sample transaction timings, clock
configuration and guarded-fault controls. The comparison left the public SDK,
runtime, dispatch and power policy unchanged. The later
[public unwind port](FAULT_UNWIND.md) is qualified separately; cached execution
remains an experimental fixture.

## Identical math, different instruction alias

[hmx_cached.c](../tests/hmx_cached.c) includes the
[correct baseline](HMX_BASELINE.md): copy and pack two distinct 32x32 FP16
activation/weight blocks, clear once, issue D load pairs, store once, and repeat
four complete transactions. Return every FP16 tile and convert the last tile to
FP32. Inputs remain resident within a job and repeat along the reduction depth.

The same payload selects uncached code at `0xf0a92000` or cached code at
`0x87a92000`. The wrapper records its PC, and LLVM disassembly verifies that
baseline and hot HMX helper calls are PC-relative. DDR operands and outputs
retain their uncached addresses; VTCM and the math are identical.

Two deterministic cases, identity and signed dense, use integer-derived dyadic
references. D=1/32/128/512, two modes and two cases produce 16 points, each with
50 measured samples. Two warmup rounds are checked and excluded. Point order
is shuffled deterministically in each round. Every measured output matches:
3,276,800 FP16 elements and 819,200 FP32 conversion elements, all exact.

Both modes use fresh direct sessions and per-job HMX power. Powered PLL/clock
registers were identical throughout the comparison; the runner stops on drift.
The counter is 19.2 MHz. No physical core/HMX frequency is inferred from the
register values or reference-counter frequency.

## Results

Signed-dense HMX windows, summed across four transactions:

| Depth blocks D | Uncached HMX us p50 | Cached HMX us p50 | Cached p95 / p99 us |
|---|---:|---:|---:|
| 1 | 49.323 | 5.156 | 5.313 / 5.573 |
| 32 | 622.344 | 5.365 | 5.417 / 5.469 |
| 128 | 2,387.969 | 13.958 | 15.156 / 15.156 |
| 512 | 9,511.146 | 56.250 | 58.438 / 58.646 |

At D=512, identical HMX work is about 169 times faster. Its cached p50 rate is
1,193 GMAC/s, distinct from the fitted resident slope of 1,269 GMAC/s. Fits
include helper/loop costs and cold/fixed transaction effects; they do not
isolate instruction latency or establish the engine's absolute ceiling.

Instruction fetch also affects packing and conversion. At signed-dense D=512,
packing p50 falls from 34.479 to 2.135 us and final-tile conversion from 215.781
to 4.427 us. Cached code publication and entry cleanup remain outside those
internal phase windows but inside the recorded worker span.

| Signed-dense D=512 boundary | Uncached p50 ms | Cached p50 ms |
|---|---:|---:|
| Worker span, including publication/cleanup | 10.062 | 1.595 |
| SDK publication to result | 10.671 | 2.216 |
| Complete submission | 30.765 | 22.264 |
| Fresh Device session | 41.570 | 32.921 |

Cached publication/cleanup and other unattributed work cost roughly 1.5 ms per
job. At D=1 the cached worker span is 1.534 ms versus 0.599 ms uncached: the
publication cost outweighs the smaller compute saving. This diagnostic protocol
must be amortized before adopting it for small public matrix submissions.
Nearest-rank p99 is the maximum of 50 samples per point; it is not sustained
tracking tail-latency qualification. Phase quantiles are not additive.

## Publication, cleanup and fault return

Build with [entry_cached.S](../tests/entry_cached.S), not the stock entry.
Before the first cached store, the C wrapper arms worker-0 status word 14.
It stages the entire fresh 4 KiB code image in the existing VTCM scratch,
overwrites full cached vectors, cleans data-cache lines, and invalidates
instruction-cache lines before calling the cached alias.

The uncached entry's common recovery path cleans/invalidates code on normal
return and selected synchronous faults. It sets status word 15 only after that
work. The host's completion validator requires the expected armed/completed
markers before restoring DDR or powering down HMX. Missing confirmation retains
resources and raises RecoveryRequired. A pipeline stall or a fault in cache
cleanup still requires deliberate recovery.

Deliberate cached traps exposed a separate unwind problem in the stock-derived
entry: recovery restored SP but left FP in an abandoned C frame. Temporary
tracing observed exception code 2 before return, followed by an overwritten
status-pointer value and another recovery entry. The inferred second return is
consistent with `deallocframe` loading its return address through the wrong FP;
Qualcomm's [stack-instruction description](https://docs.qualcomm.com/doc/80-N2040-36/80-N2040-36_REV_C_Qualcomm_Hexagon_V62_Programmers_Reference_Manual_.pdf)
defines that FP-based behavior.

The experimental entry now reconstructs FP from saved entry SP plus its
80-byte frame size and uses an explicit exception continuation. Three final
cached trap controls return exception code 2, confirm cleanup, pass DDR/power
restoration audits before acknowledgment, and are followed by correct GEMMs.
Four earlier fault diagnostics are excluded from throughput qualification.
The public SDK has since received the FP restoration and its own payload
rebuild/[qualification](FAULT_UNWIND.md). These timing results retain their
original measured payload and source hashes.

All measured sessions restored the gate, callback, dispatch window, firmware
prefix and power configuration. The service stayed active with the same PID
and zero restarts. No service restart or power-recovery script was used.

## Reproduction and next step

~~~sh
python3 handoff-tools/build_mw.py tests/entry_cached.S tests/hmx_cached.c \
    build/hmx_cached.fjob build/hmx_cached

# On the supported device, run and inspect the uncached control first:
sudo python3 tests/hmx_cached_run.py --run-on-device --probe build/hmx_cached.fjob \
    --modes uncached --samples 1 --warmup 0 --depths 1 --transactions 1

# Then inspect one cached control, followed by an explicit guarded trap check:
sudo python3 tests/hmx_cached_run.py --run-on-device --probe build/hmx_cached.fjob \
    --modes cached --samples 1 --warmup 0 --depths 1 --transactions 1
sudo python3 tests/hmx_cached_run.py --run-on-device --probe build/hmx_cached.fjob \
    --modes cached --samples 1 --warmup 0 --depths 1 --transactions 1 --verify-fault-cleanup

# Only after those pass, collect the repeated comparison:
sudo python3 tests/hmx_cached_run.py --run-on-device --probe build/hmx_cached.fjob \
    --modes uncached cached --samples 50 --warmup 2 --output /tmp/hmx-cache.json
~~~

LLVM 23.1.4 produces 3,332 bytes, SHA-256
`88e53390758b0d93650ddf73b155394033f3c9f0d6dad02ec4d6a2ff3b254a57`.
Old stock-entry cache probes are incompatible with the runner's cleanup markers.
The 40 ms host limit, 20 ms local baseline budget and 8 KiB VTCM span remain.

The next deliverable is owned resident code, tensors and prepacked weights,
with one HMX owner and HVX staging workers, followed by a real convolution.
Validate distinct operands, cache handoff, numerical error and bounded graph
segments before drawing full-model or sustained tracking conclusions. Fresh
session latency, repeated publication and transfer costs remain separate work.
