# Public SDK fault-unwind qualification

The public entry now restores FP as well as SP when abandoning nested C frames
after a guarded synchronous exception. This ports the fix discovered during
the [cached-code experiment](HMX_CACHE.md) without adding cached execution or
changing the ABI, memory reservations, power policy or 40 ms host timeout.

The rebuilt runtime remains 3,672 bytes, SHA-256
`4ee5d0122fc7cc1df5960e38cbcc58e43f60e7599364dcb70d72ca47e4fa3578`.
The bundled vector-add job and dispatch patch helper are rebuilt too. The
96-byte dispatcher is unchanged. LLVM 23.1.4 reproduces the payloads and their
updated source manifest.

## Change and migration

The SDK entry reserves an 80-byte frame and saves its SP before calling user
code. A fault may leave FP pointing into a nested C frame. Restoring SP alone
does not redirect `deallocframe`, which reads the saved FP/LR through FP. The
old recovery path could return into an abandoned call and publish completion
again instead of returning directly to firmware.

[entry.S](../src/framenpu_cdsp/sdk/entry.S) now uses a shared frame-size constant
for allocation and reconstructs `FP = saved_SP + frame_size` before restoring
VTCM/registers and deallocating the entry frame. The existing fault handler and
completion protocol remain intact. The public change is the frame-pointer
restoration, rather than the cache experiment's publication/cleanup protocol.

Existing custom `.fjob` files embed their entry code. **Rebuild them with the
updated SDK to receive this fix.** Updating the Python package does not rewrite
previously built kernels. Declarative `.fnpu` models and the wire ABI do not
change. Historical measurement reports retain their original payload and
source hashes; the new qualification is recorded separately.

## Native regression

[fault_unwind_probe.c](../tests/fault_unwind_probe.c) builds with the ordinary
public compiler and entry, once with CPU metadata and once with `--hmx`. Both
containers have identical 2,288-byte code, SHA-256
`f43abb29c8cca09b0a2aeace5fc290dfc69783b3d5d4b8c2ec26827f049b330d`.

The probe keeps one, two, four or seven recursive helper frames live below
`frame_job` (depth parameters 0/1/3/6). Volatile local arrays and use after each
recursive call prevent tail-call elimination. LLVM disassembly confirms real
frame allocations and recursive calls. It records each live SP/FP, frame order,
entry/return counts and an independently checked normal-return checksum.

The leaf either returns normally or executes `trap0(#37)`. A fault must return
SDK exception code 2 with the expected SSR cause and an ELR within the probe.
No C helper may resume after the trap: return count stays zero and the outer
result stays at its pending sentinel. HMX-marked probes also dirty the existing
8 KiB VTCM scratch, exercising the SDK backup/restore path.

[hardware_unwind.py](../tests/hardware_unwind.py) checks completion before
the host reclaims DDR or powers HMX down. Unexpected sequences, error codes or
frame evidence raise RecoveryRequired and retain resources. Expected JobFailed
is followed by a normal probe in the same Device. Borrowed DDR, callback and
power restoration are audited before acknowledgment; SSR/EVB and global health
are checked across reuse and session closure. The runner stops on failure
without recovery or service restart.

## Results — 2026-11-15

| Coverage | Result |
|---|---|
| Dispatch | Benchmark and direct |
| Power/scratch mode | CPU and HMX |
| Live helper frames | 1, 2, 4, 7 |
| Normal controls | 16 pass |
| Guarded faults | 48 pass, three repetitions per point |
| Same-session jobs after faults | 48 pass |
| Requested jobs | 112 across 64 sessions |
| Restoration audits, including direct install/removal helpers | 176 pass, 56 HMX |

Maximum live helper-stack span is 328 bytes. Maximum measured `frame_job` span
is 2.247 ms; maximum publication-to-result interval is 13.671 ms, within the
existing completion limit. This is correctness qualification, not a new latency
or throughput comparison.

The rebuilt payloads also pass the existing five padded/tiled GEMMs, activations,
custom vector addition and saved XOR model on both dispatch paths: 78 audited
submissions, including 39 HMX. Direct multi-tile correctness and precision cases,
six native argument rejections and reuse pass another 34 audits, including
30 HMX. Across these main suites, **288 restoration audits pass**, 125 HMX.
The service stays active with the same PID and zero restarts. Initial/final
gate, callback, dispatch window, firmware prefix and power snapshots match.

The [sanitized evidence](../evidence/fault-unwind-validation.json) includes
source/payload hashes, all unwind-session results, normal numerical regressions
and raw-report digests. Ordinary CI has 88 passing offline tests; the seven new
tests cover trace validation, live-frame checks, noncanonical completion resource
retention, explicit opt-in and stop-on-failure behavior. Host tests do not replace
the native fault qualification.

## Reproduction

~~~sh
python3 tools/build_kernels.py
python3 tools/build_kernels.py --source tests/fault_unwind_probe.c \
    --output build/fault_cpu.fjob
python3 tools/build_kernels.py --source tests/fault_unwind_probe.c \
    --output build/fault_hmx.fjob --hmx

# On the supported Frame, inspect a normal control and small fault/reuse pair:
sudo python3 tests/hardware_unwind.py --run-on-device --probe build/fault_cpu.fjob \
    --phase controls --depths 0
sudo python3 tests/hardware_unwind.py --run-on-device --probe build/fault_cpu.fjob \
    --phase suite --depths 0 --repetitions 1

# After those pass, run the complete CPU/HMX and backend matrix:
sudo python3 tests/hardware_unwind.py --run-on-device --probe build/fault_cpu.fjob \
    --hmx-probe build/fault_hmx.fjob --dispatch benchmark direct --phase suite \
    --output /tmp/fault-unwind.json
~~~

This qualifies guarded software traps on worker 0. It does not establish
arbitrary exception, memory-corruption or pipeline-stall recovery. Native jobs
remain trusted supervisor code. The existing RecoveryRequired contract and
explicit recovery requirements remain in effect.
