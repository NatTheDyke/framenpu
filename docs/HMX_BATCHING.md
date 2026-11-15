# Two-block HMX load qualification

The two-block packet passes full-output checks across 800 measured jobs and
32 excluded warmups. At maximum depth it reduces this diagnostic's HMX time
by about 2.52 times. Both modes execute uncached code; this is separate from
the much larger [cached-code result](HMX_CACHE.md).

The 2026-11-15 [sanitized evidence](../evidence/hmx-batch2-validation.json)
contains payload/source hashes, input hashes, timing quantiles and every
measured transaction timing. Public SDK/runtime and clock/power policy are
unchanged.

## Workload and limits

[hmx_batch2.c](../tests/hmx_batch2.c) compares two ordinary load pairs per
group with one paired instruction whose range is 4095, covering two contiguous
2 KiB blocks. Registers read by the raw HMX words stay pinned in one asm block.

To fit the existing saved/restored 8 KiB scratch, activation and weight bases
deliberately alias. Each complete transaction computes
`groups * (A0 @ A0 + A1 @ A1)`. The blocks have different deterministic values;
cases use either two signed dense blocks or identity plus signed dense. This
does not qualify a distinct activation/weight working set or a model layer.

Groups=1/16/64/256 represent reduction depths 64/1024/4096/16384. Four complete
clear/load/store transactions run per job, all on worker 0. Every FP16 tile and
the final FP32 conversion match exact integer-derived references: 3,276,800
FP16 and 819,200 FP32 measured values. There are 50 samples at each of 16
case/group/mode points, plus two checked/excluded warmup rounds, with seeded
shuffled order. The runner stops on any numerical/protocol/restoration failure
or powered clock-configuration drift.

The comparison uses the corrected experimental entry from HMX_CACHE.md to
preserve FP during fault unwinding. No cached code is published, so both
cleanup markers remain zero and the cache-maintenance loop is skipped.

## Results

Signed-dense HMX p50, summed over four transactions:

| Groups | Control us | Batch2 us | Speedup |
|---|---:|---:|---:|
| 1 | 87.448 | 53.854 | 1.624x |
| 16 | 878.490 | 369.271 | 2.379x |
| 64 | 3,426.615 | 1,375.260 | 2.492x |
| 256 | 13,618.646 | 5,414.688 | 2.515x |

At 256 groups, batch2 p95/p99 HMX time is 5,462.135/5,472.708 us.
Resident slope estimates across both cases are 4.933–4.939 GMAC/s for the
control and 12.461–12.469 GMAC/s for batch2. The control includes an extra
diagnostic mode check/helper call; it is not the earlier 7.4 GMAC/s baseline.
Batching also amortizes helper and instruction-fetch costs. The measured ratio
cannot isolate hardware packet latency or predict its gain under cached code.

Signed-dense maximum-depth publication-to-result p50 is 14.773 ms for control
and 6.564 ms for batch2. Fresh-session p50 is 45.651 and 37.401 ms. Neither
figure includes a real model's distinct operand streaming, graph operators or
sustained tracking coexistence. Nearest-rank p99 is the maximum of 50 samples.

Maximum measured worker span is 14.335 ms and maximum SDK elapsed is 14.998 ms,
within the unchanged 40 ms host limit. All sessions restored state; the service
retained its PID with zero restarts. No recovery was needed.

## Reproduction

~~~sh
python3 handoff-tools/build_mw.py tests/entry_cached.S tests/hmx_batch2.c \
    build/hmx_batch2.fjob build/hmx_batch2

# On the supported device: inspect a small control before selecting both modes.
sudo python3 tests/hmx_batch2_run.py --run-on-device --probe build/hmx_batch2.fjob \
    --modes 0 --samples 1 --warmup 0 --groups 1 --transactions 1
sudo python3 tests/hmx_batch2_run.py --run-on-device --probe build/hmx_batch2.fjob \
    --modes 0 1 --samples 50 --warmup 2 --output /tmp/hmx-batch2.json
~~~

LLVM 23.1.4 produces 3,072 bytes, SHA-256 recorded in the evidence. The next
useful batching test needs separate activation and weight storage, an owned
larger scratch/tile-cache plan and cached execution. Preserve complete
clear/load/store ownership and bounded completion while making that change.
