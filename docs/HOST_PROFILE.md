# Host overhead in the fused resident graph

The ordinary API takes **762 ms per crop**, on average: **498 ms native** and
**264 ms on the host or waiting for publication/acknowledgment**. Host state
replay costs 150 ms, much more than the remaining completion checks combined
(3.65 ms). The next host optimization should replace per-unit Python replay
with bounded arithmetic transitions while preserving independent validation.

The broader tracking target now prioritizes the
[native execution redesign](NATIVE_EXECUTION_PLAN.md). Improving host replay
alone leaves the 498 ms native cost intact.

These measurements use the qualified 12,888-byte fused image, cached code,
session-held HMX power, one owner, a 20 ms request budget and the existing
40 ms timeout. Production executor and native code are unchanged.

## Wall-time breakdown

Means in milliseconds per crop, including input checking and conversion:

| Scope | Ordinary Device | Audited transport |
|---|---:|---:|
| Native wrapper | 498.27 | 498.14 |
| Host state replay | 150.03 | 141.28 |
| Input checking and call overhead before body timer | 41.08 | 46.33 |
| FP32 conversion, validation and packing | 33.10 | 30.83 |
| Publication/wait beyond native wrapper | 19.88 | 17.72 |
| Gate claim | 0.52 | 8.67 |
| Acknowledgment | 5.15 | 14.80 |
| Held-power verification | 4.57 | 4.42 |
| Other completion validation | 3.65 | 3.52 |
| Other submission work | 4.15 | 3.82 |
| Head readback and other body work | 1.91 | 2.36 |
| **Full API wall time** | **762.30** | **771.89** |

Rounding can affect the last decimal. The underlying per-frame partitions sum
to wall time. Replay is inside `read_validate_ms`; native wrapper time is
inside `publish_wait_ms`. Those nested scopes are subtracted before adding
the table rows.

The existing `body_ms` metric starts **after** input checking and `_fp32`.
Its ordinary mean is 688.12 ms; full `owner.run()` wall time adds 74.18 ms.
This explains why earlier body-only frame figures understated the API cost.
The input measurements here use recorded NumPy FP32 arrays loaded before
timing. Packing includes float conversion and a second finite/range scan.

## What the replay measurement means

Each frame has 27 requests, including 25 native batches, and 145,458 work
units. `_batch_transition` independently applies `_kind` and `_unit` once per
unit to derive the expected completion state. A host-only cProfile replay of
one saved frame confirms 145,458 calls to each, plus 83,552 `_next_row` calls.
That cProfile pass runs after every HMX owner closes; its inflated durations
are excluded from live time attribution.

The remaining 3.65 ms of completion validation includes worker identity,
descriptor/state checks, guards, control boundaries, program and image
comparisons. Even eliminating that entire remainder would save much less
than improving replay. This profile does not time each integrity comparison
separately.

The next change should advance multiple units arithmetically within a phase,
then cross phase/layer boundaries with the same count, stop-point, fault and
terminal-state checks. Retain the current unit replay as an independent test
oracle, including partial packing/HMX groups, depthwise rows, compaction,
sigmoid and faults. Do not trust the device's reported final state directly
or raise the request budget.

After replay, target redundant input scans and conversion for contiguous typed
FP32 inputs while retaining shape, finite/range validation and support for
general iterables. Native depthwise remains the next compute bottleneck:
even removing every measured host cost leaves about 498 ms per crop.

## Method and validation

`tests/hardware_host_profile.py` verifies matching completed calibration and
heldout reports, source/image/model/corpus pins and the frozen policy before
running. Four sequential sessions use audited, ordinary, ordinary, audited
transport. Each loads weights once, then runs two warmups, two controls with
the extra timing collectors disabled and eight profiled frames. The four
recorded inputs alternate; all 48 frames compare every head bit against their
qualified crop, for 6,144 checked FP32 values.

Ordinary transport retains every production ownership, replay, integrity,
power and restoration check. Audited transport adds harness snapshots and
restoration readbacks around each job. Both modes receive independent full
resident-arena and firmware/power/service checks after each session. All pass;
the audited sessions complete 1,538 restoration audits. No failures, retries,
reboots or service restarts occur.

There are 16 profiled samples per mode. Ordinary wall time ranges from
724.08 to 802.05 ms, median 764.00 ms. Host scheduling and CPU frequency were
not controlled; same-crop control/profile differences change sign between blocks.
The larger claim/ack scopes attribute about 18 ms to audit activity, but the
whole-frame mode difference is not a precise audit-only speedup. Camera
capture, crop generation, preprocessing, IPC and sustained tracking/thermals
are outside this measurement.

Results, hashes, ranges and function counts are preserved in
[host-profile.json](../evidence/host-profile.json). Three offline tests check
that the partition counts nested scopes once and refuses inconsistent times.

```sh
sudo .venv/bin/python tests/hardware_host_profile.py --run-on-device --jobs /private/fast-jobs --model /private/model --corpus /private/corpus --calibration-report /private/calibration.json --heldout-report /private/heldout.json --output /private/host-profile.json
```
