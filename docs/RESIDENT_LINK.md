# Linked resident operators and cross-layer requests

`ResidentLandmark(graph, load_linked(folder), ...)` runs the pinned v20a graph
from one linked operator image instead of eleven modules copied into the code
slot one request at a time. A request may then cross phase and layer boundaries.
It ends between whole work units at a host tick budget, at a host stop point, at
frame completion or on a guarded fault. The eleven-module path is unchanged and
remains the default input; its modules still build to the same bytes.

Arithmetic, tensor layout, weights, the frozen numerical policy, the single HMX
owner, the 8 KiB VTCM backup, the 128 KiB borrowed job arena and the 40 ms host
timeout are unchanged. Data stays on its uncached aliases. What changes is
where operator instructions live, how often they are published, and how many
units one request may complete.

## Two images

| Image | Where | Size | Contents |
|---|---|---:|---|
| Loader | existing 4 KiB code slot, copied per request as before | 3,832 bytes | entry/fault vectors and the admin commands |
| Operators | owned DDR `0x1ff0000`, 12 KiB reservation | 9,336 bytes | the ten compute operators and `rg_batch` |

The reservation lies in previously unused pages between the activation
high-water mark (`0x1fef000`) and the packed-input page (`0x1ff4000`), with one
unused page on each side. Both the offline plan and the native program check
bound tensors below it. The image has no writable data or BSS.

The operator image is a single translation unit that includes the existing
operator sources. A handful of macros in `resident_graph.h` select the loop
bound, budget test and return path; without `RG_LINKED`/`RG_LOADER` they expand
to the previous text, and the eleven cache-entry modules reproduce their
qualified hashes. `tools/check_resident_link.py` verifies the linked outputs:
the loader's vector table and forwarding stubs (same rules as the cache entry),
its dispatch constants, the image entry at offset zero, PC-relative C
relocations only, no indirect calls or jumps other than returns, no cache or
exception-return instructions, and no embedded address of the image itself.
The same image bytes therefore run through either instruction alias.

```sh
python tools/build_resident_landmark.py --linked --output /path/to/link-jobs
python tools/check_resident_link.py --jobs /path/to/link-jobs --output /path/to/link-verification.json
```

## Entering the image

The loader entry is the cache entry with staging, publication and per-request
cleanup removed; the slot image always executes uncached. Its dispatch tail-jumps
to the operator image only for opcode 32 and only while resident state word 31
holds the seal. Otherwise the request reaches the loader's own admin code, which
rejects it. Context parameter zero selects the uncached (`0xf1ff0000`) or cached
(`0x88ff0000`) instruction alias. Fault vectors stay in the slot, so a guarded
synchronous fault inside the image returns through the same uncached recovery
path as before.

Two new loader commands bracket the image's lifetime:

- **Seal (7).** Allowed only when the immutable program is loaded and no seal is
  set. The host writes the image under the request's gate; the loader discards
  data- and instruction-cache lines for the whole reservation through the cached
  alias, then sets the seal. The host reads the image back before accepting the
  completion.
- **Retire (8).** Clears the seal and discards the same cache lines. Restoration
  commands are refused natively while the seal is set, so no original byte is
  written back under live cached instructions. Close issues it first, including
  after a confirmed fault or an unfinished frame.

Nothing writes the reservation through the cached alias, so lines are
discarded rather than written back. Between seal and retire there is no
per-request publication: the cache-entry path's roughly 1.5 ms of staging and
cleanup per request is not paid. The host compares all 12 KiB with the sealed
image before and after every request while the seal is set, and binds the image
and loader into the session identity.

## Batch requests

A batch descriptor carries the starting layer and progress (checked natively
against resident state), a unit limit, a tick budget of at most 480,000
reference ticks (25 ms), and a stop point `(layer, phase)`. `rg_batch` validates
the descriptor once, accepts the sequence once, then repeatedly runs the
operator that resident state selects. Each operator still proves it is the
expected one and that its tensors lie inside owned storage. The loop ends when:

- the frame completes or a guarded fault is recorded;
- the unit limit is reached or the tick budget has elapsed, both tested only
  between whole work units;
- a phase ends exactly at the stop point.

The host does not trust the reported position. Every unit's operator follows
from resident state alone, so the reported unit count determines the result:
`_batch_transition` replays that many units through the same independent state
machine used for single-module requests and requires the native state to match
word for word. A count that runs past the stop point, past frame completion or
past a fault is rejected, as is any PC outside the image at the selected alias.
Rejected requests must leave state and sequence unchanged.

Qualification stops once per layer, reads that layer's complete output and
applies the same local and propagated checks as before. Two extra stops exist
where a later phase overwrites what must be read: the strided in-place
downsample before compaction, and the logit before each sigmoid. Unqualified
execution uses no stop point and runs each request to its budget.

`budget_ms` defaults to the previously qualified five milliseconds. Raising it
is an explicit choice; the completion window is still 40 ms and a missed window
still poisons the session.

## Qualification and performance

Every run below uses the pinned v20a model, the four recorded crops and the
frozen policy (SHA256 `381a033d…951f7b`) of the earlier resident qualifications,
on the same firmware and boot as [the cache-entry baseline](RESIDENT_CACHE.md).

The main matrix uses cached instructions and a 20 ms budget: two sessions, two
crops each, one weight upload per session. All 308 layer checks and 20,351,360
layer-output values pass; the maximum local tolerance ratio is .80855, as
before. All 512 head values are bit-identical to the cache-entry, gather and
streamed baselines, verified from independently saved files. Each session then
repeats both crops without intermediate readback and reproduces the same head
bits. All 13,299 main restoration audits pass, the whole arena restores after
both sessions, and the DSP service keeps its PID with zero restarts.

| Audited frame, cached instructions | Cache entry, 11 modules | Linked, 5 ms | Linked, 20 ms |
|---|---:|---:|---:|
| Requests per frame | 10,959–10,964 | 5,974 | 1,566–1,567 |
| Frame body | 153.9–154.6 s | 89.6–89.7 s | 51.2–51.4 s |
| Native wrapper time per frame | about 49.5 s | 31.6 s | 30.6–30.8 s |
| Maximum wrapper / publication | 7.59 / 8.88 ms | 6.42 / 7.27 ms | 21.27 / 22.50 ms |

Without qualification readback the 20 ms frames take 48.8–49.0 seconds and
1,532–1,533 requests: **7.1× fewer requests and 3.1× less time** than the
cache-entry baseline. At the unchanged 5 ms budget the linked image alone
gives 1.72×, because phase boundaries, unit caps and per-request code
publication no longer end requests. The cache-entry figures are the historical
runs on the same boot, not an interleaved benchmark.

Request count is now set by native time, not by protocol boundaries. One frame
needs about 30.8 seconds inside the image, so a 20 ms budget cannot go below
roughly 1,540 requests. Depthwise convolution is 26.4 of those seconds (1,307
requests); the stem, every 1×1 convolution and dense head, pooling and residual
adds together take 4.3 seconds (240 requests). One request per layer, or one
per frame, therefore needs faster depthwise data access rather than more
batching: uncached program, state and tensor reads dominate each unit. Each
request also still pays about 7.7 ms of HMX power start and 1.6 ms of power
close on the host.

Supplemental runs, each restoring the full arena:

- Load, seal, retire and restore without entering the image: 445 audits.
- Six metadata rejections plus seven batch rejections (outside a frame; zero or
  excess budget; invalid stop layer/phase; zero units; reserved word), through
  the uncached and the cached alias: 458 audits each. State, sequence and every
  arena byte are unchanged after each rejection.
- Two-crop calibration at the 5 ms budget with identical heads: 24,280 audits.
  One-crop steps at 10 ms (3,058 requests, 65.4 s) and 20 ms.
- The same image through the uncached alias at 20 ms: identical heads, 92.8 s
  and 2,832 requests, every PC in `0xf1ff0000`; cached runs report `0x88ff0000`.
- Guarded faults after one HMX transaction and after one in-place depthwise
  unit: native state 4, reuse refused, retire then bounded close, full
  restoration (467 and 667 audits).
- Ordinary SDK direct/benchmark regression afterwards: 72 audits, 38 HMX.

No failed completion, reboot, gate reset, power recovery or service restart
occurred. Sanitized hashes, counts and timings are in
[resident-link-validation.json](../evidence/resident-link-validation.json).
These are audited graph runs on four crops, not sustained tracking latency,
coexistence, thermal or broader accuracy results.

## Running

```sh
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --jobs /path/to/link-jobs --model /private/model --corpus /private/corpus --load-only --controls --output /path/to/load.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --cached-code --budget-ms 20 --jobs /path/to/link-jobs --model /private/model --corpus /private/corpus --controls --unqualified-frames 2 --output /path/to/calibration.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --cached-code --budget-ms 20 --jobs /path/to/link-jobs --model /private/model --corpus /private/corpus --split heldout --calibration-report /path/to/calibration.json --output /path/to/heldout.json
```

Heldout requires a completed calibration with the same linked build, alias,
budget, sources, corpus and policy.

```python
from framenpu.resident_landmark import ResidentLandmark, load_linked

with ResidentLandmark(graph, load_linked("link-jobs"), cached_code=True, budget_ms=20) as session:
    heads = session.run(chw)
```
