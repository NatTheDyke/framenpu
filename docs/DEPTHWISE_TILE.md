# Small-map depthwise row tiling

Eleven of the sixteen depthwise layers produce rows only 7 or 14 values wide.
A work unit is one output row of one channel, so on those layers a full
gather and tap loop fills a quarter or half of a vector. The opt-in build
`--linked --dw-reuse 6 --dw-row-tile` computes four 7-wide or two 14-wide rows
of a channel in one pass. It builds on the
[coefficient and mask reuse](DEPTHWISE_REUSE.md) build and leaves the default
and reuse builds byte-identical to their qualified images.

## What changes and what does not

A separate operator, `rg_op_dwt` in `tests/resident_dw_tile.h`, handles a
layer only when its record fits: output 7 or 14 wide and square, not in place,
input row plus left padding at most 32 values, equal strides of one or two,
and input and output spans that do not overlap. Every other depthwise layer
runs the unchanged operator.

- **A unit is still one output row.** A pass puts each row in its own 32- or
  64-byte slot of one vector; each unit then stores only its own row, advances
  the same state and reaches the same budget and guarded-fault points. The
  host's unit replay is unchanged.
- **Rows computed ahead are never trusted across a request.** They wait in
  saved VTCM (`0x1880`, flags at `0x1900`) and are dropped when the request
  ends or the channel changes; the next request recomputes from the current
  row. No output row is written before its own unit.
- **Same arithmetic per lane.** Each lane sees the same taps in the same order
  with the same QFloat operations, zero identities and product and sum checks.
  The nonfinite flags are masked to the unit's own row.
- **Staging.** Input rows are staged 256 bytes apiece at the start of saved
  VTCM: the row is written at its left padding inside a zeroed pair of vectors,
  so a tap is one unaligned VTCM load and padding needs no mask. At most eleven
  rows end at `0xb00`, below the coefficient broadcasts at `0xc00`.
- **No vector on the job stack.** The pass and the per-row hand-out are
  separate functions that exchange vectors only through VTCM, so the unit loop
  holds no vector across a call.

## Wide code reservation

The tiled image is 13,420 bytes and does not fit the 12 KiB operator
reservation. The pinned plan leaves one more owned page between the image and
the packed-input scratch, so a tiled build uses a 16 KiB reservation at the
same address (`0x1ff0000..0x1ff4000`) with its own loader,
`resident_graph_loader_link_wide_v1`. A loader seals and retires exactly its
own reservation, the host takes the size from the loader name and still
requires the plan's high-water mark to lie at or below the image and the
reservation to end at or below the scratch, and it compares all 16 KiB with the
sealed image before every request. The arena itself is unchanged; this layout
leaves no free page between activations, image and scratch for this model.

## Results

All runs use the pinned v20a model, the four recorded crops, the frozen policy,
cached code, session-held HMX power and a 20 ms request budget on one boot.

| Per crop | Reuse build | Tiled build |
|---|---:|---:|
| Depthwise native time | 421 ms | 291 ms |
| of which 7- and 14-wide layers | 290 ms | 160 ms |
| Native time | 0.96 s | 0.81–0.82 s |
| Requests, timing frame | 50 | 43 |
| Frame time, timing frame | 1.28–1.38 s | 1.08–1.19 s |

Calibration and heldout qualify: 308 layer checks over 20,351,360 values pass
with the same maximum local ratio (.80855), and all heads are bit-identical to
the held-power baseline, checked independently from the saved files. Each split
also runs one frame whose tiled layers execute in requests of three or five
units, 12,590 requests that end at every offset inside a pass; its heads are
bit-identical too. The main runs pass 26,745 restoration audits. Guarded faults
after one HMX transaction, one in-place depthwise unit, one 14-wide tiled unit
and one 7-wide tiled unit invalidate the session, refuse reuse and restore the
arena (449, 462, 497 and 513 audits). Fourteen rejection controls and the
ordinary SDK regression pass. Maximum wrapper time is 20.23 ms and maximum
publication 21.96 ms. Results are in
[depthwise-row-tile-validation.json](../evidence/depthwise-row-tile-validation.json).

One earlier interrupted-pass run stopped with `DeviceBusy`: between two very
short requests a gate claim saw the stock firmware change its control word.
That stop is not a fault; the session closed with arena, callback, control and
power restored, and the identical run then passed. Ordinary frames issue about
forty requests, not thousands, but a caller should expect `DeviceBusy` from any
request and may retry the frame.

Offline, `tests/test_dw_tile.py` models staging, gathering, slot joining and
hand-out symbolically for every tiled geometry, both strides, every starting
row and channel position, and checks each consumed lane against direct
convolution indexing; it also builds the image and requires that the depthwise
functions spill no vector.

```sh
python tools/build_resident_landmark.py --linked --dw-reuse 6 --dw-row-tile --output /path/to/tile-jobs
python tools/check_resident_link.py --jobs /path/to/tile-jobs --output /path/to/tile-jobs/verify.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --cached-code --hold-power --budget-ms 20 --jobs /path/to/tile-jobs --model /private/model --corpus /private/corpus --controls --unqualified-frames 2 --interrupt-tiles --output /path/to/calibration.json
```
