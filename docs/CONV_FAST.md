# Fused 1x1 convolution path

After [row tiling](DEPTHWISE_TILE.md), the convolution layers (the 3x3 stem and
the 1x1 layers, including the heads) were the largest part of native time:
about 0.50 of 0.82 seconds per crop. The opt-in build
`--linked --dw-reuse 6 --dw-row-tile --conv-fast` brings them to 0.18 seconds.
It builds on the tiled build, uses the same 16 KiB reservation and loader, and
leaves the default, reuse and tiled builds byte-identical to their qualified
images.

## What changes and what does not

A convolution layer runs in groups of 32 positions. Each group was three
phases: pack the inputs to half floats, run HMX over 32 output channels at a
time, then a separate bias and clip unit per output channel.

- **Bias and clip move into the HMX unit.** The HMX operator already reads
  each result row back as floats and stores it. It now adds the channel bias,
  flags a non-finite sum and clips before that store. The separate unit, which
  reloaded the row it had just stored through uncached DDR, is gone. Each
  value sees the same operations in the same order: HMX result, non-finite
  check, bias add, non-finite check, clip.
- **Fewer work units, same replay discipline.** A frame is 145,458 units
  instead of 240,786; the difference is exactly the bias and clip units. The
  last HMX unit of a group now performs the state step the last bias unit
  performed, including the hand-over to the sigmoid phase on the heads. The
  host replays this shorter sequence only for an image whose build record says
  `conv_fast`; a wrong flag fails replay on the first convolution group.
- **No vector on the job stack.** One vector store and reload on the job stack
  costs about 1.4 µs, and the packing, stem and HMX units each did five or
  more. Packing and HMX row handling now run in functions of their own
  (`rg_pack_unit`, `rg_stem_unit`, `hmx_rows`) that hold no vector across a
  call; the two packed halves wait in saved VTCM during conversion. The fused
  image contains no vector stack access at all.
- **Unchanged:** tensor layout, HMX transactions, weights, the 20 ms budget
  and 40 ms timeout, guarded-fault points on every remaining unit, depthwise
  and all other operators, and the frozen policy.

## Results

All runs use the pinned v20a model, the four recorded crops, the frozen policy,
cached code, session-held HMX power and a 20 ms request budget on one boot.

| Per crop | Tiled build | Fused build |
|---|---:|---:|
| Convolution native time | 497 ms | 182 ms |
| Native time | 0.81–0.82 s | 0.50 s |
| Work units | 240,786 | 145,458 |
| Requests, timing frame | 43 | 27 |
| Frame time, timing frame | 1.12–1.21 s | 0.67–0.77 s |

Calibration and heldout qualify: 308 layer checks over 20,351,360 values pass
with the same maximum local ratio per crop as the tiled build (.80855 overall),
and all heads are bit-identical to the tiled baseline, checked independently
from the saved files. Each split also runs the interrupted-pass frame of the
tiled layers (12,590 short requests) with bit-identical heads. The main runs
pass 26,615 restoration audits. A run at a 5 ms budget, which ends requests at
different units, produces the same heads in 101 requests.

Nine guarded faults invalidate the session, refuse reuse and restore the
arena: after one stem packing unit, one stem HMX unit, one 1x1 packing unit,
one unclipped, one clipped and one head HMX unit, one in-place depthwise unit
and one 14-wide and one 7-wide tiled unit. Fourteen rejection controls, a
tiled-build control on the same boot and the ordinary SDK regression pass.
Maximum wrapper time is 20.14 ms and maximum publication 22.00 ms. Results are
in [conv-fast-validation.json](../evidence/conv-fast-validation.json).

Offline, `tests/test_resident_link.py` checks that the fused unit sequence
reaches the same state at every layer boundary and differs from the ordinary
one by exactly the bias units, runs a fused frame through the host against the
transport model and requires a wrong flag to be refused.
`tests/test_conv_fast.py` builds the image and requires that it contains no
vector stack access and that the tiled image is unchanged.

```sh
python tools/build_resident_landmark.py --linked --dw-reuse 6 --dw-row-tile --conv-fast --output /path/to/fast-jobs
python tools/check_resident_link.py --jobs /path/to/fast-jobs --output /path/to/fast-jobs/verify.json
sudo .venv/bin/python tests/hardware_resident_landmark.py --run-on-device --linked --cached-code --hold-power --budget-ms 20 --jobs /path/to/fast-jobs --model /private/model --corpus /private/corpus --controls --unqualified-frames 2 --interrupt-tiles --output /path/to/calibration.json
```
