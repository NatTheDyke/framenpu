# Four-block depthwise coefficient sharing

Four-block sharing is implemented and hardware-qualified as a selectable
linked-image build. It did **not** improve measured tail latency, so the
qualified two-block build remains the default.

Evidence: [teacher-dw4-fp16.json](../evidence/teacher-dw4-fp16.json). It includes
the complete two- and four-block run statistics, build records, per-op timing
summaries and a static assessment of sampled stride-2 computation.

## Implementation

`RS_DW_BLOCKS=4` gives the resident depthwise loop four independent
accumulators and shares each coefficient load across four consecutive output
blocks. Tap order, QFloat16 arithmetic, FP16 rounding, bias, clipping, masks
and scalar stride-2 decimation are unchanged. Both stride-1 and stride-2
depthwise ops use this loop.

The native descriptor checker requires `nj` to be divisible by the selected
block count. Every existing depthwise plan already uses `nj4`. A new device
control supplies a two-block remainder and requires native rejection by the
four-block build. The host-compiled C checker and its Python mirror are tested
in both configurations against valid plans and 4,000 descriptor mutations.

The four-block image is 5,616 bytes, 128 bytes larger than the two-block image,
and passes the relocation, vector-stack, writable-data and linked-image
checks. The loader, ordinary qualification kernels, default seven-request
image and two-block merged image remain byte-identical to their qualifications.
The descriptor format, VTCM plan and timeouts do not change.

```sh
python tools/build_teacher_link.py --stride2 --dw-blocks 4 --output /path/to/jobs4
sudo .venv/bin/python tests/hardware_teacher_link.py --run-on-device --stride2 --dw-blocks 4 --jobs /path/to/jobs4 --model /path/to/model --corpus /path/to/corpus --output /path/to/result4.json --rounds 5
```

Use `--dw-blocks 2` for the control. Omitting the option keeps two blocks.
Build records and harness flags must agree.

## Hardware comparison

Two sequential full runs on the same boot, first two blocks, then four.
Each run alternates the seven-request cached control with its one-request
merged linked frames. The block-count variants themselves are not interleaved.
Each variant has four crops and five measured rounds per crop.

| Measure | Two blocks | Four blocks |
|---|---:|---:|
| Median in-DSP tail | 15.773 ms | 15.849 ms |
| Median kernel body | 15.738 ms | 15.814 ms |
| Median host wall | 26.508 ms | 27.714 ms |
| DSP p95 | 15.904 ms | 15.897 ms |
| DSP maximum | 15.919 ms | 16.041 ms |
| Stride-1 depthwise, median sum | 3.783 ms | 3.835 ms |
| Stride-2 depthwise, median sum | 2.881 ms | 2.915 ms |
| 1x1 ops, median sum | 8.331 ms | 8.324 ms |

The four-block median is about 0.075 ms higher. These samples support keeping
two blocks as the default; they do not establish a universal regression.
Sharing more coefficient loads alone has not reduced the measured depthwise
cost. The disassembly still contains unaligned input load handling and the
same post-processing call for each output vector; their separate costs were
not isolated here.

Both variants pass all accuracy and restoration checks:

- Observed tensors 19/28/58/79 match the unchanged per-layer qualification on
  all four crops. Final tensors match on all timed and post-control frames.
- Qualification head statistics are identical between variants: maximum
  landmark drift remains 0.38544 px and presence delta 2.837e-3.
- Fault, descriptor, seal and unsealed-request controls pass. Four blocks
  also rejects the two-block remainder as expected.
- 359 restoration audits for two blocks and 360 for four (357/358 HMX),
  identical arena hashes and power/firmware checks, restored reservation,
  unchanged service PID and no restart, retry or recovery.

All 260 offline tests passed, including 19 teacher tests. Hardware timing
covers the tail only; the stem, pooling and heads remain outside it.

## Computing only stride-2 output positions

The next useful prototype is the 56x56-to-28x28 layer, `convdwclip_3`, which
still takes about 1.70 ms with two blocks. Its desired input coordinate for
tap `(dy, dx)` is `(2*y + dy - pad, 2*x + dx - pad)`, where `pad = k//2-1`.
This is exactly the current `(2*y+1, 2*x+1)` decimator's coordinate before
the depthwise tap offset. Every scalar address in all three layer geometries
was checked against that identity and the existing input grid bounds.

One HVX vector holds 32 channel-pair words. The static vector work below
counts tap multiplication/accumulation vectors, including padded lanes,
across every channel pair and tap:

| Layer | Full-resolution work now | One output row/vector | Packed sampled rows |
|---|---:|---:|---:|
| `convdwclip_3`, 56→28, k5 | 124,800 | 33,600 | 33,600 (one row) |
| `convdwclip_5`, 28→14, k3 | 18,144 | 9,072 | 4,536 (two rows) |
| `convdwclip_11`, 14→7, k5 | 38,400 | 33,600 | 9,600 (four rows) |

These are operation counts, not speed forecasts. A row of 28 outputs needs
two input vectors and a word deinterleave for each tap. Rows of 14 or 7 can
use one input vector per row, but need two or four rows packed together to
avoid mostly empty SIMD lanes. Existing `resident_graph_gather.h` and
`resident_dw_tile.h` show the relevant word-deinterleave and row-packing
instructions in a different, FP32 path; they are not qualification of a new
FP16 stride-2 kernel.

A bounded next implementation can retain the existing descriptor and output
grid, replace full-resolution arithmetic with sampled row tiles, then scatter
only the useful words. It must preserve the per-output tap order and rounding,
and prove the **whole vector-load span**, dummy-row handling, packed-lane masks
and output stores before hardware use. The scalar address check above does
not prove those new vector spans. Measure gathering, arithmetic and output
packing together, starting with `convdwclip_3`; skipping three quarters of
the logical outputs does not promise a fourfold layer speedup.

No sampled-only native kernel was added or run in this step.
