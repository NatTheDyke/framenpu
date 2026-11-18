# Banded RGB prefix on the DSP

The optional linked teacher path now runs from a normalized 3x224x224 RGB
array through all four heads in **two requests**. The first request produces
both 56x56 tail inputs: blob 8 from the convolution branch and blob 10 from
the max-pool skip branch. The second runs the existing tail, global-average
pool and FP32 heads. Only the final **512 bytes** return to the host.

Four corpus crops and twenty timed frames pass the existing accuracy
ceilings. [Hardware evidence](../evidence/teacher-stem-banded.json) contains
statistics and source/build hashes, with no private images or activations.

## Measured timing

| Work | Median DSP time including wrapper | Observed range |
|---|---:|---:|
| Banded prefix | 15.710 ms | 15.562–15.799 ms |
| Tail, pool and heads | 16.733 ms | 16.651–16.821 ms |
| Sum per frame | 32.444 ms | |
| Complete host wall | 56.754 ms | 48.308–66.485 ms |

Both requests remain below the unchanged 20 ms native budget. The complete
host measurement includes converting preloaded normalized RGB to FP16,
staging 301,056 input bytes, both audited submissions and reading the result.
The reference loader represents the pinned FP32 corpus values in FP64 arrays.
Fixed parameters and tables are installed once before timing. Corpus file
loading, camera crop/normalization, accuracy comparisons and intermediate
readback are excluded. The measurement uses the diagnostic Python/direct
dispatch harness; it is not a production native-host latency result.

The previous 29.256 ms [heads result](TEACHER_HEADS.md) starts from ncnn
blobs 8 and 10. It excludes the prefix and has a different input boundary,
so it is not an equivalent full-model baseline or a speedup comparison.

## Bands, halos and memory

Seven bands each produce eight 56-wide output rows. A band holds 20 rows
at width 112, beginning at global row `16*band-1`. DW0 needs stem rows
0..18 in local coordinates. DW1 samples local centers 2,4,..16 and odd
columns, implementing the original bottom/right TF-SAME stride-2 alignment.
The max-pool branch uses projection rows 1..16 in 2x2 windows. Masks after
every biased stage zero rows outside the actual 112-high image, including
the top and bottom bands. Band-local boundary padding cannot replace these
global image masks.

Each band executes stem -> DW0 -> projection, max-pools the projection into
blob 10, then executes expansion -> DW1 and stitches blob 8. The full RGB
input and both outputs remain resident while the two band buffers alternate.
RGB is kept as three channels. The stem uses contiguous HVX loads, even-half
shuffles and row-seam merges to form each 27-feature HMX tile; its padding
exists only in that tile. Input predicates also handle bottom/right padding.

| VTCM object | First vector | Maximum vectors |
|---|---:|---:|
| RGB FP16 | 4 | 2352 |
| Blob 8, padded 56x56 grid | 2356 | 3616 |
| Blob 10, padded 56x56 grid | 5972 | 904 |
| Band A | 6880 | 2848 |
| Band B | 9728 | 2848 |
| Parameters and scratch | 12576 | 433 |

The last possible vector end is 13009, within the existing 15296-vector
ownership. The tail reuses that same ownership in its next request. RGB
occupies the tail plan's unused final-map DDR slot. Prefix stores target the
tail's existing input slots; no host readback or repacking occurs between
requests. All fixed prefix images occupy 4704 vectors, 602,112 bytes, appended
after the tail/head parameters within the existing DDR window.

DW1 skips 36 of its 72 vector blocks because they contain no required
stride-2 outputs. Selected blocks retain the original tap order and output
bits. This reduced the qualified wide-arithmetic prefix from about 21.2 ms
to 15.7 ms, including coefficient pre-splatting. It still computes unused
lanes inside retained blocks; a fully sampled-only kernel remains future work.

## Precision and qualification

Tensor storage and HMX weights are FP16. The prefix retains original FP32
biases and adds them through QFloat32 before FP16 output storage. Its two
depthwise reductions use QFloat32 and round to FP16 before the bias add.
The existing tail's QFloat16 arithmetic is unchanged.

Early-layer error reaches every later layer. Reusing the tail's QFloat16
depthwise and FP16-bias path produced 2.422 pixels of maximum landmark drift
on the first crop, failing the frozen ceilings. Widening the depthwise and
bias-add arithmetic while retaining FP16 biases reduced this to 0.761 pixels,
but screen-head RMS was 0.293 against a 0.25 ceiling. Keeping original FP32
biases resolved that failure. No accuracy ceiling was widened.

Final worst errors against the pinned FP32 ncnn reference:

| Quantity | Measured maximum | Existing ceiling |
|---|---:|---:|
| Decoded 2D landmark displacement | 0.374492 pixels | 1 pixel |
| Screen-head absolute error | 0.372147 | 1 |
| Screen-head RMS, worst crop | 0.131274 | 0.25 |
| World-head absolute error | 0.000575264 | 0.003 |
| Presence absolute error | 0.002551616 | 0.005 |
| Handedness absolute error | 0.000162357 | 0.005 |

All five stages of each band are observed independently. Every stage is
checked against the CPU numerical contract using its actual preceding DSP
input, with a maximum bound ratio of 0.256474. DW1 is checked at its defined
stride-2 outputs; other temporary positions are discarded. Max-pool and
decimation must match the observed inputs exactly. Outputs are deterministic
across the five diagnostic passes. The unchanged per-layer reference kernels
then qualify the tail from those actual prefix outputs; observed tail
checkpoints 19/28/58/79 match byte for byte. Pool/head incremental checks use
the existing frozen caps. All twenty RGB frames match those observed results.

Six guarded stage faults, eighteen descriptor rejections, second-seal and
unsealed-request refusals pass, followed by a clean RGB frame. All 384
restoration audits (382 HMX) pass. Arena, reservation, gate, callback, power
and firmware checks restore. The service PID/restart count and device boot
remain unchanged. No recovery, restart or automatic continuation ran.

## Build and run

```sh
python tools/build_teacher_link.py --stride2 --heads --stem --output /path/to/jobs
sudo .venv/bin/python tests/hardware_teacher_link.py --run-on-device --stride2 --heads --stem --jobs /path/to/jobs --model /path/to/model --corpus /path/to/corpus --output /path/to/result.json --rounds 5
```

The image is 10,800 bytes, leaving 5,456 bytes before the seal. It passes
relocation, vector-stack, writable-data, control-flow and image-entry checks.
The loader, original 4 KiB qualification kernels, and heads-only linked image
remain byte-identical. Build and run flags must agree.

Kind 7 fixes the geometry and VTCM layout. Descriptor words 0..14 are
`7, first_band, end_band, 8, 20, rgb_ddr, parameters_ddr, 4704, 672,
out8_ddr, out10_ddr, 4, 2356, 5972, 6880`. Word 15 selects a guarded fault
after stage 1..5, or max-pool (6); zero is normal. Word 16 observes stage
1..5, with its DDR destination in word 17; both are zero normally. Words
18..31 must be zero. The native checker validates complete DDR spans and
rejects all input/parameter/output/debug aliases before execution. Partial
band ranges are diagnostic; normal frames explicitly request all seven.

Offline tests cover the actual host-compiled checker and packer, all finite
half patterns in scalar max-pooling, every RGB tile/seam/edge, global halo
stitching, vector read bounds, sampled-block coverage, VTCM liveness, composite
DDR layout and unchanged reference images. The full offline suite passes
276 tests. This is corpus qualification, not sustained tracking coexistence
or thermal/power validation.
