# Single-plane 224x224 grayscale input

The optional DSP stem now accepts **one normalized 224x224 grayscale plane**,
with no host RGB expansion. It transfers **100,352 FP16 input bytes** instead
of 301,056 bytes. All checked prefix outputs and all four heads are
**byte-identical** to the existing RGB vector path on all four corpus crops.

This is an input-storage change with the original teacher weights and
arithmetic preserved. The input resolution was already 224x224; 112x112 is
the stem's output feature-map resolution.
[Evidence](../evidence/teacher-grayscale.json) contains the complete hardware
statistics, timing samples and source/build hashes, with no private images
or activations.

## Why this preserves the teacher

The existing [reference exporter](../tools/ncnn_reference.cpp) crops a
grayscale camera frame to 224x224 using the pinned affine transform, then
uses `PIXEL_GRAY2RGB` and applies the same FP32 `1/255.f` normalization to
all three channels. The [corpus builder](../tools/prepare_conv_reference.py)
requires grayscale PNG recordings and records replicated-grayscale
normalization. The hardware comparison verifies that each crop's three
normalized input planes have identical bytes.

The new stem stores one FP16 plane in VTCM and reads that plane for all
three logical input channels when building HMX tiles. It keeps the original
27-term K order, original per-channel weights and original biases. It never
expands three full planes on the DSP. This makes the packed HMX inputs match
replicated RGB exactly, including seams and bottom/right TF-SAME padding.
Everything after input packing is unchanged.

Descriptor kind **8** explicitly identifies grayscale input; the RGB image
continues using kind 7. The grayscale native checker validates a complete
784-vector input span and rejects RGB descriptors. The parameter table,
output descriptors and buffer offsets remain the same. Live input storage
saves 200,704 bytes; the existing VTCM ownership and DDR reservation are
preserved, and the unused region is not reassigned in this change.

## Paired hardware measurements

Four crops, ten rounds per crop, 40 frames per mode, alternating RGB/gray
and gray/RGB order. Host pinned to CPU2, same boot and normal power settings.

| Measurement | Replicated RGB | One grayscale plane |
|---|---:|---:|
| FP16 input transfer | 301,056 bytes | 100,352 bytes |
| Prefix DSP time including wrapper | 10.574 ms | 10.419 ms |
| Tail, global pooling and heads | 16.768 ms | 16.761 ms |
| Complete host wall | 41.811 ms | 41.813 ms |

The prefix difference is about **0.155 ms**. There is **no material host-wall
median improvement** in this run. The host ranges overlap: 40.549–43.266 ms
for RGB and 40.409–42.969 ms for grayscale. The median of paired host
differences is +0.594 ms and their mean is +0.209 ms, while the difference
between overall medians is -0.002 ms. These different statistics and the
roughly -1.99 to +2.50 ms paired spread do not establish a stable end-to-end
latency saving.

Frame timing includes converting/staging the preloaded normalized input,
both audited submissions and reading 512 result bytes. Sealing, parameter
installation, file IO, camera preprocessing and accuracy comparisons stay
outside timing. An independent qualification phase has another 40 grayscale
frames, with median prefix 10.339 ms and host wall 42.520 ms. This remains
the diagnostic Python/direct-dispatch harness, not a production native-host
latency measurement.

## Qualification and build

The actual C packer passes an offline check of every band/tile/feature
against replicated grayscale, including row seams and image edges. Removed
input planes are poisoned to catch accidental reads. The native descriptor
checker matches its host mirror on 4,000 mutations, including exact input
end bounds, overlap rejection and input-format separation. Planner tests
prove the parameter bytes and tail plan are identical to RGB.

Hardware checks cover all five prefix stages per band, exact pooling and
decimation from actual DSP intermediates, the per-layer tail references,
checkpoints 19/28/58/79, head incremental bounds, all existing ncnn ceilings,
guarded faults, descriptor rejection and sealing. All paired prefix outputs
and heads match the unchanged RGB vector image byte for byte. Maximum
landmark displacement remains 0.374491 px. All **677 restoration audits**,
including 675 HMX jobs, pass. Arena, window, reservation, gate, firmware,
power and service state are restored; no recovery/reset runs.

```sh
python3 tools/build_teacher_link.py --output /tmp/teacher-gray \
  --stride2 --heads --stem --stem-vector --stem-gray
```

The grayscale image is **10,768 bytes**, SHA256
`b2ef31c909e7a4c1361b783069befcbc207b4b0f578f0512cec365d0f853ef5e`.
The RGB vector control remains byte-identical at 10,800 bytes, SHA256
`ee1d8256c89cdd40caeb51c67b4c668d664a6c7219bb2e73fdd79c3d1523a214`.
The loader and per-layer reference images are unchanged. Native size,
relocation, writable-data, vector-stack and control-flow checks pass.

`GrayRunner.frame_gray(gray)` takes a normalized floating-point array with
shape `(224, 224)`. It stages one FP16 plane and runs the existing two-request
pipeline. Camera crop and normalization remain separate; raw camera bytes
are not the input boundary used for these measurements.

For the hardware qualification/comparison runner, pass matching
`--stride2 --heads --stem --stem-vector --stem-gray` flags and
`--gray-baseline-jobs` pointing to a current build of the qualified RGB
vector control. Profiling and the CPU-prefix comparison remain separate
runner modes. Use the existing lease, device-health and private-corpus
procedure.

## True one-channel convolution

For replicated grayscale, real arithmetic permits folding each output
filter's RGB weights into one spatial kernel:

`Wgray[o, y, x] = Wr[o, y, x] + Wg[o, y, x] + Wb[o, y, x]`.

The bias and preprocessing stay the same. This reduces the logical stem
from 27 to 9 taps, without training a new model. Two practical limits need
measurement before adopting it:

- Rounding the summed filters to FP16 changes **124 of 216 coefficients**.
  The largest coefficient change is 0.0078125; the largest per-filter sum of
  absolute changes is 0.0323486328125. That last value bounds only the
  coefficient-rounding contribution for input magnitudes at most one,
  before accumulation/output rounding, bias and clipping. Full-model output
  accuracy still needs checking.
- The current HMX operation pads K to **32** for either 27 or 9 logical taps.
  A folded kernel would still issue one HMX transaction per spatial tile.
  It could reduce packing work, but a 3x HMX compute reduction does not follow.

The next compute experiment should compare a folded HMX stem and a direct
nine-tap HVX stem against the current exact single-plane path, first for
accuracy through all heads, then for operator and host-wall timing. The
current implementation leaves weights intact as the exact control for that
comparison.
