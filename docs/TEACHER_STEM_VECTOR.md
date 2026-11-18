# Vector pooling and band stitching

Vectorizing the 112x112 prefix's max-pool branch and stride-2 output gather
reduces their combined measured time from **5.215 ms to 0.056 ms**. In paired
complete-frame measurements, the prefix falls from **15.676 ms to 10.559 ms**
and host wall falls from **47.942 ms to 42.201 ms**. All checked prefix outputs
and four heads remain byte-identical to the scalar control.

[Evidence](../evidence/teacher-stem-vector.json) contains the timing samples,
per-band profiles, accuracy statistics, restoration audits and build hashes.
Private images and activations stay on the device.

## Paired measurements

Four crops, six rounds per crop, CPU2, unchanged normal power configuration.
The scalar and vector images execute on the same boot with rotated/reversed
mode order. Sealing and fixed parameter/table setup are outside frame timing.

| Complete RGB-to-head frame | Scalar | Vector |
|---|---:|---:|
| Prefix DSP time, including wrapper | 15.676 ms | 10.559 ms |
| Tail, global pooling and heads | 16.782 ms | 16.794 ms |
| Sum of DSP requests | 32.449 ms | 27.356 ms |
| Host wall | 47.942 ms | 42.201 ms |

Each column has 24 frames. The median of paired host-wall differences is
**5.785 ms**; the difference between column medians is **5.741 ms**, or 12.0%.
Host wall includes input FP16 conversion/staging, two audited submissions
and reading 512 bytes of results. It excludes file IO, camera preprocessing,
accuracy checks and image switching. This remains a Python/direct-dispatch
diagnostic, not a production native-host latency measurement.

Separate prefix-only samples measure 15.756 ms scalar, 10.608 ms vector,
10.587 ms vector with profiling disabled, and 10.626 ms with coarse counters.
Each mode has 24 samples. The independent qualification phase measures
10.510 ms prefix and 42.491 ms host wall across another 24 frames.

| Coarse bracket across seven bands | Scalar profile | Vector profile |
|---|---:|---:|
| 2x2 max-pool plus VTCM stitching | 2.760 ms | 0.0246 ms |
| Stride-2 gather plus VTCM stitching | 2.455 ms | 0.0310 ms |
| Combined | 5.215 ms | 0.0556 ms |

The scalar bracket values come from the preceding
[profile](TEACHER_STEM_PROFILE.md) on the same boot; the full-prefix and
full-frame comparisons above are paired in this run. The remaining operator
times stay near their previous values. Final output DDR stores still take
0.481 ms. Sampled HMX transactions plus their preceding completion fence
remain about 6.17 ms. The optimization changes output selection/movement,
with the convolution arithmetic and its fences preserved.

The previous [CPU FP32 prefix comparison](TEACHER_PREFIX_COMPARISON.md)
measured 29.025 ms host wall. That remains faster than this DSP-prefix path;
it was not rerun as part of this vector/scalar comparison.

## Implementation and qualification

`teacher_stem_vector.h` loads contiguous 128-byte vectors and deinterleaves
32-bit words, each holding two FP16 channels. Two chunks produce each
56-word output row. The second chunk blends its 24 live words with the
existing destination, preserving padding and neighboring rows.

Pooling compares finite FP16 values through signed integer order keys.
Both signed zeros map to the same key, and ties select the second operand.
This preserves the scalar maximum's signed-zero behavior and ordered 2x2
reduction. No FP16 value is converted or rounded by the gather/stitch path.

Host tests execute the actual vector C with portable intrinsic semantics:
every finite-half encoding against sign/exponent boundary values, random
finite tensors, signed-zero combinations, all row alignments and all seven
bands in nonsequential order. They compare the entire owned VTCM buffer
against the scalar implementation, and check every vector access stays
inside its channel plane. Hardware qualification checks every stage's
existing error bound, exact pooling/decimation from actual DSP intermediates,
tail checkpoints, all heads, guarded faults, descriptor rejection and sealing.
Every profile mode and single-band replay matches the scalar output bytes.
The worst landmark error remains 0.374491 px, with existing ceilings unchanged.

All **992 restoration audits**, including 990 HMX jobs, pass. The arena,
window, code reservation, gate, firmware, power and service state are restored.
The native budget, VTCM ownership and DDR reservation are unchanged.

## Build and reproduce

The optimization is selectable, preserving the scalar control build:

```sh
python3 tools/build_teacher_link.py --output /tmp/teacher-vector \
  --stride2 --heads --stem --stem-vector
```

The vector image is **10,800 bytes**, SHA256
`ee1d8256c89cdd40caeb51c67b4c668d664a6c7219bb2e73fdd79c3d1523a214`.
The scalar image remains byte-identical at 10,800 bytes, SHA256
`8c36451dd0e91b8122d867d66cbb6cdd419b312fe1125ab40fc9608c5cf327b9`.
The loader and per-layer reference images are unchanged. Both builds pass
relocation, vector-stack, writable-data, size and control-flow checks.

For profiling, add `--stem-profile` to the vector build (14,064 bytes,
SHA256 `26cc6dbf232dab29056f60c2424100ae158077143fb1730fd7782912bac78e36`).
The hardware runner takes matching `--stem-vector --stem-profile` flags,
`--profile-baseline-jobs` for the scalar build and `--profile-vector-jobs`
for the vector build without profiling code. Use the existing device lease,
health checks and private corpus procedure; no recovery or gate reset is part
of the run.
