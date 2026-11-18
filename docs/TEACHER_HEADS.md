# Resident global-average pooling and FP32 heads

The FP16 teacher tail now has an optional DSP epilogue: global-average pool
the 672x7x7 tensor, run all four FP32 heads, and apply the two sigmoids.
It stays in **one request** and returns **128 FP32 values, 512 bytes**.
The 112x112 prefix is still outside this path; inputs remain blobs 8 and 10.

Hardware qualification passed on all four corpus crops. Evidence:
[teacher-heads-fp32.json](../evidence/teacher-heads-fp32.json), statistics only.

## Timing

Twenty frames per mode, four crops x five rounds. The modes alternate within
each round, and the order also alternates between rounds. Both modes execute
the same resident convolution tail in the same linked image and one request.

| Median per crop | DSP tail + Python host pool/heads | DSP tail + DSP pool/heads |
|---|---:|---:|
| In-DSP, including wrapper | 15.750 ms | **16.744 ms** |
| Kernel body | 15.720 ms | **16.715 ms** |
| Host wall through all four outputs | 37.908 ms | **29.256 ms** |
| Activation/output readback | 301,056 B FP16 grid | **512 B FP32 outputs** |

The new epilogue descriptor takes about 1.238 ms including its FP32 parameter
copy. Removing the old final feature-map store offsets part of that cost:
the net DSP increase is about 0.994 ms. DSP samples with heads span
16.641–16.781 ms, below the unchanged 20 ms native budget.

The measured host-wall reduction is 8.652 ms (22.8%) against the **current
Python host-head reference path**. That baseline includes unpacking the
padded feature grid and the existing NumPy reference's pooling/head work,
including preparation of its weights. It is not an optimized native CPU-head
benchmark, a whole-model latency result or a tracking/power/thermal soak.
Earlier tail-only host-wall numbers excluded CPU pooling and heads, so they
are not the equivalent baseline for this comparison.

Both modes time input staging, submission and their output readback. Plan and
parameter installation when switching benchmark modes is setup outside timing.
Numerical comparisons and diagnostic checkpoint readback are outside timing.

## Arithmetic and qualification

The convolution tail retains its FP16/QFloat16 contract. Its final grid is
consumed directly in VTCM; the normal plan never stores that grid to DDR.

The epilogue converts the 49 half values per channel exactly to FP32 and
accumulates them in row-major order, followed by an FP32 `1/49` multiply.
The 672 pooled values stay in FP32. All four learned head matrices and biases
are stored as FP32, with no additional FP16 activation or weight rounding.
Head products and sums use the existing HVX QFloat32-to-FP32 helpers and
their zero-identity handling. The two scalar sigmoid outputs use the bounded
approximation already present in the qualified resident graph.

The host oracle accumulates pooling and head products in FP64 (with the
existing pooled-vector FP32 cast). Therefore the new outputs are qualified
numerically against that oracle, with frozen incremental caps in
`tests/teacher_heads_layout.py`, rather than required to match it bit for bit.

| Incremental max absolute error vs host heads of the same tail | Measured | Frozen cap |
|---|---:|---:|
| Pooled FP32 vector | 4.768e-7 | 2e-5 |
| Screen-landmark head | 1.735e-4 crop pixels | 0.01 |
| World-landmark head | 2.696e-7 | 1e-4 |
| Presence | 6.958e-7 | 2e-5 |
| Handedness | 5.309e-8 | 2e-5 |

All existing ncnn head ceilings also pass: screen coordinates max abs 1.0
and RMS 0.25, world max abs 0.003, presence/handedness max abs 0.005. The
maximum decoded 2D landmark drift is 0.385464 crop pixels, versus 0.385443
for the previous host-head calculation. Maximum world drift is 4.501e-4,
presence 2.836e-3 and handedness 3.546e-4.

On every crop, the observed convolution tensors 19/28/58/79 are byte-identical
to the unchanged per-layer qualification kernels. DSP head outputs are
byte-identical between the diagnostic frame, all timing frames and the clean
frame after controls.

## Plan, image and bounds

Build with:

```sh
python tools/build_teacher_link.py --stride2 --heads --output /path/to/jobs
```

Only the optional linked image contains the epilogue. The slot loader,
ordinary qualification kernels and default two-block merged image retain
their qualified machine code. The new image is 7,312 bytes, leaving 8,944
bytes before the seal vector. Relocation, vector-stack, writable-data,
control-flow and image-entry checks pass.

The normal plan has **46 descriptors**: two input loads, 42 tail compute ops,
one fused pool/head op and one 512-byte store. The observed plan has 51,
including the three stride-2 outputs, final tail tensor and pooled vector.
Peak VTCM remains 14,099 vectors in the existing ownership span.

Descriptor kind 6 is fixed to the pinned teacher's shapes. Words 0..14 are
`6, 336, 4, 672, 49, 7, 1, 7, 9, dpar, 2692, vin, vout, vpool, vpar`.
Word 15 selects a diagnostic trap (0 normal, 1 after GAP, 2 after heads);
words 16..31 must be zero. The FP32 image is bias[128] followed by
weights[672][128], 344,576 bytes. Result order is screen landmarks (63),
world landmarks (63), presence (1), handedness (1).

The native checker validates the full input, output, pooled and parameter
spans and rejects overlap between any two. Window bounds and unused fields
are checked before execution. The planner reuses dead tail storage only
after the final feature tensor has been reserved. No larger VTCM or DDR
ownership, native budget or host timeout is introduced.

The device harness refuses a build/flag mismatch. Fault controls after GAP
and after heads bypass the job-entry diagnostic trap so the epilogue itself
is exercised. Seventeen malformed descriptors, second-seal refusal and
unsealed-request refusal also pass, followed by a clean result.

All **238 restoration audits (236 HMX)** pass. The full arena hash, image
reservation, power configuration and firmware checks restore; the service
PID and restart count are unchanged. No retry, recovery or gate reset ran.
On an uncertain completion the harness now retains the window and sealed
image; an offline regression checks that cleanup does not write them.

Run on the device with fresh staged sources and jobs:

```sh
sudo .venv/bin/python tests/hardware_teacher_link.py --run-on-device --stride2 --heads --jobs /path/to/jobs --model /path/to/model --corpus /path/to/corpus --output /path/to/result.json --rounds 5
```

Offline checks cover the host-compiled native bounds checker and 4,000
mutations, all pairwise buffer aliases, sigmoid accuracy, full-tail planning
and numerical simulation, unchanged reference builds, diagnostic dispatch,
and uncertain-completion resource retention.

The optional [banded RGB/112x112 prefix](TEACHER_STEM.md) now produces both
tail inputs in a separate request feeding this tail-plus-heads request.
Its full-model measurements and precision contract are documented separately.
