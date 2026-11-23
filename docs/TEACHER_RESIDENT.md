# Resident teacher tail: 7 jobs per frame instead of 42

This document follows [TEACHER_TAIL_FP16.md](TEACHER_TAIL_FP16.md). That
document ran the unchanged teacher from the 56x56 stage to the heads as 42
one-layer jobs. Most of that time was job overhead and moving activations
between DDR and VTCM.

Here, one kernel runs a whole sequence of 1x1 and stride-1 depthwise layers in
a single job, with tensors kept in VTCM between them. Only the three stride-2
depthwise layers still run as their own jobs. They use the same opcode-110
kernel, with a byte-identical image.

**Result:** every check passed on all four corpus crops. Every tensor the
resident frames compared was bit-identical to the per-layer qualification run,
and the accuracy is unchanged. Per frame (median of 20 frames):

| | Per-layer jobs (previous run) | Resident (this run) |
|---|---:|---:|
| Jobs | 42 | 7 |
| In-DSP time, including wrappers | 96.2 ms | **30.0 ms** |
| Kernel body | 27.5 ms | 18.6 ms |
| Host wall time | 430 ms | 94.6 ms |

Evidence: [evidence/teacher-resident-fp16.json](../evidence/teacher-resident-fp16.json)
(statistics only).

## Kernel (`kernels/teacher_resident_probe.c`, opcode 111, 4,068 B)

**Inputs.** A job runs descriptors `params[1] .. params[2]-1` of an op table in
the arena window. Each descriptor is 128 bytes and has one of four kinds:

- 1x1 convolution (HMX)
- stride-1 depthwise (HVX QFloat16)
- load (window to VTCM)
- store (VTCM to window)

**Validation.** Each descriptor is checked natively before its op runs:

- One 14-bit cap on every VTCM offset and shape word means no product or sum
  in the checks can overflow.
- Every VTCM index the op can form must lie in the borrowed span
  `0x31002000–0x311e0000`.
- Every window access must lie inside `0x1a40000–0x1f00000`.
- For 1x1 ops, the HMX tiles must be 2 KiB aligned.

**Arithmetic.** Per output vector, the arithmetic is the same as in the
qualified per-layer kernels:

- 1x1 and depthwise share one post-processing routine: pre-splatted bias,
  clip6, residual, interior mask and finite flags.
- Outputs are written straight into a zeroed VTCM grid.

**Fitting in 4 KiB.** The shared entry stub takes 1,648 B, which leaves about
2.37 KB for C. Fitting required these changes:

- The single post-processing routine above.
- Scratch space that follows the parameter area, so it needs no descriptor
  fields of its own.
- Validated offsets turned into pointers once per op.
- One copy loop.
- A guarded fault control at the start of the job.
- **Depthwise shares each coefficient load across 2 blocks instead of 4.** The
  cost of this was not isolated in this run.

**Host side (`tests/teacher_resident_layout.py`):**

- **Plans.** It builds two plans from the same op list:
  - **Qualify:** one job per op. Every output is stored, and parameters are
    staged into the window before each job.
  - **Resident:** one job per stride-1 segment. Every parameter image stays in
    the window, written once. Only tensors that cross a job boundary, and the
    final tensor, are stored.
- **VTCM placement.** First-fit, except that tensors still needed beyond the
  next op (residuals) are placed from the top so they don't fragment the
  56x56 stage. 1x1 parameter regions are 2 KiB aligned and padded.
- **Simulator.** A lane-level simulator runs the kernel's addressing. Each job
  starts from NaN-filled VTCM, and the op table is read back from the window
  exactly as the harness writes it.
- **Check mirror.** `native_ok` mirrors the kernel's checks. The kernel's own
  check code was also compiled for the host and run against both tables.

## Results

**Accuracy (qualification phase, 42 jobs per crop).** These match the
per-layer run exactly:

- Worst ratio to the per-layer FP16 bound: 0.494.
- Landmarks within 0.385 px.
- Presence within 2.8e-3.

**Bit-exactness (resident phase, 4 crops × 5 frames, plus one frame after the
controls).** The final tensor `79` was compared each frame and was byte-equal
to qualification every time. In frame 0, every stored crossing tensor was also
compared, and equal.

**Reporting gap.** The `bit_identical_tensors` field in the evidence is empty.
The harness recorded the last frame's observation list, which is empty,
instead of frame 0's. The comparisons themselves ran, and the harness stops on
any mismatch. This is to be fixed on the next harness revision.

**Per-job time, median ms:**

| Job | Ops | Descriptors | VTCM peak (vectors) | Body | With wrapper |
|---|---|---:|---:|---:|---:|
| resident 56x56 | conv_12 … convclip_3 | 8 | 12,350 | 4.25 | 5.87 |
| stride 2 | convdwclip_3 | — | — | 2.33 | 3.97 |
| resident 28x28 | conv_16 … convclip_5 | 7 | 7,105 | 1.96 | 3.58 |
| stride 2 | convdwclip_5 | — | — | 0.85 | 2.49 |
| resident 14x14 | conv_20 … convclip_11 | 19 | 10,558 | 4.14 | 5.75 |
| stride 2 | convdwclip_11 | — | — | 0.99 | 2.63 |
| resident 7x7 | conv_32 … convdwclip_15 | 14 | 14,099 | 4.11 | 5.72 |

**Where the time goes:**

- **Job overhead** is still about 1.6 ms per job, 11.4 ms per frame.
- **Stride-2 jobs** cost 9.1 ms per frame including wrappers. Each one copies
  its full-resolution input in and its output out, and runs a scalar gather.
- **56x56 expansions** (`convclip_2`, `convclip_3`) take about 1.1 ms each,
  limited by per-row post-processing.
- **Loads and stores** at the four job boundaries cost about 1.5 ms in total.

## Protections and controls

- 319 audited restorations (317 of them HMX jobs).
- Arena hash identical before and after; DSP service unchanged.
- Native rejection of an out-of-window descriptor, and a guarded fault in a
  resident job, both followed by a clean, bit-identical frame.

**Two earlier sessions on this work stopped:**

- **First session:** the qualification harness never wrote its op table. The
  kernel rejected the first job natively. The harness's integrity validator
  then required recovery, and the device was rebooted.
- **Second session:** the 1x1 tiles were misaligned, and the per-layer bound
  check stopped the run (ratio 159) before any result was accepted.

Both bugs are fixed and covered by offline tests. The simulator now reads the
op table from the window and rejects misaligned tiles, and the kernel checks
tile alignment natively.

## Next

- Run stride 2 inside the resident kernel. That removes 3 jobs and their
  copies, about 9 ms down to about 1–2 ms. It needs code space, or a
  vector-based gather.
- Cheaper 56x56 post-processing.
- The 112x112 stem.
- Pooling and the heads on the DSP.
