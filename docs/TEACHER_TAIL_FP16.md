# FP16 teacher tail on HMX + HVX (56x56 to the heads)

This run executes the unchanged public v20a teacher from the 56x56 stage to
the heads on the DSP in FP16. It covers 42 jobs per crop: `conv_12` through
`convdwclip_15`, with the 10 residual adds fused into the projections. The
run is checked against the ncnn FP32 blobs of the four qualified corpus crops.
The global average pool and the four heads run on the host in FP32.

**Result:** every check passed. Layer for layer, the device stays within its
FP16 error bound (worst ratio 0.49). The heads stay within the frozen ceilings
on all four crops. The DSP compute is 10.1 ms per crop. Copying activations in
and out of VTCM costs more than the compute itself.

Evidence: [evidence/teacher-tail-fp16.json](../evidence/teacher-tail-fp16.json)
(device), [evidence/teacher-tail-fp16-emulation.json](../evidence/teacher-tail-fp16-emulation.json)
(host model of the same contract). Both files contain only statistics; no
tensors leave the device.

## Pieces

| File | Role |
|---|---|
| `tests/teacher_tail_layout.py` | Grid layout, op list (Split aliases resolved, adds fused), weight packing, descriptors, arena plan, the FP16 contract model, per-layer error bounds and lane-level kernel models |
| `tests/teacher_pw_probe.c` (opcode 109, 3,816 B) | HMX 1x1: B-tile pack per block, `nk` loads per output tile, then FP16 bias, optional clip6, optional FP16 residual, interior mask |
| `tests/teacher_dw_probe.c` (opcode 110, 4,044 B) | HVX QFloat16 depthwise, k3/k5, stride 1 or 2, built on the v2 inner loop |
| `tests/hardware_teacher_tail.py` | Device run: lease, refusal record, window save/restore with whole-arena hash, audited direct dispatch, controls |
| `tools/teacher_tail_accuracy.py` | Host emulation of the contract on the corpus, with no device |
| `tools/build_teacher_tail.py`, `tests/test_teacher_tail.py` | Build (with source pins) and offline tests: the kernel models match the math model on all 42 ops; both images fit in 4 KiB |

**Grid layout.** Every tensor at one resolution uses the same grid of 32-word
blocks per channel pair:

- Row stride `ws = w + 2`, with two zero guard words per row and two zero halo
  rows above and below.
- The leading offset `G` makes the interior start block aligned.

Depthwise taps read neighbouring words directly, so the kernel needs no edge
masks. Only the outputs are masked to interior lanes.

**Stride 2** (`convdwclip_3`, `_5` and `_11`, with TF-SAME padding (0,1) or
(1,2)) runs in three steps:

1. Compute the stride-1 result.
2. Gather the words at `(2y+1, 2x+1)` into a zeroed half-resolution grid, using
   a scalar gather.
3. Copy that grid out.

The host model of this decimation matched `landmark_reference.evaluate`.

## Accuracy (4 crops x 3 rounds, all rounds bit-identical)

| Measure | Device | Host emulation | Frozen ceiling |
|---|---|---|---|
| Worst per-layer `|dev - exact(dev input)| / bound` | 0.494 | — | 1 |
| Landmark 2D error, max (224 px crop) | 0.385 px | 0.330 px | `linear_42` max 1.0 |
| Landmark 2D error, worst-crop mean | 0.286 px | 0.178 px | — |
| `out0` rms (bounded from the above) | ≤ 0.22 | — | 0.25 |
| World landmarks, max abs | 4.5e-4 | 2.3e-4 | 3e-3 |
| Presence, max abs delta | 2.8e-3 | 1.2e-3 | 5e-3 |
| Handedness, max abs delta | 3.5e-4 | 7.7e-5 | 5e-3 |

The heads clear every ceiling, but presence and the `out0` rms leave only
about 1.8x and 1.1x of headroom.

**Intermediate tensors fail the per-layer tolerances.** Those tolerances were
calibrated for the FP32 path (`calibration_only`). Against them, 84 of 168
crop×layer max-abs checks and 134 rms checks are over the cap. The worst case
is `convdwclip_3`, at 6–9x its max-abs cap and a 0.33 drift against a 6.0
range, because clipping amplifies small differences near the clip edge.

The device drifts more than the host model. The model treats QFloat16
accumulation as FP32 accumulation, and that difference is the main suspect.
A gate for FP16 should therefore be set on the heads, or on new FP16
tolerances. It should not reuse the FP32 per-layer table.

## Time per crop (median of 12 runs)

| Part | ms |
|---|---|
| HMX 1x1 compute (28 jobs) | 5.87 |
| HVX depthwise compute (14 jobs) | 4.24 |
| Prologue: DDR to VTCM copies, depthwise splat | 12.58 |
| Epilogue: VTCM to DDR | 4.85 |
| In-DSP wrapper (42 jobs, about 2.3 ms each) | 96.2 |
| Host wall time, including per-layer staging | 430 |

The prologue moves about 82k activation vectors and about 17k vectors of
weights and masks. Most of the prologue and all of the epilogue is activation
traffic that a VTCM-resident, multi-op job would not need.

**Forecast, not measured:**

- Keeping activations resident, with a few jobs instead of 42, puts this
  portion at about 12–14 ms on the DSP. That is 10.1 ms of compute plus about
  2 ms of weight streaming and the splat.
- This assumes that the largest 56x56 working set fits in VTCM. A tensor of
  96 channels at 56x56 is 0.69 MiB, and this run's largest per-job footprint
  is 13,344 of 15,296 vectors.

Targets visible in the per-op table (`evidence/teacher-tail-fp16.json`):

- `convdwclip_3` takes 1.5 ms. Most of that is the full-resolution
  stride-1 pass plus the scalar gather.
- The 56x56 expansions (`convclip_2`, `convclip_3`) take 0.92 ms each and are
  bound by the per-row post-processing, not by HMX.

## Protections and controls

- 509 audited restorations (507 of them HMX jobs); arena and service unchanged
  across the run.
- One guarded fault (1x1), one out-of-window rejection per kernel, and clean
  reruns afterwards.
- Every job: magic, PC range, block count, zero finite-flag vector, zero
  padding lanes, and the timing budget.

## Still open

- 112x112 stem and blocks (`convclip_0` … `add_0` inputs). This run starts from
  ncnn blobs 8 and 10.
- GAP and the heads on the DSP.
- A resident multi-op job, to remove the wrapper and activation traffic.
- A cheaper stride-2 path.
- FP16 per-layer tolerances.
