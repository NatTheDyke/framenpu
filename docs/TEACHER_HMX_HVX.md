# The unchanged teacher on HMX + HVX

**Status:** measured and forecast on 2026-11-17. The question was whether the
public v20a teacher, with its real weights and no retraining, is now feasible.
It had been set aside for the half-width fused 3x3 student, which needs
training and a gated dataset.

**Answer: probably yes.** With the resident unit costs from v2/v3 and a
measured HVX depthwise kernel, the teacher's convolution work forecasts at
about **7–10 ms**. That is inside the 12–20 ms gate, and no training is needed.
The remaining risk is not arithmetic. It is activation traffic for the
112x112 stage, the number of jobs, and FP16 accuracy on real crops.

## Why the teacher was dropped, and what changed

[HMX_MODEL_FEASIBILITY.md](HMX_MODEL_FEASIBILITY.md) priced every operand
pair as streamed from uncached DDR at 5.4 µs. On that basis the teacher's
depthwise layers, mapped to diagonal HMX tiles, came to 172 ms. The existing
HVX depthwise path costs about 250 ms per crop
([DEPTHWISE_ARITHMETIC.md](DEPTHWISE_ARITHMETIC.md)). Almost all of that is
FP32 coefficient conversion, exact-IEEE zero and non-finite handling, 8 KiB
VTCM and uncached DDR, not multiply/add.

Three things have changed since:

1. The 1,912 KiB FeatureNN VTCM span ([VTCM_MAP.md](VTCM_MAP.md)) holds a whole
   layer's operands.
2. The resident HMX costs ([HMX_SPATIAL_BLOCK.md](HMX_SPATIAL_BLOCK.md), v2/v3)
   are 0.21 µs per load, 1.97 µs per 2 KiB weight tile and 0.42 µs per output
   tile post-process.
3. The HMX path already uses a relaxed FP16/QFloat16 contract. Depthwise can
   use the same contract.

## Teacher shape

The graph has 145.6 M MACs:

- **124.0 M** are stem, pointwise and heads: 34 1x1 layers, the 3x3/s2 stem and
  4 head products. All are HMX-native and need no reformulation.
- **21.6 M** are depthwise: 12 stride-1 layers (17.2 M) and 4 stride-2 layers
  (4.4 M). These are HVX work.

## Measured: HVX issue rate (single thread, worker 0)

[hvx_rate_probe.c](../tests/hvx_rate_probe.c) ([evidence](../evidence/hvx-issue-rate.json))
times 8 independent operations per loop iteration and reads the core cycle
counter. The core runs at **1.84 GHz**; the `c31:30` timer is 19.2 MHz.

| Operation | Core cycles each |
|---|---:|
| `vadd.w` | 0.50 |
| Aligned VTCM vector load | 2.0 |
| `valign` | 2.0 |
| QFloat16 multiply + add (pair) | 2.0 |
| Aligned VTCM vector store | 2.0 |
| Empty one-packet loop iteration | 2.3 |

One thread issues at most about one packet every two cycles. Dependent
chains stall well beyond that.

## Measured: HVX FP16 depthwise

The probes cover all 12 stride-1 depthwise layers of the teacher at their real
shapes. They use QFloat16 multiply/accumulate, an FP16 bias add and clip6,
with operands copied from the arena window into VTCM. Activations and weights
are synthetic and deterministic. Every output was checked against an FP64
oracle with an FP16/QFloat bound.

- **v1** ([hvx_dw_probe.c](../tests/hvx_dw_probe.c),
  [evidence](../evidence/hvx-depthwise-v1.json)): planar layout, per-lane edge
  masks, one scalar→vector coefficient splat per tap, four channel pairs per
  step. **8.36 ms** for the 12 layers, 55–72 cycles per tap-vector. The loop
  stalls on its dependency chain; interleaving pairs gained only 15%.
- **v2** ([hvx_dw2_probe.c](../tests/hvx_dw2_probe.c),
  [evidence](../evidence/hvx-depthwise-v2.json)): changes the layout and loop.
  - Each row carries r zero guard words, so horizontal taps need no masks.
  - Coefficients are splatted to VTCM vectors once per layer.
  - Four consecutive output blocks share each coefficient load. The compiler
    turns their unaligned loads into five shared aligned loads plus `valign`.
  - Result: **1.76 ms** for the 12 layers, **4.8x faster than v1**. That is
    9.1–9.3 cycles per tap-vector on the 14–112-wide maps and 22–27 on the
    7x7 maps, where per-channel overhead dominates.

| Layer | Shape (C×H×W, k) | v1 µs | v2 µs |
|---|---|---:|---:|
| convdwclip_0 | 24×112×112, 3 | 1,274 | 208 |
| convdwclip_2 | 96×56×56, 3 | 1,264 | 212 |
| convdwclip_4 | 144×28×28, 5 | 1,222 | 222 |
| convdwclip_6, _7 | 288×14×14, 3 | 270 each | 54 each |
| convdwclip_8 | 288×14×14, 5 | 684 | 130 |
| convdwclip_9, _10 | 384×14×14, 5 | 912 each | 172 each |
| convdwclip_12–14 | 672×7×7, 5 | 456 each | 156 each |
| convdwclip_15 | 672×7×7, 3 | 180 | 68 |
| **Total (12 layers)** | | **8,357** | **1,759** |

Outside the timed pass, the v2 coefficient splat costs 1.27 ms in total,
mostly in the 7x7 layers (8,400 vectors each). It could move into the pass,
where it would be amortised over very few blocks, or the coefficients could be
packed differently. The probe's DDR→VTCM copy-in took 3.6 ms and copy-out
2.5 ms; a real executor keeps activations in VTCM where it can (see below).

Checks: the 5-round session ran 137 audited restorations (135 HMX). Guarded
fault, out-of-window rejection and post-fault validity passed. The resident
arena window was restored and the full arena hash matched. Service, health
and power were unchanged. The worst error was 0.20 of the bound, all values
were finite, and the masked lanes stayed zero.

## Forecast for the unchanged teacher

All figures are in ms per crop.

| Part | Basis | ms |
|---|---|---:|
| Pointwise, stem, heads: weights into VTCM | 1,033 tiles × 1.97 µs (v2) | 2.0 |
| HMX loads | 6,216 × 0.21 µs (v2) | 1.3 |
| Output post-process | 3,439 tiles × 0.42 µs (v2) | 1.4 |
| Stem gather/pack | 448 K tiles × 0.49 µs (v2) | 0.2 |
| 1x1 B-operand repack (0 if the layout is block-major) | ≤ 0.49 µs per K tile | 0–1.2 |
| Depthwise, stride 1 | **measured v2** | 1.8 |
| Depthwise, stride 2 | scaled at the stride-1 rate, **not measured** | 0.5 |
| Coefficient splat | measured v2, avoidable | 0–1.3 |
| **Convolution total** | | **≈ 7–10** |
| Activation traffic | see below | 0.4–8 |
| Job overhead | 1.6 ms per job (v3), 1–3 jobs | 1.6–5 |
| Residual adds, pooling, sigmoid/heads post | not measured; small | < 0.5 |

Previous forecasts, for comparison: the half-width fused3 student at 8.0 ms of
convolution (v2), which needs training; and the teacher with diagonal HMX
depthwise at 172 ms.

**Activation traffic is the main open cost.** Moving a layer's tensors
through uncached DDR costs about 0.16 µs per 128 B in and somewhat less out.
If every teacher layer round-tripped DDR, that would be about 17.7 MB, roughly
20 ms. But from the 56x56 stage onward, a whole MBConv block fits in the
1.9 MiB span: expand, depthwise, project and the residual. Those blocks never
need to leave VTCM. Only the 112x112 stage does not fit (64×112×112 FP16 is
1.6 MB):

- Spill it whole: about 6.4 MB, roughly 8 ms.
- Fuse it in row bands (stem → dw0 → conv_10, then expand → dw1/s2 per band):
  only the input image moves, roughly 0.4 ms.

**Job count and code size.** One job image is limited to 4 KiB, and each
operator kernel is 3.4–4.0 KiB. A whole-teacher job needs the existing
resident linked-operator mechanism ([RESIDENT_LINK.md](RESIDENT_LINK.md)),
which is not yet proven with HMX/VTCM-span kernels. Without it, each operator
type needs its own job, at about 1.6 ms each.

## What remains to prove

1. **Accuracy.** The teacher has run so far only in the qualified FP32/exact
   path. FP16 storage with QFloat16 depthwise and HMX accumulation changes
   rounding everywhere. A full-graph FP16 run must be compared with the
   bit-exact baseline's four heads on the recorded calibration crop, with
   explicit landmark/presence/handedness tolerances.
2. **Stride-2 depthwise** (4 layers, 4.4 M MACs). This needs a
   deinterleave/decimation path and has not been measured.
3. **Fused 112x112 row bands**, and an executor that keeps 56x56-and-smaller
   blocks VTCM-resident.
4. **Multi-operator jobs** through the linked image, or a measured job plan.
5. **Optional:** depthwise on workers 1/2 alongside HMX on worker 0. The
   firmware itself runs FeatureNN HVX on HTID 0/1/2, so this matches the
   platform's model, but no extra worker has been used for HVX here.

The recommended next experiment is an end-to-end FP16 teacher executor for the
56x56-and-below MBConv blocks. It should hold whole blocks in VTCM and be
checked against the FP32 baseline.

## Running it

```sh
python tools/build_hvx_rate_probe.py --output /path/rjobs
python tools/build_hvx_dw_probe.py --output /path/djobs
python tools/build_hvx_dw2_probe.py --output /path/d2jobs
# On the qualified device, as root (needs numpy):
python tests/hardware_hvx_rate.py --run-on-device --jobs /path/rjobs --output rate.json
python tests/hardware_hvx_dw.py --run-on-device --kernel v1 --jobs /path/djobs --output dw1.json
python tests/hardware_hvx_dw.py --run-on-device --kernel v2 --jobs /path/d2jobs --output dw2.json
```

Offline: `python -m unittest discover -s tests -p 'test_hvx_dw.py'`. It covers
lane-level addressing models of both kernels against the oracle, plan/VTCM
fits and the 4 KiB build limit.

Note: the v1 evidence was produced before the v2 layout functions were added
to `tests/hvx_dw_layout.py`, so its pinned layout hash is the earlier one. The
v1 functions themselves are unchanged.
