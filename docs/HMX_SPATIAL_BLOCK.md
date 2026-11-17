# Fused HMX spatial-block prototype

This is the next experiment recommended by the
[HMX model feasibility study](HMX_MODEL_FEASIBILITY.md). It is a bounded,
device-gathered, packed-FP16 spatial→projection block, rather than another
prepacked operand stream. It times each part of the work separately so the
next decision rests on measured component costs.

**Status:** measured on the device on 2026-11-17 in two full runs.
[Evidence](../evidence/hmx-spatial-block.json). Every numerical, fault,
rejection and restoration check passed. **The decision gate is not met:**
the forecast for the half-width student is about 99 ms of convolution
against 12–20 ms. Even with free gathering, this 8 KiB design cannot reach
the gate.

## Results

Per 32-position output block, median of five alternated rounds (µs):

| Geometry | main | no timers | no HMX instruction | per HMX load, from main − no-HMX |
|---|---:|---:|---:|---:|
| `w050_112` | 69.3 | 57.8 | 67.2 | 0.35 |
| `w050_14` | 494.6 | 426.9 | 482.7 | 0.30 |
| `w050_7` (2 of 11 mid blocks) | 483.9 | 425.3 | 476.2 | 0.21 |
| `teacher_112` (rows 0 and 48) | 92.2 / 92.0 | 76.8 / 76.7 | 89.5 / 89.3 | 0.33 |

Per-unit costs are consistent across the geometries (median of the w050
geometries):

| Unit | µs |
|---|---:|
| Weight tile copy (16 contiguous aligned DDR vectors) | 2.27 |
| Gathered K tile (stage + pack: 32 scattered aligned DDR vectors plus shift/mask) | 9.69 |
| `hmx` phase per load (barrier, call and stamp; nearly all still present with no HMX instruction) | 1.27 |
| Mid-tile spill or reload | 1.26 |
| Stage-1 post per mid tile / stage-2 store per output tile | 0.58 / 0.37 |

Findings:

1. **HMX itself is cheap.** With HMX removed, only 0.2–0.35 µs per load
   disappears. Almost all the time goes to moving data from DDR to VTCM.
2. **The gather dominates: 67 of 99 forecast ms.** Each K tile makes 32
   scattered aligned DDR loads, about 0.3 µs each. That is twice the cost
   of a contiguous weight-tile vector. The stage/pack split is not clean.
   Vector loads do not block, so DDR latency surfaces at the first VTCM
   read of the staged data, which is in `pack`. Serial copies move the
   time between phases instead of removing it. Use stage + pack combined.
3. **Batched copies do not help in total.** Eight loads in flight before
   the stores is about 8% *slower* per block than serial copies.
4. **The phase timers cost 13–20%.** The tables above report timed
   builds; the no-timer column gives the true block time.
5. **Stage-2 weight residency, the only reuse that fits, saves 1.8–2.3 µs
   per block** (about 3%). This matches one weight-tile copy.
6. **Correctness:** output error is at most 0.2 of the oracle bound. The
   teacher deviates from exact FP64 semantics by at most 0.018, from FP16
   rounding. The spilled mid tiles are verified directly. Storing into B
   right after the last load from B produced no hazard. All four
   dirty-accumulator faults, four rejections and four clean results after a
   fault behaved as designed. 479 audited jobs restored cleanly across both
   runs. The resident arena hash, DSP service PID and restart count are
   unchanged, and the longest job took 9.7 ms.

Forecast for the fused3_w050 convolution (6,927 spatial and 1,243 1x1 K
tiles, 2,271 transactions): weights 18.7, gather 67.1, `hmx` phase 10.4,
1x1 operand copies 1.6 and post 1.3, about **99 ms in total**. That is
twice the earlier prepacked-stream forecast of 47 ms. Five layers are
approximated (the stride-2 layers and the RGB stem).

**What this means for the next decision.** Weight copies plus the
per-load barrier overhead already total about 29 ms, more than the gate,
before any gathering. A better gather can't close that gap inside 8 KiB.
Halo reuse (sharing source vectors across `dx`, sliding windows across
blocks) and a smaller barrier count still need somewhere to keep reused
data. Weight residency needs per-layer weights in VTCM, up to about 338 KiB
for the 7x7 layer. The next step therefore requires a decision: obtain
independent ownership evidence for a larger VTCM region, or go back to
architecture/traffic reduction. A training run is not justified on these
numbers.

## What one block computes

A dense 3x3 stride-1 convolution (Cin→Cmid, bias, clip6) feeds a 1x1
projection (Cmid→Cout, bias). This matches one fused3 student block. Its
input and output are both stored as **planar-packed FP16**: one 128-byte
vector per channel pair and 32-position block, holding words
`(channel 2p, channel 2p+1)` at each position. The HMX store writes this
layout as its output tile, and the gather reads it as its B operand, so a
producer's output can feed the next consumer without repacking on the host.

| Geometry | Map | Cin→Cmid→Cout | K tiles/mid block | Notes |
|---|---|---|---:|---|
| `w050_112` | 112x112, 8-row band | 16→16→8 | 5 | Fully fused in VTCM; stage-2 weights stay resident |
| `w050_14` | 14x14, whole map | 24→144→24 | 7 | Five mid blocks are spilled to arena DDR |
| `w050_7` | 7x7, whole map | 56→64 (of 336)→56 | 16 | The arena limits a job to 2 of 11 mid blocks, so stage 2 is a partial sum |
| `teacher_112` | 112x112, 8-row band | 24→24→16 | 7 | Public `convdwclip_0` placed diagonally, plus `conv_10` |

The half-width geometries use deterministic synthetic weights because no
trained student exists. The teacher control uses the public pinned v20a
weights. Its oracle is the original depthwise+projection semantics, so it
validates the new data path with real weights and no training. It is also
run on the top band to exercise image-border padding.

## Per-block pipeline and timed phases

For each 32-position output block, the kernel stamps the 19.2 MHz counter
between phases. Each stamp assigns the time since the previous stamp to one
phase, so the phases partition the measured body:

| Phase | Work |
|---|---|
| `weights` | Copy one 2 KiB weight tile from arena DDR to VTCM (stage 1 per K tile; stage 2 per mid block unless resident) |
| `stage` | Spatial gathering: eight aligned source vectors per quarter tile, from DDR to VTCM |
| `pack` | Tile packing: `valign` by the tap's word shift, column masks for `dx=±1`, zero K padding, then write B |
| `hmx` | `clear`, `barrier`+`load`, and `store` (store includes its barrier) |
| `post` | Stage-1 QFloat16 bias, clip6 and finite flags in place in B |
| `spill` | Mid-tile write to the arena and reload into B (only when Cmid > 32) |
| `store` | Stage-2 bias, tail mask, finite flags and planar FP16 output rows to the arena |
| `setup` | Per-block column/position vector load and descriptor work |

Each K tile packs 16 row pairs `k = tap*Cin + channel`, so taps can share a
tile when Cin < 32. Within a band, row padding is explicit halo rows, so the
device masks columns only. One zero guard block in front, plus enough blocks
behind, keeps every aligned load in bounds. The design follows the measured
[access costs](HMX_CACHE.md): aligned DDR vector loads, with all unaligned
work done in VTCM.

Build variants isolate three effects:

- `main`: 8-vector batched copies (eight loads in flight before the stores) with timers.
- `copy1`: serial load→store copies.
- `notime`: no phase timers, which measures instrumentation overhead.
- `noreuse` (112 geometries only): reloads the stage-2 weight tile every
  block. This measures the one weight-reuse case that fits in 8 KiB.
- `nohmx`: the same data movement with no HMX instruction (timing only, so
  outputs are not validated). It separates VTCM/DDR work from HMX execution.

## Why VTCM limits fusion and reuse

The VTCM map uses all 64 saved vectors: A 16, B 16, resident stage-2
weights 16, staging 8, scale 2, bias 3, column vector 1, flags 1, timers 1.
There is one HMX accumulator, so each transaction must stream its whole K
reduction through those tiles. Consequently:

- Stage-1 weight tiles cannot be reused across position blocks. A 3x3 layer
  needs 5–16 tiles (10–32 KiB) per mid block.
- Mid activations stay in VTCM only when Cmid ≤ 32. Larger mid widths are
  spilled per block, which costs a measured round trip.
- Gathered B tiles are rebuilt for every mid block.

The measured `weights`, `spill` and repeated gather costs show what a larger,
independently owned VTCM region would buy. The analyzer reports this as a
counterfactual. The prototype does **not** use, enlarge or assume any VTCM
beyond the entry's saved 8 KiB.

## Protections kept

The kernel uses the same mechanisms as the earlier probes:

- cached entry, worker 0 only, one HMX owner;
- the saved 8 KiB VTCM region and the 128 KiB job arena;
- an in-kernel 20 ms budget check after every block (the SDK 40 ms timeout is unchanged);
- exact header, context and input/output size checks, with wrong-geometry rejection;
- a guarded trap after the first HMX load (dirty accumulator), followed by a clean, validated result.

The harness checks restoration on every job: code, header, status, VTCM
backup, arena, callback and HMX power. It also compares the resident arena,
health and DSP service before and after. Numerical checks run after
restoration. Any unexpected result stops the run without retry, recovery,
gate clearing or service restart. Measured jobs are sized to at most about
10 ms.

## Offline validation

`tests/test_hmx_block.py` contains these checks:

- A bit-exact NumPy emulation of the device gather (block index, `valign`
  shift, masks, K padding) equals an independent im2col for every K tile of
  every block in all four geometries.
- A whole-block emulation through packed tiles stays within the oracle bounds.
- Top and bottom bands read zero padding rows.
- The payload and planar layouts round-trip.
- All 19 builds fit the 4 KiB payload, with zero vector stack accesses.
- The analyzer recovers known unit costs from a fabricated report.

Oracle bounds detect gross corruption. They are not tracking tolerances.
The teacher's propagated bound is loose (up to about 0.9), so the harness
also records the deviation from exact FP64 semantics. That deviation is
expected to be about 0.01, from FP16 rounding alone.

## Running it

```sh
python tools/build_hmx_block_probe.py --output /path/jobs
# On the qualified device, as root, with the existing resource setup:
python tests/hardware_hmx_block.py --run-on-device --jobs /path/jobs \
  --model /path/public-model --output /path/measurement.json
python tools/analyze_hmx_block.py --measurement /path/measurement.json \
  --model /path/public-model --output /path/summary.json
```

The analyzer reports per-unit costs (per weight tile, gathered K tile, HMX
load, mid/out tile and spilled tile) and the phase shares. It also reports
the effects of batching, residency and timers, and a fused3_w050
convolution forecast against the 12–20 ms decision gate, with and without
weight copies. Stride-2 and RGB-stem layers reuse stride-1 costs, and the
analyzer flags them as approximations.
