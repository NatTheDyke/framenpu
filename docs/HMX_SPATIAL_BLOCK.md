# Fused HMX spatial-block prototype

This is the next experiment recommended by the
[HMX model feasibility study](HMX_MODEL_FEASIBILITY.md). It is a bounded,
device-gathered, packed-FP16 spatial→projection block, rather than another
prepacked operand stream. It times each part of the work separately so the
next decision rests on measured component costs.

**Status:** v1 (8 KiB VTCM) was measured on the device on 2026-11-17 in two
full runs, and the v2 VTCM-resident block followed it; see
[v2](#v2-vtcm-resident-block).
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

## v2: VTCM-resident block

**Status:** measured 2026-11-17
([evidence](../evidence/hmx-resident-block.json)). The forecast is about
**8 ms of convolution, inside the 12–20 ms gate.** v2 uses the 1,912 KiB
FeatureNN worker-0/1 scratch span proven in [VTCM_MAP.md](VTCM_MAP.md).
[hmx_resident_block_probe.c](../tests/hmx_resident_block_probe.c) computes
the same block, with the same payload, output, oracle and harness as v1, but
keeps every operand in VTCM:

- **Prologue, once per job:** contiguous copies of every W1/W2 tile, the
  whole planar source band and the x|position table. It also builds a gather
  table (source vector offset, `valign` shift and mask kind for each B row
  pair), which does not depend on the position block.
- **Per position block:** the kernel packs all K tiles once from VTCM (two
  aligned VTCM loads, `valign` and a mask per row pair) and issues one
  barrier. Each mid block then runs back-to-back HMX loads straight from the
  resident weight tiles. The mid tile is stored to VTCM and biased/clipped in
  place, with one barrier for the stage. The projection then reads the
  resident W2 and mid tiles. Nothing is spilled, and the last block's mid
  tiles are copied out after the timed body only so they can be checked.
- `res1` uses single loads and `res2` the qualified paired load (range 4095,
  [HMX_BATCHING.md](HMX_BATCHING.md)). `res2_notime` has no timers.

Steady per-block cost (repeat slope, so the prologue cancels; median of five
alternated rounds, µs):

| Geometry | v1 `notime` | v2 `res2_notime` | Speedup | v2 prologue per job |
|---|---:|---:|---:|---:|
| `w050_112` | 57.7 | 6.39 | 9.0x | 71 |
| `w050_14` | 426.9 | 14.29 | 29.9x | 110 |
| `w050_7` | 422.3 | 17.51 | 24.1x | 105 |
| `teacher_112` (rows 0 / 48) | 76.5 / 76.6 | 7.48 / 7.50 | 10.2x | 92 |

v2 unit costs (timer-free medians of the w050 geometries):

| Unit | µs |
|---|---:|
| Pack per K tile (from the VTCM band) | 0.49 |
| HMX per load (clear/load/store included) | 0.21 |
| Post/store per output tile | 0.42 |
| Prologue weight tile copy | 1.97 |
| Prologue source vector copy | 0.16 |

Forecast for fused3_w050 (`forecast_resident` in the analyzer), in ms:

| Weights into VTCM | Source into VTCM | Pack | HMX | Post | **Total** |
|---:|---:|---:|---:|---:|---:|
| 2.4 | 1.2 | 1.7 | 1.7 | 1.0 | **8.0** |

Previously the per-block cost was DDR-bound; HMX itself now takes about a
quarter of it. Paired loads change the block time by less than 2% once the
operands are resident.

**Not yet modeled:** the forecast assumes one prologue per layer and
resident activations between layers (1x1 B operands are not repacked). It
does not model job splitting under the 128 KiB arena, inter-layer DDR traffic
of a real executor, stride-2 and stem layers (they reuse stride-1 costs),
residual adds, pooling, heads or host work. The next step is a layer-level
executor that reads inputs from the persistent DDR arena and measures those
costs.

**Hardware lesson:** HMX operands must be aligned more strictly than 128 B. A
first v2 build placed B and mid tiles on 128-byte boundaries (`w050_14`) and
produced non-finite output. Tiles that were 512 B or more aligned
(`w050_112`) were correct. v2 now places every operand region, and each
mid/output block's weight run, on a 4 KiB boundary. The harness stopped
cleanly on the bad build with no recovery needed. The exact minimum alignment
was not bisected.

**Checks:** one session ran 444 audited restorations and every check
passed. Output errors are within the oracle bounds, and v2 spilled-mid tiles
were checked directly. Guarded faults, wrong-geometry rejections and clean
post-fault results passed for both v1 and v2 on four fixtures (8 each). The
resident arena hash, DSP service PID and restart count were unchanged, and
the longest job took 9.7 ms.

## v3: layer-chain executor

**Status:** measured 2026-11-17 ([evidence](../evidence/hmx-layer-chain.json),
two sessions; numbers are from the 5-round one).
[hmx_layer_chain_probe.c](../tests/hmx_layer_chain_probe.c) runs whole
layers and chains them on the device. It is the first measurement of what
v2 did not model: inputs from the persistent DDR arena, inter-layer traffic,
and job splitting.

- **Runtime geometry.** The 4 KiB payload is a header plus one 32-word
  descriptor per layer. Each descriptor holds the shape and arena offsets of
  the input, W1|W2 tiles, biases, gather table, x|position table and output.
  The kernel checks every descriptor before any work: all six regions must
  lie in the arena window `[0x1a40000, 0x1f00000)`, the output and gather
  extents are bounded, and the VTCM plan must end inside the borrowed span.
  The gather and position tables are built by the host
  ([hmx_chain_layout.py](../tests/hmx_chain_layout.py)). Their contents only
  steer values: gather offsets are masked, and every write target comes from
  checked descriptor fields.
- **Chaining.** Activations use a padded planar layout (zero halo rows and
  guards, `(pad + w) % 32 == 0`). Each layer writes the next layer's input
  layout directly, so the host never repacks between layers.
- **Per layer:** the prologue copies W1|W2, biases, tables and the input band
  from the uncached DDR alias into VTCM. Blocks then run as in v2 with single
  loads, and outputs are stored straight to the arena. A job covers layers
  `params[1]..params[2]-1`. A job that reaches 20 ms stops between blocks and
  reports where it stopped.
- **Chains:** the stride-1 fused3_w050 runs. `c14` has five 14x14 layers
  (24→144→24 ×2, 24→144→32, 32→192→32 ×2). `c7` has four 7x7 layers
  (56→336→56). Each chain runs as one whole-chain job and as one job per
  layer, alternating over 5 rounds after checked warmups. Data is synthetic
  and deterministic.

Results (medians):

| Chain | Schedule | Kernel body | Wrapper (in-DSP) | Host wall |
|---|---|---:|---:|---:|
| `c14` (5 layers) | whole chain, 1 job | 1.37 ms | 3.07 ms | 9.7 ms |
| `c14` | per layer, 5 jobs | 1.47 ms | 10.0 ms | 42.9 ms |
| `c7` (4 layers) | whole chain, 1 job | 2.13 ms | 3.82 ms | 10.5 ms |
| `c7` | per layer, 4 jobs | 2.20 ms | 8.95 ms | 35.2 ms |

Per layer inside the whole-chain job (µs):

| Layer | Prologue | Blocks | Blocks per layer |
|---|---:|---:|---:|
| 14x14 24→144→24/32 | 93–99 | 129–146 | 7 |
| 14x14 32→192→32 | 137 | 154–156 | 7 |
| 7x7 56→336→56 | 419–440 | 82–99 | 2 |

What this shows:

- **Chaining on the device costs almost nothing in the kernel.** Writing to
  the arena and re-staging the input adds tens of µs per layer. Splitting
  into per-layer jobs adds only about 0.1 ms of kernel body.
- **Per-job overhead is what job splitting costs.** Each job adds about
  1.5–1.7 ms of in-DSP wrapper time beyond the body (entry, cached-code
  publication and cleanup) and about 6–8 ms of host wall time on this audited
  harness path. Whole-chain jobs are therefore much cheaper: a chain should
  run as few jobs as the 20 ms budget allows. The longest job measured was
  3.85 ms of wrapper time.
- **At 7x7 the cost is weight streaming.** Each 7x7 layer copies 198 tiles
  (396 KiB) from uncached DDR, at about 2.1 µs per tile (~0.95 GB/s). That is
  80% of the `c7` body; the HMX blocks take about 83 µs. This matches v2's
  1.97 µs per tile. The 7x7 weights are the next target: faster DDR→VTCM
  streaming (cached alias or prefetch) or keeping weights resident across
  jobs.
- **Accuracy:** every layer of every run was checked against the oracle of
  its actual device input. The worst error was 0.11 of the bound, all values
  were finite, and padding stayed zero.

**Checks:** 74 audited restorations (72 HMX) in the 5-round session and 41 in
the first. Per chain there was one guarded fault in the first block, one
out-of-window descriptor rejection and one clean post-fault chain, all
validated. The resident arena window was saved before any write and restored
afterwards, and the full arena hash then matched its initial value. The DSP
service, health and power configuration were unchanged. The kernel is
4,056 B.

**Not covered yet:** stride-2 and stem layers, residual adds, pooling,
heads, host scheduling in production (no audits), and chains longer than
the 20 ms budget (the resume path exists in the kernel but was not
exercised).

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

The v3 chain executor:

```sh
python tools/build_hmx_chain_probe.py --output /path/cjobs
# On the qualified device, as root (needs numpy):
python tests/hardware_hmx_chain.py --run-on-device --jobs /path/cjobs \
  --output /path/chain.json --rounds 5
```
