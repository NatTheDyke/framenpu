# HMX-oriented landmark model feasibility

HMX can execute the spatial work, but a model rewrite alone does not establish
the required 20–30x improvement. On 2026-11-16, distinct-operand measurements
showed that streaming packed operands dominates the tested pipeline. A smaller
fused model plus activation/weight reuse is the strongest candidate to test.
No new full graph or trained student has been implemented or timed.

The study uses only the pinned public model and deterministic synthetic
activations. It uses no private recordings, images or labels. Source/build
pins, all 360 timing samples, numerical summaries and shape accounting are in
[the evidence](../evidence/hmx-model-feasibility.json).

## Measured pipeline

[The probe](../tests/hmx_model_probe.c) uses cached code, one HMX owner and the
existing 8 KiB VTCM save/restore. Each transaction clears, loads T pairs of
32x32 FP16 operands, stores once and checks finiteness. Four separately
compiled modes avoid vector stack spills:

0. Resident arithmetic control: repeat the first VTCM operand pair T times.
1. Stream all T distinct packed operand pairs, return packed FP16.
2. Stream, convert output to FP32, add bias and clip using the existing path.
3. Stream, add rounded FP16 bias with supported QFloat16, clip and return
   packed FP16. This does not use the unavailable IEEE HVX arithmetic.

Nine cases cover dense synthetic T=1/9/25, diagonal embeddings of public
convdwclip_0/_8/_12 weights, and public conv_10/convclip_12/linear_42 weights
at T=1/4/21. Public-weight tests cover one 32-channel/output block per layer;
they are not whole-layer or real-image accuracy tests. Spatial synthetic
features respect the selected layer's shape and padding. Every pair within a
streamed reduction is distinct. Repeats within each job use the same buffers
and data again; data access remains through the existing uncached DDR alias.

Four checked warmups per case are excluded. Five rounds alternate mode order
and repeat counts 1/8. The table gives the median of five within-round slopes
`(time8-time1)/7`, in microseconds per complete transaction. Setup, wrapper,
publication and host preparation are outside the pipeline bracket.
This short experiment does not qualify sustained thermals, tail latency or
coexistence with tracking services.

| Synthetic reduction depth T | Resident control | Stream/raw half | Stream/FP32 post | Stream/FP16 post |
|---|---:|---:|---:|---:|
| 1 | 2.411 | 6.287 | 7.485 | 6.615 |
| 9 | 2.403 | 49.896 | 51.019 | 50.141 |
| 25 | 2.403 | 136.555 | 137.857 | 137.292 |

The FP16-post fit is **1.156 + 5.445 T us**, with maximum residual 0.119 us
across all nine synthetic/public-weight cases. Each streamed depth pair is
4,096 bytes, giving about 0.75 GB/s of effective pair streaming. This includes
copying, barriers, HMX helper calls and checks; it is not a measurement of the
physical memory bus limit. Resident mode0 repeats operands and still writes
DDR output; its roughly 2.4 us result is not a distinct-operand graph forecast.

FP16 postprocessing saves about 0.9 us per transaction against FP32 post,
only about 4.3 ms across the diagonal graph forecast. Its larger potential
benefit is keeping activations packed between operators, avoiding later
conversion/repacking. That cross-layer benefit is not measured here.

All 360 measured outputs and 36 warmups passed the CPU arithmetic checks.
Raw FP16 mode0/1 results matched the rounded CPU result exactly, including
all 1,024 lanes and tail zeros. FP32 post maximum deviation was 4.77e-7;
FP16/QFloat post maximum was 0.00390625 against the specified half-bias CPU
model. Bounds account for accumulation, half storage and QFloat rounding;
they detect gross corruption and are not hand-tracking tolerances. Bias
quantization and full-graph propagation remain unqualified. The clip6 probe
is a timing proxy for output processing, not the semantics of linear/sigmoid
heads. Those heads must retain their actual output operations.

## Model cost and conditional forecasts

The teacher is public v20a at commit
`03d7695fecb80f6cf8f3aa9321de130b5efd08b3`, with parameter SHA-256
`54e08f284ff6f121ff00f4a9c65afc38c0dbb22a32f4af44fc4e0fde6c4a1bbd`
and binary SHA-256
`b7c3d1be232725a5a143e2648858ec5aec9596c7b83c6a3c508479b04057e00c`.
The binary is 2,046,732 bytes. Exact accounting recovers 1,000,160 FP16
weights, 11,552 FP32 biases and 51 four-byte weight tags.

The original graph has 145.6 million logical MACs. Its existing stem,
pointwise and heads consume 203.7 million padded HMX MACs. Mapping depthwise
to diagonal 32-channel matrices adds 795.9 million physical MACs, for about
1.0 billion total. The real-valued depthwise expression is unchanged; FP16
inputs and new accumulation/store boundaries change rounding. The diagonal
mapping has prior art in
[Diagonalwise Refactorization](https://arxiv.org/abs/1803.09926); that paper's
GPU training speedups do not predict this device's inference performance.

The table applies the measured FP16-post fit to exact transaction/depth
counts. MB means decimal megabytes. These are **convolution-only forecasts
with already-packed operands**, excluding spatial lowering, pooling,
residuals, sigmoid, graph control, dispatch and changes between layouts.
They credit no reuse between transactions. They are neither measured graph
latencies nor guaranteed upper/lower bounds.

| Candidate | Raw weights/bias/tags MB | Padded HMX MACs, billion | Prepacked stream forecast ms |
|---|---:|---:|---:|
| Existing HMX layers only; depthwise excluded | 2.05 for whole teacher | 0.204 | 37.8 |
| Teacher with diagonal HMX depthwise | 2.05 | 1.000 | 172.0 |
| Replace depthwise with learned 8-channel groups | 3.54 | 1.000 | 172.0 |
| Full-width fused 3x3 spatial blocks | 8.66 | 0.695 | 118.8* |
| 0.75-width fused 3x3 | 5.17 | 0.524 | 90.2 |
| **0.5-width fused 3x3** | **2.23** | **0.268** | **47.1** |
| 0.375-width fused 3x3 | 1.47 | 0.227 | 40.2 |

\* Full-width fused3 reaches T=32, slightly beyond measured T=25; its affected
transactions are identified in the evidence. Other listed forecasts stay
within the measured depth range. Full dense substitution needs about 109 MB
of weights and 6.48 billion padded MACs; its T=525 streaming extrapolation
is not a qualified forecast. Retaining every original 5x5 kernel in fused
blocks needs 18.85 MB of weights and increases transfer work substantially.

For diagonal/group mapping, each 32-channel tile executes one load pair per
spatial tap. Four independent 8-channel groups occupy a block-diagonal 32x32
matrix, so learned group8 uses the same physical HMX work as depthwise's
diagonal zeros. Added mixing may help accuracy/capacity, but offers no timing
gain at unchanged shapes. Group32 falls back to group8 at C24 and group16 at
C144 to keep uniform legal groups; its raw weight file is already 8.49 MB.

Fused candidates replace 15 adjacent 1x1 expansion + depthwise pairs with one
dense spatial convolution; the initial depthwise also becomes dense. Fused3
uses 3x3 throughout and retains stride/output shapes. It removes intermediate
clip nonlinearities and changes 5x5 receptive fields, so cannot be obtained
by analytically multiplying the existing weights. It requires retraining or
distillation. The hardware motivation resembles
[Fused-MBConv](https://proceedings.mlr.press/v139/tan21a.html), without adopting
that paper's accuracy or performance claims. Width variants scale internal
channels, rounding up to multiples of eight; RGB input and all four head
dimensions remain fixed. No width/accuracy tradeoff has been validated.

The current 498 ms native graph includes about 25 ms of other work. Even if
that fell to 5 ms, the half-width 47.1 ms convolution forecast still requires
another **2.4–3.9x** improvement to fit a 25–17 ms total native target.
Keeping other work at 25 ms would make this candidate roughly 72 ms before
additional lowering/control, around 7x over 498 ms. A 20–30x result is not
demonstrated. Batching more crops does not remove this per-crop streaming work.

## Storage and dataflow are feasibility gates

The existing graph owns 6,152,192 bytes of resident DDR and 8 KiB saved VTCM.
Its packed weights take 2,051,904 bytes; its FP32 activation allocation span
is 4,014,080 bytes. Raw file sizes above exclude new matrix padding and
packing. Storing all diagonal zero-filled tiles would waste memory; sparse
coefficients need on-chip expansion/reuse or another justified layout.

Full-width fused/group32 weights do not fit the current DDR arena even before
activations. The half-width candidate has a plausible raw weight budget, but
still needs a real lifetime/alias/packing allocation proof. Existing in-place
depthwise/downsample alias proofs do not transfer automatically to channel
mixing or a new layout. Keep the RGB stem input compact: padding its full
224x224 tensor to 32 FP16 channels alone consumes 3.21 MB.

The diagonal model's simple stream consumes **125 MB of operands/crop**,
including repeated weights and overlapping activation patches. This is the
main measured obstacle. A useful backend must reuse coefficients across
spatial tiles, reuse patches/halos, and share packed FP16 outputs between
operators. The fused half-width stream is still about 33.5 MB/crop. A dense
32-channel 3x3/5x5 kernel alone is 18/50 KiB, so it cannot all remain in the
current VTCM alongside activations/output. No extra VTCM ownership or cache
coherency is assumed; any larger region needs independent ownership evidence.

## Public training route

The pinned weights can serve as a teacher and a weight-preserving HMX control.
The current [model card](https://huggingface.co/DeeJanuz/frametop-hands-models/blob/main/README.md)
says training code is unpublished. The public
[Frametop Hands dataset](https://huggingface.co/datasets/DeeJanuz/frametop-hands)
metadata at revision `10e42078c8d58108209e20653b0c11d49f499711` reports automatic
access gating. Its card says keypoint labels are not included; they are made
later by a teacher/triangulation/skeleton-fit pipeline. Only cards/metadata
and model weights were downloaded for this study. No access terms were
accepted and no recordings downloaded.

Public-only training is feasible in principle but is not ready to launch from
the available artifacts alone. Once dataset access is available, build a
reproducible crop/teacher-target pipeline, then distill all four heads into
the smaller fused student. Preserve screen landmarks, world landmarks,
presence and handedness semantics. Teacher agreement is not independent
ground truth; use labels/validated multiview references where available.
Split recordings by contributor/session before tuning. Set explicit limits
for decoded landmark displacement, confidence/threshold changes, handedness
and temporal jitter before selecting a model on heldout sequences. Training
compute needs its own measured pilot; no duration or accuracy promise is made.

## Recommended next experiment

Build one bounded packed-FP16 spatial-to-projection block at the half-width
candidate shapes, with real halo gathering and production-style DDR storage.
Also retain a diagonal original-weight control to validate the new data path
without training. Cover 112-, 14- and 7-wide maps; measure gather, coefficient
setup/reuse, HMX, post/store and total block time separately. Compare a CPU
precision model and test overflow, guarded faults and restored teardown.

The decision gate is whether measured blocks support <=12–20 ms total
convolution work after counting every operator, with <=5 ms for the rest.
If they do, proceed to a complete lifetime-checked experimental graph and
public-data teacher/student training. If streaming cannot improve enough,
reduce architecture/traffic further before committing to a training run.
Preserve one HMX owner, 20 ms per-request budget and 40 ms timeout throughout.

## Reproduction and validation

```sh
python tools/build_hmx_model_probe.py --output /path/jobs
# On the already-qualified device, with the existing root/resource setup:
python tests/hardware_hmx_model.py --run-on-device --jobs /path/jobs \
  --model /path/public-model --output /path/measurement.json
# Offline, no device or recording needed:
python tools/analyze_hmx_model.py --model /path/public-model \
  --measurement /path/measurement.json --output /path/model-costs.json
```

The device harness requires optional NumPy for independent CPU oracles.
The shape analyzer has no NumPy dependency. Public v20a model files are at
the pinned commit's
[model directory](https://huggingface.co/DeeJanuz/frametop-hands-models/tree/03d7695fecb80f6cf8f3aa9321de130b5efd08b3/v20a).

The feasibility session passed 407 restored submissions (405 HMX-flagged),
four dirty-accumulator guarded faults, four wrong-mode rejections and a
valid post-fault result. Maximum measured wrapper time was 2.792 ms. The
full resident arena hash matched before/after; firmware prefix, power,
gate, callback, poison flag and DSP service PID/restart count were unchanged.
The following ordinary SDK direct/benchmark regression passed another 78
restoration audits. There was no recovery, service restart or reboot.
Six new offline tests cover every packed lane, the diagonal identity,
nonlinear folding, channel-tail grouping and tile/width costs; 218 tests
pass in total. Production execution remains unchanged.
