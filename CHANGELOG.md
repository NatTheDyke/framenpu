# Changelog

## Unreleased

- Checkpointed portable HMX research fixtures, runners and sanitized evidence.
  Cached-code and two-block-load diagnostics are included as experiments with
  offline checks only; no hardware performance result is claimed for them.
- Documented the verified 5.87 MiB persistent DDR region. Resident allocation
  and graph execution remain future work. Publication checks exclude local
  handoff/recovery files and scan the portable multi-worker builder in source
  archives.
- Numerically checked single-owner resident GEMM baseline: all output tiles
  match across 400 measured jobs and 16 warmups, with a 7.41–7.44 GMAC/s load
  slope. Added separate HMX, packing, copy, conversion and host-phase timings,
  repeated latency quantiles, reproducible fixtures and sanitized evidence.
- Full-tile accumulator isolation tests across workers 0/1/2: all isolated and
  serialized controls pass; deferred stores and concurrent transactions expose
  shared HMX state. The recovered clear/load/store sequence must stay under one
  owner for the complete transaction. Added bounded synchronization, phase
  timing, exact per-worker references and sanitized hardware evidence.
- HVX output unpacking and exact FP16-to-FP32 bit conversion for every supported
  GEMM output tile, retaining scalar FP32 accumulation order. The 128-byte row
  buffer stays inside existing VTCM scratch; the runtime is 3,672 bytes.
- Native validation covered all 65,536 FP16 conversion patterns, 71 seeded
  accumulation cases, numerical regressions and 573 restoration audits. Over
  100 warm direct-dispatch samples, two-depth-tile 32x64x32 publication-to-completion
  p50 fell from 15.82 to 6.99 ms against the previous input-packing runtime.
- HVX packing for aligned full FP16 tiles with row strides 32/64, plus aligned
  vector bulk copy with scalar tails. Padding and unaligned cases retain scalar
  paths. Hardware qualification passed 136 layout cases, 53 copy cases and the
  existing numerical suites; 310 submissions passed restoration audits.
- Against the previous runtime, two-depth-tile 32x64x32 publication-to-completion
  p50 fell from 25.28 to 15.79 ms over 100 warm direct-dispatch samples. The
  runtime remains within 4 KiB at 3,520 bytes.
- Opt-in `matmul(..., tiles_per_job=2)` on direct dispatch: two depth products
  per HMX job, ordered FP32 accumulator seeds and final-only bias/ReLU. A
  32x64x32 device comparison reduced four submissions to one and median wall
  time from 60.89 ms to 38.29 ms. The 40 ms execution timeout is unchanged.
- Context flag bit 1 identifies direct dispatch; the new multi-tile opcode
  rejects the benchmark path and validates its two-product bound on the DSP.

- Opt-in reversible direct dispatch for the pinned ABI 9 firmware, bypassing
  stock benchmark arithmetic and joining workers before master completion.
- Direct-compatible SDK entry metadata, worker-0 kernel timing counters and
  explicit empty/copy/GEMM comparison tooling.
- Host lifecycle/ownership failure tests and reproducible dispatcher/helper
  builds. CPU/DSP device validation passed, including 100 warm samples,
  metadata/client compatibility and firmware restoration.
- HMX qualification passed on both dispatch paths, with all five padded/tiled
  GEMMs and the saved XOR model matching their references. A new opt-in harness
  audited 78 correctness jobs (39 HMX jobs) and separately measured 100 warm
  samples per backend. Tracking coexistence remains pending; the default
  remains the benchmark backend.
- Correctly wait for master completion when direct workers return while the
  master still owns processing state 2.

## 0.1.0 — initial experimental release

- Python Device/Job APIs and a freestanding Hexagon v75 C SDK.
- Tiled HMX GEMM, DSP accumulation, bias/ReLU, clipping and broadcast add.
- Checked Dense/ReLU/Clip models and synthetic examples.
- Firmware guards, bounded jobs and explicit recovery states.
- Real-device custom-job, tiled-GEMM and serialized XOR validation.
- Standalone packaging, development docs, CI and provenance checks.

Full FrameTop/ncnn graph inference and real-time hand tracking are unsupported.
