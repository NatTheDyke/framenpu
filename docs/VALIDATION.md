# Validation

## Real-device baseline — 2026-11-14

FrameNPU 0.1.0 ran on an authorized Steam Frame with SteamOS 0.5.8, CDSP ABI 9,
and Hexagon v75. The sanitized [numerical summary](../evidence/validation.json)
records the baseline, payload hashes and limitations.

The final hardware suite passed:
- A compiled C vector-add job produced [6, 8, 10, 12].
- An unsupported opcode raised JobFailed after cleanup; the session was reusable.
- GEMMs with (M,K,N) shapes (1,1,2), (4,2,1), (31,17,33), (32,32,32) and
  (33,65,35) exactly matched the independent tile-precision CPU reference.
- DSP ReLU/clipping and a serialized, reloaded two-layer XOR model passed.
- Final workers returned baseline SSR values, callback was zero, control was
  (1,0,1), and HMX clock/program configuration returned to its inactive baseline.

These exact inputs do not establish every FP16 rounding/exceptional-value case.
Sustained eye-tracking accuracy, latency and thermals were not measured.

The hardware-tested runtime is 2,972 bytes, SHA256
068cadcad806f1ac76f9f46c72f284ba89759c1e7990bb0e2ba9dda4c928c1df.
The included custom C job is 1,844 bytes. Both are built from included source
without proprietary SDK/runtime libraries.

## Completion-window finding

An initial monolithic (33,65,35) GEMM completed all 1,155 outputs correctly, but
the host missed the stock master's roughly 50 ms completion lease. This was
not a hardware pipeline stall. A controlled recovery verified completion,
restored HMX configuration, and restarted the DSP service once to reinitialize
abandoned shared buffers.

The backend now schedules bounded depth/output-32 HMX jobs and separate DSP
accumulation with a 40 ms host timeout and expired-lease detection. This does
not provide hard-real-time or cross-client isolation. See [ABI.md](ABI.md).

## Publication audit

The repository provides an MIT license, self-contained examples, SDK/compiler
source provenance, offline validation, packaging checks and CI. Raw captures,
machine paths, credentials and proprietary firmware are excluded from the
public source; the numerical summary is intentionally sanitized.

The publication suite passes 29 offline tests. Local verification uses
Python 3.14.9 and LLVM 23.1.4. The GitHub Actions
matrix is configured for Python 3.11–3.14; it has not run on GitHub before upload.
Offline power tests use an emulator, not live fault injection.

The release checks verify a clean exported source tree, wheel/sdist metadata,
an isolated wheel installation, CLI/model inspection and SDK resources.
Repository-maintenance changes do not themselves constitute another hardware
validation run. The included payload's byte identity can be checked with
tools/build_kernels.py --check.
