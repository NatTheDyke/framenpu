# HVX output readback

Built-in GEMM now unpacks HMX output and converts FP16 products to FP32 with
HVX integer operations. This applies automatically on both dispatch backends,
for both supported tile budgets and all supported output tile shapes. The
matrix API and model format retain their current behavior.

## Conversion and accumulation

An HMX output vector contains 32 words, with the even row in each word's low
halfword and the odd row in its high halfword. Shifting the word lanes by
0 or 16 selects one output row. Integer masks, shifts and exponent adjustment
then produce FP32 bit patterns. Subnormal fractions are normalized with
`vnormamt`; zero and exponent-31 inputs receive explicit handling.

This preserves signed zeros, infinities and all NaN payload/signaling bits at
the conversion boundary. The implementation uses no HVX floating-point
arithmetic and changes no compiler floating-point mode. The existing scalar
FP32 addition consumes each converted product in the original depth-tile
order. Its NaN handling and rounding remain those of the previous scalar
implementation. Bias and optional ReLU still apply after reduction.

The shared [SDK helper](../src/framenpu/sdk/include/fnpu_readback.h) is included
by the runtime and hardware fixture, and its source hash is part of payload
provenance. LLVM disassembly confirms vector extraction/conversion and scalar
`sfadd` accumulation. Readback has no vector stack spills; keeping it in a
separate native function confines its vector constants to that function.

## Memory and bounds

Each converted row uses 128 bytes within the existing saved/restored VTCM
scratch reservation:

| Purpose | DSP address | Bytes |
|---|---|---:|
| Packed A | `0x31000000` | 2,048 |
| Packed B | `0x31000800` | 2,048 |
| HMX output | `0x31001000` | 2,048 |
| Scale table | `0x31001800` | 256 |
| Converted FP32 row | `0x31001900` | 128 |

The row buffer is reused after the HMX store/barrier completes. The wrapper
continues saving/restoring the entire first 8 KiB of VTCM for HMX-marked jobs.
No reservation or arena expansion accompanies this change.

Readback loads complete vectors from the fixed, padded HMX output tile. It
accumulates only valid rows and columns into the logical FP32 output. Odd row
counts, partial columns, row padding and outputs aligned to four bytes are
supported without a scalar conversion fallback. The separate operand-packing
fast path retains the bounds described in [HVX.md](HVX.md).

## Device validation — 2026-11-14

The runtime is 3,672 bytes, SHA256
`93b242ba9d7f7bc41462a85f77ad0a6e95ecaee413bcaf0b87795098cf877fe8`.
The hardware-only fixture is 2,944 bytes and adds no production opcode.

The native conversion helper returned the exact expected FP32 bits for all
65,536 FP16 patterns. A separate scalar reference retained from the qualified
previous implementation matched seeded accumulation bit-for-bit in 71 cases.
These include all FP16 patterns, FP32 subnormal seeds, signed zeros, large
seeds, odd rows, partial columns, output offsets 4/64/124 bytes and row strides
through 64. Guard bytes and row padding were preserved. Finite cases also
matched an independent host FP32 reference.

Five mixed-edge/tiled GEMMs passed both tile budgets. Existing numerical suites
passed on both dispatch backends, including the saved XOR model; multi-tile
rounding/order, seed range, bias/ReLU and argument rejection checks passed.
The existing 136 packing cases and 53 copy cases also passed with this runtime.

Across four correctness runs, 573 submissions passed restoration audits,
including 180 HMX-marked submissions. Borrowed DDR, callback and power
configuration matched their gate-acquisition values before acknowledgment.
Final worker SSRs, dispatch window and first 112 KiB of firmware code matched
baseline. Control was `(1,0,1)`, callback zero, and the DSP service retained its
PID with zero restarts.

## Timing

The qualified 3,520-byte runtime with HVX input packing and scalar readback
was compared against the new runtime using identical direct-dispatch wire
requests. Each case used one excluded first call and 100 warm samples;
restoration audits were disabled during timing. The dispatcher was installed
before either runtime's measurements. Both used the same power policy, with
the scalar-readback runtime measured first.

| Publication-to-completion | Scalar readback p50 / p99 | HVX readback p50 / p99 |
|---|---:|---:|
| 32x32x32 GEMM | 10.989 / 11.217 ms | 5.479 / 6.032 ms |
| 32x64x32 GEMM, two depth tiles | 15.818 / 16.189 ms | 6.988 / 7.132 ms |

The two-depth-tile median publication interval fell about 55.8%. Complete
`Device.submit` wall-time p50 was 24.570 ms versus 14.182 ms for that case,
including upload, HMX power setup/teardown and cleanup, and excluding Python
matrix serialization. Its wall-time p99 was 35.571 ms versus 28.370 ms.
These boundaries differ from full `Device.matmul` call timing.

Raw worker-0 `frame_job` reference-tick p50 fell from 199,592 to 93,474 for
one depth tile and from 291,803 to 122,377 for two. These intervals include
packing, HMX, conversion, scalar accumulation and bias. They do not isolate
conversion throughput or establish real-time inference performance.

The 40 ms host timeout, two-depth-tile cap and memory reservations remain in
force. Scalar FP32 accumulation, initialization and bias/ReLU remain separate
optimization opportunities. Persistent model ownership and sustained tracking
coexistence/accuracy/thermals remain unvalidated. The
[sanitized summary](../evidence/readback-validation.json) records retained
percentiles, identities and limitations.

## Reproduction

Build the fixture on a workstation using the recorded LLVM toolchain:

~~~sh
python tools/build_kernels.py --source tests/readback_probe.c --output /tmp/readback_probe.fjob
~~~

Copy the source tree and fixture to the supported Frame, then explicitly opt
into hardware testing:

~~~sh
sudo python tests/hardware_readback.py --run-on-device --probe /tmp/readback_probe.fjob
sudo python tests/hardware_hmx.py --run-on-device --iterations 5
sudo python tests/hardware_multitile.py --run-on-device --correctness-only
~~~

For the timing comparison, also pass `--iterations 100 --baseline
/path/to/scalar-readback-runtime.bin` to `hardware_readback.py`. That baseline
must match SHA256
`315a3eff8b851e3a8b31cf43477aeeace9a3f26dd4a57155491c2d3e5f33027b`.
Rebuild `tests/packing_probe.c` to repeat the packing/copy regression with
`tests/hardware_hvx.py`. Neither fixture is a bundled executable resource.
