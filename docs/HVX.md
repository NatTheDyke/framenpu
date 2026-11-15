# HVX operand packing

This page records the initial input-packing change and its timing comparison.
The subsequent [HVX output readback change](READBACK.md) retains these packing
paths and adds vector output conversion with its own validation.

The built-in GEMM runtime now uses HVX to pack complete 32x32 FP16 tiles
into the HMX row-pair layout. This applies automatically to both dispatch
backends and both supported tile budgets. The matrix API and model format
are unchanged.

## Packing and bounds

HMX element `(row, column)` occupies halfword
`(row / 2) * 64 + column * 2 + (row % 2)`. Each 128-byte packed vector
alternates elements from two adjacent rows.

For a row stride of 32 FP16 values, one aligned vector load contains both
rows; `vshuff` on halfwords interleaves them. For a stride of 64, two aligned
loads contain two complete rows. A vector-pair shuffle interleaves them,
then selects columns 0..31 or 32..63 for the current depth tile. The helper
checks the source-row base alignment, independently of the selected column.
The [Qualcomm-authored instruction semantics in QEMU](https://github.com/qemu/qemu/blob/master/target/hexagon/imported/mmvec/ext.idef)
and LLVM's installed intrinsic headers provide instruction references.

The fast path requires at least 32 valid rows and columns, stride 32 or 64,
and a 128-byte-aligned source-row base. Partial tiles, other strides and
unaligned bases retain scalar packing with zero padding. Vector loads stay
within complete source rows. No input padding or widened read permission is
required. The destination remains in the existing saved/restored 8 KiB VTCM
scratch span.

Copy probe opcode 5 also transfers aligned 128-byte blocks with HVX, followed
by scalar tail bytes. Unaligned offsets retain scalar copying. HMX sequencing,
FP16 product rounding, sequential FP32 accumulation, bias/ReLU and scalar
output readback/conversion retain their previous behavior. No clock, memory
reservation or execution-budget change accompanies this optimization.

## Device validation — 2026-11-14

The runtime is 3,520 bytes, SHA256
`315a3eff8b851e3a8b31cf43477aeeace9a3f26dd4a57155491c2d3e5f33027b`.
LLVM disassembly confirms HVX loads, shuffles and stores in the packing helper
and vector block transfers in the copy probe.

The hardware fixture includes the runtime source and calls its actual packing
helper with a DDR destination. It adds no opcode to the shipped runtime.
All 65,536 halfword bit patterns passed through the stride-32 path and across
both stride-64 half selections, preserving NaNs, infinities and subnormals as
bits. Together with padded/unaligned fallbacks, this covered 136 layout cases.
Another 53 copy cases covered zero length, vector boundaries, tails, unaligned
offsets and the largest disjoint 64 KiB input/output pair in the arena.

Six GEMMs exercising combinations of fast and scalar operand packing matched
the independent reference exactly. Both existing dispatch-backend numerical
suites and the multi-tile suite also passed, including the saved XOR model,
ordered FP32 rounding, accumulator range and native argument rejection checks.
Packing bit preservation does not establish all exceptional-value arithmetic
behavior of HMX.

The three correctness runs audited restoration for 310 submissions. Borrowed
DDR spans, callback and power configuration matched their values captured at
gate acquisition. Final worker SSRs, dispatch window and first 112 KiB of
firmware code matched baseline; control was `(1,0,1)` with zero callback. The
DSP service retained its PID with zero restarts.

## Timing

The qualified 3,380-byte scalar runtime and new runtime executed identical
wire requests on direct dispatch, with 100 warm samples after one excluded
first call per case. Restoration auditing was disabled during timing. The
scalar runtime ran first, followed by HVX, under the same power policy.

| Publication-to-completion | Scalar p50 / p99 | HVX p50 / p99 |
|---|---:|---:|
| 4 KiB copy | 8.861 / 9.210 ms | 0.640 / 0.739 ms |
| 32x32x32 GEMM | 14.611 / 14.855 ms | 10.997 / 11.242 ms |
| 32x64x32 GEMM, two depth tiles | 25.281 / 25.617 ms | 15.789 / 16.018 ms |

The two-tile job's median publication interval improved about 37.5%. Complete
`Device.submit` wall-time p50 was 35.273 ms versus 23.782 ms for that case;
this includes upload, power setup/teardown and cleanup, and excludes Python
matrix serialization. These are different boundaries from the full
`Device.matmul` timings in [MULTITILE.md](MULTITILE.md).

Raw worker-0 `frame_job` reference-tick p50 fell from 268,745 to 199,238 for
one depth tile and from 473,462 to 291,505 for two. These counters include
packing, HMX and scalar readback; they do not isolate any one stage. The first
scalar copy also installed the dispatcher, so first-call times are not
comparable. This sequential comparison does not establish sustained tracking
latency, gaze coexistence or thermals. The 40 ms host timeout and two-depth-tile
cap remain in force.

See the [sanitized summary](../evidence/hvx-validation.json) for all retained
percentiles, identities and limitations. Output readback/conversion is covered
by the later [readback validation](READBACK.md); persistent weights still
require an allocation/ownership design.

## Reproduction

Build the fixture on a workstation with the recorded LLVM toolchain:

~~~sh
python tools/build_kernels.py --source tests/packing_probe.c --output /tmp/packing_probe.fjob
~~~

Copy the source tree and fixture to the supported Frame, then explicitly opt
into hardware testing:

~~~sh
sudo python tests/hardware_hvx.py --run-on-device --probe /tmp/packing_probe.fjob
sudo python tests/hardware_hmx.py --run-on-device --iterations 5
sudo python tests/hardware_multitile.py --run-on-device --correctness-only
~~~

To reproduce the comparison, additionally supply `--iterations 100 --baseline
/path/to/scalar-runtime.bin` to `hardware_hvx.py`. The baseline must match SHA256
`6a7eaa2a42bb727f6bec3b4036a8d377edc0b77dfd936dea78c3e9245d690ef2`.
The fixture is generated locally and is not a bundled executable resource.
