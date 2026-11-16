# Custom job ABI v1

Device.submit accepts a trusted Job, input bytes, output byte count, opcode and up to six uint32 parameters. JobResult contains output bytes, elapsed time including the stock benchmark, and three raw worker status rows.

Importing the library opens no hardware. The connection validates firmware and CDSP identity, receives DMA descriptors and maps the uncached alias. A reentrant lock serializes submissions within a Device; the existing atomic operation gate coordinates with participating clients, subject to its firmware timeout.

## Kernel entry and context

Implement fnpu_u32 frame_job(const fnpu_context *ctx). Return zero or an error. The SDK links the common entry at DSP VA0xf0a92000.

The wrapper accepts workers0/1/2 and runs developer code only on worker0. It saves/restores callee-saved registers, SSR and exception-vector base. HMX jobs temporarily enable SSR.XE2 and back up/restore8KiB VTCM. A handler records selected synchronous faults; it is not a complete exception subsystem or arbitrary pipeline-stall recovery. Each worker publishes completion in the same instruction packet as its final return; the host waits for all three.

The new opt-in [direct dispatcher](DISPATCH.md) enters the wrapper on context 4
only to wake workers in callback-only mode and join their returns. HMX/HVX
and `frame_job` remain on worker 0. Direct-compatible FJOBs declare the optional
`supports_direct=true` field; old files need rebuilding for that backend.
Status words 10..13 hold the low/high words of worker-0 reference start/end
counters. They cover `frame_job`, excluding entry/exit save and restore work.

The context consists of16 little-endian uint32 words:

| Word | Field |
|---:|---|
| 0 | nonzero unique sequence |
| 1 | magic0x55504e46 (FNPU) |
| 2 | ABI1 |
| 3 | flags: bit0 HMX, bit1 direct dispatch |
| 4 | used arena bytes |
| 5,6 | input offset and length |
| 7,8 | output offset and length |
| 9 | opcode |
| 10..15 | six parameter words |

Offsets are relative to arena VA0xf0a97000. Use fnpu_input/fnpu_output. Input starts at zero; output aligns to128 bytes. Combined length is bounded to128KiB. Developer kernels must not modify the header/control/callback fields.

Code reservation:0xf0a92000,4KiB. Context:0xf0a93000. Status:0xf0a93400, three64-byte rows. VTCM backup:0xf0a94000. Status word0 is sequence,1 worker ID,2 error; errors1 invalid arguments,2 captured exception,3 unsupported opcode. Exception SSR/ELR/badva use words3..5; other words belong to the wrapper.

Kernels run as trusted native supervisor code, validate their own lengths/offsets and preserve the C ABI. No libc, allocator, filesystem, dynamic linker, QuRT or FastRPC environment is supplied. No BSS or unresolved helper symbols are allowed. Mark HMX kernels with --hmx to request power/scratch/SSR setup. Do not use VTCM outside the reserved8KiB or modify unrelated firmware memory.

## File formats

FJOB contains four-byte magic, uint32 JSON-metadata length, UTF-8 metadata and raw fixed-address code. Metadata includes ABI, name, HMX flag and SHA256. Checksums detect corruption, not authenticity.

FNPU models are bounded ZIP archives with model.json and declared weight members. Manifest format is framenpu-model version1, with input feature count and sequential layers. Dense weights are FP16[K,N]; biases FP32[N]. Tensor descriptors include shape/dtype/SHA256. Duplicate/undeclared members, size mismatches, nonfinite values and unsupported operators are rejected. The loader never extracts member paths or executes model contents.

## Completion window and failures

The stock master has a roughly50ms acknowledgment budget. Firmware at0x870006bc materializes0x02faf080 (50million units in its nanosecond conversion); expiry at0x87000750..758 resets state/ownership to1. It does not wait indefinitely for custom callbacks.

By default, built-ins schedule one output/depth32 tile per HMX submission and
use separate DSP addition for depth accumulation. Opt-in
[two-tile mode](MULTITILE.md) combines two depth products for one output tile
and continues from an optional FP32 seed; opcode 6 requires the HMX and direct
flags. Host timeout is capped at40ms. Custom kernels must obey this budget.
This is not hard-real-time scheduling: a delayed host can still lose the stock lease.

On normal completion, the host clears the callback, powers HMX down, checks register/program restoration, restores all uploaded spans and acknowledges the stock command. Worker errors are raised after cleanup. Preflight validation failures write no device memory.

RecoveryRequired marks the session unusable and preserves potentially active payload/power. The stock gate can expire independently. Closing Python descriptors does not cancel DSP work. Do not overwrite a live payload or power off a possibly active engine. Deliberate power/config recovery and DSP reset must establish idle state before reuse; there is no automatic restart.

Direct mode joins workers before the master publishes stock completion. Its
40 ms host timeout and the roughly 50 ms acknowledgment window remain; it does
not add native-job cancellation. A poisoned direct session retains its firmware
edit, and a later process rejects that modified window until deliberate recovery.
Partial firmware edits are checked before helper acknowledgment.

A production backend still needs persistent model arenas, explicit tensor/power
ownership and cooperative cancellation. This release exposes a version-pinned
research interface.

The public entry now restores both SP and FP when abandoning nested C frames
after a guarded exception. [Fault-unwind qualification](FAULT_UNWIND.md) covers
48 traps and following same-session work across both dispatch paths and CPU/HMX
jobs. The rebuilt runtime, vector-add example and patch helper contain the fix.
Existing custom FJOBs must be rebuilt; their embedded entry is not upgraded by
installing a new Python package. Guarded software traps do not establish
arbitrary native-fault or pipeline-stall recovery.
