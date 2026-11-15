# FrameNPU

**Experimental custom DSP jobs and HMX inference for Steam Frame.**

FrameNPU provides a Python host API, a freestanding Hexagon v75 C SDK, and a
small data-only model format. Real-device research has verified custom C jobs,
tiled HMX matrix multiplication, DSP accumulation/activations, and a saved XOR
network loaded from disk.

This is an alpha research library. It supports one exact firmware profile,
does not isolate native jobs, and is not yet suitable for real-time hand
tracking. Read [compatibility](#compatibility-and-limits) before device use.

## Install from a checkout

From the repository root on a development workstation:

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
~~~

Python 3.11+ is required. The library has no external Python runtime
dependencies. LLVM is needed only to rebuild or compile kernels. Installing,
inspecting models, and running offline tests require no headset or root access.

For a wheel you have built or downloaded, install that wheel in your virtual
environment with python -m pip install /path/to/framenpu-0.1.0-py3-none-any.whl.
A source checkout does not assume that ignored build/release files exist.

## Create and inspect a model without hardware

~~~sh
python examples/make_xor.py
framenpu inspect-model examples/xor.fnpu --batch-size 4
framenpu inspect-ncnn examples/dense.param
~~~

The synthetic examples are included. Model inspection and ncnn graph inventory
do not execute DSP code. The ncnn command is an inventory tool, not an importer.

Supported model layers are Dense, ReLU and Clip:

~~~python
from framenpu import Dense, Model

Model(
    2,
    [
        Dense([[1, -1], [-1, 1]], activation="relu"),
        Dense([[1], [1]]),
    ],
    name="xor",
).save("xor.fnpu")
~~~

Weights are FP16 [input_features, output_features]; biases and outputs are FP32.
The loader checks supported operators, dimensions, byte lengths and SHA256.
FNPU files contain declarative data, not executable kernels or pickle.

## Run on a supported Frame

Install the package in a virtual environment on the headset. Use the
interpreter from that environment when requesting root for HMX setup:

~~~sh
.venv/bin/python -m framenpu info
sudo .venv/bin/python examples/run_xor.py
.venv/bin/python examples/run_custom.py
~~~

The examples locate their bundled files relative to the scripts, not the
working directory. Override them with --model or --job if needed.

Plain custom DSP jobs use the existing steamos service access. HMX setup needs
root access to /dev/mem. Unknown firmware is rejected before job writes.

~~~python
from framenpu import Device, Model

model = Model.load("xor.fnpu")
with Device() as npu:
    output = model.run(npu, [[0, 0], [0, 1], [1, 0], [1, 1]])
print(output)  # [[0.0], [1.0], [1.0], [0.0]]
~~~

## Matrix and custom-kernel APIs

~~~python
from framenpu import Device

with Device() as npu:
    product = npu.matmul([[1, 2, 3]], [[1], [2], [3]])  # [[14.0]]
    clipped = npu.elementwise([[-2, 3, 9]], activation="clip", maximum=6)
    summed = npu.add([[1, 2], [3, 4]], [[10, 20]])  # row broadcast
~~~

Direct dispatch also supports `npu.matmul(a, b, tiles_per_job=2)`, combining
two depth tiles per submission and preserving FP32 accumulation between
chunks. In a 32x64x32 device test, median wall time fell from 60.89 ms to
38.29 ms, with four submissions reduced to one. This option keeps the 40 ms
execution timeout and is opt-in; see [MULTITILE.md](docs/MULTITILE.md).

Full aligned tiles now use HVX packing automatically. A same-request comparison
reduced two-depth-tile 32x64x32 job publication-to-completion p50 from 25.28 ms
to 15.79 ms. Padded edges retain scalar packing. See [HVX.md](docs/HVX.md) for
correctness checks, timing boundaries and reproduction.

HVX output unpacking/conversion now applies to every supported GEMM tile shape,
while retaining the existing scalar FP32 accumulation. A subsequent comparison
reduced the same two-depth-tile job interval from 15.82 ms to 6.99 ms; see
[READBACK.md](docs/READBACK.md).

Implement [the C job ABI](src/framenpu/sdk/include/framenpu.h), as demonstrated
by [vector_add.c](examples/vector_add.c), and compile on your workstation:

~~~sh
framenpu build-job examples/vector_add.c --output build/vector_add.fjob
~~~

Copy the FJOB to your Frame and submit it:

~~~python
import struct
from framenpu import Device, Job

job = Job.load("vector_add.fjob")
with Device() as npu:
    result = npu.submit(
        job, struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8), output_size=16, params=(4,)
    )
print(struct.unpack("<4f", result.output))  # (6, 8, 10, 12)
~~~

The compiler needs Hexagon-capable clang, ld.lld and llvm-objcopy. LLVM 23.1.4
produced the included payloads. Select executable paths with CLANG, LD_LLD and
LLVM_OBJCOPY; a standard-library/compiler runtime is not linked into the job.
The wheel includes SDK sources, the header, entry wrapper and linker script.

Native jobs are trusted supervisor code, not sandboxed model data. See
[ABI.md](docs/ABI.md) and [SECURITY.md](SECURITY.md).

## Compatibility and limits

| Item | Current support |
|---|---|
| Hardware | SM8650 / Hexagon v75 Steam Frame |
| Tested firmware | SteamOS 0.5.8, CDSP ABI 9; exact hashes in [profile.py](src/framenpu/profile.py) |
| Models | Sequential Dense, ReLU and Clip |
| Kernels | At most 4 KiB native code; 128 KiB job arena |
| HMX scratch | First 8 KiB of VTCM, saved/restored |
| Matrix dimensions | Up to 256, scheduled in bounded depth/output-32 tiles |
| Arithmetic | FP16 depth-32 products, FP32 DSP accumulation across depth tiles |

The default backend executes a stock FLOPS benchmark per submission and scopes
HMX power to that job. This is unsuitable for real-time hand tracking.

An opt-in `Device(dispatch="direct")` prototype bypasses the benchmark using a
reversible 96-byte runtime dispatch edit. It keeps HMX/HVX on worker 0 and joins
all workers before completion. CPU/DSP validation passed: median warm empty-job
wall time fell from 4.29 ms to 0.73 ms across 100 samples per backend.
Both dispatch paths also pass padded/tiled HMX GEMMs and the saved XOR model;
median 32x32 GEMM submission wall time was 29.34 ms versus 25.36 ms with direct
dispatch. **Sustained tracking coexistence remains unvalidated.**
See [DISPATCH.md](docs/DISPATCH.md) for results, lifecycle, compatibility and
the explicit hardware comparison commands.

The stock dispatcher releases its completion/ownership window after roughly
50 ms even without acknowledgment; the host timeout is capped at 40 ms.
OS scheduling and other clients are not isolated. On RecoveryRequired, stop
using the session: possibly active payload/power are retained and deliberate
DSP/power recovery is required. The library never automatically restarts
tracking services. See the [failure contract](docs/ABI.md).

No proprietary firmware or hand-model weights are bundled. Power sequencing
is recovered from the exact firmware already installed on the device.

## Tests and reproducible builds

~~~sh
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check .
ruff format --check .
python tools/check_repository.py
python tools/build_kernels.py --check
python -m build
python -m twine check dist/*
python tools/check_distribution.py --dist dist
~~~

Only the kernel rebuild/check step needs LLVM. The distribution check rebuilds
from the source archive, installs the wheel in a fresh environment, and checks
the CLI, models, SDK resources and examples without a headset.

The separate hardware test is opt-in and is never run by ordinary CI:

~~~sh
sudo .venv/bin/python tests/hardware_smoke.py --run-on-device
~~~

Configured CI covers Python 3.11–3.14. [Validation notes](docs/VALIDATION.md)
record 81 passing offline tests and distinguish local results from the
configured CI matrix.

## Hand tracking and contributions

The full FrameTop v20a graph is not implemented. Use inspect-ncnn with your
own hand.ncnn.param file to inventory it. Bounded two-depth-tile jobs are
implemented with validated HVX input packing and output conversion.

A [5.87 MiB persistent DDR region](docs/MEMMAP.md) is mapped and canary-checked;
resident tensor allocation and graph execution remain unimplemented. The
[accumulator isolation tests](docs/HMX_ACCUMULATOR.md) show shared HMX state
across workers with the recovered primitives. Keep each complete clear/load/store
transaction under one HMX owner; use HVX workers for other work.

The [correct resident GEMM baseline](docs/HMX_BASELINE.md) matched every output
across 400 measured jobs and 16 warmups, with a 7.41–7.44 GMAC/s load slope.
That uncached loop was limited by instruction fetch. A subsequent
[cached-code comparison](docs/HMX_CACHE.md), at unchanged clock configuration,
passed 800 measured jobs and 32 warmups with exact outputs. For four deep
resident transactions, HMX p50 fell from 9.51 ms to 56.25 us; the fitted cached
load slope is 1,264–1,269 GMAC/s. A separate
[two-block-load diagnostic](docs/HMX_BATCHING.md) passed another 800 measured
jobs and 32 warmups, with about 2.52x improvement in its uncached loop.

These remain experimental fixtures; the public runtime is unchanged. They
reuse blocks within a job and do not establish full-model throughput. Cached
publication/cleanup costs roughly 1.5 ms per job, and fresh sessions still take
about 33 ms. Next are resident code/tensor/weight ownership, real convolution
and HVX depthwise kernels, then a validated ncnn converter. See
[HAND_TRACKING.md](docs/HAND_TRACKING.md) for the roadmap and
[HMX_THROUGHPUT.md](docs/HMX_THROUGHPUT.md) for research context.

[CONTRIBUTING.md](CONTRIBUTING.md) explains development and kernel changes.
The project is [MIT-licensed](LICENSE); see
[third-party provenance](THIRD_PARTY_NOTICES.md). This independent research
project is not endorsed by Valve or Qualcomm.
