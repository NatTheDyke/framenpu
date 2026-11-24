# framenpu

Hand-tracking models for the Steam Frame: one session API over three places to run them: the
CPU (onnxruntime), the CDSP, or the root NPU service that shares the CDSP with Valve's tracking.

| Package | What it holds |
| --- | --- |
| [framehands](https://github.com/NatTheDyke/framehands) | Hand tracking: crops, decode, fit, filters |
| **framenpu** | Models, sessions, CPU fallback, the NPU service |
| [framenpu-cdsp](https://github.com/NatTheDyke/framenpu-cdsp) | The CDSP device backend: transport, kernels, DSP payloads |

This is alpha research software, not endorsed by Valve or Qualcomm. No model weights are bundled.

## Install

~~~sh
python -m pip install "framenpu[cpu] @ git+https://github.com/NatTheDyke/framenpu"
python -m pip install git+https://github.com/NatTheDyke/framenpu-cdsp     # on the Frame
~~~

Python 3.11+. framenpu runs alone on the CPU; without framenpu-cdsp, opening the device raises
`framenpu.devices.DeviceUnavailable`.

## Use

~~~python
import framenpu

s = framenpu.open(["models/mercury-ft7/model.json"], backend="auto")  # service, else CPU
outputs = s.infer("mercury-ft7", crops)  # 1..s.max_crops(...) crops, never padded
s.close()
~~~

`framenpu serve --model M.onnx --user steamos` runs the NPU service (root, on the Frame).
`tools/fetch_models.py DIR` fetches the weights named by the
[bundled manifests](src/framenpu/manifests/README.md), by hash. The
[contracts](docs/CONTRACTS.md) define the session API, the model manifests, the device plugin ABI
and the service protocol.

## Tests

~~~sh
python -m pip install -e ".[dev]" "numpy>=1.26"
python -m unittest discover -s tests -p 'test_*.py'
ruff check . && ruff format --check .
python tools/check_repository.py
~~~

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). The project is
[MIT-licensed](LICENSE); see [third-party provenance](THIRD_PARTY_NOTICES.md).
