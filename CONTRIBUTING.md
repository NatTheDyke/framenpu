# Contributing

Development works on an ordinary workstation without a headset.

~~~sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,cpu]" "numpy>=1.26"
python -m unittest discover -s tests -p 'test_*.py' -v
ruff check .
ruff format --check .
python tools/check_repository.py
~~~

Python 3.11-3.14 are in the CI matrix. Without NumPy the numerical tests skip; the model-gated
tests need the weights (`tools/fetch_models.py DIR --env`).

Add tests for behavior, invalid inputs and meaningful failure paths. framenpu must import and run
with no device backend installed; device code belongs in
[framenpu-cdsp](https://github.com/NatTheDyke/framenpu-cdsp). Preserve the contracts in
[docs/CONTRACTS.md](docs/CONTRACTS.md) or version them explicitly. After changing
`src/framenpu/host`, rebuild the bundled library with `tools/build_host_native.py`.

Do not commit credentials, raw captures, proprietary weights or opaque binaries without
corresponding source. Contributions use the repository's MIT license.
