"""Fetch the model weights framenpu's bundled manifests name, by hash, into one folder.

    python3 tools/fetch_models.py DIR [--only NAME ...] [--env]

A manifest with a ``source`` comes from that Hugging Face repo at the recorded commit. Mercury's
own models (the jan18 keypoint model and the 160x160 detector) have no ``source``: their weights
carry no published license, so nothing re-hosts them, and they come from Monado's model
repository at the commit framenpu was checked against. Every file must match its manifest's
SHA-256; a file already in DIR with the right hash is kept, a wrong one is an error (never
overwritten). --env prints the FNPU_* variables the model-gated tests read. Standard library only.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

MANIFESTS = Path(__file__).resolve().parents[1] / "src" / "framenpu" / "manifests"
HF = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
MONADO = (
    "https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models/-/raw/"
    "37a8a81bc8f433ac6cbdf2471909d2bac74beca1/{file}"
)
MONADO_MODELS = ("mercury-keypoint-jan18", "mercury-detector-160")
# the model-gated tests' variables (tests/test_mercury_*.py, test_native_host.py)
ENV = {
    "mercury-keypoint-jan18": "FNPU_MERCURY_MODEL",
    "mercury-detector-160": "FNPU_MERCURY_DETECTOR",
    "mercury-stereo3d-tpc1": "FNPU_S3D_MODEL",
}


def manifests():
    return {
        d["name"]: d for d in (json.loads(p.read_text()) for p in sorted(MANIFESTS.glob("*.json")))
    }


def url_of(m):
    file = m["weights"]["file"]
    if m.get("source"):
        return HF.format(file=file, **m["source"])
    if m["name"] in MONADO_MODELS:
        return MONADO.format(file=file)
    raise SystemExit(f"{m['name']}: no source to fetch from")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(m, out):
    want, dest = m["weights"]["sha256"], out / m["weights"]["file"]
    if dest.exists():
        got = sha256(dest)
        if got != want:
            raise SystemExit(f"{dest}: SHA-256 {got}, the manifest says {want}; not overwriting")
        return dest, "kept"
    fd, tmp = tempfile.mkstemp(dir=out, prefix=".fetch-")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url_of(m), timeout=120) as r:
            while block := r.read(1 << 20):
                f.write(block)
        got = sha256(tmp)
        if got != want:
            raise SystemExit(f"{m['name']}: downloaded SHA-256 {got}, the manifest says {want}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest, "fetched"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="manifest names (default: all)")
    ap.add_argument("--env", action="store_true", help="print the tests' FNPU_* variables")
    a = ap.parse_args(argv)
    found = manifests()
    names = a.only or sorted(found)
    unknown = sorted(set(names) - set(found))
    if unknown:
        raise SystemExit(f"no bundled manifest named {', '.join(unknown)}")
    a.dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in names:
        paths[name], how = fetch(found[name], a.dir)
        print(f"{how} {name}: {paths[name]}", file=sys.stderr)
    if a.env:
        for name, var in ENV.items():
            if name in paths:
                print(f"export {var}={paths[name].resolve()}")


if __name__ == "__main__":
    main()
