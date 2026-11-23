"""Build an FJOB with an experimental entry, without opening a device."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from framenpu.compiler import BUILD_FLAGS, Toolchain  # noqa: E402
from framenpu.job import Job  # noqa: E402


def run(cmd):
    subprocess.run(cmd, check=True)


def build(entry, source, output, outdir):
    t = Toolchain.discover()
    sdk = ROOT / "src/framenpu_cdsp/sdk"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    objs = []
    for i, src in enumerate((entry, sdk / "hmx.S", source)):
        obj = outdir / f"{i:02d}-{Path(src).name}.o"
        run([t.clang, *BUILD_FLAGS, "-I", str(sdk / "include"), "-c", str(src), "-o", str(obj)])
        objs.append(str(obj))
    elf = outdir / "kernel.elf"
    binp = outdir / "kernel.bin"
    run([t.linker, "-T", str(sdk / "link.ld"), *objs, "-o", str(elf)])
    run([t.objcopy, "-O", "binary", "--only-section=.payload", str(elf), str(binp)])
    job = Job(binp.read_bytes(), Path(source).stem, requires_hmx=True, supports_direct=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    job.save(output)
    print(f"{job.name}: {len(job.code)} bytes -> {output}")
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("entry", "source", "output", "outdir"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    build(args.entry, args.source, args.output, args.outdir)
