"""Build the pinned resident graph modules without opening a device."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from framenpu.compiler import BUILD_FLAGS, SDK, Toolchain, _run, build_job
from framenpu import Job
from framenpu.resident_landmark import JOB_KINDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cached-entry", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {}
    tools = Toolchain.discover() if args.cached_entry else None
    for name in JOB_KINDS.values():
        source = ROOT / "tests" / f"resident_graph_{name}.c"
        output = args.output / f"resident_graph_{name}.fjob"
        if args.cached_entry:
            folder = args.output / name
            folder.mkdir(parents=True, exist_ok=True)
            objects = []
            for i, src in enumerate((ROOT / "tests/resident_graph_entry.S", SDK / "hmx.S", source)):
                obj = folder / f"{i:02d}-{src.name}.o"
                _run(
                    [
                        tools.clang,
                        *BUILD_FLAGS,
                        "-fno-jump-tables",
                        "-I",
                        str(SDK / "include"),
                        "-c",
                        str(src),
                        "-o",
                        str(obj),
                    ]
                )
                objects.append(str(obj))
                if i == 2:
                    relocations = _run(["llvm-readobj", "--relocations", str(obj)])
                    for line in relocations.splitlines():
                        if "R_HEX" in line and "PCREL" not in line:
                            raise ValueError(
                                "cached C module requires review of an absolute relocation"
                            )
                    (folder / "relocations.txt").write_text(relocations)
            elf, binary = folder / "kernel.elf", folder / "kernel.bin"
            _run([tools.linker, "-T", str(SDK / "link.ld"), *objects, "-o", str(elf)])
            _run([tools.objcopy, "-O", "binary", "--only-section=.payload", str(elf), str(binary)])
            job = Job(binary.read_bytes(), f"resident_graph_{name}_cache_v1", True, True)
            job.save(output)
        else:
            job = build_job(source, output, requires_hmx=True, build_dir=args.output / name)
        record[name] = {"bytes": len(job.code), "sha256": hashlib.sha256(job.code).hexdigest()}
    (args.output / "jobs.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.cached_entry:
        proof = {
            "format": "framenpu-resident-cache-build",
            "version": 1,
            "flags": [*BUILD_FLAGS, "-fno-jump-tables"],
            "toolchain": tools.versions(),
            "entry_sha256": hashlib.sha256(
                (ROOT / "tests/resident_graph_entry.S").read_bytes()
            ).hexdigest(),
            "sources": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    SDK / "entry.S",
                    SDK / "hmx.S",
                    SDK / "link.ld",
                    SDK / "include/framenpu.h",
                )
            },
            "module_relocations_verified": list(record),
            "modules": record,
        }
        (args.output / "cache-build.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
