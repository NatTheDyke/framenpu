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
from framenpu.resident_landmark import JOB_KINDS, CODE, CODE_BYTES

TESTS = ROOT / "tests"
LINK_SOURCES = (
    [SDK / "hmx.S", SDK / "link.ld", SDK / "include/framenpu.h", SDK / "include/fnpu_readback.h"]
    + [
        TESTS / name
        for name in (
            "resident_graph_entry.S",
            "resident_link.ld",
            "resident_graph_link.c",
            "resident_graph.h",
            "resident_graph_gather.h",
            "resident_fp16.h",
            "landmark_common.h",
        )
    ]
    + [TESTS / f"resident_graph_{name}.c" for name in JOB_KINDS.values()]
)


def _pc_relative(tools, obj, folder, label):
    relocations = _run(["llvm-readobj", "--relocations", str(obj)])
    if any("R_HEX" in line and "PCREL" not in line for line in relocations.splitlines()):
        raise ValueError(f"{label} requires review of an absolute relocation")
    (folder / f"{label}-relocations.txt").write_text(relocations)


def build_linked(output, tools=None):
    """Build the slot loader and the one linked operator image; opens no device."""
    tools = tools or Toolchain.discover()
    output.mkdir(parents=True, exist_ok=True)
    flags = [*BUILD_FLAGS, "-fno-jump-tables", "-I", str(SDK / "include")]

    def compile_(folder, index, source, *defines):
        obj = folder / f"{index:02d}-{source.name}.o"
        _run([tools.clang, *flags, *defines, "-c", str(source), "-o", str(obj)])
        return obj

    def link(folder, script, objects):
        elf, binary = folder / "kernel.elf", folder / "kernel.bin"
        _run([tools.linker, "-T", str(script), *map(str, objects), "-o", str(elf)])
        _run([tools.objcopy, "-O", "binary", "--only-section=.payload", str(elf), str(binary)])
        return binary.read_bytes()

    folder = output / "loader"
    folder.mkdir(parents=True, exist_ok=True)
    objects = [
        compile_(folder, 0, TESTS / "resident_graph_entry.S", "-DRG_LINK_ENTRY"),
        compile_(folder, 1, SDK / "hmx.S"),
        compile_(folder, 2, TESTS / "resident_graph_admin.c", "-DRG_LOADER"),
    ]
    _pc_relative(tools, objects[2], folder, "loader")
    loader = Job(
        link(folder, SDK / "link.ld", objects), "resident_graph_loader_link_v1", True, True
    )
    loader.save(output / "resident_graph_loader.fjob")

    folder = output / "linked"
    folder.mkdir(parents=True, exist_ok=True)
    objects = [
        compile_(folder, 0, TESTS / "resident_graph_link.c"),
        compile_(folder, 1, SDK / "hmx.S"),
    ]
    _pc_relative(tools, objects[0], folder, "linked")
    image = link(folder, TESTS / "resident_link.ld", objects)
    if not 0 < len(image) <= CODE_BYTES or len(image) % 4:
        raise ValueError("linked resident image size differs from its reservation")
    (output / "resident_graph_linked.bin").write_bytes(image)
    record = {
        "format": "framenpu-resident-link-build",
        "version": 1,
        "code_offset": CODE,
        "code_reservation_bytes": CODE_BYTES,
        "loader": {"bytes": len(loader.code), "sha256": hashlib.sha256(loader.code).hexdigest()},
        "linked": {"bytes": len(image), "sha256": hashlib.sha256(image).hexdigest()},
        "flags": [*BUILD_FLAGS, "-fno-jump-tables"],
        "toolchain": tools.versions(),
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in LINK_SOURCES
        },
        "c_relocations_pc_relative": ["loader", "linked"],
    }
    (output / "link.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cached-entry", action="store_true")
    parser.add_argument("--linked", action="store_true")
    args = parser.parse_args()
    if args.linked:
        if args.cached_entry:
            parser.error("--linked builds its own loader entry")
        print(json.dumps(build_linked(args.output), indent=2))
        return
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
