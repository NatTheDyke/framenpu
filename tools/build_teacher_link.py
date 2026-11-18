"""Build the teacher linked loader and operator image; opens no device.

<output>/rs and <output>/dw are the ordinary cached-entry kernels from
tools/build_teacher_resident.py (the qualification reference). <output>/loader
is the 4 KiB slot loader (tests/resident_graph_entry.S with RG_LINK_ENTRY and
TL_LINK_ENTRY, plus tests/teacher_link_admin.c). <output>/linked is the
operator image for the reservation at 0x1ff0000: tests/teacher_link.c at
offset zero, then tests/teacher_resident_probe.c and tests/teacher_dw_probe.c
compiled as tl_rs and tl_dw. --stride2 enables the resident stride-2 extension
in tl_rs only; --dw-blocks 4 restores four-block coefficient sharing there.
--s2-even (Mercury, with --stride2 only) keeps the even stride-2 phase.
The two 4 KiB qualification kernels remain unchanged.

Checks: PC-relative C relocations only, no vector stack accesses, the loader's
vector table and forwarding stubs (as the cache entry), its dispatch constants
and that it never publishes or cleans the slot, the image entry at offset zero,
no indirect calls or jumps other than returns, no cache or exception-return
instructions, no embedded address of the image and no writable data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from framenpu import Job
from framenpu.compiler import BUILD_FLAGS, SDK, Toolchain, _run
import build_teacher_resident as BR
from check_resident_cache import verify_relocations, verify_vectors
from check_resident_link import _constants, _symbols, verify_image

TESTS = ROOT / "tests"
CODE, CODE_BYTES, SEAL_AT = 0x1FF0000, 0x4000, 0x1FF3F80
SEAL = 0x544C4E4B
UNCACHED, CACHED = 0xF0000000, 0x87000000
LOADER_NAME = "teacher_link_loader_v1"
SOURCES = (
    TESTS / "resident_graph_entry.S",
    TESTS / "teacher_link_admin.c",
    TESTS / "teacher_link.c",
    TESTS / "teacher_link.ld",
    TESTS / "teacher_resident_probe.c",
    TESTS / "teacher_dw_probe.c",
    TESTS / "teacher_heads.c",
    TESTS / "teacher_heads_check.h",
    TESTS / "teacher_heads_layout.py",
    TESTS / "teacher_stem_check.h",
    TESTS / "teacher_stem_impl.h",
    TESTS / "teacher_stem_vector.h",
    TESTS / "teacher_stem_layout.py",
    TESTS / "teacher_stem_profile.h",
    TESTS / "teacher_stem_profile.py",
    TESTS / "landmark_common.h",
    TESTS / "teacher_tail_layout.py",
    TESTS / "teacher_resident_layout.py",
    TESTS / "hmx_block_layout.py",
    SDK / "hmx.S",
    SDK / "link.ld",
    SDK / "include/framenpu.h",
    SDK / "include/fnpu_readback.h",
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_loader(text, symbols):
    verify_vectors(text, symbols)
    dispatch = text.split("<resident_dispatch>:", 1)[1].split(">:", 1)[0]
    if not {UNCACHED + SEAL_AT, SEAL, UNCACHED + CODE, CACHED + CODE} <= _constants(dispatch):
        raise ValueError("loader seal/alias constants differ")
    if (
        "<frame_job>" not in dispatch
        or dispatch.count("jumpr") != 1
        or "mux(p0,r2,r1)" not in dispatch
    ):
        raise ValueError("loader dispatch contract differs")
    if re.search(r"resident_(stage|publish|cache_cleanup)", text):
        raise ValueError("loader must not publish or clean the code slot")


def build(
    output,
    *,
    stride2=False,
    dw_blocks=2,
    heads=False,
    stem=False,
    stem_profile=False,
    stem_vector=False,
    stem_gray=False,
    s2_even=False,
):
    if s2_even and (not stride2 or heads or stem or dw_blocks != 2):
        raise ValueError("even-phase stride 2 is the Mercury build: stride2 only")
    if stem_gray and (not stem or stem_profile):
        raise ValueError("grayscale requires the stem build without profiling")
    if stem_vector and not stem:
        raise ValueError("vector stitching requires the stem linked build")
    if stem_profile and not stem:
        raise ValueError("stem profiling requires the stem linked build")
    if stem and not heads:
        raise ValueError("resident stem requires the heads linked build")
    if heads and not stride2:
        raise ValueError("resident heads require the stride-2 linked build")
    if dw_blocks not in (2, 4) or (dw_blocks == 4 and not stride2):
        raise ValueError("four-block depthwise requires the stride-2 linked build")
    tools = Toolchain.discover()
    record = {"format": "framenpu-teacher-link-build", "reference": BR.build(output)}

    def compile_(folder, index, source, *defines):
        obj = folder / f"{index:02d}-{source.name}.o"
        _run(
            [
                tools.clang,
                *BUILD_FLAGS,
                "-fno-jump-tables",
                *defines,
                "-I",
                str(SDK / "include"),
                "-I",
                str(TESTS),
                "-c",
                str(source),
                "-o",
                str(obj),
            ]
        )
        if source.suffix == ".c":
            rel = _run(["llvm-readobj", "--relocations", str(obj)])
            verify_relocations(rel)
            (folder / f"{index:02d}-relocations.txt").write_text(rel)
        return obj

    def link(folder, script, objects, check):
        elf, binary = folder / "kernel.elf", folder / "kernel.bin"
        _run([tools.linker, "-T", str(script), *map(str, objects), "-o", str(elf)])
        _run([tools.objcopy, "-O", "binary", "--only-section=.payload", str(elf), str(binary)])
        text = subprocess.check_output(["llvm-objdump", "-d", str(elf)], text=True)
        (folder / "disassembly.txt").write_text(text)
        if re.search(r"vmemu?\((r29|r30)", text):
            raise ValueError(f"{folder.name}: vector stack accesses")
        sections = subprocess.check_output(["llvm-size", "-A", str(elf)], text=True)
        if any(
            r.split()[0] in (".data", ".bss", ".sdata", ".sbss") and int(r.split()[1])
            for r in sections.splitlines()
            if r.startswith(".")
        ):
            raise ValueError(f"{folder.name}: writable data")
        check(text, _symbols(elf))
        return binary.read_bytes()

    folder = output / "loader"
    folder.mkdir(parents=True, exist_ok=True)
    objects = [
        compile_(folder, 0, TESTS / "resident_graph_entry.S", "-DRG_LINK_ENTRY", "-DTL_LINK_ENTRY"),
        compile_(folder, 1, SDK / "hmx.S"),
        compile_(folder, 2, TESTS / "teacher_link_admin.c"),
    ]
    code = link(folder, SDK / "link.ld", objects, verify_loader)
    if len(code) > 4096:
        raise ValueError("loader exceeds the 4 KiB slot")
    loader = Job(code, LOADER_NAME, True, True)
    loader.save(output / "teacher_link_loader.fjob")

    folder = output / "linked"
    folder.mkdir(parents=True, exist_ok=True)
    objects = [
        compile_(folder, 0, TESTS / "teacher_link.c"),
        compile_(
            folder,
            1,
            TESTS / "teacher_resident_probe.c",
            "-Dframe_job=tl_rs",
            *(["-DRS_STRIDE2"] if stride2 else []),
            *(["-DRS_DW_BLOCKS=4"] if dw_blocks == 4 else []),
            *(["-DRS_HEADS"] if heads else []),
            *(["-DRS_STEM"] if stem else []),
            *(["-DRS_STEM_PROFILE"] if stem_profile else []),
            *(["-DRS_STEM_VECTOR"] if stem_vector else []),
            *(["-DRS_STEM_GRAY"] if stem_gray else []),
            *(["-DRS_S2_EVEN"] if s2_even else []),
        ),
        compile_(folder, 2, TESTS / "teacher_dw_probe.c", "-Dframe_job=tl_dw"),
        compile_(folder, 3, SDK / "hmx.S"),
    ]
    if heads:
        objects.append(compile_(folder, 4, TESTS / "teacher_heads.c"))

    def check_image(text, symbols):
        if symbols.get("tl_entry") != UNCACHED + CODE:
            raise ValueError("image entry is not at offset zero")
        verify_image(text, symbols | {"rg_batch": UNCACHED + CODE}, SEAL_AT - CODE)

    image = link(folder, TESTS / "teacher_link.ld", objects, check_image)
    (output / "teacher_link.bin").write_bytes(image)
    record |= {
        "stride2": stride2,
        "dw_blocks": dw_blocks,
        "heads": heads,
        "stem": stem,
        "stem_profile": stem_profile,
        "stem_vector": stem_vector,
        "stem_gray": stem_gray,
        "s2_even": s2_even,
        "loader": {"bytes": len(loader.code), "sha256": _sha(loader.code), "name": LOADER_NAME},
        "linked": {"bytes": len(image), "sha256": _sha(image)},
        "reservation": {"code": f"{CODE:#x}", "bytes": CODE_BYTES, "seal_vector": f"{SEAL_AT:#x}"},
        "flags": [*BUILD_FLAGS, "-fno-jump-tables"],
        "toolchain": tools.versions(),
        "sources": {
            str(p.relative_to(ROOT if p.is_relative_to(ROOT) else SDK.parent)): _sha(p.read_bytes())
            for p in SOURCES
        },
    }
    (output / "link.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--stride2", action="store_true", help="enable merged resident stride-2 sequences"
    )
    p.add_argument("--dw-blocks", type=int, choices=(2, 4), default=2)
    p.add_argument("--heads", action="store_true")
    p.add_argument("--stem", action="store_true")
    p.add_argument("--stem-profile", action="store_true")
    p.add_argument("--stem-vector", action="store_true")
    p.add_argument("--stem-gray", action="store_true")
    p.add_argument("--s2-even", action="store_true", help="Mercury: even stride-2 phase")
    args = p.parse_args()
    r = build(
        args.output,
        stride2=args.stride2,
        dw_blocks=args.dw_blocks,
        heads=args.heads,
        stem=args.stem,
        stem_profile=args.stem_profile,
        stem_vector=args.stem_vector,
        stem_gray=args.stem_gray,
        s2_even=args.s2_even,
    )
    print(
        json.dumps(
            {k: r[k] for k in ("loader", "linked")}
            | {k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in r["reference"].items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
