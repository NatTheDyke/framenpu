"""Import direction between the parts of the framenpu package (docs/CONTRACTS.md).

Every module is assigned to the library it moves to in the split:

- api:      framenpu proper (models, sessions, CLI); may import api modules and contracts/errors
- device:   framenpu-cdsp (transport, power, residency, the kernel ABI planner); may import only
            device modules
- mixed:    modules phase 2 splits between api and device; may import api and device
- research: earlier experiments kept out of the libraries; may import anything
- contracts: standard library only

KNOWN is today's coupling across those lines. A new crossing fails the test; so does an entry
that no longer occurs, so the list only shrinks as the split proceeds.
"""

import ast
import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
PKG, CDSP = SRC / "framenpu", SRC / "framenpu_cdsp"
ALIAS = "this name stays an alias during the library split"

LAYERS = {
    "api": {"__init__", "__main__", "cli", "mercury", "model", "onnx_graph"},
    "device": {
        "_mercury_common",
        "_mercury_layout",
        "axi",
        "compiler",
        "device",
        "dispatch",
        "errors",
        "job",
        "mercury_device",
        "power",
        "profile",
        "provenance",
        "resident",
        "transport",
    },
    "mixed": {"_mercury_host", "_native", "mercury_runtime"},
    "research": {"convolution", "landmark", "ncnn", "resident_gemm", "resident_landmark"},
    "contracts": {"contracts"},
}
ALLOWED = {
    "api": {"api", "contracts"},
    "device": {"device", "contracts"},
    "mixed": {"api", "device", "mixed", "contracts"},
    "research": {"api", "device", "mixed", "research", "contracts"},
    "contracts": set(),
}
SHARED = {"errors"}  # owned by the device backend, re-exported by framenpu: any layer may import

KNOWN = {
    # the package root still exposes the device and early compiler API
    ("__init__", "device"),
    ("__init__", "job"),
    # the early CLI drives the device directly
    ("cli", "compiler"),
    ("cli", "ncnn"),
    ("cli", "resident"),
    # the early Dense/ReLU model packs matrices with device helpers
    ("model", "device"),
    ("model", "profile"),
    # power's read-only verify path uses the native host library
    ("power", "_native"),
}


def layer_of(module):
    for name, members in LAYERS.items():
        if module in members:
            return name
    raise KeyError(module)


def sources():
    """Module name -> file, for both packages; framenpu's alias files are not sources."""
    out = {}
    for pkg in (PKG, CDSP):
        for p in sorted(pkg.glob("*.py")):
            if pkg == CDSP and p.stem == "__init__":
                continue
            if pkg == PKG and ALIAS in p.read_text():
                continue
            if p.stem in out:
                raise AssertionError(f"{p.stem} is a source in both packages")
            out[p.stem] = p
    return out


def imports(path):
    """Package-internal modules a file imports (any depth: lazy imports count)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                found.add(node.module.split(".")[0])
            elif node.level == 1:
                found.update(a.name for a in node.names)
            elif (
                node.level == 0
                and node.module
                and node.module.split(".")[0] in ("framenpu", "framenpu_cdsp")
            ):
                parts = node.module.split(".")
                found.update([parts[1]] if len(parts) > 1 else [a.name for a in node.names])
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] in ("framenpu", "framenpu_cdsp") and len(parts) > 1:
                    found.add(parts[1])
    return {m for m in found if m in sources()}


def crossings():
    out = set()
    for src, path in sources().items():
        sl = layer_of(src)
        for dst in imports(path) - {src}:
            dl = layer_of(dst)
            if sl == "contracts" or (dst not in SHARED and dl != sl and dl not in ALLOWED[sl]):
                out.add((src, dst))
    return out


class BoundaryTests(unittest.TestCase):
    def test_every_module_has_a_layer(self):
        modules = set(sources())
        assigned = set().union(*LAYERS.values())
        self.assertEqual(modules - assigned, set(), "assign new modules to a layer")
        self.assertEqual(assigned - modules, set(), "drop removed modules from LAYERS")

    def test_device_modules_live_in_framenpu_cdsp(self):
        where = {m: p.parent.name for m, p in sources().items()}
        for m, pkg in where.items():
            self.assertEqual(pkg, "framenpu_cdsp" if layer_of(m) == "device" else "framenpu", m)

    def test_old_names_are_aliases_of_the_same_module(self):
        import importlib

        for m in sorted(m for m in LAYERS["device"] if (PKG / f"{m}.py").exists()):
            self.assertIs(
                importlib.import_module(f"framenpu.{m}"),
                importlib.import_module(f"framenpu_cdsp.{m}"),
            )
        from framenpu import RecoveryRequired
        from framenpu_cdsp.errors import RecoveryRequired as moved

        self.assertIs(RecoveryRequired, moved)

    def test_no_new_crossings(self):
        self.assertEqual(crossings() - KNOWN, set())

    def test_known_crossings_still_occur(self):
        self.assertEqual(KNOWN - crossings(), set(), "a crossing was removed: drop it from KNOWN")

    def test_contracts_import_only_the_standard_library(self):
        tree = ast.parse((PKG / "contracts.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "contracts imports nothing from the package")
                self.assertNotIn(node.module.split(".")[0], {"numpy", "framenpu"})
            elif isinstance(node, ast.Import):
                for a in node.names:
                    self.assertNotIn(a.name.split(".")[0], {"numpy", "framenpu"})

    def test_device_modules_never_reach_up(self):
        up = {(s, d) for s, d in crossings() if layer_of(s) == "device"}
        self.assertEqual(up, {("power", "_native")})


if __name__ == "__main__":
    unittest.main()
