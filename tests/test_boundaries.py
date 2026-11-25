"""framenpu stands alone (docs/CONTRACTS.md): it imports no device backend, and the contracts
import only the standard library."""

import ast
import pathlib
import re
import sys
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src/framenpu"
# imported third-party packages that only some backends need: the extra that declares each
EXTRAS = {"onnxruntime": "cpu"}


def imports(path):
    """Top-level packages a file imports absolutely (any depth: lazy imports count)."""
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module.split(".")[0]


def declared():
    """(required, by extra) distribution names of pyproject.toml."""

    def names(requirements):
        return {re.match(r"[\w.-]+", r)[0].lower() for r in requirements}

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project.get("optional-dependencies", {})
    return names(project.get("dependencies", [])), {k: names(v) for k, v in extras.items()}


class BoundaryTests(unittest.TestCase):
    def test_every_third_party_import_is_declared(self):
        required, extras = declared()
        for path in PKG.rglob("*.py"):
            for name in set(imports(path)) - set(sys.stdlib_module_names) - {"framenpu"}:
                where = extras[EXTRAS[name]] if name in EXTRAS else required
                self.assertIn(name, where, f"{path.name} imports {name}, undeclared")

    def test_no_module_imports_a_backend(self):
        for path in PKG.rglob("*.py"):
            self.assertNotIn("framenpu_cdsp", set(imports(path)), path)

    def test_contracts_and_errors_import_only_the_standard_library(self):
        for name in ("contracts", "errors"):
            path = PKG / f"{name}.py"
            for node in ast.walk(ast.parse(path.read_text())):
                self.assertFalse(isinstance(node, ast.ImportFrom) and node.level, name)
            self.assertFalse({"numpy", "framenpu"} & set(imports(path)), name)


if __name__ == "__main__":
    unittest.main()
