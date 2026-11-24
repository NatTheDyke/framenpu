"""framenpu stands alone (docs/CONTRACTS.md): it imports no device backend, and the contracts
import only the standard library."""

import ast
import pathlib
import unittest

PKG = pathlib.Path(__file__).resolve().parents[1] / "src/framenpu"


def imports(path):
    """Top-level packages a file imports absolutely (any depth: lazy imports count)."""
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module.split(".")[0]


class BoundaryTests(unittest.TestCase):
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
