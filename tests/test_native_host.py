"""Host kernels (src/framenpu/host/fnpu_host.c) against NumPy.

The bundled aarch64 library is checked against its manifest everywhere. The kernels
themselves are tested through a local build of the same source for this machine
(needs clang and ld.lld). framenpu-cdsp's tests run them in whole requests.
"""

import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from framenpu import _native  # noqa: E402

TOOLS = all(shutil.which(t) for t in ("clang", "ld.lld", "llvm-nm"))
ARCH = platform.machine()


class BundledLibraryTests(unittest.TestCase):
    def test_bundled_library_matches_manifest_and_source(self):
        import check_repository

        check_repository.verify_host_native(ROOT / "src/framenpu")
        rec = json.loads((_native.DATA / "host_aarch64.json").read_text())
        self.assertEqual(rec["target"], "aarch64-linux-gnu")
        self.assertEqual(rec["abi"], _native.ABI)

    @unittest.skipUnless(TOOLS, "clang and ld.lld required")
    def test_bundled_library_rebuilds_byte_identical(self):
        import build_host_native as B

        for lib, data in (("host", _native.DATA),):
            rec = json.loads((data / f"{lib}_aarch64.json").read_text())
            if B._tool(["clang", "--version"]).splitlines()[0].strip() != rec["toolchain"]["clang"]:
                self.skipTest("a different clang built the bundled library")
            with tempfile.TemporaryDirectory() as d:
                got = B.build("aarch64", Path(d) / f"{lib}.so", lib)
            self.assertEqual(got["sha256"], rec["sha256"], lib)
            self.assertEqual(got["compiler_flags"], rec["compiler_flags"], lib)

    def test_host_refuses_a_library_that_differs_from_its_manifest(self):
        rec = json.loads((_native.DATA / "host_aarch64.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "host.so"
            bad.write_bytes((_native.DATA / "host_aarch64.so").read_bytes() + b"\0")
            with self.assertRaises(ValueError):
                _native.Host(bad, rec)

    def test_bundled_is_none_off_aarch64(self):
        if ARCH != "aarch64":
            self.assertIsNone(_native.bundled())


@unittest.skipUnless(TOOLS and ARCH in ("x86_64", "aarch64"), "clang and ld.lld required")
class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_host_native as B

        cls.tmp = tempfile.TemporaryDirectory()
        lib = Path(cls.tmp.name) / f"host_{ARCH}.so"
        cls.host = _native.Host(lib, B.build(ARCH, lib))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fp16_conversion_matches_numpy(self):
        import numpy as np

        rng = np.random.default_rng(1)
        special = np.array(
            [
                0x00000000,
                0x80000000,
                0x33000000,
                0x33000001,
                0x337FFFFF,
                0x33800000,
                0x387FC000,
                0x387FE000,
                0x387FF000,
                0x38800000,
                0x477FE000,
                0x477FEFFF,
                0x477FF000,
                0x477FF001,
                0x7F800000,
                0xFF800000,
                0x3F800000,
                0x3F801000,
                0x3F802000,
                0x3F803000,
            ],
            np.uint32,
        )
        u = np.concatenate(
            [special, rng.integers(0, 1 << 32, 1 << 20, dtype=np.uint64).astype(np.uint32)]
        )
        f = u.view(np.float32)
        f = f[~np.isnan(f)]
        out = np.empty(f.size, np.uint16)
        with np.errstate(all="ignore"):
            ref = f.astype(np.float16)
        bad = self.host.to_f16(f, out)
        np.testing.assert_array_equal(out, ref.view(np.uint16))
        self.assertEqual(bad, int(np.count_nonzero(~np.isfinite(ref))))
        h = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
        back = np.empty(h.size, np.float32)
        self.host.from_f16(h, back)
        ref = h.view(np.float16).astype(np.float32)
        same = (back.view(np.uint32) == ref.view(np.uint32)) | (np.isnan(back) & np.isnan(ref))
        self.assertTrue(same.all())

    def test_wrappers_refuse_wrong_types_and_sizes(self):
        import numpy as np

        out = np.empty(4, np.uint16)
        with self.assertRaises(TypeError):
            self.host.to_f16(np.zeros(4, np.float64), out)
        with self.assertRaises(ValueError):
            self.host.to_f16(np.zeros(5, np.float32), out)
        with self.assertRaises(TypeError):
            self.host.to_f16(np.zeros(8, np.float32)[::2], out)


if __name__ == "__main__":
    unittest.main()
