"""Offline checks of the teacher linked loader and operator image."""

import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools")]

try:
    from framenpu.compiler import Toolchain

    Toolchain.discover()
    HAVE_TOOLS = True
except Exception:  # noqa: BLE001
    HAVE_TOOLS = False


@unittest.skipUnless(HAVE_TOOLS, "Hexagon toolchain unavailable")
class TeacherLinkBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_teacher_link as BL
        import build_teacher_resident as BR

        cls.BL = BL
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.record = BL.build(cls.out / "link")
        cls.reference = BR.build(cls.out / "ref")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sizes_and_reservation(self):
        self.assertLessEqual(self.record["loader"]["bytes"], 4096)
        self.assertLessEqual(self.record["linked"]["bytes"], 0x3F80)
        image = (self.out / "link/teacher_link.bin").read_bytes()
        self.assertEqual(hashlib.sha256(image).hexdigest(), self.record["linked"]["sha256"])

    def test_reference_kernels_unchanged_by_link_build(self):
        for k in ("rs", "dw"):
            self.assertEqual(self.record["reference"][k]["sha256"], self.reference[k]["sha256"])

    def test_loader_checks_reject_a_wrong_seal_constant(self):
        text = (self.out / "link/loader/disassembly.txt").read_text()
        from check_resident_link import _symbols

        symbols = _symbols(self.out / "link/loader/kernel.elf")
        self.BL.verify_loader(text, symbols)
        with self.assertRaises(ValueError):
            self.BL.verify_loader(text.replace("0x544c4e4b", "0x544c4e4c"), symbols)

    def test_image_contains_no_slot_or_self_address(self):
        text = (self.out / "link/linked/disassembly.txt").read_text()
        self.assertNotIn("0x87a92000", text)
        self.assertNotIn("0xf0a92000", text)

    def test_admin_descriptor_layout(self):
        import hardware_teacher_link as HL

        data = struct.pack("<4I", HL.ADMIN_MAGIC, 7, HL.CODE, HL.CODE_BYTES)
        self.assertEqual(struct.unpack("<4I", data), (0x544C4144, 7, 0x1FF0000, 0x4000))
        self.assertEqual(HL.SEAL_AT, HL.CODE + HL.CODE_BYTES - 128)


if __name__ == "__main__":
    unittest.main()
