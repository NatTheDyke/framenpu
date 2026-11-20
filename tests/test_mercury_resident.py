"""Offline checks of the RS_RESIDENT plan (parameter images resident in the FeatureNN
blob region) against the legacy replay plan, on the lane simulator.

Needs NumPy and the Mercury model in FNPU_MERCURY_MODEL; no device.
"""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools")]

MODEL = os.environ.get("FNPU_MERCURY_MODEL")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@unittest.skipUnless(np is not None and MODEL, "NumPy and FNPU_MERCURY_MODEL required")
class ResidentPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from framenpu import mercury
        from framenpu import _mercury_layout as L

        cls.model = mercury.load_keypoint(MODEL, require_pin=False)
        kw = dict(prefix=True, batch=3, replay=True, pw_group=8, heads=True)
        cls.legacy = L.build_plan(cls.model, **kw)
        cls.resident = L.build_plan(cls.model, resident=True, **kw)

    def rows(self, mp):
        return np.frombuffer(mp.plan.table, "<u4").reshape(-1, 32)

    def test_layout(self):
        from framenpu import _mercury_common as C

        self.assertTrue(self.resident.resident)
        self.assertFalse(self.legacy.resident)
        self.assertEqual(len(self.rows(self.resident)), len(self.rows(self.legacy)))
        spans = []
        for r in self.rows(self.resident):
            kind = int(r[0])
            if kind in (3, 4, 7):
                self.assertEqual(int(r[21]), 0 if kind != 7 else int(r[21]))
                continue
            vpar, npar, scr = int(r[14]), int(r[10]), int(r[21])
            self.assertGreaterEqual(vpar, C.RES_LO)
            self.assertLessEqual(vpar + npar, C.RES_SPAN)
            self.assertTrue(0 < scr < C.SPAN)
            if kind in (1, 8):
                self.assertEqual((vpar % 16, scr % 16, npar % 16), (0, 0, 0))
            spans.append((vpar, vpar + npar))
        spans.sort()
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "images overlap")
        # the resident plan's borrowed-span peak can only shrink (images moved out)
        self.assertLessEqual(self.resident.vtcm_peak, self.legacy.vtcm_peak)

    def test_native_checks(self):
        from framenpu import _mercury_common as C

        for k in range(3):
            for r in self.rows(self.resident):
                row = C.replay_row(r, k, self.resident.plan.crop_stride)
                if row[0] == 7:
                    self.assertTrue(C.regrid_ok(row))
                    continue
                self.assertTrue(C.native_ok(row, s2_direct=0, pw_group=8, mheads=True, resident=True))
                if row[0] not in (3, 4):  # a resident op is refused by a legacy check
                    self.assertFalse(C.native_ok(row, s2_direct=0, pw_group=8, mheads=True))
            for r in self.rows(self.legacy):  # the resident image still accepts legacy rows
                row = C.replay_row(r, k, self.legacy.plan.crop_stride)
                if row[0] != 7:
                    self.assertTrue(C.native_ok(row, s2_direct=0, pw_group=8, mheads=True, resident=True))
        bad = self.rows(self.resident)[1].copy()
        bad[14] = C.SPAN  # the tag vector
        self.assertFalse(C.native_ok(bad, s2_direct=0, pw_group=8, mheads=True, resident=True))

    def run_plan(self, mp, case_seeds, *, resident=None, load=True, wipe_images=False):
        from framenpu import _mercury_common as C
        from framenpu import _mercury_layout as L
        import teacher_resident_layout as RL
        from test_mercury_layout import synthetic_case

        window = bytearray(C.WIN_HI - C.WIN_LO)

        def put(off, data):
            window[off - C.WIN_LO : off - C.WIN_LO + len(data)] = data

        def get(off, n):
            return bytes(window[off - C.WIN_LO : off - C.WIN_LO + n])

        for off, data in C.setup_writes(mp.plan).items():
            put(off, data)
        if wipe_images:  # prove the request reads no parameter image from the window
            for off, data in mp.plan.fixed.items():
                put(off, bytes(len(data)))
        for k, seed in enumerate(case_seeds):
            case, _ = synthetic_case(self.model, seed)
            L.stage_inputs(
                mp.crop(k), put, {L.STEM_COLS: L.stem_columns(case["image"]), L.HINT: case["hint"]}
            )
        RL.simulate(mp.plan, window, s2_phase=0, s2_direct=True, crops=len(case_seeds),
                    resident=resident, load=load)
        return [
            {b: L.read_tensor(mp.crop(k), get, b) for b in mp.plan.jobs[0].stores}
            for k in range(len(case_seeds))
        ]

    def assert_same(self, a, b):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(x.keys(), y.keys())
            for name in x:
                np.testing.assert_array_equal(x[name][0], y[name][0])
                self.assertEqual(x[name][1], y[name][1])

    def test_resident_matches_legacy_and_persists(self):
        from framenpu import _mercury_common as C

        vt = np.full((C.RES_SPAN, 32, 2), np.nan, np.float16)
        legacy = self.run_plan(self.legacy, [11, 12])
        first = self.run_plan(self.resident, [11, 12], resident=vt, load=True)
        self.assert_same(first, legacy)
        # second request: images not copied, and absent from the window
        again = self.run_plan(self.resident, [11, 12], resident=vt, load=False, wipe_images=True)
        self.assert_same(again, legacy)
        three = self.run_plan(self.resident, [13, 11, 12], resident=vt, load=False, wipe_images=True)
        self.assert_same(three[1:], legacy)

    def test_unloaded_region_is_not_used(self):
        from framenpu import _mercury_common as C

        vt = np.full((C.RES_SPAN, 32, 2), np.nan, np.float16)  # region never loaded
        out = self.run_plan(self.resident, [11], resident=vt, load=False)
        legacy = self.run_plan(self.legacy, [11])
        with self.assertRaises(AssertionError):
            self.assert_same(out, legacy)


if __name__ == "__main__":
    unittest.main()
