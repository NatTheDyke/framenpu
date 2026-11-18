"""Offline checks of the Mercury backbone plan (tests/mercury_layout.py) and of the
device harness flow (tests/hardware_mercury.py) against the lane simulator.

Needs NumPy and the pinned model in FNPU_MERCURY_MODEL; no device, no corpus.
"""

import os
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools")]

MODEL = os.environ.get("FNPU_MERCURY_MODEL")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def synthetic_case(model, seed, hinted=True):
    from framenpu import mercury
    import mercury_layout as ML

    rng = np.random.default_rng(seed)
    img = rng.uniform(-1, 1, (1, 1, 128, 128)).astype(np.float32)
    feeds = {
        "inputImg": img,
        "lastKeypoints": rng.uniform(0, 1, (1, 42)).astype(np.float32),
        "useLastKeypoints": np.full(1, float(hinted), np.float32),
    }
    fp16, t = mercury.run(model, feeds, "fp16", keep=True, host_prefix=ML.HOST_PREFIX)
    fp32 = mercury.run(model, feeds, "fp32")
    case = {
        "input": t[ML.INPUT].astype(np.float16),
        "hint": t[ML.HINT].astype(np.float16),
        "expected": t[ML.OUTPUT].astype(np.float16),
    }
    case |= {f"fp32/{k}": v for k, v in fp32.items()}
    case |= {f"fp16/{k}": v for k, v in fp16.items()}
    full, tf = mercury.run(model, feeds, "fp16", keep=True)
    case["image"] = tf["inputImg"].reshape(128, 128).astype(np.float16)
    case["expectedfull"] = tf[ML.OUTPUT].astype(np.float16)
    case |= {f"fp16full/{k}": v for k, v in full.items()}
    return case, t


class SimRunner:
    """MercuryRunner's interface over a simulated window (no device)."""

    def __init__(self):
        import teacher_tail_layout as T

        self.T = T
        self.window = bytearray(T.WIN_HI - T.WIN_LO)
        self.sealed = True
        self.events = []

    def put(self, off, data):
        self.window[off - self.T.WIN_LO : off - self.T.WIN_LO + len(data)] = data

    def get(self, off, n):
        return bytes(self.window[off - self.T.WIN_LO : off - self.T.WIN_LO + n])

    def install(self, mplan):
        import teacher_resident_layout as RL

        for off, data in RL.setup_writes(mplan.plan).items():
            self.put(off, data)

    def request(self, mplan, job, fault=False, reject=False, refused=False):
        import dataclasses
        import mercury_layout as ML
        import teacher_resident_layout as RL

        one = dataclasses.replace(mplan.plan, jobs=[job])
        if fault or refused:
            self.events.append("fault" if fault else "refused")
            return None
        if reject:
            try:
                RL.simulate(one, self.window, s2_phase=ML.S2_PHASE)
            except ValueError:
                self.events.append("reject")
                return None
            raise AssertionError("descriptor was not rejected")
        RL.simulate(one, self.window, s2_phase=ML.S2_PHASE)
        n = job.end - job.first
        return {"kind": "rs", "wall_ms": 1.0, "wrapper_ms": 1.0, "body_ms": 1.0, "op_us": [1.0] * n}

    def admin(self, command, reject=False):
        self.events.append(("admin", command, reject))

    tensor = None  # bound below
    raw = None


def _bind():
    import hardware_mercury as HM

    SimRunner.tensor = HM.MercuryRunner.tensor
    SimRunner.raw = HM.MercuryRunner.raw


@unittest.skipUnless(np is not None and MODEL, "needs NumPy and FNPU_MERCURY_MODEL")
class MercuryLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from framenpu import mercury

        cls.model = mercury.load_keypoint(MODEL)

    def test_merged_plan_fits_and_is_native(self):
        import mercury_layout as ML
        import teacher_resident_layout as RL
        import teacher_tail_layout as T

        mp = ML.build_plan(self.model)
        self.assertEqual(len(mp.ops), 36)
        self.assertEqual(mp.descriptors, 39)  # load, 36 ops, hint load, store
        self.assertLessEqual(mp.vtcm_peak, RL.SPAN)
        self.assertLessEqual(mp.plan.end, T.WIN_HI)
        self.assertEqual(mp.external, [ML.INPUT])
        self.assertTrue(mp.needs_hint)
        # hint plane sits directly after the 32 feature pairs in the window
        g = mp.grid(ML.FEATURES)
        self.assertEqual(mp.where[ML.HINT] - mp.where[ML.CONCAT], 128 * 32 * g.nbin)
        tab = np.frombuffer(mp.plan.table, "<u4").reshape(-1, 32)
        self.assertTrue(all(RL.native_ok(r) for r in tab))
        kinds = [int(r[0]) for r in tab]
        self.assertEqual(kinds.count(RL.KIND["s2"]), 2)
        self.assertEqual((kinds[0], kinds[-1]), (RL.KIND["load"], RL.KIND["store"]))

    def test_merged_simulation_bit_exact(self):
        import mercury_layout as ML

        mp = ML.build_plan(self.model)
        for seed, hinted in ((1, True), (2, False)):
            case, t = synthetic_case(self.model, seed, hinted)
            out, stray = ML.simulate(mp, t)[ML.OUTPUT]
            self.assertEqual(stray, 0)
            np.testing.assert_array_equal(out, t[ML.OUTPUT])

    def test_even_phase_matters(self):
        import mercury_layout as ML
        import teacher_resident_layout as RL
        import teacher_tail_layout as T

        mp = ML.build_plan(self.model, stop_after="Conv_33")
        _, t = synthetic_case(self.model, 3)
        window = bytearray(T.WIN_HI - T.WIN_LO)
        for off, data in RL.setup_writes(mp.plan).items():
            window[off - T.WIN_LO : off - T.WIN_LO + len(data)] = data
        ML.stage_inputs(
            mp, lambda o, d: window.__setitem__(slice(o - T.WIN_LO, o - T.WIN_LO + len(d)), d), t
        )
        RL.simulate(mp.plan, window, s2_phase=1)  # teacher phase: wrong for Mercury
        off = mp.where[mp.final] - T.WIN_LO
        g = mp.grid(mp.final)
        odd, _ = g.from_bytes(bytes(window[off : off + g.tensor_bytes(192)]), 192)
        self.assertFalse(np.array_equal(odd, t[mp.final]))

    def test_qualify_segments_cover_range(self):
        import mercury_layout as ML
        import teacher_tail_layout as T

        segs = ML.qualify_segments(self.model)
        names = [op.name for s in segs for op in s.ops]
        self.assertEqual(names, [op.name for op in ML.device_ops(self.model)])
        self.assertTrue(all(s.plan.end <= T.WIN_HI and not s.needs_hint for s in segs))
        self.assertEqual(segs[0].ops[-1].output, ML.FEATURES)
        self.assertEqual(segs[1].external, [ML.CONCAT])

    def test_prefix_plan_fits_and_is_native(self):
        import mercury_layout as ML
        import teacher_resident_layout as RL
        import teacher_tail_layout as T

        mp = ML.build_plan(self.model, prefix=True)
        self.assertEqual(len(mp.ops), 52)  # 6 prefix ops, Conv_10/14 as 6 groups each
        self.assertEqual(mp.descriptors, 55)
        self.assertLessEqual(mp.vtcm_peak, RL.SPAN)
        self.assertLessEqual(mp.plan.end, T.WIN_HI)
        self.assertEqual((mp.external, mp.host_parts), ([ML.STEM_COLS], [ML.HINT]))
        self.assertNotIn(ML.EXPAND, mp.shape.keys() & {i for op in mp.ops for i in op.inputs})
        tab = np.frombuffer(mp.plan.table, "<u4").reshape(-1, 32)
        self.assertTrue(all(RL.native_ok(r) for r in tab))
        self.assertEqual([int(r[0]) for r in tab].count(RL.KIND["s2"]), 8)

    def test_prefix_simulation_bit_exact(self):
        """Conv_0 .. Conv_139 in one request against the full FP16 contract."""
        import mercury_layout as ML

        mp = ML.build_plan(self.model, prefix=True)
        for seed, hinted in ((5, True), (6, False)):
            case, _ = synthetic_case(self.model, seed, hinted)
            out, stray = ML.simulate(
                mp, {ML.STEM_COLS: ML.stem_columns(case["image"]), ML.HINT: case["hint"]}
            )[ML.OUTPUT]
            self.assertEqual(stray, 0)
            np.testing.assert_array_equal(out, case["expectedfull"].astype(np.float64))

    def test_stem_columns_match_dense_stem(self):
        from framenpu import mercury
        import mercury_layout as ML

        case, _ = synthetic_case(self.model, 7)
        stem = self.model.op("Conv_0")
        rewritten = ML.device_ops(self.model, "Conv_0", prefix=True)[0]
        cols = ML.stem_columns(case["image"]).astype(np.float32)
        img = case["image"].astype(np.float32)[None]
        for numerics in ("fp32", "fp16"):
            np.testing.assert_array_equal(
                mercury._conv(rewritten, cols, None, numerics),
                mercury._conv(stem, img, None, numerics),
            )

    def test_prefix_qualify_segments(self):
        import mercury_layout as ML

        segs = ML.qualify_segments(self.model, prefix=True)
        names = [op.name for s in segs for op in s.ops]
        self.assertEqual(names, [op.name for op in ML.device_ops(self.model, prefix=True)])
        self.assertEqual(segs[0].ops[-1].name, "Conv_14.g5")
        self.assertEqual(segs[1].external, [ML.DOWN])

    def test_prefix_harness_flow_on_simulator(self):
        import hardware_mercury as HM
        import mercury_layout as ML

        _bind()
        case, _ = synthetic_case(self.model, 8)
        runner = SimRunner()
        manifest = {"cases": [{"name": "s", "split": "synthetic", "hinted": True, "sha256": ""}]}
        orig = HM.load_case
        HM.load_case = lambda cases, entry: case
        try:
            rows, plans = HM.run(runner, self.model, None, manifest, 2, ML.LAST, prefix=True)
        finally:
            HM.load_case = orig
        row = rows["case00"]
        self.assertTrue(row["qualify"]["matches_host_emulator"])
        self.assertLess(row["qualify"]["max_bound_ratio"], 1)
        self.assertEqual(row["qualify"]["requests"], 52)
        self.assertTrue(row["outputs_equal_fp16_contract"])
        self.assertEqual(row["controls"]["stride2_descriptor_rejections"], 48)
        self.assertEqual(len(plans["qualify_segments"]), 4)

    def test_direct_stride2_model_matches(self):
        """RS_S2_DIRECT data path (word-level model) equals stride-1 + decimation."""
        import mercury_layout as ML

        for prefix in (False, True):
            mp = ML.build_plan(self.model, prefix=prefix)
            case, t = synthetic_case(self.model, 12 + prefix)
            ins = (
                {ML.STEM_COLS: ML.stem_columns(case["image"]), ML.HINT: case["hint"]}
                if prefix
                else t
            )
            a, _ = ML.simulate(mp, ins)[ML.OUTPUT]
            b, stray = ML.simulate(mp, ins, s2_direct=True)[ML.OUTPUT]
            self.assertEqual(stray, 0)
            np.testing.assert_array_equal(a, b)

    def test_batch_plan_crops_independent(self):
        """Two crops in one request: shared parameter images, own slots, and each
        crop bit-identical to its batch-1 result (different inputs, so crop 1
        cannot pass by reading crop 0's leftover VTCM)."""
        import mercury_layout as ML
        import teacher_resident_layout as RL

        one = ML.build_plan(self.model, prefix=True)
        two = ML.build_plan(self.model, prefix=True, batch=2)
        self.assertEqual(two.descriptors, 2 * one.descriptors)
        self.assertEqual(two.vtcm_peak, one.vtcm_peak)
        self.assertEqual(two.plan.fixed, one.plan.fixed)
        self.assertEqual(two.crop(0).where, one.where)
        a, b = two.crops
        self.assertFalse(set(a.values()) & set(b.values()))
        for n in range(two.descriptors):
            self.assertTrue(
                RL.native_ok(struct.unpack_from("<32I", two.plan.table, 128 * n), s2_direct=0)
            )
        ins = []
        for seed in (21, 22):
            case, _ = synthetic_case(self.model, seed, hinted=seed % 2 == 0)
            ins.append({ML.STEM_COLS: ML.stem_columns(case["image"]), ML.HINT: case["hint"]})
        for direct in (False, True):
            got = ML.simulate(two, ins, s2_direct=direct)
            for k in range(2):
                want, _ = ML.simulate(one, ins[k], s2_direct=direct)[ML.OUTPUT]
                out, stray = got[k][ML.OUTPUT]
                self.assertEqual(stray, 0)
                np.testing.assert_array_equal(out, want)
            self.assertFalse(np.array_equal(got[0][ML.OUTPUT][0], got[1][ML.OUTPUT][0]))
        with self.assertRaises(ValueError):
            ML.build_plan(self.model, prefix=True, batch=3)  # 165 descriptors
        with self.assertRaises(ValueError):
            ML.build_plan(self.model, "qualify", prefix=True, batch=2)

    def test_batch_harness_flow_on_simulator(self):
        import hardware_mercury as HM
        import mercury_layout as ML

        _bind()
        cases = [synthetic_case(self.model, s)[0] for s in (31, 32)]
        runner = SimRunner()
        manifest = {
            "cases": [
                {"name": f"s{i}", "split": "synthetic", "hinted": True, "sha256": ""}
                for i in range(2)
            ]
        }
        orig = HM.load_case
        HM.load_case = lambda c, entry: cases[int(entry["name"][1:])]
        try:
            rows, plans = HM.run(runner, self.model, None, manifest, 2, ML.LAST, True, None, 2)
        finally:
            HM.load_case = orig
        self.assertEqual(plans["merged"]["batch"], 2)
        self.assertEqual(rows["case01"]["batch_cases"], [1, 0])
        self.assertTrue(all(r["outputs_equal_fp16_contract"] for r in rows.values()))
        s = HM.summarize(rows, len(ML.build_plan(self.model, prefix=True).ops), 2)
        self.assertEqual(s["batch"], 2)

    def test_direct_stride2_spill_check(self):
        """The direct build rejects an output plane too short for its zero-lane spill."""
        import mercury_layout as ML
        import teacher_resident_layout as RL

        mp = ML.build_plan(self.model, prefix=True)
        tab = np.frombuffer(mp.plan.table, "<u4").reshape(-1, 32).copy()
        row = next(r for r in tab if r[0] == RL.KIND["s2"] and r[19] == 8)  # 16 -> 8
        oblk, oh, ows, ow = int(row[16]), int(row[18]), int(row[17]), int(row[19])
        need = 32 * oblk + 2 + (oh - 1) * ows + 32
        row[15] = -(-(32 * oblk + (oh - 1) * ows + 2 + ow) // 32)  # enough for stride-1 only
        self.assertTrue(RL.native_ok(row))
        self.assertLess(32 * int(row[15]), need)
        self.assertFalse(RL.native_ok(row, s2_direct=0))

    def test_harness_flow_on_simulator(self):
        import hardware_mercury as HM
        import mercury_layout as ML

        _bind()
        case, t = synthetic_case(self.model, 4)
        runner = SimRunner()
        manifest = {"cases": [{"name": "s", "split": "synthetic", "hinted": True, "sha256": ""}]}
        orig = HM.load_case
        HM.load_case = lambda cases, entry: case
        try:
            rows, plans = HM.run(runner, self.model, None, manifest, 3, ML.LAST)
        finally:
            HM.load_case = orig
        row = rows["case00"]
        self.assertTrue(row["qualify"]["matches_host_emulator"])
        self.assertLess(row["qualify"]["max_bound_ratio"], 1)
        self.assertTrue(row["outputs_equal_fp16_contract"])
        self.assertEqual(row["vs_fp16_contract"]["worst_joint_px"], 0)
        c = row["controls"]
        self.assertEqual(c["stride2_descriptor_rejections"], 12)
        self.assertEqual(runner.events.count("reject"), 13)
        self.assertEqual(runner.events.count("fault"), 1)
        s = HM.summarize(rows, 36)
        self.assertEqual(s["outputs_equal_fp16_contract"], 1)
        self.assertEqual(len(plans["qualify_segments"]), 2)

    def test_pilot_plan(self):
        import mercury_layout as ML

        mp = ML.build_plan(self.model, stop_after="Conv_33")
        self.assertEqual(mp.ops[-1].name, "Conv_33")
        self.assertEqual(len(mp.ops), 5)
        self.assertFalse(mp.needs_hint)
        with self.assertRaises(ValueError):
            ML.build_plan(self.model, stop_after="Gemm_58")

    def test_depthwise_bound_covers_qf16_flush(self):
        """The resident depthwise accumulates in qf16, which drops sub-2^-14 products.
        Reconstructs the observed device failure: a small positive bias plus one product
        just under 2^-14, which the device returns as the bias alone."""
        import mercury_layout as ML

        op = next(o for o in ML.device_ops(self.model, prefix=True) if o.name == "Conv_14.g1")
        w = np.asarray(op.params["weight"][:, 0], np.float16).astype(np.float64)
        b = op.params["bias"]
        c = int(np.argmin(np.where(b > 0, b, np.inf)))
        self.assertLess(b[c], 0.01)
        x = np.zeros((32, 64, 64))
        x[c, 20, 20] = 0.9 * 2.0**-14 / abs(w[c, 1, 1])  # centre tap of output (10, 10)
        exact = ML.exact(op, x)
        device = float(np.float16(b[c]))  # the product is flushed
        allow = ML.bound(op, x)
        self.assertGreater(abs(device - exact[c, 10, 10]), 3 * 2.0**-10 * b[c] + 2.0**-20)
        self.assertLessEqual(abs(device - exact[c, 10, 10]), allow[c, 10, 10])

    def test_descriptor_rejection_word(self):
        """The pw parameter-source word the harness corrupts is word 9 (byte 36)."""
        import mercury_layout as ML
        import teacher_resident_layout as RL
        import teacher_tail_layout as T

        mp = ML.build_plan(self.model)
        table = bytearray(mp.plan.table)
        k = next(
            n for n in range(mp.descriptors) if struct.unpack_from("<I", table, 128 * n)[0] == 1
        )
        struct.pack_into("<I", table, 128 * k + 36, T.WIN_HI)
        self.assertFalse(RL.native_ok(np.frombuffer(bytes(table[128 * k : 128 * k + 128]), "<u4")))


if __name__ == "__main__":
    unittest.main()
