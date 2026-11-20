"""Packaged Mercury planner parity, session lifecycle and replay completion checks."""

import json
import os
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from framenpu import mercury
from framenpu import mercury_runtime as M
from framenpu import _mercury_common as C
from framenpu import _mercury_layout as L
from framenpu.errors import RecoveryRequired
from framenpu.resident_gemm import RECORD, SSR
from framenpu.resident_landmark import CONTROL

MODEL = os.environ.get("FNPU_MERCURY_MODEL")
JOBS = os.environ.get("FNPU_MERCURY_JOBS")
HEADS_JOBS = os.environ.get("FNPU_MERCURY_HEADS_JOBS")  # a --mheads build of the same image


class ReplayCompletionTests(unittest.TestCase):
    @staticmethod
    def result(count=3, n=55):
        words = [0] * 256
        words[:3] = [C.RS_MAGIC, M.IMAGE_CACHED + 4, n * count]
        if count > 1:
            words[48 : 48 + count] = [10 * n * (k + 1) for k in range(count)]
        words[96 : 96 + n] = [(count - 1) * 10 * n + 10 * (k + 1) for k in range(n)]
        workers = []
        for i in range(3):
            row = [0] * 16
            row[0], row[1], row[7], row[8] = 1, i, SSR[i], 0x87000000
            workers.append(tuple(row))
        # kernel_reference_ticks comes from worker[0]'s pcycle/clock metadata.
        return words, workers

    def test_real_device_evidence_and_corrupted_boundaries(self):
        for name in ("pilot", "device"):
            report = json.loads((ROOT / f"evidence/mercury-replay-{name}.json").read_text())
            self.assertTrue(report["all_checks_passed"])
        from types import SimpleNamespace

        for count in (1, 2, 3):
            words, workers = self.result(count)

            def parse(w, ws=workers):
                return M.replay_metadata(
                    SimpleNamespace(
                        output=struct.pack("<256I", *w), workers=ws, kernel_reference_ticks=2000
                    ),
                    count,
                    55,
                    6960,
                )

            self.assertEqual(parse(words)["crops"], count)
            for index in (0, 2, 3, 51, 64, 151):
                bad = words.copy()
                bad[index] += 1
                with self.subTest(count=count, word=index), self.assertRaises(RecoveryRequired):
                    parse(bad)
            if count > 1:
                for index, value in ((48, 0), (48 + count - 1, 1), (96, words[48 + count - 2])):
                    bad = words.copy()
                    bad[index] = value
                    with self.assertRaises(RecoveryRequired):
                        parse(bad)

    def test_residency_status(self):
        from types import SimpleNamespace

        words, workers = self.result(1)

        def parse(status, resident):
            w = list(words)
            w[12] = status
            return M.replay_metadata(
                SimpleNamespace(output=struct.pack("<256I", *w), workers=workers, kernel_reference_ticks=2000),
                1, 55, 6960, resident=resident,
            )

        self.assertTrue(parse(7, True)["params_copied"])
        self.assertFalse(parse(1, True)["params_copied"])
        self.assertIsNone(parse(0, False)["params_copied"])
        for status, resident in ((0, True), (3, True), (5, True), (6, True), (1, False), (7, False), (1, None)):
            with self.subTest(status=status, resident=resident), self.assertRaises(RecoveryRequired):
                parse(status, resident)

    def test_missing_clock_and_validator_errors_require_recovery(self):
        from types import SimpleNamespace

        words, workers = self.result()
        result = SimpleNamespace(
            output=struct.pack("<256I", *words), workers=workers, kernel_reference_ticks=None
        )
        with self.assertRaises(RecoveryRequired):
            M.replay_metadata(result, 3, 55, 6960)
        for error in (ValueError("invalid readback"), KeyboardInterrupt()):

            def fail(result):
                raise error

            with self.assertRaises(RecoveryRequired):
                M._completion_guard(fail)(result)


class _Power:
    def __init__(self):
        self.abandoned = False

    def abandon(self):
        self.abandoned = True


class _Lease:
    closed = False

    def close(self):
        self.closed = True


class _Device:
    """Run native plans in the independent lane simulator, model gates and seals."""

    claim_timeout = 1

    def __init__(self, dispatch):
        from types import SimpleNamespace

        self._transport = SimpleNamespace(
            memory=bytearray(0x2000000),
            poisoned=False,
            claim=self.claim,
            release_unsubmitted=self.release,
        )
        self.claimed = self.closed = self.fail = False
        self.submitted = []
        self.powers = []
        self.released = 0
        self.owner = None
        # RS_RESIDENT model: FeatureNN has loaded its blob (flag 1); VTCM persists
        struct.pack_into("<I", self._transport.memory, C.BLOB_FLAG, 1)
        self.vt = None
        self.tag = (0, 0, 0, 0)
        self.copied = []

    def claim(self, timeout):
        if self.claimed:
            raise AssertionError("overlapping gate")
        self.claimed = True

    def release(self):
        self.claimed = False

    def __enter__(self):
        return self

    def close(self):
        self.closed = True

    def _hold_power(self):
        p = _Power()
        self.powers.append(p)
        return p

    def _release_power(self, p):
        self.released += 1

    def _submit(
        self, job, data, *, opcode, params, output_size, _prepare, _completion_validator, **kwargs
    ):
        from types import SimpleNamespace
        import teacher_resident_layout as oracle

        self.claim(1)
        try:
            _prepare(self._transport)
            words, workers = ReplayCompletionTests.result()
            if opcode == M.ADMIN_OPCODE:
                after = struct.unpack_from("<I", data, 4)[0] == 7
                struct.pack_into("<I", self._transport.memory, M.SEAL_AT, M.SEAL if after else 0)
                out = struct.pack(
                    "<3I", M.ADMIN_MAGIC, M.SLOT_UNCACHED + 4, M.SEAL if after else 0
                ).ljust(256, b"\0")
            else:
                count = struct.unpack_from("<I", data, 12)[0] or 1
                self.submitted.append(count)
                p = self.owner.plan.plan
                window = self._transport.memory[C.WIN_LO : C.WIN_HI]
                status, kw = 0, {}
                if self.owner.resident_image:
                    import numpy as np

                    tag, mode = struct.unpack_from("<4I", data, 20), struct.unpack_from("<I", data, 36)[0]
                    if mode:
                        flag = struct.unpack_from("<I", self._transport.memory, C.BLOB_FLAG)[0]
                        load = bool(flag) or tag != self.tag
                        if load:
                            struct.pack_into("<I", self._transport.memory, C.BLOB_FLAG, 0)
                            self.tag = (0, 0, 0, 0)
                        if self.vt is None:
                            self.vt = np.full((oracle.RES_SPAN, 32, 2), np.nan, np.float16)
                        kw = dict(resident=self.vt, load=load)
                        status = 1 | 6 * load
                        self.copied.append(load)
                oracle.simulate(p, window, s2_phase=0, s2_direct=True, crops=count, **kw)
                if status & 4:
                    self.tag = tag
                self._transport.memory[C.WIN_LO : C.WIN_HI] = window
                words, workers = ReplayCompletionTests.result(count, len(p.table) // 128)
                words[12] = status
                out = struct.pack("<256I", *words)
                if self.fail:
                    workers[0] = tuple(list(workers[0][:2]) + [1] + list(workers[0][3:]))
            result = SimpleNamespace(output=out, workers=workers, kernel_reference_ticks=2000)
            _completion_validator(result)
            return result
        except RecoveryRequired:
            self._transport.poisoned = True
            raise
        finally:
            if not self._transport.poisoned:
                self.release()


@unittest.skipUnless(MODEL, "FNPU_MERCURY_MODEL required")
class PackagedPlannerTests(unittest.TestCase):
    def test_byte_identical_to_qualified_planner(self):
        import mercury_layout as oracle

        model = mercury.load_keypoint(MODEL)
        for stop in ("Conv_18", L.LAST):
            old = oracle.build_plan(
                model, prefix=True, stop_after=stop, batch=3, replay=True, pw_group=8
            )
            new = L.build_plan(
                model, prefix=True, stop_after=stop, batch=3, replay=True, pw_group=8
            )
            self.assertEqual(new.plan.table, old.plan.table)
            self.assertEqual(new.plan.fixed, old.plan.fixed)
            self.assertEqual(new.crops, old.crops)
            self.assertEqual(new.plan.end, old.plan.end)
            self.assertEqual(new.plan.crop_stride, old.plan.crop_stride)


@unittest.skipUnless(MODEL and JOBS, "FNPU_MERCURY_MODEL and FNPU_MERCURY_JOBS required")
class MercurySessionTests(unittest.TestCase):
    def session(self):
        from unittest.mock import patch

        lease = _Lease()
        with patch.object(M, "_lease", return_value=lease):
            session = M.MercurySession(
                MODEL, JOBS, _device_factory=_Device, _lease_factory=lambda: lease
            )
        return session, lease

    def test_retrained_weights_need_the_explicit_opt_in(self):
        import dataclasses
        from unittest.mock import patch
        from framenpu import onnx_graph
        from framenpu.errors import InvalidModel

        # same graph, other weights: only the file hash differs from the pinned Mercury model
        retrained = dataclasses.replace(onnx_graph.load(MODEL), sha256="0" * 64)
        lease = _Lease()
        with patch.object(mercury.onnx_graph, "load", return_value=retrained), \
                patch.object(M, "_lease", return_value=lease):
            with self.assertRaises(InvalidModel):
                M.MercurySession(
                    MODEL, JOBS, _device_factory=_Device, _lease_factory=lambda: lease
                )
            session = M.MercurySession(
                MODEL,
                JOBS,
                require_pin=False,
                _device_factory=_Device,
                _lease_factory=lambda: lease,
            )
        self.assertFalse(session.model.pinned)
        self.assertTrue(self.session()[0].model.pinned)

    @staticmethod
    def feeds(seed):
        import numpy as np

        rng = np.random.default_rng(seed)
        return {
            "inputImg": rng.uniform(-1, 1, (1, 1, 128, 128)).astype(np.float32),
            "lastKeypoints": rng.uniform(-1, 1, (1, 42)).astype(np.float32),
            "useLastKeypoints": np.array([seed % 2], np.float32),
        }

    def test_mixed_crops_idle_resume_and_full_restoration(self):
        import numpy as np

        session, lease = self.session()
        session.__enter__()
        dev = session.device
        dev.owner = session
        with session:
            for count in (1, 3, 2, 1):
                fs = [self.feeds(k + 31) for k in range(count)]
                got = session.infer(fs)
                for feed, out in zip(fs, got):
                    want = mercury.run(session.model, feed, "fp16")
                    for name in mercury.OUTPUTS:
                        np.testing.assert_array_equal(out[name], want[name])
            session.idle()
            self.assertEqual(dev.released, 1)
            session.infer([self.feeds(37)])
            self.assertEqual(len(dev.powers), 2)
        self.assertEqual(dev.submitted, [1, 3, 2, 1, 1])
        self.assertTrue(dev.closed and lease.closed)
        self.assertEqual(dev.released, 2)
        self.assert_restored(dev, session)

    @staticmethod
    def flag(dev):
        return struct.unpack_from("<I", dev._transport.memory, C.BLOB_FLAG)[0]

    def assert_restored(self, dev, session):
        """Everything the session wrote is restored; the blob flag stays cleared
        only when resident images were copied (FeatureNN then reloads its blob)."""
        self.assertEqual(self.flag(dev), 0 if any(dev.copied) else 1)
        mem = bytearray(dev._transport.memory)
        mem[C.BLOB_FLAG : C.BLOB_FLAG + 4] = bytes(4)
        self.assertFalse(any(mem))

    def test_resident_parameters_load_once_and_after_eviction(self):
        import numpy as np

        session, lease = self.session()
        if not session.resident_image:
            self.skipTest("FNPU_MERCURY_JOBS is not the resident image")
        self.assertTrue(session.resident and session.plan.resident)
        session.__enter__()
        dev = session.device
        dev.owner = session
        fs = [self.feeds(41), self.feeds(42)]
        want = [mercury.run(session.model, f, "fp16") for f in fs]

        def check(got, want=want):
            self.assertEqual(len(got), len(want))
            for w, out in zip(want, got):
                for name in mercury.OUTPUTS:
                    np.testing.assert_array_equal(out[name], w[name])

        with session:
            check(session.infer(fs))
            self.assertTrue(session.last_timing["params_copied"])
            self.assertEqual(self.flag(dev), 0)
            # the window's images are no longer read once resident
            for off, data in session.plan.plan.fixed.items():
                dev._transport.memory[off : off + len(data)] = bytes(len(data))
            check(session.infer(fs))
            check(session.infer(fs[1:]), want[1:])
            self.assertEqual(dev.copied, [True, False, False])
            # FeatureNN ran in between (flag 1): the next request copies again
            for off, data in session.plan.plan.fixed.items():
                dev._transport.memory[off : off + len(data)] = data
            struct.pack_into("<I", dev._transport.memory, C.BLOB_FLAG, 1)
            check(session.infer(fs))
            self.assertEqual(dev.copied, [True, False, False, True])
            self.assertEqual(self.flag(dev), 0)
        self.assert_restored(dev, session)

    def test_resident_image_runs_the_legacy_plan(self):
        import numpy as np
        from unittest.mock import patch

        lease = _Lease()
        with patch.object(M, "_lease", return_value=lease):
            session = M.MercurySession(
                MODEL, JOBS, resident=False, _device_factory=_Device, _lease_factory=lambda: lease
            )
        if not session.resident_image:
            self.skipTest("FNPU_MERCURY_JOBS is not the resident image")
        self.assertFalse(session.resident or session.plan.resident)
        session.__enter__()
        dev = session.device
        dev.owner = session
        with session:
            f = self.feeds(43)
            out = session.infer([f])[0]
            self.assertIsNone(session.last_timing["params_copied"])
            want = mercury.run(session.model, f, "fp16")
            for name in mercury.OUTPUTS:
                np.testing.assert_array_equal(out[name], want[name])
        self.assertEqual(dev.copied, [])
        self.assert_restored(dev, session)

    def test_uncertain_completion_retains_refusal_and_does_not_retry(self):
        session, lease = self.session()
        session.__enter__()
        dev = session.device
        dev.owner, dev.fail = session, True
        with self.assertRaises(RecoveryRequired):
            session.infer([self.feeds(31)])
        power = session.power
        session.close()
        self.assertTrue(power.abandoned)
        self.assertEqual(bytes(dev._transport.memory[CONTROL : CONTROL + len(RECORD)]), RECORD)
        self.assertEqual(dev.submitted, [1])
        self.assertEqual(dev.released, 0)
        self.assertTrue(lease.closed)
        with self.assertRaises(RuntimeError):
            session.infer([self.feeds(31)])


@unittest.skipUnless(MODEL and HEADS_JOBS, "FNPU_MERCURY_MODEL and FNPU_MERCURY_HEADS_JOBS required")
class HeadsSessionTests(unittest.TestCase):
    """The --mheads image: the heatmap head runs in the (simulated) kernel, and
    every output still equals the FP16 contract (the simulator is exact)."""

    def test_heatmap_head_on_the_dsp_matches_the_contract(self):
        import numpy as np
        from unittest.mock import patch

        lease = _Lease()
        with patch.object(M, "_lease", return_value=lease):
            session = M.MercurySession(
                MODEL, HEADS_JOBS, _device_factory=_Device, _lease_factory=lambda: lease
            )
        self.assertTrue(session.heads)
        rows = list(struct.iter_unpack("<32I", session.plan.plan.table))
        self.assertEqual(sum(r[0] == L.KIND_RG for r in rows), 2)
        self.assertEqual(sum(r[0] == L.KIND_P3 for r in rows), 2)
        session.__enter__()
        dev = session.device
        dev.owner = session
        with session:
            for count in (1, 3, 2):
                fs = [MercurySessionTests.feeds(k + 41) for k in range(count)]
                for feed, out in zip(fs, session.infer(fs)):
                    want = mercury.run(session.model, feed, "fp16")
                    for name in mercury.OUTPUTS:
                        np.testing.assert_array_equal(out[name], want[name])
        self.assertEqual(dev.submitted, [1, 3, 2])
        self.assert_restored(dev, session)

    def test_regrid_and_tap_checks_refuse_out_of_bounds_rows(self):
        lease = _Lease()
        from unittest.mock import patch

        with patch.object(M, "_lease", return_value=lease):
            session = M.MercurySession(
                MODEL, HEADS_JOBS, _device_factory=_Device, _lease_factory=lambda: lease
            )
        rows = [list(r) for r in struct.iter_unpack("<32I", session.plan.plan.table)]
        rg = next(r for r in rows if r[0] == L.KIND_RG and r[2] == 4)
        p3 = next(r for r in rows if r[0] == L.KIND_P3)
        self.assertTrue(C.regrid_ok(rg))
        for word, value in ((2, 10), (3, 40), (12, 4), (13, 1), (8, rg[7])):
            bad = list(rg)
            bad[word] = value
            self.assertFalse(C.regrid_ok(bad), word)
        bad = list(rg)
        bad[15] += 32 * rg[6]  # phase 0 destination past its tensor
        self.assertFalse(C.regrid_ok(bad))
        self.assertTrue(C.native_ok(p3, s2_direct=0, pw_group=8, mheads=True))
        self.assertFalse(C.native_ok(p3, s2_direct=0, pw_group=8))  # not an RS_MHEADS build
        for word, value in ((3, p3[3] + 1), (6, 0), (7, p3[7] + 8)):
            bad = list(p3)
            bad[word] = value
            self.assertFalse(C.native_ok(bad, s2_direct=0, pw_group=8, mheads=True), word)

