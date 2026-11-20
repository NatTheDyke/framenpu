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


class ReplayCompletionTests(unittest.TestCase):
    @staticmethod
    def result(count=3):
        words = [0] * 256
        words[:3] = [C.RS_MAGIC, M.IMAGE_CACHED + 4, 55 * count]
        if count > 1:
            words[48 : 48 + count] = [550 * (k + 1) for k in range(count)]
        words[96:151] = [(count - 1) * 550 + 10 * (k + 1) for k in range(55)]
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
                oracle.simulate(p, window, s2_phase=0, s2_direct=True, crops=count)
                self._transport.memory[C.WIN_LO : C.WIN_HI] = window
                words, workers = ReplayCompletionTests.result(count)
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
        self.assertFalse(any(dev._transport.memory))

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
