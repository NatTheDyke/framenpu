"""The simulated cdsp plugin (tests/sim_plugin.py) behind framenpu.open(..., "device"): the whole
device path runs with no DSP, stays on the device (no CPU fallback) and returns the FP16 contract
bit for bit. This is what framehands' and the tracker's CI select with FRAMENPU_DEVICES.

Needs NumPy, onnxruntime (the session's CPU side) and the models in FNPU_MERCURY_MODEL and
FNPU_MERCURY_DETECTOR (tools/fetch_models.py --env)."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

KEYPOINT = os.environ.get("FNPU_MERCURY_MODEL")
DETECTOR = os.environ.get("FNPU_MERCURY_DETECTOR")
try:
    import numpy as np
    import onnxruntime  # noqa: F401
except ImportError:
    np = None


@unittest.skipUnless(
    np is not None and KEYPOINT and DETECTOR,
    "NumPy, onnxruntime, FNPU_MERCURY_MODEL and FNPU_MERCURY_DETECTOR required",
)
class SimPluginTests(unittest.TestCase):
    def test_framenpu_open_runs_the_device_path_on_the_simulator(self):
        import framenpu
        from framenpu import devices, mercury

        with patch.dict(os.environ, {devices.ENV: "cdsp=sim_plugin:SimPlugin"}):
            s = framenpu.open(
                [(KEYPOINT, "mercury-keypoint-jan18"), (DETECTOR, "mercury-detector-160")],
                "device",
            )
        try:
            self.assertEqual(s.name, "framenpu")  # opened on the device, not the CPU
            rng = np.random.default_rng(5)
            kp = [
                {
                    "inputImg": rng.uniform(-1, 2, (1, 1, 128, 128)).astype(np.float32),
                    "lastKeypoints": rng.uniform(0, 1, (1, 42)).astype(np.float32),
                    "useLastKeypoints": np.array([float(k % 2)], np.float32),
                }
                for k in range(4)  # the bundled image's most crops per request
            ]
            det = [{"inputImg": rng.uniform(-1, 2, (1, 1, 160, 160)).astype(np.float32)}]
            got_kp = s.infer("mercury-keypoint-jan18", kp)
            got_det = s.infer("mercury-detector-160", det)
            self.assertEqual(s.name, "framenpu")  # and stayed there
            graph = mercury.load_keypoint(KEYPOINT)
            for feed, out in zip(kp, got_kp):
                want = mercury.run(graph, feed, "fp16")
                for name in mercury.OUTPUTS:
                    np.testing.assert_array_equal(out[name], want[name])
            dgraph = mercury.load_detector(DETECTOR)
            want = mercury.run(dgraph, det[0], "fp16")
            for name in mercury.DET_OUTPUTS:
                np.testing.assert_array_equal(got_det[0][name], want[name])
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
