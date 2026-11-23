"""The inference backends (framenpu.backends): CPU, in-process FrameNPU, service, fallback.

Moved from steam-frame-hand-tracking tests/test_inference.py with the module."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from framenpu.backends import CPUBackend, FallbackBackend, OUTPUTS, SHAPES, make_backend  # noqa: E402


def outputs(value=0):
    return {n: np.full(s, value, np.float32) for n, s in zip(OUTPUTS, SHAPES)}


class BackendTests(unittest.TestCase):
    def test_init_failure_keeps_cpu_and_does_not_retry(self):
        cpu, factory = Mock(), Mock(side_effect=PermissionError("no device access"))
        with self.assertLogs("framenpu.backends", "WARNING"):
            backend = FallbackBackend(cpu, factory)
        backend.infer([1])
        backend.infer([2, 3])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(cpu.infer.call_args_list[0].args[0], [1])
        self.assertEqual(backend.name, "cpu")

    def test_inference_failure_replays_entire_batch_on_cpu_and_latches(self):
        cpu = Mock()
        accelerator = Mock()
        accelerator.infer.side_effect = RuntimeError("completion uncertain")
        accelerator.close.side_effect = RuntimeError("state retained")
        backend = FallbackBackend(cpu, lambda: accelerator)
        feeds = [object(), object(), object()]
        with self.assertLogs("framenpu.backends", "WARNING"):
            backend.infer(feeds)
        self.assertIs(cpu.infer.call_args.args[0], feeds)
        backend.infer([object()])
        accelerator.infer.assert_called_once_with(feeds)
        accelerator.close.assert_called_once()
        backend.idle()
        accelerator.idle.assert_not_called()
        backend.close()
        cpu.close.assert_called_once()

    def test_invalid_accelerator_outputs_fall_back(self):
        for result in ([], [outputs(float("nan"))], [{"heatmap_xy": np.zeros(1)}]):
            with self.subTest(result=len(result)):
                cpu, accelerator = Mock(), Mock()
                accelerator.infer.return_value = result
                backend = FallbackBackend(cpu, lambda: accelerator)
                with self.assertLogs("framenpu.backends", "WARNING"):
                    backend.infer([object()])
                cpu.infer.assert_called_once()
                self.assertEqual(backend.name, "cpu")

    def test_valid_batches_are_not_padded_and_idle_releases_accelerator(self):
        cpu, accelerator = Mock(), Mock()
        backend = FallbackBackend(cpu, lambda: accelerator)
        for count in (1, 2, 3, 1):
            accelerator.infer.return_value = [outputs(i) for i in range(count)]
            self.assertEqual(len(backend.infer([object() for _ in range(count)])), count)
        self.assertEqual([len(c.args[0]) for c in accelerator.infer.call_args_list], [1, 2, 3, 1])
        cpu.infer.assert_not_called()
        backend.idle()
        accelerator.idle.assert_called_once()
        for count in (0, 4):
            with self.assertRaises(ValueError):
                backend.infer([None] * count)

    def test_idle_failure_disables_accelerator_before_next_observation(self):
        cpu, accelerator = Mock(), Mock()
        accelerator.idle.side_effect = RuntimeError("power verification failed")
        backend = FallbackBackend(cpu, lambda: accelerator)
        with self.assertLogs("framenpu.backends", "WARNING"):
            backend.idle()
        backend.infer([object()])
        accelerator.close.assert_called_once()
        accelerator.infer.assert_not_called()
        cpu.infer.assert_called_once()

    def test_backend_timing_disappears_on_cpu_fallback(self):
        cpu, accelerator = Mock(), Mock()
        accelerator.last_timing = {"crops": 2, "body_ms": 10.0}
        backend = FallbackBackend(cpu, lambda: accelerator)
        self.assertEqual(backend.last_timing, accelerator.last_timing)
        accelerator.idle.side_effect = RuntimeError("cannot verify power")
        with self.assertLogs("framenpu.backends", "WARNING"):
            backend.idle()
        self.assertIsNone(backend.last_timing)

    def test_cpu_default_does_not_construct_accelerator(self):
        with (
            patch("framenpu.backends.CPUBackend") as cpu,
            patch("framenpu.backends.FrameNPUBackend") as npu,
        ):
            self.assertIs(make_backend("model"), cpu.return_value)
            npu.assert_not_called()


class FrameRequestTests(unittest.TestCase):
    def backend(self, client):
        accelerator = Mock()
        accelerator.client = client
        accelerator.infer.return_value = [outputs(1)]
        return FallbackBackend(Mock(), lambda: accelerator), accelerator

    def test_frame_request_carries_the_extra_crops(self):
        client = Mock(max_frame=2, models=[0, 2])
        client.infer_frame.return_value = [[outputs(2)], ["det"]]
        backend, _ = self.backend(client)
        kp, ex = backend.infer_frame(["a"], (2, ["d"]))
        self.assertEqual(client.infer_frame.call_args.args[0], [(0, ["a"]), (2, ["d"])])
        self.assertEqual((float(kp[0][OUTPUTS[0]].flat[0]), ex), (2.0, ["det"]))

    def test_no_frame_support_runs_keypoints_alone(self):
        for client in (None, Mock(max_frame=0, models=[0, 2]), Mock(max_frame=2, models=[0])):
            backend, accelerator = self.backend(client)
            kp, ex = backend.infer_frame(["a"], (2, ["d"]))
            self.assertIsNone(ex)
            self.assertEqual(accelerator.infer.call_args.args[0], ["a"])

    def test_frame_failure_latches_to_cpu(self):
        client = Mock(max_frame=2, models=[0, 2])
        client.infer_frame.side_effect = RuntimeError("completion uncertain")
        backend, _ = self.backend(client)
        with self.assertLogs("framenpu.backends", "WARNING"):
            kp, ex = backend.infer_frame(["a"], (2, ["d"]))
        self.assertIsNone(ex)
        self.assertEqual(backend.cpu.infer.call_args.args[0], ["a"])
        self.assertEqual(backend.name, "cpu")
        client.infer_frame.side_effect = None
        client.infer_frame.return_value = [[outputs()], ["d"], ["x"]]
        backend2, _ = self.backend(client)
        client.infer_frame.return_value = [[outputs()], []]  # wrong extra count
        with self.assertLogs("framenpu.backends", "WARNING"):
            self.assertIsNone(backend2.infer_frame(["a"], (2, ["d"]))[1])


class CPUParityTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("FNPU_MERCURY_MODEL"), "FNPU_MERCURY_MODEL not set")
    def test_cpu_outputs_are_exactly_the_existing_onnx_session(self):
        backend = CPUBackend(os.environ["FNPU_MERCURY_MODEL"])
        fs = [
            {
                "inputImg": np.random.default_rng(i)
                .uniform(-1, 1, (1, 1, 128, 128))
                .astype(np.float32),
                "lastKeypoints": np.zeros((1, 42), np.float32),
                "useLastKeypoints": np.zeros(1, np.float32),
            }
            for i in range(3)
        ]
        for feeds, out in zip(fs, backend.infer(fs)):
            old = backend.sess.run(list(OUTPUTS), feeds)
            for name, value in zip(OUTPUTS, old):
                np.testing.assert_array_equal(out[name], value)


if __name__ == "__main__":
    unittest.main()
