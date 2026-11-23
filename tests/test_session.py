"""framenpu.open (framenpu.session): models by manifest name on the CPU, the service and the
in-process device, behind the CPU fallback latch."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import framenpu  # noqa: E402
from framenpu import service, session as S  # noqa: E402
from test_service import FakeSession, det_crop, feed  # noqa: E402

KP, DET, S3D = "mercury-ft7", "mercury-detector-160", "mercury-stereo3d-tpc1"


def pair_feed(v):
    return {
        "inputImg": np.full((1, 2, 128, 128), v, np.float32),
        "lastKeypoints": np.zeros((1, 84), np.float32),
        "useLastKeypoints": np.array([1.0], np.float32),
        "geom": np.ones((1, 16), np.float32),
    }


def model_dir(root, name):
    """A copy of the bundled manifest with a stand-in weights file of matching hash."""
    doc = json.loads((ROOT / "src/framenpu/manifests" / f"{name}.json").read_text())
    d = Path(root) / name
    d.mkdir()
    blob = ("weights of " + name).encode()
    (d / doc["weights"]["file"]).write_bytes(blob)
    doc["weights"]["sha256"] = hashlib.sha256(blob).hexdigest()
    (d / "model.json").write_text(json.dumps(doc))
    return d


class FakeOrt:
    """onnxruntime stand-in: outputs filled with the sum of the inputs."""

    class SessionOptions:
        pass

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = 99

    runs = []

    class InferenceSession:
        def __init__(self, path, so, providers):
            self.path = path
            self.shapes = None

        def run(self, names, feed):
            FakeOrt.runs.append((Path(self.path).name, tuple(names)))
            s = float(sum(np.asarray(v).sum() for v in feed.values()))
            doc = json.loads((Path(self.path).parent / "model.json").read_text())
            shapes = {t["name"]: t["shape"] for t in doc["outputs"]}
            return [np.full(shapes[n], s, np.float32) for n in names]


class fake_ort:
    """Swap only sys.modules["onnxruntime"] (patch.dict would restore the whole module table and
    drop modules imported meanwhile)."""

    def start(self):
        mod = types.ModuleType("onnxruntime")
        for k in ("SessionOptions", "GraphOptimizationLevel", "InferenceSession"):
            setattr(mod, k, getattr(FakeOrt, k))
        self.saved = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = mod

    def stop(self):
        if self.saved is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = self.saved


class SessionTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dirs = {n: model_dir(self.tmp.name, n) for n in (KP, DET, S3D)}
        FakeOrt.runs = []
        self.ort = fake_ort()
        self.ort.start()

    def tearDown(self):
        self.ort.stop()
        self.tmp.cleanup()


class CPUTests(SessionTestBase):
    def test_cpu_runs_each_crop_and_names_models_by_manifest(self):
        s = framenpu.open([self.dirs[KP], self.dirs[DET]], backend="cpu")
        self.assertEqual(s.models, (KP, DET))
        self.assertEqual((s.max_crops(KP), s.max_crops(DET)), (4, 4))
        out = s.infer(KP, [feed(1), feed(2)])
        self.assertEqual(
            [float(o["curls"].flat[0]) for o in out], [128 * 128 + 1, 2 * 128 * 128 + 1]
        )
        kp, det = s.infer_frame([(KP, [feed(1)]), (DET, [det_crop(1)])])
        self.assertEqual(set(det[0]), {"hand_exists", "cx", "cy", "size"})
        self.assertEqual(det[0]["cx"].shape, (1, 2))
        with self.assertRaises(ValueError):
            s.infer("mercury-unknown", [feed(1)])
        with self.assertRaises(ValueError):
            s.infer_frame([(KP, [feed(1)]), (KP, [feed(2)])])

    def test_a_weights_file_finds_its_manifest_beside_it_or_bundled(self):
        w = self.dirs[KP] / "student_ft7.onnx"
        self.assertEqual(S.Entry(w).name, KP)  # model.json beside it
        lone = Path(self.tmp.name) / "lone.onnx"
        lone.write_bytes(w.read_bytes())
        with patch.object(S, "BUNDLED", self.dirs[KP].parent / "bundled"):
            (self.dirs[KP].parent / "bundled").mkdir()
            with self.assertRaises(ValueError):
                S.Entry(lone)
            (self.dirs[KP].parent / "bundled" / "mercury-ft7.json").write_text(
                (self.dirs[KP] / "model.json").read_text()
            )
            e = S.Entry(lone)
        self.assertEqual((e.name, e.weights), (KP, lone))

    def test_an_unlisted_retrained_model_borrows_a_bundled_manifest(self):
        other = Path(self.tmp.name) / "retrained.onnx"
        other.write_bytes(b"retrained weights")
        with self.assertRaises(ValueError):
            S.Entry(other)
        e = S.Entry((other, KP))
        self.assertEqual((e.name, e.weights), (KP, other))
        self.assertEqual(
            e.manifest.weights_sha256, hashlib.sha256(b"retrained weights").hexdigest()
        )
        e.verified_weights()
        s = framenpu.open([(other, KP)], backend="cpu")
        self.assertEqual(s.models, (KP,))

    def test_bundled_manifests_load(self):
        self.assertEqual(set(S.bundled_manifests()), {KP, DET, S3D, "mercury-keypoint-jan18"})

    def test_weights_that_differ_from_the_manifest_are_refused(self):
        (self.dirs[KP] / "student_ft7.onnx").write_bytes(b"other weights")
        with self.assertRaises(ValueError):
            framenpu.open([self.dirs[KP]], backend="cpu")

    def test_auto_without_a_service_is_the_cpu(self):
        s = framenpu.open([self.dirs[KP]], socket=os.path.join(self.tmp.name, "none.sock"))
        self.assertEqual(s.name, "cpu")

    def test_bad_arguments(self):
        for args, kw in (
            (([self.dirs[KP]],), {"backend": "gpu"}),
            (([],), {}),
            (([self.dirs[KP], self.dirs[KP]],), {"backend": "cpu"}),
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                framenpu.open(*args, **kw)


class ServiceSessionTests(SessionTestBase):
    det = True

    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.tmp.name, "npu.sock")
        FakeSession.opened = FakeSession.closed = 0
        self.fail_on = None
        self.srv = service.Server(
            "model",
            "jobs",
            self.path,
            os.getuid(),
            session_factory=lambda: FakeSession(self.fail_on, det=1),
            det_model="det",
            crops=4,
        )
        self.t = threading.Thread(target=self.srv.run, daemon=True)
        self.t.start()
        for _ in range(100):
            if os.path.exists(self.path):
                break
            time.sleep(0.01)

    def tearDown(self):
        self.srv.stop = True
        self.t.join(5)
        super().tearDown()

    def open(self, names=(KP, DET)):
        return framenpu.open([self.dirs[n] for n in names], socket=self.path)

    def test_service_serves_by_name_with_one_frame_submission(self):
        s = self.open()
        self.assertEqual(s.name, "framenpu")
        self.assertEqual((s.max_crops(KP), s.max_crops(DET)), (4, 2))
        out = s.infer(KP, [feed(k) for k in (1, 2, 3, 4)])
        self.assertEqual(
            [float(o["curls"].flat[0]) for o in out], [k * 128 * 128 + 1 for k in (1, 2, 3, 4)]
        )
        kp, det = s.infer_frame([(KP, [feed(1)]), (DET, [det_crop(2), det_crop(3)])])
        self.assertEqual(float(kp[0]["heatmap_xy"].flat[0]), 128 * 128 + 1)
        self.assertEqual([float(d["cx"].flat[0]) for d in det], [2 * 160 * 160, 3 * 160 * 160])
        self.assertEqual(len(s.last_timing["requests"]), 2)
        self.assertEqual(FakeOrt.runs, [])  # nothing ran on the CPU
        s.close()

    def test_a_model_the_service_does_not_serve_runs_on_the_cpu_alone(self):
        s = self.open((KP, DET, S3D))
        self.assertEqual(s.name, "framenpu")
        self.assertEqual([s.backend_of(m) for m in (KP, DET, S3D)], ["framenpu", "framenpu", "cpu"])
        self.assertEqual(s.max_crops(S3D), 4)
        kp, j3 = s.infer_frame([(KP, [feed(1)]), (S3D, [pair_feed(2)])])
        self.assertEqual(float(kp[0]["curls"].flat[0]), 128 * 128 + 1)
        self.assertEqual(set(j3[0]), {"joints3d", "scalar_extras"})
        self.assertEqual(
            [r[0] for r in FakeOrt.runs], ["student_s3d_tpc1.onnx"]
        )  # only the student
        self.assertEqual(s.backend_of(KP), "framenpu")  # no latch

    def test_infer_many_is_one_round_trip_and_keeps_order(self):
        s = self.open()
        calls = []
        orig = s.accelerator.client.infer_many
        s.accelerator.client.infer_many = lambda reqs, **k: (
            calls.append([(m, len(f)) for m, f in reqs]) or orig(reqs, **k)
        )
        outs = s.infer_many([(DET, [det_crop(1)]), (KP, [feed(1), feed(2)]), (KP, [feed(3)])])
        self.assertEqual(calls, [[(2, 1), (0, 2), (0, 1)]])
        self.assertEqual([len(o) for o in outs], [1, 2, 1])
        self.assertEqual(float(outs[2][0]["curls"].flat[0]), 3 * 128 * 128 + 1)
        self.assertEqual(len(s.last_timing["requests"]), 3)

    def test_a_service_serving_none_of_the_models_leaves_the_cpu(self):
        with self.assertLogs("framenpu.session", "WARNING"):
            s = self.open((S3D,))
        self.assertEqual(s.name, "cpu")

    def test_a_device_fault_latches_the_session_to_cpu(self):
        s = self.open()
        self.fail_on = 5 * 128 * 128 + 1
        with self.assertLogs("framenpu.session", "WARNING"):
            out = s.infer(KP, [feed(5)])
        self.assertEqual(s.name, "cpu")
        self.assertEqual(float(out[0]["curls"].flat[0]), 5 * 128 * 128 + 1)
        self.assertEqual(
            FakeOrt.runs,
            [("student_ft7.onnx", ("heatmap_xy", "heatmap_depth", "scalar_extras", "curls"))],
        )
        self.assertEqual(s.max_crops(KP), 4)
        with self.assertRaises(ValueError):
            s.infer(KP, [])


class FallbackTests(SessionTestBase):
    def session(self, accel):
        cpu = S.CPUSession([S.Entry(self.dirs[KP])])
        return S.FallbackSession(cpu, lambda: accel)

    def test_invalid_outputs_fall_back_and_frames_replay_whole(self):
        accel = Mock(models=(KP,))
        accel.name = "framenpu"
        accel.max_crops.return_value = 3
        accel.infer_frame.return_value = [[{"heatmap_xy": np.zeros(1)}]]
        s = self.session(accel)
        with self.assertLogs("framenpu.session", "WARNING"):
            (out,) = s.infer_frame([(KP, [feed(1), feed(2)])])
        self.assertEqual(len(out), 2)
        self.assertEqual(s.name, "cpu")
        accel.close.assert_called_once()

    def test_idle_failure_disables_before_the_next_request(self):
        accel = Mock(models=(KP,))
        accel.max_crops.return_value = 3
        accel.idle.side_effect = RuntimeError("power verification failed")
        s = self.session(accel)
        with self.assertLogs("framenpu.session", "WARNING"):
            s.idle()
        s.infer(KP, [feed(1)])
        accel.infer.assert_not_called()


class DeviceSessionTests(SessionTestBase):
    def test_device_session_maps_names_to_session_models(self):
        fake = Mock(max_crops=4)
        fake.infer.return_value = ["kp"]
        fake.infer_frame.return_value = [["kp"], ["det"]]
        with patch("framenpu.mercury_runtime.MercurySession", return_value=fake) as ms:
            d = S.DeviceSession([S.Entry(self.dirs[KP]), S.Entry(self.dirs[DET])], "jobs")
        args, kw = ms.call_args
        self.assertEqual(args[1], "jobs")
        self.assertEqual(
            kw["extra_models"], [(str(self.dirs[DET] / "grayscale_detection_160x160.onnx"), 2)]
        )
        self.assertEqual((d.max_crops(KP), d.max_crops(DET)), (4, 2))
        d.infer(DET, ["c"])
        fake.infer.assert_called_with(["c"], model=1)
        d.infer_frame([(DET, ["c"]), (KP, ["k"])])
        fake.infer_frame.assert_called_with([(1, ["c"]), (0, ["k"])])
        d.infer_many([(KP, ["a"]), (DET, ["b"]), (KP, ["c"])])
        fake.infer_many.assert_called_with([(0, ["a"]), (1, ["b"]), (0, ["c"])])
        with self.assertRaises(ValueError):
            S.DeviceSession([S.Entry(self.dirs[DET])], "jobs")


if __name__ == "__main__":
    unittest.main()
