import copy
import json
import os
import tempfile
import unittest

from framenpu import contracts as K
from framenpu import mercury

SHA = "ab" * 32
REV = "0123456789abcdef0123456789abcdef01234567"


def manifest_for(name, kind, inputs, outputs, preferred=3):
    doc = {
        "manifest_version": 1,
        "name": name,
        "kind": kind,
        "weights": {"file": "model.onnx", "sha256": SHA},
        "inputs": [{"name": k, "shape": list(v)} for k, v in inputs.items()],
        "outputs": [{"name": k, "shape": list(v)} for k, v in outputs.items()],
        "batch": {"preferred": preferred},
    }
    if kind == "keypoint":
        doc["decode"] = {"heatmap_cell_offset": 0}
    return doc


FT7 = manifest_for("mercury-ft7", "keypoint", mercury.INPUTS, mercury.OUTPUTS)


class ManifestTests(unittest.TestCase):
    def test_todays_model_signatures_are_expressible(self):
        kinds = ("keypoint", "stereo3d", "detector")
        for (ins, outs), kind in zip(mercury.SIGNATURES, kinds):
            m = K.validate_manifest(manifest_for("m-" + kind, kind, ins, outs, preferred=2))
            self.assertEqual(m.input_shapes(), {k: tuple(v) for k, v in ins.items()})
            self.assertEqual(m.output_shapes(), {k: tuple(v) for k, v in outs.items()})

    def test_round_trip_through_a_file(self):
        doc = dict(FT7, source={"repo": "someone/mercury-ft7", "revision": REV})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "model.json")
            with open(p, "w") as f:
                json.dump(doc, f)
            m = K.load_manifest(p)
        self.assertEqual((m.name, m.kind, m.preferred_crops), ("mercury-ft7", "keypoint", 3))
        self.assertEqual(m.decode["heatmap_cell_offset"], 0)
        self.assertEqual(m.source["revision"], REV)

    def test_refusals_name_the_field(self):
        def bad(field, change):
            doc = copy.deepcopy(FT7)
            change(doc)
            with self.assertRaises(K.ManifestError) as cm:
                K.validate_manifest(doc)
            self.assertIn(field, str(cm.exception))

        bad("manifest_version", lambda d: d.update(manifest_version=2))
        bad("name", lambda d: d.update(name="Mercury FT7"))
        bad("kind", lambda d: d.update(kind="segmenter"))
        bad("weights.file", lambda d: d["weights"].update(file="../model.onnx"))
        bad("weights.sha256", lambda d: d["weights"].update(sha256="xyz"))
        bad("inputs[0].shape", lambda d: d["inputs"][0].update(shape=[2, 1, 128, 128]))
        bad("inputs[1].name", lambda d: d["inputs"][1].update(name="inputImg"))
        bad("outputs[0].dtype", lambda d: d["outputs"][0].update(dtype="float16"))
        bad("batch.preferred", lambda d: d["batch"].update(preferred=0))
        bad("decode.heatmap_cell_offset", lambda d: d.update(decode={}))
        bad("decode.heatmap_cell_offset", lambda d: d.update(decode={"heatmap_cell_offset": 0.25}))
        bad("source.revision", lambda d: d.update(source={"repo": "a/b", "revision": "main"}))
        bad("colour", lambda d: d.update(colour="blue"))

    def test_only_keypoint_models_must_state_a_cell_offset(self):
        det = manifest_for("mercury-det", "detector", mercury.DET_INPUTS, mercury.DET_OUTPUTS, 2)
        self.assertEqual(K.validate_manifest(det).decode, {})

    def test_not_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "model.json")
            with open(p, "w") as f:
                f.write("{")
            with self.assertRaises(K.ManifestError):
                K.load_manifest(p)


class ProtocolTests(unittest.TestCase):
    def test_session_shape(self):
        class Fake:
            models = ("mercury-ft7",)
            last_timing = None

            def max_crops(self, model):
                return 4

            def infer(self, model, crops):
                return [{} for _ in crops]

            def infer_frame(self, requests):
                return [self.infer(m, c) for m, c in requests]

            infer_many = infer_frame

            def backend_of(self, model):
                return "cpu"

            def idle(self):
                pass

            def close(self):
                pass

        self.assertIsInstance(Fake(), K.Session)
        self.assertNotIsInstance(object(), K.Session)

    def test_plugin_shape_and_abi(self):
        class Plugin:
            abi, name = 1, "fake"

            def capabilities(self):
                return K.Capabilities(abi=1, image_sha256=SHA, max_crops=4, segments=2, stream=True)

            def compile(self, graph, batch):
                return K.Refusal("host only")

            def open(self):
                raise NotImplementedError

        p = Plugin()
        self.assertIsInstance(p, K.DevicePlugin)
        self.assertTrue(K.abi_compatible(p.abi))
        self.assertFalse(K.abi_compatible(2))
        self.assertFalse(K.abi_compatible(True))
        self.assertEqual(p.capabilities().max_crops, 4)


if __name__ == "__main__":
    unittest.main()
