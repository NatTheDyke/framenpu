"""The bundled model manifests (src/framenpu/manifests/*.json) against the specs the code still carries.

Until sessions are opened by manifest, mercury.py, service.py and backends.py hold their own
copies of each model's inputs and outputs; this test keeps the four in step. With
FNPU_MERCURY_MODEL / FNPU_MERCURY_DETECTOR / FNPU_S3D_MODEL set, the files' hashes are checked
against the manifests too.
"""

import hashlib
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framenpu import backends, mercury, service  # noqa: E402
from framenpu.contracts import load_manifest  # noqa: E402

MODELS = ROOT / "src/framenpu/manifests"


def manifests():
    return {p.stem: load_manifest(p) for p in sorted(MODELS.glob("*.json"))}


def spec(tensors):
    return tuple((t.name, tuple(t.shape)) for t in tensors)


class ManifestTests(unittest.TestCase):
    def test_every_manifest_is_valid_and_named_for_its_file(self):
        found = manifests()
        self.assertEqual(
            set(found),
            {
                "mercury-ft7",
                "mercury-keypoint-jan18",
                "mercury-stereo3d-tpc1",
                "mercury-detector-160",
            },
        )
        for folder, m in found.items():
            self.assertEqual(m.name, folder)

    def test_keypoint_manifests_match_the_code(self):
        for name in ("mercury-ft7", "mercury-keypoint-jan18"):
            m = manifests()[name]
            self.assertEqual(spec(m.inputs), tuple(mercury.INPUTS.items()))
            self.assertEqual(spec(m.outputs), tuple(mercury.OUTPUTS.items()))
            self.assertEqual(spec(m.inputs), service.INPUTS)
            self.assertEqual(spec(m.outputs), service.OUTPUTS)
            self.assertEqual(spec(m.outputs), tuple(zip(backends.OUTPUTS, backends.SHAPES)))
        self.assertEqual(
            manifests()["mercury-keypoint-jan18"].weights_sha256, mercury.KEYPOINT_SHA256
        )
        self.assertEqual(manifests()["mercury-keypoint-jan18"].decode["heatmap_cell_offset"], 0.5)
        self.assertEqual(manifests()["mercury-ft7"].decode["heatmap_cell_offset"], 0)

    def test_stereo_and_detector_manifests_match_the_code(self):
        s3d, det = manifests()["mercury-stereo3d-tpc1"], manifests()["mercury-detector-160"]
        self.assertEqual(spec(s3d.inputs), tuple(mercury.S3D_INPUTS.items()))
        self.assertEqual(spec(s3d.inputs), service.S3D_INPUTS)
        self.assertEqual(spec(s3d.outputs), service.S3D_OUTPUTS)
        self.assertEqual(spec(det.inputs), tuple(mercury.DET_INPUTS.items()))
        self.assertEqual(spec(det.inputs), service.DET_INPUTS)
        self.assertEqual(spec(det.outputs), tuple(mercury.DET_OUTPUTS.items()))
        self.assertEqual(spec(det.outputs), service.DET_OUTPUTS)
        self.assertEqual(det.weights_sha256, mercury.DETECTOR_SHA256)
        # crops per request the service takes for each model
        self.assertEqual(s3d.preferred_crops, service.SPECS[1][2])
        self.assertEqual(det.preferred_crops, service.SPECS[2][2])

    def test_model_files_given_in_the_environment_match_a_manifest(self):
        known = {m.weights_sha256: n for n, m in manifests().items()}
        for var in ("FNPU_MERCURY_MODEL", "FNPU_MERCURY_DETECTOR", "FNPU_S3D_MODEL"):
            path = os.environ.get(var)
            if path:
                with self.subTest(var=var):
                    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    self.assertIn(digest, known, f"{path} matches no manifest")


if __name__ == "__main__":
    unittest.main()
