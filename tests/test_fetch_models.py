"""tools/fetch_models.py: every bundled manifest has a pinned place to fetch from, and a file that
does not match its manifest's hash is refused, never kept or overwritten. No network."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fetch_models as F  # noqa: E402


class _Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        out, self.data = self.data[:n], self.data[n:]
        return out


class FetchModelsTests(unittest.TestCase):
    def test_every_manifest_has_a_pinned_url(self):
        found = F.manifests()
        self.assertEqual(
            set(found),
            {
                "mercury-ft7",
                "mercury-stereo3d-tpc1",
                "mercury-keypoint-jan18",
                "mercury-detector-160",
            },
        )
        for name, m in found.items():
            url = F.url_of(m)
            self.assertTrue(url.endswith("/" + m["weights"]["file"]), url)
            if m.get("source"):
                self.assertIn("/resolve/%s/" % m["source"]["revision"], url)
            else:  # Mercury's own weights: from Monado at a full commit, never re-hosted
                self.assertIn(name, F.MONADO_MODELS)
                self.assertRegex(url, r"/-/raw/[0-9a-f]{40}/")

    def manifest(self, data):
        return {
            "name": "m",
            "weights": {"file": "m.onnx", "sha256": hashlib.sha256(data).hexdigest()},
            "source": {"repo": "u/m", "revision": "0" * 40},
        }

    def test_a_matching_download_is_kept_and_a_wrong_one_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            m = self.manifest(b"weights")
            with patch.object(F.urllib.request, "urlopen", return_value=_Response(b"weights")):
                self.assertEqual(F.fetch(m, out), (out / "m.onnx", "fetched"))
            self.assertEqual(F.fetch(m, out)[1], "kept")  # no download the second time
            bad = self.manifest(b"other")
            bad["weights"]["file"] = "n.onnx"
            with patch.object(F.urllib.request, "urlopen", return_value=_Response(b"weights")):
                with self.assertRaises(SystemExit):
                    F.fetch(bad, out)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["m.onnx"])  # nothing left over

    def test_an_existing_file_with_another_hash_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "m.onnx").write_bytes(b"mine")
            with self.assertRaises(SystemExit):
                F.fetch(self.manifest(b"weights"), out)
            self.assertEqual((out / "m.onnx").read_bytes(), b"mine")


if __name__ == "__main__":
    unittest.main()
