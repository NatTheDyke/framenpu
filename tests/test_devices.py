"""framenpu.devices: finding device backends by entry point, and the cdsp plugin's capabilities."""

import os
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from framenpu import devices
from framenpu.contracts import DEVICE_ENTRY_POINTS, PLUGIN_ABI, Capabilities, DevicePlugin

ROOT = Path(__file__).resolve().parents[1]
CDSP = "cdsp=framenpu_cdsp.plugin:CDSPPlugin"


class _Plugin:
    abi, name = PLUGIN_ABI, "fake"

    def __init__(self, **options):
        self.options = options

    def capabilities(self):
        return Capabilities(abi=self.abi, image_sha256="0" * 64, max_crops=1)

    def compile(self, graph, batch):
        raise NotImplementedError

    def open(self):
        raise NotImplementedError


class _OldPlugin(_Plugin):
    abi = PLUGIN_ABI + 1


class DiscoveryTests(unittest.TestCase):
    def env(self, value):
        return patch.dict(os.environ, {devices.ENV: value})

    def test_the_package_registers_the_cdsp_backend(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(
            project["entry-points"][DEVICE_ENTRY_POINTS],
            {"cdsp": "framenpu_cdsp.plugin:CDSPPlugin"},
        )

    def test_env_names_a_factory_and_load_passes_options(self):
        with self.env(f"fake={__name__}:_Plugin"):
            self.assertIn("fake", devices.available())
            plugin = devices.load("fake", jobs="J")
        self.assertIsInstance(plugin, DevicePlugin)
        self.assertEqual(plugin.options, {"jobs": "J"})

    def test_missing_backend_and_other_abi_are_unavailable(self):
        with self.env(""), patch.object(devices.metadata, "entry_points", return_value=[]):
            with self.assertRaisesRegex(devices.DeviceUnavailable, "registered: none"):
                devices.load("cdsp")
        with self.env(f"old={__name__}:_OldPlugin"):
            with self.assertRaisesRegex(devices.DeviceUnavailable, "plugin ABI"):
                devices.load("old")
        with self.env(f"bad={__name__}:DiscoveryTests"):
            with self.assertRaisesRegex(devices.DeviceUnavailable, "not a framenpu device plugin"):
                devices.load("bad")

    def test_malformed_env_is_refused(self):
        with self.env("cdsp"), self.assertRaises(ValueError):
            devices.available()


class CDSPPluginTests(unittest.TestCase):
    def test_bundled_image_capabilities(self):
        with patch.dict(os.environ, {devices.ENV: CDSP}):
            plugin = devices.load("cdsp")
        from framenpu_cdsp import mercury_device as D

        caps = plugin.capabilities()
        self.assertEqual(plugin.jobs, str(D.BUNDLED_JOBS))
        self.assertEqual(caps.image_sha256, D.CROPS4_IMAGE_SHA256)
        self.assertEqual(
            (caps.abi, caps.max_crops, caps.segments, caps.stream), (PLUGIN_ABI, 4, 2, True)
        )
        self.assertEqual(caps.resident_slots, D.RES_SLOTS)
        self.assertLessEqual({"heads", "resident", "cbias", "tiles", "deep8"}, caps.features)

    def test_compile_refuses_requests_the_image_cannot_carry(self):
        from framenpu.contracts import Refusal
        from framenpu_cdsp.plugin import CDSPPlugin

        plugin = CDSPPlugin()
        for batch in (0, 5):
            got = plugin.compile(object(), batch)  # refused before the graph is read
            self.assertIsInstance(got, Refusal)
            self.assertEqual(got.reason, "model 0's requests carry one to 4 crops")
        self.assertEqual(plugin.plans, [])

    def test_an_unqualified_folder_is_refused(self):
        from framenpu_cdsp.plugin import CDSPPlugin

        with self.assertRaises(Exception):
            CDSPPlugin(jobs=ROOT / "tests")


if __name__ == "__main__":
    unittest.main()
