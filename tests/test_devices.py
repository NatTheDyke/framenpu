"""framenpu.devices: finding device backends by entry point, and refusing unusable ones."""

import os
import unittest
from unittest.mock import patch

from framenpu import devices
from framenpu.contracts import PLUGIN_ABI, Capabilities, DevicePlugin


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


if __name__ == "__main__":
    unittest.main()
