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


class _Opened(Exception):
    pass


class ClockSettingTests(unittest.TestCase):
    """The job-scoped AXI raise and the HMX rate are plugin settings: from the option or the
    environment, refused at load when the rate is unknown, reported as features, and handed to
    each device session by the lease."""

    def plugin(self, env=None, hmx_env=None, **options):
        from framenpu_cdsp.plugin import CDSPPlugin

        with patch.dict(os.environ):
            for name, value in (("FNPU_AXI_MHZ", env), ("FNPU_HMX_MHZ", hmx_env)):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            return CDSPPlugin(**options)

    def axi(self, p):
        return p._device_options()["_axi"]

    def test_off_by_default(self):
        p = self.plugin()
        self.assertIsNone(p.axi_mhz)
        self.assertEqual(p.hmx_mhz, 192.0)
        self.assertFalse({"axi", "hmx_fast"} & p.capabilities().features)
        self.assertIs(self.axi(p), False)  # the device does not read FNPU_AXI_MHZ itself
        self.assertEqual(p._device_options()["_power_factory"]().mhz, 192.0)

    def test_the_option_and_the_environment_turn_it_on(self):
        from framenpu_cdsp.axi import AxiClock

        for p in (self.plugin(axi_mhz=748.8), self.plugin("748.8")):
            self.assertEqual(p.axi_mhz, 748.8)
            self.assertIn("axi", p.capabilities().features)
            a, b = self.axi(p), self.axi(p)
            self.assertIsInstance(a, AxiClock)
            self.assertIsNot(a, b)  # one owner per device session
            self.assertEqual(a.mhz, 748.8)
        self.assertIsNone(self.plugin("748.8", axi_mhz=None).axi_mhz)  # the option wins

    def test_an_unknown_rate_is_refused_at_load(self):
        for kw in (
            {"env": "1500"},
            {"env": "fast"},
            {"axi_mhz": 500.0},
            {"hmx_env": "672"},
            {"hmx_mhz": 806.4},
        ):
            with self.assertRaises(ValueError):
                self.plugin(**kw)

    def test_the_option_and_the_environment_raise_the_hmx_rate(self):
        for p in (self.plugin(hmx_mhz=499.2), self.plugin(hmx_env="499.2")):
            self.assertEqual(p.hmx_mhz, 499.2)
            self.assertIn("hmx_fast", p.capabilities().features)
            # the session's power owner runs at the plugin's rate, whatever the env says later
            with patch.dict(os.environ, {"FNPU_HMX_MHZ": "192"}):
                self.assertEqual(p._device_options()["_power_factory"]().mhz, 499.2)
        self.assertEqual(self.plugin(hmx_env="499.2", hmx_mhz=192.0).hmx_mhz, 192.0)

    def test_the_lease_gives_each_device_session_the_plugins_owner(self):
        from unittest.mock import MagicMock

        seen = []

        def device(**kw):
            seen.append(kw)
            raise _Opened

        for mhz in (748.8, None):
            p = self.plugin(axi_mhz=mhz, _device_factory=device, _lease_factory=MagicMock)
            with self.assertRaises(_Opened):
                p.open().open()
        self.assertEqual([k["dispatch"] for k in seen], ["direct", "direct"])
        self.assertEqual(seen[0]["_axi"].mhz, 748.8)
        self.assertIs(seen[1]["_axi"], False)
        self.assertEqual([k["_power_factory"]().mhz for k in seen], [192.0, 192.0])


if __name__ == "__main__":
    unittest.main()
