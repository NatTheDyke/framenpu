"""Device backends, found through the ``framenpu.devices`` entry-point group (docs/CONTRACTS.md).

framenpu imports no backend module: a backend registers a factory under the group (framenpu-cdsp
registers ``cdsp``), and ``load`` calls it and checks the plugin ABI. A PC or CI install without a
backend runs on the CPU alone; opening the device there raises DeviceUnavailable.

FRAMENPU_DEVICES="name=module:attr[,name=module:attr]" names factories directly, for source trees
run without installed package metadata (PYTHONPATH=src, headset harnesses); it overrides an
installed entry point of the same name.
"""

from __future__ import annotations

import importlib
import os
from importlib import metadata

from .contracts import DEVICE_ENTRY_POINTS, PLUGIN_ABI, DevicePlugin, abi_compatible

ENV = "FRAMENPU_DEVICES"


class DeviceUnavailable(RuntimeError):
    """No usable backend of that name: none installed, or one of an ABI framenpu does not support."""


def _from_env():
    out = {}
    for item in filter(None, (s.strip() for s in os.environ.get(ENV, "").split(","))):
        name, _, target = item.partition("=")
        module, _, attr = target.partition(":")
        if not (name and module and attr):
            raise ValueError(f"{ENV}: expected name=module:attr, got {item!r}")
        out[name] = (module, attr)
    return out


def _factories():
    """name -> a callable returning the backend's plugin factory (imports happen on use)"""
    out = {ep.name: ep.load for ep in metadata.entry_points(group=DEVICE_ENTRY_POINTS)}
    for name, (module, attr) in _from_env().items():
        out[name] = lambda module=module, attr=attr: getattr(importlib.import_module(module), attr)
    return out


def available():
    """Names of the registered device backends."""
    return sorted(_factories())


def load(name="cdsp", **options):
    """The backend's plugin, constructed with options (cdsp: jobs=, the image folder)."""
    factory = _factories().get(name)
    if factory is None:
        have = ", ".join(available()) or "none"
        raise DeviceUnavailable(f"no device backend {name!r} (registered: {have})")
    plugin = factory()(**options)
    if not isinstance(plugin, DevicePlugin):
        raise DeviceUnavailable(f"device backend {name!r} is not a framenpu device plugin")
    if not abi_compatible(plugin.abi):
        raise DeviceUnavailable(
            f"device backend {name!r} has plugin ABI {plugin.abi!r}; framenpu supports {PLUGIN_ABI}"
        )
    return plugin
