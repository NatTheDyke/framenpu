"""A cdsp device plugin with no DSP, for CI (framenpu's and other repositories').

    FRAMENPU_DEVICES=cdsp=sim_plugin:SimPlugin   (with framenpu's tests/ on PYTHONPATH)

The real CDSPPlugin (image checks, capabilities, plans, staging, lease) with the lane simulator
of tests/test_mercury_runtime.py standing in for the device: every request runs the plan in the
independent simulator, so outputs equal framenpu.mercury.run(..., "fp16") bit for bit (the
contract the device is qualified against). framenpu.open(..., "device") and framenpu serve then
run their whole device path on a machine without a DSP. Not a performance model: timings are
fixed.
"""

from types import SimpleNamespace

from framenpu_cdsp.plugin import CDSPPlugin

import test_mercury_runtime as T


class SimPlugin(CDSPPlugin):
    def __init__(self, jobs=None, *, resident=None, **options):
        options.pop("_device_factory", None)
        options.pop("_lease_factory", None)
        super().__init__(
            jobs,
            resident=resident,
            _device_factory=self._sim_device,
            _lease_factory=T._Lease,
            **options,
        )

    def _sim_device(self, dispatch, **kw):
        dev = T._Device(dispatch, **kw)
        f = self.features
        # the simulator reads the session's models and image record from its owner
        dev.owner = SimpleNamespace(
            models=[SimpleNamespace(plan=p.plan) for p in self.plans],
            plan=self.plans[0].plan,
            win_hi=f.win_hi,
            image_sha256=f.sha256,
            cbias=f.cbias,
            resident_image=f.resident,
            res_slots=bool(f.res_slots),
        )
        return dev
