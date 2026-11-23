"""Experimental custom Hexagon jobs and HMX inference for Steam Frame."""

from .device import Device
from .errors import (
    CompilerError,
    DeviceBusy,
    FrameNPUError,
    InvalidModel,
    JobFailed,
    RecoveryRequired,
    UnsupportedFirmware,
    UnsupportedOperator,
)
from .job import Job, JobResult
from .model import Clip, Dense, Model, ReLU

__version__ = "0.1.0"


def open(manifests, backend="auto", **options):  # noqa: A001  (framenpu.open is the session API)
    """Open models by manifest on one backend: see framenpu.session (docs/CONTRACTS.md)."""
    from .session import open as _open

    return _open(manifests, backend, **options)


__all__ = [
    "Device",
    "Job",
    "JobResult",
    "Model",
    "Dense",
    "ReLU",
    "Clip",
    "FrameNPUError",
    "CompilerError",
    "DeviceBusy",
    "RecoveryRequired",
    "JobFailed",
    "UnsupportedFirmware",
    "InvalidModel",
    "UnsupportedOperator",
    "open",
]
