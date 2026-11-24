"""Hand-tracking models on the CPU, the Steam Frame's CDSP or its NPU service."""

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

__version__ = "0.4.0"


def open(manifests, backend="auto", **options):  # noqa: A001  (framenpu.open is the session API)
    """Open models by manifest on one backend: see framenpu.session (docs/CONTRACTS.md)."""
    from .session import open as _open

    return _open(manifests, backend, **options)


__all__ = [
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
