"""Error types of framenpu and its device backends (framenpu_cdsp.errors re-exports them)."""


class FrameNPUError(RuntimeError):
    pass


class UnsupportedFirmware(FrameNPUError):
    pass


class DeviceBusy(FrameNPUError):
    pass


class RecoveryRequired(FrameNPUError):
    """The DSP may still execute: leave its payload and clocks intact; the stock gate may expire."""


class JobFailed(FrameNPUError):
    pass


class InvalidModel(ValueError):
    pass


class UnsupportedOperator(InvalidModel):
    pass


class CompilerError(FrameNPUError):
    pass
