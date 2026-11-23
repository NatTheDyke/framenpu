"""ctypes bindings for the FrameNPU layout kernels (host/fnpu_layout.c).

The host side of the Mercury kernel's tensor layout: window reads (the power verify path),
the stride-2 stem im2col and grid packing. The bundled library (data/layout_aarch64.so) is
loaded only on aarch64, and only after its size and SHA-256 match data/layout_aarch64.json;
anything else returns None and callers use the NumPy path. Tests load a local build of the
same source with ``LayoutHost(path, record)``.

Every wrapper checks dtype, contiguity and size before passing a pointer: the C side trusts
its arguments.
"""

import ctypes
import hashlib
import json
import platform
from pathlib import Path

DATA = Path(__file__).with_name("data")
ABI = 1
FORMAT = "framenpu-cdsp-layout-native"

_i32, _p = ctypes.c_int32, ctypes.c_void_p
_SIGNATURES = {
    "fnpu_layout_abi": (_i32, ()),
    "fnpu_read32": (None, (_p, _p, _i32, _p)),
    "fnpu_stem_columns": (None, (_p, _p)),
    "fnpu_grid_pack": (None, (_p, _i32, _i32, _p, _i32, _p)),
    "fnpu_grid_unpack": (None, (_p, _i32, _i32, _p, _i32, _p, _p)),
}


def _arr(a, dtype, n=None, writable=False):
    """Pointer to a C-contiguous array of exactly ``dtype`` (and ``n`` elements)."""
    import numpy as np

    if not isinstance(a, np.ndarray) or a.dtype != np.dtype(dtype) or not a.flags.c_contiguous:
        raise TypeError(f"native kernels need a C-contiguous {np.dtype(dtype)} array")
    if n is not None and a.size != n:
        raise ValueError(f"native kernel expected {n} elements, got {a.size}")
    if writable and not a.flags.writeable:
        raise ValueError("native kernel output must be writable")
    return a.ctypes.data


class LayoutHost:
    """The layout kernels of one verified library file."""

    def __init__(self, path, record):
        path = Path(path)
        data = path.read_bytes()
        if (
            record.get("format") != FORMAT
            or record.get("abi") != ABI
            or record.get("bytes") != len(data)
            or record.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError(f"{path.name} does not match its manifest")
        self.path, self.sha256 = path, record["sha256"]
        self.lib = ctypes.CDLL(str(path))
        for name, (res, args) in _SIGNATURES.items():
            f = getattr(self.lib, name)
            f.restype, f.argtypes = res, args
        if self.lib.fnpu_layout_abi() != ABI:
            raise ValueError("layout library ABI differs")

    def stem_columns(self, img16, out):
        self.lib.fnpu_stem_columns(_arr(img16, "u2", 128 * 128), _arr(out, "u2", 10 * 4096, True))

    def grid_pack(self, t16, channels, idx, nwords, out):
        hw = idx.size
        if channels % 2 or t16.size != channels * hw:
            raise ValueError("grid pack needs an even channel count and full planes")
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= nwords):
            raise ValueError("grid index outside the tensor")
        self.lib.fnpu_grid_pack(
            _arr(t16, "u2"),
            channels,
            hw,
            _arr(idx, "i4"),
            nwords,
            _arr(out, "u2", channels * nwords, True),
        )

    def grid_unpack(self, raw16, channels, idx, nwords, out, stats):
        """out (float32 channels x hw); returns (stray nonzero halves, non-finite values)."""
        hw = idx.size
        if channels % 2:
            raise ValueError("grid unpack needs an even channel count")
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= nwords):
            raise ValueError("grid index outside the tensor")
        self.lib.fnpu_grid_unpack(
            _arr(raw16, "u2", channels * nwords),
            channels,
            hw,
            _arr(idx, "i4"),
            nwords,
            _arr(out, "f4", channels * hw, True),
            _arr(stats, "i4", 2, True),
        )
        return int(stats[0]), int(stats[1])


_BUNDLED = {}


def bundled():
    """The verified bundled library on aarch64, else None (callers use NumPy)."""
    if "layout" not in _BUNDLED:
        host = None
        if platform.machine() == "aarch64":
            lib, rec = DATA / "layout_aarch64.so", DATA / "layout_aarch64.json"
            if lib.is_file() and rec.is_file():
                host = LayoutHost(lib, json.loads(rec.read_text()))
        _BUNDLED["layout"] = host
    return _BUNDLED["layout"]
