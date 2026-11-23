"""Moved to framenpu_cdsp.axi; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import axi as _module

sys.modules[__name__] = _module
