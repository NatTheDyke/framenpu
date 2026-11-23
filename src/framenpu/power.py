"""Moved to framenpu_cdsp.power; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import power as _module

sys.modules[__name__] = _module
