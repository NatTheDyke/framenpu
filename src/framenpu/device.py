"""Moved to framenpu_cdsp.device; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import device as _module

sys.modules[__name__] = _module
