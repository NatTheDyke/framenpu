"""Moved to framenpu_cdsp.resident; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import resident as _module

sys.modules[__name__] = _module
