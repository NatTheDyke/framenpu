"""Moved to framenpu_cdsp.errors; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import errors as _module

sys.modules[__name__] = _module
