"""Moved to framenpu_cdsp.profile; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import profile as _module

sys.modules[__name__] = _module
