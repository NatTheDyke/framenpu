"""Moved to framenpu_cdsp.job; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import job as _module

sys.modules[__name__] = _module
