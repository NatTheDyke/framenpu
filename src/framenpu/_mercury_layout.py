"""Moved to framenpu_cdsp._mercury_layout; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import _mercury_layout as _module

sys.modules[__name__] = _module
