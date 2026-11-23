"""Moved to framenpu_cdsp.transport; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import transport as _module

sys.modules[__name__] = _module
