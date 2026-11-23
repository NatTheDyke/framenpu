"""Moved to framenpu_cdsp.provenance; this name stays an alias during the library split."""

import sys

from framenpu_cdsp import provenance as _module

sys.modules[__name__] = _module
