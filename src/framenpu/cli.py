"""framenpu's command line: the NPU service."""

import argparse
import sys

from . import __version__


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["serve"]:
        # the NPU service parses its own options; it sets its BLAS thread count before numpy loads
        from . import service

        return service.main(argv[1:])
    parser = argparse.ArgumentParser(prog="framenpu")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "serve", help="run the NPU service (root): framenpu serve --model M [--jobs DIR] --user U"
    )
    parser.parse_args(argv)
    parser.error("options before serve are not supported")


if __name__ == "__main__":
    main()
