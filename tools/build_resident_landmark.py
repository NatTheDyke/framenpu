"""Build the pinned resident graph modules without opening a device."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from framenpu.compiler import build_job
from framenpu.resident_landmark import JOB_KINDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {}
    for name in JOB_KINDS.values():
        job = build_job(
            ROOT / "tests" / f"resident_graph_{name}.c",
            args.output / f"resident_graph_{name}.fjob",
            requires_hmx=True,
            build_dir=args.output / name,
        )
        record[name] = {"bytes": len(job.code), "sha256": hashlib.sha256(job.code).hexdigest()}
    (args.output / "jobs.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
