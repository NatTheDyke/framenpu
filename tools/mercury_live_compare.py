"""Run balanced (ABBA) tracker trials and keep aggregate timing evidence.

--order lists runs as backend[:scheduler]; scheduler is early (--merc-early,
the default) or fresh (--merc-fresh). Default: CPU/NPU/NPU/CPU, early.

Camera capture and calibration must already be ready. Raw telemetry stays in
--directory on the Frame. The output contains timings/counts/temperatures only.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mercury_live_report import summarize


def hottest():
    values = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        if (z / "type").read_text().startswith(("cpu", "gpuss")):
            values.append(int((z / "temp").read_text()) / 1000)
    return max(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("tracker", "jobs", "directory", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seconds", type=int, default=18)
    p.add_argument("--warmup", type=float, default=3)
    p.add_argument("--order", default="cpu,framenpu,framenpu,cpu")
    p.add_argument(
        "--cooldown-c",
        type=float,
        help="before each run, wait (at most 300 s) until the hottest CPU/GPU zone is at or below this",
    )
    a = p.parse_args()
    order = [t if ":" in t else t + ":early" for t in a.order.split(",")]
    for token in order:
        backend, scheduler = token.split(":")
        if backend not in ("cpu", "framenpu") or scheduler not in ("early", "fresh"):
            p.error("--order tokens are cpu|framenpu[:early|fresh]")
    if not a.directory.is_dir():
        p.error("existing private test directory required")
    report = {
        "format": "framenpu-live-tracker-abba",
        "seconds_per_run": a.seconds,
        "warmup_s": a.warmup,
        "runs": [],
        "order": order,
        "scope": "live camera pipeline, local pose only; camera detected timestamps, not motion-to-photon latency",
    }
    try:
        for i, token in enumerate(order):
            mode, scheduler = token.split(":")
            telemetry, log = (
                a.directory / f"abba-{i}-{mode}-{scheduler}.csv",
                a.directory / f"abba-{i}-{mode}-{scheduler}.log",
            )
            command = [
                sys.executable,
                "-m",
                "handtrack.live",
                "--backend",
                "mercury",
                "--merc-inference",
                mode,
                "--framenpu-jobs",
                str(a.jobs),
                "--merc-" + scheduler,
                "--flow-v2",
                "--no-http",
                "--no-bridge",
                "--seconds",
                str(a.seconds),
                "--telemetry",
                str(telemetry),
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(a.tracker) + os.pathsep + env.get("PYTHONPATH", "")
            samples, waited = [], 0.0
            if a.cooldown_c is not None:
                begin = time.monotonic()
                while hottest() > a.cooldown_c:
                    if time.monotonic() - begin > 300:
                        raise RuntimeError("device did not cool to the requested start temperature")
                    time.sleep(2)
                waited = time.monotonic() - begin
            print(
                json.dumps(
                    {"run": i, "backend": mode, "scheduler": scheduler, "phase": "starting"}
                ),
                flush=True,
            )
            with log.open("w") as output:
                process = subprocess.Popen(
                    command, env=env, stdout=output, stderr=subprocess.STDOUT
                )
                try:
                    deadline = time.monotonic() + a.seconds + 60
                    while process.poll() is None:
                        if time.monotonic() > deadline:
                            raise RuntimeError(
                                "tracker did not exit; worker startup or shutdown hung"
                            )
                        temp = hottest()
                        samples.append(temp)
                        if temp >= 92:
                            raise RuntimeError("existing tracker thermal pause threshold reached")
                        time.sleep(0.5)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=20)  # graceful DSP cleanup; never SIGKILL/retry
            summary = summarize(telemetry, a.warmup)
            row = {
                "index": i,
                "backend": mode,
                "scheduler": scheduler,
                "cooldown_s": waited,
                "exit_code": process.returncode,
                "temperature_c": {"start": samples[0], "end": samples[-1], "max": max(samples)},
                "summary": summary,
            }
            report["runs"].append(row)
            print(json.dumps(row), flush=True)
            if process.returncode or "error" in summary:
                raise RuntimeError("tracker run did not produce a valid measurement")
            if set(summary["backend_observations"]) != {mode}:
                raise RuntimeError("requested backend was not used throughout; refusing comparison")
            if "FrameNPU disabled" in log.read_text() or "FrameNPU unavailable" in log.read_text():
                raise RuntimeError("accelerator failed; no further DSP requests will be attempted")
        report["all_checks_passed"] = True
    except BaseException as exc:
        report["all_checks_passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        a.output.write_text(json.dumps(report, indent=1) + "\n")
    return int(not report["all_checks_passed"])


if __name__ == "__main__":
    sys.exit(main())
