"""Aggregate tracker telemetry locally; raw pose/camera records stay private."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def percentiles(values):
    import numpy as np

    if not values:
        return None
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def summarize(path, warmup_s=3):
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames, observations, duration = [], [], 0.0
    for entry in paths:
        rows = list(csv.reader(Path(entry).read_text().splitlines()))
        fs = [r for r in rows if r and r[0] == "F"]
        rs = [r for r in rows if r and r[0] == "R"]
        if not fs:
            return {"error": "no tracked camera frames", "observations": len(rs)}
        begin = min(int(r[1]) for r in fs) + round(warmup_s * 1e9)
        fs = [r for r in fs if int(r[1]) >= begin]
        rs = [r for r in rs if int(r[4]) >= begin and int(r[6]) > 0]
        if not fs or not rs:
            return {"error": "no observations after warmup", "observations": len(rs)}
        duration += (max(int(r[2]) for r in fs) - begin) / 1e9
        frames.extend(fs)
        observations.extend(rs)
    groups = {}
    for r in observations:
        groups.setdefault(r[6], []).append(r)
    requests = [group[0] for group in groups.values()]
    accepted = [r for r in observations if r[12] in ("mono", "stereo")]
    out = {
        "warmup_s": warmup_s,
        "duration_s": duration,
        "observations": len(observations),
        "observations_per_s": len(observations) / duration,
        "accepted": len(accepted),
        "accepted_per_s": len(accepted) / duration,
        "outcomes": dict(Counter(r[12] for r in observations)),
        "backend_observations": dict(
            Counter(r[18] if len(r) > 18 else "unknown" for r in observations)
        ),
        "capture_to_worker_end_ms": percentiles(
            [(int(r[7]) - int(r[4])) / 1e6 for r in observations]
        ),
        "capture_to_collect_ms": percentiles([(int(r[8]) - int(r[4])) / 1e6 for r in observations]),
        "fast_frame_latency_ms": percentiles([(int(r[2]) - int(r[1])) / 1e6 for r in frames]),
        "fast_frames_per_s": len({r[1] for r in frames}) / duration,
        "requests": len(requests),
        "worker_wall_ms": percentiles([(int(r[7]) - int(r[6])) / 1e6 for r in requests]),
        "worker_thread_cpu_ms_per_crop": percentiles(
            [float(r[9]) / max(1, int(r[19])) for r in requests if len(r) > 19]
        ),
        "by_crop_count": {},
    }
    for count in sorted({int(r[19]) if len(r) > 19 else int(r[11]) for r in requests}):
        selected = [r for r in requests if (int(r[19]) if len(r) > 19 else int(r[11])) == count]
        out["by_crop_count"][str(count)] = {
            "requests": len(selected),
            "worker_wall_ms": percentiles([(int(r[7]) - int(r[6])) / 1e6 for r in selected]),
            "worker_thread_cpu_ms": percentiles([float(r[9]) for r in selected]),
            "runqueue_wait_ms": percentiles([float(r[10]) for r in selected]),
            "dsp_body_ms": percentiles(
                [float(r[20]) for r in selected if len(r) > 20 and float(r[20]) > 0]
            ),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("telemetry", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument("--warmup", type=float, default=3)
    a = p.parse_args()
    report = summarize(a.telemetry, a.warmup)
    text = json.dumps(report, indent=1) + "\n"
    if a.output:
        a.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
