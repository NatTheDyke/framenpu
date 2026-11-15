"""Opt-in cached/uncached instruction-fetch A/B with identical GEMM math and clocks.

The experimental FJOB publishes a cached alias from within the SDK guarded
window. Every FP16 tile and the last FP32 conversion are checked using the
unchanged baseline decoder. Recorded wrapper PC verifies the selected alias;
final disassembly must also establish that hot helpers use PC-relative calls.
Stock entry/host power settings and the existing 8 KiB VTCM span are unchanged.
Publication/cleanup remain in outer frame timing, outside internal HMX windows.
Defaults to uncached control only. Never performs automatic recovery.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from framenpu import Device, Job
from framenpu.errors import FrameNPUError
from framenpu.job import JobResult
from hardware_hmx import PowerSnapshot, health
from hmx_multi_run import failure_snapshot
import hmx_baseline_run as baseline

MODES = {"uncached": 0, "cached": 1}
BASES = {0: 0xF0A92000, 1: 0x87A92000}


def decode(result, depth, transactions, mode, expected):
    if len(result.output) < 256:
        raise ValueError("incomplete cached result")
    pc = struct.unpack_from("<I", result.output, 28)[0]
    if not BASES[mode] <= pc < BASES[mode] + 4096:
        raise ValueError(f"wrong execution alias: requested {mode}, observed PC {pc:#x}")
    output = bytearray(result.output)
    struct.pack_into("<I", output, 28, 0)
    sample = baseline.decode(
        JobResult(bytes(output), result.elapsed_ms, result.workers), depth, transactions, expected
    )
    sample.update(mode=mode, observed_wrapper_pc=pc)
    return sample


def one(probe, data, depth, transactions, mode, expected):
    session_start = time.monotonic_ns()
    transport = baseline.TimedTransport()
    with Device(
        dispatch="direct",
        _transport=transport,
        _power_factory=lambda: baseline.TimedPower(transport.current),
    ) as device:
        start = time.monotonic_ns()
        result = device.submit(
            probe,
            data,
            output_size=256 + transactions * 2048 + 4096,
            opcode=9,
            params=(depth, transactions, mode),
        )
        submit_ms = (time.monotonic_ns() - start) / 1e6
    session_ms = (time.monotonic_ns() - session_start) / 1e6
    sample = decode(result, depth, transactions, mode, expected)
    if len(transport.benchmarks) != 1:
        raise ValueError("ambiguous timing trace")
    trace = transport.benchmarks[0]
    sample["host_phase_ms"] = {k: v for k, v in trace.items() if k.endswith("_ms")}
    sample["powered_configuration"] = trace["powered_configuration"]
    sample["submit_wall_ms"], sample["session_wall_ms"] = submit_ms, session_ms
    return sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-on-device", action="store_true")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=["uncached"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--transactions", type=int, default=4)
    parser.add_argument("--cases", choices=baseline.CASES, nargs="+", default=list(baseline.CASES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.run_on_device:
        parser.error("--run-on-device is required; this publishes cached DSP code and powers HMX")
    if not 1 <= args.samples <= 100 or not 0 <= args.warmup <= 10:
        parser.error("require 1..100 samples and 0..10 warmups")
    if not 1 <= args.transactions <= 4 or any(not 1 <= n <= 512 for n in args.depths):
        parser.error("require 1..4 transactions and depths in 1..512")
    if os.geteuid() != 0:
        parser.error("run with sudo for /dev/mem access")
    probe = Job.load(args.probe)
    if not probe.requires_hmx or not probe.supports_direct:
        parser.error("probe requires HMX and direct-entry flags")
    operands = {case: baseline.operands(case) for case in dict.fromkeys(args.cases)}
    references = {
        (case, depth): baseline.reference(operands[case][1], depth)
        for case in operands
        for depth in sorted(set(args.depths))
    }
    selections = [
        (case, depth, mode) for case, depth in references for mode in dict.fromkeys(args.modes)
    ]
    report = {
        "ref_hz": baseline.REF_HZ,
        "code_bytes": len(probe.code),
        "code_sha256": hashlib.sha256(probe.code).hexdigest(),
        "samples_per_point": args.samples,
        "warmups_per_point": args.warmup,
        "mode_names": MODES,
        "samples": [],
        "checked_warmup_jobs": 0,
        "timing_scope": "internal baseline HMX windows exclude guarded cache publication and cleanup",
    }
    failed = False
    rng = random.Random(0x43414348)
    with PowerSnapshot() as power:
        expected_power = power.read()
        try:
            report["initial_health"] = health(power, expected_power)
            for index in range(args.samples + args.warmup):
                order = list(selections)
                rng.shuffle(order)
                for case, depth, mode in order:
                    sample = one(
                        probe,
                        operands[case][0],
                        depth,
                        args.transactions,
                        MODES[mode],
                        references[(case, depth)],
                    )
                    if health(power, expected_power) != report["initial_health"]:
                        raise ValueError("post-session restoration differs")
                    sample.update(
                        case=f"{case}:{mode}", input_case=case, sample_index=index - args.warmup
                    )
                    if index < args.warmup:
                        report["checked_warmup_jobs"] += 1
                    else:
                        report["samples"].append(sample)
                if index == args.warmup - 1 or (index - args.warmup + 1) % 10 == 0:
                    print(
                        f"cached A/B: {len(report['samples'])} measured jobs",
                        file=sys.stderr,
                        flush=True,
                    )
            report["summary"] = baseline.summarize(report["samples"])
            report["final_health"] = health(power, expected_power)
            report["all_outputs_match"] = True
        except (FrameNPUError, ValueError, OSError) as exc:
            failed = True
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["after_failure"] = failure_snapshot(power)
            report["interpretation"] = "inconclusive; no recovery attempted"
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(
            json.dumps(
                {
                    "report": str(args.output),
                    "measured_jobs": len(report["samples"]),
                    "all_outputs_match": report.get("all_outputs_match", False),
                    "error": report.get("error"),
                    "resident_slopes": report.get("summary", {}).get("resident_slopes"),
                },
                indent=2,
            )
        )
    else:
        print(encoded, end="")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
