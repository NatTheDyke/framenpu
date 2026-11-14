"""Synchronous custom jobs and HMX operations. Importing opens no device."""

import hashlib
from importlib.resources import files
import math
import struct
import threading
import time
from .errors import JobFailed, RecoveryRequired
from .job import Job, JobResult
from .power import HmxPower
from .profile import PROFILE, JOB_MAGIC, JOB_ABI, HMX
from .transport import Transport


def _matrix(value, *, fp16=True):
    rows = [list(row) for row in value]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be nonempty and rectangular")
    if len(rows) > 256 or len(rows[0]) > 256:
        raise ValueError("version 0.1 supports each matrix dimension up to 256")
    if any(
        not math.isfinite(float(x)) or abs(float(x)) > (65504 if fp16 else 3.402823466e38)
        for row in rows
        for x in row
    ):
        raise ValueError("matrix elements must be finite and representable in FP16")
    return rows


def _fp32(values):
    result = [float(v) for v in values]
    if any(not math.isfinite(v) or abs(v) > 3.402823466e38 for v in result):
        raise ValueError("values must be finite FP32 numbers")
    return struct.pack("<%df" % len(result), *result)


class Device:
    def __init__(
        self, *, timeout=0.04, claim_timeout=0.5, _transport=None, _power_factory=HmxPower
    ):
        if not math.isfinite(timeout) or not 0 < timeout <= 0.04:
            raise ValueError(
                "timeout must be positive and at most 0.04 seconds (stock completion window)"
            )
        if not math.isfinite(claim_timeout) or not 0 < claim_timeout <= 30:
            raise ValueError(
                "claim_timeout must be positive and at most 0.04 seconds (stock completion window)"
            )
        self.timeout = timeout
        self.claim_timeout = claim_timeout
        self._transport = _transport
        self._power_factory = _power_factory
        self._lock = threading.RLock()
        self._sequence = time.monotonic_ns() & 0xFFFFFFFF or 1
        self._closed = False

    def __enter__(self):
        with self._lock:
            self._connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def _connect(self):
        if self._closed:
            raise RuntimeError("device is closed")
        if self._transport is None:
            self._transport = Transport()
        if self._transport.poisoned:
            raise RecoveryRequired("device session is poisoned")
        return self._transport

    def close(self):
        with self._lock:
            if self._transport:
                self._transport.close()
            self._closed = True

    def status(self):
        with self._lock:
            return self._connect().status()

    @staticmethod
    def builtin(*, hmx):
        resource = files("framenpu").joinpath("data")
        import json

        manifest = json.loads(resource.joinpath("runtime.json").read_text())
        code = resource.joinpath("runtime.bin").read_bytes()
        if (
            manifest.get("abi") != JOB_ABI
            or manifest.get("target") != "hexagonv75"
            or manifest.get("bytes") != len(code)
            or hashlib.sha256(code).hexdigest() != manifest.get("sha256")
        ):
            raise RuntimeError("packaged DSP kernel checksum mismatch; rebuild kernels")
        return Job(code, "framenpu-runtime", hmx)

    def submit(self, job, input_data=b"", *, output_size=0, opcode=0, params=()):
        """Run one trusted FJOB/C kernel using ABI-relative arena offsets."""
        if not isinstance(job, Job):
            raise TypeError("job must be a Job")
        input_data = bytes(input_data)
        if type(output_size) is not int or output_size < 0:
            raise ValueError("output_size must be a nonnegative integer")
        if type(opcode) is not int or not 0 <= opcode <= 0xFFFFFFFF:
            raise ValueError("invalid opcode")
        params = tuple(params)
        if len(params) > 6 or any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in params):
            raise ValueError("params must be at most six unsigned 32-bit words")
        p = PROFILE
        output_offset = (len(input_data) + 127) & ~127
        used = output_offset + output_size
        if used > p.arena_size:
            raise ValueError("input and output exceed the 128 KiB job arena")
        with self._lock:
            t = self._connect()
            t.claim(self.claim_timeout)
            power = None
            backup = []
            published = False
            returned = False
            try:
                spans = [(p.code, p.code_size), (p.header, 256), (p.status, 192), (p.arena, used)]
                if job.requires_hmx:
                    spans.append((p.backup, 8192))
                backup = [(off, bytes(t.memory[off : off + length])) for off, length in spans]
                self._sequence = (self._sequence + 1) & 0xFFFFFFFF or 1
                seq = self._sequence
                words = [
                    seq,
                    JOB_MAGIC,
                    JOB_ABI,
                    HMX if job.requires_hmx else 0,
                    used,
                    0,
                    len(input_data),
                    output_offset,
                    output_size,
                    opcode,
                    *params,
                    *([0] * (6 - len(params))),
                ]
                t.memory[p.code : p.code + p.code_size] = job.code.ljust(p.code_size, b"\0")
                t.memory[p.header : p.header + 256] = bytes(256)
                struct.pack_into("<16I", t.memory, p.header, *words)
                t.memory[p.status : p.status + 192] = bytes(192)
                t.memory[p.arena : p.arena + used] = input_data.ljust(used, b"\0")
                if job.requires_hmx:
                    power = self._power_factory()
                    power.start()
                started = time.monotonic()
                # Mark before publishing: an interrupted send may already have woken the DSP.
                published = True
                t.publish()
                t.wait_complete(seq, self.timeout)
                returned = True
                statuses = tuple(
                    struct.unpack_from("<16I", t.memory, p.status + 64 * i) for i in range(3)
                )
                output = bytes(t.memory[p.arena + output_offset : p.arena + used])
                result = JobResult(output, (time.monotonic() - started) * 1000, statuses)
                t.put(p.callback, 0)
                if power is not None:
                    power.close()
                    power = None
                for off, saved in backup:
                    t.memory[off : off + len(saved)] = saved
                backup = []
                t.acknowledge(self.timeout)
                failed = [(row[1], row[2], row[3:6]) for row in statuses if row[2]]
                if failed:
                    raise JobFailed(
                        "DSP kernel failed (worker, error, fault registers): %r" % failed
                    )
                return result
            except BaseException as exc:
                if (published and not returned) or isinstance(exc, RecoveryRequired):
                    t.poisoned = True
                    if power is not None:
                        power.abandon()
                    raise RecoveryRequired(
                        "DSP state requires recovery; payload and clocks were retained, but the stock operation gate may expire"
                    ) from exc
                if power is not None:
                    try:
                        power.close()
                    except BaseException as cleanup:
                        t.poisoned = True
                        raise RecoveryRequired(
                            "HMX cleanup failed; operation gate retained"
                        ) from cleanup
                for off, saved in backup:
                    t.memory[off : off + len(saved)] = saved
                t.release_unsubmitted()
                raise

    def _matmul_tile(self, a, b, *, bias=None, activation="none"):
        """FP16 tile products accumulated in FP32 on DSP; returns row-major lists."""
        a, b = _matrix(a), _matrix(b)
        m, k, n = len(a), len(a[0]), len(b[0])
        if k != len(b):
            raise ValueError("matmul inner dimensions differ")
        if activation not in ("none", "relu"):
            raise ValueError("supported fused activations: none, relu")
        bias = [0.0] * n if bias is None else list(bias)
        if len(bias) != n:
            raise ValueError("bias length must equal output columns")
        data = struct.pack("<%de" % (m * k), *(v for row in a for v in row))
        data += struct.pack("<%de" % (k * n), *(v for row in b for v in row))
        data = data.ljust((len(data) + 3) & ~3, b"\0") + _fp32(bias)
        output = self.submit(
            self.builtin(hmx=True),
            data,
            output_size=m * n * 4,
            opcode=1,
            params=(m, k, n, int(activation == "relu")),
        ).output
        values = struct.unpack("<%df" % (m * n), output)
        return [list(values[i * n : (i + 1) * n]) for i in range(m)]

    def elementwise(self, rows, *, activation="relu", minimum=0.0, maximum=6.0):
        rows = _matrix(rows, fp16=False)
        if activation not in ("relu", "clip"):
            raise ValueError("supported elementwise operations: relu, clip")
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            raise ValueError("invalid clipping bounds")
        flat = [v for row in rows for v in row]
        data = _fp32(flat) + _fp32([minimum, maximum])
        out = self.submit(
            self.builtin(hmx=False),
            data,
            output_size=len(flat) * 4,
            opcode=2,
            params=(len(flat), int(activation == "clip")),
        ).output
        values = struct.unpack("<%df" % len(flat), out)
        n = len(rows[0])
        return [list(values[i * n : (i + 1) * n]) for i in range(len(rows))]

    def add(self, a, b, *, activation="none"):
        """DSP FP32 addition; one-row b broadcasts across all rows of a."""
        a, b = _matrix(a, fp16=False), _matrix(b, fp16=False)
        if len(a[0]) != len(b[0]) or len(b) not in (1, len(a)):
            raise ValueError("add requires equal shapes or a single broadcast row")
        if activation not in ("none", "relu"):
            raise ValueError("supported fused activations: none, relu")
        av, bv = [v for row in a for v in row], [v for row in b for v in row]
        raw = self.submit(
            self.builtin(hmx=False),
            _fp32(av + bv),
            output_size=len(av) * 4,
            opcode=3,
            params=(len(av), len(bv), int(activation == "relu")),
        ).output
        vals = struct.unpack("<%df" % len(av), raw)
        n = len(a[0])
        return [list(vals[i * n : (i + 1) * n]) for i in range(len(a))]

    def matmul(self, a, b, *, bias=None, activation="none"):
        """Schedule depth-32 HMX products and DSP FP32 accumulation as bounded jobs."""
        a, b = _matrix(a), _matrix(b)
        m, k, n = len(a), len(a[0]), len(b[0])
        if k != len(b):
            raise ValueError("matmul inner dimensions differ")
        if activation not in ("none", "relu"):
            raise ValueError("supported fused activations: none, relu")
        bias = [0.0] * n if bias is None else list(bias)
        if len(bias) != n:
            raise ValueError("bias length must equal output columns")
        _fp32(bias)
        out = [[0.0] * n for _ in range(m)]
        for mr in range(0, m, 32):
            for nc in range(0, n, 32):
                acc = None
                for kb in range(0, k, 32):
                    aa = [row[kb : kb + 32] for row in a[mr : mr + 32]]
                    bb = [row[nc : nc + 32] for row in b[kb : kb + 32]]
                    partial = self._matmul_tile(
                        aa,
                        bb,
                        bias=bias[nc : nc + 32] if k <= 32 else None,
                        activation=activation if k <= 32 else "none",
                    )
                    acc = partial if acc is None else self.add(acc, partial)
                if k > 32:
                    acc = self.add(acc, [bias[nc : nc + 32]], activation=activation)
                for r, row in enumerate(acc):
                    out[mr + r][nc : nc + len(row)] = row
        return out
