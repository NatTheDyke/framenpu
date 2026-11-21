# Mercury tracker integration

The [steam-frame-hand-tracking integration patch](../integrations/steam-frame-hand-tracking.patch)
adds pluggable inference to spyjacob's existing Mercury tracker. The source base
is upstream commit `8852ec9bdd4f81fb5a359e5bf88ed25f0e6fc89e`.
CPU remains the default and fallback. The patch changes inference and its worker
lifetime; crop projection, normalization, hints, decoding, discovery and the hand
fit retain their existing behavior.

The patch adds `handtrack.inference`, an injectable `backend=` on
`MercuryKeypoint`, `run_many()`, worker configuration, idle/close hooks,
`--merc-inference cpu|framenpu`, and `--framenpu-jobs`. A request carries the
available one to three crops immediately. It never pads a request or accumulates
observations while waiting for a third crop. Longer lists are split into groups
of three and a final smaller request. Existing tracker rate and thermal caps
remain in effect. Shadow comparisons continue on the CPU worker.

## Apply and use

Apply the patch to an upstream tracker checkout at the base above:

```sh
git apply /path/to/framenpu/integrations/steam-frame-hand-tracking.patch
```

Install this FrameNPU package in the tracker's Python environment and build the
[qualified replay image](MERCURY.md#one-to-three-crops-per-request):

```sh
python -m pip install /path/to/framenpu
python /path/to/framenpu/tools/build_teacher_link.py --stride2 --s2-even \
  --s2-direct --pw-fast --dw-fast --pw-group 8 --replay --output /tmp/mercury-jobs
```

Deploy that build's `link.json`, `teacher_link.bin` and
`teacher_link_loader.fjob` together. The runtime accepts only the qualified image
`d7a78a81…` and loader `de3cc52a…`, and the pinned Mercury model `40c0daa5…`.
No weights are redistributed. In the patched tracker, select:

```sh
python -m handtrack.live --backend mercury --merc-inference framenpu \
  --framenpu-jobs /path/to/mercury-jobs
```

The DSP currently requires root access. Without device access or on unsupported
firmware, the worker logs the reason and falls back to CPU. The patched
tracker's `docs/FRAMENPU.md` includes the privileged invocation that preserves
the user's existing model/calibration paths, installation details and tests.
`--merc-inference cpu` uses the original ONNX Runtime execution provider and
settings without importing FrameNPU.

## Packaged session

Applications can use the installable API directly. Its inputs are normalized
ONNX feed dictionaries; it does not own the camera crop or decode step:

```python
from framenpu.mercury_runtime import MercurySession

with MercurySession(model_path, jobs_directory) as session:
    outputs = session.infer(available_feeds)  # actual length 1, 2 or 3
    session.idle()  # release HMX while tracking is idle
    outputs = session.infer(next_feeds)  # resumes; no accumulation timer
```

The planner and grid/descriptor helpers are packaged with FrameNPU and tested
byte-for-byte against the qualified planner; applications do not import test
modules or need a compiler at runtime. The session uploads parameters once,
seals the linked image, and owns the shared window through the existing
exclusive lease and a persistent ownership record. Input writes, completion
checks and output readback occur under the device gate. Replay end times, total
descriptor count, image/seal, worker status, output padding and finite values
are checked before acknowledging completion. `last_timing` records crop count,
cumulative crop ends, individual crop body times and total body time.

Power is held between observations. `idle()` releases it without discarding the
plan. `close()` releases power, retires the image and restores the borrowed
window, reservation and ownership record. The tracker invokes idle when no
observations remain or thermal inference pauses, and closes its worker before
pool shutdown. Its stop script allows this cleanup before terminating leftover
workers.

An accelerator failure is latched to CPU for the worker's lifetime. The whole
affected group is rerun on CPU, preserving result order. An uncertain DSP
completion leaves the refusal record, clocks and device bytes as required by
FrameNPU. There is no DSP retry, gate clear, service restart or automatic
recovery. A forced process kill cannot run cleanup.

## Validation and limits

[Device evidence](../evidence/mercury-runtime-device.json): 16 requests with
counts cycling 1, 2, 3 (31 crops), each feature tensor identical to its case's
prior replay hash. Hint staging is checked against the qualified private cases
before submitting. Idle releases power halfway through; the next request
resumes. All 20 restoration audits pass, and the full arena hash, power,
firmware and DSP service are unchanged after close.

Offline checks cover packaged table/parameter/slot parity, per-crop metadata
corruption, mixed counts, simulator outputs, idle/resume, complete restoration
and refusal retention after uncertain completion. The tracker tests cover
small immediate submissions, 3+1 splitting, blank crops, camera/side/hint
identity, CPU default, CPU output parity, geometry/decode parity and permanent
fallback after initialization, inference, output-validation or idle errors.
A direct comparison with the original upstream `MercuryKeypoint.run()` also
produced exactly equal crops and decoded results on four synthetic
observations spanning both cameras, flips and hint settings.

This validates the session on hardware and the tracker integration offline.
A subsequent [live camera comparison](MERCURY_LIVE.md) measured worker CPU
savings and a modest two-crop processing gain, while overall throughput did not
improve consistently. Short live coexistence passed; sustained thermals and
PC-rendering latency remain unqualified. The heads still run on the host and
CPU remains the default.
