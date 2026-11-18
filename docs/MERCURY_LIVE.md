# Live camera comparison

The FrameNPU integration has a measurable **inference-worker CPU saving** and
a modest two-crop processing-time improvement. These short trials do **not**
establish a consistent overall tracking-throughput improvement. CPU remains
the default.

[Evidence](../evidence/mercury-live-validation.json), measured on the Frame on
2026-11-17 with live stereo passthrough cameras and the user moving their hands.

| Median, per request | CPU | FrameNPU | Change |
|---|---:|---:|---:|
| One crop: crop + inference + decode wall | 17.87 ms | 18.48 ms | 0.60 ms slower |
| Two crops: crop + inference + decode wall | 34.09 ms | 31.23 ms | **8.4% lower** |
| Two crops: inference-worker thread CPU | 29.66 ms | 17.76 ms | **40.1% lower** |
| Two crops: DSP body | — | 10.05 ms | about 5 ms/crop |

These are pooled measurements from the complete pipeline, stratified by the
actual number of crops in each request: 388 CPU / 310 NPU two-crop requests,
and 124 CPU / 247 NPU one-crop requests after warmup. Worker wall time includes
crop geometry, normalization, hint work, inference, host heads and decoding.
Thread CPU time belongs to the Mercury inference worker, not the entire tracker
or headset. FrameNPU still spends about 21 ms of a two-crop request outside
the DSP body; host work, dispatch and scheduling remain significant.

## Complete pipeline

Four 18-second runs in **CPU / NPU / NPU / CPU** order used the same tracker
configuration: `--backend mercury --merc-early --flow-v2`, 30 Hz Mercury cap,
existing discovery and thermal caps, OpenCV/BLAS single-threaded, affinity 0–4,
SCHED_IDLE and nice 10. The first 3 seconds after tracked frames begin are
excluded from each run. The UDP bridge and HTTP viewer were disabled; the
tracker still computed local palm/hand poses.

| Pooled result after warmup | CPU | FrameNPU |
|---|---:|---:|
| Observations | 900 | 867 |
| Accepted observations/s | 32.45 | 31.04 |
| Camera detected → Mercury collect: median | 74.93 ms | 72.07 ms |
| Camera detected → Mercury collect: p95 | 107.99 ms | 98.97 ms |
| Fast path frames/s | 52.82 | 52.33 |

The latency medians differ by about 3 ms, while accepted-observation rate did
not improve. The two runs per backend ranged from 30.5–34.4 accepted crops/s
on CPU and 28.0–34.1 on NPU. Hand visibility, movement and the mix of one/two
crops varied, so these pipeline totals do not isolate an accelerator effect.
This is not motion-to-photon latency: timestamps start when `ptgrab` detects a
camera-buffer update and end at local result collection; no PC rendering was
measured.

Temperatures reached 84.8–88.8 °C across runs. Existing thermal policies were
retained, and the 92 °C pause/stop threshold was not reached. The duration and
changing workloads do not establish a temperature or sustained-power benefit.

## Identical live inputs

Before the pipeline trials, both backends processed the **same** current live
observations, alternating execution order. Thirty groups contained 53 crops.
Requests used the observations that existed immediately: 10 groups of one,
17 of two, 3 of three. There was no padding or wait for a third crop.

Discarding the first three groups for warmup leaves 9 / 16 / 2 samples for
counts 1 / 2 / 3. The median matched CPU-minus-NPU wall difference was 0.32 ms
for one crop and 3.15 ms for two crops. Three-crop samples are too few for a
performance conclusion. The paired test corroborates lower worker CPU cost;
it does not include worker IPC, optical flow or discovery costs.

Both paths used the same crop and decoder. Maximum keypoint displacement was
0.52 detector pixels in the 640×640 camera view, and none of the paired
observations changed the `is_hand >= 0.97` decision. This compares numerical
outputs, not ground-truth hand accuracy.

## Changes and checks

Live testing exposed an existing `--no-bridge` behavior that removed the palm
tracker needed by Mercury. Mercury now keeps that local state while disabling
UDP output. Telemetry records the actual backend, request crop count and DSP
body time, so silent CPU fallback cannot be mistaken for a DSP result.

Both NPU pipeline runs used FrameNPU throughout. Every run exited normally;
there was no recovery, DSP retry, gate clear or DSP service restart. The arena
hash, gate, callback, firmware/power configuration and DSP service matched the
qualified baseline after testing. Camera capture was stopped and the test's
activation flag removed. Factory calibration, recordings and raw pose/camera
records remain private on the Frame.

The camera path used upstream's passive Vulkan registry layer and `ptgrab`.
Installing the layer required one normal SteamVR restart before the trials;
`dsp_service` remained at the same PID with zero restarts. No existing tracker
installation was overwritten.

## Reproduce

Use the [integration setup](MERCURY_INTEGRATION.md), private factory calibration
and the upstream capture tools. The following commands run **on the Frame** in
the tracker environment, with root device access and the user's HOME/model
paths preserved as described in the tracker setup guide:

```sh
# Capture must already be running. No camera or pose records are exported.
python /path/to/framenpu/tools/mercury_live_bench.py \
  --tracker /path/to/patched-tracker --jobs /path/to/replay-jobs \
  --rounds 30 --seconds 25 --output /tmp/paired.json
python /path/to/framenpu/tools/mercury_live_compare.py \
  --tracker /path/to/patched-tracker --jobs /path/to/replay-jobs \
  --directory /tmp/private-live-test --seconds 18 --warmup 3 --output /tmp/abba.json
# Merge raw telemetry locally; output includes only aggregate metrics.
python /path/to/framenpu/tools/mercury_live_report.py \
  /tmp/private-live-test/abba-0-cpu.csv /tmp/private-live-test/abba-3-cpu.csv \
  --output /tmp/cpu-combined.json
```

Keep the full configuration and crop counts with timing reports. Next work
should target host processing/dispatch and establish a repeatable end-to-end
benefit before changing the default backend.
