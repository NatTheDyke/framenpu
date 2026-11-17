# Resident depthwise component timing

The linked depthwise kernel has opt-in timing builds for row/tap staging,
gathering, coefficient conversion and arithmetic, final checks/clip and stores,
state advance/sync, and the whole unit. Profiling disabled reproduces the
qualified loader and operator image byte for byte.

One component is timed per build. Request-local accumulated ticks and segment
counts occupy unused saved VTCM at `0x31001f20` through `0x31001f2b`. They are
returned through four reserved descriptor-echo words, decoded by the dedicated
harness and reset to their expected zeros before ordinary completion validation.
Native state, numerical, guard, alias and cleanup checks still run. Use these
images with the dedicated profiling harness; the ordinary qualifier expects
an unmodified descriptor echo.

## Results

The same recorded calibration crop runs through each of six variants. Every
variant first passes early layers through `convdwclip_1`, then all 77 layer
checks and three timing frames. Baseline controls run before and after the
variants. All saved heads independently match the existing held-power baseline
bit for bit. The final matrix passes 8,392 restoration audits, full-arena
restoration and unchanged firmware/power/service state.

| Component | Median ms per crop | Timed interval |
|---|---:|---|
| Staging | 12.25 | Tap staging and aligned input-row copies |
| Gathering | 135.39 | Gather addressing, bounds, loads, stride selection and masks |
| Arithmetic | 246.71 | Coefficient load/conversion, multiply, product checks and ordered add |
| Storing | 32.17 | Final nonfinite check, clip and output store |
| Bookkeeping | 10.11 | State counters, next row and per-unit state sync |
| Whole depthwise unit | 447.55 | Unit setup through state sync, excluding done/expiry checks |

The five component medians sum to 436.63 ms, about 97.6% of the whole-unit
measurement. These come from different compiled variants and must be treated
as estimates, rather than an exact partition of one instruction schedule.
Baseline depthwise wrappers total 457.18–457.53 ms in the qualified frames;
that includes entry/return and work outside the whole-unit interval.

Staging is roughly 3% of the whole-unit time. Arithmetic including coefficient
handling accounts for about 55%, gathering about 30%, and storing about 7%.
The next useful investigation is coefficient reuse/conversion and the ordered
FP32 arithmetic/zero handling, alongside gather address/mask work. Faster row
copies alone have limited headroom. Any proposed kernel change still needs
bounds checks and full numerical qualification.

## Instrumentation effects

Gather and arithmetic builds add about 7.4% and 7.2% to complete native wrapper
time compared with the median baseline timing frames. Other variants add
about 0.2–1.0%. This compares all native work, not just depthwise. Timers,
counter updates and changed scheduling contribute to that difference.
Compiler fences constrain vector work across timer boundaries; timestamp reads
do not imply a hardware pipeline-drain barrier. Loop control, initial bias,
exit checks and counter-update overhead are outside some intervals.

An initial arithmetic build used read-write vector fences. It passed the
frozen numerical limits but changed output bits. The final arithmetic build
uses input-only observations, repeats early/full validation and matches the
baseline bit for bit. Superseded arithmetic measurements are excluded.

Full component samples, per-layer rows, segment counts, image/source hashes,
overhead measurements and caveats are in
[depthwise-profile.json](../evidence/depthwise-profile.json). This is one crop
on one device and firmware; it does not establish broader accuracy, thermals
or sustained tracking performance.

## Reproduction

For each mode from one through six:

```sh
python tools/build_resident_landmark.py --linked --profile-dw 1 --output /path/to/profile1
python tools/check_resident_link.py --jobs /path/to/profile1 --output /path/to/verify1.json
```

After deploying jobs/source and preparing private references as in
[LANDMARK.md](LANDMARK.md), run the dedicated harness as root. Start with
`--stop-after convdwclip_1` before a full run of any new image:

```sh
sudo .venv/bin/python tests/hardware_depthwise_profile.py --mode 1 --profile-output /path/to/counters1.json --run-on-device --jobs /path/to/profile1 --linked --cached-code --hold-power --budget-ms 20 --model /private/model --corpus /private/corpus --crop calibration_controller --output /path/to/checked1.json --unqualified-frames 3
```

Modes 1/2/3/4/5/6 select staging/gather/arithmetic/store/bookkeeping/whole-unit.
The existing timeout, request budget and single-owner rules apply. Held power
retains its existing opt-in status and coexistence limitations.

The [arithmetic investigation](DEPTHWISE_ARITHMETIC.md) further separates the
coefficient, QFloat and check work. Its full-crop matrix is complete; a separate
batched control remains incomplete. See its device status before further work.
