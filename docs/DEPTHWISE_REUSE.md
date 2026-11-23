# Depthwise coefficient and mask reuse

The linked build can cache broadcast coefficient vectors and share product
classification work with `--dw-reuse 6`. It keeps the existing QFloat operations,
FP32 conversion boundaries, per-lane tap order, zero identities, tensor layout
and native unit/state protocol. The default build remains the qualified
baseline so that comparisons are reproducible.

## Implementation

Within a request, one channel's at-most-25 coefficients become FP32 broadcast
vectors once, then serve its successive output vectors. The 3,200-byte maximum
cache occupies saved VTCM offsets `0xc00..0x1880`. Five staged input rows end at
`0xa00`; aligned-load scratch starts at `0x1a00`. The tag at `0x1f30` is cleared
on every request and before any non-depthwise operator, including average
pooling, which may reuse part of that page. The complete cache stays inside
the existing saved/restored 8 KiB VTCM span.

Product magnitude is computed once and reused for zero and nonfinite
classification. Nonfinite predicates accumulate across taps and become the
existing vector flag at the end of the unit. Both zero corrections retain
their original order, including the product's sign when both operands are
zero. Product and final-sum checks remain active. The 40 ms host timeout,
20 ms measured request budget, one HMX owner, power policy, code reservation,
unit counting and fault boundaries are unchanged.

## Measured results

A sequential pilot uses the same calibration crop, three timing frames per
variant, and baseline controls before and after. The selected combined build
compares as follows:

| Measurement | Baseline | Broadcast + masks |
|---|---:|---:|
| Qualified depthwise native wrapper total | 457.07 ms | 420.95 ms |
| Median complete native wrapper time | 991.70 ms | 953.49 ms |
| Median timing-frame body | 1,352.80 ms | 1,269.98 ms |
| Requests per timing frame | 52 | 50 |

Depthwise time falls **7.9%** (1.086×); median frame time falls **6.1%**.
The subsequent six timing frames spanning the four-crop qualification take
1.260–1.319 seconds, all with 50 requests. This is a small sequential sample;
host variance affects frame latency, and sustained tracking is unqualified.

The first pilot also tried caching only converted scalar coefficients. It
saved about 4 ms of depthwise time. Masks alone were slightly slower. Broadcast
caching removes repeated scalar-to-vector work; the combined version was the
best observed variant. The earlier component windows did not predict removable
savings directly: about 103 ms attributed to coefficients translated into a
much smaller measured improvement. Separate-window scheduling/dependency costs
are not additive savings.

## Qualification

The selected build passes both calibration and both heldout crops under the
unchanged policy: 308 layer checks, 360 blobs and 20,351,360 layer-output values.
All 512 saved head values independently match the previous held-power baseline
bit for bit. Two sessions each load weights once across five changing/repeated
inputs. All 1,571 main restoration audits and independent full-arena checks pass.
Native HMX and in-place depthwise faults invalidate the session, refuse reuse
and restore cleanly (449/462 audits). Ordinary SDK regression passes 72 audits.
Maximum main wrapper time is 20.232 ms. Firmware, power configuration and the
DSP service restore unchanged with zero restarts.

A separate borrowed-arena fixture compares original and reused-mask arithmetic
bit for bit, including signed-zero identities. An independent integer oracle
checks NaN/Inf and zero classification over all IEEE categories plus seeded
raw words; predicate accumulation is checked with complemented inputs. All
4,096 lanes and 130 restoration audits pass. Offline tests independently cover
all accepted binary16 coefficient patterns during descending in-place expansion,
mask equivalence, VTCM bounds/overlap and explicit hardware opt-in. The suite
passes 188 tests. Disassembly shows no vector stack spills in depthwise.
The selected image is 11,572 bytes of 12,288; its loader remains 3,836 bytes.

Results, source/code hashes, pilot variants and limits are in
[depthwise-reuse-validation.json](../evidence/depthwise-reuse-validation.json).
Models, recordings and raw device captures remain private.

## Build and run

```sh
python tools/build_resident_landmark.py --linked --dw-reuse 6 --output /path/to/reuse-jobs
python tools/check_resident_link.py --jobs /path/to/reuse-jobs --output /path/to/verify.json
```

Use the same `ResidentLandmark` API and [linked qualification flow](RESIDENT_LINK.md)
with those jobs. Heldout requires a matching completed calibration report.
Start any new native image with `--stop-after convdwclip_1` before a full run.
The reported timing configuration uses `cached_code=True`, `budget_ms=20` and
opt-in `hold_power=True`; power coexistence/thermals remain unqualified.

Control flags are 0 (baseline), 1 (scalar cache), 2 (shared masks), 3 (scalar +
masks), 4 (broadcast cache), and 6 (broadcast + masks). Scalar and broadcast
caches are mutually exclusive. Component timer builds and reuse builds are
separate experiments. A zero flag reproduces the prior baseline native bytes.

Build the mask control fixture with the existing experimental-entry builder:

```sh
python handoff-tools/build_mw.py kernels/resident_graph_entry.S tests/resident_dw_reuse_probe.c /path/to/masks.fjob /path/to/masks-build
sudo .venv/bin/python tests/hardware_dw_reuse.py --run-on-device --probe /path/to/masks.fjob --output /path/to/masks.json
```
