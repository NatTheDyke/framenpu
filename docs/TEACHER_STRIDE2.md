# Teacher stride-2 layers inside the resident sequence

Status: hardware qualification passed on all four corpus crops. The tail now
runs in **one request, 15.80 ms median in-DSP**, versus 18.80 ms for the
previous seven-request linked run in [TEACHER_LINK.md](TEACHER_LINK.md).
Median host wall time is **26.69 ms**, previously 76.85 ms. Every compared
tensor is bit-identical to the unchanged per-layer qualification.

Evidence: [teacher-stride2-fp16.json](../evidence/teacher-stride2-fp16.json)
(statistics only).

The later [four-block sharing comparison](TEACHER_DW4.md) passed hardware
qualification but showed no speed gain. Two-block sharing remains the default.

## Hardware result

Four crops, five alternating rounds each: 20 seven-request cached-entry
control frames and 20 one-request merged linked frames. Each crop also has
an observed merged frame that checks all three stride-2 outputs and the
final tensor before timing.

| Median per tail | Cached control, this run | Linked, previous run | Merged linked, this run |
|---|---:|---:|---:|
| Requests | 7 | 7 | **1** |
| In-DSP, including wrapper | 30.02 ms | 18.80 ms | **15.80 ms** |
| Kernel body | 18.64 ms | 18.52 ms | **15.76 ms** |
| Host wall | 93.97 ms | 76.85 ms | **26.69 ms** |

The previous linked numbers are historical measurements, not an additional
paired control in this run. The current paired control uses the unchanged
4 KiB kernels and per-request code publication.

The merged in-DSP samples range from 15.73 to 15.94 ms (p95 15.88 ms), below
the existing 20 ms native budget. Host wall ranges from 24.99 to 48.05 ms
(p95 36.35 ms). This includes input staging and request overhead, and excludes
switching the benchmark's plan tables/parameters. The stem, pooling and heads
remain outside the timed tail; this is not a live tracking or thermal soak.

Accuracy and restoration:

- Tensors 19, 28, 58 and 79 match per-layer qualification byte for byte on all
  four observed crops. Final tensor 79 also matches in all 20 timing frames
  and the clean frame after controls.
- Qualification's worst FP16 layer-bound ratio remains 0.494. Head drift
  against ncnn remains unchanged: maximum landmark error 0.38544 px, world
  landmark absolute error 4.50e-4, presence delta 2.837e-3 and handedness
  delta 3.546e-4.
- Guarded fault, invalid parameter descriptor, 18 stride-2 descriptor
  rejections, second-seal refusal and unsealed-request refusal all pass.
- 359 restoration audits (357 HMX), identical whole-arena hashes before and
  after, reservation restored, identical power/firmware checks and unchanged
  DSP service with no restart. No retry, recovery or gate reset was used.

The three resident stride-2 descriptors now take median 1.698, 0.552 and
0.632 ms, including their parameter copy, tap splat and scalar decimation.
The two input loads and final store total about 0.741 ms per tail. The
remaining depthwise loop still shares coefficients across two blocks.

## Execution

The optional `--stride2` linked build adds descriptor kind 5 to opcode 111.
The three stride-2 layers (`convdwclip_3`, `convdwclip_5`, `convdwclip_11`)
now consume their predecessor's VTCM tensor and leave the half-resolution
result in VTCM for the next op. Their separate opcode-110 requests and the
activation transfers at those boundaries disappear.

The ordinary 4 KiB qualification kernels and the default linked image rebuild
byte-identical to the recorded baseline. The slot loader is also unchanged.
Only `tl_rs` in a `--stride2` image includes the extension; the build record
records that choice and the device harness requires a matching flag.

The first implementation keeps full-resolution depthwise computation. It
uses the resident depthwise routine, with the same tap accumulation order,
FP16 rounding, bias and clip6, writing to a full-resolution temporary grid.
The scalar decimator then copies words at `(2*y+1, 2*x+1)` into a zeroed
half-resolution grid. This preserves the qualified teacher's TF-SAME
sampling for both 3x3 and 5x5 kernels. Four-block coefficient sharing is now
selectable with `--dw-blocks 4`; a sampled-only kernel remains future work.

## Plans and placement

`tests/teacher_resident_layout.py` retains `qualify` and the seven-job
`resident` control. It adds:

| Plan | Requests | Descriptors | Window stores |
|---|---:|---:|---|
| `merged` | 1 | 45 | final tensor only |
| `merged-observe` | 1 | 48 | three stride-2 outputs and final tensor |
| `merged`, split after `convdwclip_5` | 2 | 47 | crossing tensor and final tensor |
| `merged-observe`, same split | 2 | 49 | three stride-2 outputs and final tensor |

The one-request plan has two input loads, 42 compute descriptors, and one
final store. Parameters are still staged from the owned DDR window for each
compute op. Intermediate activations remain in VTCM throughout the request.
An explicit split plans the required crossing stores and loads; it does not
assume VTCM contents survive between requests.

The linked image is 5,488 bytes with the current toolchain, leaving 10,768
bytes before the seal vector. Peak VTCM allocation, including the stride-2
temporary, is 14,099 vectors (1,804,672 bytes). The planner leaves 1,193
vectors free in addition to its four reserved vectors, within the existing
`0x31002000..0x311e0000` ownership span.

Kind 5 uses the existing descriptor words 0..14 plus:

| Word | Meaning |
|---|---|
| 15 | output grid vectors per channel pair |
| 16 | output interior block offset |
| 17 | output row stride in channel-pair words |
| 18 | output height |
| 19 | output width |
| 20 | full-resolution temporary offset, in VTCM vectors |

Native checks cover the input, compact output, temporary, parameters and tap
scratch, including every decimation address. The extended build explicitly
rejects table and parameter starts below the window before doing unsigned
range arithmetic. HMX operand alignment remains 2 KiB.

The 20 ms native budget and 40 ms completion timeout are unchanged. A request
that stops before all planned descriptors have run fails the harness. There
is no automatic retry, continuation, budget increase, or switch to two jobs.

## Validation

Offline checks cover:

- Numerical simulation of both merged variants and the two-request split
  against the per-layer contract, including all stored stride-2 outputs,
  final tensors and zero padding.
- Switching the actual harness between control, observed and merged plans
  in their shared DDR window, using a simulated submitter. The active op
  table and parameters are installed before execution.
- The actual C descriptor checker compiled for the host, tested against
  every planned descriptor and 4,000 seeded mutations, plus explicit invalid
  geometry, output, temporary, input and window bounds.
- Rebuilds of the frozen qualification and default linked images; relocation,
  vector-stack, writable-data, loader and linked-image checks.

```sh
FNPU_TEACHER_MODEL=/path/to/model python -m unittest discover -s tests -p 'test_teacher*.py' -v
python tools/build_teacher_link.py --stride2 --output /path/to/jobs
```

The device harness first qualifies with the unchanged per-layer kernels.
For each crop it runs an observed merged frame and requires all three
stride-2 outputs and the final tensor to match qualification byte for byte.
It then alternates the seven-request cached-entry control with the merged
linked plan, requiring an identical final tensor every frame. Plan installation
when switching modes is setup, outside the reported steady-frame wall time;
input staging and request submission remain timed.

Controls include the existing guarded fault, invalid parameter descriptor,
second-seal refusal, unsealed-request refusal and clean frame afterwards,
plus 18 rejections covering every extended field in the three stride-2 ops.
The existing per-request audits, image/seal checks and whole-arena restoration
remain in use. Only statistics are reported.

```sh
sudo .venv/bin/python tests/hardware_teacher_link.py --run-on-device --stride2 --jobs /path/to/jobs --model /path/to/model --corpus /path/to/corpus --output /path/to/result.json --rounds 5
```

For an explicitly planned two-request run, add
`--split-after convdwclip_5`. The two-request variant is checked offline;
only the one-request variant was exercised in this hardware run.
