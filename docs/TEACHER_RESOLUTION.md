# Crop resolution experiment

Feeding the unchanged teacher **128x128 input is a substantial regression**
on the existing crop set. On `calibration_open`, the only crop accepted by
the 224 model at a presence threshold of 0.5, presence falls from **0.999938
to 0.255875**. Its 21 landmarks move by **28.50 pixels on average**, maximum
**57.74 pixels**, relative to the 224 output in the original coordinate units.

These are **prediction differences from the 224 teacher**, not measured
ground-truth accuracy loss. The four existing crops have no annotated
keypoints, and only one is presence-positive at 224. This experiment cannot
establish a dataset-wide accuracy percentage.
[Evidence](../evidence/teacher-resolution-drift.json) preserves all crop
statistics, alternative coordinate scaling, confidence values and source/build
hashes without publishing images or landmark coordinates.

## Same accepted hand, different network input sizes

| Network input | Mean landmark drift | Maximum drift | Presence |
|---|---:|---:|---:|
| 224x224 | 0 px, reference | 0 px | 0.999938 |
| 192x192 | 5.36 px | 8.95 px | 0.999929 |
| 160x160 | 9.19 px | 21.21 px | 0.999720 |
| 128x128 | 28.50 px | 57.74 px | **0.255875** |

All rows use the same grayscale source frame and affine hand region,
recropped directly at the requested resolution. The weights are unchanged.
The final 672-channel feature map changes from 7x7 to 6x6, 5x5 and 4x4,
respectively. Global-average pooling permits all these sizes to reach the
same output heads, but successful execution does not preserve predictions.

The table retains the teacher's original 224-trained head coordinate units.
Multiplying the 128 outputs by 224/128 does not repair the result: mean
drift on the accepted hand becomes **90.96 px**, maximum **151.55 px**, and
presence remains 0.255875. The evidence includes this alternative for every
target size. A decoder change alone does not resolve the observed regression.

Across all four crops, native 128 outputs have mean displacement 73.74 px
and maximum 145.89 px. Three of those crops are presence-negative at 224,
so their regressed landmark positions are not an appropriate headline for
accepted-hand accuracy. The only presence classification change is the
accepted open hand becoming rejected. One background crop's handedness
threshold also changes; that is not a demonstrated handedness accuracy loss.

## Lower-detail input with the network kept at 224

As a control, the experiment recrops at 128, bilinearly resizes that crop
back to 224, and runs the original 224 graph. On the accepted hand this gives
**1.85 px mean drift**, **4.39 px maximum**, and presence **0.999963**.

This control has additional resampling and is not an exact decomposition of
all sources of error. It does show that changing the network's spatial size
has a much larger effect here than this particular reduction in input detail.
Upscaling keeps the network's 224 compute cost and therefore does not test
the intended DSP compute saving.

## Validation and reproducibility

`tools/teacher_resolution_accuracy.py` builds a CPU bridge against the cached,
hash-pinned strict ncnn reference. FP16 arithmetic/storage, packing, BF16,
Winograd, SGEMM and Vulkan are disabled, matching the original exporter.
Grayscale replication, affine geometry, zero border and FP32 normalization
match `ncnn_reference.cpp`. The input comes from the original raw camera
frames, rather than resizing an already sampled 224 crop for the native rows.

For each crop, the runner requires recreated 224 normalized inputs and all
four output heads to match the frozen oracle **byte for byte** before
reporting lower-resolution differences. Corpus/model hashes, crop recipe
provenance, the strict library hash and bridge source hash are checked or
recorded. Repeated runs produce identical metrics at every tested size.
Private frames and intermediate/head arrays stay local; only statistics
are written to the report. No DSP jobs, firmware or device configuration
changes are involved, and no 128 DSP timing is claimed.

```sh
OPENBLAS_NUM_THREADS=1 python tools/teacher_resolution_accuracy.py \
  --reference /path/to/strict-ncnn-reference \
  --model /path/to/model --source-corpus /path/to/convolution-corpus \
  --oracle-corpus /path/to/landmark-corpus --target 128 \
  --output /path/to/statistics.json
```

The optional source corpora and strict ncnn build are private/external inputs.
Targets 160 and 192 use the same command. Five new offline tests check
coordinate scaling, probability-head mapping, confidence-based grouping,
refusal of an altered 224 control, and the smaller graph's final/head shapes.
The offline suite passes 299 tests.

The present evidence does not support switching the unchanged teacher to
128. Further work needs a broader labeled evaluation and, if 128 remains the
target, a model adapted to that resolution. The DSP geometry should be
changed only after the candidate model's output quality is established.
