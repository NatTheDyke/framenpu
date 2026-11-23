# Teacher tail through a linked operator image

The optional resident stride-2 extension is described in
[TEACHER_STRIDE2.md](TEACHER_STRIDE2.md). It is hardware-qualified at one
request and 15.80 ms in-DSP; the results below remain the seven-request baseline.

This follows [TEACHER_RESIDENT.md](TEACHER_RESIDENT.md). That run cut the tail
from 42 jobs to 7, but every job still paid about 1.6 ms of in-DSP overhead.
Most of that overhead is publishing the 4 KiB kernel through the cached code
alias and cleaning it up again on each request
([RESIDENT_CACHE.md](RESIDENT_CACHE.md) measured 0.07 ms growing to 1.6 ms).

Here, the same two kernels run from one operator image sealed in the owned
DDR reservation used by [RESIDENT_LINK.md](RESIDENT_LINK.md). A small loader
in the code slot enters the image without publishing anything.

**Result:** every check passed. Every linked frame was bit-identical to the
per-layer qualification run. In-DSP time per frame fell from 30.0 ms to
**18.8 ms**. Per-job overhead fell from about 1.6 ms to **0.03–0.06 ms**.

| Per frame, median of 20 (same boot, frames alternated) | Cached entry | Linked image |
|---|---:|---:|
| In-DSP time, including wrappers | 30.0 ms | **18.8 ms** |
| Kernel body | 18.6 ms | 18.5 ms |
| Overhead per job | 1.61–1.64 ms | 0.03–0.06 ms |
| Host wall time | 92.0 ms | 76.9 ms |

Evidence: [evidence/teacher-link-fp16.json](../evidence/teacher-link-fp16.json)
(statistics only).

## Images

| Image | Where | Size | Contents |
|---|---|---:|---|
| Loader | 4 KiB code slot, uncached | 1,960 B | entry and fault vectors, dispatch, seal and retire commands |
| Operator image | owned DDR `0x1ff0000`, 16 KiB reservation | 4,848 B | `tl_entry`, the resident kernel (opcode 111) and the stride-2 depthwise kernel (opcode 110) |

**The loader** is `kernels/resident_graph_entry.S` built with `RG_LINK_ENTRY` and
`TL_LINK_ENTRY`, plus `kernels/teacher_link_admin.c`.

- **Dispatch.** Opcodes 110 and 111 tail-jump into the image only while the
  seal word holds `TL_SEAL`. The seal word is the last vector of the
  reservation, `0x1ff3f80`, which is outside the window the kernels can
  write. `params[0]` selects the cached (`0x88ff0000`) or uncached alias.
- **Admin (opcode 112).** Every other request reaches the admin code, which
  accepts only two commands:
  - **Seal (7):** allowed only while unsealed. It discards data and
    instruction lines over the reservation through the cached alias, then
    sets the seal.
  - **Retire (8):** clears the seal, then discards the same lines.
- **No change to existing builds.** The new dispatch sits under its own
  define. The ResidentLandmark loader and image, and the two cached-entry
  teacher kernels, rebuild byte-identical.

**The image** is `kernels/teacher_link.c` at offset zero, then
`teacher_resident_probe.c` and `teacher_dw_probe.c` compiled unchanged in
source as `tl_rs` and `tl_dw`. Each kernel still validates its own context,
opcode and descriptors.

**Build checks.** `tools/build_teacher_link.py` builds and checks everything,
reusing the ResidentLandmark link checks:

- PC-relative C relocations only.
- No vector stack accesses and no writable data.
- The loader's vector table, forwarding stubs and dispatch constants are as
  expected, and the loader never publishes or cleans the slot.
- The image entry is at offset zero.
- No indirect calls or jumps other than returns, and no cache or
  exception-return instructions.
- No embedded address of the image itself.

```sh
python tools/build_teacher_link.py --output /path/to/ljobs
sudo .venv/bin/python tests/hardware_teacher_link.py --run-on-device --jobs /path/to/ljobs --model /private/model --corpus /private/corpus --output /path/to/link.json --rounds 5
```

## Run (`tests/hardware_teacher_link.py`)

1. **Qualification.** Per-layer qualification with the ordinary cached-entry
   kernels, as before:
   - Worst ratio to the FP16 bound: 0.494.
   - Landmarks within 0.385 px.
2. **Seal.** The host writes the image and a zero seal vector, reads them
   back, and sends the seal command.
3. **Frames.** For each crop, 5 rounds that alternate a cached-entry frame
   with a linked frame.
   - **Bit-identity:** every linked frame's final tensor was compared with
     qualification and matched. On round 0 of each crop, all 7 stored
     tensors were compared too, and matched.
   - **Image check:** before and after every linked request, the host compares
     the whole reservation with what it sealed.
4. **Controls** on the first crop, then a clean, bit-identical linked frame:
   - a guarded fault inside the image (fault PC in the image's cached alias),
   - a native descriptor rejection,
   - a second seal command, refused while sealed.
5. **Retire.** After retiring, a resident request is refused by the loader
   itself, because the image is no longer entered unsealed.

**Protections:**

- 463 audited restorations (461 HMX).
- The window, and the reservation once retired, were restored. The whole-arena
  hash is identical before and after.
- The DSP service is unchanged.
- Seal and retire each take about 1.7 ms in-DSP.

## Where the time goes now

- **DSP body: 18.5 ms per frame.**
  - The stride-2 jobs still copy their full-resolution input and output
    through the window: 4.2 ms of body time across the three.
  - The depthwise loop shares coefficient loads across 2 blocks instead of 4.
    That was a code-size compromise and no longer needs to be.
- **Host: about 58 ms per frame on top of the DSP time** (8–10 ms per job).
  Most of it is HMX power-on and power-off around every request. RESIDENT_LINK
  measured this at about 5 ms and 1.6 ms. Session-held power
  ([RESIDENT_LINK.md](RESIDENT_LINK.md#session-held-hmx-power)) removes it,
  but it is opt-in, and how other HMX clients behave while it is held is
  still open.

## Next

- **Fold stride 2 into the resident kernel**, keeping tensors in VTCM. The
  image has about 11 KB free. That removes 3 jobs and their copies, and allows
  one or two requests per frame within the 20 ms budget.
- **Go back to 4-block coefficient sharing** in depthwise.
- **Session-held HMX power**, for host latency.
- **The 112x112 stem**, and pooling and the heads on the DSP.
