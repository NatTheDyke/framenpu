# VTCM ownership map

Status (2026-11-17): static map, read-only survey and an approved write
canary over `0x31002000`–`0x311e0000`, which passed. The headset was idle and
no worn-headset soak was run. Hand tracking cannot be exercised: only a
CPU-bound model is instrumented. Firmware: the pinned
`cdsp.analysis.elf` (sha256 `692f9c57…`), same boot as the HMX block runs
(dsp_service PID 1015, NRestarts 0).

## Static map (Ghidra/REA + full disassembly of every `0x31xxxxxx` operand)

Only 37 real instructions (outside data decoded as code) reference VTCM
outside gaze, all in FeatureNN (`0x870052e0`–`0x8700ad58`).

| Range | Size | Owner | Lifetime |
|---|---:|---|---|
| `0x31000000`–`0x310f0000` | 960 KiB | FeatureNN worker HTID 0 scratch (`0x31000000 + HTID*0xf0000`, `FUN_87005060`) | per request: the ping-pong bump allocator (`0x870050c0` bottom-up, `0x87005100` top-down, `0x87005140` swap) is reset on every `FUN_87005330` entry |
| `0x310f0000`–`0x311e0000` | 960 KiB | FeatureNN worker HTID 1 scratch | per request |
| `0x311e0000`–`0x316eea80` | 5,178 KiB | FeatureNN model blob, copied by op 7 (`0x870052e0`: `memcpy(0x311e0000, 0x8707c400, 0x50ea80)`), read as `0x311e0000 + offset` / `0x31492080 + offset` | persistent once op 7 has run |
| `0x316eea80`–`0x31700000` | 69 KiB | none referenced | — |
| `0x31700000`–`0x31776600` | 474 KiB | gaze (HTID 3, `FUN_87015500`; tables at `0x31700a80`, `0x3171df00`–`0x3171ee00`, copy of `0xbc84` B to `0x31760180`) | **concurrent with every job** |
| `0x31776600`–`~0x317d4600` | ~376 KiB | gaze work buffer (31 row pairs × `0x2f00`, bound from the loop at `0x87015590`) | concurrent |
| `~0x317d4600`–`0x31800000` | ~174 KiB | none referenced | — |

The HTID-2 slot formula would place worker 2's scratch on the model blob.
That slot is not a candidate.

Concurrency: FeatureNN (op 8) and direct jobs (op 4) are both dispatched by
the HTID-4 master's single CPU-request loop onto workers 0/1/2
([DISPATCH.md](DISPATCH.md)), so FeatureNN can never run during a direct job.
Gaze runs on HTID 3 independently and can.

## Read-only survey (`tests/vtcm_survey_probe.c`, `tests/hardware_vtcm_survey.py`)

The probe reads all 8 MiB on worker 0 (595 µs, 2.2 ms wrapper) and returns
the following per 4 KiB chunk: word sum, rotate-xor hash, OR and nonzero count.
A dump mode copies a raw 64 KiB window. All audits passed: per-job
restoration, resident arena hash, health, and service PID.

Headset idle, 9 h uptime:

- FeatureNN worker 0/1 slots, the model-blob range, and both unreferenced gaps:
  each sampled 64 KiB window has byte entropy 7.99 bits, is incompressible
  (zlib 1.000) and has 50.0% of bits set. That is untouched SRAM power-on
  state. **The model blob is not loaded.** FeatureNN's DDR worker slots
  `0xb22000` and `0x1022000` are entirely zero, so FeatureNN has not run this
  boot.
- Gaze float data (OR = `0xbfffffff`) is at `0x3171e000`–`0x31759000`,
  `0x3175e000`–`0x31769000` and scattered chunks up to `0x317d1000`. That
  matches the static bound.
- Nothing changed over the samples, so gaze was idle.

## Write canary (`tests/vtcm_canary_probe.c`, `tests/hardware_vtcm_canary.py`)

Run 1 filled the whole 1,912 KiB span with a seed-keyed address pattern
(seed `0x6af91db2`). The device verified it in the same job. A raw 64 KiB
window then matched the host-computed pattern exactly. Ten verify-only jobs
over 30 s each found 0 mismatched words. Before and after surveys of all
8 MiB show no change outside the span.

Run 2 was a new process and session a few minutes later: 12 verify-only jobs
over 110 s, 0 mismatches, nothing changed outside the span. Both runs had 16
audited restorations each, all checks passed, the longest wrapper took
2.0 ms, and dsp_service stayed at PID 1015 with NRestarts 0.

Costs: filling 1,912 KiB takes 150 µs (about 13 GB/s VTCM store rate), and
verifying it takes 221 µs. A residency hash check over a 338 KiB layer would
therefore take about 40 µs.

The pattern is left in place. FeatureNN scratch semantics make that harmless.

## Candidate regions

| Candidate | Size | Argument | Open evidence |
|---|---:|---|---|
| **FeatureNN worker 0/1 scratch** `0x31002000`–`0x311e0000` | **1,912 KiB** | Per-request scratch of a client that is serialized with direct jobs. Never written this boot. | Write canary and persistence **passed** (idle headset). Use it across jobs only with an in-job hash check and reload, because FeatureNN may legitimately reuse it. |
| Gap below gaze `0x316ef000`–`0x31700000` | 68 KiB | Unreferenced; noise | Worn-headset soak (gaze is adjacent) |
| Top `0x317d5000`–`0x31800000` | 172 KiB | Unreferenced beyond the gaze loop bound; noise | Worn-headset soak; exact gaze bound |
| Model-blob range | 5 MiB | Unused this boot | **Not proposed:** persistent FeatureNN state once op 7 runs |

What 1.9 MiB buys for the [spatial block](HMX_SPATIAL_BLOCK.md): every
per-layer weight set (≤ about 338 KiB) can stay resident for a whole job. Mid
activations stay in VTCM instead of spilling. A full source band can also be
staged once per job, which replaces the 32 scattered DDR loads per K tile
(67 of the 99 forecast ms) with VTCM `valign` work.
