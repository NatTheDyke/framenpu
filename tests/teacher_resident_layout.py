"""Host side of the resident teacher tail (tests/teacher_resident_probe.c,
opcode 111): op tables, VTCM and window plans, parameter images and a
lane-level simulator of the kernel. Opens no device.

A plan is a sequence of jobs. A resident job runs a range of 128-byte
descriptors (1x1, stride-1 depthwise, load, store) with tensors kept in VTCM
between ops; the three stride-2 depthwise layers run as
tests/teacher_dw_probe.c jobs (opcode 110) from and to the window.

Two plan modes:
- "resident": one resident job per stride-1 segment (4 jobs + 3 stride-2
  jobs); every parameter image lives in the window, written once; only
  tensors that cross a job boundary (and the final tensor) are stored.
- "qualify": one job per compute op, every output stored and read back, and
  each job's parameters staged into a shared window area just before it.
Both use the same kernel and per-op arithmetic, so their outputs must agree
bit for bit.
"""

from dataclasses import dataclass, field

import teacher_tail_layout as T

RS_MAGIC, RS_OPCODE = 0x31535254, 111  # "TRS1"
SPAN = T.VTCM_SPAN_VECTORS
KIND = {"pw": 1, "dw": 2, "load": 3, "store": 4}
VTCM_FIRST = 4  # vector 0 is never allocated (vres = 0 means "no residual")
FINAL = "79"
INPUTS = ("8", "10")


def _al(n, a=128):
    return (n + a - 1) // a * a


# ---------------------------------------------------------------- parameter images


def _splat_pairs(values, pairs):
    """FP16 values -> one splatted vector per channel pair (word = even | odd << 16)."""
    import numpy as np

    h = np.zeros(2 * pairs, np.float16)
    h[: len(values)] = np.asarray(values, np.float32).astype(np.float16)
    w = h.view("<u2").astype("<u4")
    word = w[0::2] | (w[1::2] << 16)
    return np.repeat(word[:, None], 32, 1).astype("<u4").tobytes()


def rs_params(graph, spec, kind, gin):
    """Parameter image (bytes) and scratch vectors for one resident compute op."""
    import numpy as np
    from framenpu.landmark import weights_and_bias

    _, b = weights_and_bias(graph, spec)
    if kind == "pw":
        tiles, _, nk, _ = T.pw_weights(graph, spec)
        cpo = spec["output_shape"][0] // 2
        image = tiles + _splat_pairs(b, cpo) + gin.mask_bytes()[: gin.nj * 128]
        image += bytes(-len(image) % 2048)  # B tiles after it stay 2 KiB aligned
        return image, nk * 16
    coef, _ = T.dw_weights(graph, spec)
    cp, kk = spec["input_shape"][0] // 2, spec["kernel"][0] ** 2
    image = coef + _splat_pairs(b, cp) + gin.mask_bytes()
    del np
    return image, cp * kk


def s2_params(graph, spec, gin):
    coef, bias = T.dw_weights(graph, spec)
    return coef, bias, gin.mask_bytes()


# ---------------------------------------------------------------- plan


@dataclass
class Job:
    kind: str  # "rs" or "s2"
    first: int = 0  # descriptor range (rs)
    end: int = 0
    op: object = None  # T.Op (s2)
    desc: bytes = b""  # opcode-110 descriptor (s2)
    ops: list = field(default_factory=list)  # compute op names, in order
    stores: list = field(default_factory=list)  # tensors written to the window
    staged: dict = field(default_factory=dict)  # window writes needed just before the job
    vtcm_peak: int = 0


@dataclass
class Plan:
    mode: str
    jobs: list
    table: bytes  # all descriptors
    table_at: int
    fixed: dict  # window writes done once (resident parameters)
    where: dict  # tensor -> window offset
    end: int  # first window byte past the plan
    ops: list
    shape_of: dict


class _Vtcm:
    def __init__(self):
        self.free = [(VTCM_FIRST, SPAN - VTCM_FIRST)]
        self.peak = 0  # most vectors in use at once

    def take(self, n, top=False, align=1):
        """First fit from the bottom (start aligned to `align` vectors), or
        (long-lived tensors) last fit from the top."""
        order = range(len(self.free) - 1, -1, -1) if top else range(len(self.free))
        for i in order:
            at, size = self.free[i]
            pad = -at % align
            if size >= n + pad:
                if top:
                    self.free[i : i + 1] = [(at, size - n)] if size > n else []
                    at += size - n
                else:
                    rest = [(at, pad)] if pad else []
                    rest += [(at + pad + n, size - pad - n)] if size > pad + n else []
                    self.free[i : i + 1] = rest
                    at += pad
                self.peak = max(self.peak, SPAN - VTCM_FIRST - sum(s for _, s in self.free))
                return at
        raise ValueError(f"VTCM plan does not fit ({n} vectors; free {self.free})")

    def give(self, at, n):
        self.free.append((at, n))
        self.free.sort()
        merged = []
        for a, s in self.free:
            if merged and merged[-1][0] + merged[-1][1] == a:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((a, s))
        self.free = merged


def _words(*w):
    import numpy as np

    out = np.zeros(32, "<u4")
    out[: len(w)] = w
    return out.tobytes()


def segments(ops, mode):
    """Job partition: lists of op indices; stride-2 ops alone."""
    groups, cur = [], []
    for i, op in enumerate(ops):
        s2 = op.kind == "dw" and op.stride2
        if s2 or mode == "qualify":
            if cur:
                groups.append(cur)
                cur = []
            groups.append([i])
        else:
            cur.append(i)
    if cur:
        groups.append(cur)
    return groups


@dataclass(frozen=True)
class _Op:
    kind: str
    name: str
    src: str
    dst: str
    res: str | None
    spec: dict
    stride2: bool


def build_plan(graph, specs, mode):
    if mode not in ("resident", "qualify"):
        raise ValueError(mode)
    by = {s["name"]: s for s in specs}
    tail, _ = T.tail_ops(specs)
    shape_of = T.shapes(specs)
    ops = [_Op(o.kind, o.name, o.src, o.dst, o.res, by[o.name],
               o.kind == "dw" and by[o.name]["stride"] == (2, 2)) for o in tail]
    groups = segments(ops, mode)
    job_of = {i: g for g, idx in enumerate(groups) for i in idx}
    grid = lambda b: T.grid_of(shape_of[b])  # noqa: E731
    pairs = lambda b: shape_of[b][0] // 2  # noqa: E731
    vec = lambda b: pairs(b) * grid(b).nbin  # noqa: E731

    # tensors that must be in the window: inputs, outputs used by a later job, the final, observed
    producer = {o.dst: job_of[i] for i, o in enumerate(ops)} | {b: -1 for b in INPUTS}
    users = {}
    for i, o in enumerate(ops):
        for b in (o.src, o.res):
            if b:
                users.setdefault(b, []).append(job_of[i])
    stored = {b for b in producer if b == FINAL or any(u != producer[b] for u in users.get(b, []))}
    if mode == "qualify":
        stored |= {o.dst for o in ops}
    stored |= set(INPUTS)

    # parameter images
    images = {}
    for i, o in enumerate(ops):
        if o.stride2:
            coef, bias, mask = s2_params(graph, o.spec, grid(o.src))
            images[i] = (coef + bias + mask, 0, (len(coef), len(bias)))
        else:
            image, scratch = rs_params(graph, o.spec, o.kind, grid(o.src))
            images[i] = (image, scratch, None)

    # window: table | parameters (fixed) or staging | tensors
    n_desc_bound = sum(4 * len(g) + 2 for g in groups)  # generous upper bound
    table_at = T.WIN_LO
    cursor = _al(table_at + 128 * n_desc_bound, 4096)
    par_at = {}
    if mode == "resident":
        for i in range(len(ops)):
            par_at[i] = cursor
            cursor = _al(cursor + len(images[i][0]))
        fixed = {par_at[i]: images[i][0] for i in range(len(ops))}
    else:
        stage = cursor
        cursor = _al(stage + max(len(im[0]) for im in images.values()), 4096)
        par_at = {i: stage for i in range(len(ops))}
        fixed = {}

    # tensor slots, reused after the last job that touches them
    last_job = {}
    for b in stored:
        js = [producer[b]] + users.get(b, [])
        last_job[b] = max(js)
    where, slot, free, live = {}, {}, [], []
    for b in sorted(stored, key=lambda b: (producer[b], b)):
        for x in [x for x in live if last_job[x] < producer[b]]:
            live.remove(x)
            free.append(slot[x])
        size = _al(vec(b) * 128, 4096)
        hit = next((k for k, (a, n) in enumerate(free) if n >= size), None)
        if hit is None:
            slot[b] = (cursor, size)
            cursor += size
        else:
            slot[b] = free.pop(hit)  # whole slot, so it is returned whole
        where[b] = slot[b][0]
        live.append(b)
    if cursor > T.WIN_HI:
        raise ValueError(f"window plan ends at {cursor:#x}")

    # descriptors
    table, jobs = [], []
    for g, idx in enumerate(groups):
        if ops[idx[0]].stride2:
            i = idx[0]
            o = ops[i]
            image, _, sizes = images[i]
            desc = T.dw_descriptor(o.spec, grid(o.src), grid(o.dst), where[o.src], par_at[i], list(sizes), where[o.dst])
            job = Job("s2", op=o, desc=desc, ops=[o.name], stores=[o.dst])
            if mode == "qualify":
                job.staged = {par_at[i]: image}
            jobs.append(job)
            continue
        job = Job("rs", first=len(table), ops=[ops[i].name for i in idx])
        vt, at = _Vtcm(), {}
        last_use = {}
        for i in idx:
            for b in (ops[i].src, ops[i].res):
                if b:
                    last_use[b] = i
        for i in idx:
            o = ops[i]
            for b in (o.src, o.res):
                if b and b not in at:
                    at[b] = vt.take(vec(b))
                    table.append(_words(KIND["load"], 0, 0, 0, 0, 0, 0, 0, 0, where[b], vec(b), 0, 0, 0, at[b]))
            later = [k for k in idx if k > i and o.dst in (ops[k].src, ops[k].res)]
            at[o.dst] = vt.take(vec(o.dst), top=bool(later) and later[-1] > i + 1)
            image, scratch, _ = images[i]
            npar = len(image) // 128
            vpar = vt.take(npar + scratch, align=16 if o.kind == "pw" else 1)  # HMX tiles: 2 KiB
            gin = grid(o.src)
            if o.kind == "pw":
                cin, cout = o.spec["input_shape"][0], o.spec["output_shape"][0]
                table.append(_words(KIND["pw"], cin // 2, cout // 2, (cin // 2 + 15) >> 4,
                                    int(o.spec["activation"] == "clip"), gin.nbin, gin.blk0, gin.nj, gin.ws,
                                    par_at[i], npar, at[o.src], at[o.dst], at[o.res] if o.res else 0, vpar))
            else:
                cp = pairs(o.src)
                table.append(_words(KIND["dw"], cp, cp, o.spec["kernel"][0], 1, gin.nbin, gin.blk0, gin.nj4, gin.ws,
                                    par_at[i], npar, at[o.src], at[o.dst], 0, vpar))
            if mode == "qualify":
                job.staged = {par_at[i]: image}
            vt.give(vpar, npar + scratch)
            if o.dst in stored:
                table.append(_words(KIND["store"], 0, 0, 0, 0, 0, 0, 0, 0, where[o.dst], vec(o.dst), 0, 0, 0, at[o.dst]))
                job.stores.append(o.dst)
            for b in {o.src, o.res} - {None}:
                if last_use[b] == i:
                    vt.give(at.pop(b), vec(b))
            if o.dst not in last_use:  # not read again in this job (stored above)
                vt.give(at.pop(o.dst), vec(o.dst))
        job.end = len(table)
        job.vtcm_peak = vt.peak
        jobs.append(job)
        if job.end - job.first > 160:
            raise ValueError("job exceeds 160 descriptors")
    if len(table) > n_desc_bound:
        raise ValueError("table bound")
    return Plan(mode, jobs, b"".join(table), table_at, fixed, where, cursor, ops, shape_of)


def setup_writes(plan):
    """Window writes required before any job of the plan: the op table and,
    for resident plans, every parameter image. Harness and simulator both use
    exactly this list."""
    return {plan.table_at: plan.table} | dict(plan.fixed)


def payload(plan):
    import numpy as np

    w = np.zeros(32, "<u4")
    w[:3] = [RS_MAGIC, plan.table_at, len(plan.table) // 128]
    return w.tobytes()


# ---------------------------------------------------------------- lane-level simulator


def simulate(plan, window):
    """Run a plan against `window` (bytearray covering WIN_LO..WIN_HI) with the
    kernel's addressing; VTCM starts each resident job filled with NaN. The
    caller performs setup_writes(plan) first, exactly as the harness does; the
    table is read back from the window for every job."""
    import numpy as np
    from hmx_block_layout import unpack

    def vecs(off, n):
        return np.frombuffer(bytes(window[off - T.WIN_LO : off - T.WIN_LO + 128 * n]), "<f2").reshape(n, 32, 2)

    def put(off, data):
        window[off - T.WIN_LO : off - T.WIN_LO + len(data)] = data

    def f16(x):
        return np.asarray(x, np.float64).astype(np.float16)

    for job in plan.jobs:
        for off, data in job.staged.items():
            put(off, data)
        tab = np.frombuffer(bytes(window[plan.table_at - T.WIN_LO : plan.table_at - T.WIN_LO + len(plan.table)]),
                            "<u4").reshape(-1, 32)
        if job.kind == "s2":
            for off, data in T.model_dw(job.desc, lambda o, n: bytes(window[o - T.WIN_LO : o - T.WIN_LO + n])).items():
                put(off, data)
            continue
        vt = np.full((SPAN, 32, 2), np.nan, np.float16)
        for row in tab[job.first : job.end]:
            kind, cpi, cpo, k, clip, nb, blk0, nj, ws, dpar, npar, vin, vout, vres, vpar = (int(x) for x in row[:15])
            if kind == 3:
                vt[vpar : vpar + npar] = vecs(dpar, npar)
                continue
            if kind == 4:
                put(dpar, vt[vpar : vpar + npar].tobytes())
                continue
            vt[vpar : vpar + npar] = vecs(dpar, npar)
            vt[vout : vout + cpo * nb] = 0
            if kind == 1:
                if (vpar | npar) % 16:
                    raise ValueError("HMX tiles not 2 KiB aligned")
                no = (cpo + 15) >> 4
                bias = vpar + no * k * 16
                raw = vt[vpar : vpar + no * k * 16].tobytes()
                A = np.block([[unpack(raw[2048 * (o * k + t) : 2048 * (o * k + t + 1)]) for t in range(k)] for o in range(no)])
                for j in range(nj):
                    blk = vt[vin + blk0 + j : vin + blk0 + j + cpi * nb : nb]  # [cpi, 32, 2]
                    B = np.zeros((k * 32, 32))
                    B[: 2 * cpi] = blk.astype(np.float64).transpose(0, 2, 1).reshape(2 * cpi, 32)
                    acc = f16(A @ B)[: 2 * cpo].reshape(cpo, 2, 32).transpose(0, 2, 1)
                    v = f16(acc.astype(np.float64) + vt[bias : bias + cpo].astype(np.float64))
                    if clip:
                        v = np.clip(v, 0, 6).astype(np.float16)
                    if vres:
                        v = f16(v.astype(np.float64) + vt[vres + blk0 + j : vres + blk0 + j + cpo * nb : nb].astype(np.float64))
                    m = vt[bias + cpo + j].view("<u4").reshape(32) != 0
                    vt[vout + blk0 + j : vout + blk0 + j + cpo * nb : nb] = np.where(m[None, :, None], v, 0)
            else:
                kk, r = k * k, k // 2
                coef = vt[vpar:].reshape(-1, 2)[: cpi * kk].reshape(cpi, kk, 2).astype(np.float32)
                bias = vpar + (cpi * kk + 31) // 32
                src = vt[vin : vin + cpi * nb].reshape(cpi, nb * 32, 2).astype(np.float32)
                o = 32 * blk0 + np.arange(nj * 32)
                acc = np.zeros((cpi, nj * 32, 2), np.float32)
                t = 0
                for dy in range(k):
                    for dx in range(k):
                        acc = acc + (src[:, o + (dy - r) * ws + (dx - r)] * coef[:, t][:, None, :]).astype(np.float32)
                        t += 1
                bv = vt[bias : bias + cpi, 0, :].astype(np.float64)  # splatted: lane 0 suffices
                v = np.clip(f16(f16(acc).astype(np.float64) + bv[:, None, :]), 0, 6)
                m = (vt[bias + cpi : bias + cpi + nj].view("<u4") != 0).reshape(-1)
                out = np.where(m[None, :, None], v, 0).astype(np.float16)
                flat = vt[vout : vout + cpi * nb].reshape(cpi, nb * 32, 2)
                flat[:, 32 * blk0 : 32 * (blk0 + nj)] = out
                vt[vout : vout + cpi * nb] = flat.reshape(-1, 32, 2)
    return window


# ---------------------------------------------------------------- native check mirror


def native_ok(row):
    """Mirror of rs_get in tests/teacher_resident_probe.c (True = accepted)."""
    m = 0xFFFFFFFF
    w = [int(x) for x in row[:15]]
    kind, cpi, cpo, k, clip, nb, blk0, nj, ws, dpar, npar, vin, vout, vres, vpar = w
    big = 0
    for i, x in enumerate(w):
        if i != 9:
            big |= x
    align = ((vpar | npar) << 3) if kind == 1 else 0
    bad = (big > 0x3FFF) or (kind - 1) & m > 3 or vpar + npar > SPAN or (dpar | align) & 127
    bad = bad or ((dpar - T.WIN_LO) & m) + 128 * npar > T.WIN_HI - T.WIN_LO or vpar == 0
    if kind > 2 or bad:
        return not bad
    kk, reach, no = k * k, (k >> 1) * (ws + 1), (cpo + 15) >> 4
    pw = kind == 1
    bias = vpar + (no * k * 16 if pw else (cpi * kk + 31) // 32)
    scr = vpar + npar
    bad = (cpi == 0 or nj == 0 or vin == 0 or vout == 0 or clip > 1 or blk0 + nj > nb or vin + cpi * nb > SPAN
           or vout + cpo * nb > SPAN or vres + cpo * nb > SPAN or bias + cpo + nj > scr
           or scr + (k * 16 if pw else cpi * kk) > SPAN)
    if pw:
        bad = bad or k != (cpi + 15) >> 4
    else:
        bad = (bad or cpi != cpo or k not in (3, 5) or nj & 1 or 32 * blk0 < reach
               or 32 * (blk0 + nj) + reach > 32 * nb)
    return not bad
