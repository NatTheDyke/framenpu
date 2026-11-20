/* Teacher tail as a resident op sequence (HMX 1x1 + HVX stride-1 depthwise),
 * FP16, on the grid layout of tests/teacher_tail_layout.py.
 *
 * A job runs ops params[1] .. params[2]-1 of an op table in the arena window
 * (128-byte descriptors, tests/teacher_resident_layout.py). Tensors stay in
 * borrowed VTCM between ops of a job at host-planned vector offsets; load and
 * store ops move whole tensors between VTCM and the window. The stride-2
 * depthwise layers run as separate tests/teacher_dw_probe.c jobs unless
 * RS_STRIDE2 is defined for the linked image.
 *
 * Arithmetic per output vector is that of tests/teacher_pw_probe.c (1x1) and
 * tests/teacher_dw_probe.c (stride 1): one shared post adds the bias (here a
 * pre-splatted vector), clips, adds the residual, masks to interior lanes and
 * stores straight into the zeroed output grid.
 *
 * Descriptor words: kind (1 = 1x1, 2 = depthwise, 3 = load, 4 = store), cpi,
 * cpo, nk (1x1) or k (depthwise), clip, nbin, blk0, nj, ws, dpar, npar, vin,
 * vout, vres (0 = none), vpar (VTCM offsets in vectors, never 0). Parameters: 1x1 = A tiles [o][t] | bias
 * vectors (cpo) | mask (nj); depthwise = tap words | bias vectors | mask.
 * Scratch (B tiles or splatted taps) follows the parameters; for 1x1 ops vpar
 * and npar are multiples of 16 vectors so A and B tiles are 2 KiB aligned. Each descriptor
 * is checked before its op runs: every VTCM index the op can form lies in the
 * borrowed span and every window access inside the window.
 *
 * Input payload: [MAGIC, table offset, table entries]. Meta: [0] magic [1] pc
 * [2] ops run; flags vector at words 64..95; ticks after op i at
 * word 96+i. Ops stop between whole ops at the 20 ms budget. */
/* RS_STRIDE2 adds kind 5: full-resolution depthwise into a temporary grid,
 * then the qualified scalar (2*y+1, 2*x+1) decimation into vout. Descriptor
 * words 15..20 are output nbin, blk0, ws, h, w and temporary vector offset.
 * RS_S2_EVEN (Mercury linked image) keeps (2*y, 2*x) instead, the phase of
 * symmetric (1,1) padding; the bounds checks cover both phases.
 * RS_S2_DIRECT computes only the kept outputs instead: per output row and
 * 32-column block it deals the three input rows into even/odd columns and
 * accumulates the same 9 qf16 products in the same order as rs_dw, so each
 * output equals the decimated stride-1 result; the temporary is unused (still
 * validated). Lanes past the row are stored as zero onto pad words or words a
 * later row rewrites; extra checks bound the reads and that spill.
 * The ordinary 4 KiB reference build does not include this extension. */
/* RS_MHEADS (Mercury heatmap head, stride-2 builds) adds two kinds:
 * kind 7 (regrid) is an exact word copy between grids. Words: 7, pairs per
 * phase np, phases nph (1..9), oh, ow, source nbin, destination nbin, vin,
 * vout (VTCM vector offsets), source row stride, destination row and column
 * steps, flags (bit 0: the source pair advances by np per phase, bit 1: the
 * destination pair does), 0, then (source word, destination word) of each
 * phase in words 14..31. The destination is zeroed first; phase t copies
 * dst[dbase + y*drow + x*dcol] = src[sbase + y*sws + x] for every pair.
 * kind 8 is a 3x3 (padding 1, same grid) convolution on HMX: kind 1 with K =
 * 9 taps x cpi pairs, where B row t*cpi + p is pair p shifted by
 * (t/3 - 1) * ws + t%3 - 1 words (unaligned vector loads). The reads are
 * bounded like the depthwise reach. Same post, parameters and scratch as 1x1. */
#ifdef RS_MHEADS
#if !defined(RS_STRIDE2) || defined(RS_STEM) || defined(RS_HEADS)
#error "RS_MHEADS is a stride-2 Mercury option (no stem prefix, no teacher heads)"
#endif
#define RG_KIND 7u
#define P3_KIND 8u
#endif
#include "landmark_common.h"
#ifdef RS_STEM
#include "teacher_stem_check.h"
#endif
#ifdef RS_HEADS
#include "teacher_heads_check.h"
void th_run(const fnpu_u32 *, const fnpu_context *);
#endif
void fnpu_hmx_clear(void);
void fnpu_hmx_load(const fnpu_u16 *, const fnpu_u16 *);
void fnpu_hmx_store(fnpu_u16 *, const void *);

#define RS_MAGIC 0x31535254u
#define RS_OPCODE 111u
#define RS_DDR 0xf0000000u
#define RS_WIN_LO 0x1a40000u
#define RS_WIN_HI 0x1f00000u
#define RS_META 1024u
#ifdef RS_REPLAY
#if !defined(RS_STRIDE2) || defined(RS_STEM) || defined(RS_HEADS)
#error "RS_REPLAY is a stride-2 Mercury option (no stem prefix, no heads)"
#endif
/* Request word 3: crops (0 or 1 = one crop, at most RS_MAX_CROPS); word 4: the
 * crop stride in bytes (4 KiB multiple). Crop k runs descriptors a..b again with
 * k * stride added to the window address of its load and store descriptors
 * (kinds 3, 4) before validation; compute ops and parameter images are shared.
 * VTCM carries over from crop to crop, so a plan must leave nothing in it that
 * the next crop reads before writing. */
#define RS_MAX_CROPS 3u
#endif
#ifndef RS_DW_BLOCKS
#define RS_DW_BLOCKS 2
#endif
#if RS_DW_BLOCKS != 2 && RS_DW_BLOCKS != 4
#error "RS_DW_BLOCKS must be 2 or 4"
#endif
#ifdef RS_STRIDE2
#define RS_MAX_KIND 5u
#else
#define RS_MAX_KIND 4u
#endif
#define SPAN ((0x311e0000u - 0x31002000u) / 128u)
#ifdef RS_S2_EVEN
#define RS_S2_P 0u /* decimation phase: output (y, x) keeps stride-1 (2y+P, 2x+P) */
#else
#define RS_S2_P 1u
#endif
/* RS_PW_PROFILE (timing instrument only): 19.2 MHz tick totals over the request
 * for 1x1 ops -- parameter copy, output zeroing, B gather, HMX clear/load/store,
 * post, barrier after the gather, an extra barrier before the gather (draining
 * the previous block's stores), a barrier after each tile's post, a second
 * barrier after the HMX store -- in meta words 3..11. Outputs are unchanged. */
#ifdef RS_PW_PROFILE
#define RS_PROF ((volatile fnpu_u32 *)LV(63)) /* zeroed with LV(57..63) per request */
#define RS_PT(v) fnpu_u32 v = rs_ticks()
#define RS_PADD(k, v) (RS_PROF[k] += rs_ticks() - (v))
#else
#define RS_PT(v)
#define RS_PADD(k, v)
#endif
#if defined(RS_S2_DIRECT) && !defined(RS_STRIDE2)
#error "RS_S2_DIRECT requires RS_STRIDE2"
#endif
#define LV(i) ((volatile HVX_Vector *)(0x31000000u + 128u * (i)))
#define RS_SCALE LV(56)
#define RS_FLAGS LV(62)
#define RS_ACC(i) LV(24 + (i)) /* 2 accumulators, qf16 zero at 4 */
#define RS_DESC LV(32)
#define FR(i) ((volatile HVX_Vector *)(0x31002000u + 128u * (i)))
#ifdef RS_PW_GROUP
#define RS_PW_SCR(k, no) (RS_PW_GROUP * ((k) + (no)) * 16) /* B tiles + output tiles */
#else
#define RS_PW_SCR(k, no) ((k) * 16)
#endif

typedef long rs_uv __attribute__((__vector_size__(128), __aligned__(4)));

typedef struct {
    fnpu_u32 kind, cpi, cpo, nk, clip, nbin, blk0, nj, ws, dpar, npar, vin, vout, vres, vpar;
    fnpu_u32 bias, scr; /* derived */
#ifdef RS_STRIDE2
    fnpu_u32 onbin, oblk0, ows, oh, ow, tmp;
#endif
} rs_op;
/* After validation of a compute op, vin, vout, vres, bias and scr hold VTCM
 * addresses (vres stays 0 when absent; plans never place anything at vector
 * 0); vpar stays a vector offset for every kind. */
#define P(x) ((volatile HVX_Vector *)(d->x))

static inline __attribute__((always_inline)) fnpu_u32 rs_ticks(void) {
    unsigned long long t;
    __asm__ volatile("%0 = c31:30" : "=r"(t)::"memory");
    return (fnpu_u32)t;
}

static __attribute__((noinline)) void rs_copy(volatile HVX_Vector *to, const volatile HVX_Vector *from, fnpu_u32 n) {
#pragma clang loop unroll(disable)
    for (fnpu_u32 i = 0; i < n; i++) to[i] = from[i];
}

static __attribute__((noinline)) void rs_zero(volatile HVX_Vector *to, fnpu_u32 n) {
#pragma clang loop unroll(disable)
    for (fnpu_u32 i = 0; i < n; i++) to[i] = Q6_V_vzero();
}

#ifdef RS_MHEADS
/* Kind 7 bounds: every word read lies in the source tensor's pairs and every
 * word written in the destination's (sizes < 2^14, so products cannot wrap). */
static __attribute__((noinline)) fnpu_u32 rg_bad(const volatile fnpu_u32 *r) {
    fnpu_u32 w[32], big = 0;
#pragma clang loop unroll(disable)
    for (fnpu_u32 i = 0; i < 32; i++) big |= w[i] = r[i];
    fnpu_u32 np = w[1], nph = w[2], oh = w[3], ow = w[4], snb = w[5], dnb = w[6], vin = w[7], vout = w[8];
    fnpu_u32 sws = w[9], drow = w[10], dcol = w[11], fl = w[12];
    if ((big > 0x3fff) | (np == 0) | (nph == 0) | (nph > 9) | (oh == 0) | (ow == 0) | (snb == 0) | (dnb == 0) |
        (vin == 0) | (vout == 0) | (fl > 3) | (w[13] != 0) | (dcol == 0))
        return 1;
    fnpu_u32 sp = (fl & 1) ? nph * np : np, dp = (fl & 2) ? nph * np : np;
    fnpu_u32 bad = (vin + sp * snb > SPAN) | (vout + dp * dnb > SPAN) |
                   ((vin < vout + dp * dnb) & (vout < vin + sp * snb)); /* no overlap */
#pragma clang loop unroll(disable)
    for (fnpu_u32 t = 0; t < nph; t++)
        bad |= (w[14 + 2 * t] + (oh - 1) * sws + ow > 32 * snb) |
               (w[15 + 2 * t] + (oh - 1) * drow + (ow - 1) * dcol >= 32 * dnb);
    return bad;
}

static __attribute__((noinline)) void rg_run(const volatile fnpu_u32 *r) {
    const fnpu_u32 np = r[1], nph = r[2], oh = r[3], ow = r[4], snb = r[5], dnb = r[6], sws = r[9];
    const fnpu_u32 drow = r[10], dcol = r[11], fl = r[12];
    const volatile fnpu_u32 *src = (const volatile fnpu_u32 *)FR(r[7]);
    volatile fnpu_u32 *dst = (volatile fnpu_u32 *)FR(r[8]);
    rs_zero((volatile HVX_Vector *)dst, ((fl & 2) ? nph * np : np) * dnb);
#pragma clang loop unroll(disable)
    for (fnpu_u32 t = 0; t < nph; t++) {
        const fnpu_u32 sb = r[14 + 2 * t], db = r[15 + 2 * t];
        const fnpu_u32 sp0 = (fl & 1) ? t * np : 0, dp0 = (fl & 2) ? t * np : 0;
#pragma clang loop unroll(disable)
        for (fnpu_u32 p = 0; p < np; p++) {
            const volatile fnpu_u32 *s = src + 32 * snb * (sp0 + p) + sb;
            volatile fnpu_u32 *o = dst + 32 * dnb * (dp0 + p) + db;
#pragma clang loop unroll(disable)
            for (fnpu_u32 y = 0; y < oh; y++, s += sws, o += drow) {
#pragma clang loop unroll(disable)
                for (fnpu_u32 x = 0; x < ow; x++) o[x * dcol] = s[x];
            }
        }
    }
}
#endif

/* Fetch descriptor i into d; nonzero if any index it implies leaves bounds. */
#ifdef RS_REPLAY
static __attribute__((noinline)) fnpu_u32 rs_get(fnpu_u32 tab, fnpu_u32 i, rs_op *d, fnpu_u32 off) {
#else
static __attribute__((noinline)) fnpu_u32 rs_get(fnpu_u32 tab, fnpu_u32 i, rs_op *d) {
#endif
    rs_copy(RS_DESC, (const volatile HVX_Vector *)(RS_DDR + tab) + i, 1);
#ifdef RS_STEM
    if (((const volatile fnpu_u32 *)RS_DESC)[0] == TS_KIND) {
        d->kind = TS_KIND;
        return ts_bad((const fnpu_u32 *)RS_DESC);
    }
#endif
#ifdef RS_HEADS
    if (((const volatile fnpu_u32 *)RS_DESC)[0] == TH_KIND) {
        fnpu_u32 *row = (fnpu_u32 *)d;
        /* Kind 6 uses the raw descriptor, never the compute-op pointer form. */
        for (fnpu_u32 w = 0; w < 16; w++) row[w] = ((const volatile fnpu_u32 *)RS_DESC)[w];
        return th_bad((const fnpu_u32 *)RS_DESC);
    }
#endif
#ifdef RS_MHEADS
    if (((const volatile fnpu_u32 *)RS_DESC)[0] == RG_KIND) {
        d->kind = RG_KIND; /* runs from the raw descriptor in RS_DESC */
        return rg_bad((const volatile fnpu_u32 *)RS_DESC);
    }
#endif
    fnpu_u32 *t = (fnpu_u32 *)d, big = 0;
#pragma clang loop unroll(disable)
    for (fnpu_u32 w = 0; w < 15; w++) big |= (t[w] = ((const volatile fnpu_u32 *)RS_DESC)[w]) & -(w != 9);
#ifdef RS_REPLAY
    /* off <= the window size (frame_job); a dpar inside the window cannot wrap */
    if (((d->kind == 3) | (d->kind == 4)) & (d->dpar <= RS_WIN_HI)) d->dpar += off;
#endif
#ifdef RS_STRIDE2
    for (fnpu_u32 w = 0; w < 6; w++) big |= ((fnpu_u32 *)&d->onbin)[w] = ((const volatile fnpu_u32 *)RS_DESC)[15 + w];
    fnpu_u32 s2 = d->kind == 5;
#endif
    /* every word but dpar < 2^14: no product or sum below can wrap */
#ifdef RS_MHEADS
    fnpu_u32 p3 = d->kind == P3_KIND;
    fnpu_u32 cpi = d->cpi, cpo = d->cpo, nb = d->nbin, nj = d->nj, k = d->nk, pw = (d->kind == 1) | p3;
    fnpu_u32 bad = (big > 0x3fff) | ((d->kind - 1 >= RS_MAX_KIND) & !p3) | (d->vpar + d->npar > SPAN) |
                   ((d->dpar | ((d->vpar | d->npar) << 3 & -pw)) & 127) |
#else
    fnpu_u32 cpi = d->cpi, cpo = d->cpo, nb = d->nbin, nj = d->nj, k = d->nk, pw = d->kind == 1;
    fnpu_u32 bad = (big > 0x3fff) | (d->kind - 1 >= RS_MAX_KIND) | (d->vpar + d->npar > SPAN) |
                   ((d->dpar | ((d->vpar | d->npar) << 3 & -(d->kind == 1))) & 127) |
#endif
                   /* 1x1: tiles 2 KiB aligned */ (d->dpar - RS_WIN_LO + 128 * d->npar > RS_WIN_HI - RS_WIN_LO);
    bad |= d->vpar == 0;
#ifdef RS_STRIDE2
    bad |= (d->dpar < RS_WIN_LO) | (d->dpar > RS_WIN_HI) |
           (d->npar > (RS_WIN_HI - d->dpar) / 128);
    if (d->kind == 3 || d->kind == 4) return bad;
    /* Reject caps before products involving the additional dimensions. */
    if (bad) return bad;
    fnpu_u32 out_nb = s2 ? d->onbin : nb;
    if (s2) {
        bad |= (d->oh == 0) | (d->ow == 0) | ((d->oh | d->ow | d->ows | d->ws) > 255) |
               (d->ows != d->ow + 2) | (d->ws != 2 * d->ow + 2) | (d->onbin == 0) |
               (d->tmp == 0) | (d->tmp + cpo * nb > SPAN) | (d->vres != 0) | (d->clip != 1);
        if (bad) return bad;
        bad |= (2 * d->oh * d->ws > 32 * nj) |
               (32 * d->oblk0 + (d->oh - 1) * d->ows + 2 + d->ow > 32 * d->onbin);
#ifdef RS_S2_DIRECT
        {
            /* last word read (row 2(oh-1)+P+1, column block xl, 66 words from tap 0)
             * and last word stored (a full vector from the last row's block xl) */
            fnpu_u32 xl = 32 * ((d->ow - 1) >> 5);
            bad |= (32 * d->blk0 + 1 + RS_S2_P + (RS_S2_P + 2 * d->oh - 1) * d->ws + 2 * xl + 65 >= 32 * nb) |
                   (32 * d->oblk0 + 2 + (d->oh - 1) * d->ows + xl + 32 > 32 * d->onbin);
        }
#endif
        d->tmp = (fnpu_u32)FR(d->tmp);
    }
#else
    if (d->kind > 2) return bad; /* load/store: frame_job addresses FR(vpar) itself */
    fnpu_u32 out_nb = nb;
#endif
    fnpu_u32 kk = k * k, reach = (k >> 1) * (d->ws + 1), no = (cpo + 15) >> 4;
    d->bias = d->vpar + (pw ? no * k * 16 : (cpi * kk + 31) / 32);
    d->scr = d->vpar + d->npar;
    bad |= (cpi == 0) | (nj == 0) | (d->vin == 0) | (d->vout == 0) | (d->clip > 1) |
           (d->blk0 + nj > nb) | (d->vin + cpi * nb > SPAN) | (d->vout + cpo * out_nb > SPAN) |
           (d->vres + cpo * nb > SPAN) | (d->bias + cpo + nj > d->scr) | (d->scr + (pw ? RS_PW_SCR(k, no) : cpi * kk) > SPAN);
#ifdef RS_MHEADS
    /* kind 8: K = 9 taps x cpi pairs; tap reads reach ws + 1 words either side */
    bad |= p3 ? (k != (9 * cpi + 15) >> 4) | (32 * d->blk0 < d->ws + 1) | (32 * (d->blk0 + nj) + d->ws + 1 > 32 * nb)
         :
#else
    bad |=
#endif
           pw ? (k != (cpi + 15) >> 4)
              : (cpi != cpo) | ((k != 3) & (k != 5)) | (nj & (RS_DW_BLOCKS - 1)) | (32 * d->blk0 < reach) |
                    (32 * (d->blk0 + nj) + reach > 32 * nb);
    /* compute ops: offsets -> VTCM addresses, except vpar (kept as an offset) */
#pragma clang loop unroll(disable)
    for (fnpu_u32 w = 11; w < 17; w++)
        if (w != 14) t[w] = t[w] ? (fnpu_u32)FR(t[w]) : 0;
    return bad;
}

/* Shared post for output vector (pair pr, block j): bias, clip, residual,
 * interior mask, finite flags; stores into the output grid. */
static __attribute__((noinline)) void rs_post(const rs_op *d, HVX_Vector v, fnpu_u32 pr, fnpu_u32 j) {
    fnpu_u32 at = pr * d->nbin + d->blk0 + j;
    HVX_Vector exp = Q6_V_vsplat_R(0x7c007c00u), flags = RS_FLAGS[0];
    flags = Q6_V_vor_VV(flags, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, exp), exp), -1));
#ifdef RS_STEM
    if (!d->dpar) {
        /* Prefix-only: early-layer rounding propagates through the whole
         * graph. Add half bias through QFloat32, keeping FP16 storage. */
        HVX_Vector one = Q6_V_vsplat_R(0x3c003c00);
        HVX_VectorPair x = Q6_Wqf32_vmpy_VhfVhf(v,one);
        v = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
            Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(x),P(bias)[2*pr+1]),
            Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(x),P(bias)[2*pr])));
    } else
#endif
    v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, P(bias)[pr]));
    if (d->clip) v = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(v, Q6_V_vzero()), Q6_V_vsplat_R(0x46004600u));
    if (d->vres) v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, P(vres)[at]));
    RS_FLAGS[0] = Q6_V_vor_VV(flags, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, exp), exp), -1));
#ifdef RS_STEM
    P(vout)[at] = Q6_V_vand_VV(v, P(bias)[(d->dpar ? d->cpo : 2*d->cpo) + j]);
#else
    P(vout)[at] = Q6_V_vand_VV(v, P(bias)[d->cpo + j]);
#endif
}

#ifdef RS_PW_FAST
#ifdef RS_STEM
#error "RS_PW_FAST has no stem prefix path"
#endif
/* rs_post for one HMX tile (output rows 16o.. of block j), same operations in
 * the same order, with the descriptor fields hoisted and the overflow flags
 * kept in a register: one RS_FLAGS update per tile instead of a
 * read-modify-write per vector (a per-op update would spill the vector).
 * Measured with RS_PW_PROFILE: the per-vector update left about 2.7 us of
 * stores to drain per tile, paid by the next barrier or HMX op. */
#ifdef RS_PW_GROUP
static __attribute__((noinline)) void rs_pw_post(const rs_op *d, fnpu_u32 o, fnpu_u32 j,
                                                 const volatile HVX_Vector *tile) {
#define RS_TILE(rr) tile[rr]
#else
static __attribute__((noinline)) void rs_pw_post(const rs_op *d, fnpu_u32 o, fnpu_u32 j) {
#define RS_TILE(rr) LV(0)[rr]
#endif
    fnpu_u32 n = d->cpo - 16 * o, nb = d->nbin, at = 16 * o * nb + d->blk0 + j;
    if (n > 16) n = 16;
    const volatile HVX_Vector *bias = P(bias) + 16 * o, *res = d->vres ? P(vres) + at : 0;
    volatile HVX_Vector *out = P(vout) + at;
    HVX_Vector exp = Q6_V_vsplat_R(0x7c007c00u), mask = P(bias)[d->cpo + j], fl = Q6_V_vzero();
    fnpu_u32 clip = d->clip;
#pragma clang loop unroll(disable)
    for (fnpu_u32 rr = 0; rr < n; rr++) {
        HVX_Vector v = RS_TILE(rr);
        fl = Q6_V_vor_VV(fl, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, exp), exp), -1));
        v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, bias[rr]));
        if (clip) v = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(v, Q6_V_vzero()), Q6_V_vsplat_R(0x46004600u));
        if (res) v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, res[rr * nb])); /* check: no qf16+hf fusion */
        fl = Q6_V_vor_VV(fl, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, exp), exp), -1));
        out[rr * nb] = Q6_V_vand_VV(v, mask);
    }
    RS_FLAGS[0] = Q6_V_vor_VV(RS_FLAGS[0], fl);
}
#undef RS_TILE
#endif
#ifdef RS_MHEADS
/* B tile rows for block src (= vin + blk0 + j): kind 1 takes pair p at the
 * block itself; kind 8 takes tap t of pair p, shifted (t/3-1)*ws + t%3-1 words. */
static __attribute__((noinline)) void rs_gather(const rs_op *d, volatile HVX_Vector *b, const volatile HVX_Vector *src) {
    fnpu_u32 p = 0;
#ifdef RS_MHEADS
    if (d->kind == P3_KIND) {
#pragma clang loop unroll(disable)
        for (fnpu_u32 t = 0; t < 9; t++) {
            const int off = ((int)(t / 3) - 1) * (int)d->ws + (int)(t % 3) - 1;
            const char *s = (const char *)src + 4 * off;
#pragma clang loop unroll(disable)
            for (fnpu_u32 q = 0; q < d->cpi; q++, p++) b[p] = *(const rs_uv *)(s + 128 * d->nbin * q);
        }
    } else
#endif
#pragma clang loop unroll(disable)
    for (; p < d->cpi; p++) b[p] = src[p * d->nbin];
#pragma clang loop unroll(disable)
    for (; p < d->nk * 16; p++) b[p] = LV(57)[0]; /* zero */
}
#define RS_GATHER(b, src) rs_gather(d, b, src)
#else
#define RS_GATHER(b, src) \
    _Pragma("clang loop unroll(disable)") for (fnpu_u32 p = 0; p < nk * 16; p++) (b)[p] = p < d->cpi ? (src)[p * d->nbin] : LV(57)[0] /* zero */
#endif

#ifdef RS_PW_GROUP
#ifndef RS_PW_FAST
#error "RS_PW_GROUP builds on RS_PW_FAST"
#endif
/* 1x1 over groups of RS_PW_GROUP column blocks: gather every block's B tile,
 * one barrier, every HMX tile of the group into its own output buffer, then
 * every post. Per tile the same HMX loads and the same rs_pw_post as rs_pw
 * below, so outputs are bit-identical. The group pays one post -> barrier/HMX
 * transition instead of one per tile (that wait is fixed per transition, about
 * 0.65 us), and a tile's post no longer waits for its own HMX writes to land.
 * Scratch: RS_PW_GROUP * (nk + no) * 16 vectors (B tiles, then output tiles;
 * each 2 KiB aligned since scr is). */
static __attribute__((noinline)) void rs_pw(const rs_op *d) {
    const fnpu_u32 nk = d->nk, no = (d->cpo + 15) >> 4, nj = d->nj;
    volatile HVX_Vector *bt = P(scr), *ot = bt + RS_PW_GROUP * nk * 16;
#pragma clang loop unroll(disable)
    for (fnpu_u32 j0 = 0; j0 < nj; j0 += RS_PW_GROUP) {
        const fnpu_u32 g = nj - j0 < RS_PW_GROUP ? nj - j0 : RS_PW_GROUP;
#ifdef RS_PW_PROFILE
        {   /* instrument: drain the previous group's post stores separately */
            RS_PT(tq);
            __asm__ volatile("barrier" ::: "memory");
            RS_PADD(6, tq);
        }
#endif
        RS_PT(tg);
#pragma clang loop unroll(disable)
        for (fnpu_u32 jj = 0; jj < g; jj++) {
            const volatile HVX_Vector *src = P(vin) + d->blk0 + j0 + jj;
            volatile HVX_Vector *b = bt + jj * nk * 16;
            RS_GATHER(b, src);
        }
        RS_PADD(2, tg);
        RS_PT(tb);
        __asm__ volatile("barrier" ::: "memory");
        RS_PADD(5, tb);
        RS_PT(th);
#pragma clang loop unroll(disable)
        for (fnpu_u32 jj = 0; jj < g; jj++) {
#pragma clang loop unroll(disable)
            for (fnpu_u32 o = 0; o < no; o++) {
                fnpu_hmx_clear();
#pragma clang loop unroll(disable)
                for (fnpu_u32 t = 0; t < nk; t++)
                    fnpu_hmx_load((const fnpu_u16 *)FR(d->vpar + (o * nk + t) * 16),
                                  (const fnpu_u16 *)(bt + jj * nk * 16 + 16 * t));
                fnpu_hmx_store((fnpu_u16 *)(ot + (jj * no + o) * 16), (const void *)RS_SCALE);
            }
        }
        RS_PADD(3, th);
        RS_PT(tp);
#pragma clang loop unroll(disable)
        for (fnpu_u32 jj = 0; jj < g; jj++) {
#pragma clang loop unroll(disable)
            for (fnpu_u32 o = 0; o < no; o++) rs_pw_post(d, o, j0 + jj, ot + (jj * no + o) * 16);
        }
        RS_PADD(4, tp);
    }
}
#else
static __attribute__((noinline)) void rs_pw(const rs_op *d) {
    volatile HVX_Vector *bt = P(scr);
    fnpu_u32 nk = d->nk;
#pragma clang loop unroll(disable)
    for (fnpu_u32 j = 0; j < d->nj; j++) {
        const volatile HVX_Vector *src = P(vin) + d->blk0 + j;
#ifdef RS_PW_PROFILE
        {   /* instrument: drain the previous block's post stores separately */
            RS_PT(tq);
            __asm__ volatile("barrier" ::: "memory");
            RS_PADD(6, tq);
        }
#endif
        RS_PT(tg);
        RS_GATHER(bt, src);
        RS_PADD(2, tg);
        RS_PT(tb);
        __asm__ volatile("barrier" ::: "memory");
        RS_PADD(5, tb);
#pragma clang loop unroll(disable)
        for (fnpu_u32 o = 0; 16 * o < d->cpo; o++) {
            RS_PT(th);
            fnpu_hmx_clear();
#pragma clang loop unroll(disable)
            for (fnpu_u32 t = 0; t < nk; t++)
                fnpu_hmx_load((const fnpu_u16 *)FR(d->vpar + (o * nk + t) * 16), (const fnpu_u16 *)(bt + 16 * t));
            fnpu_hmx_store((fnpu_u16 *)LV(0), (const void *)RS_SCALE);
            RS_PADD(3, th);
#ifdef RS_PW_PROFILE
            {   /* instrument: a second barrier after the HMX store */
                RS_PT(ts);
                __asm__ volatile("barrier" ::: "memory");
                RS_PADD(8, ts);
            }
#endif
            RS_PT(tp);
#ifdef RS_PW_FAST
            rs_pw_post(d, o, j);
#else
#pragma clang loop unroll(disable)
            for (fnpu_u32 rr = 0; rr < 16 && 16 * o + rr < d->cpo; rr++) rs_post(d, LV(0)[rr], 16 * o + rr, j);
#endif
            RS_PADD(4, tp);
#ifdef RS_PW_PROFILE
            {   /* instrument: drain this tile's post stores */
                RS_PT(tr);
                __asm__ volatile("barrier" ::: "memory");
                RS_PADD(7, tr);
            }
#endif
        }
    }
}
#endif

#ifdef RS_DW_FAST
#if RS_DW_BLOCKS != 2 || defined(RS_STEM)
#error "RS_DW_FAST is the 2-block, stem-free depthwise"
#endif
/* Keeps v in a register as its own value: stops LLVM fusing an FP16 rounding
 * (hf = qf16) into the following qf16 + hf add, which would skip that rounding. */
#define RS_FENCE(v) __asm__ volatile("" : "+v"(v))
/* rs_post's operations for a depthwise output (no residual: rs_get rejects
 * vres on depthwise), in the same order; flags accumulate in fl. */
#define RS_DW_POST(v, at, bias, mask, fl)                                                    \
    do {                                                                                  \
        HVX_Vector x_ = (v);                                                              \
        RS_FENCE(x_);                                                                     \
        fl = Q6_V_vor_VV(fl, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(x_, exp), exp), -1)); \
        x_ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(x_, bias));                         \
        if (clip) x_ = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(x_, Q6_V_vzero()), six);    \
        fl = Q6_V_vor_VV(fl, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(x_, exp), exp), -1)); \
        P(vout)[at] = Q6_V_vand_VV(x_, mask);                                             \
    } while (0)
/* rs_dw with the taps unrolled for 3x3, the post inlined and the overflow
 * flags held in a register for the whole op (one RS_FLAGS update). Same
 * products, same accumulation order, same post: bit-identical output. */
static __attribute__((noinline)) void rs_dw(const rs_op *d) {
    const fnpu_u32 k = d->nk, kk = k * k, step = 4 * (d->ws - k), nb = d->nbin, clip = d->clip;
    const fnpu_u32 base = 4 * (32 * d->blk0 - (k >> 1) * (d->ws + 1));
    const volatile fnpu_u32 *coef = (const volatile fnpu_u32 *)FR(d->vpar);
    const HVX_Vector exp = Q6_V_vsplat_R(0x7c007c00u), six = Q6_V_vsplat_R(0x46004600u);
    const HVX_Vector zq = Q6_Vqf16_vadd_VhfVhf(Q6_V_vzero(), Q6_V_vzero());
    HVX_Vector fl = Q6_V_vzero();
#pragma clang loop unroll(disable)
    for (fnpu_u32 i = 0; i < d->cpi * kk; i++) P(scr)[i] = Q6_V_vsplat_R(coef[i]);
#pragma clang loop unroll(disable)
    for (fnpu_u32 c = 0; c < d->cpi; c++) {
        const char *s = (const char *)d->vin + 128 * nb * c + base;
        const HVX_Vector *w = (const HVX_Vector *)d->scr + c * kk;
        const HVX_Vector bias = P(bias)[c];
#pragma clang loop unroll(disable)
        for (fnpu_u32 j = 0; j < d->nj; j += 2) {
            HVX_Vector a0 = zq, a1 = zq;
            const char *p = s + 128 * j;
            if (k == 3) {
                const fnpu_u32 r = 4 * d->ws;
#define RS_TAP(o, t)                                                                          \
    a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)(p + (o)), w[t]));       \
    a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)(p + (o) + 128), w[t]))
                RS_TAP(0, 0); RS_TAP(4, 1); RS_TAP(8, 2);
                RS_TAP(r, 3); RS_TAP(r + 4, 4); RS_TAP(r + 8, 5);
                RS_TAP(2 * r, 6); RS_TAP(2 * r + 4, 7); RS_TAP(2 * r + 8, 8);
#undef RS_TAP
            } else {
                const HVX_Vector *wt = w;
#pragma clang loop unroll(disable)
                for (fnpu_u32 dy = 0; dy < k; dy++, p += step) {
#pragma clang loop unroll(disable)
                    for (fnpu_u32 dx = 0; dx < k; dx++, p += 4, wt++) {
                        a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)p, *wt));
                        a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)(p + 128), *wt));
                    }
                }
            }
            fnpu_u32 at = c * nb + d->blk0 + j;
            RS_DW_POST(Q6_Vhf_equals_Vqf16(a0), at, bias, P(bias)[d->cpo + j], fl);
            RS_DW_POST(Q6_Vhf_equals_Vqf16(a1), at + 1, bias, P(bias)[d->cpo + j + 1], fl);
        }
    }
    RS_FLAGS[0] = Q6_V_vor_VV(RS_FLAGS[0], fl);
}
#else
static __attribute__((noinline)) void rs_dw(const rs_op *d) {
    const fnpu_u32 k = d->nk, kk = k * k, step = 4 * (d->ws - k);
    const fnpu_u32 base = 4 * (32 * d->blk0 - (k >> 1) * (d->ws + 1));
    const volatile fnpu_u32 *coef = (const volatile fnpu_u32 *)FR(d->vpar);
    RS_ACC(4)[0] = Q6_Vqf16_vadd_VhfVhf(Q6_V_vzero(), Q6_V_vzero());
#pragma clang loop unroll(disable)
    for (fnpu_u32 i = 0; i < d->cpi * kk; i++) P(scr)[i] = Q6_V_vsplat_R(coef[i]);
#pragma clang loop unroll(disable)
    for (fnpu_u32 c = 0; c < d->cpi; c++) {
        const char *s = (const char *)d->vin + 128 * d->nbin * c + base;
#pragma clang loop unroll(disable)
        for (fnpu_u32 j = 0; j < d->nj; j += RS_DW_BLOCKS) {
            HVX_Vector a0 = RS_ACC(4)[0], a1 = a0; /* qf16 zero */
#if RS_DW_BLOCKS == 4
            HVX_Vector a2 = a0, a3 = a0;
#endif
            const char *p = s + 128 * j;
            const HVX_Vector *w = (const HVX_Vector *)d->scr + c * kk;
#pragma clang loop unroll(disable)
            for (fnpu_u32 dy = 0; dy < k; dy++, p += step) {
#pragma clang loop unroll(disable)
                for (fnpu_u32 dx = 0; dx < k; dx++, p += 4, w++) {
                    HVX_Vector cw = *w;
                    a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)p, cw));
                    a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)(p + 128), cw));
#if RS_DW_BLOCKS == 4
                    a2 = Q6_Vqf16_vadd_Vqf16Vqf16(a2, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)(p + 256), cw));
                    a3 = Q6_Vqf16_vadd_Vqf16Vqf16(a3, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)(p + 384), cw));
#endif
                }
            }
            RS_ACC(0)[0] = a0; RS_ACC(1)[0] = a1;
#if RS_DW_BLOCKS == 4
            RS_ACC(2)[0] = a2; RS_ACC(3)[0] = a3;
#endif
#pragma clang loop unroll(disable)
            for (fnpu_u32 q = 0; q < RS_DW_BLOCKS; q++) rs_post(d, Q6_Vhf_equals_Vqf16(RS_ACC(q)[0]), c, j + q);
        }
    }
}
#endif

#if defined(RS_STRIDE2) && defined(RS_S2_DIRECT)
static __attribute__((noinline)) void rs_s2(rs_op *d) {
    const fnpu_u32 ws = d->ws, ows = d->ows, kk = 9;
    const volatile fnpu_u32 *coef = (const volatile fnpu_u32 *)FR(d->vpar);
    const HVX_Vector exp = Q6_V_vsplat_R(0x7c007c00u), top = Q6_V_vsplat_R(0x46004600u);
    rs_zero((volatile HVX_Vector *)d->vout, d->cpo * d->onbin);
    RS_ACC(4)[0] = Q6_Vqf16_vadd_VhfVhf(Q6_V_vzero(), Q6_V_vzero());
#ifdef RS_DW_FAST
    HVX_Vector fl = Q6_V_vzero();
#endif
#pragma clang loop unroll(disable)
    for (fnpu_u32 i = 0; i < d->cpi * kk; i++) P(scr)[i] = Q6_V_vsplat_R(coef[i]);
    /* word of tap (dy = 0, dx = 0) for output (0, 0): row P-1, column P-1 */
    const fnpu_u32 off0 = 32 * d->blk0 + 1 + RS_S2_P + RS_S2_P * ws - ws;
#pragma clang loop unroll(disable)
    for (fnpu_u32 c = 0; c < d->cpi; c++) {
        const char *in = (const char *)d->vin + 128 * d->nbin * c + 4 * off0;
        volatile char *out = (volatile char *)d->vout + 128 * d->onbin * c + 4 * (32 * d->oblk0 + 2);
        const HVX_Vector *w = (const HVX_Vector *)d->scr + c * kk;
        const HVX_Vector bias = P(bias)[c];
#pragma clang loop unroll(disable)
        for (fnpu_u32 y = 0; y < d->oh; y++) {
#pragma clang loop unroll(disable)
            for (fnpu_u32 x0 = 0; x0 < d->ow; x0 += 32) {
                HVX_Vector a = RS_ACC(4)[0]; /* qf16 zero */
                const char *p = in + 4 * (2 * y * ws + 2 * x0);
#pragma clang loop unroll(disable)
                for (fnpu_u32 dy = 0; dy < 3; dy++, p += 4 * ws) {
                    HVX_VectorPair e = Q6_W_vdeal_VVR(*(const rs_uv *)(p + 128), *(const rs_uv *)p, -4);
                    HVX_VectorPair f = Q6_W_vdeal_VVR(*(const rs_uv *)(p + 136), *(const rs_uv *)(p + 8), -4);
                    a = Q6_Vqf16_vadd_Vqf16Vqf16(a, Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(e), w[3 * dy]));
                    a = Q6_Vqf16_vadd_Vqf16Vqf16(a, Q6_Vqf16_vmpy_VhfVhf(Q6_V_hi_W(e), w[3 * dy + 1]));
                    a = Q6_Vqf16_vadd_Vqf16Vqf16(a, Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(f), w[3 * dy + 2]));
                }
#ifdef RS_DW_FAST
                HVX_Vector v = Q6_Vhf_equals_Vqf16(a);
                RS_FENCE(v); /* the FP16 rounding stays its own step, as below */
                v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, bias));
#else
                /* Round the accumulator to FP16 before the bias, as rs_dw/rs_post do:
                 * the volatile slot stops LLVM fusing it into a qf16+hf add. */
                RS_ACC(0)[0] = Q6_Vhf_equals_Vqf16(a);
                HVX_Vector v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(RS_ACC(0)[0], bias));
#endif
                v = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(v, Q6_V_vzero()), top);
                fnpu_u32 n = d->ow - x0;
                if (n < 32) v = Q6_V_vmux_QVV(Q6_Q_vsetq_R(4 * n), v, Q6_V_vzero());
#ifdef RS_DW_FAST
                fl = Q6_V_vor_VV(fl, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, exp), exp), -1));
#else
                RS_FLAGS[0] = Q6_V_vor_VV(RS_FLAGS[0], Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, exp), exp), -1));
#endif
                *(volatile rs_uv *)(out + 4 * (y * ows + x0)) = v;
            }
        }
    }
#ifdef RS_DW_FAST
    RS_FLAGS[0] = Q6_V_vor_VV(RS_FLAGS[0], fl);
#endif
}
#elif defined(RS_STRIDE2)
static __attribute__((noinline)) void rs_s2(rs_op *d) {
    fnpu_u32 dst = d->vout;
    rs_zero((volatile HVX_Vector *)dst, d->cpo * d->onbin);
    d->vout = d->tmp;
    rs_dw(d);
    d->vout = dst;
#ifdef RS_S2_EVEN
    /* Symmetric (1,1) padding (PyTorch/Mercury): keep stride-1 word (0, 0). */
    const volatile fnpu_u32 *s0 = (const volatile fnpu_u32 *)d->tmp + 32 * d->blk0 + 2;
#else
    const volatile fnpu_u32 *s0 = (const volatile fnpu_u32 *)d->tmp + 32 * d->blk0 + d->ws + 3;
#endif
    volatile fnpu_u32 *t0 = (volatile fnpu_u32 *)dst + 32 * d->oblk0 + 2;
#pragma clang loop unroll(disable)
    for (fnpu_u32 c = 0; c < d->cpo; c++, s0 += 32 * d->nbin, t0 += 32 * d->onbin) {
        const volatile fnpu_u32 *s = s0;
        volatile fnpu_u32 *t = t0;
#pragma clang loop unroll(disable)
        for (fnpu_u32 y = 0; y < d->oh; y++, s += 2 * d->ws, t += d->ows) {
#pragma clang loop unroll(disable)
            for (fnpu_u32 x = 0; x < d->ow; x++) t[x] = s[2 * x];
        }
    }
}
#endif

#ifdef RS_STEM
#include "teacher_stem_impl.h"
#endif
#ifdef RS_STEM_PROFILE
#include "teacher_stem_profile.h"
#endif

fnpu_u32 frame_job(const fnpu_context *c) {
    const volatile fnpu_u32 *in = fnpu_input(c);
    fnpu_u32 tab = in[1], n = in[2], a = c->params[1], b = c->params[2], i;
#ifdef RS_REPLAY
    fnpu_u32 nc = in[3] ? in[3] : 1, cs = in[4], steps = 0, stop = 0, t = 0;
    if ((nc > RS_MAX_CROPS) | (cs & 4095) | ((nc == 1) & (cs != 0)) | (cs > (RS_WIN_HI - RS_WIN_LO) / 2))
        return FNPU_BAD_ARGUMENT;
#endif
    rs_op ds[1];
    rs_op *d = ds;
#ifdef RS_STRIDE2
    if (tab < RS_WIN_LO || tab > RS_WIN_HI || n > (RS_WIN_HI - tab) / 128)
        return FNPU_BAD_ARGUMENT;
#endif
    if (!valid(c) || c->opcode != RS_OPCODE || c->params[0] != 1 || c->input_bytes != 128 ||
        c->output_bytes != RS_META || in[0] != RS_MAGIC || (tab & 127) | (n > 160) | (b > n) | /* a >= b runs no op */
        (tab - RS_WIN_LO + 128 * n > RS_WIN_HI - RS_WIN_LO))
        return FNPU_BAD_ARGUMENT;
    volatile fnpu_u32 *meta = fnpu_output(c);
    rs_zero((volatile HVX_Vector *)meta, RS_META / 128);
    rs_zero(LV(57), 7); /* zero vector, flags */
    RS_SCALE[0] = Q6_V_vsplat_R(0x3c00u);
    fault_control(c);
    fnpu_u32 t0 = rs_ticks();
#ifdef RS_REPLAY
#pragma clang loop unroll(disable)
    for (fnpu_u32 k = 0; k < nc && !stop; k++) {
#pragma clang loop unroll(disable)
    for (i = a; i < b; i++) {
        if (rs_get(tab, i, d, k * cs)) return FNPU_BAD_ARGUMENT;
#else
#pragma clang loop unroll(disable)
    for (i = a; i < b; i++) {
        if (rs_get(tab, i, d)) return FNPU_BAD_ARGUMENT;
#endif
#ifdef RS_STEM
        if (d->kind == TS_KIND) {
#ifdef RS_STEM_PROFILE
            if (((const volatile fnpu_u32 *)RS_DESC)[19]) {
                if (!tp_run(c,(const fnpu_u32 *)RS_DESC,t0)) break;
            } else
#endif
            if (!ts_run(c, (const fnpu_u32 *)RS_DESC, t0)) break;
        } else {
#endif
#ifdef RS_MHEADS
        if (d->kind == RG_KIND) {
            rg_run((const volatile fnpu_u32 *)RS_DESC);
        } else {
#endif
        volatile HVX_Vector *v = FR(d->vpar), *m = (volatile HVX_Vector *)(RS_DDR + d->dpar);
        RS_PT(tc);
        if (d->kind == 4) rs_copy(m, v, d->npar);
        else rs_copy(v, m, d->npar);
#ifdef RS_PW_PROFILE
        if (d->kind == 1) RS_PADD(0, tc);
#endif
        RS_PT(tz);
#ifdef RS_MHEADS
        if ((d->kind < 3) | (d->kind == P3_KIND)) rs_zero(P(vout), d->cpo * d->nbin);
#else
        if (d->kind < 3) rs_zero(P(vout), d->cpo * d->nbin);
#endif
#ifdef RS_PW_PROFILE
        if (d->kind == 1) RS_PADD(1, tz);
#endif
#ifdef RS_MHEADS
        if ((d->kind == 1) | (d->kind == P3_KIND)) rs_pw(d);
#else
        if (d->kind == 1) rs_pw(d);
#endif
        if (d->kind == 2) rs_dw(d);
#ifdef RS_STRIDE2
        if (d->kind == 5) rs_s2(d);
#endif
#ifdef RS_HEADS
        if (d->kind == TH_KIND) th_run((const fnpu_u32 *)RS_DESC, c);
#endif
#ifdef RS_MHEADS
        }
#endif
#ifdef RS_STEM
        }
#endif
#ifdef RS_REPLAY
        t = rs_ticks() - t0;
        meta[96 + i - a] = t; /* the last crop's ticks remain */
        steps++;
        if (t > 384000u) { stop = 1; break; }
    }
    if (nc > 1) meta[48 + k] = t; /* crop k's end */
    }
#else
        fnpu_u32 t = rs_ticks() - t0;
        meta[96 + i - a] = t;
        if (t > 384000u) { i++; break; }
    }
#endif
    fnpu_u32 pc;
    __asm__ volatile("%0 = pc" : "=r"(pc)::"memory");
#ifdef RS_REPLAY
    meta[0] = RS_MAGIC; meta[1] = pc; meta[2] = steps; /* descriptors run, all crops */
#else
    meta[0] = RS_MAGIC; meta[1] = pc; meta[2] = i - a;
#endif
#ifdef RS_PW_PROFILE
    for (fnpu_u32 k = 0; k < 9; k++) meta[3 + k] = RS_PROF[k];
#endif
    ((volatile HVX_Vector *)meta)[2] = RS_FLAGS[0];
    __asm__ volatile("barrier" ::: "memory");
    return FNPU_OK;
}
