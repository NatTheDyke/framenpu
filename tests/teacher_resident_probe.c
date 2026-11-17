/* Teacher tail as a resident op sequence (HMX 1x1 + HVX stride-1 depthwise),
 * FP16, on the grid layout of tests/teacher_tail_layout.py.
 *
 * A job runs ops params[1] .. params[2]-1 of an op table in the arena window
 * (128-byte descriptors, tests/teacher_resident_layout.py). Tensors stay in
 * borrowed VTCM between ops of a job at host-planned vector offsets; load and
 * store ops move whole tensors between VTCM and the window. The stride-2
 * depthwise layers run as separate tests/teacher_dw_probe.c jobs.
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
#include "landmark_common.h"
void fnpu_hmx_clear(void);
void fnpu_hmx_load(const fnpu_u16 *, const fnpu_u16 *);
void fnpu_hmx_store(fnpu_u16 *, const void *);

#define RS_MAGIC 0x31535254u
#define RS_OPCODE 111u
#define RS_DDR 0xf0000000u
#define RS_WIN_LO 0x1a40000u
#define RS_WIN_HI 0x1f00000u
#define RS_META 1024u
#define SPAN ((0x311e0000u - 0x31002000u) / 128u)
#define LV(i) ((volatile HVX_Vector *)(0x31000000u + 128u * (i)))
#define RS_SCALE LV(56)
#define RS_FLAGS LV(62)
#define RS_ACC(i) LV(24 + (i)) /* 2 accumulators, qf16 zero at 4 */
#define RS_DESC LV(32)
#define FR(i) ((volatile HVX_Vector *)(0x31002000u + 128u * (i)))

typedef long rs_uv __attribute__((__vector_size__(128), __aligned__(4)));

typedef struct {
    fnpu_u32 kind, cpi, cpo, nk, clip, nbin, blk0, nj, ws, dpar, npar, vin, vout, vres, vpar;
    fnpu_u32 bias, scr; /* derived */
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

/* Fetch descriptor i into d; nonzero if any index it implies leaves bounds. */
static __attribute__((noinline)) fnpu_u32 rs_get(fnpu_u32 tab, fnpu_u32 i, rs_op *d) {
    rs_copy(RS_DESC, (const volatile HVX_Vector *)(RS_DDR + tab) + i, 1);
    fnpu_u32 *t = (fnpu_u32 *)d, big = 0;
#pragma clang loop unroll(disable)
    for (fnpu_u32 w = 0; w < 15; w++) big |= (t[w] = ((const volatile fnpu_u32 *)RS_DESC)[w]) & -(w != 9);
    /* every word but dpar < 2^14: no product or sum below can wrap */
    fnpu_u32 cpi = d->cpi, cpo = d->cpo, nb = d->nbin, nj = d->nj, k = d->nk, pw = d->kind == 1;
    fnpu_u32 bad = (big > 0x3fff) | (d->kind - 1 > 3) | (d->vpar + d->npar > SPAN) |
                   ((d->dpar | ((d->vpar | d->npar) << 3 & -(d->kind == 1))) & 127) | /* 1x1: tiles 2 KiB aligned */ (d->dpar - RS_WIN_LO + 128 * d->npar > RS_WIN_HI - RS_WIN_LO);
    bad |= d->vpar == 0;
    if (d->kind > 2) return bad; /* load/store: frame_job addresses FR(vpar) itself */
    fnpu_u32 kk = k * k, reach = (k >> 1) * (d->ws + 1), no = (cpo + 15) >> 4;
    d->bias = d->vpar + (pw ? no * k * 16 : (cpi * kk + 31) / 32);
    d->scr = d->vpar + d->npar;
    bad |= (cpi == 0) | (nj == 0) | (d->vin == 0) | (d->vout == 0) | (d->clip > 1) |
           (d->blk0 + nj > nb) | (d->vin + cpi * nb > SPAN) | (d->vout + cpo * nb > SPAN) |
           (d->vres + cpo * nb > SPAN) | (d->bias + cpo + nj > d->scr) | (d->scr + (pw ? k * 16 : cpi * kk) > SPAN);
    bad |= pw ? (k != (cpi + 15) >> 4)
              : (cpi != cpo) | ((k != 3) & (k != 5)) | (nj & 1) | (32 * d->blk0 < reach) |
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
    v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, P(bias)[pr]));
    if (d->clip) v = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(v, Q6_V_vzero()), Q6_V_vsplat_R(0x46004600u));
    if (d->vres) v = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(v, P(vres)[at]));
    RS_FLAGS[0] = Q6_V_vor_VV(flags, Q6_V_vand_QR(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(v, exp), exp), -1));
    P(vout)[at] = Q6_V_vand_VV(v, P(bias)[d->cpo + j]);
}

static __attribute__((noinline)) void rs_pw(const rs_op *d) {
    volatile HVX_Vector *bt = P(scr);
    fnpu_u32 nk = d->nk;
#pragma clang loop unroll(disable)
    for (fnpu_u32 j = 0; j < d->nj; j++) {
        const volatile HVX_Vector *src = P(vin) + d->blk0 + j;
#pragma clang loop unroll(disable)
        for (fnpu_u32 p = 0; p < nk * 16; p++) bt[p] = p < d->cpi ? src[p * d->nbin] : LV(57)[0]; /* zero */
        __asm__ volatile("barrier" ::: "memory");
#pragma clang loop unroll(disable)
        for (fnpu_u32 o = 0; 16 * o < d->cpo; o++) {
            fnpu_hmx_clear();
#pragma clang loop unroll(disable)
            for (fnpu_u32 t = 0; t < nk; t++)
                fnpu_hmx_load((const fnpu_u16 *)FR(d->vpar + (o * nk + t) * 16), (const fnpu_u16 *)(bt + 16 * t));
            fnpu_hmx_store((fnpu_u16 *)LV(0), (const void *)RS_SCALE);
#pragma clang loop unroll(disable)
            for (fnpu_u32 rr = 0; rr < 16 && 16 * o + rr < d->cpo; rr++) rs_post(d, LV(0)[rr], 16 * o + rr, j);
        }
    }
}

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
        for (fnpu_u32 j = 0; j < d->nj; j += 2) {
            HVX_Vector a0 = RS_ACC(4)[0], a1 = a0; /* qf16 zero */
            const char *p = s + 128 * j;
            const HVX_Vector *w = (const HVX_Vector *)d->scr + c * kk;
#pragma clang loop unroll(disable)
            for (fnpu_u32 dy = 0; dy < k; dy++, p += step) {
#pragma clang loop unroll(disable)
                for (fnpu_u32 dx = 0; dx < k; dx++, p += 4, w++) {
                    HVX_Vector cw = *w;
                    a0 = Q6_Vqf16_vadd_Vqf16Vqf16(a0, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)p, cw));
                    a1 = Q6_Vqf16_vadd_Vqf16Vqf16(a1, Q6_Vqf16_vmpy_VhfVhf(*(const rs_uv *)(p + 128), cw));
                }
            }
            RS_ACC(0)[0] = a0; RS_ACC(1)[0] = a1;
#pragma clang loop unroll(disable)
            for (fnpu_u32 q = 0; q < 2; q++) rs_post(d, Q6_Vhf_equals_Vqf16(RS_ACC(q)[0]), c, j + q);
        }
    }
}

fnpu_u32 frame_job(const fnpu_context *c) {
    const volatile fnpu_u32 *in = fnpu_input(c);
    fnpu_u32 tab = in[1], n = in[2], a = c->params[1], b = c->params[2], i;
    rs_op ds[1];
    rs_op *d = ds;
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
#pragma clang loop unroll(disable)
    for (i = a; i < b; i++) {
        if (rs_get(tab, i, d)) return FNPU_BAD_ARGUMENT;
        volatile HVX_Vector *v = FR(d->vpar), *m = (volatile HVX_Vector *)(RS_DDR + d->dpar);
        if (d->kind == 4) rs_copy(m, v, d->npar);
        else rs_copy(v, m, d->npar);
        if (d->kind < 3) rs_zero(P(vout), d->cpo * d->nbin);
        if (d->kind == 1) rs_pw(d);
        if (d->kind == 2) rs_dw(d);
        fnpu_u32 t = rs_ticks() - t0;
        meta[96 + i - a] = t;
        if (t > 384000u) { i++; break; }
    }
    fnpu_u32 pc;
    __asm__ volatile("%0 = pc" : "=r"(pc)::"memory");
    meta[0] = RS_MAGIC; meta[1] = pc; meta[2] = i - a;
    ((volatile HVX_Vector *)meta)[2] = RS_FLAGS[0];
    __asm__ volatile("barrier" ::: "memory");
    return FNPU_OK;
}
