/* FrameNPU host kernels: the CPU side of a Mercury request, in C.
 *
 * The model half: elementwise FP16 conversion and the host layers (dense, 1x1, pool, hint).
 * The kernel-layout half (window reads, stem columns, grid pack and unpack) is
 * framenpu_cdsp/host/fnpu_layout.c.
 *
 * Loaded with ctypes by framenpu/_native.py. Freestanding: no libc, no Python API,
 * no allocation, no global state. Every function reads and writes only the caller's
 * buffers, whose sizes the caller has checked. Built by tools/build_host_native.py
 * with -ffp-contract=off so that every product and sum rounds exactly as written,
 * on every architecture.
 *
 * Numerics follow framenpu.mercury's NumPy reference:
 *   - FP32 -> FP16 is round-to-nearest-even (NumPy's astype(float16)).
 *   - dense: FP64 products and sums of FP32 values, plus the FP32 bias in FP64, then
 *     one rounding to FP32 (the reference's FP64 matmul; only the summation order may
 *     differ, which can move a result by one FP32 ulp in rare cases).
 *   - conv1x1 on a 1x1 map (the reference's FP32 matmul): FP64 accumulation, rounded
 *     to FP32, plus the FP32 bias in FP32. This is at least as exact as an FP32 BLAS,
 *     whose summation order the reference does not fix either.
 *   - average pools of FP16-valued inputs: the FP64 sum of at most 36 FP16 values
 *     is exact, so the result equals NumPy's mean exactly.
 *   - conv1x1 on a spatial FP16 map (the reference's "hmx" unit): FP64 products and sums
 *     of FP16 values rounded once to FP16, plus the FP16 bias in FP32, rounded to FP16.
 */
#include "fnpu_common.h"

/* ---------------------------------------------------------------- exported */

EXPORT i32 fnpu_abi(void) { return 5; }

/* Count of non-finite values in x[0..n). */
EXPORT i32 fnpu_nonfinite32(const float *x, i32 n) {
    i32 bad = 0;
    for (i32 i = 0; i < n; i++)
        bad += !finite32(x[i]);
    return bad;
}

/* y = FP16(x), returning the count of non-finite results (overflow included). */
EXPORT i32 fnpu_to_f16(const float *x, u16 *y, i32 n) {
    i32 bad = 0;
    for (i32 i = 0; i < n; i++) {
        y[i] = f2h(x[i]);
        bad += !finite16(y[i]);
    }
    return bad;
}

/* y = FP32(x) for FP16 x, exact. */
EXPORT void fnpu_from_f16(const u16 *x, float *y, i32 n) {
    for (i32 i = 0; i < n; i++)
        y[i] = h2f(x[i]);
}

/* For each of nb inputs (rows of x, n_in each): acc[b][i] = sum over k = 0 .. n_in-1, in
 * order, of FP64(wt[k][i]) * FP64(x[b][k]); wt is FP32, transposed (n_in x n_out). The product
 * of two FP32 values is exact in FP64, so widening here equals FP64 weights. One pass over the
 * weights serves every input (these layers are bound by weight bandwidth); each output's sum
 * keeps the same order, so a batch equals nb single calls bit for bit. (Holding blocks of
 * outputs in registers measured slower on the Frame.) More than three inputs run as three,
 * then the rest (same per-output order, so still equal to single calls). */
static void dot_columns(const float *wt, const float *x, i32 nb, i32 n_out, i32 n_in,
                        double *acc) {
    if (nb > 3) {
        dot_columns(wt, x, 3, n_out, n_in, acc);
        dot_columns(wt, x + 3 * (usize)n_in, nb - 3, n_out, n_in, acc + 3 * (usize)n_out);
        return;
    }
    for (i32 i = 0; i < nb * n_out; i++)
        acc[i] = 0.0;
    double *a0 = acc, *a1 = acc + n_out, *a2 = acc + 2 * n_out;
    for (i32 k = 0; k < n_in; k++) {
        const float *r = wt + (usize)k * (usize)n_out;
        double x0 = (double)x[k];
        if (nb == 1) {
            for (i32 i = 0; i < n_out; i++)
                a0[i] += (double)r[i] * x0;
        } else if (nb == 2) {
            double x1 = (double)x[n_in + k];
            for (i32 i = 0; i < n_out; i++) {
                double w = (double)r[i];
                a0[i] += w * x0;
                a1[i] += w * x1;
            }
        } else {
            double x1 = (double)x[n_in + k], x2 = (double)x[2 * n_in + k];
            for (i32 i = 0; i < n_out; i++) {
                double w = (double)r[i];
                a0[i] += w * x0;
                a1[i] += w * x1;
                a2[i] += w * x2;
            }
        }
    }
}

/* Dense (mercury._step "dense") of nb inputs: y[b] = FP32(sum FP64(w) * FP64(x[b]) +
 * FP64(bias)), then relu6. wt: the FP32 weight transposed (n_in x n_out); x: nb x n_in;
 * acc: nb x n_out doubles of scratch; y: nb x n_out. 1 <= nb <= 4. */
EXPORT void fnpu_dense(const float *wt, const float *b, const float *x, i32 nb, i32 n_out,
                       i32 n_in, i32 act_relu6, double *acc, float *y) {
    dot_columns(wt, x, nb, n_out, n_in, acc);
    for (i32 j = 0; j < nb * n_out; j++) {
        float v = (float)(acc[j] + (double)b[j % n_out]);
        y[j] = act_relu6 ? relu6(v) : v;
    }
}

/* 1x1 conv of a 1x1 map ("fp32" unit) of nb inputs: y[b] = FP32(sum FP64 w*x[b]) + bias in
 * FP32, then relu6. Same layout and scratch as fnpu_dense. */
EXPORT void fnpu_conv1x1(const float *wt, const float *b, const float *x, i32 nb, i32 n_out,
                         i32 n_in, i32 act_relu6, double *acc, float *y) {
    dot_columns(wt, x, nb, n_out, n_in, acc);
    for (i32 j = 0; j < nb * n_out; j++) {
        float v = (float)acc[j] + b[j % n_out];
        y[j] = act_relu6 ? relu6(v) : v;
    }
}

/* Average pool (mercury._step "avgpool"): non-overlapping kh x kw windows of the top-left
 * oh*kh x ow*kw corner of c planes of h x w; FP64 sum / count, rounded to FP32. */
EXPORT void fnpu_avgpool(const float *x, i32 c, i32 h, i32 w, i32 kh, i32 kw, i32 oh, i32 ow,
                         float *y) {
    double n = (double)(kh * kw);
    for (i32 ch = 0; ch < c; ch++)
        for (i32 oy = 0; oy < oh; oy++)
            for (i32 ox = 0; ox < ow; ox++) {
                double s = 0.0;
                for (i32 dy = 0; dy < kh; dy++)
                    for (i32 dx = 0; dx < kw; dx++)
                        s += (double)x[((usize)ch * h + oy * kh + dy) * w + ox * kw + dx];
                y[((usize)ch * oh + oy) * ow + ox] = (float)(s / n);
            }
}

/* Hint planes (Gemm_58 -> Reshape_63 -> Mul_70) of nb crops:
 * out[b] = FP16(FP32(relu6 dense(kp[b])) * flag[b]). Returns the count of non-finite FP16
 * values. kp: nb x n_in; acc: nb x n_out doubles; tmp, out: nb x n_out. */
EXPORT i32 fnpu_hint(const float *wt, const float *b, const float *kp, const float *flag, i32 nb,
                     i32 n_out, i32 n_in, i32 act_relu6, double *acc, float *tmp, u16 *out) {
    fnpu_dense(wt, b, kp, nb, n_out, n_in, act_relu6, acc, tmp);
    i32 bad = 0;
    for (i32 j = 0; j < nb * n_out; j++) {
        out[j] = f2h(tmp[j] * flag[j / n_out]);
        bad += !finite16(out[j]);
    }
    return bad;
}

/* 1x1 conv of an FP16 map (mercury._conv, unit "hmx"): y[o][p] = FP16(FP32(FP16(sum over
 * k = 0 .. cin-1, in order, of FP64(w[o][k]) * FP64(x[k][p]))) + b[o]), then relu6. w (cout x
 * cin), b and x (cin planes of hw) hold FP16 values as FP32; acc: hw doubles; y: cout x hw. */
EXPORT void fnpu_conv1x1_f16(const float *w, const float *b, const float *x, i32 cout, i32 cin,
                             i32 hw, i32 act_relu6, double *acc, float *y) {
    for (i32 o = 0; o < cout; o++) {
        for (i32 p = 0; p < hw; p++)
            acc[p] = 0.0;
        const float *wr = w + (usize)o * (usize)cin;
        for (i32 k = 0; k < cin; k++) {
            double wk = (double)wr[k];
            const float *xr = x + (usize)k * (usize)hw;
            for (i32 p = 0; p < hw; p++)
                acc[p] += wk * (double)xr[p];
        }
        float *yr = y + (usize)o * (usize)hw;
        for (i32 p = 0; p < hw; p++) {
            float v = h2f(f2h(h2f(d2h(acc[p])) + b[o]));
            yr[p] = act_relu6 ? relu6(v) : v;
        }
    }
}
