/* FrameNPU layout kernels: the host side of the Mercury kernel's tensor layout, in C.
 *
 * Window reads, the stride-2 stem im2col and the grid packing of FP16 tensors into the
 * kernel's word layout (framenpu_cdsp._mercury_layout). Loaded with ctypes by
 * framenpu_cdsp/_layout_native.py; freestanding like the model library (fnpu_host.c), built by
 * tools/build_host_native.py --lib layout.
 */
#include "fnpu_common.h"

/* ---------------------------------------------------------------- exported */

EXPORT i32 fnpu_layout_abi(void) { return 1; }

/* out[i] = base[word[i]] for i = 0 .. n-1, in order: one 32-bit load per entry (volatile:
 * never merged, widened, reordered or skipped). Read-only; used on a PROT_READ mapping of
 * the HMX power registers. The caller validates every word index against the mapping. */
EXPORT void fnpu_read32(const volatile u32 *base, const i32 *word, i32 n, u32 *out) {
    for (i32 i = 0; i < n; i++)
        out[i] = base[word[i]];
}

/* Stride-2 stem im2col of an FP16 128x128 image (layout.stem_columns): plane
 * 3*dy+dx holds image[2y-1+dy][2x-1+dx], zero outside; plane 9 is zero. */
EXPORT void fnpu_stem_columns(const u16 *img, u16 *cols) {
    for (i32 dy = 0; dy < 3; dy++)
        for (i32 dx = 0; dx < 3; dx++) {
            u16 *plane = cols + (3 * dy + dx) * 4096;
            for (i32 y = 0; y < 64; y++) {
                i32 iy = 2 * y - 1 + dy;
                for (i32 x = 0; x < 64; x++) {
                    i32 ix = 2 * x - 1 + dx;
                    plane[64 * y + x] =
                        (iy >= 0 && iy < 128 && ix >= 0 && ix < 128) ? img[128 * iy + ix] : 0;
                }
            }
        }
    for (i32 i = 0; i < 4096; i++)
        cols[9 * 4096 + i] = 0;
}

/* Grid packing (Grid.to_bytes). t: c planes of hw FP16 values; idx: the word of each
 * position; out: c/2 pairs of nwords u32 words, channel 2p+j in half j of pair p. */
EXPORT void fnpu_grid_pack(const u16 *t, i32 c, i32 hw, const i32 *idx, i32 nwords, u16 *out) {
    for (i32 i = 0; i < c * nwords; i++)
        out[i] = 0;
    for (i32 p = 0; p < c / 2; p++)
        for (i32 j = 0; j < 2; j++) {
            const u16 *src = t + (2 * p + j) * hw;
            u16 *dst = out + 2 * p * nwords + j;
            for (i32 k = 0; k < hw; k++)
                dst[2 * idx[k]] = src[k];
        }
}

/* Grid unpacking (Grid.from_bytes) into c FP32 planes of hw values. stats[0] = nonzero
 * halves outside the interior (stray), stats[1] = non-finite interior values. */
EXPORT void fnpu_grid_unpack(const u16 *raw, i32 c, i32 hw, const i32 *idx, i32 nwords,
                             float *out, i32 *stats) {
    i32 nonzero = 0, inner = 0, bad = 0;
    for (i32 i = 0; i < c * nwords; i++)
        nonzero += (raw[i] & 0x7fffu) != 0;
    for (i32 p = 0; p < c / 2; p++)
        for (i32 j = 0; j < 2; j++) {
            const u16 *src = raw + 2 * p * nwords + j;
            float *dst = out + (2 * p + j) * hw;
            for (i32 k = 0; k < hw; k++) {
                u16 h = src[2 * idx[k]];
                inner += (h & 0x7fffu) != 0;
                bad += !finite16(h);
                dst[k] = h2f(h);
            }
        }
    stats[0] = nonzero - inner;
    stats[1] = bad;
}
