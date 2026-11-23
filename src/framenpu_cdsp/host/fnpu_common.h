/* FrameNPU host kernels: the types, freestanding memset/memcpy and FP16 helpers shared by the
 * model library (framenpu/host/fnpu_host.c) and the layout library
 * (framenpu_cdsp/host/fnpu_layout.c). The two packages carry identical copies of this file
 * (tests/test_native_host.py checks), so each library builds from its own package alone.
 * Rounding follows framenpu.mercury's NumPy reference (see fnpu_host.c).
 */
#ifndef FNPU_COMMON_H
#define FNPU_COMMON_H

typedef unsigned short u16;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long usize;

#define EXPORT __attribute__((visibility("default")))

/* The compiler may lower loops to these; the library carries its own (no_builtin keeps
 * them from being lowered to calls to themselves). */
__attribute__((visibility("hidden"), no_builtin("memset"))) void *memset(void *d, int c,
                                                                         usize n) {
    unsigned char *p = d;
    unsigned long v = (unsigned char)c * 0x0101010101010101ul;
    while (n && ((usize)p & 7u)) {
        *p++ = (unsigned char)c;
        n--;
    }
    for (; n >= 8; n -= 8, p += 8)
        *(unsigned long *)p = v;
    while (n--)
        *p++ = (unsigned char)c;
    return d;
}

__attribute__((visibility("hidden"), no_builtin("memcpy"))) void *memcpy(void *d, const void *s,
                                                                         usize n) {
    unsigned char *p = d;
    const unsigned char *q = s;
    while (n--)
        *p++ = *q++;
    return d;
}

static inline u32 bits(float f) {
    union { float f; u32 u; } v = {f};
    return v.u;
}

static inline float from_bits(u32 u) {
    union { u32 u; float f; } v = {u};
    return v.f;
}

static inline int finite32(float f) { return (bits(f) & 0x7f800000u) != 0x7f800000u; }
static inline int finite16(u16 h) { return (h & 0x7c00u) != 0x7c00u; }

/* FP32 -> FP16, round to nearest even, IEEE overflow to infinity. */
static inline u16 f2h(float f) {
    u32 x = bits(f), sign = (x >> 16) & 0x8000u, ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u)
        return (u16)(sign | 0x7e00u);  /* NaN (callers reject it) */
    if (ax >= 0x477ff000u)
        return (u16)(sign | 0x7c00u);  /* >= 65520 rounds to infinity */
    if (ax >= 0x38800000u) {          /* normal FP16 */
        u32 m = ax + 0xfffu + ((ax >> 13) & 1u);
        return (u16)(sign | ((m - 0x38000000u) >> 13));
    }
    u32 e = ax >> 23;
    if (e < 102u)
        return (u16)sign;              /* below 2^-25 rounds to zero */
    u32 mant = (ax & 0x7fffffu) | 0x800000u, shift = 126u - e;
    u32 q = mant >> shift, rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u)))
        q++;
    return (u16)(sign | q);
}

/* FP64 -> FP16 with a single rounding (NumPy's float64.astype(float16)): round to odd
 * into FP32 first, whose 24 bits leave enough guard bits that the RNE step to FP16 is
 * the correctly rounded result. */
static inline u16 d2h(double d) {
    if (d != d)
        return 0x7e00u;
    double a = d < 0.0 ? -d : d;
    if (a >= 65520.0)
        return (u16)((d < 0.0 ? 0x8000u : 0u) | 0x7c00u);
    float f = (float)d;
    if ((double)f != d) {
        u32 u = bits(f);
        double af = f < 0.0f ? -(double)f : (double)f;
        if (af > a)
            u -= 1u;                   /* truncate: one step toward zero */
        f = from_bits(u | 1u);         /* sticky bit: odd */
    }
    return f2h(f);
}

/* FP16 -> FP32, exact. */
static inline float h2f(u16 h) {
    u32 s = (u32)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0) {
        float v = (float)m * 5.9604644775390625e-8f;  /* m * 2^-24, exact */
        return from_bits(bits(v) | s);
    }
    if (e == 31)
        return from_bits(s | 0x7f800000u | (m << 13));
    return from_bits(s | ((e + 112u) << 23) | (m << 13));
}

static inline float relu6(float v) { return v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v); }

#endif
