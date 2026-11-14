#include "framenpu.h"

void fnpu_hmx_clear(void);
void fnpu_hmx_load(const fnpu_u16 *, const fnpu_u16 *);
void fnpu_hmx_store(fnpu_u16 *, const void *);

static float from_half(fnpu_u16 h) {
    union { fnpu_u32 u; float f; } v;
    fnpu_u32 sign = (h & 0x8000u) << 16;
    fnpu_u32 exponent = (h >> 10) & 31;
    fnpu_u32 fraction = h & 1023u;
    if (!exponent) {
        if (!fraction) { v.u = sign; return v.f; }
        int shift = 0;
        while (!(fraction & 1024u)) { fraction <<= 1; shift++; }
        v.u = sign | ((113u - shift) << 23) | ((fraction & 1023u) << 13);
    } else {
        v.u = sign | ((exponent == 31 ? 255u : exponent + 112u) << 23) | (fraction << 13);
    }
    return v.f;
}

static fnpu_u32 packed(fnpu_u32 row, fnpu_u32 col) {
    return (row >> 1) * 64 + col * 2 + (row & 1);
}

static fnpu_u32 gemm(const fnpu_context *ctx) {
    fnpu_u32 m = ctx->params[0], k = ctx->params[1], n = ctx->params[2];
    fnpu_u32 relu = ctx->params[3];
    if (!m || !k || !n || m > 32 || k > 32 || n > 32 || relu > 1 ||
        !(ctx->flags & FNPU_FLAG_HMX))
        return FNPU_BAD_ARGUMENT;
    fnpu_u32 bias_offset = ((m*k + k*n)*2 + 3) & ~3u;
    if (ctx->input_bytes != bias_offset + n*4 || ctx->output_bytes != m*n*4)
        return FNPU_BAD_ARGUMENT;
    const fnpu_u16 *a = fnpu_input(ctx);
    const fnpu_u16 *b = a + m*k;
    const float *bias = (const float *)((const unsigned char *)a + bias_offset);
    float *out = fnpu_output(ctx);
    volatile fnpu_u16 *ta = (volatile fnpu_u16 *)0x31000000u;
    volatile fnpu_u16 *tb = (volatile fnpu_u16 *)0x31000800u;
    volatile fnpu_u16 *tc = (volatile fnpu_u16 *)0x31001000u;
    volatile fnpu_u32 *scales = (volatile fnpu_u32 *)0x31001800u;
    for (fnpu_u32 i=0;i<32;i++) scales[i]=0x3c00u;
    for (fnpu_u32 i=32;i<64;i++) scales[i]=0;
    for (fnpu_u32 i=0;i<m*n;i++) out[i]=0.0f;
    for (fnpu_u32 mr=0;mr<m;mr+=32) {
        for (fnpu_u32 nc=0;nc<n;nc+=32) {
            for (fnpu_u32 kb=0;kb<k;kb+=32) {
                for (fnpu_u32 r=0;r<32;r++) {
                    for (fnpu_u32 c=0;c<32;c++) {
                        ta[packed(r,c)] = mr+r<m && kb+c<k ? a[(mr+r)*k+kb+c] : 0;
                        tb[packed(r,c)] = kb+r<k && nc+c<n ? b[(kb+r)*n+nc+c] : 0;
                    }
                }
                __asm__ volatile("barrier" ::: "memory");
                fnpu_hmx_clear();
                fnpu_hmx_load((const fnpu_u16 *)ta, (const fnpu_u16 *)tb);
                fnpu_hmx_store((fnpu_u16 *)tc, (const void *)scales);
                /* Readback serializes completion before these VTCM tiles are reused.
                   Each depth-32 product rounds to FP16, then sums into FP32 DDR. */
                for (fnpu_u32 r=0;r<32 && mr+r<m;r++) {
                    for (fnpu_u32 c=0;c<32 && nc+c<n;c++) {
                        fnpu_u16 h=tc[packed(r,c)];
                        out[(mr+r)*n+nc+c] += from_half(h);
                    }
                }
            }
        }
    }
    for (fnpu_u32 r=0;r<m;r++) {
        for (fnpu_u32 c=0;c<n;c++) {
            float value=out[r*n+c]+bias[c];
            out[r*n+c]=relu && value<0.0f ? 0.0f : value;
        }
    }
    return FNPU_OK;
}

static fnpu_u32 elementwise(const fnpu_context *ctx) {
    fnpu_u32 count=ctx->params[0], clip=ctx->params[1];
    if (count>32768 || clip>1 || ctx->input_bytes!=count*4+8 || ctx->output_bytes!=count*4)
        return FNPU_BAD_ARGUMENT;
    const float *in=fnpu_input(ctx);
    float *out=fnpu_output(ctx);
    float low=clip ? in[count] : 0.0f;
    float high=in[count+1];
    for (fnpu_u32 i=0;i<count;i++) {
        float v=in[i];
        if (v<low) v=low;
        if (clip && v>high) v=high;
        out[i]=v;
    }
    return FNPU_OK;
}

static fnpu_u32 add(const fnpu_context *ctx) {
    fnpu_u32 count=ctx->params[0], bc=ctx->params[1], relu=ctx->params[2];
    if (!bc || count>32768 || bc>count || relu>1 ||
        ctx->input_bytes!=(count+bc)*4 || ctx->output_bytes!=count*4)
        return FNPU_BAD_ARGUMENT;
    const float *a=fnpu_input(ctx);
    const float *b=a+count;
    float *out=fnpu_output(ctx);
    fnpu_u32 cursor=0;
    for (fnpu_u32 i=0;i<count;i++) {
        float v=a[i]+b[cursor];
        cursor++;
        if (cursor==bc) cursor=0;
        out[i]=relu && v<0.0f ? 0.0f : v;
    }
    return FNPU_OK;
}

fnpu_u32 frame_job(const fnpu_context *ctx) {
    if (ctx->magic!=FNPU_MAGIC || ctx->abi!=FNPU_ABI ||
        ctx->arena_bytes>FNPU_ARENA_SIZE ||
        ctx->input_offset>ctx->arena_bytes ||
        ctx->input_bytes>ctx->arena_bytes-ctx->input_offset ||
        ctx->output_offset>ctx->arena_bytes ||
        ctx->output_bytes>ctx->arena_bytes-ctx->output_offset ||
        (ctx->input_offset&3) || (ctx->output_offset&3))
        return FNPU_BAD_ARGUMENT;
    if (ctx->opcode==1) return gemm(ctx);
    if (ctx->opcode==2) return elementwise(ctx);
    if (ctx->opcode==3) return add(ctx);
    return FNPU_UNSUPPORTED;
}
