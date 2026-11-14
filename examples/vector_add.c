#include "framenpu.h"

fnpu_u32 frame_job(const fnpu_context *ctx) {
    fnpu_u32 n=ctx->params[0];
    if (ctx->magic!=FNPU_MAGIC || ctx->abi!=FNPU_ABI ||
        n>1024 || ctx->input_bytes!=n*8 || ctx->output_bytes!=n*4 ||
        ctx->input_offset>ctx->arena_bytes ||
        ctx->input_bytes>ctx->arena_bytes-ctx->input_offset ||
        ctx->output_offset>ctx->arena_bytes ||
        ctx->output_bytes>ctx->arena_bytes-ctx->output_offset)
        return FNPU_BAD_ARGUMENT;
    const float *in=fnpu_input(ctx);
    float *out=fnpu_output(ctx);
    for (fnpu_u32 i=0;i<n;i++) out[i]=in[i]+in[n+i];
    return FNPU_OK;
}
