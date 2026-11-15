/* Diagnostic cached/uncached instruction-fetch A/B. Stock SDK entry and power.
 * params[2]=0: uncached, 1: cached. Other params and layout are hmx_baseline.c.
 * Run publication only after the SDK has installed its guarded window and
 * saved 8 KiB VTCM. Stage all 4 KiB of fresh code before any cached writes:
 * dcinva also cleans, so it cannot safely discard dirty aliases after upload.
 * Full 128-byte cached stores precede clean/invalidate of their cache lines.
 * The existing baseline then overwrites this temporary VTCM staging with input.
 * Successful cached execution returns to an uncached outer frame, which
 * invalidates its cached code before normal host DDR restoration.
 * Experimental only: a fault can bypass this C cleanup; do not promote this
 * entry protocol to the public SDK without fault-path cache qualification.
 */
#define frame_job baseline_frame_job
#include "hmx_baseline.c"
#undef frame_job

#define UNCACHED_CODE 0xf0a92000u
#define CACHED_CODE 0x87a92000u
#define ALIAS_DELTA 0x69000000u

static __attribute__((noinline)) void invalidate_code(void) {
    for (fnpu_u32 offset = 0; offset < 4096; offset += 32) {
        fnpu_u32 address = CACHED_CODE + offset;
        __asm__ volatile("dccleaninva(%0)\n\tsyncht\n\ticinva(%0)\n\tisync"
                         :: "r"(address) : "memory");
    }
}

static __attribute__((noinline)) void publish_cached(void) {
    copy_vectors((void *)0x31000000u, (const void *)UNCACHED_CODE, 32);
    for (fnpu_u32 i = 0; i < 32; i++) {
        ((volatile HVX_Vector *)CACHED_CODE)[i] = ((const volatile HVX_Vector *)0x31000000u)[i];
        __asm__ volatile("barrier" ::: "memory");
        for (fnpu_u32 part = 0; part < 128; part += 32) {
            fnpu_u32 address = CACHED_CODE + i * 128 + part;
            __asm__ volatile("dccleaninva(%0)\n\tsyncht" :: "r"(address) : "memory");
        }
    }
    for (fnpu_u32 offset = 0; offset < 4096; offset += 32) {
        fnpu_u32 address = CACHED_CODE + offset;
        __asm__ volatile("icinva(%0)\n\tisync" :: "r"(address) : "memory");
    }
}

fnpu_u32 frame_job(const fnpu_context *ctx) {
    fnpu_u32 pc;
    __asm__ volatile("%0 = pc" : "=r"(pc) :: "memory");
    if (ctx->params[2] > 1) return FNPU_BAD_ARGUMENT;
    if (ctx->params[2] && pc >= UNCACHED_CODE && pc < UNCACHED_CODE + 4096) {
        publish_cached();
        fnpu_u32 target = (fnpu_u32)frame_job;
        /* Keep alias arithmetic opaque: the compiler can otherwise recover
         * the original function identity and emit an uncached recursive call. */
        __asm__ volatile("%0 = sub(%0,%1)" : "+r"(target) : "r"(ALIAS_DELTA) : "memory");
        fnpu_u32 (*cached)(const fnpu_context *) =
            (fnpu_u32 (*)(const fnpu_context *))target;
        fnpu_u32 result = cached(ctx);
        invalidate_code();
        return result;
    }
    fnpu_u32 result = baseline_frame_job(ctx);
    if (result == FNPU_OK) ((volatile fnpu_u32 *)fnpu_output(ctx))[7] = pc;
    return result;
}
