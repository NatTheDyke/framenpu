/* Fixed, pinned resident protocol; all persistent pointers use uncached DDR.
 * Host revalidates the immutable program under the firmware gate and checks
 * every completion before reclamation. Data stays uncached; the opt-in cache
 * entry may execute this same image through the cached instruction alias. */
#ifndef FNPU_RESIDENT_GRAPH_H
#define FNPU_RESIDENT_GRAPH_H
#include "landmark_common.h"
typedef unsigned long long rg_u64;
#define RG_START 0x1a22000u
#define RG_END 0x2000000u
#define RG_CONTROL 0x1a23000u
#define RG_PROGRAM 0x1a24000u
#define RG_WEIGHTS 0x1a26000u
#define RG_WEIGHT_BYTES 2051904u
#define RG_ACT (RG_WEIGHTS+2051968u+128u)
#define RG_GUARD (RG_END-4096u)
#define RG_SCRATCH (RG_GUARD-45056u)
#define RG_STATE ((volatile fnpu_u32 *)0xf1a23080u)
#define RG_PTR(off) ((unsigned char *)(0xf0000000u+(off)))
#define RG_MAGIC 0x524c4d4bu
/* Linked execution (docs/RESIDENT_LINK.md): the ten compute operators form one
 * read-only image in owned DDR below the packed-input page. RG_LOADER builds
 * the slot-resident admin module that seals/retires it; RG_LINKED builds the
 * image itself. Without either macro every module compiles exactly as before. */
#define RG_CODE 0x1ff0000u
#define RG_CODE_BYTES 0x3000u
#define RG_SEAL 0x434f4445u
#if defined(RG_LINKED)||defined(RG_LOADER)
#define RG_LIMIT RG_CODE
#else
#define RG_LIMIT (RG_SCRATCH-128u)
#endif
#ifdef RG_LINKED
#define RG_OPCODE 32
#define RG_RESERVED 26
#define RG_BATCH_KIND 12u
#define RG_UNIT_LIMIT 0x100000u
#define RG_BUDGET_MAX 480000u
/* Request-local limits live in the already saved/restored VTCM page. */
#define RG_BATCH_START (*(volatile rg_u64 *)0x31001f00u)
#define RG_BATCH_BUDGET (*(volatile fnpu_u32 *)0x31001f08u)
#define RG_BATCH_UNITS (*(volatile fnpu_u32 *)0x31001f0cu)
#define RG_MORE(n,d) ((void)(n),(void)(d),RG_STATE[30]<RG_BATCH_UNITS)
#define RG_EXPIRED(start) ((void)(start),rg_ticks()-RG_BATCH_START>RG_BATCH_BUDGET)
#define RG_RETURN(c,result) ((void)(c),(fnpu_u32)(result))
#define RG_ONE() (RG_STATE[30]++)
#else
#define RG_OPCODE 31
#define RG_RESERVED 23
#define RG_MORE(n,d) ((n)<(d)[19])
#define RG_EXPIRED(start) (rg_ticks()-(start)>96000)
#define RG_RETURN(c,result) rg_finish(c,result)
#define RG_ONE() (RG_STATE[30]=1)
#endif
static rg_u64 rg_ticks(void) {
    rg_u64 t;__asm__ volatile("%0 = c31:30":"=r"(t)::"memory");return t;
}
static void rg_echo(const fnpu_context *c) {
    const volatile fnpu_u32 *d=fnpu_input(c);volatile fnpu_u32 *o=fnpu_output(c);
    for(fnpu_u32 i=0;i<32;i++) o[i]=d[i];
    fnpu_u32 pc;__asm__ volatile("%0 = pc":"=r"(pc)::"memory");o[31]=pc;
}
static __attribute__((noinline)) fnpu_u32 rg_finish(const fnpu_context *c,fnpu_u32 result) {
    volatile fnpu_u32 *o=fnpu_output(c);
    for(fnpu_u32 i=0;i<32;i++) o[32+i]=RG_STATE[i];
    __asm__ volatile("barrier" ::: "memory");return result;
}
static __attribute__((noinline)) int rg_header(const fnpu_context *c,fnpu_u32 kind) {
    if(!valid(c)||c->opcode!=RG_OPCODE||c->input_bytes!=128||c->output_bytes!=256) return 0;
    if(c->params[0]>1) return 0;
    for(fnpu_u32 i=1;i<6;i++) if(c->params[i]) return 0;
    rg_echo(c);
    const volatile fnpu_u32 *d=fnpu_input(c);
    if(d[0]!=RG_MAGIC||d[1]!=1||d[3]!=kind||d[30]>1) return 0;
    for(fnpu_u32 i=RG_RESERVED;i<32;i++) if(i!=30&&d[i]) return 0;
    if(d[2]==0) return kind==0 && !RG_STATE[0] && d[8]==1 && !d[9] && (d[4]|d[5]|d[6]|d[7]);
    if(!RG_STATE[0]||RG_STATE[0]==5) return 0;
    for(fnpu_u32 i=4;i<18;i++) if(i!=8&&i!=9&&RG_STATE[i]!=d[i]) return 0;
    rg_u64 seq=((rg_u64)RG_STATE[9]<<32)|RG_STATE[8];
    return seq!=~(rg_u64)0 && d[8]==(fnpu_u32)(seq+1) && d[9]==(fnpu_u32)((seq+1)>>32);
}
#ifdef RG_LINKED
/* The batch entry accepts the sequence once; operators must not reset it. */
static void rg_begin(const fnpu_context *c) {
    const volatile fnpu_u32 *d=fnpu_input(c);
    RG_STATE[8]=d[8];RG_STATE[9]=d[9];RG_STATE[29]=0;RG_STATE[30]=0;
}
static void rg_accept(const fnpu_context *c) {(void)c;}
#else
static void rg_accept(const fnpu_context *c) {
    const volatile fnpu_u32 *d=fnpu_input(c);
    RG_STATE[8]=d[8];RG_STATE[9]=d[9];RG_STATE[29]=0;RG_STATE[30]=0;
}
#endif
static const volatile fnpu_u32 *rg_layer(void) {
    return (const volatile fnpu_u32 *)RG_PTR(RG_PROGRAM+RG_STATE[19]*128);
}
static int rg_span(fnpu_u32 off,fnpu_u32 bytes) {
    return off>=RG_ACT && off<=RG_LIMIT && bytes<=RG_LIMIT-off;
}
#ifdef RG_LINKED
/* Module selected by resident state alone; identical to the single-module rule. */
static fnpu_u32 rg_kind(void) {
    const volatile fnpu_u32 *r=rg_layer();
    if(r[0]==1||r[0]==2) return RG_STATE[24]==0?(r[0]==1?9u:10u):RG_STATE[24]==1?1u:RG_STATE[24]==3?8u:6u;
    if(RG_STATE[24]) return 7;
    if(r[0]==5&&r[23]==2) return 11;
    return r[0];
}
/* rg_batch validated the descriptor once; each operator still proves that it
 * is the module the resident state expects and that its tensors are owned. */
static __attribute__((noinline)) int rg_run(const fnpu_context *c,fnpu_u32 kind) {
    (void)c;
    if(RG_STATE[0]!=3||RG_STATE[19]>=63) return 0;
    const volatile fnpu_u32 *r=rg_layer();
    return kind==rg_kind()&&rg_span(r[1],r[21])&&rg_span(r[3],r[22]);
}
#else
static __attribute__((noinline)) int rg_run(const fnpu_context *c,fnpu_u32 kind) {
    if(!rg_header(c,kind)) return 0;
    const volatile fnpu_u32 *d=fnpu_input(c);
    if(d[2]!=5||RG_STATE[0]!=3||RG_STATE[27]!=602112||RG_STATE[19]>=63||
       d[18]!=RG_STATE[19]||!d[19]||d[19]>(kind==8||kind==9||kind==10?256u:32u)||d[21]||d[22]) return 0;
    const volatile fnpu_u32 *r=rg_layer();
    fnpu_u32 expected=r[0];
    if(r[0]==1||r[0]==2) expected=RG_STATE[24]==0?(r[0]==1?9u:10u):RG_STATE[24]==1?1u:RG_STATE[24]==3?8u:6u;
    else if(RG_STATE[24]) expected=7;
    else if(r[0]==5&&r[23]==2) expected=11;
    if(d[20]!=(kind==8||kind==9||kind==10?RG_STATE[18]:RG_STATE[20])) return 0;
    /* Shape/stride products were validated by the admin module before READY.
     * The host checks the immutable program hash under each request's gate. */
    if(kind!=expected||!rg_span(r[1],r[21])||!rg_span(r[3],r[22])) return 0;
    return 1;
}
#endif
static void rg_bad(HVX_Vector v) {
    volatile HVX_Vector *scratch=(volatile HVX_Vector *)0x31001c00u;
    *scratch=nonfinite(v);
    const volatile fnpu_u32 *words=(const volatile fnpu_u32 *)scratch;
    fnpu_u32 bits=0;
    for(fnpu_u32 i=0;i<32;i++) bits|=words[i];
    RG_STATE[29]|=bits;
}
static void rg_fault(const fnpu_context *c) {
    const volatile fnpu_u32 *d=fnpu_input(c);
    if(d[30]) {
        const volatile fnpu_u32 *r=rg_layer();
        *(volatile fnpu_u32 *)RG_PTR(r[3])=0xdecafbadu;
        RG_STATE[0]=4;
        rg_finish(c,FNPU_EXCEPTION);
        __asm__ volatile("barrier\n\ttrap0(#37)" ::: "memory");
        ((volatile fnpu_u32 *)fnpu_output(c))[30]=2;
    }
}
static void rg_done(const volatile fnpu_u32 *r) {
    RG_STATE[20]=RG_STATE[21]=RG_STATE[22]=RG_STATE[23]=0;
    if(!RG_STATE[24]&&r[23]==1) RG_STATE[24]=1;
    else {
        RG_STATE[24]=0;RG_STATE[19]++;
        if(RG_STATE[19]==63) RG_STATE[0]=2;
    }
}
static HVX_Vector rg_load(const float *p,fnpu_u32 count) {
    if(count==32) return *(const volatile HVX_UVector *)p;
    volatile float *tmp=(volatile float *)0x31001d00u;
    for(fnpu_u32 i=0;i<32;i++) tmp[i]=i<count?p[i]:0.f;
    return *(volatile HVX_Vector *)tmp;
}
static void rg_store(float *p,fnpu_u32 count,HVX_Vector v) {
    if(count==32) *(volatile HVX_UVector *)p=v;
    else {
        volatile float *tmp=(volatile float *)0x31001d00u;
        *(volatile HVX_Vector *)tmp=v;
        for(fnpu_u32 i=0;i<count;i++) p[i]=tmp[i];
    }
}
#endif
