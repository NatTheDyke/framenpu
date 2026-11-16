#include "resident_graph.h"
fnpu_u32 frame_job(const fnpu_context *c) {
    if(!rg_run(c,11)) return rg_finish(c,FNPU_BAD_ARGUMENT);
    const volatile fnpu_u32 *r=rg_layer(),*d=fnpu_input(c);
    if(r[23]!=2||r[19]!=49||r[8]!=1||r[9]!=1) return rg_finish(c,FNPU_BAD_ARGUMENT);
    rg_accept(c);rg_u64 start=rg_ticks();
    for(fnpu_u32 n=0;n<d[19];n++) {
        fnpu_u32 first=RG_STATE[20]*32,count=r[7]-first;if(count>32) count=32;
        HVX_Vector sum=Q6_V_vzero(),bad=Q6_V_vzero();
        volatile float *tmp=(volatile float *)0x31001e00u;
        const float *in=(const float *)RG_PTR(r[1]);
        for(fnpu_u32 t=0;t<49;t++) {
            for(fnpu_u32 i=0;i<32;i++) tmp[i]=i<count?in[(first+i)*49+t]:0.f;
            HVX_Vector v=*(volatile HVX_Vector *)tmp;
            bad=Q6_V_vor_VV(bad,nonfinite(v));sum=addf(sum,v);
        }
        union {float f;fnpu_u32 u;} scale;scale.f=1.f/49.f;
        sum=mulf(sum,Q6_V_vsplat_R(scale.u));bad=Q6_V_vor_VV(bad,nonfinite(sum));rg_bad(bad);
        rg_store((float *)RG_PTR(r[3]+first*4),count,sum);
        RG_STATE[20]++;RG_STATE[30]++;rg_fault(c);
        if(RG_STATE[20]==r[24]) {rg_done(r);break;}
        if(rg_ticks()-start>96000) break;
    }
    return rg_finish(c,FNPU_OK);
}
