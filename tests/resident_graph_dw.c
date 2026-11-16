#include "resident_graph_gather.h"
fnpu_u32 frame_job(const fnpu_context *c) {
    if(!rg_run(c,3)) return rg_finish(c,FNPU_BAD_ARGUMENT);
    const volatile fnpu_u32 *r=rg_layer(),*d=fnpu_input(c);
    const fnpu_u16 *weight=(const fnpu_u16 *)RG_PTR(r[16]);
    rg_accept(c);rg_u64 start=rg_ticks();
    for(fnpu_u32 n=0;n<d[19];n++) {
        fnpu_u32 channel=RG_STATE[21],y=RG_STATE[22],x=RG_STATE[23];
        fnpu_u32 count=r[9]-x;if(count>32) count=32;
        HVX_Vector sum=Q6_V_vsplat_R(((const volatile fnpu_u32 *)RG_PTR(r[17]))[channel]);
        HVX_Vector bad=Q6_V_vzero();
        for(fnpu_u32 ky=0;ky<r[10];ky++) for(fnpu_u32 kx=0;kx<r[11];kx++) {
            HVX_Vector input=rg_gather(r,channel,y,x,ky,kx,count);
            HVX_Vector coefficient=Q6_V_vsplat_R(half_bits(weight[(channel*r[10]+ky)*r[11]+kx]));
            HVX_Vector product=mulf(input,coefficient);
            bad=Q6_V_vor_VV(bad,nonfinite(product));
            sum=addf(sum,product);
        }
        bad=Q6_V_vor_VV(bad,nonfinite(sum));rg_bad(bad);
        if(r[18]) sum=clip6(sum);
        rg_store((float *)RG_PTR(r[3]+(channel*r[20]+y*r[9]+x)*4),count,sum);
        RG_STATE[20]++;RG_STATE[30]++;rg_next_row(r);
        rg_fault(c);
        if(RG_STATE[20]==r[24]) {rg_done(r);break;}
        if(rg_ticks()-start>96000) break;
    }
    return rg_finish(c,FNPU_OK);
}
