#include "resident_graph.h"
/* Loads may cross a row only while inside the same owned input tensor;
 * the row/count mask removes those lanes before arithmetic. */
static HVX_Vector rg_gather(const volatile fnpu_u32 *r,fnpu_u32 channel,
                            fnpu_u32 y,fnpu_u32 x,fnpu_u32 ky,fnpu_u32 kx,fnpu_u32 count) {
    int iy=(int)y*(int)r[12]-(int)r[14]+(int)ky;
    int ix=(int)x*(int)r[13]-(int)r[15]+(int)kx;
    if(iy<0||iy>=(int)r[5]||ix>=(int)r[6]) return Q6_V_vzero();
    fnpu_u32 stride=r[13];
    const float *in=(const float *)RG_PTR(r[1]);
    if(ix>=0) {
        fnpu_u32 base=channel*r[19]+iy*r[6]+ix;
        fnpu_u32 need=stride==1?32:64;
        if(base+need<=r[4]*r[19]) {
            HVX_Vector value=*(const volatile HVX_UVector *)(in+base);
            if(stride==2) value=Q6_V_lo_W(Q6_W_vdeal_VVR(*(const volatile HVX_UVector *)(in+base+32),value,-4));
            fnpu_u32 valid=r[6]-ix;
            if(stride==2) valid=(valid+1)>>1;
            if(valid>count) valid=count;
            /* vsetq(#128) wraps to zero; a full vector needs no mask. */
            return valid==32?value:Q6_V_vmux_QVV(Q6_Q_vsetq_R(valid*4),value,Q6_V_vzero());
        }
    }
    volatile float *tmp=(volatile float *)0x31001e00u;
    for(fnpu_u32 i=0;i<32;i++) {
        int col=ix+(int)(i*stride);
        tmp[i]=i<count&&col>=0&&col<(int)r[6]?in[channel*r[19]+iy*r[6]+col]:0.f;
    }
    return *(volatile HVX_Vector *)tmp;
}
static void rg_next_row(const volatile fnpu_u32 *r) {
    fnpu_u32 x=RG_STATE[23]+32,y=RG_STATE[22],ch=RG_STATE[21];
    if(x>=r[9]) {x=0;if(++y==r[8]) {y=0;ch++;}}
    RG_STATE[21]=ch;RG_STATE[22]=y;RG_STATE[23]=x;
}
static fnpu_u32 half_bits(fnpu_u32 h) {
    /* The pinned loader proves all depthwise coefficients normal or zero. */
    fnpu_u32 sign=(h&0x8000)<<16,mag=h&0x7fff;
    return sign|(mag?(mag<<13)+0x38000000u:0);
}
