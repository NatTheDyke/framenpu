/* Linked resident operator image: the ten compute modules as one read-only,
 * position-independent program in owned DDR (RG_CODE). The slot loader jumps
 * to rg_batch at image offset zero through the selected instruction alias.
 * One request may cross phase and layer boundaries. It stops between whole
 * work units at the host's tick budget or unit limit, at the host's
 * (layer, phase) stop point, at frame completion, or on a guarded fault.
 * Arithmetic, tensor layout and per-unit state changes are those of the
 * separately qualified modules; the host replays the unit count through an
 * independent state machine before any resource is reclaimed. */
#define RG_LINKED 1
#include "resident_graph.h"
fnpu_u32 rg_batch(const fnpu_context *c) __attribute__((section(".text.rg_entry")));
#define frame_job rg_op_hmx
#include "resident_graph_hmx.c"
#undef frame_job
#define frame_job rg_op_dw
#include "resident_graph_dw.c"
#undef frame_job
#define frame_job rg_op_add
#include "resident_graph_add.c"
#undef frame_job
#define frame_job rg_op_pool
#include "resident_graph_pool.c"
#undef frame_job
#define frame_job rg_op_sigmoid
#include "resident_graph_sigmoid.c"
#undef frame_job
#define frame_job rg_op_compact
#include "resident_graph_compact.c"
#undef frame_job
#define frame_job rg_op_post
#include "resident_graph_post.c"
#undef frame_job
#define frame_job rg_op_avg
#include "resident_graph_avg.c"
#undef frame_job
#define frame_job rg_op_pack
#define gather rg_pack_gather
#include "resident_graph_pack.c"
#undef frame_job
#undef gather
#undef RG_STEM
#define RG_STEM 1
#define frame_job rg_op_stem
#define gather rg_stem_gather
#include "resident_graph_pack.c"
#undef frame_job
#undef gather
fnpu_u32 rg_batch(const fnpu_context *c) {
    rg_open();
    if(!rg_header(c,RG_BATCH_KIND)) return rg_finish(c,FNPU_BAD_ARGUMENT);
    const volatile fnpu_u32 *d=fnpu_input(c);
    if(d[2]!=5||RG_STATE[0]!=3||RG_STATE[27]!=602112||RG_STATE[19]>=63||RG_STATE[31]!=RG_SEAL||
       d[18]!=RG_STATE[19]||!d[19]||d[19]>RG_UNIT_LIMIT||d[21]||d[22]||
       !d[23]||d[23]>RG_BUDGET_MAX||d[24]>63||d[25]>3) return rg_finish(c,FNPU_BAD_ARGUMENT);
    fnpu_u32 kind=rg_kind();
    if(d[20]!=(kind==8||kind==9||kind==10?RG_STATE[18]:RG_STATE[20])) return rg_finish(c,FNPU_BAD_ARGUMENT);
    fnpu_u32 stop_layer=d[24],stop_phase=d[25];
    rg_begin(c);
    RG_BATCH_BUDGET=d[23];RG_BATCH_UNITS=d[19];RG_BATCH_FAULT=d[30];RG_BATCH_START=rg_ticks();
    for(;;) {
        fnpu_u32 result;
        kind=rg_kind();
        if(kind==1) result=rg_op_hmx(c);
        else if(kind==3) result=rg_op_dw(c);
        else if(kind==4) result=rg_op_add(c);
        else if(kind==5) result=rg_op_pool(c);
        else if(kind==6) result=rg_op_sigmoid(c);
        else if(kind==7) result=rg_op_compact(c);
        else if(kind==8) result=rg_op_post(c);
        else if(kind==9) result=rg_op_pack(c);
        else if(kind==10) result=rg_op_stem(c);
        else if(kind==11) result=rg_op_avg(c);
        else result=FNPU_BAD_ARGUMENT;
        if(result) return rg_finish(c,result);
        if(RG_STATE[0]!=3||RG_STATE[30]>=RG_BATCH_UNITS||
           (RG_STATE[19]==stop_layer&&RG_STATE[24]==stop_phase)||
           rg_ticks()-RG_BATCH_START>RG_BATCH_BUDGET) break;
    }
    return rg_finish(c,FNPU_OK);
}
