/* Teacher linked operator image entry, at image offset zero. The slot loader
 * (tests/teacher_link_admin.c) tail-jumps here for opcodes 110 and 111 while
 * the image is sealed, with r0 = context and the uncached entry's return in
 * r31. tests/teacher_resident_probe.c (opcode 111, built as tl_rs) and
 * tests/teacher_dw_probe.c (opcode 110, built as tl_dw) are linked unchanged
 * in source; each still validates its own context, opcode and descriptors. */
#include "landmark_common.h"

fnpu_u32 tl_rs(const fnpu_context *c);
fnpu_u32 tl_dw(const fnpu_context *c);

__attribute__((section(".text.rg_entry"))) fnpu_u32 tl_entry(const fnpu_context *c) {
    return c->opcode == 111u ? tl_rs(c) : tl_dw(c);
}
