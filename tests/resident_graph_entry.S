/* Resident graph cache entry v1. Context params[0] selects 0=uncached,
 * 1=cached code; every persistent data pointer remains uncached. Public SDK
 * payloads are unchanged. Cache publication uses only already-saved VTCM.
 * Worker-0 status14 arms before cached stores; status15 confirms cleanup on
 * normal and guarded fault return, before code/power may be reclaimed.
 * Tail jumps preserve the uncached entry return address. Forwarding stubs
 * preserve the original r0 target and clobber no additional registers.
 */
#define FNPU_ENTRY_FRAME_BYTES 80
.section .text.entry,"ax",@progbits
.global custom_job
custom_job:
 { allocframe(#FNPU_ENTRY_FRAME_BYTES) }
 { memd(r29+#0) = r17:16 }
 { memd(r29+#8) = r19:18 }
 { memd(r29+#16) = r21:20 }
 { memd(r29+#24) = r23:22 }
 { memd(r29+#32) = r25:24 }
 { memd(r29+#40) = r27:26 }
 { r0 = htid }
 { p0 = cmp.eq(r0,#4) }
 { if (p0) jump direct_master }
 { p0 = cmp.gtu(r0,#2) }
 { if (p0) jump skip }
 { r16 = ##0xf0a93400 }
 { r16 = addasl(r16,r0,#6) }
 { memw(r16+#4) = r0 }
 { r17 = ssr }
 { memw(r16+#28) = r17 }
 { p0 = cmp.eq(r0,#0) }
 { if (!p0) jump finish }
 { r18 = evb }
 { memw(r16+#32) = r18 }
 { memw(r16+#24) = r29 }
 { r0 = ##vectors }
 { evb = r0 }
 { isync }
 { r19 = ##0xf0a93000 }
 { r0 = memw(r19+#12) }
 { p0 = tstbit(r0,#0) }
 { if (!p0) jump execute }
 { r0 = ##0x31000000 }
 { r1 = ##0xf0a94000 }
 { r2 = #64 }
backup_loop:
 { v0 = vmem(r0+#0) }
 { vmem(r1+#0) = v0 }
 { r0 = add(r0,#128) }
 { r1 = add(r1,#128) }
 { r2 = add(r2,#-1) }
 { p0 = cmp.eq(r2,#0) }
 { if (!p0) jump backup_loop }
 { barrier }
 { r0 = #1 }
 { memw(r16+#36) = r0 }
 { r0 = setbit(r17,#26) }
 { ssr = r0 }
 { isync }
execute:
 { r1:0 = c31:30 }
 { memd(r16+#40) = r1:0 }
 { r0 = r19 }
 { call resident_dispatch }
 { memw(r16+#8) = r0 }
recover:
 { r16 = ##0xf0a93400 }
 { r0 = memw(r16+#56) }
 { p0 = cmp.eq(r0,#0) }
 { if (p0) jump resident_cache_recovered }
 { call resident_cache_cleanup }
resident_cache_recovered:
 { r1:0 = c31:30 }
 { memd(r16+#48) = r1:0 }
 { r29 = memw(r16+#24) }
 /* A synchronous fault abandons C frames. deallocframe reads through FP, so
  * restoring SP alone can return into an abandoned frame and publish twice. */
 { r30 = add(r29,#FNPU_ENTRY_FRAME_BYTES) }
 { r17 = memw(r16+#28) }
 { r18 = memw(r16+#32) }
 { r0 = memw(r16+#36) }
 { p0 = cmp.eq(r0,#0) }
 { if (p0) jump registers }
 { r0 = ##0x31000000 }
 { r1 = ##0xf0a94000 }
 { r2 = #64 }
restore_loop:
 { v0 = vmem(r1+#0) }
 { vmem(r0+#0) = v0 }
 { r0 = add(r0,#128) }
 { r1 = add(r1,#128) }
 { r2 = add(r2,#-1) }
 { p0 = cmp.eq(r2,#0) }
 { if (!p0) jump restore_loop }
 { barrier }
registers:
 { evb = r18 }
 { ssr = r17 }
 { isync }
finish:
 { barrier }
 { r0 = r16 }
 { r1 = ##0xf0a93000 }
 { r1 = memw(r1+#0) }
 { r17:16 = memd(r29+#0) }
 { r19:18 = memd(r29+#8) }
 { r21:20 = memd(r29+#16) }
 { r23:22 = memd(r29+#24) }
 { r25:24 = memd(r29+#32) }
 { r27:26 = memd(r29+#40) }
 { deallocframe }
 { memw(r0+#0) = r1; jumpr r31 }
/* The patched CPU-request case calls here on context 4. Keep HVX/HMX on
 * worker 0: the master only wakes workers in the callback-only mode and joins
 * their final return packets before publishing stock completion. Never return
 * just because the host went away or a deadline elapsed while work is active.
 */
direct_master:
 { r16 = ##0xf0b12000 }
 { r17 = ##0xf0a93400 }
 { r18 = ##0xf0a93000 }
 { r19 = memw(r18+#0) }
 { r0 = memw(r16+#24) }
 { r0 = clrbit(r0,#0) }
 { memw(r16+#24) = r0 }
 { memw(r16+#0) = #6 }
 { barrier }
 { r0 = #7 }
 { resume(r0) }
direct_join:
 { r0 = memw(r17+#0) }
 { p0 = cmp.eq(r0,r19) }
 { if (!p0) jump direct_pause }
 { r0 = memw(r17+#64) }
 { p0 = cmp.eq(r0,r19) }
 { if (!p0) jump direct_pause }
 { r0 = memw(r17+#128) }
 { p0 = cmp.eq(r0,r19) }
 { if (!p0) jump direct_pause }
 { barrier }
 { memw(r16+#0) = #3 }
 { jump skip }
direct_pause:
 { pause(#64) }
 { jump direct_join }
skip:
 { r17:16 = memd(r29+#0) }
 { r19:18 = memd(r29+#8) }
 { r21:20 = memd(r29+#16) }
 { r23:22 = memd(r29+#24) }
 { r25:24 = memd(r29+#32) }
 { r27:26 = memd(r29+#40) }
 { dealloc_return }
.balign 256
vectors:
 { jump forward0 }
 { jump forward1 }
 { jump fault }
 { jump forward3 }
 { jump fault }
 { jump forward5 }
 { jump fault }
 { jump forward7 }
 { jump fault }
 { jump forward9 }
 { jump forward10 }
 { jump forward11 }
 { jump forward12 }
 { jump forward13 }
 { jump forward14 }
 { jump forward15 }
 { jump forward16 }
 { jump forward17 }
 { jump forward18 }
 { jump forward19 }
 { jump forward20 }
 { jump forward21 }
 { jump forward22 }
 { jump forward23 }
 { jump forward24 }
 { jump forward25 }
 { jump forward26 }
 { jump forward27 }
 { jump forward28 }
 { jump forward29 }
 { jump forward30 }
 { jump forward31 }
 { jump forward32 }
 { jump forward33 }
 { jump forward34 }
 { jump forward35 }
 { jump forward36 }
 { jump forward37 }
 { jump forward38 }
 { jump forward39 }
 { jump forward40 }
 { jump forward41 }
 { jump forward42 }
 { jump forward43 }
 { jump forward44 }
 { jump forward45 }
 { jump forward46 }
 { jump forward47 }
 { jump forward48 }
 { jump forward49 }
 { jump forward50 }
 { jump forward51 }
 { jump forward52 }
 { jump forward53 }
 { jump forward54 }
 { jump forward55 }
 { jump forward56 }
 { jump forward57 }
 { jump forward58 }
 { jump forward59 }
 { jump forward60 }
 { jump forward61 }
 { jump forward62 }
 { jump forward63 }
/* Each target receives exactly the original r0 = EVB + vector*4. */
.macro resident_forward index
forward\index:
 { r0 = #\index; jump resident_forward_common }
.endm
 resident_forward 0
 resident_forward 1
 resident_forward 3
 resident_forward 5
 resident_forward 7
 resident_forward 9
 resident_forward 10
 resident_forward 11
 resident_forward 12
 resident_forward 13
 resident_forward 14
 resident_forward 15
 resident_forward 16
 resident_forward 17
 resident_forward 18
 resident_forward 19
 resident_forward 20
 resident_forward 21
 resident_forward 22
 resident_forward 23
 resident_forward 24
 resident_forward 25
 resident_forward 26
 resident_forward 27
 resident_forward 28
 resident_forward 29
 resident_forward 30
 resident_forward 31
 resident_forward 32
 resident_forward 33
 resident_forward 34
 resident_forward 35
 resident_forward 36
 resident_forward 37
 resident_forward 38
 resident_forward 39
 resident_forward 40
 resident_forward 41
 resident_forward 42
 resident_forward 43
 resident_forward 44
 resident_forward 45
 resident_forward 46
 resident_forward 47
 resident_forward 48
 resident_forward 49
 resident_forward 50
 resident_forward 51
 resident_forward 52
 resident_forward 53
 resident_forward 54
 resident_forward 55
 resident_forward 56
 resident_forward 57
 resident_forward 58
 resident_forward 59
 resident_forward 60
 resident_forward 61
 resident_forward 62
 resident_forward 63
resident_forward_common:
 { r0 = asl(r0,#2) }
 { r0 = add(r0,##0x87000000) }
 { jumpr r0 }
.balign 16
fault:
 { r0 = ##0xf0a93400 }
 { r1 = #2 }
 { memw(r0+#8) = r1 }
 { r1 = ssr }
 { memw(r0+#12) = r1 }
 { r1 = elr }
 { memw(r0+#16) = r1 }
 { r1 = badva }
 { memw(r0+#20) = r1 }
 { r1 = ##resident_cache_fault_return }
 { elr = r1 }
 { rte }

.balign 16
resident_cache_fault_return:
 { r16 = ##0xf0a93400 }
 { r0 = #2 }
 { memw(r16+#8) = r0 }
 { jump recover }

/* Called only from uncached recover, with the entry return saved by allocframe.
 * Its r31 is disposable; deallocframe restores the firmware caller's return. */
resident_cache_cleanup:
 { r0 = ##0x87a92000 }
 { r1 = #128 }
resident_cache_clean_loop:
 { dccleaninva(r0) }
 { syncht }
 { icinva(r0) }
 { isync }
 { r0 = add(r0,#32) }
 { r1 = add(r1,#-1) }
 { p0 = cmp.eq(r1,#0) }
 { if (!p0) jump resident_cache_clean_loop }
 { barrier }
 { memw(r16+#60) = #1 }
 { jumpr r31 }

/* No nested call: retain the uncached entry's return address in r31, then
 * tail-jump to frame_job through the selected instruction alias. */
resident_dispatch:
 { r1 = memw(r19+#40) }
 { p0 = cmp.eq(r1,#1) }
 { if (!p0) jump frame_job }
 { r1 = memw(r19+#12) }
 { p0 = tstbit(r1,#0) }
 { if (!p0) jump frame_job }
 { r0 = ##0xf0a92000 }
 { r1 = ##0x31000000 }
 { r2 = #32 }
resident_stage_loop:
 { v0 = vmem(r0+#0) }
 { vmem(r1+#0) = v0 }
 { r0 = add(r0,#128) }
 { r1 = add(r1,#128) }
 { r2 = add(r2,#-1) }
 { p0 = cmp.eq(r2,#0) }
 { if (!p0) jump resident_stage_loop }
 { barrier }
 { memw(r16+#56) = #1 }
 { barrier }
 { r0 = ##0x87a92000 }
 { r1 = ##0x31000000 }
 { r2 = #32 }
resident_publish_vector:
 { v0 = vmem(r1+#0) }
 { vmem(r0+#0) = v0 }
 { barrier }
 { r3 = r0 }
 { r4 = #4 }
resident_publish_line:
 { dccleaninva(r3) }
 { syncht }
 { r3 = add(r3,#32) }
 { r4 = add(r4,#-1) }
 { p0 = cmp.eq(r4,#0) }
 { if (!p0) jump resident_publish_line }
 { r0 = add(r0,#128) }
 { r1 = add(r1,#128) }
 { r2 = add(r2,#-1) }
 { p0 = cmp.eq(r2,#0) }
 { if (!p0) jump resident_publish_vector }
 { r0 = ##0x87a92000 }
 { r1 = #128 }
resident_publish_instruction:
 { icinva(r0) }
 { isync }
 { r0 = add(r0,#32) }
 { r1 = add(r1,#-1) }
 { p0 = cmp.eq(r1,#0) }
 { if (!p0) jump resident_publish_instruction }
 { r0 = r19 }
 { r1 = ##(frame_job-0x69000000) }
 { jumpr r1 }
