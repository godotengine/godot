@ This file was created from a .asm file
@  using the ads2gas.pl script.
	.equ DO1STROUNDING, 0
@
@  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
@
@  Use of this source code is governed by a BSD-style license
@  that can be found in the LICENSE file in the root of the source
@  tree. An additional intellectual property rights grant can be found
@  in the file PATENTS.  All contributing project authors may
@  be found in the AUTHORS file in the root of the source tree.
@


    .global vpx_push_neon 
	.type vpx_push_neon, function
    .global vpx_pop_neon 
	.type vpx_pop_neon, function

   .arm
   .eabi_attribute 24, 1 @Tag_ABI_align_needed
   .eabi_attribute 25, 1 @Tag_ABI_align_preserved

.text
.p2align 2

_vpx_push_neon:
	vpx_push_neon: @ PROC
    vst1.i64            {d8, d9, d10, d11}, [r0]!
    vst1.i64            {d12, d13, d14, d15}, [r0]!
    bx              lr

	.size vpx_push_neon, .-vpx_push_neon    @ ENDP

_vpx_pop_neon:
	vpx_pop_neon: @ PROC
    vld1.i64            {d8, d9, d10, d11}, [r0]!
    vld1.i64            {d12, d13, d14, d15}, [r0]!
    bx              lr

	.size vpx_pop_neon, .-vpx_pop_neon    @ ENDP


	.section	.note.GNU-stack,"",%progbits
