;
;  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;


%include "vpx_ports/x86_abi_support.asm"

SECTION .text

;int vp8_block_error_sse2(short *coeff_ptr,  short *dcoef_ptr)
globalsym(vp8_block_error_sse2)
sym(vp8_block_error_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 2
    push rsi
    push rdi
    ; end prologue

        mov         rsi,        arg(0) ;coeff_ptr
        mov         rdi,        arg(1) ;dcoef_ptr

        movdqa      xmm0,       [rsi]
        movdqa      xmm1,       [rdi]

        movdqa      xmm2,       [rsi+16]
        movdqa      xmm3,       [rdi+16]

        psubw       xmm0,       xmm1
        psubw       xmm2,       xmm3

        pmaddwd     xmm0,       xmm0
        pmaddwd     xmm2,       xmm2

        paddd       xmm0,       xmm2

        pxor        xmm5,       xmm5
        movdqa      xmm1,       xmm0

        punpckldq   xmm0,       xmm5
        punpckhdq   xmm1,       xmm5

        paddd       xmm0,       xmm1
        movdqa      xmm1,       xmm0

        psrldq      xmm0,       8
        paddd       xmm0,       xmm1

        movq        rax,        xmm0

    pop rdi
    pop rsi
    ; begin epilog
    UNSHADOW_ARGS
    pop         rbp
    ret

;int vp8_mbblock_error_sse2_impl(short *coeff_ptr, short *dcoef_ptr, int dc);
globalsym(vp8_mbblock_error_sse2_impl)
sym(vp8_mbblock_error_sse2_impl):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 3
    SAVE_XMM 6
    push rsi
    push rdi
    ; end prolog


        mov         rsi,        arg(0) ;coeff_ptr
        pxor        xmm6,       xmm6

        mov         rdi,        arg(1) ;dcoef_ptr
        pxor        xmm4,       xmm4

        movd        xmm5,       dword ptr arg(2) ;dc
        por         xmm5,       xmm4

        pcmpeqw     xmm5,       xmm6
        mov         rcx,        16

.mberror_loop:
        movdqa      xmm0,       [rsi]
        movdqa      xmm1,       [rdi]

        movdqa      xmm2,       [rsi+16]
        movdqa      xmm3,       [rdi+16]


        psubw       xmm2,       xmm3
        pmaddwd     xmm2,       xmm2

        psubw       xmm0,       xmm1
        pand        xmm0,       xmm5

        pmaddwd     xmm0,       xmm0
        add         rsi,        32

        add         rdi,        32

        sub         rcx,        1
        paddd       xmm4,       xmm2

        paddd       xmm4,       xmm0
        jnz         .mberror_loop

        movdqa      xmm0,       xmm4
        punpckldq   xmm0,       xmm6

        punpckhdq   xmm4,       xmm6
        paddd       xmm0,       xmm4

        movdqa      xmm1,       xmm0
        psrldq      xmm0,       8

        paddd       xmm0,       xmm1
        movq        rax,        xmm0

    pop rdi
    pop rsi
    ; begin epilog
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;int vp8_mbuverror_sse2_impl(short *s_ptr, short *d_ptr);
globalsym(vp8_mbuverror_sse2_impl)
sym(vp8_mbuverror_sse2_impl):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 2
    push rsi
    push rdi
    ; end prolog


        mov             rsi,        arg(0) ;s_ptr
        mov             rdi,        arg(1) ;d_ptr

        mov             rcx,        16
        pxor            xmm3,       xmm3

.mbuverror_loop:

        movdqa          xmm1,       [rsi]
        movdqa          xmm2,       [rdi]

        psubw           xmm1,       xmm2
        pmaddwd         xmm1,       xmm1

        paddd           xmm3,       xmm1

        add             rsi,        16
        add             rdi,        16

        dec             rcx
        jnz             .mbuverror_loop

        pxor        xmm0,           xmm0
        movdqa      xmm1,           xmm3

        movdqa      xmm2,           xmm1
        punpckldq   xmm1,           xmm0

        punpckhdq   xmm2,           xmm0
        paddd       xmm1,           xmm2

        movdqa      xmm2,           xmm1

        psrldq      xmm1,           8
        paddd       xmm1,           xmm2

        movq            rax,            xmm1

    pop rdi
    pop rsi
    ; begin epilog
    UNSHADOW_ARGS
    pop         rbp
    ret
