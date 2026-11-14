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

; tabulate_ssim - sums sum_s,sum_r,sum_sq_s,sum_sq_r, sum_sxr
%macro TABULATE_SSIM 0
        paddusw         xmm15, xmm3  ; sum_s
        paddusw         xmm14, xmm4  ; sum_r
        movdqa          xmm1, xmm3
        pmaddwd         xmm1, xmm1
        paddd           xmm13, xmm1 ; sum_sq_s
        movdqa          xmm2, xmm4
        pmaddwd         xmm2, xmm2
        paddd           xmm12, xmm2 ; sum_sq_r
        pmaddwd         xmm3, xmm4
        paddd           xmm11, xmm3  ; sum_sxr
%endmacro

; Sum across the register %1 starting with q words
%macro SUM_ACROSS_Q 1
        movdqa          xmm2,%1
        punpckldq       %1,xmm0
        punpckhdq       xmm2,xmm0
        paddq           %1,xmm2
        movdqa          xmm2,%1
        punpcklqdq      %1,xmm0
        punpckhqdq      xmm2,xmm0
        paddq           %1,xmm2
%endmacro

; Sum across the register %1 starting with q words
%macro SUM_ACROSS_W 1
        movdqa          xmm1, %1
        punpcklwd       %1,xmm0
        punpckhwd       xmm1,xmm0
        paddd           %1, xmm1
        SUM_ACROSS_Q    %1
%endmacro

SECTION .text

;void vpx_ssim_parms_8x8_sse2(
;    unsigned char *s,
;    int sp,
;    unsigned char *r,
;    int rp
;    uint32_t *sum_s,
;    uint32_t *sum_r,
;    uint32_t *sum_sq_s,
;    uint32_t *sum_sq_r,
;    uint32_t *sum_sxr);
;
; TODO: Use parm passing through structure, probably don't need the pxors
; ( calling app will initialize to 0 ) could easily fit everything in sse2
; without too much hastle, and can probably do better estimates with psadw
; or pavgb At this point this is just meant to be first pass for calculating
; all the parms needed for 16x16 ssim so we can play with dssim as distortion
; in mode selection code.
globalsym(vpx_ssim_parms_8x8_sse2)
sym(vpx_ssim_parms_8x8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 9
    SAVE_XMM 15
    push        rsi
    push        rdi
    ; end prolog

    mov             rsi,        arg(0) ;s
    mov             rcx,        arg(1) ;sp
    mov             rdi,        arg(2) ;r
    mov             rax,        arg(3) ;rp

    pxor            xmm0, xmm0
    pxor            xmm15,xmm15  ;sum_s
    pxor            xmm14,xmm14  ;sum_r
    pxor            xmm13,xmm13  ;sum_sq_s
    pxor            xmm12,xmm12  ;sum_sq_r
    pxor            xmm11,xmm11  ;sum_sxr

    mov             rdx, 8      ;row counter
.NextRow:

    ;grab source and reference pixels
    movq            xmm3, [rsi]
    movq            xmm4, [rdi]
    punpcklbw       xmm3, xmm0 ; low_s
    punpcklbw       xmm4, xmm0 ; low_r

    TABULATE_SSIM

    add             rsi, rcx   ; next s row
    add             rdi, rax   ; next r row

    dec             rdx        ; counter
    jnz .NextRow

    SUM_ACROSS_W    xmm15
    SUM_ACROSS_W    xmm14
    SUM_ACROSS_Q    xmm13
    SUM_ACROSS_Q    xmm12
    SUM_ACROSS_Q    xmm11

    mov             rdi,arg(4)
    movd            [rdi], xmm15;
    mov             rdi,arg(5)
    movd            [rdi], xmm14;
    mov             rdi,arg(6)
    movd            [rdi], xmm13;
    mov             rdi,arg(7)
    movd            [rdi], xmm12;
    mov             rdi,arg(8)
    movd            [rdi], xmm11;

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
