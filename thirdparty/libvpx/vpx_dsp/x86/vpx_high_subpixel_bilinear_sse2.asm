;
;  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

%include "vpx_ports/x86_abi_support.asm"

%macro HIGH_GET_PARAM_4 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr
    mov         rcx, 0x00000040

    movdqa      xmm3, [rdx]                 ;load filters
    pshuflw     xmm4, xmm3, 11111111b       ;k3
    psrldq      xmm3, 8
    pshuflw     xmm3, xmm3, 0b              ;k4
    punpcklwd   xmm4, xmm3                  ;k3k4

    movq        xmm3, rcx                   ;rounding
    pshufd      xmm3, xmm3, 0

    mov         rdx, 0x00010001
    movsxd      rcx, DWORD PTR arg(6)       ;bd
    movq        xmm5, rdx
    movq        xmm2, rcx
    pshufd      xmm5, xmm5, 0b
    movdqa      xmm1, xmm5
    psllw       xmm5, xmm2
    psubw       xmm5, xmm1                  ;max value (for clamping)
    pxor        xmm2, xmm2                  ;min value (for clamping)

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
%endm

%macro HIGH_APPLY_FILTER_4 1

    punpcklwd   xmm0, xmm1                  ;two row in one register
    pmaddwd     xmm0, xmm4                  ;multiply the filter factors

    paddd       xmm0, xmm3                  ;rounding
    psrad       xmm0, 7                     ;shift
    packssdw    xmm0, xmm0                  ;pack to word

    ;clamp the values
    pminsw      xmm0, xmm5
    pmaxsw      xmm0, xmm2

%if %1
    movq        xmm1, [rdi]
    pavgw       xmm0, xmm1
%endif

    movq        [rdi], xmm0
    lea         rsi, [rsi + 2*rax]
    lea         rdi, [rdi + 2*rdx]
    dec         rcx
%endm

%if VPX_ARCH_X86_64
%macro HIGH_GET_PARAM 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr
    mov         rcx, 0x00000040

    movdqa      xmm6, [rdx]                 ;load filters

    pshuflw     xmm7, xmm6, 11111111b       ;k3
    pshufhw     xmm6, xmm6, 0b              ;k4
    psrldq      xmm6, 8
    punpcklwd   xmm7, xmm6                  ;k3k4k3k4k3k4k3k4

    movq        xmm4, rcx                   ;rounding
    pshufd      xmm4, xmm4, 0

    mov         rdx, 0x00010001
    movsxd      rcx, DWORD PTR arg(6)       ;bd
    movq        xmm8, rdx
    movq        xmm5, rcx
    pshufd      xmm8, xmm8, 0b
    movdqa      xmm1, xmm8
    psllw       xmm8, xmm5
    psubw       xmm8, xmm1                  ;max value (for clamping)
    pxor        xmm5, xmm5                  ;min value (for clamping)

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
%endm

%macro HIGH_APPLY_FILTER_8 1
    movdqa      xmm6, xmm0
    punpckhwd   xmm6, xmm1
    punpcklwd   xmm0, xmm1
    pmaddwd     xmm6, xmm7
    pmaddwd     xmm0, xmm7

    paddd       xmm6, xmm4                  ;rounding
    paddd       xmm0, xmm4                  ;rounding
    psrad       xmm6, 7                     ;shift
    psrad       xmm0, 7                     ;shift
    packssdw    xmm0, xmm6                  ;pack back to word

    ;clamp the values
    pminsw      xmm0, xmm8
    pmaxsw      xmm0, xmm5

%if %1
    movdqu      xmm1, [rdi]
    pavgw       xmm0, xmm1
%endif
    movdqu      [rdi], xmm0                 ;store the result

    lea         rsi, [rsi + 2*rax]
    lea         rdi, [rdi + 2*rdx]
    dec         rcx
%endm

%macro HIGH_APPLY_FILTER_16 1
    movdqa      xmm9, xmm0
    movdqa      xmm6, xmm2
    punpckhwd   xmm9, xmm1
    punpckhwd   xmm6, xmm3
    punpcklwd   xmm0, xmm1
    punpcklwd   xmm2, xmm3

    pmaddwd     xmm9, xmm7
    pmaddwd     xmm6, xmm7
    pmaddwd     xmm0, xmm7
    pmaddwd     xmm2, xmm7

    paddd       xmm9, xmm4                  ;rounding
    paddd       xmm6, xmm4
    paddd       xmm0, xmm4
    paddd       xmm2, xmm4

    psrad       xmm9, 7                     ;shift
    psrad       xmm6, 7
    psrad       xmm0, 7
    psrad       xmm2, 7

    packssdw    xmm0, xmm9                  ;pack back to word
    packssdw    xmm2, xmm6                  ;pack back to word

    ;clamp the values
    pminsw      xmm0, xmm8
    pmaxsw      xmm0, xmm5
    pminsw      xmm2, xmm8
    pmaxsw      xmm2, xmm5

%if %1
    movdqu      xmm1, [rdi]
    movdqu      xmm3, [rdi + 16]
    pavgw       xmm0, xmm1
    pavgw       xmm2, xmm3
%endif
    movdqu      [rdi], xmm0               ;store the result
    movdqu      [rdi + 16], xmm2          ;store the result

    lea         rsi, [rsi + 2*rax]
    lea         rdi, [rdi + 2*rdx]
    dec         rcx
%endm
%endif

SECTION .text

globalsym(vpx_highbd_filter_block1d4_v2_sse2)
sym(vpx_highbd_filter_block1d4_v2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM_4
.loop:
    movq        xmm0, [rsi]                 ;load src
    movq        xmm1, [rsi + 2*rax]

    HIGH_APPLY_FILTER_4 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

%if VPX_ARCH_X86_64
globalsym(vpx_highbd_filter_block1d8_v2_sse2)
sym(vpx_highbd_filter_block1d8_v2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 8
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu      xmm0, [rsi]                 ;0
    movdqu      xmm1, [rsi + 2*rax]         ;1

    HIGH_APPLY_FILTER_8 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d16_v2_sse2)
sym(vpx_highbd_filter_block1d16_v2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 9
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu        xmm0, [rsi]               ;0
    movdqu        xmm2, [rsi + 16]
    movdqu        xmm1, [rsi + 2*rax]       ;1
    movdqu        xmm3, [rsi + 2*rax + 16]

    HIGH_APPLY_FILTER_16 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
%endif

globalsym(vpx_highbd_filter_block1d4_v2_avg_sse2)
sym(vpx_highbd_filter_block1d4_v2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM_4
.loop:
    movq        xmm0, [rsi]                 ;load src
    movq        xmm1, [rsi + 2*rax]

    HIGH_APPLY_FILTER_4 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

%if VPX_ARCH_X86_64
globalsym(vpx_highbd_filter_block1d8_v2_avg_sse2)
sym(vpx_highbd_filter_block1d8_v2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 8
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu      xmm0, [rsi]                 ;0
    movdqu      xmm1, [rsi + 2*rax]         ;1

    HIGH_APPLY_FILTER_8 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d16_v2_avg_sse2)
sym(vpx_highbd_filter_block1d16_v2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 9
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu        xmm0, [rsi]               ;0
    movdqu        xmm1, [rsi + 2*rax]       ;1
    movdqu        xmm2, [rsi + 16]
    movdqu        xmm3, [rsi + 2*rax + 16]

    HIGH_APPLY_FILTER_16 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
%endif

globalsym(vpx_highbd_filter_block1d4_h2_sse2)
sym(vpx_highbd_filter_block1d4_h2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM_4
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqa      xmm1, xmm0
    psrldq      xmm1, 2

    HIGH_APPLY_FILTER_4 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

%if VPX_ARCH_X86_64
globalsym(vpx_highbd_filter_block1d8_h2_sse2)
sym(vpx_highbd_filter_block1d8_h2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 8
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqu      xmm1, [rsi + 2]

    HIGH_APPLY_FILTER_8 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d16_h2_sse2)
sym(vpx_highbd_filter_block1d16_h2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 9
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu      xmm0,   [rsi]               ;load src
    movdqu      xmm1,   [rsi + 2]
    movdqu      xmm2,   [rsi + 16]
    movdqu      xmm3,   [rsi + 18]

    HIGH_APPLY_FILTER_16 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
%endif

globalsym(vpx_highbd_filter_block1d4_h2_avg_sse2)
sym(vpx_highbd_filter_block1d4_h2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM_4
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqa      xmm1, xmm0
    psrldq      xmm1, 2

    HIGH_APPLY_FILTER_4 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

%if VPX_ARCH_X86_64
globalsym(vpx_highbd_filter_block1d8_h2_avg_sse2)
sym(vpx_highbd_filter_block1d8_h2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 8
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqu      xmm1, [rsi + 2]

    HIGH_APPLY_FILTER_8 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d16_h2_avg_sse2)
sym(vpx_highbd_filter_block1d16_h2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 9
    push        rsi
    push        rdi
    ; end prolog

    HIGH_GET_PARAM
.loop:
    movdqu      xmm0,   [rsi]               ;load src
    movdqu      xmm1,   [rsi + 2]
    movdqu      xmm2,   [rsi + 16]
    movdqu      xmm3,   [rsi + 18]

    HIGH_APPLY_FILTER_16 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
%endif
