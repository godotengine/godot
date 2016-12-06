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

%macro GET_PARAM_4 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr
    mov         rcx, 0x0400040

    movdqa      xmm3, [rdx]                 ;load filters
    pshuflw     xmm4, xmm3, 11111111b       ;k3
    psrldq      xmm3, 8
    pshuflw     xmm3, xmm3, 0b              ;k4
    punpcklqdq  xmm4, xmm3                  ;k3k4

    movq        xmm3, rcx                   ;rounding
    pshufd      xmm3, xmm3, 0

    pxor        xmm2, xmm2

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
%endm

%macro APPLY_FILTER_4 1

    punpckldq   xmm0, xmm1                  ;two row in one register
    punpcklbw   xmm0, xmm2                  ;unpack to word
    pmullw      xmm0, xmm4                  ;multiply the filter factors

    movdqa      xmm1, xmm0
    psrldq      xmm1, 8
    paddsw      xmm0, xmm1

    paddsw      xmm0, xmm3                  ;rounding
    psraw       xmm0, 7                     ;shift
    packuswb    xmm0, xmm0                  ;pack to byte

%if %1
    movd        xmm1, [rdi]
    pavgb       xmm0, xmm1
%endif

    movd        [rdi], xmm0
    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
%endm

%macro GET_PARAM 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr
    mov         rcx, 0x0400040

    movdqa      xmm7, [rdx]                 ;load filters

    pshuflw     xmm6, xmm7, 11111111b       ;k3
    pshufhw     xmm7, xmm7, 0b              ;k4
    punpcklwd   xmm6, xmm6
    punpckhwd   xmm7, xmm7

    movq        xmm4, rcx                   ;rounding
    pshufd      xmm4, xmm4, 0

    pxor        xmm5, xmm5

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
%endm

%macro APPLY_FILTER_8 1
    punpcklbw   xmm0, xmm5
    punpcklbw   xmm1, xmm5

    pmullw      xmm0, xmm6
    pmullw      xmm1, xmm7
    paddsw      xmm0, xmm1
    paddsw      xmm0, xmm4                  ;rounding
    psraw       xmm0, 7                     ;shift
    packuswb    xmm0, xmm0                  ;pack back to byte
%if %1
    movq        xmm1, [rdi]
    pavgb       xmm0, xmm1
%endif
    movq        [rdi], xmm0                 ;store the result

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
%endm

%macro APPLY_FILTER_16 1
    punpcklbw   xmm0, xmm5
    punpcklbw   xmm1, xmm5
    punpckhbw   xmm2, xmm5
    punpckhbw   xmm3, xmm5

    pmullw      xmm0, xmm6
    pmullw      xmm1, xmm7
    pmullw      xmm2, xmm6
    pmullw      xmm3, xmm7

    paddsw      xmm0, xmm1
    paddsw      xmm2, xmm3

    paddsw      xmm0, xmm4                  ;rounding
    paddsw      xmm2, xmm4
    psraw       xmm0, 7                     ;shift
    psraw       xmm2, 7
    packuswb    xmm0, xmm2                  ;pack back to byte
%if %1
    movdqu      xmm1, [rdi]
    pavgb       xmm0, xmm1
%endif
    movdqu      [rdi], xmm0                 ;store the result

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
%endm

global sym(vpx_filter_block1d4_v2_sse2) PRIVATE
sym(vpx_filter_block1d4_v2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM_4
.loop:
    movd        xmm0, [rsi]                 ;load src
    movd        xmm1, [rsi + rax]

    APPLY_FILTER_4 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d8_v2_sse2) PRIVATE
sym(vpx_filter_block1d8_v2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movq        xmm0, [rsi]                 ;0
    movq        xmm1, [rsi + rax]           ;1

    APPLY_FILTER_8 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d16_v2_sse2) PRIVATE
sym(vpx_filter_block1d16_v2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movdqu        xmm0, [rsi]               ;0
    movdqu        xmm1, [rsi + rax]         ;1
    movdqa        xmm2, xmm0
    movdqa        xmm3, xmm1

    APPLY_FILTER_16 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d4_v2_avg_sse2) PRIVATE
sym(vpx_filter_block1d4_v2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM_4
.loop:
    movd        xmm0, [rsi]                 ;load src
    movd        xmm1, [rsi + rax]

    APPLY_FILTER_4 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d8_v2_avg_sse2) PRIVATE
sym(vpx_filter_block1d8_v2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movq        xmm0, [rsi]                 ;0
    movq        xmm1, [rsi + rax]           ;1

    APPLY_FILTER_8 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d16_v2_avg_sse2) PRIVATE
sym(vpx_filter_block1d16_v2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movdqu        xmm0, [rsi]               ;0
    movdqu        xmm1, [rsi + rax]         ;1
    movdqa        xmm2, xmm0
    movdqa        xmm3, xmm1

    APPLY_FILTER_16 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d4_h2_sse2) PRIVATE
sym(vpx_filter_block1d4_h2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM_4
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqa      xmm1, xmm0
    psrldq      xmm1, 1

    APPLY_FILTER_4 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d8_h2_sse2) PRIVATE
sym(vpx_filter_block1d8_h2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqa      xmm1, xmm0
    psrldq      xmm1, 1

    APPLY_FILTER_8 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d16_h2_sse2) PRIVATE
sym(vpx_filter_block1d16_h2_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movdqu      xmm0,   [rsi]               ;load src
    movdqu      xmm1,   [rsi + 1]
    movdqa      xmm2, xmm0
    movdqa      xmm3, xmm1

    APPLY_FILTER_16 0
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d4_h2_avg_sse2) PRIVATE
sym(vpx_filter_block1d4_h2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM_4
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqa      xmm1, xmm0
    psrldq      xmm1, 1

    APPLY_FILTER_4 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d8_h2_avg_sse2) PRIVATE
sym(vpx_filter_block1d8_h2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movdqu      xmm0, [rsi]                 ;load src
    movdqa      xmm1, xmm0
    psrldq      xmm1, 1

    APPLY_FILTER_8 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d16_h2_avg_sse2) PRIVATE
sym(vpx_filter_block1d16_h2_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    GET_PARAM
.loop:
    movdqu      xmm0,   [rsi]               ;load src
    movdqu      xmm1,   [rsi + 1]
    movdqa      xmm2, xmm0
    movdqa      xmm3, xmm1

    APPLY_FILTER_16 1
    jnz         .loop

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
