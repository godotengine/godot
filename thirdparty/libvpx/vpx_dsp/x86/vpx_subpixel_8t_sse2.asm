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

;Note: tap3 and tap4 have to be applied and added after other taps to avoid
;overflow.

%macro GET_FILTERS_4 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rcx, 0x0400040

    movdqa      xmm7, [rdx]                 ;load filters
    pshuflw     xmm0, xmm7, 0b              ;k0
    pshuflw     xmm1, xmm7, 01010101b       ;k1
    pshuflw     xmm2, xmm7, 10101010b       ;k2
    pshuflw     xmm3, xmm7, 11111111b       ;k3
    psrldq      xmm7, 8
    pshuflw     xmm4, xmm7, 0b              ;k4
    pshuflw     xmm5, xmm7, 01010101b       ;k5
    pshuflw     xmm6, xmm7, 10101010b       ;k6
    pshuflw     xmm7, xmm7, 11111111b       ;k7

    punpcklqdq  xmm0, xmm1
    punpcklqdq  xmm2, xmm3
    punpcklqdq  xmm5, xmm4
    punpcklqdq  xmm6, xmm7

    movdqa      k0k1, xmm0
    movdqa      k2k3, xmm2
    movdqa      k5k4, xmm5
    movdqa      k6k7, xmm6

    movq        xmm6, rcx
    pshufd      xmm6, xmm6, 0
    movdqa      krd, xmm6

    pxor        xmm7, xmm7
    movdqa      zero, xmm7
%endm

%macro APPLY_FILTER_4 1
    punpckldq   xmm0, xmm1                  ;two row in one register
    punpckldq   xmm6, xmm7
    punpckldq   xmm2, xmm3
    punpckldq   xmm5, xmm4

    punpcklbw   xmm0, zero                  ;unpack to word
    punpcklbw   xmm6, zero
    punpcklbw   xmm2, zero
    punpcklbw   xmm5, zero

    pmullw      xmm0, k0k1                  ;multiply the filter factors
    pmullw      xmm6, k6k7
    pmullw      xmm2, k2k3
    pmullw      xmm5, k5k4

    paddsw      xmm0, xmm6                  ;sum
    movdqa      xmm1, xmm0
    psrldq      xmm1, 8
    paddsw      xmm0, xmm1
    paddsw      xmm0, xmm2
    psrldq      xmm2, 8
    paddsw      xmm0, xmm5
    psrldq      xmm5, 8
    paddsw      xmm0, xmm2
    paddsw      xmm0, xmm5

    paddsw      xmm0, krd                   ;rounding
    psraw       xmm0, 7                     ;shift
    packuswb    xmm0, xmm0                  ;pack to byte

%if %1
    movd        xmm1, [rdi]
    pavgb       xmm0, xmm1
%endif
    movd        [rdi], xmm0
%endm

%macro GET_FILTERS 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr
    mov         rcx, 0x0400040

    movdqa      xmm7, [rdx]                 ;load filters
    pshuflw     xmm0, xmm7, 0b              ;k0
    pshuflw     xmm1, xmm7, 01010101b       ;k1
    pshuflw     xmm2, xmm7, 10101010b       ;k2
    pshuflw     xmm3, xmm7, 11111111b       ;k3
    pshufhw     xmm4, xmm7, 0b              ;k4
    pshufhw     xmm5, xmm7, 01010101b       ;k5
    pshufhw     xmm6, xmm7, 10101010b       ;k6
    pshufhw     xmm7, xmm7, 11111111b       ;k7

    punpcklwd   xmm0, xmm0
    punpcklwd   xmm1, xmm1
    punpcklwd   xmm2, xmm2
    punpcklwd   xmm3, xmm3
    punpckhwd   xmm4, xmm4
    punpckhwd   xmm5, xmm5
    punpckhwd   xmm6, xmm6
    punpckhwd   xmm7, xmm7

    movdqa      k0,   xmm0                  ;store filter factors on stack
    movdqa      k1,   xmm1
    movdqa      k2,   xmm2
    movdqa      k3,   xmm3
    movdqa      k4,   xmm4
    movdqa      k5,   xmm5
    movdqa      k6,   xmm6
    movdqa      k7,   xmm7

    movq        xmm6, rcx
    pshufd      xmm6, xmm6, 0
    movdqa      krd, xmm6                   ;rounding

    pxor        xmm7, xmm7
    movdqa      zero, xmm7
%endm

%macro LOAD_VERT_8 1
    movq        xmm0, [rsi + %1]            ;0
    movq        xmm1, [rsi + rax + %1]      ;1
    movq        xmm6, [rsi + rdx * 2 + %1]  ;6
    lea         rsi,  [rsi + rax]
    movq        xmm7, [rsi + rdx * 2 + %1]  ;7
    movq        xmm2, [rsi + rax + %1]      ;2
    movq        xmm3, [rsi + rax * 2 + %1]  ;3
    movq        xmm4, [rsi + rdx + %1]      ;4
    movq        xmm5, [rsi + rax * 4 + %1]  ;5
%endm

%macro APPLY_FILTER_8 2
    punpcklbw   xmm0, zero
    punpcklbw   xmm1, zero
    punpcklbw   xmm6, zero
    punpcklbw   xmm7, zero
    punpcklbw   xmm2, zero
    punpcklbw   xmm5, zero
    punpcklbw   xmm3, zero
    punpcklbw   xmm4, zero

    pmullw      xmm0, k0
    pmullw      xmm1, k1
    pmullw      xmm6, k6
    pmullw      xmm7, k7
    pmullw      xmm2, k2
    pmullw      xmm5, k5
    pmullw      xmm3, k3
    pmullw      xmm4, k4

    paddsw      xmm0, xmm1
    paddsw      xmm0, xmm6
    paddsw      xmm0, xmm7
    paddsw      xmm0, xmm2
    paddsw      xmm0, xmm5
    paddsw      xmm0, xmm3
    paddsw      xmm0, xmm4

    paddsw      xmm0, krd                   ;rounding
    psraw       xmm0, 7                     ;shift
    packuswb    xmm0, xmm0                  ;pack back to byte
%if %1
    movq        xmm1, [rdi + %2]
    pavgb       xmm0, xmm1
%endif
    movq        [rdi + %2], xmm0
%endm

;void vpx_filter_block1d4_v8_sse2
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    short *filter
;)
global sym(vpx_filter_block1d4_v8_sse2) PRIVATE
sym(vpx_filter_block1d4_v8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 6
    %define k0k1 [rsp + 16 * 0]
    %define k2k3 [rsp + 16 * 1]
    %define k5k4 [rsp + 16 * 2]
    %define k6k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define zero [rsp + 16 * 5]

    GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movd        xmm0, [rsi]                 ;load src: row 0
    movd        xmm1, [rsi + rax]           ;1
    movd        xmm6, [rsi + rdx * 2]       ;6
    lea         rsi,  [rsi + rax]
    movd        xmm7, [rsi + rdx * 2]       ;7
    movd        xmm2, [rsi + rax]           ;2
    movd        xmm3, [rsi + rax * 2]       ;3
    movd        xmm4, [rsi + rdx]           ;4
    movd        xmm5, [rsi + rax * 4]       ;5

    APPLY_FILTER_4 0

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 6
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_filter_block1d8_v8_sse2
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    short *filter
;)
global sym(vpx_filter_block1d8_v8_sse2) PRIVATE
sym(vpx_filter_block1d8_v8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    LOAD_VERT_8 0
    APPLY_FILTER_8 0, 0

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_filter_block1d16_v8_sse2
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    short *filter
;)
global sym(vpx_filter_block1d16_v8_sse2) PRIVATE
sym(vpx_filter_block1d16_v8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    LOAD_VERT_8 0
    APPLY_FILTER_8 0, 0
    sub         rsi, rax

    LOAD_VERT_8 8
    APPLY_FILTER_8 0, 8
    add         rdi, rbx

    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d4_v8_avg_sse2) PRIVATE
sym(vpx_filter_block1d4_v8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 6
    %define k0k1 [rsp + 16 * 0]
    %define k2k3 [rsp + 16 * 1]
    %define k5k4 [rsp + 16 * 2]
    %define k6k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define zero [rsp + 16 * 5]

    GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movd        xmm0, [rsi]                 ;load src: row 0
    movd        xmm1, [rsi + rax]           ;1
    movd        xmm6, [rsi + rdx * 2]       ;6
    lea         rsi,  [rsi + rax]
    movd        xmm7, [rsi + rdx * 2]       ;7
    movd        xmm2, [rsi + rax]           ;2
    movd        xmm3, [rsi + rax * 2]       ;3
    movd        xmm4, [rsi + rdx]           ;4
    movd        xmm5, [rsi + rax * 4]       ;5

    APPLY_FILTER_4 1

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 6
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d8_v8_avg_sse2) PRIVATE
sym(vpx_filter_block1d8_v8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
.loop:
    LOAD_VERT_8 0
    APPLY_FILTER_8 1, 0

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d16_v8_avg_sse2) PRIVATE
sym(vpx_filter_block1d16_v8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
.loop:
    LOAD_VERT_8 0
    APPLY_FILTER_8 1, 0
    sub         rsi, rax

    LOAD_VERT_8 8
    APPLY_FILTER_8 1, 8
    add         rdi, rbx

    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_filter_block1d4_h8_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    short *filter
;)
global sym(vpx_filter_block1d4_h8_sse2) PRIVATE
sym(vpx_filter_block1d4_h8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 6
    %define k0k1 [rsp + 16 * 0]
    %define k2k3 [rsp + 16 * 1]
    %define k5k4 [rsp + 16 * 2]
    %define k6k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define zero [rsp + 16 * 5]

    GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 3]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm3, 3
    psrldq      xmm5, 5
    psrldq      xmm4, 4

    APPLY_FILTER_4 0

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 6
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_filter_block1d8_h8_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    short *filter
;)
global sym(vpx_filter_block1d8_h8_sse2) PRIVATE
sym(vpx_filter_block1d8_h8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 3]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm5, 5
    psrldq      xmm3, 3
    psrldq      xmm4, 4

    APPLY_FILTER_8 0, 0

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_filter_block1d16_h8_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    short *filter
;)
global sym(vpx_filter_block1d16_h8_sse2) PRIVATE
sym(vpx_filter_block1d16_h8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 3]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm5, 5
    psrldq      xmm3, 3
    psrldq      xmm4, 4

    APPLY_FILTER_8 0, 0

    movdqu      xmm0,   [rsi + 5]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm5, 5
    psrldq      xmm3, 3
    psrldq      xmm4, 4

    APPLY_FILTER_8 0, 8

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d4_h8_avg_sse2) PRIVATE
sym(vpx_filter_block1d4_h8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 6
    %define k0k1 [rsp + 16 * 0]
    %define k2k3 [rsp + 16 * 1]
    %define k5k4 [rsp + 16 * 2]
    %define k6k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define zero [rsp + 16 * 5]

    GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 3]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm3, 3
    psrldq      xmm5, 5
    psrldq      xmm4, 4

    APPLY_FILTER_4 1

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 6
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d8_h8_avg_sse2) PRIVATE
sym(vpx_filter_block1d8_h8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 3]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm5, 5
    psrldq      xmm3, 3
    psrldq      xmm4, 4

    APPLY_FILTER_8 1, 0

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

global sym(vpx_filter_block1d16_h8_avg_sse2) PRIVATE
sym(vpx_filter_block1d16_h8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 10
    %define k0 [rsp + 16 * 0]
    %define k1 [rsp + 16 * 1]
    %define k2 [rsp + 16 * 2]
    %define k3 [rsp + 16 * 3]
    %define k4 [rsp + 16 * 4]
    %define k5 [rsp + 16 * 5]
    %define k6 [rsp + 16 * 6]
    %define k7 [rsp + 16 * 7]
    %define krd [rsp + 16 * 8]
    %define zero [rsp + 16 * 9]

    GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 3]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm5, 5
    psrldq      xmm3, 3
    psrldq      xmm4, 4

    APPLY_FILTER_8 1, 0

    movdqu      xmm0,   [rsi + 5]           ;load src

    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm0
    movdqa      xmm7, xmm0
    movdqa      xmm2, xmm0
    movdqa      xmm5, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm4, xmm0

    psrldq      xmm1, 1
    psrldq      xmm6, 6
    psrldq      xmm7, 7
    psrldq      xmm2, 2
    psrldq      xmm5, 5
    psrldq      xmm3, 3
    psrldq      xmm4, 4

    APPLY_FILTER_8 1, 8

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 10
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
