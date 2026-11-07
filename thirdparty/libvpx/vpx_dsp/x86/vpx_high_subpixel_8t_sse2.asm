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

;Note: tap3 and tap4 have to be applied and added after other taps to avoid
;overflow.

%macro HIGH_GET_FILTERS_4 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rcx, 0x00000040

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

    punpcklwd   xmm0, xmm6
    punpcklwd   xmm2, xmm5
    punpcklwd   xmm3, xmm4
    punpcklwd   xmm1, xmm7

    movdqa      k0k6, xmm0
    movdqa      k2k5, xmm2
    movdqa      k3k4, xmm3
    movdqa      k1k7, xmm1

    movq        xmm6, rcx
    pshufd      xmm6, xmm6, 0
    movdqa      krd, xmm6

    ;Compute max and min values of a pixel
    mov         rdx, 0x00010001
    movsxd      rcx, DWORD PTR arg(6)      ;bd
    movq        xmm0, rdx
    movq        xmm1, rcx
    pshufd      xmm0, xmm0, 0b
    movdqa      xmm2, xmm0
    psllw       xmm0, xmm1
    psubw       xmm0, xmm2
    pxor        xmm1, xmm1
    movdqa      max, xmm0                  ;max value (for clamping)
    movdqa      min, xmm1                  ;min value (for clamping)

%endm

%macro HIGH_APPLY_FILTER_4 1
    punpcklwd   xmm0, xmm6                  ;two row in one register
    punpcklwd   xmm1, xmm7
    punpcklwd   xmm2, xmm5
    punpcklwd   xmm3, xmm4

    pmaddwd     xmm0, k0k6                  ;multiply the filter factors
    pmaddwd     xmm1, k1k7
    pmaddwd     xmm2, k2k5
    pmaddwd     xmm3, k3k4

    paddd       xmm0, xmm1                  ;sum
    paddd       xmm0, xmm2
    paddd       xmm0, xmm3

    paddd       xmm0, krd                   ;rounding
    psrad       xmm0, 7                     ;shift
    packssdw    xmm0, xmm0                  ;pack to word

    ;clamp the values
    pminsw      xmm0, max
    pmaxsw      xmm0, min

%if %1
    movq        xmm1, [rdi]
    pavgw       xmm0, xmm1
%endif
    movq        [rdi], xmm0
%endm

%macro HIGH_GET_FILTERS 0
    mov         rdx, arg(5)                 ;filter ptr
    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr
    mov         rcx, 0x00000040

    movdqa      xmm7, [rdx]                 ;load filters
    pshuflw     xmm0, xmm7, 0b              ;k0
    pshuflw     xmm1, xmm7, 01010101b       ;k1
    pshuflw     xmm2, xmm7, 10101010b       ;k2
    pshuflw     xmm3, xmm7, 11111111b       ;k3
    pshufhw     xmm4, xmm7, 0b              ;k4
    pshufhw     xmm5, xmm7, 01010101b       ;k5
    pshufhw     xmm6, xmm7, 10101010b       ;k6
    pshufhw     xmm7, xmm7, 11111111b       ;k7
    punpcklqdq  xmm2, xmm2
    punpcklqdq  xmm3, xmm3
    punpcklwd   xmm0, xmm1
    punpckhwd   xmm6, xmm7
    punpckhwd   xmm2, xmm5
    punpckhwd   xmm3, xmm4

    movdqa      k0k1, xmm0                  ;store filter factors on stack
    movdqa      k6k7, xmm6
    movdqa      k2k5, xmm2
    movdqa      k3k4, xmm3

    movq        xmm6, rcx
    pshufd      xmm6, xmm6, 0
    movdqa      krd, xmm6                   ;rounding

    ;Compute max and min values of a pixel
    mov         rdx, 0x00010001
    movsxd      rcx, DWORD PTR arg(6)       ;bd
    movq        xmm0, rdx
    movq        xmm1, rcx
    pshufd      xmm0, xmm0, 0b
    movdqa      xmm2, xmm0
    psllw       xmm0, xmm1
    psubw       xmm0, xmm2
    pxor        xmm1, xmm1
    movdqa      max, xmm0                  ;max value (for clamping)
    movdqa      min, xmm1                  ;min value (for clamping)
%endm

%macro LOAD_VERT_8 1
    movdqu      xmm0, [rsi + %1]            ;0
    movdqu      xmm1, [rsi + rax + %1]      ;1
    movdqu      xmm6, [rsi + rdx * 2 + %1]  ;6
    lea         rsi,  [rsi + rax]
    movdqu      xmm7, [rsi + rdx * 2 + %1]  ;7
    movdqu      xmm2, [rsi + rax + %1]      ;2
    movdqu      xmm3, [rsi + rax * 2 + %1]  ;3
    movdqu      xmm4, [rsi + rdx + %1]      ;4
    movdqu      xmm5, [rsi + rax * 4 + %1]  ;5
%endm

%macro HIGH_APPLY_FILTER_8 2
    movdqu      temp, xmm4
    movdqa      xmm4, xmm0
    punpcklwd   xmm0, xmm1
    punpckhwd   xmm4, xmm1
    movdqa      xmm1, xmm6
    punpcklwd   xmm6, xmm7
    punpckhwd   xmm1, xmm7
    movdqa      xmm7, xmm2
    punpcklwd   xmm2, xmm5
    punpckhwd   xmm7, xmm5

    movdqu      xmm5, temp
    movdqu      temp, xmm4
    movdqa      xmm4, xmm3
    punpcklwd   xmm3, xmm5
    punpckhwd   xmm4, xmm5
    movdqu      xmm5, temp

    pmaddwd     xmm0, k0k1
    pmaddwd     xmm5, k0k1
    pmaddwd     xmm6, k6k7
    pmaddwd     xmm1, k6k7
    pmaddwd     xmm2, k2k5
    pmaddwd     xmm7, k2k5
    pmaddwd     xmm3, k3k4
    pmaddwd     xmm4, k3k4

    paddd       xmm0, xmm6
    paddd       xmm0, xmm2
    paddd       xmm0, xmm3
    paddd       xmm5, xmm1
    paddd       xmm5, xmm7
    paddd       xmm5, xmm4

    paddd       xmm0, krd                   ;rounding
    paddd       xmm5, krd
    psrad       xmm0, 7                     ;shift
    psrad       xmm5, 7
    packssdw    xmm0, xmm5                  ;pack back to word

    ;clamp the values
    pminsw      xmm0, max
    pmaxsw      xmm0, min

%if %1
    movdqu      xmm1, [rdi + %2]
    pavgw       xmm0, xmm1
%endif
    movdqu      [rdi + %2], xmm0
%endm

SECTION .text

;void vpx_highbd_filter_block1d4_v8_sse2
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    short *filter
;)
globalsym(vpx_highbd_filter_block1d4_v8_sse2)
sym(vpx_highbd_filter_block1d4_v8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 7
    %define k0k6 [rsp + 16 * 0]
    %define k2k5 [rsp + 16 * 1]
    %define k3k4 [rsp + 16 * 2]
    %define k1k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define max [rsp + 16 * 5]
    %define min [rsp + 16 * 6]

    HIGH_GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rbx, [rbx + rbx]
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movq        xmm0, [rsi]                 ;load src: row 0
    movq        xmm1, [rsi + rax]           ;1
    movq        xmm6, [rsi + rdx * 2]       ;6
    lea         rsi,  [rsi + rax]
    movq        xmm7, [rsi + rdx * 2]       ;7
    movq        xmm2, [rsi + rax]           ;2
    movq        xmm3, [rsi + rax * 2]       ;3
    movq        xmm4, [rsi + rdx]           ;4
    movq        xmm5, [rsi + rax * 4]       ;5

    HIGH_APPLY_FILTER_4 0

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 7
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_highbd_filter_block1d8_v8_sse2
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    short *filter
;)
globalsym(vpx_highbd_filter_block1d8_v8_sse2)
sym(vpx_highbd_filter_block1d8_v8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rbx, [rbx + rbx]
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    LOAD_VERT_8 0
    HIGH_APPLY_FILTER_8 0, 0

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_highbd_filter_block1d16_v8_sse2
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    short *filter
;)
globalsym(vpx_highbd_filter_block1d16_v8_sse2)
sym(vpx_highbd_filter_block1d16_v8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rbx, [rbx + rbx]
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    LOAD_VERT_8 0
    HIGH_APPLY_FILTER_8 0, 0
    sub         rsi, rax

    LOAD_VERT_8 16
    HIGH_APPLY_FILTER_8 0, 16
    add         rdi, rbx

    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d4_v8_avg_sse2)
sym(vpx_highbd_filter_block1d4_v8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 7
    %define k0k6 [rsp + 16 * 0]
    %define k2k5 [rsp + 16 * 1]
    %define k3k4 [rsp + 16 * 2]
    %define k1k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define max [rsp + 16 * 5]
    %define min [rsp + 16 * 6]

    HIGH_GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rbx, [rbx + rbx]
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movq        xmm0, [rsi]                 ;load src: row 0
    movq        xmm1, [rsi + rax]           ;1
    movq        xmm6, [rsi + rdx * 2]       ;6
    lea         rsi,  [rsi + rax]
    movq        xmm7, [rsi + rdx * 2]       ;7
    movq        xmm2, [rsi + rax]           ;2
    movq        xmm3, [rsi + rax * 2]       ;3
    movq        xmm4, [rsi + rdx]           ;4
    movq        xmm5, [rsi + rax * 4]       ;5

    HIGH_APPLY_FILTER_4 1

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 7
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d8_v8_avg_sse2)
sym(vpx_highbd_filter_block1d8_v8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rbx, [rbx + rbx]
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
.loop:
    LOAD_VERT_8 0
    HIGH_APPLY_FILTER_8 1, 0

    lea         rdi, [rdi + rbx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d16_v8_avg_sse2)
sym(vpx_highbd_filter_block1d16_v8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    push        rbx
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rbx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rbx, [rbx + rbx]
    lea         rdx, [rax + rax * 2]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height
.loop:
    LOAD_VERT_8 0
    HIGH_APPLY_FILTER_8 1, 0
    sub         rsi, rax

    LOAD_VERT_8 16
    HIGH_APPLY_FILTER_8 1, 16
    add         rdi, rbx

    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp
    pop rbx
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_highbd_filter_block1d4_h8_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    short *filter
;)
globalsym(vpx_highbd_filter_block1d4_h8_sse2)
sym(vpx_highbd_filter_block1d4_h8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 7
    %define k0k6 [rsp + 16 * 0]
    %define k2k5 [rsp + 16 * 1]
    %define k3k4 [rsp + 16 * 2]
    %define k1k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define max [rsp + 16 * 5]
    %define min [rsp + 16 * 6]

    HIGH_GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rdx, [rdx + rdx]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 6]           ;load src
    movdqu      xmm4,   [rsi + 2]
    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm4
    movdqa      xmm7, xmm4
    movdqa      xmm2, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm5, xmm4

    psrldq      xmm1, 2
    psrldq      xmm6, 4
    psrldq      xmm7, 6
    psrldq      xmm2, 4
    psrldq      xmm3, 6
    psrldq      xmm5, 2

    HIGH_APPLY_FILTER_4 0

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 7
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_highbd_filter_block1d8_h8_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    short *filter
;)
globalsym(vpx_highbd_filter_block1d8_h8_sse2)
sym(vpx_highbd_filter_block1d8_h8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rdx, [rdx + rdx]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 6]           ;load src
    movdqu      xmm1,   [rsi - 4]
    movdqu      xmm2,   [rsi - 2]
    movdqu      xmm3,   [rsi]
    movdqu      xmm4,   [rsi + 2]
    movdqu      xmm5,   [rsi + 4]
    movdqu      xmm6,   [rsi + 6]
    movdqu      xmm7,   [rsi + 8]

    HIGH_APPLY_FILTER_8 0, 0

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vpx_highbd_filter_block1d16_h8_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    short *filter
;)
globalsym(vpx_highbd_filter_block1d16_h8_sse2)
sym(vpx_highbd_filter_block1d16_h8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rdx, [rdx + rdx]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 6]           ;load src
    movdqu      xmm1,   [rsi - 4]
    movdqu      xmm2,   [rsi - 2]
    movdqu      xmm3,   [rsi]
    movdqu      xmm4,   [rsi + 2]
    movdqu      xmm5,   [rsi + 4]
    movdqu      xmm6,   [rsi + 6]
    movdqu      xmm7,   [rsi + 8]

    HIGH_APPLY_FILTER_8 0, 0

    movdqu      xmm0,   [rsi + 10]           ;load src
    movdqu      xmm1,   [rsi + 12]
    movdqu      xmm2,   [rsi + 14]
    movdqu      xmm3,   [rsi + 16]
    movdqu      xmm4,   [rsi + 18]
    movdqu      xmm5,   [rsi + 20]
    movdqu      xmm6,   [rsi + 22]
    movdqu      xmm7,   [rsi + 24]

    HIGH_APPLY_FILTER_8 0, 16

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d4_h8_avg_sse2)
sym(vpx_highbd_filter_block1d4_h8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 7
    %define k0k6 [rsp + 16 * 0]
    %define k2k5 [rsp + 16 * 1]
    %define k3k4 [rsp + 16 * 2]
    %define k1k7 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define max [rsp + 16 * 5]
    %define min [rsp + 16 * 6]

    HIGH_GET_FILTERS_4

    mov         rsi, arg(0)                 ;src_ptr
    mov         rdi, arg(2)                 ;output_ptr

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rdx, [rdx + rdx]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 6]           ;load src
    movdqu      xmm4,   [rsi + 2]
    movdqa      xmm1, xmm0
    movdqa      xmm6, xmm4
    movdqa      xmm7, xmm4
    movdqa      xmm2, xmm0
    movdqa      xmm3, xmm0
    movdqa      xmm5, xmm4

    psrldq      xmm1, 2
    psrldq      xmm6, 4
    psrldq      xmm7, 6
    psrldq      xmm2, 4
    psrldq      xmm3, 6
    psrldq      xmm5, 2

    HIGH_APPLY_FILTER_4 1

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 7
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d8_h8_avg_sse2)
sym(vpx_highbd_filter_block1d8_h8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rdx, [rdx + rdx]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 6]           ;load src
    movdqu      xmm1,   [rsi - 4]
    movdqu      xmm2,   [rsi - 2]
    movdqu      xmm3,   [rsi]
    movdqu      xmm4,   [rsi + 2]
    movdqu      xmm5,   [rsi + 4]
    movdqu      xmm6,   [rsi + 6]
    movdqu      xmm7,   [rsi + 8]

    HIGH_APPLY_FILTER_8 1, 0

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

globalsym(vpx_highbd_filter_block1d16_h8_avg_sse2)
sym(vpx_highbd_filter_block1d16_h8_avg_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 16 * 8
    %define k0k1 [rsp + 16 * 0]
    %define k6k7 [rsp + 16 * 1]
    %define k2k5 [rsp + 16 * 2]
    %define k3k4 [rsp + 16 * 3]
    %define krd [rsp + 16 * 4]
    %define temp [rsp + 16 * 5]
    %define max [rsp + 16 * 6]
    %define min [rsp + 16 * 7]

    HIGH_GET_FILTERS

    movsxd      rax, DWORD PTR arg(1)       ;pixels_per_line
    movsxd      rdx, DWORD PTR arg(3)       ;out_pitch
    lea         rax, [rax + rax]            ;bytes per line
    lea         rdx, [rdx + rdx]
    movsxd      rcx, DWORD PTR arg(4)       ;output_height

.loop:
    movdqu      xmm0,   [rsi - 6]           ;load src
    movdqu      xmm1,   [rsi - 4]
    movdqu      xmm2,   [rsi - 2]
    movdqu      xmm3,   [rsi]
    movdqu      xmm4,   [rsi + 2]
    movdqu      xmm5,   [rsi + 4]
    movdqu      xmm6,   [rsi + 6]
    movdqu      xmm7,   [rsi + 8]

    HIGH_APPLY_FILTER_8 1, 0

    movdqu      xmm0,   [rsi + 10]           ;load src
    movdqu      xmm1,   [rsi + 12]
    movdqu      xmm2,   [rsi + 14]
    movdqu      xmm3,   [rsi + 16]
    movdqu      xmm4,   [rsi + 18]
    movdqu      xmm5,   [rsi + 20]
    movdqu      xmm6,   [rsi + 22]
    movdqu      xmm7,   [rsi + 24]

    HIGH_APPLY_FILTER_8 1, 16

    lea         rsi, [rsi + rax]
    lea         rdi, [rdi + rdx]
    dec         rcx
    jnz         .loop

    add rsp, 16 * 8
    pop rsp

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
