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

%define BLOCK_HEIGHT_WIDTH 4
%define VP8_FILTER_WEIGHT 128
%define VP8_FILTER_SHIFT  7

SECTION .text

;/************************************************************************************
; Notes: filter_block1d_h6 applies a 6 tap filter horizontally to the input pixels. The
; input pixel array has output_height rows. This routine assumes that output_height is an
; even number. This function handles 8 pixels in horizontal direction, calculating ONE
; rows each iteration to take advantage of the 128 bits operations.
;
; This is an implementation of some of the SSE optimizations first seen in ffvp8
;
;*************************************************************************************/
;void vp8_filter_block1d8_h6_ssse3
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    unsigned int    vp8_filter_index
;)
globalsym(vp8_filter_block1d8_h6_ssse3)
sym(vp8_filter_block1d8_h6_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movsxd      rdx, DWORD PTR arg(5)   ;table index
    xor         rsi, rsi
    shl         rdx, 4

    movdqa      xmm7, [GLOBAL(rd)]

    lea         rax, [GLOBAL(k0_k5)]
    add         rax, rdx
    mov         rdi, arg(2)             ;output_ptr

    cmp         esi, DWORD PTR [rax]
    je          vp8_filter_block1d8_h4_ssse3

    movdqa      xmm4, XMMWORD PTR [rax]         ;k0_k5
    movdqa      xmm5, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm6, XMMWORD PTR [rax+128]     ;k1_k3

    mov         rsi, arg(0)             ;src_ptr
    movsxd      rax, dword ptr arg(1)   ;src_pixels_per_line
    movsxd      rcx, dword ptr arg(4)   ;output_height

    movsxd      rdx, dword ptr arg(3)   ;output_pitch

    sub         rdi, rdx
;xmm3 free
.filter_block1d8_h6_rowloop_ssse3:
    movq        xmm0,   MMWORD PTR [rsi - 2]    ; -2 -1  0  1  2  3  4  5

    movq        xmm2,   MMWORD PTR [rsi + 3]    ;  3  4  5  6  7  8  9 10

    punpcklbw   xmm0,   xmm2                    ; -2  3 -1  4  0  5  1  6  2  7  3  8  4  9  5 10

    movdqa      xmm1,   xmm0
    pmaddubsw   xmm0,   xmm4

    movdqa      xmm2,   xmm1
    pshufb      xmm1,   [GLOBAL(shuf2bfrom1)]

    pshufb      xmm2,   [GLOBAL(shuf3bfrom1)]
    pmaddubsw   xmm1,   xmm5

    lea         rdi,    [rdi + rdx]
    pmaddubsw   xmm2,   xmm6

    lea         rsi,    [rsi + rax]
    dec         rcx

    paddsw      xmm0,   xmm1
    paddsw      xmm2,   xmm7

    paddsw      xmm0,   xmm2

    psraw       xmm0,   7

    packuswb    xmm0,   xmm0

    movq        MMWORD Ptr [rdi], xmm0
    jnz         .filter_block1d8_h6_rowloop_ssse3

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

vp8_filter_block1d8_h4_ssse3:
    movdqa      xmm5, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm6, XMMWORD PTR [rax+128]     ;k1_k3

    movdqa      xmm3, XMMWORD PTR [GLOBAL(shuf2bfrom1)]
    movdqa      xmm4, XMMWORD PTR [GLOBAL(shuf3bfrom1)]

    mov         rsi, arg(0)             ;src_ptr

    movsxd      rax, dword ptr arg(1)   ;src_pixels_per_line
    movsxd      rcx, dword ptr arg(4)   ;output_height

    movsxd      rdx, dword ptr arg(3)   ;output_pitch

    sub         rdi, rdx

.filter_block1d8_h4_rowloop_ssse3:
    movq        xmm0,   MMWORD PTR [rsi - 2]    ; -2 -1  0  1  2  3  4  5

    movq        xmm1,   MMWORD PTR [rsi + 3]    ;  3  4  5  6  7  8  9 10

    punpcklbw   xmm0,   xmm1                    ; -2  3 -1  4  0  5  1  6  2  7  3  8  4  9  5 10

    movdqa      xmm2,   xmm0
    pshufb      xmm0,   xmm3

    pshufb      xmm2,   xmm4
    pmaddubsw   xmm0,   xmm5

    lea         rdi,    [rdi + rdx]
    pmaddubsw   xmm2,   xmm6

    lea         rsi,    [rsi + rax]
    dec         rcx

    paddsw      xmm0,   xmm7

    paddsw      xmm0,   xmm2

    psraw       xmm0,   7

    packuswb    xmm0,   xmm0

    movq        MMWORD Ptr [rdi], xmm0

    jnz         .filter_block1d8_h4_rowloop_ssse3

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
;void vp8_filter_block1d16_h6_ssse3
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    unsigned int    vp8_filter_index
;)
globalsym(vp8_filter_block1d16_h6_ssse3)
sym(vp8_filter_block1d16_h6_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movsxd      rdx, DWORD PTR arg(5)           ;table index
    xor         rsi, rsi
    shl         rdx, 4      ;

    lea         rax, [GLOBAL(k0_k5)]
    add         rax, rdx

    mov         rdi, arg(2)                     ;output_ptr

    mov         rsi, arg(0)                     ;src_ptr

    movdqa      xmm4, XMMWORD PTR [rax]         ;k0_k5
    movdqa      xmm5, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm6, XMMWORD PTR [rax+128]     ;k1_k3

    movsxd      rax, dword ptr arg(1)           ;src_pixels_per_line
    movsxd      rcx, dword ptr arg(4)           ;output_height
    movsxd      rdx, dword ptr arg(3)           ;output_pitch

.filter_block1d16_h6_rowloop_ssse3:
    movq        xmm0,   MMWORD PTR [rsi - 2]    ; -2 -1  0  1  2  3  4  5

    movq        xmm3,   MMWORD PTR [rsi + 3]    ;  3  4  5  6  7  8  9 10

    punpcklbw   xmm0,   xmm3                    ; -2  3 -1  4  0  5  1  6  2  7  3  8  4  9  5 10

    movdqa      xmm1,   xmm0
    pmaddubsw   xmm0,   xmm4

    movdqa      xmm2,   xmm1
    pshufb      xmm1,   [GLOBAL(shuf2bfrom1)]

    pshufb      xmm2,   [GLOBAL(shuf3bfrom1)]
    movq        xmm3,   MMWORD PTR [rsi +  6]

    pmaddubsw   xmm1,   xmm5
    movq        xmm7,   MMWORD PTR [rsi + 11]

    pmaddubsw   xmm2,   xmm6
    punpcklbw   xmm3,   xmm7

    paddsw      xmm0,   xmm1
    movdqa      xmm1,   xmm3

    pmaddubsw   xmm3,   xmm4
    paddsw      xmm0,   xmm2

    movdqa      xmm2,   xmm1
    paddsw      xmm0,   [GLOBAL(rd)]

    pshufb      xmm1,   [GLOBAL(shuf2bfrom1)]
    pshufb      xmm2,   [GLOBAL(shuf3bfrom1)]

    psraw       xmm0,   7
    pmaddubsw   xmm1,   xmm5

    pmaddubsw   xmm2,   xmm6
    packuswb    xmm0,   xmm0

    lea         rsi,    [rsi + rax]
    paddsw      xmm3,   xmm1

    paddsw      xmm3,   xmm2

    paddsw      xmm3,   [GLOBAL(rd)]

    psraw       xmm3,   7

    packuswb    xmm3,   xmm3

    punpcklqdq  xmm0,   xmm3

    movdqa      XMMWORD Ptr [rdi], xmm0

    lea         rdi,    [rdi + rdx]
    dec         rcx
    jnz         .filter_block1d16_h6_rowloop_ssse3

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vp8_filter_block1d4_h6_ssse3
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    unsigned int    output_pitch,
;    unsigned int    output_height,
;    unsigned int    vp8_filter_index
;)
globalsym(vp8_filter_block1d4_h6_ssse3)
sym(vp8_filter_block1d4_h6_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movsxd      rdx, DWORD PTR arg(5)   ;table index
    xor         rsi, rsi
    shl         rdx, 4      ;

    lea         rax, [GLOBAL(k0_k5)]
    add         rax, rdx
    movdqa      xmm7, [GLOBAL(rd)]

    cmp         esi, DWORD PTR [rax]
    je          .vp8_filter_block1d4_h4_ssse3

    movdqa      xmm4, XMMWORD PTR [rax]         ;k0_k5
    movdqa      xmm5, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm6, XMMWORD PTR [rax+128]     ;k1_k3

    mov         rsi, arg(0)             ;src_ptr
    mov         rdi, arg(2)             ;output_ptr
    movsxd      rax, dword ptr arg(1)   ;src_pixels_per_line
    movsxd      rcx, dword ptr arg(4)   ;output_height

    movsxd      rdx, dword ptr arg(3)   ;output_pitch

;xmm3 free
.filter_block1d4_h6_rowloop_ssse3:
    movdqu      xmm0,   XMMWORD PTR [rsi - 2]

    movdqa      xmm1, xmm0
    pshufb      xmm0, [GLOBAL(shuf1b)]

    movdqa      xmm2, xmm1
    pshufb      xmm1, [GLOBAL(shuf2b)]
    pmaddubsw   xmm0, xmm4
    pshufb      xmm2, [GLOBAL(shuf3b)]
    pmaddubsw   xmm1, xmm5

;--
    pmaddubsw   xmm2, xmm6

    lea         rsi,    [rsi + rax]
;--
    paddsw      xmm0, xmm1
    paddsw      xmm0, xmm7
    pxor        xmm1, xmm1
    paddsw      xmm0, xmm2
    psraw       xmm0, 7
    packuswb    xmm0, xmm0

    movd        DWORD PTR [rdi], xmm0

    add         rdi, rdx
    dec         rcx
    jnz         .filter_block1d4_h6_rowloop_ssse3

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

.vp8_filter_block1d4_h4_ssse3:
    movdqa      xmm5, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm6, XMMWORD PTR [rax+128]     ;k1_k3
    movdqa      xmm0, XMMWORD PTR [GLOBAL(shuf2b)]
    movdqa      xmm3, XMMWORD PTR [GLOBAL(shuf3b)]

    mov         rsi, arg(0)             ;src_ptr
    mov         rdi, arg(2)             ;output_ptr
    movsxd      rax, dword ptr arg(1)   ;src_pixels_per_line
    movsxd      rcx, dword ptr arg(4)   ;output_height

    movsxd      rdx, dword ptr arg(3)   ;output_pitch

.filter_block1d4_h4_rowloop_ssse3:
    movdqu      xmm1,   XMMWORD PTR [rsi - 2]

    movdqa      xmm2, xmm1
    pshufb      xmm1, xmm0 ;;[GLOBAL(shuf2b)]
    pshufb      xmm2, xmm3 ;;[GLOBAL(shuf3b)]
    pmaddubsw   xmm1, xmm5

;--
    pmaddubsw   xmm2, xmm6

    lea         rsi,    [rsi + rax]
;--
    paddsw      xmm1, xmm7
    paddsw      xmm1, xmm2
    psraw       xmm1, 7
    packuswb    xmm1, xmm1

    movd        DWORD PTR [rdi], xmm1

    add         rdi, rdx
    dec         rcx
    jnz         .filter_block1d4_h4_rowloop_ssse3

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret



;void vp8_filter_block1d16_v6_ssse3
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    unsigned int   vp8_filter_index
;)
globalsym(vp8_filter_block1d16_v6_ssse3)
sym(vp8_filter_block1d16_v6_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movsxd      rdx, DWORD PTR arg(5)   ;table index
    xor         rsi, rsi
    shl         rdx, 4      ;

    lea         rax, [GLOBAL(k0_k5)]
    add         rax, rdx

    cmp         esi, DWORD PTR [rax]
    je          .vp8_filter_block1d16_v4_ssse3

    movdqa      xmm5, XMMWORD PTR [rax]         ;k0_k5
    movdqa      xmm6, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm7, XMMWORD PTR [rax+128]     ;k1_k3

    mov         rsi, arg(0)             ;src_ptr
    movsxd      rdx, DWORD PTR arg(1)   ;pixels_per_line
    mov         rdi, arg(2)             ;output_ptr

%if ABI_IS_32BIT=0
    movsxd      r8, DWORD PTR arg(3)    ;out_pitch
%endif
    mov         rax, rsi
    movsxd      rcx, DWORD PTR arg(4)   ;output_height
    add         rax, rdx


.vp8_filter_block1d16_v6_ssse3_loop:
    movq        xmm1, MMWORD PTR [rsi]                  ;A
    movq        xmm2, MMWORD PTR [rsi + rdx]            ;B
    movq        xmm3, MMWORD PTR [rsi + rdx * 2]        ;C
    movq        xmm4, MMWORD PTR [rax + rdx * 2]        ;D
    movq        xmm0, MMWORD PTR [rsi + rdx * 4]        ;E

    punpcklbw   xmm2, xmm4                  ;B D
    punpcklbw   xmm3, xmm0                  ;C E

    movq        xmm0, MMWORD PTR [rax + rdx * 4]        ;F

    pmaddubsw   xmm3, xmm6
    punpcklbw   xmm1, xmm0                  ;A F
    pmaddubsw   xmm2, xmm7
    pmaddubsw   xmm1, xmm5

    paddsw      xmm2, xmm3
    paddsw      xmm2, xmm1
    paddsw      xmm2, [GLOBAL(rd)]
    psraw       xmm2, 7
    packuswb    xmm2, xmm2

    movq        MMWORD PTR [rdi], xmm2          ;store the results

    movq        xmm1, MMWORD PTR [rsi + 8]                  ;A
    movq        xmm2, MMWORD PTR [rsi + rdx + 8]            ;B
    movq        xmm3, MMWORD PTR [rsi + rdx * 2 + 8]        ;C
    movq        xmm4, MMWORD PTR [rax + rdx * 2 + 8]        ;D
    movq        xmm0, MMWORD PTR [rsi + rdx * 4 + 8]        ;E

    punpcklbw   xmm2, xmm4                  ;B D
    punpcklbw   xmm3, xmm0                  ;C E

    movq        xmm0, MMWORD PTR [rax + rdx * 4 + 8]        ;F
    pmaddubsw   xmm3, xmm6
    punpcklbw   xmm1, xmm0                  ;A F
    pmaddubsw   xmm2, xmm7
    pmaddubsw   xmm1, xmm5

    add         rsi,  rdx
    add         rax,  rdx
;--
;--
    paddsw      xmm2, xmm3
    paddsw      xmm2, xmm1
    paddsw      xmm2, [GLOBAL(rd)]
    psraw       xmm2, 7
    packuswb    xmm2, xmm2

    movq        MMWORD PTR [rdi+8], xmm2

%if ABI_IS_32BIT
    add         rdi,        DWORD PTR arg(3) ;out_pitch
%else
    add         rdi,        r8
%endif
    dec         rcx
    jnz         .vp8_filter_block1d16_v6_ssse3_loop

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

.vp8_filter_block1d16_v4_ssse3:
    movdqa      xmm6, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm7, XMMWORD PTR [rax+128]     ;k1_k3

    mov         rsi, arg(0)             ;src_ptr
    movsxd      rdx, DWORD PTR arg(1)   ;pixels_per_line
    mov         rdi, arg(2)             ;output_ptr

%if ABI_IS_32BIT=0
    movsxd      r8, DWORD PTR arg(3)    ;out_pitch
%endif
    mov         rax, rsi
    movsxd      rcx, DWORD PTR arg(4)   ;output_height
    add         rax, rdx

.vp8_filter_block1d16_v4_ssse3_loop:
    movq        xmm2, MMWORD PTR [rsi + rdx]            ;B
    movq        xmm3, MMWORD PTR [rsi + rdx * 2]        ;C
    movq        xmm4, MMWORD PTR [rax + rdx * 2]        ;D
    movq        xmm0, MMWORD PTR [rsi + rdx * 4]        ;E

    punpcklbw   xmm2, xmm4                  ;B D
    punpcklbw   xmm3, xmm0                  ;C E

    pmaddubsw   xmm3, xmm6
    pmaddubsw   xmm2, xmm7
    movq        xmm5, MMWORD PTR [rsi + rdx + 8]            ;B
    movq        xmm1, MMWORD PTR [rsi + rdx * 2 + 8]        ;C
    movq        xmm4, MMWORD PTR [rax + rdx * 2 + 8]        ;D
    movq        xmm0, MMWORD PTR [rsi + rdx * 4 + 8]        ;E

    paddsw      xmm2, [GLOBAL(rd)]
    paddsw      xmm2, xmm3
    psraw       xmm2, 7
    packuswb    xmm2, xmm2

    punpcklbw   xmm5, xmm4                  ;B D
    punpcklbw   xmm1, xmm0                  ;C E

    pmaddubsw   xmm1, xmm6
    pmaddubsw   xmm5, xmm7

    movdqa      xmm4, [GLOBAL(rd)]
    add         rsi,  rdx
    add         rax,  rdx
;--
;--
    paddsw      xmm5, xmm1
    paddsw      xmm5, xmm4
    psraw       xmm5, 7
    packuswb    xmm5, xmm5

    punpcklqdq  xmm2, xmm5

    movdqa       XMMWORD PTR [rdi], xmm2

%if ABI_IS_32BIT
    add         rdi,        DWORD PTR arg(3) ;out_pitch
%else
    add         rdi,        r8
%endif
    dec         rcx
    jnz         .vp8_filter_block1d16_v4_ssse3_loop

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vp8_filter_block1d8_v6_ssse3
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    unsigned int   vp8_filter_index
;)
globalsym(vp8_filter_block1d8_v6_ssse3)
sym(vp8_filter_block1d8_v6_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movsxd      rdx, DWORD PTR arg(5)   ;table index
    xor         rsi, rsi
    shl         rdx, 4      ;

    lea         rax, [GLOBAL(k0_k5)]
    add         rax, rdx

    movsxd      rdx, DWORD PTR arg(1)   ;pixels_per_line
    mov         rdi, arg(2)             ;output_ptr
%if ABI_IS_32BIT=0
    movsxd      r8, DWORD PTR arg(3)    ; out_pitch
%endif
    movsxd      rcx, DWORD PTR arg(4)   ;[output_height]

    cmp         esi, DWORD PTR [rax]
    je          .vp8_filter_block1d8_v4_ssse3

    movdqa      xmm5, XMMWORD PTR [rax]         ;k0_k5
    movdqa      xmm6, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm7, XMMWORD PTR [rax+128]     ;k1_k3

    mov         rsi, arg(0)             ;src_ptr

    mov         rax, rsi
    add         rax, rdx

.vp8_filter_block1d8_v6_ssse3_loop:
    movq        xmm1, MMWORD PTR [rsi]                  ;A
    movq        xmm2, MMWORD PTR [rsi + rdx]            ;B
    movq        xmm3, MMWORD PTR [rsi + rdx * 2]        ;C
    movq        xmm4, MMWORD PTR [rax + rdx * 2]        ;D
    movq        xmm0, MMWORD PTR [rsi + rdx * 4]        ;E

    punpcklbw   xmm2, xmm4                  ;B D
    punpcklbw   xmm3, xmm0                  ;C E

    movq        xmm0, MMWORD PTR [rax + rdx * 4]        ;F
    movdqa      xmm4, [GLOBAL(rd)]

    pmaddubsw   xmm3, xmm6
    punpcklbw   xmm1, xmm0                  ;A F
    pmaddubsw   xmm2, xmm7
    pmaddubsw   xmm1, xmm5
    add         rsi,  rdx
    add         rax,  rdx
;--
;--
    paddsw      xmm2, xmm3
    paddsw      xmm2, xmm1
    paddsw      xmm2, xmm4
    psraw       xmm2, 7
    packuswb    xmm2, xmm2

    movq        MMWORD PTR [rdi], xmm2

%if ABI_IS_32BIT
    add         rdi,        DWORD PTR arg(3) ;[out_pitch]
%else
    add         rdi,        r8
%endif
    dec         rcx
    jnz         .vp8_filter_block1d8_v6_ssse3_loop

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

.vp8_filter_block1d8_v4_ssse3:
    movdqa      xmm6, XMMWORD PTR [rax+256]     ;k2_k4
    movdqa      xmm7, XMMWORD PTR [rax+128]     ;k1_k3
    movdqa      xmm5, [GLOBAL(rd)]

    mov         rsi, arg(0)             ;src_ptr

    mov         rax, rsi
    add         rax, rdx

.vp8_filter_block1d8_v4_ssse3_loop:
    movq        xmm2, MMWORD PTR [rsi + rdx]            ;B
    movq        xmm3, MMWORD PTR [rsi + rdx * 2]        ;C
    movq        xmm4, MMWORD PTR [rax + rdx * 2]        ;D
    movq        xmm0, MMWORD PTR [rsi + rdx * 4]        ;E

    punpcklbw   xmm2, xmm4                  ;B D
    punpcklbw   xmm3, xmm0                  ;C E

    pmaddubsw   xmm3, xmm6
    pmaddubsw   xmm2, xmm7
    add         rsi,  rdx
    add         rax,  rdx
;--
;--
    paddsw      xmm2, xmm3
    paddsw      xmm2, xmm5
    psraw       xmm2, 7
    packuswb    xmm2, xmm2

    movq        MMWORD PTR [rdi], xmm2

%if ABI_IS_32BIT
    add         rdi,        DWORD PTR arg(3) ;[out_pitch]
%else
    add         rdi,        r8
%endif
    dec         rcx
    jnz         .vp8_filter_block1d8_v4_ssse3_loop

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
;void vp8_filter_block1d4_v6_ssse3
;(
;    unsigned char *src_ptr,
;    unsigned int   src_pitch,
;    unsigned char *output_ptr,
;    unsigned int   out_pitch,
;    unsigned int   output_height,
;    unsigned int   vp8_filter_index
;)
globalsym(vp8_filter_block1d4_v6_ssse3)
sym(vp8_filter_block1d4_v6_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movsxd      rdx, DWORD PTR arg(5)   ;table index
    xor         rsi, rsi
    shl         rdx, 4      ;

    lea         rax, [GLOBAL(k0_k5)]
    add         rax, rdx

    movsxd      rdx, DWORD PTR arg(1)   ;pixels_per_line
    mov         rdi, arg(2)             ;output_ptr
%if ABI_IS_32BIT=0
    movsxd      r8, DWORD PTR arg(3)    ; out_pitch
%endif
    movsxd      rcx, DWORD PTR arg(4)   ;[output_height]

    cmp         esi, DWORD PTR [rax]
    je          .vp8_filter_block1d4_v4_ssse3

    movq        mm5, MMWORD PTR [rax]         ;k0_k5
    movq        mm6, MMWORD PTR [rax+256]     ;k2_k4
    movq        mm7, MMWORD PTR [rax+128]     ;k1_k3

    mov         rsi, arg(0)             ;src_ptr

    mov         rax, rsi
    add         rax, rdx

.vp8_filter_block1d4_v6_ssse3_loop:
    movd        mm1, DWORD PTR [rsi]                  ;A
    movd        mm2, DWORD PTR [rsi + rdx]            ;B
    movd        mm3, DWORD PTR [rsi + rdx * 2]        ;C
    movd        mm4, DWORD PTR [rax + rdx * 2]        ;D
    movd        mm0, DWORD PTR [rsi + rdx * 4]        ;E

    punpcklbw   mm2, mm4                  ;B D
    punpcklbw   mm3, mm0                  ;C E

    movd        mm0, DWORD PTR [rax + rdx * 4]        ;F

    movq        mm4, [GLOBAL(rd)]

    pmaddubsw   mm3, mm6
    punpcklbw   mm1, mm0                  ;A F
    pmaddubsw   mm2, mm7
    pmaddubsw   mm1, mm5
    add         rsi,  rdx
    add         rax,  rdx
;--
;--
    paddsw      mm2, mm3
    paddsw      mm2, mm1
    paddsw      mm2, mm4
    psraw       mm2, 7
    packuswb    mm2, mm2

    movd        DWORD PTR [rdi], mm2

%if ABI_IS_32BIT
    add         rdi,        DWORD PTR arg(3) ;[out_pitch]
%else
    add         rdi,        r8
%endif
    dec         rcx
    jnz         .vp8_filter_block1d4_v6_ssse3_loop

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret

.vp8_filter_block1d4_v4_ssse3:
    movq        mm6, MMWORD PTR [rax+256]     ;k2_k4
    movq        mm7, MMWORD PTR [rax+128]     ;k1_k3
    movq        mm5, MMWORD PTR [GLOBAL(rd)]

    mov         rsi, arg(0)             ;src_ptr

    mov         rax, rsi
    add         rax, rdx

.vp8_filter_block1d4_v4_ssse3_loop:
    movd        mm2, DWORD PTR [rsi + rdx]            ;B
    movd        mm3, DWORD PTR [rsi + rdx * 2]        ;C
    movd        mm4, DWORD PTR [rax + rdx * 2]        ;D
    movd        mm0, DWORD PTR [rsi + rdx * 4]        ;E

    punpcklbw   mm2, mm4                  ;B D
    punpcklbw   mm3, mm0                  ;C E

    pmaddubsw   mm3, mm6
    pmaddubsw   mm2, mm7
    add         rsi,  rdx
    add         rax,  rdx
;--
;--
    paddsw      mm2, mm3
    paddsw      mm2, mm5
    psraw       mm2, 7
    packuswb    mm2, mm2

    movd        DWORD PTR [rdi], mm2

%if ABI_IS_32BIT
    add         rdi,        DWORD PTR arg(3) ;[out_pitch]
%else
    add         rdi,        r8
%endif
    dec         rcx
    jnz         .vp8_filter_block1d4_v4_ssse3_loop

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vp8_bilinear_predict16x16_ssse3
;(
;    unsigned char  *src_ptr,
;    int   src_pixels_per_line,
;    int  xoffset,
;    int  yoffset,
;    unsigned char *dst_ptr,
;    int dst_pitch
;)
globalsym(vp8_bilinear_predict16x16_ssse3)
sym(vp8_bilinear_predict16x16_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        lea         rcx,        [GLOBAL(vp8_bilinear_filters_ssse3)]
        movsxd      rax,        dword ptr arg(2)    ; xoffset

        cmp         rax,        0                   ; skip first_pass filter if xoffset=0
        je          .b16x16_sp_only

        shl         rax,        4
        lea         rax,        [rax + rcx]         ; HFilter

        mov         rdi,        arg(4)              ; dst_ptr
        mov         rsi,        arg(0)              ; src_ptr
        movsxd      rdx,        dword ptr arg(5)    ; dst_pitch

        movdqa      xmm1,       [rax]

        movsxd      rax,        dword ptr arg(3)    ; yoffset

        cmp         rax,        0                   ; skip second_pass filter if yoffset=0
        je          .b16x16_fp_only

        shl         rax,        4
        lea         rax,        [rax + rcx]         ; VFilter

        lea         rcx,        [rdi+rdx*8]
        lea         rcx,        [rcx+rdx*8]
        movsxd      rdx,        dword ptr arg(1)    ; src_pixels_per_line

        movdqa      xmm2,       [rax]

%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(5)    ; dst_pitch
%endif
        movq        xmm3,       [rsi]               ; 00 01 02 03 04 05 06 07
        movq        xmm5,       [rsi+1]             ; 01 02 03 04 05 06 07 08

        punpcklbw   xmm3,       xmm5                ; 00 01 01 02 02 03 03 04 04 05 05 06 06 07 07 08
        movq        xmm4,       [rsi+8]             ; 08 09 10 11 12 13 14 15

        movq        xmm5,       [rsi+9]             ; 09 10 11 12 13 14 15 16

        lea         rsi,        [rsi + rdx]         ; next line

        pmaddubsw   xmm3,       xmm1                ; 00 02 04 06 08 10 12 14

        punpcklbw   xmm4,       xmm5                ; 08 09 09 10 10 11 11 12 12 13 13 14 14 15 15 16
        pmaddubsw   xmm4,       xmm1                ; 01 03 05 07 09 11 13 15

        paddw       xmm3,       [GLOBAL(rd)]        ; xmm3 += round value
        psraw       xmm3,       VP8_FILTER_SHIFT    ; xmm3 /= 128

        paddw       xmm4,       [GLOBAL(rd)]        ; xmm4 += round value
        psraw       xmm4,       VP8_FILTER_SHIFT    ; xmm4 /= 128

        movdqa      xmm7,       xmm3
        packuswb    xmm7,       xmm4                ; 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15

.next_row:
        movq        xmm6,       [rsi]               ; 00 01 02 03 04 05 06 07
        movq        xmm5,       [rsi+1]             ; 01 02 03 04 05 06 07 08

        punpcklbw   xmm6,       xmm5
        movq        xmm4,       [rsi+8]             ; 08 09 10 11 12 13 14 15

        movq        xmm5,       [rsi+9]             ; 09 10 11 12 13 14 15 16
        lea         rsi,        [rsi + rdx]         ; next line

        pmaddubsw   xmm6,       xmm1

        punpcklbw   xmm4,       xmm5
        pmaddubsw   xmm4,       xmm1

        paddw       xmm6,       [GLOBAL(rd)]        ; xmm6 += round value
        psraw       xmm6,       VP8_FILTER_SHIFT    ; xmm6 /= 128

        paddw       xmm4,       [GLOBAL(rd)]        ; xmm4 += round value
        psraw       xmm4,       VP8_FILTER_SHIFT    ; xmm4 /= 128

        packuswb    xmm6,       xmm4
        movdqa      xmm5,       xmm7

        punpcklbw   xmm5,       xmm6
        pmaddubsw   xmm5,       xmm2

        punpckhbw   xmm7,       xmm6
        pmaddubsw   xmm7,       xmm2

        paddw       xmm5,       [GLOBAL(rd)]        ; xmm5 += round value
        psraw       xmm5,       VP8_FILTER_SHIFT    ; xmm5 /= 128

        paddw       xmm7,       [GLOBAL(rd)]        ; xmm7 += round value
        psraw       xmm7,       VP8_FILTER_SHIFT    ; xmm7 /= 128

        packuswb    xmm5,       xmm7
        movdqa      xmm7,       xmm6

        movdqa      [rdi],      xmm5                ; store the results in the destination
%if ABI_IS_32BIT
        add         rdi,        DWORD PTR arg(5)    ; dst_pitch
%else
        add         rdi,        r8
%endif

        cmp         rdi,        rcx
        jne         .next_row

        jmp         .done

.b16x16_sp_only:
        movsxd      rax,        dword ptr arg(3)    ; yoffset
        shl         rax,        4
        lea         rax,        [rax + rcx]         ; VFilter

        mov         rdi,        arg(4)              ; dst_ptr
        mov         rsi,        arg(0)              ; src_ptr
        movsxd      rdx,        dword ptr arg(5)    ; dst_pitch

        movdqa      xmm1,       [rax]               ; VFilter

        lea         rcx,        [rdi+rdx*8]
        lea         rcx,        [rcx+rdx*8]
        movsxd      rax,        dword ptr arg(1)    ; src_pixels_per_line

        ; get the first horizontal line done
        movq        xmm4,       [rsi]               ; load row 0
        movq        xmm2,       [rsi + 8]           ; load row 0

        lea         rsi,        [rsi + rax]         ; next line
.next_row_sp:
        movq        xmm3,       [rsi]               ; load row + 1
        movq        xmm5,       [rsi + 8]           ; load row + 1

        punpcklbw   xmm4,       xmm3
        punpcklbw   xmm2,       xmm5

        pmaddubsw   xmm4,       xmm1
        movq        xmm7,       [rsi + rax]         ; load row + 2

        pmaddubsw   xmm2,       xmm1
        movq        xmm6,       [rsi + rax + 8]     ; load row + 2

        punpcklbw   xmm3,       xmm7
        punpcklbw   xmm5,       xmm6

        pmaddubsw   xmm3,       xmm1
        paddw       xmm4,       [GLOBAL(rd)]

        pmaddubsw   xmm5,       xmm1
        paddw       xmm2,       [GLOBAL(rd)]

        psraw       xmm4,       VP8_FILTER_SHIFT
        psraw       xmm2,       VP8_FILTER_SHIFT

        packuswb    xmm4,       xmm2
        paddw       xmm3,       [GLOBAL(rd)]

        movdqa      [rdi],      xmm4                ; store row 0
        paddw       xmm5,       [GLOBAL(rd)]

        psraw       xmm3,       VP8_FILTER_SHIFT
        psraw       xmm5,       VP8_FILTER_SHIFT

        packuswb    xmm3,       xmm5
        movdqa      xmm4,       xmm7

        movdqa      [rdi + rdx],xmm3                ; store row 1
        lea         rsi,        [rsi + 2*rax]

        movdqa      xmm2,       xmm6
        lea         rdi,        [rdi + 2*rdx]

        cmp         rdi,        rcx
        jne         .next_row_sp

        jmp         .done

.b16x16_fp_only:
        lea         rcx,        [rdi+rdx*8]
        lea         rcx,        [rcx+rdx*8]
        movsxd      rax,        dword ptr arg(1)    ; src_pixels_per_line

.next_row_fp:
        movq        xmm2,       [rsi]               ; 00 01 02 03 04 05 06 07
        movq        xmm4,       [rsi+1]             ; 01 02 03 04 05 06 07 08

        punpcklbw   xmm2,       xmm4
        movq        xmm3,       [rsi+8]             ; 08 09 10 11 12 13 14 15

        pmaddubsw   xmm2,       xmm1
        movq        xmm4,       [rsi+9]             ; 09 10 11 12 13 14 15 16

        lea         rsi,        [rsi + rax]         ; next line
        punpcklbw   xmm3,       xmm4

        pmaddubsw   xmm3,       xmm1
        movq        xmm5,       [rsi]

        paddw       xmm2,       [GLOBAL(rd)]
        movq        xmm7,       [rsi+1]

        movq        xmm6,       [rsi+8]
        psraw       xmm2,       VP8_FILTER_SHIFT

        punpcklbw   xmm5,       xmm7
        movq        xmm7,       [rsi+9]

        paddw       xmm3,       [GLOBAL(rd)]
        pmaddubsw   xmm5,       xmm1

        psraw       xmm3,       VP8_FILTER_SHIFT
        punpcklbw   xmm6,       xmm7

        packuswb    xmm2,       xmm3
        pmaddubsw   xmm6,       xmm1

        movdqa      [rdi],      xmm2                ; store the results in the destination
        paddw       xmm5,       [GLOBAL(rd)]

        lea         rdi,        [rdi + rdx]         ; dst_pitch
        psraw       xmm5,       VP8_FILTER_SHIFT

        paddw       xmm6,       [GLOBAL(rd)]
        psraw       xmm6,       VP8_FILTER_SHIFT

        packuswb    xmm5,       xmm6
        lea         rsi,        [rsi + rax]         ; next line

        movdqa      [rdi],      xmm5                ; store the results in the destination
        lea         rdi,        [rdi + rdx]         ; dst_pitch

        cmp         rdi,        rcx

        jne         .next_row_fp

.done:
    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vp8_bilinear_predict8x8_ssse3
;(
;    unsigned char  *src_ptr,
;    int   src_pixels_per_line,
;    int  xoffset,
;    int  yoffset,
;    unsigned char *dst_ptr,
;    int dst_pitch
;)
globalsym(vp8_bilinear_predict8x8_ssse3)
sym(vp8_bilinear_predict8x8_ssse3):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ALIGN_STACK 16, rax
    sub         rsp, 144                         ; reserve 144 bytes

        lea         rcx,        [GLOBAL(vp8_bilinear_filters_ssse3)]

        mov         rsi,        arg(0) ;src_ptr
        movsxd      rdx,        dword ptr arg(1) ;src_pixels_per_line

    ;Read 9-line unaligned data in and put them on stack. This gives a big
    ;performance boost.
        movdqu      xmm0,       [rsi]
        lea         rax,        [rdx + rdx*2]
        movdqu      xmm1,       [rsi+rdx]
        movdqu      xmm2,       [rsi+rdx*2]
        add         rsi,        rax
        movdqu      xmm3,       [rsi]
        movdqu      xmm4,       [rsi+rdx]
        movdqu      xmm5,       [rsi+rdx*2]
        add         rsi,        rax
        movdqu      xmm6,       [rsi]
        movdqu      xmm7,       [rsi+rdx]

        movdqa      XMMWORD PTR [rsp],            xmm0

        movdqu      xmm0,       [rsi+rdx*2]

        movdqa      XMMWORD PTR [rsp+16],         xmm1
        movdqa      XMMWORD PTR [rsp+32],         xmm2
        movdqa      XMMWORD PTR [rsp+48],         xmm3
        movdqa      XMMWORD PTR [rsp+64],         xmm4
        movdqa      XMMWORD PTR [rsp+80],         xmm5
        movdqa      XMMWORD PTR [rsp+96],         xmm6
        movdqa      XMMWORD PTR [rsp+112],        xmm7
        movdqa      XMMWORD PTR [rsp+128],        xmm0

        movsxd      rax,        dword ptr arg(2)    ; xoffset
        cmp         rax,        0                   ; skip first_pass filter if xoffset=0
        je          .b8x8_sp_only

        shl         rax,        4
        add         rax,        rcx                 ; HFilter

        mov         rdi,        arg(4)              ; dst_ptr
        movsxd      rdx,        dword ptr arg(5)    ; dst_pitch

        movdqa      xmm0,       [rax]

        movsxd      rax,        dword ptr arg(3)    ; yoffset
        cmp         rax,        0                   ; skip second_pass filter if yoffset=0
        je          .b8x8_fp_only

        shl         rax,        4
        lea         rax,        [rax + rcx]         ; VFilter

        lea         rcx,        [rdi+rdx*8]

        movdqa      xmm1,       [rax]

        ; get the first horizontal line done
        movdqa      xmm3,       [rsp]               ; 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
        movdqa      xmm5,       xmm3                ; 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 xx

        psrldq      xmm5,       1
        lea         rsp,        [rsp + 16]          ; next line

        punpcklbw   xmm3,       xmm5                ; 00 01 01 02 02 03 03 04 04 05 05 06 06 07 07 08
        pmaddubsw   xmm3,       xmm0                ; 00 02 04 06 08 10 12 14

        paddw       xmm3,       [GLOBAL(rd)]        ; xmm3 += round value
        psraw       xmm3,       VP8_FILTER_SHIFT    ; xmm3 /= 128

        movdqa      xmm7,       xmm3
        packuswb    xmm7,       xmm7                ; 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15

.next_row:
        movdqa      xmm6,       [rsp]               ; 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
        lea         rsp,        [rsp + 16]          ; next line

        movdqa      xmm5,       xmm6

        psrldq      xmm5,       1

        punpcklbw   xmm6,       xmm5
        pmaddubsw   xmm6,       xmm0

        paddw       xmm6,       [GLOBAL(rd)]        ; xmm6 += round value
        psraw       xmm6,       VP8_FILTER_SHIFT    ; xmm6 /= 128

        packuswb    xmm6,       xmm6

        punpcklbw   xmm7,       xmm6
        pmaddubsw   xmm7,       xmm1

        paddw       xmm7,       [GLOBAL(rd)]        ; xmm7 += round value
        psraw       xmm7,       VP8_FILTER_SHIFT    ; xmm7 /= 128

        packuswb    xmm7,       xmm7

        movq        [rdi],      xmm7                ; store the results in the destination
        lea         rdi,        [rdi + rdx]

        movdqa      xmm7,       xmm6

        cmp         rdi,        rcx
        jne         .next_row

        jmp         .done8x8

.b8x8_sp_only:
        movsxd      rax,        dword ptr arg(3)    ; yoffset
        shl         rax,        4
        lea         rax,        [rax + rcx]         ; VFilter

        mov         rdi,        arg(4) ;dst_ptr
        movsxd      rdx,        dword ptr arg(5)    ; dst_pitch

        movdqa      xmm0,       [rax]               ; VFilter

        movq        xmm1,       XMMWORD PTR [rsp]
        movq        xmm2,       XMMWORD PTR [rsp+16]

        movq        xmm3,       XMMWORD PTR [rsp+32]
        punpcklbw   xmm1,       xmm2

        movq        xmm4,       XMMWORD PTR [rsp+48]
        punpcklbw   xmm2,       xmm3

        movq        xmm5,       XMMWORD PTR [rsp+64]
        punpcklbw   xmm3,       xmm4

        movq        xmm6,       XMMWORD PTR [rsp+80]
        punpcklbw   xmm4,       xmm5

        movq        xmm7,       XMMWORD PTR [rsp+96]
        punpcklbw   xmm5,       xmm6

        ; Because the source register (xmm0) is always treated as signed by
        ; pmaddubsw, the constant '128' is treated as '-128'.
        pmaddubsw   xmm1,       xmm0
        pmaddubsw   xmm2,       xmm0

        pmaddubsw   xmm3,       xmm0
        pmaddubsw   xmm4,       xmm0

        pmaddubsw   xmm5,       xmm0
        punpcklbw   xmm6,       xmm7

        pmaddubsw   xmm6,       xmm0
        paddw       xmm1,       [GLOBAL(rd)]

        paddw       xmm2,       [GLOBAL(rd)]
        psraw       xmm1,       VP8_FILTER_SHIFT

        paddw       xmm3,       [GLOBAL(rd)]
        psraw       xmm2,       VP8_FILTER_SHIFT

        paddw       xmm4,       [GLOBAL(rd)]
        psraw       xmm3,       VP8_FILTER_SHIFT

        paddw       xmm5,       [GLOBAL(rd)]
        psraw       xmm4,       VP8_FILTER_SHIFT

        paddw       xmm6,       [GLOBAL(rd)]
        psraw       xmm5,       VP8_FILTER_SHIFT

        psraw       xmm6,       VP8_FILTER_SHIFT

        ; Having multiplied everything by '-128' and obtained negative
        ; numbers, the unsigned saturation truncates those values to 0,
        ; resulting in incorrect handling of xoffset == 0 && yoffset == 0
        packuswb    xmm1,       xmm1

        packuswb    xmm2,       xmm2
        movq        [rdi],      xmm1

        packuswb    xmm3,       xmm3
        movq        [rdi+rdx],  xmm2

        packuswb    xmm4,       xmm4
        movq        xmm1,       XMMWORD PTR [rsp+112]

        lea         rdi,        [rdi + 2*rdx]
        movq        xmm2,       XMMWORD PTR [rsp+128]

        packuswb    xmm5,       xmm5
        movq        [rdi],      xmm3

        packuswb    xmm6,       xmm6
        movq        [rdi+rdx],  xmm4

        lea         rdi,        [rdi + 2*rdx]
        punpcklbw   xmm7,       xmm1

        movq        [rdi],      xmm5
        pmaddubsw   xmm7,       xmm0

        movq        [rdi+rdx],  xmm6
        punpcklbw   xmm1,       xmm2

        pmaddubsw   xmm1,       xmm0
        paddw       xmm7,       [GLOBAL(rd)]

        psraw       xmm7,       VP8_FILTER_SHIFT
        paddw       xmm1,       [GLOBAL(rd)]

        psraw       xmm1,       VP8_FILTER_SHIFT
        packuswb    xmm7,       xmm7

        packuswb    xmm1,       xmm1
        lea         rdi,        [rdi + 2*rdx]

        movq        [rdi],      xmm7

        movq        [rdi+rdx],  xmm1
        lea         rsp,        [rsp + 144]

        jmp         .done8x8

.b8x8_fp_only:
        lea         rcx,        [rdi+rdx*8]

.next_row_fp:
        movdqa      xmm1,       XMMWORD PTR [rsp]
        movdqa      xmm3,       XMMWORD PTR [rsp+16]

        movdqa      xmm2,       xmm1
        movdqa      xmm5,       XMMWORD PTR [rsp+32]

        psrldq      xmm2,       1
        movdqa      xmm7,       XMMWORD PTR [rsp+48]

        movdqa      xmm4,       xmm3
        psrldq      xmm4,       1

        movdqa      xmm6,       xmm5
        psrldq      xmm6,       1

        punpcklbw   xmm1,       xmm2
        pmaddubsw   xmm1,       xmm0

        punpcklbw   xmm3,       xmm4
        pmaddubsw   xmm3,       xmm0

        punpcklbw   xmm5,       xmm6
        pmaddubsw   xmm5,       xmm0

        movdqa      xmm2,       xmm7
        psrldq      xmm2,       1

        punpcklbw   xmm7,       xmm2
        pmaddubsw   xmm7,       xmm0

        paddw       xmm1,       [GLOBAL(rd)]
        psraw       xmm1,       VP8_FILTER_SHIFT

        paddw       xmm3,       [GLOBAL(rd)]
        psraw       xmm3,       VP8_FILTER_SHIFT

        paddw       xmm5,       [GLOBAL(rd)]
        psraw       xmm5,       VP8_FILTER_SHIFT

        paddw       xmm7,       [GLOBAL(rd)]
        psraw       xmm7,       VP8_FILTER_SHIFT

        packuswb    xmm1,       xmm1
        packuswb    xmm3,       xmm3

        packuswb    xmm5,       xmm5
        movq        [rdi],      xmm1

        packuswb    xmm7,       xmm7
        movq        [rdi+rdx],  xmm3

        lea         rdi,        [rdi + 2*rdx]
        movq        [rdi],      xmm5

        lea         rsp,        [rsp + 4*16]
        movq        [rdi+rdx],  xmm7

        lea         rdi,        [rdi + 2*rdx]
        cmp         rdi,        rcx

        jne         .next_row_fp

        lea         rsp,        [rsp + 16]

.done8x8:
    ;add rsp, 144
    pop         rsp
    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
shuf1b:
    db 0, 5, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 6, 11, 7, 12
shuf2b:
    db 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11
shuf3b:
    db 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10

align 16
shuf2bfrom1:
    db  4, 8, 6, 1, 8, 3, 1, 5, 3, 7, 5, 9, 7,11, 9,13
align 16
shuf3bfrom1:
    db  2, 6, 4, 8, 6, 1, 8, 3, 1, 5, 3, 7, 5, 9, 7,11

align 16
rd:
    times 8 dw 0x40

align 16
k0_k5:
    times 8 db 0, 0             ;placeholder
    times 8 db 0, 0
    times 8 db 2, 1
    times 8 db 0, 0
    times 8 db 3, 3
    times 8 db 0, 0
    times 8 db 1, 2
    times 8 db 0, 0
k1_k3:
    times 8 db  0,    0         ;placeholder
    times 8 db  -6,  12
    times 8 db -11,  36
    times 8 db  -9,  50
    times 8 db -16,  77
    times 8 db  -6,  93
    times 8 db  -8, 108
    times 8 db  -1, 123
k2_k4:
    times 8 db 128,    0        ;placeholder
    times 8 db 123,   -1
    times 8 db 108,   -8
    times 8 db  93,   -6
    times 8 db  77,  -16
    times 8 db  50,   -9
    times 8 db  36,  -11
    times 8 db  12,   -6
align 16
vp8_bilinear_filters_ssse3:
    times 8 db 128, 0
    times 8 db 112, 16
    times 8 db 96,  32
    times 8 db 80,  48
    times 8 db 64,  64
    times 8 db 48,  80
    times 8 db 32,  96
    times 8 db 16,  112

