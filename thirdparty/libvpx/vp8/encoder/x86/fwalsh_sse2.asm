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

;void vp8_short_walsh4x4_sse2(short *input, short *output, int pitch)
globalsym(vp8_short_walsh4x4_sse2)
sym(vp8_short_walsh4x4_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 3
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    mov     rsi, arg(0)           ; input
    mov     rdi, arg(1)           ; output
    movsxd  rdx, dword ptr arg(2) ; pitch

    ; first for loop
    movq    xmm0, MMWORD PTR [rsi]           ; load input
    movq    xmm1, MMWORD PTR [rsi + rdx]
    lea     rsi,  [rsi + rdx*2]
    movq    xmm2, MMWORD PTR [rsi]
    movq    xmm3, MMWORD PTR [rsi + rdx]

    punpcklwd xmm0,  xmm1
    punpcklwd xmm2,  xmm3

    movdqa    xmm1, xmm0
    punpckldq xmm0, xmm2           ; ip[1] ip[0]
    punpckhdq xmm1, xmm2           ; ip[3] ip[2]

    movdqa    xmm2, xmm0
    paddw     xmm0, xmm1
    psubw     xmm2, xmm1

    psllw     xmm0, 2              ; d1  a1
    psllw     xmm2, 2              ; c1  b1

    movdqa    xmm1, xmm0
    punpcklqdq xmm0, xmm2          ; b1  a1
    punpckhqdq xmm1, xmm2          ; c1  d1

    pxor      xmm6, xmm6
    movq      xmm6, xmm0
    pxor      xmm7, xmm7
    pcmpeqw   xmm7, xmm6
    paddw     xmm7, [GLOBAL(c1)]

    movdqa    xmm2, xmm0
    paddw     xmm0, xmm1           ; b1+c1  a1+d1
    psubw     xmm2, xmm1           ; b1-c1  a1-d1
    paddw     xmm0, xmm7           ; b1+c1  a1+d1+(a1!=0)

    ; second for loop
    ; input: 13  9  5  1 12  8  4  0 (xmm0)
    ;        14 10  6  2 15 11  7  3 (xmm2)
    ; after shuffle:
    ;        13  5  9  1 12  4  8  0 (xmm0)
    ;        14  6 10  2 15  7 11  3 (xmm1)
    pshuflw   xmm3, xmm0, 0xd8
    pshufhw   xmm0, xmm3, 0xd8
    pshuflw   xmm3, xmm2, 0xd8
    pshufhw   xmm1, xmm3, 0xd8

    movdqa    xmm2, xmm0
    pmaddwd   xmm0, [GLOBAL(c1)]    ; d11 a11 d10 a10
    pmaddwd   xmm2, [GLOBAL(cn1)]   ; c11 b11 c10 b10
    movdqa    xmm3, xmm1
    pmaddwd   xmm1, [GLOBAL(c1)]    ; d12 a12 d13 a13
    pmaddwd   xmm3, [GLOBAL(cn1)]   ; c12 b12 c13 b13

    pshufd    xmm4, xmm0, 0xd8      ; d11 d10 a11 a10
    pshufd    xmm5, xmm2, 0xd8      ; c11 c10 b11 b10
    pshufd    xmm6, xmm1, 0x72      ; d13 d12 a13 a12
    pshufd    xmm7, xmm3, 0x72      ; c13 c12 b13 b12

    movdqa    xmm0, xmm4
    punpcklqdq xmm0, xmm5           ; b11 b10 a11 a10
    punpckhqdq xmm4, xmm5           ; c11 c10 d11 d10
    movdqa    xmm1, xmm6
    punpcklqdq xmm1, xmm7           ; b13 b12 a13 a12
    punpckhqdq xmm6, xmm7           ; c13 c12 d13 d12

    movdqa    xmm2, xmm0
    paddd     xmm0, xmm4            ; b21 b20 a21 a20
    psubd     xmm2, xmm4            ; c21 c20 d21 d20
    movdqa    xmm3, xmm1
    paddd     xmm1, xmm6            ; b23 b22 a23 a22
    psubd     xmm3, xmm6            ; c23 c22 d23 d22

    pxor      xmm4, xmm4
    movdqa    xmm5, xmm4
    pcmpgtd   xmm4, xmm0
    pcmpgtd   xmm5, xmm2
    pand      xmm4, [GLOBAL(cd1)]
    pand      xmm5, [GLOBAL(cd1)]

    pxor      xmm6, xmm6
    movdqa    xmm7, xmm6
    pcmpgtd   xmm6, xmm1
    pcmpgtd   xmm7, xmm3
    pand      xmm6, [GLOBAL(cd1)]
    pand      xmm7, [GLOBAL(cd1)]

    paddd     xmm0, xmm4
    paddd     xmm2, xmm5
    paddd     xmm0, [GLOBAL(cd3)]
    paddd     xmm2, [GLOBAL(cd3)]
    paddd     xmm1, xmm6
    paddd     xmm3, xmm7
    paddd     xmm1, [GLOBAL(cd3)]
    paddd     xmm3, [GLOBAL(cd3)]

    psrad     xmm0, 3
    psrad     xmm1, 3
    psrad     xmm2, 3
    psrad     xmm3, 3
    movdqa    xmm4, xmm0
    punpcklqdq xmm0, xmm1           ; a23 a22 a21 a20
    punpckhqdq xmm4, xmm1           ; b23 b22 b21 b20
    movdqa    xmm5, xmm2
    punpckhqdq xmm2, xmm3           ; c23 c22 c21 c20
    punpcklqdq xmm5, xmm3           ; d23 d22 d21 d20

    packssdw  xmm0, xmm4            ; b23 b22 b21 b20 a23 a22 a21 a20
    packssdw  xmm2, xmm5            ; d23 d22 d21 d20 c23 c22 c21 c20

    movdqa  XMMWORD PTR [rdi], xmm0
    movdqa  XMMWORD PTR [rdi + 16], xmm2

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
c1:
    dw 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001
align 16
cn1:
    dw 0x0001, 0xffff, 0x0001, 0xffff, 0x0001, 0xffff, 0x0001, 0xffff
align 16
cd1:
    dd 0x00000001, 0x00000001, 0x00000001, 0x00000001
align 16
cd3:
    dd 0x00000003, 0x00000003, 0x00000003, 0x00000003
