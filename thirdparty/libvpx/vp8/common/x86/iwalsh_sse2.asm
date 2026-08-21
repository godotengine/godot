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

;void vp8_short_inv_walsh4x4_sse2(short *input, short *mb_dqcoeff)
globalsym(vp8_short_inv_walsh4x4_sse2)
sym(vp8_short_inv_walsh4x4_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 2
    ; end prolog

    mov         rcx, arg(0)
    mov         rdx, arg(1)
    mov         rax, 30003h

    movdqa      xmm0, [rcx + 0]     ;ip[4] ip[0]
    movdqa      xmm1, [rcx + 16]    ;ip[12] ip[8]


    pshufd      xmm2, xmm1, 4eh     ;ip[8] ip[12]
    movdqa      xmm3, xmm0          ;ip[4] ip[0]

    paddw       xmm0, xmm2          ;ip[4]+ip[8] ip[0]+ip[12] aka b1 a1
    psubw       xmm3, xmm2          ;ip[4]-ip[8] ip[0]-ip[12] aka c1 d1

    movdqa      xmm4, xmm0
    punpcklqdq  xmm0, xmm3          ;d1 a1
    punpckhqdq  xmm4, xmm3          ;c1 b1

    movdqa      xmm1, xmm4          ;c1 b1
    paddw       xmm4, xmm0          ;dl+cl a1+b1 aka op[4] op[0]
    psubw       xmm0, xmm1          ;d1-c1 a1-b1 aka op[12] op[8]

    ;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ; 13 12 11 10 03 02 01 00
    ;
    ; 33 32 31 30 23 22 21 20
    ;
    movdqa      xmm3, xmm4          ; 13 12 11 10 03 02 01 00
    punpcklwd   xmm4, xmm0          ; 23 03 22 02 21 01 20 00
    punpckhwd   xmm3, xmm0          ; 33 13 32 12 31 11 30 10
    movdqa      xmm1, xmm4          ; 23 03 22 02 21 01 20 00
    punpcklwd   xmm4, xmm3          ; 31 21 11 01 30 20 10 00
    punpckhwd   xmm1, xmm3          ; 33 23 13 03 32 22 12 02
    ;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    movd        xmm0, eax
    pshufd      xmm2, xmm1, 4eh     ;ip[8] ip[12]
    movdqa      xmm3, xmm4          ;ip[4] ip[0]

    pshufd      xmm0, xmm0, 0       ;03 03 03 03 03 03 03 03

    paddw       xmm4, xmm2          ;ip[4]+ip[8] ip[0]+ip[12] aka b1 a1
    psubw       xmm3, xmm2          ;ip[4]-ip[8] ip[0]-ip[12] aka c1 d1

    movdqa      xmm5, xmm4
    punpcklqdq  xmm4, xmm3          ;d1 a1
    punpckhqdq  xmm5, xmm3          ;c1 b1

    movdqa      xmm1, xmm5          ;c1 b1
    paddw       xmm5, xmm4          ;dl+cl a1+b1 aka op[4] op[0]
    psubw       xmm4, xmm1          ;d1-c1 a1-b1 aka op[12] op[8]

    paddw       xmm5, xmm0
    paddw       xmm4, xmm0
    psraw       xmm5, 3
    psraw       xmm4, 3

    movd        eax, xmm5
    movd        ecx, xmm4
    psrldq      xmm5, 4
    psrldq      xmm4, 4
    mov         word ptr[rdx+32*0], ax
    mov         word ptr[rdx+32*2], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*4], ax
    mov         word ptr[rdx+32*6], cx
    movd        eax, xmm5
    movd        ecx, xmm4
    psrldq      xmm5, 4
    psrldq      xmm4, 4
    mov         word ptr[rdx+32*8], ax
    mov         word ptr[rdx+32*10], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*12], ax
    mov         word ptr[rdx+32*14], cx

    movd        eax, xmm5
    movd        ecx, xmm4
    psrldq      xmm5, 4
    psrldq      xmm4, 4
    mov         word ptr[rdx+32*1], ax
    mov         word ptr[rdx+32*3], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*5], ax
    mov         word ptr[rdx+32*7], cx
    movd        eax, xmm5
    movd        ecx, xmm4
    mov         word ptr[rdx+32*9], ax
    mov         word ptr[rdx+32*11], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*13], ax
    mov         word ptr[rdx+32*15], cx

    ; begin epilog
    UNSHADOW_ARGS
    pop         rbp
    ret
