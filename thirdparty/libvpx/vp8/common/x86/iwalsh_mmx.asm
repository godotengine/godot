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

;void vp8_short_inv_walsh4x4_mmx(short *input, short *output)
global sym(vp8_short_inv_walsh4x4_mmx) PRIVATE
sym(vp8_short_inv_walsh4x4_mmx):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 2
    ; end prolog

    mov         rdx, arg(0)
    mov         rax, 30003h

    movq        mm0, [rdx + 0]    ;ip[0]
    movq        mm1, [rdx + 8]    ;ip[4]
    movq        mm7, rax

    movq        mm2, [rdx + 16]   ;ip[8]
    movq        mm3, [rdx + 24]   ;ip[12]
    punpcklwd   mm7, mm7          ;0003000300030003h
    mov         rdx, arg(1)

    movq        mm4, mm0
    movq        mm5, mm1

    paddw       mm4, mm3          ;ip[0] + ip[12] aka al
    paddw       mm5, mm2          ;ip[4] + ip[8] aka bl

    movq        mm6, mm4          ;temp al
    paddw       mm4, mm5          ;al + bl
    psubw       mm6, mm5          ;al - bl

    psubw       mm0, mm3          ;ip[0] - ip[12] aka d1
    psubw       mm1, mm2          ;ip[4] - ip[8] aka c1

    movq        mm5, mm0          ;temp dl
    paddw       mm0, mm1          ;dl + cl
    psubw       mm5, mm1          ;dl - cl

    ; 03 02 01 00
    ; 13 12 11 10
    ; 23 22 21 20
    ; 33 32 31 30

    movq        mm3, mm4          ; 03 02 01 00
    punpcklwd   mm4, mm0          ; 11 01 10 00
    punpckhwd   mm3, mm0          ; 13 03 12 02

    movq        mm1, mm6          ; 23 22 21 20
    punpcklwd   mm6, mm5          ; 31 21 30 20
    punpckhwd   mm1, mm5          ; 33 23 32 22

    movq        mm0, mm4          ; 11 01 10 00
    movq        mm2, mm3          ; 13 03 12 02

    punpckldq   mm0, mm6          ; 30 20 10 00 aka ip[0]
    punpckhdq   mm4, mm6          ; 31 21 11 01 aka ip[4]

    punpckldq   mm2, mm1          ; 32 22 12 02 aka ip[8]
    punpckhdq   mm3, mm1          ; 33 23 13 03 aka ip[12]
;~~~~~~~~~~~~~~~~~~~~~
    movq        mm1, mm0
    movq        mm5, mm4
    paddw       mm1, mm3          ;ip[0] + ip[12] aka al
    paddw       mm5, mm2          ;ip[4] + ip[8] aka bl

    movq        mm6, mm1          ;temp al
    paddw       mm1, mm5          ;al + bl
    psubw       mm6, mm5          ;al - bl
    paddw       mm1, mm7
    paddw       mm6, mm7
    psraw       mm1, 3
    psraw       mm6, 3

    psubw       mm0, mm3          ;ip[0] - ip[12] aka d1
    psubw       mm4, mm2          ;ip[4] - ip[8] aka c1

    movq        mm5, mm0          ;temp dl
    paddw       mm0, mm4          ;dl + cl
    psubw       mm5, mm4          ;dl - cl
    paddw       mm0, mm7
    paddw       mm5, mm7
    psraw       mm0, 3
    psraw       mm5, 3
;~~~~~~~~~~~~~~~~~~~~~

    movd        eax, mm1
    movd        ecx, mm0
    psrlq       mm0, 32
    psrlq       mm1, 32
    mov         word ptr[rdx+32*0], ax
    mov         word ptr[rdx+32*1], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*4], ax
    mov         word ptr[rdx+32*5], cx
    movd        eax, mm1
    movd        ecx, mm0
    mov         word ptr[rdx+32*8], ax
    mov         word ptr[rdx+32*9], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*12], ax
    mov         word ptr[rdx+32*13], cx

    movd        eax, mm6
    movd        ecx, mm5
    psrlq       mm5, 32
    psrlq       mm6, 32
    mov         word ptr[rdx+32*2], ax
    mov         word ptr[rdx+32*3], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*6], ax
    mov         word ptr[rdx+32*7], cx
    movd        eax, mm6
    movd        ecx, mm5
    mov         word ptr[rdx+32*10], ax
    mov         word ptr[rdx+32*11], cx
    shr         eax, 16
    shr         ecx, 16
    mov         word ptr[rdx+32*14], ax
    mov         word ptr[rdx+32*15], cx

    ; begin epilog
    UNSHADOW_ARGS
    pop         rbp
    ret

