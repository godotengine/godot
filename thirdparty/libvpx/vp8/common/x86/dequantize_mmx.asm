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


;void vp8_dequantize_b_impl_mmx(short *sq, short *dq, short *q)
global sym(vp8_dequantize_b_impl_mmx) PRIVATE
sym(vp8_dequantize_b_impl_mmx):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 3
    push        rsi
    push        rdi
    ; end prolog

        mov       rsi, arg(0) ;sq
        mov       rdi, arg(1) ;dq
        mov       rax, arg(2) ;q

        movq      mm1, [rsi]
        pmullw    mm1, [rax+0]            ; mm4 *= kernel 0 modifiers.
        movq      [rdi], mm1

        movq      mm1, [rsi+8]
        pmullw    mm1, [rax+8]            ; mm4 *= kernel 0 modifiers.
        movq      [rdi+8], mm1

        movq      mm1, [rsi+16]
        pmullw    mm1, [rax+16]            ; mm4 *= kernel 0 modifiers.
        movq      [rdi+16], mm1

        movq      mm1, [rsi+24]
        pmullw    mm1, [rax+24]            ; mm4 *= kernel 0 modifiers.
        movq      [rdi+24], mm1

    ; begin epilog
    pop rdi
    pop rsi
    UNSHADOW_ARGS
    pop         rbp
    ret


;void dequant_idct_add_mmx(
;short *input,            0
;short *dq,               1
;unsigned char *dest,     2
;int stride)              3
global sym(vp8_dequant_idct_add_mmx) PRIVATE
sym(vp8_dequant_idct_add_mmx):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 4
    GET_GOT     rbx
    push        rdi
    ; end prolog

        mov         rax,    arg(0) ;input
        mov         rdx,    arg(1) ;dq


        movq        mm0,    [rax   ]
        pmullw      mm0,    [rdx]

        movq        mm1,    [rax +8]
        pmullw      mm1,    [rdx +8]

        movq        mm2,    [rax+16]
        pmullw      mm2,    [rdx+16]

        movq        mm3,    [rax+24]
        pmullw      mm3,    [rdx+24]

        mov         rdx,    arg(2) ;dest

        pxor        mm7,    mm7


        movq        [rax],   mm7
        movq        [rax+8], mm7

        movq        [rax+16],mm7
        movq        [rax+24],mm7


        movsxd      rdi,            dword ptr arg(3) ;stride

        psubw       mm0,            mm2             ; b1= 0-2
        paddw       mm2,            mm2             ;

        movq        mm5,            mm1
        paddw       mm2,            mm0             ; a1 =0+2

        pmulhw      mm5,            [GLOBAL(x_s1sqr2)];
        paddw       mm5,            mm1             ; ip1 * sin(pi/8) * sqrt(2)

        movq        mm7,            mm3             ;
        pmulhw      mm7,            [GLOBAL(x_c1sqr2less1)];

        paddw       mm7,            mm3             ; ip3 * cos(pi/8) * sqrt(2)
        psubw       mm7,            mm5             ; c1

        movq        mm5,            mm1
        movq        mm4,            mm3

        pmulhw      mm5,            [GLOBAL(x_c1sqr2less1)]
        paddw       mm5,            mm1

        pmulhw      mm3,            [GLOBAL(x_s1sqr2)]
        paddw       mm3,            mm4

        paddw       mm3,            mm5             ; d1
        movq        mm6,            mm2             ; a1

        movq        mm4,            mm0             ; b1
        paddw       mm2,            mm3             ;0

        paddw       mm4,            mm7             ;1
        psubw       mm0,            mm7             ;2

        psubw       mm6,            mm3             ;3

        movq        mm1,            mm2             ; 03 02 01 00
        movq        mm3,            mm4             ; 23 22 21 20

        punpcklwd   mm1,            mm0             ; 11 01 10 00
        punpckhwd   mm2,            mm0             ; 13 03 12 02

        punpcklwd   mm3,            mm6             ; 31 21 30 20
        punpckhwd   mm4,            mm6             ; 33 23 32 22

        movq        mm0,            mm1             ; 11 01 10 00
        movq        mm5,            mm2             ; 13 03 12 02

        punpckldq   mm0,            mm3             ; 30 20 10 00
        punpckhdq   mm1,            mm3             ; 31 21 11 01

        punpckldq   mm2,            mm4             ; 32 22 12 02
        punpckhdq   mm5,            mm4             ; 33 23 13 03

        movq        mm3,            mm5             ; 33 23 13 03

        psubw       mm0,            mm2             ; b1= 0-2
        paddw       mm2,            mm2             ;

        movq        mm5,            mm1
        paddw       mm2,            mm0             ; a1 =0+2

        pmulhw      mm5,            [GLOBAL(x_s1sqr2)];
        paddw       mm5,            mm1             ; ip1 * sin(pi/8) * sqrt(2)

        movq        mm7,            mm3             ;
        pmulhw      mm7,            [GLOBAL(x_c1sqr2less1)];

        paddw       mm7,            mm3             ; ip3 * cos(pi/8) * sqrt(2)
        psubw       mm7,            mm5             ; c1

        movq        mm5,            mm1
        movq        mm4,            mm3

        pmulhw      mm5,            [GLOBAL(x_c1sqr2less1)]
        paddw       mm5,            mm1

        pmulhw      mm3,            [GLOBAL(x_s1sqr2)]
        paddw       mm3,            mm4

        paddw       mm3,            mm5             ; d1
        paddw       mm0,            [GLOBAL(fours)]

        paddw       mm2,            [GLOBAL(fours)]
        movq        mm6,            mm2             ; a1

        movq        mm4,            mm0             ; b1
        paddw       mm2,            mm3             ;0

        paddw       mm4,            mm7             ;1
        psubw       mm0,            mm7             ;2

        psubw       mm6,            mm3             ;3
        psraw       mm2,            3

        psraw       mm0,            3
        psraw       mm4,            3

        psraw       mm6,            3

        movq        mm1,            mm2             ; 03 02 01 00
        movq        mm3,            mm4             ; 23 22 21 20

        punpcklwd   mm1,            mm0             ; 11 01 10 00
        punpckhwd   mm2,            mm0             ; 13 03 12 02

        punpcklwd   mm3,            mm6             ; 31 21 30 20
        punpckhwd   mm4,            mm6             ; 33 23 32 22

        movq        mm0,            mm1             ; 11 01 10 00
        movq        mm5,            mm2             ; 13 03 12 02

        punpckldq   mm0,            mm3             ; 30 20 10 00
        punpckhdq   mm1,            mm3             ; 31 21 11 01

        punpckldq   mm2,            mm4             ; 32 22 12 02
        punpckhdq   mm5,            mm4             ; 33 23 13 03

        pxor        mm7,            mm7

        movd        mm4,            [rdx]
        punpcklbw   mm4,            mm7
        paddsw      mm0,            mm4
        packuswb    mm0,            mm7
        movd        [rdx],          mm0

        movd        mm4,            [rdx+rdi]
        punpcklbw   mm4,            mm7
        paddsw      mm1,            mm4
        packuswb    mm1,            mm7
        movd        [rdx+rdi],      mm1

        movd        mm4,            [rdx+2*rdi]
        punpcklbw   mm4,            mm7
        paddsw      mm2,            mm4
        packuswb    mm2,            mm7
        movd        [rdx+rdi*2],    mm2

        add         rdx,            rdi

        movd        mm4,            [rdx+2*rdi]
        punpcklbw   mm4,            mm7
        paddsw      mm5,            mm4
        packuswb    mm5,            mm7
        movd        [rdx+rdi*2],    mm5

    ; begin epilog
    pop rdi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
x_s1sqr2:
    times 4 dw 0x8A8C
align 16
x_c1sqr2less1:
    times 4 dw 0x4E7B
align 16
fours:
    times 4 dw 0x0004
