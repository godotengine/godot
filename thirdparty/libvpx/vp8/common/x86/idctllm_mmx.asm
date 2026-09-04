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

; /****************************************************************************
; * Notes:
; *
; * This implementation makes use of 16 bit fixed point version of two multiply
; * constants:
; *        1.   sqrt(2) * cos (pi/8)
; *        2.   sqrt(2) * sin (pi/8)
; * Because the first constant is bigger than 1, to maintain the same 16 bit
; * fixed point precision as the second one, we use a trick of
; *        x * a = x + x*(a-1)
; * so
; *        x * sqrt(2) * cos (pi/8) = x + x * (sqrt(2) *cos(pi/8)-1).
; *
; * For the second constant, because of the 16bit version is 35468, which
; * is bigger than 32768, in signed 16 bit multiply, it becomes a negative
; * number.
; *        (x * (unsigned)35468 >> 16) = x * (signed)35468 >> 16 + x
; *
; **************************************************************************/

SECTION .text

;void vp8_short_idct4x4llm_mmx(short *input, unsigned char *pred,
;int pitch, unsigned char *dest,int stride)
globalsym(vp8_short_idct4x4llm_mmx)
sym(vp8_short_idct4x4llm_mmx):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    mov         rax,    arg(0)              ;input
    mov         rsi,    arg(1)              ;pred

    movq        mm0,    [rax   ]
    movq        mm1,    [rax+ 8]
    movq        mm2,    [rax+16]
    movq        mm3,    [rax+24]

%if 0
    pxor        mm7,    mm7
    movq        [rax],   mm7
    movq        [rax+8], mm7
    movq        [rax+16],mm7
    movq        [rax+24],mm7
%endif
    movsxd      rax,    dword ptr arg(2)    ;pitch
    mov         rdx,    arg(3)              ;dest
    movsxd      rdi,    dword ptr arg(4)    ;stride


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

    movd        mm4,            [rsi]
    punpcklbw   mm4,            mm7
    paddsw      mm0,            mm4
    packuswb    mm0,            mm7
    movd        [rdx],          mm0

    movd        mm4,            [rsi+rax]
    punpcklbw   mm4,            mm7
    paddsw      mm1,            mm4
    packuswb    mm1,            mm7
    movd        [rdx+rdi],      mm1

    movd        mm4,            [rsi+2*rax]
    punpcklbw   mm4,            mm7
    paddsw      mm2,            mm4
    packuswb    mm2,            mm7
    movd        [rdx+rdi*2],    mm2

    add         rdx,            rdi
    add         rsi,            rax

    movd        mm4,            [rsi+2*rax]
    punpcklbw   mm4,            mm7
    paddsw      mm5,            mm4
    packuswb    mm5,            mm7
    movd        [rdx+rdi*2],    mm5

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vp8_dc_only_idct_add_mmx(
;short input_dc,
;unsigned char *pred_ptr,
;int pred_stride,
;unsigned char *dst_ptr,
;int stride)
globalsym(vp8_dc_only_idct_add_mmx)
sym(vp8_dc_only_idct_add_mmx):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    GET_GOT     rbx
    ; end prolog

        movd        mm5,            arg(0) ;input_dc
        mov         rax,            arg(1) ;pred_ptr
        movsxd      rdx,            dword ptr arg(2) ;pred_stride

        pxor        mm0,            mm0

        paddw       mm5,            [GLOBAL(fours)]
        lea         rcx,            [rdx + rdx*2]

        psraw       mm5,            3

        punpcklwd   mm5,            mm5

        punpckldq   mm5,            mm5

        movd        mm1,            [rax]
        movd        mm2,            [rax+rdx]
        movd        mm3,            [rax+2*rdx]
        movd        mm4,            [rax+rcx]

        mov         rax,            arg(3) ;d -- destination
        movsxd      rdx,            dword ptr arg(4) ;dst_stride

        punpcklbw   mm1,            mm0
        paddsw      mm1,            mm5
        packuswb    mm1,            mm0              ; pack and unpack to saturate
        lea         rcx,            [rdx + rdx*2]

        punpcklbw   mm2,            mm0
        paddsw      mm2,            mm5
        packuswb    mm2,            mm0              ; pack and unpack to saturate

        punpcklbw   mm3,            mm0
        paddsw      mm3,            mm5
        packuswb    mm3,            mm0              ; pack and unpack to saturate

        punpcklbw   mm4,            mm0
        paddsw      mm4,            mm5
        packuswb    mm4,            mm0              ; pack and unpack to saturate

        movd        [rax],          mm1
        movd        [rax+rdx],      mm2
        movd        [rax+2*rdx],    mm3
        movd        [rax+rcx],      mm4

    ; begin epilog
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
