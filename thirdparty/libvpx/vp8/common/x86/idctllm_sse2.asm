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

;void vp8_idct_dequant_0_2x_sse2
; (
;   short *qcoeff       - 0
;   short *dequant      - 1
;   unsigned char *dst  - 2
;   int dst_stride      - 3
; )

SECTION .text

globalsym(vp8_idct_dequant_0_2x_sse2)
sym(vp8_idct_dequant_0_2x_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 4
    GET_GOT     rbx
    ; end prolog

        mov         rdx,            arg(1) ; dequant
        mov         rax,            arg(0) ; qcoeff

        movd        xmm4,           [rax]
        movd        xmm5,           [rdx]

        pinsrw      xmm4,           [rax+32],   4
        pinsrw      xmm5,           [rdx],      4

        pmullw      xmm4,           xmm5

    ; Zero out xmm5, for use unpacking
        pxor        xmm5,           xmm5

    ; clear coeffs
        movd        [rax],          xmm5
        movd        [rax+32],       xmm5
;pshufb
        mov         rax,            arg(2) ; dst
        movsxd      rdx,            dword ptr arg(3) ; dst_stride

        pshuflw     xmm4,           xmm4,       00000000b
        pshufhw     xmm4,           xmm4,       00000000b

        lea         rcx,            [rdx + rdx*2]
        paddw       xmm4,           [GLOBAL(fours)]

        psraw       xmm4,           3

        movq        xmm0,           [rax]
        movq        xmm1,           [rax+rdx]
        movq        xmm2,           [rax+2*rdx]
        movq        xmm3,           [rax+rcx]

        punpcklbw   xmm0,           xmm5
        punpcklbw   xmm1,           xmm5
        punpcklbw   xmm2,           xmm5
        punpcklbw   xmm3,           xmm5


    ; Add to predict buffer
        paddw       xmm0,           xmm4
        paddw       xmm1,           xmm4
        paddw       xmm2,           xmm4
        paddw       xmm3,           xmm4

    ; pack up before storing
        packuswb    xmm0,           xmm5
        packuswb    xmm1,           xmm5
        packuswb    xmm2,           xmm5
        packuswb    xmm3,           xmm5

    ; store blocks back out
        movq        [rax],          xmm0
        movq        [rax + rdx],    xmm1

        lea         rax,            [rax + 2*rdx]

        movq        [rax],          xmm2
        movq        [rax + rdx],    xmm3

    ; begin epilog
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vp8_idct_dequant_full_2x_sse2
; (
;   short *qcoeff       - 0
;   short *dequant      - 1
;   unsigned char *dst  - 2
;   int dst_stride      - 3
; )
globalsym(vp8_idct_dequant_full_2x_sse2)
sym(vp8_idct_dequant_full_2x_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 4
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    ; special case when 2 blocks have 0 or 1 coeffs
    ; dc is set as first coeff, so no need to load qcoeff
        mov         rax,            arg(0) ; qcoeff
        mov         rdx,            arg(1)  ; dequant
        mov         rdi,            arg(2) ; dst


    ; Zero out xmm7, for use unpacking
        pxor        xmm7,           xmm7


    ; note the transpose of xmm1 and xmm2, necessary for shuffle
    ;   to spit out sensicle data
        movdqa      xmm0,           [rax]
        movdqa      xmm2,           [rax+16]
        movdqa      xmm1,           [rax+32]
        movdqa      xmm3,           [rax+48]

    ; Clear out coeffs
        movdqa      [rax],          xmm7
        movdqa      [rax+16],       xmm7
        movdqa      [rax+32],       xmm7
        movdqa      [rax+48],       xmm7

    ; dequantize qcoeff buffer
        pmullw      xmm0,           [rdx]
        pmullw      xmm2,           [rdx+16]
        pmullw      xmm1,           [rdx]
        pmullw      xmm3,           [rdx+16]
        movsxd      rdx,            dword ptr arg(3) ; dst_stride

    ; repack so block 0 row x and block 1 row x are together
        movdqa      xmm4,           xmm0
        punpckldq   xmm0,           xmm1
        punpckhdq   xmm4,           xmm1

        pshufd      xmm0,           xmm0,       11011000b
        pshufd      xmm1,           xmm4,       11011000b

        movdqa      xmm4,           xmm2
        punpckldq   xmm2,           xmm3
        punpckhdq   xmm4,           xmm3

        pshufd      xmm2,           xmm2,       11011000b
        pshufd      xmm3,           xmm4,       11011000b

    ; first pass
        psubw       xmm0,           xmm2        ; b1 = 0-2
        paddw       xmm2,           xmm2        ;

        movdqa      xmm5,           xmm1
        paddw       xmm2,           xmm0        ; a1 = 0+2

        pmulhw      xmm5,           [GLOBAL(x_s1sqr2)]
        lea         rcx,            [rdx + rdx*2]   ;dst_stride * 3
        paddw       xmm5,           xmm1        ; ip1 * sin(pi/8) * sqrt(2)

        movdqa      xmm7,           xmm3
        pmulhw      xmm7,           [GLOBAL(x_c1sqr2less1)]

        paddw       xmm7,           xmm3        ; ip3 * cos(pi/8) * sqrt(2)
        psubw       xmm7,           xmm5        ; c1

        movdqa      xmm5,           xmm1
        movdqa      xmm4,           xmm3

        pmulhw      xmm5,           [GLOBAL(x_c1sqr2less1)]
        paddw       xmm5,           xmm1

        pmulhw      xmm3,           [GLOBAL(x_s1sqr2)]
        paddw       xmm3,           xmm4

        paddw       xmm3,           xmm5        ; d1
        movdqa      xmm6,           xmm2        ; a1

        movdqa      xmm4,           xmm0        ; b1
        paddw       xmm2,           xmm3        ;0

        paddw       xmm4,           xmm7        ;1
        psubw       xmm0,           xmm7        ;2

        psubw       xmm6,           xmm3        ;3

    ; transpose for the second pass
        movdqa      xmm7,           xmm2        ; 103 102 101 100 003 002 001 000
        punpcklwd   xmm2,           xmm0        ; 007 003 006 002 005 001 004 000
        punpckhwd   xmm7,           xmm0        ; 107 103 106 102 105 101 104 100

        movdqa      xmm5,           xmm4        ; 111 110 109 108 011 010 009 008
        punpcklwd   xmm4,           xmm6        ; 015 011 014 010 013 009 012 008
        punpckhwd   xmm5,           xmm6        ; 115 111 114 110 113 109 112 108


        movdqa      xmm1,           xmm2        ; 007 003 006 002 005 001 004 000
        punpckldq   xmm2,           xmm4        ; 013 009 005 001 012 008 004 000
        punpckhdq   xmm1,           xmm4        ; 015 011 007 003 014 010 006 002

        movdqa      xmm6,           xmm7        ; 107 103 106 102 105 101 104 100
        punpckldq   xmm7,           xmm5        ; 113 109 105 101 112 108 104 100
        punpckhdq   xmm6,           xmm5        ; 115 111 107 103 114 110 106 102


        movdqa      xmm5,           xmm2        ; 013 009 005 001 012 008 004 000
        punpckldq   xmm2,           xmm7        ; 112 108 012 008 104 100 004 000
        punpckhdq   xmm5,           xmm7        ; 113 109 013 009 105 101 005 001

        movdqa      xmm7,           xmm1        ; 015 011 007 003 014 010 006 002
        punpckldq   xmm1,           xmm6        ; 114 110 014 010 106 102 006 002
        punpckhdq   xmm7,           xmm6        ; 115 111 015 011 107 103 007 003

        pshufd      xmm0,           xmm2,       11011000b
        pshufd      xmm2,           xmm1,       11011000b

        pshufd      xmm1,           xmm5,       11011000b
        pshufd      xmm3,           xmm7,       11011000b

    ; second pass
        psubw       xmm0,           xmm2            ; b1 = 0-2
        paddw       xmm2,           xmm2

        movdqa      xmm5,           xmm1
        paddw       xmm2,           xmm0            ; a1 = 0+2

        pmulhw      xmm5,           [GLOBAL(x_s1sqr2)]
        paddw       xmm5,           xmm1            ; ip1 * sin(pi/8) * sqrt(2)

        movdqa      xmm7,           xmm3
        pmulhw      xmm7,           [GLOBAL(x_c1sqr2less1)]

        paddw       xmm7,           xmm3            ; ip3 * cos(pi/8) * sqrt(2)
        psubw       xmm7,           xmm5            ; c1

        movdqa      xmm5,           xmm1
        movdqa      xmm4,           xmm3

        pmulhw      xmm5,           [GLOBAL(x_c1sqr2less1)]
        paddw       xmm5,           xmm1

        pmulhw      xmm3,           [GLOBAL(x_s1sqr2)]
        paddw       xmm3,           xmm4

        paddw       xmm3,           xmm5            ; d1
        paddw       xmm0,           [GLOBAL(fours)]

        paddw       xmm2,           [GLOBAL(fours)]
        movdqa      xmm6,           xmm2            ; a1

        movdqa      xmm4,           xmm0            ; b1
        paddw       xmm2,           xmm3            ;0

        paddw       xmm4,           xmm7            ;1
        psubw       xmm0,           xmm7            ;2

        psubw       xmm6,           xmm3            ;3
        psraw       xmm2,           3

        psraw       xmm0,           3
        psraw       xmm4,           3

        psraw       xmm6,           3

    ; transpose to save
        movdqa      xmm7,           xmm2        ; 103 102 101 100 003 002 001 000
        punpcklwd   xmm2,           xmm0        ; 007 003 006 002 005 001 004 000
        punpckhwd   xmm7,           xmm0        ; 107 103 106 102 105 101 104 100

        movdqa      xmm5,           xmm4        ; 111 110 109 108 011 010 009 008
        punpcklwd   xmm4,           xmm6        ; 015 011 014 010 013 009 012 008
        punpckhwd   xmm5,           xmm6        ; 115 111 114 110 113 109 112 108


        movdqa      xmm1,           xmm2        ; 007 003 006 002 005 001 004 000
        punpckldq   xmm2,           xmm4        ; 013 009 005 001 012 008 004 000
        punpckhdq   xmm1,           xmm4        ; 015 011 007 003 014 010 006 002

        movdqa      xmm6,           xmm7        ; 107 103 106 102 105 101 104 100
        punpckldq   xmm7,           xmm5        ; 113 109 105 101 112 108 104 100
        punpckhdq   xmm6,           xmm5        ; 115 111 107 103 114 110 106 102


        movdqa      xmm5,           xmm2        ; 013 009 005 001 012 008 004 000
        punpckldq   xmm2,           xmm7        ; 112 108 012 008 104 100 004 000
        punpckhdq   xmm5,           xmm7        ; 113 109 013 009 105 101 005 001

        movdqa      xmm7,           xmm1        ; 015 011 007 003 014 010 006 002
        punpckldq   xmm1,           xmm6        ; 114 110 014 010 106 102 006 002
        punpckhdq   xmm7,           xmm6        ; 115 111 015 011 107 103 007 003

        pshufd      xmm0,           xmm2,       11011000b
        pshufd      xmm2,           xmm1,       11011000b

        pshufd      xmm1,           xmm5,       11011000b
        pshufd      xmm3,           xmm7,       11011000b

        pxor        xmm7,           xmm7

    ; Load up predict blocks
        movq        xmm4,           [rdi]
        movq        xmm5,           [rdi+rdx]

        punpcklbw   xmm4,           xmm7
        punpcklbw   xmm5,           xmm7

        paddw       xmm0,           xmm4
        paddw       xmm1,           xmm5

        movq        xmm4,           [rdi+2*rdx]
        movq        xmm5,           [rdi+rcx]

        punpcklbw   xmm4,           xmm7
        punpcklbw   xmm5,           xmm7

        paddw       xmm2,           xmm4
        paddw       xmm3,           xmm5

.finish:

    ; pack up before storing
        packuswb    xmm0,           xmm7
        packuswb    xmm1,           xmm7
        packuswb    xmm2,           xmm7
        packuswb    xmm3,           xmm7

    ; store blocks back out
        movq        [rdi],          xmm0
        movq        [rdi + rdx],    xmm1
        movq        [rdi + rdx*2],  xmm2
        movq        [rdi + rcx],    xmm3

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

;void vp8_idct_dequant_dc_0_2x_sse2
; (
;   short *qcoeff       - 0
;   short *dequant      - 1
;   unsigned char *dst  - 2
;   int dst_stride      - 3
;   short *dc           - 4
; )
globalsym(vp8_idct_dequant_dc_0_2x_sse2)
sym(vp8_idct_dequant_dc_0_2x_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    GET_GOT     rbx
    push        rdi
    ; end prolog

    ; special case when 2 blocks have 0 or 1 coeffs
    ; dc is set as first coeff, so no need to load qcoeff
        mov         rax,            arg(0) ; qcoeff

        mov         rdi,            arg(2) ; dst
        mov         rdx,            arg(4) ; dc

    ; Zero out xmm5, for use unpacking
        pxor        xmm5,           xmm5

    ; load up 2 dc words here == 2*16 = doubleword
        movd        xmm4,           [rdx]

        movsxd      rdx,            dword ptr arg(3) ; dst_stride
        lea         rcx, [rdx + rdx*2]
    ; Load up predict blocks
        movq        xmm0,           [rdi]
        movq        xmm1,           [rdi+rdx*1]
        movq        xmm2,           [rdi+rdx*2]
        movq        xmm3,           [rdi+rcx]

    ; Duplicate and expand dc across
        punpcklwd   xmm4,           xmm4
        punpckldq   xmm4,           xmm4

    ; Rounding to dequant and downshift
        paddw       xmm4,           [GLOBAL(fours)]
        psraw       xmm4,           3

    ; Predict buffer needs to be expanded from bytes to words
        punpcklbw   xmm0,           xmm5
        punpcklbw   xmm1,           xmm5
        punpcklbw   xmm2,           xmm5
        punpcklbw   xmm3,           xmm5

    ; Add to predict buffer
        paddw       xmm0,           xmm4
        paddw       xmm1,           xmm4
        paddw       xmm2,           xmm4
        paddw       xmm3,           xmm4

    ; pack up before storing
        packuswb    xmm0,           xmm5
        packuswb    xmm1,           xmm5
        packuswb    xmm2,           xmm5
        packuswb    xmm3,           xmm5

    ; store blocks back out
        movq        [rdi],          xmm0
        movq        [rdi + rdx],    xmm1
        movq        [rdi + rdx*2],  xmm2
        movq        [rdi + rcx],    xmm3

    ; begin epilog
    pop         rdi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret
;void vp8_idct_dequant_dc_full_2x_sse2
; (
;   short *qcoeff       - 0
;   short *dequant      - 1
;   unsigned char *dst  - 2
;   int dst_stride      - 3
;   short *dc           - 4
; )
globalsym(vp8_idct_dequant_dc_full_2x_sse2)
sym(vp8_idct_dequant_dc_full_2x_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    SAVE_XMM 7
    GET_GOT     rbx
    push        rdi
    ; end prolog

    ; special case when 2 blocks have 0 or 1 coeffs
    ; dc is set as first coeff, so no need to load qcoeff
        mov         rax,            arg(0) ; qcoeff
        mov         rdx,            arg(1)  ; dequant

        mov         rdi,            arg(2) ; dst

    ; Zero out xmm7, for use unpacking
        pxor        xmm7,           xmm7


    ; note the transpose of xmm1 and xmm2, necessary for shuffle
    ;   to spit out sensicle data
        movdqa      xmm0,           [rax]
        movdqa      xmm2,           [rax+16]
        movdqa      xmm1,           [rax+32]
        movdqa      xmm3,           [rax+48]

    ; Clear out coeffs
        movdqa      [rax],          xmm7
        movdqa      [rax+16],       xmm7
        movdqa      [rax+32],       xmm7
        movdqa      [rax+48],       xmm7

    ; dequantize qcoeff buffer
        pmullw      xmm0,           [rdx]
        pmullw      xmm2,           [rdx+16]
        pmullw      xmm1,           [rdx]
        pmullw      xmm3,           [rdx+16]

    ; DC component
        mov         rdx,            arg(4)

    ; repack so block 0 row x and block 1 row x are together
        movdqa      xmm4,           xmm0
        punpckldq   xmm0,           xmm1
        punpckhdq   xmm4,           xmm1

        pshufd      xmm0,           xmm0,       11011000b
        pshufd      xmm1,           xmm4,       11011000b

        movdqa      xmm4,           xmm2
        punpckldq   xmm2,           xmm3
        punpckhdq   xmm4,           xmm3

        pshufd      xmm2,           xmm2,       11011000b
        pshufd      xmm3,           xmm4,       11011000b

    ; insert DC component
        pinsrw      xmm0,           [rdx],      0
        pinsrw      xmm0,           [rdx+2],    4

    ; first pass
        psubw       xmm0,           xmm2        ; b1 = 0-2
        paddw       xmm2,           xmm2        ;

        movdqa      xmm5,           xmm1
        paddw       xmm2,           xmm0        ; a1 = 0+2

        pmulhw      xmm5,           [GLOBAL(x_s1sqr2)]
        paddw       xmm5,           xmm1        ; ip1 * sin(pi/8) * sqrt(2)

        movdqa      xmm7,           xmm3
        pmulhw      xmm7,           [GLOBAL(x_c1sqr2less1)]

        paddw       xmm7,           xmm3        ; ip3 * cos(pi/8) * sqrt(2)
        psubw       xmm7,           xmm5        ; c1

        movdqa      xmm5,           xmm1
        movdqa      xmm4,           xmm3

        pmulhw      xmm5,           [GLOBAL(x_c1sqr2less1)]
        paddw       xmm5,           xmm1

        pmulhw      xmm3,           [GLOBAL(x_s1sqr2)]
        paddw       xmm3,           xmm4

        paddw       xmm3,           xmm5        ; d1
        movdqa      xmm6,           xmm2        ; a1

        movdqa      xmm4,           xmm0        ; b1
        paddw       xmm2,           xmm3        ;0

        paddw       xmm4,           xmm7        ;1
        psubw       xmm0,           xmm7        ;2

        psubw       xmm6,           xmm3        ;3

    ; transpose for the second pass
        movdqa      xmm7,           xmm2        ; 103 102 101 100 003 002 001 000
        punpcklwd   xmm2,           xmm0        ; 007 003 006 002 005 001 004 000
        punpckhwd   xmm7,           xmm0        ; 107 103 106 102 105 101 104 100

        movdqa      xmm5,           xmm4        ; 111 110 109 108 011 010 009 008
        punpcklwd   xmm4,           xmm6        ; 015 011 014 010 013 009 012 008
        punpckhwd   xmm5,           xmm6        ; 115 111 114 110 113 109 112 108


        movdqa      xmm1,           xmm2        ; 007 003 006 002 005 001 004 000
        punpckldq   xmm2,           xmm4        ; 013 009 005 001 012 008 004 000
        punpckhdq   xmm1,           xmm4        ; 015 011 007 003 014 010 006 002

        movdqa      xmm6,           xmm7        ; 107 103 106 102 105 101 104 100
        punpckldq   xmm7,           xmm5        ; 113 109 105 101 112 108 104 100
        punpckhdq   xmm6,           xmm5        ; 115 111 107 103 114 110 106 102


        movdqa      xmm5,           xmm2        ; 013 009 005 001 012 008 004 000
        punpckldq   xmm2,           xmm7        ; 112 108 012 008 104 100 004 000
        punpckhdq   xmm5,           xmm7        ; 113 109 013 009 105 101 005 001

        movdqa      xmm7,           xmm1        ; 015 011 007 003 014 010 006 002
        punpckldq   xmm1,           xmm6        ; 114 110 014 010 106 102 006 002
        punpckhdq   xmm7,           xmm6        ; 115 111 015 011 107 103 007 003

        pshufd      xmm0,           xmm2,       11011000b
        pshufd      xmm2,           xmm1,       11011000b

        pshufd      xmm1,           xmm5,       11011000b
        pshufd      xmm3,           xmm7,       11011000b

    ; second pass
        psubw       xmm0,           xmm2            ; b1 = 0-2
        paddw       xmm2,           xmm2

        movdqa      xmm5,           xmm1
        paddw       xmm2,           xmm0            ; a1 = 0+2

        pmulhw      xmm5,           [GLOBAL(x_s1sqr2)]
        paddw       xmm5,           xmm1            ; ip1 * sin(pi/8) * sqrt(2)

        movdqa      xmm7,           xmm3
        pmulhw      xmm7,           [GLOBAL(x_c1sqr2less1)]

        paddw       xmm7,           xmm3            ; ip3 * cos(pi/8) * sqrt(2)
        psubw       xmm7,           xmm5            ; c1

        movdqa      xmm5,           xmm1
        movdqa      xmm4,           xmm3

        pmulhw      xmm5,           [GLOBAL(x_c1sqr2less1)]
        paddw       xmm5,           xmm1

        pmulhw      xmm3,           [GLOBAL(x_s1sqr2)]
        paddw       xmm3,           xmm4

        paddw       xmm3,           xmm5            ; d1
        paddw       xmm0,           [GLOBAL(fours)]

        paddw       xmm2,           [GLOBAL(fours)]
        movdqa      xmm6,           xmm2            ; a1

        movdqa      xmm4,           xmm0            ; b1
        paddw       xmm2,           xmm3            ;0

        paddw       xmm4,           xmm7            ;1
        psubw       xmm0,           xmm7            ;2

        psubw       xmm6,           xmm3            ;3
        psraw       xmm2,           3

        psraw       xmm0,           3
        psraw       xmm4,           3

        psraw       xmm6,           3

    ; transpose to save
        movdqa      xmm7,           xmm2        ; 103 102 101 100 003 002 001 000
        punpcklwd   xmm2,           xmm0        ; 007 003 006 002 005 001 004 000
        punpckhwd   xmm7,           xmm0        ; 107 103 106 102 105 101 104 100

        movdqa      xmm5,           xmm4        ; 111 110 109 108 011 010 009 008
        punpcklwd   xmm4,           xmm6        ; 015 011 014 010 013 009 012 008
        punpckhwd   xmm5,           xmm6        ; 115 111 114 110 113 109 112 108


        movdqa      xmm1,           xmm2        ; 007 003 006 002 005 001 004 000
        punpckldq   xmm2,           xmm4        ; 013 009 005 001 012 008 004 000
        punpckhdq   xmm1,           xmm4        ; 015 011 007 003 014 010 006 002

        movdqa      xmm6,           xmm7        ; 107 103 106 102 105 101 104 100
        punpckldq   xmm7,           xmm5        ; 113 109 105 101 112 108 104 100
        punpckhdq   xmm6,           xmm5        ; 115 111 107 103 114 110 106 102


        movdqa      xmm5,           xmm2        ; 013 009 005 001 012 008 004 000
        punpckldq   xmm2,           xmm7        ; 112 108 012 008 104 100 004 000
        punpckhdq   xmm5,           xmm7        ; 113 109 013 009 105 101 005 001

        movdqa      xmm7,           xmm1        ; 015 011 007 003 014 010 006 002
        punpckldq   xmm1,           xmm6        ; 114 110 014 010 106 102 006 002
        punpckhdq   xmm7,           xmm6        ; 115 111 015 011 107 103 007 003

        pshufd      xmm0,           xmm2,       11011000b
        pshufd      xmm2,           xmm1,       11011000b

        pshufd      xmm1,           xmm5,       11011000b
        pshufd      xmm3,           xmm7,       11011000b

        pxor        xmm7,           xmm7

    ; Load up predict blocks
        movsxd      rdx,            dword ptr arg(3) ; dst_stride
        movq        xmm4,           [rdi]
        movq        xmm5,           [rdi+rdx]
        lea         rcx,            [rdx + rdx*2]

        punpcklbw   xmm4,           xmm7
        punpcklbw   xmm5,           xmm7

        paddw       xmm0,           xmm4
        paddw       xmm1,           xmm5

        movq        xmm4,           [rdi+rdx*2]
        movq        xmm5,           [rdi+rcx]

        punpcklbw   xmm4,           xmm7
        punpcklbw   xmm5,           xmm7

        paddw       xmm2,           xmm4
        paddw       xmm3,           xmm5

.finish:

    ; pack up before storing
        packuswb    xmm0,           xmm7
        packuswb    xmm1,           xmm7
        packuswb    xmm2,           xmm7
        packuswb    xmm3,           xmm7

    ; Load destination stride before writing out,
    ;   doesn't need to persist
        movsxd      rdx,            dword ptr arg(3) ; dst_stride

    ; store blocks back out
        movq        [rdi],          xmm0
        movq        [rdi + rdx],    xmm1

        lea         rdi,            [rdi + 2*rdx]

        movq        [rdi],          xmm2
        movq        [rdi + rdx],    xmm3


    ; begin epilog
    pop         rdi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
fours:
    times 8 dw 0x0004
align 16
x_s1sqr2:
    times 8 dw 0x8A8C
align 16
x_c1sqr2less1:
    times 8 dw 0x4E7B
