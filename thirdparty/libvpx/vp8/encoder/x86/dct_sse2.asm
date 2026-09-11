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

%macro STACK_FRAME_CREATE 0
%if ABI_IS_32BIT
  %define       input       rsi
  %define       output      rdi
  %define       pitch       rax
    push        rbp
    mov         rbp, rsp
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    mov         rsi, arg(0)
    mov         rdi, arg(1)

    movsxd      rax, dword ptr arg(2)
    lea         rcx, [rsi + rax*2]
%else
  %if LIBVPX_YASM_WIN64
    %define     input       rcx
    %define     output      rdx
    %define     pitch       r8
    SAVE_XMM 7, u
  %else
    %define     input       rdi
    %define     output      rsi
    %define     pitch       rdx
  %endif
%endif
%endmacro

%macro STACK_FRAME_DESTROY 0
  %define     input
  %define     output
  %define     pitch

%if ABI_IS_32BIT
    pop         rdi
    pop         rsi
    RESTORE_GOT
    pop         rbp
%else
  %if LIBVPX_YASM_WIN64
    RESTORE_XMM
  %endif
%endif
    ret
%endmacro

SECTION .text

;void vp8_short_fdct4x4_sse2(short *input, short *output, int pitch)
globalsym(vp8_short_fdct4x4_sse2)
sym(vp8_short_fdct4x4_sse2):

    STACK_FRAME_CREATE

    movq        xmm0, MMWORD PTR[input        ] ;03 02 01 00
    movq        xmm2, MMWORD PTR[input+  pitch] ;13 12 11 10
    lea         input,          [input+2*pitch]
    movq        xmm1, MMWORD PTR[input        ] ;23 22 21 20
    movq        xmm3, MMWORD PTR[input+  pitch] ;33 32 31 30

    punpcklqdq  xmm0, xmm2                      ;13 12 11 10 03 02 01 00
    punpcklqdq  xmm1, xmm3                      ;33 32 31 30 23 22 21 20

    movdqa      xmm2, xmm0
    punpckldq   xmm0, xmm1                      ;23 22 03 02 21 20 01 00
    punpckhdq   xmm2, xmm1                      ;33 32 13 12 31 30 11 10
    movdqa      xmm1, xmm0
    punpckldq   xmm0, xmm2                      ;31 21 30 20 11 10 01 00
    pshufhw     xmm1, xmm1, 0b1h                ;22 23 02 03 xx xx xx xx
    pshufhw     xmm2, xmm2, 0b1h                ;32 33 12 13 xx xx xx xx

    punpckhdq   xmm1, xmm2                      ;32 33 22 23 12 13 02 03
    movdqa      xmm3, xmm0
    paddw       xmm0, xmm1                      ;b1 a1 b1 a1 b1 a1 b1 a1
    psubw       xmm3, xmm1                      ;c1 d1 c1 d1 c1 d1 c1 d1
    psllw       xmm0, 3                         ;b1 <<= 3 a1 <<= 3
    psllw       xmm3, 3                         ;c1 <<= 3 d1 <<= 3

    movdqa      xmm1, xmm0
    pmaddwd     xmm0, XMMWORD PTR[GLOBAL(_mult_add)]    ;a1 + b1
    pmaddwd     xmm1, XMMWORD PTR[GLOBAL(_mult_sub)]    ;a1 - b1
    movdqa      xmm4, xmm3
    pmaddwd     xmm3, XMMWORD PTR[GLOBAL(_5352_2217)]   ;c1*2217 + d1*5352
    pmaddwd     xmm4, XMMWORD PTR[GLOBAL(_2217_neg5352)];d1*2217 - c1*5352

    paddd       xmm3, XMMWORD PTR[GLOBAL(_14500)]
    paddd       xmm4, XMMWORD PTR[GLOBAL(_7500)]
    psrad       xmm3, 12            ;(c1 * 2217 + d1 * 5352 +  14500)>>12
    psrad       xmm4, 12            ;(d1 * 2217 - c1 * 5352 +   7500)>>12

    packssdw    xmm0, xmm1                      ;op[2] op[0]
    packssdw    xmm3, xmm4                      ;op[3] op[1]
    ; 23 22 21 20 03 02 01 00
    ;
    ; 33 32 31 30 13 12 11 10
    ;
    movdqa      xmm2, xmm0
    punpcklqdq  xmm0, xmm3                      ;13 12 11 10 03 02 01 00
    punpckhqdq  xmm2, xmm3                      ;23 22 21 20 33 32 31 30

    movdqa      xmm3, xmm0
    punpcklwd   xmm0, xmm2                      ;32 30 22 20 12 10 02 00
    punpckhwd   xmm3, xmm2                      ;33 31 23 21 13 11 03 01
    movdqa      xmm2, xmm0
    punpcklwd   xmm0, xmm3                      ;13 12 11 10 03 02 01 00
    punpckhwd   xmm2, xmm3                      ;33 32 31 30 23 22 21 20

    movdqa      xmm5, XMMWORD PTR[GLOBAL(_7)]
    pshufd      xmm2, xmm2, 04eh
    movdqa      xmm3, xmm0
    paddw       xmm0, xmm2                      ;b1 b1 b1 b1 a1 a1 a1 a1
    psubw       xmm3, xmm2                      ;c1 c1 c1 c1 d1 d1 d1 d1

    pshufd      xmm0, xmm0, 0d8h                ;b1 b1 a1 a1 b1 b1 a1 a1
    movdqa      xmm2, xmm3                      ;save d1 for compare
    pshufd      xmm3, xmm3, 0d8h                ;c1 c1 d1 d1 c1 c1 d1 d1
    pshuflw     xmm0, xmm0, 0d8h                ;b1 b1 a1 a1 b1 a1 b1 a1
    pshuflw     xmm3, xmm3, 0d8h                ;c1 c1 d1 d1 c1 d1 c1 d1
    pshufhw     xmm0, xmm0, 0d8h                ;b1 a1 b1 a1 b1 a1 b1 a1
    pshufhw     xmm3, xmm3, 0d8h                ;c1 d1 c1 d1 c1 d1 c1 d1
    movdqa      xmm1, xmm0
    pmaddwd     xmm0, XMMWORD PTR[GLOBAL(_mult_add)] ;a1 + b1
    pmaddwd     xmm1, XMMWORD PTR[GLOBAL(_mult_sub)] ;a1 - b1

    pxor        xmm4, xmm4                      ;zero out for compare
    paddd       xmm0, xmm5
    paddd       xmm1, xmm5
    pcmpeqw     xmm2, xmm4
    psrad       xmm0, 4                         ;(a1 + b1 + 7)>>4
    psrad       xmm1, 4                         ;(a1 - b1 + 7)>>4
    pandn       xmm2, XMMWORD PTR[GLOBAL(_cmp_mask)] ;clear upper,
                                                     ;and keep bit 0 of lower

    movdqa      xmm4, xmm3
    pmaddwd     xmm3, XMMWORD PTR[GLOBAL(_5352_2217)]    ;c1*2217 + d1*5352
    pmaddwd     xmm4, XMMWORD PTR[GLOBAL(_2217_neg5352)] ;d1*2217 - c1*5352
    paddd       xmm3, XMMWORD PTR[GLOBAL(_12000)]
    paddd       xmm4, XMMWORD PTR[GLOBAL(_51000)]
    packssdw    xmm0, xmm1                      ;op[8] op[0]
    psrad       xmm3, 16                ;(c1 * 2217 + d1 * 5352 +  12000)>>16
    psrad       xmm4, 16                ;(d1 * 2217 - c1 * 5352 +  51000)>>16

    packssdw    xmm3, xmm4                      ;op[12] op[4]
    movdqa      xmm1, xmm0
    paddw       xmm3, xmm2                      ;op[4] += (d1!=0)
    punpcklqdq  xmm0, xmm3                      ;op[4] op[0]
    punpckhqdq  xmm1, xmm3                      ;op[12] op[8]

    movdqa      XMMWORD PTR[output +  0], xmm0
    movdqa      XMMWORD PTR[output + 16], xmm1

    STACK_FRAME_DESTROY

;void vp8_short_fdct8x4_sse2(short *input, short *output, int pitch)
globalsym(vp8_short_fdct8x4_sse2)
sym(vp8_short_fdct8x4_sse2):

    STACK_FRAME_CREATE

        ; read the input data
        movdqa      xmm0,       [input        ]
        movdqa      xmm2,       [input+  pitch]
        lea         input,      [input+2*pitch]
        movdqa      xmm4,       [input        ]
        movdqa      xmm3,       [input+  pitch]

        ; transpose for the first stage
        movdqa      xmm1,       xmm0        ; 00 01 02 03 04 05 06 07
        movdqa      xmm5,       xmm4        ; 20 21 22 23 24 25 26 27

        punpcklwd   xmm0,       xmm2        ; 00 10 01 11 02 12 03 13
        punpckhwd   xmm1,       xmm2        ; 04 14 05 15 06 16 07 17

        punpcklwd   xmm4,       xmm3        ; 20 30 21 31 22 32 23 33
        punpckhwd   xmm5,       xmm3        ; 24 34 25 35 26 36 27 37

        movdqa      xmm2,       xmm0        ; 00 10 01 11 02 12 03 13
        punpckldq   xmm0,       xmm4        ; 00 10 20 30 01 11 21 31

        punpckhdq   xmm2,       xmm4        ; 02 12 22 32 03 13 23 33

        movdqa      xmm4,       xmm1        ; 04 14 05 15 06 16 07 17
        punpckldq   xmm4,       xmm5        ; 04 14 24 34 05 15 25 35

        punpckhdq   xmm1,       xmm5        ; 06 16 26 36 07 17 27 37
        movdqa      xmm3,       xmm2        ; 02 12 22 32 03 13 23 33

        punpckhqdq  xmm3,       xmm1        ; 03 13 23 33 07 17 27 37
        punpcklqdq  xmm2,       xmm1        ; 02 12 22 32 06 16 26 36

        movdqa      xmm1,       xmm0        ; 00 10 20 30 01 11 21 31
        punpcklqdq  xmm0,       xmm4        ; 00 10 20 30 04 14 24 34

        punpckhqdq  xmm1,       xmm4        ; 01 11 21 32 05 15 25 35

        ; xmm0 0
        ; xmm1 1
        ; xmm2 2
        ; xmm3 3

        ; first stage
        movdqa      xmm5,       xmm0
        movdqa      xmm4,       xmm1

        paddw       xmm0,       xmm3        ; a1 = 0 + 3
        paddw       xmm1,       xmm2        ; b1 = 1 + 2

        psubw       xmm4,       xmm2        ; c1 = 1 - 2
        psubw       xmm5,       xmm3        ; d1 = 0 - 3

        psllw       xmm5,        3
        psllw       xmm4,        3

        psllw       xmm0,        3
        psllw       xmm1,        3

        ; output 0 and 2
        movdqa      xmm2,       xmm0        ; a1

        paddw       xmm0,       xmm1        ; op[0] = a1 + b1
        psubw       xmm2,       xmm1        ; op[2] = a1 - b1

        ; output 1 and 3
        ; interleave c1, d1
        movdqa      xmm1,       xmm5        ; d1
        punpcklwd   xmm1,       xmm4        ; c1 d1
        punpckhwd   xmm5,       xmm4        ; c1 d1

        movdqa      xmm3,       xmm1
        movdqa      xmm4,       xmm5

        pmaddwd     xmm1,       XMMWORD PTR[GLOBAL (_5352_2217)]    ; c1*2217 + d1*5352
        pmaddwd     xmm4,       XMMWORD PTR[GLOBAL (_5352_2217)]    ; c1*2217 + d1*5352

        pmaddwd     xmm3,       XMMWORD PTR[GLOBAL(_2217_neg5352)]  ; d1*2217 - c1*5352
        pmaddwd     xmm5,       XMMWORD PTR[GLOBAL(_2217_neg5352)]  ; d1*2217 - c1*5352

        paddd       xmm1,       XMMWORD PTR[GLOBAL(_14500)]
        paddd       xmm4,       XMMWORD PTR[GLOBAL(_14500)]
        paddd       xmm3,       XMMWORD PTR[GLOBAL(_7500)]
        paddd       xmm5,       XMMWORD PTR[GLOBAL(_7500)]

        psrad       xmm1,       12          ; (c1 * 2217 + d1 * 5352 +  14500)>>12
        psrad       xmm4,       12          ; (c1 * 2217 + d1 * 5352 +  14500)>>12
        psrad       xmm3,       12          ; (d1 * 2217 - c1 * 5352 +   7500)>>12
        psrad       xmm5,       12          ; (d1 * 2217 - c1 * 5352 +   7500)>>12

        packssdw    xmm1,       xmm4        ; op[1]
        packssdw    xmm3,       xmm5        ; op[3]

        ; done with vertical
        ; transpose for the second stage
        movdqa      xmm4,       xmm0         ; 00 10 20 30 04 14 24 34
        movdqa      xmm5,       xmm2         ; 02 12 22 32 06 16 26 36

        punpcklwd   xmm0,       xmm1         ; 00 01 10 11 20 21 30 31
        punpckhwd   xmm4,       xmm1         ; 04 05 14 15 24 25 34 35

        punpcklwd   xmm2,       xmm3         ; 02 03 12 13 22 23 32 33
        punpckhwd   xmm5,       xmm3         ; 06 07 16 17 26 27 36 37

        movdqa      xmm1,       xmm0         ; 00 01 10 11 20 21 30 31
        punpckldq   xmm0,       xmm2         ; 00 01 02 03 10 11 12 13

        punpckhdq   xmm1,       xmm2         ; 20 21 22 23 30 31 32 33

        movdqa      xmm2,       xmm4         ; 04 05 14 15 24 25 34 35
        punpckldq   xmm2,       xmm5         ; 04 05 06 07 14 15 16 17

        punpckhdq   xmm4,       xmm5         ; 24 25 26 27 34 35 36 37
        movdqa      xmm3,       xmm1         ; 20 21 22 23 30 31 32 33

        punpckhqdq  xmm3,       xmm4         ; 30 31 32 33 34 35 36 37
        punpcklqdq  xmm1,       xmm4         ; 20 21 22 23 24 25 26 27

        movdqa      xmm4,       xmm0         ; 00 01 02 03 10 11 12 13
        punpcklqdq  xmm0,       xmm2         ; 00 01 02 03 04 05 06 07

        punpckhqdq  xmm4,       xmm2         ; 10 11 12 13 14 15 16 17

        ; xmm0 0
        ; xmm1 4
        ; xmm2 1
        ; xmm3 3

        movdqa      xmm5,       xmm0
        movdqa      xmm2,       xmm1

        paddw       xmm0,       xmm3        ; a1 = 0 + 3
        paddw       xmm1,       xmm4        ; b1 = 1 + 2

        psubw       xmm4,       xmm2        ; c1 = 1 - 2
        psubw       xmm5,       xmm3        ; d1 = 0 - 3

        pxor        xmm6,       xmm6        ; zero out for compare

        pcmpeqw     xmm6,       xmm5        ; d1 != 0

        pandn       xmm6,       XMMWORD PTR[GLOBAL(_cmp_mask8x4)]   ; clear upper,
                                                                    ; and keep bit 0 of lower

        ; output 0 and 2
        movdqa      xmm2,       xmm0        ; a1

        paddw       xmm0,       xmm1        ; a1 + b1
        psubw       xmm2,       xmm1        ; a1 - b1

        paddw       xmm0,       XMMWORD PTR[GLOBAL(_7w)]
        paddw       xmm2,       XMMWORD PTR[GLOBAL(_7w)]

        psraw       xmm0,       4           ; op[0] = (a1 + b1 + 7)>>4
        psraw       xmm2,       4           ; op[8] = (a1 - b1 + 7)>>4

        ; output 1 and 3
        ; interleave c1, d1
        movdqa      xmm1,       xmm5        ; d1
        punpcklwd   xmm1,       xmm4        ; c1 d1
        punpckhwd   xmm5,       xmm4        ; c1 d1

        movdqa      xmm3,       xmm1
        movdqa      xmm4,       xmm5

        pmaddwd     xmm1,       XMMWORD PTR[GLOBAL (_5352_2217)]    ; c1*2217 + d1*5352
        pmaddwd     xmm4,       XMMWORD PTR[GLOBAL (_5352_2217)]    ; c1*2217 + d1*5352

        pmaddwd     xmm3,       XMMWORD PTR[GLOBAL(_2217_neg5352)]  ; d1*2217 - c1*5352
        pmaddwd     xmm5,       XMMWORD PTR[GLOBAL(_2217_neg5352)]  ; d1*2217 - c1*5352

        paddd       xmm1,       XMMWORD PTR[GLOBAL(_12000)]
        paddd       xmm4,       XMMWORD PTR[GLOBAL(_12000)]
        paddd       xmm3,       XMMWORD PTR[GLOBAL(_51000)]
        paddd       xmm5,       XMMWORD PTR[GLOBAL(_51000)]

        psrad       xmm1,       16          ; (c1 * 2217 + d1 * 5352 +  14500)>>16
        psrad       xmm4,       16          ; (c1 * 2217 + d1 * 5352 +  14500)>>16
        psrad       xmm3,       16          ; (d1 * 2217 - c1 * 5352 +   7500)>>16
        psrad       xmm5,       16          ; (d1 * 2217 - c1 * 5352 +   7500)>>16

        packssdw    xmm1,       xmm4        ; op[4]
        packssdw    xmm3,       xmm5        ; op[12]

        paddw       xmm1,       xmm6        ; op[4] += (d1!=0)

        movdqa      xmm4,       xmm0
        movdqa      xmm5,       xmm2

        punpcklqdq  xmm0,       xmm1
        punpckhqdq  xmm4,       xmm1

        punpcklqdq  xmm2,       xmm3
        punpckhqdq  xmm5,       xmm3

        movdqa      XMMWORD PTR[output + 0 ],  xmm0
        movdqa      XMMWORD PTR[output + 16],  xmm2
        movdqa      XMMWORD PTR[output + 32],  xmm4
        movdqa      XMMWORD PTR[output + 48],  xmm5

    STACK_FRAME_DESTROY

SECTION_RODATA
align 16
_5352_2217:
    dw 5352
    dw 2217
    dw 5352
    dw 2217
    dw 5352
    dw 2217
    dw 5352
    dw 2217
align 16
_2217_neg5352:
    dw 2217
    dw -5352
    dw 2217
    dw -5352
    dw 2217
    dw -5352
    dw 2217
    dw -5352
align 16
_mult_add:
    times 8 dw 1
align 16
_cmp_mask:
    times 4 dw 1
    times 4 dw 0
align 16
_cmp_mask8x4:
    times 8 dw 1
align 16
_mult_sub:
    dw 1
    dw -1
    dw 1
    dw -1
    dw 1
    dw -1
    dw 1
    dw -1
align 16
_7:
    times 4 dd 7
align 16
_7w:
    times 8 dw 7
align 16
_14500:
    times 4 dd 14500
align 16
_7500:
    times 4 dd 7500
align 16
_12000:
    times 4 dd 12000
align 16
_51000:
    times 4 dd 51000
