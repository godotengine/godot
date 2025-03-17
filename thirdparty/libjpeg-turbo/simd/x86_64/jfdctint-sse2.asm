;
; jfdctint.asm - accurate integer FDCT (64-bit SSE2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2020, 2024, D. R. Commander.
; Copyright (C) 2023, Aliaksiej Kandracienka.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.
;
; This file contains a slower but more accurate integer implementation of the
; forward DCT (Discrete Cosine Transform). The following code is based
; directly on the IJG's original jfdctint.c; see the jfdctint.c for
; more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%define CONST_BITS  13
%define PASS1_BITS  2

%define DESCALE_P1  (CONST_BITS - PASS1_BITS)
%define DESCALE_P2  (CONST_BITS + PASS1_BITS)

%if CONST_BITS == 13
F_0_298 equ  2446  ; FIX(0.298631336)
F_0_390 equ  3196  ; FIX(0.390180644)
F_0_541 equ  4433  ; FIX(0.541196100)
F_0_765 equ  6270  ; FIX(0.765366865)
F_0_899 equ  7373  ; FIX(0.899976223)
F_1_175 equ  9633  ; FIX(1.175875602)
F_1_501 equ 12299  ; FIX(1.501321110)
F_1_847 equ 15137  ; FIX(1.847759065)
F_1_961 equ 16069  ; FIX(1.961570560)
F_2_053 equ 16819  ; FIX(2.053119869)
F_2_562 equ 20995  ; FIX(2.562915447)
F_3_072 equ 25172  ; FIX(3.072711026)
%else
; NASM cannot do compile-time arithmetic on floating-point constants.
%define DESCALE(x, n)  (((x) + (1 << ((n) - 1))) >> (n))
F_0_298 equ DESCALE( 320652955, 30 - CONST_BITS)  ; FIX(0.298631336)
F_0_390 equ DESCALE( 418953276, 30 - CONST_BITS)  ; FIX(0.390180644)
F_0_541 equ DESCALE( 581104887, 30 - CONST_BITS)  ; FIX(0.541196100)
F_0_765 equ DESCALE( 821806413, 30 - CONST_BITS)  ; FIX(0.765366865)
F_0_899 equ DESCALE( 966342111, 30 - CONST_BITS)  ; FIX(0.899976223)
F_1_175 equ DESCALE(1262586813, 30 - CONST_BITS)  ; FIX(1.175875602)
F_1_501 equ DESCALE(1612031267, 30 - CONST_BITS)  ; FIX(1.501321110)
F_1_847 equ DESCALE(1984016188, 30 - CONST_BITS)  ; FIX(1.847759065)
F_1_961 equ DESCALE(2106220350, 30 - CONST_BITS)  ; FIX(1.961570560)
F_2_053 equ DESCALE(2204520673, 30 - CONST_BITS)  ; FIX(2.053119869)
F_2_562 equ DESCALE(2751909506, 30 - CONST_BITS)  ; FIX(2.562915447)
F_3_072 equ DESCALE(3299298341, 30 - CONST_BITS)  ; FIX(3.072711026)
%endif

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_fdct_islow_sse2)

EXTN(jconst_fdct_islow_sse2):

PW_F130_F054   times 4 dw  (F_0_541 + F_0_765),  F_0_541
PW_F054_MF130  times 4 dw  F_0_541, (F_0_541 - F_1_847)
PW_MF078_F117  times 4 dw  (F_1_175 - F_1_961),  F_1_175
PW_F117_F078   times 4 dw  F_1_175, (F_1_175 - F_0_390)
PW_MF060_MF089 times 4 dw  (F_0_298 - F_0_899), -F_0_899
PW_MF089_F060  times 4 dw -F_0_899, (F_1_501 - F_0_899)
PW_MF050_MF256 times 4 dw  (F_2_053 - F_2_562), -F_2_562
PW_MF256_F050  times 4 dw -F_2_562, (F_3_072 - F_2_562)
PD_DESCALE_P1  times 4 dd  1 << (DESCALE_P1 - 1)
PD_DESCALE_P2  times 4 dd  1 << (DESCALE_P2 - 1)
PW_DESCALE_P2X times 8 dw  1 << (PASS1_BITS - 1)

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64
;
; Perform the forward DCT on one block of samples.
;
; GLOBAL(void)
; jsimd_fdct_islow_sse2(DCTELEM *data)
;

; r10 = DCTELEM *data

%define wk(i)   r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD  ; xmmword wk[WK_NUM]
%define WK_NUM  6

    align       32
    GLOBAL_FUNCTION(jsimd_fdct_islow_sse2)

EXTN(jsimd_fdct_islow_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    push        r15
    and         rsp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    ; Allocate stack space for wk array.  r15 is used to access it.
    mov         r15, rsp
    sub         rsp, byte (SIZEOF_XMMWORD * WK_NUM)
    COLLECT_ARGS 1

    ; ---- Pass 1: process rows.

    mov         rdx, r10                ; (DCTELEM *)

    movdqa      xmm0, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_DCTELEM)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_DCTELEM)]
    movdqa      xmm2, XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_DCTELEM)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_DCTELEM)]

    ; xmm0=(00 01 02 03 04 05 06 07), xmm2=(20 21 22 23 24 25 26 27)
    ; xmm1=(10 11 12 13 14 15 16 17), xmm3=(30 31 32 33 34 35 36 37)

    movdqa      xmm4, xmm0              ; transpose coefficients(phase 1)
    punpcklwd   xmm0, xmm1              ; xmm0=(00 10 01 11 02 12 03 13)
    punpckhwd   xmm4, xmm1              ; xmm4=(04 14 05 15 06 16 07 17)
    movdqa      xmm5, xmm2              ; transpose coefficients(phase 1)
    punpcklwd   xmm2, xmm3              ; xmm2=(20 30 21 31 22 32 23 33)
    punpckhwd   xmm5, xmm3              ; xmm5=(24 34 25 35 26 36 27 37)

    movdqa      xmm6, XMMWORD [XMMBLOCK(4,0,rdx,SIZEOF_DCTELEM)]
    movdqa      xmm7, XMMWORD [XMMBLOCK(5,0,rdx,SIZEOF_DCTELEM)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(6,0,rdx,SIZEOF_DCTELEM)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(7,0,rdx,SIZEOF_DCTELEM)]

    ; xmm6=( 4 12 20 28 36 44 52 60), xmm1=( 6 14 22 30 38 46 54 62)
    ; xmm7=( 5 13 21 29 37 45 53 61), xmm3=( 7 15 23 31 39 47 55 63)

    movdqa      XMMWORD [wk(0)], xmm2   ; wk(0)=(20 30 21 31 22 32 23 33)
    movdqa      XMMWORD [wk(1)], xmm5   ; wk(1)=(24 34 25 35 26 36 27 37)

    movdqa      xmm2, xmm6              ; transpose coefficients(phase 1)
    punpcklwd   xmm6, xmm7              ; xmm6=(40 50 41 51 42 52 43 53)
    punpckhwd   xmm2, xmm7              ; xmm2=(44 54 45 55 46 56 47 57)
    movdqa      xmm5, xmm1              ; transpose coefficients(phase 1)
    punpcklwd   xmm1, xmm3              ; xmm1=(60 70 61 71 62 72 63 73)
    punpckhwd   xmm5, xmm3              ; xmm5=(64 74 65 75 66 76 67 77)

    movdqa      xmm7, xmm6              ; transpose coefficients(phase 2)
    punpckldq   xmm6, xmm1              ; xmm6=(40 50 60 70 41 51 61 71)
    punpckhdq   xmm7, xmm1              ; xmm7=(42 52 62 72 43 53 63 73)
    movdqa      xmm3, xmm2              ; transpose coefficients(phase 2)
    punpckldq   xmm2, xmm5              ; xmm2=(44 54 64 74 45 55 65 75)
    punpckhdq   xmm3, xmm5              ; xmm3=(46 56 66 76 47 57 67 77)

    movdqa      xmm1, XMMWORD [wk(0)]   ; xmm1=(20 30 21 31 22 32 23 33)
    movdqa      xmm5, XMMWORD [wk(1)]   ; xmm5=(24 34 25 35 26 36 27 37)
    movdqa      XMMWORD [wk(2)], xmm7   ; wk(2)=(42 52 62 72 43 53 63 73)
    movdqa      XMMWORD [wk(3)], xmm2   ; wk(3)=(44 54 64 74 45 55 65 75)

    movdqa      xmm7, xmm0              ; transpose coefficients(phase 2)
    punpckldq   xmm0, xmm1              ; xmm0=(00 10 20 30 01 11 21 31)
    punpckhdq   xmm7, xmm1              ; xmm7=(02 12 22 32 03 13 23 33)
    movdqa      xmm2, xmm4              ; transpose coefficients(phase 2)
    punpckldq   xmm4, xmm5              ; xmm4=(04 14 24 34 05 15 25 35)
    punpckhdq   xmm2, xmm5              ; xmm2=(06 16 26 36 07 17 27 37)

    movdqa      xmm1, xmm0              ; transpose coefficients(phase 3)
    punpcklqdq  xmm0, xmm6              ; xmm0=(00 10 20 30 40 50 60 70)=data0
    punpckhqdq  xmm1, xmm6              ; xmm1=(01 11 21 31 41 51 61 71)=data1
    movdqa      xmm5, xmm2              ; transpose coefficients(phase 3)
    punpcklqdq  xmm2, xmm3              ; xmm2=(06 16 26 36 46 56 66 76)=data6
    punpckhqdq  xmm5, xmm3              ; xmm5=(07 17 27 37 47 57 67 77)=data7

    movdqa      xmm6, xmm1
    movdqa      xmm3, xmm0
    psubw       xmm1, xmm2              ; xmm1=data1-data6=tmp6
    psubw       xmm0, xmm5              ; xmm0=data0-data7=tmp7
    paddw       xmm6, xmm2              ; xmm6=data1+data6=tmp1
    paddw       xmm3, xmm5              ; xmm3=data0+data7=tmp0

    movdqa      xmm2, XMMWORD [wk(2)]   ; xmm2=(42 52 62 72 43 53 63 73)
    movdqa      xmm5, XMMWORD [wk(3)]   ; xmm5=(44 54 64 74 45 55 65 75)
    movdqa      XMMWORD [wk(0)], xmm1   ; wk(0)=tmp6
    movdqa      XMMWORD [wk(1)], xmm0   ; wk(1)=tmp7

    movdqa      xmm1, xmm7              ; transpose coefficients(phase 3)
    punpcklqdq  xmm7, xmm2              ; xmm7=(02 12 22 32 42 52 62 72)=data2
    punpckhqdq  xmm1, xmm2              ; xmm1=(03 13 23 33 43 53 63 73)=data3
    movdqa      xmm0, xmm4              ; transpose coefficients(phase 3)
    punpcklqdq  xmm4, xmm5              ; xmm4=(04 14 24 34 44 54 64 74)=data4
    punpckhqdq  xmm0, xmm5              ; xmm0=(05 15 25 35 45 55 65 75)=data5

    movdqa      xmm2, xmm1
    movdqa      xmm5, xmm7
    paddw       xmm1, xmm4              ; xmm1=data3+data4=tmp3
    paddw       xmm7, xmm0              ; xmm7=data2+data5=tmp2
    psubw       xmm2, xmm4              ; xmm2=data3-data4=tmp4
    psubw       xmm5, xmm0              ; xmm5=data2-data5=tmp5

    ; -- Even part

    movdqa      xmm4, xmm3
    movdqa      xmm0, xmm6
    paddw       xmm3, xmm1              ; xmm3=tmp10
    paddw       xmm6, xmm7              ; xmm6=tmp11
    psubw       xmm4, xmm1              ; xmm4=tmp13
    psubw       xmm0, xmm7              ; xmm0=tmp12

    movdqa      xmm1, xmm3
    paddw       xmm3, xmm6              ; xmm3=tmp10+tmp11
    psubw       xmm1, xmm6              ; xmm1=tmp10-tmp11

    psllw       xmm3, PASS1_BITS        ; xmm3=data0
    psllw       xmm1, PASS1_BITS        ; xmm1=data4

    movdqa      XMMWORD [wk(2)], xmm3   ; wk(2)=data0
    movdqa      XMMWORD [wk(3)], xmm1   ; wk(3)=data4

    ; (Original)
    ; z1 = (tmp12 + tmp13) * 0.541196100;
    ; data2 = z1 + tmp13 * 0.765366865;
    ; data6 = z1 + tmp12 * -1.847759065;
    ;
    ; (This implementation)
    ; data2 = tmp13 * (0.541196100 + 0.765366865) + tmp12 * 0.541196100;
    ; data6 = tmp13 * 0.541196100 + tmp12 * (0.541196100 - 1.847759065);

    movdqa      xmm7, xmm4              ; xmm4=tmp13
    movdqa      xmm6, xmm4
    punpcklwd   xmm7, xmm0              ; xmm0=tmp12
    punpckhwd   xmm6, xmm0
    movdqa      xmm4, xmm7
    movdqa      xmm0, xmm6
    pmaddwd     xmm7, [rel PW_F130_F054]   ; xmm7=data2L
    pmaddwd     xmm6, [rel PW_F130_F054]   ; xmm6=data2H
    pmaddwd     xmm4, [rel PW_F054_MF130]  ; xmm4=data6L
    pmaddwd     xmm0, [rel PW_F054_MF130]  ; xmm0=data6H

    paddd       xmm7, [rel PD_DESCALE_P1]
    paddd       xmm6, [rel PD_DESCALE_P1]
    psrad       xmm7, DESCALE_P1
    psrad       xmm6, DESCALE_P1
    paddd       xmm4, [rel PD_DESCALE_P1]
    paddd       xmm0, [rel PD_DESCALE_P1]
    psrad       xmm4, DESCALE_P1
    psrad       xmm0, DESCALE_P1

    packssdw    xmm7, xmm6              ; xmm7=data2
    packssdw    xmm4, xmm0              ; xmm4=data6

    movdqa      XMMWORD [wk(4)], xmm7   ; wk(4)=data2
    movdqa      XMMWORD [wk(5)], xmm4   ; wk(5)=data6

    ; -- Odd part

    movdqa      xmm3, XMMWORD [wk(0)]   ; xmm3=tmp6
    movdqa      xmm1, XMMWORD [wk(1)]   ; xmm1=tmp7

    movdqa      xmm6, xmm2              ; xmm2=tmp4
    movdqa      xmm0, xmm5              ; xmm5=tmp5
    paddw       xmm6, xmm3              ; xmm6=z3
    paddw       xmm0, xmm1              ; xmm0=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movdqa      xmm7, xmm6
    movdqa      xmm4, xmm6
    punpcklwd   xmm7, xmm0
    punpckhwd   xmm4, xmm0
    movdqa      xmm6, xmm7
    movdqa      xmm0, xmm4
    pmaddwd     xmm7, [rel PW_MF078_F117]  ; xmm7=z3L
    pmaddwd     xmm4, [rel PW_MF078_F117]  ; xmm4=z3H
    pmaddwd     xmm6, [rel PW_F117_F078]   ; xmm6=z4L
    pmaddwd     xmm0, [rel PW_F117_F078]   ; xmm0=z4H

    movdqa      XMMWORD [wk(0)], xmm7   ; wk(0)=z3L
    movdqa      XMMWORD [wk(1)], xmm4   ; wk(1)=z3H

    ; (Original)
    ; z1 = tmp4 + tmp7;  z2 = tmp5 + tmp6;
    ; tmp4 = tmp4 * 0.298631336;  tmp5 = tmp5 * 2.053119869;
    ; tmp6 = tmp6 * 3.072711026;  tmp7 = tmp7 * 1.501321110;
    ; z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
    ; data7 = tmp4 + z1 + z3;  data5 = tmp5 + z2 + z4;
    ; data3 = tmp6 + z2 + z3;  data1 = tmp7 + z1 + z4;
    ;
    ; (This implementation)
    ; tmp4 = tmp4 * (0.298631336 - 0.899976223) + tmp7 * -0.899976223;
    ; tmp5 = tmp5 * (2.053119869 - 2.562915447) + tmp6 * -2.562915447;
    ; tmp6 = tmp5 * -2.562915447 + tmp6 * (3.072711026 - 2.562915447);
    ; tmp7 = tmp4 * -0.899976223 + tmp7 * (1.501321110 - 0.899976223);
    ; data7 = tmp4 + z3;  data5 = tmp5 + z4;
    ; data3 = tmp6 + z3;  data1 = tmp7 + z4;

    movdqa      xmm7, xmm2
    movdqa      xmm4, xmm2
    punpcklwd   xmm7, xmm1
    punpckhwd   xmm4, xmm1
    movdqa      xmm2, xmm7
    movdqa      xmm1, xmm4
    pmaddwd     xmm7, [rel PW_MF060_MF089]  ; xmm7=tmp4L
    pmaddwd     xmm4, [rel PW_MF060_MF089]  ; xmm4=tmp4H
    pmaddwd     xmm2, [rel PW_MF089_F060]   ; xmm2=tmp7L
    pmaddwd     xmm1, [rel PW_MF089_F060]   ; xmm1=tmp7H

    paddd       xmm7, XMMWORD [wk(0)]   ; xmm7=data7L
    paddd       xmm4, XMMWORD [wk(1)]   ; xmm4=data7H
    paddd       xmm2, xmm6              ; xmm2=data1L
    paddd       xmm1, xmm0              ; xmm1=data1H

    paddd       xmm7, [rel PD_DESCALE_P1]
    paddd       xmm4, [rel PD_DESCALE_P1]
    psrad       xmm7, DESCALE_P1
    psrad       xmm4, DESCALE_P1
    paddd       xmm2, [rel PD_DESCALE_P1]
    paddd       xmm1, [rel PD_DESCALE_P1]
    psrad       xmm2, DESCALE_P1
    psrad       xmm1, DESCALE_P1

    packssdw    xmm7, xmm4              ; xmm7=data7
    packssdw    xmm2, xmm1              ; xmm2=data1

    movdqa      xmm4, xmm5
    movdqa      xmm1, xmm5
    punpcklwd   xmm4, xmm3
    punpckhwd   xmm1, xmm3
    movdqa      xmm5, xmm4
    movdqa      xmm3, xmm1
    pmaddwd     xmm4, [rel PW_MF050_MF256]  ; xmm4=tmp5L
    pmaddwd     xmm1, [rel PW_MF050_MF256]  ; xmm1=tmp5H
    pmaddwd     xmm5, [rel PW_MF256_F050]   ; xmm5=tmp6L
    pmaddwd     xmm3, [rel PW_MF256_F050]   ; xmm3=tmp6H

    paddd       xmm4, xmm6              ; xmm4=data5L
    paddd       xmm1, xmm0              ; xmm1=data5H
    paddd       xmm5, XMMWORD [wk(0)]   ; xmm5=data3L
    paddd       xmm3, XMMWORD [wk(1)]   ; xmm3=data3H

    paddd       xmm4, [rel PD_DESCALE_P1]
    paddd       xmm1, [rel PD_DESCALE_P1]
    psrad       xmm4, DESCALE_P1
    psrad       xmm1, DESCALE_P1
    paddd       xmm5, [rel PD_DESCALE_P1]
    paddd       xmm3, [rel PD_DESCALE_P1]
    psrad       xmm5, DESCALE_P1
    psrad       xmm3, DESCALE_P1

    packssdw    xmm4, xmm1              ; xmm4=data5
    packssdw    xmm5, xmm3              ; xmm5=data3

    ; ---- Pass 2: process columns.

    movdqa      xmm6, XMMWORD [wk(2)]   ; xmm6=col0
    movdqa      xmm0, XMMWORD [wk(4)]   ; xmm0=col2

    ; xmm6=(00 10 20 30 40 50 60 70), xmm0=(02 12 22 32 42 52 62 72)
    ; xmm2=(01 11 21 31 41 51 61 71), xmm5=(03 13 23 33 43 53 63 73)

    movdqa      xmm1, xmm6              ; transpose coefficients(phase 1)
    punpcklwd   xmm6, xmm2              ; xmm6=(00 01 10 11 20 21 30 31)
    punpckhwd   xmm1, xmm2              ; xmm1=(40 41 50 51 60 61 70 71)
    movdqa      xmm3, xmm0              ; transpose coefficients(phase 1)
    punpcklwd   xmm0, xmm5              ; xmm0=(02 03 12 13 22 23 32 33)
    punpckhwd   xmm3, xmm5              ; xmm3=(42 43 52 53 62 63 72 73)

    movdqa      xmm2, XMMWORD [wk(3)]   ; xmm2=col4
    movdqa      xmm5, XMMWORD [wk(5)]   ; xmm5=col6

    ; xmm2=(04 14 24 34 44 54 64 74), xmm5=(06 16 26 36 46 56 66 76)
    ; xmm4=(05 15 25 35 45 55 65 75), xmm7=(07 17 27 37 47 57 67 77)

    movdqa      XMMWORD [wk(0)], xmm0   ; wk(0)=(02 03 12 13 22 23 32 33)
    movdqa      XMMWORD [wk(1)], xmm3   ; wk(1)=(42 43 52 53 62 63 72 73)

    movdqa      xmm0, xmm2              ; transpose coefficients(phase 1)
    punpcklwd   xmm2, xmm4              ; xmm2=(04 05 14 15 24 25 34 35)
    punpckhwd   xmm0, xmm4              ; xmm0=(44 45 54 55 64 65 74 75)
    movdqa      xmm3, xmm5              ; transpose coefficients(phase 1)
    punpcklwd   xmm5, xmm7              ; xmm5=(06 07 16 17 26 27 36 37)
    punpckhwd   xmm3, xmm7              ; xmm3=(46 47 56 57 66 67 76 77)

    movdqa      xmm4, xmm2              ; transpose coefficients(phase 2)
    punpckldq   xmm2, xmm5              ; xmm2=(04 05 06 07 14 15 16 17)
    punpckhdq   xmm4, xmm5              ; xmm4=(24 25 26 27 34 35 36 37)
    movdqa      xmm7, xmm0              ; transpose coefficients(phase 2)
    punpckldq   xmm0, xmm3              ; xmm0=(44 45 46 47 54 55 56 57)
    punpckhdq   xmm7, xmm3              ; xmm7=(64 65 66 67 74 75 76 77)

    movdqa      xmm5, XMMWORD [wk(0)]   ; xmm5=(02 03 12 13 22 23 32 33)
    movdqa      xmm3, XMMWORD [wk(1)]   ; xmm3=(42 43 52 53 62 63 72 73)
    movdqa      XMMWORD [wk(2)], xmm4   ; wk(2)=(24 25 26 27 34 35 36 37)
    movdqa      XMMWORD [wk(3)], xmm0   ; wk(3)=(44 45 46 47 54 55 56 57)

    movdqa      xmm4, xmm6              ; transpose coefficients(phase 2)
    punpckldq   xmm6, xmm5              ; xmm6=(00 01 02 03 10 11 12 13)
    punpckhdq   xmm4, xmm5              ; xmm4=(20 21 22 23 30 31 32 33)
    movdqa      xmm0, xmm1              ; transpose coefficients(phase 2)
    punpckldq   xmm1, xmm3              ; xmm1=(40 41 42 43 50 51 52 53)
    punpckhdq   xmm0, xmm3              ; xmm0=(60 61 62 63 70 71 72 73)

    movdqa      xmm5, xmm6              ; transpose coefficients(phase 3)
    punpcklqdq  xmm6, xmm2              ; xmm6=(00 01 02 03 04 05 06 07)=data0
    punpckhqdq  xmm5, xmm2              ; xmm5=(10 11 12 13 14 15 16 17)=data1
    movdqa      xmm3, xmm0              ; transpose coefficients(phase 3)
    punpcklqdq  xmm0, xmm7              ; xmm0=(60 61 62 63 64 65 66 67)=data6
    punpckhqdq  xmm3, xmm7              ; xmm3=(70 71 72 73 74 75 76 77)=data7

    movdqa      xmm2, xmm5
    movdqa      xmm7, xmm6
    psubw       xmm5, xmm0              ; xmm5=data1-data6=tmp6
    psubw       xmm6, xmm3              ; xmm6=data0-data7=tmp7
    paddw       xmm2, xmm0              ; xmm2=data1+data6=tmp1
    paddw       xmm7, xmm3              ; xmm7=data0+data7=tmp0

    movdqa      xmm0, XMMWORD [wk(2)]   ; xmm0=(24 25 26 27 34 35 36 37)
    movdqa      xmm3, XMMWORD [wk(3)]   ; xmm3=(44 45 46 47 54 55 56 57)
    movdqa      XMMWORD [wk(0)], xmm5   ; wk(0)=tmp6
    movdqa      XMMWORD [wk(1)], xmm6   ; wk(1)=tmp7

    movdqa      xmm5, xmm4              ; transpose coefficients(phase 3)
    punpcklqdq  xmm4, xmm0              ; xmm4=(20 21 22 23 24 25 26 27)=data2
    punpckhqdq  xmm5, xmm0              ; xmm5=(30 31 32 33 34 35 36 37)=data3
    movdqa      xmm6, xmm1              ; transpose coefficients(phase 3)
    punpcklqdq  xmm1, xmm3              ; xmm1=(40 41 42 43 44 45 46 47)=data4
    punpckhqdq  xmm6, xmm3              ; xmm6=(50 51 52 53 54 55 56 57)=data5

    movdqa      xmm0, xmm5
    movdqa      xmm3, xmm4
    paddw       xmm5, xmm1              ; xmm5=data3+data4=tmp3
    paddw       xmm4, xmm6              ; xmm4=data2+data5=tmp2
    psubw       xmm0, xmm1              ; xmm0=data3-data4=tmp4
    psubw       xmm3, xmm6              ; xmm3=data2-data5=tmp5

    ; -- Even part

    movdqa      xmm1, xmm7
    movdqa      xmm6, xmm2
    paddw       xmm7, xmm5              ; xmm7=tmp10
    paddw       xmm2, xmm4              ; xmm2=tmp11
    psubw       xmm1, xmm5              ; xmm1=tmp13
    psubw       xmm6, xmm4              ; xmm6=tmp12

    movdqa      xmm5, xmm7
    paddw       xmm7, xmm2              ; xmm7=tmp10+tmp11
    psubw       xmm5, xmm2              ; xmm5=tmp10-tmp11

    paddw       xmm7, [rel PW_DESCALE_P2X]
    paddw       xmm5, [rel PW_DESCALE_P2X]
    psraw       xmm7, PASS1_BITS        ; xmm7=data0
    psraw       xmm5, PASS1_BITS        ; xmm5=data4

    movdqa      XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_DCTELEM)], xmm7
    movdqa      XMMWORD [XMMBLOCK(4,0,rdx,SIZEOF_DCTELEM)], xmm5

    ; (Original)
    ; z1 = (tmp12 + tmp13) * 0.541196100;
    ; data2 = z1 + tmp13 * 0.765366865;
    ; data6 = z1 + tmp12 * -1.847759065;
    ;
    ; (This implementation)
    ; data2 = tmp13 * (0.541196100 + 0.765366865) + tmp12 * 0.541196100;
    ; data6 = tmp13 * 0.541196100 + tmp12 * (0.541196100 - 1.847759065);

    movdqa      xmm4, xmm1              ; xmm1=tmp13
    movdqa      xmm2, xmm1
    punpcklwd   xmm4, xmm6              ; xmm6=tmp12
    punpckhwd   xmm2, xmm6
    movdqa      xmm1, xmm4
    movdqa      xmm6, xmm2
    pmaddwd     xmm4, [rel PW_F130_F054]   ; xmm4=data2L
    pmaddwd     xmm2, [rel PW_F130_F054]   ; xmm2=data2H
    pmaddwd     xmm1, [rel PW_F054_MF130]  ; xmm1=data6L
    pmaddwd     xmm6, [rel PW_F054_MF130]  ; xmm6=data6H

    paddd       xmm4, [rel PD_DESCALE_P2]
    paddd       xmm2, [rel PD_DESCALE_P2]
    psrad       xmm4, DESCALE_P2
    psrad       xmm2, DESCALE_P2
    paddd       xmm1, [rel PD_DESCALE_P2]
    paddd       xmm6, [rel PD_DESCALE_P2]
    psrad       xmm1, DESCALE_P2
    psrad       xmm6, DESCALE_P2

    packssdw    xmm4, xmm2              ; xmm4=data2
    packssdw    xmm1, xmm6              ; xmm1=data6

    movdqa      XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_DCTELEM)], xmm4
    movdqa      XMMWORD [XMMBLOCK(6,0,rdx,SIZEOF_DCTELEM)], xmm1

    ; -- Odd part

    movdqa      xmm7, XMMWORD [wk(0)]   ; xmm7=tmp6
    movdqa      xmm5, XMMWORD [wk(1)]   ; xmm5=tmp7

    movdqa      xmm2, xmm0              ; xmm0=tmp4
    movdqa      xmm6, xmm3              ; xmm3=tmp5
    paddw       xmm2, xmm7              ; xmm2=z3
    paddw       xmm6, xmm5              ; xmm6=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movdqa      xmm4, xmm2
    movdqa      xmm1, xmm2
    punpcklwd   xmm4, xmm6
    punpckhwd   xmm1, xmm6
    movdqa      xmm2, xmm4
    movdqa      xmm6, xmm1
    pmaddwd     xmm4, [rel PW_MF078_F117]  ; xmm4=z3L
    pmaddwd     xmm1, [rel PW_MF078_F117]  ; xmm1=z3H
    pmaddwd     xmm2, [rel PW_F117_F078]   ; xmm2=z4L
    pmaddwd     xmm6, [rel PW_F117_F078]   ; xmm6=z4H

    movdqa      XMMWORD [wk(0)], xmm4   ; wk(0)=z3L
    movdqa      XMMWORD [wk(1)], xmm1   ; wk(1)=z3H

    ; (Original)
    ; z1 = tmp4 + tmp7;  z2 = tmp5 + tmp6;
    ; tmp4 = tmp4 * 0.298631336;  tmp5 = tmp5 * 2.053119869;
    ; tmp6 = tmp6 * 3.072711026;  tmp7 = tmp7 * 1.501321110;
    ; z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
    ; data7 = tmp4 + z1 + z3;  data5 = tmp5 + z2 + z4;
    ; data3 = tmp6 + z2 + z3;  data1 = tmp7 + z1 + z4;
    ;
    ; (This implementation)
    ; tmp4 = tmp4 * (0.298631336 - 0.899976223) + tmp7 * -0.899976223;
    ; tmp5 = tmp5 * (2.053119869 - 2.562915447) + tmp6 * -2.562915447;
    ; tmp6 = tmp5 * -2.562915447 + tmp6 * (3.072711026 - 2.562915447);
    ; tmp7 = tmp4 * -0.899976223 + tmp7 * (1.501321110 - 0.899976223);
    ; data7 = tmp4 + z3;  data5 = tmp5 + z4;
    ; data3 = tmp6 + z3;  data1 = tmp7 + z4;

    movdqa      xmm4, xmm0
    movdqa      xmm1, xmm0
    punpcklwd   xmm4, xmm5
    punpckhwd   xmm1, xmm5
    movdqa      xmm0, xmm4
    movdqa      xmm5, xmm1
    pmaddwd     xmm4, [rel PW_MF060_MF089]  ; xmm4=tmp4L
    pmaddwd     xmm1, [rel PW_MF060_MF089]  ; xmm1=tmp4H
    pmaddwd     xmm0, [rel PW_MF089_F060]   ; xmm0=tmp7L
    pmaddwd     xmm5, [rel PW_MF089_F060]   ; xmm5=tmp7H

    paddd       xmm4,  XMMWORD [wk(0)]  ; xmm4=data7L
    paddd       xmm1,  XMMWORD [wk(1)]  ; xmm1=data7H
    paddd       xmm0, xmm2              ; xmm0=data1L
    paddd       xmm5, xmm6              ; xmm5=data1H

    paddd       xmm4, [rel PD_DESCALE_P2]
    paddd       xmm1, [rel PD_DESCALE_P2]
    psrad       xmm4, DESCALE_P2
    psrad       xmm1, DESCALE_P2
    paddd       xmm0, [rel PD_DESCALE_P2]
    paddd       xmm5, [rel PD_DESCALE_P2]
    psrad       xmm0, DESCALE_P2
    psrad       xmm5, DESCALE_P2

    packssdw    xmm4, xmm1              ; xmm4=data7
    packssdw    xmm0, xmm5              ; xmm0=data1

    movdqa      XMMWORD [XMMBLOCK(7,0,rdx,SIZEOF_DCTELEM)], xmm4
    movdqa      XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_DCTELEM)], xmm0

    movdqa      xmm1, xmm3
    movdqa      xmm5, xmm3
    punpcklwd   xmm1, xmm7
    punpckhwd   xmm5, xmm7
    movdqa      xmm3, xmm1
    movdqa      xmm7, xmm5
    pmaddwd     xmm1, [rel PW_MF050_MF256]  ; xmm1=tmp5L
    pmaddwd     xmm5, [rel PW_MF050_MF256]  ; xmm5=tmp5H
    pmaddwd     xmm3, [rel PW_MF256_F050]   ; xmm3=tmp6L
    pmaddwd     xmm7, [rel PW_MF256_F050]   ; xmm7=tmp6H

    paddd       xmm1, xmm2              ; xmm1=data5L
    paddd       xmm5, xmm6              ; xmm5=data5H
    paddd       xmm3, XMMWORD [wk(0)]   ; xmm3=data3L
    paddd       xmm7, XMMWORD [wk(1)]   ; xmm7=data3H

    paddd       xmm1, [rel PD_DESCALE_P2]
    paddd       xmm5, [rel PD_DESCALE_P2]
    psrad       xmm1, DESCALE_P2
    psrad       xmm5, DESCALE_P2
    paddd       xmm3, [rel PD_DESCALE_P2]
    paddd       xmm7, [rel PD_DESCALE_P2]
    psrad       xmm3, DESCALE_P2
    psrad       xmm7, DESCALE_P2

    packssdw    xmm1, xmm5              ; xmm1=data5
    packssdw    xmm3, xmm7              ; xmm3=data3

    movdqa      XMMWORD [XMMBLOCK(5,0,rdx,SIZEOF_DCTELEM)], xmm1
    movdqa      XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_DCTELEM)], xmm3

    UNCOLLECT_ARGS 1
    lea         rsp, [rbp-8]
    pop         r15
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
