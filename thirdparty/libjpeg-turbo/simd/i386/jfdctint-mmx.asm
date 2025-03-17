;
; jfdctint.asm - accurate integer FDCT (MMX)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2016, 2020, 2024, D. R. Commander.
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
    GLOBAL_DATA(jconst_fdct_islow_mmx)

EXTN(jconst_fdct_islow_mmx):

PW_F130_F054   times 2 dw  (F_0_541 + F_0_765),  F_0_541
PW_F054_MF130  times 2 dw  F_0_541, (F_0_541 - F_1_847)
PW_MF078_F117  times 2 dw  (F_1_175 - F_1_961),  F_1_175
PW_F117_F078   times 2 dw  F_1_175, (F_1_175 - F_0_390)
PW_MF060_MF089 times 2 dw  (F_0_298 - F_0_899), -F_0_899
PW_MF089_F060  times 2 dw -F_0_899, (F_1_501 - F_0_899)
PW_MF050_MF256 times 2 dw  (F_2_053 - F_2_562), -F_2_562
PW_MF256_F050  times 2 dw -F_2_562, (F_3_072 - F_2_562)
PD_DESCALE_P1  times 2 dd  1 << (DESCALE_P1 - 1)
PD_DESCALE_P2  times 2 dd  1 << (DESCALE_P2 - 1)
PW_DESCALE_P2X times 4 dw  1 << (PASS1_BITS - 1)

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform the forward DCT on one block of samples.
;
; GLOBAL(void)
; jsimd_fdct_islow_mmx(DCTELEM *data)
;

%define data(b)       (b) + 8           ; DCTELEM *data

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_MMWORD  ; mmword wk[WK_NUM]
%define WK_NUM        2

    align       32
    GLOBAL_FUNCTION(jsimd_fdct_islow_mmx)

EXTN(jsimd_fdct_islow_mmx):
    push        ebp
    mov         eax, esp                    ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_MMWORD)  ; align to 64 bits
    mov         [esp], eax
    mov         ebp, esp                    ; ebp = aligned ebp
    lea         esp, [wk(0)]
    PUSHPIC     ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
;   push        esi                     ; unused
;   push        edi                     ; unused

    GET_GOT     ebx                     ; get GOT address

    ; ---- Pass 1: process rows.

    mov         edx, POINTER [data(eax)]  ; (DCTELEM *)
    mov         ecx, DCTSIZE/4
    ALIGNX      16, 7
.rowloop:

    movq        mm0, MMWORD [MMBLOCK(2,0,edx,SIZEOF_DCTELEM)]
    movq        mm1, MMWORD [MMBLOCK(3,0,edx,SIZEOF_DCTELEM)]
    movq        mm2, MMWORD [MMBLOCK(2,1,edx,SIZEOF_DCTELEM)]
    movq        mm3, MMWORD [MMBLOCK(3,1,edx,SIZEOF_DCTELEM)]

    ; mm0=(20 21 22 23), mm2=(24 25 26 27)
    ; mm1=(30 31 32 33), mm3=(34 35 36 37)

    movq        mm4, mm0                ; transpose coefficients(phase 1)
    punpcklwd   mm0, mm1                ; mm0=(20 30 21 31)
    punpckhwd   mm4, mm1                ; mm4=(22 32 23 33)
    movq        mm5, mm2                ; transpose coefficients(phase 1)
    punpcklwd   mm2, mm3                ; mm2=(24 34 25 35)
    punpckhwd   mm5, mm3                ; mm5=(26 36 27 37)

    movq        mm6, MMWORD [MMBLOCK(0,0,edx,SIZEOF_DCTELEM)]
    movq        mm7, MMWORD [MMBLOCK(1,0,edx,SIZEOF_DCTELEM)]
    movq        mm1, MMWORD [MMBLOCK(0,1,edx,SIZEOF_DCTELEM)]
    movq        mm3, MMWORD [MMBLOCK(1,1,edx,SIZEOF_DCTELEM)]

    ; mm6=(00 01 02 03), mm1=(04 05 06 07)
    ; mm7=(10 11 12 13), mm3=(14 15 16 17)

    movq        MMWORD [wk(0)], mm4     ; wk(0)=(22 32 23 33)
    movq        MMWORD [wk(1)], mm2     ; wk(1)=(24 34 25 35)

    movq        mm4, mm6                ; transpose coefficients(phase 1)
    punpcklwd   mm6, mm7                ; mm6=(00 10 01 11)
    punpckhwd   mm4, mm7                ; mm4=(02 12 03 13)
    movq        mm2, mm1                ; transpose coefficients(phase 1)
    punpcklwd   mm1, mm3                ; mm1=(04 14 05 15)
    punpckhwd   mm2, mm3                ; mm2=(06 16 07 17)

    movq        mm7, mm6                ; transpose coefficients(phase 2)
    punpckldq   mm6, mm0                ; mm6=(00 10 20 30)=data0
    punpckhdq   mm7, mm0                ; mm7=(01 11 21 31)=data1
    movq        mm3, mm2                ; transpose coefficients(phase 2)
    punpckldq   mm2, mm5                ; mm2=(06 16 26 36)=data6
    punpckhdq   mm3, mm5                ; mm3=(07 17 27 37)=data7

    movq        mm0, mm7
    movq        mm5, mm6
    psubw       mm7, mm2                ; mm7=data1-data6=tmp6
    psubw       mm6, mm3                ; mm6=data0-data7=tmp7
    paddw       mm0, mm2                ; mm0=data1+data6=tmp1
    paddw       mm5, mm3                ; mm5=data0+data7=tmp0

    movq        mm2, MMWORD [wk(0)]     ; mm2=(22 32 23 33)
    movq        mm3, MMWORD [wk(1)]     ; mm3=(24 34 25 35)
    movq        MMWORD [wk(0)], mm7     ; wk(0)=tmp6
    movq        MMWORD [wk(1)], mm6     ; wk(1)=tmp7

    movq        mm7, mm4                ; transpose coefficients(phase 2)
    punpckldq   mm4, mm2                ; mm4=(02 12 22 32)=data2
    punpckhdq   mm7, mm2                ; mm7=(03 13 23 33)=data3
    movq        mm6, mm1                ; transpose coefficients(phase 2)
    punpckldq   mm1, mm3                ; mm1=(04 14 24 34)=data4
    punpckhdq   mm6, mm3                ; mm6=(05 15 25 35)=data5

    movq        mm2, mm7
    movq        mm3, mm4
    paddw       mm7, mm1                ; mm7=data3+data4=tmp3
    paddw       mm4, mm6                ; mm4=data2+data5=tmp2
    psubw       mm2, mm1                ; mm2=data3-data4=tmp4
    psubw       mm3, mm6                ; mm3=data2-data5=tmp5

    ; -- Even part

    movq        mm1, mm5
    movq        mm6, mm0
    paddw       mm5, mm7                ; mm5=tmp10
    paddw       mm0, mm4                ; mm0=tmp11
    psubw       mm1, mm7                ; mm1=tmp13
    psubw       mm6, mm4                ; mm6=tmp12

    movq        mm7, mm5
    paddw       mm5, mm0                ; mm5=tmp10+tmp11
    psubw       mm7, mm0                ; mm7=tmp10-tmp11

    psllw       mm5, PASS1_BITS         ; mm5=data0
    psllw       mm7, PASS1_BITS         ; mm7=data4

    movq        MMWORD [MMBLOCK(0,0,edx,SIZEOF_DCTELEM)], mm5
    movq        MMWORD [MMBLOCK(0,1,edx,SIZEOF_DCTELEM)], mm7

    ; (Original)
    ; z1 = (tmp12 + tmp13) * 0.541196100;
    ; data2 = z1 + tmp13 * 0.765366865;
    ; data6 = z1 + tmp12 * -1.847759065;
    ;
    ; (This implementation)
    ; data2 = tmp13 * (0.541196100 + 0.765366865) + tmp12 * 0.541196100;
    ; data6 = tmp13 * 0.541196100 + tmp12 * (0.541196100 - 1.847759065);

    movq        mm4, mm1                ; mm1=tmp13
    movq        mm0, mm1
    punpcklwd   mm4, mm6                ; mm6=tmp12
    punpckhwd   mm0, mm6
    movq        mm1, mm4
    movq        mm6, mm0
    pmaddwd     mm4, [GOTOFF(ebx,PW_F130_F054)]   ; mm4=data2L
    pmaddwd     mm0, [GOTOFF(ebx,PW_F130_F054)]   ; mm0=data2H
    pmaddwd     mm1, [GOTOFF(ebx,PW_F054_MF130)]  ; mm1=data6L
    pmaddwd     mm6, [GOTOFF(ebx,PW_F054_MF130)]  ; mm6=data6H

    paddd       mm4, [GOTOFF(ebx,PD_DESCALE_P1)]
    paddd       mm0, [GOTOFF(ebx,PD_DESCALE_P1)]
    psrad       mm4, DESCALE_P1
    psrad       mm0, DESCALE_P1
    paddd       mm1, [GOTOFF(ebx,PD_DESCALE_P1)]
    paddd       mm6, [GOTOFF(ebx,PD_DESCALE_P1)]
    psrad       mm1, DESCALE_P1
    psrad       mm6, DESCALE_P1

    packssdw    mm4, mm0                ; mm4=data2
    packssdw    mm1, mm6                ; mm1=data6

    movq        MMWORD [MMBLOCK(2,0,edx,SIZEOF_DCTELEM)], mm4
    movq        MMWORD [MMBLOCK(2,1,edx,SIZEOF_DCTELEM)], mm1

    ; -- Odd part

    movq        mm5, MMWORD [wk(0)]     ; mm5=tmp6
    movq        mm7, MMWORD [wk(1)]     ; mm7=tmp7

    movq        mm0, mm2                ; mm2=tmp4
    movq        mm6, mm3                ; mm3=tmp5
    paddw       mm0, mm5                ; mm0=z3
    paddw       mm6, mm7                ; mm6=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movq        mm4, mm0
    movq        mm1, mm0
    punpcklwd   mm4, mm6
    punpckhwd   mm1, mm6
    movq        mm0, mm4
    movq        mm6, mm1
    pmaddwd     mm4, [GOTOFF(ebx,PW_MF078_F117)]  ; mm4=z3L
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF078_F117)]  ; mm1=z3H
    pmaddwd     mm0, [GOTOFF(ebx,PW_F117_F078)]   ; mm0=z4L
    pmaddwd     mm6, [GOTOFF(ebx,PW_F117_F078)]   ; mm6=z4H

    movq        MMWORD [wk(0)], mm4     ; wk(0)=z3L
    movq        MMWORD [wk(1)], mm1     ; wk(1)=z3H

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

    movq        mm4, mm2
    movq        mm1, mm2
    punpcklwd   mm4, mm7
    punpckhwd   mm1, mm7
    movq        mm2, mm4
    movq        mm7, mm1
    pmaddwd     mm4, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm4=tmp4L
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm1=tmp4H
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF089_F060)]   ; mm2=tmp7L
    pmaddwd     mm7, [GOTOFF(ebx,PW_MF089_F060)]   ; mm7=tmp7H

    paddd       mm4, MMWORD [wk(0)]     ; mm4=data7L
    paddd       mm1, MMWORD [wk(1)]     ; mm1=data7H
    paddd       mm2, mm0                ; mm2=data1L
    paddd       mm7, mm6                ; mm7=data1H

    paddd       mm4, [GOTOFF(ebx,PD_DESCALE_P1)]
    paddd       mm1, [GOTOFF(ebx,PD_DESCALE_P1)]
    psrad       mm4, DESCALE_P1
    psrad       mm1, DESCALE_P1
    paddd       mm2, [GOTOFF(ebx,PD_DESCALE_P1)]
    paddd       mm7, [GOTOFF(ebx,PD_DESCALE_P1)]
    psrad       mm2, DESCALE_P1
    psrad       mm7, DESCALE_P1

    packssdw    mm4, mm1                ; mm4=data7
    packssdw    mm2, mm7                ; mm2=data1

    movq        MMWORD [MMBLOCK(3,1,edx,SIZEOF_DCTELEM)], mm4
    movq        MMWORD [MMBLOCK(1,0,edx,SIZEOF_DCTELEM)], mm2

    movq        mm1, mm3
    movq        mm7, mm3
    punpcklwd   mm1, mm5
    punpckhwd   mm7, mm5
    movq        mm3, mm1
    movq        mm5, mm7
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm1=tmp5L
    pmaddwd     mm7, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm7=tmp5H
    pmaddwd     mm3, [GOTOFF(ebx,PW_MF256_F050)]   ; mm3=tmp6L
    pmaddwd     mm5, [GOTOFF(ebx,PW_MF256_F050)]   ; mm5=tmp6H

    paddd       mm1, mm0                ; mm1=data5L
    paddd       mm7, mm6                ; mm7=data5H
    paddd       mm3, MMWORD [wk(0)]     ; mm3=data3L
    paddd       mm5, MMWORD [wk(1)]     ; mm5=data3H

    paddd       mm1, [GOTOFF(ebx,PD_DESCALE_P1)]
    paddd       mm7, [GOTOFF(ebx,PD_DESCALE_P1)]
    psrad       mm1, DESCALE_P1
    psrad       mm7, DESCALE_P1
    paddd       mm3, [GOTOFF(ebx,PD_DESCALE_P1)]
    paddd       mm5, [GOTOFF(ebx,PD_DESCALE_P1)]
    psrad       mm3, DESCALE_P1
    psrad       mm5, DESCALE_P1

    packssdw    mm1, mm7                ; mm1=data5
    packssdw    mm3, mm5                ; mm3=data3

    movq        MMWORD [MMBLOCK(1,1,edx,SIZEOF_DCTELEM)], mm1
    movq        MMWORD [MMBLOCK(3,0,edx,SIZEOF_DCTELEM)], mm3

    add         edx, byte 4*DCTSIZE*SIZEOF_DCTELEM
    dec         ecx
    jnz         near .rowloop

    ; ---- Pass 2: process columns.

    mov         edx, POINTER [data(eax)]  ; (DCTELEM *)
    mov         ecx, DCTSIZE/4
    ALIGNX      16, 7
.columnloop:

    movq        mm0, MMWORD [MMBLOCK(2,0,edx,SIZEOF_DCTELEM)]
    movq        mm1, MMWORD [MMBLOCK(3,0,edx,SIZEOF_DCTELEM)]
    movq        mm2, MMWORD [MMBLOCK(6,0,edx,SIZEOF_DCTELEM)]
    movq        mm3, MMWORD [MMBLOCK(7,0,edx,SIZEOF_DCTELEM)]

    ; mm0=(02 12 22 32), mm2=(42 52 62 72)
    ; mm1=(03 13 23 33), mm3=(43 53 63 73)

    movq        mm4, mm0                ; transpose coefficients(phase 1)
    punpcklwd   mm0, mm1                ; mm0=(02 03 12 13)
    punpckhwd   mm4, mm1                ; mm4=(22 23 32 33)
    movq        mm5, mm2                ; transpose coefficients(phase 1)
    punpcklwd   mm2, mm3                ; mm2=(42 43 52 53)
    punpckhwd   mm5, mm3                ; mm5=(62 63 72 73)

    movq        mm6, MMWORD [MMBLOCK(0,0,edx,SIZEOF_DCTELEM)]
    movq        mm7, MMWORD [MMBLOCK(1,0,edx,SIZEOF_DCTELEM)]
    movq        mm1, MMWORD [MMBLOCK(4,0,edx,SIZEOF_DCTELEM)]
    movq        mm3, MMWORD [MMBLOCK(5,0,edx,SIZEOF_DCTELEM)]

    ; mm6=(00 10 20 30), mm1=(40 50 60 70)
    ; mm7=(01 11 21 31), mm3=(41 51 61 71)

    movq        MMWORD [wk(0)], mm4     ; wk(0)=(22 23 32 33)
    movq        MMWORD [wk(1)], mm2     ; wk(1)=(42 43 52 53)

    movq        mm4, mm6                ; transpose coefficients(phase 1)
    punpcklwd   mm6, mm7                ; mm6=(00 01 10 11)
    punpckhwd   mm4, mm7                ; mm4=(20 21 30 31)
    movq        mm2, mm1                ; transpose coefficients(phase 1)
    punpcklwd   mm1, mm3                ; mm1=(40 41 50 51)
    punpckhwd   mm2, mm3                ; mm2=(60 61 70 71)

    movq        mm7, mm6                ; transpose coefficients(phase 2)
    punpckldq   mm6, mm0                ; mm6=(00 01 02 03)=data0
    punpckhdq   mm7, mm0                ; mm7=(10 11 12 13)=data1
    movq        mm3, mm2                ; transpose coefficients(phase 2)
    punpckldq   mm2, mm5                ; mm2=(60 61 62 63)=data6
    punpckhdq   mm3, mm5                ; mm3=(70 71 72 73)=data7

    movq        mm0, mm7
    movq        mm5, mm6
    psubw       mm7, mm2                ; mm7=data1-data6=tmp6
    psubw       mm6, mm3                ; mm6=data0-data7=tmp7
    paddw       mm0, mm2                ; mm0=data1+data6=tmp1
    paddw       mm5, mm3                ; mm5=data0+data7=tmp0

    movq        mm2, MMWORD [wk(0)]     ; mm2=(22 23 32 33)
    movq        mm3, MMWORD [wk(1)]     ; mm3=(42 43 52 53)
    movq        MMWORD [wk(0)], mm7     ; wk(0)=tmp6
    movq        MMWORD [wk(1)], mm6     ; wk(1)=tmp7

    movq        mm7, mm4                ; transpose coefficients(phase 2)
    punpckldq   mm4, mm2                ; mm4=(20 21 22 23)=data2
    punpckhdq   mm7, mm2                ; mm7=(30 31 32 33)=data3
    movq        mm6, mm1                ; transpose coefficients(phase 2)
    punpckldq   mm1, mm3                ; mm1=(40 41 42 43)=data4
    punpckhdq   mm6, mm3                ; mm6=(50 51 52 53)=data5

    movq        mm2, mm7
    movq        mm3, mm4
    paddw       mm7, mm1                ; mm7=data3+data4=tmp3
    paddw       mm4, mm6                ; mm4=data2+data5=tmp2
    psubw       mm2, mm1                ; mm2=data3-data4=tmp4
    psubw       mm3, mm6                ; mm3=data2-data5=tmp5

    ; -- Even part

    movq        mm1, mm5
    movq        mm6, mm0
    paddw       mm5, mm7                ; mm5=tmp10
    paddw       mm0, mm4                ; mm0=tmp11
    psubw       mm1, mm7                ; mm1=tmp13
    psubw       mm6, mm4                ; mm6=tmp12

    movq        mm7, mm5
    paddw       mm5, mm0                ; mm5=tmp10+tmp11
    psubw       mm7, mm0                ; mm7=tmp10-tmp11

    paddw       mm5, [GOTOFF(ebx,PW_DESCALE_P2X)]
    paddw       mm7, [GOTOFF(ebx,PW_DESCALE_P2X)]
    psraw       mm5, PASS1_BITS         ; mm5=data0
    psraw       mm7, PASS1_BITS         ; mm7=data4

    movq        MMWORD [MMBLOCK(0,0,edx,SIZEOF_DCTELEM)], mm5
    movq        MMWORD [MMBLOCK(4,0,edx,SIZEOF_DCTELEM)], mm7

    ; (Original)
    ; z1 = (tmp12 + tmp13) * 0.541196100;
    ; data2 = z1 + tmp13 * 0.765366865;
    ; data6 = z1 + tmp12 * -1.847759065;
    ;
    ; (This implementation)
    ; data2 = tmp13 * (0.541196100 + 0.765366865) + tmp12 * 0.541196100;
    ; data6 = tmp13 * 0.541196100 + tmp12 * (0.541196100 - 1.847759065);

    movq        mm4, mm1                ; mm1=tmp13
    movq        mm0, mm1
    punpcklwd   mm4, mm6                ; mm6=tmp12
    punpckhwd   mm0, mm6
    movq        mm1, mm4
    movq        mm6, mm0
    pmaddwd     mm4, [GOTOFF(ebx,PW_F130_F054)]   ; mm4=data2L
    pmaddwd     mm0, [GOTOFF(ebx,PW_F130_F054)]   ; mm0=data2H
    pmaddwd     mm1, [GOTOFF(ebx,PW_F054_MF130)]  ; mm1=data6L
    pmaddwd     mm6, [GOTOFF(ebx,PW_F054_MF130)]  ; mm6=data6H

    paddd       mm4, [GOTOFF(ebx,PD_DESCALE_P2)]
    paddd       mm0, [GOTOFF(ebx,PD_DESCALE_P2)]
    psrad       mm4, DESCALE_P2
    psrad       mm0, DESCALE_P2
    paddd       mm1, [GOTOFF(ebx,PD_DESCALE_P2)]
    paddd       mm6, [GOTOFF(ebx,PD_DESCALE_P2)]
    psrad       mm1, DESCALE_P2
    psrad       mm6, DESCALE_P2

    packssdw    mm4, mm0                ; mm4=data2
    packssdw    mm1, mm6                ; mm1=data6

    movq        MMWORD [MMBLOCK(2,0,edx,SIZEOF_DCTELEM)], mm4
    movq        MMWORD [MMBLOCK(6,0,edx,SIZEOF_DCTELEM)], mm1

    ; -- Odd part

    movq        mm5, MMWORD [wk(0)]     ; mm5=tmp6
    movq        mm7, MMWORD [wk(1)]     ; mm7=tmp7

    movq        mm0, mm2                ; mm2=tmp4
    movq        mm6, mm3                ; mm3=tmp5
    paddw       mm0, mm5                ; mm0=z3
    paddw       mm6, mm7                ; mm6=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movq        mm4, mm0
    movq        mm1, mm0
    punpcklwd   mm4, mm6
    punpckhwd   mm1, mm6
    movq        mm0, mm4
    movq        mm6, mm1
    pmaddwd     mm4, [GOTOFF(ebx,PW_MF078_F117)]  ; mm4=z3L
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF078_F117)]  ; mm1=z3H
    pmaddwd     mm0, [GOTOFF(ebx,PW_F117_F078)]   ; mm0=z4L
    pmaddwd     mm6, [GOTOFF(ebx,PW_F117_F078)]   ; mm6=z4H

    movq        MMWORD [wk(0)], mm4     ; wk(0)=z3L
    movq        MMWORD [wk(1)], mm1     ; wk(1)=z3H

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

    movq        mm4, mm2
    movq        mm1, mm2
    punpcklwd   mm4, mm7
    punpckhwd   mm1, mm7
    movq        mm2, mm4
    movq        mm7, mm1
    pmaddwd     mm4, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm4=tmp4L
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm1=tmp4H
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF089_F060)]   ; mm2=tmp7L
    pmaddwd     mm7, [GOTOFF(ebx,PW_MF089_F060)]   ; mm7=tmp7H

    paddd       mm4, MMWORD [wk(0)]     ; mm4=data7L
    paddd       mm1, MMWORD [wk(1)]     ; mm1=data7H
    paddd       mm2, mm0                ; mm2=data1L
    paddd       mm7, mm6                ; mm7=data1H

    paddd       mm4, [GOTOFF(ebx,PD_DESCALE_P2)]
    paddd       mm1, [GOTOFF(ebx,PD_DESCALE_P2)]
    psrad       mm4, DESCALE_P2
    psrad       mm1, DESCALE_P2
    paddd       mm2, [GOTOFF(ebx,PD_DESCALE_P2)]
    paddd       mm7, [GOTOFF(ebx,PD_DESCALE_P2)]
    psrad       mm2, DESCALE_P2
    psrad       mm7, DESCALE_P2

    packssdw    mm4, mm1                ; mm4=data7
    packssdw    mm2, mm7                ; mm2=data1

    movq        MMWORD [MMBLOCK(7,0,edx,SIZEOF_DCTELEM)], mm4
    movq        MMWORD [MMBLOCK(1,0,edx,SIZEOF_DCTELEM)], mm2

    movq        mm1, mm3
    movq        mm7, mm3
    punpcklwd   mm1, mm5
    punpckhwd   mm7, mm5
    movq        mm3, mm1
    movq        mm5, mm7
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm1=tmp5L
    pmaddwd     mm7, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm7=tmp5H
    pmaddwd     mm3, [GOTOFF(ebx,PW_MF256_F050)]   ; mm3=tmp6L
    pmaddwd     mm5, [GOTOFF(ebx,PW_MF256_F050)]   ; mm5=tmp6H

    paddd       mm1, mm0                ; mm1=data5L
    paddd       mm7, mm6                ; mm7=data5H
    paddd       mm3, MMWORD [wk(0)]     ; mm3=data3L
    paddd       mm5, MMWORD [wk(1)]     ; mm5=data3H

    paddd       mm1, [GOTOFF(ebx,PD_DESCALE_P2)]
    paddd       mm7, [GOTOFF(ebx,PD_DESCALE_P2)]
    psrad       mm1, DESCALE_P2
    psrad       mm7, DESCALE_P2
    paddd       mm3, [GOTOFF(ebx,PD_DESCALE_P2)]
    paddd       mm5, [GOTOFF(ebx,PD_DESCALE_P2)]
    psrad       mm3, DESCALE_P2
    psrad       mm5, DESCALE_P2

    packssdw    mm1, mm7                ; mm1=data5
    packssdw    mm3, mm5                ; mm3=data3

    movq        MMWORD [MMBLOCK(5,0,edx,SIZEOF_DCTELEM)], mm1
    movq        MMWORD [MMBLOCK(3,0,edx,SIZEOF_DCTELEM)], mm3

    add         edx, byte 4*SIZEOF_DCTELEM
    dec         ecx
    jnz         near .columnloop

    emms                                ; empty MMX state

;   pop         edi                     ; unused
;   pop         esi                     ; unused
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    POPPIC      ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
