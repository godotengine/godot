;
; jidctint.asm - accurate integer IDCT (MMX)
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
; inverse DCT (Discrete Cosine Transform). The following code is based
; directly on the IJG's original jidctint.c; see the jidctint.c for
; more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%define CONST_BITS  13
%define PASS1_BITS  2

%define DESCALE_P1  (CONST_BITS - PASS1_BITS)
%define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)

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
    GLOBAL_DATA(jconst_idct_islow_mmx)

EXTN(jconst_idct_islow_mmx):

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
PB_CENTERJSAMP times 8 db  CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform dequantization and inverse DCT on one block of coefficients.
;
; GLOBAL(void)
; jsimd_idct_islow_mmx(void *dct_table, JCOEFPTR coef_block,
;                      JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; jpeg_component_info *compptr
%define coef_block(b)  (b) + 12         ; JCOEFPTR coef_block
%define output_buf(b)  (b) + 16         ; JSAMPARRAY output_buf
%define output_col(b)  (b) + 20         ; JDIMENSION output_col

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_MMWORD
                                        ; mmword wk[WK_NUM]
%define WK_NUM         12
%define workspace      wk(0) - DCTSIZE2 * SIZEOF_JCOEF
                                        ; JCOEF workspace[DCTSIZE2]

    align       32
    GLOBAL_FUNCTION(jsimd_idct_islow_mmx)

EXTN(jsimd_idct_islow_mmx):
    push        ebp
    mov         eax, esp                    ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_MMWORD)  ; align to 64 bits
    mov         [esp], eax
    mov         ebp, esp                    ; ebp = aligned ebp
    lea         esp, [workspace]
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    GET_GOT     ebx                     ; get GOT address

    ; ---- Pass 1: process columns from input, store into work array.

;   mov         eax, [original_ebp]
    mov         edx, POINTER [dct_table(eax)]    ; quantptr
    mov         esi, JCOEFPTR [coef_block(eax)]  ; inptr
    lea         edi, [workspace]                 ; JCOEF *wsptr
    mov         ecx, DCTSIZE/4                   ; ctr
    ALIGNX      16, 7
.columnloop:
%ifndef NO_ZERO_COLUMN_TEST_ISLOW_MMX
    mov         eax, dword [DWBLOCK(1,0,esi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,esi,SIZEOF_JCOEF)]
    jnz         short .columnDCT

    movq        mm0, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    por         mm0, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    por         mm1, MMWORD [MMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    por         mm0, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    por         mm1, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    por         mm0, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    por         mm1, mm0
    packsswb    mm1, mm1
    movd        eax, mm1
    test        eax, eax
    jnz         short .columnDCT

    ; -- AC terms all zero

    movq        mm0, MMWORD [MMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    pmullw      mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    psllw       mm0, PASS1_BITS

    movq        mm2, mm0                ; mm0=in0=(00 01 02 03)
    punpcklwd   mm0, mm0                ; mm0=(00 00 01 01)
    punpckhwd   mm2, mm2                ; mm2=(02 02 03 03)

    movq        mm1, mm0
    punpckldq   mm0, mm0                ; mm0=(00 00 00 00)
    punpckhdq   mm1, mm1                ; mm1=(01 01 01 01)
    movq        mm3, mm2
    punpckldq   mm2, mm2                ; mm2=(02 02 02 02)
    punpckhdq   mm3, mm3                ; mm3=(03 03 03 03)

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_JCOEF)], mm0
    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_JCOEF)], mm0
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_JCOEF)], mm1
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_JCOEF)], mm1
    movq        MMWORD [MMBLOCK(2,0,edi,SIZEOF_JCOEF)], mm2
    movq        MMWORD [MMBLOCK(2,1,edi,SIZEOF_JCOEF)], mm2
    movq        MMWORD [MMBLOCK(3,0,edi,SIZEOF_JCOEF)], mm3
    movq        MMWORD [MMBLOCK(3,1,edi,SIZEOF_JCOEF)], mm3
    jmp         near .nextcolumn
    ALIGNX      16, 7
%endif
.columnDCT:

    ; -- Even part

    movq        mm0, MMWORD [MMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    pmullw      mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm1, MMWORD [MMBLOCK(2,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    movq        mm2, MMWORD [MMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    pmullw      mm2, MMWORD [MMBLOCK(4,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm3, MMWORD [MMBLOCK(6,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    ; (Original)
    ; z1 = (z2 + z3) * 0.541196100;
    ; tmp2 = z1 + z3 * -1.847759065;
    ; tmp3 = z1 + z2 * 0.765366865;
    ;
    ; (This implementation)
    ; tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065);
    ; tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100;

    movq        mm4, mm1                ; mm1=in2=z2
    movq        mm5, mm1
    punpcklwd   mm4, mm3                ; mm3=in6=z3
    punpckhwd   mm5, mm3
    movq        mm1, mm4
    movq        mm3, mm5
    pmaddwd     mm4, [GOTOFF(ebx,PW_F130_F054)]   ; mm4=tmp3L
    pmaddwd     mm5, [GOTOFF(ebx,PW_F130_F054)]   ; mm5=tmp3H
    pmaddwd     mm1, [GOTOFF(ebx,PW_F054_MF130)]  ; mm1=tmp2L
    pmaddwd     mm3, [GOTOFF(ebx,PW_F054_MF130)]  ; mm3=tmp2H

    movq        mm6, mm0
    paddw       mm0, mm2                ; mm0=in0+in4
    psubw       mm6, mm2                ; mm6=in0-in4

    pxor        mm7, mm7
    pxor        mm2, mm2
    punpcklwd   mm7, mm0                ; mm7=tmp0L
    punpckhwd   mm2, mm0                ; mm2=tmp0H
    psrad       mm7, (16-CONST_BITS)    ; psrad mm7,16 & pslld mm7,CONST_BITS
    psrad       mm2, (16-CONST_BITS)    ; psrad mm2,16 & pslld mm2,CONST_BITS

    movq        mm0, mm7
    paddd       mm7, mm4                ; mm7=tmp10L
    psubd       mm0, mm4                ; mm0=tmp13L
    movq        mm4, mm2
    paddd       mm2, mm5                ; mm2=tmp10H
    psubd       mm4, mm5                ; mm4=tmp13H

    movq        MMWORD [wk(0)], mm7     ; wk(0)=tmp10L
    movq        MMWORD [wk(1)], mm2     ; wk(1)=tmp10H
    movq        MMWORD [wk(2)], mm0     ; wk(2)=tmp13L
    movq        MMWORD [wk(3)], mm4     ; wk(3)=tmp13H

    pxor        mm5, mm5
    pxor        mm7, mm7
    punpcklwd   mm5, mm6                ; mm5=tmp1L
    punpckhwd   mm7, mm6                ; mm7=tmp1H
    psrad       mm5, (16-CONST_BITS)    ; psrad mm5,16 & pslld mm5,CONST_BITS
    psrad       mm7, (16-CONST_BITS)    ; psrad mm7,16 & pslld mm7,CONST_BITS

    movq        mm2, mm5
    paddd       mm5, mm1                ; mm5=tmp11L
    psubd       mm2, mm1                ; mm2=tmp12L
    movq        mm0, mm7
    paddd       mm7, mm3                ; mm7=tmp11H
    psubd       mm0, mm3                ; mm0=tmp12H

    movq        MMWORD [wk(4)], mm5     ; wk(4)=tmp11L
    movq        MMWORD [wk(5)], mm7     ; wk(5)=tmp11H
    movq        MMWORD [wk(6)], mm2     ; wk(6)=tmp12L
    movq        MMWORD [wk(7)], mm0     ; wk(7)=tmp12H

    ; -- Odd part

    movq        mm4, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm6, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    pmullw      mm4, MMWORD [MMBLOCK(1,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm6, MMWORD [MMBLOCK(3,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    movq        mm1, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    pmullw      mm1, MMWORD [MMBLOCK(5,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm3, MMWORD [MMBLOCK(7,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    movq        mm5, mm6
    movq        mm7, mm4
    paddw       mm5, mm3                ; mm5=z3
    paddw       mm7, mm1                ; mm7=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movq        mm2, mm5
    movq        mm0, mm5
    punpcklwd   mm2, mm7
    punpckhwd   mm0, mm7
    movq        mm5, mm2
    movq        mm7, mm0
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF078_F117)]  ; mm2=z3L
    pmaddwd     mm0, [GOTOFF(ebx,PW_MF078_F117)]  ; mm0=z3H
    pmaddwd     mm5, [GOTOFF(ebx,PW_F117_F078)]   ; mm5=z4L
    pmaddwd     mm7, [GOTOFF(ebx,PW_F117_F078)]   ; mm7=z4H

    movq        MMWORD [wk(10)], mm2    ; wk(10)=z3L
    movq        MMWORD [wk(11)], mm0    ; wk(11)=z3H

    ; (Original)
    ; z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
    ; tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
    ; tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
    ; z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
    ; tmp0 += z1 + z3;  tmp1 += z2 + z4;
    ; tmp2 += z2 + z3;  tmp3 += z1 + z4;
    ;
    ; (This implementation)
    ; tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
    ; tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
    ; tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
    ; tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
    ; tmp0 += z3;  tmp1 += z4;
    ; tmp2 += z3;  tmp3 += z4;

    movq        mm2, mm3
    movq        mm0, mm3
    punpcklwd   mm2, mm4
    punpckhwd   mm0, mm4
    movq        mm3, mm2
    movq        mm4, mm0
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm2=tmp0L
    pmaddwd     mm0, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm0=tmp0H
    pmaddwd     mm3, [GOTOFF(ebx,PW_MF089_F060)]   ; mm3=tmp3L
    pmaddwd     mm4, [GOTOFF(ebx,PW_MF089_F060)]   ; mm4=tmp3H

    paddd       mm2, MMWORD [wk(10)]    ; mm2=tmp0L
    paddd       mm0, MMWORD [wk(11)]    ; mm0=tmp0H
    paddd       mm3, mm5                ; mm3=tmp3L
    paddd       mm4, mm7                ; mm4=tmp3H

    movq        MMWORD [wk(8)], mm2     ; wk(8)=tmp0L
    movq        MMWORD [wk(9)], mm0     ; wk(9)=tmp0H

    movq        mm2, mm1
    movq        mm0, mm1
    punpcklwd   mm2, mm6
    punpckhwd   mm0, mm6
    movq        mm1, mm2
    movq        mm6, mm0
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm2=tmp1L
    pmaddwd     mm0, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm0=tmp1H
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF256_F050)]   ; mm1=tmp2L
    pmaddwd     mm6, [GOTOFF(ebx,PW_MF256_F050)]   ; mm6=tmp2H

    paddd       mm2, mm5                ; mm2=tmp1L
    paddd       mm0, mm7                ; mm0=tmp1H
    paddd       mm1, MMWORD [wk(10)]    ; mm1=tmp2L
    paddd       mm6, MMWORD [wk(11)]    ; mm6=tmp2H

    movq        MMWORD [wk(10)], mm2    ; wk(10)=tmp1L
    movq        MMWORD [wk(11)], mm0    ; wk(11)=tmp1H

    ; -- Final output stage

    movq        mm5, MMWORD [wk(0)]     ; mm5=tmp10L
    movq        mm7, MMWORD [wk(1)]     ; mm7=tmp10H

    movq        mm2, mm5
    movq        mm0, mm7
    paddd       mm5, mm3                ; mm5=data0L
    paddd       mm7, mm4                ; mm7=data0H
    psubd       mm2, mm3                ; mm2=data7L
    psubd       mm0, mm4                ; mm0=data7H

    movq        mm3, [GOTOFF(ebx,PD_DESCALE_P1)]  ; mm3=[PD_DESCALE_P1]

    paddd       mm5, mm3
    paddd       mm7, mm3
    psrad       mm5, DESCALE_P1
    psrad       mm7, DESCALE_P1
    paddd       mm2, mm3
    paddd       mm0, mm3
    psrad       mm2, DESCALE_P1
    psrad       mm0, DESCALE_P1

    packssdw    mm5, mm7                ; mm5=data0=(00 01 02 03)
    packssdw    mm2, mm0                ; mm2=data7=(70 71 72 73)

    movq        mm4, MMWORD [wk(4)]     ; mm4=tmp11L
    movq        mm3, MMWORD [wk(5)]     ; mm3=tmp11H

    movq        mm7, mm4
    movq        mm0, mm3
    paddd       mm4, mm1                ; mm4=data1L
    paddd       mm3, mm6                ; mm3=data1H
    psubd       mm7, mm1                ; mm7=data6L
    psubd       mm0, mm6                ; mm0=data6H

    movq        mm1, [GOTOFF(ebx,PD_DESCALE_P1)]  ; mm1=[PD_DESCALE_P1]

    paddd       mm4, mm1
    paddd       mm3, mm1
    psrad       mm4, DESCALE_P1
    psrad       mm3, DESCALE_P1
    paddd       mm7, mm1
    paddd       mm0, mm1
    psrad       mm7, DESCALE_P1
    psrad       mm0, DESCALE_P1

    packssdw    mm4, mm3                ; mm4=data1=(10 11 12 13)
    packssdw    mm7, mm0                ; mm7=data6=(60 61 62 63)

    movq        mm6, mm5                ; transpose coefficients(phase 1)
    punpcklwd   mm5, mm4                ; mm5=(00 10 01 11)
    punpckhwd   mm6, mm4                ; mm6=(02 12 03 13)
    movq        mm1, mm7                ; transpose coefficients(phase 1)
    punpcklwd   mm7, mm2                ; mm7=(60 70 61 71)
    punpckhwd   mm1, mm2                ; mm1=(62 72 63 73)

    movq        mm3, MMWORD [wk(6)]     ; mm3=tmp12L
    movq        mm0, MMWORD [wk(7)]     ; mm0=tmp12H
    movq        mm4, MMWORD [wk(10)]    ; mm4=tmp1L
    movq        mm2, MMWORD [wk(11)]    ; mm2=tmp1H

    movq        MMWORD [wk(0)], mm5     ; wk(0)=(00 10 01 11)
    movq        MMWORD [wk(1)], mm6     ; wk(1)=(02 12 03 13)
    movq        MMWORD [wk(4)], mm7     ; wk(4)=(60 70 61 71)
    movq        MMWORD [wk(5)], mm1     ; wk(5)=(62 72 63 73)

    movq        mm5, mm3
    movq        mm6, mm0
    paddd       mm3, mm4                ; mm3=data2L
    paddd       mm0, mm2                ; mm0=data2H
    psubd       mm5, mm4                ; mm5=data5L
    psubd       mm6, mm2                ; mm6=data5H

    movq        mm7, [GOTOFF(ebx,PD_DESCALE_P1)]  ; mm7=[PD_DESCALE_P1]

    paddd       mm3, mm7
    paddd       mm0, mm7
    psrad       mm3, DESCALE_P1
    psrad       mm0, DESCALE_P1
    paddd       mm5, mm7
    paddd       mm6, mm7
    psrad       mm5, DESCALE_P1
    psrad       mm6, DESCALE_P1

    packssdw    mm3, mm0                ; mm3=data2=(20 21 22 23)
    packssdw    mm5, mm6                ; mm5=data5=(50 51 52 53)

    movq        mm1, MMWORD [wk(2)]     ; mm1=tmp13L
    movq        mm4, MMWORD [wk(3)]     ; mm4=tmp13H
    movq        mm2, MMWORD [wk(8)]     ; mm2=tmp0L
    movq        mm7, MMWORD [wk(9)]     ; mm7=tmp0H

    movq        mm0, mm1
    movq        mm6, mm4
    paddd       mm1, mm2                ; mm1=data3L
    paddd       mm4, mm7                ; mm4=data3H
    psubd       mm0, mm2                ; mm0=data4L
    psubd       mm6, mm7                ; mm6=data4H

    movq        mm2, [GOTOFF(ebx,PD_DESCALE_P1)]  ; mm2=[PD_DESCALE_P1]

    paddd       mm1, mm2
    paddd       mm4, mm2
    psrad       mm1, DESCALE_P1
    psrad       mm4, DESCALE_P1
    paddd       mm0, mm2
    paddd       mm6, mm2
    psrad       mm0, DESCALE_P1
    psrad       mm6, DESCALE_P1

    packssdw    mm1, mm4                ; mm1=data3=(30 31 32 33)
    packssdw    mm0, mm6                ; mm0=data4=(40 41 42 43)

    movq        mm7, MMWORD [wk(0)]     ; mm7=(00 10 01 11)
    movq        mm2, MMWORD [wk(1)]     ; mm2=(02 12 03 13)

    movq        mm4, mm3                ; transpose coefficients(phase 1)
    punpcklwd   mm3, mm1                ; mm3=(20 30 21 31)
    punpckhwd   mm4, mm1                ; mm4=(22 32 23 33)
    movq        mm6, mm0                ; transpose coefficients(phase 1)
    punpcklwd   mm0, mm5                ; mm0=(40 50 41 51)
    punpckhwd   mm6, mm5                ; mm6=(42 52 43 53)

    movq        mm1, mm7                ; transpose coefficients(phase 2)
    punpckldq   mm7, mm3                ; mm7=(00 10 20 30)
    punpckhdq   mm1, mm3                ; mm1=(01 11 21 31)
    movq        mm5, mm2                ; transpose coefficients(phase 2)
    punpckldq   mm2, mm4                ; mm2=(02 12 22 32)
    punpckhdq   mm5, mm4                ; mm5=(03 13 23 33)

    movq        mm3, MMWORD [wk(4)]     ; mm3=(60 70 61 71)
    movq        mm4, MMWORD [wk(5)]     ; mm4=(62 72 63 73)

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_JCOEF)], mm7
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_JCOEF)], mm1
    movq        MMWORD [MMBLOCK(2,0,edi,SIZEOF_JCOEF)], mm2
    movq        MMWORD [MMBLOCK(3,0,edi,SIZEOF_JCOEF)], mm5

    movq        mm7, mm0                ; transpose coefficients(phase 2)
    punpckldq   mm0, mm3                ; mm0=(40 50 60 70)
    punpckhdq   mm7, mm3                ; mm7=(41 51 61 71)
    movq        mm1, mm6                ; transpose coefficients(phase 2)
    punpckldq   mm6, mm4                ; mm6=(42 52 62 72)
    punpckhdq   mm1, mm4                ; mm1=(43 53 63 73)

    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_JCOEF)], mm0
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_JCOEF)], mm7
    movq        MMWORD [MMBLOCK(2,1,edi,SIZEOF_JCOEF)], mm6
    movq        MMWORD [MMBLOCK(3,1,edi,SIZEOF_JCOEF)], mm1

.nextcolumn:
    add         esi, byte 4*SIZEOF_JCOEF            ; coef_block
    add         edx, byte 4*SIZEOF_ISLOW_MULT_TYPE  ; quantptr
    add         edi, byte 4*DCTSIZE*SIZEOF_JCOEF    ; wsptr
    dec         ecx                                 ; ctr
    jnz         near .columnloop

    ; ---- Pass 2: process rows from work array, store into output array.

    mov         eax, [original_ebp]
    lea         esi, [workspace]                   ; JCOEF *wsptr
    mov         edi, JSAMPARRAY [output_buf(eax)]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [output_col(eax)]
    mov         ecx, DCTSIZE/4                     ; ctr
    ALIGNX      16, 7
.rowloop:

    ; -- Even part

    movq        mm0, MMWORD [MMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    movq        mm2, MMWORD [MMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]

    ; (Original)
    ; z1 = (z2 + z3) * 0.541196100;
    ; tmp2 = z1 + z3 * -1.847759065;
    ; tmp3 = z1 + z2 * 0.765366865;
    ;
    ; (This implementation)
    ; tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065);
    ; tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100;

    movq        mm4, mm1                ; mm1=in2=z2
    movq        mm5, mm1
    punpcklwd   mm4, mm3                ; mm3=in6=z3
    punpckhwd   mm5, mm3
    movq        mm1, mm4
    movq        mm3, mm5
    pmaddwd     mm4, [GOTOFF(ebx,PW_F130_F054)]   ; mm4=tmp3L
    pmaddwd     mm5, [GOTOFF(ebx,PW_F130_F054)]   ; mm5=tmp3H
    pmaddwd     mm1, [GOTOFF(ebx,PW_F054_MF130)]  ; mm1=tmp2L
    pmaddwd     mm3, [GOTOFF(ebx,PW_F054_MF130)]  ; mm3=tmp2H

    movq        mm6, mm0
    paddw       mm0, mm2                ; mm0=in0+in4
    psubw       mm6, mm2                ; mm6=in0-in4

    pxor        mm7, mm7
    pxor        mm2, mm2
    punpcklwd   mm7, mm0                ; mm7=tmp0L
    punpckhwd   mm2, mm0                ; mm2=tmp0H
    psrad       mm7, (16-CONST_BITS)    ; psrad mm7,16 & pslld mm7,CONST_BITS
    psrad       mm2, (16-CONST_BITS)    ; psrad mm2,16 & pslld mm2,CONST_BITS

    movq        mm0, mm7
    paddd       mm7, mm4                ; mm7=tmp10L
    psubd       mm0, mm4                ; mm0=tmp13L
    movq        mm4, mm2
    paddd       mm2, mm5                ; mm2=tmp10H
    psubd       mm4, mm5                ; mm4=tmp13H

    movq        MMWORD [wk(0)], mm7     ; wk(0)=tmp10L
    movq        MMWORD [wk(1)], mm2     ; wk(1)=tmp10H
    movq        MMWORD [wk(2)], mm0     ; wk(2)=tmp13L
    movq        MMWORD [wk(3)], mm4     ; wk(3)=tmp13H

    pxor        mm5, mm5
    pxor        mm7, mm7
    punpcklwd   mm5, mm6                ; mm5=tmp1L
    punpckhwd   mm7, mm6                ; mm7=tmp1H
    psrad       mm5, (16-CONST_BITS)    ; psrad mm5,16 & pslld mm5,CONST_BITS
    psrad       mm7, (16-CONST_BITS)    ; psrad mm7,16 & pslld mm7,CONST_BITS

    movq        mm2, mm5
    paddd       mm5, mm1                ; mm5=tmp11L
    psubd       mm2, mm1                ; mm2=tmp12L
    movq        mm0, mm7
    paddd       mm7, mm3                ; mm7=tmp11H
    psubd       mm0, mm3                ; mm0=tmp12H

    movq        MMWORD [wk(4)], mm5     ; wk(4)=tmp11L
    movq        MMWORD [wk(5)], mm7     ; wk(5)=tmp11H
    movq        MMWORD [wk(6)], mm2     ; wk(6)=tmp12L
    movq        MMWORD [wk(7)], mm0     ; wk(7)=tmp12H

    ; -- Odd part

    movq        mm4, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm6, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]

    movq        mm5, mm6
    movq        mm7, mm4
    paddw       mm5, mm3                ; mm5=z3
    paddw       mm7, mm1                ; mm7=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movq        mm2, mm5
    movq        mm0, mm5
    punpcklwd   mm2, mm7
    punpckhwd   mm0, mm7
    movq        mm5, mm2
    movq        mm7, mm0
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF078_F117)]  ; mm2=z3L
    pmaddwd     mm0, [GOTOFF(ebx,PW_MF078_F117)]  ; mm0=z3H
    pmaddwd     mm5, [GOTOFF(ebx,PW_F117_F078)]   ; mm5=z4L
    pmaddwd     mm7, [GOTOFF(ebx,PW_F117_F078)]   ; mm7=z4H

    movq        MMWORD [wk(10)], mm2    ; wk(10)=z3L
    movq        MMWORD [wk(11)], mm0    ; wk(11)=z3H

    ; (Original)
    ; z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
    ; tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
    ; tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
    ; z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
    ; tmp0 += z1 + z3;  tmp1 += z2 + z4;
    ; tmp2 += z2 + z3;  tmp3 += z1 + z4;
    ;
    ; (This implementation)
    ; tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
    ; tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
    ; tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
    ; tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
    ; tmp0 += z3;  tmp1 += z4;
    ; tmp2 += z3;  tmp3 += z4;

    movq        mm2, mm3
    movq        mm0, mm3
    punpcklwd   mm2, mm4
    punpckhwd   mm0, mm4
    movq        mm3, mm2
    movq        mm4, mm0
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm2=tmp0L
    pmaddwd     mm0, [GOTOFF(ebx,PW_MF060_MF089)]  ; mm0=tmp0H
    pmaddwd     mm3, [GOTOFF(ebx,PW_MF089_F060)]   ; mm3=tmp3L
    pmaddwd     mm4, [GOTOFF(ebx,PW_MF089_F060)]   ; mm4=tmp3H

    paddd       mm2, MMWORD [wk(10)]    ; mm2=tmp0L
    paddd       mm0, MMWORD [wk(11)]    ; mm0=tmp0H
    paddd       mm3, mm5                ; mm3=tmp3L
    paddd       mm4, mm7                ; mm4=tmp3H

    movq        MMWORD [wk(8)], mm2     ; wk(8)=tmp0L
    movq        MMWORD [wk(9)], mm0     ; wk(9)=tmp0H

    movq        mm2, mm1
    movq        mm0, mm1
    punpcklwd   mm2, mm6
    punpckhwd   mm0, mm6
    movq        mm1, mm2
    movq        mm6, mm0
    pmaddwd     mm2, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm2=tmp1L
    pmaddwd     mm0, [GOTOFF(ebx,PW_MF050_MF256)]  ; mm0=tmp1H
    pmaddwd     mm1, [GOTOFF(ebx,PW_MF256_F050)]   ; mm1=tmp2L
    pmaddwd     mm6, [GOTOFF(ebx,PW_MF256_F050)]   ; mm6=tmp2H

    paddd       mm2, mm5                ; mm2=tmp1L
    paddd       mm0, mm7                ; mm0=tmp1H
    paddd       mm1, MMWORD [wk(10)]    ; mm1=tmp2L
    paddd       mm6, MMWORD [wk(11)]    ; mm6=tmp2H

    movq        MMWORD [wk(10)], mm2    ; wk(10)=tmp1L
    movq        MMWORD [wk(11)], mm0    ; wk(11)=tmp1H

    ; -- Final output stage

    movq        mm5, MMWORD [wk(0)]     ; mm5=tmp10L
    movq        mm7, MMWORD [wk(1)]     ; mm7=tmp10H

    movq        mm2, mm5
    movq        mm0, mm7
    paddd       mm5, mm3                ; mm5=data0L
    paddd       mm7, mm4                ; mm7=data0H
    psubd       mm2, mm3                ; mm2=data7L
    psubd       mm0, mm4                ; mm0=data7H

    movq        mm3, [GOTOFF(ebx,PD_DESCALE_P2)]  ; mm3=[PD_DESCALE_P2]

    paddd       mm5, mm3
    paddd       mm7, mm3
    psrad       mm5, DESCALE_P2
    psrad       mm7, DESCALE_P2
    paddd       mm2, mm3
    paddd       mm0, mm3
    psrad       mm2, DESCALE_P2
    psrad       mm0, DESCALE_P2

    packssdw    mm5, mm7                ; mm5=data0=(00 10 20 30)
    packssdw    mm2, mm0                ; mm2=data7=(07 17 27 37)

    movq        mm4, MMWORD [wk(4)]     ; mm4=tmp11L
    movq        mm3, MMWORD [wk(5)]     ; mm3=tmp11H

    movq        mm7, mm4
    movq        mm0, mm3
    paddd       mm4, mm1                ; mm4=data1L
    paddd       mm3, mm6                ; mm3=data1H
    psubd       mm7, mm1                ; mm7=data6L
    psubd       mm0, mm6                ; mm0=data6H

    movq        mm1, [GOTOFF(ebx,PD_DESCALE_P2)]  ; mm1=[PD_DESCALE_P2]

    paddd       mm4, mm1
    paddd       mm3, mm1
    psrad       mm4, DESCALE_P2
    psrad       mm3, DESCALE_P2
    paddd       mm7, mm1
    paddd       mm0, mm1
    psrad       mm7, DESCALE_P2
    psrad       mm0, DESCALE_P2

    packssdw    mm4, mm3                ; mm4=data1=(01 11 21 31)
    packssdw    mm7, mm0                ; mm7=data6=(06 16 26 36)

    packsswb    mm5, mm7                ; mm5=(00 10 20 30 06 16 26 36)
    packsswb    mm4, mm2                ; mm4=(01 11 21 31 07 17 27 37)

    movq        mm6, MMWORD [wk(6)]     ; mm6=tmp12L
    movq        mm1, MMWORD [wk(7)]     ; mm1=tmp12H
    movq        mm3, MMWORD [wk(10)]    ; mm3=tmp1L
    movq        mm0, MMWORD [wk(11)]    ; mm0=tmp1H

    movq        MMWORD [wk(0)], mm5     ; wk(0)=(00 10 20 30 06 16 26 36)
    movq        MMWORD [wk(1)], mm4     ; wk(1)=(01 11 21 31 07 17 27 37)

    movq        mm7, mm6
    movq        mm2, mm1
    paddd       mm6, mm3                ; mm6=data2L
    paddd       mm1, mm0                ; mm1=data2H
    psubd       mm7, mm3                ; mm7=data5L
    psubd       mm2, mm0                ; mm2=data5H

    movq        mm5, [GOTOFF(ebx,PD_DESCALE_P2)]  ; mm5=[PD_DESCALE_P2]

    paddd       mm6, mm5
    paddd       mm1, mm5
    psrad       mm6, DESCALE_P2
    psrad       mm1, DESCALE_P2
    paddd       mm7, mm5
    paddd       mm2, mm5
    psrad       mm7, DESCALE_P2
    psrad       mm2, DESCALE_P2

    packssdw    mm6, mm1                ; mm6=data2=(02 12 22 32)
    packssdw    mm7, mm2                ; mm7=data5=(05 15 25 35)

    movq        mm4, MMWORD [wk(2)]     ; mm4=tmp13L
    movq        mm3, MMWORD [wk(3)]     ; mm3=tmp13H
    movq        mm0, MMWORD [wk(8)]     ; mm0=tmp0L
    movq        mm5, MMWORD [wk(9)]     ; mm5=tmp0H

    movq        mm1, mm4
    movq        mm2, mm3
    paddd       mm4, mm0                ; mm4=data3L
    paddd       mm3, mm5                ; mm3=data3H
    psubd       mm1, mm0                ; mm1=data4L
    psubd       mm2, mm5                ; mm2=data4H

    movq        mm0, [GOTOFF(ebx,PD_DESCALE_P2)]  ; mm0=[PD_DESCALE_P2]

    paddd       mm4, mm0
    paddd       mm3, mm0
    psrad       mm4, DESCALE_P2
    psrad       mm3, DESCALE_P2
    paddd       mm1, mm0
    paddd       mm2, mm0
    psrad       mm1, DESCALE_P2
    psrad       mm2, DESCALE_P2

    movq        mm5, [GOTOFF(ebx,PB_CENTERJSAMP)]  ; mm5=[PB_CENTERJSAMP]

    packssdw    mm4, mm3                ; mm4=data3=(03 13 23 33)
    packssdw    mm1, mm2                ; mm1=data4=(04 14 24 34)

    movq        mm0, MMWORD [wk(0)]     ; mm0=(00 10 20 30 06 16 26 36)
    movq        mm3, MMWORD [wk(1)]     ; mm3=(01 11 21 31 07 17 27 37)

    packsswb    mm6, mm1                ; mm6=(02 12 22 32 04 14 24 34)
    packsswb    mm4, mm7                ; mm4=(03 13 23 33 05 15 25 35)

    paddb       mm0, mm5
    paddb       mm3, mm5
    paddb       mm6, mm5
    paddb       mm4, mm5

    movq        mm2, mm0                ; transpose coefficients(phase 1)
    punpcklbw   mm0, mm3                ; mm0=(00 01 10 11 20 21 30 31)
    punpckhbw   mm2, mm3                ; mm2=(06 07 16 17 26 27 36 37)
    movq        mm1, mm6                ; transpose coefficients(phase 1)
    punpcklbw   mm6, mm4                ; mm6=(02 03 12 13 22 23 32 33)
    punpckhbw   mm1, mm4                ; mm1=(04 05 14 15 24 25 34 35)

    movq        mm7, mm0                ; transpose coefficients(phase 2)
    punpcklwd   mm0, mm6                ; mm0=(00 01 02 03 10 11 12 13)
    punpckhwd   mm7, mm6                ; mm7=(20 21 22 23 30 31 32 33)
    movq        mm5, mm1                ; transpose coefficients(phase 2)
    punpcklwd   mm1, mm2                ; mm1=(04 05 06 07 14 15 16 17)
    punpckhwd   mm5, mm2                ; mm5=(24 25 26 27 34 35 36 37)

    movq        mm3, mm0                ; transpose coefficients(phase 3)
    punpckldq   mm0, mm1                ; mm0=(00 01 02 03 04 05 06 07)
    punpckhdq   mm3, mm1                ; mm3=(10 11 12 13 14 15 16 17)
    movq        mm4, mm7                ; transpose coefficients(phase 3)
    punpckldq   mm7, mm5                ; mm7=(20 21 22 23 24 25 26 27)
    punpckhdq   mm4, mm5                ; mm4=(30 31 32 33 34 35 36 37)

    PUSHPIC     ebx                     ; save GOT address

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]
    mov         ebx, JSAMPROW [edi+1*SIZEOF_JSAMPROW]
    movq        MMWORD [edx+eax*SIZEOF_JSAMPLE], mm0
    movq        MMWORD [ebx+eax*SIZEOF_JSAMPLE], mm3
    mov         edx, JSAMPROW [edi+2*SIZEOF_JSAMPROW]
    mov         ebx, JSAMPROW [edi+3*SIZEOF_JSAMPROW]
    movq        MMWORD [edx+eax*SIZEOF_JSAMPLE], mm7
    movq        MMWORD [ebx+eax*SIZEOF_JSAMPLE], mm4

    POPPIC      ebx                     ; restore GOT address

    add         esi, byte 4*SIZEOF_JCOEF     ; wsptr
    add         edi, byte 4*SIZEOF_JSAMPROW
    dec         ecx                          ; ctr
    jnz         near .rowloop

    emms                                ; empty MMX state

    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    pop         ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
