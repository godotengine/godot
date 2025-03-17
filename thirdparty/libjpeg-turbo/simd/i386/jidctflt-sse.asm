;
; jidctflt.asm - floating-point IDCT (SSE & MMX)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2016, 2024, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.
;
; This file contains a floating-point implementation of the inverse DCT
; (Discrete Cosine Transform). The following code is based directly on
; the IJG's original jidctflt.c; see the jidctflt.c for more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%macro UNPCKLPS2 2  ; %1=(0 1 2 3) / %2=(4 5 6 7) => %1=(0 1 4 5)
    shufps      %1, %2, 0x44
%endmacro

%macro UNPCKHPS2 2  ; %1=(0 1 2 3) / %2=(4 5 6 7) => %1=(2 3 6 7)
    shufps      %1, %2, 0xEE
%endmacro

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_idct_float_sse)

EXTN(jconst_idct_float_sse):

PD_1_414       times 4 dd  1.414213562373095048801689
PD_1_847       times 4 dd  1.847759065022573512256366
PD_1_082       times 4 dd  1.082392200292393968799446
PD_M2_613      times 4 dd -2.613125929752753055713286
PD_0_125       times 4 dd  0.125        ; 1/8
PB_CENTERJSAMP times 8 db  CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform dequantization and inverse DCT on one block of coefficients.
;
; GLOBAL(void)
; jsimd_idct_float_sse(void *dct_table, JCOEFPTR coef_block,
;                      JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; void *dct_table
%define coef_block(b)  (b) + 12         ; JCOEFPTR coef_block
%define output_buf(b)  (b) + 16         ; JSAMPARRAY output_buf
%define output_col(b)  (b) + 20         ; JDIMENSION output_col

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_XMMWORD
                                        ; xmmword wk[WK_NUM]
%define WK_NUM         2
%define workspace      wk(0) - DCTSIZE2 * SIZEOF_FAST_FLOAT
                                        ; FAST_FLOAT workspace[DCTSIZE2]

    align       32
    GLOBAL_FUNCTION(jsimd_idct_float_sse)

EXTN(jsimd_idct_float_sse):
    push        ebp
    mov         eax, esp                     ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    mov         [esp], eax
    mov         ebp, esp                     ; ebp = aligned ebp
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
    lea         edi, [workspace]                 ; FAST_FLOAT *wsptr
    mov         ecx, DCTSIZE/4                   ; ctr
    ALIGNX      16, 7
.columnloop:
%ifndef NO_ZERO_COLUMN_TEST_FLOAT_SSE
    mov         eax, dword [DWBLOCK(1,0,esi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,esi,SIZEOF_JCOEF)]
    jnz         near .columnDCT

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

    punpckhwd   mm1, mm0                   ; mm1=(** 02 ** 03)
    punpcklwd   mm0, mm0                   ; mm0=(00 00 01 01)
    psrad       mm1, (DWORD_BIT-WORD_BIT)  ; mm1=in0H=(02 03)
    psrad       mm0, (DWORD_BIT-WORD_BIT)  ; mm0=in0L=(00 01)
    cvtpi2ps    xmm3, mm1                  ; xmm3=(02 03 ** **)
    cvtpi2ps    xmm0, mm0                  ; xmm0=(00 01 ** **)
    movlhps     xmm0, xmm3                 ; xmm0=in0=(00 01 02 03)

    mulps       xmm0, XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movaps      xmm1, xmm0
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0

    shufps      xmm0, xmm0, 0x00        ; xmm0=(00 00 00 00)
    shufps      xmm1, xmm1, 0x55        ; xmm1=(01 01 01 01)
    shufps      xmm2, xmm2, 0xAA        ; xmm2=(02 02 02 02)
    shufps      xmm3, xmm3, 0xFF        ; xmm3=(03 03 03 03)

    movaps      XMMWORD [XMMBLOCK(0,0,edi,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(0,1,edi,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(1,0,edi,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(1,1,edi,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(2,0,edi,SIZEOF_FAST_FLOAT)], xmm2
    movaps      XMMWORD [XMMBLOCK(2,1,edi,SIZEOF_FAST_FLOAT)], xmm2
    movaps      XMMWORD [XMMBLOCK(3,0,edi,SIZEOF_FAST_FLOAT)], xmm3
    movaps      XMMWORD [XMMBLOCK(3,1,edi,SIZEOF_FAST_FLOAT)], xmm3
    jmp         near .nextcolumn
    ALIGNX      16, 7
%endif
.columnDCT:

    ; -- Even part

    movq        mm0, MMWORD [MMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    movq        mm2, MMWORD [MMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]

    punpckhwd   mm4, mm0                ; mm4=(** 02 ** 03)
    punpcklwd   mm0, mm0                ; mm0=(00 00 01 01)
    punpckhwd   mm5, mm1                ; mm5=(** 22 ** 23)
    punpcklwd   mm1, mm1                ; mm1=(20 20 21 21)

    psrad       mm4, (DWORD_BIT-WORD_BIT)  ; mm4=in0H=(02 03)
    psrad       mm0, (DWORD_BIT-WORD_BIT)  ; mm0=in0L=(00 01)
    cvtpi2ps    xmm4, mm4                  ; xmm4=(02 03 ** **)
    cvtpi2ps    xmm0, mm0                  ; xmm0=(00 01 ** **)
    psrad       mm5, (DWORD_BIT-WORD_BIT)  ; mm5=in2H=(22 23)
    psrad       mm1, (DWORD_BIT-WORD_BIT)  ; mm1=in2L=(20 21)
    cvtpi2ps    xmm5, mm5                  ; xmm5=(22 23 ** **)
    cvtpi2ps    xmm1, mm1                  ; xmm1=(20 21 ** **)

    punpckhwd   mm6, mm2                ; mm6=(** 42 ** 43)
    punpcklwd   mm2, mm2                ; mm2=(40 40 41 41)
    punpckhwd   mm7, mm3                ; mm7=(** 62 ** 63)
    punpcklwd   mm3, mm3                ; mm3=(60 60 61 61)

    psrad       mm6, (DWORD_BIT-WORD_BIT)  ; mm6=in4H=(42 43)
    psrad       mm2, (DWORD_BIT-WORD_BIT)  ; mm2=in4L=(40 41)
    cvtpi2ps    xmm6, mm6                  ; xmm6=(42 43 ** **)
    cvtpi2ps    xmm2, mm2                  ; xmm2=(40 41 ** **)
    psrad       mm7, (DWORD_BIT-WORD_BIT)  ; mm7=in6H=(62 63)
    psrad       mm3, (DWORD_BIT-WORD_BIT)  ; mm3=in6L=(60 61)
    cvtpi2ps    xmm7, mm7                  ; xmm7=(62 63 ** **)
    cvtpi2ps    xmm3, mm3                  ; xmm3=(60 61 ** **)

    movlhps     xmm0, xmm4              ; xmm0=in0=(00 01 02 03)
    movlhps     xmm1, xmm5              ; xmm1=in2=(20 21 22 23)
    mulps       xmm0, XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm1, XMMWORD [XMMBLOCK(2,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movlhps     xmm2, xmm6              ; xmm2=in4=(40 41 42 43)
    movlhps     xmm3, xmm7              ; xmm3=in6=(60 61 62 63)
    mulps       xmm2, XMMWORD [XMMBLOCK(4,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm3, XMMWORD [XMMBLOCK(6,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movaps      xmm4, xmm0
    movaps      xmm5, xmm1
    subps       xmm0, xmm2              ; xmm0=tmp11
    subps       xmm1, xmm3
    addps       xmm4, xmm2              ; xmm4=tmp10
    addps       xmm5, xmm3              ; xmm5=tmp13

    mulps       xmm1, [GOTOFF(ebx,PD_1_414)]
    subps       xmm1, xmm5              ; xmm1=tmp12

    movaps      xmm6, xmm4
    movaps      xmm7, xmm0
    subps       xmm4, xmm5              ; xmm4=tmp3
    subps       xmm0, xmm1              ; xmm0=tmp2
    addps       xmm6, xmm5              ; xmm6=tmp0
    addps       xmm7, xmm1              ; xmm7=tmp1

    movaps      XMMWORD [wk(1)], xmm4   ; tmp3
    movaps      XMMWORD [wk(0)], xmm0   ; tmp2

    ; -- Odd part

    movq        mm4, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm0, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    movq        mm5, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]

    punpckhwd   mm6, mm4                ; mm6=(** 12 ** 13)
    punpcklwd   mm4, mm4                ; mm4=(10 10 11 11)
    punpckhwd   mm2, mm0                ; mm2=(** 32 ** 33)
    punpcklwd   mm0, mm0                ; mm0=(30 30 31 31)

    psrad       mm6, (DWORD_BIT-WORD_BIT)  ; mm6=in1H=(12 13)
    psrad       mm4, (DWORD_BIT-WORD_BIT)  ; mm4=in1L=(10 11)
    cvtpi2ps    xmm4, mm6                  ; xmm4=(12 13 ** **)
    cvtpi2ps    xmm2, mm4                  ; xmm2=(10 11 ** **)
    psrad       mm2, (DWORD_BIT-WORD_BIT)  ; mm2=in3H=(32 33)
    psrad       mm0, (DWORD_BIT-WORD_BIT)  ; mm0=in3L=(30 31)
    cvtpi2ps    xmm0, mm2                  ; xmm0=(32 33 ** **)
    cvtpi2ps    xmm3, mm0                  ; xmm3=(30 31 ** **)

    punpckhwd   mm7, mm5                ; mm7=(** 52 ** 53)
    punpcklwd   mm5, mm5                ; mm5=(50 50 51 51)
    punpckhwd   mm3, mm1                ; mm3=(** 72 ** 73)
    punpcklwd   mm1, mm1                ; mm1=(70 70 71 71)

    movlhps     xmm2, xmm4              ; xmm2=in1=(10 11 12 13)
    movlhps     xmm3, xmm0              ; xmm3=in3=(30 31 32 33)

    psrad       mm7, (DWORD_BIT-WORD_BIT)  ; mm7=in5H=(52 53)
    psrad       mm5, (DWORD_BIT-WORD_BIT)  ; mm5=in5L=(50 51)
    cvtpi2ps    xmm4, mm7                  ; xmm4=(52 53 ** **)
    cvtpi2ps    xmm5, mm5                  ; xmm5=(50 51 ** **)
    psrad       mm3, (DWORD_BIT-WORD_BIT)  ; mm3=in7H=(72 73)
    psrad       mm1, (DWORD_BIT-WORD_BIT)  ; mm1=in7L=(70 71)
    cvtpi2ps    xmm0, mm3                  ; xmm0=(72 73 ** **)
    cvtpi2ps    xmm1, mm1                  ; xmm1=(70 71 ** **)

    mulps       xmm2, XMMWORD [XMMBLOCK(1,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm3, XMMWORD [XMMBLOCK(3,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movlhps     xmm5, xmm4              ; xmm5=in5=(50 51 52 53)
    movlhps     xmm1, xmm0              ; xmm1=in7=(70 71 72 73)
    mulps       xmm5, XMMWORD [XMMBLOCK(5,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm1, XMMWORD [XMMBLOCK(7,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movaps      xmm4, xmm2
    movaps      xmm0, xmm5
    addps       xmm2, xmm1              ; xmm2=z11
    addps       xmm5, xmm3              ; xmm5=z13
    subps       xmm4, xmm1              ; xmm4=z12
    subps       xmm0, xmm3              ; xmm0=z10

    movaps      xmm1, xmm2
    subps       xmm2, xmm5
    addps       xmm1, xmm5              ; xmm1=tmp7

    mulps       xmm2, [GOTOFF(ebx,PD_1_414)]  ; xmm2=tmp11

    movaps      xmm3, xmm0
    addps       xmm0, xmm4
    mulps       xmm0, [GOTOFF(ebx,PD_1_847)]   ; xmm0=z5
    mulps       xmm3, [GOTOFF(ebx,PD_M2_613)]  ; xmm3=(z10 * -2.613125930)
    mulps       xmm4, [GOTOFF(ebx,PD_1_082)]   ; xmm4=(z12 * 1.082392200)
    addps       xmm3, xmm0                     ; xmm3=tmp12
    subps       xmm4, xmm0                     ; xmm4=tmp10

    ; -- Final output stage

    subps       xmm3, xmm1              ; xmm3=tmp6
    movaps      xmm5, xmm6
    movaps      xmm0, xmm7
    addps       xmm6, xmm1              ; xmm6=data0=(00 01 02 03)
    addps       xmm7, xmm3              ; xmm7=data1=(10 11 12 13)
    subps       xmm5, xmm1              ; xmm5=data7=(70 71 72 73)
    subps       xmm0, xmm3              ; xmm0=data6=(60 61 62 63)
    subps       xmm2, xmm3              ; xmm2=tmp5

    movaps      xmm1, xmm6              ; transpose coefficients(phase 1)
    unpcklps    xmm6, xmm7              ; xmm6=(00 10 01 11)
    unpckhps    xmm1, xmm7              ; xmm1=(02 12 03 13)
    movaps      xmm3, xmm0              ; transpose coefficients(phase 1)
    unpcklps    xmm0, xmm5              ; xmm0=(60 70 61 71)
    unpckhps    xmm3, xmm5              ; xmm3=(62 72 63 73)

    movaps      xmm7, XMMWORD [wk(0)]   ; xmm7=tmp2
    movaps      xmm5, XMMWORD [wk(1)]   ; xmm5=tmp3

    movaps      XMMWORD [wk(0)], xmm0   ; wk(0)=(60 70 61 71)
    movaps      XMMWORD [wk(1)], xmm3   ; wk(1)=(62 72 63 73)

    addps       xmm4, xmm2              ; xmm4=tmp4
    movaps      xmm0, xmm7
    movaps      xmm3, xmm5
    addps       xmm7, xmm2              ; xmm7=data2=(20 21 22 23)
    addps       xmm5, xmm4              ; xmm5=data4=(40 41 42 43)
    subps       xmm0, xmm2              ; xmm0=data5=(50 51 52 53)
    subps       xmm3, xmm4              ; xmm3=data3=(30 31 32 33)

    movaps      xmm2, xmm7              ; transpose coefficients(phase 1)
    unpcklps    xmm7, xmm3              ; xmm7=(20 30 21 31)
    unpckhps    xmm2, xmm3              ; xmm2=(22 32 23 33)
    movaps      xmm4, xmm5              ; transpose coefficients(phase 1)
    unpcklps    xmm5, xmm0              ; xmm5=(40 50 41 51)
    unpckhps    xmm4, xmm0              ; xmm4=(42 52 43 53)

    movaps      xmm3, xmm6              ; transpose coefficients(phase 2)
    UNPCKLPS2   xmm6, xmm7              ; xmm6=(00 10 20 30)
    UNPCKHPS2   xmm3, xmm7              ; xmm3=(01 11 21 31)
    movaps      xmm0, xmm1              ; transpose coefficients(phase 2)
    UNPCKLPS2   xmm1, xmm2              ; xmm1=(02 12 22 32)
    UNPCKHPS2   xmm0, xmm2              ; xmm0=(03 13 23 33)

    movaps      xmm7, XMMWORD [wk(0)]   ; xmm7=(60 70 61 71)
    movaps      xmm2, XMMWORD [wk(1)]   ; xmm2=(62 72 63 73)

    movaps      XMMWORD [XMMBLOCK(0,0,edi,SIZEOF_FAST_FLOAT)], xmm6
    movaps      XMMWORD [XMMBLOCK(1,0,edi,SIZEOF_FAST_FLOAT)], xmm3
    movaps      XMMWORD [XMMBLOCK(2,0,edi,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(3,0,edi,SIZEOF_FAST_FLOAT)], xmm0

    movaps      xmm6, xmm5              ; transpose coefficients(phase 2)
    UNPCKLPS2   xmm5, xmm7              ; xmm5=(40 50 60 70)
    UNPCKHPS2   xmm6, xmm7              ; xmm6=(41 51 61 71)
    movaps      xmm3, xmm4              ; transpose coefficients(phase 2)
    UNPCKLPS2   xmm4, xmm2              ; xmm4=(42 52 62 72)
    UNPCKHPS2   xmm3, xmm2              ; xmm3=(43 53 63 73)

    movaps      XMMWORD [XMMBLOCK(0,1,edi,SIZEOF_FAST_FLOAT)], xmm5
    movaps      XMMWORD [XMMBLOCK(1,1,edi,SIZEOF_FAST_FLOAT)], xmm6
    movaps      XMMWORD [XMMBLOCK(2,1,edi,SIZEOF_FAST_FLOAT)], xmm4
    movaps      XMMWORD [XMMBLOCK(3,1,edi,SIZEOF_FAST_FLOAT)], xmm3

.nextcolumn:
    add         esi, byte 4*SIZEOF_JCOEF               ; coef_block
    add         edx, byte 4*SIZEOF_FLOAT_MULT_TYPE     ; quantptr
    add         edi,      4*DCTSIZE*SIZEOF_FAST_FLOAT  ; wsptr
    dec         ecx                                    ; ctr
    jnz         near .columnloop

    ; -- Prefetch the next coefficient block

    prefetchnta [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 0*32]
    prefetchnta [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 1*32]
    prefetchnta [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 2*32]
    prefetchnta [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 3*32]

    ; ---- Pass 2: process rows from work array, store into output array.

    mov         eax, [original_ebp]
    lea         esi, [workspace]                   ; FAST_FLOAT *wsptr
    mov         edi, JSAMPARRAY [output_buf(eax)]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [output_col(eax)]
    mov         ecx, DCTSIZE/4                     ; ctr
    ALIGNX      16, 7
.rowloop:

    ; -- Even part

    movaps      xmm0, XMMWORD [XMMBLOCK(0,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(2,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm2, XMMWORD [XMMBLOCK(4,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(6,0,esi,SIZEOF_FAST_FLOAT)]

    movaps      xmm4, xmm0
    movaps      xmm5, xmm1
    subps       xmm0, xmm2              ; xmm0=tmp11
    subps       xmm1, xmm3
    addps       xmm4, xmm2              ; xmm4=tmp10
    addps       xmm5, xmm3              ; xmm5=tmp13

    mulps       xmm1, [GOTOFF(ebx,PD_1_414)]
    subps       xmm1, xmm5              ; xmm1=tmp12

    movaps      xmm6, xmm4
    movaps      xmm7, xmm0
    subps       xmm4, xmm5              ; xmm4=tmp3
    subps       xmm0, xmm1              ; xmm0=tmp2
    addps       xmm6, xmm5              ; xmm6=tmp0
    addps       xmm7, xmm1              ; xmm7=tmp1

    movaps      XMMWORD [wk(1)], xmm4   ; tmp3
    movaps      XMMWORD [wk(0)], xmm0   ; tmp2

    ; -- Odd part

    movaps      xmm2, XMMWORD [XMMBLOCK(1,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(3,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm5, XMMWORD [XMMBLOCK(5,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(7,0,esi,SIZEOF_FAST_FLOAT)]

    movaps      xmm4, xmm2
    movaps      xmm0, xmm5
    addps       xmm2, xmm1              ; xmm2=z11
    addps       xmm5, xmm3              ; xmm5=z13
    subps       xmm4, xmm1              ; xmm4=z12
    subps       xmm0, xmm3              ; xmm0=z10

    movaps      xmm1, xmm2
    subps       xmm2, xmm5
    addps       xmm1, xmm5              ; xmm1=tmp7

    mulps       xmm2, [GOTOFF(ebx,PD_1_414)]  ; xmm2=tmp11

    movaps      xmm3, xmm0
    addps       xmm0, xmm4
    mulps       xmm0, [GOTOFF(ebx,PD_1_847)]   ; xmm0=z5
    mulps       xmm3, [GOTOFF(ebx,PD_M2_613)]  ; xmm3=(z10 * -2.613125930)
    mulps       xmm4, [GOTOFF(ebx,PD_1_082)]   ; xmm4=(z12 * 1.082392200)
    addps       xmm3, xmm0                     ; xmm3=tmp12
    subps       xmm4, xmm0                     ; xmm4=tmp10

    ; -- Final output stage

    subps       xmm3, xmm1              ; xmm3=tmp6
    movaps      xmm5, xmm6
    movaps      xmm0, xmm7
    addps       xmm6, xmm1              ; xmm6=data0=(00 10 20 30)
    addps       xmm7, xmm3              ; xmm7=data1=(01 11 21 31)
    subps       xmm5, xmm1              ; xmm5=data7=(07 17 27 37)
    subps       xmm0, xmm3              ; xmm0=data6=(06 16 26 36)
    subps       xmm2, xmm3              ; xmm2=tmp5

    movaps      xmm1, [GOTOFF(ebx,PD_0_125)]  ; xmm1=[PD_0_125]

    mulps       xmm6, xmm1              ; descale(1/8)
    mulps       xmm7, xmm1              ; descale(1/8)
    mulps       xmm5, xmm1              ; descale(1/8)
    mulps       xmm0, xmm1              ; descale(1/8)

    movhlps     xmm3, xmm6
    movhlps     xmm1, xmm7
    cvtps2pi    mm0, xmm6               ; round to int32, mm0=data0L=(00 10)
    cvtps2pi    mm1, xmm7               ; round to int32, mm1=data1L=(01 11)
    cvtps2pi    mm2, xmm3               ; round to int32, mm2=data0H=(20 30)
    cvtps2pi    mm3, xmm1               ; round to int32, mm3=data1H=(21 31)
    packssdw    mm0, mm2                ; mm0=data0=(00 10 20 30)
    packssdw    mm1, mm3                ; mm1=data1=(01 11 21 31)

    movhlps     xmm6, xmm5
    movhlps     xmm7, xmm0
    cvtps2pi    mm4, xmm5               ; round to int32, mm4=data7L=(07 17)
    cvtps2pi    mm5, xmm0               ; round to int32, mm5=data6L=(06 16)
    cvtps2pi    mm6, xmm6               ; round to int32, mm6=data7H=(27 37)
    cvtps2pi    mm7, xmm7               ; round to int32, mm7=data6H=(26 36)
    packssdw    mm4, mm6                ; mm4=data7=(07 17 27 37)
    packssdw    mm5, mm7                ; mm5=data6=(06 16 26 36)

    packsswb    mm0, mm5                ; mm0=(00 10 20 30 06 16 26 36)
    packsswb    mm1, mm4                ; mm1=(01 11 21 31 07 17 27 37)

    movaps      xmm3, XMMWORD [wk(0)]   ; xmm3=tmp2
    movaps      xmm1, XMMWORD [wk(1)]   ; xmm1=tmp3

    movaps      xmm6, [GOTOFF(ebx,PD_0_125)]  ; xmm6=[PD_0_125]

    addps       xmm4, xmm2              ; xmm4=tmp4
    movaps      xmm5, xmm3
    movaps      xmm0, xmm1
    addps       xmm3, xmm2              ; xmm3=data2=(02 12 22 32)
    addps       xmm1, xmm4              ; xmm1=data4=(04 14 24 34)
    subps       xmm5, xmm2              ; xmm5=data5=(05 15 25 35)
    subps       xmm0, xmm4              ; xmm0=data3=(03 13 23 33)

    mulps       xmm3, xmm6              ; descale(1/8)
    mulps       xmm1, xmm6              ; descale(1/8)
    mulps       xmm5, xmm6              ; descale(1/8)
    mulps       xmm0, xmm6              ; descale(1/8)

    movhlps     xmm7, xmm3
    movhlps     xmm2, xmm1
    cvtps2pi    mm2, xmm3               ; round to int32, mm2=data2L=(02 12)
    cvtps2pi    mm3, xmm1               ; round to int32, mm3=data4L=(04 14)
    cvtps2pi    mm6, xmm7               ; round to int32, mm6=data2H=(22 32)
    cvtps2pi    mm7, xmm2               ; round to int32, mm7=data4H=(24 34)
    packssdw    mm2, mm6                ; mm2=data2=(02 12 22 32)
    packssdw    mm3, mm7                ; mm3=data4=(04 14 24 34)

    movhlps     xmm4, xmm5
    movhlps     xmm6, xmm0
    cvtps2pi    mm5, xmm5               ; round to int32, mm5=data5L=(05 15)
    cvtps2pi    mm4, xmm0               ; round to int32, mm4=data3L=(03 13)
    cvtps2pi    mm6, xmm4               ; round to int32, mm6=data5H=(25 35)
    cvtps2pi    mm7, xmm6               ; round to int32, mm7=data3H=(23 33)
    packssdw    mm5, mm6                ; mm5=data5=(05 15 25 35)
    packssdw    mm4, mm7                ; mm4=data3=(03 13 23 33)

    movq        mm6, [GOTOFF(ebx,PB_CENTERJSAMP)]  ; mm6=[PB_CENTERJSAMP]

    packsswb    mm2, mm3                ; mm2=(02 12 22 32 04 14 24 34)
    packsswb    mm4, mm5                ; mm4=(03 13 23 33 05 15 25 35)

    paddb       mm0, mm6
    paddb       mm1, mm6
    paddb       mm2, mm6
    paddb       mm4, mm6

    movq        mm7, mm0                ; transpose coefficients(phase 1)
    punpcklbw   mm0, mm1                ; mm0=(00 01 10 11 20 21 30 31)
    punpckhbw   mm7, mm1                ; mm7=(06 07 16 17 26 27 36 37)
    movq        mm3, mm2                ; transpose coefficients(phase 1)
    punpcklbw   mm2, mm4                ; mm2=(02 03 12 13 22 23 32 33)
    punpckhbw   mm3, mm4                ; mm3=(04 05 14 15 24 25 34 35)

    movq        mm5, mm0                ; transpose coefficients(phase 2)
    punpcklwd   mm0, mm2                ; mm0=(00 01 02 03 10 11 12 13)
    punpckhwd   mm5, mm2                ; mm5=(20 21 22 23 30 31 32 33)
    movq        mm6, mm3                ; transpose coefficients(phase 2)
    punpcklwd   mm3, mm7                ; mm3=(04 05 06 07 14 15 16 17)
    punpckhwd   mm6, mm7                ; mm6=(24 25 26 27 34 35 36 37)

    movq        mm1, mm0                ; transpose coefficients(phase 3)
    punpckldq   mm0, mm3                ; mm0=(00 01 02 03 04 05 06 07)
    punpckhdq   mm1, mm3                ; mm1=(10 11 12 13 14 15 16 17)
    movq        mm4, mm5                ; transpose coefficients(phase 3)
    punpckldq   mm5, mm6                ; mm5=(20 21 22 23 24 25 26 27)
    punpckhdq   mm4, mm6                ; mm4=(30 31 32 33 34 35 36 37)

    PUSHPIC     ebx                     ; save GOT address

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]
    mov         ebx, JSAMPROW [edi+1*SIZEOF_JSAMPROW]
    movq        MMWORD [edx+eax*SIZEOF_JSAMPLE], mm0
    movq        MMWORD [ebx+eax*SIZEOF_JSAMPLE], mm1
    mov         edx, JSAMPROW [edi+2*SIZEOF_JSAMPROW]
    mov         ebx, JSAMPROW [edi+3*SIZEOF_JSAMPROW]
    movq        MMWORD [edx+eax*SIZEOF_JSAMPLE], mm5
    movq        MMWORD [ebx+eax*SIZEOF_JSAMPLE], mm4

    POPPIC      ebx                     ; restore GOT address

    add         esi, byte 4*SIZEOF_FAST_FLOAT  ; wsptr
    add         edi, byte 4*SIZEOF_JSAMPROW
    dec         ecx                            ; ctr
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
