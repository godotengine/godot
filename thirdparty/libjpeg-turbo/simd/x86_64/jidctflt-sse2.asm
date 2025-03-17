;
; jidctflt.asm - floating-point IDCT (64-bit SSE & SSE2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2024, D. R. Commander.
; Copyright (C) 2018, Matthias Räncker.
; Copyright (C) 2023, Aliaksiej Kandracienka.
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
    GLOBAL_DATA(jconst_idct_float_sse2)

EXTN(jconst_idct_float_sse2):

PD_1_414        times 4  dd  1.414213562373095048801689
PD_1_847        times 4  dd  1.847759065022573512256366
PD_1_082        times 4  dd  1.082392200292393968799446
PD_M2_613       times 4  dd -2.613125929752753055713286
PD_RNDINT_MAGIC times 4  dd  100663296.0  ; (float)(0x00C00000 << 3)
PB_CENTERJSAMP  times 16 db  CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64
;
; Perform dequantization and inverse DCT on one block of coefficients.
;
; GLOBAL(void)
; jsimd_idct_float_sse2(void *dct_table, JCOEFPTR coef_block,
;                       JSAMPARRAY output_buf, JDIMENSION output_col)
;

; r10 = void *dct_table
; r11 = JCOEFPTR coef_block
; r12 = JSAMPARRAY output_buf
; r13d = JDIMENSION output_col

%define wk(i)         r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD
                                        ; xmmword wk[WK_NUM]
%define WK_NUM        2
%define workspace     wk(0) - DCTSIZE2 * SIZEOF_FAST_FLOAT
                                        ; FAST_FLOAT workspace[DCTSIZE2]

    align       32
    GLOBAL_FUNCTION(jsimd_idct_float_sse2)

EXTN(jsimd_idct_float_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    push        r15
    and         rsp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    ; Allocate stack space for wk array.  r15 is used to access it.
    mov         r15, rsp
    lea         rsp, [workspace]
    COLLECT_ARGS 4
    push        rbx

    ; ---- Pass 1: process columns from input, store into work array.

    mov         rdx, r10                ; quantptr
    mov         rsi, r11                ; inptr
    lea         rdi, [workspace]        ; FAST_FLOAT *wsptr
    mov         rcx, DCTSIZE/4          ; ctr
.columnloop:
%ifndef NO_ZERO_COLUMN_TEST_FLOAT_SSE
    mov         eax, dword [DWBLOCK(1,0,rsi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,rsi,SIZEOF_JCOEF)]
    jnz         near .columnDCT

    movq        xmm1, XMM_MMWORD [MMBLOCK(1,0,rsi,SIZEOF_JCOEF)]
    movq        xmm2, XMM_MMWORD [MMBLOCK(2,0,rsi,SIZEOF_JCOEF)]
    movq        xmm3, XMM_MMWORD [MMBLOCK(3,0,rsi,SIZEOF_JCOEF)]
    movq        xmm4, XMM_MMWORD [MMBLOCK(4,0,rsi,SIZEOF_JCOEF)]
    movq        xmm5, XMM_MMWORD [MMBLOCK(5,0,rsi,SIZEOF_JCOEF)]
    movq        xmm6, XMM_MMWORD [MMBLOCK(6,0,rsi,SIZEOF_JCOEF)]
    movq        xmm7, XMM_MMWORD [MMBLOCK(7,0,rsi,SIZEOF_JCOEF)]
    por         xmm1, xmm2
    por         xmm3, xmm4
    por         xmm5, xmm6
    por         xmm1, xmm3
    por         xmm5, xmm7
    por         xmm1, xmm5
    packsswb    xmm1, xmm1
    movd        eax, xmm1
    test        rax, rax
    jnz         short .columnDCT

    ; -- AC terms all zero

    movq        xmm0, XMM_MMWORD [MMBLOCK(0,0,rsi,SIZEOF_JCOEF)]

    punpcklwd   xmm0, xmm0                  ; xmm0=(00 00 01 01 02 02 03 03)
    psrad       xmm0, (DWORD_BIT-WORD_BIT)  ; xmm0=in0=(00 01 02 03)
    cvtdq2ps    xmm0, xmm0                  ; xmm0=in0=(00 01 02 03)

    mulps       xmm0, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]

    movaps      xmm1, xmm0
    movaps      xmm2, xmm0
    movaps      xmm3, xmm0

    shufps      xmm0, xmm0, 0x00        ; xmm0=(00 00 00 00)
    shufps      xmm1, xmm1, 0x55        ; xmm1=(01 01 01 01)
    shufps      xmm2, xmm2, 0xAA        ; xmm2=(02 02 02 02)
    shufps      xmm3, xmm3, 0xFF        ; xmm3=(03 03 03 03)

    movaps      XMMWORD [XMMBLOCK(0,0,rdi,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(0,1,rdi,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(1,0,rdi,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(1,1,rdi,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(2,0,rdi,SIZEOF_FAST_FLOAT)], xmm2
    movaps      XMMWORD [XMMBLOCK(2,1,rdi,SIZEOF_FAST_FLOAT)], xmm2
    movaps      XMMWORD [XMMBLOCK(3,0,rdi,SIZEOF_FAST_FLOAT)], xmm3
    movaps      XMMWORD [XMMBLOCK(3,1,rdi,SIZEOF_FAST_FLOAT)], xmm3
    jmp         near .nextcolumn
%endif
.columnDCT:

    ; -- Even part

    movq        xmm0, XMM_MMWORD [MMBLOCK(0,0,rsi,SIZEOF_JCOEF)]
    movq        xmm1, XMM_MMWORD [MMBLOCK(2,0,rsi,SIZEOF_JCOEF)]
    movq        xmm2, XMM_MMWORD [MMBLOCK(4,0,rsi,SIZEOF_JCOEF)]
    movq        xmm3, XMM_MMWORD [MMBLOCK(6,0,rsi,SIZEOF_JCOEF)]

    punpcklwd   xmm0, xmm0                  ; xmm0=(00 00 01 01 02 02 03 03)
    punpcklwd   xmm1, xmm1                  ; xmm1=(20 20 21 21 22 22 23 23)
    psrad       xmm0, (DWORD_BIT-WORD_BIT)  ; xmm0=in0=(00 01 02 03)
    psrad       xmm1, (DWORD_BIT-WORD_BIT)  ; xmm1=in2=(20 21 22 23)
    cvtdq2ps    xmm0, xmm0                  ; xmm0=in0=(00 01 02 03)
    cvtdq2ps    xmm1, xmm1                  ; xmm1=in2=(20 21 22 23)

    punpcklwd   xmm2, xmm2                  ; xmm2=(40 40 41 41 42 42 43 43)
    punpcklwd   xmm3, xmm3                  ; xmm3=(60 60 61 61 62 62 63 63)
    psrad       xmm2, (DWORD_BIT-WORD_BIT)  ; xmm2=in4=(40 41 42 43)
    psrad       xmm3, (DWORD_BIT-WORD_BIT)  ; xmm3=in6=(60 61 62 63)
    cvtdq2ps    xmm2, xmm2                  ; xmm2=in4=(40 41 42 43)
    cvtdq2ps    xmm3, xmm3                  ; xmm3=in6=(60 61 62 63)

    mulps       xmm0, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm1, XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm2, XMMWORD [XMMBLOCK(4,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm3, XMMWORD [XMMBLOCK(6,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]

    movaps      xmm4, xmm0
    movaps      xmm5, xmm1
    subps       xmm0, xmm2              ; xmm0=tmp11
    subps       xmm1, xmm3
    addps       xmm4, xmm2              ; xmm4=tmp10
    addps       xmm5, xmm3              ; xmm5=tmp13

    mulps       xmm1, [rel PD_1_414]
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

    movq        xmm2, XMM_MMWORD [MMBLOCK(1,0,rsi,SIZEOF_JCOEF)]
    movq        xmm3, XMM_MMWORD [MMBLOCK(3,0,rsi,SIZEOF_JCOEF)]
    movq        xmm5, XMM_MMWORD [MMBLOCK(5,0,rsi,SIZEOF_JCOEF)]
    movq        xmm1, XMM_MMWORD [MMBLOCK(7,0,rsi,SIZEOF_JCOEF)]

    punpcklwd   xmm2, xmm2                  ; xmm2=(10 10 11 11 12 12 13 13)
    punpcklwd   xmm3, xmm3                  ; xmm3=(30 30 31 31 32 32 33 33)
    psrad       xmm2, (DWORD_BIT-WORD_BIT)  ; xmm2=in1=(10 11 12 13)
    psrad       xmm3, (DWORD_BIT-WORD_BIT)  ; xmm3=in3=(30 31 32 33)
    cvtdq2ps    xmm2, xmm2                  ; xmm2=in1=(10 11 12 13)
    cvtdq2ps    xmm3, xmm3                  ; xmm3=in3=(30 31 32 33)

    punpcklwd   xmm5, xmm5                  ; xmm5=(50 50 51 51 52 52 53 53)
    punpcklwd   xmm1, xmm1                  ; xmm1=(70 70 71 71 72 72 73 73)
    psrad       xmm5, (DWORD_BIT-WORD_BIT)  ; xmm5=in5=(50 51 52 53)
    psrad       xmm1, (DWORD_BIT-WORD_BIT)  ; xmm1=in7=(70 71 72 73)
    cvtdq2ps    xmm5, xmm5                  ; xmm5=in5=(50 51 52 53)
    cvtdq2ps    xmm1, xmm1                  ; xmm1=in7=(70 71 72 73)

    mulps       xmm2, XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm3, XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm5, XMMWORD [XMMBLOCK(5,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]
    mulps       xmm1, XMMWORD [XMMBLOCK(7,0,rdx,SIZEOF_FLOAT_MULT_TYPE)]

    movaps      xmm4, xmm2
    movaps      xmm0, xmm5
    addps       xmm2, xmm1              ; xmm2=z11
    addps       xmm5, xmm3              ; xmm5=z13
    subps       xmm4, xmm1              ; xmm4=z12
    subps       xmm0, xmm3              ; xmm0=z10

    movaps      xmm1, xmm2
    subps       xmm2, xmm5
    addps       xmm1, xmm5              ; xmm1=tmp7

    mulps       xmm2, [rel PD_1_414]    ; xmm2=tmp11

    movaps      xmm3, xmm0
    addps       xmm0, xmm4
    mulps       xmm0, [rel PD_1_847]    ; xmm0=z5
    mulps       xmm3, [rel PD_M2_613]   ; xmm3=(z10 * -2.613125930)
    mulps       xmm4, [rel PD_1_082]    ; xmm4=(z12 * 1.082392200)
    addps       xmm3, xmm0              ; xmm3=tmp12
    subps       xmm4, xmm0              ; xmm4=tmp10

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

    movaps      XMMWORD [XMMBLOCK(0,0,rdi,SIZEOF_FAST_FLOAT)], xmm6
    movaps      XMMWORD [XMMBLOCK(1,0,rdi,SIZEOF_FAST_FLOAT)], xmm3
    movaps      XMMWORD [XMMBLOCK(2,0,rdi,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(3,0,rdi,SIZEOF_FAST_FLOAT)], xmm0

    movaps      xmm6, xmm5              ; transpose coefficients(phase 2)
    UNPCKLPS2   xmm5, xmm7              ; xmm5=(40 50 60 70)
    UNPCKHPS2   xmm6, xmm7              ; xmm6=(41 51 61 71)
    movaps      xmm3, xmm4              ; transpose coefficients(phase 2)
    UNPCKLPS2   xmm4, xmm2              ; xmm4=(42 52 62 72)
    UNPCKHPS2   xmm3, xmm2              ; xmm3=(43 53 63 73)

    movaps      XMMWORD [XMMBLOCK(0,1,rdi,SIZEOF_FAST_FLOAT)], xmm5
    movaps      XMMWORD [XMMBLOCK(1,1,rdi,SIZEOF_FAST_FLOAT)], xmm6
    movaps      XMMWORD [XMMBLOCK(2,1,rdi,SIZEOF_FAST_FLOAT)], xmm4
    movaps      XMMWORD [XMMBLOCK(3,1,rdi,SIZEOF_FAST_FLOAT)], xmm3

.nextcolumn:
    add         rsi, byte 4*SIZEOF_JCOEF               ; coef_block
    add         rdx, byte 4*SIZEOF_FLOAT_MULT_TYPE     ; quantptr
    add         rdi,      4*DCTSIZE*SIZEOF_FAST_FLOAT  ; wsptr
    dec         rcx                                    ; ctr
    jnz         near .columnloop

    ; -- Prefetch the next coefficient block

    prefetchnta [rsi + (DCTSIZE2-8)*SIZEOF_JCOEF + 0*32]
    prefetchnta [rsi + (DCTSIZE2-8)*SIZEOF_JCOEF + 1*32]
    prefetchnta [rsi + (DCTSIZE2-8)*SIZEOF_JCOEF + 2*32]
    prefetchnta [rsi + (DCTSIZE2-8)*SIZEOF_JCOEF + 3*32]

    ; ---- Pass 2: process rows from work array, store into output array.

    lea         rsi, [workspace]        ; FAST_FLOAT *wsptr
    mov         rdi, r12                ; (JSAMPROW *)
    mov         eax, r13d
    mov         rcx, DCTSIZE/4          ; ctr
.rowloop:

    ; -- Even part

    movaps      xmm0, XMMWORD [XMMBLOCK(0,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(2,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm2, XMMWORD [XMMBLOCK(4,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(6,0,rsi,SIZEOF_FAST_FLOAT)]

    movaps      xmm4, xmm0
    movaps      xmm5, xmm1
    subps       xmm0, xmm2              ; xmm0=tmp11
    subps       xmm1, xmm3
    addps       xmm4, xmm2              ; xmm4=tmp10
    addps       xmm5, xmm3              ; xmm5=tmp13

    mulps       xmm1, [rel PD_1_414]
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

    movaps      xmm2, XMMWORD [XMMBLOCK(1,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(3,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm5, XMMWORD [XMMBLOCK(5,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(7,0,rsi,SIZEOF_FAST_FLOAT)]

    movaps      xmm4, xmm2
    movaps      xmm0, xmm5
    addps       xmm2, xmm1              ; xmm2=z11
    addps       xmm5, xmm3              ; xmm5=z13
    subps       xmm4, xmm1              ; xmm4=z12
    subps       xmm0, xmm3              ; xmm0=z10

    movaps      xmm1, xmm2
    subps       xmm2, xmm5
    addps       xmm1, xmm5              ; xmm1=tmp7

    mulps       xmm2, [rel PD_1_414]    ; xmm2=tmp11

    movaps      xmm3, xmm0
    addps       xmm0, xmm4
    mulps       xmm0, [rel PD_1_847]    ; xmm0=z5
    mulps       xmm3, [rel PD_M2_613]   ; xmm3=(z10 * -2.613125930)
    mulps       xmm4, [rel PD_1_082]    ; xmm4=(z12 * 1.082392200)
    addps       xmm3, xmm0              ; xmm3=tmp12
    subps       xmm4, xmm0              ; xmm4=tmp10

    ; -- Final output stage

    subps       xmm3, xmm1              ; xmm3=tmp6
    movaps      xmm5, xmm6
    movaps      xmm0, xmm7
    addps       xmm6, xmm1              ; xmm6=data0=(00 10 20 30)
    addps       xmm7, xmm3              ; xmm7=data1=(01 11 21 31)
    subps       xmm5, xmm1              ; xmm5=data7=(07 17 27 37)
    subps       xmm0, xmm3              ; xmm0=data6=(06 16 26 36)
    subps       xmm2, xmm3              ; xmm2=tmp5

    movaps      xmm1, [rel PD_RNDINT_MAGIC]  ; xmm1=[rel PD_RNDINT_MAGIC]
    pcmpeqd     xmm3, xmm3
    psrld       xmm3, WORD_BIT          ; xmm3={0xFFFF 0x0000 0xFFFF 0x0000 ..}

    addps       xmm6, xmm1              ; xmm6=roundint(data0/8)=(00 ** 10 ** 20 ** 30 **)
    addps       xmm7, xmm1              ; xmm7=roundint(data1/8)=(01 ** 11 ** 21 ** 31 **)
    addps       xmm0, xmm1              ; xmm0=roundint(data6/8)=(06 ** 16 ** 26 ** 36 **)
    addps       xmm5, xmm1              ; xmm5=roundint(data7/8)=(07 ** 17 ** 27 ** 37 **)

    pand        xmm6, xmm3              ; xmm6=(00 -- 10 -- 20 -- 30 --)
    pslld       xmm7, WORD_BIT          ; xmm7=(-- 01 -- 11 -- 21 -- 31)
    pand        xmm0, xmm3              ; xmm0=(06 -- 16 -- 26 -- 36 --)
    pslld       xmm5, WORD_BIT          ; xmm5=(-- 07 -- 17 -- 27 -- 37)
    por         xmm6, xmm7              ; xmm6=(00 01 10 11 20 21 30 31)
    por         xmm0, xmm5              ; xmm0=(06 07 16 17 26 27 36 37)

    movaps      xmm1,  XMMWORD [wk(0)]  ; xmm1=tmp2
    movaps      xmm3,  XMMWORD [wk(1)]  ; xmm3=tmp3

    addps       xmm4, xmm2              ; xmm4=tmp4
    movaps      xmm7, xmm1
    movaps      xmm5, xmm3
    addps       xmm1, xmm2              ; xmm1=data2=(02 12 22 32)
    addps       xmm3, xmm4              ; xmm3=data4=(04 14 24 34)
    subps       xmm7, xmm2              ; xmm7=data5=(05 15 25 35)
    subps       xmm5, xmm4              ; xmm5=data3=(03 13 23 33)

    movaps      xmm2, [rel PD_RNDINT_MAGIC]  ; xmm2=[rel PD_RNDINT_MAGIC]
    pcmpeqd     xmm4, xmm4
    psrld       xmm4, WORD_BIT          ; xmm4={0xFFFF 0x0000 0xFFFF 0x0000 ..}

    addps       xmm3, xmm2              ; xmm3=roundint(data4/8)=(04 ** 14 ** 24 ** 34 **)
    addps       xmm7, xmm2              ; xmm7=roundint(data5/8)=(05 ** 15 ** 25 ** 35 **)
    addps       xmm1, xmm2              ; xmm1=roundint(data2/8)=(02 ** 12 ** 22 ** 32 **)
    addps       xmm5, xmm2              ; xmm5=roundint(data3/8)=(03 ** 13 ** 23 ** 33 **)

    pand        xmm3, xmm4              ; xmm3=(04 -- 14 -- 24 -- 34 --)
    pslld       xmm7, WORD_BIT          ; xmm7=(-- 05 -- 15 -- 25 -- 35)
    pand        xmm1, xmm4              ; xmm1=(02 -- 12 -- 22 -- 32 --)
    pslld       xmm5, WORD_BIT          ; xmm5=(-- 03 -- 13 -- 23 -- 33)
    por         xmm3, xmm7              ; xmm3=(04 05 14 15 24 25 34 35)
    por         xmm1, xmm5              ; xmm1=(02 03 12 13 22 23 32 33)

    movdqa      xmm2, [rel PB_CENTERJSAMP]  ; xmm2=[rel PB_CENTERJSAMP]

    packsswb    xmm6, xmm3        ; xmm6=(00 01 10 11 20 21 30 31 04 05 14 15 24 25 34 35)
    packsswb    xmm1, xmm0        ; xmm1=(02 03 12 13 22 23 32 33 06 07 16 17 26 27 36 37)
    paddb       xmm6, xmm2
    paddb       xmm1, xmm2

    movdqa      xmm4, xmm6        ; transpose coefficients(phase 2)
    punpcklwd   xmm6, xmm1        ; xmm6=(00 01 02 03 10 11 12 13 20 21 22 23 30 31 32 33)
    punpckhwd   xmm4, xmm1        ; xmm4=(04 05 06 07 14 15 16 17 24 25 26 27 34 35 36 37)

    movdqa      xmm7, xmm6        ; transpose coefficients(phase 3)
    punpckldq   xmm6, xmm4        ; xmm6=(00 01 02 03 04 05 06 07 10 11 12 13 14 15 16 17)
    punpckhdq   xmm7, xmm4        ; xmm7=(20 21 22 23 24 25 26 27 30 31 32 33 34 35 36 37)

    pshufd      xmm5, xmm6, 0x4E  ; xmm5=(10 11 12 13 14 15 16 17 00 01 02 03 04 05 06 07)
    pshufd      xmm3, xmm7, 0x4E  ; xmm3=(30 31 32 33 34 35 36 37 20 21 22 23 24 25 26 27)

    mov         rdxp, JSAMPROW [rdi+0*SIZEOF_JSAMPROW]
    mov         rbxp, JSAMPROW [rdi+2*SIZEOF_JSAMPROW]
    movq        XMM_MMWORD [rdx+rax*SIZEOF_JSAMPLE], xmm6
    movq        XMM_MMWORD [rbx+rax*SIZEOF_JSAMPLE], xmm7
    mov         rdxp, JSAMPROW [rdi+1*SIZEOF_JSAMPROW]
    mov         rbxp, JSAMPROW [rdi+3*SIZEOF_JSAMPROW]
    movq        XMM_MMWORD [rdx+rax*SIZEOF_JSAMPLE], xmm5
    movq        XMM_MMWORD [rbx+rax*SIZEOF_JSAMPLE], xmm3

    add         rsi, byte 4*SIZEOF_FAST_FLOAT  ; wsptr
    add         rdi, byte 4*SIZEOF_JSAMPROW
    dec         rcx                            ; ctr
    jnz         near .rowloop

    pop         rbx
    UNCOLLECT_ARGS 4
    lea         rsp, [rbp-8]
    pop         r15
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
