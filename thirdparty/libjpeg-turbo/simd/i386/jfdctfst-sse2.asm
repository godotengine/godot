;
; jfdctfst.asm - fast integer FDCT (SSE2)
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
; This file contains a fast, not so accurate integer implementation of
; the forward DCT (Discrete Cosine Transform). The following code is
; based directly on the IJG's original jfdctfst.c; see the jfdctfst.c
; for more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%define CONST_BITS  8  ; 14 is also OK.

%if CONST_BITS == 8
F_0_382 equ  98  ; FIX(0.382683433)
F_0_541 equ 139  ; FIX(0.541196100)
F_0_707 equ 181  ; FIX(0.707106781)
F_1_306 equ 334  ; FIX(1.306562965)
%else
; NASM cannot do compile-time arithmetic on floating-point constants.
%define DESCALE(x, n)  (((x) + (1 << ((n) - 1))) >> (n))
F_0_382 equ DESCALE( 410903207, 30 - CONST_BITS)  ; FIX(0.382683433)
F_0_541 equ DESCALE( 581104887, 30 - CONST_BITS)  ; FIX(0.541196100)
F_0_707 equ DESCALE( 759250124, 30 - CONST_BITS)  ; FIX(0.707106781)
F_1_306 equ DESCALE(1402911301, 30 - CONST_BITS)  ; FIX(1.306562965)
%endif

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

; PRE_MULTIPLY_SCALE_BITS <= 2 (to avoid overflow)
; CONST_BITS + CONST_SHIFT + PRE_MULTIPLY_SCALE_BITS == 16 (for pmulhw)

%define PRE_MULTIPLY_SCALE_BITS  2
%define CONST_SHIFT              (16 - PRE_MULTIPLY_SCALE_BITS - CONST_BITS)

    ALIGNZ      32
    GLOBAL_DATA(jconst_fdct_ifast_sse2)

EXTN(jconst_fdct_ifast_sse2):

PW_F0707 times 8 dw F_0_707 << CONST_SHIFT
PW_F0382 times 8 dw F_0_382 << CONST_SHIFT
PW_F0541 times 8 dw F_0_541 << CONST_SHIFT
PW_F1306 times 8 dw F_1_306 << CONST_SHIFT

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform the forward DCT on one block of samples.
;
; GLOBAL(void)
; jsimd_fdct_ifast_sse2(DCTELEM *data)
;

%define data(b)       (b) + 8           ; DCTELEM *data

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_XMMWORD
                                        ; xmmword wk[WK_NUM]
%define WK_NUM        2

    align       32
    GLOBAL_FUNCTION(jsimd_fdct_ifast_sse2)

EXTN(jsimd_fdct_ifast_sse2):
    push        ebp
    mov         eax, esp                     ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    mov         [esp], eax
    mov         ebp, esp                     ; ebp = aligned ebp
    lea         esp, [wk(0)]
    PUSHPIC     ebx
;   push        ecx                     ; unused
;   push        edx                     ; need not be preserved
;   push        esi                     ; unused
;   push        edi                     ; unused

    GET_GOT     ebx                     ; get GOT address

    ; ---- Pass 1: process rows.

    mov         edx, POINTER [data(eax)]  ; (DCTELEM *)

    movdqa      xmm0, XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_DCTELEM)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(1,0,edx,SIZEOF_DCTELEM)]
    movdqa      xmm2, XMMWORD [XMMBLOCK(2,0,edx,SIZEOF_DCTELEM)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(3,0,edx,SIZEOF_DCTELEM)]

    ; xmm0=(00 01 02 03 04 05 06 07), xmm2=(20 21 22 23 24 25 26 27)
    ; xmm1=(10 11 12 13 14 15 16 17), xmm3=(30 31 32 33 34 35 36 37)

    movdqa      xmm4, xmm0              ; transpose coefficients(phase 1)
    punpcklwd   xmm0, xmm1              ; xmm0=(00 10 01 11 02 12 03 13)
    punpckhwd   xmm4, xmm1              ; xmm4=(04 14 05 15 06 16 07 17)
    movdqa      xmm5, xmm2              ; transpose coefficients(phase 1)
    punpcklwd   xmm2, xmm3              ; xmm2=(20 30 21 31 22 32 23 33)
    punpckhwd   xmm5, xmm3              ; xmm5=(24 34 25 35 26 36 27 37)

    movdqa      xmm6, XMMWORD [XMMBLOCK(4,0,edx,SIZEOF_DCTELEM)]
    movdqa      xmm7, XMMWORD [XMMBLOCK(5,0,edx,SIZEOF_DCTELEM)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(6,0,edx,SIZEOF_DCTELEM)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(7,0,edx,SIZEOF_DCTELEM)]

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
    movdqa      XMMWORD [wk(0)], xmm7   ; wk(0)=(42 52 62 72 43 53 63 73)
    movdqa      XMMWORD [wk(1)], xmm2   ; wk(1)=(44 54 64 74 45 55 65 75)

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

    movdqa      xmm2, XMMWORD [wk(0)]   ; xmm2=(42 52 62 72 43 53 63 73)
    movdqa      xmm5, XMMWORD [wk(1)]   ; xmm5=(44 54 64 74 45 55 65 75)
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
    psubw       xmm3, xmm1              ; xmm3=tmp13
    psubw       xmm6, xmm7              ; xmm6=tmp12
    paddw       xmm4, xmm1              ; xmm4=tmp10
    paddw       xmm0, xmm7              ; xmm0=tmp11

    paddw       xmm6, xmm3
    psllw       xmm6, PRE_MULTIPLY_SCALE_BITS
    pmulhw      xmm6, [GOTOFF(ebx,PW_F0707)]  ; xmm6=z1

    movdqa      xmm1, xmm4
    movdqa      xmm7, xmm3
    psubw       xmm4, xmm0              ; xmm4=data4
    psubw       xmm3, xmm6              ; xmm3=data6
    paddw       xmm1, xmm0              ; xmm1=data0
    paddw       xmm7, xmm6              ; xmm7=data2

    movdqa      xmm0, XMMWORD [wk(0)]   ; xmm0=tmp6
    movdqa      xmm6, XMMWORD [wk(1)]   ; xmm6=tmp7
    movdqa      XMMWORD [wk(0)], xmm4   ; wk(0)=data4
    movdqa      XMMWORD [wk(1)], xmm3   ; wk(1)=data6

    ; -- Odd part

    paddw       xmm2, xmm5              ; xmm2=tmp10
    paddw       xmm5, xmm0              ; xmm5=tmp11
    paddw       xmm0, xmm6              ; xmm0=tmp12, xmm6=tmp7

    psllw       xmm2, PRE_MULTIPLY_SCALE_BITS
    psllw       xmm0, PRE_MULTIPLY_SCALE_BITS

    psllw       xmm5, PRE_MULTIPLY_SCALE_BITS
    pmulhw      xmm5, [GOTOFF(ebx,PW_F0707)]  ; xmm5=z3

    movdqa      xmm4, xmm2                    ; xmm4=tmp10
    psubw       xmm2, xmm0
    pmulhw      xmm2, [GOTOFF(ebx,PW_F0382)]  ; xmm2=z5
    pmulhw      xmm4, [GOTOFF(ebx,PW_F0541)]  ; xmm4=MULTIPLY(tmp10,FIX_0_541196)
    pmulhw      xmm0, [GOTOFF(ebx,PW_F1306)]  ; xmm0=MULTIPLY(tmp12,FIX_1_306562)
    paddw       xmm4, xmm2                    ; xmm4=z2
    paddw       xmm0, xmm2                    ; xmm0=z4

    movdqa      xmm3, xmm6
    psubw       xmm6, xmm5              ; xmm6=z13
    paddw       xmm3, xmm5              ; xmm3=z11

    movdqa      xmm2, xmm6
    movdqa      xmm5, xmm3
    psubw       xmm6, xmm4              ; xmm6=data3
    psubw       xmm3, xmm0              ; xmm3=data7
    paddw       xmm2, xmm4              ; xmm2=data5
    paddw       xmm5, xmm0              ; xmm5=data1

    ; ---- Pass 2: process columns.

;   mov         edx, POINTER [data(eax)]  ; (DCTELEM *)

    ; xmm1=(00 10 20 30 40 50 60 70), xmm7=(02 12 22 32 42 52 62 72)
    ; xmm5=(01 11 21 31 41 51 61 71), xmm6=(03 13 23 33 43 53 63 73)

    movdqa      xmm4, xmm1              ; transpose coefficients(phase 1)
    punpcklwd   xmm1, xmm5              ; xmm1=(00 01 10 11 20 21 30 31)
    punpckhwd   xmm4, xmm5              ; xmm4=(40 41 50 51 60 61 70 71)
    movdqa      xmm0, xmm7              ; transpose coefficients(phase 1)
    punpcklwd   xmm7, xmm6              ; xmm7=(02 03 12 13 22 23 32 33)
    punpckhwd   xmm0, xmm6              ; xmm0=(42 43 52 53 62 63 72 73)

    movdqa      xmm5, XMMWORD [wk(0)]   ; xmm5=col4
    movdqa      xmm6, XMMWORD [wk(1)]   ; xmm6=col6

    ; xmm5=(04 14 24 34 44 54 64 74), xmm6=(06 16 26 36 46 56 66 76)
    ; xmm2=(05 15 25 35 45 55 65 75), xmm3=(07 17 27 37 47 57 67 77)

    movdqa      XMMWORD [wk(0)], xmm7   ; wk(0)=(02 03 12 13 22 23 32 33)
    movdqa      XMMWORD [wk(1)], xmm0   ; wk(1)=(42 43 52 53 62 63 72 73)

    movdqa      xmm7, xmm5              ; transpose coefficients(phase 1)
    punpcklwd   xmm5, xmm2              ; xmm5=(04 05 14 15 24 25 34 35)
    punpckhwd   xmm7, xmm2              ; xmm7=(44 45 54 55 64 65 74 75)
    movdqa      xmm0, xmm6              ; transpose coefficients(phase 1)
    punpcklwd   xmm6, xmm3              ; xmm6=(06 07 16 17 26 27 36 37)
    punpckhwd   xmm0, xmm3              ; xmm0=(46 47 56 57 66 67 76 77)

    movdqa      xmm2, xmm5              ; transpose coefficients(phase 2)
    punpckldq   xmm5, xmm6              ; xmm5=(04 05 06 07 14 15 16 17)
    punpckhdq   xmm2, xmm6              ; xmm2=(24 25 26 27 34 35 36 37)
    movdqa      xmm3, xmm7              ; transpose coefficients(phase 2)
    punpckldq   xmm7, xmm0              ; xmm7=(44 45 46 47 54 55 56 57)
    punpckhdq   xmm3, xmm0              ; xmm3=(64 65 66 67 74 75 76 77)

    movdqa      xmm6, XMMWORD [wk(0)]   ; xmm6=(02 03 12 13 22 23 32 33)
    movdqa      xmm0, XMMWORD [wk(1)]   ; xmm0=(42 43 52 53 62 63 72 73)
    movdqa      XMMWORD [wk(0)], xmm2   ; wk(0)=(24 25 26 27 34 35 36 37)
    movdqa      XMMWORD [wk(1)], xmm7   ; wk(1)=(44 45 46 47 54 55 56 57)

    movdqa      xmm2, xmm1              ; transpose coefficients(phase 2)
    punpckldq   xmm1, xmm6              ; xmm1=(00 01 02 03 10 11 12 13)
    punpckhdq   xmm2, xmm6              ; xmm2=(20 21 22 23 30 31 32 33)
    movdqa      xmm7, xmm4              ; transpose coefficients(phase 2)
    punpckldq   xmm4, xmm0              ; xmm4=(40 41 42 43 50 51 52 53)
    punpckhdq   xmm7, xmm0              ; xmm7=(60 61 62 63 70 71 72 73)

    movdqa      xmm6, xmm1              ; transpose coefficients(phase 3)
    punpcklqdq  xmm1, xmm5              ; xmm1=(00 01 02 03 04 05 06 07)=data0
    punpckhqdq  xmm6, xmm5              ; xmm6=(10 11 12 13 14 15 16 17)=data1
    movdqa      xmm0, xmm7              ; transpose coefficients(phase 3)
    punpcklqdq  xmm7, xmm3              ; xmm7=(60 61 62 63 64 65 66 67)=data6
    punpckhqdq  xmm0, xmm3              ; xmm0=(70 71 72 73 74 75 76 77)=data7

    movdqa      xmm5, xmm6
    movdqa      xmm3, xmm1
    psubw       xmm6, xmm7              ; xmm6=data1-data6=tmp6
    psubw       xmm1, xmm0              ; xmm1=data0-data7=tmp7
    paddw       xmm5, xmm7              ; xmm5=data1+data6=tmp1
    paddw       xmm3, xmm0              ; xmm3=data0+data7=tmp0

    movdqa      xmm7, XMMWORD [wk(0)]   ; xmm7=(24 25 26 27 34 35 36 37)
    movdqa      xmm0, XMMWORD [wk(1)]   ; xmm0=(44 45 46 47 54 55 56 57)
    movdqa      XMMWORD [wk(0)], xmm6   ; wk(0)=tmp6
    movdqa      XMMWORD [wk(1)], xmm1   ; wk(1)=tmp7

    movdqa      xmm6, xmm2              ; transpose coefficients(phase 3)
    punpcklqdq  xmm2, xmm7              ; xmm2=(20 21 22 23 24 25 26 27)=data2
    punpckhqdq  xmm6, xmm7              ; xmm6=(30 31 32 33 34 35 36 37)=data3
    movdqa      xmm1, xmm4              ; transpose coefficients(phase 3)
    punpcklqdq  xmm4, xmm0              ; xmm4=(40 41 42 43 44 45 46 47)=data4
    punpckhqdq  xmm1, xmm0              ; xmm1=(50 51 52 53 54 55 56 57)=data5

    movdqa      xmm7, xmm6
    movdqa      xmm0, xmm2
    paddw       xmm6, xmm4              ; xmm6=data3+data4=tmp3
    paddw       xmm2, xmm1              ; xmm2=data2+data5=tmp2
    psubw       xmm7, xmm4              ; xmm7=data3-data4=tmp4
    psubw       xmm0, xmm1              ; xmm0=data2-data5=tmp5

    ; -- Even part

    movdqa      xmm4, xmm3
    movdqa      xmm1, xmm5
    psubw       xmm3, xmm6              ; xmm3=tmp13
    psubw       xmm5, xmm2              ; xmm5=tmp12
    paddw       xmm4, xmm6              ; xmm4=tmp10
    paddw       xmm1, xmm2              ; xmm1=tmp11

    paddw       xmm5, xmm3
    psllw       xmm5, PRE_MULTIPLY_SCALE_BITS
    pmulhw      xmm5, [GOTOFF(ebx,PW_F0707)]  ; xmm5=z1

    movdqa      xmm6, xmm4
    movdqa      xmm2, xmm3
    psubw       xmm4, xmm1              ; xmm4=data4
    psubw       xmm3, xmm5              ; xmm3=data6
    paddw       xmm6, xmm1              ; xmm6=data0
    paddw       xmm2, xmm5              ; xmm2=data2

    movdqa      XMMWORD [XMMBLOCK(4,0,edx,SIZEOF_DCTELEM)], xmm4
    movdqa      XMMWORD [XMMBLOCK(6,0,edx,SIZEOF_DCTELEM)], xmm3
    movdqa      XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_DCTELEM)], xmm6
    movdqa      XMMWORD [XMMBLOCK(2,0,edx,SIZEOF_DCTELEM)], xmm2

    ; -- Odd part

    movdqa      xmm1, XMMWORD [wk(0)]   ; xmm1=tmp6
    movdqa      xmm5, XMMWORD [wk(1)]   ; xmm5=tmp7

    paddw       xmm7, xmm0              ; xmm7=tmp10
    paddw       xmm0, xmm1              ; xmm0=tmp11
    paddw       xmm1, xmm5              ; xmm1=tmp12, xmm5=tmp7

    psllw       xmm7, PRE_MULTIPLY_SCALE_BITS
    psllw       xmm1, PRE_MULTIPLY_SCALE_BITS

    psllw       xmm0, PRE_MULTIPLY_SCALE_BITS
    pmulhw      xmm0, [GOTOFF(ebx,PW_F0707)]  ; xmm0=z3

    movdqa      xmm4, xmm7                    ; xmm4=tmp10
    psubw       xmm7, xmm1
    pmulhw      xmm7, [GOTOFF(ebx,PW_F0382)]  ; xmm7=z5
    pmulhw      xmm4, [GOTOFF(ebx,PW_F0541)]  ; xmm4=MULTIPLY(tmp10,FIX_0_541196)
    pmulhw      xmm1, [GOTOFF(ebx,PW_F1306)]  ; xmm1=MULTIPLY(tmp12,FIX_1_306562)
    paddw       xmm4, xmm7                    ; xmm4=z2
    paddw       xmm1, xmm7                    ; xmm1=z4

    movdqa      xmm3, xmm5
    psubw       xmm5, xmm0              ; xmm5=z13
    paddw       xmm3, xmm0              ; xmm3=z11

    movdqa      xmm6, xmm5
    movdqa      xmm2, xmm3
    psubw       xmm5, xmm4              ; xmm5=data3
    psubw       xmm3, xmm1              ; xmm3=data7
    paddw       xmm6, xmm4              ; xmm6=data5
    paddw       xmm2, xmm1              ; xmm2=data1

    movdqa      XMMWORD [XMMBLOCK(3,0,edx,SIZEOF_DCTELEM)], xmm5
    movdqa      XMMWORD [XMMBLOCK(7,0,edx,SIZEOF_DCTELEM)], xmm3
    movdqa      XMMWORD [XMMBLOCK(5,0,edx,SIZEOF_DCTELEM)], xmm6
    movdqa      XMMWORD [XMMBLOCK(1,0,edx,SIZEOF_DCTELEM)], xmm2

;   pop         edi                     ; unused
;   pop         esi                     ; unused
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; unused
    POPPIC      ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
