;
; jfdctflt.asm - floating-point FDCT (64-bit SSE)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2024, D. R. Commander.
; Copyright (C) 2023, Aliaksiej Kandracienka.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.
;
; This file contains a floating-point implementation of the forward DCT
; (Discrete Cosine Transform). The following code is based directly on
; the IJG's original jfdctflt.c; see the jfdctflt.c for more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%macro  unpcklps2 2  ; %1=(0 1 2 3) / %2=(4 5 6 7) => %1=(0 1 4 5)
    shufps      %1, %2, 0x44
%endmacro

%macro  unpckhps2 2  ; %1=(0 1 2 3) / %2=(4 5 6 7) => %1=(2 3 6 7)
    shufps      %1, %2, 0xEE
%endmacro

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_fdct_float_sse)

EXTN(jconst_fdct_float_sse):

PD_0_382 times 4 dd 0.382683432365089771728460
PD_0_707 times 4 dd 0.707106781186547524400844
PD_0_541 times 4 dd 0.541196100146196984399723
PD_1_306 times 4 dd 1.306562964876376527856643

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64
;
; Perform the forward DCT on one block of samples.
;
; GLOBAL(void)
; jsimd_fdct_float_sse(FAST_FLOAT *data)
;

; r10 = FAST_FLOAT *data

%define wk(i)   r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD  ; xmmword wk[WK_NUM]
%define WK_NUM  2

    align       32
    GLOBAL_FUNCTION(jsimd_fdct_float_sse)

EXTN(jsimd_fdct_float_sse):
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

    mov         rdx, r10                ; (FAST_FLOAT *)
    mov         rcx, DCTSIZE/4
.rowloop:

    movaps      xmm0, XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm2, XMMWORD [XMMBLOCK(2,1,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(3,1,rdx,SIZEOF_FAST_FLOAT)]

    ; xmm0=(20 21 22 23), xmm2=(24 25 26 27)
    ; xmm1=(30 31 32 33), xmm3=(34 35 36 37)

    movaps      xmm4, xmm0              ; transpose coefficients(phase 1)
    unpcklps    xmm0, xmm1              ; xmm0=(20 30 21 31)
    unpckhps    xmm4, xmm1              ; xmm4=(22 32 23 33)
    movaps      xmm5, xmm2              ; transpose coefficients(phase 1)
    unpcklps    xmm2, xmm3              ; xmm2=(24 34 25 35)
    unpckhps    xmm5, xmm3              ; xmm5=(26 36 27 37)

    movaps      xmm6, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm7, XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(0,1,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(1,1,rdx,SIZEOF_FAST_FLOAT)]

    ; xmm6=(00 01 02 03), xmm1=(04 05 06 07)
    ; xmm7=(10 11 12 13), xmm3=(14 15 16 17)

    movaps      XMMWORD [wk(0)], xmm4   ; wk(0)=(22 32 23 33)
    movaps      XMMWORD [wk(1)], xmm2   ; wk(1)=(24 34 25 35)

    movaps      xmm4, xmm6              ; transpose coefficients(phase 1)
    unpcklps    xmm6, xmm7              ; xmm6=(00 10 01 11)
    unpckhps    xmm4, xmm7              ; xmm4=(02 12 03 13)
    movaps      xmm2, xmm1              ; transpose coefficients(phase 1)
    unpcklps    xmm1, xmm3              ; xmm1=(04 14 05 15)
    unpckhps    xmm2, xmm3              ; xmm2=(06 16 07 17)

    movaps      xmm7, xmm6              ; transpose coefficients(phase 2)
    unpcklps2   xmm6, xmm0              ; xmm6=(00 10 20 30)=data0
    unpckhps2   xmm7, xmm0              ; xmm7=(01 11 21 31)=data1
    movaps      xmm3, xmm2              ; transpose coefficients(phase 2)
    unpcklps2   xmm2, xmm5              ; xmm2=(06 16 26 36)=data6
    unpckhps2   xmm3, xmm5              ; xmm3=(07 17 27 37)=data7

    movaps      xmm0, xmm7
    movaps      xmm5, xmm6
    subps       xmm7, xmm2              ; xmm7=data1-data6=tmp6
    subps       xmm6, xmm3              ; xmm6=data0-data7=tmp7
    addps       xmm0, xmm2              ; xmm0=data1+data6=tmp1
    addps       xmm5, xmm3              ; xmm5=data0+data7=tmp0

    movaps      xmm2, XMMWORD [wk(0)]   ; xmm2=(22 32 23 33)
    movaps      xmm3, XMMWORD [wk(1)]   ; xmm3=(24 34 25 35)
    movaps      XMMWORD [wk(0)], xmm7   ; wk(0)=tmp6
    movaps      XMMWORD [wk(1)], xmm6   ; wk(1)=tmp7

    movaps      xmm7, xmm4              ; transpose coefficients(phase 2)
    unpcklps2   xmm4, xmm2              ; xmm4=(02 12 22 32)=data2
    unpckhps2   xmm7, xmm2              ; xmm7=(03 13 23 33)=data3
    movaps      xmm6, xmm1              ; transpose coefficients(phase 2)
    unpcklps2   xmm1, xmm3              ; xmm1=(04 14 24 34)=data4
    unpckhps2   xmm6, xmm3              ; xmm6=(05 15 25 35)=data5

    movaps      xmm2, xmm7
    movaps      xmm3, xmm4
    addps       xmm7, xmm1              ; xmm7=data3+data4=tmp3
    addps       xmm4, xmm6              ; xmm4=data2+data5=tmp2
    subps       xmm2, xmm1              ; xmm2=data3-data4=tmp4
    subps       xmm3, xmm6              ; xmm3=data2-data5=tmp5

    ; -- Even part

    movaps      xmm1, xmm5
    movaps      xmm6, xmm0
    subps       xmm5, xmm7              ; xmm5=tmp13
    subps       xmm0, xmm4              ; xmm0=tmp12
    addps       xmm1, xmm7              ; xmm1=tmp10
    addps       xmm6, xmm4              ; xmm6=tmp11

    addps       xmm0, xmm5
    mulps       xmm0, [rel PD_0_707]    ; xmm0=z1

    movaps      xmm7, xmm1
    movaps      xmm4, xmm5
    subps       xmm1, xmm6              ; xmm1=data4
    subps       xmm5, xmm0              ; xmm5=data6
    addps       xmm7, xmm6              ; xmm7=data0
    addps       xmm4, xmm0              ; xmm4=data2

    movaps      XMMWORD [XMMBLOCK(0,1,rdx,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(2,1,rdx,SIZEOF_FAST_FLOAT)], xmm5
    movaps      XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_FAST_FLOAT)], xmm7
    movaps      XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_FAST_FLOAT)], xmm4

    ; -- Odd part

    movaps      xmm6, XMMWORD [wk(0)]   ; xmm6=tmp6
    movaps      xmm0, XMMWORD [wk(1)]   ; xmm0=tmp7

    addps       xmm2, xmm3              ; xmm2=tmp10
    addps       xmm3, xmm6              ; xmm3=tmp11
    addps       xmm6, xmm0              ; xmm6=tmp12, xmm0=tmp7

    mulps       xmm3, [rel PD_0_707]    ; xmm3=z3

    movaps      xmm1, xmm2              ; xmm1=tmp10
    subps       xmm2, xmm6
    mulps       xmm2, [rel PD_0_382]    ; xmm2=z5
    mulps       xmm1, [rel PD_0_541]    ; xmm1=MULTIPLY(tmp10,FIX_0_541196)
    mulps       xmm6, [rel PD_1_306]    ; xmm6=MULTIPLY(tmp12,FIX_1_306562)
    addps       xmm1, xmm2              ; xmm1=z2
    addps       xmm6, xmm2              ; xmm6=z4

    movaps      xmm5, xmm0
    subps       xmm0, xmm3              ; xmm0=z13
    addps       xmm5, xmm3              ; xmm5=z11

    movaps      xmm7, xmm0
    movaps      xmm4, xmm5
    subps       xmm0, xmm1              ; xmm0=data3
    subps       xmm5, xmm6              ; xmm5=data7
    addps       xmm7, xmm1              ; xmm7=data5
    addps       xmm4, xmm6              ; xmm4=data1

    movaps      XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(3,1,rdx,SIZEOF_FAST_FLOAT)], xmm5
    movaps      XMMWORD [XMMBLOCK(1,1,rdx,SIZEOF_FAST_FLOAT)], xmm7
    movaps      XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_FAST_FLOAT)], xmm4

    add         rdx, 4*DCTSIZE*SIZEOF_FAST_FLOAT
    dec         rcx
    jnz         near .rowloop

    ; ---- Pass 2: process columns.

    mov         rdx, r10                ; (FAST_FLOAT *)
    mov         rcx, DCTSIZE/4
.columnloop:

    movaps      xmm0, XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm2, XMMWORD [XMMBLOCK(6,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(7,0,rdx,SIZEOF_FAST_FLOAT)]

    ; xmm0=(02 12 22 32), xmm2=(42 52 62 72)
    ; xmm1=(03 13 23 33), xmm3=(43 53 63 73)

    movaps      xmm4, xmm0              ; transpose coefficients(phase 1)
    unpcklps    xmm0, xmm1              ; xmm0=(02 03 12 13)
    unpckhps    xmm4, xmm1              ; xmm4=(22 23 32 33)
    movaps      xmm5, xmm2              ; transpose coefficients(phase 1)
    unpcklps    xmm2, xmm3              ; xmm2=(42 43 52 53)
    unpckhps    xmm5, xmm3              ; xmm5=(62 63 72 73)

    movaps      xmm6, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm7, XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(4,0,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(5,0,rdx,SIZEOF_FAST_FLOAT)]

    ; xmm6=(00 10 20 30), xmm1=(40 50 60 70)
    ; xmm7=(01 11 21 31), xmm3=(41 51 61 71)

    movaps      XMMWORD [wk(0)], xmm4   ; wk(0)=(22 23 32 33)
    movaps      XMMWORD [wk(1)], xmm2   ; wk(1)=(42 43 52 53)

    movaps      xmm4, xmm6              ; transpose coefficients(phase 1)
    unpcklps    xmm6, xmm7              ; xmm6=(00 01 10 11)
    unpckhps    xmm4, xmm7              ; xmm4=(20 21 30 31)
    movaps      xmm2, xmm1              ; transpose coefficients(phase 1)
    unpcklps    xmm1, xmm3              ; xmm1=(40 41 50 51)
    unpckhps    xmm2, xmm3              ; xmm2=(60 61 70 71)

    movaps      xmm7, xmm6              ; transpose coefficients(phase 2)
    unpcklps2   xmm6, xmm0              ; xmm6=(00 01 02 03)=data0
    unpckhps2   xmm7, xmm0              ; xmm7=(10 11 12 13)=data1
    movaps      xmm3, xmm2              ; transpose coefficients(phase 2)
    unpcklps2   xmm2, xmm5              ; xmm2=(60 61 62 63)=data6
    unpckhps2   xmm3, xmm5              ; xmm3=(70 71 72 73)=data7

    movaps      xmm0, xmm7
    movaps      xmm5, xmm6
    subps       xmm7, xmm2              ; xmm7=data1-data6=tmp6
    subps       xmm6, xmm3              ; xmm6=data0-data7=tmp7
    addps       xmm0, xmm2              ; xmm0=data1+data6=tmp1
    addps       xmm5, xmm3              ; xmm5=data0+data7=tmp0

    movaps      xmm2, XMMWORD [wk(0)]   ; xmm2=(22 23 32 33)
    movaps      xmm3, XMMWORD [wk(1)]   ; xmm3=(42 43 52 53)
    movaps      XMMWORD [wk(0)], xmm7   ; wk(0)=tmp6
    movaps      XMMWORD [wk(1)], xmm6   ; wk(1)=tmp7

    movaps      xmm7, xmm4              ; transpose coefficients(phase 2)
    unpcklps2   xmm4, xmm2              ; xmm4=(20 21 22 23)=data2
    unpckhps2   xmm7, xmm2              ; xmm7=(30 31 32 33)=data3
    movaps      xmm6, xmm1              ; transpose coefficients(phase 2)
    unpcklps2   xmm1, xmm3              ; xmm1=(40 41 42 43)=data4
    unpckhps2   xmm6, xmm3              ; xmm6=(50 51 52 53)=data5

    movaps      xmm2, xmm7
    movaps      xmm3, xmm4
    addps       xmm7, xmm1              ; xmm7=data3+data4=tmp3
    addps       xmm4, xmm6              ; xmm4=data2+data5=tmp2
    subps       xmm2, xmm1              ; xmm2=data3-data4=tmp4
    subps       xmm3, xmm6              ; xmm3=data2-data5=tmp5

    ; -- Even part

    movaps      xmm1, xmm5
    movaps      xmm6, xmm0
    subps       xmm5, xmm7              ; xmm5=tmp13
    subps       xmm0, xmm4              ; xmm0=tmp12
    addps       xmm1, xmm7              ; xmm1=tmp10
    addps       xmm6, xmm4              ; xmm6=tmp11

    addps       xmm0, xmm5
    mulps       xmm0, [rel PD_0_707]    ; xmm0=z1

    movaps      xmm7, xmm1
    movaps      xmm4, xmm5
    subps       xmm1, xmm6              ; xmm1=data4
    subps       xmm5, xmm0              ; xmm5=data6
    addps       xmm7, xmm6              ; xmm7=data0
    addps       xmm4, xmm0              ; xmm4=data2

    movaps      XMMWORD [XMMBLOCK(4,0,rdx,SIZEOF_FAST_FLOAT)], xmm1
    movaps      XMMWORD [XMMBLOCK(6,0,rdx,SIZEOF_FAST_FLOAT)], xmm5
    movaps      XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_FAST_FLOAT)], xmm7
    movaps      XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_FAST_FLOAT)], xmm4

    ; -- Odd part

    movaps      xmm6, XMMWORD [wk(0)]   ; xmm6=tmp6
    movaps      xmm0, XMMWORD [wk(1)]   ; xmm0=tmp7

    addps       xmm2, xmm3              ; xmm2=tmp10
    addps       xmm3, xmm6              ; xmm3=tmp11
    addps       xmm6, xmm0              ; xmm6=tmp12, xmm0=tmp7

    mulps       xmm3, [rel PD_0_707]    ; xmm3=z3

    movaps      xmm1, xmm2              ; xmm1=tmp10
    subps       xmm2, xmm6
    mulps       xmm2, [rel PD_0_382]    ; xmm2=z5
    mulps       xmm1, [rel PD_0_541]    ; xmm1=MULTIPLY(tmp10,FIX_0_541196)
    mulps       xmm6, [rel PD_1_306]    ; xmm6=MULTIPLY(tmp12,FIX_1_306562)
    addps       xmm1, xmm2              ; xmm1=z2
    addps       xmm6, xmm2              ; xmm6=z4

    movaps      xmm5, xmm0
    subps       xmm0, xmm3              ; xmm0=z13
    addps       xmm5, xmm3              ; xmm5=z11

    movaps      xmm7, xmm0
    movaps      xmm4, xmm5
    subps       xmm0, xmm1              ; xmm0=data3
    subps       xmm5, xmm6              ; xmm5=data7
    addps       xmm7, xmm1              ; xmm7=data5
    addps       xmm4, xmm6              ; xmm4=data1

    movaps      XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(7,0,rdx,SIZEOF_FAST_FLOAT)], xmm5
    movaps      XMMWORD [XMMBLOCK(5,0,rdx,SIZEOF_FAST_FLOAT)], xmm7
    movaps      XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_FAST_FLOAT)], xmm4

    add         rdx, byte 4*SIZEOF_FAST_FLOAT
    dec         rcx
    jnz         near .columnloop

    UNCOLLECT_ARGS 1
    lea         rsp, [rbp-8]
    pop         r15
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
