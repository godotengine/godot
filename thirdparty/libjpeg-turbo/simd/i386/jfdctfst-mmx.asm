;
; jfdctfst.asm - fast integer FDCT (MMX)
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
    GLOBAL_DATA(jconst_fdct_ifast_mmx)

EXTN(jconst_fdct_ifast_mmx):

PW_F0707 times 4 dw F_0_707 << CONST_SHIFT
PW_F0382 times 4 dw F_0_382 << CONST_SHIFT
PW_F0541 times 4 dw F_0_541 << CONST_SHIFT
PW_F1306 times 4 dw F_1_306 << CONST_SHIFT

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform the forward DCT on one block of samples.
;
; GLOBAL(void)
; jsimd_fdct_ifast_mmx(DCTELEM *data)
;

%define data(b)       (b) + 8           ; DCTELEM *data

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_MMWORD  ; mmword wk[WK_NUM]
%define WK_NUM        2

    align       32
    GLOBAL_FUNCTION(jsimd_fdct_ifast_mmx)

EXTN(jsimd_fdct_ifast_mmx):
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
    psubw       mm5, mm7                ; mm5=tmp13
    psubw       mm0, mm4                ; mm0=tmp12
    paddw       mm1, mm7                ; mm1=tmp10
    paddw       mm6, mm4                ; mm6=tmp11

    paddw       mm0, mm5
    psllw       mm0, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm0, [GOTOFF(ebx,PW_F0707)]  ; mm0=z1

    movq        mm7, mm1
    movq        mm4, mm5
    psubw       mm1, mm6                ; mm1=data4
    psubw       mm5, mm0                ; mm5=data6
    paddw       mm7, mm6                ; mm7=data0
    paddw       mm4, mm0                ; mm4=data2

    movq        MMWORD [MMBLOCK(0,1,edx,SIZEOF_DCTELEM)], mm1
    movq        MMWORD [MMBLOCK(2,1,edx,SIZEOF_DCTELEM)], mm5
    movq        MMWORD [MMBLOCK(0,0,edx,SIZEOF_DCTELEM)], mm7
    movq        MMWORD [MMBLOCK(2,0,edx,SIZEOF_DCTELEM)], mm4

    ; -- Odd part

    movq        mm6, MMWORD [wk(0)]     ; mm6=tmp6
    movq        mm0, MMWORD [wk(1)]     ; mm0=tmp7

    paddw       mm2, mm3                ; mm2=tmp10
    paddw       mm3, mm6                ; mm3=tmp11
    paddw       mm6, mm0                ; mm6=tmp12, mm0=tmp7

    psllw       mm2, PRE_MULTIPLY_SCALE_BITS
    psllw       mm6, PRE_MULTIPLY_SCALE_BITS

    psllw       mm3, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm3, [GOTOFF(ebx,PW_F0707)]  ; mm3=z3

    movq        mm1, mm2                     ; mm1=tmp10
    psubw       mm2, mm6
    pmulhw      mm2, [GOTOFF(ebx,PW_F0382)]  ; mm2=z5
    pmulhw      mm1, [GOTOFF(ebx,PW_F0541)]  ; mm1=MULTIPLY(tmp10,FIX_0_54119610)
    pmulhw      mm6, [GOTOFF(ebx,PW_F1306)]  ; mm6=MULTIPLY(tmp12,FIX_1_30656296)
    paddw       mm1, mm2                     ; mm1=z2
    paddw       mm6, mm2                     ; mm6=z4

    movq        mm5, mm0
    psubw       mm0, mm3                ; mm0=z13
    paddw       mm5, mm3                ; mm5=z11

    movq        mm7, mm0
    movq        mm4, mm5
    psubw       mm0, mm1                ; mm0=data3
    psubw       mm5, mm6                ; mm5=data7
    paddw       mm7, mm1                ; mm7=data5
    paddw       mm4, mm6                ; mm4=data1

    movq        MMWORD [MMBLOCK(3,0,edx,SIZEOF_DCTELEM)], mm0
    movq        MMWORD [MMBLOCK(3,1,edx,SIZEOF_DCTELEM)], mm5
    movq        MMWORD [MMBLOCK(1,1,edx,SIZEOF_DCTELEM)], mm7
    movq        MMWORD [MMBLOCK(1,0,edx,SIZEOF_DCTELEM)], mm4

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
    psubw       mm5, mm7                ; mm5=tmp13
    psubw       mm0, mm4                ; mm0=tmp12
    paddw       mm1, mm7                ; mm1=tmp10
    paddw       mm6, mm4                ; mm6=tmp11

    paddw       mm0, mm5
    psllw       mm0, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm0, [GOTOFF(ebx,PW_F0707)]  ; mm0=z1

    movq        mm7, mm1
    movq        mm4, mm5
    psubw       mm1, mm6                ; mm1=data4
    psubw       mm5, mm0                ; mm5=data6
    paddw       mm7, mm6                ; mm7=data0
    paddw       mm4, mm0                ; mm4=data2

    movq        MMWORD [MMBLOCK(4,0,edx,SIZEOF_DCTELEM)], mm1
    movq        MMWORD [MMBLOCK(6,0,edx,SIZEOF_DCTELEM)], mm5
    movq        MMWORD [MMBLOCK(0,0,edx,SIZEOF_DCTELEM)], mm7
    movq        MMWORD [MMBLOCK(2,0,edx,SIZEOF_DCTELEM)], mm4

    ; -- Odd part

    movq        mm6, MMWORD [wk(0)]     ; mm6=tmp6
    movq        mm0, MMWORD [wk(1)]     ; mm0=tmp7

    paddw       mm2, mm3                ; mm2=tmp10
    paddw       mm3, mm6                ; mm3=tmp11
    paddw       mm6, mm0                ; mm6=tmp12, mm0=tmp7

    psllw       mm2, PRE_MULTIPLY_SCALE_BITS
    psllw       mm6, PRE_MULTIPLY_SCALE_BITS

    psllw       mm3, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm3, [GOTOFF(ebx,PW_F0707)]  ; mm3=z3

    movq        mm1, mm2                     ; mm1=tmp10
    psubw       mm2, mm6
    pmulhw      mm2, [GOTOFF(ebx,PW_F0382)]  ; mm2=z5
    pmulhw      mm1, [GOTOFF(ebx,PW_F0541)]  ; mm1=MULTIPLY(tmp10,FIX_0_54119610)
    pmulhw      mm6, [GOTOFF(ebx,PW_F1306)]  ; mm6=MULTIPLY(tmp12,FIX_1_30656296)
    paddw       mm1, mm2                     ; mm1=z2
    paddw       mm6, mm2                     ; mm6=z4

    movq        mm5, mm0
    psubw       mm0, mm3                ; mm0=z13
    paddw       mm5, mm3                ; mm5=z11

    movq        mm7, mm0
    movq        mm4, mm5
    psubw       mm0, mm1                ; mm0=data3
    psubw       mm5, mm6                ; mm5=data7
    paddw       mm7, mm1                ; mm7=data5
    paddw       mm4, mm6                ; mm4=data1

    movq        MMWORD [MMBLOCK(3,0,edx,SIZEOF_DCTELEM)], mm0
    movq        MMWORD [MMBLOCK(7,0,edx,SIZEOF_DCTELEM)], mm5
    movq        MMWORD [MMBLOCK(5,0,edx,SIZEOF_DCTELEM)], mm7
    movq        MMWORD [MMBLOCK(1,0,edx,SIZEOF_DCTELEM)], mm4

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
