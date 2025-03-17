;
; jfdctflt.asm - floating-point FDCT (3DNow!)
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
; This file contains a floating-point implementation of the forward DCT
; (Discrete Cosine Transform). The following code is based directly on
; the IJG's original jfdctflt.c; see the jfdctflt.c for more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_fdct_float_3dnow)

EXTN(jconst_fdct_float_3dnow):

PD_0_382 times 2 dd 0.382683432365089771728460
PD_0_707 times 2 dd 0.707106781186547524400844
PD_0_541 times 2 dd 0.541196100146196984399723
PD_1_306 times 2 dd 1.306562964876376527856643

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform the forward DCT on one block of samples.
;
; GLOBAL(void)
; jsimd_fdct_float_3dnow(FAST_FLOAT *data)
;

%define data(b)       (b) + 8           ; FAST_FLOAT *data

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_MMWORD  ; mmword wk[WK_NUM]
%define WK_NUM        2

    align       32
    GLOBAL_FUNCTION(jsimd_fdct_float_3dnow)

EXTN(jsimd_fdct_float_3dnow):
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

    mov         edx, POINTER [data(eax)]  ; (FAST_FLOAT *)
    mov         ecx, DCTSIZE/2
    ALIGNX      16, 7
.rowloop:

    movq        mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm1, MMWORD [MMBLOCK(1,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm2, MMWORD [MMBLOCK(0,3,edx,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(1,3,edx,SIZEOF_FAST_FLOAT)]

    ; mm0=(00 01), mm1=(10 11), mm2=(06 07), mm3=(16 17)

    movq        mm4, mm0                ; transpose coefficients
    punpckldq   mm0, mm1                ; mm0=(00 10)=data0
    punpckhdq   mm4, mm1                ; mm4=(01 11)=data1
    movq        mm5, mm2                ; transpose coefficients
    punpckldq   mm2, mm3                ; mm2=(06 16)=data6
    punpckhdq   mm5, mm3                ; mm5=(07 17)=data7

    movq        mm6, mm4
    movq        mm7, mm0
    pfsub       mm4, mm2                ; mm4=data1-data6=tmp6
    pfsub       mm0, mm5                ; mm0=data0-data7=tmp7
    pfadd       mm6, mm2                ; mm6=data1+data6=tmp1
    pfadd       mm7, mm5                ; mm7=data0+data7=tmp0

    movq        mm1, MMWORD [MMBLOCK(0,1,edx,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(1,1,edx,SIZEOF_FAST_FLOAT)]
    movq        mm2, MMWORD [MMBLOCK(0,2,edx,SIZEOF_FAST_FLOAT)]
    movq        mm5, MMWORD [MMBLOCK(1,2,edx,SIZEOF_FAST_FLOAT)]

    ; mm1=(02 03), mm3=(12 13), mm2=(04 05), mm5=(14 15)

    movq        MMWORD [wk(0)], mm4     ; wk(0)=tmp6
    movq        MMWORD [wk(1)], mm0     ; wk(1)=tmp7

    movq        mm4, mm1                ; transpose coefficients
    punpckldq   mm1, mm3                ; mm1=(02 12)=data2
    punpckhdq   mm4, mm3                ; mm4=(03 13)=data3
    movq        mm0, mm2                ; transpose coefficients
    punpckldq   mm2, mm5                ; mm2=(04 14)=data4
    punpckhdq   mm0, mm5                ; mm0=(05 15)=data5

    movq        mm3, mm4
    movq        mm5, mm1
    pfadd       mm4, mm2                ; mm4=data3+data4=tmp3
    pfadd       mm1, mm0                ; mm1=data2+data5=tmp2
    pfsub       mm3, mm2                ; mm3=data3-data4=tmp4
    pfsub       mm5, mm0                ; mm5=data2-data5=tmp5

    ; -- Even part

    movq        mm2, mm7
    movq        mm0, mm6
    pfsub       mm7, mm4                ; mm7=tmp13
    pfsub       mm6, mm1                ; mm6=tmp12
    pfadd       mm2, mm4                ; mm2=tmp10
    pfadd       mm0, mm1                ; mm0=tmp11

    pfadd       mm6, mm7
    pfmul       mm6, [GOTOFF(ebx,PD_0_707)]  ; mm6=z1

    movq        mm4, mm2
    movq        mm1, mm7
    pfsub       mm2, mm0                ; mm2=data4
    pfsub       mm7, mm6                ; mm7=data6
    pfadd       mm4, mm0                ; mm4=data0
    pfadd       mm1, mm6                ; mm1=data2

    movq        MMWORD [MMBLOCK(0,2,edx,SIZEOF_FAST_FLOAT)], mm2
    movq        MMWORD [MMBLOCK(0,3,edx,SIZEOF_FAST_FLOAT)], mm7
    movq        MMWORD [MMBLOCK(0,0,edx,SIZEOF_FAST_FLOAT)], mm4
    movq        MMWORD [MMBLOCK(0,1,edx,SIZEOF_FAST_FLOAT)], mm1

    ; -- Odd part

    movq        mm0, MMWORD [wk(0)]     ; mm0=tmp6
    movq        mm6, MMWORD [wk(1)]     ; mm6=tmp7

    pfadd       mm3, mm5                ; mm3=tmp10
    pfadd       mm5, mm0                ; mm5=tmp11
    pfadd       mm0, mm6                ; mm0=tmp12, mm6=tmp7

    pfmul       mm5, [GOTOFF(ebx,PD_0_707)]  ; mm5=z3

    movq        mm2, mm3                     ; mm2=tmp10
    pfsub       mm3, mm0
    pfmul       mm3, [GOTOFF(ebx,PD_0_382)]  ; mm3=z5
    pfmul       mm2, [GOTOFF(ebx,PD_0_541)]  ; mm2=MULTIPLY(tmp10,FIX_0_54119610)
    pfmul       mm0, [GOTOFF(ebx,PD_1_306)]  ; mm0=MULTIPLY(tmp12,FIX_1_30656296)
    pfadd       mm2, mm3                     ; mm2=z2
    pfadd       mm0, mm3                     ; mm0=z4

    movq        mm7, mm6
    pfsub       mm6, mm5                ; mm6=z13
    pfadd       mm7, mm5                ; mm7=z11

    movq        mm4, mm6
    movq        mm1, mm7
    pfsub       mm6, mm2                ; mm6=data3
    pfsub       mm7, mm0                ; mm7=data7
    pfadd       mm4, mm2                ; mm4=data5
    pfadd       mm1, mm0                ; mm1=data1

    movq        MMWORD [MMBLOCK(1,1,edx,SIZEOF_FAST_FLOAT)], mm6
    movq        MMWORD [MMBLOCK(1,3,edx,SIZEOF_FAST_FLOAT)], mm7
    movq        MMWORD [MMBLOCK(1,2,edx,SIZEOF_FAST_FLOAT)], mm4
    movq        MMWORD [MMBLOCK(1,0,edx,SIZEOF_FAST_FLOAT)], mm1

    add         edx, byte 2*DCTSIZE*SIZEOF_FAST_FLOAT
    dec         ecx
    jnz         near .rowloop

    ; ---- Pass 2: process columns.

    mov         edx, POINTER [data(eax)]  ; (FAST_FLOAT *)
    mov         ecx, DCTSIZE/2
    ALIGNX      16, 7
.columnloop:

    movq        mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm1, MMWORD [MMBLOCK(1,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm2, MMWORD [MMBLOCK(6,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(7,0,edx,SIZEOF_FAST_FLOAT)]

    ; mm0=(00 10), mm1=(01 11), mm2=(60 70), mm3=(61 71)

    movq        mm4, mm0                ; transpose coefficients
    punpckldq   mm0, mm1                ; mm0=(00 01)=data0
    punpckhdq   mm4, mm1                ; mm4=(10 11)=data1
    movq        mm5, mm2                ; transpose coefficients
    punpckldq   mm2, mm3                ; mm2=(60 61)=data6
    punpckhdq   mm5, mm3                ; mm5=(70 71)=data7

    movq        mm6, mm4
    movq        mm7, mm0
    pfsub       mm4, mm2                ; mm4=data1-data6=tmp6
    pfsub       mm0, mm5                ; mm0=data0-data7=tmp7
    pfadd       mm6, mm2                ; mm6=data1+data6=tmp1
    pfadd       mm7, mm5                ; mm7=data0+data7=tmp0

    movq        mm1, MMWORD [MMBLOCK(2,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(3,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm2, MMWORD [MMBLOCK(4,0,edx,SIZEOF_FAST_FLOAT)]
    movq        mm5, MMWORD [MMBLOCK(5,0,edx,SIZEOF_FAST_FLOAT)]

    ; mm1=(20 30), mm3=(21 31), mm2=(40 50), mm5=(41 51)

    movq        MMWORD [wk(0)], mm4     ; wk(0)=tmp6
    movq        MMWORD [wk(1)], mm0     ; wk(1)=tmp7

    movq        mm4, mm1                ; transpose coefficients
    punpckldq   mm1, mm3                ; mm1=(20 21)=data2
    punpckhdq   mm4, mm3                ; mm4=(30 31)=data3
    movq        mm0, mm2                ; transpose coefficients
    punpckldq   mm2, mm5                ; mm2=(40 41)=data4
    punpckhdq   mm0, mm5                ; mm0=(50 51)=data5

    movq        mm3, mm4
    movq        mm5, mm1
    pfadd       mm4, mm2                ; mm4=data3+data4=tmp3
    pfadd       mm1, mm0                ; mm1=data2+data5=tmp2
    pfsub       mm3, mm2                ; mm3=data3-data4=tmp4
    pfsub       mm5, mm0                ; mm5=data2-data5=tmp5

    ; -- Even part

    movq        mm2, mm7
    movq        mm0, mm6
    pfsub       mm7, mm4                ; mm7=tmp13
    pfsub       mm6, mm1                ; mm6=tmp12
    pfadd       mm2, mm4                ; mm2=tmp10
    pfadd       mm0, mm1                ; mm0=tmp11

    pfadd       mm6, mm7
    pfmul       mm6, [GOTOFF(ebx,PD_0_707)]  ; mm6=z1

    movq        mm4, mm2
    movq        mm1, mm7
    pfsub       mm2, mm0                ; mm2=data4
    pfsub       mm7, mm6                ; mm7=data6
    pfadd       mm4, mm0                ; mm4=data0
    pfadd       mm1, mm6                ; mm1=data2

    movq        MMWORD [MMBLOCK(4,0,edx,SIZEOF_FAST_FLOAT)], mm2
    movq        MMWORD [MMBLOCK(6,0,edx,SIZEOF_FAST_FLOAT)], mm7
    movq        MMWORD [MMBLOCK(0,0,edx,SIZEOF_FAST_FLOAT)], mm4
    movq        MMWORD [MMBLOCK(2,0,edx,SIZEOF_FAST_FLOAT)], mm1

    ; -- Odd part

    movq        mm0, MMWORD [wk(0)]     ; mm0=tmp6
    movq        mm6, MMWORD [wk(1)]     ; mm6=tmp7

    pfadd       mm3, mm5                ; mm3=tmp10
    pfadd       mm5, mm0                ; mm5=tmp11
    pfadd       mm0, mm6                ; mm0=tmp12, mm6=tmp7

    pfmul       mm5, [GOTOFF(ebx,PD_0_707)]  ; mm5=z3

    movq        mm2, mm3                     ; mm2=tmp10
    pfsub       mm3, mm0
    pfmul       mm3, [GOTOFF(ebx,PD_0_382)]  ; mm3=z5
    pfmul       mm2, [GOTOFF(ebx,PD_0_541)]  ; mm2=MULTIPLY(tmp10,FIX_0_54119610)
    pfmul       mm0, [GOTOFF(ebx,PD_1_306)]  ; mm0=MULTIPLY(tmp12,FIX_1_30656296)
    pfadd       mm2, mm3                     ; mm2=z2
    pfadd       mm0, mm3                     ; mm0=z4

    movq        mm7, mm6
    pfsub       mm6, mm5                ; mm6=z13
    pfadd       mm7, mm5                ; mm7=z11

    movq        mm4, mm6
    movq        mm1, mm7
    pfsub       mm6, mm2                ; mm6=data3
    pfsub       mm7, mm0                ; mm7=data7
    pfadd       mm4, mm2                ; mm4=data5
    pfadd       mm1, mm0                ; mm1=data1

    movq        MMWORD [MMBLOCK(3,0,edx,SIZEOF_FAST_FLOAT)], mm6
    movq        MMWORD [MMBLOCK(7,0,edx,SIZEOF_FAST_FLOAT)], mm7
    movq        MMWORD [MMBLOCK(5,0,edx,SIZEOF_FAST_FLOAT)], mm4
    movq        MMWORD [MMBLOCK(1,0,edx,SIZEOF_FAST_FLOAT)], mm1

    add         edx, byte 2*SIZEOF_FAST_FLOAT
    dec         ecx
    jnz         near .columnloop

    femms                               ; empty MMX/3DNow! state

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
