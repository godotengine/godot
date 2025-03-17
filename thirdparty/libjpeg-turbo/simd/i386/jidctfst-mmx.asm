;
; jidctfst.asm - fast integer IDCT (MMX)
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
; the inverse DCT (Discrete Cosine Transform). The following code is
; based directly on the IJG's original jidctfst.c; see the jidctfst.c
; for more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%define CONST_BITS  8  ; 14 is also OK.
%define PASS1_BITS  2

%if IFAST_SCALE_BITS != PASS1_BITS
%error "'IFAST_SCALE_BITS' must be equal to 'PASS1_BITS'."
%endif

%if CONST_BITS == 8
F_1_082 equ 277              ; FIX(1.082392200)
F_1_414 equ 362              ; FIX(1.414213562)
F_1_847 equ 473              ; FIX(1.847759065)
F_2_613 equ 669              ; FIX(2.613125930)
F_1_613 equ (F_2_613 - 256)  ; FIX(2.613125930) - FIX(1)
%else
; NASM cannot do compile-time arithmetic on floating-point constants.
%define DESCALE(x, n)  (((x) + (1 << ((n) - 1))) >> (n))
F_1_082 equ DESCALE(1162209775, 30 - CONST_BITS)  ; FIX(1.082392200)
F_1_414 equ DESCALE(1518500249, 30 - CONST_BITS)  ; FIX(1.414213562)
F_1_847 equ DESCALE(1984016188, 30 - CONST_BITS)  ; FIX(1.847759065)
F_2_613 equ DESCALE(2805822602, 30 - CONST_BITS)  ; FIX(2.613125930)
F_1_613 equ (F_2_613 - (1 << CONST_BITS))       ; FIX(2.613125930) - FIX(1)
%endif

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

; PRE_MULTIPLY_SCALE_BITS <= 2 (to avoid overflow)
; CONST_BITS + CONST_SHIFT + PRE_MULTIPLY_SCALE_BITS == 16 (for pmulhw)

%define PRE_MULTIPLY_SCALE_BITS  2
%define CONST_SHIFT              (16 - PRE_MULTIPLY_SCALE_BITS - CONST_BITS)

    ALIGNZ      32
    GLOBAL_DATA(jconst_idct_ifast_mmx)

EXTN(jconst_idct_ifast_mmx):

PW_F1414       times 4 dw  F_1_414 << CONST_SHIFT
PW_F1847       times 4 dw  F_1_847 << CONST_SHIFT
PW_MF1613      times 4 dw -F_1_613 << CONST_SHIFT
PW_F1082       times 4 dw  F_1_082 << CONST_SHIFT
PB_CENTERJSAMP times 8 db  CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform dequantization and inverse DCT on one block of coefficients.
;
; GLOBAL(void)
; jsimd_idct_ifast_mmx(void *dct_table, JCOEFPTR coef_block,
;                      JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; jpeg_component_info *compptr
%define coef_block(b)  (b) + 12         ; JCOEFPTR coef_block
%define output_buf(b)  (b) + 16         ; JSAMPARRAY output_buf
%define output_col(b)  (b) + 20         ; JDIMENSION output_col

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_MMWORD
                                        ; mmword wk[WK_NUM]
%define WK_NUM         2
%define workspace      wk(0) - DCTSIZE2 * SIZEOF_JCOEF
                                        ; JCOEF workspace[DCTSIZE2]

    align       32
    GLOBAL_FUNCTION(jsimd_idct_ifast_mmx)

EXTN(jsimd_idct_ifast_mmx):
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
%ifndef NO_ZERO_COLUMN_TEST_IFAST_MMX
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
    pmullw      mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_IFAST_MULT_TYPE)]

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
    pmullw      mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_IFAST_MULT_TYPE)]
    pmullw      mm1, MMWORD [MMBLOCK(2,0,edx,SIZEOF_IFAST_MULT_TYPE)]
    movq        mm2, MMWORD [MMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    pmullw      mm2, MMWORD [MMBLOCK(4,0,edx,SIZEOF_IFAST_MULT_TYPE)]
    pmullw      mm3, MMWORD [MMBLOCK(6,0,edx,SIZEOF_IFAST_MULT_TYPE)]

    movq        mm4, mm0
    movq        mm5, mm1
    psubw       mm0, mm2                ; mm0=tmp11
    psubw       mm1, mm3
    paddw       mm4, mm2                ; mm4=tmp10
    paddw       mm5, mm3                ; mm5=tmp13

    psllw       mm1, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm1, [GOTOFF(ebx,PW_F1414)]
    psubw       mm1, mm5                ; mm1=tmp12

    movq        mm6, mm4
    movq        mm7, mm0
    psubw       mm4, mm5                ; mm4=tmp3
    psubw       mm0, mm1                ; mm0=tmp2
    paddw       mm6, mm5                ; mm6=tmp0
    paddw       mm7, mm1                ; mm7=tmp1

    movq        MMWORD [wk(1)], mm4     ; wk(1)=tmp3
    movq        MMWORD [wk(0)], mm0     ; wk(0)=tmp2

    ; -- Odd part

    movq        mm2, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    pmullw      mm2, MMWORD [MMBLOCK(1,0,edx,SIZEOF_IFAST_MULT_TYPE)]
    pmullw      mm3, MMWORD [MMBLOCK(3,0,edx,SIZEOF_IFAST_MULT_TYPE)]
    movq        mm5, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    pmullw      mm5, MMWORD [MMBLOCK(5,0,edx,SIZEOF_IFAST_MULT_TYPE)]
    pmullw      mm1, MMWORD [MMBLOCK(7,0,edx,SIZEOF_IFAST_MULT_TYPE)]

    movq        mm4, mm2
    movq        mm0, mm5
    psubw       mm2, mm1                ; mm2=z12
    psubw       mm5, mm3                ; mm5=z10
    paddw       mm4, mm1                ; mm4=z11
    paddw       mm0, mm3                ; mm0=z13

    movq        mm1, mm5                ; mm1=z10(unscaled)
    psllw       mm2, PRE_MULTIPLY_SCALE_BITS
    psllw       mm5, PRE_MULTIPLY_SCALE_BITS

    movq        mm3, mm4
    psubw       mm4, mm0
    paddw       mm3, mm0                ; mm3=tmp7

    psllw       mm4, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm4, [GOTOFF(ebx,PW_F1414)]  ; mm4=tmp11

    ; To avoid overflow...
    ;
    ; (Original)
    ; tmp12 = -2.613125930 * z10 + z5;
    ;
    ; (This implementation)
    ; tmp12 = (-1.613125930 - 1) * z10 + z5;
    ;       = -1.613125930 * z10 - z10 + z5;

    movq        mm0, mm5
    paddw       mm5, mm2
    pmulhw      mm5, [GOTOFF(ebx,PW_F1847)]   ; mm5=z5
    pmulhw      mm0, [GOTOFF(ebx,PW_MF1613)]
    pmulhw      mm2, [GOTOFF(ebx,PW_F1082)]
    psubw       mm0, mm1
    psubw       mm2, mm5                ; mm2=tmp10
    paddw       mm0, mm5                ; mm0=tmp12

    ; -- Final output stage

    psubw       mm0, mm3                ; mm0=tmp6
    movq        mm1, mm6
    movq        mm5, mm7
    paddw       mm6, mm3                ; mm6=data0=(00 01 02 03)
    paddw       mm7, mm0                ; mm7=data1=(10 11 12 13)
    psubw       mm1, mm3                ; mm1=data7=(70 71 72 73)
    psubw       mm5, mm0                ; mm5=data6=(60 61 62 63)
    psubw       mm4, mm0                ; mm4=tmp5

    movq        mm3, mm6                ; transpose coefficients(phase 1)
    punpcklwd   mm6, mm7                ; mm6=(00 10 01 11)
    punpckhwd   mm3, mm7                ; mm3=(02 12 03 13)
    movq        mm0, mm5                ; transpose coefficients(phase 1)
    punpcklwd   mm5, mm1                ; mm5=(60 70 61 71)
    punpckhwd   mm0, mm1                ; mm0=(62 72 63 73)

    movq        mm7, MMWORD [wk(0)]     ; mm7=tmp2
    movq        mm1, MMWORD [wk(1)]     ; mm1=tmp3

    movq        MMWORD [wk(0)], mm5     ; wk(0)=(60 70 61 71)
    movq        MMWORD [wk(1)], mm0     ; wk(1)=(62 72 63 73)

    paddw       mm2, mm4                ; mm2=tmp4
    movq        mm5, mm7
    movq        mm0, mm1
    paddw       mm7, mm4                ; mm7=data2=(20 21 22 23)
    paddw       mm1, mm2                ; mm1=data4=(40 41 42 43)
    psubw       mm5, mm4                ; mm5=data5=(50 51 52 53)
    psubw       mm0, mm2                ; mm0=data3=(30 31 32 33)

    movq        mm4, mm7                ; transpose coefficients(phase 1)
    punpcklwd   mm7, mm0                ; mm7=(20 30 21 31)
    punpckhwd   mm4, mm0                ; mm4=(22 32 23 33)
    movq        mm2, mm1                ; transpose coefficients(phase 1)
    punpcklwd   mm1, mm5                ; mm1=(40 50 41 51)
    punpckhwd   mm2, mm5                ; mm2=(42 52 43 53)

    movq        mm0, mm6                ; transpose coefficients(phase 2)
    punpckldq   mm6, mm7                ; mm6=(00 10 20 30)
    punpckhdq   mm0, mm7                ; mm0=(01 11 21 31)
    movq        mm5, mm3                ; transpose coefficients(phase 2)
    punpckldq   mm3, mm4                ; mm3=(02 12 22 32)
    punpckhdq   mm5, mm4                ; mm5=(03 13 23 33)

    movq        mm7, MMWORD [wk(0)]     ; mm7=(60 70 61 71)
    movq        mm4, MMWORD [wk(1)]     ; mm4=(62 72 63 73)

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_JCOEF)], mm6
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_JCOEF)], mm0
    movq        MMWORD [MMBLOCK(2,0,edi,SIZEOF_JCOEF)], mm3
    movq        MMWORD [MMBLOCK(3,0,edi,SIZEOF_JCOEF)], mm5

    movq        mm6, mm1                ; transpose coefficients(phase 2)
    punpckldq   mm1, mm7                ; mm1=(40 50 60 70)
    punpckhdq   mm6, mm7                ; mm6=(41 51 61 71)
    movq        mm0, mm2                ; transpose coefficients(phase 2)
    punpckldq   mm2, mm4                ; mm2=(42 52 62 72)
    punpckhdq   mm0, mm4                ; mm0=(43 53 63 73)

    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_JCOEF)], mm1
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_JCOEF)], mm6
    movq        MMWORD [MMBLOCK(2,1,edi,SIZEOF_JCOEF)], mm2
    movq        MMWORD [MMBLOCK(3,1,edi,SIZEOF_JCOEF)], mm0

.nextcolumn:
    add         esi, byte 4*SIZEOF_JCOEF            ; coef_block
    add         edx, byte 4*SIZEOF_IFAST_MULT_TYPE  ; quantptr
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

    movq        mm4, mm0
    movq        mm5, mm1
    psubw       mm0, mm2                ; mm0=tmp11
    psubw       mm1, mm3
    paddw       mm4, mm2                ; mm4=tmp10
    paddw       mm5, mm3                ; mm5=tmp13

    psllw       mm1, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm1, [GOTOFF(ebx,PW_F1414)]
    psubw       mm1, mm5                ; mm1=tmp12

    movq        mm6, mm4
    movq        mm7, mm0
    psubw       mm4, mm5                ; mm4=tmp3
    psubw       mm0, mm1                ; mm0=tmp2
    paddw       mm6, mm5                ; mm6=tmp0
    paddw       mm7, mm1                ; mm7=tmp1

    movq        MMWORD [wk(1)], mm4     ; wk(1)=tmp3
    movq        MMWORD [wk(0)], mm0     ; wk(0)=tmp2

    ; -- Odd part

    movq        mm2, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    movq        mm5, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]

    movq        mm4, mm2
    movq        mm0, mm5
    psubw       mm2, mm1                ; mm2=z12
    psubw       mm5, mm3                ; mm5=z10
    paddw       mm4, mm1                ; mm4=z11
    paddw       mm0, mm3                ; mm0=z13

    movq        mm1, mm5                ; mm1=z10(unscaled)
    psllw       mm2, PRE_MULTIPLY_SCALE_BITS
    psllw       mm5, PRE_MULTIPLY_SCALE_BITS

    movq        mm3, mm4
    psubw       mm4, mm0
    paddw       mm3, mm0                ; mm3=tmp7

    psllw       mm4, PRE_MULTIPLY_SCALE_BITS
    pmulhw      mm4, [GOTOFF(ebx,PW_F1414)]  ; mm4=tmp11

    ; To avoid overflow...
    ;
    ; (Original)
    ; tmp12 = -2.613125930 * z10 + z5;
    ;
    ; (This implementation)
    ; tmp12 = (-1.613125930 - 1) * z10 + z5;
    ;       = -1.613125930 * z10 - z10 + z5;

    movq        mm0, mm5
    paddw       mm5, mm2
    pmulhw      mm5, [GOTOFF(ebx,PW_F1847)]   ; mm5=z5
    pmulhw      mm0, [GOTOFF(ebx,PW_MF1613)]
    pmulhw      mm2, [GOTOFF(ebx,PW_F1082)]
    psubw       mm0, mm1
    psubw       mm2, mm5                ; mm2=tmp10
    paddw       mm0, mm5                ; mm0=tmp12

    ; -- Final output stage

    psubw       mm0, mm3                ; mm0=tmp6
    movq        mm1, mm6
    movq        mm5, mm7
    paddw       mm6, mm3                ; mm6=data0=(00 10 20 30)
    paddw       mm7, mm0                ; mm7=data1=(01 11 21 31)
    psraw       mm6, (PASS1_BITS+3)     ; descale
    psraw       mm7, (PASS1_BITS+3)     ; descale
    psubw       mm1, mm3                ; mm1=data7=(07 17 27 37)
    psubw       mm5, mm0                ; mm5=data6=(06 16 26 36)
    psraw       mm1, (PASS1_BITS+3)     ; descale
    psraw       mm5, (PASS1_BITS+3)     ; descale
    psubw       mm4, mm0                ; mm4=tmp5

    packsswb    mm6, mm5                ; mm6=(00 10 20 30 06 16 26 36)
    packsswb    mm7, mm1                ; mm7=(01 11 21 31 07 17 27 37)

    movq        mm3, MMWORD [wk(0)]     ; mm3=tmp2
    movq        mm0, MMWORD [wk(1)]     ; mm0=tmp3

    paddw       mm2, mm4                ; mm2=tmp4
    movq        mm5, mm3
    movq        mm1, mm0
    paddw       mm3, mm4                ; mm3=data2=(02 12 22 32)
    paddw       mm0, mm2                ; mm0=data4=(04 14 24 34)
    psraw       mm3, (PASS1_BITS+3)     ; descale
    psraw       mm0, (PASS1_BITS+3)     ; descale
    psubw       mm5, mm4                ; mm5=data5=(05 15 25 35)
    psubw       mm1, mm2                ; mm1=data3=(03 13 23 33)
    psraw       mm5, (PASS1_BITS+3)     ; descale
    psraw       mm1, (PASS1_BITS+3)     ; descale

    movq        mm4, [GOTOFF(ebx,PB_CENTERJSAMP)]  ; mm4=[PB_CENTERJSAMP]

    packsswb    mm3, mm0                ; mm3=(02 12 22 32 04 14 24 34)
    packsswb    mm1, mm5                ; mm1=(03 13 23 33 05 15 25 35)

    paddb       mm6, mm4
    paddb       mm7, mm4
    paddb       mm3, mm4
    paddb       mm1, mm4

    movq        mm2, mm6                ; transpose coefficients(phase 1)
    punpcklbw   mm6, mm7                ; mm6=(00 01 10 11 20 21 30 31)
    punpckhbw   mm2, mm7                ; mm2=(06 07 16 17 26 27 36 37)
    movq        mm0, mm3                ; transpose coefficients(phase 1)
    punpcklbw   mm3, mm1                ; mm3=(02 03 12 13 22 23 32 33)
    punpckhbw   mm0, mm1                ; mm0=(04 05 14 15 24 25 34 35)

    movq        mm5, mm6                ; transpose coefficients(phase 2)
    punpcklwd   mm6, mm3                ; mm6=(00 01 02 03 10 11 12 13)
    punpckhwd   mm5, mm3                ; mm5=(20 21 22 23 30 31 32 33)
    movq        mm4, mm0                ; transpose coefficients(phase 2)
    punpcklwd   mm0, mm2                ; mm0=(04 05 06 07 14 15 16 17)
    punpckhwd   mm4, mm2                ; mm4=(24 25 26 27 34 35 36 37)

    movq        mm7, mm6                ; transpose coefficients(phase 3)
    punpckldq   mm6, mm0                ; mm6=(00 01 02 03 04 05 06 07)
    punpckhdq   mm7, mm0                ; mm7=(10 11 12 13 14 15 16 17)
    movq        mm1, mm5                ; transpose coefficients(phase 3)
    punpckldq   mm5, mm4                ; mm5=(20 21 22 23 24 25 26 27)
    punpckhdq   mm1, mm4                ; mm1=(30 31 32 33 34 35 36 37)

    PUSHPIC     ebx                     ; save GOT address

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]
    mov         ebx, JSAMPROW [edi+1*SIZEOF_JSAMPROW]
    movq        MMWORD [edx+eax*SIZEOF_JSAMPLE], mm6
    movq        MMWORD [ebx+eax*SIZEOF_JSAMPLE], mm7
    mov         edx, JSAMPROW [edi+2*SIZEOF_JSAMPROW]
    mov         ebx, JSAMPROW [edi+3*SIZEOF_JSAMPROW]
    movq        MMWORD [edx+eax*SIZEOF_JSAMPLE], mm5
    movq        MMWORD [ebx+eax*SIZEOF_JSAMPLE], mm1

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
