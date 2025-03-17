;
; jidctflt.asm - floating-point IDCT (3DNow! & MMX)
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
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_idct_float_3dnow)

EXTN(jconst_idct_float_3dnow):

PD_1_414        times 2 dd 1.414213562373095048801689
PD_1_847        times 2 dd 1.847759065022573512256366
PD_1_082        times 2 dd 1.082392200292393968799446
PD_2_613        times 2 dd 2.613125929752753055713286
PD_RNDINT_MAGIC times 2 dd 100663296.0  ; (float)(0x00C00000 << 3)
PB_CENTERJSAMP  times 8 db CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform dequantization and inverse DCT on one block of coefficients.
;
; GLOBAL(void)
; jsimd_idct_float_3dnow(void *dct_table, JCOEFPTR coef_block,
;                        JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; void *dct_table
%define coef_block(b)  (b) + 12         ; JCOEFPTR coef_block
%define output_buf(b)  (b) + 16         ; JSAMPARRAY output_buf
%define output_col(b)  (b) + 20         ; JDIMENSION output_col

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_MMWORD
                                        ; mmword wk[WK_NUM]
%define WK_NUM         2
%define workspace      wk(0) - DCTSIZE2 * SIZEOF_FAST_FLOAT
                                        ; FAST_FLOAT workspace[DCTSIZE2]

    align       32
    GLOBAL_FUNCTION(jsimd_idct_float_3dnow)

EXTN(jsimd_idct_float_3dnow):
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
    lea         edi, [workspace]                 ; FAST_FLOAT *wsptr
    mov         ecx, DCTSIZE/2                   ; ctr
    ALIGNX      16, 7
.columnloop:
%ifndef NO_ZERO_COLUMN_TEST_FLOAT_3DNOW
    mov         eax, dword [DWBLOCK(1,0,esi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,esi,SIZEOF_JCOEF)]
    jnz         short .columnDCT

    PUSHPIC     ebx                     ; save GOT address
    mov         ebx, dword [DWBLOCK(3,0,esi,SIZEOF_JCOEF)]
    mov         eax, dword [DWBLOCK(4,0,esi,SIZEOF_JCOEF)]
    or          ebx, dword [DWBLOCK(5,0,esi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(6,0,esi,SIZEOF_JCOEF)]
    or          ebx, dword [DWBLOCK(7,0,esi,SIZEOF_JCOEF)]
    or          eax, ebx
    POPPIC      ebx                     ; restore GOT address
    jnz         short .columnDCT

    ; -- AC terms all zero

    movd        mm0, dword [DWBLOCK(0,0,esi,SIZEOF_JCOEF)]

    punpcklwd   mm0, mm0
    psrad       mm0, (DWORD_BIT-WORD_BIT)
    pi2fd       mm0, mm0

    pfmul       mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movq        mm1, mm0
    punpckldq   mm0, mm0
    punpckhdq   mm1, mm1

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_FAST_FLOAT)], mm0
    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_FAST_FLOAT)], mm0
    movq        MMWORD [MMBLOCK(0,2,edi,SIZEOF_FAST_FLOAT)], mm0
    movq        MMWORD [MMBLOCK(0,3,edi,SIZEOF_FAST_FLOAT)], mm0
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_FAST_FLOAT)], mm1
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_FAST_FLOAT)], mm1
    movq        MMWORD [MMBLOCK(1,2,edi,SIZEOF_FAST_FLOAT)], mm1
    movq        MMWORD [MMBLOCK(1,3,edi,SIZEOF_FAST_FLOAT)], mm1
    jmp         near .nextcolumn
    ALIGNX      16, 7
%endif
.columnDCT:

    ; -- Even part

    movd        mm0, dword [DWBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movd        mm1, dword [DWBLOCK(2,0,esi,SIZEOF_JCOEF)]
    movd        mm2, dword [DWBLOCK(4,0,esi,SIZEOF_JCOEF)]
    movd        mm3, dword [DWBLOCK(6,0,esi,SIZEOF_JCOEF)]

    punpcklwd   mm0, mm0
    punpcklwd   mm1, mm1
    psrad       mm0, (DWORD_BIT-WORD_BIT)
    psrad       mm1, (DWORD_BIT-WORD_BIT)
    pi2fd       mm0, mm0
    pi2fd       mm1, mm1

    pfmul       mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    pfmul       mm1, MMWORD [MMBLOCK(2,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    punpcklwd   mm2, mm2
    punpcklwd   mm3, mm3
    psrad       mm2, (DWORD_BIT-WORD_BIT)
    psrad       mm3, (DWORD_BIT-WORD_BIT)
    pi2fd       mm2, mm2
    pi2fd       mm3, mm3

    pfmul       mm2, MMWORD [MMBLOCK(4,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    pfmul       mm3, MMWORD [MMBLOCK(6,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movq        mm4, mm0
    movq        mm5, mm1
    pfsub       mm0, mm2                ; mm0=tmp11
    pfsub       mm1, mm3
    pfadd       mm4, mm2                ; mm4=tmp10
    pfadd       mm5, mm3                ; mm5=tmp13

    pfmul       mm1, [GOTOFF(ebx,PD_1_414)]
    pfsub       mm1, mm5                ; mm1=tmp12

    movq        mm6, mm4
    movq        mm7, mm0
    pfsub       mm4, mm5                ; mm4=tmp3
    pfsub       mm0, mm1                ; mm0=tmp2
    pfadd       mm6, mm5                ; mm6=tmp0
    pfadd       mm7, mm1                ; mm7=tmp1

    movq        MMWORD [wk(1)], mm4     ; tmp3
    movq        MMWORD [wk(0)], mm0     ; tmp2

    ; -- Odd part

    movd        mm2, dword [DWBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movd        mm3, dword [DWBLOCK(3,0,esi,SIZEOF_JCOEF)]
    movd        mm5, dword [DWBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movd        mm1, dword [DWBLOCK(7,0,esi,SIZEOF_JCOEF)]

    punpcklwd   mm2, mm2
    punpcklwd   mm3, mm3
    psrad       mm2, (DWORD_BIT-WORD_BIT)
    psrad       mm3, (DWORD_BIT-WORD_BIT)
    pi2fd       mm2, mm2
    pi2fd       mm3, mm3

    pfmul       mm2, MMWORD [MMBLOCK(1,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    pfmul       mm3, MMWORD [MMBLOCK(3,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    punpcklwd   mm5, mm5
    punpcklwd   mm1, mm1
    psrad       mm5, (DWORD_BIT-WORD_BIT)
    psrad       mm1, (DWORD_BIT-WORD_BIT)
    pi2fd       mm5, mm5
    pi2fd       mm1, mm1

    pfmul       mm5, MMWORD [MMBLOCK(5,0,edx,SIZEOF_FLOAT_MULT_TYPE)]
    pfmul       mm1, MMWORD [MMBLOCK(7,0,edx,SIZEOF_FLOAT_MULT_TYPE)]

    movq        mm4, mm2
    movq        mm0, mm5
    pfadd       mm2, mm1                ; mm2=z11
    pfadd       mm5, mm3                ; mm5=z13
    pfsub       mm4, mm1                ; mm4=z12
    pfsub       mm0, mm3                ; mm0=z10

    movq        mm1, mm2
    pfsub       mm2, mm5
    pfadd       mm1, mm5                ; mm1=tmp7

    pfmul       mm2, [GOTOFF(ebx,PD_1_414)]  ; mm2=tmp11

    movq        mm3, mm0
    pfadd       mm0, mm4
    pfmul       mm0, [GOTOFF(ebx,PD_1_847)]  ; mm0=z5
    pfmul       mm3, [GOTOFF(ebx,PD_2_613)]  ; mm3=(z10 * 2.613125930)
    pfmul       mm4, [GOTOFF(ebx,PD_1_082)]  ; mm4=(z12 * 1.082392200)
    pfsubr      mm3, mm0                     ; mm3=tmp12
    pfsub       mm4, mm0                     ; mm4=tmp10

    ; -- Final output stage

    pfsub       mm3, mm1                ; mm3=tmp6
    movq        mm5, mm6
    movq        mm0, mm7
    pfadd       mm6, mm1                ; mm6=data0=(00 01)
    pfadd       mm7, mm3                ; mm7=data1=(10 11)
    pfsub       mm5, mm1                ; mm5=data7=(70 71)
    pfsub       mm0, mm3                ; mm0=data6=(60 61)
    pfsub       mm2, mm3                ; mm2=tmp5

    movq        mm1, mm6                ; transpose coefficients
    punpckldq   mm6, mm7                ; mm6=(00 10)
    punpckhdq   mm1, mm7                ; mm1=(01 11)
    movq        mm3, mm0                ; transpose coefficients
    punpckldq   mm0, mm5                ; mm0=(60 70)
    punpckhdq   mm3, mm5                ; mm3=(61 71)

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_FAST_FLOAT)], mm6
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_FAST_FLOAT)], mm1
    movq        MMWORD [MMBLOCK(0,3,edi,SIZEOF_FAST_FLOAT)], mm0
    movq        MMWORD [MMBLOCK(1,3,edi,SIZEOF_FAST_FLOAT)], mm3

    movq        mm7, MMWORD [wk(0)]     ; mm7=tmp2
    movq        mm5, MMWORD [wk(1)]     ; mm5=tmp3

    pfadd       mm4, mm2                ; mm4=tmp4
    movq        mm6, mm7
    movq        mm1, mm5
    pfadd       mm7, mm2                ; mm7=data2=(20 21)
    pfadd       mm5, mm4                ; mm5=data4=(40 41)
    pfsub       mm6, mm2                ; mm6=data5=(50 51)
    pfsub       mm1, mm4                ; mm1=data3=(30 31)

    movq        mm0, mm7                ; transpose coefficients
    punpckldq   mm7, mm1                ; mm7=(20 30)
    punpckhdq   mm0, mm1                ; mm0=(21 31)
    movq        mm3, mm5                ; transpose coefficients
    punpckldq   mm5, mm6                ; mm5=(40 50)
    punpckhdq   mm3, mm6                ; mm3=(41 51)

    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_FAST_FLOAT)], mm7
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_FAST_FLOAT)], mm0
    movq        MMWORD [MMBLOCK(0,2,edi,SIZEOF_FAST_FLOAT)], mm5
    movq        MMWORD [MMBLOCK(1,2,edi,SIZEOF_FAST_FLOAT)], mm3

.nextcolumn:
    add         esi, byte 2*SIZEOF_JCOEF               ; coef_block
    add         edx, byte 2*SIZEOF_FLOAT_MULT_TYPE     ; quantptr
    add         edi, byte 2*DCTSIZE*SIZEOF_FAST_FLOAT  ; wsptr
    dec         ecx                                    ; ctr
    jnz         near .columnloop

    ; -- Prefetch the next coefficient block

    prefetch [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 0*32]
    prefetch [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 1*32]
    prefetch [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 2*32]
    prefetch [esi + (DCTSIZE2-8)*SIZEOF_JCOEF + 3*32]

    ; ---- Pass 2: process rows from work array, store into output array.

    mov         eax, [original_ebp]
    lea         esi, [workspace]                   ; FAST_FLOAT *wsptr
    mov         edi, JSAMPARRAY [output_buf(eax)]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [output_col(eax)]
    mov         ecx, DCTSIZE/2                     ; ctr
    ALIGNX      16, 7
.rowloop:

    ; -- Even part

    movq        mm0, MMWORD [MMBLOCK(0,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm1, MMWORD [MMBLOCK(2,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm2, MMWORD [MMBLOCK(4,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(6,0,esi,SIZEOF_FAST_FLOAT)]

    movq        mm4, mm0
    movq        mm5, mm1
    pfsub       mm0, mm2                ; mm0=tmp11
    pfsub       mm1, mm3
    pfadd       mm4, mm2                ; mm4=tmp10
    pfadd       mm5, mm3                ; mm5=tmp13

    pfmul       mm1, [GOTOFF(ebx,PD_1_414)]
    pfsub       mm1, mm5                ; mm1=tmp12

    movq        mm6, mm4
    movq        mm7, mm0
    pfsub       mm4, mm5                ; mm4=tmp3
    pfsub       mm0, mm1                ; mm0=tmp2
    pfadd       mm6, mm5                ; mm6=tmp0
    pfadd       mm7, mm1                ; mm7=tmp1

    movq        MMWORD [wk(1)], mm4     ; tmp3
    movq        MMWORD [wk(0)], mm0     ; tmp2

    ; -- Odd part

    movq        mm2, MMWORD [MMBLOCK(1,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(3,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm5, MMWORD [MMBLOCK(5,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm1, MMWORD [MMBLOCK(7,0,esi,SIZEOF_FAST_FLOAT)]

    movq        mm4, mm2
    movq        mm0, mm5
    pfadd       mm2, mm1                ; mm2=z11
    pfadd       mm5, mm3                ; mm5=z13
    pfsub       mm4, mm1                ; mm4=z12
    pfsub       mm0, mm3                ; mm0=z10

    movq        mm1, mm2
    pfsub       mm2, mm5
    pfadd       mm1, mm5                ; mm1=tmp7

    pfmul       mm2, [GOTOFF(ebx,PD_1_414)]  ; mm2=tmp11

    movq        mm3, mm0
    pfadd       mm0, mm4
    pfmul       mm0, [GOTOFF(ebx,PD_1_847)]  ; mm0=z5
    pfmul       mm3, [GOTOFF(ebx,PD_2_613)]  ; mm3=(z10 * 2.613125930)
    pfmul       mm4, [GOTOFF(ebx,PD_1_082)]  ; mm4=(z12 * 1.082392200)
    pfsubr      mm3, mm0                     ; mm3=tmp12
    pfsub       mm4, mm0                     ; mm4=tmp10

    ; -- Final output stage

    pfsub       mm3, mm1                ; mm3=tmp6
    movq        mm5, mm6
    movq        mm0, mm7
    pfadd       mm6, mm1                ; mm6=data0=(00 10)
    pfadd       mm7, mm3                ; mm7=data1=(01 11)
    pfsub       mm5, mm1                ; mm5=data7=(07 17)
    pfsub       mm0, mm3                ; mm0=data6=(06 16)
    pfsub       mm2, mm3                ; mm2=tmp5

    movq        mm1, [GOTOFF(ebx,PD_RNDINT_MAGIC)]  ; mm1=[PD_RNDINT_MAGIC]
    pcmpeqd     mm3, mm3
    psrld       mm3, WORD_BIT           ; mm3={0xFFFF 0x0000 0xFFFF 0x0000}

    pfadd       mm6, mm1                ; mm6=roundint(data0/8)=(00 ** 10 **)
    pfadd       mm7, mm1                ; mm7=roundint(data1/8)=(01 ** 11 **)
    pfadd       mm0, mm1                ; mm0=roundint(data6/8)=(06 ** 16 **)
    pfadd       mm5, mm1                ; mm5=roundint(data7/8)=(07 ** 17 **)

    pand        mm6, mm3                ; mm6=(00 -- 10 --)
    pslld       mm7, WORD_BIT           ; mm7=(-- 01 -- 11)
    pand        mm0, mm3                ; mm0=(06 -- 16 --)
    pslld       mm5, WORD_BIT           ; mm5=(-- 07 -- 17)
    por         mm6, mm7                ; mm6=(00 01 10 11)
    por         mm0, mm5                ; mm0=(06 07 16 17)

    movq        mm1, MMWORD [wk(0)]     ; mm1=tmp2
    movq        mm3, MMWORD [wk(1)]     ; mm3=tmp3

    pfadd       mm4, mm2                ; mm4=tmp4
    movq        mm7, mm1
    movq        mm5, mm3
    pfadd       mm1, mm2                ; mm1=data2=(02 12)
    pfadd       mm3, mm4                ; mm3=data4=(04 14)
    pfsub       mm7, mm2                ; mm7=data5=(05 15)
    pfsub       mm5, mm4                ; mm5=data3=(03 13)

    movq        mm2, [GOTOFF(ebx,PD_RNDINT_MAGIC)]  ; mm2=[PD_RNDINT_MAGIC]
    pcmpeqd     mm4, mm4
    psrld       mm4, WORD_BIT           ; mm4={0xFFFF 0x0000 0xFFFF 0x0000}

    pfadd       mm3, mm2                ; mm3=roundint(data4/8)=(04 ** 14 **)
    pfadd       mm7, mm2                ; mm7=roundint(data5/8)=(05 ** 15 **)
    pfadd       mm1, mm2                ; mm1=roundint(data2/8)=(02 ** 12 **)
    pfadd       mm5, mm2                ; mm5=roundint(data3/8)=(03 ** 13 **)

    pand        mm3, mm4                ; mm3=(04 -- 14 --)
    pslld       mm7, WORD_BIT           ; mm7=(-- 05 -- 15)
    pand        mm1, mm4                ; mm1=(02 -- 12 --)
    pslld       mm5, WORD_BIT           ; mm5=(-- 03 -- 13)
    por         mm3, mm7                ; mm3=(04 05 14 15)
    por         mm1, mm5                ; mm1=(02 03 12 13)

    movq        mm2, [GOTOFF(ebx,PB_CENTERJSAMP)]  ; mm2=[PB_CENTERJSAMP]

    packsswb    mm6, mm3                ; mm6=(00 01 10 11 04 05 14 15)
    packsswb    mm1, mm0                ; mm1=(02 03 12 13 06 07 16 17)
    paddb       mm6, mm2
    paddb       mm1, mm2

    movq        mm4, mm6                ; transpose coefficients(phase 2)
    punpcklwd   mm6, mm1                ; mm6=(00 01 02 03 10 11 12 13)
    punpckhwd   mm4, mm1                ; mm4=(04 05 06 07 14 15 16 17)

    movq        mm7, mm6                ; transpose coefficients(phase 3)
    punpckldq   mm6, mm4                ; mm6=(00 01 02 03 04 05 06 07)
    punpckhdq   mm7, mm4                ; mm7=(10 11 12 13 14 15 16 17)

    PUSHPIC     ebx                     ; save GOT address

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]
    mov         ebx, JSAMPROW [edi+1*SIZEOF_JSAMPROW]
    movq        MMWORD [edx+eax*SIZEOF_JSAMPLE], mm6
    movq        MMWORD [ebx+eax*SIZEOF_JSAMPLE], mm7

    POPPIC      ebx                     ; restore GOT address

    add         esi, byte 2*SIZEOF_FAST_FLOAT  ; wsptr
    add         edi, byte 2*SIZEOF_JSAMPROW
    dec         ecx                            ; ctr
    jnz         near .rowloop

    femms                               ; empty MMX/3DNow! state

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
