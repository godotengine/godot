;
; jidctred.asm - reduced-size IDCT (MMX)
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
; This file contains inverse-DCT routines that produce reduced-size
; output: either 4x4 or 2x2 pixels from an 8x8 DCT block.
; The following code is based directly on the IJG's original jidctred.c;
; see the jidctred.c for more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%define CONST_BITS    13
%define PASS1_BITS    2

%define DESCALE_P1_4  (CONST_BITS - PASS1_BITS + 1)
%define DESCALE_P2_4  (CONST_BITS + PASS1_BITS + 3 + 1)
%define DESCALE_P1_2  (CONST_BITS - PASS1_BITS + 2)
%define DESCALE_P2_2  (CONST_BITS + PASS1_BITS + 3 + 2)

%if CONST_BITS == 13
F_0_211 equ  1730  ; FIX(0.211164243)
F_0_509 equ  4176  ; FIX(0.509795579)
F_0_601 equ  4926  ; FIX(0.601344887)
F_0_720 equ  5906  ; FIX(0.720959822)
F_0_765 equ  6270  ; FIX(0.765366865)
F_0_850 equ  6967  ; FIX(0.850430095)
F_0_899 equ  7373  ; FIX(0.899976223)
F_1_061 equ  8697  ; FIX(1.061594337)
F_1_272 equ 10426  ; FIX(1.272758580)
F_1_451 equ 11893  ; FIX(1.451774981)
F_1_847 equ 15137  ; FIX(1.847759065)
F_2_172 equ 17799  ; FIX(2.172734803)
F_2_562 equ 20995  ; FIX(2.562915447)
F_3_624 equ 29692  ; FIX(3.624509785)
%else
; NASM cannot do compile-time arithmetic on floating-point constants.
%define DESCALE(x, n)  (((x) + (1 << ((n) - 1))) >> (n))
F_0_211 equ DESCALE( 226735879, 30 - CONST_BITS)  ; FIX(0.211164243)
F_0_509 equ DESCALE( 547388834, 30 - CONST_BITS)  ; FIX(0.509795579)
F_0_601 equ DESCALE( 645689155, 30 - CONST_BITS)  ; FIX(0.601344887)
F_0_720 equ DESCALE( 774124714, 30 - CONST_BITS)  ; FIX(0.720959822)
F_0_765 equ DESCALE( 821806413, 30 - CONST_BITS)  ; FIX(0.765366865)
F_0_850 equ DESCALE( 913142361, 30 - CONST_BITS)  ; FIX(0.850430095)
F_0_899 equ DESCALE( 966342111, 30 - CONST_BITS)  ; FIX(0.899976223)
F_1_061 equ DESCALE(1139878239, 30 - CONST_BITS)  ; FIX(1.061594337)
F_1_272 equ DESCALE(1366614119, 30 - CONST_BITS)  ; FIX(1.272758580)
F_1_451 equ DESCALE(1558831516, 30 - CONST_BITS)  ; FIX(1.451774981)
F_1_847 equ DESCALE(1984016188, 30 - CONST_BITS)  ; FIX(1.847759065)
F_2_172 equ DESCALE(2332956230, 30 - CONST_BITS)  ; FIX(2.172734803)
F_2_562 equ DESCALE(2751909506, 30 - CONST_BITS)  ; FIX(2.562915447)
F_3_624 equ DESCALE(3891787747, 30 - CONST_BITS)  ; FIX(3.624509785)
%endif

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_idct_red_mmx)

EXTN(jconst_idct_red_mmx):

PW_F184_MF076   times 2 dw  F_1_847, -F_0_765
PW_F256_F089    times 2 dw  F_2_562,  F_0_899
PW_F106_MF217   times 2 dw  F_1_061, -F_2_172
PW_MF060_MF050  times 2 dw -F_0_601, -F_0_509
PW_F145_MF021   times 2 dw  F_1_451, -F_0_211
PW_F362_MF127   times 2 dw  F_3_624, -F_1_272
PW_F085_MF072   times 2 dw  F_0_850, -F_0_720
PD_DESCALE_P1_4 times 2 dd  1 << (DESCALE_P1_4 - 1)
PD_DESCALE_P2_4 times 2 dd  1 << (DESCALE_P2_4 - 1)
PD_DESCALE_P1_2 times 2 dd  1 << (DESCALE_P1_2 - 1)
PD_DESCALE_P2_2 times 2 dd  1 << (DESCALE_P2_2 - 1)
PB_CENTERJSAMP  times 8 db  CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform dequantization and inverse DCT on one block of coefficients,
; producing a reduced-size 4x4 output block.
;
; GLOBAL(void)
; jsimd_idct_4x4_mmx(void *dct_table, JCOEFPTR coef_block,
;                    JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; void *dct_table
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
    GLOBAL_FUNCTION(jsimd_idct_4x4_mmx)

EXTN(jsimd_idct_4x4_mmx):
    push        ebp
    mov         eax, esp                    ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_MMWORD)  ; align to 64 bits
    mov         [esp], eax
    mov         ebp, esp                    ; ebp = aligned ebp
    lea         esp, [workspace]
    PUSHPIC     ebx
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
%ifndef NO_ZERO_COLUMN_TEST_4X4_MMX
    mov         eax, dword [DWBLOCK(1,0,esi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,esi,SIZEOF_JCOEF)]
    jnz         short .columnDCT

    movq        mm0, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    por         mm0, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    por         mm1, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    por         mm0, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    por         mm1, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    por         mm0, mm1
    packsswb    mm0, mm0
    movd        eax, mm0
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
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_JCOEF)], mm1
    movq        MMWORD [MMBLOCK(2,0,edi,SIZEOF_JCOEF)], mm2
    movq        MMWORD [MMBLOCK(3,0,edi,SIZEOF_JCOEF)], mm3
    jmp         near .nextcolumn
    ALIGNX      16, 7
%endif
.columnDCT:

    ; -- Odd part

    movq        mm0, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    pmullw      mm0, MMWORD [MMBLOCK(1,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm1, MMWORD [MMBLOCK(3,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    movq        mm2, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    pmullw      mm2, MMWORD [MMBLOCK(5,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm3, MMWORD [MMBLOCK(7,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    movq        mm4, mm0
    movq        mm5, mm0
    punpcklwd   mm4, mm1
    punpckhwd   mm5, mm1
    movq        mm0, mm4
    movq        mm1, mm5
    pmaddwd     mm4, [GOTOFF(ebx,PW_F256_F089)]   ; mm4=(tmp2L)
    pmaddwd     mm5, [GOTOFF(ebx,PW_F256_F089)]   ; mm5=(tmp2H)
    pmaddwd     mm0, [GOTOFF(ebx,PW_F106_MF217)]  ; mm0=(tmp0L)
    pmaddwd     mm1, [GOTOFF(ebx,PW_F106_MF217)]  ; mm1=(tmp0H)

    movq        mm6, mm2
    movq        mm7, mm2
    punpcklwd   mm6, mm3
    punpckhwd   mm7, mm3
    movq        mm2, mm6
    movq        mm3, mm7
    pmaddwd     mm6, [GOTOFF(ebx,PW_MF060_MF050)]  ; mm6=(tmp2L)
    pmaddwd     mm7, [GOTOFF(ebx,PW_MF060_MF050)]  ; mm7=(tmp2H)
    pmaddwd     mm2, [GOTOFF(ebx,PW_F145_MF021)]   ; mm2=(tmp0L)
    pmaddwd     mm3, [GOTOFF(ebx,PW_F145_MF021)]   ; mm3=(tmp0H)

    paddd       mm6, mm4                ; mm6=tmp2L
    paddd       mm7, mm5                ; mm7=tmp2H
    paddd       mm2, mm0                ; mm2=tmp0L
    paddd       mm3, mm1                ; mm3=tmp0H

    movq        MMWORD [wk(0)], mm2     ; wk(0)=tmp0L
    movq        MMWORD [wk(1)], mm3     ; wk(1)=tmp0H

    ; -- Even part

    movq        mm4, MMWORD [MMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movq        mm5, MMWORD [MMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    movq        mm0, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    pmullw      mm4, MMWORD [MMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm5, MMWORD [MMBLOCK(2,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm0, MMWORD [MMBLOCK(6,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    pxor        mm1, mm1
    pxor        mm2, mm2
    punpcklwd   mm1, mm4                ; mm1=tmp0L
    punpckhwd   mm2, mm4                ; mm2=tmp0H
    psrad       mm1, (16-CONST_BITS-1)  ; psrad mm1,16 & pslld mm1,CONST_BITS+1
    psrad       mm2, (16-CONST_BITS-1)  ; psrad mm2,16 & pslld mm2,CONST_BITS+1

    movq        mm3, mm5                ; mm5=in2=z2
    punpcklwd   mm5, mm0                ; mm0=in6=z3
    punpckhwd   mm3, mm0
    pmaddwd     mm5, [GOTOFF(ebx,PW_F184_MF076)]  ; mm5=tmp2L
    pmaddwd     mm3, [GOTOFF(ebx,PW_F184_MF076)]  ; mm3=tmp2H

    movq        mm4, mm1
    movq        mm0, mm2
    paddd       mm1, mm5                ; mm1=tmp10L
    paddd       mm2, mm3                ; mm2=tmp10H
    psubd       mm4, mm5                ; mm4=tmp12L
    psubd       mm0, mm3                ; mm0=tmp12H

    ; -- Final output stage

    movq        mm5, mm1
    movq        mm3, mm2
    paddd       mm1, mm6                ; mm1=data0L
    paddd       mm2, mm7                ; mm2=data0H
    psubd       mm5, mm6                ; mm5=data3L
    psubd       mm3, mm7                ; mm3=data3H

    movq        mm6, [GOTOFF(ebx,PD_DESCALE_P1_4)]  ; mm6=[PD_DESCALE_P1_4]

    paddd       mm1, mm6
    paddd       mm2, mm6
    psrad       mm1, DESCALE_P1_4
    psrad       mm2, DESCALE_P1_4
    paddd       mm5, mm6
    paddd       mm3, mm6
    psrad       mm5, DESCALE_P1_4
    psrad       mm3, DESCALE_P1_4

    packssdw    mm1, mm2                ; mm1=data0=(00 01 02 03)
    packssdw    mm5, mm3                ; mm5=data3=(30 31 32 33)

    movq        mm7, MMWORD [wk(0)]     ; mm7=tmp0L
    movq        mm6, MMWORD [wk(1)]     ; mm6=tmp0H

    movq        mm2, mm4
    movq        mm3, mm0
    paddd       mm4, mm7                ; mm4=data1L
    paddd       mm0, mm6                ; mm0=data1H
    psubd       mm2, mm7                ; mm2=data2L
    psubd       mm3, mm6                ; mm3=data2H

    movq        mm7, [GOTOFF(ebx,PD_DESCALE_P1_4)]  ; mm7=[PD_DESCALE_P1_4]

    paddd       mm4, mm7
    paddd       mm0, mm7
    psrad       mm4, DESCALE_P1_4
    psrad       mm0, DESCALE_P1_4
    paddd       mm2, mm7
    paddd       mm3, mm7
    psrad       mm2, DESCALE_P1_4
    psrad       mm3, DESCALE_P1_4

    packssdw    mm4, mm0                ; mm4=data1=(10 11 12 13)
    packssdw    mm2, mm3                ; mm2=data2=(20 21 22 23)

    movq        mm6, mm1                ; transpose coefficients(phase 1)
    punpcklwd   mm1, mm4                ; mm1=(00 10 01 11)
    punpckhwd   mm6, mm4                ; mm6=(02 12 03 13)
    movq        mm7, mm2                ; transpose coefficients(phase 1)
    punpcklwd   mm2, mm5                ; mm2=(20 30 21 31)
    punpckhwd   mm7, mm5                ; mm7=(22 32 23 33)

    movq        mm0, mm1                ; transpose coefficients(phase 2)
    punpckldq   mm1, mm2                ; mm1=(00 10 20 30)
    punpckhdq   mm0, mm2                ; mm0=(01 11 21 31)
    movq        mm3, mm6                ; transpose coefficients(phase 2)
    punpckldq   mm6, mm7                ; mm6=(02 12 22 32)
    punpckhdq   mm3, mm7                ; mm3=(03 13 23 33)

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_JCOEF)], mm1
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_JCOEF)], mm0
    movq        MMWORD [MMBLOCK(2,0,edi,SIZEOF_JCOEF)], mm6
    movq        MMWORD [MMBLOCK(3,0,edi,SIZEOF_JCOEF)], mm3

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

    ; -- Odd part

    movq        mm0, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    movq        mm2, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]

    movq        mm4, mm0
    movq        mm5, mm0
    punpcklwd   mm4, mm1
    punpckhwd   mm5, mm1
    movq        mm0, mm4
    movq        mm1, mm5
    pmaddwd     mm4, [GOTOFF(ebx,PW_F256_F089)]   ; mm4=(tmp2L)
    pmaddwd     mm5, [GOTOFF(ebx,PW_F256_F089)]   ; mm5=(tmp2H)
    pmaddwd     mm0, [GOTOFF(ebx,PW_F106_MF217)]  ; mm0=(tmp0L)
    pmaddwd     mm1, [GOTOFF(ebx,PW_F106_MF217)]  ; mm1=(tmp0H)

    movq        mm6, mm2
    movq        mm7, mm2
    punpcklwd   mm6, mm3
    punpckhwd   mm7, mm3
    movq        mm2, mm6
    movq        mm3, mm7
    pmaddwd     mm6, [GOTOFF(ebx,PW_MF060_MF050)]  ; mm6=(tmp2L)
    pmaddwd     mm7, [GOTOFF(ebx,PW_MF060_MF050)]  ; mm7=(tmp2H)
    pmaddwd     mm2, [GOTOFF(ebx,PW_F145_MF021)]   ; mm2=(tmp0L)
    pmaddwd     mm3, [GOTOFF(ebx,PW_F145_MF021)]   ; mm3=(tmp0H)

    paddd       mm6, mm4                ; mm6=tmp2L
    paddd       mm7, mm5                ; mm7=tmp2H
    paddd       mm2, mm0                ; mm2=tmp0L
    paddd       mm3, mm1                ; mm3=tmp0H

    movq        MMWORD [wk(0)], mm2     ; wk(0)=tmp0L
    movq        MMWORD [wk(1)], mm3     ; wk(1)=tmp0H

    ; -- Even part

    movq        mm4, MMWORD [MMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movq        mm5, MMWORD [MMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    movq        mm0, MMWORD [MMBLOCK(6,0,esi,SIZEOF_JCOEF)]

    pxor        mm1, mm1
    pxor        mm2, mm2
    punpcklwd   mm1, mm4                ; mm1=tmp0L
    punpckhwd   mm2, mm4                ; mm2=tmp0H
    psrad       mm1, (16-CONST_BITS-1)  ; psrad mm1,16 & pslld mm1,CONST_BITS+1
    psrad       mm2, (16-CONST_BITS-1)  ; psrad mm2,16 & pslld mm2,CONST_BITS+1

    movq        mm3, mm5                ; mm5=in2=z2
    punpcklwd   mm5, mm0                ; mm0=in6=z3
    punpckhwd   mm3, mm0
    pmaddwd     mm5, [GOTOFF(ebx,PW_F184_MF076)]  ; mm5=tmp2L
    pmaddwd     mm3, [GOTOFF(ebx,PW_F184_MF076)]  ; mm3=tmp2H

    movq        mm4, mm1
    movq        mm0, mm2
    paddd       mm1, mm5                ; mm1=tmp10L
    paddd       mm2, mm3                ; mm2=tmp10H
    psubd       mm4, mm5                ; mm4=tmp12L
    psubd       mm0, mm3                ; mm0=tmp12H

    ; -- Final output stage

    movq        mm5, mm1
    movq        mm3, mm2
    paddd       mm1, mm6                ; mm1=data0L
    paddd       mm2, mm7                ; mm2=data0H
    psubd       mm5, mm6                ; mm5=data3L
    psubd       mm3, mm7                ; mm3=data3H

    movq        mm6, [GOTOFF(ebx,PD_DESCALE_P2_4)]  ; mm6=[PD_DESCALE_P2_4]

    paddd       mm1, mm6
    paddd       mm2, mm6
    psrad       mm1, DESCALE_P2_4
    psrad       mm2, DESCALE_P2_4
    paddd       mm5, mm6
    paddd       mm3, mm6
    psrad       mm5, DESCALE_P2_4
    psrad       mm3, DESCALE_P2_4

    packssdw    mm1, mm2                ; mm1=data0=(00 10 20 30)
    packssdw    mm5, mm3                ; mm5=data3=(03 13 23 33)

    movq        mm7, MMWORD [wk(0)]     ; mm7=tmp0L
    movq        mm6, MMWORD [wk(1)]     ; mm6=tmp0H

    movq        mm2, mm4
    movq        mm3, mm0
    paddd       mm4, mm7                ; mm4=data1L
    paddd       mm0, mm6                ; mm0=data1H
    psubd       mm2, mm7                ; mm2=data2L
    psubd       mm3, mm6                ; mm3=data2H

    movq        mm7, [GOTOFF(ebx,PD_DESCALE_P2_4)]  ; mm7=[PD_DESCALE_P2_4]

    paddd       mm4, mm7
    paddd       mm0, mm7
    psrad       mm4, DESCALE_P2_4
    psrad       mm0, DESCALE_P2_4
    paddd       mm2, mm7
    paddd       mm3, mm7
    psrad       mm2, DESCALE_P2_4
    psrad       mm3, DESCALE_P2_4

    packssdw    mm4, mm0                ; mm4=data1=(01 11 21 31)
    packssdw    mm2, mm3                ; mm2=data2=(02 12 22 32)

    movq        mm6, [GOTOFF(ebx,PB_CENTERJSAMP)]  ; mm6=[PB_CENTERJSAMP]

    packsswb    mm1, mm2                ; mm1=(00 10 20 30 02 12 22 32)
    packsswb    mm4, mm5                ; mm4=(01 11 21 31 03 13 23 33)
    paddb       mm1, mm6
    paddb       mm4, mm6

    movq        mm7, mm1                ; transpose coefficients(phase 1)
    punpcklbw   mm1, mm4                ; mm1=(00 01 10 11 20 21 30 31)
    punpckhbw   mm7, mm4                ; mm7=(02 03 12 13 22 23 32 33)

    movq        mm0, mm1                ; transpose coefficients(phase 2)
    punpcklwd   mm1, mm7                ; mm1=(00 01 02 03 10 11 12 13)
    punpckhwd   mm0, mm7                ; mm0=(20 21 22 23 30 31 32 33)

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]
    mov         esi, JSAMPROW [edi+2*SIZEOF_JSAMPROW]
    movd        dword [edx+eax*SIZEOF_JSAMPLE], mm1
    movd        dword [esi+eax*SIZEOF_JSAMPLE], mm0

    psrlq       mm1, 4*BYTE_BIT
    psrlq       mm0, 4*BYTE_BIT

    mov         edx, JSAMPROW [edi+1*SIZEOF_JSAMPROW]
    mov         esi, JSAMPROW [edi+3*SIZEOF_JSAMPROW]
    movd        dword [edx+eax*SIZEOF_JSAMPLE], mm1
    movd        dword [esi+eax*SIZEOF_JSAMPLE], mm0

    emms                                ; empty MMX state

    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    POPPIC      ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
    ret

; --------------------------------------------------------------------------
;
; Perform dequantization and inverse DCT on one block of coefficients,
; producing a reduced-size 2x2 output block.
;
; GLOBAL(void)
; jsimd_idct_2x2_mmx(void *dct_table, JCOEFPTR coef_block,
;                    JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; void *dct_table
%define coef_block(b)  (b) + 12         ; JCOEFPTR coef_block
%define output_buf(b)  (b) + 16         ; JSAMPARRAY output_buf
%define output_col(b)  (b) + 20         ; JDIMENSION output_col

    align       32
    GLOBAL_FUNCTION(jsimd_idct_2x2_mmx)

EXTN(jsimd_idct_2x2_mmx):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    GET_GOT     ebx                     ; get GOT address

    ; ---- Pass 1: process columns from input.

    mov         edx, POINTER [dct_table(ebp)]    ; quantptr
    mov         esi, JCOEFPTR [coef_block(ebp)]  ; inptr

    ; | input:                  | result:        |
    ; | 00 01 ** 03 ** 05 ** 07 |                |
    ; | 10 11 ** 13 ** 15 ** 17 |                |
    ; | ** ** ** ** ** ** ** ** |                |
    ; | 30 31 ** 33 ** 35 ** 37 | A0 A1 A3 A5 A7 |
    ; | ** ** ** ** ** ** ** ** | B0 B1 B3 B5 B7 |
    ; | 50 51 ** 53 ** 55 ** 57 |                |
    ; | ** ** ** ** ** ** ** ** |                |
    ; | 70 71 ** 73 ** 75 ** 77 |                |

    ; -- Odd part

    movq        mm0, MMWORD [MMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    pmullw      mm0, MMWORD [MMBLOCK(1,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm1, MMWORD [MMBLOCK(3,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    movq        mm2, MMWORD [MMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movq        mm3, MMWORD [MMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    pmullw      mm2, MMWORD [MMBLOCK(5,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm3, MMWORD [MMBLOCK(7,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    ; mm0=(10 11 ** 13), mm1=(30 31 ** 33)
    ; mm2=(50 51 ** 53), mm3=(70 71 ** 73)

    pcmpeqd     mm7, mm7
    pslld       mm7, WORD_BIT           ; mm7={0x0000 0xFFFF 0x0000 0xFFFF}

    movq        mm4, mm0                ; mm4=(10 11 ** 13)
    movq        mm5, mm2                ; mm5=(50 51 ** 53)
    punpcklwd   mm4, mm1                ; mm4=(10 30 11 31)
    punpcklwd   mm5, mm3                ; mm5=(50 70 51 71)
    pmaddwd     mm4, [GOTOFF(ebx,PW_F362_MF127)]
    pmaddwd     mm5, [GOTOFF(ebx,PW_F085_MF072)]

    psrld       mm0, WORD_BIT           ; mm0=(11 -- 13 --)
    pand        mm1, mm7                ; mm1=(-- 31 -- 33)
    psrld       mm2, WORD_BIT           ; mm2=(51 -- 53 --)
    pand        mm3, mm7                ; mm3=(-- 71 -- 73)
    por         mm0, mm1                ; mm0=(11 31 13 33)
    por         mm2, mm3                ; mm2=(51 71 53 73)
    pmaddwd     mm0, [GOTOFF(ebx,PW_F362_MF127)]
    pmaddwd     mm2, [GOTOFF(ebx,PW_F085_MF072)]

    paddd       mm4, mm5                ; mm4=tmp0[col0 col1]

    movq        mm6, MMWORD [MMBLOCK(1,1,esi,SIZEOF_JCOEF)]
    movq        mm1, MMWORD [MMBLOCK(3,1,esi,SIZEOF_JCOEF)]
    pmullw      mm6, MMWORD [MMBLOCK(1,1,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm1, MMWORD [MMBLOCK(3,1,edx,SIZEOF_ISLOW_MULT_TYPE)]
    movq        mm3, MMWORD [MMBLOCK(5,1,esi,SIZEOF_JCOEF)]
    movq        mm5, MMWORD [MMBLOCK(7,1,esi,SIZEOF_JCOEF)]
    pmullw      mm3, MMWORD [MMBLOCK(5,1,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm5, MMWORD [MMBLOCK(7,1,edx,SIZEOF_ISLOW_MULT_TYPE)]

    ; mm6=(** 15 ** 17), mm1=(** 35 ** 37)
    ; mm3=(** 55 ** 57), mm5=(** 75 ** 77)

    psrld       mm6, WORD_BIT           ; mm6=(15 -- 17 --)
    pand        mm1, mm7                ; mm1=(-- 35 -- 37)
    psrld       mm3, WORD_BIT           ; mm3=(55 -- 57 --)
    pand        mm5, mm7                ; mm5=(-- 75 -- 77)
    por         mm6, mm1                ; mm6=(15 35 17 37)
    por         mm3, mm5                ; mm3=(55 75 57 77)
    pmaddwd     mm6, [GOTOFF(ebx,PW_F362_MF127)]
    pmaddwd     mm3, [GOTOFF(ebx,PW_F085_MF072)]

    paddd       mm0, mm2                ; mm0=tmp0[col1 col3]
    paddd       mm6, mm3                ; mm6=tmp0[col5 col7]

    ; -- Even part

    movq        mm1, MMWORD [MMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movq        mm5, MMWORD [MMBLOCK(0,1,esi,SIZEOF_JCOEF)]
    pmullw      mm1, MMWORD [MMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      mm5, MMWORD [MMBLOCK(0,1,edx,SIZEOF_ISLOW_MULT_TYPE)]

    ; mm1=(00 01 ** 03), mm5=(** 05 ** 07)

    movq        mm2, mm1                      ; mm2=(00 01 ** 03)
    pslld       mm1, WORD_BIT                 ; mm1=(-- 00 -- **)
    psrad       mm1, (WORD_BIT-CONST_BITS-2)  ; mm1=tmp10[col0 ****]

    pand        mm2, mm7                      ; mm2=(-- 01 -- 03)
    pand        mm5, mm7                      ; mm5=(-- 05 -- 07)
    psrad       mm2, (WORD_BIT-CONST_BITS-2)  ; mm2=tmp10[col1 col3]
    psrad       mm5, (WORD_BIT-CONST_BITS-2)  ; mm5=tmp10[col5 col7]

    ; -- Final output stage

    movq        mm3, mm1
    paddd       mm1, mm4                ; mm1=data0[col0 ****]=(A0 **)
    psubd       mm3, mm4                ; mm3=data1[col0 ****]=(B0 **)
    punpckldq   mm1, mm3                ; mm1=(A0 B0)

    movq        mm7, [GOTOFF(ebx,PD_DESCALE_P1_2)]  ; mm7=[PD_DESCALE_P1_2]

    movq        mm4, mm2
    movq        mm3, mm5
    paddd       mm2, mm0                ; mm2=data0[col1 col3]=(A1 A3)
    paddd       mm5, mm6                ; mm5=data0[col5 col7]=(A5 A7)
    psubd       mm4, mm0                ; mm4=data1[col1 col3]=(B1 B3)
    psubd       mm3, mm6                ; mm3=data1[col5 col7]=(B5 B7)

    paddd       mm1, mm7
    psrad       mm1, DESCALE_P1_2

    paddd       mm2, mm7
    paddd       mm5, mm7
    psrad       mm2, DESCALE_P1_2
    psrad       mm5, DESCALE_P1_2
    paddd       mm4, mm7
    paddd       mm3, mm7
    psrad       mm4, DESCALE_P1_2
    psrad       mm3, DESCALE_P1_2

    ; ---- Pass 2: process rows, store into output array.

    mov         edi, JSAMPARRAY [output_buf(ebp)]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [output_col(ebp)]

    ; | input:| result:|
    ; | A0 B0 |        |
    ; | A1 B1 | C0 C1  |
    ; | A3 B3 | D0 D1  |
    ; | A5 B5 |        |
    ; | A7 B7 |        |

    ; -- Odd part

    packssdw    mm2, mm4                ; mm2=(A1 A3 B1 B3)
    packssdw    mm5, mm3                ; mm5=(A5 A7 B5 B7)
    pmaddwd     mm2, [GOTOFF(ebx,PW_F362_MF127)]
    pmaddwd     mm5, [GOTOFF(ebx,PW_F085_MF072)]

    paddd       mm2, mm5                ; mm2=tmp0[row0 row1]

    ; -- Even part

    pslld       mm1, (CONST_BITS+2)     ; mm1=tmp10[row0 row1]

    ; -- Final output stage

    movq        mm0, [GOTOFF(ebx,PD_DESCALE_P2_2)]  ; mm0=[PD_DESCALE_P2_2]

    movq        mm6, mm1
    paddd       mm1, mm2                ; mm1=data0[row0 row1]=(C0 C1)
    psubd       mm6, mm2                ; mm6=data1[row0 row1]=(D0 D1)

    paddd       mm1, mm0
    paddd       mm6, mm0
    psrad       mm1, DESCALE_P2_2
    psrad       mm6, DESCALE_P2_2

    movq        mm7, mm1                ; transpose coefficients
    punpckldq   mm1, mm6                ; mm1=(C0 D0)
    punpckhdq   mm7, mm6                ; mm7=(C1 D1)

    packssdw    mm1, mm7                ; mm1=(C0 D0 C1 D1)
    packsswb    mm1, mm1                ; mm1=(C0 D0 C1 D1 C0 D0 C1 D1)
    paddb       mm1, [GOTOFF(ebx,PB_CENTERJSAMP)]

    movd        ecx, mm1
    movd        ebx, mm1                ; ebx=(C0 D0 C1 D1)
    shr         ecx, 2*BYTE_BIT         ; ecx=(C1 D1 -- --)

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]
    mov         esi, JSAMPROW [edi+1*SIZEOF_JSAMPROW]
    mov         word [edx+eax*SIZEOF_JSAMPLE], bx
    mov         word [esi+eax*SIZEOF_JSAMPLE], cx

    emms                                ; empty MMX state

    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    pop         ebx
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
