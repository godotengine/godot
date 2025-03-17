;
; jidctred.asm - reduced-size IDCT (64-bit SSE2)
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
    GLOBAL_DATA(jconst_idct_red_sse2)

EXTN(jconst_idct_red_sse2):

PW_F184_MF076   times 4  dw  F_1_847, -F_0_765
PW_F256_F089    times 4  dw  F_2_562,  F_0_899
PW_F106_MF217   times 4  dw  F_1_061, -F_2_172
PW_MF060_MF050  times 4  dw -F_0_601, -F_0_509
PW_F145_MF021   times 4  dw  F_1_451, -F_0_211
PW_F362_MF127   times 4  dw  F_3_624, -F_1_272
PW_F085_MF072   times 4  dw  F_0_850, -F_0_720
PD_DESCALE_P1_4 times 4  dd  1 << (DESCALE_P1_4 - 1)
PD_DESCALE_P2_4 times 4  dd  1 << (DESCALE_P2_4 - 1)
PD_DESCALE_P1_2 times 4  dd  1 << (DESCALE_P1_2 - 1)
PD_DESCALE_P2_2 times 4  dd  1 << (DESCALE_P2_2 - 1)
PB_CENTERJSAMP  times 16 db  CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64
;
; Perform dequantization and inverse DCT on one block of coefficients,
; producing a reduced-size 4x4 output block.
;
; GLOBAL(void)
; jsimd_idct_4x4_sse2(void *dct_table, JCOEFPTR coef_block,
;                     JSAMPARRAY output_buf, JDIMENSION output_col)
;

; r10 = void *dct_table
; r11 = JCOEFPTR coef_block
; r12 = JSAMPARRAY output_buf
; r13d = JDIMENSION output_col

%define wk(i)         r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD
                                        ; xmmword wk[WK_NUM]
%define WK_NUM        2

    align       32
    GLOBAL_FUNCTION(jsimd_idct_4x4_sse2)

EXTN(jsimd_idct_4x4_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    push        r15
    and         rsp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    ; Allocate stack space for wk array.  r15 is used to access it.
    mov         r15, rsp
    sub         rsp, byte (SIZEOF_XMMWORD * WK_NUM)
    COLLECT_ARGS 4

    ; ---- Pass 1: process columns from input.

    mov         rdx, r10                ; quantptr
    mov         rsi, r11                ; inptr

%ifndef NO_ZERO_COLUMN_TEST_4X4_SSE2
    mov         eax, dword [DWBLOCK(1,0,rsi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,rsi,SIZEOF_JCOEF)]
    jnz         short .columnDCT

    movdqa      xmm0, XMMWORD [XMMBLOCK(1,0,rsi,SIZEOF_JCOEF)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(2,0,rsi,SIZEOF_JCOEF)]
    por         xmm0, XMMWORD [XMMBLOCK(3,0,rsi,SIZEOF_JCOEF)]
    por         xmm1, XMMWORD [XMMBLOCK(5,0,rsi,SIZEOF_JCOEF)]
    por         xmm0, XMMWORD [XMMBLOCK(6,0,rsi,SIZEOF_JCOEF)]
    por         xmm1, XMMWORD [XMMBLOCK(7,0,rsi,SIZEOF_JCOEF)]
    por         xmm0, xmm1
    packsswb    xmm0, xmm0
    packsswb    xmm0, xmm0
    movd        eax, xmm0
    test        rax, rax
    jnz         short .columnDCT

    ; -- AC terms all zero

    movdqa      xmm0, XMMWORD [XMMBLOCK(0,0,rsi,SIZEOF_JCOEF)]
    pmullw      xmm0, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]

    psllw       xmm0, PASS1_BITS

    movdqa      xmm3, xmm0        ; xmm0=in0=(00 01 02 03 04 05 06 07)
    punpcklwd   xmm0, xmm0        ; xmm0=(00 00 01 01 02 02 03 03)
    punpckhwd   xmm3, xmm3        ; xmm3=(04 04 05 05 06 06 07 07)

    pshufd      xmm1, xmm0, 0x50  ; xmm1=[col0 col1]=(00 00 00 00 01 01 01 01)
    pshufd      xmm0, xmm0, 0xFA  ; xmm0=[col2 col3]=(02 02 02 02 03 03 03 03)
    pshufd      xmm6, xmm3, 0x50  ; xmm6=[col4 col5]=(04 04 04 04 05 05 05 05)
    pshufd      xmm3, xmm3, 0xFA  ; xmm3=[col6 col7]=(06 06 06 06 07 07 07 07)

    jmp         near .column_end
%endif
.columnDCT:

    ; -- Odd part

    movdqa      xmm0, XMMWORD [XMMBLOCK(1,0,rsi,SIZEOF_JCOEF)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(3,0,rsi,SIZEOF_JCOEF)]
    pmullw      xmm0, XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm1, XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    movdqa      xmm2, XMMWORD [XMMBLOCK(5,0,rsi,SIZEOF_JCOEF)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(7,0,rsi,SIZEOF_JCOEF)]
    pmullw      xmm2, XMMWORD [XMMBLOCK(5,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm3, XMMWORD [XMMBLOCK(7,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]

    movdqa      xmm4, xmm0
    movdqa      xmm5, xmm0
    punpcklwd   xmm4, xmm1
    punpckhwd   xmm5, xmm1
    movdqa      xmm0, xmm4
    movdqa      xmm1, xmm5
    pmaddwd     xmm4, [rel PW_F256_F089]   ; xmm4=(tmp2L)
    pmaddwd     xmm5, [rel PW_F256_F089]   ; xmm5=(tmp2H)
    pmaddwd     xmm0, [rel PW_F106_MF217]  ; xmm0=(tmp0L)
    pmaddwd     xmm1, [rel PW_F106_MF217]  ; xmm1=(tmp0H)

    movdqa      xmm6, xmm2
    movdqa      xmm7, xmm2
    punpcklwd   xmm6, xmm3
    punpckhwd   xmm7, xmm3
    movdqa      xmm2, xmm6
    movdqa      xmm3, xmm7
    pmaddwd     xmm6, [rel PW_MF060_MF050]  ; xmm6=(tmp2L)
    pmaddwd     xmm7, [rel PW_MF060_MF050]  ; xmm7=(tmp2H)
    pmaddwd     xmm2, [rel PW_F145_MF021]   ; xmm2=(tmp0L)
    pmaddwd     xmm3, [rel PW_F145_MF021]   ; xmm3=(tmp0H)

    paddd       xmm6, xmm4              ; xmm6=tmp2L
    paddd       xmm7, xmm5              ; xmm7=tmp2H
    paddd       xmm2, xmm0              ; xmm2=tmp0L
    paddd       xmm3, xmm1              ; xmm3=tmp0H

    movdqa      XMMWORD [wk(0)], xmm2   ; wk(0)=tmp0L
    movdqa      XMMWORD [wk(1)], xmm3   ; wk(1)=tmp0H

    ; -- Even part

    movdqa      xmm4, XMMWORD [XMMBLOCK(0,0,rsi,SIZEOF_JCOEF)]
    movdqa      xmm5, XMMWORD [XMMBLOCK(2,0,rsi,SIZEOF_JCOEF)]
    movdqa      xmm0, XMMWORD [XMMBLOCK(6,0,rsi,SIZEOF_JCOEF)]
    pmullw      xmm4, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm5, XMMWORD [XMMBLOCK(2,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm0, XMMWORD [XMMBLOCK(6,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]

    pxor        xmm1, xmm1
    pxor        xmm2, xmm2
    punpcklwd   xmm1, xmm4               ; xmm1=tmp0L
    punpckhwd   xmm2, xmm4               ; xmm2=tmp0H
    psrad       xmm1, (16-CONST_BITS-1)  ; psrad xmm1,16 & pslld xmm1,CONST_BITS+1
    psrad       xmm2, (16-CONST_BITS-1)  ; psrad xmm2,16 & pslld xmm2,CONST_BITS+1

    movdqa      xmm3, xmm5              ; xmm5=in2=z2
    punpcklwd   xmm5, xmm0              ; xmm0=in6=z3
    punpckhwd   xmm3, xmm0
    pmaddwd     xmm5, [rel PW_F184_MF076]  ; xmm5=tmp2L
    pmaddwd     xmm3, [rel PW_F184_MF076]  ; xmm3=tmp2H

    movdqa      xmm4, xmm1
    movdqa      xmm0, xmm2
    paddd       xmm1, xmm5              ; xmm1=tmp10L
    paddd       xmm2, xmm3              ; xmm2=tmp10H
    psubd       xmm4, xmm5              ; xmm4=tmp12L
    psubd       xmm0, xmm3              ; xmm0=tmp12H

    ; -- Final output stage

    movdqa      xmm5, xmm1
    movdqa      xmm3, xmm2
    paddd       xmm1, xmm6              ; xmm1=data0L
    paddd       xmm2, xmm7              ; xmm2=data0H
    psubd       xmm5, xmm6              ; xmm5=data3L
    psubd       xmm3, xmm7              ; xmm3=data3H

    movdqa      xmm6, [rel PD_DESCALE_P1_4]  ; xmm6=[rel PD_DESCALE_P1_4]

    paddd       xmm1, xmm6
    paddd       xmm2, xmm6
    psrad       xmm1, DESCALE_P1_4
    psrad       xmm2, DESCALE_P1_4
    paddd       xmm5, xmm6
    paddd       xmm3, xmm6
    psrad       xmm5, DESCALE_P1_4
    psrad       xmm3, DESCALE_P1_4

    packssdw    xmm1, xmm2              ; xmm1=data0=(00 01 02 03 04 05 06 07)
    packssdw    xmm5, xmm3              ; xmm5=data3=(30 31 32 33 34 35 36 37)

    movdqa      xmm7, XMMWORD [wk(0)]   ; xmm7=tmp0L
    movdqa      xmm6, XMMWORD [wk(1)]   ; xmm6=tmp0H

    movdqa      xmm2, xmm4
    movdqa      xmm3, xmm0
    paddd       xmm4, xmm7              ; xmm4=data1L
    paddd       xmm0, xmm6              ; xmm0=data1H
    psubd       xmm2, xmm7              ; xmm2=data2L
    psubd       xmm3, xmm6              ; xmm3=data2H

    movdqa      xmm7, [rel PD_DESCALE_P1_4]  ; xmm7=[rel PD_DESCALE_P1_4]

    paddd       xmm4, xmm7
    paddd       xmm0, xmm7
    psrad       xmm4, DESCALE_P1_4
    psrad       xmm0, DESCALE_P1_4
    paddd       xmm2, xmm7
    paddd       xmm3, xmm7
    psrad       xmm2, DESCALE_P1_4
    psrad       xmm3, DESCALE_P1_4

    packssdw    xmm4, xmm0        ; xmm4=data1=(10 11 12 13 14 15 16 17)
    packssdw    xmm2, xmm3        ; xmm2=data2=(20 21 22 23 24 25 26 27)

    movdqa      xmm6, xmm1        ; transpose coefficients(phase 1)
    punpcklwd   xmm1, xmm4        ; xmm1=(00 10 01 11 02 12 03 13)
    punpckhwd   xmm6, xmm4        ; xmm6=(04 14 05 15 06 16 07 17)
    movdqa      xmm7, xmm2        ; transpose coefficients(phase 1)
    punpcklwd   xmm2, xmm5        ; xmm2=(20 30 21 31 22 32 23 33)
    punpckhwd   xmm7, xmm5        ; xmm7=(24 34 25 35 26 36 27 37)

    movdqa      xmm0, xmm1        ; transpose coefficients(phase 2)
    punpckldq   xmm1, xmm2        ; xmm1=[col0 col1]=(00 10 20 30 01 11 21 31)
    punpckhdq   xmm0, xmm2        ; xmm0=[col2 col3]=(02 12 22 32 03 13 23 33)
    movdqa      xmm3, xmm6        ; transpose coefficients(phase 2)
    punpckldq   xmm6, xmm7        ; xmm6=[col4 col5]=(04 14 24 34 05 15 25 35)
    punpckhdq   xmm3, xmm7        ; xmm3=[col6 col7]=(06 16 26 36 07 17 27 37)
.column_end:

    ; -- Prefetch the next coefficient block

    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 0*32]
    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 1*32]
    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 2*32]
    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 3*32]

    ; ---- Pass 2: process rows, store into output array.

    mov         rdi, r12                ; (JSAMPROW *)
    mov         eax, r13d

    ; -- Even part

    pxor        xmm4, xmm4
    punpcklwd   xmm4, xmm1               ; xmm4=tmp0
    psrad       xmm4, (16-CONST_BITS-1)  ; psrad xmm4,16 & pslld xmm4,CONST_BITS+1

    ; -- Odd part

    punpckhwd   xmm1, xmm0
    punpckhwd   xmm6, xmm3
    movdqa      xmm5, xmm1
    movdqa      xmm2, xmm6
    pmaddwd     xmm1, [rel PW_F256_F089]    ; xmm1=(tmp2)
    pmaddwd     xmm6, [rel PW_MF060_MF050]  ; xmm6=(tmp2)
    pmaddwd     xmm5, [rel PW_F106_MF217]   ; xmm5=(tmp0)
    pmaddwd     xmm2, [rel PW_F145_MF021]   ; xmm2=(tmp0)

    paddd       xmm6, xmm1              ; xmm6=tmp2
    paddd       xmm2, xmm5              ; xmm2=tmp0

    ; -- Even part

    punpcklwd   xmm0, xmm3
    pmaddwd     xmm0, [rel PW_F184_MF076]  ; xmm0=tmp2

    movdqa      xmm7, xmm4
    paddd       xmm4, xmm0              ; xmm4=tmp10
    psubd       xmm7, xmm0              ; xmm7=tmp12

    ; -- Final output stage

    movdqa      xmm1, [rel PD_DESCALE_P2_4]  ; xmm1=[rel PD_DESCALE_P2_4]

    movdqa      xmm5, xmm4
    movdqa      xmm3, xmm7
    paddd       xmm4, xmm6              ; xmm4=data0=(00 10 20 30)
    paddd       xmm7, xmm2              ; xmm7=data1=(01 11 21 31)
    psubd       xmm5, xmm6              ; xmm5=data3=(03 13 23 33)
    psubd       xmm3, xmm2              ; xmm3=data2=(02 12 22 32)

    paddd       xmm4, xmm1
    paddd       xmm7, xmm1
    psrad       xmm4, DESCALE_P2_4
    psrad       xmm7, DESCALE_P2_4
    paddd       xmm5, xmm1
    paddd       xmm3, xmm1
    psrad       xmm5, DESCALE_P2_4
    psrad       xmm3, DESCALE_P2_4

    packssdw    xmm4, xmm3              ; xmm4=(00 10 20 30 02 12 22 32)
    packssdw    xmm7, xmm5              ; xmm7=(01 11 21 31 03 13 23 33)

    movdqa      xmm0, xmm4              ; transpose coefficients(phase 1)
    punpcklwd   xmm4, xmm7              ; xmm4=(00 01 10 11 20 21 30 31)
    punpckhwd   xmm0, xmm7              ; xmm0=(02 03 12 13 22 23 32 33)

    movdqa      xmm6, xmm4              ; transpose coefficients(phase 2)
    punpckldq   xmm4, xmm0              ; xmm4=(00 01 02 03 10 11 12 13)
    punpckhdq   xmm6, xmm0              ; xmm6=(20 21 22 23 30 31 32 33)

    packsswb    xmm4, xmm6              ; xmm4=(00 01 02 03 10 11 12 13 20 ..)
    paddb       xmm4, [rel PB_CENTERJSAMP]

    pshufd      xmm2, xmm4, 0x39        ; xmm2=(10 11 12 13 20 21 22 23 30 ..)
    pshufd      xmm1, xmm4, 0x4E        ; xmm1=(20 21 22 23 30 31 32 33 00 ..)
    pshufd      xmm3, xmm4, 0x93        ; xmm3=(30 31 32 33 00 01 02 03 10 ..)

    mov         rdxp, JSAMPROW [rdi+0*SIZEOF_JSAMPROW]
    mov         rsip, JSAMPROW [rdi+1*SIZEOF_JSAMPROW]
    movd        XMM_DWORD [rdx+rax*SIZEOF_JSAMPLE], xmm4
    movd        XMM_DWORD [rsi+rax*SIZEOF_JSAMPLE], xmm2
    mov         rdxp, JSAMPROW [rdi+2*SIZEOF_JSAMPROW]
    mov         rsip, JSAMPROW [rdi+3*SIZEOF_JSAMPROW]
    movd        XMM_DWORD [rdx+rax*SIZEOF_JSAMPLE], xmm1
    movd        XMM_DWORD [rsi+rax*SIZEOF_JSAMPLE], xmm3

    UNCOLLECT_ARGS 4
    lea         rsp, [rbp-8]
    pop         r15
    pop         rbp
    ret

; --------------------------------------------------------------------------
;
; Perform dequantization and inverse DCT on one block of coefficients,
; producing a reduced-size 2x2 output block.
;
; GLOBAL(void)
; jsimd_idct_2x2_sse2(void *dct_table, JCOEFPTR coef_block,
;                     JSAMPARRAY output_buf, JDIMENSION output_col)
;

; r10 = void *dct_table
; r11 = JCOEFPTR coef_block
; r12 = JSAMPARRAY output_buf
; r13d = JDIMENSION output_col

    align       32
    GLOBAL_FUNCTION(jsimd_idct_2x2_sse2)

EXTN(jsimd_idct_2x2_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 4
    push        rbx

    ; ---- Pass 1: process columns from input.

    mov         rdx, r10                ; quantptr
    mov         rsi, r11                ; inptr

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

    movdqa      xmm0, XMMWORD [XMMBLOCK(1,0,rsi,SIZEOF_JCOEF)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(3,0,rsi,SIZEOF_JCOEF)]
    pmullw      xmm0, XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm1, XMMWORD [XMMBLOCK(3,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    movdqa      xmm2, XMMWORD [XMMBLOCK(5,0,rsi,SIZEOF_JCOEF)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(7,0,rsi,SIZEOF_JCOEF)]
    pmullw      xmm2, XMMWORD [XMMBLOCK(5,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm3, XMMWORD [XMMBLOCK(7,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]

    ; xmm0=(10 11 ** 13 ** 15 ** 17), xmm1=(30 31 ** 33 ** 35 ** 37)
    ; xmm2=(50 51 ** 53 ** 55 ** 57), xmm3=(70 71 ** 73 ** 75 ** 77)

    pcmpeqd     xmm7, xmm7
    pslld       xmm7, WORD_BIT          ; xmm7={0x0000 0xFFFF 0x0000 0xFFFF ..}

    movdqa      xmm4, xmm0              ; xmm4=(10 11 ** 13 ** 15 ** 17)
    movdqa      xmm5, xmm2              ; xmm5=(50 51 ** 53 ** 55 ** 57)
    punpcklwd   xmm4, xmm1              ; xmm4=(10 30 11 31 ** ** 13 33)
    punpcklwd   xmm5, xmm3              ; xmm5=(50 70 51 71 ** ** 53 73)
    pmaddwd     xmm4, [rel PW_F362_MF127]
    pmaddwd     xmm5, [rel PW_F085_MF072]

    psrld       xmm0, WORD_BIT          ; xmm0=(11 -- 13 -- 15 -- 17 --)
    pand        xmm1, xmm7              ; xmm1=(-- 31 -- 33 -- 35 -- 37)
    psrld       xmm2, WORD_BIT          ; xmm2=(51 -- 53 -- 55 -- 57 --)
    pand        xmm3, xmm7              ; xmm3=(-- 71 -- 73 -- 75 -- 77)
    por         xmm0, xmm1              ; xmm0=(11 31 13 33 15 35 17 37)
    por         xmm2, xmm3              ; xmm2=(51 71 53 73 55 75 57 77)
    pmaddwd     xmm0, [rel PW_F362_MF127]
    pmaddwd     xmm2, [rel PW_F085_MF072]

    paddd       xmm4, xmm5              ; xmm4=tmp0[col0 col1 **** col3]
    paddd       xmm0, xmm2              ; xmm0=tmp0[col1 col3 col5 col7]

    ; -- Even part

    movdqa      xmm6, XMMWORD [XMMBLOCK(0,0,rsi,SIZEOF_JCOEF)]
    pmullw      xmm6, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_ISLOW_MULT_TYPE)]

    ; xmm6=(00 01 ** 03 ** 05 ** 07)

    movdqa      xmm1, xmm6              ; xmm1=(00 01 ** 03 ** 05 ** 07)
    pslld       xmm6, WORD_BIT          ; xmm6=(-- 00 -- ** -- ** -- **)
    pand        xmm1, xmm7              ; xmm1=(-- 01 -- 03 -- 05 -- 07)
    psrad       xmm6, (WORD_BIT-CONST_BITS-2)  ; xmm6=tmp10[col0 **** **** ****]
    psrad       xmm1, (WORD_BIT-CONST_BITS-2)  ; xmm1=tmp10[col1 col3 col5 col7]

    ; -- Final output stage

    movdqa      xmm3, xmm6
    movdqa      xmm5, xmm1
    paddd       xmm6, xmm4      ; xmm6=data0[col0 **** **** ****]=(A0 ** ** **)
    paddd       xmm1, xmm0      ; xmm1=data0[col1 col3 col5 col7]=(A1 A3 A5 A7)
    psubd       xmm3, xmm4      ; xmm3=data1[col0 **** **** ****]=(B0 ** ** **)
    psubd       xmm5, xmm0      ; xmm5=data1[col1 col3 col5 col7]=(B1 B3 B5 B7)

    movdqa      xmm2, [rel PD_DESCALE_P1_2]  ; xmm2=[rel PD_DESCALE_P1_2]

    punpckldq   xmm6, xmm3              ; xmm6=(A0 B0 ** **)

    movdqa      xmm7, xmm1
    punpcklqdq  xmm1, xmm5              ; xmm1=(A1 A3 B1 B3)
    punpckhqdq  xmm7, xmm5              ; xmm7=(A5 A7 B5 B7)

    paddd       xmm6, xmm2
    psrad       xmm6, DESCALE_P1_2

    paddd       xmm1, xmm2
    paddd       xmm7, xmm2
    psrad       xmm1, DESCALE_P1_2
    psrad       xmm7, DESCALE_P1_2

    ; -- Prefetch the next coefficient block

    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 0*32]
    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 1*32]
    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 2*32]
    prefetchnta [rsi + DCTSIZE2*SIZEOF_JCOEF + 3*32]

    ; ---- Pass 2: process rows, store into output array.

    mov         rdi, r12                ; (JSAMPROW *)
    mov         eax, r13d

    ; | input:| result:|
    ; | A0 B0 |        |
    ; | A1 B1 | C0 C1  |
    ; | A3 B3 | D0 D1  |
    ; | A5 B5 |        |
    ; | A7 B7 |        |

    ; -- Odd part

    packssdw    xmm1, xmm1              ; xmm1=(A1 A3 B1 B3 A1 A3 B1 B3)
    packssdw    xmm7, xmm7              ; xmm7=(A5 A7 B5 B7 A5 A7 B5 B7)
    pmaddwd     xmm1, [rel PW_F362_MF127]
    pmaddwd     xmm7, [rel PW_F085_MF072]

    paddd       xmm1, xmm7              ; xmm1=tmp0[row0 row1 row0 row1]

    ; -- Even part

    pslld       xmm6, (CONST_BITS+2)    ; xmm6=tmp10[row0 row1 **** ****]

    ; -- Final output stage

    movdqa      xmm4, xmm6
    paddd       xmm6, xmm1     ; xmm6=data0[row0 row1 **** ****]=(C0 C1 ** **)
    psubd       xmm4, xmm1     ; xmm4=data1[row0 row1 **** ****]=(D0 D1 ** **)

    punpckldq   xmm6, xmm4     ; xmm6=(C0 D0 C1 D1)

    paddd       xmm6, [rel PD_DESCALE_P2_2]
    psrad       xmm6, DESCALE_P2_2

    packssdw    xmm6, xmm6              ; xmm6=(C0 D0 C1 D1 C0 D0 C1 D1)
    packsswb    xmm6, xmm6              ; xmm6=(C0 D0 C1 D1 C0 D0 C1 D1 ..)
    paddb       xmm6, [rel PB_CENTERJSAMP]

    pextrw      ebx, xmm6, 0x00         ; ebx=(C0 D0 -- --)
    pextrw      ecx, xmm6, 0x01         ; ecx=(C1 D1 -- --)

    mov         rdxp, JSAMPROW [rdi+0*SIZEOF_JSAMPROW]
    mov         rsip, JSAMPROW [rdi+1*SIZEOF_JSAMPROW]
    mov         word [rdx+rax*SIZEOF_JSAMPLE], bx
    mov         word [rsi+rax*SIZEOF_JSAMPLE], cx

    pop         rbx
    UNCOLLECT_ARGS 4
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
