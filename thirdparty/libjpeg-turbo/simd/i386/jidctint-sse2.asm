;
; jidctint.asm - accurate integer IDCT (SSE2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2016, 2020, 2024, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.
;
; This file contains a slower but more accurate integer implementation of the
; inverse DCT (Discrete Cosine Transform). The following code is based
; directly on the IJG's original jidctint.c; see the jidctint.c for
; more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%define CONST_BITS  13
%define PASS1_BITS  2

%define DESCALE_P1  (CONST_BITS - PASS1_BITS)
%define DESCALE_P2  (CONST_BITS + PASS1_BITS + 3)

%if CONST_BITS == 13
F_0_298 equ  2446  ; FIX(0.298631336)
F_0_390 equ  3196  ; FIX(0.390180644)
F_0_541 equ  4433  ; FIX(0.541196100)
F_0_765 equ  6270  ; FIX(0.765366865)
F_0_899 equ  7373  ; FIX(0.899976223)
F_1_175 equ  9633  ; FIX(1.175875602)
F_1_501 equ 12299  ; FIX(1.501321110)
F_1_847 equ 15137  ; FIX(1.847759065)
F_1_961 equ 16069  ; FIX(1.961570560)
F_2_053 equ 16819  ; FIX(2.053119869)
F_2_562 equ 20995  ; FIX(2.562915447)
F_3_072 equ 25172  ; FIX(3.072711026)
%else
; NASM cannot do compile-time arithmetic on floating-point constants.
%define DESCALE(x, n)  (((x) + (1 << ((n) - 1))) >> (n))
F_0_298 equ DESCALE( 320652955, 30 - CONST_BITS)  ; FIX(0.298631336)
F_0_390 equ DESCALE( 418953276, 30 - CONST_BITS)  ; FIX(0.390180644)
F_0_541 equ DESCALE( 581104887, 30 - CONST_BITS)  ; FIX(0.541196100)
F_0_765 equ DESCALE( 821806413, 30 - CONST_BITS)  ; FIX(0.765366865)
F_0_899 equ DESCALE( 966342111, 30 - CONST_BITS)  ; FIX(0.899976223)
F_1_175 equ DESCALE(1262586813, 30 - CONST_BITS)  ; FIX(1.175875602)
F_1_501 equ DESCALE(1612031267, 30 - CONST_BITS)  ; FIX(1.501321110)
F_1_847 equ DESCALE(1984016188, 30 - CONST_BITS)  ; FIX(1.847759065)
F_1_961 equ DESCALE(2106220350, 30 - CONST_BITS)  ; FIX(1.961570560)
F_2_053 equ DESCALE(2204520673, 30 - CONST_BITS)  ; FIX(2.053119869)
F_2_562 equ DESCALE(2751909506, 30 - CONST_BITS)  ; FIX(2.562915447)
F_3_072 equ DESCALE(3299298341, 30 - CONST_BITS)  ; FIX(3.072711026)
%endif

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_idct_islow_sse2)

EXTN(jconst_idct_islow_sse2):

PW_F130_F054   times 4  dw  (F_0_541 + F_0_765),  F_0_541
PW_F054_MF130  times 4  dw  F_0_541, (F_0_541 - F_1_847)
PW_MF078_F117  times 4  dw  (F_1_175 - F_1_961),  F_1_175
PW_F117_F078   times 4  dw  F_1_175, (F_1_175 - F_0_390)
PW_MF060_MF089 times 4  dw  (F_0_298 - F_0_899), -F_0_899
PW_MF089_F060  times 4  dw -F_0_899, (F_1_501 - F_0_899)
PW_MF050_MF256 times 4  dw  (F_2_053 - F_2_562), -F_2_562
PW_MF256_F050  times 4  dw -F_2_562, (F_3_072 - F_2_562)
PD_DESCALE_P1  times 4  dd  1 << (DESCALE_P1 - 1)
PD_DESCALE_P2  times 4  dd  1 << (DESCALE_P2 - 1)
PB_CENTERJSAMP times 16 db  CENTERJSAMPLE

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform dequantization and inverse DCT on one block of coefficients.
;
; GLOBAL(void)
; jsimd_idct_islow_sse2(void *dct_table, JCOEFPTR coef_block,
;                       JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; jpeg_component_info *compptr
%define coef_block(b)  (b) + 12         ; JCOEFPTR coef_block
%define output_buf(b)  (b) + 16         ; JSAMPARRAY output_buf
%define output_col(b)  (b) + 20         ; JDIMENSION output_col

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_XMMWORD
                                        ; xmmword wk[WK_NUM]
%define WK_NUM         12

    align       32
    GLOBAL_FUNCTION(jsimd_idct_islow_sse2)

EXTN(jsimd_idct_islow_sse2):
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
    push        esi
    push        edi

    GET_GOT     ebx                     ; get GOT address

    ; ---- Pass 1: process columns from input.

;   mov         eax, [original_ebp]
    mov         edx, POINTER [dct_table(eax)]    ; quantptr
    mov         esi, JCOEFPTR [coef_block(eax)]  ; inptr

%ifndef NO_ZERO_COLUMN_TEST_ISLOW_SSE2
    mov         eax, dword [DWBLOCK(1,0,esi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,esi,SIZEOF_JCOEF)]
    jnz         near .columnDCT

    movdqa      xmm0, XMMWORD [XMMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    por         xmm0, XMMWORD [XMMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    por         xmm1, XMMWORD [XMMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    por         xmm0, XMMWORD [XMMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    por         xmm1, XMMWORD [XMMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    por         xmm0, XMMWORD [XMMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    por         xmm1, xmm0
    packsswb    xmm1, xmm1
    packsswb    xmm1, xmm1
    movd        eax, xmm1
    test        eax, eax
    jnz         short .columnDCT

    ; -- AC terms all zero

    movdqa      xmm5, XMMWORD [XMMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    pmullw      xmm5, XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    psllw       xmm5, PASS1_BITS

    movdqa      xmm4, xmm5              ; xmm5=in0=(00 01 02 03 04 05 06 07)
    punpcklwd   xmm5, xmm5              ; xmm5=(00 00 01 01 02 02 03 03)
    punpckhwd   xmm4, xmm4              ; xmm4=(04 04 05 05 06 06 07 07)

    pshufd      xmm7, xmm5, 0x00        ; xmm7=col0=(00 00 00 00 00 00 00 00)
    pshufd      xmm6, xmm5, 0x55        ; xmm6=col1=(01 01 01 01 01 01 01 01)
    pshufd      xmm1, xmm5, 0xAA        ; xmm1=col2=(02 02 02 02 02 02 02 02)
    pshufd      xmm5, xmm5, 0xFF        ; xmm5=col3=(03 03 03 03 03 03 03 03)
    pshufd      xmm0, xmm4, 0x00        ; xmm0=col4=(04 04 04 04 04 04 04 04)
    pshufd      xmm3, xmm4, 0x55        ; xmm3=col5=(05 05 05 05 05 05 05 05)
    pshufd      xmm2, xmm4, 0xAA        ; xmm2=col6=(06 06 06 06 06 06 06 06)
    pshufd      xmm4, xmm4, 0xFF        ; xmm4=col7=(07 07 07 07 07 07 07 07)

    movdqa      XMMWORD [wk(8)], xmm6   ; wk(8)=col1
    movdqa      XMMWORD [wk(9)], xmm5   ; wk(9)=col3
    movdqa      XMMWORD [wk(10)], xmm3  ; wk(10)=col5
    movdqa      XMMWORD [wk(11)], xmm4  ; wk(11)=col7
    jmp         near .column_end
    ALIGNX      16, 7
%endif
.columnDCT:

    ; -- Even part

    movdqa      xmm0, XMMWORD [XMMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    pmullw      xmm0, XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm1, XMMWORD [XMMBLOCK(2,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    movdqa      xmm2, XMMWORD [XMMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    pmullw      xmm2, XMMWORD [XMMBLOCK(4,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm3, XMMWORD [XMMBLOCK(6,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    ; (Original)
    ; z1 = (z2 + z3) * 0.541196100;
    ; tmp2 = z1 + z3 * -1.847759065;
    ; tmp3 = z1 + z2 * 0.765366865;
    ;
    ; (This implementation)
    ; tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065);
    ; tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100;

    movdqa      xmm4, xmm1              ; xmm1=in2=z2
    movdqa      xmm5, xmm1
    punpcklwd   xmm4, xmm3              ; xmm3=in6=z3
    punpckhwd   xmm5, xmm3
    movdqa      xmm1, xmm4
    movdqa      xmm3, xmm5
    pmaddwd     xmm4, [GOTOFF(ebx,PW_F130_F054)]   ; xmm4=tmp3L
    pmaddwd     xmm5, [GOTOFF(ebx,PW_F130_F054)]   ; xmm5=tmp3H
    pmaddwd     xmm1, [GOTOFF(ebx,PW_F054_MF130)]  ; xmm1=tmp2L
    pmaddwd     xmm3, [GOTOFF(ebx,PW_F054_MF130)]  ; xmm3=tmp2H

    movdqa      xmm6, xmm0
    paddw       xmm0, xmm2              ; xmm0=in0+in4
    psubw       xmm6, xmm2              ; xmm6=in0-in4

    pxor        xmm7, xmm7
    pxor        xmm2, xmm2
    punpcklwd   xmm7, xmm0              ; xmm7=tmp0L
    punpckhwd   xmm2, xmm0              ; xmm2=tmp0H
    psrad       xmm7, (16-CONST_BITS)   ; psrad xmm7,16 & pslld xmm7,CONST_BITS
    psrad       xmm2, (16-CONST_BITS)   ; psrad xmm2,16 & pslld xmm2,CONST_BITS

    movdqa      xmm0, xmm7
    paddd       xmm7, xmm4              ; xmm7=tmp10L
    psubd       xmm0, xmm4              ; xmm0=tmp13L
    movdqa      xmm4, xmm2
    paddd       xmm2, xmm5              ; xmm2=tmp10H
    psubd       xmm4, xmm5              ; xmm4=tmp13H

    movdqa      XMMWORD [wk(0)], xmm7   ; wk(0)=tmp10L
    movdqa      XMMWORD [wk(1)], xmm2   ; wk(1)=tmp10H
    movdqa      XMMWORD [wk(2)], xmm0   ; wk(2)=tmp13L
    movdqa      XMMWORD [wk(3)], xmm4   ; wk(3)=tmp13H

    pxor        xmm5, xmm5
    pxor        xmm7, xmm7
    punpcklwd   xmm5, xmm6              ; xmm5=tmp1L
    punpckhwd   xmm7, xmm6              ; xmm7=tmp1H
    psrad       xmm5, (16-CONST_BITS)   ; psrad xmm5,16 & pslld xmm5,CONST_BITS
    psrad       xmm7, (16-CONST_BITS)   ; psrad xmm7,16 & pslld xmm7,CONST_BITS

    movdqa      xmm2, xmm5
    paddd       xmm5, xmm1              ; xmm5=tmp11L
    psubd       xmm2, xmm1              ; xmm2=tmp12L
    movdqa      xmm0, xmm7
    paddd       xmm7, xmm3              ; xmm7=tmp11H
    psubd       xmm0, xmm3              ; xmm0=tmp12H

    movdqa      XMMWORD [wk(4)], xmm5   ; wk(4)=tmp11L
    movdqa      XMMWORD [wk(5)], xmm7   ; wk(5)=tmp11H
    movdqa      XMMWORD [wk(6)], xmm2   ; wk(6)=tmp12L
    movdqa      XMMWORD [wk(7)], xmm0   ; wk(7)=tmp12H

    ; -- Odd part

    movdqa      xmm4, XMMWORD [XMMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movdqa      xmm6, XMMWORD [XMMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    pmullw      xmm4, XMMWORD [XMMBLOCK(1,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm6, XMMWORD [XMMBLOCK(3,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    movdqa      xmm3, XMMWORD [XMMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    pmullw      xmm1, XMMWORD [XMMBLOCK(5,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    pmullw      xmm3, XMMWORD [XMMBLOCK(7,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    movdqa      xmm5, xmm6
    movdqa      xmm7, xmm4
    paddw       xmm5, xmm3              ; xmm5=z3
    paddw       xmm7, xmm1              ; xmm7=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movdqa      xmm2, xmm5
    movdqa      xmm0, xmm5
    punpcklwd   xmm2, xmm7
    punpckhwd   xmm0, xmm7
    movdqa      xmm5, xmm2
    movdqa      xmm7, xmm0
    pmaddwd     xmm2, [GOTOFF(ebx,PW_MF078_F117)]  ; xmm2=z3L
    pmaddwd     xmm0, [GOTOFF(ebx,PW_MF078_F117)]  ; xmm0=z3H
    pmaddwd     xmm5, [GOTOFF(ebx,PW_F117_F078)]   ; xmm5=z4L
    pmaddwd     xmm7, [GOTOFF(ebx,PW_F117_F078)]   ; xmm7=z4H

    movdqa      XMMWORD [wk(10)], xmm2  ; wk(10)=z3L
    movdqa      XMMWORD [wk(11)], xmm0  ; wk(11)=z3H

    ; (Original)
    ; z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
    ; tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
    ; tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
    ; z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
    ; tmp0 += z1 + z3;  tmp1 += z2 + z4;
    ; tmp2 += z2 + z3;  tmp3 += z1 + z4;
    ;
    ; (This implementation)
    ; tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
    ; tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
    ; tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
    ; tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
    ; tmp0 += z3;  tmp1 += z4;
    ; tmp2 += z3;  tmp3 += z4;

    movdqa      xmm2, xmm3
    movdqa      xmm0, xmm3
    punpcklwd   xmm2, xmm4
    punpckhwd   xmm0, xmm4
    movdqa      xmm3, xmm2
    movdqa      xmm4, xmm0
    pmaddwd     xmm2, [GOTOFF(ebx,PW_MF060_MF089)]  ; xmm2=tmp0L
    pmaddwd     xmm0, [GOTOFF(ebx,PW_MF060_MF089)]  ; xmm0=tmp0H
    pmaddwd     xmm3, [GOTOFF(ebx,PW_MF089_F060)]   ; xmm3=tmp3L
    pmaddwd     xmm4, [GOTOFF(ebx,PW_MF089_F060)]   ; xmm4=tmp3H

    paddd       xmm2, XMMWORD [wk(10)]  ; xmm2=tmp0L
    paddd       xmm0, XMMWORD [wk(11)]  ; xmm0=tmp0H
    paddd       xmm3, xmm5              ; xmm3=tmp3L
    paddd       xmm4, xmm7              ; xmm4=tmp3H

    movdqa      XMMWORD [wk(8)], xmm2   ; wk(8)=tmp0L
    movdqa      XMMWORD [wk(9)], xmm0   ; wk(9)=tmp0H

    movdqa      xmm2, xmm1
    movdqa      xmm0, xmm1
    punpcklwd   xmm2, xmm6
    punpckhwd   xmm0, xmm6
    movdqa      xmm1, xmm2
    movdqa      xmm6, xmm0
    pmaddwd     xmm2, [GOTOFF(ebx,PW_MF050_MF256)]  ; xmm2=tmp1L
    pmaddwd     xmm0, [GOTOFF(ebx,PW_MF050_MF256)]  ; xmm0=tmp1H
    pmaddwd     xmm1, [GOTOFF(ebx,PW_MF256_F050)]   ; xmm1=tmp2L
    pmaddwd     xmm6, [GOTOFF(ebx,PW_MF256_F050)]   ; xmm6=tmp2H

    paddd       xmm2, xmm5              ; xmm2=tmp1L
    paddd       xmm0, xmm7              ; xmm0=tmp1H
    paddd       xmm1, XMMWORD [wk(10)]  ; xmm1=tmp2L
    paddd       xmm6, XMMWORD [wk(11)]  ; xmm6=tmp2H

    movdqa      XMMWORD [wk(10)], xmm2  ; wk(10)=tmp1L
    movdqa      XMMWORD [wk(11)], xmm0  ; wk(11)=tmp1H

    ; -- Final output stage

    movdqa      xmm5, XMMWORD [wk(0)]   ; xmm5=tmp10L
    movdqa      xmm7, XMMWORD [wk(1)]   ; xmm7=tmp10H

    movdqa      xmm2, xmm5
    movdqa      xmm0, xmm7
    paddd       xmm5, xmm3              ; xmm5=data0L
    paddd       xmm7, xmm4              ; xmm7=data0H
    psubd       xmm2, xmm3              ; xmm2=data7L
    psubd       xmm0, xmm4              ; xmm0=data7H

    movdqa      xmm3, [GOTOFF(ebx,PD_DESCALE_P1)]  ; xmm3=[PD_DESCALE_P1]

    paddd       xmm5, xmm3
    paddd       xmm7, xmm3
    psrad       xmm5, DESCALE_P1
    psrad       xmm7, DESCALE_P1
    paddd       xmm2, xmm3
    paddd       xmm0, xmm3
    psrad       xmm2, DESCALE_P1
    psrad       xmm0, DESCALE_P1

    packssdw    xmm5, xmm7              ; xmm5=data0=(00 01 02 03 04 05 06 07)
    packssdw    xmm2, xmm0              ; xmm2=data7=(70 71 72 73 74 75 76 77)

    movdqa      xmm4, XMMWORD [wk(4)]   ; xmm4=tmp11L
    movdqa      xmm3, XMMWORD [wk(5)]   ; xmm3=tmp11H

    movdqa      xmm7, xmm4
    movdqa      xmm0, xmm3
    paddd       xmm4, xmm1              ; xmm4=data1L
    paddd       xmm3, xmm6              ; xmm3=data1H
    psubd       xmm7, xmm1              ; xmm7=data6L
    psubd       xmm0, xmm6              ; xmm0=data6H

    movdqa      xmm1, [GOTOFF(ebx,PD_DESCALE_P1)]  ; xmm1=[PD_DESCALE_P1]

    paddd       xmm4, xmm1
    paddd       xmm3, xmm1
    psrad       xmm4, DESCALE_P1
    psrad       xmm3, DESCALE_P1
    paddd       xmm7, xmm1
    paddd       xmm0, xmm1
    psrad       xmm7, DESCALE_P1
    psrad       xmm0, DESCALE_P1

    packssdw    xmm4, xmm3              ; xmm4=data1=(10 11 12 13 14 15 16 17)
    packssdw    xmm7, xmm0              ; xmm7=data6=(60 61 62 63 64 65 66 67)

    movdqa      xmm6, xmm5              ; transpose coefficients(phase 1)
    punpcklwd   xmm5, xmm4              ; xmm5=(00 10 01 11 02 12 03 13)
    punpckhwd   xmm6, xmm4              ; xmm6=(04 14 05 15 06 16 07 17)
    movdqa      xmm1, xmm7              ; transpose coefficients(phase 1)
    punpcklwd   xmm7, xmm2              ; xmm7=(60 70 61 71 62 72 63 73)
    punpckhwd   xmm1, xmm2              ; xmm1=(64 74 65 75 66 76 67 77)

    movdqa      xmm3, XMMWORD [wk(6)]   ; xmm3=tmp12L
    movdqa      xmm0, XMMWORD [wk(7)]   ; xmm0=tmp12H
    movdqa      xmm4, XMMWORD [wk(10)]  ; xmm4=tmp1L
    movdqa      xmm2, XMMWORD [wk(11)]  ; xmm2=tmp1H

    movdqa      XMMWORD [wk(0)], xmm5   ; wk(0)=(00 10 01 11 02 12 03 13)
    movdqa      XMMWORD [wk(1)], xmm6   ; wk(1)=(04 14 05 15 06 16 07 17)
    movdqa      XMMWORD [wk(4)], xmm7   ; wk(4)=(60 70 61 71 62 72 63 73)
    movdqa      XMMWORD [wk(5)], xmm1   ; wk(5)=(64 74 65 75 66 76 67 77)

    movdqa      xmm5, xmm3
    movdqa      xmm6, xmm0
    paddd       xmm3, xmm4              ; xmm3=data2L
    paddd       xmm0, xmm2              ; xmm0=data2H
    psubd       xmm5, xmm4              ; xmm5=data5L
    psubd       xmm6, xmm2              ; xmm6=data5H

    movdqa      xmm7, [GOTOFF(ebx,PD_DESCALE_P1)]  ; xmm7=[PD_DESCALE_P1]

    paddd       xmm3, xmm7
    paddd       xmm0, xmm7
    psrad       xmm3, DESCALE_P1
    psrad       xmm0, DESCALE_P1
    paddd       xmm5, xmm7
    paddd       xmm6, xmm7
    psrad       xmm5, DESCALE_P1
    psrad       xmm6, DESCALE_P1

    packssdw    xmm3, xmm0              ; xmm3=data2=(20 21 22 23 24 25 26 27)
    packssdw    xmm5, xmm6              ; xmm5=data5=(50 51 52 53 54 55 56 57)

    movdqa      xmm1, XMMWORD [wk(2)]   ; xmm1=tmp13L
    movdqa      xmm4, XMMWORD [wk(3)]   ; xmm4=tmp13H
    movdqa      xmm2, XMMWORD [wk(8)]   ; xmm2=tmp0L
    movdqa      xmm7, XMMWORD [wk(9)]   ; xmm7=tmp0H

    movdqa      xmm0, xmm1
    movdqa      xmm6, xmm4
    paddd       xmm1, xmm2              ; xmm1=data3L
    paddd       xmm4, xmm7              ; xmm4=data3H
    psubd       xmm0, xmm2              ; xmm0=data4L
    psubd       xmm6, xmm7              ; xmm6=data4H

    movdqa      xmm2, [GOTOFF(ebx,PD_DESCALE_P1)]  ; xmm2=[PD_DESCALE_P1]

    paddd       xmm1, xmm2
    paddd       xmm4, xmm2
    psrad       xmm1, DESCALE_P1
    psrad       xmm4, DESCALE_P1
    paddd       xmm0, xmm2
    paddd       xmm6, xmm2
    psrad       xmm0, DESCALE_P1
    psrad       xmm6, DESCALE_P1

    packssdw    xmm1, xmm4              ; xmm1=data3=(30 31 32 33 34 35 36 37)
    packssdw    xmm0, xmm6              ; xmm0=data4=(40 41 42 43 44 45 46 47)

    movdqa      xmm7, XMMWORD [wk(0)]   ; xmm7=(00 10 01 11 02 12 03 13)
    movdqa      xmm2, XMMWORD [wk(1)]   ; xmm2=(04 14 05 15 06 16 07 17)

    movdqa      xmm4, xmm3              ; transpose coefficients(phase 1)
    punpcklwd   xmm3, xmm1              ; xmm3=(20 30 21 31 22 32 23 33)
    punpckhwd   xmm4, xmm1              ; xmm4=(24 34 25 35 26 36 27 37)
    movdqa      xmm6, xmm0              ; transpose coefficients(phase 1)
    punpcklwd   xmm0, xmm5              ; xmm0=(40 50 41 51 42 52 43 53)
    punpckhwd   xmm6, xmm5              ; xmm6=(44 54 45 55 46 56 47 57)

    movdqa      xmm1, xmm7              ; transpose coefficients(phase 2)
    punpckldq   xmm7, xmm3              ; xmm7=(00 10 20 30 01 11 21 31)
    punpckhdq   xmm1, xmm3              ; xmm1=(02 12 22 32 03 13 23 33)
    movdqa      xmm5, xmm2              ; transpose coefficients(phase 2)
    punpckldq   xmm2, xmm4              ; xmm2=(04 14 24 34 05 15 25 35)
    punpckhdq   xmm5, xmm4              ; xmm5=(06 16 26 36 07 17 27 37)

    movdqa      xmm3, XMMWORD [wk(4)]   ; xmm3=(60 70 61 71 62 72 63 73)
    movdqa      xmm4, XMMWORD [wk(5)]   ; xmm4=(64 74 65 75 66 76 67 77)

    movdqa      XMMWORD [wk(6)], xmm2   ; wk(6)=(04 14 24 34 05 15 25 35)
    movdqa      XMMWORD [wk(7)], xmm5   ; wk(7)=(06 16 26 36 07 17 27 37)

    movdqa      xmm2, xmm0              ; transpose coefficients(phase 2)
    punpckldq   xmm0, xmm3              ; xmm0=(40 50 60 70 41 51 61 71)
    punpckhdq   xmm2, xmm3              ; xmm2=(42 52 62 72 43 53 63 73)
    movdqa      xmm5, xmm6              ; transpose coefficients(phase 2)
    punpckldq   xmm6, xmm4              ; xmm6=(44 54 64 74 45 55 65 75)
    punpckhdq   xmm5, xmm4              ; xmm5=(46 56 66 76 47 57 67 77)

    movdqa      xmm3, xmm7              ; transpose coefficients(phase 3)
    punpcklqdq  xmm7, xmm0              ; xmm7=col0=(00 10 20 30 40 50 60 70)
    punpckhqdq  xmm3, xmm0              ; xmm3=col1=(01 11 21 31 41 51 61 71)
    movdqa      xmm4, xmm1              ; transpose coefficients(phase 3)
    punpcklqdq  xmm1, xmm2              ; xmm1=col2=(02 12 22 32 42 52 62 72)
    punpckhqdq  xmm4, xmm2              ; xmm4=col3=(03 13 23 33 43 53 63 73)

    movdqa      xmm0, XMMWORD [wk(6)]   ; xmm0=(04 14 24 34 05 15 25 35)
    movdqa      xmm2, XMMWORD [wk(7)]   ; xmm2=(06 16 26 36 07 17 27 37)

    movdqa      XMMWORD [wk(8)], xmm3   ; wk(8)=col1
    movdqa      XMMWORD [wk(9)], xmm4   ; wk(9)=col3

    movdqa      xmm3, xmm0              ; transpose coefficients(phase 3)
    punpcklqdq  xmm0, xmm6              ; xmm0=col4=(04 14 24 34 44 54 64 74)
    punpckhqdq  xmm3, xmm6              ; xmm3=col5=(05 15 25 35 45 55 65 75)
    movdqa      xmm4, xmm2              ; transpose coefficients(phase 3)
    punpcklqdq  xmm2, xmm5              ; xmm2=col6=(06 16 26 36 46 56 66 76)
    punpckhqdq  xmm4, xmm5              ; xmm4=col7=(07 17 27 37 47 57 67 77)

    movdqa      XMMWORD [wk(10)], xmm3  ; wk(10)=col5
    movdqa      XMMWORD [wk(11)], xmm4  ; wk(11)=col7
.column_end:

    ; -- Prefetch the next coefficient block

    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 0*32]
    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 1*32]
    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 2*32]
    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 3*32]

    ; ---- Pass 2: process rows from work array, store into output array.

    mov         eax, [original_ebp]
    mov         edi, JSAMPARRAY [output_buf(eax)]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [output_col(eax)]

    ; -- Even part

    ; xmm7=col0, xmm1=col2, xmm0=col4, xmm2=col6

    ; (Original)
    ; z1 = (z2 + z3) * 0.541196100;
    ; tmp2 = z1 + z3 * -1.847759065;
    ; tmp3 = z1 + z2 * 0.765366865;
    ;
    ; (This implementation)
    ; tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065);
    ; tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100;

    movdqa      xmm6, xmm1              ; xmm1=in2=z2
    movdqa      xmm5, xmm1
    punpcklwd   xmm6, xmm2              ; xmm2=in6=z3
    punpckhwd   xmm5, xmm2
    movdqa      xmm1, xmm6
    movdqa      xmm2, xmm5
    pmaddwd     xmm6, [GOTOFF(ebx,PW_F130_F054)]   ; xmm6=tmp3L
    pmaddwd     xmm5, [GOTOFF(ebx,PW_F130_F054)]   ; xmm5=tmp3H
    pmaddwd     xmm1, [GOTOFF(ebx,PW_F054_MF130)]  ; xmm1=tmp2L
    pmaddwd     xmm2, [GOTOFF(ebx,PW_F054_MF130)]  ; xmm2=tmp2H

    movdqa      xmm3, xmm7
    paddw       xmm7, xmm0              ; xmm7=in0+in4
    psubw       xmm3, xmm0              ; xmm3=in0-in4

    pxor        xmm4, xmm4
    pxor        xmm0, xmm0
    punpcklwd   xmm4, xmm7              ; xmm4=tmp0L
    punpckhwd   xmm0, xmm7              ; xmm0=tmp0H
    psrad       xmm4, (16-CONST_BITS)   ; psrad xmm4,16 & pslld xmm4,CONST_BITS
    psrad       xmm0, (16-CONST_BITS)   ; psrad xmm0,16 & pslld xmm0,CONST_BITS

    movdqa      xmm7, xmm4
    paddd       xmm4, xmm6              ; xmm4=tmp10L
    psubd       xmm7, xmm6              ; xmm7=tmp13L
    movdqa      xmm6, xmm0
    paddd       xmm0, xmm5              ; xmm0=tmp10H
    psubd       xmm6, xmm5              ; xmm6=tmp13H

    movdqa      XMMWORD [wk(0)], xmm4   ; wk(0)=tmp10L
    movdqa      XMMWORD [wk(1)], xmm0   ; wk(1)=tmp10H
    movdqa      XMMWORD [wk(2)], xmm7   ; wk(2)=tmp13L
    movdqa      XMMWORD [wk(3)], xmm6   ; wk(3)=tmp13H

    pxor        xmm5, xmm5
    pxor        xmm4, xmm4
    punpcklwd   xmm5, xmm3              ; xmm5=tmp1L
    punpckhwd   xmm4, xmm3              ; xmm4=tmp1H
    psrad       xmm5, (16-CONST_BITS)   ; psrad xmm5,16 & pslld xmm5,CONST_BITS
    psrad       xmm4, (16-CONST_BITS)   ; psrad xmm4,16 & pslld xmm4,CONST_BITS

    movdqa      xmm0, xmm5
    paddd       xmm5, xmm1              ; xmm5=tmp11L
    psubd       xmm0, xmm1              ; xmm0=tmp12L
    movdqa      xmm7, xmm4
    paddd       xmm4, xmm2              ; xmm4=tmp11H
    psubd       xmm7, xmm2              ; xmm7=tmp12H

    movdqa      XMMWORD [wk(4)], xmm5   ; wk(4)=tmp11L
    movdqa      XMMWORD [wk(5)], xmm4   ; wk(5)=tmp11H
    movdqa      XMMWORD [wk(6)], xmm0   ; wk(6)=tmp12L
    movdqa      XMMWORD [wk(7)], xmm7   ; wk(7)=tmp12H

    ; -- Odd part

    movdqa      xmm6, XMMWORD [wk(9)]   ; xmm6=col3
    movdqa      xmm3, XMMWORD [wk(8)]   ; xmm3=col1
    movdqa      xmm1, XMMWORD [wk(11)]  ; xmm1=col7
    movdqa      xmm2, XMMWORD [wk(10)]  ; xmm2=col5

    movdqa      xmm5, xmm6
    movdqa      xmm4, xmm3
    paddw       xmm5, xmm1              ; xmm5=z3
    paddw       xmm4, xmm2              ; xmm4=z4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    movdqa      xmm0, xmm5
    movdqa      xmm7, xmm5
    punpcklwd   xmm0, xmm4
    punpckhwd   xmm7, xmm4
    movdqa      xmm5, xmm0
    movdqa      xmm4, xmm7
    pmaddwd     xmm0, [GOTOFF(ebx,PW_MF078_F117)]  ; xmm0=z3L
    pmaddwd     xmm7, [GOTOFF(ebx,PW_MF078_F117)]  ; xmm7=z3H
    pmaddwd     xmm5, [GOTOFF(ebx,PW_F117_F078)]   ; xmm5=z4L
    pmaddwd     xmm4, [GOTOFF(ebx,PW_F117_F078)]   ; xmm4=z4H

    movdqa      XMMWORD [wk(10)], xmm0  ; wk(10)=z3L
    movdqa      XMMWORD [wk(11)], xmm7  ; wk(11)=z3H

    ; (Original)
    ; z1 = tmp0 + tmp3;  z2 = tmp1 + tmp2;
    ; tmp0 = tmp0 * 0.298631336;  tmp1 = tmp1 * 2.053119869;
    ; tmp2 = tmp2 * 3.072711026;  tmp3 = tmp3 * 1.501321110;
    ; z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
    ; tmp0 += z1 + z3;  tmp1 += z2 + z4;
    ; tmp2 += z2 + z3;  tmp3 += z1 + z4;
    ;
    ; (This implementation)
    ; tmp0 = tmp0 * (0.298631336 - 0.899976223) + tmp3 * -0.899976223;
    ; tmp1 = tmp1 * (2.053119869 - 2.562915447) + tmp2 * -2.562915447;
    ; tmp2 = tmp1 * -2.562915447 + tmp2 * (3.072711026 - 2.562915447);
    ; tmp3 = tmp0 * -0.899976223 + tmp3 * (1.501321110 - 0.899976223);
    ; tmp0 += z3;  tmp1 += z4;
    ; tmp2 += z3;  tmp3 += z4;

    movdqa      xmm0, xmm1
    movdqa      xmm7, xmm1
    punpcklwd   xmm0, xmm3
    punpckhwd   xmm7, xmm3
    movdqa      xmm1, xmm0
    movdqa      xmm3, xmm7
    pmaddwd     xmm0, [GOTOFF(ebx,PW_MF060_MF089)]  ; xmm0=tmp0L
    pmaddwd     xmm7, [GOTOFF(ebx,PW_MF060_MF089)]  ; xmm7=tmp0H
    pmaddwd     xmm1, [GOTOFF(ebx,PW_MF089_F060)]   ; xmm1=tmp3L
    pmaddwd     xmm3, [GOTOFF(ebx,PW_MF089_F060)]   ; xmm3=tmp3H

    paddd       xmm0, XMMWORD [wk(10)]  ; xmm0=tmp0L
    paddd       xmm7, XMMWORD [wk(11)]  ; xmm7=tmp0H
    paddd       xmm1, xmm5              ; xmm1=tmp3L
    paddd       xmm3, xmm4              ; xmm3=tmp3H

    movdqa      XMMWORD [wk(8)], xmm0   ; wk(8)=tmp0L
    movdqa      XMMWORD [wk(9)], xmm7   ; wk(9)=tmp0H

    movdqa      xmm0, xmm2
    movdqa      xmm7, xmm2
    punpcklwd   xmm0, xmm6
    punpckhwd   xmm7, xmm6
    movdqa      xmm2, xmm0
    movdqa      xmm6, xmm7
    pmaddwd     xmm0, [GOTOFF(ebx,PW_MF050_MF256)]  ; xmm0=tmp1L
    pmaddwd     xmm7, [GOTOFF(ebx,PW_MF050_MF256)]  ; xmm7=tmp1H
    pmaddwd     xmm2, [GOTOFF(ebx,PW_MF256_F050)]   ; xmm2=tmp2L
    pmaddwd     xmm6, [GOTOFF(ebx,PW_MF256_F050)]   ; xmm6=tmp2H

    paddd       xmm0, xmm5              ; xmm0=tmp1L
    paddd       xmm7, xmm4              ; xmm7=tmp1H
    paddd       xmm2, XMMWORD [wk(10)]  ; xmm2=tmp2L
    paddd       xmm6, XMMWORD [wk(11)]  ; xmm6=tmp2H

    movdqa      XMMWORD [wk(10)], xmm0  ; wk(10)=tmp1L
    movdqa      XMMWORD [wk(11)], xmm7  ; wk(11)=tmp1H

    ; -- Final output stage

    movdqa      xmm5, XMMWORD [wk(0)]   ; xmm5=tmp10L
    movdqa      xmm4, XMMWORD [wk(1)]   ; xmm4=tmp10H

    movdqa      xmm0, xmm5
    movdqa      xmm7, xmm4
    paddd       xmm5, xmm1              ; xmm5=data0L
    paddd       xmm4, xmm3              ; xmm4=data0H
    psubd       xmm0, xmm1              ; xmm0=data7L
    psubd       xmm7, xmm3              ; xmm7=data7H

    movdqa      xmm1, [GOTOFF(ebx,PD_DESCALE_P2)]  ; xmm1=[PD_DESCALE_P2]

    paddd       xmm5, xmm1
    paddd       xmm4, xmm1
    psrad       xmm5, DESCALE_P2
    psrad       xmm4, DESCALE_P2
    paddd       xmm0, xmm1
    paddd       xmm7, xmm1
    psrad       xmm0, DESCALE_P2
    psrad       xmm7, DESCALE_P2

    packssdw    xmm5, xmm4              ; xmm5=data0=(00 10 20 30 40 50 60 70)
    packssdw    xmm0, xmm7              ; xmm0=data7=(07 17 27 37 47 57 67 77)

    movdqa      xmm3, XMMWORD [wk(4)]   ; xmm3=tmp11L
    movdqa      xmm1, XMMWORD [wk(5)]   ; xmm1=tmp11H

    movdqa      xmm4, xmm3
    movdqa      xmm7, xmm1
    paddd       xmm3, xmm2              ; xmm3=data1L
    paddd       xmm1, xmm6              ; xmm1=data1H
    psubd       xmm4, xmm2              ; xmm4=data6L
    psubd       xmm7, xmm6              ; xmm7=data6H

    movdqa      xmm2, [GOTOFF(ebx,PD_DESCALE_P2)]  ; xmm2=[PD_DESCALE_P2]

    paddd       xmm3, xmm2
    paddd       xmm1, xmm2
    psrad       xmm3, DESCALE_P2
    psrad       xmm1, DESCALE_P2
    paddd       xmm4, xmm2
    paddd       xmm7, xmm2
    psrad       xmm4, DESCALE_P2
    psrad       xmm7, DESCALE_P2

    packssdw    xmm3, xmm1              ; xmm3=data1=(01 11 21 31 41 51 61 71)
    packssdw    xmm4, xmm7              ; xmm4=data6=(06 16 26 36 46 56 66 76)

    packsswb    xmm5, xmm4              ; xmm5=(00 10 20 30 40 50 60 70 06 16 26 36 46 56 66 76)
    packsswb    xmm3, xmm0              ; xmm3=(01 11 21 31 41 51 61 71 07 17 27 37 47 57 67 77)

    movdqa      xmm6, XMMWORD [wk(6)]   ; xmm6=tmp12L
    movdqa      xmm2, XMMWORD [wk(7)]   ; xmm2=tmp12H
    movdqa      xmm1, XMMWORD [wk(10)]  ; xmm1=tmp1L
    movdqa      xmm7, XMMWORD [wk(11)]  ; xmm7=tmp1H

    movdqa      XMMWORD [wk(0)], xmm5   ; wk(0)=(00 10 20 30 40 50 60 70 06 16 26 36 46 56 66 76)
    movdqa      XMMWORD [wk(1)], xmm3   ; wk(1)=(01 11 21 31 41 51 61 71 07 17 27 37 47 57 67 77)

    movdqa      xmm4, xmm6
    movdqa      xmm0, xmm2
    paddd       xmm6, xmm1              ; xmm6=data2L
    paddd       xmm2, xmm7              ; xmm2=data2H
    psubd       xmm4, xmm1              ; xmm4=data5L
    psubd       xmm0, xmm7              ; xmm0=data5H

    movdqa      xmm5, [GOTOFF(ebx,PD_DESCALE_P2)]  ; xmm5=[PD_DESCALE_P2]

    paddd       xmm6, xmm5
    paddd       xmm2, xmm5
    psrad       xmm6, DESCALE_P2
    psrad       xmm2, DESCALE_P2
    paddd       xmm4, xmm5
    paddd       xmm0, xmm5
    psrad       xmm4, DESCALE_P2
    psrad       xmm0, DESCALE_P2

    packssdw    xmm6, xmm2              ; xmm6=data2=(02 12 22 32 42 52 62 72)
    packssdw    xmm4, xmm0              ; xmm4=data5=(05 15 25 35 45 55 65 75)

    movdqa      xmm3, XMMWORD [wk(2)]   ; xmm3=tmp13L
    movdqa      xmm1, XMMWORD [wk(3)]   ; xmm1=tmp13H
    movdqa      xmm7, XMMWORD [wk(8)]   ; xmm7=tmp0L
    movdqa      xmm5, XMMWORD [wk(9)]   ; xmm5=tmp0H

    movdqa      xmm2, xmm3
    movdqa      xmm0, xmm1
    paddd       xmm3, xmm7              ; xmm3=data3L
    paddd       xmm1, xmm5              ; xmm1=data3H
    psubd       xmm2, xmm7              ; xmm2=data4L
    psubd       xmm0, xmm5              ; xmm0=data4H

    movdqa      xmm7, [GOTOFF(ebx,PD_DESCALE_P2)]  ; xmm7=[PD_DESCALE_P2]

    paddd       xmm3, xmm7
    paddd       xmm1, xmm7
    psrad       xmm3, DESCALE_P2
    psrad       xmm1, DESCALE_P2
    paddd       xmm2, xmm7
    paddd       xmm0, xmm7
    psrad       xmm2, DESCALE_P2
    psrad       xmm0, DESCALE_P2

    movdqa      xmm5, [GOTOFF(ebx,PB_CENTERJSAMP)]  ; xmm5=[PB_CENTERJSAMP]

    packssdw    xmm3, xmm1             ; xmm3=data3=(03 13 23 33 43 53 63 73)
    packssdw    xmm2, xmm0             ; xmm2=data4=(04 14 24 34 44 54 64 74)

    movdqa      xmm7, XMMWORD [wk(0)]  ; xmm7=(00 10 20 30 40 50 60 70 06 16 26 36 46 56 66 76)
    movdqa      xmm1, XMMWORD [wk(1)]  ; xmm1=(01 11 21 31 41 51 61 71 07 17 27 37 47 57 67 77)

    packsswb    xmm6, xmm2             ; xmm6=(02 12 22 32 42 52 62 72 04 14 24 34 44 54 64 74)
    packsswb    xmm3, xmm4             ; xmm3=(03 13 23 33 43 53 63 73 05 15 25 35 45 55 65 75)

    paddb       xmm7, xmm5
    paddb       xmm1, xmm5
    paddb       xmm6, xmm5
    paddb       xmm3, xmm5

    movdqa      xmm0, xmm7        ; transpose coefficients(phase 1)
    punpcklbw   xmm7, xmm1        ; xmm7=(00 01 10 11 20 21 30 31 40 41 50 51 60 61 70 71)
    punpckhbw   xmm0, xmm1        ; xmm0=(06 07 16 17 26 27 36 37 46 47 56 57 66 67 76 77)
    movdqa      xmm2, xmm6        ; transpose coefficients(phase 1)
    punpcklbw   xmm6, xmm3        ; xmm6=(02 03 12 13 22 23 32 33 42 43 52 53 62 63 72 73)
    punpckhbw   xmm2, xmm3        ; xmm2=(04 05 14 15 24 25 34 35 44 45 54 55 64 65 74 75)

    movdqa      xmm4, xmm7        ; transpose coefficients(phase 2)
    punpcklwd   xmm7, xmm6        ; xmm7=(00 01 02 03 10 11 12 13 20 21 22 23 30 31 32 33)
    punpckhwd   xmm4, xmm6        ; xmm4=(40 41 42 43 50 51 52 53 60 61 62 63 70 71 72 73)
    movdqa      xmm5, xmm2        ; transpose coefficients(phase 2)
    punpcklwd   xmm2, xmm0        ; xmm2=(04 05 06 07 14 15 16 17 24 25 26 27 34 35 36 37)
    punpckhwd   xmm5, xmm0        ; xmm5=(44 45 46 47 54 55 56 57 64 65 66 67 74 75 76 77)

    movdqa      xmm1, xmm7        ; transpose coefficients(phase 3)
    punpckldq   xmm7, xmm2        ; xmm7=(00 01 02 03 04 05 06 07 10 11 12 13 14 15 16 17)
    punpckhdq   xmm1, xmm2        ; xmm1=(20 21 22 23 24 25 26 27 30 31 32 33 34 35 36 37)
    movdqa      xmm3, xmm4        ; transpose coefficients(phase 3)
    punpckldq   xmm4, xmm5        ; xmm4=(40 41 42 43 44 45 46 47 50 51 52 53 54 55 56 57)
    punpckhdq   xmm3, xmm5        ; xmm3=(60 61 62 63 64 65 66 67 70 71 72 73 74 75 76 77)

    pshufd      xmm6, xmm7, 0x4E  ; xmm6=(10 11 12 13 14 15 16 17 00 01 02 03 04 05 06 07)
    pshufd      xmm0, xmm1, 0x4E  ; xmm0=(30 31 32 33 34 35 36 37 20 21 22 23 24 25 26 27)
    pshufd      xmm2, xmm4, 0x4E  ; xmm2=(50 51 52 53 54 55 56 57 40 41 42 43 44 45 46 47)
    pshufd      xmm5, xmm3, 0x4E  ; xmm5=(70 71 72 73 74 75 76 77 60 61 62 63 64 65 66 67)

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]
    mov         esi, JSAMPROW [edi+2*SIZEOF_JSAMPROW]
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm7
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm1
    mov         edx, JSAMPROW [edi+4*SIZEOF_JSAMPROW]
    mov         esi, JSAMPROW [edi+6*SIZEOF_JSAMPROW]
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm4
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm3

    mov         edx, JSAMPROW [edi+1*SIZEOF_JSAMPROW]
    mov         esi, JSAMPROW [edi+3*SIZEOF_JSAMPROW]
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm6
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm0
    mov         edx, JSAMPROW [edi+5*SIZEOF_JSAMPROW]
    mov         esi, JSAMPROW [edi+7*SIZEOF_JSAMPROW]
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm2
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm5

    pop         edi
    pop         esi
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
