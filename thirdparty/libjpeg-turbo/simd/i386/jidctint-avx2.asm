;
; jidctint.asm - accurate integer IDCT (AVX2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2018, 2020, 2024, D. R. Commander.
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
; In-place 8x8x16-bit inverse matrix transpose using AVX2 instructions
; %1-%4: Input/output registers
; %5-%8: Temp registers

%macro DOTRANSPOSE 8
    ; %5=(00 10 20 30 40 50 60 70  01 11 21 31 41 51 61 71)
    ; %6=(03 13 23 33 43 53 63 73  02 12 22 32 42 52 62 72)
    ; %7=(04 14 24 34 44 54 64 74  05 15 25 35 45 55 65 75)
    ; %8=(07 17 27 37 47 57 67 77  06 16 26 36 46 56 66 76)

    vpermq      %5, %1, 0xD8
    vpermq      %6, %2, 0x72
    vpermq      %7, %3, 0xD8
    vpermq      %8, %4, 0x72
    ; transpose coefficients(phase 1)
    ; %5=(00 10 20 30 01 11 21 31  40 50 60 70 41 51 61 71)
    ; %6=(02 12 22 32 03 13 23 33  42 52 62 72 43 53 63 73)
    ; %7=(04 14 24 34 05 15 25 35  44 54 64 74 45 55 65 75)
    ; %8=(06 16 26 36 07 17 27 37  46 56 66 76 47 57 67 77)

    vpunpcklwd  %1, %5, %6
    vpunpckhwd  %2, %5, %6
    vpunpcklwd  %3, %7, %8
    vpunpckhwd  %4, %7, %8
    ; transpose coefficients(phase 2)
    ; %1=(00 02 10 12 20 22 30 32  40 42 50 52 60 62 70 72)
    ; %2=(01 03 11 13 21 23 31 33  41 43 51 53 61 63 71 73)
    ; %3=(04 06 14 16 24 26 34 36  44 46 54 56 64 66 74 76)
    ; %4=(05 07 15 17 25 27 35 37  45 47 55 57 65 67 75 77)

    vpunpcklwd  %5, %1, %2
    vpunpcklwd  %6, %3, %4
    vpunpckhwd  %7, %1, %2
    vpunpckhwd  %8, %3, %4
    ; transpose coefficients(phase 3)
    ; %5=(00 01 02 03 10 11 12 13  40 41 42 43 50 51 52 53)
    ; %6=(04 05 06 07 14 15 16 17  44 45 46 47 54 55 56 57)
    ; %7=(20 21 22 23 30 31 32 33  60 61 62 63 70 71 72 73)
    ; %8=(24 25 26 27 34 35 36 37  64 65 66 67 74 75 76 77)

    vpunpcklqdq %1, %5, %6
    vpunpckhqdq %2, %5, %6
    vpunpcklqdq %3, %7, %8
    vpunpckhqdq %4, %7, %8
    ; transpose coefficients(phase 4)
    ; %1=(00 01 02 03 04 05 06 07  40 41 42 43 44 45 46 47)
    ; %2=(10 11 12 13 14 15 16 17  50 51 52 53 54 55 56 57)
    ; %3=(20 21 22 23 24 25 26 27  60 61 62 63 64 65 66 67)
    ; %4=(30 31 32 33 34 35 36 37  70 71 72 73 74 75 76 77)
%endmacro

; --------------------------------------------------------------------------
; In-place 8x8x16-bit accurate integer inverse DCT using AVX2 instructions
; %1-%4:  Input/output registers
; %5-%12: Temp registers
; %9:     Pass (1 or 2)

%macro DODCT 13
    ; -- Even part

    ; (Original)
    ; z1 = (z2 + z3) * 0.541196100;
    ; tmp2 = z1 + z3 * -1.847759065;
    ; tmp3 = z1 + z2 * 0.765366865;
    ;
    ; (This implementation)
    ; tmp2 = z2 * 0.541196100 + z3 * (0.541196100 - 1.847759065);
    ; tmp3 = z2 * (0.541196100 + 0.765366865) + z3 * 0.541196100;

    vperm2i128  %6, %3, %3, 0x01        ; %6=in6_2
    vpunpcklwd  %5, %3, %6              ; %5=in26_62L
    vpunpckhwd  %6, %3, %6              ; %6=in26_62H
    vpmaddwd    %5, %5, [GOTOFF(ebx,PW_F130_F054_MF130_F054)]  ; %5=tmp3_2L
    vpmaddwd    %6, %6, [GOTOFF(ebx,PW_F130_F054_MF130_F054)]  ; %6=tmp3_2H

    vperm2i128  %7, %1, %1, 0x01        ; %7=in4_0
    vpsignw     %1, %1, [GOTOFF(ebx,PW_1_NEG1)]
    vpaddw      %7, %7, %1              ; %7=(in0+in4)_(in0-in4)

    vpxor       %1, %1, %1
    vpunpcklwd  %8, %1, %7              ; %8=tmp0_1L
    vpunpckhwd  %1, %1, %7              ; %1=tmp0_1H
    vpsrad      %8, %8, (16-CONST_BITS)  ; vpsrad %8,16 & vpslld %8,CONST_BITS
    vpsrad      %1, %1, (16-CONST_BITS)  ; vpsrad %1,16 & vpslld %1,CONST_BITS

    vpsubd      %3, %8, %5
    vmovdqu     %11, %3                 ; %11=tmp0_1L-tmp3_2L=tmp13_12L
    vpaddd      %3, %8, %5
    vmovdqu     %9, %3                  ; %9=tmp0_1L+tmp3_2L=tmp10_11L
    vpsubd      %3, %1, %6
    vmovdqu     %12, %3                 ; %12=tmp0_1H-tmp3_2H=tmp13_12H
    vpaddd      %3, %1, %6
    vmovdqu     %10, %3                 ; %10=tmp0_1H+tmp3_2H=tmp10_11H

    ; -- Odd part

    vpaddw      %1, %4, %2              ; %1=in7_5+in3_1=z3_4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    vperm2i128  %8, %1, %1, 0x01        ; %8=z4_3
    vpunpcklwd  %7, %1, %8              ; %7=z34_43L
    vpunpckhwd  %8, %1, %8              ; %8=z34_43H
    vpmaddwd    %7, %7, [GOTOFF(ebx,PW_MF078_F117_F078_F117)]  ; %7=z3_4L
    vpmaddwd    %8, %8, [GOTOFF(ebx,PW_MF078_F117_F078_F117)]  ; %8=z3_4H

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

    vperm2i128  %2, %2, %2, 0x01        ; %2=in1_3
    vpunpcklwd  %3, %4, %2              ; %3=in71_53L
    vpunpckhwd  %4, %4, %2              ; %4=in71_53H

    vpmaddwd    %5, %3, [GOTOFF(ebx,PW_MF060_MF089_MF050_MF256)]  ; %5=tmp0_1L
    vpmaddwd    %6, %4, [GOTOFF(ebx,PW_MF060_MF089_MF050_MF256)]  ; %6=tmp0_1H
    vpaddd      %5, %5, %7              ; %5=tmp0_1L+z3_4L=tmp0_1L
    vpaddd      %6, %6, %8              ; %6=tmp0_1H+z3_4H=tmp0_1H

    vpmaddwd    %3, %3, [GOTOFF(ebx,PW_MF089_F060_MF256_F050)]  ; %3=tmp3_2L
    vpmaddwd    %4, %4, [GOTOFF(ebx,PW_MF089_F060_MF256_F050)]  ; %4=tmp3_2H
    vperm2i128  %7, %7, %7, 0x01        ; %7=z4_3L
    vperm2i128  %8, %8, %8, 0x01        ; %8=z4_3H
    vpaddd      %7, %3, %7              ; %7=tmp3_2L+z4_3L=tmp3_2L
    vpaddd      %8, %4, %8              ; %8=tmp3_2H+z4_3H=tmp3_2H

    ; -- Final output stage

    vmovdqu     %3, %9
    vmovdqu     %4, %10

    vpaddd      %1, %3, %7              ; %1=tmp10_11L+tmp3_2L=data0_1L
    vpaddd      %2, %4, %8              ; %2=tmp10_11H+tmp3_2H=data0_1H
    vpaddd      %1, %1, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpaddd      %2, %2, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpsrad      %1, %1, DESCALE_P %+ %13
    vpsrad      %2, %2, DESCALE_P %+ %13
    vpackssdw   %1, %1, %2              ; %1=data0_1

    vpsubd      %3, %3, %7              ; %3=tmp10_11L-tmp3_2L=data7_6L
    vpsubd      %4, %4, %8              ; %4=tmp10_11H-tmp3_2H=data7_6H
    vpaddd      %3, %3, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpaddd      %4, %4, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpsrad      %3, %3, DESCALE_P %+ %13
    vpsrad      %4, %4, DESCALE_P %+ %13
    vpackssdw   %4, %3, %4              ; %4=data7_6

    vmovdqu     %7, %11
    vmovdqu     %8, %12

    vpaddd      %2, %7, %5              ; %7=tmp13_12L+tmp0_1L=data3_2L
    vpaddd      %3, %8, %6              ; %8=tmp13_12H+tmp0_1H=data3_2H
    vpaddd      %2, %2, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpaddd      %3, %3, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpsrad      %2, %2, DESCALE_P %+ %13
    vpsrad      %3, %3, DESCALE_P %+ %13
    vpackssdw   %2, %2, %3              ; %2=data3_2

    vpsubd      %3, %7, %5              ; %7=tmp13_12L-tmp0_1L=data4_5L
    vpsubd      %6, %8, %6              ; %8=tmp13_12H-tmp0_1H=data4_5H
    vpaddd      %3, %3, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpaddd      %6, %6, [GOTOFF(ebx,PD_DESCALE_P %+ %13)]
    vpsrad      %3, %3, DESCALE_P %+ %13
    vpsrad      %6, %6, DESCALE_P %+ %13
    vpackssdw   %3, %3, %6              ; %3=data4_5
%endmacro

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_idct_islow_avx2)

EXTN(jconst_idct_islow_avx2):

PW_F130_F054_MF130_F054    times 4  dw  (F_0_541 + F_0_765),  F_0_541
                           times 4  dw  (F_0_541 - F_1_847),  F_0_541
PW_MF078_F117_F078_F117    times 4  dw  (F_1_175 - F_1_961),  F_1_175
                           times 4  dw  (F_1_175 - F_0_390),  F_1_175
PW_MF060_MF089_MF050_MF256 times 4  dw  (F_0_298 - F_0_899), -F_0_899
                           times 4  dw  (F_2_053 - F_2_562), -F_2_562
PW_MF089_F060_MF256_F050   times 4  dw -F_0_899, (F_1_501 - F_0_899)
                           times 4  dw -F_2_562, (F_3_072 - F_2_562)
PD_DESCALE_P1              times 8  dd  1 << (DESCALE_P1 - 1)
PD_DESCALE_P2              times 8  dd  1 << (DESCALE_P2 - 1)
PB_CENTERJSAMP             times 32 db  CENTERJSAMPLE
PW_1_NEG1                  times 8  dw  1
                           times 8  dw -1

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform dequantization and inverse DCT on one block of coefficients.
;
; GLOBAL(void)
; jsimd_idct_islow_avx2(void *dct_table, JCOEFPTR coef_block,
;                       JSAMPARRAY output_buf, JDIMENSION output_col)
;

%define dct_table(b)   (b) + 8          ; jpeg_component_info *compptr
%define coef_block(b)  (b) + 12         ; JCOEFPTR coef_block
%define output_buf(b)  (b) + 16         ; JSAMPARRAY output_buf
%define output_col(b)  (b) + 20         ; JDIMENSION output_col

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_YMMWORD
                                        ; ymmword wk[WK_NUM]
%define WK_NUM         4

    align       32
    GLOBAL_FUNCTION(jsimd_idct_islow_avx2)

EXTN(jsimd_idct_islow_avx2):
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

    ; ---- Pass 1: process columns.

;   mov         eax, [original_ebp]
    mov         edx, POINTER [dct_table(eax)]    ; quantptr
    mov         esi, JCOEFPTR [coef_block(eax)]  ; inptr

%ifndef NO_ZERO_COLUMN_TEST_ISLOW_AVX2
    mov         eax, dword [DWBLOCK(1,0,esi,SIZEOF_JCOEF)]
    or          eax, dword [DWBLOCK(2,0,esi,SIZEOF_JCOEF)]
    jnz         near .columnDCT

    movdqa      xmm0, XMMWORD [XMMBLOCK(1,0,esi,SIZEOF_JCOEF)]
    movdqa      xmm1, XMMWORD [XMMBLOCK(2,0,esi,SIZEOF_JCOEF)]
    vpor        xmm0, xmm0, XMMWORD [XMMBLOCK(3,0,esi,SIZEOF_JCOEF)]
    vpor        xmm1, xmm1, XMMWORD [XMMBLOCK(4,0,esi,SIZEOF_JCOEF)]
    vpor        xmm0, xmm0, XMMWORD [XMMBLOCK(5,0,esi,SIZEOF_JCOEF)]
    vpor        xmm1, xmm1, XMMWORD [XMMBLOCK(6,0,esi,SIZEOF_JCOEF)]
    vpor        xmm0, xmm0, XMMWORD [XMMBLOCK(7,0,esi,SIZEOF_JCOEF)]
    vpor        xmm1, xmm1, xmm0
    vpacksswb   xmm1, xmm1, xmm1
    vpacksswb   xmm1, xmm1, xmm1
    movd        eax, xmm1
    test        eax, eax
    jnz         short .columnDCT

    ; -- AC terms all zero

    movdqa      xmm5, XMMWORD [XMMBLOCK(0,0,esi,SIZEOF_JCOEF)]
    vpmullw     xmm5, xmm5, XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    vpsllw      xmm5, xmm5, PASS1_BITS

    vpunpcklwd  xmm4, xmm5, xmm5        ; xmm4=(00 00 01 01 02 02 03 03)
    vpunpckhwd  xmm5, xmm5, xmm5        ; xmm5=(04 04 05 05 06 06 07 07)
    vinserti128 ymm4, ymm4, xmm5, 1

    vpshufd     ymm0, ymm4, 0x00        ; ymm0=col0_4=(00 00 00 00 00 00 00 00  04 04 04 04 04 04 04 04)
    vpshufd     ymm1, ymm4, 0x55        ; ymm1=col1_5=(01 01 01 01 01 01 01 01  05 05 05 05 05 05 05 05)
    vpshufd     ymm2, ymm4, 0xAA        ; ymm2=col2_6=(02 02 02 02 02 02 02 02  06 06 06 06 06 06 06 06)
    vpshufd     ymm3, ymm4, 0xFF        ; ymm3=col3_7=(03 03 03 03 03 03 03 03  07 07 07 07 07 07 07 07)

    jmp         near .column_end
    ALIGNX      16, 7
%endif
.columnDCT:

    vmovdqu     ymm4, YMMWORD [YMMBLOCK(0,0,esi,SIZEOF_JCOEF)]  ; ymm4=in0_1
    vmovdqu     ymm5, YMMWORD [YMMBLOCK(2,0,esi,SIZEOF_JCOEF)]  ; ymm5=in2_3
    vmovdqu     ymm6, YMMWORD [YMMBLOCK(4,0,esi,SIZEOF_JCOEF)]  ; ymm6=in4_5
    vmovdqu     ymm7, YMMWORD [YMMBLOCK(6,0,esi,SIZEOF_JCOEF)]  ; ymm7=in6_7
    vpmullw     ymm4, ymm4, YMMWORD [YMMBLOCK(0,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    vpmullw     ymm5, ymm5, YMMWORD [YMMBLOCK(2,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    vpmullw     ymm6, ymm6, YMMWORD [YMMBLOCK(4,0,edx,SIZEOF_ISLOW_MULT_TYPE)]
    vpmullw     ymm7, ymm7, YMMWORD [YMMBLOCK(6,0,edx,SIZEOF_ISLOW_MULT_TYPE)]

    vperm2i128  ymm0, ymm4, ymm6, 0x20  ; ymm0=in0_4
    vperm2i128  ymm1, ymm5, ymm4, 0x31  ; ymm1=in3_1
    vperm2i128  ymm2, ymm5, ymm7, 0x20  ; ymm2=in2_6
    vperm2i128  ymm3, ymm7, ymm6, 0x31  ; ymm3=in7_5

    DODCT ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, XMMWORD [wk(0)], XMMWORD [wk(1)], XMMWORD [wk(2)], XMMWORD [wk(3)], 1
    ; ymm0=data0_1, ymm1=data3_2, ymm2=data4_5, ymm3=data7_6

    DOTRANSPOSE ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7
    ; ymm0=data0_4, ymm1=data1_5, ymm2=data2_6, ymm3=data3_7

.column_end:

    ; -- Prefetch the next coefficient block

    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 0*32]
    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 1*32]
    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 2*32]
    prefetchnta [esi + DCTSIZE2*SIZEOF_JCOEF + 3*32]

    ; ---- Pass 2: process rows.

    mov         eax, [original_ebp]
    mov         edi, JSAMPARRAY [output_buf(eax)]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [output_col(eax)]

    vperm2i128  ymm4, ymm3, ymm1, 0x31  ; ymm3=in7_5
    vperm2i128  ymm1, ymm3, ymm1, 0x20  ; ymm1=in3_1

    DODCT ymm0, ymm1, ymm2, ymm4, ymm3, ymm5, ymm6, ymm7, XMMWORD [wk(0)], XMMWORD [wk(1)], XMMWORD [wk(2)], XMMWORD [wk(3)], 2
    ; ymm0=data0_1, ymm1=data3_2, ymm2=data4_5, ymm4=data7_6

    DOTRANSPOSE ymm0, ymm1, ymm2, ymm4, ymm3, ymm5, ymm6, ymm7
    ; ymm0=data0_4, ymm1=data1_5, ymm2=data2_6, ymm4=data3_7

    vpacksswb   ymm0, ymm0, ymm1        ; ymm0=data01_45
    vpacksswb   ymm1, ymm2, ymm4        ; ymm1=data23_67
    vpaddb      ymm0, ymm0, [GOTOFF(ebx,PB_CENTERJSAMP)]
    vpaddb      ymm1, ymm1, [GOTOFF(ebx,PB_CENTERJSAMP)]

    vextracti128 xmm6, ymm1, 1          ; xmm3=data67
    vextracti128 xmm4, ymm0, 1          ; xmm2=data45
    vextracti128 xmm2, ymm1, 0          ; xmm1=data23
    vextracti128 xmm0, ymm0, 0          ; xmm0=data01

    vpshufd     xmm1, xmm0, 0x4E  ; xmm1=(10 11 12 13 14 15 16 17 00 01 02 03 04 05 06 07)
    vpshufd     xmm3, xmm2, 0x4E  ; xmm3=(30 31 32 33 34 35 36 37 20 21 22 23 24 25 26 27)
    vpshufd     xmm5, xmm4, 0x4E  ; xmm5=(50 51 52 53 54 55 56 57 40 41 42 43 44 45 46 47)
    vpshufd     xmm7, xmm6, 0x4E  ; xmm7=(70 71 72 73 74 75 76 77 60 61 62 63 64 65 66 67)

    vzeroupper

    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         esi, JSAMPROW [edi+1*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm0
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm1

    mov         edx, JSAMPROW [edi+2*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         esi, JSAMPROW [edi+3*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm2
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm3

    mov         edx, JSAMPROW [edi+4*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         esi, JSAMPROW [edi+5*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm4
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm5

    mov         edx, JSAMPROW [edi+6*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         esi, JSAMPROW [edi+7*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE], xmm6
    movq        XMM_MMWORD [esi+eax*SIZEOF_JSAMPLE], xmm7

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
