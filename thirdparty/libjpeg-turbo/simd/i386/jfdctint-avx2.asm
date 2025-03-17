;
; jfdctint.asm - accurate integer FDCT (AVX2)
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
; forward DCT (Discrete Cosine Transform). The following code is based
; directly on the IJG's original jfdctint.c; see the jfdctint.c for
; more details.

%include "jsimdext.inc"
%include "jdct.inc"

; --------------------------------------------------------------------------

%define CONST_BITS  13
%define PASS1_BITS  2

%define DESCALE_P1  (CONST_BITS - PASS1_BITS)
%define DESCALE_P2  (CONST_BITS + PASS1_BITS)

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
; In-place 8x8x16-bit matrix transpose using AVX2 instructions
; %1-%4: Input/output registers
; %5-%8: Temp registers

%macro DOTRANSPOSE 8
    ; %1=(00 01 02 03 04 05 06 07  40 41 42 43 44 45 46 47)
    ; %2=(10 11 12 13 14 15 16 17  50 51 52 53 54 55 56 57)
    ; %3=(20 21 22 23 24 25 26 27  60 61 62 63 64 65 66 67)
    ; %4=(30 31 32 33 34 35 36 37  70 71 72 73 74 75 76 77)

    vpunpcklwd  %5, %1, %2
    vpunpckhwd  %6, %1, %2
    vpunpcklwd  %7, %3, %4
    vpunpckhwd  %8, %3, %4
    ; transpose coefficients(phase 1)
    ; %5=(00 10 01 11 02 12 03 13  40 50 41 51 42 52 43 53)
    ; %6=(04 14 05 15 06 16 07 17  44 54 45 55 46 56 47 57)
    ; %7=(20 30 21 31 22 32 23 33  60 70 61 71 62 72 63 73)
    ; %8=(24 34 25 35 26 36 27 37  64 74 65 75 66 76 67 77)

    vpunpckldq  %1, %5, %7
    vpunpckhdq  %2, %5, %7
    vpunpckldq  %3, %6, %8
    vpunpckhdq  %4, %6, %8
    ; transpose coefficients(phase 2)
    ; %1=(00 10 20 30 01 11 21 31  40 50 60 70 41 51 61 71)
    ; %2=(02 12 22 32 03 13 23 33  42 52 62 72 43 53 63 73)
    ; %3=(04 14 24 34 05 15 25 35  44 54 64 74 45 55 65 75)
    ; %4=(06 16 26 36 07 17 27 37  46 56 66 76 47 57 67 77)

    vpermq      %1, %1, 0x8D
    vpermq      %2, %2, 0x8D
    vpermq      %3, %3, 0xD8
    vpermq      %4, %4, 0xD8
    ; transpose coefficients(phase 3)
    ; %1=(01 11 21 31 41 51 61 71  00 10 20 30 40 50 60 70)
    ; %2=(03 13 23 33 43 53 63 73  02 12 22 32 42 52 62 72)
    ; %3=(04 14 24 34 44 54 64 74  05 15 25 35 45 55 65 75)
    ; %4=(06 16 26 36 46 56 66 76  07 17 27 37 47 57 67 77)
%endmacro

; --------------------------------------------------------------------------
; In-place 8x8x16-bit accurate integer forward DCT using AVX2 instructions
; %1-%4: Input/output registers
; %5-%8: Temp registers
; %9:    Pass (1 or 2)

%macro DODCT 9
    vpsubw      %5, %1, %4              ; %5=data1_0-data6_7=tmp6_7
    vpaddw      %6, %1, %4              ; %6=data1_0+data6_7=tmp1_0
    vpaddw      %7, %2, %3              ; %7=data3_2+data4_5=tmp3_2
    vpsubw      %8, %2, %3              ; %8=data3_2-data4_5=tmp4_5

    ; -- Even part

    vperm2i128  %6, %6, %6, 0x01        ; %6=tmp0_1
    vpaddw      %1, %6, %7              ; %1=tmp0_1+tmp3_2=tmp10_11
    vpsubw      %6, %6, %7              ; %6=tmp0_1-tmp3_2=tmp13_12

    vperm2i128  %7, %1, %1, 0x01        ; %7=tmp11_10
    vpsignw     %1, %1, [GOTOFF(ebx, PW_1_NEG1)]  ; %1=tmp10_neg11
    vpaddw      %7, %7, %1              ; %7=(tmp10+tmp11)_(tmp10-tmp11)
%if %9 == 1
    vpsllw      %1, %7, PASS1_BITS      ; %1=data0_4
%else
    vpaddw      %7, %7, [GOTOFF(ebx, PW_DESCALE_P2X)]
    vpsraw      %1, %7, PASS1_BITS      ; %1=data0_4
%endif

    ; (Original)
    ; z1 = (tmp12 + tmp13) * 0.541196100;
    ; data2 = z1 + tmp13 * 0.765366865;
    ; data6 = z1 + tmp12 * -1.847759065;
    ;
    ; (This implementation)
    ; data2 = tmp13 * (0.541196100 + 0.765366865) + tmp12 * 0.541196100;
    ; data6 = tmp13 * 0.541196100 + tmp12 * (0.541196100 - 1.847759065);

    vperm2i128  %7, %6, %6, 0x01        ; %7=tmp12_13
    vpunpcklwd  %2, %6, %7
    vpunpckhwd  %6, %6, %7
    vpmaddwd    %2, %2, [GOTOFF(ebx, PW_F130_F054_MF130_F054)]  ; %2=data2_6L
    vpmaddwd    %6, %6, [GOTOFF(ebx, PW_F130_F054_MF130_F054)]  ; %6=data2_6H

    vpaddd      %2, %2, [GOTOFF(ebx, PD_DESCALE_P %+ %9)]
    vpaddd      %6, %6, [GOTOFF(ebx, PD_DESCALE_P %+ %9)]
    vpsrad      %2, %2, DESCALE_P %+ %9
    vpsrad      %6, %6, DESCALE_P %+ %9

    vpackssdw   %3, %2, %6              ; %6=data2_6

    ; -- Odd part

    vpaddw      %7, %8, %5              ; %7=tmp4_5+tmp6_7=z3_4

    ; (Original)
    ; z5 = (z3 + z4) * 1.175875602;
    ; z3 = z3 * -1.961570560;  z4 = z4 * -0.390180644;
    ; z3 += z5;  z4 += z5;
    ;
    ; (This implementation)
    ; z3 = z3 * (1.175875602 - 1.961570560) + z4 * 1.175875602;
    ; z4 = z3 * 1.175875602 + z4 * (1.175875602 - 0.390180644);

    vperm2i128  %2, %7, %7, 0x01        ; %2=z4_3
    vpunpcklwd  %6, %7, %2
    vpunpckhwd  %7, %7, %2
    vpmaddwd    %6, %6, [GOTOFF(ebx, PW_MF078_F117_F078_F117)]  ; %6=z3_4L
    vpmaddwd    %7, %7, [GOTOFF(ebx, PW_MF078_F117_F078_F117)]  ; %7=z3_4H

    ; (Original)
    ; z1 = tmp4 + tmp7;  z2 = tmp5 + tmp6;
    ; tmp4 = tmp4 * 0.298631336;  tmp5 = tmp5 * 2.053119869;
    ; tmp6 = tmp6 * 3.072711026;  tmp7 = tmp7 * 1.501321110;
    ; z1 = z1 * -0.899976223;  z2 = z2 * -2.562915447;
    ; data7 = tmp4 + z1 + z3;  data5 = tmp5 + z2 + z4;
    ; data3 = tmp6 + z2 + z3;  data1 = tmp7 + z1 + z4;
    ;
    ; (This implementation)
    ; tmp4 = tmp4 * (0.298631336 - 0.899976223) + tmp7 * -0.899976223;
    ; tmp5 = tmp5 * (2.053119869 - 2.562915447) + tmp6 * -2.562915447;
    ; tmp6 = tmp5 * -2.562915447 + tmp6 * (3.072711026 - 2.562915447);
    ; tmp7 = tmp4 * -0.899976223 + tmp7 * (1.501321110 - 0.899976223);
    ; data7 = tmp4 + z3;  data5 = tmp5 + z4;
    ; data3 = tmp6 + z3;  data1 = tmp7 + z4;

    vperm2i128  %4, %5, %5, 0x01        ; %4=tmp7_6
    vpunpcklwd  %2, %8, %4
    vpunpckhwd  %4, %8, %4
    vpmaddwd    %2, %2, [GOTOFF(ebx, PW_MF060_MF089_MF050_MF256)]  ; %2=tmp4_5L
    vpmaddwd    %4, %4, [GOTOFF(ebx, PW_MF060_MF089_MF050_MF256)]  ; %4=tmp4_5H

    vpaddd      %2, %2, %6              ; %2=data7_5L
    vpaddd      %4, %4, %7              ; %4=data7_5H

    vpaddd      %2, %2, [GOTOFF(ebx, PD_DESCALE_P %+ %9)]
    vpaddd      %4, %4, [GOTOFF(ebx, PD_DESCALE_P %+ %9)]
    vpsrad      %2, %2, DESCALE_P %+ %9
    vpsrad      %4, %4, DESCALE_P %+ %9

    vpackssdw   %4, %2, %4              ; %4=data7_5

    vperm2i128  %2, %8, %8, 0x01        ; %2=tmp5_4
    vpunpcklwd  %8, %5, %2
    vpunpckhwd  %5, %5, %2
    vpmaddwd    %8, %8, [GOTOFF(ebx, PW_F050_MF256_F060_MF089)]  ; %8=tmp6_7L
    vpmaddwd    %5, %5, [GOTOFF(ebx, PW_F050_MF256_F060_MF089)]  ; %5=tmp6_7H

    vpaddd      %8, %8, %6              ; %8=data3_1L
    vpaddd      %5, %5, %7              ; %5=data3_1H

    vpaddd      %8, %8, [GOTOFF(ebx, PD_DESCALE_P %+ %9)]
    vpaddd      %5, %5, [GOTOFF(ebx, PD_DESCALE_P %+ %9)]
    vpsrad      %8, %8, DESCALE_P %+ %9
    vpsrad      %5, %5, DESCALE_P %+ %9

    vpackssdw   %2, %8, %5              ; %2=data3_1
%endmacro

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_fdct_islow_avx2)

EXTN(jconst_fdct_islow_avx2):

PW_F130_F054_MF130_F054    times 4  dw  (F_0_541 + F_0_765),  F_0_541
                           times 4  dw  (F_0_541 - F_1_847),  F_0_541
PW_MF078_F117_F078_F117    times 4  dw  (F_1_175 - F_1_961),  F_1_175
                           times 4  dw  (F_1_175 - F_0_390),  F_1_175
PW_MF060_MF089_MF050_MF256 times 4  dw  (F_0_298 - F_0_899), -F_0_899
                           times 4  dw  (F_2_053 - F_2_562), -F_2_562
PW_F050_MF256_F060_MF089   times 4  dw  (F_3_072 - F_2_562), -F_2_562
                           times 4  dw  (F_1_501 - F_0_899), -F_0_899
PD_DESCALE_P1              times 8  dd  1 << (DESCALE_P1 - 1)
PD_DESCALE_P2              times 8  dd  1 << (DESCALE_P2 - 1)
PW_DESCALE_P2X             times 16 dw  1 << (PASS1_BITS - 1)
PW_1_NEG1                  times 8  dw  1
                           times 8  dw -1

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Perform the forward DCT on one block of samples.
;
; GLOBAL(void)
; jsimd_fdct_islow_avx2(DCTELEM *data)
;

%define data(b)       (b) + 8           ; DCTELEM *data

    align       32
    GLOBAL_FUNCTION(jsimd_fdct_islow_avx2)

EXTN(jsimd_fdct_islow_avx2):
    push        ebp
    mov         ebp, esp
    PUSHPIC     ebx
;   push        ecx                     ; unused
;   push        edx                     ; need not be preserved
;   push        esi                     ; unused
;   push        edi                     ; unused

    GET_GOT     ebx                     ; get GOT address

    ; ---- Pass 1: process rows.

    mov         edx, POINTER [data(ebp)]  ; (DCTELEM *)

    vmovdqu     ymm4, YMMWORD [YMMBLOCK(0,0,edx,SIZEOF_DCTELEM)]
    vmovdqu     ymm5, YMMWORD [YMMBLOCK(2,0,edx,SIZEOF_DCTELEM)]
    vmovdqu     ymm6, YMMWORD [YMMBLOCK(4,0,edx,SIZEOF_DCTELEM)]
    vmovdqu     ymm7, YMMWORD [YMMBLOCK(6,0,edx,SIZEOF_DCTELEM)]
    ; ymm4=(00 01 02 03 04 05 06 07  10 11 12 13 14 15 16 17)
    ; ymm5=(20 21 22 23 24 25 26 27  30 31 32 33 34 35 36 37)
    ; ymm6=(40 41 42 43 44 45 46 47  50 51 52 53 54 55 56 57)
    ; ymm7=(60 61 62 63 64 65 66 67  70 71 72 73 74 75 76 77)

    vperm2i128  ymm0, ymm4, ymm6, 0x20
    vperm2i128  ymm1, ymm4, ymm6, 0x31
    vperm2i128  ymm2, ymm5, ymm7, 0x20
    vperm2i128  ymm3, ymm5, ymm7, 0x31
    ; ymm0=(00 01 02 03 04 05 06 07  40 41 42 43 44 45 46 47)
    ; ymm1=(10 11 12 13 14 15 16 17  50 51 52 53 54 55 56 57)
    ; ymm2=(20 21 22 23 24 25 26 27  60 61 62 63 64 65 66 67)
    ; ymm3=(30 31 32 33 34 35 36 37  70 71 72 73 74 75 76 77)

    DOTRANSPOSE ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7

    DODCT       ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, 1
    ; ymm0=data0_4, ymm1=data3_1, ymm2=data2_6, ymm3=data7_5

    ; ---- Pass 2: process columns.

    vperm2i128  ymm4, ymm1, ymm3, 0x20  ; ymm4=data3_7
    vperm2i128  ymm1, ymm1, ymm3, 0x31  ; ymm1=data1_5

    DOTRANSPOSE ymm0, ymm1, ymm2, ymm4, ymm3, ymm5, ymm6, ymm7

    DODCT       ymm0, ymm1, ymm2, ymm4, ymm3, ymm5, ymm6, ymm7, 2
    ; ymm0=data0_4, ymm1=data3_1, ymm2=data2_6, ymm4=data7_5

    vperm2i128 ymm3, ymm0, ymm1, 0x30   ; ymm3=data0_1
    vperm2i128 ymm5, ymm2, ymm1, 0x20   ; ymm5=data2_3
    vperm2i128 ymm6, ymm0, ymm4, 0x31   ; ymm6=data4_5
    vperm2i128 ymm7, ymm2, ymm4, 0x21   ; ymm7=data6_7

    vmovdqu     YMMWORD [YMMBLOCK(0,0,edx,SIZEOF_DCTELEM)], ymm3
    vmovdqu     YMMWORD [YMMBLOCK(2,0,edx,SIZEOF_DCTELEM)], ymm5
    vmovdqu     YMMWORD [YMMBLOCK(4,0,edx,SIZEOF_DCTELEM)], ymm6
    vmovdqu     YMMWORD [YMMBLOCK(6,0,edx,SIZEOF_DCTELEM)], ymm7

    vzeroupper
;   pop         edi                     ; unused
;   pop         esi                     ; unused
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; unused
    POPPIC      ebx
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
