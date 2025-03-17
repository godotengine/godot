;
; jdmrgext.asm - merged upsampling/color conversion (AVX2)
;
; Copyright 2009, 2012 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2012, 2016, 2024, D. R. Commander.
; Copyright (C) 2015, Intel Corporation.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jcolsamp.inc"

; --------------------------------------------------------------------------
;
; Upsample and color convert for the case of 2:1 horizontal and 1:1 vertical.
;
; GLOBAL(void)
; jsimd_h2v1_merged_upsample_avx2(JDIMENSION output_width,
;                                 JSAMPIMAGE input_buf,
;                                 JDIMENSION in_row_group_ctr,
;                                 JSAMPARRAY output_buf);
;

%define output_width(b)      (b) + 8    ; JDIMENSION output_width
%define input_buf(b)         (b) + 12   ; JSAMPIMAGE input_buf
%define in_row_group_ctr(b)  (b) + 16   ; JDIMENSION in_row_group_ctr
%define output_buf(b)        (b) + 20   ; JSAMPARRAY output_buf

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_YMMWORD
                                        ; ymmword wk[WK_NUM]
%define WK_NUM        3
%define gotptr        wk(0) - SIZEOF_POINTER  ; void * gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_merged_upsample_avx2)

EXTN(jsimd_h2v1_merged_upsample_avx2):
    push        ebp
    mov         eax, esp                     ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_YMMWORD)  ; align to 256 bits
    mov         [esp], eax
    mov         ebp, esp                     ; ebp = aligned ebp
    lea         esp, [wk(0)]
    PUSHPIC     eax                     ; make a room for GOT address
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    GET_GOT     ebx                     ; get GOT address
    MOVPIC      POINTER [gotptr], ebx   ; save GOT address

    mov         ecx, JDIMENSION [output_width(eax)]  ; col
    test        ecx, ecx
    jz          near .return

    push        ecx

    mov         edi, JSAMPIMAGE [input_buf(eax)]
    mov         ecx, JDIMENSION [in_row_group_ctr(eax)]
    mov         esi, JSAMPARRAY [edi+0*SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [edi+1*SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [edi+2*SIZEOF_JSAMPARRAY]
    mov         edi, JSAMPARRAY [output_buf(eax)]
    mov         esi, JSAMPROW [esi+ecx*SIZEOF_JSAMPROW]  ; inptr0
    mov         ebx, JSAMPROW [ebx+ecx*SIZEOF_JSAMPROW]  ; inptr1
    mov         edx, JSAMPROW [edx+ecx*SIZEOF_JSAMPROW]  ; inptr2
    mov         edi, JSAMPROW [edi]                      ; outptr

    pop         ecx                     ; col

    ALIGNX      16, 7
.columnloop:
    MOVPIC      eax, POINTER [gotptr]   ; load GOT address (eax)

    vmovdqu     ymm6, YMMWORD [ebx]     ; ymm6=Cb(0123456789ABCDEFGHIJKLMNOPQRSTUV)
    vmovdqu     ymm7, YMMWORD [edx]     ; ymm7=Cr(0123456789ABCDEFGHIJKLMNOPQRSTUV)

    vpxor       ymm1, ymm1, ymm1        ; ymm1=(all 0's)
    vpcmpeqw    ymm3, ymm3, ymm3
    vpsllw      ymm3, ymm3, 7           ; ymm3={0xFF80 0xFF80 0xFF80 0xFF80 ..}

    vpermq      ymm6, ymm6, 0xd8        ; ymm6=Cb(01234567GHIJKLMN89ABCDEFOPQRSTUV)
    vpermq      ymm7, ymm7, 0xd8        ; ymm7=Cr(01234567GHIJKLMN89ABCDEFOPQRSTUV)
    vpunpcklbw  ymm4, ymm6, ymm1        ; ymm4=Cb(0123456789ABCDEF)=CbL
    vpunpckhbw  ymm6, ymm6, ymm1        ; ymm6=Cb(GHIJKLMNOPQRSTUV)=CbH
    vpunpcklbw  ymm0, ymm7, ymm1        ; ymm0=Cr(0123456789ABCDEF)=CrL
    vpunpckhbw  ymm7, ymm7, ymm1        ; ymm7=Cr(GHIJKLMNOPQRSTUV)=CrH

    vpaddw      ymm5, ymm6, ymm3
    vpaddw      ymm2, ymm4, ymm3
    vpaddw      ymm1, ymm7, ymm3
    vpaddw      ymm3, ymm0, ymm3

    ; (Original)
    ; R = Y                + 1.40200 * Cr
    ; G = Y - 0.34414 * Cb - 0.71414 * Cr
    ; B = Y + 1.77200 * Cb
    ;
    ; (This implementation)
    ; R = Y                + 0.40200 * Cr + Cr
    ; G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
    ; B = Y - 0.22800 * Cb + Cb + Cb

    vpaddw      ymm6, ymm5, ymm5             ; ymm6=2*CbH
    vpaddw      ymm4, ymm2, ymm2             ; ymm4=2*CbL
    vpaddw      ymm7, ymm1, ymm1             ; ymm7=2*CrH
    vpaddw      ymm0, ymm3, ymm3             ; ymm0=2*CrL

    vpmulhw     ymm6, ymm6, [GOTOFF(eax,PW_MF0228)]  ; ymm6=(2*CbH * -FIX(0.22800))
    vpmulhw     ymm4, ymm4, [GOTOFF(eax,PW_MF0228)]  ; ymm4=(2*CbL * -FIX(0.22800))
    vpmulhw     ymm7, ymm7, [GOTOFF(eax,PW_F0402)]   ; ymm7=(2*CrH * FIX(0.40200))
    vpmulhw     ymm0, ymm0, [GOTOFF(eax,PW_F0402)]   ; ymm0=(2*CrL * FIX(0.40200))

    vpaddw      ymm6, ymm6, [GOTOFF(eax,PW_ONE)]
    vpaddw      ymm4, ymm4, [GOTOFF(eax,PW_ONE)]
    vpsraw      ymm6, ymm6, 1                     ; ymm6=(CbH * -FIX(0.22800))
    vpsraw      ymm4, ymm4, 1                     ; ymm4=(CbL * -FIX(0.22800))
    vpaddw      ymm7, ymm7, [GOTOFF(eax,PW_ONE)]
    vpaddw      ymm0, ymm0, [GOTOFF(eax,PW_ONE)]
    vpsraw      ymm7, ymm7, 1                     ; ymm7=(CrH * FIX(0.40200))
    vpsraw      ymm0, ymm0, 1                     ; ymm0=(CrL * FIX(0.40200))

    vpaddw      ymm6, ymm6, ymm5
    vpaddw      ymm4, ymm4, ymm2
    vpaddw      ymm6, ymm6, ymm5                  ; ymm6=(CbH * FIX(1.77200))=(B-Y)H
    vpaddw      ymm4, ymm4, ymm2                  ; ymm4=(CbL * FIX(1.77200))=(B-Y)L
    vpaddw      ymm7, ymm7, ymm1                  ; ymm7=(CrH * FIX(1.40200))=(R-Y)H
    vpaddw      ymm0, ymm0, ymm3                  ; ymm0=(CrL * FIX(1.40200))=(R-Y)L

    vmovdqa     YMMWORD [wk(0)], ymm6             ; wk(0)=(B-Y)H
    vmovdqa     YMMWORD [wk(1)], ymm7             ; wk(1)=(R-Y)H

    vpunpckhwd  ymm6, ymm5, ymm1
    vpunpcklwd  ymm5, ymm5, ymm1
    vpmaddwd    ymm5, ymm5, [GOTOFF(eax,PW_MF0344_F0285)]
    vpmaddwd    ymm6, ymm6, [GOTOFF(eax,PW_MF0344_F0285)]
    vpunpckhwd  ymm7, ymm2, ymm3
    vpunpcklwd  ymm2, ymm2, ymm3
    vpmaddwd    ymm2, ymm2, [GOTOFF(eax,PW_MF0344_F0285)]
    vpmaddwd    ymm7, ymm7, [GOTOFF(eax,PW_MF0344_F0285)]

    vpaddd      ymm5, ymm5, [GOTOFF(eax,PD_ONEHALF)]
    vpaddd      ymm6, ymm6, [GOTOFF(eax,PD_ONEHALF)]
    vpsrad      ymm5, ymm5, SCALEBITS
    vpsrad      ymm6, ymm6, SCALEBITS
    vpaddd      ymm2, ymm2, [GOTOFF(eax,PD_ONEHALF)]
    vpaddd      ymm7, ymm7, [GOTOFF(eax,PD_ONEHALF)]
    vpsrad      ymm2, ymm2, SCALEBITS
    vpsrad      ymm7, ymm7, SCALEBITS

    vpackssdw   ymm5, ymm5, ymm6        ; ymm5=CbH*-FIX(0.344)+CrH*FIX(0.285)
    vpackssdw   ymm2, ymm2, ymm7        ; ymm2=CbL*-FIX(0.344)+CrL*FIX(0.285)
    vpsubw      ymm5, ymm5, ymm1        ; ymm5=CbH*-FIX(0.344)+CrH*-FIX(0.714)=(G-Y)H
    vpsubw      ymm2, ymm2, ymm3        ; ymm2=CbL*-FIX(0.344)+CrL*-FIX(0.714)=(G-Y)L

    vmovdqa     YMMWORD [wk(2)], ymm5   ; wk(2)=(G-Y)H

    mov         al, 2                   ; Yctr
    jmp         short .Yloop_1st
    ALIGNX      16, 7

.Yloop_2nd:
    vmovdqa     ymm0, YMMWORD [wk(1)]   ; ymm0=(R-Y)H
    vmovdqa     ymm2, YMMWORD [wk(2)]   ; ymm2=(G-Y)H
    vmovdqa     ymm4, YMMWORD [wk(0)]   ; ymm4=(B-Y)H
    ALIGNX      16, 7

.Yloop_1st:
    vmovdqu     ymm7, YMMWORD [esi]     ; ymm7=Y(0123456789ABCDEFGHIJKLMNOPQRSTUV)

    vpcmpeqw    ymm6, ymm6, ymm6
    vpsrlw      ymm6, ymm6, BYTE_BIT    ; ymm6={0xFF 0x00 0xFF 0x00 ..}
    vpand       ymm6, ymm6, ymm7        ; ymm6=Y(02468ACEGIKMOQSU)=YE
    vpsrlw      ymm7, ymm7, BYTE_BIT    ; ymm7=Y(13579BDFHJLNPRTV)=YO

    vmovdqa     ymm1, ymm0              ; ymm1=ymm0=(R-Y)(L/H)
    vmovdqa     ymm3, ymm2              ; ymm3=ymm2=(G-Y)(L/H)
    vmovdqa     ymm5, ymm4              ; ymm5=ymm4=(B-Y)(L/H)

    vpaddw      ymm0, ymm0, ymm6        ; ymm0=((R-Y)+YE)=RE=R(02468ACEGIKMOQSU)
    vpaddw      ymm1, ymm1, ymm7        ; ymm1=((R-Y)+YO)=RO=R(13579BDFHJLNPRTV)
    vpackuswb   ymm0, ymm0, ymm0        ; ymm0=R(02468ACE********GIKMOQSU********)
    vpackuswb   ymm1, ymm1, ymm1        ; ymm1=R(13579BDF********HJLNPRTV********)

    vpaddw      ymm2, ymm2, ymm6        ; ymm2=((G-Y)+YE)=GE=G(02468ACEGIKMOQSU)
    vpaddw      ymm3, ymm3, ymm7        ; ymm3=((G-Y)+YO)=GO=G(13579BDFHJLNPRTV)
    vpackuswb   ymm2, ymm2, ymm2        ; ymm2=G(02468ACE********GIKMOQSU********)
    vpackuswb   ymm3, ymm3, ymm3        ; ymm3=G(13579BDF********HJLNPRTV********)

    vpaddw      ymm4, ymm4, ymm6        ; ymm4=((B-Y)+YE)=BE=B(02468ACEGIKMOQSU)
    vpaddw      ymm5, ymm5, ymm7        ; ymm5=((B-Y)+YO)=BO=B(13579BDFHJLNPRTV)
    vpackuswb   ymm4, ymm4, ymm4        ; ymm4=B(02468ACE********GIKMOQSU********)
    vpackuswb   ymm5, ymm5, ymm5        ; ymm5=B(13579BDF********HJLNPRTV********)

%if RGB_PIXELSIZE == 3  ; ---------------

    ; ymmA=(00 02 04 06 08 0A 0C 0E ** 0G 0I 0K 0M 0O 0Q 0S 0U **)
    ; ymmB=(01 03 05 07 09 0B 0D 0F ** 0H 0J 0L 0N 0P 0R 0T 0V **)
    ; ymmC=(10 12 14 16 18 1A 1C 1E ** 1G 1I 1K 1M 1O 1Q 1S 1U **)
    ; ymmD=(11 13 15 17 19 1B 1D 1F ** 1H 1J 1L 1N 1P 1R 1T 1V **)
    ; ymmE=(20 22 24 26 28 2A 2C 2E ** 2G 2I 2K 2M 2O 2Q 2S 2U **)
    ; ymmF=(21 23 25 27 29 2B 2D 2F ** 2H 2J 2L 2N 2P 2R 2T 2V **)
    ; ymmG=(** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **)
    ; ymmH=(** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **)

    vpunpcklbw  ymmA, ymmA, ymmC        ; ymmA=(00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E
                                        ;       0G 1G 0I 1I 0K 1K 0M 1M 0O 1O 0Q 1Q 0S 1S 0U 1U)
    vpunpcklbw  ymmE, ymmE, ymmB        ; ymmE=(20 01 22 03 24 05 26 07 28 09 2A 0B 2C 0D 2E 0F
                                        ;       2G 0H 2I 0J 2K 0L 2M 0N 2O 0P 2Q 0R 2S 0T 2U 0V)
    vpunpcklbw  ymmD, ymmD, ymmF        ; ymmD=(11 21 13 23 15 25 17 27 19 29 1B 2B 1D 2D 1F 2F
                                        ;       1H 2H 1J 2J 1L 2L 1N 2N 1P 2P 1R 2R 1T 2T 1V 2V)

    vpsrldq     ymmH, ymmA, 2           ; ymmH=(02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E 0G 1G
                                        ;       0I 1I 0K 1K 0M 1M 0O 1O 0Q 1Q 0S 1S 0U 1U -- --)
    vpunpckhwd  ymmG, ymmA, ymmE        ; ymmG=(08 18 28 09 0A 1A 2A 0B 0C 1C 2C 0D 0E 1E 2E 0F
                                        ;       0O 1O 2O 0P 0Q 1Q 2Q 0R 0S 1S 2S 0T 0U 1U 2U 0V)
    vpunpcklwd  ymmA, ymmA, ymmE        ; ymmA=(00 10 20 01 02 12 22 03 04 14 24 05 06 16 26 07
                                        ;       0G 1G 2G 0H 0I 1I 2I 0J 0K 1K 2K 0L 0M 1M 2M 0N)

    vpsrldq     ymmE, ymmE, 2           ; ymmE=(22 03 24 05 26 07 28 09 2A 0B 2C 0D 2E 0F 2G 0H
                                        ;       2I 0J 2K 0L 2M 0N 2O 0P 2Q 0R 2S 0T 2U 0V -- --)

    vpsrldq     ymmB, ymmD, 2           ; ymmB=(13 23 15 25 17 27 19 29 1B 2B 1D 2D 1F 2F 1H 2H
                                        ;       1J 2J 1L 2L 1N 2N 1P 2P 1R 2R 1T 2T 1V 2V -- --)
    vpunpckhwd  ymmC, ymmD, ymmH        ; ymmC=(19 29 0A 1A 1B 2B 0C 1C 1D 2D 0E 1E 1F 2F 0G 1G
                                        ;       1P 2P 0Q 1Q 1R 2R 0S 1S 1T 2T 0U 1U 1V 2V -- --)
    vpunpcklwd  ymmD, ymmD, ymmH        ; ymmD=(11 21 02 12 13 23 04 14 15 25 06 16 17 27 08 18
                                        ;       1H 2H 0I 1I 1J 2J 0K 1K 1L 2L 0M 1M 1N 2N 0O 1O)

    vpunpckhwd  ymmF, ymmE, ymmB        ; ymmF=(2A 0B 1B 2B 2C 0D 1D 2D 2E 0F 1F 2F 2G 0H 1H 2H
                                        ;       2Q 0R 1R 2R 2S 0T 1T 2T 2U 0V 1V 2V -- -- -- --)
    vpunpcklwd  ymmE, ymmE, ymmB        ; ymmE=(22 03 13 23 24 05 15 25 26 07 17 27 28 09 19 29
                                        ;       2I 0J 1J 2J 2K 0L 1L 2L 2M 0N 1N 2N 2O 0P 1P 2P)

    vpshufd     ymmH, ymmA, 0x4E        ; ymmH=(04 14 24 05 06 16 26 07 00 10 20 01 02 12 22 03
                                        ;       0K 1K 2K 0L 0M 1M 2M 0N 0G 1G 2G 0H 0I 1I 2I 0J)
    vpunpckldq  ymmA, ymmA, ymmD        ; ymmA=(00 10 20 01 11 21 02 12 02 12 22 03 13 23 04 14
                                        ;       0G 1G 2G 0H 1H 2H 0I 1I 0I 1I 2I 0J 1J 2J 0K 1K)
    vpunpckhdq  ymmD, ymmD, ymmE        ; ymmD=(15 25 06 16 26 07 17 27 17 27 08 18 28 09 19 29
                                        ;       1L 2L 0M 1M 2M 0N 1N 2N 1N 2N 0O 1O 2O 0P 1P 2P)
    vpunpckldq  ymmE, ymmE, ymmH        ; ymmE=(22 03 13 23 04 14 24 05 24 05 15 25 06 16 26 07
                                        ;       2I 0J 1J 2J 0K 1K 2K 0L 2K 0L 1L 2L 0M 1M 2M 0N)

    vpshufd     ymmH, ymmG, 0x4E        ; ymmH=(0C 1C 2C 0D 0E 1E 2E 0F 08 18 28 09 0A 1A 2A 0B
                                        ;       0S 1S 2S 0T 0U 1U 2U 0V 0O 1O 2O 0P 0Q 1Q 2Q 0R)
    vpunpckldq  ymmG, ymmG, ymmC        ; ymmG=(08 18 28 09 19 29 0A 1A 0A 1A 2A 0B 1B 2B 0C 1C
                                        ;       0O 1O 2O 0P 1P 2P 0Q 1Q 0Q 1Q 2Q 0R 1R 2R 0S 1S)
    vpunpckhdq  ymmC, ymmC, ymmF        ; ymmC=(1D 2D 0E 1E 2E 0F 1F 2F 1F 2F 0G 1G 2G 0H 1H 2H
                                        ;       1T 2T 0U 1U 2U 0V 1V 2V 1V 2V -- -- -- -- -- --)
    vpunpckldq  ymmF, ymmF, ymmH        ; ymmF=(2A 0B 1B 2B 0C 1C 2C 0D 2C 0D 1D 2D 0E 1E 2E 0F
                                        ;       2Q 0R 1R 2R 0S 1S 2S 0T 2S 0T 1T 2T 0U 1U 2U 0V)

    vpunpcklqdq ymmH, ymmA, ymmE        ; ymmH=(00 10 20 01 11 21 02 12 22 03 13 23 04 14 24 05
                                        ;       0G 1G 2G 0H 1H 2H 0I 1I 2I 0J 1J 2J 0K 1K 2K 0L)
    vpunpcklqdq ymmG, ymmD, ymmG        ; ymmG=(15 25 06 16 26 07 17 27 08 18 28 09 19 29 0A 1A
                                        ;       1L 2L 0M 1M 2M 0N 1N 2N 0O 1O 2O 0P 1P 2P 0Q 1Q)
    vpunpcklqdq ymmC, ymmF, ymmC        ; ymmC=(2A 0B 1B 2B 0C 1C 2C 0D 1D 2D 0E 1E 2E 0F 1F 2F
                                        ;       2Q 0R 1R 2R 0S 1S 2S 0T 1T 2T 0U 1U 2U 0V 1V 2V)

    vperm2i128  ymmA, ymmH, ymmG, 0x20  ; ymmA=(00 10 20 01 11 21 02 12 22 03 13 23 04 14 24 05
                                        ;       15 25 06 16 26 07 17 27 08 18 28 09 19 29 0A 1A)
    vperm2i128  ymmD, ymmC, ymmH, 0x30  ; ymmD=(2A 0B 1B 2B 0C 1C 2C 0D 1D 2D 0E 1E 2E 0F 1F 2F
                                        ;       0G 1G 2G 0H 1H 2H 0I 1I 2I 0J 1J 2J 0K 1K 2K 0L)
    vperm2i128  ymmF, ymmG, ymmC, 0x31  ; ymmF=(1L 2L 0M 1M 2M 0N 1N 2N 0O 1O 2O 0P 1P 2P 0Q 1Q
                                        ;       2Q 0R 1R 2R 0S 1S 2S 0T 1T 2T 0U 1U 2U 0V 1V 2V)

    cmp         ecx, byte SIZEOF_YMMWORD
    jb          short .column_st64

    test        edi, SIZEOF_YMMWORD-1
    jnz         short .out1
    ; --(aligned)-------------------
    vmovntdq    YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    vmovntdq    YMMWORD [edi+1*SIZEOF_YMMWORD], ymmD
    vmovntdq    YMMWORD [edi+2*SIZEOF_YMMWORD], ymmF
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymmD
    vmovdqu     YMMWORD [edi+2*SIZEOF_YMMWORD], ymmF
.out0:
    add         edi, byte RGB_PIXELSIZE*SIZEOF_YMMWORD  ; outptr
    sub         ecx, byte SIZEOF_YMMWORD
    jz          near .endcolumn

    add         esi, byte SIZEOF_YMMWORD  ; inptr0
    dec         al                        ; Yctr
    jnz         near .Yloop_2nd

    add         ebx, byte SIZEOF_YMMWORD  ; inptr1
    add         edx, byte SIZEOF_YMMWORD  ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st64:
    lea         ecx, [ecx+ecx*2]            ; imul ecx, RGB_PIXELSIZE
    cmp         ecx, byte 2*SIZEOF_YMMWORD
    jb          short .column_st32
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymmD
    add         edi, byte 2*SIZEOF_YMMWORD  ; outptr
    vmovdqa     ymmA, ymmF
    sub         ecx, byte 2*SIZEOF_YMMWORD
    jmp         short .column_st31
.column_st32:
    cmp         ecx, byte SIZEOF_YMMWORD
    jb          short .column_st31
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    add         edi, byte SIZEOF_YMMWORD    ; outptr
    vmovdqa     ymmA, ymmD
    sub         ecx, byte SIZEOF_YMMWORD
    jmp         short .column_st31
.column_st31:
    cmp         ecx, byte SIZEOF_XMMWORD
    jb          short .column_st15
    vmovdqu     XMMWORD [edi+0*SIZEOF_XMMWORD], xmmA
    add         edi, byte SIZEOF_XMMWORD    ; outptr
    vperm2i128  ymmA, ymmA, ymmA, 1
    sub         ecx, byte SIZEOF_XMMWORD
.column_st15:
    ; Store the lower 8 bytes of xmmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st7
    vmovq       XMM_MMWORD [edi], xmmA
    add         edi, byte SIZEOF_MMWORD
    sub         ecx, byte SIZEOF_MMWORD
    vpsrldq     xmmA, xmmA, SIZEOF_MMWORD
.column_st7:
    ; Store the lower 4 bytes of xmmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_DWORD
    jb          short .column_st3
    vmovd       XMM_DWORD [edi], xmmA
    add         edi, byte SIZEOF_DWORD
    sub         ecx, byte SIZEOF_DWORD
    vpsrldq     xmmA, xmmA, SIZEOF_DWORD
.column_st3:
    ; Store the lower 2 bytes of eax to the output when it has enough
    ; space.
    vmovd       eax, xmmA
    cmp         ecx, byte SIZEOF_WORD
    jb          short .column_st1
    mov         word [edi], ax
    add         edi, byte SIZEOF_WORD
    sub         ecx, byte SIZEOF_WORD
    shr         eax, 16
.column_st1:
    ; Store the lower 1 byte of eax to the output when it has enough
    ; space.
    test        ecx, ecx
    jz          short .endcolumn
    mov         byte [edi], al

%else  ; RGB_PIXELSIZE == 4 ; -----------

%ifdef RGBX_FILLER_0XFF
    vpcmpeqb    ymm6, ymm6, ymm6        ; ymm6=XE=X(02468ACE********GIKMOQSU********)
    vpcmpeqb    ymm7, ymm7, ymm7        ; ymm7=XO=X(13579BDF********HJLNPRTV********)
%else
    vpxor       ymm6, ymm6, ymm6        ; ymm6=XE=X(02468ACE********GIKMOQSU********)
    vpxor       ymm7, ymm7, ymm7        ; ymm7=XO=X(13579BDF********HJLNPRTV********)
%endif
    ; ymmA=(00 02 04 06 08 0A 0C 0E ** 0G 0I 0K 0M 0O 0Q 0S 0U **)
    ; ymmB=(01 03 05 07 09 0B 0D 0F ** 0H 0J 0L 0N 0P 0R 0T 0V **)
    ; ymmC=(10 12 14 16 18 1A 1C 1E ** 1G 1I 1K 1M 1O 1Q 1S 1U **)
    ; ymmD=(11 13 15 17 19 1B 1D 1F ** 1H 1J 1L 1N 1P 1R 1T 1V **)
    ; ymmE=(20 22 24 26 28 2A 2C 2E ** 2G 2I 2K 2M 2O 2Q 2S 2U **)
    ; ymmF=(21 23 25 27 29 2B 2D 2F ** 2H 2J 2L 2N 2P 2R 2T 2V **)
    ; ymmG=(30 32 34 36 38 3A 3C 3E ** 3G 3I 3K 3M 3O 3Q 3S 3U **)
    ; ymmH=(31 33 35 37 39 3B 3D 3F ** 3H 3J 3L 3N 3P 3R 3T 3V **)

    vpunpcklbw  ymmA, ymmA, ymmC        ; ymmA=(00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E
                                        ;       0G 1G 0I 1I 0K 1K 0M 1M 0O 1O 0Q 1Q 0S 1S 0U 1U)
    vpunpcklbw  ymmE, ymmE, ymmG        ; ymmE=(20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E
                                        ;       2G 3G 2I 3I 2K 3K 2M 3M 2O 3O 2Q 3Q 2S 3S 2U 3U)
    vpunpcklbw  ymmB, ymmB, ymmD        ; ymmB=(01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F
                                        ;       0H 1H 0J 1J 0L 1L 0N 1N 0P 1P 0R 1R 0T 1T 0V 1V)
    vpunpcklbw  ymmF, ymmF, ymmH        ; ymmF=(21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F
                                        ;       2H 3H 2J 3J 2L 3L 2N 3N 2P 3P 2R 3R 2T 3T 2V 3V)

    vpunpckhwd  ymmC, ymmA, ymmE        ; ymmC=(08 18 28 38 0A 1A 2A 3A 0C 1C 2C 3C 0E 1E 2E 3E
                                        ;       0O 1O 2O 3O 0Q 1Q 2Q 3Q 0S 1S 2S 3S 0U 1U 2U 3U)
    vpunpcklwd  ymmA, ymmA, ymmE        ; ymmA=(00 10 20 30 02 12 22 32 04 14 24 34 06 16 26 36
                                        ;       0G 1G 2G 3G 0I 1I 2I 3I 0K 1K 2K 3K 0M 1M 2M 3M)
    vpunpckhwd  ymmG, ymmB, ymmF        ; ymmG=(09 19 29 39 0B 1B 2B 3B 0D 1D 2D 3D 0F 1F 2F 3F
                                        ;       0P 1P 2P 3P 0R 1R 2R 3R 0T 1T 2T 3T 0V 1V 2V 3V)
    vpunpcklwd  ymmB, ymmB, ymmF        ; ymmB=(01 11 21 31 03 13 23 33 05 15 25 35 07 17 27 37
                                        ;       0H 1H 2H 3H 0J 1J 2J 3J 0L 1L 2L 3L 0N 1N 2N 3N)

    vpunpckhdq  ymmE, ymmA, ymmB        ; ymmE=(04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37
                                        ;       0K 1K 2K 3K 0L 1L 2L 3L 0M 1M 2M 3M 0N 1N 2N 3N)
    vpunpckldq  ymmB, ymmA, ymmB        ; ymmB=(00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
                                        ;       0G 1G 2G 3G 0H 1H 2H 3H 0I 1I 2I 3I 0J 1J 2J 3J)
    vpunpckhdq  ymmF, ymmC, ymmG        ; ymmF=(0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F
                                        ;       0S 1S 2S 3S 0T 1T 2T 3T 0U 1U 2U 3U 0V 1V 2V 3V)
    vpunpckldq  ymmG, ymmC, ymmG        ; ymmG=(08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B
                                        ;       0O 1O 2O 3O 0P 1P 2P 3P 0Q 1Q 2Q 3Q 0R 1R 2R 3R)

    vperm2i128  ymmA, ymmB, ymmE, 0x20  ; ymmA=(00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
                                        ;       04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37)
    vperm2i128  ymmD, ymmG, ymmF, 0x20  ; ymmD=(08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B
                                        ;       0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F)
    vperm2i128  ymmC, ymmB, ymmE, 0x31  ; ymmC=(0G 1G 2G 3G 0H 1H 2H 3H 0I 1I 2I 3I 0J 1J 2J 3J
                                        ;       0K 1K 2K 3K 0L 1L 2L 3L 0M 1M 2M 3M 0N 1N 2N 3N)
    vperm2i128  ymmH, ymmG, ymmF, 0x31  ; ymmH=(0O 1O 2O 3O 0P 1P 2P 3P 0Q 1Q 2Q 3Q 0R 1R 2R 3R
                                        ;       0S 1S 2S 3S 0T 1T 2T 3T 0U 1U 2U 3U 0V 1V 2V 3V)

    cmp         ecx, byte SIZEOF_YMMWORD
    jb          short .column_st64

    test        edi, SIZEOF_YMMWORD-1
    jnz         short .out1
    ; --(aligned)-------------------
    vmovntdq    YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    vmovntdq    YMMWORD [edi+1*SIZEOF_YMMWORD], ymmD
    vmovntdq    YMMWORD [edi+2*SIZEOF_YMMWORD], ymmC
    vmovntdq    YMMWORD [edi+3*SIZEOF_YMMWORD], ymmH
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymmD
    vmovdqu     YMMWORD [edi+2*SIZEOF_YMMWORD], ymmC
    vmovdqu     YMMWORD [edi+3*SIZEOF_YMMWORD], ymmH
.out0:
    add         edi, RGB_PIXELSIZE*SIZEOF_YMMWORD  ; outptr
    sub         ecx, byte SIZEOF_YMMWORD
    jz          near .endcolumn

    add         esi, byte SIZEOF_YMMWORD  ; inptr0
    dec         al
    jnz         near .Yloop_2nd

    add         ebx, byte SIZEOF_YMMWORD  ; inptr1
    add         edx, byte SIZEOF_YMMWORD  ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st64:
    cmp         ecx, byte SIZEOF_YMMWORD/2
    jb          short .column_st32
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymmD
    add         edi, byte 2*SIZEOF_YMMWORD  ; outptr
    vmovdqa     ymmA, ymmC
    vmovdqa     ymmD, ymmH
    sub         ecx, byte SIZEOF_YMMWORD/2
.column_st32:
    cmp         ecx, byte SIZEOF_YMMWORD/4
    jb          short .column_st16
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymmA
    add         edi, byte SIZEOF_YMMWORD    ; outptr
    vmovdqa     ymmA, ymmD
    sub         ecx, byte SIZEOF_YMMWORD/4
.column_st16:
    cmp         ecx, byte SIZEOF_YMMWORD/8
    jb          short .column_st15
    vmovdqu     XMMWORD [edi+0*SIZEOF_XMMWORD], xmmA
    add         edi, byte SIZEOF_XMMWORD    ; outptr
    vperm2i128  ymmA, ymmA, ymmA, 1
    sub         ecx, byte SIZEOF_YMMWORD/8
.column_st15:
    ; Store two pixels (8 bytes) of ymmA to the output when it has enough
    ; space.
    cmp         ecx, byte SIZEOF_YMMWORD/16
    jb          short .column_st7
    vmovq       MMWORD [edi], xmmA
    add         edi, byte SIZEOF_YMMWORD/16*4
    sub         ecx, byte SIZEOF_YMMWORD/16
    vpsrldq     xmmA, SIZEOF_YMMWORD/16*4
.column_st7:
    ; Store one pixel (4 bytes) of ymmA to the output when it has enough
    ; space.
    test        ecx, ecx
    jz          short .endcolumn
    vmovd       XMM_DWORD [edi], xmmA

%endif  ; RGB_PIXELSIZE ; ---------------

.endcolumn:
    sfence                              ; flush the write buffer

.return:
    vzeroupper
    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    pop         ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
    ret

; --------------------------------------------------------------------------
;
; Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
;
; GLOBAL(void)
; jsimd_h2v2_merged_upsample_avx2(JDIMENSION output_width,
;                                 JSAMPIMAGE input_buf,
;                                 JDIMENSION in_row_group_ctr,
;                                 JSAMPARRAY output_buf);
;

%define output_width(b)      (b) + 8    ; JDIMENSION output_width
%define input_buf(b)         (b) + 12   ; JSAMPIMAGE input_buf
%define in_row_group_ctr(b)  (b) + 16   ; JDIMENSION in_row_group_ctr
%define output_buf(b)        (b) + 20   ; JSAMPARRAY output_buf

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_merged_upsample_avx2)

EXTN(jsimd_h2v2_merged_upsample_avx2):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         eax, POINTER [output_width(ebp)]

    mov         edi, JSAMPIMAGE [input_buf(ebp)]
    mov         ecx, JDIMENSION [in_row_group_ctr(ebp)]
    mov         esi, JSAMPARRAY [edi+0*SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [edi+1*SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [edi+2*SIZEOF_JSAMPARRAY]
    mov         edi, JSAMPARRAY [output_buf(ebp)]
    lea         esi, [esi+ecx*SIZEOF_JSAMPROW]

    push        edx                     ; inptr2
    push        ebx                     ; inptr1
    push        esi                     ; inptr00
    mov         ebx, esp

    push        edi                     ; output_buf (outptr0)
    push        ecx                     ; in_row_group_ctr
    push        ebx                     ; input_buf
    push        eax                     ; output_width

    call        near EXTN(jsimd_h2v1_merged_upsample_avx2)

    add         esi, byte SIZEOF_JSAMPROW  ; inptr01
    add         edi, byte SIZEOF_JSAMPROW  ; outptr1
    mov         POINTER [ebx+0*SIZEOF_POINTER], esi
    mov         POINTER [ebx-1*SIZEOF_POINTER], edi

    call        near EXTN(jsimd_h2v1_merged_upsample_avx2)

    add         esp, byte 7*SIZEOF_DWORD

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
