;
; jccolext.asm - colorspace conversion (AVX2)
;
; Copyright (C) 2015, Intel Corporation.
; Copyright (C) 2016, 2024, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jcolsamp.inc"

; --------------------------------------------------------------------------
;
; Convert some rows of samples to the output colorspace.
;
; GLOBAL(void)
; jsimd_rgb_ycc_convert_avx2(JDIMENSION img_width, JSAMPARRAY input_buf,
;                            JSAMPIMAGE output_buf, JDIMENSION output_row,
;                            int num_rows);
;

%define img_width(b)   (b) + 8          ; JDIMENSION img_width
%define input_buf(b)   (b) + 12         ; JSAMPARRAY input_buf
%define output_buf(b)  (b) + 16         ; JSAMPIMAGE output_buf
%define output_row(b)  (b) + 20         ; JDIMENSION output_row
%define num_rows(b)    (b) + 24         ; int num_rows

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_YMMWORD
                                        ; ymmword wk[WK_NUM]
%define WK_NUM         8
%define gotptr         wk(0) - SIZEOF_POINTER  ; void * gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_rgb_ycc_convert_avx2)

EXTN(jsimd_rgb_ycc_convert_avx2):
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

    mov         ecx, JDIMENSION [img_width(eax)]
    test        ecx, ecx
    jz          near .return

    push        ecx

    mov         esi, JSAMPIMAGE [output_buf(eax)]
    mov         ecx, JDIMENSION [output_row(eax)]
    mov         edi, JSAMPARRAY [esi+0*SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [esi+1*SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [esi+2*SIZEOF_JSAMPARRAY]
    lea         edi, [edi+ecx*SIZEOF_JSAMPROW]
    lea         ebx, [ebx+ecx*SIZEOF_JSAMPROW]
    lea         edx, [edx+ecx*SIZEOF_JSAMPROW]

    pop         ecx

    mov         esi, JSAMPARRAY [input_buf(eax)]
    mov         eax, INT [num_rows(eax)]
    test        eax, eax
    jle         near .return
    ALIGNX      16, 7
.rowloop:
    PUSHPIC     eax
    push        edx
    push        ebx
    push        edi
    push        esi
    push        ecx                     ; col

    mov         esi, JSAMPROW [esi]     ; inptr
    mov         edi, JSAMPROW [edi]     ; outptr0
    mov         ebx, JSAMPROW [ebx]     ; outptr1
    mov         edx, JSAMPROW [edx]     ; outptr2
    MOVPIC      eax, POINTER [gotptr]   ; load GOT address (eax)

    cmp         ecx, byte SIZEOF_YMMWORD
    jae         near .columnloop
    ALIGNX      16, 7

%if RGB_PIXELSIZE == 3  ; ---------------

.column_ld1:
    push        eax
    push        edx
    lea         ecx, [ecx+ecx*2]        ; imul ecx,RGB_PIXELSIZE
    test        cl, SIZEOF_BYTE
    jz          short .column_ld2
    sub         ecx, byte SIZEOF_BYTE
    movzx       eax, byte [esi+ecx]
.column_ld2:
    test        cl, SIZEOF_WORD
    jz          short .column_ld4
    sub         ecx, byte SIZEOF_WORD
    movzx       edx, word [esi+ecx]
    shl         eax, WORD_BIT
    or          eax, edx
.column_ld4:
    vmovd       xmmA, eax
    pop         edx
    pop         eax
    test        cl, SIZEOF_DWORD
    jz          short .column_ld8
    sub         ecx, byte SIZEOF_DWORD
    vmovd       xmmF, XMM_DWORD [esi+ecx]
    vpslldq     xmmA, xmmA, SIZEOF_DWORD
    vpor        xmmA, xmmA, xmmF
.column_ld8:
    test        cl, SIZEOF_MMWORD
    jz          short .column_ld16
    sub         ecx, byte SIZEOF_MMWORD
    vmovq       xmmB, XMM_MMWORD [esi+ecx]
    vpslldq     xmmA, xmmA, SIZEOF_MMWORD
    vpor        xmmA, xmmA, xmmB
.column_ld16:
    test        cl, SIZEOF_XMMWORD
    jz          short .column_ld32
    sub         ecx, byte SIZEOF_XMMWORD
    vmovdqu     xmmB, XMM_MMWORD [esi+ecx]
    vperm2i128  ymmA, ymmA, ymmA, 1
    vpor        ymmA, ymmB
.column_ld32:
    test        cl, SIZEOF_YMMWORD
    jz          short .column_ld64
    sub         ecx, byte SIZEOF_YMMWORD
    vmovdqa     ymmF, ymmA
    vmovdqu     ymmA, YMMWORD [esi+0*SIZEOF_YMMWORD]
.column_ld64:
    test        cl, 2*SIZEOF_YMMWORD
    mov         ecx, SIZEOF_YMMWORD
    jz          short .rgb_ycc_cnv
    vmovdqa     ymmB, ymmA
    vmovdqu     ymmA, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [esi+1*SIZEOF_YMMWORD]
    jmp         short .rgb_ycc_cnv
    ALIGNX      16, 7

.columnloop:
    vmovdqu     ymmA, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [esi+1*SIZEOF_YMMWORD]
    vmovdqu     ymmB, YMMWORD [esi+2*SIZEOF_YMMWORD]

.rgb_ycc_cnv:
    ; ymmA=(00 10 20 01 11 21 02 12 22 03 13 23 04 14 24 05
    ;       15 25 06 16 26 07 17 27 08 18 28 09 19 29 0A 1A)
    ; ymmF=(2A 0B 1B 2B 0C 1C 2C 0D 1D 2D 0E 1E 2E 0F 1F 2F
    ;       0G 1G 2G 0H 1H 2H 0I 1I 2I 0J 1J 2J 0K 1K 2K 0L)
    ; ymmB=(1L 2L 0M 1M 2M 0N 1N 2N 0O 1O 2O 0P 1P 2P 0Q 1Q
    ;       2Q 0R 1R 2R 0S 1S 2S 0T 1T 2T 0U 1U 2U 0V 1V 2V)

    vmovdqu     ymmC, ymmA
    vinserti128 ymmA, ymmF, xmmA, 0  ; ymmA=(00 10 20 01 11 21 02 12 22 03 13 23 04 14 24 05
                                     ;       0G 1G 2G 0H 1H 2H 0I 1I 2I 0J 1J 2J 0K 1K 2K 0L)
    vinserti128 ymmC, ymmC, xmmB, 0  ; ymmC=(1L 2L 0M 1M 2M 0N 1N 2N 0O 1O 2O 0P 1P 2P 0Q 1Q
                                     ;       15 25 06 16 26 07 17 27 08 18 28 09 19 29 0A 1A)
    vinserti128 ymmB, ymmB, xmmF, 0  ; ymmB=(2A 0B 1B 2B 0C 1C 2C 0D 1D 2D 0E 1E 2E 0F 1F 2F
                                     ;       2Q 0R 1R 2R 0S 1S 2S 0T 1T 2T 0U 1U 2U 0V 1V 2V)
    vperm2i128  ymmF, ymmC, ymmC, 1  ; ymmF=(15 25 06 16 26 07 17 27 08 18 28 09 19 29 0A 1A
                                     ;       1L 2L 0M 1M 2M 0N 1N 2N 0O 1O 2O 0P 1P 2P 0Q 1Q)

    vmovdqa     ymmG, ymmA
    vpslldq     ymmA, ymmA, 8     ; ymmA=(-- -- -- -- -- -- -- -- 00 10 20 01 11 21 02 12
                                  ;       22 03 13 23 04 14 24 05 0G 1G 2G 0H 1H 2H 0I 1I)
    vpsrldq     ymmG, ymmG, 8     ; ymmG=(22 03 13 23 04 14 24 05 0G 1G 2G 0H 1H 2H 0I 1I
                                  ;       2I 0J 1J 2J 0K 1K 2K 0L -- -- -- -- -- -- -- --)

    vpunpckhbw  ymmA, ymmA, ymmF  ; ymmA=(00 08 10 18 20 28 01 09 11 19 21 29 02 0A 12 1A
                                  ;       0G 0O 1G 1O 2G 2O 0H 0P 1H 1P 2H 2P 0I 0Q 1I 1Q)
    vpslldq     ymmF, ymmF, 8     ; ymmF=(-- -- -- -- -- -- -- -- 15 25 06 16 26 07 17 27
                                  ;       08 18 28 09 19 29 0A 1A 1L 2L 0M 1M 2M 0N 1N 2N)

    vpunpcklbw  ymmG, ymmG, ymmB  ; ymmG=(22 2A 03 0B 13 1B 23 2B 04 0C 14 1C 24 2C 05 0D
                                  ;       2I 2Q 0J 0R 1J 1R 2J 2R 0K 0S 1K 1S 2K 2S 0L 0T)
    vpunpckhbw  ymmF, ymmF, ymmB  ; ymmF=(15 1D 25 2D 06 0E 16 1E 26 2E 07 0F 17 1F 27 2F
                                  ;       1L 1T 2L 2T 0M 0U 1M 1U 2M 2U 0N 0V 1N 1V 2N 2V)

    vmovdqa     ymmD, ymmA
    vpslldq     ymmA, ymmA, 8     ; ymmA=(-- -- -- -- -- -- -- -- 00 08 10 18 20 28 01 09
                                  ;       11 19 21 29 02 0A 12 1A 0G 0O 1G 1O 2G 2O 0H 0P)
    vpsrldq     ymmD, ymmD, 8     ; ymmD=(11 19 21 29 02 0A 12 1A 0G 0O 1G 1O 2G 2O 0H 0P
                                  ;       1H 1P 2H 2P 0I 0Q 1I 1Q -- -- -- -- -- -- -- --)

    vpunpckhbw  ymmA, ymmA, ymmG  ; ymmA=(00 04 08 0C 10 14 18 1C 20 24 28 2C 01 05 09 0D
                                  ;       0G 0K 0O 0S 1G 1K 1O 1S 2G 2K 2O 2S 0H 0L 0P 0T)
    vpslldq     ymmG, ymmG, 8     ; ymmG=(-- -- -- -- -- -- -- -- 22 2A 03 0B 13 1B 23 2B
                                  ;       04 0C 14 1C 24 2C 05 0D 2I 2Q 0J 0R 1J 1R 2J 2R)

    vpunpcklbw  ymmD, ymmD, ymmF  ; ymmD=(11 15 19 1D 21 25 29 2D 02 06 0A 0E 12 16 1A 1E
                                  ;       1H 1L 1P 1T 2H 2L 2P 2T 0I 0M 0Q 0U 1I 1M 1Q 1U)
    vpunpckhbw  ymmG, ymmG, ymmF  ; ymmG=(22 26 2A 2E 03 07 0B 0F 13 17 1B 1F 23 27 2B 2F
                                  ;       2I 2M 2Q 2U 0J 0N 0R 0V 1J 1N 1R 1V 2J 2N 2R 2V)

    vmovdqa     ymmE, ymmA
    vpslldq     ymmA, ymmA, 8     ; ymmA=(-- -- -- -- -- -- -- -- 00 04 08 0C 10 14 18 1C
                                  ;       20 24 28 2C 01 05 09 0D 0G 0K 0O 0S 1G 1K 1O 1S)
    vpsrldq     ymmE, ymmE, 8     ; ymmE=(20 24 28 2C 01 05 09 0D 0G 0K 0O 0S 1G 1K 1O 1S
                                  ;       2G 2K 2O 2S 0H 0L 0P 0T -- -- -- -- -- -- -- --)

    vpunpckhbw  ymmA, ymmA, ymmD  ; ymmA=(00 02 04 06 08 0A 0C 0E 10 12 14 16 18 1A 1C 1E
                                  ;       0G 0I 0K 0M 0O 0Q 0S 0U 1G 1I 1K 1M 1O 1Q 1S 1U)
    vpslldq     ymmD, ymmD, 8     ; ymmD=(-- -- -- -- -- -- -- -- 11 15 19 1D 21 25 29 2D
                                  ;       02 06 0A 0E 12 16 1A 1E 1H 1L 1P 1T 2H 2L 2P 2T)

    vpunpcklbw  ymmE, ymmE, ymmG  ; ymmE=(20 22 24 26 28 2A 2C 2E 01 03 05 07 09 0B 0D 0F
                                  ;       2G 2I 2K 2M 2O 2Q 2S 2U 0H 0J 0L 0N 0P 0R 0T 0V)
    vpunpckhbw  ymmD, ymmD, ymmG  ; ymmD=(11 13 15 17 19 1B 1D 1F 21 23 25 27 29 2B 2D 2F
                                  ;       1H 1J 1L 1N 1P 1R 1T 1V 2H 2J 2L 2N 2P 2R 2T 2V)

    vpxor       ymmH, ymmH, ymmH

    vmovdqa     ymmC, ymmA
    vpunpcklbw  ymmA, ymmA, ymmH  ; ymmA=(00 02 04 06 08 0A 0C 0E 0G 0I 0K 0M 0O 0Q 0S 0U)
    vpunpckhbw  ymmC, ymmC, ymmH  ; ymmC=(10 12 14 16 18 1A 1C 1E 1G 1I 1K 1M 1O 1Q 1S 1U)

    vmovdqa     ymmB, ymmE
    vpunpcklbw  ymmE, ymmE, ymmH  ; ymmE=(20 22 24 26 28 2A 2C 2E 2G 2I 2K 2M 2O 2Q 2S 2U)
    vpunpckhbw  ymmB, ymmB, ymmH  ; ymmB=(01 03 05 07 09 0B 0D 0F 0H 0J 0L 0N 0P 0R 0T 0V)

    vmovdqa     ymmF, ymmD
    vpunpcklbw  ymmD, ymmD, ymmH  ; ymmD=(11 13 15 17 19 1B 1D 1F 1H 1J 1L 1N 1P 1R 1T 1V)
    vpunpckhbw  ymmF, ymmF, ymmH  ; ymmF=(21 23 25 27 29 2B 2D 2F 2H 2J 2L 2N 2P 2R 2T 2V)

%else  ; RGB_PIXELSIZE == 4 ; -----------

.column_ld1:
    test        cl, SIZEOF_XMMWORD/16
    jz          short .column_ld2
    sub         ecx, byte SIZEOF_XMMWORD/16
    vmovd       xmmA, XMM_DWORD [esi+ecx*RGB_PIXELSIZE]
.column_ld2:
    test        cl, SIZEOF_XMMWORD/8
    jz          short .column_ld4
    sub         ecx, byte SIZEOF_XMMWORD/8
    vmovq       xmmF, XMM_MMWORD [esi+ecx*RGB_PIXELSIZE]
    vpslldq     xmmA, xmmA, SIZEOF_MMWORD
    vpor        xmmA, xmmA, xmmF
.column_ld4:
    test        cl, SIZEOF_XMMWORD/4
    jz          short .column_ld8
    sub         ecx, byte SIZEOF_XMMWORD/4
    vmovdqa     xmmF, xmmA
    vperm2i128  ymmF, ymmF, ymmF, 1
    vmovdqu     xmmA, XMMWORD [esi+ecx*RGB_PIXELSIZE]
    vpor        ymmA, ymmA, ymmF
.column_ld8:
    test        cl, SIZEOF_XMMWORD/2
    jz          short .column_ld16
    sub         ecx, byte SIZEOF_XMMWORD/2
    vmovdqa     ymmF, ymmA
    vmovdqu     ymmA, YMMWORD [esi+ecx*RGB_PIXELSIZE]
.column_ld16:
    test        cl, SIZEOF_XMMWORD
    mov         ecx, SIZEOF_YMMWORD
    jz          short .rgb_ycc_cnv
    vmovdqa     ymmE, ymmA
    vmovdqa     ymmH, ymmF
    vmovdqu     ymmA, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [esi+1*SIZEOF_YMMWORD]
    jmp         short .rgb_ycc_cnv
    ALIGNX      16, 7

.columnloop:
    vmovdqu     ymmA, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     ymmF, YMMWORD [esi+1*SIZEOF_YMMWORD]
    vmovdqu     ymmE, YMMWORD [esi+2*SIZEOF_YMMWORD]
    vmovdqu     ymmH, YMMWORD [esi+3*SIZEOF_YMMWORD]

.rgb_ycc_cnv:
    ; ymmA=(00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
    ;       04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37)
    ; ymmF=(08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B
    ;       0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F)
    ; ymmE=(0G 1G 2G 3G 0H 1H 2H 3H 0I 1I 2I 3I 0J 1J 2J 3J
    ;       0K 1K 2K 3K 0L 1L 2L 3L 0M 1M 2M 3M 0N 1N 2N 3N)
    ; ymmH=(0O 1O 2O 3O 0P 1P 2P 3P 0Q 1Q 2Q 3Q 0R 1R 2R 3R
    ;       0S 1S 2S 3S 0T 1T 2T 3T 0U 1U 2U 3U 0V 1V 2V 3V)

    vmovdqa     ymmB, ymmA
    vinserti128 ymmA, ymmA, xmmE, 1     ; ymmA=(00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
                                        ;       0G 1G 2G 3G 0H 1H 2H 3H 0I 1I 2I 3I 0J 1J 2J 3J)
    vperm2i128  ymmE, ymmB, ymmE, 0x31  ; ymmE=(04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37
                                        ;       0K 1K 2K 3K 0L 1L 2L 3L 0M 1M 2M 3M 0N 1N 2N 3N)

    vmovdqa     ymmB, ymmF
    vinserti128 ymmF, ymmF, xmmH, 1     ; ymmF=(08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B
                                        ;       0O 1O 2O 3O 0P 1P 2P 3P 0Q 1Q 2Q 3Q 0R 1R 2R 3R)
    vperm2i128  ymmH, ymmB, ymmH, 0x31  ; ymmH=(0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F
                                        ;       0S 1S 2S 3S 0T 1T 2T 3T 0U 1U 2U 3U 0V 1V 2V 3V)

    vmovdqa     ymmD, ymmA
    vpunpcklbw  ymmA, ymmA, ymmE      ; ymmA=(00 04 10 14 20 24 30 34 01 05 11 15 21 25 31 35
                                      ;       0G 0K 1G 1K 2G 2K 3G 3K 0H 0L 1H 1L 2H 2L 3H 3L)
    vpunpckhbw  ymmD, ymmD, ymmE      ; ymmD=(02 06 12 16 22 26 32 36 03 07 13 17 23 27 33 37
                                      ;       0I 0M 1I 1M 2I 2M 3I 3M 0J 0N 1J 1N 2J 2N 3J 3N)

    vmovdqa     ymmC, ymmF
    vpunpcklbw  ymmF, ymmF, ymmH      ; ymmF=(08 0C 18 1C 28 2C 38 3C 09 0D 19 1D 29 2D 39 3D
                                      ;       0O 0S 1O 1S 2O 2S 3O 3S 0P 0T 1P 1T 2P 2T 3P 3T)
    vpunpckhbw  ymmC, ymmC, ymmH      ; ymmC=(0A 0E 1A 1E 2A 2E 3A 3E 0B 0F 1B 1F 2B 2F 3B 3F
                                      ;       0Q 0U 1Q 1U 2Q 2U 3Q 3U 0R 0V 1R 1V 2R 2V 3R 3V)

    vmovdqa     ymmB, ymmA
    vpunpcklwd  ymmA, ymmA, ymmF      ; ymmA=(00 04 08 0C 10 14 18 1C 20 24 28 2C 30 34 38 3C
                                      ;       0G 0K 0O 0S 1G 1K 1O 1S 2G 2K 2O 2S 3G 3K 3O 3S)
    vpunpckhwd  ymmB, ymmB, ymmF      ; ymmB=(01 05 09 0D 11 15 19 1D 21 25 29 2D 31 35 39 3D
                                      ;       0H 0L 0P 0T 1H 1L 1P 1T 2H 2L 2P 2T 3H 3L 3P 3T)

    vmovdqa     ymmG, ymmD
    vpunpcklwd  ymmD, ymmD, ymmC      ; ymmD=(02 06 0A 0E 12 16 1A 1E 22 26 2A 2E 32 36 3A 3E
                                      ;       0I 0M 0Q 0U 1I 1M 1Q 1U 2I 2M 2Q 2U 3I 3M 3Q 3U)
    vpunpckhwd  ymmG, ymmG, ymmC      ; ymmG=(03 07 0B 0F 13 17 1B 1F 23 27 2B 2F 33 37 3B 3F
                                      ;       0J 0N 0R 0V 1J 1N 1R 1V 2J 2N 2R 2V 3J 3N 3R 3V)

    vmovdqa     ymmE, ymmA
    vpunpcklbw  ymmA, ymmA, ymmD      ; ymmA=(00 02 04 06 08 0A 0C 0E 10 12 14 16 18 1A 1C 1E
                                      ;       0G 0I 0K 0M 0O 0Q 0S 0U 1G 1I 1K 1M 1O 1Q 1S 1U)
    vpunpckhbw  ymmE, ymmE, ymmD      ; ymmE=(20 22 24 26 28 2A 2C 2E 30 32 34 36 38 3A 3C 3E
                                      ;       2G 2I 2K 2M 2O 2Q 2S 2U 3G 3I 3K 3M 3O 3Q 3S 3U)

    vmovdqa     ymmH, ymmB
    vpunpcklbw  ymmB, ymmB, ymmG      ; ymmB=(01 03 05 07 09 0B 0D 0F 11 13 15 17 19 1B 1D 1F
                                      ;       0H 0J 0L 0N 0P 0R 0T 0V 1H 1J 1L 1N 1P 1R 1T 1V)
    vpunpckhbw  ymmH, ymmH, ymmG      ; ymmH=(21 23 25 27 29 2B 2D 2F 31 33 35 37 39 3B 3D 3F
                                      ;       2H 2J 2L 2N 2P 2R 2T 2V 3H 3J 3L 3N 3P 3R 3T 3V)

    vpxor       ymmF, ymmF, ymmF

    vmovdqa     ymmC, ymmA
    vpunpcklbw  ymmA, ymmA, ymmF      ; ymmA=(00 02 04 06 08 0A 0C 0E 0G 0I 0K 0M 0O 0Q 0S 0U)
    vpunpckhbw  ymmC, ymmC, ymmF      ; ymmC=(10 12 14 16 18 1A 1C 1E 1G 1I 1K 1M 1O 1Q 1S 1U)

    vmovdqa     ymmD, ymmB
    vpunpcklbw  ymmB, ymmB, ymmF      ; ymmB=(01 03 05 07 09 0B 0D 0F 0H 0J 0L 0N 0P 0R 0T 0V)
    vpunpckhbw  ymmD, ymmD, ymmF      ; ymmD=(11 13 15 17 19 1B 1D 1F 1H 1J 1L 1N 1P 1R 1T 1V)

    vmovdqa     ymmG, ymmE
    vpunpcklbw  ymmE, ymmE, ymmF      ; ymmE=(20 22 24 26 28 2A 2C 2E 2G 2I 2K 2M 2O 2Q 2S 2U)
    vpunpckhbw  ymmG, ymmG, ymmF      ; ymmG=(30 32 34 36 38 3A 3C 3E 3G 3I 3K 3M 3O 3Q 3S 3U)

    vpunpcklbw  ymmF, ymmF, ymmH
    vpunpckhbw  ymmH, ymmH, ymmH
    vpsrlw      ymmF, ymmF, BYTE_BIT  ; ymmF=(21 23 25 27 29 2B 2D 2F 2H 2J 2L 2N 2P 2R 2T 2V)
    vpsrlw      ymmH, ymmH, BYTE_BIT  ; ymmH=(31 33 35 37 39 3B 3D 3F 3H 3J 3L 3N 3P 3R 3T 3V)

%endif  ; RGB_PIXELSIZE ; ---------------

    ; ymm0=R(02468ACEGIKMOQSU)=RE, ymm2=G(02468ACEGIKMOQSU)=GE, ymm4=B(02468ACEGIKMOQSU)=BE
    ; ymm1=R(13579BDFHJLNPRTV)=RO, ymm3=G(13579BDFHJLNPRTV)=GO, ymm5=B(13579BDFHJLNPRTV)=BO

    ; (Original)
    ; Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    ; Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + CENTERJSAMPLE
    ; Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + CENTERJSAMPLE
    ;
    ; (This implementation)
    ; Y  =  0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G
    ; Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + CENTERJSAMPLE
    ; Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + CENTERJSAMPLE

    vmovdqa     YMMWORD [wk(0)], ymm0   ; wk(0)=RE
    vmovdqa     YMMWORD [wk(1)], ymm1   ; wk(1)=RO
    vmovdqa     YMMWORD [wk(2)], ymm4   ; wk(2)=BE
    vmovdqa     YMMWORD [wk(3)], ymm5   ; wk(3)=BO

    vmovdqa     ymm6, ymm1
    vpunpcklwd  ymm1, ymm1, ymm3
    vpunpckhwd  ymm6, ymm6, ymm3
    vmovdqa     ymm7, ymm1
    vmovdqa     ymm4, ymm6
    vpmaddwd    ymm1, ymm1, [GOTOFF(eax,PW_F0299_F0337)]  ; ymm1=ROL*FIX(0.299)+GOL*FIX(0.337)
    vpmaddwd    ymm6, ymm6, [GOTOFF(eax,PW_F0299_F0337)]  ; ymm6=ROH*FIX(0.299)+GOH*FIX(0.337)
    vpmaddwd    ymm7, ymm7, [GOTOFF(eax,PW_MF016_MF033)]  ; ymm7=ROL*-FIX(0.168)+GOL*-FIX(0.331)
    vpmaddwd    ymm4, ymm4, [GOTOFF(eax,PW_MF016_MF033)]  ; ymm4=ROH*-FIX(0.168)+GOH*-FIX(0.331)

    vmovdqa     YMMWORD [wk(4)], ymm1   ; wk(4)=ROL*FIX(0.299)+GOL*FIX(0.337)
    vmovdqa     YMMWORD [wk(5)], ymm6   ; wk(5)=ROH*FIX(0.299)+GOH*FIX(0.337)

    vpxor       ymm1, ymm1, ymm1
    vpxor       ymm6, ymm6, ymm6
    vpunpcklwd  ymm1, ymm1, ymm5        ; ymm1=BOL
    vpunpckhwd  ymm6, ymm6, ymm5        ; ymm6=BOH
    vpsrld      ymm1, ymm1, 1           ; ymm1=BOL*FIX(0.500)
    vpsrld      ymm6, ymm6, 1           ; ymm6=BOH*FIX(0.500)

    vmovdqa     ymm5, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; ymm5=[PD_ONEHALFM1_CJ]

    vpaddd      ymm7, ymm7, ymm1
    vpaddd      ymm4, ymm4, ymm6
    vpaddd      ymm7, ymm7, ymm5
    vpaddd      ymm4, ymm4, ymm5
    vpsrld      ymm7, ymm7, SCALEBITS   ; ymm7=CbOL
    vpsrld      ymm4, ymm4, SCALEBITS   ; ymm4=CbOH
    vpackssdw   ymm7, ymm7, ymm4        ; ymm7=CbO

    vmovdqa     ymm1, YMMWORD [wk(2)]   ; ymm1=BE

    vmovdqa     ymm6, ymm0
    vpunpcklwd  ymm0, ymm0, ymm2
    vpunpckhwd  ymm6, ymm6, ymm2
    vmovdqa     ymm5, ymm0
    vmovdqa     ymm4, ymm6
    vpmaddwd    ymm0, ymm0, [GOTOFF(eax,PW_F0299_F0337)]  ; ymm0=REL*FIX(0.299)+GEL*FIX(0.337)
    vpmaddwd    ymm6, ymm6, [GOTOFF(eax,PW_F0299_F0337)]  ; ymm6=REH*FIX(0.299)+GEH*FIX(0.337)
    vpmaddwd    ymm5, ymm5, [GOTOFF(eax,PW_MF016_MF033)]  ; ymm5=REL*-FIX(0.168)+GEL*-FIX(0.331)
    vpmaddwd    ymm4, ymm4, [GOTOFF(eax,PW_MF016_MF033)]  ; ymm4=REH*-FIX(0.168)+GEH*-FIX(0.331)

    vmovdqa     YMMWORD [wk(6)], ymm0   ; wk(6)=REL*FIX(0.299)+GEL*FIX(0.337)
    vmovdqa     YMMWORD [wk(7)], ymm6   ; wk(7)=REH*FIX(0.299)+GEH*FIX(0.337)

    vpxor       ymm0, ymm0, ymm0
    vpxor       ymm6, ymm6, ymm6
    vpunpcklwd  ymm0, ymm0, ymm1        ; ymm0=BEL
    vpunpckhwd  ymm6, ymm6, ymm1        ; ymm6=BEH
    vpsrld      ymm0, ymm0, 1           ; ymm0=BEL*FIX(0.500)
    vpsrld      ymm6, ymm6, 1           ; ymm6=BEH*FIX(0.500)

    vmovdqa     ymm1, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; ymm1=[PD_ONEHALFM1_CJ]

    vpaddd      ymm5, ymm5, ymm0
    vpaddd      ymm4, ymm4, ymm6
    vpaddd      ymm5, ymm5, ymm1
    vpaddd      ymm4, ymm4, ymm1
    vpsrld      ymm5, ymm5, SCALEBITS   ; ymm5=CbEL
    vpsrld      ymm4, ymm4, SCALEBITS   ; ymm4=CbEH
    vpackssdw   ymm5, ymm5, ymm4        ; ymm5=CbE

    vpsllw      ymm7, ymm7, BYTE_BIT
    vpor        ymm5, ymm5, ymm7        ; ymm5=Cb
    vmovdqu     YMMWORD [ebx], ymm5     ; Save Cb

    vmovdqa     ymm0, YMMWORD [wk(3)]   ; ymm0=BO
    vmovdqa     ymm6, YMMWORD [wk(2)]   ; ymm6=BE
    vmovdqa     ymm1, YMMWORD [wk(1)]   ; ymm1=RO

    vmovdqa     ymm4, ymm0
    vpunpcklwd  ymm0, ymm0, ymm3
    vpunpckhwd  ymm4, ymm4, ymm3
    vmovdqa     ymm7, ymm0
    vmovdqa     ymm5, ymm4
    vpmaddwd    ymm0, ymm0, [GOTOFF(eax,PW_F0114_F0250)]  ; ymm0=BOL*FIX(0.114)+GOL*FIX(0.250)
    vpmaddwd    ymm4, ymm4, [GOTOFF(eax,PW_F0114_F0250)]  ; ymm4=BOH*FIX(0.114)+GOH*FIX(0.250)
    vpmaddwd    ymm7, ymm7, [GOTOFF(eax,PW_MF008_MF041)]  ; ymm7=BOL*-FIX(0.081)+GOL*-FIX(0.418)
    vpmaddwd    ymm5, ymm5, [GOTOFF(eax,PW_MF008_MF041)]  ; ymm5=BOH*-FIX(0.081)+GOH*-FIX(0.418)

    vmovdqa     ymm3, [GOTOFF(eax,PD_ONEHALF)]            ; ymm3=[PD_ONEHALF]

    vpaddd      ymm0, ymm0, YMMWORD [wk(4)]
    vpaddd      ymm4, ymm4, YMMWORD [wk(5)]
    vpaddd      ymm0, ymm0, ymm3
    vpaddd      ymm4, ymm4, ymm3
    vpsrld      ymm0, ymm0, SCALEBITS   ; ymm0=YOL
    vpsrld      ymm4, ymm4, SCALEBITS   ; ymm4=YOH
    vpackssdw   ymm0, ymm0, ymm4        ; ymm0=YO

    vpxor       ymm3, ymm3, ymm3
    vpxor       ymm4, ymm4, ymm4
    vpunpcklwd  ymm3, ymm3, ymm1        ; ymm3=ROL
    vpunpckhwd  ymm4, ymm4, ymm1        ; ymm4=ROH
    vpsrld      ymm3, ymm3, 1           ; ymm3=ROL*FIX(0.500)
    vpsrld      ymm4, ymm4, 1           ; ymm4=ROH*FIX(0.500)

    vmovdqa     ymm1, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; ymm1=[PD_ONEHALFM1_CJ]

    vpaddd      ymm7, ymm7, ymm3
    vpaddd      ymm5, ymm5, ymm4
    vpaddd      ymm7, ymm7, ymm1
    vpaddd      ymm5, ymm5, ymm1
    vpsrld      ymm7, ymm7, SCALEBITS   ; ymm7=CrOL
    vpsrld      ymm5, ymm5, SCALEBITS   ; ymm5=CrOH
    vpackssdw   ymm7, ymm7, ymm5        ; ymm7=CrO

    vmovdqa     ymm3, YMMWORD [wk(0)]   ; ymm3=RE

    vmovdqa     ymm4, ymm6
    vpunpcklwd  ymm6, ymm6, ymm2
    vpunpckhwd  ymm4, ymm4, ymm2
    vmovdqa     ymm1, ymm6
    vmovdqa     ymm5, ymm4
    vpmaddwd    ymm6, ymm6, [GOTOFF(eax,PW_F0114_F0250)]  ; ymm6=BEL*FIX(0.114)+GEL*FIX(0.250)
    vpmaddwd    ymm4, ymm4, [GOTOFF(eax,PW_F0114_F0250)]  ; ymm4=BEH*FIX(0.114)+GEH*FIX(0.250)
    vpmaddwd    ymm1, ymm1, [GOTOFF(eax,PW_MF008_MF041)]  ; ymm1=BEL*-FIX(0.081)+GEL*-FIX(0.418)
    vpmaddwd    ymm5, ymm5, [GOTOFF(eax,PW_MF008_MF041)]  ; ymm5=BEH*-FIX(0.081)+GEH*-FIX(0.418)

    vmovdqa     ymm2, [GOTOFF(eax,PD_ONEHALF)]            ; ymm2=[PD_ONEHALF]

    vpaddd      ymm6, ymm6, YMMWORD [wk(6)]
    vpaddd      ymm4, ymm4, YMMWORD [wk(7)]
    vpaddd      ymm6, ymm6, ymm2
    vpaddd      ymm4, ymm4, ymm2
    vpsrld      ymm6, ymm6, SCALEBITS   ; ymm6=YEL
    vpsrld      ymm4, ymm4, SCALEBITS   ; ymm4=YEH
    vpackssdw   ymm6, ymm6, ymm4        ; ymm6=YE

    vpsllw      ymm0, ymm0, BYTE_BIT
    vpor        ymm6, ymm6, ymm0        ; ymm6=Y
    vmovdqu     YMMWORD [edi], ymm6     ; Save Y

    vpxor       ymm2, ymm2, ymm2
    vpxor       ymm4, ymm4, ymm4
    vpunpcklwd  ymm2, ymm2, ymm3        ; ymm2=REL
    vpunpckhwd  ymm4, ymm4, ymm3        ; ymm4=REH
    vpsrld      ymm2, ymm2, 1           ; ymm2=REL*FIX(0.500)
    vpsrld      ymm4, ymm4, 1           ; ymm4=REH*FIX(0.500)

    vmovdqa     ymm0, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; ymm0=[PD_ONEHALFM1_CJ]

    vpaddd      ymm1, ymm1, ymm2
    vpaddd      ymm5, ymm5, ymm4
    vpaddd      ymm1, ymm1, ymm0
    vpaddd      ymm5, ymm5, ymm0
    vpsrld      ymm1, ymm1, SCALEBITS   ; ymm1=CrEL
    vpsrld      ymm5, ymm5, SCALEBITS   ; ymm5=CrEH
    vpackssdw   ymm1, ymm1, ymm5        ; ymm1=CrE

    vpsllw      ymm7, ymm7, BYTE_BIT
    vpor        ymm1, ymm1, ymm7        ; ymm1=Cr
    vmovdqu     YMMWORD [edx], ymm1     ; Save Cr

    sub         ecx, byte SIZEOF_YMMWORD
    add         esi, RGB_PIXELSIZE*SIZEOF_YMMWORD  ; inptr
    add         edi, byte SIZEOF_YMMWORD           ; outptr0
    add         ebx, byte SIZEOF_YMMWORD           ; outptr1
    add         edx, byte SIZEOF_YMMWORD           ; outptr2
    cmp         ecx, byte SIZEOF_YMMWORD
    jae         near .columnloop
    test        ecx, ecx
    jnz         near .column_ld1

    pop         ecx                     ; col
    pop         esi
    pop         edi
    pop         ebx
    pop         edx
    POPPIC      eax

    add         esi, byte SIZEOF_JSAMPROW  ; input_buf
    add         edi, byte SIZEOF_JSAMPROW
    add         ebx, byte SIZEOF_JSAMPROW
    add         edx, byte SIZEOF_JSAMPROW
    dec         eax                        ; num_rows
    jg          near .rowloop

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

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
