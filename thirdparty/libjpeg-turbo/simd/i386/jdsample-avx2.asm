;
; jdsample.asm - upsampling (AVX2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2015, Intel Corporation.
; Copyright (C) 2016, 2024, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jsimdext.inc"

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_fancy_upsample_avx2)

EXTN(jconst_fancy_upsample_avx2):

PW_ONE   times 16 dw 1
PW_TWO   times 16 dw 2
PW_THREE times 16 dw 3
PW_SEVEN times 16 dw 7
PW_EIGHT times 16 dw 8

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Fancy processing for the common case of 2:1 horizontal and 1:1 vertical.
;
; The upsampling algorithm is linear interpolation between pixel centers,
; also known as a "triangle filter".  This is a good compromise between
; speed and visual quality.  The centers of the output pixels are 1/4 and 3/4
; of the way between input pixel centers.
;
; GLOBAL(void)
; jsimd_h2v1_fancy_upsample_avx2(int max_v_samp_factor,
;                                JDIMENSION downsampled_width,
;                                JSAMPARRAY input_data,
;                                JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define downsamp_width(b)   (b) + 12    ; JDIMENSION downsampled_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_fancy_upsample_avx2)

EXTN(jsimd_h2v1_fancy_upsample_avx2):
    push        ebp
    mov         ebp, esp
    PUSHPIC     ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    GET_GOT     ebx                     ; get GOT address

    mov         eax, JDIMENSION [downsamp_width(ebp)]  ; colctr
    test        eax, eax
    jz          near .return

    mov         ecx, INT [max_v_samp(ebp)]  ; rowctr
    test        ecx, ecx
    jz          near .return

    mov         esi, JSAMPARRAY [input_data(ebp)]    ; input_data
    mov         edi, POINTER [output_data_ptr(ebp)]
    mov         edi, JSAMPARRAY [edi]                ; output_data
    ALIGNX      16, 7
.rowloop:
    push        eax                     ; colctr
    push        edi
    push        esi

    mov         esi, JSAMPROW [esi]     ; inptr
    mov         edi, JSAMPROW [edi]     ; outptr

    test        eax, SIZEOF_YMMWORD-1
    jz          short .skip
    mov         dl, JSAMPLE [esi+(eax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [esi+eax*SIZEOF_JSAMPLE], dl    ; insert a dummy sample
.skip:
    vpxor       ymm0, ymm0, ymm0                ; ymm0=(all 0's)
    vpcmpeqb    xmm7, xmm7, xmm7
    vpsrldq     xmm7, xmm7, (SIZEOF_XMMWORD-1)  ; (ff -- -- -- ... -- --) LSB is ff
    vpand       ymm7, ymm7, YMMWORD [esi+0*SIZEOF_YMMWORD]

    add         eax, byte SIZEOF_YMMWORD-1
    and         eax, byte -SIZEOF_YMMWORD
    cmp         eax, byte SIZEOF_YMMWORD
    ja          short .columnloop
    ALIGNX      16, 7

.columnloop_last:
    vpcmpeqb    xmm6, xmm6, xmm6
    vpslldq     xmm6, xmm6, (SIZEOF_XMMWORD-1)
    vperm2i128  ymm6, ymm6, ymm6, 1             ; (---- ---- ... ---- ---- ff) MSB is ff
    vpand       ymm6, ymm6, YMMWORD [esi+0*SIZEOF_YMMWORD]
    jmp         short .upsample
    ALIGNX      16, 7

.columnloop:
    vmovdqu     ymm6, YMMWORD [esi+1*SIZEOF_YMMWORD]
    vperm2i128  ymm6, ymm0, ymm6, 0x20
    vpslldq     ymm6, ymm6, 15

.upsample:
    vmovdqu     ymm1, YMMWORD [esi+0*SIZEOF_YMMWORD]  ; ymm1=( 0  1  2 ... 29 30 31)

    vperm2i128  ymm2, ymm0, ymm1, 0x20
    vpalignr    ymm2, ymm1, ymm2, 15            ; ymm2=(--  0  1 ... 28 29 30)
    vperm2i128  ymm4, ymm0, ymm1, 0x03
    vpalignr    ymm3, ymm4, ymm1, 1             ; ymm3=( 1  2  3 ... 30 31 --)

    vpor        ymm2, ymm2, ymm7                ; ymm2=(-1  0  1 ... 28 29 30)
    vpor        ymm3, ymm3, ymm6                ; ymm3=( 1  2  3 ... 30 31 32)

    vpsrldq     ymm7, ymm4, (SIZEOF_XMMWORD-1)  ; ymm7=(31 -- -- ... -- -- --)

    vpunpckhbw  ymm4, ymm1, ymm0                ; ymm4=( 8  9 10 11 12 13 14 15 24 25 26 27 28 29 30 31)
    vpunpcklbw  ymm5, ymm1, ymm0                ; ymm5=( 0  1  2  3  4  5  6  7 16 17 18 19 20 21 22 23)
    vperm2i128  ymm1, ymm5, ymm4, 0x20          ; ymm1=( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vperm2i128  ymm4, ymm5, ymm4, 0x31          ; ymm4=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpunpckhbw  ymm5, ymm2, ymm0                ; ymm5=( 7  8  9 10 11 12 13 14 23 24 25 26 27 28 29 30)
    vpunpcklbw  ymm6, ymm2, ymm0                ; ymm6=(-1  0  1  2  3  4  5  6 15 16 17 18 19 20 21 22)
    vperm2i128  ymm2, ymm6, ymm5, 0x20          ; ymm2=(-1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14)
    vperm2i128  ymm5, ymm6, ymm5, 0x31          ; ymm5=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

    vpunpckhbw  ymm6, ymm3, ymm0                ; ymm6=( 1  2  3  4  5  6  7  8 17 18 19 20 21 22 23 24)
    vpunpcklbw  ymm0, ymm3, ymm0                ; ymm0=( 9 10 11 12 13 14 15 16 25 26 27 28 29 30 31 32)
    vperm2i128  ymm3, ymm0, ymm6, 0x20          ; ymm3=( 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16)
    vperm2i128  ymm6, ymm0, ymm6, 0x31          ; ymm6=(17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)

    vpxor       ymm0, ymm0, ymm0                ; ymm0=(all 0's)

    vpmullw     ymm1, ymm1, [GOTOFF(ebx,PW_THREE)]
    vpmullw     ymm4, ymm4, [GOTOFF(ebx,PW_THREE)]
    vpaddw      ymm2, ymm2, [GOTOFF(ebx,PW_ONE)]
    vpaddw      ymm5, ymm5, [GOTOFF(ebx,PW_ONE)]
    vpaddw      ymm3, ymm3, [GOTOFF(ebx,PW_TWO)]
    vpaddw      ymm6, ymm6, [GOTOFF(ebx,PW_TWO)]

    vpaddw      ymm2, ymm2, ymm1
    vpaddw      ymm5, ymm5, ymm4
    vpsrlw      ymm2, ymm2, 2                   ; ymm2=OutLE=( 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30)
    vpsrlw      ymm5, ymm5, 2                   ; ymm5=OutHE=(32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62)
    vpaddw      ymm3, ymm3, ymm1
    vpaddw      ymm6, ymm6, ymm4
    vpsrlw      ymm3, ymm3, 2                   ; ymm3=OutLO=( 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31)
    vpsrlw      ymm6, ymm6, 2                   ; ymm6=OutHO=(33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63)

    vpsllw      ymm3, ymm3, BYTE_BIT
    vpsllw      ymm6, ymm6, BYTE_BIT
    vpor        ymm2, ymm2, ymm3                ; ymm2=OutL=( 0  1  2 ... 29 30 31)
    vpor        ymm5, ymm5, ymm6                ; ymm5=OutH=(32 33 34 ... 61 62 63)

    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymm2
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymm5

    sub         eax, byte SIZEOF_YMMWORD
    add         esi, byte 1*SIZEOF_YMMWORD  ; inptr
    add         edi, byte 2*SIZEOF_YMMWORD  ; outptr
    cmp         eax, byte SIZEOF_YMMWORD
    ja          near .columnloop
    test        eax, eax
    jnz         near .columnloop_last

    pop         esi
    pop         edi
    pop         eax

    add         esi, byte SIZEOF_JSAMPROW  ; input_data
    add         edi, byte SIZEOF_JSAMPROW  ; output_data
    dec         ecx                        ; rowctr
    jg          near .rowloop

.return:
    vzeroupper
    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    POPPIC      ebx
    pop         ebp
    ret

; --------------------------------------------------------------------------
;
; Fancy processing for the common case of 2:1 horizontal and 2:1 vertical.
; Again a triangle filter; see comments for h2v1 case, above.
;
; GLOBAL(void)
; jsimd_h2v2_fancy_upsample_avx2(int max_v_samp_factor,
;                                JDIMENSION downsampled_width,
;                                JSAMPARRAY input_data,
;                                JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define downsamp_width(b)   (b) + 12    ; JDIMENSION downsampled_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_YMMWORD
                                        ; ymmword wk[WK_NUM]
%define WK_NUM        4
%define gotptr        wk(0) - SIZEOF_POINTER  ; void *gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_fancy_upsample_avx2)

EXTN(jsimd_h2v2_fancy_upsample_avx2):
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

    mov         edx, eax                ; edx = original ebp
    mov         eax, JDIMENSION [downsamp_width(edx)]  ; colctr
    test        eax, eax
    jz          near .return

    mov         ecx, INT [max_v_samp(edx)]  ; rowctr
    test        ecx, ecx
    jz          near .return

    mov         esi, JSAMPARRAY [input_data(edx)]    ; input_data
    mov         edi, POINTER [output_data_ptr(edx)]
    mov         edi, JSAMPARRAY [edi]                ; output_data
    ALIGNX      16, 7
.rowloop:
    push        eax                     ; colctr
    push        ecx
    push        edi
    push        esi

    mov         ecx, JSAMPROW [esi-1*SIZEOF_JSAMPROW]  ; inptr1(above)
    mov         ebx, JSAMPROW [esi+0*SIZEOF_JSAMPROW]  ; inptr0
    mov         esi, JSAMPROW [esi+1*SIZEOF_JSAMPROW]  ; inptr1(below)
    mov         edx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]  ; outptr0
    mov         edi, JSAMPROW [edi+1*SIZEOF_JSAMPROW]  ; outptr1

    test        eax, SIZEOF_YMMWORD-1
    jz          short .skip
    push        edx
    mov         dl, JSAMPLE [ecx+(eax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [ecx+eax*SIZEOF_JSAMPLE], dl
    mov         dl, JSAMPLE [ebx+(eax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [ebx+eax*SIZEOF_JSAMPLE], dl
    mov         dl, JSAMPLE [esi+(eax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [esi+eax*SIZEOF_JSAMPLE], dl    ; insert a dummy sample
    pop         edx
.skip:
    ; -- process the first column block

    vmovdqu     ymm0, YMMWORD [ebx+0*SIZEOF_YMMWORD]  ; ymm0=row[ 0][0]
    vmovdqu     ymm1, YMMWORD [ecx+0*SIZEOF_YMMWORD]  ; ymm1=row[-1][0]
    vmovdqu     ymm2, YMMWORD [esi+0*SIZEOF_YMMWORD]  ; ymm2=row[+1][0]

    PUSHPIC     ebx
    MOVPIC      ebx, POINTER [gotptr]   ; load GOT address

    vpxor       ymm3, ymm3, ymm3        ; ymm3=(all 0's)

    vpunpckhbw  ymm4, ymm0, ymm3        ; ymm4=row[ 0]( 8  9 10 11 12 13 14 15 24 25 26 27 28 29 30 31)
    vpunpcklbw  ymm5, ymm0, ymm3        ; ymm5=row[ 0]( 0  1  2  3  4  5  6  7 16 17 18 19 20 21 22 23)
    vperm2i128  ymm0, ymm5, ymm4, 0x20  ; ymm0=row[ 0]( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vperm2i128  ymm4, ymm5, ymm4, 0x31  ; ymm4=row[ 0](16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpunpckhbw  ymm5, ymm1, ymm3        ; ymm5=row[-1]( 8  9 10 11 12 13 14 15 24 25 26 27 28 29 30 31)
    vpunpcklbw  ymm6, ymm1, ymm3        ; ymm6=row[-1]( 0  1  2  3  4  5  6  7 16 17 18 19 20 21 22 23)
    vperm2i128  ymm1, ymm6, ymm5, 0x20  ; ymm1=row[-1]( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vperm2i128  ymm5, ymm6, ymm5, 0x31  ; ymm5=row[-1](16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpunpckhbw  ymm6, ymm2, ymm3        ; ymm6=row[+1]( 8  9 10 11 12 13 14 15 24 25 26 27 28 29 30 31)
    vpunpcklbw  ymm3, ymm2, ymm3        ; ymm3=row[+1]( 0  1  2  3  4  5  6  7 16 17 18 19 20 21 22 23)
    vperm2i128  ymm2, ymm3, ymm6, 0x20  ; ymm2=row[+1]( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vperm2i128  ymm6, ymm3, ymm6, 0x31  ; ymm6=row[+1](16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpmullw     ymm0, ymm0, [GOTOFF(ebx,PW_THREE)]
    vpmullw     ymm4, ymm4, [GOTOFF(ebx,PW_THREE)]

    vpcmpeqb    xmm7, xmm7, xmm7
    vpsrldq     xmm7, xmm7, (SIZEOF_XMMWORD-2)  ; (ffff ---- ---- ... ---- ----) LSB is ffff

    vpaddw      ymm1, ymm1, ymm0        ; ymm1=Int0L=( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vpaddw      ymm5, ymm5, ymm4        ; ymm5=Int0H=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
    vpaddw      ymm2, ymm2, ymm0        ; ymm2=Int1L=( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vpaddw      ymm6, ymm6, ymm4        ; ymm6=Int1H=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vmovdqu     YMMWORD [edx+0*SIZEOF_YMMWORD], ymm1  ; temporarily save
    vmovdqu     YMMWORD [edx+1*SIZEOF_YMMWORD], ymm5  ; the intermediate data
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymm2
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymm6

    vpand       ymm1, ymm1, ymm7        ; ymm1=( 0 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --)
    vpand       ymm2, ymm2, ymm7        ; ymm2=( 0 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --)

    vmovdqa     YMMWORD [wk(0)], ymm1
    vmovdqa     YMMWORD [wk(1)], ymm2

    POPPIC      ebx

    add         eax, byte SIZEOF_YMMWORD-1
    and         eax, byte -SIZEOF_YMMWORD
    cmp         eax, byte SIZEOF_YMMWORD
    ja          short .columnloop
    ALIGNX      16, 7

.columnloop_last:
    ; -- process the last column block

    PUSHPIC     ebx
    MOVPIC      ebx, POINTER [gotptr]   ; load GOT address

    vpcmpeqb    xmm1, xmm1, xmm1
    vpslldq     xmm1, xmm1, (SIZEOF_XMMWORD-2)
    vperm2i128  ymm1, ymm1, ymm1, 1             ; (---- ---- ... ---- ---- ffff) MSB is ffff

    vpand       ymm2, ymm1, YMMWORD [edi+1*SIZEOF_YMMWORD]
    vpand       ymm1, ymm1, YMMWORD [edx+1*SIZEOF_YMMWORD]

    vmovdqa     YMMWORD [wk(2)], ymm1          ; ymm1=(-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 31)
    vmovdqa     YMMWORD [wk(3)], ymm2          ; ymm2=(-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 31)

    jmp         near .upsample
    ALIGNX      16, 7

.columnloop:
    ; -- process the next column block

    vmovdqu     ymm0, YMMWORD [ebx+1*SIZEOF_YMMWORD]  ; ymm0=row[ 0][1]
    vmovdqu     ymm1, YMMWORD [ecx+1*SIZEOF_YMMWORD]  ; ymm1=row[-1][1]
    vmovdqu     ymm2, YMMWORD [esi+1*SIZEOF_YMMWORD]  ; ymm2=row[+1][1]

    PUSHPIC     ebx
    MOVPIC      ebx, POINTER [gotptr]   ; load GOT address

    vpxor       ymm3, ymm3, ymm3        ; ymm3=(all 0's)

    vpunpckhbw  ymm4, ymm0, ymm3        ; ymm4=row[ 0]( 8  9 10 11 12 13 14 15 24 25 26 27 28 29 30 31)
    vpunpcklbw  ymm5, ymm0, ymm3        ; ymm5=row[ 0]( 0  1  2  3  4  5  6  7 16 17 18 19 20 21 22 23)
    vperm2i128  ymm0, ymm5, ymm4, 0x20  ; ymm0=row[ 0]( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vperm2i128  ymm4, ymm5, ymm4, 0x31  ; ymm4=row[ 0](16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpunpckhbw  ymm5, ymm1, ymm3        ; ymm5=row[-1]( 8  9 10 11 12 13 14 15 24 25 26 27 28 29 30 31)
    vpunpcklbw  ymm6, ymm1, ymm3        ; ymm6=row[-1]( 0  1  2  3  4  5  6  7 16 17 18 19 20 21 22 23)
    vperm2i128  ymm1, ymm6, ymm5, 0x20  ; ymm1=row[-1]( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vperm2i128  ymm5, ymm6, ymm5, 0x31  ; ymm5=row[-1](16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpunpckhbw  ymm6, ymm2, ymm3        ; ymm6=row[+1]( 8  9 10 11 12 13 14 15 24 25 26 27 28 29 30 31)
    vpunpcklbw  ymm7, ymm2, ymm3        ; ymm7=row[+1]( 0  1  2  3  4  5  6  7 16 17 18 19 20 21 22 23)
    vperm2i128  ymm2, ymm7, ymm6, 0x20  ; ymm2=row[+1]( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vperm2i128  ymm6, ymm7, ymm6, 0x31  ; ymm6=row[+1](16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpmullw     ymm0, ymm0, [GOTOFF(ebx,PW_THREE)]
    vpmullw     ymm4, ymm4, [GOTOFF(ebx,PW_THREE)]

    vpaddw      ymm1, ymm1, ymm0        ; ymm1=Int0L=( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vpaddw      ymm5, ymm5, ymm4        ; ymm5=Int0H=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
    vpaddw      ymm2, ymm2, ymm0        ; ymm2=Int1L=( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vpaddw      ymm6, ymm6, ymm4        ; ymm6=Int1H=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vmovdqu     YMMWORD [edx+2*SIZEOF_YMMWORD], ymm1  ; temporarily save
    vmovdqu     YMMWORD [edx+3*SIZEOF_YMMWORD], ymm5  ; the intermediate data
    vmovdqu     YMMWORD [edi+2*SIZEOF_YMMWORD], ymm2
    vmovdqu     YMMWORD [edi+3*SIZEOF_YMMWORD], ymm6

    vperm2i128  ymm1, ymm3, ymm1, 0x20
    vpslldq     ymm1, ymm1, 14          ; ymm1=(-- -- -- -- -- -- -- -- -- -- -- -- -- -- --  0)
    vperm2i128  ymm2, ymm3, ymm2, 0x20
    vpslldq     ymm2, ymm2, 14          ; ymm2=(-- -- -- -- -- -- -- -- -- -- -- -- -- -- --  0)

    vmovdqa     YMMWORD [wk(2)], ymm1
    vmovdqa     YMMWORD [wk(3)], ymm2

.upsample:
    ; -- process the upper row

    vmovdqu     ymm7, YMMWORD [edx+0*SIZEOF_YMMWORD]  ; ymm7=Int0L=( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vmovdqu     ymm3, YMMWORD [edx+1*SIZEOF_YMMWORD]  ; ymm3=Int0H=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpxor       ymm1, ymm1, ymm1        ; ymm1=(all 0's)

    vperm2i128  ymm0, ymm1, ymm7, 0x03
    vpalignr    ymm0, ymm0, ymm7, 2     ; ymm0=( 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 --)
    vperm2i128  ymm4, ymm1, ymm3, 0x20
    vpslldq     ymm4, ymm4, 14          ; ymm4=(-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 16)

    vperm2i128  ymm5, ymm1, ymm7, 0x03
    vpsrldq     ymm5, ymm5, 14          ; ymm5=(15 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --)
    vperm2i128  ymm6, ymm1, ymm3, 0x20
    vpalignr    ymm6, ymm3, ymm6, 14    ; ymm6=(-- 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

    vpor        ymm0, ymm0, ymm4        ; ymm0=( 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16)
    vpor        ymm5, ymm5, ymm6        ; ymm5=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

    vperm2i128  ymm2, ymm1, ymm3, 0x03
    vpalignr    ymm2, ymm2, ymm3, 2     ; ymm2=(17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --)
    vperm2i128  ymm4, ymm1, ymm3, 0x03
    vpsrldq     ymm4, ymm4, 14          ; ymm4=(31 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --)
    vperm2i128  ymm1, ymm1, ymm7, 0x20
    vpalignr    ymm1, ymm7, ymm1, 14    ; ymm1=(--  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14)

    vpor        ymm1, ymm1, YMMWORD [wk(0)]  ; ymm1=(-1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14)
    vpor        ymm2, ymm2, YMMWORD [wk(2)]  ; ymm2=(17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)

    vmovdqa     YMMWORD [wk(0)], ymm4

    vpmullw     ymm7, ymm7, [GOTOFF(ebx,PW_THREE)]
    vpmullw     ymm3, ymm3, [GOTOFF(ebx,PW_THREE)]
    vpaddw      ymm1, ymm1, [GOTOFF(ebx,PW_EIGHT)]
    vpaddw      ymm5, ymm5, [GOTOFF(ebx,PW_EIGHT)]
    vpaddw      ymm0, ymm0, [GOTOFF(ebx,PW_SEVEN)]
    vpaddw      ymm2, [GOTOFF(ebx,PW_SEVEN)]

    vpaddw      ymm1, ymm1, ymm7
    vpaddw      ymm5, ymm5, ymm3
    vpsrlw      ymm1, ymm1, 4           ; ymm1=Out0LE=( 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30)
    vpsrlw      ymm5, ymm5, 4           ; ymm5=Out0HE=(32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62)
    vpaddw      ymm0, ymm0, ymm7
    vpaddw      ymm2, ymm2, ymm3
    vpsrlw      ymm0, ymm0, 4           ; ymm0=Out0LO=( 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31)
    vpsrlw      ymm2, ymm2, 4           ; ymm2=Out0HO=(33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63)

    vpsllw      ymm0, ymm0, BYTE_BIT
    vpsllw      ymm2, ymm2, BYTE_BIT
    vpor        ymm1, ymm1, ymm0        ; ymm1=Out0L=( 0  1  2 ... 29 30 31)
    vpor        ymm5, ymm5, ymm2        ; ymm5=Out0H=(32 33 34 ... 61 62 63)

    vmovdqu     YMMWORD [edx+0*SIZEOF_YMMWORD], ymm1
    vmovdqu     YMMWORD [edx+1*SIZEOF_YMMWORD], ymm5

    ; -- process the lower row

    vmovdqu     ymm6, YMMWORD [edi+0*SIZEOF_YMMWORD]  ; ymm6=Int1L=( 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15)
    vmovdqu     ymm4, YMMWORD [edi+1*SIZEOF_YMMWORD]  ; ymm4=Int1H=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

    vpxor       ymm1, ymm1, ymm1        ; ymm1=(all 0's)

    vperm2i128  ymm7, ymm1, ymm6, 0x03
    vpalignr    ymm7, ymm7, ymm6, 2     ; ymm7=( 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 --)
    vperm2i128  ymm3, ymm1, ymm4, 0x20
    vpslldq     ymm3, ymm3, 14          ; ymm3=(-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 16)

    vperm2i128  ymm0, ymm1, ymm6, 0x03
    vpsrldq     ymm0, ymm0, 14          ; ymm0=(15 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --)
    vperm2i128  ymm2, ymm1, ymm4, 0x20
    vpalignr    ymm2, ymm4, ymm2, 14    ; ymm2=(-- 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

    vpor        ymm7, ymm7, ymm3        ; ymm7=( 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16)
    vpor        ymm0, ymm0, ymm2        ; ymm0=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

    vperm2i128  ymm5, ymm1, ymm4, 0x03
    vpalignr    ymm5, ymm5, ymm4, 2     ; ymm5=(17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --)
    vperm2i128  ymm3, ymm1, ymm4, 0x03
    vpsrldq     ymm3, ymm3, 14          ; ymm3=(31 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --)
    vperm2i128  ymm1, ymm1, ymm6, 0x20
    vpalignr    ymm1, ymm6, ymm1, 14    ; ymm1=(--  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14)

    vpor        ymm1, ymm1, YMMWORD [wk(1)]  ; ymm1=(-1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14)
    vpor        ymm5, ymm5, YMMWORD [wk(3)]  ; ymm5=(17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)

    vmovdqa     YMMWORD [wk(1)], ymm3

    vpmullw     ymm6, ymm6, [GOTOFF(ebx,PW_THREE)]
    vpmullw     ymm4, ymm4, [GOTOFF(ebx,PW_THREE)]
    vpaddw      ymm1, ymm1, [GOTOFF(ebx,PW_EIGHT)]
    vpaddw      ymm0, ymm0, [GOTOFF(ebx,PW_EIGHT)]
    vpaddw      ymm7, ymm7, [GOTOFF(ebx,PW_SEVEN)]
    vpaddw      ymm5, ymm5, [GOTOFF(ebx,PW_SEVEN)]

    vpaddw      ymm1, ymm1, ymm6
    vpaddw      ymm0, ymm0, ymm4
    vpsrlw      ymm1, ymm1, 4           ; ymm1=Out1LE=( 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30)
    vpsrlw      ymm0, ymm0, 4           ; ymm0=Out1HE=(32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62)
    vpaddw      ymm7, ymm7, ymm6
    vpaddw      ymm5, ymm5, ymm4
    vpsrlw      ymm7, ymm7, 4           ; ymm7=Out1LO=( 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31)
    vpsrlw      ymm5, ymm5, 4           ; ymm5=Out1HO=(33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63)

    vpsllw      ymm7, ymm7, BYTE_BIT
    vpsllw      ymm5, ymm5, BYTE_BIT
    vpor        ymm1, ymm1, ymm7        ; ymm1=Out1L=( 0  1  2 ... 29 30 31)
    vpor        ymm0, ymm0, ymm5        ; ymm0=Out1H=(32 33 34 ... 61 62 63)

    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymm1
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymm0

    POPPIC      ebx

    sub         eax, byte SIZEOF_YMMWORD
    add         ecx, byte 1*SIZEOF_YMMWORD  ; inptr1(above)
    add         ebx, byte 1*SIZEOF_YMMWORD  ; inptr0
    add         esi, byte 1*SIZEOF_YMMWORD  ; inptr1(below)
    add         edx, byte 2*SIZEOF_YMMWORD  ; outptr0
    add         edi, byte 2*SIZEOF_YMMWORD  ; outptr1
    cmp         eax, byte SIZEOF_YMMWORD
    ja          near .columnloop
    test        eax, eax
    jnz         near .columnloop_last

    pop         esi
    pop         edi
    pop         ecx
    pop         eax

    add         esi, byte 1*SIZEOF_JSAMPROW  ; input_data
    add         edi, byte 2*SIZEOF_JSAMPROW  ; output_data
    sub         ecx, byte 2                  ; rowctr
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

; --------------------------------------------------------------------------
;
; Fast processing for the common case of 2:1 horizontal and 1:1 vertical.
; It's still a box filter.
;
; GLOBAL(void)
; jsimd_h2v1_upsample_avx2(int max_v_samp_factor, JDIMENSION output_width,
;                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define output_width(b)     (b) + 12    ; JDIMENSION output_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_upsample_avx2)

EXTN(jsimd_h2v1_upsample_avx2):
    push        ebp
    mov         ebp, esp
;   push        ebx                     ; unused
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         edx, JDIMENSION [output_width(ebp)]
    add         edx, byte (SIZEOF_YMMWORD-1)
    and         edx, -SIZEOF_YMMWORD
    jz          short .return

    mov         ecx, INT [max_v_samp(ebp)]  ; rowctr
    test        ecx, ecx
    jz          short .return

    mov         esi, JSAMPARRAY [input_data(ebp)]    ; input_data
    mov         edi, POINTER [output_data_ptr(ebp)]
    mov         edi, JSAMPARRAY [edi]                ; output_data
    ALIGNX      16, 7
.rowloop:
    push        edi
    push        esi

    mov         esi, JSAMPROW [esi]     ; inptr
    mov         edi, JSAMPROW [edi]     ; outptr
    mov         eax, edx                ; colctr
    ALIGNX      16, 7
.columnloop:

    cmp         eax, byte SIZEOF_YMMWORD
    ja          near .above_16

    vmovdqu     xmm0, XMMWORD [esi+0*SIZEOF_YMMWORD]
    vpunpckhbw  xmm1, xmm0, xmm0
    vpunpcklbw  xmm0, xmm0, xmm0

    vmovdqu     XMMWORD [edi+0*SIZEOF_XMMWORD], xmm0
    vmovdqu     XMMWORD [edi+1*SIZEOF_XMMWORD], xmm1

    jmp         short .nextrow

.above_16:
    vmovdqu     ymm0, YMMWORD [esi+0*SIZEOF_YMMWORD]

    vpermq      ymm0, ymm0, 0xd8
    vpunpckhbw  ymm1, ymm0, ymm0
    vpunpcklbw  ymm0, ymm0, ymm0

    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymm0
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymm1

    sub         eax, byte 2*SIZEOF_YMMWORD
    jz          short .nextrow

    add         esi, byte SIZEOF_YMMWORD    ; inptr
    add         edi, byte 2*SIZEOF_YMMWORD  ; outptr
    jmp         short .columnloop
    ALIGNX      16, 7

.nextrow:
    pop         esi
    pop         edi

    add         esi, byte SIZEOF_JSAMPROW  ; input_data
    add         edi, byte SIZEOF_JSAMPROW  ; output_data
    dec         ecx                        ; rowctr
    jg          short .rowloop

.return:
    vzeroupper
    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
;   pop         ebx                     ; unused
    pop         ebp
    ret

; --------------------------------------------------------------------------
;
; Fast processing for the common case of 2:1 horizontal and 2:1 vertical.
; It's still a box filter.
;
; GLOBAL(void)
; jsimd_h2v2_upsample_avx2(int max_v_samp_factor, JDIMENSION output_width,
;                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define output_width(b)     (b) + 12    ; JDIMENSION output_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_upsample_avx2)

EXTN(jsimd_h2v2_upsample_avx2):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         edx, JDIMENSION [output_width(ebp)]
    add         edx, byte (SIZEOF_YMMWORD-1)
    and         edx, -SIZEOF_YMMWORD
    jz          near .return

    mov         ecx, INT [max_v_samp(ebp)]  ; rowctr
    test        ecx, ecx
    jz          near .return

    mov         esi, JSAMPARRAY [input_data(ebp)]    ; input_data
    mov         edi, POINTER [output_data_ptr(ebp)]
    mov         edi, JSAMPARRAY [edi]                ; output_data
    ALIGNX      16, 7
.rowloop:
    push        edi
    push        esi

    mov         esi, JSAMPROW [esi]                    ; inptr
    mov         ebx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]  ; outptr0
    mov         edi, JSAMPROW [edi+1*SIZEOF_JSAMPROW]  ; outptr1
    mov         eax, edx                               ; colctr
    ALIGNX      16, 7
.columnloop:

    cmp         eax, byte SIZEOF_YMMWORD
    ja          short .above_16

    vmovdqu     xmm0, XMMWORD [esi+0*SIZEOF_XMMWORD]
    vpunpckhbw  xmm1, xmm0, xmm0
    vpunpcklbw  xmm0, xmm0, xmm0

    vmovdqu     XMMWORD [ebx+0*SIZEOF_XMMWORD], xmm0
    vmovdqu     XMMWORD [ebx+1*SIZEOF_XMMWORD], xmm1
    vmovdqu     XMMWORD [edi+0*SIZEOF_XMMWORD], xmm0
    vmovdqu     XMMWORD [edi+1*SIZEOF_XMMWORD], xmm1

    jmp         near .nextrow

.above_16:
    vmovdqu     ymm0, YMMWORD [esi+0*SIZEOF_YMMWORD]

    vpermq      ymm0, ymm0, 0xd8
    vpunpckhbw  ymm1, ymm0, ymm0
    vpunpcklbw  ymm0, ymm0, ymm0

    vmovdqu     YMMWORD [ebx+0*SIZEOF_YMMWORD], ymm0
    vmovdqu     YMMWORD [ebx+1*SIZEOF_YMMWORD], ymm1
    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymm0
    vmovdqu     YMMWORD [edi+1*SIZEOF_YMMWORD], ymm1

    sub         eax, byte 2*SIZEOF_YMMWORD
    jz          short .nextrow

    add         esi, byte SIZEOF_YMMWORD  ; inptr
    add         ebx, 2*SIZEOF_YMMWORD     ; outptr0
    add         edi, 2*SIZEOF_YMMWORD     ; outptr1
    jmp         short .columnloop
    ALIGNX      16, 7

.nextrow:
    pop         esi
    pop         edi

    add         esi, byte 1*SIZEOF_JSAMPROW  ; input_data
    add         edi, byte 2*SIZEOF_JSAMPROW  ; output_data
    sub         ecx, byte 2                  ; rowctr
    jg          near .rowloop

.return:
    vzeroupper
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
