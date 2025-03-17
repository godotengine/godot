;
; jcsample.asm - downsampling (AVX2)
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
    SECTION     SEG_TEXT
    BITS        32
;
; Downsample pixel values of a single component.
; This version handles the common case of 2:1 horizontal and 1:1 vertical,
; without smoothing.
;
; GLOBAL(void)
; jsimd_h2v1_downsample_avx2(JDIMENSION image_width, int max_v_samp_factor,
;                            JDIMENSION v_samp_factor,
;                            JDIMENSION width_in_blocks, JSAMPARRAY input_data,
;                            JSAMPARRAY output_data);
;

%define img_width(b)    (b) + 8         ; JDIMENSION image_width
%define max_v_samp(b)   (b) + 12        ; int max_v_samp_factor
%define v_samp(b)       (b) + 16        ; JDIMENSION v_samp_factor
%define width_blks(b)   (b) + 20        ; JDIMENSION width_in_blocks
%define input_data(b)   (b) + 24        ; JSAMPARRAY input_data
%define output_data(b)  (b) + 28        ; JSAMPARRAY output_data

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_downsample_avx2)

EXTN(jsimd_h2v1_downsample_avx2):
    push        ebp
    mov         ebp, esp
;   push        ebx                     ; unused
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         ecx, JDIMENSION [width_blks(ebp)]
    shl         ecx, 3                  ; imul ecx,DCTSIZE (ecx = output_cols)
    jz          near .return

    mov         edx, JDIMENSION [img_width(ebp)]

    ; -- expand_right_edge

    push        ecx
    shl         ecx, 1                  ; output_cols * 2
    sub         ecx, edx
    jle         short .expand_end

    mov         eax, INT [max_v_samp(ebp)]
    test        eax, eax
    jle         short .expand_end

    cld
    mov         esi, JSAMPARRAY [input_data(ebp)]  ; input_data
    ALIGNX      16, 7
.expandloop:
    push        eax
    push        ecx

    mov         edi, JSAMPROW [esi]
    add         edi, edx
    mov         al, JSAMPLE [edi-1]

    rep stosb

    pop         ecx
    pop         eax

    add         esi, byte SIZEOF_JSAMPROW
    dec         eax
    jg          short .expandloop

.expand_end:
    pop         ecx                     ; output_cols

    ; -- h2v1_downsample

    mov         eax, JDIMENSION [v_samp(ebp)]  ; rowctr
    test        eax, eax
    jle         near .return

    mov         edx, 0x00010000         ; bias pattern
    vmovd       xmm7, edx
    vpshufd     xmm7, xmm7, 0x00        ; xmm7={0, 1, 0, 1, 0, 1, 0, 1}
    vperm2i128  ymm7, ymm7, ymm7, 0     ; ymm7={xmm7, xmm7}
    vpcmpeqw    ymm6, ymm6, ymm6
    vpsrlw      ymm6, ymm6, BYTE_BIT    ; ymm6={0xFF 0x00 0xFF 0x00 ..}

    mov         esi, JSAMPARRAY [input_data(ebp)]   ; input_data
    mov         edi, JSAMPARRAY [output_data(ebp)]  ; output_data
    ALIGNX      16, 7
.rowloop:
    push        ecx
    push        edi
    push        esi

    mov         esi, JSAMPROW [esi]     ; inptr
    mov         edi, JSAMPROW [edi]     ; outptr

    cmp         ecx, byte SIZEOF_YMMWORD
    jae         short .columnloop
    ALIGNX      16, 7

.columnloop_r24:
    ; ecx can possibly be 8, 16, 24
    cmp         ecx, 24
    jne         .columnloop_r16
    vmovdqu     ymm0, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     xmm1, XMMWORD [esi+1*SIZEOF_YMMWORD]
    mov         ecx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r16:
    cmp         ecx, 16
    jne         .columnloop_r8
    vmovdqu     ymm0, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vpxor       ymm1, ymm1, ymm1
    mov         ecx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r8:
    vmovdqu     xmm0, XMMWORD[esi+0*SIZEOF_YMMWORD]
    vpxor       ymm1, ymm1, ymm1
    mov         ecx, SIZEOF_YMMWORD
    jmp         short .downsample
    ALIGNX      16, 7

.columnloop:
    vmovdqu     ymm0, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [esi+1*SIZEOF_YMMWORD]

.downsample:
    vpsrlw      ymm2, ymm0, BYTE_BIT
    vpand       ymm0, ymm0, ymm6
    vpsrlw      ymm3, ymm1, BYTE_BIT
    vpand       ymm1, ymm1, ymm6

    vpaddw      ymm0, ymm0, ymm2
    vpaddw      ymm1, ymm1, ymm3
    vpaddw      ymm0, ymm0, ymm7
    vpaddw      ymm1, ymm1, ymm7
    vpsrlw      ymm0, ymm0, 1
    vpsrlw      ymm1, ymm1, 1

    vpackuswb   ymm0, ymm0, ymm1
    vpermq      ymm0, ymm0, 0xd8

    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymm0

    sub         ecx, byte SIZEOF_YMMWORD    ; outcol
    add         esi, byte 2*SIZEOF_YMMWORD  ; inptr
    add         edi, byte 1*SIZEOF_YMMWORD  ; outptr
    cmp         ecx, byte SIZEOF_YMMWORD
    jae         short .columnloop
    test        ecx, ecx
    jnz         near .columnloop_r24

    pop         esi
    pop         edi
    pop         ecx

    add         esi, byte SIZEOF_JSAMPROW  ; input_data
    add         edi, byte SIZEOF_JSAMPROW  ; output_data
    dec         eax                        ; rowctr
    jg          near .rowloop

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
; Downsample pixel values of a single component.
; This version handles the standard case of 2:1 horizontal and 2:1 vertical,
; without smoothing.
;
; GLOBAL(void)
; jsimd_h2v2_downsample_avx2(JDIMENSION image_width, int max_v_samp_factor,
;                            JDIMENSION v_samp_factor,
;                            JDIMENSION width_in_blocks, JSAMPARRAY input_data,
;                            JSAMPARRAY output_data);
;

%define img_width(b)    (b) + 8         ; JDIMENSION image_width
%define max_v_samp(b)   (b) + 12        ; int max_v_samp_factor
%define v_samp(b)       (b) + 16        ; JDIMENSION v_samp_factor
%define width_blks(b)   (b) + 20        ; JDIMENSION width_in_blocks
%define input_data(b)   (b) + 24        ; JSAMPARRAY input_data
%define output_data(b)  (b) + 28        ; JSAMPARRAY output_data

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_downsample_avx2)

EXTN(jsimd_h2v2_downsample_avx2):
    push        ebp
    mov         ebp, esp
;   push        ebx                     ; unused
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         ecx, JDIMENSION [width_blks(ebp)]
    shl         ecx, 3                  ; imul ecx,DCTSIZE (ecx = output_cols)
    jz          near .return

    mov         edx, JDIMENSION [img_width(ebp)]

    ; -- expand_right_edge

    push        ecx
    shl         ecx, 1                  ; output_cols * 2
    sub         ecx, edx
    jle         short .expand_end

    mov         eax, INT [max_v_samp(ebp)]
    test        eax, eax
    jle         short .expand_end

    cld
    mov         esi, JSAMPARRAY [input_data(ebp)]  ; input_data
    ALIGNX      16, 7
.expandloop:
    push        eax
    push        ecx

    mov         edi, JSAMPROW [esi]
    add         edi, edx
    mov         al, JSAMPLE [edi-1]

    rep stosb

    pop         ecx
    pop         eax

    add         esi, byte SIZEOF_JSAMPROW
    dec         eax
    jg          short .expandloop

.expand_end:
    pop         ecx                     ; output_cols

    ; -- h2v2_downsample

    mov         eax, JDIMENSION [v_samp(ebp)]  ; rowctr
    test        eax, eax
    jle         near .return

    mov         edx, 0x00020001         ; bias pattern
    vmovd       xmm7, edx
    vpcmpeqw    ymm6, ymm6, ymm6
    vpshufd     xmm7, xmm7, 0x00        ; ymm7={1, 2, 1, 2, 1, 2, 1, 2}
    vperm2i128  ymm7, ymm7, ymm7, 0
    vpsrlw      ymm6, ymm6, BYTE_BIT    ; ymm6={0xFF 0x00 0xFF 0x00 ..}

    mov         esi, JSAMPARRAY [input_data(ebp)]   ; input_data
    mov         edi, JSAMPARRAY [output_data(ebp)]  ; output_data
    ALIGNX      16, 7
.rowloop:
    push        ecx
    push        edi
    push        esi

    mov         edx, JSAMPROW [esi+0*SIZEOF_JSAMPROW]  ; inptr0
    mov         esi, JSAMPROW [esi+1*SIZEOF_JSAMPROW]  ; inptr1
    mov         edi, JSAMPROW [edi]                    ; outptr

    cmp         ecx, byte SIZEOF_YMMWORD
    jae         short .columnloop
    ALIGNX      16, 7

.columnloop_r24:
    cmp         ecx, 24
    jne         .columnloop_r16
    vmovdqu     ymm0, YMMWORD [edx+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     xmm2, XMMWORD [edx+1*SIZEOF_YMMWORD]
    vmovdqu     xmm3, XMMWORD [esi+1*SIZEOF_YMMWORD]
    mov         ecx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r16:
    cmp         ecx, 16
    jne         .columnloop_r8
    vmovdqu     ymm0, YMMWORD [edx+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vpxor       ymm2, ymm2, ymm2
    vpxor       ymm3, ymm3, ymm3
    mov         ecx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r8:
    vmovdqu     xmm0, XMMWORD [edx+0*SIZEOF_XMMWORD]
    vmovdqu     xmm1, XMMWORD [esi+0*SIZEOF_XMMWORD]
    vpxor       ymm2, ymm2, ymm2
    vpxor       ymm3, ymm3, ymm3
    mov         ecx, SIZEOF_YMMWORD
    jmp         short .downsample
    ALIGNX      16, 7

.columnloop:
    vmovdqu     ymm0, YMMWORD [edx+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [esi+0*SIZEOF_YMMWORD]
    vmovdqu     ymm2, YMMWORD [edx+1*SIZEOF_YMMWORD]
    vmovdqu     ymm3, YMMWORD [esi+1*SIZEOF_YMMWORD]

.downsample:
    vpand       ymm4, ymm0, ymm6
    vpsrlw      ymm0, ymm0, BYTE_BIT
    vpand       ymm5, ymm1, ymm6
    vpsrlw      ymm1, ymm1, BYTE_BIT
    vpaddw      ymm0, ymm0, ymm4
    vpaddw      ymm1, ymm1, ymm5

    vpand       ymm4, ymm2, ymm6
    vpsrlw      ymm2, ymm2, BYTE_BIT
    vpand       ymm5, ymm3, ymm6
    vpsrlw      ymm3, ymm3, BYTE_BIT
    vpaddw      ymm2, ymm2, ymm4
    vpaddw      ymm3, ymm3, ymm5

    vpaddw      ymm0, ymm0, ymm1
    vpaddw      ymm2, ymm2, ymm3
    vpaddw      ymm0, ymm0, ymm7
    vpaddw      ymm2, ymm2, ymm7
    vpsrlw      ymm0, ymm0, 2
    vpsrlw      ymm2, ymm2, 2

    vpackuswb   ymm0, ymm0, ymm2
    vpermq      ymm0, ymm0, 0xd8

    vmovdqu     YMMWORD [edi+0*SIZEOF_YMMWORD], ymm0

    sub         ecx, byte SIZEOF_YMMWORD    ; outcol
    add         edx, byte 2*SIZEOF_YMMWORD  ; inptr0
    add         esi, byte 2*SIZEOF_YMMWORD  ; inptr1
    add         edi, byte 1*SIZEOF_YMMWORD  ; outptr
    cmp         ecx, byte SIZEOF_YMMWORD
    jae         near .columnloop
    test        ecx, ecx
    jnz         near .columnloop_r24

    pop         esi
    pop         edi
    pop         ecx

    add         esi, byte 2*SIZEOF_JSAMPROW  ; input_data
    add         edi, byte 1*SIZEOF_JSAMPROW  ; output_data
    dec         eax                          ; rowctr
    jg          near .rowloop

.return:
    vzeroupper
    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
;   pop         ebx                     ; unused
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
