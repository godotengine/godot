;
; jcsample.asm - downsampling (MMX)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
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
; jsimd_h2v1_downsample_mmx(JDIMENSION image_width, int max_v_samp_factor,
;                           JDIMENSION v_samp_factor,
;                           JDIMENSION width_in_blocks, JSAMPARRAY input_data,
;                           JSAMPARRAY output_data);
;

%define img_width(b)    (b) + 8         ; JDIMENSION image_width
%define max_v_samp(b)   (b) + 12        ; int max_v_samp_factor
%define v_samp(b)       (b) + 16        ; JDIMENSION v_samp_factor
%define width_blks(b)   (b) + 20        ; JDIMENSION width_in_blocks
%define input_data(b)   (b) + 24        ; JSAMPARRAY input_data
%define output_data(b)  (b) + 28        ; JSAMPARRAY output_data

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_downsample_mmx)

EXTN(jsimd_h2v1_downsample_mmx):
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
    movd        mm7, edx
    pcmpeqw     mm6, mm6
    punpckldq   mm7, mm7                ; mm7={0, 1, 0, 1}
    psrlw       mm6, BYTE_BIT           ; mm6={0xFF 0x00 0xFF 0x00 ..}

    mov         esi, JSAMPARRAY [input_data(ebp)]   ; input_data
    mov         edi, JSAMPARRAY [output_data(ebp)]  ; output_data
    ALIGNX      16, 7
.rowloop:
    push        ecx
    push        edi
    push        esi

    mov         esi, JSAMPROW [esi]     ; inptr
    mov         edi, JSAMPROW [edi]     ; outptr
    ALIGNX      16, 7
.columnloop:

    movq        mm0, MMWORD [esi+0*SIZEOF_MMWORD]
    movq        mm1, MMWORD [esi+1*SIZEOF_MMWORD]
    movq        mm2, mm0
    movq        mm3, mm1

    pand        mm0, mm6
    psrlw       mm2, BYTE_BIT
    pand        mm1, mm6
    psrlw       mm3, BYTE_BIT

    paddw       mm0, mm2
    paddw       mm1, mm3
    paddw       mm0, mm7
    paddw       mm1, mm7
    psrlw       mm0, 1
    psrlw       mm1, 1

    packuswb    mm0, mm1

    movq        MMWORD [edi+0*SIZEOF_MMWORD], mm0

    add         esi, byte 2*SIZEOF_MMWORD  ; inptr
    add         edi, byte 1*SIZEOF_MMWORD  ; outptr
    sub         ecx, byte SIZEOF_MMWORD    ; outcol
    jnz         short .columnloop

    pop         esi
    pop         edi
    pop         ecx

    add         esi, byte SIZEOF_JSAMPROW  ; input_data
    add         edi, byte SIZEOF_JSAMPROW  ; output_data
    dec         eax                        ; rowctr
    jg          short .rowloop

    emms                                ; empty MMX state

.return:
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
; jsimd_h2v2_downsample_mmx(JDIMENSION image_width, int max_v_samp_factor,
;                           JDIMENSION v_samp_factor,
;                           JDIMENSION width_in_blocks, JSAMPARRAY input_data,
;                           JSAMPARRAY output_data);
;

%define img_width(b)    (b) + 8         ; JDIMENSION image_width
%define max_v_samp(b)   (b) + 12        ; int max_v_samp_factor
%define v_samp(b)       (b) + 16        ; JDIMENSION v_samp_factor
%define width_blks(b)   (b) + 20        ; JDIMENSION width_in_blocks
%define input_data(b)   (b) + 24        ; JSAMPARRAY input_data
%define output_data(b)  (b) + 28        ; JSAMPARRAY output_data

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_downsample_mmx)

EXTN(jsimd_h2v2_downsample_mmx):
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
    movd        mm7, edx
    pcmpeqw     mm6, mm6
    punpckldq   mm7, mm7                ; mm7={1, 2, 1, 2}
    psrlw       mm6, BYTE_BIT           ; mm6={0xFF 0x00 0xFF 0x00 ..}

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
    ALIGNX      16, 7
.columnloop:

    movq        mm0, MMWORD [edx+0*SIZEOF_MMWORD]
    movq        mm1, MMWORD [esi+0*SIZEOF_MMWORD]
    movq        mm2, MMWORD [edx+1*SIZEOF_MMWORD]
    movq        mm3, MMWORD [esi+1*SIZEOF_MMWORD]

    movq        mm4, mm0
    movq        mm5, mm1
    pand        mm0, mm6
    psrlw       mm4, BYTE_BIT
    pand        mm1, mm6
    psrlw       mm5, BYTE_BIT
    paddw       mm0, mm4
    paddw       mm1, mm5

    movq        mm4, mm2
    movq        mm5, mm3
    pand        mm2, mm6
    psrlw       mm4, BYTE_BIT
    pand        mm3, mm6
    psrlw       mm5, BYTE_BIT
    paddw       mm2, mm4
    paddw       mm3, mm5

    paddw       mm0, mm1
    paddw       mm2, mm3
    paddw       mm0, mm7
    paddw       mm2, mm7
    psrlw       mm0, 2
    psrlw       mm2, 2

    packuswb    mm0, mm2

    movq        MMWORD [edi+0*SIZEOF_MMWORD], mm0

    add         edx, byte 2*SIZEOF_MMWORD  ; inptr0
    add         esi, byte 2*SIZEOF_MMWORD  ; inptr1
    add         edi, byte 1*SIZEOF_MMWORD  ; outptr
    sub         ecx, byte SIZEOF_MMWORD    ; outcol
    jnz         near .columnloop

    pop         esi
    pop         edi
    pop         ecx

    add         esi, byte 2*SIZEOF_JSAMPROW  ; input_data
    add         edi, byte 1*SIZEOF_JSAMPROW  ; output_data
    dec         eax                          ; rowctr
    jg          near .rowloop

    emms                                ; empty MMX state

.return:
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
