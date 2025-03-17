;
; jquanti.asm - sample data conversion and quantization (SSE2)
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
%include "jdct.inc"

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32
;
; Load data into workspace, applying unsigned->signed conversion
;
; GLOBAL(void)
; jsimd_convsamp_sse2(JSAMPARRAY sample_data, JDIMENSION start_col,
;                     DCTELEM *workspace);
;

%define sample_data  ebp + 8            ; JSAMPARRAY sample_data
%define start_col    ebp + 12           ; JDIMENSION start_col
%define workspace    ebp + 16           ; DCTELEM *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_convsamp_sse2)

EXTN(jsimd_convsamp_sse2):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    pxor        xmm6, xmm6              ; xmm6=(all 0's)
    pcmpeqw     xmm7, xmm7
    psllw       xmm7, 7                 ; xmm7={0xFF80 0xFF80 0xFF80 0xFF80 ..}

    mov         esi, JSAMPARRAY [sample_data]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [start_col]
    mov         edi, POINTER [workspace]       ; (DCTELEM *)
    mov         ecx, DCTSIZE/4
    ALIGNX      16, 7
.convloop:
    mov         ebx, JSAMPROW [esi+0*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+1*SIZEOF_JSAMPROW]  ; (JSAMPLE *)

    movq        xmm0, XMM_MMWORD [ebx+eax*SIZEOF_JSAMPLE]  ; xmm0=(01234567)
    movq        xmm1, XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE]  ; xmm1=(89ABCDEF)

    mov         ebx, JSAMPROW [esi+2*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+3*SIZEOF_JSAMPROW]  ; (JSAMPLE *)

    movq        xmm2, XMM_MMWORD [ebx+eax*SIZEOF_JSAMPLE]  ; xmm2=(GHIJKLMN)
    movq        xmm3, XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE]  ; xmm3=(OPQRSTUV)

    punpcklbw   xmm0, xmm6              ; xmm0=(01234567)
    punpcklbw   xmm1, xmm6              ; xmm1=(89ABCDEF)
    paddw       xmm0, xmm7
    paddw       xmm1, xmm7
    punpcklbw   xmm2, xmm6              ; xmm2=(GHIJKLMN)
    punpcklbw   xmm3, xmm6              ; xmm3=(OPQRSTUV)
    paddw       xmm2, xmm7
    paddw       xmm3, xmm7

    movdqa      XMMWORD [XMMBLOCK(0,0,edi,SIZEOF_DCTELEM)], xmm0
    movdqa      XMMWORD [XMMBLOCK(1,0,edi,SIZEOF_DCTELEM)], xmm1
    movdqa      XMMWORD [XMMBLOCK(2,0,edi,SIZEOF_DCTELEM)], xmm2
    movdqa      XMMWORD [XMMBLOCK(3,0,edi,SIZEOF_DCTELEM)], xmm3

    add         esi, byte 4*SIZEOF_JSAMPROW
    add         edi, byte 4*DCTSIZE*SIZEOF_DCTELEM
    dec         ecx
    jnz         short .convloop

    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; need not be preserved
    pop         ebx
    pop         ebp
    ret

; --------------------------------------------------------------------------
;
; Quantize/descale the coefficients, and store into coef_block
;
; This implementation is based on an algorithm described in
;   "Optimizing subroutines in assembly language:
;   An optimization guide for x86 platforms" (https://agner.org/optimize).
;
; GLOBAL(void)
; jsimd_quantize_sse2(JCOEFPTR coef_block, DCTELEM *divisors,
;                     DCTELEM *workspace);
;

%define RECIPROCAL(m, n, b) \
  XMMBLOCK(DCTSIZE * 0 + (m), (n), (b), SIZEOF_DCTELEM)
%define CORRECTION(m, n, b) \
  XMMBLOCK(DCTSIZE * 1 + (m), (n), (b), SIZEOF_DCTELEM)
%define SCALE(m, n, b) \
  XMMBLOCK(DCTSIZE * 2 + (m), (n), (b), SIZEOF_DCTELEM)

%define coef_block  ebp + 8             ; JCOEFPTR coef_block
%define divisors    ebp + 12            ; DCTELEM *divisors
%define workspace   ebp + 16            ; DCTELEM *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_quantize_sse2)

EXTN(jsimd_quantize_sse2):
    push        ebp
    mov         ebp, esp
;   push        ebx                     ; unused
;   push        ecx                     ; unused
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         esi, POINTER [workspace]
    mov         edx, POINTER [divisors]
    mov         edi, JCOEFPTR [coef_block]
    mov         eax, DCTSIZE2/32
    ALIGNX      16, 7
.quantloop:
    movdqa      xmm4, XMMWORD [XMMBLOCK(0,0,esi,SIZEOF_DCTELEM)]
    movdqa      xmm5, XMMWORD [XMMBLOCK(1,0,esi,SIZEOF_DCTELEM)]
    movdqa      xmm6, XMMWORD [XMMBLOCK(2,0,esi,SIZEOF_DCTELEM)]
    movdqa      xmm7, XMMWORD [XMMBLOCK(3,0,esi,SIZEOF_DCTELEM)]
    movdqa      xmm0, xmm4
    movdqa      xmm1, xmm5
    movdqa      xmm2, xmm6
    movdqa      xmm3, xmm7
    psraw       xmm4, (WORD_BIT-1)
    psraw       xmm5, (WORD_BIT-1)
    psraw       xmm6, (WORD_BIT-1)
    psraw       xmm7, (WORD_BIT-1)
    pxor        xmm0, xmm4
    pxor        xmm1, xmm5
    pxor        xmm2, xmm6
    pxor        xmm3, xmm7
    psubw       xmm0, xmm4              ; if (xmm0 < 0) xmm0 = -xmm0;
    psubw       xmm1, xmm5              ; if (xmm1 < 0) xmm1 = -xmm1;
    psubw       xmm2, xmm6              ; if (xmm2 < 0) xmm2 = -xmm2;
    psubw       xmm3, xmm7              ; if (xmm3 < 0) xmm3 = -xmm3;

    paddw       xmm0, XMMWORD [CORRECTION(0,0,edx)]  ; correction + roundfactor
    paddw       xmm1, XMMWORD [CORRECTION(1,0,edx)]
    paddw       xmm2, XMMWORD [CORRECTION(2,0,edx)]
    paddw       xmm3, XMMWORD [CORRECTION(3,0,edx)]
    pmulhuw     xmm0, XMMWORD [RECIPROCAL(0,0,edx)]  ; reciprocal
    pmulhuw     xmm1, XMMWORD [RECIPROCAL(1,0,edx)]
    pmulhuw     xmm2, XMMWORD [RECIPROCAL(2,0,edx)]
    pmulhuw     xmm3, XMMWORD [RECIPROCAL(3,0,edx)]
    pmulhuw     xmm0, XMMWORD [SCALE(0,0,edx)]       ; scale
    pmulhuw     xmm1, XMMWORD [SCALE(1,0,edx)]
    pmulhuw     xmm2, XMMWORD [SCALE(2,0,edx)]
    pmulhuw     xmm3, XMMWORD [SCALE(3,0,edx)]

    pxor        xmm0, xmm4
    pxor        xmm1, xmm5
    pxor        xmm2, xmm6
    pxor        xmm3, xmm7
    psubw       xmm0, xmm4
    psubw       xmm1, xmm5
    psubw       xmm2, xmm6
    psubw       xmm3, xmm7
    movdqa      XMMWORD [XMMBLOCK(0,0,edi,SIZEOF_DCTELEM)], xmm0
    movdqa      XMMWORD [XMMBLOCK(1,0,edi,SIZEOF_DCTELEM)], xmm1
    movdqa      XMMWORD [XMMBLOCK(2,0,edi,SIZEOF_DCTELEM)], xmm2
    movdqa      XMMWORD [XMMBLOCK(3,0,edi,SIZEOF_DCTELEM)], xmm3

    add         esi, byte 32*SIZEOF_DCTELEM
    add         edx, byte 32*SIZEOF_DCTELEM
    add         edi, byte 32*SIZEOF_JCOEF
    dec         eax
    jnz         near .quantloop

    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
;   pop         ecx                     ; unused
;   pop         ebx                     ; unused
    pop         ebp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
