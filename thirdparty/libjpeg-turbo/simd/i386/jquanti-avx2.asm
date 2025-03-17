;
; jquanti.asm - sample data conversion and quantization (AVX2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2016, 2018, 2024, D. R. Commander.
; Copyright (C) 2016, Matthieu Darbois.
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
; jsimd_convsamp_avx2(JSAMPARRAY sample_data, JDIMENSION start_col,
;                     DCTELEM *workspace);
;

%define sample_data  ebp + 8            ; JSAMPARRAY sample_data
%define start_col    ebp + 12           ; JDIMENSION start_col
%define workspace    ebp + 16           ; DCTELEM *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_convsamp_avx2)

EXTN(jsimd_convsamp_avx2):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         esi, JSAMPARRAY [sample_data]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [start_col]
    mov         edi, POINTER [workspace]       ; (DCTELEM *)

    mov         ebx, JSAMPROW [esi+0*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+1*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm0, XMM_MMWORD [ebx+eax*SIZEOF_JSAMPLE]
    movq        xmm1, XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE]

    mov         ebx, JSAMPROW [esi+2*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+3*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm2, XMM_MMWORD [ebx+eax*SIZEOF_JSAMPLE]
    movq        xmm3, XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE]

    mov         ebx, JSAMPROW [esi+4*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+5*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm4, XMM_MMWORD [ebx+eax*SIZEOF_JSAMPLE]
    movq        xmm5, XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE]

    mov         ebx, JSAMPROW [esi+6*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+7*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm6, XMM_MMWORD [ebx+eax*SIZEOF_JSAMPLE]
    movq        xmm7, XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE]

    vinserti128 ymm0, ymm0, xmm1, 1
    vinserti128 ymm2, ymm2, xmm3, 1
    vinserti128 ymm4, ymm4, xmm5, 1
    vinserti128 ymm6, ymm6, xmm7, 1

    vpxor       ymm1, ymm1, ymm1        ; ymm1=(all 0's)
    vpunpcklbw  ymm0, ymm0, ymm1
    vpunpcklbw  ymm2, ymm2, ymm1
    vpunpcklbw  ymm4, ymm4, ymm1
    vpunpcklbw  ymm6, ymm6, ymm1

    vpcmpeqw    ymm7, ymm7, ymm7
    vpsllw      ymm7, ymm7, 7           ; ymm7={0xFF80 0xFF80 0xFF80 0xFF80 ..}

    vpaddw      ymm0, ymm0, ymm7
    vpaddw      ymm2, ymm2, ymm7
    vpaddw      ymm4, ymm4, ymm7
    vpaddw      ymm6, ymm6, ymm7

    vmovdqu     YMMWORD [YMMBLOCK(0,0,edi,SIZEOF_DCTELEM)], ymm0
    vmovdqu     YMMWORD [YMMBLOCK(2,0,edi,SIZEOF_DCTELEM)], ymm2
    vmovdqu     YMMWORD [YMMBLOCK(4,0,edi,SIZEOF_DCTELEM)], ymm4
    vmovdqu     YMMWORD [YMMBLOCK(6,0,edi,SIZEOF_DCTELEM)], ymm6

    vzeroupper
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
; jsimd_quantize_avx2(JCOEFPTR coef_block, DCTELEM *divisors,
;                     DCTELEM *workspace);
;

%define RECIPROCAL(m, n, b) \
  YMMBLOCK(DCTSIZE * 0 + (m), (n), (b), SIZEOF_DCTELEM)
%define CORRECTION(m, n, b) \
  YMMBLOCK(DCTSIZE * 1 + (m), (n), (b), SIZEOF_DCTELEM)
%define SCALE(m, n, b) \
  YMMBLOCK(DCTSIZE * 2 + (m), (n), (b), SIZEOF_DCTELEM)

%define coef_block  ebp + 8             ; JCOEFPTR coef_block
%define divisors    ebp + 12            ; DCTELEM *divisors
%define workspace   ebp + 16            ; DCTELEM *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_quantize_avx2)

EXTN(jsimd_quantize_avx2):
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

    vmovdqu     ymm4, [YMMBLOCK(0,0,esi,SIZEOF_DCTELEM)]
    vmovdqu     ymm5, [YMMBLOCK(2,0,esi,SIZEOF_DCTELEM)]
    vmovdqu     ymm6, [YMMBLOCK(4,0,esi,SIZEOF_DCTELEM)]
    vmovdqu     ymm7, [YMMBLOCK(6,0,esi,SIZEOF_DCTELEM)]
    vpabsw      ymm0, ymm4
    vpabsw      ymm1, ymm5
    vpabsw      ymm2, ymm6
    vpabsw      ymm3, ymm7

    vpaddw      ymm0, YMMWORD [CORRECTION(0,0,edx)]  ; correction + roundfactor
    vpaddw      ymm1, YMMWORD [CORRECTION(2,0,edx)]
    vpaddw      ymm2, YMMWORD [CORRECTION(4,0,edx)]
    vpaddw      ymm3, YMMWORD [CORRECTION(6,0,edx)]
    vpmulhuw    ymm0, YMMWORD [RECIPROCAL(0,0,edx)]  ; reciprocal
    vpmulhuw    ymm1, YMMWORD [RECIPROCAL(2,0,edx)]
    vpmulhuw    ymm2, YMMWORD [RECIPROCAL(4,0,edx)]
    vpmulhuw    ymm3, YMMWORD [RECIPROCAL(6,0,edx)]
    vpmulhuw    ymm0, YMMWORD [SCALE(0,0,edx)]       ; scale
    vpmulhuw    ymm1, YMMWORD [SCALE(2,0,edx)]
    vpmulhuw    ymm2, YMMWORD [SCALE(4,0,edx)]
    vpmulhuw    ymm3, YMMWORD [SCALE(6,0,edx)]

    vpsignw     ymm0, ymm0, ymm4
    vpsignw     ymm1, ymm1, ymm5
    vpsignw     ymm2, ymm2, ymm6
    vpsignw     ymm3, ymm3, ymm7

    vmovdqu     [YMMBLOCK(0,0,edi,SIZEOF_DCTELEM)], ymm0
    vmovdqu     [YMMBLOCK(2,0,edi,SIZEOF_DCTELEM)], ymm1
    vmovdqu     [YMMBLOCK(4,0,edi,SIZEOF_DCTELEM)], ymm2
    vmovdqu     [YMMBLOCK(6,0,edi,SIZEOF_DCTELEM)], ymm3

    vzeroupper
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
