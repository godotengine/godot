;
; jquanti.asm - sample data conversion and quantization (64-bit AVX2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2018, 2024, D. R. Commander.
; Copyright (C) 2016, Matthieu Darbois.
; Copyright (C) 2018, Matthias Räncker.
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
    BITS        64
;
; Load data into workspace, applying unsigned->signed conversion
;
; GLOBAL(void)
; jsimd_convsamp_avx2(JSAMPARRAY sample_data, JDIMENSION start_col,
;                     DCTELEM *workspace);
;

; r10 = JSAMPARRAY sample_data
; r11d = JDIMENSION start_col
; r12 = DCTELEM *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_convsamp_avx2)

EXTN(jsimd_convsamp_avx2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 3

    mov         eax, r11d

    mov         rsip, JSAMPROW [r10+0*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         rdip, JSAMPROW [r10+1*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm0, XMM_MMWORD [rsi+rax*SIZEOF_JSAMPLE]
    pinsrq      xmm0, XMM_MMWORD [rdi+rax*SIZEOF_JSAMPLE], 1

    mov         rsip, JSAMPROW [r10+2*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         rdip, JSAMPROW [r10+3*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm1, XMM_MMWORD [rsi+rax*SIZEOF_JSAMPLE]
    pinsrq      xmm1, XMM_MMWORD [rdi+rax*SIZEOF_JSAMPLE], 1

    mov         rsip, JSAMPROW [r10+4*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         rdip, JSAMPROW [r10+5*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm2, XMM_MMWORD [rsi+rax*SIZEOF_JSAMPLE]
    pinsrq      xmm2, XMM_MMWORD [rdi+rax*SIZEOF_JSAMPLE], 1

    mov         rsip, JSAMPROW [r10+6*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         rdip, JSAMPROW [r10+7*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    movq        xmm3, XMM_MMWORD [rsi+rax*SIZEOF_JSAMPLE]
    pinsrq      xmm3, XMM_MMWORD [rdi+rax*SIZEOF_JSAMPLE], 1

    vpmovzxbw   ymm0, xmm0              ; ymm0=(00 01 02 03 04 05 06 07 10 11 12 13 14 15 16 17)
    vpmovzxbw   ymm1, xmm1              ; ymm1=(20 21 22 23 24 25 26 27 30 31 32 33 34 35 36 37)
    vpmovzxbw   ymm2, xmm2              ; ymm2=(40 41 42 43 44 45 46 47 50 51 52 53 54 55 56 57)
    vpmovzxbw   ymm3, xmm3              ; ymm3=(60 61 62 63 64 65 66 67 70 71 72 73 74 75 76 77)

    vpcmpeqw    ymm7, ymm7, ymm7
    vpsllw      ymm7, ymm7, 7           ; ymm7={0xFF80 0xFF80 0xFF80 0xFF80 ..}

    vpaddw      ymm0, ymm0, ymm7
    vpaddw      ymm1, ymm1, ymm7
    vpaddw      ymm2, ymm2, ymm7
    vpaddw      ymm3, ymm3, ymm7

    vmovdqu     YMMWORD [YMMBLOCK(0,0,r12,SIZEOF_DCTELEM)], ymm0
    vmovdqu     YMMWORD [YMMBLOCK(2,0,r12,SIZEOF_DCTELEM)], ymm1
    vmovdqu     YMMWORD [YMMBLOCK(4,0,r12,SIZEOF_DCTELEM)], ymm2
    vmovdqu     YMMWORD [YMMBLOCK(6,0,r12,SIZEOF_DCTELEM)], ymm3

    vzeroupper
    UNCOLLECT_ARGS 3
    pop         rbp
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

; r10 = JCOEFPTR coef_block
; r11 = DCTELEM *divisors
; r12 = DCTELEM *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_quantize_avx2)

EXTN(jsimd_quantize_avx2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 3

    vmovdqu     ymm4, [YMMBLOCK(0,0,r12,SIZEOF_DCTELEM)]
    vmovdqu     ymm5, [YMMBLOCK(2,0,r12,SIZEOF_DCTELEM)]
    vmovdqu     ymm6, [YMMBLOCK(4,0,r12,SIZEOF_DCTELEM)]
    vmovdqu     ymm7, [YMMBLOCK(6,0,r12,SIZEOF_DCTELEM)]
    vpabsw      ymm0, ymm4
    vpabsw      ymm1, ymm5
    vpabsw      ymm2, ymm6
    vpabsw      ymm3, ymm7

    vpaddw      ymm0, YMMWORD [CORRECTION(0,0,r11)]  ; correction + roundfactor
    vpaddw      ymm1, YMMWORD [CORRECTION(2,0,r11)]
    vpaddw      ymm2, YMMWORD [CORRECTION(4,0,r11)]
    vpaddw      ymm3, YMMWORD [CORRECTION(6,0,r11)]
    vpmulhuw    ymm0, YMMWORD [RECIPROCAL(0,0,r11)]  ; reciprocal
    vpmulhuw    ymm1, YMMWORD [RECIPROCAL(2,0,r11)]
    vpmulhuw    ymm2, YMMWORD [RECIPROCAL(4,0,r11)]
    vpmulhuw    ymm3, YMMWORD [RECIPROCAL(6,0,r11)]
    vpmulhuw    ymm0, YMMWORD [SCALE(0,0,r11)]       ; scale
    vpmulhuw    ymm1, YMMWORD [SCALE(2,0,r11)]
    vpmulhuw    ymm2, YMMWORD [SCALE(4,0,r11)]
    vpmulhuw    ymm3, YMMWORD [SCALE(6,0,r11)]

    vpsignw     ymm0, ymm0, ymm4
    vpsignw     ymm1, ymm1, ymm5
    vpsignw     ymm2, ymm2, ymm6
    vpsignw     ymm3, ymm3, ymm7

    vmovdqu     [YMMBLOCK(0,0,r10,SIZEOF_DCTELEM)], ymm0
    vmovdqu     [YMMBLOCK(2,0,r10,SIZEOF_DCTELEM)], ymm1
    vmovdqu     [YMMBLOCK(4,0,r10,SIZEOF_DCTELEM)], ymm2
    vmovdqu     [YMMBLOCK(6,0,r10,SIZEOF_DCTELEM)], ymm3

    vzeroupper
    UNCOLLECT_ARGS 3
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
