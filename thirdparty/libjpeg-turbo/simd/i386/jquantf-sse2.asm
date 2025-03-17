;
; jquantf.asm - sample data conversion and quantization (SSE & SSE2)
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
; jsimd_convsamp_float_sse2(JSAMPARRAY sample_data, JDIMENSION start_col,
;                           FAST_FLOAT *workspace);
;

%define sample_data  ebp + 8            ; JSAMPARRAY sample_data
%define start_col    ebp + 12           ; JDIMENSION start_col
%define workspace    ebp + 16           ; FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_convsamp_float_sse2)

EXTN(jsimd_convsamp_float_sse2):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    pcmpeqw     xmm7, xmm7
    psllw       xmm7, 7
    packsswb    xmm7, xmm7              ; xmm7 = PB_CENTERJSAMPLE (0x808080..)

    mov         esi, JSAMPARRAY [sample_data]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [start_col]
    mov         edi, POINTER [workspace]       ; (DCTELEM *)
    mov         ecx, DCTSIZE/2
    ALIGNX      16, 7
.convloop:
    mov         ebx, JSAMPROW [esi+0*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+1*SIZEOF_JSAMPROW]  ; (JSAMPLE *)

    movq        xmm0, XMM_MMWORD [ebx+eax*SIZEOF_JSAMPLE]
    movq        xmm1, XMM_MMWORD [edx+eax*SIZEOF_JSAMPLE]

    psubb       xmm0, xmm7              ; xmm0=(01234567)
    psubb       xmm1, xmm7              ; xmm1=(89ABCDEF)

    punpcklbw   xmm0, xmm0              ; xmm0=(*0*1*2*3*4*5*6*7)
    punpcklbw   xmm1, xmm1              ; xmm1=(*8*9*A*B*C*D*E*F)

    punpcklwd   xmm2, xmm0              ; xmm2=(***0***1***2***3)
    punpckhwd   xmm0, xmm0              ; xmm0=(***4***5***6***7)
    punpcklwd   xmm3, xmm1              ; xmm3=(***8***9***A***B)
    punpckhwd   xmm1, xmm1              ; xmm1=(***C***D***E***F)

    psrad       xmm2, (DWORD_BIT-BYTE_BIT)  ; xmm2=(0123)
    psrad       xmm0, (DWORD_BIT-BYTE_BIT)  ; xmm0=(4567)
    cvtdq2ps    xmm2, xmm2                  ; xmm2=(0123)
    cvtdq2ps    xmm0, xmm0                  ; xmm0=(4567)
    psrad       xmm3, (DWORD_BIT-BYTE_BIT)  ; xmm3=(89AB)
    psrad       xmm1, (DWORD_BIT-BYTE_BIT)  ; xmm1=(CDEF)
    cvtdq2ps    xmm3, xmm3                  ; xmm3=(89AB)
    cvtdq2ps    xmm1, xmm1                  ; xmm1=(CDEF)

    movaps      XMMWORD [XMMBLOCK(0,0,edi,SIZEOF_FAST_FLOAT)], xmm2
    movaps      XMMWORD [XMMBLOCK(0,1,edi,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(1,0,edi,SIZEOF_FAST_FLOAT)], xmm3
    movaps      XMMWORD [XMMBLOCK(1,1,edi,SIZEOF_FAST_FLOAT)], xmm1

    add         esi, byte 2*SIZEOF_JSAMPROW
    add         edi, byte 2*DCTSIZE*SIZEOF_FAST_FLOAT
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
; GLOBAL(void)
; jsimd_quantize_float_sse2(JCOEFPTR coef_block, FAST_FLOAT *divisors,
;                           FAST_FLOAT *workspace);
;

%define coef_block  ebp + 8             ; JCOEFPTR coef_block
%define divisors    ebp + 12            ; FAST_FLOAT *divisors
%define workspace   ebp + 16            ; FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_quantize_float_sse2)

EXTN(jsimd_quantize_float_sse2):
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
    mov         eax, DCTSIZE2/16
    ALIGNX      16, 7
.quantloop:
    movaps      xmm0, XMMWORD [XMMBLOCK(0,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(0,1,esi,SIZEOF_FAST_FLOAT)]
    mulps       xmm0, XMMWORD [XMMBLOCK(0,0,edx,SIZEOF_FAST_FLOAT)]
    mulps       xmm1, XMMWORD [XMMBLOCK(0,1,edx,SIZEOF_FAST_FLOAT)]
    movaps      xmm2, XMMWORD [XMMBLOCK(1,0,esi,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(1,1,esi,SIZEOF_FAST_FLOAT)]
    mulps       xmm2, XMMWORD [XMMBLOCK(1,0,edx,SIZEOF_FAST_FLOAT)]
    mulps       xmm3, XMMWORD [XMMBLOCK(1,1,edx,SIZEOF_FAST_FLOAT)]

    cvtps2dq    xmm0, xmm0
    cvtps2dq    xmm1, xmm1
    cvtps2dq    xmm2, xmm2
    cvtps2dq    xmm3, xmm3

    packssdw    xmm0, xmm1
    packssdw    xmm2, xmm3

    movdqa      XMMWORD [XMMBLOCK(0,0,edi,SIZEOF_JCOEF)], xmm0
    movdqa      XMMWORD [XMMBLOCK(1,0,edi,SIZEOF_JCOEF)], xmm2

    add         esi, byte 16*SIZEOF_FAST_FLOAT
    add         edx, byte 16*SIZEOF_FAST_FLOAT
    add         edi, byte 16*SIZEOF_JCOEF
    dec         eax
    jnz         short .quantloop

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
