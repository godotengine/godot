;
; jquant.asm - sample data conversion and quantization (SSE & MMX)
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
; jsimd_convsamp_float_sse(JSAMPARRAY sample_data, JDIMENSION start_col,
;                          FAST_FLOAT *workspace);
;

%define sample_data  ebp + 8            ; JSAMPARRAY sample_data
%define start_col    ebp + 12           ; JDIMENSION start_col
%define workspace    ebp + 16           ; FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_convsamp_float_sse)

EXTN(jsimd_convsamp_float_sse):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    pcmpeqw     mm7, mm7
    psllw       mm7, 7
    packsswb    mm7, mm7                ; mm7 = PB_CENTERJSAMPLE (0x808080..)

    mov         esi, JSAMPARRAY [sample_data]  ; (JSAMPROW *)
    mov         eax, JDIMENSION [start_col]
    mov         edi, POINTER [workspace]       ; (DCTELEM *)
    mov         ecx, DCTSIZE/2
    ALIGNX      16, 7
.convloop:
    mov         ebx, JSAMPROW [esi+0*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         edx, JSAMPROW [esi+1*SIZEOF_JSAMPROW]  ; (JSAMPLE *)

    movq        mm0, MMWORD [ebx+eax*SIZEOF_JSAMPLE]
    movq        mm1, MMWORD [edx+eax*SIZEOF_JSAMPLE]

    psubb       mm0, mm7                ; mm0=(01234567)
    psubb       mm1, mm7                ; mm1=(89ABCDEF)

    punpcklbw   mm2, mm0                ; mm2=(*0*1*2*3)
    punpckhbw   mm0, mm0                ; mm0=(*4*5*6*7)
    punpcklbw   mm3, mm1                ; mm3=(*8*9*A*B)
    punpckhbw   mm1, mm1                ; mm1=(*C*D*E*F)

    punpcklwd   mm4, mm2                ; mm4=(***0***1)
    punpckhwd   mm2, mm2                ; mm2=(***2***3)
    punpcklwd   mm5, mm0                ; mm5=(***4***5)
    punpckhwd   mm0, mm0                ; mm0=(***6***7)

    psrad       mm4, (DWORD_BIT-BYTE_BIT)  ; mm4=(01)
    psrad       mm2, (DWORD_BIT-BYTE_BIT)  ; mm2=(23)
    cvtpi2ps    xmm0, mm4                  ; xmm0=(01**)
    cvtpi2ps    xmm1, mm2                  ; xmm1=(23**)
    psrad       mm5, (DWORD_BIT-BYTE_BIT)  ; mm5=(45)
    psrad       mm0, (DWORD_BIT-BYTE_BIT)  ; mm0=(67)
    cvtpi2ps    xmm2, mm5                  ; xmm2=(45**)
    cvtpi2ps    xmm3, mm0                  ; xmm3=(67**)

    punpcklwd   mm6, mm3                ; mm6=(***8***9)
    punpckhwd   mm3, mm3                ; mm3=(***A***B)
    punpcklwd   mm4, mm1                ; mm4=(***C***D)
    punpckhwd   mm1, mm1                ; mm1=(***E***F)

    psrad       mm6, (DWORD_BIT-BYTE_BIT)  ; mm6=(89)
    psrad       mm3, (DWORD_BIT-BYTE_BIT)  ; mm3=(AB)
    cvtpi2ps    xmm4, mm6                  ; xmm4=(89**)
    cvtpi2ps    xmm5, mm3                  ; xmm5=(AB**)
    psrad       mm4, (DWORD_BIT-BYTE_BIT)  ; mm4=(CD)
    psrad       mm1, (DWORD_BIT-BYTE_BIT)  ; mm1=(EF)
    cvtpi2ps    xmm6, mm4                  ; xmm6=(CD**)
    cvtpi2ps    xmm7, mm1                  ; xmm7=(EF**)

    movlhps     xmm0, xmm1              ; xmm0=(0123)
    movlhps     xmm2, xmm3              ; xmm2=(4567)
    movlhps     xmm4, xmm5              ; xmm4=(89AB)
    movlhps     xmm6, xmm7              ; xmm6=(CDEF)

    movaps      XMMWORD [XMMBLOCK(0,0,edi,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(0,1,edi,SIZEOF_FAST_FLOAT)], xmm2
    movaps      XMMWORD [XMMBLOCK(1,0,edi,SIZEOF_FAST_FLOAT)], xmm4
    movaps      XMMWORD [XMMBLOCK(1,1,edi,SIZEOF_FAST_FLOAT)], xmm6

    add         esi, byte 2*SIZEOF_JSAMPROW
    add         edi, byte 2*DCTSIZE*SIZEOF_FAST_FLOAT
    dec         ecx
    jnz         near .convloop

    emms                                ; empty MMX state

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
; jsimd_quantize_float_sse(JCOEFPTR coef_block, FAST_FLOAT *divisors,
;                          FAST_FLOAT *workspace);
;

%define coef_block  ebp + 8             ; JCOEFPTR coef_block
%define divisors    ebp + 12            ; FAST_FLOAT *divisors
%define workspace   ebp + 16            ; FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_quantize_float_sse)

EXTN(jsimd_quantize_float_sse):
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

    movhlps     xmm4, xmm0
    movhlps     xmm5, xmm1

    cvtps2pi    mm0, xmm0
    cvtps2pi    mm1, xmm1
    cvtps2pi    mm4, xmm4
    cvtps2pi    mm5, xmm5

    movhlps     xmm6, xmm2
    movhlps     xmm7, xmm3

    cvtps2pi    mm2, xmm2
    cvtps2pi    mm3, xmm3
    cvtps2pi    mm6, xmm6
    cvtps2pi    mm7, xmm7

    packssdw    mm0, mm4
    packssdw    mm1, mm5
    packssdw    mm2, mm6
    packssdw    mm3, mm7

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_JCOEF)], mm0
    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_JCOEF)], mm1
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_JCOEF)], mm2
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_JCOEF)], mm3

    add         esi, byte 16*SIZEOF_FAST_FLOAT
    add         edx, byte 16*SIZEOF_FAST_FLOAT
    add         edi, byte 16*SIZEOF_JCOEF
    dec         eax
    jnz         short .quantloop

    emms                                ; empty MMX state

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
