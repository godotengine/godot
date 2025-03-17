;
; jquantf.asm - sample data conversion and quantization (64-bit SSE & SSE2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2024, D. R. Commander.
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
; jsimd_convsamp_float_sse2(JSAMPARRAY sample_data, JDIMENSION start_col,
;                           FAST_FLOAT *workspace);
;

; r10 = JSAMPARRAY sample_data
; r11d = JDIMENSION start_col
; r12 = FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_convsamp_float_sse2)

EXTN(jsimd_convsamp_float_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 3
    push        rbx

    pcmpeqw     xmm7, xmm7
    psllw       xmm7, 7
    packsswb    xmm7, xmm7              ; xmm7 = PB_CENTERJSAMPLE (0x808080..)

    mov         rsi, r10
    mov         eax, r11d
    mov         rdi, r12
    mov         rcx, DCTSIZE/2
.convloop:
    mov         rbxp, JSAMPROW [rsi+0*SIZEOF_JSAMPROW]  ; (JSAMPLE *)
    mov         rdxp, JSAMPROW [rsi+1*SIZEOF_JSAMPROW]  ; (JSAMPLE *)

    movq        xmm0, XMM_MMWORD [rbx+rax*SIZEOF_JSAMPLE]
    movq        xmm1, XMM_MMWORD [rdx+rax*SIZEOF_JSAMPLE]

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

    movaps      XMMWORD [XMMBLOCK(0,0,rdi,SIZEOF_FAST_FLOAT)], xmm2
    movaps      XMMWORD [XMMBLOCK(0,1,rdi,SIZEOF_FAST_FLOAT)], xmm0
    movaps      XMMWORD [XMMBLOCK(1,0,rdi,SIZEOF_FAST_FLOAT)], xmm3
    movaps      XMMWORD [XMMBLOCK(1,1,rdi,SIZEOF_FAST_FLOAT)], xmm1

    add         rsi, byte 2*SIZEOF_JSAMPROW
    add         rdi, byte 2*DCTSIZE*SIZEOF_FAST_FLOAT
    dec         rcx
    jnz         short .convloop

    pop         rbx
    UNCOLLECT_ARGS 3
    pop         rbp
    ret

; --------------------------------------------------------------------------
;
; Quantize/descale the coefficients, and store into coef_block
;
; GLOBAL(void)
; jsimd_quantize_float_sse2(JCOEFPTR coef_block, FAST_FLOAT *divisors,
;                           FAST_FLOAT *workspace);
;

; r10 = JCOEFPTR coef_block
; r11 = FAST_FLOAT *divisors
; r12 = FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_quantize_float_sse2)

EXTN(jsimd_quantize_float_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 3

    mov         rsi, r12
    mov         rdx, r11
    mov         rdi, r10
    mov         rax, DCTSIZE2/16
.quantloop:
    movaps      xmm0, XMMWORD [XMMBLOCK(0,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm1, XMMWORD [XMMBLOCK(0,1,rsi,SIZEOF_FAST_FLOAT)]
    mulps       xmm0, XMMWORD [XMMBLOCK(0,0,rdx,SIZEOF_FAST_FLOAT)]
    mulps       xmm1, XMMWORD [XMMBLOCK(0,1,rdx,SIZEOF_FAST_FLOAT)]
    movaps      xmm2, XMMWORD [XMMBLOCK(1,0,rsi,SIZEOF_FAST_FLOAT)]
    movaps      xmm3, XMMWORD [XMMBLOCK(1,1,rsi,SIZEOF_FAST_FLOAT)]
    mulps       xmm2, XMMWORD [XMMBLOCK(1,0,rdx,SIZEOF_FAST_FLOAT)]
    mulps       xmm3, XMMWORD [XMMBLOCK(1,1,rdx,SIZEOF_FAST_FLOAT)]

    cvtps2dq    xmm0, xmm0
    cvtps2dq    xmm1, xmm1
    cvtps2dq    xmm2, xmm2
    cvtps2dq    xmm3, xmm3

    packssdw    xmm0, xmm1
    packssdw    xmm2, xmm3

    movdqa      XMMWORD [XMMBLOCK(0,0,rdi,SIZEOF_JCOEF)], xmm0
    movdqa      XMMWORD [XMMBLOCK(1,0,rdi,SIZEOF_JCOEF)], xmm2

    add         rsi, byte 16*SIZEOF_FAST_FLOAT
    add         rdx, byte 16*SIZEOF_FAST_FLOAT
    add         rdi, byte 16*SIZEOF_JCOEF
    dec         rax
    jnz         short .quantloop

    UNCOLLECT_ARGS 3
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
