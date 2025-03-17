;
; jquant.asm - sample data conversion and quantization (3DNow! & MMX)
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
; jsimd_convsamp_float_3dnow(JSAMPARRAY sample_data, JDIMENSION start_col,
;                            FAST_FLOAT *workspace);
;

%define sample_data  ebp + 8            ; JSAMPARRAY sample_data
%define start_col    ebp + 12           ; JDIMENSION start_col
%define workspace    ebp + 16           ; FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_convsamp_float_3dnow)

EXTN(jsimd_convsamp_float_3dnow):
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
    pi2fd       mm4, mm4
    pi2fd       mm2, mm2
    psrad       mm5, (DWORD_BIT-BYTE_BIT)  ; mm5=(45)
    psrad       mm0, (DWORD_BIT-BYTE_BIT)  ; mm0=(67)
    pi2fd       mm5, mm5
    pi2fd       mm0, mm0

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_FAST_FLOAT)], mm4
    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_FAST_FLOAT)], mm2
    movq        MMWORD [MMBLOCK(0,2,edi,SIZEOF_FAST_FLOAT)], mm5
    movq        MMWORD [MMBLOCK(0,3,edi,SIZEOF_FAST_FLOAT)], mm0

    punpcklwd   mm6, mm3                ; mm6=(***8***9)
    punpckhwd   mm3, mm3                ; mm3=(***A***B)
    punpcklwd   mm4, mm1                ; mm4=(***C***D)
    punpckhwd   mm1, mm1                ; mm1=(***E***F)

    psrad       mm6, (DWORD_BIT-BYTE_BIT)  ; mm6=(89)
    psrad       mm3, (DWORD_BIT-BYTE_BIT)  ; mm3=(AB)
    pi2fd       mm6, mm6
    pi2fd       mm3, mm3
    psrad       mm4, (DWORD_BIT-BYTE_BIT)  ; mm4=(CD)
    psrad       mm1, (DWORD_BIT-BYTE_BIT)  ; mm1=(EF)
    pi2fd       mm4, mm4
    pi2fd       mm1, mm1

    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_FAST_FLOAT)], mm6
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_FAST_FLOAT)], mm3
    movq        MMWORD [MMBLOCK(1,2,edi,SIZEOF_FAST_FLOAT)], mm4
    movq        MMWORD [MMBLOCK(1,3,edi,SIZEOF_FAST_FLOAT)], mm1

    add         esi, byte 2*SIZEOF_JSAMPROW
    add         edi, byte 2*DCTSIZE*SIZEOF_FAST_FLOAT
    dec         ecx
    jnz         near .convloop

    femms                               ; empty MMX/3DNow! state

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
; jsimd_quantize_float_3dnow(JCOEFPTR coef_block, FAST_FLOAT *divisors,
;                            FAST_FLOAT *workspace);
;

%define coef_block  ebp + 8             ; JCOEFPTR coef_block
%define divisors    ebp + 12            ; FAST_FLOAT *divisors
%define workspace   ebp + 16            ; FAST_FLOAT *workspace

    align       32
    GLOBAL_FUNCTION(jsimd_quantize_float_3dnow)

EXTN(jsimd_quantize_float_3dnow):
    push        ebp
    mov         ebp, esp
;   push        ebx                     ; unused
;   push        ecx                     ; unused
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         eax, 0x4B400000         ; (float)0x00C00000 (rndint_magic)
    movd        mm7, eax
    punpckldq   mm7, mm7                ; mm7={12582912.0F 12582912.0F}

    mov         esi, POINTER [workspace]
    mov         edx, POINTER [divisors]
    mov         edi, JCOEFPTR [coef_block]
    mov         eax, DCTSIZE2/16
    ALIGNX      16, 7
.quantloop:
    movq        mm0, MMWORD [MMBLOCK(0,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm1, MMWORD [MMBLOCK(0,1,esi,SIZEOF_FAST_FLOAT)]
    pfmul       mm0, MMWORD [MMBLOCK(0,0,edx,SIZEOF_FAST_FLOAT)]
    pfmul       mm1, MMWORD [MMBLOCK(0,1,edx,SIZEOF_FAST_FLOAT)]
    movq        mm2, MMWORD [MMBLOCK(0,2,esi,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(0,3,esi,SIZEOF_FAST_FLOAT)]
    pfmul       mm2, MMWORD [MMBLOCK(0,2,edx,SIZEOF_FAST_FLOAT)]
    pfmul       mm3, MMWORD [MMBLOCK(0,3,edx,SIZEOF_FAST_FLOAT)]

    pfadd       mm0, mm7                ; mm0=(00 ** 01 **)
    pfadd       mm1, mm7                ; mm1=(02 ** 03 **)
    pfadd       mm2, mm7                ; mm0=(04 ** 05 **)
    pfadd       mm3, mm7                ; mm1=(06 ** 07 **)

    movq        mm4, mm0
    punpcklwd   mm0, mm1                ; mm0=(00 02 ** **)
    punpckhwd   mm4, mm1                ; mm4=(01 03 ** **)
    movq        mm5, mm2
    punpcklwd   mm2, mm3                ; mm2=(04 06 ** **)
    punpckhwd   mm5, mm3                ; mm5=(05 07 ** **)

    punpcklwd   mm0, mm4                ; mm0=(00 01 02 03)
    punpcklwd   mm2, mm5                ; mm2=(04 05 06 07)

    movq        mm6, MMWORD [MMBLOCK(1,0,esi,SIZEOF_FAST_FLOAT)]
    movq        mm1, MMWORD [MMBLOCK(1,1,esi,SIZEOF_FAST_FLOAT)]
    pfmul       mm6, MMWORD [MMBLOCK(1,0,edx,SIZEOF_FAST_FLOAT)]
    pfmul       mm1, MMWORD [MMBLOCK(1,1,edx,SIZEOF_FAST_FLOAT)]
    movq        mm3, MMWORD [MMBLOCK(1,2,esi,SIZEOF_FAST_FLOAT)]
    movq        mm4, MMWORD [MMBLOCK(1,3,esi,SIZEOF_FAST_FLOAT)]
    pfmul       mm3, MMWORD [MMBLOCK(1,2,edx,SIZEOF_FAST_FLOAT)]
    pfmul       mm4, MMWORD [MMBLOCK(1,3,edx,SIZEOF_FAST_FLOAT)]

    pfadd       mm6, mm7                ; mm0=(10 ** 11 **)
    pfadd       mm1, mm7                ; mm4=(12 ** 13 **)
    pfadd       mm3, mm7                ; mm0=(14 ** 15 **)
    pfadd       mm4, mm7                ; mm4=(16 ** 17 **)

    movq        mm5, mm6
    punpcklwd   mm6, mm1                ; mm6=(10 12 ** **)
    punpckhwd   mm5, mm1                ; mm5=(11 13 ** **)
    movq        mm1, mm3
    punpcklwd   mm3, mm4                ; mm3=(14 16 ** **)
    punpckhwd   mm1, mm4                ; mm1=(15 17 ** **)

    punpcklwd   mm6, mm5                ; mm6=(10 11 12 13)
    punpcklwd   mm3, mm1                ; mm3=(14 15 16 17)

    movq        MMWORD [MMBLOCK(0,0,edi,SIZEOF_JCOEF)], mm0
    movq        MMWORD [MMBLOCK(0,1,edi,SIZEOF_JCOEF)], mm2
    movq        MMWORD [MMBLOCK(1,0,edi,SIZEOF_JCOEF)], mm6
    movq        MMWORD [MMBLOCK(1,1,edi,SIZEOF_JCOEF)], mm3

    add         esi, byte 16*SIZEOF_FAST_FLOAT
    add         edx, byte 16*SIZEOF_FAST_FLOAT
    add         edi, byte 16*SIZEOF_JCOEF
    dec         eax
    jnz         near .quantloop

    femms                               ; empty MMX/3DNow! state

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
