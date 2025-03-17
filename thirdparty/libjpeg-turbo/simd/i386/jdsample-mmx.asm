;
; jdsample.asm - upsampling (MMX)
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
    SECTION     SEG_CONST

    ALIGNZ      32
    GLOBAL_DATA(jconst_fancy_upsample_mmx)

EXTN(jconst_fancy_upsample_mmx):

PW_ONE   times 4 dw 1
PW_TWO   times 4 dw 2
PW_THREE times 4 dw 3
PW_SEVEN times 4 dw 7
PW_EIGHT times 4 dw 8

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
; jsimd_h2v1_fancy_upsample_mmx(int max_v_samp_factor,
;                               JDIMENSION downsampled_width,
;                               JSAMPARRAY input_data,
;                               JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define downsamp_width(b)   (b) + 12    ; JDIMENSION downsampled_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_fancy_upsample_mmx)

EXTN(jsimd_h2v1_fancy_upsample_mmx):
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

    test        eax, SIZEOF_MMWORD-1
    jz          short .skip
    mov         dl, JSAMPLE [esi+(eax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [esi+eax*SIZEOF_JSAMPLE], dl    ; insert a dummy sample
.skip:
    pxor        mm0, mm0                ; mm0=(all 0's)
    pcmpeqb     mm7, mm7
    psrlq       mm7, (SIZEOF_MMWORD-1)*BYTE_BIT
    pand        mm7,  MMWORD [esi+0*SIZEOF_MMWORD]

    add         eax, byte SIZEOF_MMWORD-1
    and         eax, byte -SIZEOF_MMWORD
    cmp         eax, byte SIZEOF_MMWORD
    ja          short .columnloop
    ALIGNX      16, 7

.columnloop_last:
    pcmpeqb     mm6, mm6
    psllq       mm6, (SIZEOF_MMWORD-1)*BYTE_BIT
    pand        mm6, MMWORD [esi+0*SIZEOF_MMWORD]
    jmp         short .upsample
    ALIGNX      16, 7

.columnloop:
    movq        mm6, MMWORD [esi+1*SIZEOF_MMWORD]
    psllq       mm6, (SIZEOF_MMWORD-1)*BYTE_BIT

.upsample:
    movq        mm1, MMWORD [esi+0*SIZEOF_MMWORD]
    movq        mm2, mm1
    movq        mm3, mm1                ; mm1=( 0 1 2 3 4 5 6 7)
    psllq       mm2, BYTE_BIT           ; mm2=( - 0 1 2 3 4 5 6)
    psrlq       mm3, BYTE_BIT           ; mm3=( 1 2 3 4 5 6 7 -)

    por         mm2, mm7                ; mm2=(-1 0 1 2 3 4 5 6)
    por         mm3, mm6                ; mm3=( 1 2 3 4 5 6 7 8)

    movq        mm7, mm1
    psrlq       mm7, (SIZEOF_MMWORD-1)*BYTE_BIT  ; mm7=( 7 - - - - - - -)

    movq        mm4, mm1
    punpcklbw   mm1, mm0                ; mm1=( 0 1 2 3)
    punpckhbw   mm4, mm0                ; mm4=( 4 5 6 7)
    movq        mm5, mm2
    punpcklbw   mm2, mm0                ; mm2=(-1 0 1 2)
    punpckhbw   mm5, mm0                ; mm5=( 3 4 5 6)
    movq        mm6, mm3
    punpcklbw   mm3, mm0                ; mm3=( 1 2 3 4)
    punpckhbw   mm6, mm0                ; mm6=( 5 6 7 8)

    pmullw      mm1, [GOTOFF(ebx,PW_THREE)]
    pmullw      mm4, [GOTOFF(ebx,PW_THREE)]
    paddw       mm2, [GOTOFF(ebx,PW_ONE)]
    paddw       mm5, [GOTOFF(ebx,PW_ONE)]
    paddw       mm3, [GOTOFF(ebx,PW_TWO)]
    paddw       mm6, [GOTOFF(ebx,PW_TWO)]

    paddw       mm2, mm1
    paddw       mm5, mm4
    psrlw       mm2, 2                  ; mm2=OutLE=( 0  2  4  6)
    psrlw       mm5, 2                  ; mm5=OutHE=( 8 10 12 14)
    paddw       mm3, mm1
    paddw       mm6, mm4
    psrlw       mm3, 2                  ; mm3=OutLO=( 1  3  5  7)
    psrlw       mm6, 2                  ; mm6=OutHO=( 9 11 13 15)

    psllw       mm3, BYTE_BIT
    psllw       mm6, BYTE_BIT
    por         mm2, mm3                ; mm2=OutL=( 0  1  2  3  4  5  6  7)
    por         mm5, mm6                ; mm5=OutH=( 8  9 10 11 12 13 14 15)

    movq        MMWORD [edi+0*SIZEOF_MMWORD], mm2
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mm5

    sub         eax, byte SIZEOF_MMWORD
    add         esi, byte 1*SIZEOF_MMWORD  ; inptr
    add         edi, byte 2*SIZEOF_MMWORD  ; outptr
    cmp         eax, byte SIZEOF_MMWORD
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

    emms                                ; empty MMX state

.return:
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
; jsimd_h2v2_fancy_upsample_mmx(int max_v_samp_factor,
;                               JDIMENSION downsampled_width,
;                               JSAMPARRAY input_data,
;                               JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define downsamp_width(b)   (b) + 12    ; JDIMENSION downsampled_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_MMWORD  ; mmword wk[WK_NUM]
%define WK_NUM        4
%define gotptr        wk(0) - SIZEOF_POINTER  ; void *gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_fancy_upsample_mmx)

EXTN(jsimd_h2v2_fancy_upsample_mmx):
    push        ebp
    mov         eax, esp                    ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_MMWORD)  ; align to 64 bits
    mov         [esp], eax
    mov         ebp, esp                    ; ebp = aligned ebp
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

    test        eax, SIZEOF_MMWORD-1
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

    movq        mm0, MMWORD [ebx+0*SIZEOF_MMWORD]  ; mm0=row[ 0][0]
    movq        mm1, MMWORD [ecx+0*SIZEOF_MMWORD]  ; mm1=row[-1][0]
    movq        mm2, MMWORD [esi+0*SIZEOF_MMWORD]  ; mm2=row[+1][0]

    PUSHPIC     ebx
    MOVPIC      ebx, POINTER [gotptr]   ; load GOT address

    pxor        mm3, mm3                ; mm3=(all 0's)
    movq        mm4, mm0
    punpcklbw   mm0, mm3                ; mm0=row[ 0][0]( 0 1 2 3)
    punpckhbw   mm4, mm3                ; mm4=row[ 0][0]( 4 5 6 7)
    movq        mm5, mm1
    punpcklbw   mm1, mm3                ; mm1=row[-1][0]( 0 1 2 3)
    punpckhbw   mm5, mm3                ; mm5=row[-1][0]( 4 5 6 7)
    movq        mm6, mm2
    punpcklbw   mm2, mm3                ; mm2=row[+1][0]( 0 1 2 3)
    punpckhbw   mm6, mm3                ; mm6=row[+1][0]( 4 5 6 7)

    pmullw      mm0, [GOTOFF(ebx,PW_THREE)]
    pmullw      mm4, [GOTOFF(ebx,PW_THREE)]

    pcmpeqb     mm7, mm7
    psrlq       mm7, (SIZEOF_MMWORD-2)*BYTE_BIT

    paddw       mm1, mm0                ; mm1=Int0L=( 0 1 2 3)
    paddw       mm5, mm4                ; mm5=Int0H=( 4 5 6 7)
    paddw       mm2, mm0                ; mm2=Int1L=( 0 1 2 3)
    paddw       mm6, mm4                ; mm6=Int1H=( 4 5 6 7)

    movq        MMWORD [edx+0*SIZEOF_MMWORD], mm1  ; temporarily save
    movq        MMWORD [edx+1*SIZEOF_MMWORD], mm5  ; the intermediate data
    movq        MMWORD [edi+0*SIZEOF_MMWORD], mm2
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mm6

    pand        mm1, mm7                ; mm1=( 0 - - -)
    pand        mm2, mm7                ; mm2=( 0 - - -)

    movq        MMWORD [wk(0)], mm1
    movq        MMWORD [wk(1)], mm2

    POPPIC      ebx

    add         eax, byte SIZEOF_MMWORD-1
    and         eax, byte -SIZEOF_MMWORD
    cmp         eax, byte SIZEOF_MMWORD
    ja          short .columnloop
    ALIGNX      16, 7

.columnloop_last:
    ; -- process the last column block

    PUSHPIC     ebx
    MOVPIC      ebx, POINTER [gotptr]   ; load GOT address

    pcmpeqb     mm1, mm1
    psllq       mm1, (SIZEOF_MMWORD-2)*BYTE_BIT
    movq        mm2, mm1

    pand        mm1, MMWORD [edx+1*SIZEOF_MMWORD]  ; mm1=( - - - 7)
    pand        mm2, MMWORD [edi+1*SIZEOF_MMWORD]  ; mm2=( - - - 7)

    movq        MMWORD [wk(2)], mm1
    movq        MMWORD [wk(3)], mm2

    jmp         short .upsample
    ALIGNX      16, 7

.columnloop:
    ; -- process the next column block

    movq        mm0, MMWORD [ebx+1*SIZEOF_MMWORD]  ; mm0=row[ 0][1]
    movq        mm1, MMWORD [ecx+1*SIZEOF_MMWORD]  ; mm1=row[-1][1]
    movq        mm2, MMWORD [esi+1*SIZEOF_MMWORD]  ; mm2=row[+1][1]

    PUSHPIC     ebx
    MOVPIC      ebx, POINTER [gotptr]   ; load GOT address

    pxor        mm3, mm3                ; mm3=(all 0's)
    movq        mm4, mm0
    punpcklbw   mm0, mm3                ; mm0=row[ 0][1]( 0 1 2 3)
    punpckhbw   mm4, mm3                ; mm4=row[ 0][1]( 4 5 6 7)
    movq        mm5, mm1
    punpcklbw   mm1, mm3                ; mm1=row[-1][1]( 0 1 2 3)
    punpckhbw   mm5, mm3                ; mm5=row[-1][1]( 4 5 6 7)
    movq        mm6, mm2
    punpcklbw   mm2, mm3                ; mm2=row[+1][1]( 0 1 2 3)
    punpckhbw   mm6, mm3                ; mm6=row[+1][1]( 4 5 6 7)

    pmullw      mm0, [GOTOFF(ebx,PW_THREE)]
    pmullw      mm4, [GOTOFF(ebx,PW_THREE)]

    paddw       mm1, mm0                ; mm1=Int0L=( 0 1 2 3)
    paddw       mm5, mm4                ; mm5=Int0H=( 4 5 6 7)
    paddw       mm2, mm0                ; mm2=Int1L=( 0 1 2 3)
    paddw       mm6, mm4                ; mm6=Int1H=( 4 5 6 7)

    movq        MMWORD [edx+2*SIZEOF_MMWORD], mm1  ; temporarily save
    movq        MMWORD [edx+3*SIZEOF_MMWORD], mm5  ; the intermediate data
    movq        MMWORD [edi+2*SIZEOF_MMWORD], mm2
    movq        MMWORD [edi+3*SIZEOF_MMWORD], mm6

    psllq       mm1, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm1=( - - - 0)
    psllq       mm2, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm2=( - - - 0)

    movq        MMWORD [wk(2)], mm1
    movq        MMWORD [wk(3)], mm2

.upsample:
    ; -- process the upper row

    movq        mm7, MMWORD [edx+0*SIZEOF_MMWORD]  ; mm7=Int0L=( 0 1 2 3)
    movq        mm3, MMWORD [edx+1*SIZEOF_MMWORD]  ; mm3=Int0H=( 4 5 6 7)

    movq        mm0, mm7
    movq        mm4, mm3
    psrlq       mm0, 2*BYTE_BIT                  ; mm0=( 1 2 3 -)
    psllq       mm4, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm4=( - - - 4)
    movq        mm5, mm7
    movq        mm6, mm3
    psrlq       mm5, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm5=( 3 - - -)
    psllq       mm6, 2*BYTE_BIT                  ; mm6=( - 4 5 6)

    por         mm0, mm4                         ; mm0=( 1 2 3 4)
    por         mm5, mm6                         ; mm5=( 3 4 5 6)

    movq        mm1, mm7
    movq        mm2, mm3
    psllq       mm1, 2*BYTE_BIT                  ; mm1=( - 0 1 2)
    psrlq       mm2, 2*BYTE_BIT                  ; mm2=( 5 6 7 -)
    movq        mm4, mm3
    psrlq       mm4, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm4=( 7 - - -)

    por         mm1, MMWORD [wk(0)]              ; mm1=(-1 0 1 2)
    por         mm2, MMWORD [wk(2)]              ; mm2=( 5 6 7 8)

    movq        MMWORD [wk(0)], mm4

    pmullw      mm7, [GOTOFF(ebx,PW_THREE)]
    pmullw      mm3, [GOTOFF(ebx,PW_THREE)]
    paddw       mm1, [GOTOFF(ebx,PW_EIGHT)]
    paddw       mm5, [GOTOFF(ebx,PW_EIGHT)]
    paddw       mm0, [GOTOFF(ebx,PW_SEVEN)]
    paddw       mm2, [GOTOFF(ebx,PW_SEVEN)]

    paddw       mm1, mm7
    paddw       mm5, mm3
    psrlw       mm1, 4                  ; mm1=Out0LE=( 0  2  4  6)
    psrlw       mm5, 4                  ; mm5=Out0HE=( 8 10 12 14)
    paddw       mm0, mm7
    paddw       mm2, mm3
    psrlw       mm0, 4                  ; mm0=Out0LO=( 1  3  5  7)
    psrlw       mm2, 4                  ; mm2=Out0HO=( 9 11 13 15)

    psllw       mm0, BYTE_BIT
    psllw       mm2, BYTE_BIT
    por         mm1, mm0                ; mm1=Out0L=( 0  1  2  3  4  5  6  7)
    por         mm5, mm2                ; mm5=Out0H=( 8  9 10 11 12 13 14 15)

    movq        MMWORD [edx+0*SIZEOF_MMWORD], mm1
    movq        MMWORD [edx+1*SIZEOF_MMWORD], mm5

    ; -- process the lower row

    movq        mm6, MMWORD [edi+0*SIZEOF_MMWORD]  ; mm6=Int1L=( 0 1 2 3)
    movq        mm4, MMWORD [edi+1*SIZEOF_MMWORD]  ; mm4=Int1H=( 4 5 6 7)

    movq        mm7, mm6
    movq        mm3, mm4
    psrlq       mm7, 2*BYTE_BIT                  ; mm7=( 1 2 3 -)
    psllq       mm3, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm3=( - - - 4)
    movq        mm0, mm6
    movq        mm2, mm4
    psrlq       mm0, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm0=( 3 - - -)
    psllq       mm2, 2*BYTE_BIT                  ; mm2=( - 4 5 6)

    por         mm7, mm3                         ; mm7=( 1 2 3 4)
    por         mm0, mm2                         ; mm0=( 3 4 5 6)

    movq        mm1, mm6
    movq        mm5, mm4
    psllq       mm1, 2*BYTE_BIT                  ; mm1=( - 0 1 2)
    psrlq       mm5, 2*BYTE_BIT                  ; mm5=( 5 6 7 -)
    movq        mm3, mm4
    psrlq       mm3, (SIZEOF_MMWORD-2)*BYTE_BIT  ; mm3=( 7 - - -)

    por         mm1, MMWORD [wk(1)]              ; mm1=(-1 0 1 2)
    por         mm5, MMWORD [wk(3)]              ; mm5=( 5 6 7 8)

    movq        MMWORD [wk(1)], mm3

    pmullw      mm6, [GOTOFF(ebx,PW_THREE)]
    pmullw      mm4, [GOTOFF(ebx,PW_THREE)]
    paddw       mm1, [GOTOFF(ebx,PW_EIGHT)]
    paddw       mm0, [GOTOFF(ebx,PW_EIGHT)]
    paddw       mm7, [GOTOFF(ebx,PW_SEVEN)]
    paddw       mm5, [GOTOFF(ebx,PW_SEVEN)]

    paddw       mm1, mm6
    paddw       mm0, mm4
    psrlw       mm1, 4                  ; mm1=Out1LE=( 0  2  4  6)
    psrlw       mm0, 4                  ; mm0=Out1HE=( 8 10 12 14)
    paddw       mm7, mm6
    paddw       mm5, mm4
    psrlw       mm7, 4                  ; mm7=Out1LO=( 1  3  5  7)
    psrlw       mm5, 4                  ; mm5=Out1HO=( 9 11 13 15)

    psllw       mm7, BYTE_BIT
    psllw       mm5, BYTE_BIT
    por         mm1, mm7                ; mm1=Out1L=( 0  1  2  3  4  5  6  7)
    por         mm0, mm5                ; mm0=Out1H=( 8  9 10 11 12 13 14 15)

    movq        MMWORD [edi+0*SIZEOF_MMWORD], mm1
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mm0

    POPPIC      ebx

    sub         eax, byte SIZEOF_MMWORD
    add         ecx, byte 1*SIZEOF_MMWORD  ; inptr1(above)
    add         ebx, byte 1*SIZEOF_MMWORD  ; inptr0
    add         esi, byte 1*SIZEOF_MMWORD  ; inptr1(below)
    add         edx, byte 2*SIZEOF_MMWORD  ; outptr0
    add         edi, byte 2*SIZEOF_MMWORD  ; outptr1
    cmp         eax, byte SIZEOF_MMWORD
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

    emms                                ; empty MMX state

.return:
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
; jsimd_h2v1_upsample_mmx(int max_v_samp_factor, JDIMENSION output_width,
;                         JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define output_width(b)     (b) + 12    ; JDIMENSION output_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_upsample_mmx)

EXTN(jsimd_h2v1_upsample_mmx):
    push        ebp
    mov         ebp, esp
;   push        ebx                     ; unused
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         edx, JDIMENSION [output_width(ebp)]
    add         edx, byte (2*SIZEOF_MMWORD)-1
    and         edx, byte -(2*SIZEOF_MMWORD)
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

    movq        mm0, MMWORD [esi+0*SIZEOF_MMWORD]

    movq        mm1, mm0
    punpcklbw   mm0, mm0
    punpckhbw   mm1, mm1

    movq        MMWORD [edi+0*SIZEOF_MMWORD], mm0
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mm1

    sub         eax, byte 2*SIZEOF_MMWORD
    jz          short .nextrow

    movq        mm2, MMWORD [esi+1*SIZEOF_MMWORD]

    movq        mm3, mm2
    punpcklbw   mm2, mm2
    punpckhbw   mm3, mm3

    movq        MMWORD [edi+2*SIZEOF_MMWORD], mm2
    movq        MMWORD [edi+3*SIZEOF_MMWORD], mm3

    sub         eax, byte 2*SIZEOF_MMWORD
    jz          short .nextrow

    add         esi, byte 2*SIZEOF_MMWORD  ; inptr
    add         edi, byte 4*SIZEOF_MMWORD  ; outptr
    jmp         short .columnloop
    ALIGNX      16, 7

.nextrow:
    pop         esi
    pop         edi

    add         esi, byte SIZEOF_JSAMPROW  ; input_data
    add         edi, byte SIZEOF_JSAMPROW  ; output_data
    dec         ecx                        ; rowctr
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
; Fast processing for the common case of 2:1 horizontal and 2:1 vertical.
; It's still a box filter.
;
; GLOBAL(void)
; jsimd_h2v2_upsample_mmx(int max_v_samp_factor, JDIMENSION output_width,
;                         JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr);
;

%define max_v_samp(b)       (b) + 8     ; int max_v_samp_factor
%define output_width(b)     (b) + 12    ; JDIMENSION output_width
%define input_data(b)       (b) + 16    ; JSAMPARRAY input_data
%define output_data_ptr(b)  (b) + 20    ; JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_upsample_mmx)

EXTN(jsimd_h2v2_upsample_mmx):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         edx, JDIMENSION [output_width(ebp)]
    add         edx, byte (2*SIZEOF_MMWORD)-1
    and         edx, byte -(2*SIZEOF_MMWORD)
    jz          near .return

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

    mov         esi, JSAMPROW [esi]                    ; inptr
    mov         ebx, JSAMPROW [edi+0*SIZEOF_JSAMPROW]  ; outptr0
    mov         edi, JSAMPROW [edi+1*SIZEOF_JSAMPROW]  ; outptr1
    mov         eax, edx                               ; colctr
    ALIGNX      16, 7
.columnloop:

    movq        mm0, MMWORD [esi+0*SIZEOF_MMWORD]

    movq        mm1, mm0
    punpcklbw   mm0, mm0
    punpckhbw   mm1, mm1

    movq        MMWORD [ebx+0*SIZEOF_MMWORD], mm0
    movq        MMWORD [ebx+1*SIZEOF_MMWORD], mm1
    movq        MMWORD [edi+0*SIZEOF_MMWORD], mm0
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mm1

    sub         eax, byte 2*SIZEOF_MMWORD
    jz          short .nextrow

    movq        mm2, MMWORD [esi+1*SIZEOF_MMWORD]

    movq        mm3, mm2
    punpcklbw   mm2, mm2
    punpckhbw   mm3, mm3

    movq        MMWORD [ebx+2*SIZEOF_MMWORD], mm2
    movq        MMWORD [ebx+3*SIZEOF_MMWORD], mm3
    movq        MMWORD [edi+2*SIZEOF_MMWORD], mm2
    movq        MMWORD [edi+3*SIZEOF_MMWORD], mm3

    sub         eax, byte 2*SIZEOF_MMWORD
    jz          short .nextrow

    add         esi, byte 2*SIZEOF_MMWORD  ; inptr
    add         ebx, byte 4*SIZEOF_MMWORD  ; outptr0
    add         edi, byte 4*SIZEOF_MMWORD  ; outptr1
    jmp         short .columnloop
    ALIGNX      16, 7

.nextrow:
    pop         esi
    pop         edi

    add         esi, byte 1*SIZEOF_JSAMPROW  ; input_data
    add         edi, byte 2*SIZEOF_JSAMPROW  ; output_data
    sub         ecx, byte 2                  ; rowctr
    jg          short .rowloop

    emms                                ; empty MMX state

.return:
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
