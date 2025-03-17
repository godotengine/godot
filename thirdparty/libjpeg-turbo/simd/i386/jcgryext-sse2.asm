;
; jcgryext.asm - grayscale colorspace conversion (SSE2)
;
; Copyright (C) 2011, 2016, 2024, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jcolsamp.inc"

; --------------------------------------------------------------------------
;
; Convert some rows of samples to the output colorspace.
;
; GLOBAL(void)
; jsimd_rgb_gray_convert_sse2(JDIMENSION img_width, JSAMPARRAY input_buf,
;                             JSAMPIMAGE output_buf, JDIMENSION output_row,
;                             int num_rows);
;

%define img_width(b)   (b) + 8          ; JDIMENSION img_width
%define input_buf(b)   (b) + 12         ; JSAMPARRAY input_buf
%define output_buf(b)  (b) + 16         ; JSAMPIMAGE output_buf
%define output_row(b)  (b) + 20         ; JDIMENSION output_row
%define num_rows(b)    (b) + 24         ; int num_rows

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_XMMWORD
                                        ; xmmword wk[WK_NUM]
%define WK_NUM         2
%define gotptr         wk(0) - SIZEOF_POINTER  ; void * gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_rgb_gray_convert_sse2)

EXTN(jsimd_rgb_gray_convert_sse2):
    push        ebp
    mov         eax, esp                     ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    mov         [esp], eax
    mov         ebp, esp                     ; ebp = aligned ebp
    lea         esp, [wk(0)]
    PUSHPIC     eax                     ; make a room for GOT address
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    GET_GOT     ebx                     ; get GOT address
    MOVPIC      POINTER [gotptr], ebx   ; save GOT address

    mov         ecx, JDIMENSION [img_width(eax)]
    test        ecx, ecx
    jz          near .return

    push        ecx

    mov         esi, JSAMPIMAGE [output_buf(eax)]
    mov         ecx, JDIMENSION [output_row(eax)]
    mov         edi, JSAMPARRAY [esi+0*SIZEOF_JSAMPARRAY]
    lea         edi, [edi+ecx*SIZEOF_JSAMPROW]

    pop         ecx

    mov         esi, JSAMPARRAY [input_buf(eax)]
    mov         eax, INT [num_rows(eax)]
    test        eax, eax
    jle         near .return
    ALIGNX      16, 7
.rowloop:
    PUSHPIC     eax
    push        edi
    push        esi
    push        ecx                     ; col

    mov         esi, JSAMPROW [esi]     ; inptr
    mov         edi, JSAMPROW [edi]     ; outptr0
    MOVPIC      eax, POINTER [gotptr]   ; load GOT address (eax)

    cmp         ecx, byte SIZEOF_XMMWORD
    jae         near .columnloop
    ALIGNX      16, 7

%if RGB_PIXELSIZE == 3  ; ---------------

.column_ld1:
    push        eax
    push        edx
    lea         ecx, [ecx+ecx*2]        ; imul ecx,RGB_PIXELSIZE
    test        cl, SIZEOF_BYTE
    jz          short .column_ld2
    sub         ecx, byte SIZEOF_BYTE
    movzx       eax, byte [esi+ecx]
.column_ld2:
    test        cl, SIZEOF_WORD
    jz          short .column_ld4
    sub         ecx, byte SIZEOF_WORD
    movzx       edx, word [esi+ecx]
    shl         eax, WORD_BIT
    or          eax, edx
.column_ld4:
    movd        xmmA, eax
    pop         edx
    pop         eax
    test        cl, SIZEOF_DWORD
    jz          short .column_ld8
    sub         ecx, byte SIZEOF_DWORD
    movd        xmmF, XMM_DWORD [esi+ecx]
    pslldq      xmmA, SIZEOF_DWORD
    por         xmmA, xmmF
.column_ld8:
    test        cl, SIZEOF_MMWORD
    jz          short .column_ld16
    sub         ecx, byte SIZEOF_MMWORD
    movq        xmmB, XMM_MMWORD [esi+ecx]
    pslldq      xmmA, SIZEOF_MMWORD
    por         xmmA, xmmB
.column_ld16:
    test        cl, SIZEOF_XMMWORD
    jz          short .column_ld32
    movdqa      xmmF, xmmA
    movdqu      xmmA, XMMWORD [esi+0*SIZEOF_XMMWORD]
    mov         ecx, SIZEOF_XMMWORD
    jmp         short .rgb_gray_cnv
.column_ld32:
    test        cl, 2*SIZEOF_XMMWORD
    mov         ecx, SIZEOF_XMMWORD
    jz          short .rgb_gray_cnv
    movdqa      xmmB, xmmA
    movdqu      xmmA, XMMWORD [esi+0*SIZEOF_XMMWORD]
    movdqu      xmmF, XMMWORD [esi+1*SIZEOF_XMMWORD]
    jmp         short .rgb_gray_cnv
    ALIGNX      16, 7

.columnloop:
    movdqu      xmmA, XMMWORD [esi+0*SIZEOF_XMMWORD]
    movdqu      xmmF, XMMWORD [esi+1*SIZEOF_XMMWORD]
    movdqu      xmmB, XMMWORD [esi+2*SIZEOF_XMMWORD]

.rgb_gray_cnv:
    ; xmmA=(00 10 20 01 11 21 02 12 22 03 13 23 04 14 24 05)
    ; xmmF=(15 25 06 16 26 07 17 27 08 18 28 09 19 29 0A 1A)
    ; xmmB=(2A 0B 1B 2B 0C 1C 2C 0D 1D 2D 0E 1E 2E 0F 1F 2F)

    movdqa      xmmG, xmmA
    pslldq      xmmA, 8     ; xmmA=(-- -- -- -- -- -- -- -- 00 10 20 01 11 21 02 12)
    psrldq      xmmG, 8     ; xmmG=(22 03 13 23 04 14 24 05 -- -- -- -- -- -- -- --)

    punpckhbw   xmmA, xmmF  ; xmmA=(00 08 10 18 20 28 01 09 11 19 21 29 02 0A 12 1A)
    pslldq      xmmF, 8     ; xmmF=(-- -- -- -- -- -- -- -- 15 25 06 16 26 07 17 27)

    punpcklbw   xmmG, xmmB  ; xmmG=(22 2A 03 0B 13 1B 23 2B 04 0C 14 1C 24 2C 05 0D)
    punpckhbw   xmmF, xmmB  ; xmmF=(15 1D 25 2D 06 0E 16 1E 26 2E 07 0F 17 1F 27 2F)

    movdqa      xmmD, xmmA
    pslldq      xmmA, 8     ; xmmA=(-- -- -- -- -- -- -- -- 00 08 10 18 20 28 01 09)
    psrldq      xmmD, 8     ; xmmD=(11 19 21 29 02 0A 12 1A -- -- -- -- -- -- -- --)

    punpckhbw   xmmA, xmmG  ; xmmA=(00 04 08 0C 10 14 18 1C 20 24 28 2C 01 05 09 0D)
    pslldq      xmmG, 8     ; xmmG=(-- -- -- -- -- -- -- -- 22 2A 03 0B 13 1B 23 2B)

    punpcklbw   xmmD, xmmF  ; xmmD=(11 15 19 1D 21 25 29 2D 02 06 0A 0E 12 16 1A 1E)
    punpckhbw   xmmG, xmmF  ; xmmG=(22 26 2A 2E 03 07 0B 0F 13 17 1B 1F 23 27 2B 2F)

    movdqa      xmmE, xmmA
    pslldq      xmmA, 8     ; xmmA=(-- -- -- -- -- -- -- -- 00 04 08 0C 10 14 18 1C)
    psrldq      xmmE, 8     ; xmmE=(20 24 28 2C 01 05 09 0D -- -- -- -- -- -- -- --)

    punpckhbw   xmmA, xmmD  ; xmmA=(00 02 04 06 08 0A 0C 0E 10 12 14 16 18 1A 1C 1E)
    pslldq      xmmD, 8     ; xmmD=(-- -- -- -- -- -- -- -- 11 15 19 1D 21 25 29 2D)

    punpcklbw   xmmE, xmmG  ; xmmE=(20 22 24 26 28 2A 2C 2E 01 03 05 07 09 0B 0D 0F)
    punpckhbw   xmmD, xmmG  ; xmmD=(11 13 15 17 19 1B 1D 1F 21 23 25 27 29 2B 2D 2F)

    pxor        xmmH, xmmH

    movdqa      xmmC, xmmA
    punpcklbw   xmmA, xmmH  ; xmmA=(00 02 04 06 08 0A 0C 0E)
    punpckhbw   xmmC, xmmH  ; xmmC=(10 12 14 16 18 1A 1C 1E)

    movdqa      xmmB, xmmE
    punpcklbw   xmmE, xmmH  ; xmmE=(20 22 24 26 28 2A 2C 2E)
    punpckhbw   xmmB, xmmH  ; xmmB=(01 03 05 07 09 0B 0D 0F)

    movdqa      xmmF, xmmD
    punpcklbw   xmmD, xmmH  ; xmmD=(11 13 15 17 19 1B 1D 1F)
    punpckhbw   xmmF, xmmH  ; xmmF=(21 23 25 27 29 2B 2D 2F)

%else  ; RGB_PIXELSIZE == 4 ; -----------

.column_ld1:
    test        cl, SIZEOF_XMMWORD/16
    jz          short .column_ld2
    sub         ecx, byte SIZEOF_XMMWORD/16
    movd        xmmA, XMM_DWORD [esi+ecx*RGB_PIXELSIZE]
.column_ld2:
    test        cl, SIZEOF_XMMWORD/8
    jz          short .column_ld4
    sub         ecx, byte SIZEOF_XMMWORD/8
    movq        xmmE, XMM_MMWORD [esi+ecx*RGB_PIXELSIZE]
    pslldq      xmmA, SIZEOF_MMWORD
    por         xmmA, xmmE
.column_ld4:
    test        cl, SIZEOF_XMMWORD/4
    jz          short .column_ld8
    sub         ecx, byte SIZEOF_XMMWORD/4
    movdqa      xmmE, xmmA
    movdqu      xmmA, XMMWORD [esi+ecx*RGB_PIXELSIZE]
.column_ld8:
    test        cl, SIZEOF_XMMWORD/2
    mov         ecx, SIZEOF_XMMWORD
    jz          short .rgb_gray_cnv
    movdqa      xmmF, xmmA
    movdqa      xmmH, xmmE
    movdqu      xmmA, XMMWORD [esi+0*SIZEOF_XMMWORD]
    movdqu      xmmE, XMMWORD [esi+1*SIZEOF_XMMWORD]
    jmp         short .rgb_gray_cnv
    ALIGNX      16, 7

.columnloop:
    movdqu      xmmA, XMMWORD [esi+0*SIZEOF_XMMWORD]
    movdqu      xmmE, XMMWORD [esi+1*SIZEOF_XMMWORD]
    movdqu      xmmF, XMMWORD [esi+2*SIZEOF_XMMWORD]
    movdqu      xmmH, XMMWORD [esi+3*SIZEOF_XMMWORD]

.rgb_gray_cnv:
    ; xmmA=(00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33)
    ; xmmE=(04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37)
    ; xmmF=(08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B)
    ; xmmH=(0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F)

    movdqa      xmmD, xmmA
    punpcklbw   xmmA, xmmE      ; xmmA=(00 04 10 14 20 24 30 34 01 05 11 15 21 25 31 35)
    punpckhbw   xmmD, xmmE      ; xmmD=(02 06 12 16 22 26 32 36 03 07 13 17 23 27 33 37)

    movdqa      xmmC, xmmF
    punpcklbw   xmmF, xmmH      ; xmmF=(08 0C 18 1C 28 2C 38 3C 09 0D 19 1D 29 2D 39 3D)
    punpckhbw   xmmC, xmmH      ; xmmC=(0A 0E 1A 1E 2A 2E 3A 3E 0B 0F 1B 1F 2B 2F 3B 3F)

    movdqa      xmmB, xmmA
    punpcklwd   xmmA, xmmF      ; xmmA=(00 04 08 0C 10 14 18 1C 20 24 28 2C 30 34 38 3C)
    punpckhwd   xmmB, xmmF      ; xmmB=(01 05 09 0D 11 15 19 1D 21 25 29 2D 31 35 39 3D)

    movdqa      xmmG, xmmD
    punpcklwd   xmmD, xmmC      ; xmmD=(02 06 0A 0E 12 16 1A 1E 22 26 2A 2E 32 36 3A 3E)
    punpckhwd   xmmG, xmmC      ; xmmG=(03 07 0B 0F 13 17 1B 1F 23 27 2B 2F 33 37 3B 3F)

    movdqa      xmmE, xmmA
    punpcklbw   xmmA, xmmD      ; xmmA=(00 02 04 06 08 0A 0C 0E 10 12 14 16 18 1A 1C 1E)
    punpckhbw   xmmE, xmmD      ; xmmE=(20 22 24 26 28 2A 2C 2E 30 32 34 36 38 3A 3C 3E)

    movdqa      xmmH, xmmB
    punpcklbw   xmmB, xmmG      ; xmmB=(01 03 05 07 09 0B 0D 0F 11 13 15 17 19 1B 1D 1F)
    punpckhbw   xmmH, xmmG      ; xmmH=(21 23 25 27 29 2B 2D 2F 31 33 35 37 39 3B 3D 3F)

    pxor        xmmF, xmmF

    movdqa      xmmC, xmmA
    punpcklbw   xmmA, xmmF      ; xmmA=(00 02 04 06 08 0A 0C 0E)
    punpckhbw   xmmC, xmmF      ; xmmC=(10 12 14 16 18 1A 1C 1E)

    movdqa      xmmD, xmmB
    punpcklbw   xmmB, xmmF      ; xmmB=(01 03 05 07 09 0B 0D 0F)
    punpckhbw   xmmD, xmmF      ; xmmD=(11 13 15 17 19 1B 1D 1F)

    movdqa      xmmG, xmmE
    punpcklbw   xmmE, xmmF      ; xmmE=(20 22 24 26 28 2A 2C 2E)
    punpckhbw   xmmG, xmmF      ; xmmG=(30 32 34 36 38 3A 3C 3E)

    punpcklbw   xmmF, xmmH
    punpckhbw   xmmH, xmmH
    psrlw       xmmF, BYTE_BIT  ; xmmF=(21 23 25 27 29 2B 2D 2F)
    psrlw       xmmH, BYTE_BIT  ; xmmH=(31 33 35 37 39 3B 3D 3F)

%endif  ; RGB_PIXELSIZE ; ---------------

    ; xmm0=R(02468ACE)=RE, xmm2=G(02468ACE)=GE, xmm4=B(02468ACE)=BE
    ; xmm1=R(13579BDF)=RO, xmm3=G(13579BDF)=GO, xmm5=B(13579BDF)=BO

    ; (Original)
    ; Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    ;
    ; (This implementation)
    ; Y  =  0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G

    movdqa      xmm6, xmm1
    punpcklwd   xmm1, xmm3
    punpckhwd   xmm6, xmm3
    pmaddwd     xmm1, [GOTOFF(eax,PW_F0299_F0337)]  ; xmm1=ROL*FIX(0.299)+GOL*FIX(0.337)
    pmaddwd     xmm6, [GOTOFF(eax,PW_F0299_F0337)]  ; xmm6=ROH*FIX(0.299)+GOH*FIX(0.337)

    movdqa      xmm7, xmm6              ; xmm7=ROH*FIX(0.299)+GOH*FIX(0.337)

    movdqa      xmm6, xmm0
    punpcklwd   xmm0, xmm2
    punpckhwd   xmm6, xmm2
    pmaddwd     xmm0, [GOTOFF(eax,PW_F0299_F0337)]  ; xmm0=REL*FIX(0.299)+GEL*FIX(0.337)
    pmaddwd     xmm6, [GOTOFF(eax,PW_F0299_F0337)]  ; xmm6=REH*FIX(0.299)+GEH*FIX(0.337)

    movdqa      XMMWORD [wk(0)], xmm0   ; wk(0)=REL*FIX(0.299)+GEL*FIX(0.337)
    movdqa      XMMWORD [wk(1)], xmm6   ; wk(1)=REH*FIX(0.299)+GEH*FIX(0.337)

    movdqa      xmm0, xmm5              ; xmm0=BO
    movdqa      xmm6, xmm4              ; xmm6=BE

    movdqa      xmm4, xmm0
    punpcklwd   xmm0, xmm3
    punpckhwd   xmm4, xmm3
    pmaddwd     xmm0, [GOTOFF(eax,PW_F0114_F0250)]  ; xmm0=BOL*FIX(0.114)+GOL*FIX(0.250)
    pmaddwd     xmm4, [GOTOFF(eax,PW_F0114_F0250)]  ; xmm4=BOH*FIX(0.114)+GOH*FIX(0.250)

    movdqa      xmm3, [GOTOFF(eax,PD_ONEHALF)]      ; xmm3=[PD_ONEHALF]

    paddd       xmm0, xmm1
    paddd       xmm4, xmm7
    paddd       xmm0, xmm3
    paddd       xmm4, xmm3
    psrld       xmm0, SCALEBITS         ; xmm0=YOL
    psrld       xmm4, SCALEBITS         ; xmm4=YOH
    packssdw    xmm0, xmm4              ; xmm0=YO

    movdqa      xmm4, xmm6
    punpcklwd   xmm6, xmm2
    punpckhwd   xmm4, xmm2
    pmaddwd     xmm6, [GOTOFF(eax,PW_F0114_F0250)]  ; xmm6=BEL*FIX(0.114)+GEL*FIX(0.250)
    pmaddwd     xmm4, [GOTOFF(eax,PW_F0114_F0250)]  ; xmm4=BEH*FIX(0.114)+GEH*FIX(0.250)

    movdqa      xmm2, [GOTOFF(eax,PD_ONEHALF)]      ; xmm2=[PD_ONEHALF]

    paddd       xmm6, XMMWORD [wk(0)]
    paddd       xmm4, XMMWORD [wk(1)]
    paddd       xmm6, xmm2
    paddd       xmm4, xmm2
    psrld       xmm6, SCALEBITS         ; xmm6=YEL
    psrld       xmm4, SCALEBITS         ; xmm4=YEH
    packssdw    xmm6, xmm4              ; xmm6=YE

    psllw       xmm0, BYTE_BIT
    por         xmm6, xmm0              ; xmm6=Y
    movdqa      XMMWORD [edi], xmm6     ; Save Y

    sub         ecx, byte SIZEOF_XMMWORD
    add         esi, byte RGB_PIXELSIZE*SIZEOF_XMMWORD  ; inptr
    add         edi, byte SIZEOF_XMMWORD                ; outptr0
    cmp         ecx, byte SIZEOF_XMMWORD
    jae         near .columnloop
    test        ecx, ecx
    jnz         near .column_ld1

    pop         ecx                     ; col
    pop         esi
    pop         edi
    POPPIC      eax

    add         esi, byte SIZEOF_JSAMPROW  ; input_buf
    add         edi, byte SIZEOF_JSAMPROW
    dec         eax                        ; num_rows
    jg          near .rowloop

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

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
