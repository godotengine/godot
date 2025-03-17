;
; jccolext.asm - colorspace conversion (MMX)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2016, 2024, D. R. Commander.
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
; jsimd_rgb_ycc_convert_mmx(JDIMENSION img_width, JSAMPARRAY input_buf,
;                           JSAMPIMAGE output_buf, JDIMENSION output_row,
;                           int num_rows);
;

%define img_width(b)   (b) + 8          ; JDIMENSION img_width
%define input_buf(b)   (b) + 12         ; JSAMPARRAY input_buf
%define output_buf(b)  (b) + 16         ; JSAMPIMAGE output_buf
%define output_row(b)  (b) + 20         ; JDIMENSION output_row
%define num_rows(b)    (b) + 24         ; int num_rows

%define original_ebp   ebp + 0
%define wk(i)          ebp - (WK_NUM - (i)) * SIZEOF_MMWORD
                                        ; mmword wk[WK_NUM]
%define WK_NUM         8
%define gotptr         wk(0) - SIZEOF_POINTER  ; void * gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_rgb_ycc_convert_mmx)

EXTN(jsimd_rgb_ycc_convert_mmx):
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

    mov         ecx, JDIMENSION [img_width(eax)]  ; num_cols
    test        ecx, ecx
    jz          near .return

    push        ecx

    mov         esi, JSAMPIMAGE [output_buf(eax)]
    mov         ecx, JDIMENSION [output_row(eax)]
    mov         edi, JSAMPARRAY [esi+0*SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [esi+1*SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [esi+2*SIZEOF_JSAMPARRAY]
    lea         edi, [edi+ecx*SIZEOF_JSAMPROW]
    lea         ebx, [ebx+ecx*SIZEOF_JSAMPROW]
    lea         edx, [edx+ecx*SIZEOF_JSAMPROW]

    pop         ecx

    mov         esi, JSAMPARRAY [input_buf(eax)]
    mov         eax, INT [num_rows(eax)]
    test        eax, eax
    jle         near .return
    ALIGNX      16, 7
.rowloop:
    PUSHPIC     eax
    push        edx
    push        ebx
    push        edi
    push        esi
    push        ecx                     ; col

    mov         esi, JSAMPROW [esi]     ; inptr
    mov         edi, JSAMPROW [edi]     ; outptr0
    mov         ebx, JSAMPROW [ebx]     ; outptr1
    mov         edx, JSAMPROW [edx]     ; outptr2
    MOVPIC      eax, POINTER [gotptr]   ; load GOT address (eax)

    cmp         ecx, byte SIZEOF_MMWORD
    jae         short .columnloop
    ALIGNX      16, 7

%if RGB_PIXELSIZE == 3  ; ---------------

.column_ld1:
    push        eax
    push        edx
    lea         ecx, [ecx+ecx*2]        ; imul ecx,RGB_PIXELSIZE
    test        cl, SIZEOF_BYTE
    jz          short .column_ld2
    sub         ecx, byte SIZEOF_BYTE
    xor         eax, eax
    mov         al, byte [esi+ecx]
.column_ld2:
    test        cl, SIZEOF_WORD
    jz          short .column_ld4
    sub         ecx, byte SIZEOF_WORD
    xor         edx, edx
    mov         dx, word [esi+ecx]
    shl         eax, WORD_BIT
    or          eax, edx
.column_ld4:
    movd        mmA, eax
    pop         edx
    pop         eax
    test        cl, SIZEOF_DWORD
    jz          short .column_ld8
    sub         ecx, byte SIZEOF_DWORD
    movd        mmG, dword [esi+ecx]
    psllq       mmA, DWORD_BIT
    por         mmA, mmG
.column_ld8:
    test        cl, SIZEOF_MMWORD
    jz          short .column_ld16
    movq        mmG, mmA
    movq        mmA, MMWORD [esi+0*SIZEOF_MMWORD]
    mov         ecx, SIZEOF_MMWORD
    jmp         short .rgb_ycc_cnv
.column_ld16:
    test        cl, 2*SIZEOF_MMWORD
    mov         ecx, SIZEOF_MMWORD
    jz          short .rgb_ycc_cnv
    movq        mmF, mmA
    movq        mmA, MMWORD [esi+0*SIZEOF_MMWORD]
    movq        mmG, MMWORD [esi+1*SIZEOF_MMWORD]
    jmp         short .rgb_ycc_cnv
    ALIGNX      16, 7

.columnloop:
    movq        mmA, MMWORD [esi+0*SIZEOF_MMWORD]
    movq        mmG, MMWORD [esi+1*SIZEOF_MMWORD]
    movq        mmF, MMWORD [esi+2*SIZEOF_MMWORD]

.rgb_ycc_cnv:
    ; mmA=(00 10 20 01 11 21 02 12)
    ; mmG=(22 03 13 23 04 14 24 05)
    ; mmF=(15 25 06 16 26 07 17 27)

    movq        mmD, mmA
    psllq       mmA, 4*BYTE_BIT         ; mmA=(-- -- -- -- 00 10 20 01)
    psrlq       mmD, 4*BYTE_BIT         ; mmD=(11 21 02 12 -- -- -- --)

    punpckhbw   mmA, mmG                ; mmA=(00 04 10 14 20 24 01 05)
    psllq       mmG, 4*BYTE_BIT         ; mmG=(-- -- -- -- 22 03 13 23)

    punpcklbw   mmD, mmF                ; mmD=(11 15 21 25 02 06 12 16)
    punpckhbw   mmG, mmF                ; mmG=(22 26 03 07 13 17 23 27)

    movq        mmE, mmA
    psllq       mmA, 4*BYTE_BIT         ; mmA=(-- -- -- -- 00 04 10 14)
    psrlq       mmE, 4*BYTE_BIT         ; mmE=(20 24 01 05 -- -- -- --)

    punpckhbw   mmA, mmD                ; mmA=(00 02 04 06 10 12 14 16)
    psllq       mmD, 4*BYTE_BIT         ; mmD=(-- -- -- -- 11 15 21 25)

    punpcklbw   mmE, mmG                ; mmE=(20 22 24 26 01 03 05 07)
    punpckhbw   mmD, mmG                ; mmD=(11 13 15 17 21 23 25 27)

    pxor        mmH, mmH

    movq        mmC, mmA
    punpcklbw   mmA, mmH                ; mmA=(00 02 04 06)
    punpckhbw   mmC, mmH                ; mmC=(10 12 14 16)

    movq        mmB, mmE
    punpcklbw   mmE, mmH                ; mmE=(20 22 24 26)
    punpckhbw   mmB, mmH                ; mmB=(01 03 05 07)

    movq        mmF, mmD
    punpcklbw   mmD, mmH                ; mmD=(11 13 15 17)
    punpckhbw   mmF, mmH                ; mmF=(21 23 25 27)

%else  ; RGB_PIXELSIZE == 4 ; -----------

.column_ld1:
    test        cl, SIZEOF_MMWORD/8
    jz          short .column_ld2
    sub         ecx, byte SIZEOF_MMWORD/8
    movd        mmA, dword [esi+ecx*RGB_PIXELSIZE]
.column_ld2:
    test        cl, SIZEOF_MMWORD/4
    jz          short .column_ld4
    sub         ecx, byte SIZEOF_MMWORD/4
    movq        mmF, mmA
    movq        mmA, MMWORD [esi+ecx*RGB_PIXELSIZE]
.column_ld4:
    test        cl, SIZEOF_MMWORD/2
    mov         ecx, SIZEOF_MMWORD
    jz          short .rgb_ycc_cnv
    movq        mmD, mmA
    movq        mmC, mmF
    movq        mmA, MMWORD [esi+0*SIZEOF_MMWORD]
    movq        mmF, MMWORD [esi+1*SIZEOF_MMWORD]
    jmp         short .rgb_ycc_cnv
    ALIGNX      16, 7

.columnloop:
    movq        mmA, MMWORD [esi+0*SIZEOF_MMWORD]
    movq        mmF, MMWORD [esi+1*SIZEOF_MMWORD]
    movq        mmD, MMWORD [esi+2*SIZEOF_MMWORD]
    movq        mmC, MMWORD [esi+3*SIZEOF_MMWORD]

.rgb_ycc_cnv:
    ; mmA=(00 10 20 30 01 11 21 31)
    ; mmF=(02 12 22 32 03 13 23 33)
    ; mmD=(04 14 24 34 05 15 25 35)
    ; mmC=(06 16 26 36 07 17 27 37)

    movq        mmB, mmA
    punpcklbw   mmA, mmF                ; mmA=(00 02 10 12 20 22 30 32)
    punpckhbw   mmB, mmF                ; mmB=(01 03 11 13 21 23 31 33)

    movq        mmG, mmD
    punpcklbw   mmD, mmC                ; mmD=(04 06 14 16 24 26 34 36)
    punpckhbw   mmG, mmC                ; mmG=(05 07 15 17 25 27 35 37)

    movq        mmE, mmA
    punpcklwd   mmA, mmD                ; mmA=(00 02 04 06 10 12 14 16)
    punpckhwd   mmE, mmD                ; mmE=(20 22 24 26 30 32 34 36)

    movq        mmH, mmB
    punpcklwd   mmB, mmG                ; mmB=(01 03 05 07 11 13 15 17)
    punpckhwd   mmH, mmG                ; mmH=(21 23 25 27 31 33 35 37)

    pxor        mmF, mmF

    movq        mmC, mmA
    punpcklbw   mmA, mmF                ; mmA=(00 02 04 06)
    punpckhbw   mmC, mmF                ; mmC=(10 12 14 16)

    movq        mmD, mmB
    punpcklbw   mmB, mmF                ; mmB=(01 03 05 07)
    punpckhbw   mmD, mmF                ; mmD=(11 13 15 17)

    movq        mmG, mmE
    punpcklbw   mmE, mmF                ; mmE=(20 22 24 26)
    punpckhbw   mmG, mmF                ; mmG=(30 32 34 36)

    punpcklbw   mmF, mmH
    punpckhbw   mmH, mmH
    psrlw       mmF, BYTE_BIT           ; mmF=(21 23 25 27)
    psrlw       mmH, BYTE_BIT           ; mmH=(31 33 35 37)

%endif  ; RGB_PIXELSIZE ; ---------------

    ; mm0=(R0 R2 R4 R6)=RE, mm2=(G0 G2 G4 G6)=GE, mm4=(B0 B2 B4 B6)=BE
    ; mm1=(R1 R3 R5 R7)=RO, mm3=(G1 G3 G5 G7)=GO, mm5=(B1 B3 B5 B7)=BO

    ; (Original)
    ; Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    ; Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + CENTERJSAMPLE
    ; Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + CENTERJSAMPLE
    ;
    ; (This implementation)
    ; Y  =  0.29900 * R + 0.33700 * G + 0.11400 * B + 0.25000 * G
    ; Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B + CENTERJSAMPLE
    ; Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B + CENTERJSAMPLE

    movq        MMWORD [wk(0)], mm0     ; wk(0)=RE
    movq        MMWORD [wk(1)], mm1     ; wk(1)=RO
    movq        MMWORD [wk(2)], mm4     ; wk(2)=BE
    movq        MMWORD [wk(3)], mm5     ; wk(3)=BO

    movq        mm6, mm1
    punpcklwd   mm1, mm3
    punpckhwd   mm6, mm3
    movq        mm7, mm1
    movq        mm4, mm6
    pmaddwd     mm1, [GOTOFF(eax,PW_F0299_F0337)]  ; mm1=ROL*FIX(0.299)+GOL*FIX(0.337)
    pmaddwd     mm6, [GOTOFF(eax,PW_F0299_F0337)]  ; mm6=ROH*FIX(0.299)+GOH*FIX(0.337)
    pmaddwd     mm7, [GOTOFF(eax,PW_MF016_MF033)]  ; mm7=ROL*-FIX(0.168)+GOL*-FIX(0.331)
    pmaddwd     mm4, [GOTOFF(eax,PW_MF016_MF033)]  ; mm4=ROH*-FIX(0.168)+GOH*-FIX(0.331)

    movq        MMWORD [wk(4)], mm1     ; wk(4)=ROL*FIX(0.299)+GOL*FIX(0.337)
    movq        MMWORD [wk(5)], mm6     ; wk(5)=ROH*FIX(0.299)+GOH*FIX(0.337)

    pxor        mm1, mm1
    pxor        mm6, mm6
    punpcklwd   mm1, mm5                ; mm1=BOL
    punpckhwd   mm6, mm5                ; mm6=BOH
    psrld       mm1, 1                  ; mm1=BOL*FIX(0.500)
    psrld       mm6, 1                  ; mm6=BOH*FIX(0.500)

    movq        mm5, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; mm5=[PD_ONEHALFM1_CJ]

    paddd       mm7, mm1
    paddd       mm4, mm6
    paddd       mm7, mm5
    paddd       mm4, mm5
    psrld       mm7, SCALEBITS          ; mm7=CbOL
    psrld       mm4, SCALEBITS          ; mm4=CbOH
    packssdw    mm7, mm4                ; mm7=CbO

    movq        mm1, MMWORD [wk(2)]     ; mm1=BE

    movq        mm6, mm0
    punpcklwd   mm0, mm2
    punpckhwd   mm6, mm2
    movq        mm5, mm0
    movq        mm4, mm6
    pmaddwd     mm0, [GOTOFF(eax,PW_F0299_F0337)]  ; mm0=REL*FIX(0.299)+GEL*FIX(0.337)
    pmaddwd     mm6, [GOTOFF(eax,PW_F0299_F0337)]  ; mm6=REH*FIX(0.299)+GEH*FIX(0.337)
    pmaddwd     mm5, [GOTOFF(eax,PW_MF016_MF033)]  ; mm5=REL*-FIX(0.168)+GEL*-FIX(0.331)
    pmaddwd     mm4, [GOTOFF(eax,PW_MF016_MF033)]  ; mm4=REH*-FIX(0.168)+GEH*-FIX(0.331)

    movq        MMWORD [wk(6)], mm0     ; wk(6)=REL*FIX(0.299)+GEL*FIX(0.337)
    movq        MMWORD [wk(7)], mm6     ; wk(7)=REH*FIX(0.299)+GEH*FIX(0.337)

    pxor        mm0, mm0
    pxor        mm6, mm6
    punpcklwd   mm0, mm1                ; mm0=BEL
    punpckhwd   mm6, mm1                ; mm6=BEH
    psrld       mm0, 1                  ; mm0=BEL*FIX(0.500)
    psrld       mm6, 1                  ; mm6=BEH*FIX(0.500)

    movq        mm1, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; mm1=[PD_ONEHALFM1_CJ]

    paddd       mm5, mm0
    paddd       mm4, mm6
    paddd       mm5, mm1
    paddd       mm4, mm1
    psrld       mm5, SCALEBITS          ; mm5=CbEL
    psrld       mm4, SCALEBITS          ; mm4=CbEH
    packssdw    mm5, mm4                ; mm5=CbE

    psllw       mm7, BYTE_BIT
    por         mm5, mm7                ; mm5=Cb
    movq        MMWORD [ebx], mm5       ; Save Cb

    movq        mm0, MMWORD [wk(3)]     ; mm0=BO
    movq        mm6, MMWORD [wk(2)]     ; mm6=BE
    movq        mm1, MMWORD [wk(1)]     ; mm1=RO

    movq        mm4, mm0
    punpcklwd   mm0, mm3
    punpckhwd   mm4, mm3
    movq        mm7, mm0
    movq        mm5, mm4
    pmaddwd     mm0, [GOTOFF(eax,PW_F0114_F0250)]  ; mm0=BOL*FIX(0.114)+GOL*FIX(0.250)
    pmaddwd     mm4, [GOTOFF(eax,PW_F0114_F0250)]  ; mm4=BOH*FIX(0.114)+GOH*FIX(0.250)
    pmaddwd     mm7, [GOTOFF(eax,PW_MF008_MF041)]  ; mm7=BOL*-FIX(0.081)+GOL*-FIX(0.418)
    pmaddwd     mm5, [GOTOFF(eax,PW_MF008_MF041)]  ; mm5=BOH*-FIX(0.081)+GOH*-FIX(0.418)

    movq        mm3, [GOTOFF(eax,PD_ONEHALF)]  ; mm3=[PD_ONEHALF]

    paddd       mm0, MMWORD [wk(4)]
    paddd       mm4, MMWORD [wk(5)]
    paddd       mm0, mm3
    paddd       mm4, mm3
    psrld       mm0, SCALEBITS          ; mm0=YOL
    psrld       mm4, SCALEBITS          ; mm4=YOH
    packssdw    mm0, mm4                ; mm0=YO

    pxor        mm3, mm3
    pxor        mm4, mm4
    punpcklwd   mm3, mm1                ; mm3=ROL
    punpckhwd   mm4, mm1                ; mm4=ROH
    psrld       mm3, 1                  ; mm3=ROL*FIX(0.500)
    psrld       mm4, 1                  ; mm4=ROH*FIX(0.500)

    movq        mm1, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; mm1=[PD_ONEHALFM1_CJ]

    paddd       mm7, mm3
    paddd       mm5, mm4
    paddd       mm7, mm1
    paddd       mm5, mm1
    psrld       mm7, SCALEBITS          ; mm7=CrOL
    psrld       mm5, SCALEBITS          ; mm5=CrOH
    packssdw    mm7, mm5                ; mm7=CrO

    movq        mm3, MMWORD [wk(0)]     ; mm3=RE

    movq        mm4, mm6
    punpcklwd   mm6, mm2
    punpckhwd   mm4, mm2
    movq        mm1, mm6
    movq        mm5, mm4
    pmaddwd     mm6, [GOTOFF(eax,PW_F0114_F0250)]  ; mm6=BEL*FIX(0.114)+GEL*FIX(0.250)
    pmaddwd     mm4, [GOTOFF(eax,PW_F0114_F0250)]  ; mm4=BEH*FIX(0.114)+GEH*FIX(0.250)
    pmaddwd     mm1, [GOTOFF(eax,PW_MF008_MF041)]  ; mm1=BEL*-FIX(0.081)+GEL*-FIX(0.418)
    pmaddwd     mm5, [GOTOFF(eax,PW_MF008_MF041)]  ; mm5=BEH*-FIX(0.081)+GEH*-FIX(0.418)

    movq        mm2, [GOTOFF(eax,PD_ONEHALF)]      ; mm2=[PD_ONEHALF]

    paddd       mm6, MMWORD [wk(6)]
    paddd       mm4, MMWORD [wk(7)]
    paddd       mm6, mm2
    paddd       mm4, mm2
    psrld       mm6, SCALEBITS          ; mm6=YEL
    psrld       mm4, SCALEBITS          ; mm4=YEH
    packssdw    mm6, mm4                ; mm6=YE

    psllw       mm0, BYTE_BIT
    por         mm6, mm0                ; mm6=Y
    movq        MMWORD [edi], mm6       ; Save Y

    pxor        mm2, mm2
    pxor        mm4, mm4
    punpcklwd   mm2, mm3                ; mm2=REL
    punpckhwd   mm4, mm3                ; mm4=REH
    psrld       mm2, 1                  ; mm2=REL*FIX(0.500)
    psrld       mm4, 1                  ; mm4=REH*FIX(0.500)

    movq        mm0, [GOTOFF(eax,PD_ONEHALFM1_CJ)]  ; mm0=[PD_ONEHALFM1_CJ]

    paddd       mm1, mm2
    paddd       mm5, mm4
    paddd       mm1, mm0
    paddd       mm5, mm0
    psrld       mm1, SCALEBITS          ; mm1=CrEL
    psrld       mm5, SCALEBITS          ; mm5=CrEH
    packssdw    mm1, mm5                ; mm1=CrE

    psllw       mm7, BYTE_BIT
    por         mm1, mm7                ; mm1=Cr
    movq        MMWORD [edx], mm1       ; Save Cr

    sub         ecx, byte SIZEOF_MMWORD
    add         esi, byte RGB_PIXELSIZE*SIZEOF_MMWORD  ; inptr
    add         edi, byte SIZEOF_MMWORD                ; outptr0
    add         ebx, byte SIZEOF_MMWORD                ; outptr1
    add         edx, byte SIZEOF_MMWORD                ; outptr2
    cmp         ecx, byte SIZEOF_MMWORD
    jae         near .columnloop
    test        ecx, ecx
    jnz         near .column_ld1

    pop         ecx                     ; col
    pop         esi
    pop         edi
    pop         ebx
    pop         edx
    POPPIC      eax

    add         esi, byte SIZEOF_JSAMPROW  ; input_buf
    add         edi, byte SIZEOF_JSAMPROW
    add         ebx, byte SIZEOF_JSAMPROW
    add         edx, byte SIZEOF_JSAMPROW
    dec         eax                        ; num_rows
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

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
