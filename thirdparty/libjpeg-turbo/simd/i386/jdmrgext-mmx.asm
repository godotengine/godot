;
; jdmrgext.asm - merged upsampling/color conversion (MMX)
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
; Upsample and color convert for the case of 2:1 horizontal and 1:1 vertical.
;
; GLOBAL(void)
; jsimd_h2v1_merged_upsample_mmx(JDIMENSION output_width, JSAMPIMAGE input_buf,
;                                JDIMENSION in_row_group_ctr,
;                                JSAMPARRAY output_buf);
;

%define output_width(b)      (b) + 8    ; JDIMENSION output_width
%define input_buf(b)         (b) + 12   ; JSAMPIMAGE input_buf
%define in_row_group_ctr(b)  (b) + 16   ; JDIMENSION in_row_group_ctr
%define output_buf(b)        (b) + 20   ; JSAMPARRAY output_buf

%define original_ebp  ebp + 0
%define wk(i)         ebp - (WK_NUM - (i)) * SIZEOF_MMWORD  ; mmword wk[WK_NUM]
%define WK_NUM        3
%define gotptr        wk(0) - SIZEOF_POINTER  ; void * gotptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_merged_upsample_mmx)

EXTN(jsimd_h2v1_merged_upsample_mmx):
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

    mov         ecx, JDIMENSION [output_width(eax)]  ; col
    test        ecx, ecx
    jz          near .return

    push        ecx

    mov         edi, JSAMPIMAGE [input_buf(eax)]
    mov         ecx, JDIMENSION [in_row_group_ctr(eax)]
    mov         esi, JSAMPARRAY [edi+0*SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [edi+1*SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [edi+2*SIZEOF_JSAMPARRAY]
    mov         edi, JSAMPARRAY [output_buf(eax)]
    mov         esi, JSAMPROW [esi+ecx*SIZEOF_JSAMPROW]  ; inptr0
    mov         ebx, JSAMPROW [ebx+ecx*SIZEOF_JSAMPROW]  ; inptr1
    mov         edx, JSAMPROW [edx+ecx*SIZEOF_JSAMPROW]  ; inptr2
    mov         edi, JSAMPROW [edi]                      ; outptr

    pop         ecx                     ; col

    ALIGNX      16, 7
.columnloop:
    MOVPIC      eax, POINTER [gotptr]   ; load GOT address (eax)

    movq        mm6, MMWORD [ebx]       ; mm6=Cb(01234567)
    movq        mm7, MMWORD [edx]       ; mm7=Cr(01234567)

    pxor        mm1, mm1                ; mm1=(all 0's)
    pcmpeqw     mm3, mm3
    psllw       mm3, 7                  ; mm3={0xFF80 0xFF80 0xFF80 0xFF80}

    movq        mm4, mm6
    punpckhbw   mm6, mm1                ; mm6=Cb(4567)=CbH
    punpcklbw   mm4, mm1                ; mm4=Cb(0123)=CbL
    movq        mm0, mm7
    punpckhbw   mm7, mm1                ; mm7=Cr(4567)=CrH
    punpcklbw   mm0, mm1                ; mm0=Cr(0123)=CrL

    paddw       mm6, mm3
    paddw       mm4, mm3
    paddw       mm7, mm3
    paddw       mm0, mm3

    ; (Original)
    ; R = Y                + 1.40200 * Cr
    ; G = Y - 0.34414 * Cb - 0.71414 * Cr
    ; B = Y + 1.77200 * Cb
    ;
    ; (This implementation)
    ; R = Y                + 0.40200 * Cr + Cr
    ; G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
    ; B = Y - 0.22800 * Cb + Cb + Cb

    movq        mm5, mm6                ; mm5=CbH
    movq        mm2, mm4                ; mm2=CbL
    paddw       mm6, mm6                ; mm6=2*CbH
    paddw       mm4, mm4                ; mm4=2*CbL
    movq        mm1, mm7                ; mm1=CrH
    movq        mm3, mm0                ; mm3=CrL
    paddw       mm7, mm7                ; mm7=2*CrH
    paddw       mm0, mm0                ; mm0=2*CrL

    pmulhw      mm6, [GOTOFF(eax,PW_MF0228)]  ; mm6=(2*CbH * -FIX(0.22800))
    pmulhw      mm4, [GOTOFF(eax,PW_MF0228)]  ; mm4=(2*CbL * -FIX(0.22800))
    pmulhw      mm7, [GOTOFF(eax,PW_F0402)]   ; mm7=(2*CrH * FIX(0.40200))
    pmulhw      mm0, [GOTOFF(eax,PW_F0402)]   ; mm0=(2*CrL * FIX(0.40200))

    paddw       mm6, [GOTOFF(eax,PW_ONE)]
    paddw       mm4, [GOTOFF(eax,PW_ONE)]
    psraw       mm6, 1                  ; mm6=(CbH * -FIX(0.22800))
    psraw       mm4, 1                  ; mm4=(CbL * -FIX(0.22800))
    paddw       mm7, [GOTOFF(eax,PW_ONE)]
    paddw       mm0, [GOTOFF(eax,PW_ONE)]
    psraw       mm7, 1                  ; mm7=(CrH * FIX(0.40200))
    psraw       mm0, 1                  ; mm0=(CrL * FIX(0.40200))

    paddw       mm6, mm5
    paddw       mm4, mm2
    paddw       mm6, mm5                ; mm6=(CbH * FIX(1.77200))=(B-Y)H
    paddw       mm4, mm2                ; mm4=(CbL * FIX(1.77200))=(B-Y)L
    paddw       mm7, mm1                ; mm7=(CrH * FIX(1.40200))=(R-Y)H
    paddw       mm0, mm3                ; mm0=(CrL * FIX(1.40200))=(R-Y)L

    movq        MMWORD [wk(0)], mm6     ; wk(0)=(B-Y)H
    movq        MMWORD [wk(1)], mm7     ; wk(1)=(R-Y)H

    movq        mm6, mm5
    movq        mm7, mm2
    punpcklwd   mm5, mm1
    punpckhwd   mm6, mm1
    pmaddwd     mm5, [GOTOFF(eax,PW_MF0344_F0285)]
    pmaddwd     mm6, [GOTOFF(eax,PW_MF0344_F0285)]
    punpcklwd   mm2, mm3
    punpckhwd   mm7, mm3
    pmaddwd     mm2, [GOTOFF(eax,PW_MF0344_F0285)]
    pmaddwd     mm7, [GOTOFF(eax,PW_MF0344_F0285)]

    paddd       mm5, [GOTOFF(eax,PD_ONEHALF)]
    paddd       mm6, [GOTOFF(eax,PD_ONEHALF)]
    psrad       mm5, SCALEBITS
    psrad       mm6, SCALEBITS
    paddd       mm2, [GOTOFF(eax,PD_ONEHALF)]
    paddd       mm7, [GOTOFF(eax,PD_ONEHALF)]
    psrad       mm2, SCALEBITS
    psrad       mm7, SCALEBITS

    packssdw    mm5, mm6                ; mm5=CbH*-FIX(0.344)+CrH*FIX(0.285)
    packssdw    mm2, mm7                ; mm2=CbL*-FIX(0.344)+CrL*FIX(0.285)
    psubw       mm5, mm1                ; mm5=CbH*-FIX(0.344)+CrH*-FIX(0.714)=(G-Y)H
    psubw       mm2, mm3                ; mm2=CbL*-FIX(0.344)+CrL*-FIX(0.714)=(G-Y)L

    movq        MMWORD [wk(2)], mm5     ; wk(2)=(G-Y)H

    mov         al, 2                   ; Yctr
    jmp         short .Yloop_1st
    ALIGNX      16, 7

.Yloop_2nd:
    movq        mm0, MMWORD [wk(1)]     ; mm0=(R-Y)H
    movq        mm2, MMWORD [wk(2)]     ; mm2=(G-Y)H
    movq        mm4, MMWORD [wk(0)]     ; mm4=(B-Y)H
    ALIGNX      16, 7

.Yloop_1st:
    movq        mm7, MMWORD [esi]       ; mm7=Y(01234567)

    pcmpeqw     mm6, mm6
    psrlw       mm6, BYTE_BIT           ; mm6={0xFF 0x00 0xFF 0x00 ..}
    pand        mm6, mm7                ; mm6=Y(0246)=YE
    psrlw       mm7, BYTE_BIT           ; mm7=Y(1357)=YO

    movq        mm1, mm0                ; mm1=mm0=(R-Y)(L/H)
    movq        mm3, mm2                ; mm3=mm2=(G-Y)(L/H)
    movq        mm5, mm4                ; mm5=mm4=(B-Y)(L/H)

    paddw       mm0, mm6                ; mm0=((R-Y)+YE)=RE=(R0 R2 R4 R6)
    paddw       mm1, mm7                ; mm1=((R-Y)+YO)=RO=(R1 R3 R5 R7)
    packuswb    mm0, mm0                ; mm0=(R0 R2 R4 R6 ** ** ** **)
    packuswb    mm1, mm1                ; mm1=(R1 R3 R5 R7 ** ** ** **)

    paddw       mm2, mm6                ; mm2=((G-Y)+YE)=GE=(G0 G2 G4 G6)
    paddw       mm3, mm7                ; mm3=((G-Y)+YO)=GO=(G1 G3 G5 G7)
    packuswb    mm2, mm2                ; mm2=(G0 G2 G4 G6 ** ** ** **)
    packuswb    mm3, mm3                ; mm3=(G1 G3 G5 G7 ** ** ** **)

    paddw       mm4, mm6                ; mm4=((B-Y)+YE)=BE=(B0 B2 B4 B6)
    paddw       mm5, mm7                ; mm5=((B-Y)+YO)=BO=(B1 B3 B5 B7)
    packuswb    mm4, mm4                ; mm4=(B0 B2 B4 B6 ** ** ** **)
    packuswb    mm5, mm5                ; mm5=(B1 B3 B5 B7 ** ** ** **)

%if RGB_PIXELSIZE == 3  ; ---------------

    ; mmA=(00 02 04 06 ** ** ** **), mmB=(01 03 05 07 ** ** ** **)
    ; mmC=(10 12 14 16 ** ** ** **), mmD=(11 13 15 17 ** ** ** **)
    ; mmE=(20 22 24 26 ** ** ** **), mmF=(21 23 25 27 ** ** ** **)
    ; mmG=(** ** ** ** ** ** ** **), mmH=(** ** ** ** ** ** ** **)

    punpcklbw   mmA, mmC                ; mmA=(00 10 02 12 04 14 06 16)
    punpcklbw   mmE, mmB                ; mmE=(20 01 22 03 24 05 26 07)
    punpcklbw   mmD, mmF                ; mmD=(11 21 13 23 15 25 17 27)

    movq        mmG, mmA
    movq        mmH, mmA
    punpcklwd   mmA, mmE                ; mmA=(00 10 20 01 02 12 22 03)
    punpckhwd   mmG, mmE                ; mmG=(04 14 24 05 06 16 26 07)

    psrlq       mmH, 2*BYTE_BIT         ; mmH=(02 12 04 14 06 16 -- --)
    psrlq       mmE, 2*BYTE_BIT         ; mmE=(22 03 24 05 26 07 -- --)

    movq        mmC, mmD
    movq        mmB, mmD
    punpcklwd   mmD, mmH                ; mmD=(11 21 02 12 13 23 04 14)
    punpckhwd   mmC, mmH                ; mmC=(15 25 06 16 17 27 -- --)

    psrlq       mmB, 2*BYTE_BIT         ; mmB=(13 23 15 25 17 27 -- --)

    movq        mmF, mmE
    punpcklwd   mmE, mmB                ; mmE=(22 03 13 23 24 05 15 25)
    punpckhwd   mmF, mmB                ; mmF=(26 07 17 27 -- -- -- --)

    punpckldq   mmA, mmD                ; mmA=(00 10 20 01 11 21 02 12)
    punpckldq   mmE, mmG                ; mmE=(22 03 13 23 04 14 24 05)
    punpckldq   mmC, mmF                ; mmC=(15 25 06 16 26 07 17 27)

    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st16

    movq        MMWORD [edi+0*SIZEOF_MMWORD], mmA
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mmE
    movq        MMWORD [edi+2*SIZEOF_MMWORD], mmC

    sub         ecx, byte SIZEOF_MMWORD
    jz          near .endcolumn

    add         edi, byte RGB_PIXELSIZE*SIZEOF_MMWORD  ; outptr
    add         esi, byte SIZEOF_MMWORD                ; inptr0
    dec         al                                     ; Yctr
    jnz         near .Yloop_2nd

    add         ebx, byte SIZEOF_MMWORD                ; inptr1
    add         edx, byte SIZEOF_MMWORD                ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st16:
    lea         ecx, [ecx+ecx*2]        ; imul ecx, RGB_PIXELSIZE
    cmp         ecx, byte 2*SIZEOF_MMWORD
    jb          short .column_st8
    movq        MMWORD [edi+0*SIZEOF_MMWORD], mmA
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mmE
    movq        mmA, mmC
    sub         ecx, byte 2*SIZEOF_MMWORD
    add         edi, byte 2*SIZEOF_MMWORD
    jmp         short .column_st4
.column_st8:
    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st4
    movq        MMWORD [edi+0*SIZEOF_MMWORD], mmA
    movq        mmA, mmE
    sub         ecx, byte SIZEOF_MMWORD
    add         edi, byte SIZEOF_MMWORD
.column_st4:
    movd        eax, mmA
    cmp         ecx, byte SIZEOF_DWORD
    jb          short .column_st2
    mov         dword [edi+0*SIZEOF_DWORD], eax
    psrlq       mmA, DWORD_BIT
    movd        eax, mmA
    sub         ecx, byte SIZEOF_DWORD
    add         edi, byte SIZEOF_DWORD
.column_st2:
    cmp         ecx, byte SIZEOF_WORD
    jb          short .column_st1
    mov         word [edi+0*SIZEOF_WORD], ax
    shr         eax, WORD_BIT
    sub         ecx, byte SIZEOF_WORD
    add         edi, byte SIZEOF_WORD
.column_st1:
    cmp         ecx, byte SIZEOF_BYTE
    jb          short .endcolumn
    mov         byte [edi+0*SIZEOF_BYTE], al

%else  ; RGB_PIXELSIZE == 4 ; -----------

%ifdef RGBX_FILLER_0XFF
    pcmpeqb     mm6, mm6                ; mm6=(X0 X2 X4 X6 ** ** ** **)
    pcmpeqb     mm7, mm7                ; mm7=(X1 X3 X5 X7 ** ** ** **)
%else
    pxor        mm6, mm6                ; mm6=(X0 X2 X4 X6 ** ** ** **)
    pxor        mm7, mm7                ; mm7=(X1 X3 X5 X7 ** ** ** **)
%endif
    ; mmA=(00 02 04 06 ** ** ** **), mmB=(01 03 05 07 ** ** ** **)
    ; mmC=(10 12 14 16 ** ** ** **), mmD=(11 13 15 17 ** ** ** **)
    ; mmE=(20 22 24 26 ** ** ** **), mmF=(21 23 25 27 ** ** ** **)
    ; mmG=(30 32 34 36 ** ** ** **), mmH=(31 33 35 37 ** ** ** **)

    punpcklbw   mmA, mmC                ; mmA=(00 10 02 12 04 14 06 16)
    punpcklbw   mmE, mmG                ; mmE=(20 30 22 32 24 34 26 36)
    punpcklbw   mmB, mmD                ; mmB=(01 11 03 13 05 15 07 17)
    punpcklbw   mmF, mmH                ; mmF=(21 31 23 33 25 35 27 37)

    movq        mmC, mmA
    punpcklwd   mmA, mmE                ; mmA=(00 10 20 30 02 12 22 32)
    punpckhwd   mmC, mmE                ; mmC=(04 14 24 34 06 16 26 36)
    movq        mmG, mmB
    punpcklwd   mmB, mmF                ; mmB=(01 11 21 31 03 13 23 33)
    punpckhwd   mmG, mmF                ; mmG=(05 15 25 35 07 17 27 37)

    movq        mmD, mmA
    punpckldq   mmA, mmB                ; mmA=(00 10 20 30 01 11 21 31)
    punpckhdq   mmD, mmB                ; mmD=(02 12 22 32 03 13 23 33)
    movq        mmH, mmC
    punpckldq   mmC, mmG                ; mmC=(04 14 24 34 05 15 25 35)
    punpckhdq   mmH, mmG                ; mmH=(06 16 26 36 07 17 27 37)

    cmp         ecx, byte SIZEOF_MMWORD
    jb          short .column_st16

    movq        MMWORD [edi+0*SIZEOF_MMWORD], mmA
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mmD
    movq        MMWORD [edi+2*SIZEOF_MMWORD], mmC
    movq        MMWORD [edi+3*SIZEOF_MMWORD], mmH

    sub         ecx, byte SIZEOF_MMWORD
    jz          short .endcolumn

    add         edi, byte RGB_PIXELSIZE*SIZEOF_MMWORD  ; outptr
    add         esi, byte SIZEOF_MMWORD                ; inptr0
    dec         al                                     ; Yctr
    jnz         near .Yloop_2nd

    add         ebx, byte SIZEOF_MMWORD                ; inptr1
    add         edx, byte SIZEOF_MMWORD                ; inptr2
    jmp         near .columnloop
    ALIGNX      16, 7

.column_st16:
    cmp         ecx, byte SIZEOF_MMWORD/2
    jb          short .column_st8
    movq        MMWORD [edi+0*SIZEOF_MMWORD], mmA
    movq        MMWORD [edi+1*SIZEOF_MMWORD], mmD
    movq        mmA, mmC
    movq        mmD, mmH
    sub         ecx, byte SIZEOF_MMWORD/2
    add         edi, byte 2*SIZEOF_MMWORD
.column_st8:
    cmp         ecx, byte SIZEOF_MMWORD/4
    jb          short .column_st4
    movq        MMWORD [edi+0*SIZEOF_MMWORD], mmA
    movq        mmA, mmD
    sub         ecx, byte SIZEOF_MMWORD/4
    add         edi, byte 1*SIZEOF_MMWORD
.column_st4:
    cmp         ecx, byte SIZEOF_MMWORD/8
    jb          short .endcolumn
    movd        dword [edi+0*SIZEOF_DWORD], mmA

%endif  ; RGB_PIXELSIZE ; ---------------

.endcolumn:
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
; Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
;
; GLOBAL(void)
; jsimd_h2v2_merged_upsample_mmx(JDIMENSION output_width, JSAMPIMAGE input_buf,
;                                JDIMENSION in_row_group_ctr,
;                                JSAMPARRAY output_buf);
;

%define output_width(b)      (b) + 8    ; JDIMENSION output_width
%define input_buf(b)         (b) + 12   ; JSAMPIMAGE input_buf
%define in_row_group_ctr(b)  (b) + 16   ; JDIMENSION in_row_group_ctr
%define output_buf(b)        (b) + 20   ; JSAMPARRAY output_buf

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_merged_upsample_mmx)

EXTN(jsimd_h2v2_merged_upsample_mmx):
    push        ebp
    mov         ebp, esp
    push        ebx
;   push        ecx                     ; need not be preserved
;   push        edx                     ; need not be preserved
    push        esi
    push        edi

    mov         eax, JDIMENSION [output_width(ebp)]

    mov         edi, JSAMPIMAGE [input_buf(ebp)]
    mov         ecx, JDIMENSION [in_row_group_ctr(ebp)]
    mov         esi, JSAMPARRAY [edi+0*SIZEOF_JSAMPARRAY]
    mov         ebx, JSAMPARRAY [edi+1*SIZEOF_JSAMPARRAY]
    mov         edx, JSAMPARRAY [edi+2*SIZEOF_JSAMPARRAY]
    mov         edi, JSAMPARRAY [output_buf(ebp)]
    lea         esi, [esi+ecx*SIZEOF_JSAMPROW]

    push        edx                     ; inptr2
    push        ebx                     ; inptr1
    push        esi                     ; inptr00
    mov         ebx, esp

    push        edi                     ; output_buf (outptr0)
    push        ecx                     ; in_row_group_ctr
    push        ebx                     ; input_buf
    push        eax                     ; output_width

    call        near EXTN(jsimd_h2v1_merged_upsample_mmx)

    add         esi, byte SIZEOF_JSAMPROW  ; inptr01
    add         edi, byte SIZEOF_JSAMPROW  ; outptr1
    mov         POINTER [ebx+0*SIZEOF_POINTER], esi
    mov         POINTER [ebx-1*SIZEOF_POINTER], edi

    call        near EXTN(jsimd_h2v1_merged_upsample_mmx)

    add         esp, byte 7*SIZEOF_DWORD

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
