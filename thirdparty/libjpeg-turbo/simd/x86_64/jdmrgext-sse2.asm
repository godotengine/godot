;
; jdmrgext.asm - merged upsampling/color conversion (64-bit SSE2)
;
; Copyright 2009, 2012 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2012, 2016, 2024, D. R. Commander.
; Copyright (C) 2018, Matthias Räncker.
; Copyright (C) 2023, Aliaksiej Kandracienka.
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
; jsimd_h2v1_merged_upsample_sse2(JDIMENSION output_width,
;                                 JSAMPIMAGE input_buf,
;                                 JDIMENSION in_row_group_ctr,
;                                 JSAMPARRAY output_buf);
;

; r10d = JDIMENSION output_width
; r11 = JSAMPIMAGE input_buf
; r12d = JDIMENSION in_row_group_ctr
; r13 = JSAMPARRAY output_buf

%define wk(i)   r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD  ; xmmword wk[WK_NUM]
%define WK_NUM  3

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_merged_upsample_sse2)

EXTN(jsimd_h2v1_merged_upsample_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    push        r15
    and         rsp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    ; Allocate stack space for wk array.  r15 is used to access it.
    mov         r15, rsp
    sub         rsp, byte (SIZEOF_XMMWORD * WK_NUM)
    COLLECT_ARGS 4
    push        rbx

    mov         ecx, r10d               ; col
    test        rcx, rcx
    jz          near .return

    push        rcx

    mov         rdi, r11
    mov         ecx, r12d
    mov         rsip, JSAMPARRAY [rdi+0*SIZEOF_JSAMPARRAY]
    mov         rbxp, JSAMPARRAY [rdi+1*SIZEOF_JSAMPARRAY]
    mov         rdxp, JSAMPARRAY [rdi+2*SIZEOF_JSAMPARRAY]
    mov         rdi, r13
    mov         rsip, JSAMPROW [rsi+rcx*SIZEOF_JSAMPROW]  ; inptr0
    mov         rbxp, JSAMPROW [rbx+rcx*SIZEOF_JSAMPROW]  ; inptr1
    mov         rdxp, JSAMPROW [rdx+rcx*SIZEOF_JSAMPROW]  ; inptr2
    mov         rdip, JSAMPROW [rdi]                      ; outptr

    pop         rcx                     ; col

.columnloop:

    movdqa      xmm6, XMMWORD [rbx]     ; xmm6=Cb(0123456789ABCDEF)
    movdqa      xmm7, XMMWORD [rdx]     ; xmm7=Cr(0123456789ABCDEF)

    pxor        xmm1, xmm1              ; xmm1=(all 0's)
    pcmpeqw     xmm3, xmm3
    psllw       xmm3, 7                 ; xmm3={0xFF80 0xFF80 0xFF80 0xFF80 ..}

    movdqa      xmm4, xmm6
    punpckhbw   xmm6, xmm1              ; xmm6=Cb(89ABCDEF)=CbH
    punpcklbw   xmm4, xmm1              ; xmm4=Cb(01234567)=CbL
    movdqa      xmm0, xmm7
    punpckhbw   xmm7, xmm1              ; xmm7=Cr(89ABCDEF)=CrH
    punpcklbw   xmm0, xmm1              ; xmm0=Cr(01234567)=CrL

    paddw       xmm6, xmm3
    paddw       xmm4, xmm3
    paddw       xmm7, xmm3
    paddw       xmm0, xmm3

    ; (Original)
    ; R = Y                + 1.40200 * Cr
    ; G = Y - 0.34414 * Cb - 0.71414 * Cr
    ; B = Y + 1.77200 * Cb
    ;
    ; (This implementation)
    ; R = Y                + 0.40200 * Cr + Cr
    ; G = Y - 0.34414 * Cb + 0.28586 * Cr - Cr
    ; B = Y - 0.22800 * Cb + Cb + Cb

    movdqa      xmm5, xmm6              ; xmm5=CbH
    movdqa      xmm2, xmm4              ; xmm2=CbL
    paddw       xmm6, xmm6              ; xmm6=2*CbH
    paddw       xmm4, xmm4              ; xmm4=2*CbL
    movdqa      xmm1, xmm7              ; xmm1=CrH
    movdqa      xmm3, xmm0              ; xmm3=CrL
    paddw       xmm7, xmm7              ; xmm7=2*CrH
    paddw       xmm0, xmm0              ; xmm0=2*CrL

    pmulhw      xmm6, [rel PW_MF0228]   ; xmm6=(2*CbH * -FIX(0.22800))
    pmulhw      xmm4, [rel PW_MF0228]   ; xmm4=(2*CbL * -FIX(0.22800))
    pmulhw      xmm7, [rel PW_F0402]    ; xmm7=(2*CrH * FIX(0.40200))
    pmulhw      xmm0, [rel PW_F0402]    ; xmm0=(2*CrL * FIX(0.40200))

    paddw       xmm6, [rel PW_ONE]
    paddw       xmm4, [rel PW_ONE]
    psraw       xmm6, 1                 ; xmm6=(CbH * -FIX(0.22800))
    psraw       xmm4, 1                 ; xmm4=(CbL * -FIX(0.22800))
    paddw       xmm7, [rel PW_ONE]
    paddw       xmm0, [rel PW_ONE]
    psraw       xmm7, 1                 ; xmm7=(CrH * FIX(0.40200))
    psraw       xmm0, 1                 ; xmm0=(CrL * FIX(0.40200))

    paddw       xmm6, xmm5
    paddw       xmm4, xmm2
    paddw       xmm6, xmm5              ; xmm6=(CbH * FIX(1.77200))=(B-Y)H
    paddw       xmm4, xmm2              ; xmm4=(CbL * FIX(1.77200))=(B-Y)L
    paddw       xmm7, xmm1              ; xmm7=(CrH * FIX(1.40200))=(R-Y)H
    paddw       xmm0, xmm3              ; xmm0=(CrL * FIX(1.40200))=(R-Y)L

    movdqa      XMMWORD [wk(0)], xmm6   ; wk(0)=(B-Y)H
    movdqa      XMMWORD [wk(1)], xmm7   ; wk(1)=(R-Y)H

    movdqa      xmm6, xmm5
    movdqa      xmm7, xmm2
    punpcklwd   xmm5, xmm1
    punpckhwd   xmm6, xmm1
    pmaddwd     xmm5, [rel PW_MF0344_F0285]
    pmaddwd     xmm6, [rel PW_MF0344_F0285]
    punpcklwd   xmm2, xmm3
    punpckhwd   xmm7, xmm3
    pmaddwd     xmm2, [rel PW_MF0344_F0285]
    pmaddwd     xmm7, [rel PW_MF0344_F0285]

    paddd       xmm5, [rel PD_ONEHALF]
    paddd       xmm6, [rel PD_ONEHALF]
    psrad       xmm5, SCALEBITS
    psrad       xmm6, SCALEBITS
    paddd       xmm2, [rel PD_ONEHALF]
    paddd       xmm7, [rel PD_ONEHALF]
    psrad       xmm2, SCALEBITS
    psrad       xmm7, SCALEBITS

    packssdw    xmm5, xmm6              ; xmm5=CbH*-FIX(0.344)+CrH*FIX(0.285)
    packssdw    xmm2, xmm7              ; xmm2=CbL*-FIX(0.344)+CrL*FIX(0.285)
    psubw       xmm5, xmm1              ; xmm5=CbH*-FIX(0.344)+CrH*-FIX(0.714)=(G-Y)H
    psubw       xmm2, xmm3              ; xmm2=CbL*-FIX(0.344)+CrL*-FIX(0.714)=(G-Y)L

    movdqa      XMMWORD [wk(2)], xmm5   ; wk(2)=(G-Y)H

    mov         al, 2                   ; Yctr
    jmp         short .Yloop_1st

.Yloop_2nd:
    movdqa      xmm0, XMMWORD [wk(1)]   ; xmm0=(R-Y)H
    movdqa      xmm2, XMMWORD [wk(2)]   ; xmm2=(G-Y)H
    movdqa      xmm4, XMMWORD [wk(0)]   ; xmm4=(B-Y)H

.Yloop_1st:
    movdqa      xmm7, XMMWORD [rsi]     ; xmm7=Y(0123456789ABCDEF)

    pcmpeqw     xmm6, xmm6
    psrlw       xmm6, BYTE_BIT          ; xmm6={0xFF 0x00 0xFF 0x00 ..}
    pand        xmm6, xmm7              ; xmm6=Y(02468ACE)=YE
    psrlw       xmm7, BYTE_BIT          ; xmm7=Y(13579BDF)=YO

    movdqa      xmm1, xmm0              ; xmm1=xmm0=(R-Y)(L/H)
    movdqa      xmm3, xmm2              ; xmm3=xmm2=(G-Y)(L/H)
    movdqa      xmm5, xmm4              ; xmm5=xmm4=(B-Y)(L/H)

    paddw       xmm0, xmm6              ; xmm0=((R-Y)+YE)=RE=R(02468ACE)
    paddw       xmm1, xmm7              ; xmm1=((R-Y)+YO)=RO=R(13579BDF)
    packuswb    xmm0, xmm0              ; xmm0=R(02468ACE********)
    packuswb    xmm1, xmm1              ; xmm1=R(13579BDF********)

    paddw       xmm2, xmm6              ; xmm2=((G-Y)+YE)=GE=G(02468ACE)
    paddw       xmm3, xmm7              ; xmm3=((G-Y)+YO)=GO=G(13579BDF)
    packuswb    xmm2, xmm2              ; xmm2=G(02468ACE********)
    packuswb    xmm3, xmm3              ; xmm3=G(13579BDF********)

    paddw       xmm4, xmm6              ; xmm4=((B-Y)+YE)=BE=B(02468ACE)
    paddw       xmm5, xmm7              ; xmm5=((B-Y)+YO)=BO=B(13579BDF)
    packuswb    xmm4, xmm4              ; xmm4=B(02468ACE********)
    packuswb    xmm5, xmm5              ; xmm5=B(13579BDF********)

%if RGB_PIXELSIZE == 3  ; ---------------

    ; xmmA=(00 02 04 06 08 0A 0C 0E **), xmmB=(01 03 05 07 09 0B 0D 0F **)
    ; xmmC=(10 12 14 16 18 1A 1C 1E **), xmmD=(11 13 15 17 19 1B 1D 1F **)
    ; xmmE=(20 22 24 26 28 2A 2C 2E **), xmmF=(21 23 25 27 29 2B 2D 2F **)
    ; xmmG=(** ** ** ** ** ** ** ** **), xmmH=(** ** ** ** ** ** ** ** **)

    punpcklbw   xmmA, xmmC        ; xmmA=(00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E)
    punpcklbw   xmmE, xmmB        ; xmmE=(20 01 22 03 24 05 26 07 28 09 2A 0B 2C 0D 2E 0F)
    punpcklbw   xmmD, xmmF        ; xmmD=(11 21 13 23 15 25 17 27 19 29 1B 2B 1D 2D 1F 2F)

    movdqa      xmmG, xmmA
    movdqa      xmmH, xmmA
    punpcklwd   xmmA, xmmE        ; xmmA=(00 10 20 01 02 12 22 03 04 14 24 05 06 16 26 07)
    punpckhwd   xmmG, xmmE        ; xmmG=(08 18 28 09 0A 1A 2A 0B 0C 1C 2C 0D 0E 1E 2E 0F)

    psrldq      xmmH, 2           ; xmmH=(02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E -- --)
    psrldq      xmmE, 2           ; xmmE=(22 03 24 05 26 07 28 09 2A 0B 2C 0D 2E 0F -- --)

    movdqa      xmmC, xmmD
    movdqa      xmmB, xmmD
    punpcklwd   xmmD, xmmH        ; xmmD=(11 21 02 12 13 23 04 14 15 25 06 16 17 27 08 18)
    punpckhwd   xmmC, xmmH        ; xmmC=(19 29 0A 1A 1B 2B 0C 1C 1D 2D 0E 1E 1F 2F -- --)

    psrldq      xmmB, 2           ; xmmB=(13 23 15 25 17 27 19 29 1B 2B 1D 2D 1F 2F -- --)

    movdqa      xmmF, xmmE
    punpcklwd   xmmE, xmmB        ; xmmE=(22 03 13 23 24 05 15 25 26 07 17 27 28 09 19 29)
    punpckhwd   xmmF, xmmB        ; xmmF=(2A 0B 1B 2B 2C 0D 1D 2D 2E 0F 1F 2F -- -- -- --)

    pshufd      xmmH, xmmA, 0x4E  ; xmmH=(04 14 24 05 06 16 26 07 00 10 20 01 02 12 22 03)
    movdqa      xmmB, xmmE
    punpckldq   xmmA, xmmD        ; xmmA=(00 10 20 01 11 21 02 12 02 12 22 03 13 23 04 14)
    punpckldq   xmmE, xmmH        ; xmmE=(22 03 13 23 04 14 24 05 24 05 15 25 06 16 26 07)
    punpckhdq   xmmD, xmmB        ; xmmD=(15 25 06 16 26 07 17 27 17 27 08 18 28 09 19 29)

    pshufd      xmmH, xmmG, 0x4E  ; xmmH=(0C 1C 2C 0D 0E 1E 2E 0F 08 18 28 09 0A 1A 2A 0B)
    movdqa      xmmB, xmmF
    punpckldq   xmmG, xmmC        ; xmmG=(08 18 28 09 19 29 0A 1A 0A 1A 2A 0B 1B 2B 0C 1C)
    punpckldq   xmmF, xmmH        ; xmmF=(2A 0B 1B 2B 0C 1C 2C 0D 2C 0D 1D 2D 0E 1E 2E 0F)
    punpckhdq   xmmC, xmmB        ; xmmC=(1D 2D 0E 1E 2E 0F 1F 2F 1F 2F -- -- -- -- -- --)

    punpcklqdq  xmmA, xmmE        ; xmmA=(00 10 20 01 11 21 02 12 22 03 13 23 04 14 24 05)
    punpcklqdq  xmmD, xmmG        ; xmmD=(15 25 06 16 26 07 17 27 08 18 28 09 19 29 0A 1A)
    punpcklqdq  xmmF, xmmC        ; xmmF=(2A 0B 1B 2B 0C 1C 2C 0D 1D 2D 0E 1E 2E 0F 1F 2F)

    cmp         rcx, byte SIZEOF_XMMWORD
    jb          short .column_st32

    test        rdi, SIZEOF_XMMWORD-1
    jnz         short .out1
    ; --(aligned)-------------------
    movntdq     XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    movntdq     XMMWORD [rdi+1*SIZEOF_XMMWORD], xmmD
    movntdq     XMMWORD [rdi+2*SIZEOF_XMMWORD], xmmF
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    movdqu      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmmD
    movdqu      XMMWORD [rdi+2*SIZEOF_XMMWORD], xmmF
.out0:
    add         rdi, byte RGB_PIXELSIZE*SIZEOF_XMMWORD  ; outptr
    sub         rcx, byte SIZEOF_XMMWORD
    jz          near .endcolumn

    add         rsi, byte SIZEOF_XMMWORD  ; inptr0
    dec         al                        ; Yctr
    jnz         near .Yloop_2nd

    add         rbx, byte SIZEOF_XMMWORD  ; inptr1
    add         rdx, byte SIZEOF_XMMWORD  ; inptr2
    jmp         near .columnloop

.column_st32:
    lea         rcx, [rcx+rcx*2]            ; imul ecx, RGB_PIXELSIZE
    cmp         rcx, byte 2*SIZEOF_XMMWORD
    jb          short .column_st16
    movdqu      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmmD
    add         rdi, byte 2*SIZEOF_XMMWORD  ; outptr
    movdqa      xmmA, xmmF
    sub         rcx, byte 2*SIZEOF_XMMWORD
    jmp         short .column_st15
.column_st16:
    cmp         rcx, byte SIZEOF_XMMWORD
    jb          short .column_st15
    movdqu      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    add         rdi, byte SIZEOF_XMMWORD    ; outptr
    movdqa      xmmA, xmmD
    sub         rcx, byte SIZEOF_XMMWORD
.column_st15:
    ; Store the lower 8 bytes of xmmA to the output when it has enough
    ; space.
    cmp         rcx, byte SIZEOF_MMWORD
    jb          short .column_st7
    movq        XMM_MMWORD [rdi], xmmA
    add         rdi, byte SIZEOF_MMWORD
    sub         rcx, byte SIZEOF_MMWORD
    psrldq      xmmA, SIZEOF_MMWORD
.column_st7:
    ; Store the lower 4 bytes of xmmA to the output when it has enough
    ; space.
    cmp         rcx, byte SIZEOF_DWORD
    jb          short .column_st3
    movd        XMM_DWORD [rdi], xmmA
    add         rdi, byte SIZEOF_DWORD
    sub         rcx, byte SIZEOF_DWORD
    psrldq      xmmA, SIZEOF_DWORD
.column_st3:
    ; Store the lower 2 bytes of rax to the output when it has enough
    ; space.
    movd        eax, xmmA
    cmp         rcx, byte SIZEOF_WORD
    jb          short .column_st1
    mov         word [rdi], ax
    add         rdi, byte SIZEOF_WORD
    sub         rcx, byte SIZEOF_WORD
    shr         rax, 16
.column_st1:
    ; Store the lower 1 byte of rax to the output when it has enough
    ; space.
    test        rcx, rcx
    jz          short .endcolumn
    mov         byte [rdi], al

%else  ; RGB_PIXELSIZE == 4 ; -----------

%ifdef RGBX_FILLER_0XFF
    pcmpeqb     xmm6, xmm6              ; xmm6=XE=X(02468ACE********)
    pcmpeqb     xmm7, xmm7              ; xmm7=XO=X(13579BDF********)
%else
    pxor        xmm6, xmm6              ; xmm6=XE=X(02468ACE********)
    pxor        xmm7, xmm7              ; xmm7=XO=X(13579BDF********)
%endif
    ; xmmA=(00 02 04 06 08 0A 0C 0E **), xmmB=(01 03 05 07 09 0B 0D 0F **)
    ; xmmC=(10 12 14 16 18 1A 1C 1E **), xmmD=(11 13 15 17 19 1B 1D 1F **)
    ; xmmE=(20 22 24 26 28 2A 2C 2E **), xmmF=(21 23 25 27 29 2B 2D 2F **)
    ; xmmG=(30 32 34 36 38 3A 3C 3E **), xmmH=(31 33 35 37 39 3B 3D 3F **)

    punpcklbw   xmmA, xmmC  ; xmmA=(00 10 02 12 04 14 06 16 08 18 0A 1A 0C 1C 0E 1E)
    punpcklbw   xmmE, xmmG  ; xmmE=(20 30 22 32 24 34 26 36 28 38 2A 3A 2C 3C 2E 3E)
    punpcklbw   xmmB, xmmD  ; xmmB=(01 11 03 13 05 15 07 17 09 19 0B 1B 0D 1D 0F 1F)
    punpcklbw   xmmF, xmmH  ; xmmF=(21 31 23 33 25 35 27 37 29 39 2B 3B 2D 3D 2F 3F)

    movdqa      xmmC, xmmA
    punpcklwd   xmmA, xmmE  ; xmmA=(00 10 20 30 02 12 22 32 04 14 24 34 06 16 26 36)
    punpckhwd   xmmC, xmmE  ; xmmC=(08 18 28 38 0A 1A 2A 3A 0C 1C 2C 3C 0E 1E 2E 3E)
    movdqa      xmmG, xmmB
    punpcklwd   xmmB, xmmF  ; xmmB=(01 11 21 31 03 13 23 33 05 15 25 35 07 17 27 37)
    punpckhwd   xmmG, xmmF  ; xmmG=(09 19 29 39 0B 1B 2B 3B 0D 1D 2D 3D 0F 1F 2F 3F)

    movdqa      xmmD, xmmA
    punpckldq   xmmA, xmmB  ; xmmA=(00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33)
    punpckhdq   xmmD, xmmB  ; xmmD=(04 14 24 34 05 15 25 35 06 16 26 36 07 17 27 37)
    movdqa      xmmH, xmmC
    punpckldq   xmmC, xmmG  ; xmmC=(08 18 28 38 09 19 29 39 0A 1A 2A 3A 0B 1B 2B 3B)
    punpckhdq   xmmH, xmmG  ; xmmH=(0C 1C 2C 3C 0D 1D 2D 3D 0E 1E 2E 3E 0F 1F 2F 3F)

    cmp         rcx, byte SIZEOF_XMMWORD
    jb          short .column_st32

    test        rdi, SIZEOF_XMMWORD-1
    jnz         short .out1
    ; --(aligned)-------------------
    movntdq     XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    movntdq     XMMWORD [rdi+1*SIZEOF_XMMWORD], xmmD
    movntdq     XMMWORD [rdi+2*SIZEOF_XMMWORD], xmmC
    movntdq     XMMWORD [rdi+3*SIZEOF_XMMWORD], xmmH
    jmp         short .out0
.out1:  ; --(unaligned)-----------------
    movdqu      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmmD
    movdqu      XMMWORD [rdi+2*SIZEOF_XMMWORD], xmmC
    movdqu      XMMWORD [rdi+3*SIZEOF_XMMWORD], xmmH
.out0:
    add         rdi, byte RGB_PIXELSIZE*SIZEOF_XMMWORD  ; outptr
    sub         rcx, byte SIZEOF_XMMWORD
    jz          near .endcolumn

    add         rsi, byte SIZEOF_XMMWORD  ; inptr0
    dec         al                        ; Yctr
    jnz         near .Yloop_2nd

    add         rbx, byte SIZEOF_XMMWORD  ; inptr1
    add         rdx, byte SIZEOF_XMMWORD  ; inptr2
    jmp         near .columnloop

.column_st32:
    cmp         rcx, byte SIZEOF_XMMWORD/2
    jb          short .column_st16
    movdqu      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    movdqu      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmmD
    add         rdi, byte 2*SIZEOF_XMMWORD  ; outptr
    movdqa      xmmA, xmmC
    movdqa      xmmD, xmmH
    sub         rcx, byte SIZEOF_XMMWORD/2
.column_st16:
    cmp         rcx, byte SIZEOF_XMMWORD/4
    jb          short .column_st15
    movdqu      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmmA
    add         rdi, byte SIZEOF_XMMWORD    ; outptr
    movdqa      xmmA, xmmD
    sub         rcx, byte SIZEOF_XMMWORD/4
.column_st15:
    ; Store two pixels (8 bytes) of xmmA to the output when it has enough
    ; space.
    cmp         rcx, byte SIZEOF_XMMWORD/8
    jb          short .column_st7
    movq        XMM_MMWORD [rdi], xmmA
    add         rdi, byte SIZEOF_XMMWORD/8*4
    sub         rcx, byte SIZEOF_XMMWORD/8
    psrldq      xmmA, SIZEOF_XMMWORD/8*4
.column_st7:
    ; Store one pixel (4 bytes) of xmmA to the output when it has enough
    ; space.
    test        rcx, rcx
    jz          short .endcolumn
    movd        XMM_DWORD [rdi], xmmA

%endif  ; RGB_PIXELSIZE ; ---------------

.endcolumn:
    sfence                              ; flush the write buffer

.return:
    pop         rbx
    UNCOLLECT_ARGS 4
    lea         rsp, [rbp-8]
    pop         r15
    pop         rbp
    ret

; --------------------------------------------------------------------------
;
; Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
;
; GLOBAL(void)
; jsimd_h2v2_merged_upsample_sse2(JDIMENSION output_width,
;                                 JSAMPIMAGE input_buf,
;                                 JDIMENSION in_row_group_ctr,
;                                 JSAMPARRAY output_buf);
;

; r10d = JDIMENSION output_width
; r11 = JSAMPIMAGE input_buf
; r12d = JDIMENSION in_row_group_ctr
; r13 = JSAMPARRAY output_buf

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_merged_upsample_sse2)

EXTN(jsimd_h2v2_merged_upsample_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 4
    push        rbx

    mov         eax, r10d

    mov         rdi, r11
    mov         ecx, r12d
    mov         rsip, JSAMPARRAY [rdi+0*SIZEOF_JSAMPARRAY]
    mov         rbxp, JSAMPARRAY [rdi+1*SIZEOF_JSAMPARRAY]
    mov         rdxp, JSAMPARRAY [rdi+2*SIZEOF_JSAMPARRAY]
    mov         rdi, r13
    lea         rsi, [rsi+rcx*SIZEOF_JSAMPROW]

    sub         rsp, SIZEOF_JSAMPARRAY*4
    mov         JSAMPARRAY [rsp+0*SIZEOF_JSAMPARRAY], rsip  ; intpr00
    mov         JSAMPARRAY [rsp+1*SIZEOF_JSAMPARRAY], rbxp  ; intpr1
    mov         JSAMPARRAY [rsp+2*SIZEOF_JSAMPARRAY], rdxp  ; intpr2
    mov         rbx, rsp

    push        rdi
    push        rcx
    push        rax

    %ifdef WIN64
    mov         r8, rcx
    mov         r9, rdi
    mov         rcx, rax
    mov         rdx, rbx
    %else
    mov         rdx, rcx
    mov         rcx, rdi
    mov         rdi, rax
    mov         rsi, rbx
    %endif

    call        EXTN(jsimd_h2v1_merged_upsample_sse2)

    pop         rax
    pop         rcx
    pop         rdi
    mov         rsip, JSAMPARRAY [rsp+0*SIZEOF_JSAMPARRAY]
    mov         rbxp, JSAMPARRAY [rsp+1*SIZEOF_JSAMPARRAY]
    mov         rdxp, JSAMPARRAY [rsp+2*SIZEOF_JSAMPARRAY]

    add         rdi, byte SIZEOF_JSAMPROW  ; outptr1
    add         rsi, byte SIZEOF_JSAMPROW  ; inptr01

    mov         JSAMPARRAY [rsp+0*SIZEOF_JSAMPARRAY], rsip  ; intpr00
    mov         JSAMPARRAY [rsp+1*SIZEOF_JSAMPARRAY], rbxp  ; intpr1
    mov         JSAMPARRAY [rsp+2*SIZEOF_JSAMPARRAY], rdxp  ; intpr2
    mov         rbx, rsp

    push        rdi
    push        rcx
    push        rax

    %ifdef WIN64
    mov         r8, rcx
    mov         r9, rdi
    mov         rcx, rax
    mov         rdx, rbx
    %else
    mov         rdx, rcx
    mov         rcx, rdi
    mov         rdi, rax
    mov         rsi, rbx
    %endif

    call        EXTN(jsimd_h2v1_merged_upsample_sse2)

    pop         rax
    pop         rcx
    pop         rdi
    mov         rsip, JSAMPARRAY [rsp+0*SIZEOF_JSAMPARRAY]
    mov         rbxp, JSAMPARRAY [rsp+1*SIZEOF_JSAMPARRAY]
    mov         rdxp, JSAMPARRAY [rsp+2*SIZEOF_JSAMPARRAY]
    add         rsp, SIZEOF_JSAMPARRAY*4

    pop         rbx
    UNCOLLECT_ARGS 4
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
