;
; jdsample.asm - upsampling (64-bit SSE2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2024, D. R. Commander.
; Copyright (C) 2018, Matthias Räncker.
; Copyright (C) 2023, Aliaksiej Kandracienka.
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
    GLOBAL_DATA(jconst_fancy_upsample_sse2)

EXTN(jconst_fancy_upsample_sse2):

PW_ONE   times 8 dw 1
PW_TWO   times 8 dw 2
PW_THREE times 8 dw 3
PW_SEVEN times 8 dw 7
PW_EIGHT times 8 dw 8

    ALIGNZ      32

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64
;
; Fancy processing for the common case of 2:1 horizontal and 1:1 vertical.
;
; The upsampling algorithm is linear interpolation between pixel centers,
; also known as a "triangle filter".  This is a good compromise between
; speed and visual quality.  The centers of the output pixels are 1/4 and 3/4
; of the way between input pixel centers.
;
; GLOBAL(void)
; jsimd_h2v1_fancy_upsample_sse2(int max_v_samp_factor,
;                                JDIMENSION downsampled_width,
;                                JSAMPARRAY input_data,
;                                JSAMPARRAY *output_data_ptr);
;

; r10 = int max_v_samp_factor
; r11d = JDIMENSION downsampled_width
; r12 = JSAMPARRAY input_data
; r13 = JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_fancy_upsample_sse2)

EXTN(jsimd_h2v1_fancy_upsample_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 4

    mov         eax, r11d               ; colctr
    test        rax, rax
    jz          near .return

    mov         rcx, r10                ; rowctr
    test        rcx, rcx
    jz          near .return

    mov         rsi, r12                ; input_data
    mov         rdi, r13
    mov         rdip, JSAMPARRAY [rdi]  ; output_data
.rowloop:
    push        rax                     ; colctr
    push        rdi
    push        rsi

    mov         rsip, JSAMPROW [rsi]    ; inptr
    mov         rdip, JSAMPROW [rdi]    ; outptr

    test        rax, SIZEOF_XMMWORD-1
    jz          short .skip
    mov         dl, JSAMPLE [rsi+(rax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [rsi+rax*SIZEOF_JSAMPLE], dl    ; insert a dummy sample
.skip:
    pxor        xmm0, xmm0              ; xmm0=(all 0's)
    pcmpeqb     xmm7, xmm7
    psrldq      xmm7, (SIZEOF_XMMWORD-1)
    pand        xmm7, XMMWORD [rsi+0*SIZEOF_XMMWORD]

    add         rax, byte SIZEOF_XMMWORD-1
    and         rax, byte -SIZEOF_XMMWORD
    cmp         rax, byte SIZEOF_XMMWORD
    ja          short .columnloop

.columnloop_last:
    pcmpeqb     xmm6, xmm6
    pslldq      xmm6, (SIZEOF_XMMWORD-1)
    pand        xmm6, XMMWORD [rsi+0*SIZEOF_XMMWORD]
    jmp         short .upsample

.columnloop:
    movdqa      xmm6, XMMWORD [rsi+1*SIZEOF_XMMWORD]
    pslldq      xmm6, (SIZEOF_XMMWORD-1)

.upsample:
    movdqa      xmm1, XMMWORD [rsi+0*SIZEOF_XMMWORD]
    movdqa      xmm2, xmm1
    movdqa      xmm3, xmm1                ; xmm1=( 0  1  2 ... 13 14 15)
    pslldq      xmm2, 1                   ; xmm2=(--  0  1 ... 12 13 14)
    psrldq      xmm3, 1                   ; xmm3=( 1  2  3 ... 14 15 --)

    por         xmm2, xmm7                ; xmm2=(-1  0  1 ... 12 13 14)
    por         xmm3, xmm6                ; xmm3=( 1  2  3 ... 14 15 16)

    movdqa      xmm7, xmm1
    psrldq      xmm7, (SIZEOF_XMMWORD-1)  ; xmm7=(15 -- -- ... -- -- --)

    movdqa      xmm4, xmm1
    punpcklbw   xmm1, xmm0                ; xmm1=( 0  1  2  3  4  5  6  7)
    punpckhbw   xmm4, xmm0                ; xmm4=( 8  9 10 11 12 13 14 15)
    movdqa      xmm5, xmm2
    punpcklbw   xmm2, xmm0                ; xmm2=(-1  0  1  2  3  4  5  6)
    punpckhbw   xmm5, xmm0                ; xmm5=( 7  8  9 10 11 12 13 14)
    movdqa      xmm6, xmm3
    punpcklbw   xmm3, xmm0                ; xmm3=( 1  2  3  4  5  6  7  8)
    punpckhbw   xmm6, xmm0                ; xmm6=( 9 10 11 12 13 14 15 16)

    pmullw      xmm1, [rel PW_THREE]
    pmullw      xmm4, [rel PW_THREE]
    paddw       xmm2, [rel PW_ONE]
    paddw       xmm5, [rel PW_ONE]
    paddw       xmm3, [rel PW_TWO]
    paddw       xmm6, [rel PW_TWO]

    paddw       xmm2, xmm1
    paddw       xmm5, xmm4
    psrlw       xmm2, 2                 ; xmm2=OutLE=( 0  2  4  6  8 10 12 14)
    psrlw       xmm5, 2                 ; xmm5=OutHE=(16 18 20 22 24 26 28 30)
    paddw       xmm3, xmm1
    paddw       xmm6, xmm4
    psrlw       xmm3, 2                 ; xmm3=OutLO=( 1  3  5  7  9 11 13 15)
    psrlw       xmm6, 2                 ; xmm6=OutHO=(17 19 21 23 25 27 29 31)

    psllw       xmm3, BYTE_BIT
    psllw       xmm6, BYTE_BIT
    por         xmm2, xmm3              ; xmm2=OutL=( 0  1  2 ... 13 14 15)
    por         xmm5, xmm6              ; xmm5=OutH=(16 17 18 ... 29 30 31)

    movdqa      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmm2
    movdqa      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmm5

    sub         rax, byte SIZEOF_XMMWORD
    add         rsi, byte 1*SIZEOF_XMMWORD  ; inptr
    add         rdi, byte 2*SIZEOF_XMMWORD  ; outptr
    cmp         rax, byte SIZEOF_XMMWORD
    ja          near .columnloop
    test        eax, eax
    jnz         near .columnloop_last

    pop         rsi
    pop         rdi
    pop         rax

    add         rsi, byte SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte SIZEOF_JSAMPROW  ; output_data
    dec         rcx                        ; rowctr
    jg          near .rowloop

.return:
    UNCOLLECT_ARGS 4
    pop         rbp
    ret

; --------------------------------------------------------------------------
;
; Fancy processing for the common case of 2:1 horizontal and 2:1 vertical.
; Again a triangle filter; see comments for h2v1 case, above.
;
; GLOBAL(void)
; jsimd_h2v2_fancy_upsample_sse2(int max_v_samp_factor,
;                                JDIMENSION downsampled_width,
;                                JSAMPARRAY input_data,
;                                JSAMPARRAY *output_data_ptr);
;

; r10 = int max_v_samp_factor
; r11d = JDIMENSION downsampled_width
; r12 = JSAMPARRAY input_data
; r13 = JSAMPARRAY *output_data_ptr

%define wk(i)   r15 - (WK_NUM - (i)) * SIZEOF_XMMWORD  ; xmmword wk[WK_NUM]
%define WK_NUM  4

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_fancy_upsample_sse2)

EXTN(jsimd_h2v2_fancy_upsample_sse2):
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

    mov         eax, r11d               ; colctr
    test        rax, rax
    jz          near .return

    mov         rcx, r10                ; rowctr
    test        rcx, rcx
    jz          near .return

    mov         rsi, r12                ; input_data
    mov         rdi, r13
    mov         rdip, JSAMPARRAY [rdi]  ; output_data
.rowloop:
    push        rax                     ; colctr
    push        rcx
    push        rdi
    push        rsi

    mov         rcxp, JSAMPROW [rsi-1*SIZEOF_JSAMPROW]  ; inptr1(above)
    mov         rbxp, JSAMPROW [rsi+0*SIZEOF_JSAMPROW]  ; inptr0
    mov         rsip, JSAMPROW [rsi+1*SIZEOF_JSAMPROW]  ; inptr1(below)
    mov         rdxp, JSAMPROW [rdi+0*SIZEOF_JSAMPROW]  ; outptr0
    mov         rdip, JSAMPROW [rdi+1*SIZEOF_JSAMPROW]  ; outptr1

    test        rax, SIZEOF_XMMWORD-1
    jz          short .skip
    push        rdx
    mov         dl, JSAMPLE [rcx+(rax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [rcx+rax*SIZEOF_JSAMPLE], dl
    mov         dl, JSAMPLE [rbx+(rax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [rbx+rax*SIZEOF_JSAMPLE], dl
    mov         dl, JSAMPLE [rsi+(rax-1)*SIZEOF_JSAMPLE]
    mov         JSAMPLE [rsi+rax*SIZEOF_JSAMPLE], dl    ; insert a dummy sample
    pop         rdx
.skip:
    ; -- process the first column block

    movdqa      xmm0, XMMWORD [rbx+0*SIZEOF_XMMWORD]  ; xmm0=row[ 0][0]
    movdqa      xmm1, XMMWORD [rcx+0*SIZEOF_XMMWORD]  ; xmm1=row[-1][0]
    movdqa      xmm2, XMMWORD [rsi+0*SIZEOF_XMMWORD]  ; xmm2=row[+1][0]

    pxor        xmm3, xmm3              ; xmm3=(all 0's)
    movdqa      xmm4, xmm0
    punpcklbw   xmm0, xmm3              ; xmm0=row[ 0]( 0  1  2  3  4  5  6  7)
    punpckhbw   xmm4, xmm3              ; xmm4=row[ 0]( 8  9 10 11 12 13 14 15)
    movdqa      xmm5, xmm1
    punpcklbw   xmm1, xmm3              ; xmm1=row[-1]( 0  1  2  3  4  5  6  7)
    punpckhbw   xmm5, xmm3              ; xmm5=row[-1]( 8  9 10 11 12 13 14 15)
    movdqa      xmm6, xmm2
    punpcklbw   xmm2, xmm3              ; xmm2=row[+1]( 0  1  2  3  4  5  6  7)
    punpckhbw   xmm6, xmm3              ; xmm6=row[+1]( 8  9 10 11 12 13 14 15)

    pmullw      xmm0, [rel PW_THREE]
    pmullw      xmm4, [rel PW_THREE]

    pcmpeqb     xmm7, xmm7
    psrldq      xmm7, (SIZEOF_XMMWORD-2)

    paddw       xmm1, xmm0              ; xmm1=Int0L=( 0  1  2  3  4  5  6  7)
    paddw       xmm5, xmm4              ; xmm5=Int0H=( 8  9 10 11 12 13 14 15)
    paddw       xmm2, xmm0              ; xmm2=Int1L=( 0  1  2  3  4  5  6  7)
    paddw       xmm6, xmm4              ; xmm6=Int1H=( 8  9 10 11 12 13 14 15)

    movdqa      XMMWORD [rdx+0*SIZEOF_XMMWORD], xmm1  ; temporarily save
    movdqa      XMMWORD [rdx+1*SIZEOF_XMMWORD], xmm5  ; the intermediate data
    movdqa      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmm2
    movdqa      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmm6

    pand        xmm1, xmm7              ; xmm1=( 0 -- -- -- -- -- -- --)
    pand        xmm2, xmm7              ; xmm2=( 0 -- -- -- -- -- -- --)

    movdqa      XMMWORD [wk(0)], xmm1
    movdqa      XMMWORD [wk(1)], xmm2

    add         rax, byte SIZEOF_XMMWORD-1
    and         rax, byte -SIZEOF_XMMWORD
    cmp         rax, byte SIZEOF_XMMWORD
    ja          short .columnloop

.columnloop_last:
    ; -- process the last column block

    pcmpeqb     xmm1, xmm1
    pslldq      xmm1, (SIZEOF_XMMWORD-2)
    movdqa      xmm2, xmm1

    pand        xmm1, XMMWORD [rdx+1*SIZEOF_XMMWORD]
    pand        xmm2, XMMWORD [rdi+1*SIZEOF_XMMWORD]

    movdqa      XMMWORD [wk(2)], xmm1   ; xmm1=(-- -- -- -- -- -- -- 15)
    movdqa      XMMWORD [wk(3)], xmm2   ; xmm2=(-- -- -- -- -- -- -- 15)

    jmp         near .upsample

.columnloop:
    ; -- process the next column block

    movdqa      xmm0, XMMWORD [rbx+1*SIZEOF_XMMWORD]  ; xmm0=row[ 0][1]
    movdqa      xmm1, XMMWORD [rcx+1*SIZEOF_XMMWORD]  ; xmm1=row[-1][1]
    movdqa      xmm2, XMMWORD [rsi+1*SIZEOF_XMMWORD]  ; xmm2=row[+1][1]

    pxor        xmm3, xmm3              ; xmm3=(all 0's)
    movdqa      xmm4, xmm0
    punpcklbw   xmm0, xmm3              ; xmm0=row[ 0]( 0  1  2  3  4  5  6  7)
    punpckhbw   xmm4, xmm3              ; xmm4=row[ 0]( 8  9 10 11 12 13 14 15)
    movdqa      xmm5, xmm1
    punpcklbw   xmm1, xmm3              ; xmm1=row[-1]( 0  1  2  3  4  5  6  7)
    punpckhbw   xmm5, xmm3              ; xmm5=row[-1]( 8  9 10 11 12 13 14 15)
    movdqa      xmm6, xmm2
    punpcklbw   xmm2, xmm3              ; xmm2=row[+1]( 0  1  2  3  4  5  6  7)
    punpckhbw   xmm6, xmm3              ; xmm6=row[+1]( 8  9 10 11 12 13 14 15)

    pmullw      xmm0, [rel PW_THREE]
    pmullw      xmm4, [rel PW_THREE]

    paddw       xmm1, xmm0              ; xmm1=Int0L=( 0  1  2  3  4  5  6  7)
    paddw       xmm5, xmm4              ; xmm5=Int0H=( 8  9 10 11 12 13 14 15)
    paddw       xmm2, xmm0              ; xmm2=Int1L=( 0  1  2  3  4  5  6  7)
    paddw       xmm6, xmm4              ; xmm6=Int1H=( 8  9 10 11 12 13 14 15)

    movdqa      XMMWORD [rdx+2*SIZEOF_XMMWORD], xmm1  ; temporarily save
    movdqa      XMMWORD [rdx+3*SIZEOF_XMMWORD], xmm5  ; the intermediate data
    movdqa      XMMWORD [rdi+2*SIZEOF_XMMWORD], xmm2
    movdqa      XMMWORD [rdi+3*SIZEOF_XMMWORD], xmm6

    pslldq      xmm1, (SIZEOF_XMMWORD-2)  ; xmm1=(-- -- -- -- -- -- --  0)
    pslldq      xmm2, (SIZEOF_XMMWORD-2)  ; xmm2=(-- -- -- -- -- -- --  0)

    movdqa      XMMWORD [wk(2)], xmm1
    movdqa      XMMWORD [wk(3)], xmm2

.upsample:
    ; -- process the upper row

    movdqa      xmm7, XMMWORD [rdx+0*SIZEOF_XMMWORD]
    movdqa      xmm3, XMMWORD [rdx+1*SIZEOF_XMMWORD]

    movdqa      xmm0, xmm7                ; xmm7=Int0L=( 0  1  2  3  4  5  6  7)
    movdqa      xmm4, xmm3                ; xmm3=Int0H=( 8  9 10 11 12 13 14 15)
    psrldq      xmm0, 2                   ; xmm0=( 1  2  3  4  5  6  7 --)
    pslldq      xmm4, (SIZEOF_XMMWORD-2)  ; xmm4=(-- -- -- -- -- -- --  8)
    movdqa      xmm5, xmm7
    movdqa      xmm6, xmm3
    psrldq      xmm5, (SIZEOF_XMMWORD-2)  ; xmm5=( 7 -- -- -- -- -- -- --)
    pslldq      xmm6, 2                   ; xmm6=(--  8  9 10 11 12 13 14)

    por         xmm0, xmm4                ; xmm0=( 1  2  3  4  5  6  7  8)
    por         xmm5, xmm6                ; xmm5=( 7  8  9 10 11 12 13 14)

    movdqa      xmm1, xmm7
    movdqa      xmm2, xmm3
    pslldq      xmm1, 2                   ; xmm1=(--  0  1  2  3  4  5  6)
    psrldq      xmm2, 2                   ; xmm2=( 9 10 11 12 13 14 15 --)
    movdqa      xmm4, xmm3
    psrldq      xmm4, (SIZEOF_XMMWORD-2)  ; xmm4=(15 -- -- -- -- -- -- --)

    por         xmm1, XMMWORD [wk(0)]     ; xmm1=(-1  0  1  2  3  4  5  6)
    por         xmm2, XMMWORD [wk(2)]     ; xmm2=( 9 10 11 12 13 14 15 16)

    movdqa      XMMWORD [wk(0)], xmm4

    pmullw      xmm7, [rel PW_THREE]
    pmullw      xmm3, [rel PW_THREE]
    paddw       xmm1, [rel PW_EIGHT]
    paddw       xmm5, [rel PW_EIGHT]
    paddw       xmm0, [rel PW_SEVEN]
    paddw       xmm2, [rel PW_SEVEN]

    paddw       xmm1, xmm7
    paddw       xmm5, xmm3
    psrlw       xmm1, 4                 ; xmm1=Out0LE=( 0  2  4  6  8 10 12 14)
    psrlw       xmm5, 4                 ; xmm5=Out0HE=(16 18 20 22 24 26 28 30)
    paddw       xmm0, xmm7
    paddw       xmm2, xmm3
    psrlw       xmm0, 4                 ; xmm0=Out0LO=( 1  3  5  7  9 11 13 15)
    psrlw       xmm2, 4                 ; xmm2=Out0HO=(17 19 21 23 25 27 29 31)

    psllw       xmm0, BYTE_BIT
    psllw       xmm2, BYTE_BIT
    por         xmm1, xmm0              ; xmm1=Out0L=( 0  1  2 ... 13 14 15)
    por         xmm5, xmm2              ; xmm5=Out0H=(16 17 18 ... 29 30 31)

    movdqa      XMMWORD [rdx+0*SIZEOF_XMMWORD], xmm1
    movdqa      XMMWORD [rdx+1*SIZEOF_XMMWORD], xmm5

    ; -- process the lower row

    movdqa      xmm6, XMMWORD [rdi+0*SIZEOF_XMMWORD]
    movdqa      xmm4, XMMWORD [rdi+1*SIZEOF_XMMWORD]

    movdqa      xmm7, xmm6                ; xmm6=Int1L=( 0  1  2  3  4  5  6  7)
    movdqa      xmm3, xmm4                ; xmm4=Int1H=( 8  9 10 11 12 13 14 15)
    psrldq      xmm7, 2                   ; xmm7=( 1  2  3  4  5  6  7 --)
    pslldq      xmm3, (SIZEOF_XMMWORD-2)  ; xmm3=(-- -- -- -- -- -- --  8)
    movdqa      xmm0, xmm6
    movdqa      xmm2, xmm4
    psrldq      xmm0, (SIZEOF_XMMWORD-2)  ; xmm0=( 7 -- -- -- -- -- -- --)
    pslldq      xmm2, 2                   ; xmm2=(--  8  9 10 11 12 13 14)

    por         xmm7, xmm3                ; xmm7=( 1  2  3  4  5  6  7  8)
    por         xmm0, xmm2                ; xmm0=( 7  8  9 10 11 12 13 14)

    movdqa      xmm1, xmm6
    movdqa      xmm5, xmm4
    pslldq      xmm1, 2                   ; xmm1=(--  0  1  2  3  4  5  6)
    psrldq      xmm5, 2                   ; xmm5=( 9 10 11 12 13 14 15 --)
    movdqa      xmm3, xmm4
    psrldq      xmm3, (SIZEOF_XMMWORD-2)  ; xmm3=(15 -- -- -- -- -- -- --)

    por         xmm1, XMMWORD [wk(1)]     ; xmm1=(-1  0  1  2  3  4  5  6)
    por         xmm5, XMMWORD [wk(3)]     ; xmm5=( 9 10 11 12 13 14 15 16)

    movdqa      XMMWORD [wk(1)], xmm3

    pmullw      xmm6, [rel PW_THREE]
    pmullw      xmm4, [rel PW_THREE]
    paddw       xmm1, [rel PW_EIGHT]
    paddw       xmm0, [rel PW_EIGHT]
    paddw       xmm7, [rel PW_SEVEN]
    paddw       xmm5, [rel PW_SEVEN]

    paddw       xmm1, xmm6
    paddw       xmm0, xmm4
    psrlw       xmm1, 4                 ; xmm1=Out1LE=( 0  2  4  6  8 10 12 14)
    psrlw       xmm0, 4                 ; xmm0=Out1HE=(16 18 20 22 24 26 28 30)
    paddw       xmm7, xmm6
    paddw       xmm5, xmm4
    psrlw       xmm7, 4                 ; xmm7=Out1LO=( 1  3  5  7  9 11 13 15)
    psrlw       xmm5, 4                 ; xmm5=Out1HO=(17 19 21 23 25 27 29 31)

    psllw       xmm7, BYTE_BIT
    psllw       xmm5, BYTE_BIT
    por         xmm1, xmm7              ; xmm1=Out1L=( 0  1  2 ... 13 14 15)
    por         xmm0, xmm5              ; xmm0=Out1H=(16 17 18 ... 29 30 31)

    movdqa      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmm1
    movdqa      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmm0

    sub         rax, byte SIZEOF_XMMWORD
    add         rcx, byte 1*SIZEOF_XMMWORD  ; inptr1(above)
    add         rbx, byte 1*SIZEOF_XMMWORD  ; inptr0
    add         rsi, byte 1*SIZEOF_XMMWORD  ; inptr1(below)
    add         rdx, byte 2*SIZEOF_XMMWORD  ; outptr0
    add         rdi, byte 2*SIZEOF_XMMWORD  ; outptr1
    cmp         rax, byte SIZEOF_XMMWORD
    ja          near .columnloop
    test        rax, rax
    jnz         near .columnloop_last

    pop         rsi
    pop         rdi
    pop         rcx
    pop         rax

    add         rsi, byte 1*SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte 2*SIZEOF_JSAMPROW  ; output_data
    sub         rcx, byte 2                  ; rowctr
    jg          near .rowloop

.return:
    pop         rbx
    UNCOLLECT_ARGS 4
    lea         rsp, [rbp-8]
    pop         r15
    pop         rbp
    ret

; --------------------------------------------------------------------------
;
; Fast processing for the common case of 2:1 horizontal and 1:1 vertical.
; It's still a box filter.
;
; GLOBAL(void)
; jsimd_h2v1_upsample_sse2(int max_v_samp_factor, JDIMENSION output_width,
;                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr);
;

; r10 = int max_v_samp_factor
; r11d = JDIMENSION output_width
; r12 = JSAMPARRAY input_data
; r13 = JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_upsample_sse2)

EXTN(jsimd_h2v1_upsample_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 4

    mov         edx, r11d
    add         rdx, byte (2*SIZEOF_XMMWORD)-1
    and         rdx, byte -(2*SIZEOF_XMMWORD)
    jz          near .return

    mov         rcx, r10                ; rowctr
    test        rcx, rcx
    jz          short .return

    mov         rsi, r12                ; input_data
    mov         rdi, r13
    mov         rdip, JSAMPARRAY [rdi]  ; output_data
.rowloop:
    push        rdi
    push        rsi

    mov         rsip, JSAMPROW [rsi]    ; inptr
    mov         rdip, JSAMPROW [rdi]    ; outptr
    mov         rax, rdx                ; colctr
.columnloop:

    movdqa      xmm0, XMMWORD [rsi+0*SIZEOF_XMMWORD]

    movdqa      xmm1, xmm0
    punpcklbw   xmm0, xmm0
    punpckhbw   xmm1, xmm1

    movdqa      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmm0
    movdqa      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmm1

    sub         rax, byte 2*SIZEOF_XMMWORD
    jz          short .nextrow

    movdqa      xmm2, XMMWORD [rsi+1*SIZEOF_XMMWORD]

    movdqa      xmm3, xmm2
    punpcklbw   xmm2, xmm2
    punpckhbw   xmm3, xmm3

    movdqa      XMMWORD [rdi+2*SIZEOF_XMMWORD], xmm2
    movdqa      XMMWORD [rdi+3*SIZEOF_XMMWORD], xmm3

    sub         rax, byte 2*SIZEOF_XMMWORD
    jz          short .nextrow

    add         rsi, byte 2*SIZEOF_XMMWORD  ; inptr
    add         rdi, byte 4*SIZEOF_XMMWORD  ; outptr
    jmp         short .columnloop

.nextrow:
    pop         rsi
    pop         rdi

    add         rsi, byte SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte SIZEOF_JSAMPROW  ; output_data
    dec         rcx                        ; rowctr
    jg          short .rowloop

.return:
    UNCOLLECT_ARGS 4
    pop         rbp
    ret

; --------------------------------------------------------------------------
;
; Fast processing for the common case of 2:1 horizontal and 2:1 vertical.
; It's still a box filter.
;
; GLOBAL(void)
; jsimd_h2v2_upsample_sse2(int max_v_samp_factor, JDIMENSION output_width,
;                          JSAMPARRAY input_data, JSAMPARRAY *output_data_ptr);
;

; r10 = int max_v_samp_factor
; r11d = JDIMENSION output_width
; r12 = JSAMPARRAY input_data
; r13 = JSAMPARRAY *output_data_ptr

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_upsample_sse2)

EXTN(jsimd_h2v2_upsample_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 4
    push        rbx

    mov         edx, r11d
    add         rdx, byte (2*SIZEOF_XMMWORD)-1
    and         rdx, byte -(2*SIZEOF_XMMWORD)
    jz          near .return

    mov         rcx, r10                ; rowctr
    test        rcx, rcx
    jz          near .return

    mov         rsi, r12                ; input_data
    mov         rdi, r13
    mov         rdip, JSAMPARRAY [rdi]  ; output_data
.rowloop:
    push        rdi
    push        rsi

    mov         rsip, JSAMPROW [rsi]                   ; inptr
    mov         rbxp, JSAMPROW [rdi+0*SIZEOF_JSAMPROW] ; outptr0
    mov         rdip, JSAMPROW [rdi+1*SIZEOF_JSAMPROW] ; outptr1
    mov         rax, rdx                               ; colctr
.columnloop:

    movdqa      xmm0, XMMWORD [rsi+0*SIZEOF_XMMWORD]

    movdqa      xmm1, xmm0
    punpcklbw   xmm0, xmm0
    punpckhbw   xmm1, xmm1

    movdqa      XMMWORD [rbx+0*SIZEOF_XMMWORD], xmm0
    movdqa      XMMWORD [rbx+1*SIZEOF_XMMWORD], xmm1
    movdqa      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmm0
    movdqa      XMMWORD [rdi+1*SIZEOF_XMMWORD], xmm1

    sub         rax, byte 2*SIZEOF_XMMWORD
    jz          short .nextrow

    movdqa      xmm2, XMMWORD [rsi+1*SIZEOF_XMMWORD]

    movdqa      xmm3, xmm2
    punpcklbw   xmm2, xmm2
    punpckhbw   xmm3, xmm3

    movdqa      XMMWORD [rbx+2*SIZEOF_XMMWORD], xmm2
    movdqa      XMMWORD [rbx+3*SIZEOF_XMMWORD], xmm3
    movdqa      XMMWORD [rdi+2*SIZEOF_XMMWORD], xmm2
    movdqa      XMMWORD [rdi+3*SIZEOF_XMMWORD], xmm3

    sub         rax, byte 2*SIZEOF_XMMWORD
    jz          short .nextrow

    add         rsi, byte 2*SIZEOF_XMMWORD  ; inptr
    add         rbx, byte 4*SIZEOF_XMMWORD  ; outptr0
    add         rdi, byte 4*SIZEOF_XMMWORD  ; outptr1
    jmp         short .columnloop

.nextrow:
    pop         rsi
    pop         rdi

    add         rsi, byte 1*SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte 2*SIZEOF_JSAMPROW  ; output_data
    sub         rcx, byte 2                  ; rowctr
    jg          near .rowloop

.return:
    pop         rbx
    UNCOLLECT_ARGS 4
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
