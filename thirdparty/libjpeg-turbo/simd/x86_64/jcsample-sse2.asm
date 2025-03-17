;
; jcsample.asm - downsampling (64-bit SSE2)
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

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64
;
; Downsample pixel values of a single component.
; This version handles the common case of 2:1 horizontal and 1:1 vertical,
; without smoothing.
;
; GLOBAL(void)
; jsimd_h2v1_downsample_sse2(JDIMENSION image_width, int max_v_samp_factor,
;                            JDIMENSION v_samp_factor,
;                            JDIMENSION width_in_blocks, JSAMPARRAY input_data,
;                            JSAMPARRAY output_data);
;

; r10d = JDIMENSION image_width
; r11 = int max_v_samp_factor
; r12d = JDIMENSION v_samp_factor
; r13d = JDIMENSION width_in_blocks
; r14 = JSAMPARRAY input_data
; r15 = JSAMPARRAY output_data

    align       32
    GLOBAL_FUNCTION(jsimd_h2v1_downsample_sse2)

EXTN(jsimd_h2v1_downsample_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 6

    mov         ecx, r13d
    shl         rcx, 3                  ; imul rcx,DCTSIZE (rcx = output_cols)
    jz          near .return

    mov         edx, r10d

    ; -- expand_right_edge

    push        rcx
    shl         rcx, 1                  ; output_cols * 2
    sub         rcx, rdx
    jle         short .expand_end

    mov         rax, r11
    test        rax, rax
    jle         short .expand_end

    cld
    mov         rsi, r14                ; input_data
.expandloop:
    push        rax
    push        rcx

    mov         rdip, JSAMPROW [rsi]
    add         rdi, rdx
    mov         al, JSAMPLE [rdi-1]

    rep stosb

    pop         rcx
    pop         rax

    add         rsi, byte SIZEOF_JSAMPROW
    dec         rax
    jg          short .expandloop

.expand_end:
    pop         rcx                     ; output_cols

    ; -- h2v1_downsample

    mov         eax, r12d               ; rowctr
    test        eax, eax
    jle         near .return

    mov         rdx, 0x00010000         ; bias pattern
    movd        xmm7, edx
    pcmpeqw     xmm6, xmm6
    pshufd      xmm7, xmm7, 0x00        ; xmm7={0, 1, 0, 1, 0, 1, 0, 1}
    psrlw       xmm6, BYTE_BIT          ; xmm6={0xFF 0x00 0xFF 0x00 ..}

    mov         rsi, r14                ; input_data
    mov         rdi, r15                ; output_data
.rowloop:
    push        rcx
    push        rdi
    push        rsi

    mov         rsip, JSAMPROW [rsi]    ; inptr
    mov         rdip, JSAMPROW [rdi]    ; outptr

    cmp         rcx, byte SIZEOF_XMMWORD
    jae         short .columnloop

.columnloop_r8:
    movdqa      xmm0, XMMWORD [rsi+0*SIZEOF_XMMWORD]
    pxor        xmm1, xmm1
    mov         rcx, SIZEOF_XMMWORD
    jmp         short .downsample

.columnloop:
    movdqa      xmm0, XMMWORD [rsi+0*SIZEOF_XMMWORD]
    movdqa      xmm1, XMMWORD [rsi+1*SIZEOF_XMMWORD]

.downsample:
    movdqa      xmm2, xmm0
    movdqa      xmm3, xmm1

    pand        xmm0, xmm6
    psrlw       xmm2, BYTE_BIT
    pand        xmm1, xmm6
    psrlw       xmm3, BYTE_BIT

    paddw       xmm0, xmm2
    paddw       xmm1, xmm3
    paddw       xmm0, xmm7
    paddw       xmm1, xmm7
    psrlw       xmm0, 1
    psrlw       xmm1, 1

    packuswb    xmm0, xmm1

    movdqa      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmm0

    sub         rcx, byte SIZEOF_XMMWORD    ; outcol
    add         rsi, byte 2*SIZEOF_XMMWORD  ; inptr
    add         rdi, byte 1*SIZEOF_XMMWORD  ; outptr
    cmp         rcx, byte SIZEOF_XMMWORD
    jae         short .columnloop
    test        rcx, rcx
    jnz         short .columnloop_r8

    pop         rsi
    pop         rdi
    pop         rcx

    add         rsi, byte SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte SIZEOF_JSAMPROW  ; output_data
    dec         rax                        ; rowctr
    jg          near .rowloop

.return:
    UNCOLLECT_ARGS 6
    pop         rbp
    ret

; --------------------------------------------------------------------------
;
; Downsample pixel values of a single component.
; This version handles the standard case of 2:1 horizontal and 2:1 vertical,
; without smoothing.
;
; GLOBAL(void)
; jsimd_h2v2_downsample_sse2(JDIMENSION image_width, int max_v_samp_factor,
;                            JDIMENSION v_samp_factor,
;                            JDIMENSION width_in_blocks, JSAMPARRAY input_data,
;                            JSAMPARRAY output_data);
;

; r10d = JDIMENSION image_width
; r11 = int max_v_samp_factor
; r12d = JDIMENSION v_samp_factor
; r13d = JDIMENSION width_in_blocks
; r14 = JSAMPARRAY input_data
; r15 = JSAMPARRAY output_data

    align       32
    GLOBAL_FUNCTION(jsimd_h2v2_downsample_sse2)

EXTN(jsimd_h2v2_downsample_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    COLLECT_ARGS 6

    mov         ecx, r13d
    shl         rcx, 3                  ; imul rcx,DCTSIZE (rcx = output_cols)
    jz          near .return

    mov         edx, r10d

    ; -- expand_right_edge

    push        rcx
    shl         rcx, 1                  ; output_cols * 2
    sub         rcx, rdx
    jle         short .expand_end

    mov         rax, r11
    test        rax, rax
    jle         short .expand_end

    cld
    mov         rsi, r14                ; input_data
.expandloop:
    push        rax
    push        rcx

    mov         rdip, JSAMPROW [rsi]
    add         rdi, rdx
    mov         al, JSAMPLE [rdi-1]

    rep stosb

    pop         rcx
    pop         rax

    add         rsi, byte SIZEOF_JSAMPROW
    dec         rax
    jg          short .expandloop

.expand_end:
    pop         rcx                     ; output_cols

    ; -- h2v2_downsample

    mov         eax, r12d               ; rowctr
    test        rax, rax
    jle         near .return

    mov         rdx, 0x00020001         ; bias pattern
    movd        xmm7, edx
    pcmpeqw     xmm6, xmm6
    pshufd      xmm7, xmm7, 0x00        ; xmm7={1, 2, 1, 2, 1, 2, 1, 2}
    psrlw       xmm6, BYTE_BIT          ; xmm6={0xFF 0x00 0xFF 0x00 ..}

    mov         rsi, r14                ; input_data
    mov         rdi, r15                ; output_data
.rowloop:
    push        rcx
    push        rdi
    push        rsi

    mov         rdxp, JSAMPROW [rsi+0*SIZEOF_JSAMPROW]  ; inptr0
    mov         rsip, JSAMPROW [rsi+1*SIZEOF_JSAMPROW]  ; inptr1
    mov         rdip, JSAMPROW [rdi]                    ; outptr

    cmp         rcx, byte SIZEOF_XMMWORD
    jae         short .columnloop

.columnloop_r8:
    movdqa      xmm0, XMMWORD [rdx+0*SIZEOF_XMMWORD]
    movdqa      xmm1, XMMWORD [rsi+0*SIZEOF_XMMWORD]
    pxor        xmm2, xmm2
    pxor        xmm3, xmm3
    mov         rcx, SIZEOF_XMMWORD
    jmp         short .downsample

.columnloop:
    movdqa      xmm0, XMMWORD [rdx+0*SIZEOF_XMMWORD]
    movdqa      xmm1, XMMWORD [rsi+0*SIZEOF_XMMWORD]
    movdqa      xmm2, XMMWORD [rdx+1*SIZEOF_XMMWORD]
    movdqa      xmm3, XMMWORD [rsi+1*SIZEOF_XMMWORD]

.downsample:
    movdqa      xmm4, xmm0
    movdqa      xmm5, xmm1
    pand        xmm0, xmm6
    psrlw       xmm4, BYTE_BIT
    pand        xmm1, xmm6
    psrlw       xmm5, BYTE_BIT
    paddw       xmm0, xmm4
    paddw       xmm1, xmm5

    movdqa      xmm4, xmm2
    movdqa      xmm5, xmm3
    pand        xmm2, xmm6
    psrlw       xmm4, BYTE_BIT
    pand        xmm3, xmm6
    psrlw       xmm5, BYTE_BIT
    paddw       xmm2, xmm4
    paddw       xmm3, xmm5

    paddw       xmm0, xmm1
    paddw       xmm2, xmm3
    paddw       xmm0, xmm7
    paddw       xmm2, xmm7
    psrlw       xmm0, 2
    psrlw       xmm2, 2

    packuswb    xmm0, xmm2

    movdqa      XMMWORD [rdi+0*SIZEOF_XMMWORD], xmm0

    sub         rcx, byte SIZEOF_XMMWORD    ; outcol
    add         rdx, byte 2*SIZEOF_XMMWORD  ; inptr0
    add         rsi, byte 2*SIZEOF_XMMWORD  ; inptr1
    add         rdi, byte 1*SIZEOF_XMMWORD  ; outptr
    cmp         rcx, byte SIZEOF_XMMWORD
    jae         near .columnloop
    test        rcx, rcx
    jnz         near .columnloop_r8

    pop         rsi
    pop         rdi
    pop         rcx

    add         rsi, byte 2*SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte 1*SIZEOF_JSAMPROW  ; output_data
    dec         rax                          ; rowctr
    jg          near .rowloop

.return:
    UNCOLLECT_ARGS 6
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
