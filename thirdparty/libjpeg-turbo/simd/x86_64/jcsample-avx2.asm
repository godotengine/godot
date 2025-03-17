;
; jcsample.asm - downsampling (64-bit AVX2)
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2009, 2016, 2024, D. R. Commander.
; Copyright (C) 2015, Intel Corporation.
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
; jsimd_h2v1_downsample_avx2(JDIMENSION image_width, int max_v_samp_factor,
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
    GLOBAL_FUNCTION(jsimd_h2v1_downsample_avx2)

EXTN(jsimd_h2v1_downsample_avx2):
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
    vmovd       xmm7, edx
    vpshufd     xmm7, xmm7, 0x00        ; xmm7={0, 1, 0, 1, 0, 1, 0, 1}
    vperm2i128  ymm7, ymm7, ymm7, 0     ; ymm7={xmm7, xmm7}
    vpcmpeqw    ymm6, ymm6, ymm6
    vpsrlw      ymm6, ymm6, BYTE_BIT    ; ymm6={0xFF 0x00 0xFF 0x00 ..}

    mov         rsi, r14                ; input_data
    mov         rdi, r15                ; output_data
.rowloop:
    push        rcx
    push        rdi
    push        rsi

    mov         rsip, JSAMPROW [rsi]    ; inptr
    mov         rdip, JSAMPROW [rdi]    ; outptr

    cmp         rcx, byte SIZEOF_YMMWORD
    jae         short .columnloop

.columnloop_r24:
    ; rcx can possibly be 8, 16, 24
    cmp         rcx, 24
    jne         .columnloop_r16
    vmovdqu     ymm0, YMMWORD [rsi+0*SIZEOF_YMMWORD]
    vmovdqu     xmm1, XMMWORD [rsi+1*SIZEOF_YMMWORD]
    mov         rcx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r16:
    cmp         rcx, 16
    jne         .columnloop_r8
    vmovdqu     ymm0, YMMWORD [rsi+0*SIZEOF_YMMWORD]
    vpxor       ymm1, ymm1, ymm1
    mov         rcx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r8:
    vmovdqu     xmm0, XMMWORD[rsi+0*SIZEOF_YMMWORD]
    vpxor       ymm1, ymm1, ymm1
    mov         rcx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop:
    vmovdqu     ymm0, YMMWORD [rsi+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [rsi+1*SIZEOF_YMMWORD]

.downsample:
    vpsrlw      ymm2, ymm0, BYTE_BIT
    vpand       ymm0, ymm0, ymm6
    vpsrlw      ymm3, ymm1, BYTE_BIT
    vpand       ymm1, ymm1, ymm6

    vpaddw      ymm0, ymm0, ymm2
    vpaddw      ymm1, ymm1, ymm3
    vpaddw      ymm0, ymm0, ymm7
    vpaddw      ymm1, ymm1, ymm7
    vpsrlw      ymm0, ymm0, 1
    vpsrlw      ymm1, ymm1, 1

    vpackuswb   ymm0, ymm0, ymm1
    vpermq      ymm0, ymm0, 0xd8

    vmovdqu     YMMWORD [rdi+0*SIZEOF_YMMWORD], ymm0

    sub         rcx, byte SIZEOF_YMMWORD    ; outcol
    add         rsi, byte 2*SIZEOF_YMMWORD  ; inptr
    add         rdi, byte 1*SIZEOF_YMMWORD  ; outptr
    cmp         rcx, byte SIZEOF_YMMWORD
    jae         short .columnloop
    test        rcx, rcx
    jnz         near .columnloop_r24

    pop         rsi
    pop         rdi
    pop         rcx

    add         rsi, byte SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte SIZEOF_JSAMPROW  ; output_data
    dec         rax                        ; rowctr
    jg          near .rowloop

.return:
    vzeroupper
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
; jsimd_h2v2_downsample_avx2(JDIMENSION image_width, int max_v_samp_factor,
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
    GLOBAL_FUNCTION(jsimd_h2v2_downsample_avx2)

EXTN(jsimd_h2v2_downsample_avx2):
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
    vmovd       xmm7, edx
    vpcmpeqw    ymm6, ymm6, ymm6
    vpshufd     xmm7, xmm7, 0x00        ; ymm7={1, 2, 1, 2, 1, 2, 1, 2}
    vperm2i128  ymm7, ymm7, ymm7, 0
    vpsrlw      ymm6, ymm6, BYTE_BIT    ; ymm6={0xFF 0x00 0xFF 0x00 ..}

    mov         rsi, r14                ; input_data
    mov         rdi, r15                ; output_data
.rowloop:
    push        rcx
    push        rdi
    push        rsi

    mov         rdxp, JSAMPROW [rsi+0*SIZEOF_JSAMPROW]  ; inptr0
    mov         rsip, JSAMPROW [rsi+1*SIZEOF_JSAMPROW]  ; inptr1
    mov         rdip, JSAMPROW [rdi]                    ; outptr

    cmp         rcx, byte SIZEOF_YMMWORD
    jae         short .columnloop

.columnloop_r24:
    cmp         rcx, 24
    jne         .columnloop_r16
    vmovdqu     ymm0, YMMWORD [rdx+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [rsi+0*SIZEOF_YMMWORD]
    vmovdqu     xmm2, XMMWORD [rdx+1*SIZEOF_YMMWORD]
    vmovdqu     xmm3, XMMWORD [rsi+1*SIZEOF_YMMWORD]
    mov         rcx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r16:
    cmp         rcx, 16
    jne         .columnloop_r8
    vmovdqu     ymm0, YMMWORD [rdx+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [rsi+0*SIZEOF_YMMWORD]
    vpxor       ymm2, ymm2, ymm2
    vpxor       ymm3, ymm3, ymm3
    mov         rcx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop_r8:
    vmovdqu     xmm0, XMMWORD [rdx+0*SIZEOF_XMMWORD]
    vmovdqu     xmm1, XMMWORD [rsi+0*SIZEOF_XMMWORD]
    vpxor       ymm2, ymm2, ymm2
    vpxor       ymm3, ymm3, ymm3
    mov         rcx, SIZEOF_YMMWORD
    jmp         short .downsample

.columnloop:
    vmovdqu     ymm0, YMMWORD [rdx+0*SIZEOF_YMMWORD]
    vmovdqu     ymm1, YMMWORD [rsi+0*SIZEOF_YMMWORD]
    vmovdqu     ymm2, YMMWORD [rdx+1*SIZEOF_YMMWORD]
    vmovdqu     ymm3, YMMWORD [rsi+1*SIZEOF_YMMWORD]

.downsample:
    vpand       ymm4, ymm0, ymm6
    vpsrlw      ymm0, ymm0, BYTE_BIT
    vpand       ymm5, ymm1, ymm6
    vpsrlw      ymm1, ymm1, BYTE_BIT
    vpaddw      ymm0, ymm0, ymm4
    vpaddw      ymm1, ymm1, ymm5

    vpand       ymm4, ymm2, ymm6
    vpsrlw      ymm2, ymm2, BYTE_BIT
    vpand       ymm5, ymm3, ymm6
    vpsrlw      ymm3, ymm3, BYTE_BIT
    vpaddw      ymm2, ymm2, ymm4
    vpaddw      ymm3, ymm3, ymm5

    vpaddw      ymm0, ymm0, ymm1
    vpaddw      ymm2, ymm2, ymm3
    vpaddw      ymm0, ymm0, ymm7
    vpaddw      ymm2, ymm2, ymm7
    vpsrlw      ymm0, ymm0, 2
    vpsrlw      ymm2, ymm2, 2

    vpackuswb   ymm0, ymm0, ymm2
    vpermq      ymm0, ymm0, 0xd8

    vmovdqu     YMMWORD [rdi+0*SIZEOF_YMMWORD], ymm0

    sub         rcx, byte SIZEOF_YMMWORD    ; outcol
    add         rdx, byte 2*SIZEOF_YMMWORD  ; inptr0
    add         rsi, byte 2*SIZEOF_YMMWORD  ; inptr1
    add         rdi, byte 1*SIZEOF_YMMWORD  ; outptr
    cmp         rcx, byte SIZEOF_YMMWORD
    jae         near .columnloop
    test        rcx, rcx
    jnz         near .columnloop_r24

    pop         rsi
    pop         rdi
    pop         rcx

    add         rsi, byte 2*SIZEOF_JSAMPROW  ; input_data
    add         rdi, byte 1*SIZEOF_JSAMPROW  ; output_data
    dec         rax                          ; rowctr
    jg          near .rowloop

.return:
    vzeroupper
    UNCOLLECT_ARGS 6
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
