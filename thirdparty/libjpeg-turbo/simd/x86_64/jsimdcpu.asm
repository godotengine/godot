;
; jsimdcpu.asm - SIMD instruction support check
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2016, D. R. Commander.
; Copyright (C) 2023, Aliaksiej Kandracienka.
;
; Based on
; x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.

%include "jsimdext.inc"

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64
;
; Check if the CPU supports SIMD instructions
;
; GLOBAL(unsigned int)
; jpeg_simd_cpu_support(void)
;

    align       32
    GLOBAL_FUNCTION(jpeg_simd_cpu_support)

EXTN(jpeg_simd_cpu_support):
    push        rbp
    mov         rbp, rsp
    push        rbx
    push        rdi

    xor         rdi, rdi                ; simd support flag

    ; Assume that all x86-64 processors support SSE & SSE2 instructions
    or          rdi, JSIMD_SSE2
    or          rdi, JSIMD_SSE

    ; Check whether CPUID leaf 07H is supported
    ; (leaf 07H is used to check for AVX2 instruction support)
    mov         rax, 0
    cpuid
    cmp         rax, 7
    jl          short .return           ; Maximum leaf < 07H

    ; Check for AVX2 instruction support
    mov         rax, 7
    xor         rcx, rcx
    cpuid
    mov         rax, rbx                ; rax = Extended feature flags

    test        rax, 1<<5               ; bit5:AVX2
    jz          short .return

    ; Check for AVX2 O/S support
    mov         rax, 1
    xor         rcx, rcx
    cpuid
    test        rcx, 1<<27
    jz          short .return           ; O/S does not support XSAVE
    test        rcx, 1<<28
    jz          short .return           ; CPU does not support AVX2

    xor         rcx, rcx
    xgetbv
    and         rax, 6
    cmp         rax, 6                  ; O/S does not manage XMM/YMM state
                                        ; using XSAVE
    jnz         short .return

    or          rdi, JSIMD_AVX2

.return:
    mov         rax, rdi

    pop         rdi
    pop         rbx
    pop         rbp
    ret

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
