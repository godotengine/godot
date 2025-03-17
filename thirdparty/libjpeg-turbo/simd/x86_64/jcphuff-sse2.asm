;
; jcphuff-sse2.asm - prepare data for progressive Huffman encoding
; (64-bit SSE2)
;
; Copyright (C) 2016, 2018, Matthieu Darbois
; Copyright (C) 2023, Aliaksiej Kandracienka.
; Copyright (C) 2024, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.
;
; This file contains an SSE2 implementation of data preparation for progressive
; Huffman encoding.  See jcphuff.c for more details.

%include "jsimdext.inc"

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64

; --------------------------------------------------------------------------
; Macros to load data for jsimd_encode_mcu_AC_first_prepare_sse2() and
; jsimd_encode_mcu_AC_refine_prepare_sse2()

%macro LOAD16 0
    pxor        N0, N0
    pxor        N1, N1

    mov         T0d, INT [LUT +  0*SIZEOF_INT]
    mov         T1d, INT [LUT +  8*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 0
    pinsrw      X1, word [BLOCK + T1 * 2], 0

    mov         T0d, INT [LUT +  1*SIZEOF_INT]
    mov         T1d, INT [LUT +  9*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 1
    pinsrw      X1, word [BLOCK + T1 * 2], 1

    mov         T0d, INT [LUT +  2*SIZEOF_INT]
    mov         T1d, INT [LUT + 10*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 2
    pinsrw      X1, word [BLOCK + T1 * 2], 2

    mov         T0d, INT [LUT +  3*SIZEOF_INT]
    mov         T1d, INT [LUT + 11*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 3
    pinsrw      X1, word [BLOCK + T1 * 2], 3

    mov         T0d, INT [LUT +  4*SIZEOF_INT]
    mov         T1d, INT [LUT + 12*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 4
    pinsrw      X1, word [BLOCK + T1 * 2], 4

    mov         T0d, INT [LUT +  5*SIZEOF_INT]
    mov         T1d, INT [LUT + 13*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 5
    pinsrw      X1, word [BLOCK + T1 * 2], 5

    mov         T0d, INT [LUT +  6*SIZEOF_INT]
    mov         T1d, INT [LUT + 14*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 6
    pinsrw      X1, word [BLOCK + T1 * 2], 6

    mov         T0d, INT [LUT +  7*SIZEOF_INT]
    mov         T1d, INT [LUT + 15*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 7
    pinsrw      X1, word [BLOCK + T1 * 2], 7
%endmacro

%macro LOAD15 0
    pxor        N0, N0
    pxor        N1, N1
    pxor        X1, X1

    mov         T0d, INT [LUT +  0*SIZEOF_INT]
    mov         T1d, INT [LUT +  8*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 0
    pinsrw      X1, word [BLOCK + T1 * 2], 0

    mov         T0d, INT [LUT +  1*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 1

    mov         T0d, INT [LUT +  2*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 2

    mov         T0d, INT [LUT +  3*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 3

    mov         T0d, INT [LUT +  4*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 4

    mov         T0d, INT [LUT +  5*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 5

    mov         T0d, INT [LUT +  6*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 6

    mov         T0d, INT [LUT +  7*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 7

    cmp         LENEND, 2
    jl          %%.ELOAD15
    mov         T1d, INT [LUT +  9*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 1

    cmp         LENEND, 3
    jl          %%.ELOAD15
    mov         T1d, INT [LUT + 10*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 2

    cmp         LENEND, 4
    jl          %%.ELOAD15
    mov         T1d, INT [LUT + 11*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 3

    cmp         LENEND, 5
    jl          %%.ELOAD15
    mov         T1d, INT [LUT + 12*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 4

    cmp         LENEND, 6
    jl          %%.ELOAD15
    mov         T1d, INT [LUT + 13*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 5

    cmp         LENEND, 7
    jl          %%.ELOAD15
    mov         T1d, INT [LUT + 14*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 6
%%.ELOAD15:
%endmacro

%macro LOAD8 0
    pxor        N0, N0

    mov         T0d, INT [LUT +  0*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 0

    mov         T0d, INT [LUT +  1*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 1

    mov         T0d, INT [LUT +  2*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 2

    mov         T0d, INT [LUT +  3*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 3

    mov         T0d, INT [LUT +  4*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 4

    mov         T0d, INT [LUT +  5*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 5

    mov         T0d, INT [LUT +  6*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 6

    mov         T0d, INT [LUT +  7*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 7
%endmacro

%macro LOAD7 0
    pxor        N0, N0
    pxor        X0, X0

    mov         T1d, INT [LUT +  0*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 0

    cmp         LENEND, 2
    jl          %%.ELOAD7
    mov         T1d, INT [LUT +  1*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 1

    cmp         LENEND, 3
    jl          %%.ELOAD7
    mov         T1d, INT [LUT +  2*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 2

    cmp         LENEND, 4
    jl          %%.ELOAD7
    mov         T1d, INT [LUT +  3*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 3

    cmp         LENEND, 5
    jl          %%.ELOAD7
    mov         T1d, INT [LUT +  4*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 4

    cmp         LENEND, 6
    jl          %%.ELOAD7
    mov         T1d, INT [LUT +  5*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 5

    cmp         LENEND, 7
    jl          %%.ELOAD7
    mov         T1d, INT [LUT +  6*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 6
%%.ELOAD7:
%endmacro

%macro REDUCE0 0
    movdqa      xmm0, XMMWORD [VALUES + ( 0*2)]
    movdqa      xmm1, XMMWORD [VALUES + ( 8*2)]
    movdqa      xmm2, XMMWORD [VALUES + (16*2)]
    movdqa      xmm3, XMMWORD [VALUES + (24*2)]
    movdqa      xmm4, XMMWORD [VALUES + (32*2)]
    movdqa      xmm5, XMMWORD [VALUES + (40*2)]
    movdqa      xmm6, XMMWORD [VALUES + (48*2)]
    movdqa      xmm7, XMMWORD [VALUES + (56*2)]

    pcmpeqw     xmm0, ZERO
    pcmpeqw     xmm1, ZERO
    pcmpeqw     xmm2, ZERO
    pcmpeqw     xmm3, ZERO
    pcmpeqw     xmm4, ZERO
    pcmpeqw     xmm5, ZERO
    pcmpeqw     xmm6, ZERO
    pcmpeqw     xmm7, ZERO

    packsswb    xmm0, xmm1
    packsswb    xmm2, xmm3
    packsswb    xmm4, xmm5
    packsswb    xmm6, xmm7

    pmovmskb    eax, xmm0
    pmovmskb    ecx, xmm2
    pmovmskb    edx, xmm4
    pmovmskb    esi, xmm6

    shl         rcx, 16
    shl         rdx, 32
    shl         rsi, 48

    or          rax, rcx
    or          rdx, rsi
    or          rax, rdx

    not         rax

    mov         MMWORD [r15], rax
%endmacro

;
; Prepare data for jsimd_encode_mcu_AC_first().
;
; GLOBAL(void)
; jsimd_encode_mcu_AC_first_prepare_sse2(const JCOEF *block,
;                                        const int *jpeg_natural_order_start,
;                                        int Sl, int Al, JCOEF *values,
;                                        size_t *zerobits)
;
; r10 = const JCOEF *block
; r11 = const int *jpeg_natural_order_start
; r12 = int Sl
; r13 = int Al
; r14 = JCOEF *values
; r15 = size_t *zerobits

%define ZERO    xmm9
%define X0      xmm0
%define X1      xmm1
%define N0      xmm2
%define N1      xmm3
%define AL      xmm4
%define K       eax
%define LUT     r11
%define T0      rcx
%define T0d     ecx
%define T1      rdx
%define T1d     edx
%define BLOCK   r10
%define VALUES  r14
%define LEN     r12d
%define LENEND  r13d

    align       32
    GLOBAL_FUNCTION(jsimd_encode_mcu_AC_first_prepare_sse2)

EXTN(jsimd_encode_mcu_AC_first_prepare_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    and         rsp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    sub         rsp, SIZEOF_XMMWORD
    movdqa      XMMWORD [rsp], ZERO
    COLLECT_ARGS 6

    movd        AL, r13d
    pxor        ZERO, ZERO
    mov         K, LEN
    mov         LENEND, LEN
    and         K, -16
    and         LENEND, 7
    shr         K, 4
    jz          .ELOOP16
.BLOOP16:
    LOAD16
    pcmpgtw     N0, X0
    pcmpgtw     N1, X1
    paddw       X0, N0
    paddw       X1, N1
    pxor        X0, N0
    pxor        X1, N1
    psrlw       X0, AL
    psrlw       X1, AL
    pxor        N0, X0
    pxor        N1, X1
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    movdqa      XMMWORD [VALUES + (8) * 2], X1
    movdqa      XMMWORD [VALUES + (0 + DCTSIZE2) * 2], N0
    movdqa      XMMWORD [VALUES + (8 + DCTSIZE2) * 2], N1
    add         VALUES, 16*2
    add         LUT, 16*SIZEOF_INT
    dec         K
    jnz         .BLOOP16
    test        LEN, 15
    je          .PADDING
.ELOOP16:
    test        LEN, 8
    jz          .TRY7
    test        LEN, 7
    jz          .TRY8

    LOAD15
    pcmpgtw     N0, X0
    pcmpgtw     N1, X1
    paddw       X0, N0
    paddw       X1, N1
    pxor        X0, N0
    pxor        X1, N1
    psrlw       X0, AL
    psrlw       X1, AL
    pxor        N0, X0
    pxor        N1, X1
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    movdqa      XMMWORD [VALUES + (8) * 2], X1
    movdqa      XMMWORD [VALUES + (0 + DCTSIZE2) * 2], N0
    movdqa      XMMWORD [VALUES + (8 + DCTSIZE2) * 2], N1
    add         VALUES, 16*2
    jmp         .PADDING
.TRY8:
    LOAD8
    pcmpgtw     N0, X0
    paddw       X0, N0
    pxor        X0, N0
    psrlw       X0, AL
    pxor        N0, X0
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    movdqa      XMMWORD [VALUES + (0 + DCTSIZE2) * 2], N0
    add         VALUES, 8*2
    jmp         .PADDING
.TRY7:
    LOAD7
    pcmpgtw     N0, X0
    paddw       X0, N0
    pxor        X0, N0
    psrlw       X0, AL
    pxor        N0, X0
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    movdqa      XMMWORD [VALUES + (0 + DCTSIZE2) * 2], N0
    add         VALUES, 8*2
.PADDING:
    mov         K, LEN
    add         K, 7
    and         K, -8
    shr         K, 3
    sub         K, DCTSIZE2/8
    jz          .EPADDING
    align       16
.ZEROLOOP:
    movdqa      XMMWORD [VALUES + 0], ZERO
    add         VALUES, 8*2
    inc         K
    jnz         .ZEROLOOP
.EPADDING:
    sub         VALUES, DCTSIZE2*2

    REDUCE0

    UNCOLLECT_ARGS 6
    movdqa      ZERO, XMMWORD [rsp]
    mov         rsp, rbp
    pop         rbp
    ret

%undef ZERO
%undef X0
%undef X1
%undef N0
%undef N1
%undef AL
%undef K
%undef LUT
%undef T0
%undef T0d
%undef T1
%undef T1d
%undef BLOCK
%undef VALUES
%undef LEN
%undef LENEND

;
; Prepare data for jsimd_encode_mcu_AC_refine().
;
; GLOBAL(int)
; jsimd_encode_mcu_AC_refine_prepare_sse2(const JCOEF *block,
;                                         const int *jpeg_natural_order_start,
;                                         int Sl, int Al, JCOEF *absvalues,
;                                         size_t *bits)
;
; r10 = const JCOEF *block
; r11 = const int *jpeg_natural_order_start
; r12 = int Sl
; r13 = int Al
; r14 = JCOEF *values
; r15 = size_t *bits

%define ZERO    xmm9
%define ONE     xmm5
%define X0      xmm0
%define X1      xmm1
%define N0      xmm2
%define N1      xmm3
%define AL      xmm4
%define K       eax
%define KK      r9d
%define EOB     r8d
%define SIGN    rdi
%define LUT     r11
%define T0      rcx
%define T0d     ecx
%define T1      rdx
%define T1d     edx
%define BLOCK   r10
%define VALUES  r14
%define LEN     r12d
%define LENEND  r13d

    align       32
    GLOBAL_FUNCTION(jsimd_encode_mcu_AC_refine_prepare_sse2)

EXTN(jsimd_encode_mcu_AC_refine_prepare_sse2):
    ENDBR64
    push        rbp
    mov         rbp, rsp
    and         rsp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    sub         rsp, SIZEOF_XMMWORD
    movdqa      XMMWORD [rsp], ZERO
    COLLECT_ARGS 6

    xor         SIGN, SIGN
    xor         EOB, EOB
    xor         KK, KK
    movd        AL, r13d
    pxor        ZERO, ZERO
    pcmpeqw     ONE, ONE
    psrlw       ONE, 15
    mov         K, LEN
    mov         LENEND, LEN
    and         K, -16
    and         LENEND, 7
    shr         K, 4
    jz          .ELOOPR16
.BLOOPR16:
    LOAD16
    pcmpgtw     N0, X0
    pcmpgtw     N1, X1
    paddw       X0, N0
    paddw       X1, N1
    pxor        X0, N0
    pxor        X1, N1
    psrlw       X0, AL
    psrlw       X1, AL
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    movdqa      XMMWORD [VALUES + (8) * 2], X1
    pcmpeqw     X0, ONE
    pcmpeqw     X1, ONE
    packsswb    N0, N1
    packsswb    X0, X1
    pmovmskb    T0d, N0                 ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    pmovmskb    T1d, X0                 ; idx = _mm_movemask_epi8(x1);
    shr         SIGN, 16                ; make room for sizebits
    shl         T0, 48
    or          SIGN, T0
    bsr         T1d, T1d                ;  idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER16            ; if (idx) {
    mov         EOB, KK
    add         EOB, T1d                ; EOB = k + idx;
.CONTINUER16:
    add         VALUES, 16*2
    add         LUT, 16*SIZEOF_INT
    add         KK, 16
    dec         K
    jnz         .BLOOPR16
    test        LEN, 15
    je          .PADDINGR
.ELOOPR16:
    test        LEN, 8
    jz          .TRYR7
    test        LEN, 7
    jz          .TRYR8

    LOAD15
    pcmpgtw     N0, X0
    pcmpgtw     N1, X1
    paddw       X0, N0
    paddw       X1, N1
    pxor        X0, N0
    pxor        X1, N1
    psrlw       X0, AL
    psrlw       X1, AL
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    movdqa      XMMWORD [VALUES + (8) * 2], X1
    pcmpeqw     X0, ONE
    pcmpeqw     X1, ONE
    packsswb    N0, N1
    packsswb    X0, X1
    pmovmskb    T0d, N0                 ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    pmovmskb    T1d, X0                 ; idx = _mm_movemask_epi8(x1);
    shr         SIGN, 16                ; make room for sizebits
    shl         T0, 48
    or          SIGN, T0
    bsr         T1d, T1d                ;  idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER15            ; if (idx) {
    mov         EOB, KK
    add         EOB, T1d                ; EOB = k + idx;
.CONTINUER15:
    add         VALUES, 16*2
    jmp         .PADDINGR
.TRYR8:
    LOAD8

    pcmpgtw     N0, X0
    paddw       X0, N0
    pxor        X0, N0
    psrlw       X0, AL
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    pcmpeqw     X0, ONE
    packsswb    N0, ZERO
    packsswb    X0, ZERO
    pmovmskb    T0d, N0                 ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    pmovmskb    T1d, X0                 ; idx = _mm_movemask_epi8(x1);
    shr         SIGN, 8                 ; make room for sizebits
    shl         T0, 56
    or          SIGN, T0
    bsr         T1d, T1d                ;  idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER8             ; if (idx) {
    mov         EOB, KK
    add         EOB, T1d                ; EOB = k + idx;
.CONTINUER8:
    add         VALUES, 8*2
    jmp         .PADDINGR
.TRYR7:
    LOAD7

    pcmpgtw     N0, X0
    paddw       X0, N0
    pxor        X0, N0
    psrlw       X0, AL
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    pcmpeqw     X0, ONE
    packsswb    N0, ZERO
    packsswb    X0, ZERO
    pmovmskb    T0d, N0                 ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    pmovmskb    T1d, X0                 ; idx = _mm_movemask_epi8(x1);
    shr         SIGN, 8                 ; make room for sizebits
    shl         T0, 56
    or          SIGN, T0
    bsr         T1d, T1d                ; idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER7             ; if (idx) {
    mov         EOB, KK
    add         EOB, T1d                ; EOB = k + idx;
.CONTINUER7:
    add         VALUES, 8*2
.PADDINGR:
    mov         K, LEN
    add         K, 7
    and         K, -8
    shr         K, 3
    sub         K, DCTSIZE2/8
    jz          .EPADDINGR
    align       16
.ZEROLOOPR:
    movdqa      XMMWORD [VALUES + 0], ZERO
    shr         SIGN, 8
    add         VALUES, 8*2
    inc         K
    jnz         .ZEROLOOPR
.EPADDINGR:
    not         SIGN
    sub         VALUES, DCTSIZE2*2
    mov         MMWORD [r15+SIZEOF_MMWORD], SIGN

    REDUCE0

    mov         eax, EOB
    UNCOLLECT_ARGS 6
    movdqa      ZERO, XMMWORD [rsp]
    mov         rsp, rbp
    pop         rbp
    ret

%undef ZERO
%undef ONE
%undef X0
%undef X1
%undef N0
%undef N1
%undef AL
%undef K
%undef KK
%undef EOB
%undef SIGN
%undef LUT
%undef T0
%undef T0d
%undef T1
%undef T1d
%undef BLOCK
%undef VALUES
%undef LEN
%undef LENEND

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
