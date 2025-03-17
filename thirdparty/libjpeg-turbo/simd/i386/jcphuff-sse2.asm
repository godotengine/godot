;
; jcphuff-sse2.asm - prepare data for progressive Huffman encoding (SSE2)
;
; Copyright (C) 2016, 2018, Matthieu Darbois
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
    BITS        32

; --------------------------------------------------------------------------
; Macros to load data for jsimd_encode_mcu_AC_first_prepare_sse2() and
; jsimd_encode_mcu_AC_refine_prepare_sse2()

%macro LOAD16 0
    pxor        N0, N0
    pxor        N1, N1

    mov         T0, INT [LUT +  0*SIZEOF_INT]
    mov         T1, INT [LUT +  8*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 0
    pinsrw      X1, word [BLOCK + T1 * 2], 0

    mov         T0, INT [LUT +  1*SIZEOF_INT]
    mov         T1, INT [LUT +  9*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 1
    pinsrw      X1, word [BLOCK + T1 * 2], 1

    mov         T0, INT [LUT +  2*SIZEOF_INT]
    mov         T1, INT [LUT + 10*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 2
    pinsrw      X1, word [BLOCK + T1 * 2], 2

    mov         T0, INT [LUT +  3*SIZEOF_INT]
    mov         T1, INT [LUT + 11*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 3
    pinsrw      X1, word [BLOCK + T1 * 2], 3

    mov         T0, INT [LUT +  4*SIZEOF_INT]
    mov         T1, INT [LUT + 12*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 4
    pinsrw      X1, word [BLOCK + T1 * 2], 4

    mov         T0, INT [LUT +  5*SIZEOF_INT]
    mov         T1, INT [LUT + 13*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 5
    pinsrw      X1, word [BLOCK + T1 * 2], 5

    mov         T0, INT [LUT +  6*SIZEOF_INT]
    mov         T1, INT [LUT + 14*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 6
    pinsrw      X1, word [BLOCK + T1 * 2], 6

    mov         T0, INT [LUT +  7*SIZEOF_INT]
    mov         T1, INT [LUT + 15*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 7
    pinsrw      X1, word [BLOCK + T1 * 2], 7
%endmacro

%macro LOAD15 0
    pxor        N0, N0
    pxor        N1, N1
    pxor        X1, X1

    mov         T0, INT [LUT +  0*SIZEOF_INT]
    mov         T1, INT [LUT +  8*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 0
    pinsrw      X1, word [BLOCK + T1 * 2], 0

    mov         T0, INT [LUT +  1*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 1

    mov         T0, INT [LUT +  2*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 2

    mov         T0, INT [LUT +  3*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 3

    mov         T0, INT [LUT +  4*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 4

    mov         T0, INT [LUT +  5*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 5

    mov         T0, INT [LUT +  6*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 6

    mov         T0, INT [LUT +  7*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 7

    cmp         LENEND, 2
    jl          %%.ELOAD15
    mov         T1, INT [LUT +  9*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 1

    cmp         LENEND, 3
    jl          %%.ELOAD15
    mov         T1, INT [LUT + 10*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 2

    cmp         LENEND, 4
    jl          %%.ELOAD15
    mov         T1, INT [LUT + 11*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 3

    cmp         LENEND, 5
    jl          %%.ELOAD15
    mov         T1, INT [LUT + 12*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 4

    cmp         LENEND, 6
    jl          %%.ELOAD15
    mov         T1, INT [LUT + 13*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 5

    cmp         LENEND, 7
    jl          %%.ELOAD15
    mov         T1, INT [LUT + 14*SIZEOF_INT]
    pinsrw      X1, word [BLOCK + T1 * 2], 6
%%.ELOAD15:
%endmacro

%macro LOAD8 0
    pxor        N0, N0

    mov         T0, INT [LUT +  0*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 0

    mov         T0, INT [LUT +  1*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 1

    mov         T0, INT [LUT +  2*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 2

    mov         T0, INT [LUT +  3*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 3

    mov         T0, INT [LUT +  4*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 4

    mov         T0, INT [LUT +  5*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 5

    mov         T0, INT [LUT +  6*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 6

    mov         T0, INT [LUT +  7*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T0 * 2], 7
%endmacro

%macro LOAD7 0
    pxor        N0, N0
    pxor        X0, X0

    mov         T1, INT [LUT +  0*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 0

    cmp         LENEND, 2
    jl          %%.ELOAD7
    mov         T1, INT [LUT +  1*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 1

    cmp         LENEND, 3
    jl          %%.ELOAD7
    mov         T1, INT [LUT +  2*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 2

    cmp         LENEND, 4
    jl          %%.ELOAD7
    mov         T1, INT [LUT +  3*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 3

    cmp         LENEND, 5
    jl          %%.ELOAD7
    mov         T1, INT [LUT +  4*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 4

    cmp         LENEND, 6
    jl          %%.ELOAD7
    mov         T1, INT [LUT +  5*SIZEOF_INT]
    pinsrw      X0, word [BLOCK + T1 * 2], 5

    cmp         LENEND, 7
    jl          %%.ELOAD7
    mov         T1, INT [LUT +  6*SIZEOF_INT]
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

    pcmpeqw     xmm0, ZERO
    pcmpeqw     xmm1, ZERO
    pcmpeqw     xmm2, ZERO
    pcmpeqw     xmm3, ZERO
    pcmpeqw     xmm4, ZERO
    pcmpeqw     xmm5, ZERO
    pcmpeqw     xmm6, ZERO
    pcmpeqw     xmm7, XMMWORD [VALUES + (56*2)]

    packsswb    xmm0, xmm1
    packsswb    xmm2, xmm3
    packsswb    xmm4, xmm5
    packsswb    xmm6, xmm7

    pmovmskb    eax, xmm0
    pmovmskb    ecx, xmm2
    pmovmskb    edx, xmm4
    pmovmskb    esi, xmm6

    shl         ecx, 16
    shl         esi, 16

    or          eax, ecx
    or          edx, esi

    not         eax
    not         edx

    mov         edi, ZEROBITS

    mov         INT [edi], eax
    mov         INT [edi+SIZEOF_INT], edx
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
; eax + 8 = const JCOEF *block
; eax + 12 = const int *jpeg_natural_order_start
; eax + 16 = int Sl
; eax + 20 = int Al
; eax + 24 = JCOEF *values
; eax + 28 = size_t *zerobits

%define ZERO    xmm7
%define X0      xmm0
%define X1      xmm1
%define N0      xmm2
%define N1      xmm3
%define AL      xmm4
%define K       eax
%define LENEND  eax
%define LUT     ebx
%define T0      ecx
%define T1      edx
%define BLOCK   esi
%define VALUES  edi
%define LEN     ebp

%define ZEROBITS  INT [esp + 5 * 4]

    align       32
    GLOBAL_FUNCTION(jsimd_encode_mcu_AC_first_prepare_sse2)

EXTN(jsimd_encode_mcu_AC_first_prepare_sse2):
    push        ebp
    mov         eax, esp                     ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    mov         [esp], eax
    mov         ebp, esp                     ; ebp = aligned ebp
    sub         esp, 4
    push        ebx
    push        ecx
;   push        edx                     ; need not be preserved
    push        esi
    push        edi
    push        ebp

    mov         BLOCK, INT [eax + 8]
    mov         LUT, INT [eax + 12]
    mov         VALUES, INT [eax + 24]
    movd        AL, INT [eax + 20]
    mov         T0, INT [eax + 28]
    mov         ZEROBITS, T0
    mov         LEN, INT [eax + 16]
    pxor        ZERO, ZERO
    mov         K, LEN
    and         K, -16
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
    mov         LENEND, LEN
    and         LENEND, 7

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

    pop         ebp
    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
    pop         ecx
    pop         ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
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
%undef T1
%undef BLOCK
%undef VALUES
%undef LEN

;
; Prepare data for jsimd_encode_mcu_AC_refine().
;
; GLOBAL(int)
; jsimd_encode_mcu_AC_refine_prepare_sse2(const JCOEF *block,
;                                         const int *jpeg_natural_order_start,
;                                         int Sl, int Al, JCOEF *absvalues,
;                                         size_t *bits)
;
; eax + 8 = const JCOEF *block
; eax + 12 = const int *jpeg_natural_order_start
; eax + 16 = int Sl
; eax + 20 = int Al
; eax + 24 = JCOEF *values
; eax + 28 = size_t *bits

%define ZERO    xmm7
%define ONE     xmm5
%define X0      xmm0
%define X1      xmm1
%define N0      xmm2
%define N1      xmm3
%define AL      xmm4
%define K       eax
%define LENEND  eax
%define LUT     ebx
%define T0      ecx
%define T0w      cx
%define T1      edx
%define BLOCK   esi
%define VALUES  edi
%define KK      ebp

%define ZEROBITS  INT [esp + 5 * 4]
%define EOB       INT [esp + 5 * 4 + 4]
%define LEN       INT [esp + 5 * 4 + 8]

    align       32
    GLOBAL_FUNCTION(jsimd_encode_mcu_AC_refine_prepare_sse2)

EXTN(jsimd_encode_mcu_AC_refine_prepare_sse2):
    push        ebp
    mov         eax, esp                     ; eax = original ebp
    sub         esp, byte 4
    and         esp, byte (-SIZEOF_XMMWORD)  ; align to 128 bits
    mov         [esp], eax
    mov         ebp, esp                     ; ebp = aligned ebp
    sub         esp, 16
    push        ebx
    push        ecx
;   push        edx                     ; need not be preserved
    push        esi
    push        edi
    push        ebp

    pcmpeqw     ONE, ONE
    psrlw       ONE, 15
    mov         BLOCK, INT [eax + 8]
    mov         LUT, INT [eax + 12]
    mov         VALUES, INT [eax + 24]
    movd        AL, INT [eax + 20]
    mov         T0, INT [eax + 28]
    mov         K,  INT [eax + 16]
    mov         INT [T0 + 2 * SIZEOF_INT], -1
    mov         INT [T0 + 3 * SIZEOF_INT], -1
    mov         ZEROBITS, T0
    mov         LEN, K
    pxor        ZERO, ZERO
    and         K, -16
    mov         EOB, 0
    xor         KK, KK
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
    pmovmskb    T0, N0                  ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    mov         T1, ZEROBITS
    not         T0
    mov         word [T1 + 2 * SIZEOF_INT + KK], T0w
    pmovmskb    T1, X0                  ; idx = _mm_movemask_epi8(x1);
    bsr         T1, T1                  ;  idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER16            ; if (idx) {
    lea         T1, [T1+KK*8]
    mov         EOB, T1                 ; EOB = k + idx;
.CONTINUER16:
    add         VALUES, 16*2
    add         LUT, 16*SIZEOF_INT
    add         KK, 2
    dec         K
    jnz         .BLOOPR16
    test        LEN, 15
    je          .PADDINGR
.ELOOPR16:
    mov         LENEND, LEN

    test        LENEND, 8
    jz          .TRYR7
    test        LENEND, 7
    jz          .TRYR8

    and         LENEND, 7
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
    pmovmskb    T0, N0                  ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    mov         T1, ZEROBITS
    not         T0
    mov         word [T1 + 2 * SIZEOF_INT + KK], T0w
    pmovmskb    T1, X0                  ; idx = _mm_movemask_epi8(x1);
    bsr         T1, T1                  ;  idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER15            ; if (idx) {
    lea         T1, [T1+KK*8]
    mov         EOB, T1                 ; EOB = k + idx;
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
    pmovmskb    T0, N0                  ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    mov         T1, ZEROBITS
    not         T0
    mov         word [T1 + 2 * SIZEOF_INT + KK], T0w
    pmovmskb    T1, X0                  ; idx = _mm_movemask_epi8(x1);
    bsr         T1, T1                  ;  idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER8             ; if (idx) {
    lea         T1, [T1+KK*8]
    mov         EOB, T1                 ; EOB = k + idx;
.CONTINUER8:
    add         VALUES, 8*2
    jmp         .PADDINGR
.TRYR7:
    and         LENEND, 7
    LOAD7

    pcmpgtw     N0, X0
    paddw       X0, N0
    pxor        X0, N0
    psrlw       X0, AL
    movdqa      XMMWORD [VALUES + (0) * 2], X0
    pcmpeqw     X0, ONE
    packsswb    N0, ZERO
    packsswb    X0, ZERO
    pmovmskb    T0, N0                  ; lsignbits.val16u[k>>4] = _mm_movemask_epi8(neg);
    mov         T1, ZEROBITS
    not         T0
    mov         word [T1 + 2 * SIZEOF_INT + KK], T0w
    pmovmskb    T1, X0                  ; idx = _mm_movemask_epi8(x1);
    bsr         T1, T1                  ;  idx = 16 - (__builtin_clz(idx)>>1);
    jz          .CONTINUER7             ; if (idx) {
    lea         T1, [T1+KK*8]
    mov         EOB, T1                 ; EOB = k + idx;
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
    add         VALUES, 8*2
    inc         K
    jnz         .ZEROLOOPR
.EPADDINGR:
    sub         VALUES, DCTSIZE2*2

    REDUCE0

    mov         eax, EOB

    pop         ebp
    pop         edi
    pop         esi
;   pop         edx                     ; need not be preserved
    pop         ecx
    pop         ebx
    mov         esp, ebp                ; esp <- aligned ebp
    pop         esp                     ; esp <- original ebp
    pop         ebp
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
%undef T1
%undef BLOCK
%undef VALUES
%undef LEN
%undef LENEND

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
