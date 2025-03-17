;
; jchuff-sse2.asm - Huffman entropy encoding (SSE2)
;
; Copyright (C) 2009-2011, 2014-2017, 2019, 2024, D. R. Commander.
; Copyright (C) 2015, Matthieu Darbois.
; Copyright (C) 2018, Matthias Räncker.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc
;
; This file should be assembled with NASM (Netwide Assembler) or Yasm.
;
; This file contains an SSE2 implementation for Huffman coding of one block.
; The following code is based on jchuff.c; see jchuff.c for more details.

%include "jsimdext.inc"

struc working_state
.next_output_byte:   resp 1     ; => next byte to write in buffer
.free_in_buffer:     resp 1     ; # of byte spaces remaining in buffer
.cur.put_buffer.simd resq 1     ; current bit accumulation buffer
.cur.free_bits       resd 1     ; # of bits available in it
.cur.last_dc_val     resd 4     ; last DC coef for each component
.cinfo:              resp 1     ; dump_buffer needs access to this
endstruc

struc c_derived_tbl
.ehufco:             resd 256   ; code for each symbol
.ehufsi:             resb 256   ; length of code for each symbol
; If no code has been allocated for a symbol S, ehufsi[S] contains 0
endstruc

; --------------------------------------------------------------------------
    SECTION     SEG_CONST

    GLOBAL_DATA(jconst_huff_encode_one_block)

EXTN(jconst_huff_encode_one_block):

    ALIGNZ      32

jpeg_mask_bits dq 0x0000, 0x0001, 0x0003, 0x0007
               dq 0x000f, 0x001f, 0x003f, 0x007f
               dq 0x00ff, 0x01ff, 0x03ff, 0x07ff
               dq 0x0fff, 0x1fff, 0x3fff, 0x7fff

times 1 << 14 db 15
times 1 << 13 db 14
times 1 << 12 db 13
times 1 << 11 db 12
times 1 << 10 db 11
times 1 <<  9 db 10
times 1 <<  8 db  9
times 1 <<  7 db  8
times 1 <<  6 db  7
times 1 <<  5 db  6
times 1 <<  4 db  5
times 1 <<  3 db  4
times 1 <<  2 db  3
times 1 <<  1 db  2
times 1 <<  0 db  1
times 1       db  0
GLOBAL_DATA(jpeg_nbits_table)
EXTN(jpeg_nbits_table):
times 1       db  0
times 1 <<  0 db  1
times 1 <<  1 db  2
times 1 <<  2 db  3
times 1 <<  3 db  4
times 1 <<  4 db  5
times 1 <<  5 db  6
times 1 <<  6 db  7
times 1 <<  7 db  8
times 1 <<  8 db  9
times 1 <<  9 db 10
times 1 << 10 db 11
times 1 << 11 db 12
times 1 << 12 db 13
times 1 << 13 db 14
times 1 << 14 db 15

    ALIGNZ      32

%ifdef PIC
%define NBITS(x)      nbits_base + x
%else
%define NBITS(x)      EXTN(jpeg_nbits_table) + x
%endif
%define MASK_BITS(x)  NBITS((x) * 8) + (jpeg_mask_bits - EXTN(jpeg_nbits_table))

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        32

%define mm_put_buffer     mm0
%define mm_all_0xff       mm1
%define mm_temp           mm2
%define mm_nbits          mm3
%define mm_code_bits      mm3
%define mm_code           mm4
%define mm_overflow_bits  mm5
%define mm_save_nbits     mm6

; Shorthand used to describe SIMD operations:
; wN:  xmmN treated as eight signed 16-bit values
; wN[i]:  perform the same operation on all eight signed 16-bit values, i=0..7
; bN:  xmmN treated as 16 unsigned 8-bit values, or
;      mmN treated as eight unsigned 8-bit values
; bN[i]:  perform the same operation on all unsigned 8-bit values,
;         i=0..15 (SSE register) or i=0..7 (MMX register)
; Contents of SIMD registers are shown in memory order.

; Fill the bit buffer to capacity with the leading bits from code, then output
; the bit buffer and put the remaining bits from code into the bit buffer.
;
; Usage:
; code - contains the bits to shift into the bit buffer (LSB-aligned)
; %1 - temp register
; %2 - low byte of temp register
; %3 - second byte of temp register
; %4-%8 (optional) - extra instructions to execute before the macro completes
; %9 - the label to which to jump when the macro completes
;
; Upon completion, free_bits will be set to the number of remaining bits from
; code, and put_buffer will contain those remaining bits.  temp and code will
; be clobbered.
;
; This macro encodes any 0xFF bytes as 0xFF 0x00, as does the EMIT_BYTE()
; macro in jchuff.c.

%macro EMIT_QWORD 9
%define %%temp   %1
%define %%tempb  %2
%define %%temph  %3
    add         nbits, free_bits             ; nbits += free_bits;
    neg         free_bits                    ; free_bits = -free_bits;
    movq        mm_temp, mm_code             ; temp = code;
    movd        mm_nbits, nbits              ; nbits --> MMX register
    movd        mm_overflow_bits, free_bits  ; overflow_bits (temp register) = free_bits;
    neg         free_bits                    ; free_bits = -free_bits;
    psllq       mm_put_buffer, mm_nbits      ; put_buffer <<= nbits;
    psrlq       mm_temp, mm_overflow_bits    ; temp >>= overflow_bits;
    add         free_bits, 64                ; free_bits += 64;
    por         mm_temp, mm_put_buffer       ; temp |= put_buffer;
%ifidn %%temp, nbits_base
    movd        mm_save_nbits, nbits_base    ; save nbits_base
%endif
    movq        mm_code_bits, mm_temp        ; code_bits (temp register) = temp;
    movq        mm_put_buffer, mm_code       ; put_buffer = code;
    pcmpeqb     mm_temp, mm_all_0xff         ; b_temp[i] = (b_temp[i] == 0xFF ? 0xFF : 0);
    movq        mm_code, mm_code_bits        ; code = code_bits;
    psrlq       mm_code_bits, 32             ; code_bits >>= 32;
    pmovmskb    nbits, mm_temp               ; nbits = 0;  nbits |= ((b_temp[i] >> 7) << i);
    movd        %%temp, mm_code_bits         ; temp = code_bits;
    bswap       %%temp                       ; temp = htonl(temp);
    test        nbits, nbits                 ; if (nbits != 0)  /* Some 0xFF bytes */
    jnz         %%.SLOW                      ;   goto %%.SLOW
    mov         dword [buffer], %%temp       ; *(uint32_t)buffer = temp;
%ifidn %%temp, nbits_base
    movd        nbits_base, mm_save_nbits    ; restore nbits_base
%endif
    %4
    movd        nbits, mm_code               ; nbits = (uint32_t)(code);
    %5
    bswap       nbits                        ; nbits = htonl(nbits);
    mov         dword [buffer + 4], nbits    ; *(uint32_t)(buffer + 4) = nbits;
    lea         buffer, [buffer + 8]         ; buffer += 8;
    %6
    %7
    %8
    jmp %9                                   ; return
%%.SLOW:
    ; Execute the equivalent of the EMIT_BYTE() macro in jchuff.c for all 8
    ; bytes in the qword.
    mov         byte [buffer], %%tempb     ; buffer[0] = temp[0];
    cmp         %%tempb, 0xFF              ; Set CF if temp[0] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (temp[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], %%temph     ; buffer[0] = temp[1];
    cmp         %%temph, 0xFF              ; Set CF if temp[1] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (temp[1] < 0xFF ? 1 : 0));
    shr         %%temp, 16                 ; temp >>= 16;
    mov         byte [buffer], %%tempb     ; buffer[0] = temp[0];
    cmp         %%tempb, 0xFF              ; Set CF if temp[0] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (temp[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], %%temph     ; buffer[0] = temp[1];
    cmp         %%temph, 0xFF              ; Set CF if temp[1] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (temp[1] < 0xFF ? 1 : 0));
    movd        nbits, mm_code             ; nbits (temp register) = (uint32_t)(code)
%ifidn %%temp, nbits_base
    movd        nbits_base, mm_save_nbits  ; restore nbits_base
%endif
    bswap       nbits                      ; nbits = htonl(nbits)
    mov         byte [buffer], nbitsb      ; buffer[0] = nbits[0];
    cmp         nbitsb, 0xFF               ; Set CF if nbits[0] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (nbits[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], nbitsh      ; buffer[0] = nbits[1];
    cmp         nbitsh, 0xFF               ; Set CF if nbits[1] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (nbits[1] < 0xFF ? 1 : 0));
    shr         nbits, 16                  ; nbits >>= 16;
    mov         byte [buffer], nbitsb      ; buffer[0] = nbits[0];
    cmp         nbitsb, 0xFF               ; Set CF if nbits[0] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (nbits[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], nbitsh      ; buffer[0] = nbits[1];
    %4
    cmp         nbitsh, 0xFF               ; Set CF if nbits[1] < 0xFF
    mov         byte [buffer+1], 0         ; buffer[1] = 0;
    sbb         buffer, -2                 ; buffer -= (-2 + (nbits[1] < 0xFF ? 1 : 0));
    %5
    %6
    %7
    %8
    jmp %9                                 ; return;
%endmacro

%macro PUSH 1
    push        %1
%assign stack_offset  stack_offset + 4
%endmacro

%macro POP 1
    pop         %1
%assign stack_offset  stack_offset - 4
%endmacro

; If PIC is defined, load the address of a symbol defined in this file into a
; register.  Equivalent to
;   GET_GOT     %1
;   lea         %1, [GOTOFF(%1, %2)]
; without using the GOT.
;
; Usage:
; %1 - register into which to load the address of the symbol
; %2 - symbol whose address should be loaded
; %3 - optional multi-line macro to execute before the symbol address is loaded
; %4 - optional multi-line macro to execute after the symbol address is loaded
;
; If PIC is not defined, then %3 and %4 are executed in order.

%macro GET_SYM 2-4
%ifdef PIC
    call        %%.geteip
%%.ref:
    %4
    add         %1, %2 - %%.ref
    jmp         short %%.done
    align       32
%%.geteip:
    %3          4               ; must adjust stack pointer because of call
    mov         %1, POINTER [esp]
    ret
    align       32
%%.done:
%else
    %3          0
    %4
%endif
%endmacro

;
; Encode a single block's worth of coefficients.
;
; GLOBAL(JOCTET *)
; jsimd_huff_encode_one_block_sse2(working_state *state, JOCTET *buffer,
;                                  JCOEFPTR block, int last_dc_val,
;                                  c_derived_tbl *dctbl, c_derived_tbl *actbl)
;
; Stack layout:
; Function args
; Return address
; Saved ebx
; Saved ebp
; Saved esi
; Saved edi <-- esp_save
; ...
; esp_save
; t_ 64*2 bytes (aligned to 128 bytes)
;
; esp is used (as t) to point into t_ (data in lower indices is not used once
; esp passes over them, so this is signal-safe.)  Aligning to 128 bytes allows
; us to find the rest of the data again.
;
; NOTES:
; When shuffling data, we try to avoid pinsrw as much as possible, since it is
; slow on many CPUs.  Its reciprocal throughput (issue latency) is 1 even on
; modern CPUs, so chains of pinsrw instructions (even with different outputs)
; can limit performance.  pinsrw is a VectorPath instruction on AMD K8 and
; requires 2 µops (with memory operand) on Intel.  In either case, only one
; pinsrw instruction can be decoded per cycle (and nothing else if they are
; back-to-back), so out-of-order execution cannot be used to work around long
; pinsrw chains (though for Sandy Bridge and later, this may be less of a
; problem if the code runs from the µop cache.)
;
; We use tzcnt instead of bsf without checking for support.  The instruction is
; executed as bsf on CPUs that don't support tzcnt (encoding is equivalent to
; rep bsf.)  The destination (first) operand of bsf (and tzcnt on some CPUs) is
; an input dependency (although the behavior is not formally defined, Intel
; CPUs usually leave the destination unmodified if the source is zero.)  This
; can prevent out-of-order execution, so we clear the destination before
; invoking tzcnt.
;
; Initial register allocation
; eax - frame --> buffer
; ebx - nbits_base (PIC) / emit_temp
; ecx - dctbl --> size --> state
; edx - block --> nbits
; esi - code_temp --> state --> actbl
; edi - index_temp --> free_bits
; esp - t
; ebp - index

%define frame       eax
%ifdef PIC
%define nbits_base  ebx
%endif
%define emit_temp   ebx
%define emit_tempb  bl
%define emit_temph  bh
%define dctbl       ecx
%define block       edx
%define code_temp   esi
%define index_temp  edi
%define t           esp
%define index       ebp

%assign save_frame  DCTSIZE2 * SIZEOF_WORD

; Step 1: Re-arrange input data according to jpeg_natural_order
; xx 01 02 03 04 05 06 07      xx 01 08 16 09 02 03 10
; 08 09 10 11 12 13 14 15      17 24 32 25 18 11 04 05
; 16 17 18 19 20 21 22 23      12 19 26 33 40 48 41 34
; 24 25 26 27 28 29 30 31 ==>  27 20 13 06 07 14 21 28
; 32 33 34 35 36 37 38 39      35 42 49 56 57 50 43 36
; 40 41 42 43 44 45 46 47      29 22 15 23 30 37 44 51
; 48 49 50 51 52 53 54 55      58 59 52 45 38 31 39 46
; 56 57 58 59 60 61 62 63      53 60 61 54 47 55 62 63

    align       32
    GLOBAL_FUNCTION(jsimd_huff_encode_one_block_sse2)

EXTN(jsimd_huff_encode_one_block_sse2):

%assign stack_offset      0
%define arg_state         4 + stack_offset
%define arg_buffer        8 + stack_offset
%define arg_block        12 + stack_offset
%define arg_last_dc_val  16 + stack_offset
%define arg_dctbl        20 + stack_offset
%define arg_actbl        24 + stack_offset

                                                          ;X: X = code stream
    mov         block, [esp + arg_block]
    PUSH        ebx
    PUSH        ebp
    movups      xmm3, XMMWORD [block + 0 * SIZEOF_WORD]   ;D: w3 = xx 01 02 03 04 05 06 07
    PUSH        esi
    PUSH        edi
    movdqa      xmm0, xmm3                                ;A: w0 = xx 01 02 03 04 05 06 07
    mov         frame, esp
    lea         t, [frame - (save_frame + 4)]
    movups      xmm1, XMMWORD [block + 8 * SIZEOF_WORD]   ;B: w1 = 08 09 10 11 12 13 14 15
    and         t, -DCTSIZE2 * SIZEOF_WORD                                             ; t = &t_[0]
    mov         [t + save_frame], frame
    pxor        xmm4, xmm4                                ;A: w4[i] = 0;
    punpckldq   xmm0, xmm1                                ;A: w0 = xx 01 08 09 02 03 10 11
    pshuflw     xmm0, xmm0, 11001001b                     ;A: w0 = 01 08 xx 09 02 03 10 11
    pinsrw      xmm0, word [block + 16 * SIZEOF_WORD], 2  ;A: w0 = 01 08 16 09 02 03 10 11
    punpckhdq   xmm3, xmm1                                ;D: w3 = 04 05 12 13 06 07 14 15
    punpcklqdq  xmm1, xmm3                                ;B: w1 = 08 09 10 11 04 05 12 13
    pinsrw      xmm0, word [block + 17 * SIZEOF_WORD], 7  ;A: w0 = 01 08 16 09 02 03 10 17
                                                          ;A:      (Row 0, offset 1)
    pcmpgtw     xmm4, xmm0                                ;A: w4[i] = (w0[i] < 0 ? -1 : 0);
    paddw       xmm0, xmm4                                ;A: w0[i] += w4[i];
    movaps      XMMWORD [t + 0 * SIZEOF_WORD], xmm0       ;A: t[i] = w0[i];

    movq        xmm2, qword [block + 24 * SIZEOF_WORD]    ;B: w2 = 24 25 26 27 -- -- -- --
    pshuflw     xmm2, xmm2, 11011000b                     ;B: w2 = 24 26 25 27 -- -- -- --
    pslldq      xmm1, 1 * SIZEOF_WORD                     ;B: w1 = -- 08 09 10 11 04 05 12
    movups      xmm5, XMMWORD [block + 48 * SIZEOF_WORD]  ;H: w5 = 48 49 50 51 52 53 54 55
    movsd       xmm1, xmm2                                ;B: w1 = 24 26 25 27 11 04 05 12
    punpcklqdq  xmm2, xmm5                                ;C: w2 = 24 26 25 27 48 49 50 51
    pinsrw      xmm1, word [block + 32 * SIZEOF_WORD], 1  ;B: w1 = 24 32 25 27 11 04 05 12
    pxor        xmm4, xmm4                                ;A: w4[i] = 0;
    psrldq      xmm3, 2 * SIZEOF_WORD                     ;D: w3 = 12 13 06 07 14 15 -- --
    pcmpeqw     xmm0, xmm4                                ;A: w0[i] = (w0[i] == 0 ? -1 : 0);
    pinsrw      xmm1, word [block + 18 * SIZEOF_WORD], 3  ;B: w1 = 24 32 25 18 11 04 05 12
                                                          ;        (Row 1, offset 1)
    pcmpgtw     xmm4, xmm1                                ;B: w4[i] = (w1[i] < 0 ? -1 : 0);
    paddw       xmm1, xmm4                                ;B: w1[i] += w4[i];
    movaps      XMMWORD [t + 8 * SIZEOF_WORD], xmm1       ;B: t[i+8] = w1[i];
    pxor        xmm4, xmm4                                ;B: w4[i] = 0;
    pcmpeqw     xmm1, xmm4                                ;B: w1[i] = (w1[i] == 0 ? -1 : 0);

    packsswb    xmm0, xmm1                                ;AB: b0[i] = w0[i], b0[i+8] = w1[i]
                                                          ;    w/ signed saturation

    pinsrw      xmm3, word [block + 20 * SIZEOF_WORD], 0  ;D: w3 = 20 13 06 07 14 15 -- --
    pinsrw      xmm3, word [block + 21 * SIZEOF_WORD], 5  ;D: w3 = 20 13 06 07 14 21 -- --
    pinsrw      xmm3, word [block + 28 * SIZEOF_WORD], 6  ;D: w3 = 20 13 06 07 14 21 28 --
    pinsrw      xmm3, word [block + 35 * SIZEOF_WORD], 7  ;D: w3 = 20 13 06 07 14 21 28 35
                                                          ;        (Row 3, offset 1)
    pcmpgtw     xmm4, xmm3                                ;D: w4[i] = (w3[i] < 0 ? -1 : 0);
    paddw       xmm3, xmm4                                ;D: w3[i] += w4[i];
    movaps      XMMWORD [t + 24 * SIZEOF_WORD], xmm3      ;D: t[i+24] = w3[i];
    pxor        xmm4, xmm4                                ;D: w4[i] = 0;
    pcmpeqw     xmm3, xmm4                                ;D: w3[i] = (w3[i] == 0 ? -1 : 0);

    pinsrw      xmm2, word [block + 19 * SIZEOF_WORD], 0  ;C: w2 = 19 26 25 27 48 49 50 51
    pinsrw      xmm2, word [block + 33 * SIZEOF_WORD], 2  ;C: w2 = 19 26 33 27 48 49 50 51
    pinsrw      xmm2, word [block + 40 * SIZEOF_WORD], 3  ;C: w2 = 19 26 33 40 48 49 50 51
    pinsrw      xmm2, word [block + 41 * SIZEOF_WORD], 5  ;C: w2 = 19 26 33 40 48 41 50 51
    pinsrw      xmm2, word [block + 34 * SIZEOF_WORD], 6  ;C: w2 = 19 26 33 40 48 41 34 51
    pinsrw      xmm2, word [block + 27 * SIZEOF_WORD], 7  ;C: w2 = 19 26 33 40 48 41 34 27
                                                          ;        (Row 2, offset 1)
    pcmpgtw     xmm4, xmm2                                ;C: w4[i] = (w2[i] < 0 ? -1 : 0);
    paddw       xmm2, xmm4                                ;C: w2[i] += w4[i];
    movsx       code_temp, word [block]                   ;Z:     code_temp = block[0];

; %1 - stack pointer adjustment
%macro GET_SYM_BEFORE 1
    movaps      XMMWORD [t + 16 * SIZEOF_WORD + %1], xmm2
                                                          ;C: t[i+16] = w2[i];
    pxor        xmm4, xmm4                                ;C: w4[i] = 0;
    pcmpeqw     xmm2, xmm4                                ;C: w2[i] = (w2[i] == 0 ? -1 : 0);
    sub         code_temp, [frame + arg_last_dc_val]      ;Z:     code_temp -= last_dc_val;

    packsswb    xmm2, xmm3                                ;CD: b2[i] = w2[i], b2[i+8] = w3[i]
                                                          ;    w/ signed saturation

    movdqa      xmm3, xmm5                                ;H: w3 = 48 49 50 51 52 53 54 55
    pmovmskb    index_temp, xmm2                          ;Z:     index_temp = 0;  index_temp |= ((b2[i] >> 7) << i);
    pmovmskb    index, xmm0                               ;Z:     index = 0;  index |= ((b0[i] >> 7) << i);
    movups      xmm0, XMMWORD [block + 56 * SIZEOF_WORD]  ;H: w0 = 56 57 58 59 60 61 62 63
    punpckhdq   xmm3, xmm0                                ;H: w3 = 52 53 60 61 54 55 62 63
    shl         index_temp, 16                            ;Z:     index_temp <<= 16;
    psrldq      xmm3, 1 * SIZEOF_WORD                     ;H: w3 = 53 60 61 54 55 62 63 --
    pxor        xmm2, xmm2                                ;H: w2[i] = 0;
    pshuflw     xmm3, xmm3, 00111001b                     ;H: w3 = 60 61 54 53 55 62 63 --
    or          index, index_temp                         ;Z:     index |= index_temp;
%undef index_temp
%define free_bits  edi
%endmacro

%macro GET_SYM_AFTER 0
    movq        xmm1, qword [block + 44 * SIZEOF_WORD]    ;G: w1 = 44 45 46 47 -- -- -- --
    unpcklps    xmm5, xmm0                                ;E: w5 = 48 49 56 57 50 51 58 59
    pxor        xmm0, xmm0                                ;H: w0[i] = 0;
    not         index                                     ;Z:     index = ~index;
    pinsrw      xmm3, word [block + 47 * SIZEOF_WORD], 3  ;H: w3 = 60 61 54 47 55 62 63 --
                                                          ;        (Row 7, offset 1)
    pcmpgtw     xmm2, xmm3                                ;H: w2[i] = (w3[i] < 0 ? -1 : 0);
    mov         dctbl, [frame + arg_dctbl]
    paddw       xmm3, xmm2                                ;H: w3[i] += w2[i];
    movaps      XMMWORD [t + 56 * SIZEOF_WORD], xmm3      ;H: t[i+56] = w3[i];
    movq        xmm4, qword [block + 36 * SIZEOF_WORD]    ;G: w4 = 36 37 38 39 -- -- -- --
    pcmpeqw     xmm3, xmm0                                ;H: w3[i] = (w3[i] == 0 ? -1 : 0);
    punpckldq   xmm4, xmm1                                ;G: w4 = 36 37 44 45 38 39 46 47
    movdqa      xmm1, xmm4                                ;F: w1 = 36 37 44 45 38 39 46 47
    pcmpeqw     mm_all_0xff, mm_all_0xff                  ;Z:     all_0xff[i] = 0xFF;
%endmacro

    GET_SYM     nbits_base, EXTN(jpeg_nbits_table), GET_SYM_BEFORE, GET_SYM_AFTER

    psrldq      xmm4, 1 * SIZEOF_WORD                     ;G: w4 = 37 44 45 38 39 46 47 --
    shufpd      xmm1, xmm5, 10b                           ;F: w1 = 36 37 44 45 50 51 58 59
    pshufhw     xmm4, xmm4, 11010011b                     ;G: w4 = 37 44 45 38 -- 39 46 --
    pslldq      xmm1, 1 * SIZEOF_WORD                     ;F: w1 = -- 36 37 44 45 50 51 58
    pinsrw      xmm4, word [block + 59 * SIZEOF_WORD], 0  ;G: w4 = 59 44 45 38 -- 39 46 --
    pshufd      xmm1, xmm1, 11011000b                     ;F: w1 = -- 36 45 50 37 44 51 58
    cmp         code_temp, 1 << 31                        ;Z:     Set CF if code_temp < 0x80000000,
                                                          ;Z:     i.e. if code_temp is positive
    pinsrw      xmm4, word [block + 52 * SIZEOF_WORD], 1  ;G: w4 = 59 52 45 38 -- 39 46 --
    movlps      xmm1, qword [block + 20 * SIZEOF_WORD]    ;F: w1 = 20 21 22 23 37 44 51 58
    pinsrw      xmm4, word [block + 31 * SIZEOF_WORD], 4  ;G: w4 = 59 52 45 38 31 39 46 --
    pshuflw     xmm1, xmm1, 01110010b                     ;F: w1 = 22 20 23 21 37 44 51 58
    pinsrw      xmm4, word [block + 53 * SIZEOF_WORD], 7  ;G: w4 = 59 52 45 38 31 39 46 53
                                                          ;        (Row 6, offset 1)
    adc         code_temp, -1                             ;Z:     code_temp += -1 + (code_temp >= 0 ? 1 : 0);
    pxor        xmm2, xmm2                                ;G: w2[i] = 0;
    pcmpgtw     xmm0, xmm4                                ;G: w0[i] = (w4[i] < 0 ? -1 : 0);
    pinsrw      xmm1, word [block + 15 * SIZEOF_WORD], 1  ;F: w1 = 22 15 23 21 37 44 51 58
    paddw       xmm4, xmm0                                ;G: w4[i] += w0[i];
    movaps      XMMWORD [t + 48 * SIZEOF_WORD], xmm4      ;G: t[48+i] = w4[i];
    movd        mm_temp, code_temp                        ;Z:     temp = code_temp
    pinsrw      xmm1, word [block + 30 * SIZEOF_WORD], 3  ;F: w1 = 22 15 23 30 37 44 51 58
                                                          ;        (Row 5, offset 1)
    pcmpeqw     xmm4, xmm2                                ;G: w4[i] = (w4[i] == 0 ? -1 : 0);

    packsswb    xmm4, xmm3                                ;GH: b4[i] = w4[i], b4[i+8] = w3[i]
                                                          ;    w/ signed saturation

    lea         t, [t - SIZEOF_WORD]                      ;Z:     t = &t[-1]
    pxor        xmm0, xmm0                                ;F: w0[i] = 0;
    pcmpgtw     xmm2, xmm1                                ;F: w2[i] = (w1[i] < 0 ? -1 : 0);
    paddw       xmm1, xmm2                                ;F: w1[i] += w2[i];
    movaps      XMMWORD [t + (40+1) * SIZEOF_WORD], xmm1  ;F: t[40+i] = w1[i];
    pcmpeqw     xmm1, xmm0                                ;F: w1[i] = (w1[i] == 0 ? -1 : 0);
    pinsrw      xmm5, word [block + 42 * SIZEOF_WORD], 0  ;E: w5 = 42 49 56 57 50 51 58 59
    pinsrw      xmm5, word [block + 43 * SIZEOF_WORD], 5  ;E: w5 = 42 49 56 57 50 43 58 59
    pinsrw      xmm5, word [block + 36 * SIZEOF_WORD], 6  ;E: w5 = 42 49 56 57 50 43 36 59
    pinsrw      xmm5, word [block + 29 * SIZEOF_WORD], 7  ;E: w5 = 42 49 56 57 50 43 36 29
                                                          ;        (Row 4, offset 1)
%undef block
%define nbits  edx
%define nbitsb  dl
%define nbitsh  dh
    movzx       nbits, byte [NBITS(code_temp)]            ;Z:     nbits = JPEG_NBITS(code_temp);
%undef code_temp
%define state  esi
    pxor        xmm2, xmm2                                ;E: w2[i] = 0;
    mov         state, [frame + arg_state]
    movd        mm_nbits, nbits                           ;Z:     nbits --> MMX register
    pcmpgtw     xmm0, xmm5                                ;E: w0[i] = (w5[i] < 0 ? -1 : 0);
    movd        mm_code, dword [dctbl + c_derived_tbl.ehufco + nbits * 4]
                                                          ;Z:     code = dctbl->ehufco[nbits];
%define size  ecx
%define sizeb  cl
%define sizeh  ch
    paddw       xmm5, xmm0                                ;E: w5[i] += w0[i];
    movaps      XMMWORD [t + (32+1) * SIZEOF_WORD], xmm5  ;E: t[32+i] = w5[i];
    movzx       size, byte [dctbl + c_derived_tbl.ehufsi + nbits]
                                                          ;Z:     size = dctbl->ehufsi[nbits];
%undef dctbl
    pcmpeqw     xmm5, xmm2                                ;E: w5[i] = (w5[i] == 0 ? -1 : 0);

    packsswb    xmm5, xmm1                                ;EF: b5[i] = w5[i], b5[i+8] = w1[i]
                                                          ;    w/ signed saturation

    movq        mm_put_buffer, [state + working_state.cur.put_buffer.simd]
                                                          ;Z:     put_buffer = state->cur.put_buffer.simd;
    mov         free_bits, [state + working_state.cur.free_bits]
                                                          ;Z:     free_bits = state->cur.free_bits;
%undef state
%define actbl  esi
    mov         actbl, [frame + arg_actbl]
%define buffer  eax
    mov         buffer, [frame + arg_buffer]
%undef frame
    jmp        .BEGIN

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
; size <= 32, so this is not really a loop
.BRLOOP1:                                                 ; .BRLOOP1:
    movzx       nbits, byte [actbl + c_derived_tbl.ehufsi + 0xf0]
                                                          ; nbits = actbl->ehufsi[0xf0];
    movd        mm_code, dword [actbl + c_derived_tbl.ehufco + 0xf0 * 4]
                                                          ; code = actbl->ehufco[0xf0];
    and         index, 0x7ffffff                          ; clear index if size == 32
    sub         size, 16                                  ; size -= 16;
    sub         free_bits, nbits                          ; if ((free_bits -= nbits) <= 0)
    jle         .EMIT_BRLOOP1                             ;   goto .EMIT_BRLOOP1;
    movd        mm_nbits, nbits                           ; nbits --> MMX register
    psllq       mm_put_buffer, mm_nbits                   ; put_buffer <<= nbits;
    por         mm_put_buffer, mm_code                    ; put_buffer |= code;
    jmp         .ERLOOP1                                  ; goto .ERLOOP1;

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
%ifdef PIC
    times 6     nop
%else
    times 2     nop
%endif
.BLOOP1:                                                  ; do {  /* size = # of zero bits/elements to skip */
; if size == 32, index remains unchanged.  Correct in .BRLOOP.
    shr         index, sizeb                              ;   index >>= size;
    lea         t, [t + size * SIZEOF_WORD]               ;   t += size;
    cmp         size, 16                                  ;   if (size > 16)
    jg          .BRLOOP1                                  ;     goto .BRLOOP1;
.ERLOOP1:                                                 ; .ERLOOP1:
    movsx       nbits, word [t]                           ;   nbits = *t;
%ifdef PIC
    add         size, size                                ;   size += size;
%else
    lea         size, [size * 2]                          ;   size += size;
%endif
    movd        mm_temp, nbits                            ;   temp = nbits;
    movzx       nbits, byte [NBITS(nbits)]                ;   nbits = JPEG_NBITS(nbits);
    lea         size, [size * 8 + nbits]                  ;   size = size * 8 + nbits;
    movd        mm_nbits, nbits                           ;   nbits --> MMX register
    movd        mm_code, dword [actbl + c_derived_tbl.ehufco + (size - 16) * 4]
                                                          ;   code = actbl->ehufco[size-16];
    movzx       size, byte [actbl + c_derived_tbl.ehufsi + (size - 16)]
                                                          ;   size = actbl->ehufsi[size-16];
.BEGIN:                                                   ; .BEGIN:
    pand        mm_temp, [MASK_BITS(nbits)]               ;   temp &= (1 << nbits) - 1;
    psllq       mm_code, mm_nbits                         ;   code <<= nbits;
    add         nbits, size                               ;   nbits += size;
    por         mm_code, mm_temp                          ;   code |= temp;
    sub         free_bits, nbits                          ;   if ((free_bits -= nbits) <= 0)
    jle         .EMIT_ERLOOP1                             ;     insert code, flush buffer, init size, goto .BLOOP1
    xor         size, size                                ;   size = 0;  /* kill tzcnt input dependency */
    tzcnt       size, index                               ;   size = # of trailing 0 bits in index
    movd        mm_nbits, nbits                           ;   nbits --> MMX register
    psllq       mm_put_buffer, mm_nbits                   ;   put_buffer <<= nbits;
    inc         size                                      ;   ++size;
    por         mm_put_buffer, mm_code                    ;   put_buffer |= code;
    test        index, index
    jnz         .BLOOP1                                   ; } while (index != 0);
; Round 2
; t points to the last used word, possibly below t_ if the previous index had 32 zero bits.
.ELOOP1:                                                  ; .ELOOP1:
    pmovmskb    size, xmm4                                ; size = 0;  size |= ((b4[i] >> 7) << i);
    pmovmskb    index, xmm5                               ; index = 0;  index |= ((b5[i] >> 7) << i);
    shl         size, 16                                  ; size <<= 16;
    or          index, size                               ; index |= size;
    not         index                                     ; index = ~index;
    lea         nbits, [t + (1 + DCTSIZE2) * SIZEOF_WORD]
                                                          ; nbits = t + 1 + 64;
    and         nbits, -DCTSIZE2 * SIZEOF_WORD            ; nbits &= -128;  /* now points to &t_[64] */
    sub         nbits, t                                  ; nbits -= t;
    shr         nbits, 1                                  ; nbits >>= 1;  /* # of leading 0 bits in old index + 33 */
    tzcnt       size, index                               ; size = # of trailing 0 bits in index
    inc         size                                      ; ++size;
    test        index, index                              ; if (index == 0)
    jz          .ELOOP2                                   ;   goto .ELOOP2;
; NOTE: size == 32 cannot happen, since the last element is always 0.
    shr         index, sizeb                              ; index >>= size;
    lea         size, [size + nbits - 33]                 ; size = size + nbits - 33;
    lea         t, [t + size * SIZEOF_WORD]               ; t += size;
    cmp         size, 16                                  ; if (size <= 16)
    jle         .ERLOOP2                                  ;   goto .ERLOOP2;
.BRLOOP2:                                                 ; do {
    movzx       nbits, byte [actbl + c_derived_tbl.ehufsi + 0xf0]
                                                          ;   nbits = actbl->ehufsi[0xf0];
    sub         size, 16                                  ;   size -= 16;
    movd        mm_code, dword [actbl + c_derived_tbl.ehufco + 0xf0 * 4]
                                                          ;   code = actbl->ehufco[0xf0];
    sub         free_bits, nbits                          ;   if ((free_bits -= nbits) <= 0)
    jle         .EMIT_BRLOOP2                             ;     insert code and flush put_buffer
    movd        mm_nbits, nbits                           ;   else { nbits --> MMX register
    psllq       mm_put_buffer, mm_nbits                   ;     put_buffer <<= nbits;
    por         mm_put_buffer, mm_code                    ;     put_buffer |= code;
    cmp         size, 16                                  ;     if (size <= 16)
    jle        .ERLOOP2                                   ;       goto .ERLOOP2;
    jmp        .BRLOOP2                                   ; } while (1);

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align      16
.BLOOP2:                                                  ; do {  /* size = # of zero bits/elements to skip */
    shr         index, sizeb                              ;   index >>= size;
    lea         t, [t + size * SIZEOF_WORD]               ;   t += size;
    cmp         size, 16                                  ;   if (size > 16)
    jg          .BRLOOP2                                  ;     goto .BRLOOP2;
.ERLOOP2:                                                 ; .ERLOOP2:
    movsx       nbits, word [t]                           ;   nbits = *t;
    add         size, size                                ;   size += size;
    movd        mm_temp, nbits                            ;   temp = nbits;
    movzx       nbits, byte [NBITS(nbits)]                ;   nbits = JPEG_NBITS(nbits);
    movd        mm_nbits, nbits                           ;   nbits --> MMX register
    lea         size, [size * 8 + nbits]                  ;   size = size * 8 + nbits;
    movd        mm_code, dword [actbl + c_derived_tbl.ehufco + (size - 16) * 4]
                                                          ;   code = actbl->ehufco[size-16];
    movzx       size, byte [actbl + c_derived_tbl.ehufsi + (size - 16)]
                                                          ;   size = actbl->ehufsi[size-16];
    psllq       mm_code, mm_nbits                         ;   code <<= nbits;
    pand        mm_temp, [MASK_BITS(nbits)]               ;   temp &= (1 << nbits) - 1;
    lea         nbits, [nbits + size]                     ;   nbits += size;
    por         mm_code, mm_temp                          ;   code |= temp;
    xor         size, size                                ;   size = 0;  /* kill tzcnt input dependency */
    sub         free_bits, nbits                          ;   if ((free_bits -= nbits) <= 0)
    jle         .EMIT_ERLOOP2                             ;     insert code, flush buffer, init size, goto .BLOOP2
    tzcnt       size, index                               ;   size = # of trailing 0 bits in index
    movd        mm_nbits, nbits                           ;   nbits --> MMX register
    psllq       mm_put_buffer, mm_nbits                   ;   put_buffer <<= nbits;
    inc         size                                      ;   ++size;
    por         mm_put_buffer, mm_code                    ;   put_buffer |= code;
    test        index, index
    jnz         .BLOOP2                                   ; } while (index != 0);
.ELOOP2:                                                  ; .ELOOP2:
    mov         nbits, t                                  ; nbits = t;
    lea         t, [t + SIZEOF_WORD]                      ; t = &t[1];
    and         nbits, DCTSIZE2 * SIZEOF_WORD - 1         ; nbits &= 127;
    and         t, -DCTSIZE2 * SIZEOF_WORD                ; t &= -128;  /* t = &t_[0]; */
    cmp         nbits, (DCTSIZE2 - 2) * SIZEOF_WORD       ; if (nbits != 62 * 2)
    je          .EFN                                      ; {
    movd        mm_code, dword [actbl + c_derived_tbl.ehufco + 0]
                                                          ;   code = actbl->ehufco[0];
    movzx       nbits, byte [actbl + c_derived_tbl.ehufsi + 0]
                                                          ;   nbits = actbl->ehufsi[0];
    sub         free_bits, nbits                          ;   if ((free_bits -= nbits) <= 0)
    jg          .EFN_SKIP_EMIT_CODE                       ;   {
    EMIT_QWORD  size, sizeb, sizeh, , , , , , .EFN        ;     insert code, flush put_buffer
    align       16
.EFN_SKIP_EMIT_CODE:                                      ;   } else {
    movd        mm_nbits, nbits                           ;     nbits --> MMX register
    psllq       mm_put_buffer, mm_nbits                   ;     put_buffer <<= nbits;
    por         mm_put_buffer, mm_code                    ;     put_buffer |= code;
.EFN:                                                     ; } }
%define frame  esp
    mov         frame, [t + save_frame]
%define state  ecx
    mov         state, [frame + arg_state]
    movq        [state + working_state.cur.put_buffer.simd], mm_put_buffer
                                                          ; state->cur.put_buffer.simd = put_buffer;
    emms
    mov         [state + working_state.cur.free_bits], free_bits
                                                          ; state->cur.free_bits = free_bits;
    POP         edi
    POP         esi
    POP         ebp
    POP         ebx
    ret

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
.EMIT_BRLOOP1:
    EMIT_QWORD  emit_temp, emit_tempb, emit_temph, , , , , , \
      .ERLOOP1

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
.EMIT_ERLOOP1:
    EMIT_QWORD  size, sizeb, sizeh, \
      { xor     size, size }, \
      { tzcnt   size, index }, \
      { inc     size }, \
      { test    index, index }, \
      { jnz     .BLOOP1 }, \
      .ELOOP1

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
.EMIT_BRLOOP2:
    EMIT_QWORD  emit_temp, emit_tempb, emit_temph, , , , \
      { cmp     size, 16 }, \
      { jle     .ERLOOP2 }, \
      .BRLOOP2

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
.EMIT_ERLOOP2:
    EMIT_QWORD  size, sizeb, sizeh, \
      { xor     size, size }, \
      { tzcnt   size, index }, \
      { inc     size }, \
      { test    index, index }, \
      { jnz     .BLOOP2 }, \
      .ELOOP2

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
