;
; jchuff-sse2.asm - Huffman entropy encoding (64-bit SSE2)
;
; Copyright (C) 2009-2011, 2014-2016, 2019, 2021, 2023-2024, D. R. Commander.
; Copyright (C) 2015, Matthieu Darbois.
; Copyright (C) 2018, Matthias Räncker.
; Copyright (C) 2023, Aliaksiej Kandracienka.
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

    ALIGNZ      32
    GLOBAL_DATA(jconst_huff_encode_one_block)

EXTN(jconst_huff_encode_one_block):

jpeg_mask_bits dd 0x0000, 0x0001, 0x0003, 0x0007
               dd 0x000f, 0x001f, 0x003f, 0x007f
               dd 0x00ff, 0x01ff, 0x03ff, 0x07ff
               dd 0x0fff, 0x1fff, 0x3fff, 0x7fff

    ALIGNZ      32

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
times 1 << 15 db 16

    ALIGNZ      32

%define NBITS(x)      nbits_base + x
%define MASK_BITS(x)  NBITS((x) * 4) + (jpeg_mask_bits - EXTN(jpeg_nbits_table))

; --------------------------------------------------------------------------
    SECTION     SEG_TEXT
    BITS        64

; Shorthand used to describe SIMD operations:
; wN:  xmmN treated as eight signed 16-bit values
; wN[i]:  perform the same operation on all eight signed 16-bit values, i=0..7
; bN:  xmmN treated as 16 unsigned 8-bit values
; bN[i]:  perform the same operation on all 16 unsigned 8-bit values, i=0..15
; Contents of SIMD registers are shown in memory order.

; Fill the bit buffer to capacity with the leading bits from code, then output
; the bit buffer and put the remaining bits from code into the bit buffer.
;
; Usage:
; code - contains the bits to shift into the bit buffer (LSB-aligned)
; %1 - the label to which to jump when the macro completes
; %2 (optional) - extra instructions to execute after nbits has been set
;
; Upon completion, free_bits will be set to the number of remaining bits from
; code, and put_buffer will contain those remaining bits.  temp and code will
; be clobbered.
;
; This macro encodes any 0xFF bytes as 0xFF 0x00, as does the EMIT_BYTE()
; macro in jchuff.c.

%macro EMIT_QWORD 1-2
    add         nbitsb, free_bitsb      ; nbits += free_bits;
    neg         free_bitsb              ; free_bits = -free_bits;
    mov         tempd, code             ; temp = code;
    shl         put_buffer, nbitsb      ; put_buffer <<= nbits;
    mov         nbitsb, free_bitsb      ; nbits = free_bits;
    neg         free_bitsb              ; free_bits = -free_bits;
    shr         tempd, nbitsb           ; temp >>= nbits;
    or          tempq, put_buffer       ; temp |= put_buffer;
    movq        xmm0, tempq             ; xmm0.u64 = { temp, 0 };
    bswap       tempq                   ; temp = htonl(temp);
    mov         put_buffer, codeq       ; put_buffer = code;
    pcmpeqb     xmm0, xmm1              ; b0[i] = (b0[i] == 0xFF ? 0xFF : 0);
    %2
    pmovmskb    code, xmm0              ; code = 0;  code |= ((b0[i] >> 7) << i);
    mov         qword [buffer], tempq   ; memcpy(buffer, &temp, 8);
                                        ; (speculative; will be overwritten if
                                        ; code contains any 0xFF bytes)
    add         free_bitsb, 64          ; free_bits += 64;
    add         bufferp, 8              ; buffer += 8;
    test        code, code              ; if (code == 0)  /* No 0xFF bytes */
    jz          %1                      ;   return;
    ; Execute the equivalent of the EMIT_BYTE() macro in jchuff.c for all 8
    ; bytes in the qword.
    cmp         tempb, 0xFF             ; Set CF if temp[0] < 0xFF
    mov         byte [buffer-7], 0      ; buffer[-7] = 0;
    sbb         bufferp, 6              ; buffer -= (6 + (temp[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], temph    ; buffer[0] = temp[1];
    cmp         temph, 0xFF             ; Set CF if temp[1] < 0xFF
    mov         byte [buffer+1], 0      ; buffer[1] = 0;
    sbb         bufferp, -2             ; buffer -= (-2 + (temp[1] < 0xFF ? 1 : 0));
    shr         tempq, 16               ; temp >>= 16;
    mov         byte [buffer], tempb    ; buffer[0] = temp[0];
    cmp         tempb, 0xFF             ; Set CF if temp[0] < 0xFF
    mov         byte [buffer+1], 0      ; buffer[1] = 0;
    sbb         bufferp, -2             ; buffer -= (-2 + (temp[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], temph    ; buffer[0] = temp[1];
    cmp         temph, 0xFF             ; Set CF if temp[1] < 0xFF
    mov         byte [buffer+1], 0      ; buffer[1] = 0;
    sbb         bufferp, -2             ; buffer -= (-2 + (temp[1] < 0xFF ? 1 : 0));
    shr         tempq, 16               ; temp >>= 16;
    mov         byte [buffer], tempb    ; buffer[0] = temp[0];
    cmp         tempb, 0xFF             ; Set CF if temp[0] < 0xFF
    mov         byte [buffer+1], 0      ; buffer[1] = 0;
    sbb         bufferp, -2             ; buffer -= (-2 + (temp[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], temph    ; buffer[0] = temp[1];
    cmp         temph, 0xFF             ; Set CF if temp[1] < 0xFF
    mov         byte [buffer+1], 0      ; buffer[1] = 0;
    sbb         bufferp, -2             ; buffer -= (-2 + (temp[1] < 0xFF ? 1 : 0));
    shr         tempd, 16               ; temp >>= 16;
    mov         byte [buffer], tempb    ; buffer[0] = temp[0];
    cmp         tempb, 0xFF             ; Set CF if temp[0] < 0xFF
    mov         byte [buffer+1], 0      ; buffer[1] = 0;
    sbb         bufferp, -2             ; buffer -= (-2 + (temp[0] < 0xFF ? 1 : 0));
    mov         byte [buffer], temph    ; buffer[0] = temp[1];
    cmp         temph, 0xFF             ; Set CF if temp[1] < 0xFF
    mov         byte [buffer+1], 0      ; buffer[1] = 0;
    sbb         bufferp, -2             ; buffer -= (-2 + (temp[1] < 0xFF ? 1 : 0));
    jmp         %1                      ; return;
%endmacro

;
; Encode a single block's worth of coefficients.
;
; GLOBAL(JOCTET *)
; jsimd_huff_encode_one_block_sse2(working_state *state, JOCTET *buffer,
;                                  JCOEFPTR block, int last_dc_val,
;                                  c_derived_tbl *dctbl, c_derived_tbl *actbl)
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
; rax - buffer
; rbx - temp
; rcx - nbits
; rdx - code
; rsi - nbits_base
; rdi - t
; r8  - dctbl --> code_temp
; r9  - actbl
; r10 - state
; r11 - index
; r12 - put_buffer
; r15 - block --> free_bits

%define buffer       rax
%ifdef WIN64
%define bufferp      rax
%else
%define bufferp      raxp
%endif
%define tempq        rbx
%define tempd        ebx
%define tempb        bl
%define temph        bh
%define nbitsq       rcx
%define nbits        ecx
%define nbitsb       cl
%define codeq        rdx
%define code         edx
%define nbits_base   rsi
%define t            rdi
%define td           edi
%define dctbl        r8
%define actbl        r9
%define state        r10
%define index        r11
%define indexd       r11d
%define put_buffer   r12
%define put_bufferd  r12d
%define block        r15

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
    ENDBR64
    push        rbp
    mov         rbp, rsp

%ifdef WIN64

; rcx = working_state *state
; rdx = JOCTET *buffer
; r8 = JCOEFPTR block
; r9 = int last_dc_val
; [rbp+48] = c_derived_tbl *dctbl
; [rbp+56] = c_derived_tbl *actbl

                                                          ;X: X = code stream
    mov         buffer, rdx
    push        r15
    mov         block, r8
    movups      xmm3, XMMWORD [block + 0 * SIZEOF_WORD]   ;D: w3 = xx 01 02 03 04 05 06 07
    push        rbx
    movdqa      xmm0, xmm3                                ;A: w0 = xx 01 02 03 04 05 06 07
    push        rsi
    push        rdi
    push        r12
    movups      xmm1, XMMWORD [block + 8 * SIZEOF_WORD]   ;B: w1 = 08 09 10 11 12 13 14 15
    mov         state, rcx
    movsx       code, word [block]                        ;Z:     code = block[0];
    pxor        xmm4, xmm4                                ;A: w4[i] = 0;
    sub         code, r9d                                 ;Z:     code -= last_dc_val;
    mov         dctbl, POINTER [rbp+48]
    mov         actbl, POINTER [rbp+56]
    punpckldq   xmm0, xmm1                                ;A: w0 = xx 01 08 09 02 03 10 11
    lea         nbits_base, [rel EXTN(jpeg_nbits_table)]

%else

; rdi = working_state *state
; rsi = JOCTET *buffer
; rdx = JCOEFPTR block
; rcx = int last_dc_val
; r8 = c_derived_tbl *dctbl
; r9 = c_derived_tbl *actbl

                                                          ;X: X = code stream
    push        r15
    mov         block, rdx
    movups      xmm3, XMMWORD [block + 0 * SIZEOF_WORD]   ;D: w3 = xx 01 02 03 04 05 06 07
    push        rbx
    movdqa      xmm0, xmm3                                ;A: w0 = xx 01 02 03 04 05 06 07
    push        r12
    mov         state, rdi
    mov         buffer, rsi
    movups      xmm1, XMMWORD [block + 8 * SIZEOF_WORD]   ;B: w1 = 08 09 10 11 12 13 14 15
    movsx       codeq, word [block]                       ;Z:     code = block[0];
    lea         nbits_base, [rel EXTN(jpeg_nbits_table)]
    pxor        xmm4, xmm4                                ;A: w4[i] = 0;
    sub         codeq, rcx                                ;Z:     code -= last_dc_val;
    punpckldq   xmm0, xmm1                                ;A: w0 = xx 01 08 09 02 03 10 11

%endif

    ; Allocate stack space for t array, and realign stack.
    add         rsp, -DCTSIZE2 * SIZEOF_WORD - 8
    mov         t, rsp

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
    cmp         code, 1 << 31                             ;Z:     Set CF if code < 0x80000000,
                                                          ;Z:     i.e. if code is positive
    pinsrw      xmm2, word [block + 33 * SIZEOF_WORD], 2  ;C: w2 = 19 26 33 27 48 49 50 51
    pinsrw      xmm2, word [block + 40 * SIZEOF_WORD], 3  ;C: w2 = 19 26 33 40 48 49 50 51
    adc         code, -1                                  ;Z:     code += -1 + (code >= 0 ? 1 : 0);
    pinsrw      xmm2, word [block + 41 * SIZEOF_WORD], 5  ;C: w2 = 19 26 33 40 48 41 50 51
    pinsrw      xmm2, word [block + 34 * SIZEOF_WORD], 6  ;C: w2 = 19 26 33 40 48 41 34 51
    movsxd      codeq, code                               ;Z:     sign extend code
    pinsrw      xmm2, word [block + 27 * SIZEOF_WORD], 7  ;C: w2 = 19 26 33 40 48 41 34 27
                                                          ;        (Row 2, offset 1)
    pcmpgtw     xmm4, xmm2                                ;C: w4[i] = (w2[i] < 0 ? -1 : 0);
    paddw       xmm2, xmm4                                ;C: w2[i] += w4[i];
    movaps      XMMWORD [t + 16 * SIZEOF_WORD], xmm2      ;C: t[i+16] = w2[i];
    pxor        xmm4, xmm4                                ;C: w4[i] = 0;
    pcmpeqw     xmm2, xmm4                                ;C: w2[i] = (w2[i] == 0 ? -1 : 0);

    packsswb    xmm2, xmm3                                ;CD: b2[i] = w2[i], b2[i+8] = w3[i]
                                                          ;    w/ signed saturation

    movzx       nbitsq, byte [NBITS(codeq)]               ;Z:     nbits = JPEG_NBITS(code);
    movdqa      xmm3, xmm5                                ;H: w3 = 48 49 50 51 52 53 54 55
    pmovmskb    tempd, xmm2                               ;Z:     temp = 0;  temp |= ((b2[i] >> 7) << i);
    pmovmskb    put_bufferd, xmm0                         ;Z:     put_buffer = 0;  put_buffer |= ((b0[i] >> 7) << i);
    movups      xmm0, XMMWORD [block + 56 * SIZEOF_WORD]  ;H: w0 = 56 57 58 59 60 61 62 63
    punpckhdq   xmm3, xmm0                                ;H: w3 = 52 53 60 61 54 55 62 63
    shl         tempd, 16                                 ;Z:     temp <<= 16;
    psrldq      xmm3, 1 * SIZEOF_WORD                     ;H: w3 = 53 60 61 54 55 62 63 --
    pxor        xmm2, xmm2                                ;H: w2[i] = 0;
    or          put_bufferd, tempd                        ;Z:     put_buffer |= temp;
    pshuflw     xmm3, xmm3, 00111001b                     ;H: w3 = 60 61 54 53 55 62 63 --
    movq        xmm1, qword [block + 44 * SIZEOF_WORD]    ;G: w1 = 44 45 46 47 -- -- -- --
    unpcklps    xmm5, xmm0                                ;E: w5 = 48 49 56 57 50 51 58 59
    pxor        xmm0, xmm0                                ;H: w0[i] = 0;
    pinsrw      xmm3, word [block + 47 * SIZEOF_WORD], 3  ;H: w3 = 60 61 54 47 55 62 63 --
                                                          ;        (Row 7, offset 1)
    pcmpgtw     xmm2, xmm3                                ;H: w2[i] = (w3[i] < 0 ? -1 : 0);
    paddw       xmm3, xmm2                                ;H: w3[i] += w2[i];
    movaps      XMMWORD [t + 56 * SIZEOF_WORD], xmm3      ;H: t[i+56] = w3[i];
    movq        xmm4, qword [block + 36 * SIZEOF_WORD]    ;G: w4 = 36 37 38 39 -- -- -- --
    pcmpeqw     xmm3, xmm0                                ;H: w3[i] = (w3[i] == 0 ? -1 : 0);
    punpckldq   xmm4, xmm1                                ;G: w4 = 36 37 44 45 38 39 46 47
    mov         tempd, [dctbl + c_derived_tbl.ehufco + nbitsq * 4]
                                                          ;Z:     temp = dctbl->ehufco[nbits];
    movdqa      xmm1, xmm4                                ;F: w1 = 36 37 44 45 38 39 46 47
    psrldq      xmm4, 1 * SIZEOF_WORD                     ;G: w4 = 37 44 45 38 39 46 47 --
    shufpd      xmm1, xmm5, 10b                           ;F: w1 = 36 37 44 45 50 51 58 59
    and         code, dword [MASK_BITS(nbitsq)]           ;Z:     code &= (1 << nbits) - 1;
    pshufhw     xmm4, xmm4, 11010011b                     ;G: w4 = 37 44 45 38 -- 39 46 --
    pslldq      xmm1, 1 * SIZEOF_WORD                     ;F: w1 = -- 36 37 44 45 50 51 58
    shl         tempq, nbitsb                             ;Z:     temp <<= nbits;
    pinsrw      xmm4, word [block + 59 * SIZEOF_WORD], 0  ;G: w4 = 59 44 45 38 -- 39 46 --
    pshufd      xmm1, xmm1, 11011000b                     ;F: w1 = -- 36 45 50 37 44 51 58
    pinsrw      xmm4, word [block + 52 * SIZEOF_WORD], 1  ;G: w4 = 59 52 45 38 -- 39 46 --
    or          code, tempd                               ;Z:     code |= temp;
    movlps      xmm1, qword [block + 20 * SIZEOF_WORD]    ;F: w1 = 20 21 22 23 37 44 51 58
    pinsrw      xmm4, word [block + 31 * SIZEOF_WORD], 4  ;G: w4 = 59 52 45 38 31 39 46 --
    pshuflw     xmm1, xmm1, 01110010b                     ;F: w1 = 22 20 23 21 37 44 51 58
    pinsrw      xmm4, word [block + 53 * SIZEOF_WORD], 7  ;G: w4 = 59 52 45 38 31 39 46 53
                                                          ;        (Row 6, offset 1)
    pxor        xmm2, xmm2                                ;G: w2[i] = 0;
    pcmpgtw     xmm0, xmm4                                ;G: w0[i] = (w4[i] < 0 ? -1 : 0);
    pinsrw      xmm1, word [block + 15 * SIZEOF_WORD], 1  ;F: w1 = 22 15 23 21 37 44 51 58
    paddw       xmm4, xmm0                                ;G: w4[i] += w0[i];
    movaps      XMMWORD [t + 48 * SIZEOF_WORD], xmm4      ;G: t[48+i] = w4[i];
    pinsrw      xmm1, word [block + 30 * SIZEOF_WORD], 3  ;F: w1 = 22 15 23 30 37 44 51 58
                                                          ;        (Row 5, offset 1)
    pcmpeqw     xmm4, xmm2                                ;G: w4[i] = (w4[i] == 0 ? -1 : 0);
    pinsrw      xmm5, word [block + 42 * SIZEOF_WORD], 0  ;E: w5 = 42 49 56 57 50 51 58 59

    packsswb    xmm4, xmm3                                ;GH: b4[i] = w4[i], b4[i+8] = w3[i]
                                                          ;    w/ signed saturation

    pxor        xmm0, xmm0                                ;F: w0[i] = 0;
    pinsrw      xmm5, word [block + 43 * SIZEOF_WORD], 5  ;E: w5 = 42 49 56 57 50 43 58 59
    pcmpgtw     xmm2, xmm1                                ;F: w2[i] = (w1[i] < 0 ? -1 : 0);
    pmovmskb    tempd, xmm4                               ;Z:     temp = 0;  temp |= ((b4[i] >> 7) << i);
    pinsrw      xmm5, word [block + 36 * SIZEOF_WORD], 6  ;E: w5 = 42 49 56 57 50 43 36 59
    paddw       xmm1, xmm2                                ;F: w1[i] += w2[i];
    movaps      XMMWORD [t + 40 * SIZEOF_WORD], xmm1      ;F: t[40+i] = w1[i];
    pinsrw      xmm5, word [block + 29 * SIZEOF_WORD], 7  ;E: w5 = 42 49 56 57 50 43 36 29
                                                          ;        (Row 4, offset 1)
%undef block
%define free_bitsq  r15
%define free_bitsd  r15d
%define free_bitsb  r15b
    pcmpeqw     xmm1, xmm0                                ;F: w1[i] = (w1[i] == 0 ? -1 : 0);
    shl         tempq, 48                                 ;Z:     temp <<= 48;
    pxor        xmm2, xmm2                                ;E: w2[i] = 0;
    pcmpgtw     xmm0, xmm5                                ;E: w0[i] = (w5[i] < 0 ? -1 : 0);
    paddw       xmm5, xmm0                                ;E: w5[i] += w0[i];
    or          tempq, put_buffer                         ;Z:     temp |= put_buffer;
    movaps      XMMWORD [t + 32 * SIZEOF_WORD], xmm5      ;E: t[32+i] = w5[i];
    lea         t, [dword t - 2]                          ;Z:     t = &t[-1];
    pcmpeqw     xmm5, xmm2                                ;E: w5[i] = (w5[i] == 0 ? -1 : 0);

    packsswb    xmm5, xmm1                                ;EF: b5[i] = w5[i], b5[i+8] = w1[i]
                                                          ;    w/ signed saturation

    add         nbitsb, byte [dctbl + c_derived_tbl.ehufsi + nbitsq]
                                                          ;Z:     nbits += dctbl->ehufsi[nbits];
%undef dctbl
%define code_temp  r8d
    pmovmskb    indexd, xmm5                              ;Z:     index = 0;  index |= ((b5[i] >> 7) << i);
    mov         free_bitsd, [state+working_state.cur.free_bits]
                                                          ;Z:     free_bits = state->cur.free_bits;
    pcmpeqw     xmm1, xmm1                                ;Z:     b1[i] = 0xFF;
    shl         index, 32                                 ;Z:     index <<= 32;
    mov         put_buffer, [state+working_state.cur.put_buffer.simd]
                                                          ;Z:     put_buffer = state->cur.put_buffer.simd;
    or          index, tempq                              ;Z:     index |= temp;
    not         index                                     ;Z:     index = ~index;
    sub         free_bitsb, nbitsb                        ;Z:     if ((free_bits -= nbits) >= 0)
    jnl         .ENTRY_SKIP_EMIT_CODE                     ;Z:       goto .ENTRY_SKIP_EMIT_CODE;
    align       16
.EMIT_CODE:                                               ;Z:     .EMIT_CODE:
    EMIT_QWORD  .BLOOP_COND                               ;Z:     insert code, flush buffer, goto .BLOOP_COND

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
.BRLOOP:                                                  ; do {
    lea         code_temp, [nbitsq - 16]                  ;   code_temp = nbits - 16;
    movzx       nbits, byte [actbl + c_derived_tbl.ehufsi + 0xf0]
                                                          ;   nbits = actbl->ehufsi[0xf0];
    mov         code, [actbl + c_derived_tbl.ehufco + 0xf0 * 4]
                                                          ;   code = actbl->ehufco[0xf0];
    sub         free_bitsb, nbitsb                        ;   if ((free_bits -= nbits) <= 0)
    jle         .EMIT_BRLOOP_CODE                         ;     goto .EMIT_BRLOOP_CODE;
    shl         put_buffer, nbitsb                        ;   put_buffer <<= nbits;
    mov         nbits, code_temp                          ;   nbits = code_temp;
    or          put_buffer, codeq                         ;   put_buffer |= code;
    cmp         nbits, 16                                 ;   if (nbits <= 16)
    jle         .ERLOOP                                   ;     break;
    jmp         .BRLOOP                                   ; } while (1);

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
    times 5     nop
.ENTRY_SKIP_EMIT_CODE:                                    ; .ENTRY_SKIP_EMIT_CODE:
    shl         put_buffer, nbitsb                        ; put_buffer <<= nbits;
    or          put_buffer, codeq                         ; put_buffer |= code;
.BLOOP_COND:                                              ; .BLOOP_COND:
    test        index, index                              ; if (index != 0)
    jz          .ELOOP                                    ; {
.BLOOP:                                                   ;   do {
    xor         nbits, nbits                              ;     nbits = 0;  /* kill tzcnt input dependency */
    tzcnt       nbitsq, index                             ;     nbits = # of trailing 0 bits in index
    inc         nbits                                     ;     ++nbits;
    lea         t, [t + nbitsq * 2]                       ;     t = &t[nbits];
    shr         index, nbitsb                             ;     index >>= nbits;
.EMIT_BRLOOP_CODE_END:                                    ; .EMIT_BRLOOP_CODE_END:
    cmp         nbits, 16                                 ;     if (nbits > 16)
    jg          .BRLOOP                                   ;       goto .BRLOOP;
.ERLOOP:                                                  ; .ERLOOP:
    movsx       codeq, word [t]                           ;     code = *t;
    lea         tempd, [nbitsq * 2]                       ;     temp = nbits * 2;
    movzx       nbits, byte [NBITS(codeq)]                ;     nbits = JPEG_NBITS(code);
    lea         tempd, [nbitsq + tempq * 8]               ;     temp = temp * 8 + nbits;
    mov         code_temp, [actbl + c_derived_tbl.ehufco + (tempq - 16) * 4]
                                                          ;     code_temp = actbl->ehufco[temp-16];
    shl         code_temp, nbitsb                         ;     code_temp <<= nbits;
    and         code, dword [MASK_BITS(nbitsq)]           ;     code &= (1 << nbits) - 1;
    add         nbitsb, [actbl + c_derived_tbl.ehufsi + (tempq - 16)]
                                                          ;     free_bits -= actbl->ehufsi[temp-16];
    or          code, code_temp                           ;     code |= code_temp;
    sub         free_bitsb, nbitsb                        ;     if ((free_bits -= nbits) <= 0)
    jle         .EMIT_CODE                                ;       goto .EMIT_CODE;
    shl         put_buffer, nbitsb                        ;     put_buffer <<= nbits;
    or          put_buffer, codeq                         ;     put_buffer |= code;
    test        index, index
    jnz         .BLOOP                                    ;   } while (index != 0);
.ELOOP:                                                   ; }  /* index != 0 */
    sub         td, esp                                   ; t -= &t_[0];
    cmp         td, (DCTSIZE2 - 2) * SIZEOF_WORD          ; if (t != 62)
    je          .EFN                                      ; {
    movzx       nbits, byte [actbl + c_derived_tbl.ehufsi + 0]
                                                          ;   nbits = actbl->ehufsi[0];
    mov         code, [actbl + c_derived_tbl.ehufco + 0]  ;   code = actbl->ehufco[0];
    sub         free_bitsb, nbitsb                        ;   if ((free_bits -= nbits) <= 0)
    jg          .EFN_SKIP_EMIT_CODE                       ;   {
    EMIT_QWORD  .EFN                                      ;     insert code, flush buffer
    align       16
.EFN_SKIP_EMIT_CODE:                                      ;   } else {
    shl         put_buffer, nbitsb                        ;     put_buffer <<= nbits;
    or          put_buffer, codeq                         ;     put_buffer |= code;
.EFN:                                                     ; } }
    mov         [state + working_state.cur.put_buffer.simd], put_buffer
                                                          ; state->cur.put_buffer.simd = put_buffer;
    mov         byte [state + working_state.cur.free_bits], free_bitsb
                                                          ; state->cur.free_bits = free_bits;
    sub         rsp, -DCTSIZE2 * SIZEOF_WORD - 8
    pop         r12
%ifdef WIN64
    pop         rdi
    pop         rsi
    pop         rbx
%else
    pop         rbx
%endif
    pop         r15
    pop         rbp
    ret

; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    align       16
.EMIT_BRLOOP_CODE:
    EMIT_QWORD  .EMIT_BRLOOP_CODE_END, { mov nbits, code_temp }
                                                          ; insert code, flush buffer,
                                                          ; nbits = code_temp, goto .EMIT_BRLOOP_CODE_END

; For some reason, the OS X linker does not honor the request to align the
; segment unless we do this.
    align       32
