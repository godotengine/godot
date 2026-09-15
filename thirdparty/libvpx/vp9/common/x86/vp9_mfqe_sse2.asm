;
;  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

;  This file is a duplicate of mfqe_sse2.asm in VP8.
;  TODO(jackychen): Find a way to fix the duplicate.
%include "vpx_ports/x86_abi_support.asm"

SECTION .text

;void vp9_filter_by_weight16x16_sse2
;(
;    unsigned char *src,
;    int            src_stride,
;    unsigned char *dst,
;    int            dst_stride,
;    int            src_weight
;)
globalsym(vp9_filter_by_weight16x16_sse2)
sym(vp9_filter_by_weight16x16_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    SAVE_XMM 6
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movd        xmm0, arg(4)                ; src_weight
    pshuflw     xmm0, xmm0, 0x0             ; replicate to all low words
    punpcklqdq  xmm0, xmm0                  ; replicate to all hi words

    movdqa      xmm1, [GLOBAL(tMFQE)]
    psubw       xmm1, xmm0                  ; dst_weight

    mov         rax, arg(0)                 ; src
    mov         rsi, arg(1)                 ; src_stride
    mov         rdx, arg(2)                 ; dst
    mov         rdi, arg(3)                 ; dst_stride

    mov         rcx, 16                     ; loop count
    pxor        xmm6, xmm6

.combine:
    movdqa      xmm2, [rax]
    movdqa      xmm4, [rdx]
    add         rax, rsi

    ; src * src_weight
    movdqa      xmm3, xmm2
    punpcklbw   xmm2, xmm6
    punpckhbw   xmm3, xmm6
    pmullw      xmm2, xmm0
    pmullw      xmm3, xmm0

    ; dst * dst_weight
    movdqa      xmm5, xmm4
    punpcklbw   xmm4, xmm6
    punpckhbw   xmm5, xmm6
    pmullw      xmm4, xmm1
    pmullw      xmm5, xmm1

    ; sum, round and shift
    paddw       xmm2, xmm4
    paddw       xmm3, xmm5
    paddw       xmm2, [GLOBAL(tMFQE_round)]
    paddw       xmm3, [GLOBAL(tMFQE_round)]
    psrlw       xmm2, 4
    psrlw       xmm3, 4

    packuswb    xmm2, xmm3
    movdqa      [rdx], xmm2
    add         rdx, rdi

    dec         rcx
    jnz         .combine

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp

    ret

;void vp9_filter_by_weight8x8_sse2
;(
;    unsigned char *src,
;    int            src_stride,
;    unsigned char *dst,
;    int            dst_stride,
;    int            src_weight
;)
globalsym(vp9_filter_by_weight8x8_sse2)
sym(vp9_filter_by_weight8x8_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    movd        xmm0, arg(4)                ; src_weight
    pshuflw     xmm0, xmm0, 0x0             ; replicate to all low words
    punpcklqdq  xmm0, xmm0                  ; replicate to all hi words

    movdqa      xmm1, [GLOBAL(tMFQE)]
    psubw       xmm1, xmm0                  ; dst_weight

    mov         rax, arg(0)                 ; src
    mov         rsi, arg(1)                 ; src_stride
    mov         rdx, arg(2)                 ; dst
    mov         rdi, arg(3)                 ; dst_stride

    mov         rcx, 8                      ; loop count
    pxor        xmm4, xmm4

.combine:
    movq        xmm2, [rax]
    movq        xmm3, [rdx]
    add         rax, rsi

    ; src * src_weight
    punpcklbw   xmm2, xmm4
    pmullw      xmm2, xmm0

    ; dst * dst_weight
    punpcklbw   xmm3, xmm4
    pmullw      xmm3, xmm1

    ; sum, round and shift
    paddw       xmm2, xmm3
    paddw       xmm2, [GLOBAL(tMFQE_round)]
    psrlw       xmm2, 4

    packuswb    xmm2, xmm4
    movq        [rdx], xmm2
    add         rdx, rdi

    dec         rcx
    jnz         .combine

    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp

    ret

;void vp9_variance_and_sad_16x16_sse2 | arg
;(
;    unsigned char *src1,          0
;    int            stride1,       1
;    unsigned char *src2,          2
;    int            stride2,       3
;    unsigned int  *variance,      4
;    unsigned int  *sad,           5
;)
globalsym(vp9_variance_and_sad_16x16_sse2)
sym(vp9_variance_and_sad_16x16_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

    mov         rax,        arg(0)          ; src1
    mov         rcx,        arg(1)          ; stride1
    mov         rdx,        arg(2)          ; src2
    mov         rdi,        arg(3)          ; stride2

    mov         rsi,        16              ; block height

    ; Prep accumulator registers
    pxor        xmm3, xmm3                  ; SAD
    pxor        xmm4, xmm4                  ; sum of src2
    pxor        xmm5, xmm5                  ; sum of src2^2

    ; Because we're working with the actual output frames
    ; we can't depend on any kind of data alignment.
.accumulate:
    movdqa      xmm0, [rax]                 ; src1
    movdqa      xmm1, [rdx]                 ; src2
    add         rax, rcx                    ; src1 + stride1
    add         rdx, rdi                    ; src2 + stride2

    ; SAD(src1, src2)
    psadbw      xmm0, xmm1
    paddusw     xmm3, xmm0

    ; SUM(src2)
    pxor        xmm2, xmm2
    psadbw      xmm2, xmm1                  ; sum src2 by misusing SAD against 0
    paddusw     xmm4, xmm2

    ; pmaddubsw would be ideal if it took two unsigned values. instead,
    ; it expects a signed and an unsigned value. so instead we zero extend
    ; and operate on words.
    pxor        xmm2, xmm2
    movdqa      xmm0, xmm1
    punpcklbw   xmm0, xmm2
    punpckhbw   xmm1, xmm2
    pmaddwd     xmm0, xmm0
    pmaddwd     xmm1, xmm1
    paddd       xmm5, xmm0
    paddd       xmm5, xmm1

    sub         rsi,        1
    jnz         .accumulate

    ; phaddd only operates on adjacent double words.
    ; Finalize SAD and store
    movdqa      xmm0, xmm3
    psrldq      xmm0, 8
    paddusw     xmm0, xmm3
    paddd       xmm0, [GLOBAL(t128)]
    psrld       xmm0, 8

    mov         rax,  arg(5)
    movd        [rax], xmm0

    ; Accumulate sum of src2
    movdqa      xmm0, xmm4
    psrldq      xmm0, 8
    paddusw     xmm0, xmm4
    ; Square src2. Ignore high value
    pmuludq     xmm0, xmm0
    psrld       xmm0, 8

    ; phaddw could be used to sum adjacent values but we want
    ; all the values summed. promote to doubles, accumulate,
    ; shift and sum
    pxor        xmm2, xmm2
    movdqa      xmm1, xmm5
    punpckldq   xmm1, xmm2
    punpckhdq   xmm5, xmm2
    paddd       xmm1, xmm5
    movdqa      xmm2, xmm1
    psrldq      xmm1, 8
    paddd       xmm1, xmm2

    psubd       xmm1, xmm0

    ; (variance + 128) >> 8
    paddd       xmm1, [GLOBAL(t128)]
    psrld       xmm1, 8
    mov         rax,  arg(4)

    movd        [rax], xmm1


    ; begin epilog
    pop         rdi
    pop         rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
t128:
%ifndef __NASM_VER__
    ddq 128
%elif CONFIG_BIG_ENDIAN
    dq  0, 128
%else
    dq  128, 0
%endif
align 16
tMFQE: ; 1 << MFQE_PRECISION
    times 8 dw 0x10
align 16
tMFQE_round: ; 1 << (MFQE_PRECISION - 1)
    times 8 dw 0x08
