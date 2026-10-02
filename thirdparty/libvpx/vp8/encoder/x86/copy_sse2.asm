;
;  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;


%include "vpx_ports/x86_abi_support.asm"

SECTION .text

;void vp8_copy32xn_sse2(
;    unsigned char *src_ptr,
;    int  src_stride,
;    unsigned char *dst_ptr,
;    int  dst_stride,
;    int height);
globalsym(vp8_copy32xn_sse2)
sym(vp8_copy32xn_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    SAVE_XMM 7
    push        rsi
    push        rdi
    ; end prolog

        mov             rsi,        arg(0) ;src_ptr
        mov             rdi,        arg(2) ;dst_ptr

        movsxd          rax,        dword ptr arg(1) ;src_stride
        movsxd          rdx,        dword ptr arg(3) ;dst_stride
        movsxd          rcx,        dword ptr arg(4) ;height

.block_copy_sse2_loopx4:
        movdqu          xmm0,       XMMWORD PTR [rsi]
        movdqu          xmm1,       XMMWORD PTR [rsi + 16]
        movdqu          xmm2,       XMMWORD PTR [rsi + rax]
        movdqu          xmm3,       XMMWORD PTR [rsi + rax + 16]

        lea             rsi,        [rsi+rax*2]

        movdqu          xmm4,       XMMWORD PTR [rsi]
        movdqu          xmm5,       XMMWORD PTR [rsi + 16]
        movdqu          xmm6,       XMMWORD PTR [rsi + rax]
        movdqu          xmm7,       XMMWORD PTR [rsi + rax + 16]

        lea             rsi,    [rsi+rax*2]

        movdqa          XMMWORD PTR [rdi], xmm0
        movdqa          XMMWORD PTR [rdi + 16], xmm1
        movdqa          XMMWORD PTR [rdi + rdx], xmm2
        movdqa          XMMWORD PTR [rdi + rdx + 16], xmm3

        lea             rdi,    [rdi+rdx*2]

        movdqa          XMMWORD PTR [rdi], xmm4
        movdqa          XMMWORD PTR [rdi + 16], xmm5
        movdqa          XMMWORD PTR [rdi + rdx], xmm6
        movdqa          XMMWORD PTR [rdi + rdx + 16], xmm7

        lea             rdi,    [rdi+rdx*2]

        sub             rcx,     4
        cmp             rcx,     4
        jge             .block_copy_sse2_loopx4

        cmp             rcx, 0
        je              .copy_is_done

.block_copy_sse2_loop:
        movdqu          xmm0,       XMMWORD PTR [rsi]
        movdqu          xmm1,       XMMWORD PTR [rsi + 16]
        lea             rsi,    [rsi+rax]

        movdqa          XMMWORD PTR [rdi], xmm0
        movdqa          XMMWORD PTR [rdi + 16], xmm1
        lea             rdi,    [rdi+rdx]

        sub             rcx,     1
        jne             .block_copy_sse2_loop

.copy_is_done:
    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
