;
;  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;


%include "vpx_ports/x86_abi_support.asm"

SECTION .text

;unsigned int vpx_highbd_calc16x16var_sse2
;(
;    unsigned char   *  src_ptr,
;    int             src_stride,
;    unsigned char   *  ref_ptr,
;    int             ref_stride,
;    unsigned int    *  SSE,
;    int             *  Sum
;)
globalsym(vpx_highbd_calc16x16var_sse2)
sym(vpx_highbd_calc16x16var_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push rbx
    push rsi
    push rdi
    ; end prolog

        mov         rsi,            arg(0) ;[src_ptr]
        mov         rdi,            arg(2) ;[ref_ptr]

        movsxd      rax,            DWORD PTR arg(1) ;[src_stride]
        movsxd      rdx,            DWORD PTR arg(3) ;[ref_stride]
        add         rax,            rax ; source stride in bytes
        add         rdx,            rdx ; recon stride in bytes

        ; Prefetch data
        prefetcht0      [rsi]
        prefetcht0      [rsi+16]
        prefetcht0      [rsi+rax]
        prefetcht0      [rsi+rax+16]
        lea             rbx,    [rsi+rax*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+16]
        prefetcht0      [rbx+rax]
        prefetcht0      [rbx+rax+16]

        prefetcht0      [rdi]
        prefetcht0      [rdi+16]
        prefetcht0      [rdi+rdx]
        prefetcht0      [rdi+rdx+16]
        lea             rbx,    [rdi+rdx*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+16]
        prefetcht0      [rbx+rdx]
        prefetcht0      [rbx+rdx+16]

        pxor        xmm0,           xmm0     ; clear xmm0 for unpack
        pxor        xmm7,           xmm7     ; clear xmm7 for accumulating diffs

        pxor        xmm6,           xmm6     ; clear xmm6 for accumulating sse
        mov         rcx,            16

.var16loop:
        movdqu      xmm1,           XMMWORD PTR [rsi]
        movdqu      xmm2,           XMMWORD PTR [rdi]

        lea             rbx,    [rsi+rax*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+16]
        prefetcht0      [rbx+rax]
        prefetcht0      [rbx+rax+16]
        lea             rbx,    [rdi+rdx*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+16]
        prefetcht0      [rbx+rdx]
        prefetcht0      [rbx+rdx+16]

        pxor        xmm5,           xmm5

        psubw       xmm1,           xmm2
        movdqu      xmm3,           XMMWORD PTR [rsi+16]
        paddw       xmm5,           xmm1
        pmaddwd     xmm1,           xmm1
        movdqu      xmm2,           XMMWORD PTR [rdi+16]
        paddd       xmm6,           xmm1

        psubw       xmm3,           xmm2
        movdqu      xmm1,           XMMWORD PTR [rsi+rax]
        paddw       xmm5,           xmm3
        pmaddwd     xmm3,           xmm3
        movdqu      xmm2,           XMMWORD PTR [rdi+rdx]
        paddd       xmm6,           xmm3

        psubw       xmm1,           xmm2
        movdqu      xmm3,           XMMWORD PTR [rsi+rax+16]
        paddw       xmm5,           xmm1
        pmaddwd     xmm1,           xmm1
        movdqu      xmm2,           XMMWORD PTR [rdi+rdx+16]
        paddd       xmm6,           xmm1

        psubw       xmm3,           xmm2
        paddw       xmm5,           xmm3
        pmaddwd     xmm3,           xmm3
        paddd       xmm6,           xmm3

        movdqa      xmm1,           xmm5
        movdqa      xmm2,           xmm5
        pcmpgtw     xmm1,           xmm0
        pcmpeqw     xmm2,           xmm0
        por         xmm1,           xmm2
        pcmpeqw     xmm1,           xmm0
        movdqa      xmm2,           xmm5
        punpcklwd   xmm5,           xmm1
        punpckhwd   xmm2,           xmm1
        paddd       xmm7,           xmm5
        paddd       xmm7,           xmm2

        lea         rsi,            [rsi + 2*rax]
        lea         rdi,            [rdi + 2*rdx]
        sub         rcx,            2
        jnz         .var16loop

        movdqa      xmm4,           xmm6
        punpckldq   xmm6,           xmm0

        punpckhdq   xmm4,           xmm0
        movdqa      xmm5,           xmm7

        paddd       xmm6,           xmm4
        punpckldq   xmm7,           xmm0

        punpckhdq   xmm5,           xmm0
        paddd       xmm7,           xmm5

        movdqa      xmm4,           xmm6
        movdqa      xmm5,           xmm7

        psrldq      xmm4,           8
        psrldq      xmm5,           8

        paddd       xmm6,           xmm4
        paddd       xmm7,           xmm5

        mov         rdi,            arg(4)   ; [SSE]
        mov         rax,            arg(5)   ; [Sum]

        movd DWORD PTR [rdi],       xmm6
        movd DWORD PTR [rax],       xmm7


    ; begin epilog
    pop rdi
    pop rsi
    pop rbx
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;unsigned int vpx_highbd_calc8x8var_sse2
;(
;    unsigned char   *  src_ptr,
;    int             src_stride,
;    unsigned char   *  ref_ptr,
;    int             ref_stride,
;    unsigned int    *  SSE,
;    int             *  Sum
;)
globalsym(vpx_highbd_calc8x8var_sse2)
sym(vpx_highbd_calc8x8var_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    push rbx
    push rsi
    push rdi
    ; end prolog

        mov         rsi,            arg(0) ;[src_ptr]
        mov         rdi,            arg(2) ;[ref_ptr]

        movsxd      rax,            DWORD PTR arg(1) ;[src_stride]
        movsxd      rdx,            DWORD PTR arg(3) ;[ref_stride]
        add         rax,            rax ; source stride in bytes
        add         rdx,            rdx ; recon stride in bytes

        ; Prefetch data
        prefetcht0      [rsi]
        prefetcht0      [rsi+rax]
        lea             rbx,    [rsi+rax*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+rax]

        prefetcht0      [rdi]
        prefetcht0      [rdi+rdx]
        lea             rbx,    [rdi+rdx*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+rdx]

        pxor        xmm0,           xmm0     ; clear xmm0 for unpack
        pxor        xmm7,           xmm7     ; clear xmm7 for accumulating diffs

        pxor        xmm6,           xmm6     ; clear xmm6 for accumulating sse
        mov         rcx,            8

.var8loop:
        movdqu      xmm1,           XMMWORD PTR [rsi]
        movdqu      xmm2,           XMMWORD PTR [rdi]

        lea             rbx,    [rsi+rax*4]
        prefetcht0      [rbx]
        prefetcht0      [rbx+rax]
        lea             rbx,    [rbx+rax*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+rax]
        lea             rbx,    [rdi+rdx*4]
        prefetcht0      [rbx]
        prefetcht0      [rbx+rdx]
        lea             rbx,    [rbx+rdx*2]
        prefetcht0      [rbx]
        prefetcht0      [rbx+rdx]

        pxor        xmm5,           xmm5

        psubw       xmm1,           xmm2
        movdqu      xmm3,           XMMWORD PTR [rsi+rax]
        paddw       xmm5,           xmm1
        pmaddwd     xmm1,           xmm1
        movdqu      xmm2,           XMMWORD PTR [rdi+rdx]
        paddd       xmm6,           xmm1

        lea         rsi,            [rsi + 2*rax]
        lea         rdi,            [rdi + 2*rdx]

        psubw       xmm3,           xmm2
        movdqu      xmm1,           XMMWORD PTR [rsi]
        paddw       xmm5,           xmm3
        pmaddwd     xmm3,           xmm3
        movdqu      xmm2,           XMMWORD PTR [rdi]
        paddd       xmm6,           xmm3

        psubw       xmm1,           xmm2
        movdqu      xmm3,           XMMWORD PTR [rsi+rax]
        paddw       xmm5,           xmm1
        pmaddwd     xmm1,           xmm1
        movdqu      xmm2,           XMMWORD PTR [rdi+rdx]
        paddd       xmm6,           xmm1

        psubw       xmm3,           xmm2
        paddw       xmm5,           xmm3
        pmaddwd     xmm3,           xmm3
        paddd       xmm6,           xmm3

        movdqa      xmm1,           xmm5
        movdqa      xmm2,           xmm5
        pcmpgtw     xmm1,           xmm0
        pcmpeqw     xmm2,           xmm0
        por         xmm1,           xmm2
        pcmpeqw     xmm1,           xmm0
        movdqa      xmm2,           xmm5
        punpcklwd   xmm5,           xmm1
        punpckhwd   xmm2,           xmm1
        paddd       xmm7,           xmm5
        paddd       xmm7,           xmm2

        lea         rsi,            [rsi + 2*rax]
        lea         rdi,            [rdi + 2*rdx]
        sub         rcx,            4
        jnz         .var8loop

        movdqa      xmm4,           xmm6
        punpckldq   xmm6,           xmm0

        punpckhdq   xmm4,           xmm0
        movdqa      xmm5,           xmm7

        paddd       xmm6,           xmm4
        punpckldq   xmm7,           xmm0

        punpckhdq   xmm5,           xmm0
        paddd       xmm7,           xmm5

        movdqa      xmm4,           xmm6
        movdqa      xmm5,           xmm7

        psrldq      xmm4,           8
        psrldq      xmm5,           8

        paddd       xmm6,           xmm4
        paddd       xmm7,           xmm5

        mov         rdi,            arg(4)   ; [SSE]
        mov         rax,            arg(5)   ; [Sum]

        movd DWORD PTR [rdi],       xmm6
        movd DWORD PTR [rax],       xmm7

    ; begin epilog
    pop rdi
    pop rsi
    pop rbx
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret
