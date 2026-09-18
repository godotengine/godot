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

%macro STACK_FRAME_CREATE_X3 0
%if ABI_IS_32BIT
  %define     src_ptr       rsi
  %define     src_stride    rax
  %define     ref_ptr       rdi
  %define     ref_stride    rdx
  %define     end_ptr       rcx
  %define     ret_var       rbx
  %define     result_ptr    arg(4)
  %define     max_sad       arg(4)
  %define     height        dword ptr arg(4)
    push        rbp
    mov         rbp,        rsp
    push        rsi
    push        rdi
    push        rbx

    mov         rsi,        arg(0)              ; src_ptr
    mov         rdi,        arg(2)              ; ref_ptr

    movsxd      rax,        dword ptr arg(1)    ; src_stride
    movsxd      rdx,        dword ptr arg(3)    ; ref_stride
%else
  %if LIBVPX_YASM_WIN64
    SAVE_XMM 7, u
    %define     src_ptr     rcx
    %define     src_stride  rdx
    %define     ref_ptr     r8
    %define     ref_stride  r9
    %define     end_ptr     r10
    %define     ret_var     r11
    %define     result_ptr  [rsp+xmm_stack_space+8+4*8]
    %define     max_sad     [rsp+xmm_stack_space+8+4*8]
    %define     height      dword ptr [rsp+xmm_stack_space+8+4*8]
  %else
    %define     src_ptr     rdi
    %define     src_stride  rsi
    %define     ref_ptr     rdx
    %define     ref_stride  rcx
    %define     end_ptr     r9
    %define     ret_var     r10
    %define     result_ptr  r8
    %define     max_sad     r8
    %define     height      r8
  %endif
%endif

%endmacro

%macro STACK_FRAME_DESTROY_X3 0
  %define     src_ptr
  %define     src_stride
  %define     ref_ptr
  %define     ref_stride
  %define     end_ptr
  %define     ret_var
  %define     result_ptr
  %define     max_sad
  %define     height

%if ABI_IS_32BIT
    pop         rbx
    pop         rdi
    pop         rsi
    pop         rbp
%else
  %if LIBVPX_YASM_WIN64
    RESTORE_XMM
  %endif
%endif
    ret
%endmacro

SECTION .text

;void vp8_copy32xn_sse3(
;    unsigned char *src_ptr,
;    int  src_stride,
;    unsigned char *dst_ptr,
;    int  dst_stride,
;    int height);
globalsym(vp8_copy32xn_sse3)
sym(vp8_copy32xn_sse3):

    STACK_FRAME_CREATE_X3

.block_copy_sse3_loopx4:
        lea             end_ptr,    [src_ptr+src_stride*2]

        movdqu          xmm0,       XMMWORD PTR [src_ptr]
        movdqu          xmm1,       XMMWORD PTR [src_ptr + 16]
        movdqu          xmm2,       XMMWORD PTR [src_ptr + src_stride]
        movdqu          xmm3,       XMMWORD PTR [src_ptr + src_stride + 16]
        movdqu          xmm4,       XMMWORD PTR [end_ptr]
        movdqu          xmm5,       XMMWORD PTR [end_ptr + 16]
        movdqu          xmm6,       XMMWORD PTR [end_ptr + src_stride]
        movdqu          xmm7,       XMMWORD PTR [end_ptr + src_stride + 16]

        lea             src_ptr,    [src_ptr+src_stride*4]

        lea             end_ptr,    [ref_ptr+ref_stride*2]

        movdqa          XMMWORD PTR [ref_ptr], xmm0
        movdqa          XMMWORD PTR [ref_ptr + 16], xmm1
        movdqa          XMMWORD PTR [ref_ptr + ref_stride], xmm2
        movdqa          XMMWORD PTR [ref_ptr + ref_stride + 16], xmm3
        movdqa          XMMWORD PTR [end_ptr], xmm4
        movdqa          XMMWORD PTR [end_ptr + 16], xmm5
        movdqa          XMMWORD PTR [end_ptr + ref_stride], xmm6
        movdqa          XMMWORD PTR [end_ptr + ref_stride + 16], xmm7

        lea             ref_ptr,    [ref_ptr+ref_stride*4]

        sub             height,     4
        cmp             height,     4
        jge             .block_copy_sse3_loopx4

        ;Check to see if there is more rows need to be copied.
        cmp             height, 0
        je              .copy_is_done

.block_copy_sse3_loop:
        movdqu          xmm0,       XMMWORD PTR [src_ptr]
        movdqu          xmm1,       XMMWORD PTR [src_ptr + 16]
        lea             src_ptr,    [src_ptr+src_stride]

        movdqa          XMMWORD PTR [ref_ptr], xmm0
        movdqa          XMMWORD PTR [ref_ptr + 16], xmm1
        lea             ref_ptr,    [ref_ptr+ref_stride]

        sub             height,     1
        jne             .block_copy_sse3_loop

.copy_is_done:
    STACK_FRAME_DESTROY_X3
