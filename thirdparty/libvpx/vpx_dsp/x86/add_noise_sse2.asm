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

;void vpx_plane_add_noise_sse2(uint8_t *start, const int8_t *noise,
;                              int blackclamp, int whiteclamp,
;                              int width, int height, int pitch)
globalsym(vpx_plane_add_noise_sse2)
sym(vpx_plane_add_noise_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    GET_GOT     rbx
    push        rsi
    push        rdi

    mov         rdx, 0x01010101
    mov         rax, arg(2)
    mul         rdx
    movq        xmm3, rax
    pshufd      xmm3, xmm3, 0  ; xmm3 is 16 copies of char in blackclamp

    mov         rdx, 0x01010101
    mov         rax, arg(3)
    mul         rdx
    movq        xmm4, rax
    pshufd      xmm4, xmm4, 0  ; xmm4 is 16 copies of char in whiteclamp

    movdqu      xmm5, xmm3     ; both clamp = black clamp + white clamp
    paddusb     xmm5, xmm4

.addnoise_loop:
    call sym(LIBVPX_RAND) WRT_PLT
    mov     rcx, arg(1) ;noise
    and     rax, 0xff
    add     rcx, rax

    mov     rdi, rcx
    movsxd  rcx, dword arg(4) ;[Width]
    mov     rsi, arg(0) ;Pos
    xor     rax, rax

.addnoise_nextset:
      movdqu      xmm1,[rsi+rax]         ; get the source

      psubusb     xmm1, xmm3 ; subtract black clamp
      paddusb     xmm1, xmm5 ; add both clamp
      psubusb     xmm1, xmm4 ; subtract whiteclamp

      movdqu      xmm2,[rdi+rax]         ; get the noise for this line
      paddb       xmm1,xmm2              ; add it in
      movdqu      [rsi+rax],xmm1         ; store the result

      add         rax,16                 ; move to the next line

      cmp         rax, rcx
      jl          .addnoise_nextset

    movsxd  rax, dword arg(6) ; Pitch
    add     arg(0), rax ; Start += Pitch
    sub     dword arg(5), 1   ; Height -= 1
    jg      .addnoise_loop

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
rd42:
    times 8 dw 0x04
four8s:
    times 4 dd 8
