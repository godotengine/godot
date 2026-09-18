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

; void vp8_temporal_filter_apply_sse2 | arg
;  (unsigned char  *frame1,           |  0
;   unsigned int    stride,           |  1
;   unsigned char  *frame2,           |  2
;   unsigned int    block_size,       |  3
;   int             strength,         |  4
;   int             filter_weight,    |  5
;   unsigned int   *accumulator,      |  6
;   unsigned short *count)            |  7
globalsym(vp8_temporal_filter_apply_sse2)
sym(vp8_temporal_filter_apply_sse2):

    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 8
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ALIGN_STACK 16, rax
    %define block_size    0
    %define strength      16
    %define filter_weight 32
    %define rounding_bit  48
    %define rbp_backup    64
    %define stack_size    80
    sub         rsp,           stack_size
    mov         [rsp + rbp_backup], rbp
    ; end prolog

        mov         rdx,            arg(3)
        mov         [rsp + block_size], rdx
        movd        xmm6,            arg(4)
        movdqa      [rsp + strength], xmm6 ; where strength is used, all 16 bytes are read

        ; calculate the rounding bit outside the loop
        ; 0x8000 >> (16 - strength)
        mov         rdx,            16
        sub         rdx,            arg(4) ; 16 - strength
        movq        xmm4,           rdx    ; can't use rdx w/ shift
        movdqa      xmm5,           [GLOBAL(_const_top_bit)]
        psrlw       xmm5,           xmm4
        movdqa      [rsp + rounding_bit], xmm5

        mov         rsi,            arg(0) ; src/frame1
        mov         rdx,            arg(2) ; predictor frame
        mov         rdi,            arg(6) ; accumulator
        mov         rax,            arg(7) ; count

        ; dup the filter weight and store for later
        movd        xmm0,           arg(5) ; filter_weight
        pshuflw     xmm0,           xmm0, 0
        punpcklwd   xmm0,           xmm0
        movdqa      [rsp + filter_weight], xmm0

        mov         rbp,            arg(1) ; stride
        pxor        xmm7,           xmm7   ; zero for extraction

        lea         rcx,            [rdx + 16*16*1]
        cmp         dword ptr [rsp + block_size], 8
        jne         .temporal_filter_apply_load_16
        lea         rcx,            [rdx + 8*8*1]

.temporal_filter_apply_load_8:
        movq        xmm0,           [rsi]  ; first row
        lea         rsi,            [rsi + rbp] ; += stride
        punpcklbw   xmm0,           xmm7   ; src[ 0- 7]
        movq        xmm1,           [rsi]  ; second row
        lea         rsi,            [rsi + rbp] ; += stride
        punpcklbw   xmm1,           xmm7   ; src[ 8-15]
        jmp         .temporal_filter_apply_load_finished

.temporal_filter_apply_load_16:
        movdqa      xmm0,           [rsi]  ; src (frame1)
        lea         rsi,            [rsi + rbp] ; += stride
        movdqa      xmm1,           xmm0
        punpcklbw   xmm0,           xmm7   ; src[ 0- 7]
        punpckhbw   xmm1,           xmm7   ; src[ 8-15]

.temporal_filter_apply_load_finished:
        movdqa      xmm2,           [rdx]  ; predictor (frame2)
        movdqa      xmm3,           xmm2
        punpcklbw   xmm2,           xmm7   ; pred[ 0- 7]
        punpckhbw   xmm3,           xmm7   ; pred[ 8-15]

        ; modifier = src_byte - pixel_value
        psubw       xmm0,           xmm2   ; src - pred[ 0- 7]
        psubw       xmm1,           xmm3   ; src - pred[ 8-15]

        ; modifier *= modifier
        pmullw      xmm0,           xmm0   ; modifer[ 0- 7]^2
        pmullw      xmm1,           xmm1   ; modifer[ 8-15]^2

        ; modifier *= 3
        pmullw      xmm0,           [GLOBAL(_const_3w)]
        pmullw      xmm1,           [GLOBAL(_const_3w)]

        ; modifer += 0x8000 >> (16 - strength)
        paddw       xmm0,           [rsp + rounding_bit]
        paddw       xmm1,           [rsp + rounding_bit]

        ; modifier >>= strength
        psrlw       xmm0,           [rsp + strength]
        psrlw       xmm1,           [rsp + strength]

        ; modifier = 16 - modifier
        ; saturation takes care of modifier > 16
        movdqa      xmm3,           [GLOBAL(_const_16w)]
        movdqa      xmm2,           [GLOBAL(_const_16w)]
        psubusw     xmm3,           xmm1
        psubusw     xmm2,           xmm0

        ; modifier *= filter_weight
        pmullw      xmm2,           [rsp + filter_weight]
        pmullw      xmm3,           [rsp + filter_weight]

        ; count
        movdqa      xmm4,           [rax]
        movdqa      xmm5,           [rax+16]
        ; += modifier
        paddw       xmm4,           xmm2
        paddw       xmm5,           xmm3
        ; write back
        movdqa      [rax],          xmm4
        movdqa      [rax+16],       xmm5
        lea         rax,            [rax + 16*2] ; count += 16*(sizeof(short))

        ; load and extract the predictor up to shorts
        pxor        xmm7,           xmm7
        movdqa      xmm0,           [rdx]
        lea         rdx,            [rdx + 16*1] ; pred += 16*(sizeof(char))
        movdqa      xmm1,           xmm0
        punpcklbw   xmm0,           xmm7   ; pred[ 0- 7]
        punpckhbw   xmm1,           xmm7   ; pred[ 8-15]

        ; modifier *= pixel_value
        pmullw      xmm0,           xmm2
        pmullw      xmm1,           xmm3

        ; expand to double words
        movdqa      xmm2,           xmm0
        punpcklwd   xmm0,           xmm7   ; [ 0- 3]
        punpckhwd   xmm2,           xmm7   ; [ 4- 7]
        movdqa      xmm3,           xmm1
        punpcklwd   xmm1,           xmm7   ; [ 8-11]
        punpckhwd   xmm3,           xmm7   ; [12-15]

        ; accumulator
        movdqa      xmm4,           [rdi]
        movdqa      xmm5,           [rdi+16]
        movdqa      xmm6,           [rdi+32]
        movdqa      xmm7,           [rdi+48]
        ; += modifier
        paddd       xmm4,           xmm0
        paddd       xmm5,           xmm2
        paddd       xmm6,           xmm1
        paddd       xmm7,           xmm3
        ; write back
        movdqa      [rdi],          xmm4
        movdqa      [rdi+16],       xmm5
        movdqa      [rdi+32],       xmm6
        movdqa      [rdi+48],       xmm7
        lea         rdi,            [rdi + 16*4] ; accumulator += 16*(sizeof(int))

        cmp         rdx,            rcx
        je          .temporal_filter_apply_epilog
        pxor        xmm7,           xmm7   ; zero for extraction
        cmp         dword ptr [rsp + block_size], 16
        je          .temporal_filter_apply_load_16
        jmp         .temporal_filter_apply_load_8

.temporal_filter_apply_epilog:
    ; begin epilog
    mov         rbp,            [rsp + rbp_backup]
    add         rsp,            stack_size
    pop         rsp
    pop         rdi
    pop         rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret

SECTION_RODATA
align 16
_const_3w:
    times 8 dw 3
align 16
_const_top_bit:
    times 8 dw 1<<15
align 16
_const_16w:
    times 8 dw 16
