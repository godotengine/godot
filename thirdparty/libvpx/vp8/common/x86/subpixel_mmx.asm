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

%define BLOCK_HEIGHT_WIDTH 4
%define vp8_filter_weight 128
%define VP8_FILTER_SHIFT  7

SECTION .text

;void vp8_filter_block1d_h6_mmx
;(
;    unsigned char   *src_ptr,
;    unsigned short  *output_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned int    pixel_step,
;    unsigned int    output_height,
;    unsigned int    output_width,
;    short           * vp8_filter
;)
globalsym(vp8_filter_block1d_h6_mmx)
sym(vp8_filter_block1d_h6_mmx):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rdx,    arg(6) ;vp8_filter

        movq        mm1,    [rdx + 16]             ; do both the negative taps first!!!
        movq        mm2,    [rdx + 32]         ;
        movq        mm6,    [rdx + 48]        ;
        movq        mm7,    [rdx + 64]        ;

        mov         rdi,    arg(1) ;output_ptr
        mov         rsi,    arg(0) ;src_ptr
        movsxd      rcx,    dword ptr arg(4) ;output_height
        movsxd      rax,    dword ptr arg(5) ;output_width      ; destination pitch?
        pxor        mm0,    mm0              ; mm0 = 00000000

.nextrow:
        movq        mm3,    [rsi-2]          ; mm3 = p-2..p5
        movq        mm4,    mm3              ; mm4 = p-2..p5
        psrlq       mm3,    8                ; mm3 = p-1..p5
        punpcklbw   mm3,    mm0              ; mm3 = p-1..p2
        pmullw      mm3,    mm1              ; mm3 *= kernel 1 modifiers.

        movq        mm5,    mm4              ; mm5 = p-2..p5
        punpckhbw   mm4,    mm0              ; mm5 = p2..p5
        pmullw      mm4,    mm7              ; mm5 *= kernel 4 modifiers
        paddsw      mm3,    mm4              ; mm3 += mm5

        movq        mm4,    mm5              ; mm4 = p-2..p5;
        psrlq       mm5,    16               ; mm5 = p0..p5;
        punpcklbw   mm5,    mm0              ; mm5 = p0..p3
        pmullw      mm5,    mm2              ; mm5 *= kernel 2 modifiers
        paddsw      mm3,    mm5              ; mm3 += mm5

        movq        mm5,    mm4              ; mm5 = p-2..p5
        psrlq       mm4,    24               ; mm4 = p1..p5
        punpcklbw   mm4,    mm0              ; mm4 = p1..p4
        pmullw      mm4,    mm6              ; mm5 *= kernel 3 modifiers
        paddsw      mm3,    mm4              ; mm3 += mm5

        ; do outer positive taps
        movd        mm4,    [rsi+3]
        punpcklbw   mm4,    mm0              ; mm5 = p3..p6
        pmullw      mm4,    [rdx+80]         ; mm5 *= kernel 0 modifiers
        paddsw      mm3,    mm4              ; mm3 += mm5

        punpcklbw   mm5,    mm0              ; mm5 = p-2..p1
        pmullw      mm5,    [rdx]            ; mm5 *= kernel 5 modifiers
        paddsw      mm3,    mm5              ; mm3 += mm5

        paddsw      mm3,    [GLOBAL(rd)]              ; mm3 += round value
        psraw       mm3,    VP8_FILTER_SHIFT     ; mm3 /= 128
        packuswb    mm3,    mm0              ; pack and unpack to saturate
        punpcklbw   mm3,    mm0              ;

        movq        [rdi],  mm3              ; store the results in the destination

%if ABI_IS_32BIT
        add         rsi,    dword ptr arg(2) ;src_pixels_per_line ; next line
        add         rdi,    rax;
%else
        movsxd      r8,     dword ptr arg(2) ;src_pixels_per_line
        add         rdi,    rax;

        add         rsi,    r8               ; next line
%endif

        dec         rcx                      ; decrement count
        jnz         .nextrow                 ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_filter_block1dc_v6_mmx
;(
;   short *src_ptr,
;   unsigned char *output_ptr,
;    int output_pitch,
;   unsigned int pixels_per_line,
;   unsigned int pixel_step,
;   unsigned int output_height,
;   unsigned int output_width,
;   short * vp8_filter
;)
globalsym(vp8_filter_block1dc_v6_mmx)
sym(vp8_filter_block1dc_v6_mmx):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 8
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        movq      mm5, [GLOBAL(rd)]
        push        rbx
        mov         rbx, arg(7) ;vp8_filter
        movq      mm1, [rbx + 16]             ; do both the negative taps first!!!
        movq      mm2, [rbx + 32]         ;
        movq      mm6, [rbx + 48]        ;
        movq      mm7, [rbx + 64]        ;

        movsxd      rdx, dword ptr arg(3) ;pixels_per_line
        mov         rdi, arg(1) ;output_ptr
        mov         rsi, arg(0) ;src_ptr
        sub         rsi, rdx
        sub         rsi, rdx
        movsxd      rcx, DWORD PTR arg(5) ;output_height
        movsxd      rax, DWORD PTR arg(2) ;output_pitch      ; destination pitch?
        pxor        mm0, mm0              ; mm0 = 00000000


.nextrow_cv:
        movq        mm3, [rsi+rdx]        ; mm3 = p0..p8  = row -1
        pmullw      mm3, mm1              ; mm3 *= kernel 1 modifiers.


        movq        mm4, [rsi + 4*rdx]      ; mm4 = p0..p3  = row 2
        pmullw      mm4, mm7              ; mm4 *= kernel 4 modifiers.
        paddsw      mm3, mm4              ; mm3 += mm4

        movq        mm4, [rsi + 2*rdx]           ; mm4 = p0..p3  = row 0
        pmullw      mm4, mm2              ; mm4 *= kernel 2 modifiers.
        paddsw      mm3, mm4              ; mm3 += mm4

        movq        mm4, [rsi]            ; mm4 = p0..p3  = row -2
        pmullw      mm4, [rbx]            ; mm4 *= kernel 0 modifiers.
        paddsw      mm3, mm4              ; mm3 += mm4


        add         rsi, rdx              ; move source forward 1 line to avoid 3 * pitch
        movq        mm4, [rsi + 2*rdx]     ; mm4 = p0..p3  = row 1
        pmullw      mm4, mm6              ; mm4 *= kernel 3 modifiers.
        paddsw      mm3, mm4              ; mm3 += mm4

        movq        mm4, [rsi + 4*rdx]    ; mm4 = p0..p3  = row 3
        pmullw      mm4, [rbx +80]        ; mm4 *= kernel 3 modifiers.
        paddsw      mm3, mm4              ; mm3 += mm4


        paddsw      mm3, mm5               ; mm3 += round value
        psraw       mm3, VP8_FILTER_SHIFT     ; mm3 /= 128
        packuswb    mm3, mm0              ; pack and saturate

        movd        [rdi],mm3             ; store the results in the destination
        ; the subsequent iterations repeat 3 out of 4 of these reads.  Since the
        ; recon block should be in cache this shouldn't cost much.  Its obviously
        ; avoidable!!!.
        lea         rdi,  [rdi+rax] ;
        dec         rcx                   ; decrement count
        jnz         .nextrow_cv           ; next row

        pop         rbx

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    UNSHADOW_ARGS
    pop         rbp
    ret


SECTION_RODATA
align 16
rd:
    times 4 dw 0x40

align 16
global HIDDEN_DATA(sym(vp8_six_tap_x86))
sym(vp8_six_tap_x86):
    times 8 dw 0
    times 8 dw 0
    times 8 dw 128
    times 8 dw 0
    times 8 dw 0
    times 8 dw 0

    times 8 dw 0
    times 8 dw -6
    times 8 dw 123
    times 8 dw 12
    times 8 dw -1
    times 8 dw 0

    times 8 dw 2
    times 8 dw -11
    times 8 dw 108
    times 8 dw 36
    times 8 dw -8
    times 8 dw 1

    times 8 dw 0
    times 8 dw -9
    times 8 dw 93
    times 8 dw 50
    times 8 dw -6
    times 8 dw 0

    times 8 dw 3
    times 8 dw -16
    times 8 dw 77
    times 8 dw 77
    times 8 dw -16
    times 8 dw 3

    times 8 dw 0
    times 8 dw -6
    times 8 dw 50
    times 8 dw 93
    times 8 dw -9
    times 8 dw 0

    times 8 dw 1
    times 8 dw -8
    times 8 dw 36
    times 8 dw 108
    times 8 dw -11
    times 8 dw 2

    times 8 dw 0
    times 8 dw -1
    times 8 dw 12
    times 8 dw 123
    times 8 dw -6
    times 8 dw 0


