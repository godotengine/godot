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
%define VP8_FILTER_WEIGHT 128
%define VP8_FILTER_SHIFT  7

SECTION .text

;/************************************************************************************
; Notes: filter_block1d_h6 applies a 6 tap filter horizontally to the input pixels. The
; input pixel array has output_height rows. This routine assumes that output_height is an
; even number. This function handles 8 pixels in horizontal direction, calculating ONE
; rows each iteration to take advantage of the 128 bits operations.
;*************************************************************************************/
;void vp8_filter_block1d8_h6_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned short *output_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned int    pixel_step,
;    unsigned int    output_height,
;    unsigned int    output_width,
;    short           *vp8_filter
;)
globalsym(vp8_filter_block1d8_h6_sse2)
sym(vp8_filter_block1d8_h6_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rdx,        arg(6) ;vp8_filter
        mov         rsi,        arg(0) ;src_ptr

        mov         rdi,        arg(1) ;output_ptr

        movsxd      rcx,        dword ptr arg(4) ;output_height
        movsxd      rax,        dword ptr arg(2) ;src_pixels_per_line            ; Pitch for Source
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(5) ;output_width
%endif
        pxor        xmm0,       xmm0                        ; clear xmm0 for unpack

.filter_block1d8_h6_rowloop:
        movq        xmm3,       MMWORD PTR [rsi - 2]
        movq        xmm1,       MMWORD PTR [rsi + 6]

        prefetcht2  [rsi+rax-2]

        pslldq      xmm1,       8
        por         xmm1,       xmm3

        movdqa      xmm4,       xmm1
        movdqa      xmm5,       xmm1

        movdqa      xmm6,       xmm1
        movdqa      xmm7,       xmm1

        punpcklbw   xmm3,       xmm0                        ; xx05 xx04 xx03 xx02 xx01 xx01 xx-1 xx-2
        psrldq      xmm4,       1                           ; xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00 -1

        pmullw      xmm3,       XMMWORD PTR [rdx]           ; x[-2] * H[-2]; Tap 1
        punpcklbw   xmm4,       xmm0                        ; xx06 xx05 xx04 xx03 xx02 xx01 xx00 xx-1

        psrldq      xmm5,       2                           ; xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00
        pmullw      xmm4,       XMMWORD PTR [rdx+16]        ; x[-1] * H[-1]; Tap 2


        punpcklbw   xmm5,       xmm0                        ; xx07 xx06 xx05 xx04 xx03 xx02 xx01 xx00
        psrldq      xmm6,       3                           ; xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01

        pmullw      xmm5,       [rdx+32]                    ; x[ 0] * H[ 0]; Tap 3

        punpcklbw   xmm6,       xmm0                        ; xx08 xx07 xx06 xx05 xx04 xx03 xx02 xx01
        psrldq      xmm7,       4                           ; xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02

        pmullw      xmm6,       [rdx+48]                    ; x[ 1] * h[ 1] ; Tap 4

        punpcklbw   xmm7,       xmm0                        ; xx09 xx08 xx07 xx06 xx05 xx04 xx03 xx02
        psrldq      xmm1,       5                           ; xx xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03


        pmullw      xmm7,       [rdx+64]                    ; x[ 2] * h[ 2] ; Tap 5

        punpcklbw   xmm1,       xmm0                        ; xx0a xx09 xx08 xx07 xx06 xx05 xx04 xx03
        pmullw      xmm1,       [rdx+80]                    ; x[ 3] * h[ 3] ; Tap 6


        paddsw      xmm4,       xmm7
        paddsw      xmm4,       xmm5

        paddsw      xmm4,       xmm3
        paddsw      xmm4,       xmm6

        paddsw      xmm4,       xmm1
        paddsw      xmm4,       [GLOBAL(rd)]

        psraw       xmm4,       7

        packuswb    xmm4,       xmm0
        punpcklbw   xmm4,       xmm0

        movdqa      XMMWORD Ptr [rdi],         xmm4
        lea         rsi,        [rsi + rax]

%if ABI_IS_32BIT
        add         rdi,        DWORD Ptr arg(5) ;[output_width]
%else
        add         rdi,        r8
%endif
        dec         rcx

        jnz         .filter_block1d8_h6_rowloop                ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_filter_block1d16_h6_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned short *output_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned int    pixel_step,
;    unsigned int    output_height,
;    unsigned int    output_width,
;    short           *vp8_filter
;)
;/************************************************************************************
; Notes: filter_block1d_h6 applies a 6 tap filter horizontally to the input pixels. The
; input pixel array has output_height rows. This routine assumes that output_height is an
; even number. This function handles 8 pixels in horizontal direction, calculating ONE
; rows each iteration to take advantage of the 128 bits operations.
;*************************************************************************************/
globalsym(vp8_filter_block1d16_h6_sse2)
sym(vp8_filter_block1d16_h6_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 7
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rdx,        arg(6) ;vp8_filter
        mov         rsi,        arg(0) ;src_ptr

        mov         rdi,        arg(1) ;output_ptr

        movsxd      rcx,        dword ptr arg(4) ;output_height
        movsxd      rax,        dword ptr arg(2) ;src_pixels_per_line            ; Pitch for Source
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(5) ;output_width
%endif

        pxor        xmm0,       xmm0                        ; clear xmm0 for unpack

.filter_block1d16_h6_sse2_rowloop:
        movq        xmm3,       MMWORD PTR [rsi - 2]
        movq        xmm1,       MMWORD PTR [rsi + 6]

        ; Load from 11 to avoid reading out of bounds.
        movq        xmm2,       MMWORD PTR [rsi +11]
        ; The lower bits are not cleared before 'or'ing with xmm1,
        ; but that is OK because the values in the overlapping positions
        ; are already equal to the ones in xmm1.
        pslldq      xmm2,       5

        por         xmm2,       xmm1
        prefetcht2  [rsi+rax-2]

        pslldq      xmm1,       8
        por         xmm1,       xmm3

        movdqa      xmm4,       xmm1
        movdqa      xmm5,       xmm1

        movdqa      xmm6,       xmm1
        movdqa      xmm7,       xmm1

        punpcklbw   xmm3,       xmm0                        ; xx05 xx04 xx03 xx02 xx01 xx01 xx-1 xx-2
        psrldq      xmm4,       1                           ; xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00 -1

        pmullw      xmm3,       XMMWORD PTR [rdx]           ; x[-2] * H[-2]; Tap 1
        punpcklbw   xmm4,       xmm0                        ; xx06 xx05 xx04 xx03 xx02 xx01 xx00 xx-1

        psrldq      xmm5,       2                           ; xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00
        pmullw      xmm4,       XMMWORD PTR [rdx+16]        ; x[-1] * H[-1]; Tap 2


        punpcklbw   xmm5,       xmm0                        ; xx07 xx06 xx05 xx04 xx03 xx02 xx01 xx00
        psrldq      xmm6,       3                           ; xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01

        pmullw      xmm5,       [rdx+32]                    ; x[ 0] * H[ 0]; Tap 3

        punpcklbw   xmm6,       xmm0                        ; xx08 xx07 xx06 xx05 xx04 xx03 xx02 xx01
        psrldq      xmm7,       4                           ; xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02

        pmullw      xmm6,       [rdx+48]                    ; x[ 1] * h[ 1] ; Tap 4

        punpcklbw   xmm7,       xmm0                        ; xx09 xx08 xx07 xx06 xx05 xx04 xx03 xx02
        psrldq      xmm1,       5                           ; xx xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03


        pmullw      xmm7,       [rdx+64]                    ; x[ 2] * h[ 2] ; Tap 5

        punpcklbw   xmm1,       xmm0                        ; xx0a xx09 xx08 xx07 xx06 xx05 xx04 xx03
        pmullw      xmm1,       [rdx+80]                    ; x[ 3] * h[ 3] ; Tap 6

        paddsw      xmm4,       xmm7
        paddsw      xmm4,       xmm5

        paddsw      xmm4,       xmm3
        paddsw      xmm4,       xmm6

        paddsw      xmm4,       xmm1
        paddsw      xmm4,       [GLOBAL(rd)]

        psraw       xmm4,       7

        packuswb    xmm4,       xmm0
        punpcklbw   xmm4,       xmm0

        movdqa      XMMWORD Ptr [rdi],         xmm4

        movdqa      xmm3,       xmm2
        movdqa      xmm4,       xmm2

        movdqa      xmm5,       xmm2
        movdqa      xmm6,       xmm2

        movdqa      xmm7,       xmm2

        punpcklbw   xmm3,       xmm0                        ; xx05 xx04 xx03 xx02 xx01 xx01 xx-1 xx-2
        psrldq      xmm4,       1                           ; xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00 -1

        pmullw      xmm3,       XMMWORD PTR [rdx]           ; x[-2] * H[-2]; Tap 1
        punpcklbw   xmm4,       xmm0                        ; xx06 xx05 xx04 xx03 xx02 xx01 xx00 xx-1

        psrldq      xmm5,       2                           ; xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00
        pmullw      xmm4,       XMMWORD PTR [rdx+16]        ; x[-1] * H[-1]; Tap 2


        punpcklbw   xmm5,       xmm0                        ; xx07 xx06 xx05 xx04 xx03 xx02 xx01 xx00
        psrldq      xmm6,       3                           ; xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01

        pmullw      xmm5,       [rdx+32]                    ; x[ 0] * H[ 0]; Tap 3

        punpcklbw   xmm6,       xmm0                        ; xx08 xx07 xx06 xx05 xx04 xx03 xx02 xx01
        psrldq      xmm7,       4                           ; xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02

        pmullw      xmm6,       [rdx+48]                    ; x[ 1] * h[ 1] ; Tap 4

        punpcklbw   xmm7,       xmm0                        ; xx09 xx08 xx07 xx06 xx05 xx04 xx03 xx02
        psrldq      xmm2,       5                           ; xx xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03

        pmullw      xmm7,       [rdx+64]                    ; x[ 2] * h[ 2] ; Tap 5

        punpcklbw   xmm2,       xmm0                        ; xx0a xx09 xx08 xx07 xx06 xx05 xx04 xx03
        pmullw      xmm2,       [rdx+80]                    ; x[ 3] * h[ 3] ; Tap 6


        paddsw      xmm4,       xmm7
        paddsw      xmm4,       xmm5

        paddsw      xmm4,       xmm3
        paddsw      xmm4,       xmm6

        paddsw      xmm4,       xmm2
        paddsw      xmm4,       [GLOBAL(rd)]

        psraw       xmm4,       7

        packuswb    xmm4,       xmm0
        punpcklbw   xmm4,       xmm0

        movdqa      XMMWORD Ptr [rdi+16],      xmm4

        lea         rsi,        [rsi + rax]
%if ABI_IS_32BIT
        add         rdi,        DWORD Ptr arg(5) ;[output_width]
%else
        add         rdi,        r8
%endif

        dec         rcx
        jnz         .filter_block1d16_h6_sse2_rowloop                ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_filter_block1d8_v6_sse2
;(
;    short *src_ptr,
;    unsigned char *output_ptr,
;    int dst_ptich,
;    unsigned int pixels_per_line,
;    unsigned int pixel_step,
;    unsigned int output_height,
;    unsigned int output_width,
;    short * vp8_filter
;)
;/************************************************************************************
; Notes: filter_block1d8_v6 applies a 6 tap filter vertically to the input pixels. The
; input pixel array has output_height rows.
;*************************************************************************************/
globalsym(vp8_filter_block1d8_v6_sse2)
sym(vp8_filter_block1d8_v6_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 8
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rax,        arg(7) ;vp8_filter
        movsxd      rdx,        dword ptr arg(3) ;pixels_per_line

        mov         rdi,        arg(1) ;output_ptr
        mov         rsi,        arg(0) ;src_ptr

        sub         rsi,        rdx
        sub         rsi,        rdx

        movsxd      rcx,        DWORD PTR arg(5) ;[output_height]
        pxor        xmm0,       xmm0                        ; clear xmm0

        movdqa      xmm7,       XMMWORD PTR [GLOBAL(rd)]
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(2) ; dst_ptich
%endif

.vp8_filter_block1d8_v6_sse2_loop:
        movdqa      xmm1,       XMMWORD PTR [rsi]
        pmullw      xmm1,       [rax]

        movdqa      xmm2,       XMMWORD PTR [rsi + rdx]
        pmullw      xmm2,       [rax + 16]

        movdqa      xmm3,       XMMWORD PTR [rsi + rdx * 2]
        pmullw      xmm3,       [rax + 32]

        movdqa      xmm5,       XMMWORD PTR [rsi + rdx * 4]
        pmullw      xmm5,       [rax + 64]

        add         rsi,        rdx
        movdqa      xmm4,       XMMWORD PTR [rsi + rdx * 2]

        pmullw      xmm4,       [rax + 48]
        movdqa      xmm6,       XMMWORD PTR [rsi + rdx * 4]

        pmullw      xmm6,       [rax + 80]

        paddsw      xmm2,       xmm5
        paddsw      xmm2,       xmm3

        paddsw      xmm2,       xmm1
        paddsw      xmm2,       xmm4

        paddsw      xmm2,       xmm6
        paddsw      xmm2,       xmm7

        psraw       xmm2,       7
        packuswb    xmm2,       xmm0              ; pack and saturate

        movq        QWORD PTR [rdi], xmm2         ; store the results in the destination
%if ABI_IS_32BIT
        add         rdi,        DWORD PTR arg(2) ;[dst_ptich]
%else
        add         rdi,        r8
%endif
        dec         rcx         ; decrement count
        jnz         .vp8_filter_block1d8_v6_sse2_loop               ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_filter_block1d16_v6_sse2
;(
;    unsigned short *src_ptr,
;    unsigned char *output_ptr,
;    int dst_ptich,
;    unsigned int pixels_per_line,
;    unsigned int pixel_step,
;    unsigned int output_height,
;    unsigned int output_width,
;    const short    *vp8_filter
;)
;/************************************************************************************
; Notes: filter_block1d16_v6 applies a 6 tap filter vertically to the input pixels. The
; input pixel array has output_height rows.
;*************************************************************************************/
globalsym(vp8_filter_block1d16_v6_sse2)
sym(vp8_filter_block1d16_v6_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 8
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rax,        arg(7) ;vp8_filter
        movsxd      rdx,        dword ptr arg(3) ;pixels_per_line

        mov         rdi,        arg(1) ;output_ptr
        mov         rsi,        arg(0) ;src_ptr

        sub         rsi,        rdx
        sub         rsi,        rdx

        movsxd      rcx,        DWORD PTR arg(5) ;[output_height]
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(2) ; dst_ptich
%endif

.vp8_filter_block1d16_v6_sse2_loop:
; The order for adding 6-tap is 2 5 3 1 4 6. Read in data in that order.
        movdqa      xmm1,       XMMWORD PTR [rsi + rdx]       ; line 2
        movdqa      xmm2,       XMMWORD PTR [rsi + rdx + 16]
        pmullw      xmm1,       [rax + 16]
        pmullw      xmm2,       [rax + 16]

        movdqa      xmm3,       XMMWORD PTR [rsi + rdx * 4]       ; line 5
        movdqa      xmm4,       XMMWORD PTR [rsi + rdx * 4 + 16]
        pmullw      xmm3,       [rax + 64]
        pmullw      xmm4,       [rax + 64]

        movdqa      xmm5,       XMMWORD PTR [rsi + rdx * 2]       ; line 3
        movdqa      xmm6,       XMMWORD PTR [rsi + rdx * 2 + 16]
        pmullw      xmm5,       [rax + 32]
        pmullw      xmm6,       [rax + 32]

        movdqa      xmm7,       XMMWORD PTR [rsi]       ; line 1
        movdqa      xmm0,       XMMWORD PTR [rsi + 16]
        pmullw      xmm7,       [rax]
        pmullw      xmm0,       [rax]

        paddsw      xmm1,       xmm3
        paddsw      xmm2,       xmm4
        paddsw      xmm1,       xmm5
        paddsw      xmm2,       xmm6
        paddsw      xmm1,       xmm7
        paddsw      xmm2,       xmm0

        add         rsi,        rdx

        movdqa      xmm3,       XMMWORD PTR [rsi + rdx * 2]       ; line 4
        movdqa      xmm4,       XMMWORD PTR [rsi + rdx * 2 + 16]
        pmullw      xmm3,       [rax + 48]
        pmullw      xmm4,       [rax + 48]

        movdqa      xmm5,       XMMWORD PTR [rsi + rdx * 4]       ; line 6
        movdqa      xmm6,       XMMWORD PTR [rsi + rdx * 4 + 16]
        pmullw      xmm5,       [rax + 80]
        pmullw      xmm6,       [rax + 80]

        movdqa      xmm7,       XMMWORD PTR [GLOBAL(rd)]
        pxor        xmm0,       xmm0                        ; clear xmm0

        paddsw      xmm1,       xmm3
        paddsw      xmm2,       xmm4
        paddsw      xmm1,       xmm5
        paddsw      xmm2,       xmm6

        paddsw      xmm1,       xmm7
        paddsw      xmm2,       xmm7

        psraw       xmm1,       7
        psraw       xmm2,       7

        packuswb    xmm1,       xmm2              ; pack and saturate
        movdqa      XMMWORD PTR [rdi], xmm1       ; store the results in the destination
%if ABI_IS_32BIT
        add         rdi,        DWORD PTR arg(2) ;[dst_ptich]
%else
        add         rdi,        r8
%endif
        dec         rcx         ; decrement count
        jnz         .vp8_filter_block1d16_v6_sse2_loop              ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_filter_block1d8_h6_only_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    int dst_ptich,
;    unsigned int    output_height,
;    const short    *vp8_filter
;)
; First-pass filter only when yoffset==0
globalsym(vp8_filter_block1d8_h6_only_sse2)
sym(vp8_filter_block1d8_h6_only_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rdx,        arg(5) ;vp8_filter
        mov         rsi,        arg(0) ;src_ptr

        mov         rdi,        arg(2) ;output_ptr

        movsxd      rcx,        dword ptr arg(4) ;output_height
        movsxd      rax,        dword ptr arg(1) ;src_pixels_per_line            ; Pitch for Source
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(3) ;dst_ptich
%endif
        pxor        xmm0,       xmm0                        ; clear xmm0 for unpack

.filter_block1d8_h6_only_rowloop:
        movq        xmm3,       MMWORD PTR [rsi - 2]
        movq        xmm1,       MMWORD PTR [rsi + 6]

        prefetcht2  [rsi+rax-2]

        pslldq      xmm1,       8
        por         xmm1,       xmm3

        movdqa      xmm4,       xmm1
        movdqa      xmm5,       xmm1

        movdqa      xmm6,       xmm1
        movdqa      xmm7,       xmm1

        punpcklbw   xmm3,       xmm0                        ; xx05 xx04 xx03 xx02 xx01 xx01 xx-1 xx-2
        psrldq      xmm4,       1                           ; xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00 -1

        pmullw      xmm3,       XMMWORD PTR [rdx]           ; x[-2] * H[-2]; Tap 1
        punpcklbw   xmm4,       xmm0                        ; xx06 xx05 xx04 xx03 xx02 xx01 xx00 xx-1

        psrldq      xmm5,       2                           ; xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00
        pmullw      xmm4,       XMMWORD PTR [rdx+16]        ; x[-1] * H[-1]; Tap 2


        punpcklbw   xmm5,       xmm0                        ; xx07 xx06 xx05 xx04 xx03 xx02 xx01 xx00
        psrldq      xmm6,       3                           ; xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01

        pmullw      xmm5,       [rdx+32]                    ; x[ 0] * H[ 0]; Tap 3

        punpcklbw   xmm6,       xmm0                        ; xx08 xx07 xx06 xx05 xx04 xx03 xx02 xx01
        psrldq      xmm7,       4                           ; xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02

        pmullw      xmm6,       [rdx+48]                    ; x[ 1] * h[ 1] ; Tap 4

        punpcklbw   xmm7,       xmm0                        ; xx09 xx08 xx07 xx06 xx05 xx04 xx03 xx02
        psrldq      xmm1,       5                           ; xx xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03


        pmullw      xmm7,       [rdx+64]                    ; x[ 2] * h[ 2] ; Tap 5

        punpcklbw   xmm1,       xmm0                        ; xx0a xx09 xx08 xx07 xx06 xx05 xx04 xx03
        pmullw      xmm1,       [rdx+80]                    ; x[ 3] * h[ 3] ; Tap 6


        paddsw      xmm4,       xmm7
        paddsw      xmm4,       xmm5

        paddsw      xmm4,       xmm3
        paddsw      xmm4,       xmm6

        paddsw      xmm4,       xmm1
        paddsw      xmm4,       [GLOBAL(rd)]

        psraw       xmm4,       7

        packuswb    xmm4,       xmm0

        movq        QWORD PTR [rdi],   xmm4       ; store the results in the destination
        lea         rsi,        [rsi + rax]

%if ABI_IS_32BIT
        add         rdi,        DWORD Ptr arg(3) ;dst_ptich
%else
        add         rdi,        r8
%endif
        dec         rcx

        jnz         .filter_block1d8_h6_only_rowloop               ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_filter_block1d16_h6_only_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char  *output_ptr,
;    int dst_ptich,
;    unsigned int    output_height,
;    const short    *vp8_filter
;)
; First-pass filter only when yoffset==0
globalsym(vp8_filter_block1d16_h6_only_sse2)
sym(vp8_filter_block1d16_h6_only_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rdx,        arg(5) ;vp8_filter
        mov         rsi,        arg(0) ;src_ptr

        mov         rdi,        arg(2) ;output_ptr

        movsxd      rcx,        dword ptr arg(4) ;output_height
        movsxd      rax,        dword ptr arg(1) ;src_pixels_per_line            ; Pitch for Source
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(3) ;dst_ptich
%endif

        pxor        xmm0,       xmm0                        ; clear xmm0 for unpack

.filter_block1d16_h6_only_sse2_rowloop:
        movq        xmm3,       MMWORD PTR [rsi - 2]
        movq        xmm1,       MMWORD PTR [rsi + 6]

        movq        xmm2,       MMWORD PTR [rsi +14]
        pslldq      xmm2,       8

        por         xmm2,       xmm1
        prefetcht2  [rsi+rax-2]

        pslldq      xmm1,       8
        por         xmm1,       xmm3

        movdqa      xmm4,       xmm1
        movdqa      xmm5,       xmm1

        movdqa      xmm6,       xmm1
        movdqa      xmm7,       xmm1

        punpcklbw   xmm3,       xmm0                        ; xx05 xx04 xx03 xx02 xx01 xx01 xx-1 xx-2
        psrldq      xmm4,       1                           ; xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00 -1

        pmullw      xmm3,       XMMWORD PTR [rdx]           ; x[-2] * H[-2]; Tap 1
        punpcklbw   xmm4,       xmm0                        ; xx06 xx05 xx04 xx03 xx02 xx01 xx00 xx-1

        psrldq      xmm5,       2                           ; xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00
        pmullw      xmm4,       XMMWORD PTR [rdx+16]        ; x[-1] * H[-1]; Tap 2

        punpcklbw   xmm5,       xmm0                        ; xx07 xx06 xx05 xx04 xx03 xx02 xx01 xx00
        psrldq      xmm6,       3                           ; xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01

        pmullw      xmm5,       [rdx+32]                    ; x[ 0] * H[ 0]; Tap 3

        punpcklbw   xmm6,       xmm0                        ; xx08 xx07 xx06 xx05 xx04 xx03 xx02 xx01
        psrldq      xmm7,       4                           ; xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02

        pmullw      xmm6,       [rdx+48]                    ; x[ 1] * h[ 1] ; Tap 4

        punpcklbw   xmm7,       xmm0                        ; xx09 xx08 xx07 xx06 xx05 xx04 xx03 xx02
        psrldq      xmm1,       5                           ; xx xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03

        pmullw      xmm7,       [rdx+64]                    ; x[ 2] * h[ 2] ; Tap 5

        punpcklbw   xmm1,       xmm0                        ; xx0a xx09 xx08 xx07 xx06 xx05 xx04 xx03
        pmullw      xmm1,       [rdx+80]                    ; x[ 3] * h[ 3] ; Tap 6

        paddsw      xmm4,       xmm7
        paddsw      xmm4,       xmm5

        paddsw      xmm4,       xmm3
        paddsw      xmm4,       xmm6

        paddsw      xmm4,       xmm1
        paddsw      xmm4,       [GLOBAL(rd)]

        psraw       xmm4,       7

        packuswb    xmm4,       xmm0                        ; lower 8 bytes

        movq        QWORD Ptr [rdi],         xmm4           ; store the results in the destination

        movdqa      xmm3,       xmm2
        movdqa      xmm4,       xmm2

        movdqa      xmm5,       xmm2
        movdqa      xmm6,       xmm2

        movdqa      xmm7,       xmm2

        punpcklbw   xmm3,       xmm0                        ; xx05 xx04 xx03 xx02 xx01 xx01 xx-1 xx-2
        psrldq      xmm4,       1                           ; xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00 -1

        pmullw      xmm3,       XMMWORD PTR [rdx]           ; x[-2] * H[-2]; Tap 1
        punpcklbw   xmm4,       xmm0                        ; xx06 xx05 xx04 xx03 xx02 xx01 xx00 xx-1

        psrldq      xmm5,       2                           ; xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00
        pmullw      xmm4,       XMMWORD PTR [rdx+16]        ; x[-1] * H[-1]; Tap 2

        punpcklbw   xmm5,       xmm0                        ; xx07 xx06 xx05 xx04 xx03 xx02 xx01 xx00
        psrldq      xmm6,       3                           ; xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01

        pmullw      xmm5,       [rdx+32]                    ; x[ 0] * H[ 0]; Tap 3

        punpcklbw   xmm6,       xmm0                        ; xx08 xx07 xx06 xx05 xx04 xx03 xx02 xx01
        psrldq      xmm7,       4                           ; xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03 02

        pmullw      xmm6,       [rdx+48]                    ; x[ 1] * h[ 1] ; Tap 4

        punpcklbw   xmm7,       xmm0                        ; xx09 xx08 xx07 xx06 xx05 xx04 xx03 xx02
        psrldq      xmm2,       5                           ; xx xx xx xx xx 0d 0c 0b 0a 09 08 07 06 05 04 03

        pmullw      xmm7,       [rdx+64]                    ; x[ 2] * h[ 2] ; Tap 5

        punpcklbw   xmm2,       xmm0                        ; xx0a xx09 xx08 xx07 xx06 xx05 xx04 xx03
        pmullw      xmm2,       [rdx+80]                    ; x[ 3] * h[ 3] ; Tap 6

        paddsw      xmm4,       xmm7
        paddsw      xmm4,       xmm5

        paddsw      xmm4,       xmm3
        paddsw      xmm4,       xmm6

        paddsw      xmm4,       xmm2
        paddsw      xmm4,       [GLOBAL(rd)]

        psraw       xmm4,       7

        packuswb    xmm4,       xmm0                        ; higher 8 bytes

        movq        QWORD Ptr [rdi+8],      xmm4            ; store the results in the destination

        lea         rsi,        [rsi + rax]
%if ABI_IS_32BIT
        add         rdi,        DWORD Ptr arg(3) ;dst_ptich
%else
        add         rdi,        r8
%endif

        dec         rcx
        jnz         .filter_block1d16_h6_only_sse2_rowloop               ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_filter_block1d8_v6_only_sse2
;(
;    unsigned char *src_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned char *output_ptr,
;    int dst_ptich,
;    unsigned int output_height,
;    const short    *vp8_filter
;)
; Second-pass filter only when xoffset==0
globalsym(vp8_filter_block1d8_v6_only_sse2)
sym(vp8_filter_block1d8_v6_only_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 6
    SAVE_XMM 7
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rsi,        arg(0) ;src_ptr
        mov         rdi,        arg(2) ;output_ptr

        movsxd      rcx,        dword ptr arg(4) ;output_height
        movsxd      rdx,        dword ptr arg(1) ;src_pixels_per_line

        mov         rax,        arg(5) ;vp8_filter

        pxor        xmm0,       xmm0                        ; clear xmm0

        movdqa      xmm7,       XMMWORD PTR [GLOBAL(rd)]
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(3) ; dst_ptich
%endif

.vp8_filter_block1d8_v6_only_sse2_loop:
        movq        xmm1,       MMWORD PTR [rsi]
        movq        xmm2,       MMWORD PTR [rsi + rdx]
        movq        xmm3,       MMWORD PTR [rsi + rdx * 2]
        movq        xmm5,       MMWORD PTR [rsi + rdx * 4]
        add         rsi,        rdx
        movq        xmm4,       MMWORD PTR [rsi + rdx * 2]
        movq        xmm6,       MMWORD PTR [rsi + rdx * 4]

        punpcklbw   xmm1,       xmm0
        pmullw      xmm1,       [rax]

        punpcklbw   xmm2,       xmm0
        pmullw      xmm2,       [rax + 16]

        punpcklbw   xmm3,       xmm0
        pmullw      xmm3,       [rax + 32]

        punpcklbw   xmm5,       xmm0
        pmullw      xmm5,       [rax + 64]

        punpcklbw   xmm4,       xmm0
        pmullw      xmm4,       [rax + 48]

        punpcklbw   xmm6,       xmm0
        pmullw      xmm6,       [rax + 80]

        paddsw      xmm2,       xmm5
        paddsw      xmm2,       xmm3

        paddsw      xmm2,       xmm1
        paddsw      xmm2,       xmm4

        paddsw      xmm2,       xmm6
        paddsw      xmm2,       xmm7

        psraw       xmm2,       7
        packuswb    xmm2,       xmm0              ; pack and saturate

        movq        QWORD PTR [rdi], xmm2         ; store the results in the destination
%if ABI_IS_32BIT
        add         rdi,        DWORD PTR arg(3) ;[dst_ptich]
%else
        add         rdi,        r8
%endif
        dec         rcx         ; decrement count
        jnz         .vp8_filter_block1d8_v6_only_sse2_loop              ; next row

    ; begin epilog
    pop rdi
    pop rsi
    RESTORE_GOT
    RESTORE_XMM
    UNSHADOW_ARGS
    pop         rbp
    ret


;void vp8_unpack_block1d16_h6_sse2
;(
;    unsigned char  *src_ptr,
;    unsigned short *output_ptr,
;    unsigned int    src_pixels_per_line,
;    unsigned int    output_height,
;    unsigned int    output_width
;)
globalsym(vp8_unpack_block1d16_h6_sse2)
sym(vp8_unpack_block1d16_h6_sse2):
    push        rbp
    mov         rbp, rsp
    SHADOW_ARGS_TO_STACK 5
    GET_GOT     rbx
    push        rsi
    push        rdi
    ; end prolog

        mov         rsi,        arg(0) ;src_ptr
        mov         rdi,        arg(1) ;output_ptr

        movsxd      rcx,        dword ptr arg(3) ;output_height
        movsxd      rax,        dword ptr arg(2) ;src_pixels_per_line            ; Pitch for Source

        pxor        xmm0,       xmm0                        ; clear xmm0 for unpack
%if ABI_IS_32BIT=0
        movsxd      r8,         dword ptr arg(4) ;output_width            ; Pitch for Source
%endif

.unpack_block1d16_h6_sse2_rowloop:
        movq        xmm1,       MMWORD PTR [rsi]            ; 0d 0c 0b 0a 09 08 07 06 05 04 03 02 01 00 -1 -2
        movq        xmm3,       MMWORD PTR [rsi+8]          ; make copy of xmm1

        punpcklbw   xmm3,       xmm0                        ; xx05 xx04 xx03 xx02 xx01 xx01 xx-1 xx-2
        punpcklbw   xmm1,       xmm0

        movdqa      XMMWORD Ptr [rdi],         xmm1
        movdqa      XMMWORD Ptr [rdi + 16],    xmm3

        lea         rsi,        [rsi + rax]
%if ABI_IS_32BIT
        add         rdi,        DWORD Ptr arg(4) ;[output_width]
%else
        add         rdi,        r8
%endif
        dec         rcx
        jnz         .unpack_block1d16_h6_sse2_rowloop               ; next row

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
    times 8 dw 0x40
