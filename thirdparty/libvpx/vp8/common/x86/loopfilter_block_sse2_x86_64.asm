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

%macro LF_ABS 2
        ; %1 value not preserved
        ; %2 value preserved
        ; output in %1
        movdqa      scratch1, %2            ; v2

        psubusb     scratch1, %1            ; v2 - v1
        psubusb     %1, %2                  ; v1 - v2
        por         %1, scratch1            ; abs(v2 - v1)
%endmacro

%macro LF_FILTER_HEV_MASK 8-9

        LF_ABS      %1, %2                  ; abs(p3 - p2)
        LF_ABS      %2, %3                  ; abs(p2 - p1)
        pmaxub      %1, %2                  ; accumulate mask
%if %0 == 8
        movdqa      scratch2, %3            ; save p1
        LF_ABS      scratch2, %4            ; abs(p1 - p0)
%endif
        LF_ABS      %4, %5                  ; abs(p0 - q0)
        LF_ABS      %5, %6                  ; abs(q0 - q1)
%if %0 == 8
        pmaxub      %5, scratch2            ; accumulate hev
%else
        pmaxub      %5, %9
%endif
        pmaxub      %1, %5                  ; accumulate mask

        LF_ABS      %3, %6                  ; abs(p1 - q1)
        LF_ABS      %6, %7                  ; abs(q1 - q2)
        pmaxub      %1, %6                  ; accumulate mask
        LF_ABS      %7, %8                  ; abs(q2 - q3)
        pmaxub      %1, %7                  ; accumulate mask

        paddusb     %4, %4                  ; 2 * abs(p0 - q0)
        pand        %3, [GLOBAL(tfe)]
        psrlw       %3, 1                   ; abs(p1 - q1) / 2
        paddusb     %4, %3                  ; abs(p0 - q0) * 2 + abs(p1 - q1) / 2

        psubusb     %1, [limit]
        psubusb     %4, [blimit]
        por         %1, %4
        pcmpeqb     %1, zero                ; mask

        psubusb     %5, [thresh]
        pcmpeqb     %5, zero                ; ~hev
%endmacro

%macro LF_FILTER 6
        ; %1-%4: p1-q1
        ; %5: mask
        ; %6: hev

        movdqa      scratch2, %6            ; save hev

        pxor        %1, [GLOBAL(t80)]       ; ps1
        pxor        %4, [GLOBAL(t80)]       ; qs1
        movdqa      scratch1, %1
        psubsb      scratch1, %4            ; signed_char_clamp(ps1 - qs1)
        pandn       scratch2, scratch1      ; vp8_filter &= hev

        pxor        %2, [GLOBAL(t80)]       ; ps0
        pxor        %3, [GLOBAL(t80)]       ; qs0
        movdqa      scratch1, %3
        psubsb      scratch1, %2            ; qs0 - ps0
        paddsb      scratch2, scratch1      ; vp8_filter += (qs0 - ps0)
        paddsb      scratch2, scratch1      ; vp8_filter += (qs0 - ps0)
        paddsb      scratch2, scratch1      ; vp8_filter += (qs0 - ps0)
        pand        %5, scratch2            ; &= mask

        movdqa      scratch2, %5
        paddsb      %5, [GLOBAL(t4)]        ; Filter1
        paddsb      scratch2, [GLOBAL(t3)]  ; Filter2

        ; Filter1 >> 3
        movdqa      scratch1, zero
        pcmpgtb     scratch1, %5
        psrlw       %5, 3
        pand        scratch1, [GLOBAL(te0)]
        pand        %5, [GLOBAL(t1f)]
        por         %5, scratch1

        psubsb      %3, %5                  ; qs0 - Filter1
        pxor        %3, [GLOBAL(t80)]

        ; Filter2 >> 3
        movdqa      scratch1, zero
        pcmpgtb     scratch1, scratch2
        psrlw       scratch2, 3
        pand        scratch1, [GLOBAL(te0)]
        pand        scratch2, [GLOBAL(t1f)]
        por         scratch2, scratch1

        paddsb      %2, scratch2            ; ps0 + Filter2
        pxor        %2, [GLOBAL(t80)]

        ; outer tap adjustments
        paddsb      %5, [GLOBAL(t1)]
        movdqa      scratch1, zero
        pcmpgtb     scratch1, %5
        psrlw       %5, 1
        pand        scratch1, [GLOBAL(t80)]
        pand        %5, [GLOBAL(t7f)]
        por         %5, scratch1
        pand        %5, %6                  ; vp8_filter &= ~hev

        psubsb      %4, %5                  ; qs1 - vp8_filter
        pxor        %4, [GLOBAL(t80)]

        paddsb      %1, %5                  ; ps1 + vp8_filter
        pxor        %1, [GLOBAL(t80)]
%endmacro

SECTION .text

;void vp8_loop_filter_bh_y_sse2
;(
;    unsigned char *src_ptr,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh
;)
globalsym(vp8_loop_filter_bh_y_sse2)
sym(vp8_loop_filter_bh_y_sse2):

%if LIBVPX_YASM_WIN64
    %define src      rcx ; src_ptr
    %define stride   rdx ; src_pixel_step
    %define blimit   r8
    %define limit    r9
    %define thresh   r10

    %define spp      rax
    %define stride3  r11
    %define stride5  r12
    %define stride7  r13

    push    rbp
    mov     rbp, rsp
    SAVE_XMM 11
    push    r12
    push    r13
    mov     thresh, arg(4)
%else
    %define src      rdi ; src_ptr
    %define stride   rsi ; src_pixel_step
    %define blimit   rdx
    %define limit    rcx
    %define thresh   r8

    %define spp      rax
    %define stride3  r9
    %define stride5  r10
    %define stride7  r11
%endif

    %define scratch1 xmm5
    %define scratch2 xmm6
    %define zero     xmm7

    %define i0       [src]
    %define i1       [spp]
    %define i2       [src + 2 * stride]
    %define i3       [spp + 2 * stride]
    %define i4       [src + 4 * stride]
    %define i5       [spp + 4 * stride]
    %define i6       [src + 2 * stride3]
    %define i7       [spp + 2 * stride3]
    %define i8       [src + 8 * stride]
    %define i9       [spp + 8 * stride]
    %define i10      [src + 2 * stride5]
    %define i11      [spp + 2 * stride5]
    %define i12      [src + 4 * stride3]
    %define i13      [spp + 4 * stride3]
    %define i14      [src + 2 * stride7]
    %define i15      [spp + 2 * stride7]

    ; prep work
    lea         spp, [src + stride]
    lea         stride3, [stride + 2 * stride]
    lea         stride5, [stride3 + 2 * stride]
    lea         stride7, [stride3 + 4 * stride]
    pxor        zero, zero

        ; load the first set into registers
        movdqa       xmm0, i0
        movdqa       xmm1, i1
        movdqa       xmm2, i2
        movdqa       xmm3, i3
        movdqa       xmm4, i4
        movdqa       xmm8, i5
        movdqa       xmm9, i6   ; q2, will contain abs(p1-p0)
        movdqa       xmm10, i7
LF_FILTER_HEV_MASK xmm0, xmm1, xmm2, xmm3, xmm4, xmm8, xmm9, xmm10

        movdqa       xmm1, i2
        movdqa       xmm2, i3
        movdqa       xmm3, i4
        movdqa       xmm8, i5
LF_FILTER xmm1, xmm2, xmm3, xmm8, xmm0, xmm4
        movdqa       i2, xmm1
        movdqa       i3, xmm2

; second set
        movdqa       i4, xmm3
        movdqa       i5, xmm8

        movdqa       xmm0, i6
        movdqa       xmm1, i7
        movdqa       xmm2, i8
        movdqa       xmm4, i9
        movdqa       xmm10, i10   ; q2, will contain abs(p1-p0)
        movdqa       xmm11, i11
LF_FILTER_HEV_MASK xmm3, xmm8, xmm0, xmm1, xmm2, xmm4, xmm10, xmm11, xmm9

        movdqa       xmm0, i6
        movdqa       xmm1, i7
        movdqa       xmm4, i8
        movdqa       xmm8, i9
LF_FILTER xmm0, xmm1, xmm4, xmm8, xmm3, xmm2
        movdqa       i6, xmm0
        movdqa       i7, xmm1

; last set
        movdqa       i8, xmm4
        movdqa       i9, xmm8

        movdqa       xmm0, i10
        movdqa       xmm1, i11
        movdqa       xmm2, i12
        movdqa       xmm3, i13
        movdqa       xmm9, i14   ; q2, will contain abs(p1-p0)
        movdqa       xmm11, i15
LF_FILTER_HEV_MASK xmm4, xmm8, xmm0, xmm1, xmm2, xmm3, xmm9, xmm11, xmm10

        movdqa       xmm0, i10
        movdqa       xmm1, i11
        movdqa       xmm3, i12
        movdqa       xmm8, i13
LF_FILTER xmm0, xmm1, xmm3, xmm8, xmm4, xmm2
        movdqa       i10, xmm0
        movdqa       i11, xmm1
        movdqa       i12, xmm3
        movdqa       i13, xmm8

%if LIBVPX_YASM_WIN64
    pop    r13
    pop    r12
    RESTORE_XMM
    pop    rbp
%endif

    ret


;void vp8_loop_filter_bv_y_sse2
;(
;    unsigned char *src_ptr,
;    int            src_pixel_step,
;    const char    *blimit,
;    const char    *limit,
;    const char    *thresh
;)

globalsym(vp8_loop_filter_bv_y_sse2)
sym(vp8_loop_filter_bv_y_sse2):

%if LIBVPX_YASM_WIN64
    %define src      rcx ; src_ptr
    %define stride   rdx ; src_pixel_step
    %define blimit   r8
    %define limit    r9
    %define thresh   r10

    %define spp      rax
    %define stride3  r11
    %define stride5  r12
    %define stride7  r13

    push    rbp
    mov     rbp, rsp
    SAVE_XMM 15
    push    r12
    push    r13
    mov     thresh, arg(4)
%else
    %define src      rdi
    %define stride   rsi
    %define blimit   rdx
    %define limit    rcx
    %define thresh   r8

    %define spp      rax
    %define stride3  r9
    %define stride5  r10
    %define stride7  r11
%endif

    %define scratch1 xmm5
    %define scratch2 xmm6
    %define zero     xmm7

    %define s0       [src]
    %define s1       [spp]
    %define s2       [src + 2 * stride]
    %define s3       [spp + 2 * stride]
    %define s4       [src + 4 * stride]
    %define s5       [spp + 4 * stride]
    %define s6       [src + 2 * stride3]
    %define s7       [spp + 2 * stride3]
    %define s8       [src + 8 * stride]
    %define s9       [spp + 8 * stride]
    %define s10      [src + 2 * stride5]
    %define s11      [spp + 2 * stride5]
    %define s12      [src + 4 * stride3]
    %define s13      [spp + 4 * stride3]
    %define s14      [src + 2 * stride7]
    %define s15      [spp + 2 * stride7]

    %define i0       [rsp]
    %define i1       [rsp + 16]
    %define i2       [rsp + 32]
    %define i3       [rsp + 48]
    %define i4       [rsp + 64]
    %define i5       [rsp + 80]
    %define i6       [rsp + 96]
    %define i7       [rsp + 112]
    %define i8       [rsp + 128]
    %define i9       [rsp + 144]
    %define i10      [rsp + 160]
    %define i11      [rsp + 176]
    %define i12      [rsp + 192]
    %define i13      [rsp + 208]
    %define i14      [rsp + 224]
    %define i15      [rsp + 240]

    ALIGN_STACK 16, rax

    ; reserve stack space
    %define      temp_storage  0 ; size is 256 (16*16)
    %define      stack_size 256
    sub          rsp, stack_size

    ; prep work
    lea         spp, [src + stride]
    lea         stride3, [stride + 2 * stride]
    lea         stride5, [stride3 + 2 * stride]
    lea         stride7, [stride3 + 4 * stride]

        ; 8-f
        movdqa      xmm0, s8
        movdqa      xmm1, xmm0
        punpcklbw   xmm0, s9                ; 80 90
        punpckhbw   xmm1, s9                ; 88 98

        movdqa      xmm2, s10
        movdqa      xmm3, xmm2
        punpcklbw   xmm2, s11 ; a0 b0
        punpckhbw   xmm3, s11 ; a8 b8

        movdqa      xmm4, xmm0
        punpcklwd   xmm0, xmm2              ; 80 90 a0 b0
        punpckhwd   xmm4, xmm2              ; 84 94 a4 b4

        movdqa      xmm2, xmm1
        punpcklwd   xmm1, xmm3              ; 88 98 a8 b8
        punpckhwd   xmm2, xmm3              ; 8c 9c ac bc

        ; using xmm[0124]
        ; work on next 4 rows

        movdqa      xmm3, s12
        movdqa      xmm5, xmm3
        punpcklbw   xmm3, s13 ; c0 d0
        punpckhbw   xmm5, s13 ; c8 d8

        movdqa      xmm6, s14
        movdqa      xmm7, xmm6
        punpcklbw   xmm6, s15 ; e0 f0
        punpckhbw   xmm7, s15 ; e8 f8

        movdqa      xmm8, xmm3
        punpcklwd   xmm3, xmm6              ; c0 d0 e0 f0
        punpckhwd   xmm8, xmm6              ; c4 d4 e4 f4

        movdqa      xmm6, xmm5
        punpcklwd   xmm5, xmm7              ; c8 d8 e8 f8
        punpckhwd   xmm6, xmm7              ; cc dc ec fc

        ; pull the third and fourth sets together

        movdqa      xmm7, xmm0
        punpckldq   xmm0, xmm3              ; 80 90 a0 b0 c0 d0 e0 f0
        punpckhdq   xmm7, xmm3              ; 82 92 a2 b2 c2 d2 e2 f2

        movdqa      xmm3, xmm4
        punpckldq   xmm4, xmm8              ; 84 94 a4 b4 c4 d4 e4 f4
        punpckhdq   xmm3, xmm8              ; 86 96 a6 b6 c6 d6 e6 f6

        movdqa      xmm8, xmm1
        punpckldq   xmm1, xmm5              ; 88 88 a8 b8 c8 d8 e8 f8
        punpckhdq   xmm8, xmm5              ; 8a 9a aa ba ca da ea fa

        movdqa      xmm5, xmm2
        punpckldq   xmm2, xmm6              ; 8c 9c ac bc cc dc ec fc
        punpckhdq   xmm5, xmm6              ; 8e 9e ae be ce de ee fe

        ; save the calculations. we only have 15 registers ...
        movdqa      i0, xmm0
        movdqa      i1, xmm7
        movdqa      i2, xmm4
        movdqa      i3, xmm3
        movdqa      i4, xmm1
        movdqa      i5, xmm8
        movdqa      i6, xmm2
        movdqa      i7, xmm5

        ; 0-7
        movdqa      xmm0, s0
        movdqa      xmm1, xmm0
        punpcklbw   xmm0, s1 ; 00 10
        punpckhbw   xmm1, s1 ; 08 18

        movdqa      xmm2, s2
        movdqa      xmm3, xmm2
        punpcklbw   xmm2, s3 ; 20 30
        punpckhbw   xmm3, s3 ; 28 38

        movdqa      xmm4, xmm0
        punpcklwd   xmm0, xmm2              ; 00 10 20 30
        punpckhwd   xmm4, xmm2              ; 04 14 24 34

        movdqa      xmm2, xmm1
        punpcklwd   xmm1, xmm3              ; 08 18 28 38
        punpckhwd   xmm2, xmm3              ; 0c 1c 2c 3c

        ; using xmm[0124]
        ; work on next 4 rows

        movdqa      xmm3, s4
        movdqa      xmm5, xmm3
        punpcklbw   xmm3, s5 ; 40 50
        punpckhbw   xmm5, s5 ; 48 58

        movdqa      xmm6, s6
        movdqa      xmm7, xmm6
        punpcklbw   xmm6, s7   ; 60 70
        punpckhbw   xmm7, s7   ; 68 78

        movdqa      xmm8, xmm3
        punpcklwd   xmm3, xmm6              ; 40 50 60 70
        punpckhwd   xmm8, xmm6              ; 44 54 64 74

        movdqa      xmm6, xmm5
        punpcklwd   xmm5, xmm7              ; 48 58 68 78
        punpckhwd   xmm6, xmm7              ; 4c 5c 6c 7c

        ; pull the first two sets together

        movdqa      xmm7, xmm0
        punpckldq   xmm0, xmm3              ; 00 10 20 30 40 50 60 70
        punpckhdq   xmm7, xmm3              ; 02 12 22 32 42 52 62 72

        movdqa      xmm3, xmm4
        punpckldq   xmm4, xmm8              ; 04 14 24 34 44 54 64 74
        punpckhdq   xmm3, xmm8              ; 06 16 26 36 46 56 66 76

        movdqa      xmm8, xmm1
        punpckldq   xmm1, xmm5              ; 08 18 28 38 48 58 68 78
        punpckhdq   xmm8, xmm5              ; 0a 1a 2a 3a 4a 5a 6a 7a

        movdqa      xmm5, xmm2
        punpckldq   xmm2, xmm6              ; 0c 1c 2c 3c 4c 5c 6c 7c
        punpckhdq   xmm5, xmm6              ; 0e 1e 2e 3e 4e 5e 6e 7e
        ; final combination

        movdqa      xmm6, xmm0
        punpcklqdq  xmm0, i0
        punpckhqdq  xmm6, i0

        movdqa      xmm9, xmm7
        punpcklqdq  xmm7, i1
        punpckhqdq  xmm9, i1

        movdqa      xmm10, xmm4
        punpcklqdq  xmm4, i2
        punpckhqdq  xmm10, i2

        movdqa      xmm11, xmm3
        punpcklqdq  xmm3, i3
        punpckhqdq  xmm11, i3

        movdqa      xmm12, xmm1
        punpcklqdq  xmm1, i4
        punpckhqdq  xmm12, i4

        movdqa      xmm13, xmm8
        punpcklqdq  xmm8, i5
        punpckhqdq  xmm13, i5

        movdqa      xmm14, xmm2
        punpcklqdq  xmm2, i6
        punpckhqdq  xmm14, i6

        movdqa      xmm15, xmm5
        punpcklqdq  xmm5, i7
        punpckhqdq  xmm15, i7

        movdqa      i0, xmm0
        movdqa      i1, xmm6
        movdqa      i2, xmm7
        movdqa      i3, xmm9
        movdqa      i4, xmm4
        movdqa      i5, xmm10
        movdqa      i6, xmm3
        movdqa      i7, xmm11
        movdqa      i8, xmm1
        movdqa      i9, xmm12
        movdqa      i10, xmm8
        movdqa      i11, xmm13
        movdqa      i12, xmm2
        movdqa      i13, xmm14
        movdqa      i14, xmm5
        movdqa      i15, xmm15

; TRANSPOSED DATA AVAILABLE ON THE STACK

        movdqa      xmm12, xmm6
        movdqa      xmm13, xmm7

        pxor        zero, zero

LF_FILTER_HEV_MASK xmm0, xmm12, xmm13, xmm9, xmm4, xmm10, xmm3, xmm11

        movdqa       xmm1, i2
        movdqa       xmm2, i3
        movdqa       xmm8, i4
        movdqa       xmm9, i5
LF_FILTER xmm1, xmm2, xmm8, xmm9, xmm0, xmm4
        movdqa       i2, xmm1
        movdqa       i3, xmm2

; second set
        movdqa       i4, xmm8
        movdqa       i5, xmm9

        movdqa       xmm0, i6
        movdqa       xmm1, i7
        movdqa       xmm2, i8
        movdqa       xmm4, i9
        movdqa       xmm10, i10   ; q2, will contain abs(p1-p0)
        movdqa       xmm11, i11
LF_FILTER_HEV_MASK xmm8, xmm9, xmm0, xmm1, xmm2, xmm4, xmm10, xmm11, xmm3

        movdqa       xmm0, i6
        movdqa       xmm1, i7
        movdqa       xmm3, i8
        movdqa       xmm4, i9
LF_FILTER xmm0, xmm1, xmm3, xmm4, xmm8, xmm2
        movdqa       i6, xmm0
        movdqa       i7, xmm1

; last set
        movdqa       i8, xmm3
        movdqa       i9, xmm4

        movdqa       xmm0, i10
        movdqa       xmm1, i11
        movdqa       xmm2, i12
        movdqa       xmm8, i13
        movdqa       xmm9, i14   ; q2, will contain abs(p1-p0)
        movdqa       xmm11, i15
LF_FILTER_HEV_MASK xmm3, xmm4, xmm0, xmm1, xmm2, xmm8, xmm9, xmm11, xmm10

        movdqa       xmm0, i10
        movdqa       xmm1, i11
        movdqa       xmm4, i12
        movdqa       xmm8, i13
LF_FILTER xmm0, xmm1, xmm4, xmm8, xmm3, xmm2
        movdqa       i10, xmm0
        movdqa       i11, xmm1
        movdqa       i12, xmm4
        movdqa       i13, xmm8


; RESHUFFLE AND WRITE OUT
        ; 8-f
        movdqa      xmm0, i8
        movdqa      xmm1, xmm0
        punpcklbw   xmm0, i9                ; 80 90
        punpckhbw   xmm1, i9                ; 88 98

        movdqa      xmm2, i10
        movdqa      xmm3, xmm2
        punpcklbw   xmm2, i11               ; a0 b0
        punpckhbw   xmm3, i11               ; a8 b8

        movdqa      xmm4, xmm0
        punpcklwd   xmm0, xmm2              ; 80 90 a0 b0
        punpckhwd   xmm4, xmm2              ; 84 94 a4 b4

        movdqa      xmm2, xmm1
        punpcklwd   xmm1, xmm3              ; 88 98 a8 b8
        punpckhwd   xmm2, xmm3              ; 8c 9c ac bc

        ; using xmm[0124]
        ; work on next 4 rows

        movdqa      xmm3, i12
        movdqa      xmm5, xmm3
        punpcklbw   xmm3, i13               ; c0 d0
        punpckhbw   xmm5, i13               ; c8 d8

        movdqa      xmm6, i14
        movdqa      xmm7, xmm6
        punpcklbw   xmm6, i15               ; e0 f0
        punpckhbw   xmm7, i15               ; e8 f8

        movdqa      xmm8, xmm3
        punpcklwd   xmm3, xmm6              ; c0 d0 e0 f0
        punpckhwd   xmm8, xmm6              ; c4 d4 e4 f4

        movdqa      xmm6, xmm5
        punpcklwd   xmm5, xmm7              ; c8 d8 e8 f8
        punpckhwd   xmm6, xmm7              ; cc dc ec fc

        ; pull the third and fourth sets together

        movdqa      xmm7, xmm0
        punpckldq   xmm0, xmm3              ; 80 90 a0 b0 c0 d0 e0 f0
        punpckhdq   xmm7, xmm3              ; 82 92 a2 b2 c2 d2 e2 f2

        movdqa      xmm3, xmm4
        punpckldq   xmm4, xmm8              ; 84 94 a4 b4 c4 d4 e4 f4
        punpckhdq   xmm3, xmm8              ; 86 96 a6 b6 c6 d6 e6 f6

        movdqa      xmm8, xmm1
        punpckldq   xmm1, xmm5              ; 88 88 a8 b8 c8 d8 e8 f8
        punpckhdq   xmm8, xmm5              ; 8a 9a aa ba ca da ea fa

        movdqa      xmm5, xmm2
        punpckldq   xmm2, xmm6              ; 8c 9c ac bc cc dc ec fc
        punpckhdq   xmm5, xmm6              ; 8e 9e ae be ce de ee fe

        ; save the calculations. we only have 15 registers ...
        movdqa      i8, xmm0
        movdqa      i9, xmm7
        movdqa      i10, xmm4
        movdqa      i11, xmm3
        movdqa      i12, xmm1
        movdqa      i13, xmm8
        movdqa      i14, xmm2
        movdqa      i15, xmm5

        ; 0-7
        movdqa      xmm0, i0
        movdqa      xmm1, xmm0
        punpcklbw   xmm0, i1                ; 00 10
        punpckhbw   xmm1, i1                ; 08 18

        movdqa      xmm2, i2
        movdqa      xmm3, xmm2
        punpcklbw   xmm2, i3                ; 20 30
        punpckhbw   xmm3, i3                ; 28 38

        movdqa      xmm4, xmm0
        punpcklwd   xmm0, xmm2              ; 00 10 20 30
        punpckhwd   xmm4, xmm2              ; 04 14 24 34

        movdqa      xmm2, xmm1
        punpcklwd   xmm1, xmm3              ; 08 18 28 38
        punpckhwd   xmm2, xmm3              ; 0c 1c 2c 3c

        ; using xmm[0124]
        ; work on next 4 rows

        movdqa      xmm3, i4
        movdqa      xmm5, xmm3
        punpcklbw   xmm3, i5                ; 40 50
        punpckhbw   xmm5, i5                ; 48 58

        movdqa      xmm6, i6
        movdqa      xmm7, xmm6
        punpcklbw   xmm6, i7                ; 60 70
        punpckhbw   xmm7, i7                ; 68 78

        movdqa      xmm8, xmm3
        punpcklwd   xmm3, xmm6              ; 40 50 60 70
        punpckhwd   xmm8, xmm6              ; 44 54 64 74

        movdqa      xmm6, xmm5
        punpcklwd   xmm5, xmm7              ; 48 58 68 78
        punpckhwd   xmm6, xmm7              ; 4c 5c 6c 7c

        ; pull the first two sets together

        movdqa      xmm7, xmm0
        punpckldq   xmm0, xmm3              ; 00 10 20 30 40 50 60 70
        punpckhdq   xmm7, xmm3              ; 02 12 22 32 42 52 62 72

        movdqa      xmm3, xmm4
        punpckldq   xmm4, xmm8              ; 04 14 24 34 44 54 64 74
        punpckhdq   xmm3, xmm8              ; 06 16 26 36 46 56 66 76

        movdqa      xmm8, xmm1
        punpckldq   xmm1, xmm5              ; 08 18 28 38 48 58 68 78
        punpckhdq   xmm8, xmm5              ; 0a 1a 2a 3a 4a 5a 6a 7a

        movdqa      xmm5, xmm2
        punpckldq   xmm2, xmm6              ; 0c 1c 2c 3c 4c 5c 6c 7c
        punpckhdq   xmm5, xmm6              ; 0e 1e 2e 3e 4e 5e 6e 7e
        ; final combination

        movdqa      xmm6, xmm0
        punpcklqdq  xmm0, i8
        punpckhqdq  xmm6, i8

        movdqa      xmm9, xmm7
        punpcklqdq  xmm7, i9
        punpckhqdq  xmm9, i9

        movdqa      xmm10, xmm4
        punpcklqdq  xmm4, i10
        punpckhqdq  xmm10, i10

        movdqa      xmm11, xmm3
        punpcklqdq  xmm3, i11
        punpckhqdq  xmm11, i11

        movdqa      xmm12, xmm1
        punpcklqdq  xmm1, i12
        punpckhqdq  xmm12, i12

        movdqa      xmm13, xmm8
        punpcklqdq  xmm8, i13
        punpckhqdq  xmm13, i13

        movdqa      xmm14, xmm2
        punpcklqdq  xmm2, i14
        punpckhqdq  xmm14, i14

        movdqa      xmm15, xmm5
        punpcklqdq  xmm5, i15
        punpckhqdq  xmm15, i15

        movdqa      s0, xmm0
        movdqa      s1, xmm6
        movdqa      s2, xmm7
        movdqa      s3, xmm9
        movdqa      s4, xmm4
        movdqa      s5, xmm10
        movdqa      s6, xmm3
        movdqa      s7, xmm11
        movdqa      s8, xmm1
        movdqa      s9, xmm12
        movdqa      s10, xmm8
        movdqa      s11, xmm13
        movdqa      s12, xmm2
        movdqa      s13, xmm14
        movdqa      s14, xmm5
        movdqa      s15, xmm15

    ; free stack space
    add          rsp, stack_size

    ; un-ALIGN_STACK
    pop          rsp

%if LIBVPX_YASM_WIN64
    pop    r13
    pop    r12
    RESTORE_XMM
    pop    rbp
%endif

    ret

SECTION_RODATA
align 16
te0:
    times 16 db 0xe0
align 16
t7f:
    times 16 db 0x7f
align 16
tfe:
    times 16 db 0xfe
align 16
t1f:
    times 16 db 0x1f
align 16
t80:
    times 16 db 0x80
align 16
t1:
    times 16 db 0x01
align 16
t3:
    times 16 db 0x03
align 16
t4:
    times 16 db 0x04
