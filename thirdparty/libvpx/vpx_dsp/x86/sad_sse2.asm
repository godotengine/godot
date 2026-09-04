;
;  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

%include "third_party/x86inc/x86inc.asm"

SECTION .text

; Macro Arguments
; Arg 1: Width
; Arg 2: Height
; Arg 3: Number of general purpose registers
; Arg 4: Type of function: if 0, normal sad; if 1, avg; if 2, skip rows
%macro SAD_FN 4
%if %4 == 0 ; normal sad
%if %3 == 5
cglobal sad%1x%2, 4, %3, 5, src, src_stride, ref, ref_stride, n_rows
%else ; %3 == 7
cglobal sad%1x%2, 4, %3, 6, src, src_stride, ref, ref_stride, \
                            src_stride3, ref_stride3, n_rows
%endif ; %3 == 5/7

%elif %4 == 2 ; skip
%if %3 == 5
cglobal sad_skip_%1x%2, 4, %3, 5, src, src_stride, ref, ref_stride, n_rows
%else ; %3 == 7
cglobal sad_skip_%1x%2, 4, %3, 6, src, src_stride, ref, ref_stride, \
                            src_stride3, ref_stride3, n_rows
%endif ; %3 == 5/7

%else
%if %3 == 5
cglobal sad%1x%2_avg, 5, 1 + %3, 5, src, src_stride, ref, ref_stride, \
                                    second_pred, n_rows
%else ; %3 == 7
cglobal sad%1x%2_avg, 5, VPX_ARCH_X86_64 + %3, 6, src, src_stride, \
                                              ref, ref_stride, \
                                              second_pred, \
                                              src_stride3, ref_stride3
%if VPX_ARCH_X86_64
%define n_rowsd r7d
%else ; x86-32
%define n_rowsd dword r0m
%endif ; x86-32/64
%endif ; %3 == 5/7
%endif ; sad/avg/skip
%if %4 == 2; skip rows so double the stride
lea           src_strided, [src_strided*2]
lea           ref_strided, [ref_strided*2]
%endif ; %4 skip
  movsxdifnidn src_strideq, src_strided
  movsxdifnidn ref_strideq, ref_strided
%if %3 == 7
  lea         src_stride3q, [src_strideq*3]
  lea         ref_stride3q, [ref_strideq*3]
%endif ; %3 == 7
%endmacro

; unsigned int vpx_sad64x64_sse2(uint8_t *src, int src_stride,
;                                uint8_t *ref, int ref_stride);
%macro SAD64XN 1-2 0
  SAD_FN 64, %1, 5, %2
%if %2 == 2
  mov              n_rowsd, %1/2
%else
  mov              n_rowsd, %1
%endif
  pxor                  m0, m0
.loop:
  movu                  m1, [refq]
  movu                  m2, [refq+16]
  movu                  m3, [refq+32]
  movu                  m4, [refq+48]
%if %2 == 1
  pavgb                 m1, [second_predq+mmsize*0]
  pavgb                 m2, [second_predq+mmsize*1]
  pavgb                 m3, [second_predq+mmsize*2]
  pavgb                 m4, [second_predq+mmsize*3]
  lea         second_predq, [second_predq+mmsize*4]
%endif
  psadbw                m1, [srcq]
  psadbw                m2, [srcq+16]
  psadbw                m3, [srcq+32]
  psadbw                m4, [srcq+48]
  paddd                 m1, m2
  paddd                 m3, m4
  add                 refq, ref_strideq
  paddd                 m0, m1
  add                 srcq, src_strideq
  paddd                 m0, m3
  dec              n_rowsd
  jg .loop

  movhlps               m1, m0
  paddd                 m0, m1
%if %2 == 2 ; we skipped rows, so now we need to double the sad
  pslld                 m0, 1
%endif
  movd                 eax, m0
  RET
%endmacro

INIT_XMM sse2
SAD64XN 64 ; sad64x64_sse2
SAD64XN 32 ; sad64x32_sse2
SAD64XN 64, 1 ; sad64x64_avg_sse2
SAD64XN 32, 1 ; sad64x32_avg_sse2
SAD64XN  64, 2  ; sad64x64_skip_sse2
SAD64XN  32, 2  ; sad64x32_skip_sse2

; unsigned int vpx_sad32x32_sse2(uint8_t *src, int src_stride,
;                                uint8_t *ref, int ref_stride);
%macro SAD32XN 1-2 0
  SAD_FN 32, %1, 5, %2
%if %2 == 2
  mov              n_rowsd, %1/4
%else
  mov              n_rowsd, %1/2
%endif
  pxor                  m0, m0
.loop:
  movu                  m1, [refq]
  movu                  m2, [refq+16]
  movu                  m3, [refq+ref_strideq]
  movu                  m4, [refq+ref_strideq+16]
%if %2 == 1
  pavgb                 m1, [second_predq+mmsize*0]
  pavgb                 m2, [second_predq+mmsize*1]
  pavgb                 m3, [second_predq+mmsize*2]
  pavgb                 m4, [second_predq+mmsize*3]
  lea         second_predq, [second_predq+mmsize*4]
%endif
  psadbw                m1, [srcq]
  psadbw                m2, [srcq+16]
  psadbw                m3, [srcq+src_strideq]
  psadbw                m4, [srcq+src_strideq+16]
  paddd                 m1, m2
  paddd                 m3, m4
  lea                 refq, [refq+ref_strideq*2]
  paddd                 m0, m1
  lea                 srcq, [srcq+src_strideq*2]
  paddd                 m0, m3
  dec              n_rowsd
  jg .loop

  movhlps               m1, m0
  paddd                 m0, m1
%if %2 == 2 ; we skipped rows, so now we need to double the sad
  pslld                 m0, 1
%endif
  movd                 eax, m0
  RET
%endmacro

INIT_XMM sse2
SAD32XN 64 ; sad32x64_sse2
SAD32XN 32 ; sad32x32_sse2
SAD32XN 16 ; sad32x16_sse2
SAD32XN 64, 1 ; sad32x64_avg_sse2
SAD32XN 32, 1 ; sad32x32_avg_sse2
SAD32XN 16, 1 ; sad32x16_avg_sse2
SAD32XN 64, 2 ; sad32x64_skip_sse2
SAD32XN 32, 2 ; sad32x32_skip_sse2
SAD32XN 16, 2 ; sad32x16_skip_sse2

; unsigned int vpx_sad16x{8,16}_sse2(uint8_t *src, int src_stride,
;                                    uint8_t *ref, int ref_stride);
%macro SAD16XN 1-2 0
  SAD_FN 16, %1, 7, %2
%if %2 == 2
  mov              n_rowsd, %1/8
%else
  mov              n_rowsd, %1/4
%endif
  pxor                  m0, m0

.loop:
  movu                  m1, [refq]
  movu                  m2, [refq+ref_strideq]
  movu                  m3, [refq+ref_strideq*2]
  movu                  m4, [refq+ref_stride3q]
%if %2 == 1
  pavgb                 m1, [second_predq+mmsize*0]
  pavgb                 m2, [second_predq+mmsize*1]
  pavgb                 m3, [second_predq+mmsize*2]
  pavgb                 m4, [second_predq+mmsize*3]
  lea         second_predq, [second_predq+mmsize*4]
%endif
  psadbw                m1, [srcq]
  psadbw                m2, [srcq+src_strideq]
  psadbw                m3, [srcq+src_strideq*2]
  psadbw                m4, [srcq+src_stride3q]
  paddd                 m1, m2
  paddd                 m3, m4
  lea                 refq, [refq+ref_strideq*4]
  paddd                 m0, m1
  lea                 srcq, [srcq+src_strideq*4]
  paddd                 m0, m3
  dec              n_rowsd
  jg .loop

  movhlps               m1, m0
  paddd                 m0, m1
%if %2 == 2 ; we skipped rows, so now we need to double the sad
  pslld                 m0, 1
%endif
  movd                 eax, m0
  RET
%endmacro

INIT_XMM sse2
SAD16XN 32 ; sad16x32_sse2
SAD16XN 16 ; sad16x16_sse2
SAD16XN  8 ; sad16x8_sse2
SAD16XN 32, 1 ; sad16x32_avg_sse2
SAD16XN 16, 1 ; sad16x16_avg_sse2
SAD16XN  8, 1 ; sad16x8_avg_sse2
SAD16XN 32, 2 ; sad16x32_skip_sse2
SAD16XN 16, 2 ; sad16x16_skip_sse2
SAD16XN  8, 2 ; sad16x8_skip_sse2

; unsigned int vpx_sad8x{8,16}_sse2(uint8_t *src, int src_stride,
;                                   uint8_t *ref, int ref_stride);
%macro SAD8XN 1-2 0
  SAD_FN 8, %1, 7, %2
%if %2 == 2
  mov              n_rowsd, %1/8
%else
  mov              n_rowsd, %1/4
%endif
  pxor                  m0, m0

.loop:
  movh                  m1, [refq]
  movhps                m1, [refq+ref_strideq]
  movh                  m2, [refq+ref_strideq*2]
  movhps                m2, [refq+ref_stride3q]
%if %2 == 1
  pavgb                 m1, [second_predq+mmsize*0]
  pavgb                 m2, [second_predq+mmsize*1]
  lea         second_predq, [second_predq+mmsize*2]
%endif
  movh                  m3, [srcq]
  movhps                m3, [srcq+src_strideq]
  movh                  m4, [srcq+src_strideq*2]
  movhps                m4, [srcq+src_stride3q]
  psadbw                m1, m3
  psadbw                m2, m4
  lea                 refq, [refq+ref_strideq*4]
  paddd                 m0, m1
  lea                 srcq, [srcq+src_strideq*4]
  paddd                 m0, m2
  dec              n_rowsd
  jg .loop

  movhlps               m1, m0
  paddd                 m0, m1
%if %2 == 2 ; we skipped rows, so now we need to double the sad
  pslld                 m0, 1
%endif
  movd                 eax, m0
  RET
%endmacro

INIT_XMM sse2
SAD8XN 16 ; sad8x16_sse2
SAD8XN  8 ; sad8x8_sse2
SAD8XN  4 ; sad8x4_sse2
SAD8XN 16, 1 ; sad8x16_avg_sse2
SAD8XN  8, 1 ; sad8x8_avg_sse2
SAD8XN  4, 1 ; sad8x4_avg_sse2
SAD8XN 16, 2 ; sad8x16_skip_sse2
SAD8XN  8, 2 ; sad8x8_skip_sse2

; unsigned int vpx_sad4x{4, 8}_sse2(uint8_t *src, int src_stride,
;                                   uint8_t *ref, int ref_stride);
%macro SAD4XN 1-2 0
  SAD_FN 4, %1, 7, %2
%if %2 == 2
  mov              n_rowsd, %1/8
%else
  mov              n_rowsd, %1/4
%endif
  pxor                  m0, m0

.loop:
  movd                  m1, [refq]
  movd                  m2, [refq+ref_strideq]
  movd                  m3, [refq+ref_strideq*2]
  movd                  m4, [refq+ref_stride3q]
  punpckldq             m1, m2
  punpckldq             m3, m4
  movlhps               m1, m3
%if %2 == 1
  pavgb                 m1, [second_predq+mmsize*0]
  lea         second_predq, [second_predq+mmsize*1]
%endif
  movd                  m2, [srcq]
  movd                  m5, [srcq+src_strideq]
  movd                  m4, [srcq+src_strideq*2]
  movd                  m3, [srcq+src_stride3q]
  punpckldq             m2, m5
  punpckldq             m4, m3
  movlhps               m2, m4
  psadbw                m1, m2
  lea                 refq, [refq+ref_strideq*4]
  paddd                 m0, m1
  lea                 srcq, [srcq+src_strideq*4]
  dec              n_rowsd
  jg .loop

  movhlps               m1, m0
  paddd                 m0, m1
%if %2 == 2 ; we skipped rows, so now we need to double the sad
  pslld                 m0, 1
%endif
  movd                 eax, m0
  RET
%endmacro

INIT_XMM sse2
SAD4XN  8 ; sad4x8_sse
SAD4XN  4 ; sad4x4_sse
SAD4XN  8, 1 ; sad4x8_avg_sse
SAD4XN  4, 1 ; sad4x4_avg_sse
SAD4XN  8, 2 ; sad4x8_skip_sse
