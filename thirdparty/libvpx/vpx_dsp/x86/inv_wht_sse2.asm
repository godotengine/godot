;
;  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

%include "third_party/x86inc/x86inc.asm"

SECTION .text

%macro REORDER_INPUTS 0
  ; a c d b  to  a b c d
  SWAP 1, 3, 2
%endmacro

%macro TRANSFORM_COLS 0
  ; input:
  ; m0 a
  ; m1 b
  ; m2 c
  ; m3 d
  paddw           m0,        m2
  psubw           m3,        m1

  ; wide subtract
  punpcklwd       m4,        m0
  punpcklwd       m5,        m3
  psrad           m4,        16
  psrad           m5,        16
  psubd           m4,        m5
  psrad           m4,        1
  packssdw        m4,        m4             ; e

  psubw           m5,        m4,        m1  ; b
  psubw           m4,        m2             ; c
  psubw           m0,        m5
  paddw           m3,        m4
                                ; m0 a
  SWAP            1,         5  ; m1 b
  SWAP            2,         4  ; m2 c
                                ; m3 d
%endmacro

%macro TRANSPOSE_4X4 0
  punpcklwd       m0,        m2
  punpcklwd       m1,        m3
  mova            m2,        m0
  punpcklwd       m0,        m1
  punpckhwd       m2,        m1
  pshufd          m1,        m0, 0x0e
  pshufd          m3,        m2, 0x0e
%endmacro

; transpose a 4x4 int16 matrix in xmm0 and xmm1 to the bottom half of xmm0-xmm3
%macro TRANSPOSE_4X4_WIDE 0
  mova            m3, m0
  punpcklwd       m0, m1
  punpckhwd       m3, m1
  mova            m2, m0
  punpcklwd       m0, m3
  punpckhwd       m2, m3
  pshufd          m1, m0, 0x0e
  pshufd          m3, m2, 0x0e
%endmacro

%macro ADD_STORE_4P_2X 5  ; src1, src2, tmp1, tmp2, zero
  movd            m%3,       [outputq]
  movd            m%4,       [outputq + strideq]
  punpcklbw       m%3,       m%5
  punpcklbw       m%4,       m%5
  paddw           m%1,       m%3
  paddw           m%2,       m%4
  packuswb        m%1,       m%5
  packuswb        m%2,       m%5
  movd            [outputq], m%1
  movd            [outputq + strideq], m%2
%endmacro

INIT_XMM sse2
cglobal iwht4x4_16_add, 3, 3, 7, input, output, stride
%if CONFIG_VP9_HIGHBITDEPTH
  mova            m0,        [inputq +  0]
  packssdw        m0,        [inputq + 16]
  mova            m1,        [inputq + 32]
  packssdw        m1,        [inputq + 48]
%else
  mova            m0,        [inputq +  0]
  mova            m1,        [inputq + 16]
%endif
  psraw           m0,        2
  psraw           m1,        2

  TRANSPOSE_4X4_WIDE
  REORDER_INPUTS
  TRANSFORM_COLS
  TRANSPOSE_4X4
  REORDER_INPUTS
  TRANSFORM_COLS

  pxor            m4, m4
  ADD_STORE_4P_2X  0, 1, 5, 6, 4
  lea             outputq, [outputq + 2 * strideq]
  ADD_STORE_4P_2X  2, 3, 5, 6, 4

  RET
