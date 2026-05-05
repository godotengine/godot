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

SECTION_RODATA

pw_11585x2: times 8 dw 23170
pd_8192:    times 4 dd 8192

%macro TRANSFORM_COEFFS 2
pw_%1_%2:   dw  %1,  %2,  %1,  %2,  %1,  %2,  %1,  %2
pw_%2_m%1:  dw  %2, -%1,  %2, -%1,  %2, -%1,  %2, -%1
%endmacro

TRANSFORM_COEFFS 11585,  11585
TRANSFORM_COEFFS 15137,   6270
TRANSFORM_COEFFS 16069,   3196
TRANSFORM_COEFFS  9102,  13623

SECTION .text

%if VPX_ARCH_X86_64
INIT_XMM ssse3
cglobal fdct8x8, 3, 5, 13, input, output, stride

  mova               m8, [GLOBAL(pd_8192)]
  mova              m12, [GLOBAL(pw_11585x2)]

  lea                r3, [2 * strideq]
  lea                r4, [4 * strideq]
  mova               m0, [inputq]
  mova               m1, [inputq + r3]
  lea                inputq, [inputq + r4]
  mova               m2, [inputq]
  mova               m3, [inputq + r3]
  lea                inputq, [inputq + r4]
  mova               m4, [inputq]
  mova               m5, [inputq + r3]
  lea                inputq, [inputq + r4]
  mova               m6, [inputq]
  mova               m7, [inputq + r3]

  ; left shift by 2 to increase forward transformation precision
  psllw              m0, 2
  psllw              m1, 2
  psllw              m2, 2
  psllw              m3, 2
  psllw              m4, 2
  psllw              m5, 2
  psllw              m6, 2
  psllw              m7, 2

  ; column transform
  ; stage 1
  paddw m10, m0, m7
  psubw m0, m7

  paddw m9, m1, m6
  psubw m1, m6

  paddw m7, m2, m5
  psubw m2, m5

  paddw m6, m3, m4
  psubw m3, m4

  ; stage 2
  paddw m5, m9, m7
  psubw m9, m7

  paddw m4, m10, m6
  psubw m10, m6

  paddw m7, m1, m2
  psubw m1, m2

  ; stage 3
  paddw m6, m4, m5
  psubw m4, m5

  pmulhrsw m1, m12
  pmulhrsw m7, m12

  ; sin(pi / 8), cos(pi / 8)
  punpcklwd m2, m10, m9
  punpckhwd m10, m9
  pmaddwd m5, m2, [GLOBAL(pw_15137_6270)]
  pmaddwd m2, [GLOBAL(pw_6270_m15137)]
  pmaddwd m9, m10, [GLOBAL(pw_15137_6270)]
  pmaddwd m10, [GLOBAL(pw_6270_m15137)]
  paddd m5, m8
  paddd m2, m8
  paddd m9, m8
  paddd m10, m8
  psrad m5, 14
  psrad m2, 14
  psrad m9, 14
  psrad m10, 14
  packssdw m5, m9
  packssdw m2, m10

  pmulhrsw m6, m12
  pmulhrsw m4, m12

  paddw m9, m3, m1
  psubw m3, m1

  paddw m10, m0, m7
  psubw m0, m7

  ; stage 4
  ; sin(pi / 16), cos(pi / 16)
  punpcklwd m1, m10, m9
  punpckhwd m10, m9
  pmaddwd m7, m1, [GLOBAL(pw_16069_3196)]
  pmaddwd m1, [GLOBAL(pw_3196_m16069)]
  pmaddwd m9, m10, [GLOBAL(pw_16069_3196)]
  pmaddwd m10, [GLOBAL(pw_3196_m16069)]
  paddd m7, m8
  paddd m1, m8
  paddd m9, m8
  paddd m10, m8
  psrad m7, 14
  psrad m1, 14
  psrad m9, 14
  psrad m10, 14
  packssdw m7, m9
  packssdw m1, m10

  ; sin(3 * pi / 16), cos(3 * pi / 16)
  punpcklwd m11, m0, m3
  punpckhwd m0, m3
  pmaddwd m9, m11, [GLOBAL(pw_9102_13623)]
  pmaddwd m11, [GLOBAL(pw_13623_m9102)]
  pmaddwd m3, m0, [GLOBAL(pw_9102_13623)]
  pmaddwd m0, [GLOBAL(pw_13623_m9102)]
  paddd m9, m8
  paddd m11, m8
  paddd m3, m8
  paddd m0, m8
  psrad m9, 14
  psrad m11, 14
  psrad m3, 14
  psrad m0, 14
  packssdw m9, m3
  packssdw m11, m0

  ; transpose
  ; stage 1
  punpcklwd m0, m6, m7
  punpcklwd m3, m5, m11
  punpckhwd m6, m7
  punpckhwd m5, m11
  punpcklwd m7, m4, m9
  punpcklwd m10, m2, m1
  punpckhwd m4, m9
  punpckhwd m2, m1

  ; stage 2
  punpckldq m9, m0, m3
  punpckldq m1, m6, m5
  punpckhdq m0, m3
  punpckhdq m6, m5
  punpckldq m3, m7, m10
  punpckldq m5, m4, m2
  punpckhdq m7, m10
  punpckhdq m4, m2

  ; stage 3
  punpcklqdq m10, m9, m3
  punpckhqdq m9, m3
  punpcklqdq m2, m0, m7
  punpckhqdq m0, m7
  punpcklqdq m3, m1, m5
  punpckhqdq m1, m5
  punpcklqdq m7, m6, m4
  punpckhqdq m6, m4

  ; row transform
  ; stage 1
  paddw m5, m10, m6
  psubw m10, m6

  paddw m4, m9, m7
  psubw m9, m7

  paddw m6, m2, m1
  psubw m2, m1

  paddw m7, m0, m3
  psubw m0, m3

  ;stage 2
  paddw m1, m5, m7
  psubw m5, m7

  paddw m3, m4, m6
  psubw m4, m6

  paddw m7, m9, m2
  psubw m9, m2

  ; stage 3
  punpcklwd m6, m1, m3
  punpckhwd m1, m3
  pmaddwd m2, m6, [GLOBAL(pw_11585_11585)]
  pmaddwd m6, [GLOBAL(pw_11585_m11585)]
  pmaddwd m3, m1, [GLOBAL(pw_11585_11585)]
  pmaddwd m1, [GLOBAL(pw_11585_m11585)]
  paddd m2, m8
  paddd m6, m8
  paddd m3, m8
  paddd m1, m8
  psrad m2, 14
  psrad m6, 14
  psrad m3, 14
  psrad m1, 14
  packssdw m2, m3
  packssdw m6, m1

  pmulhrsw m7, m12
  pmulhrsw m9, m12

  punpcklwd m3, m5, m4
  punpckhwd m5, m4
  pmaddwd m1, m3, [GLOBAL(pw_15137_6270)]
  pmaddwd m3, [GLOBAL(pw_6270_m15137)]
  pmaddwd m4, m5, [GLOBAL(pw_15137_6270)]
  pmaddwd m5, [GLOBAL(pw_6270_m15137)]
  paddd m1, m8
  paddd m3, m8
  paddd m4, m8
  paddd m5, m8
  psrad m1, 14
  psrad m3, 14
  psrad m4, 14
  psrad m5, 14
  packssdw m1, m4
  packssdw m3, m5

  paddw m4, m0, m9
  psubw m0, m9

  paddw m5, m10, m7
  psubw m10, m7

  ; stage 4
  punpcklwd m9, m5, m4
  punpckhwd m5, m4
  pmaddwd m7, m9, [GLOBAL(pw_16069_3196)]
  pmaddwd m9, [GLOBAL(pw_3196_m16069)]
  pmaddwd m4, m5, [GLOBAL(pw_16069_3196)]
  pmaddwd m5, [GLOBAL(pw_3196_m16069)]
  paddd m7, m8
  paddd m9, m8
  paddd m4, m8
  paddd m5, m8
  psrad m7, 14
  psrad m9, 14
  psrad m4, 14
  psrad m5, 14
  packssdw m7, m4
  packssdw m9, m5

  punpcklwd m4, m10, m0
  punpckhwd m10, m0
  pmaddwd m5, m4, [GLOBAL(pw_9102_13623)]
  pmaddwd m4, [GLOBAL(pw_13623_m9102)]
  pmaddwd m0, m10, [GLOBAL(pw_9102_13623)]
  pmaddwd m10, [GLOBAL(pw_13623_m9102)]
  paddd m5, m8
  paddd m4, m8
  paddd m0, m8
  paddd m10, m8
  psrad m5, 14
  psrad m4, 14
  psrad m0, 14
  psrad m10, 14
  packssdw m5, m0
  packssdw m4, m10

  ; transpose
  ; stage 1
  punpcklwd m0, m2, m7
  punpcklwd m10, m1, m4
  punpckhwd m2, m7
  punpckhwd m1, m4
  punpcklwd m7, m6, m5
  punpcklwd m4, m3, m9
  punpckhwd m6, m5
  punpckhwd m3, m9

  ; stage 2
  punpckldq m5, m0, m10
  punpckldq m9, m2, m1
  punpckhdq m0, m10
  punpckhdq m2, m1
  punpckldq m10, m7, m4
  punpckldq m1, m6, m3
  punpckhdq m7, m4
  punpckhdq m6, m3

  ; stage 3
  punpcklqdq m4, m5, m10
  punpckhqdq m5, m10
  punpcklqdq m3, m0, m7
  punpckhqdq m0, m7
  punpcklqdq m10, m9, m1
  punpckhqdq m9, m1
  punpcklqdq m7, m2, m6
  punpckhqdq m2, m6

  psraw m1, m4, 15
  psraw m6, m5, 15
  psraw m8, m3, 15
  psraw m11, m0, 15

  psubw m4, m1
  psubw m5, m6
  psubw m3, m8
  psubw m0, m11

  psraw m4, 1
  psraw m5, 1
  psraw m3, 1
  psraw m0, 1

  psraw m1, m10, 15
  psraw m6, m9, 15
  psraw m8, m7, 15
  psraw m11, m2, 15

  psubw m10, m1
  psubw m9, m6
  psubw m7, m8
  psubw m2, m11

  psraw m10, 1
  psraw m9, 1
  psraw m7, 1
  psraw m2, 1

  mova              [outputq +   0], m4
  mova              [outputq +  16], m5
  mova              [outputq +  32], m3
  mova              [outputq +  48], m0
  mova              [outputq +  64], m10
  mova              [outputq +  80], m9
  mova              [outputq +  96], m7
  mova              [outputq + 112], m2

  RET
%endif
