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

; void vpx_subtract_block(int rows, int cols,
;                         int16_t *diff, ptrdiff_t diff_stride,
;                         const uint8_t *src, ptrdiff_t src_stride,
;                         const uint8_t *pred, ptrdiff_t pred_stride)

INIT_XMM sse2
cglobal subtract_block, 7, 7, 8, \
                        rows, cols, diff, diff_stride, src, src_stride, \
                        pred, pred_stride
%define pred_str colsq
  pxor                  m7, m7         ; dedicated zero register
  cmp                colsd, 4
  je .case_4
  cmp                colsd, 8
  je .case_8
  cmp                colsd, 16
  je .case_16
  cmp                colsd, 32
  je .case_32

%macro loop16 6
  mova                  m0, [srcq+%1]
  mova                  m4, [srcq+%2]
  mova                  m1, [predq+%3]
  mova                  m5, [predq+%4]
  punpckhbw             m2, m0, m7
  punpckhbw             m3, m1, m7
  punpcklbw             m0, m7
  punpcklbw             m1, m7
  psubw                 m2, m3
  psubw                 m0, m1
  punpckhbw             m1, m4, m7
  punpckhbw             m3, m5, m7
  punpcklbw             m4, m7
  punpcklbw             m5, m7
  psubw                 m1, m3
  psubw                 m4, m5
  mova [diffq+mmsize*0+%5], m0
  mova [diffq+mmsize*1+%5], m2
  mova [diffq+mmsize*0+%6], m4
  mova [diffq+mmsize*1+%6], m1
%endmacro

  mov             pred_str, pred_stridemp
.loop_64:
  loop16 0*mmsize, 1*mmsize, 0*mmsize, 1*mmsize, 0*mmsize, 2*mmsize
  loop16 2*mmsize, 3*mmsize, 2*mmsize, 3*mmsize, 4*mmsize, 6*mmsize
  lea                diffq, [diffq+diff_strideq*2]
  add                predq, pred_str
  add                 srcq, src_strideq
  dec                rowsd
  jg .loop_64
  RET

.case_32:
  mov             pred_str, pred_stridemp
.loop_32:
  loop16 0, mmsize, 0, mmsize, 0, 2*mmsize
  lea                diffq, [diffq+diff_strideq*2]
  add                predq, pred_str
  add                 srcq, src_strideq
  dec                rowsd
  jg .loop_32
  RET

.case_16:
  mov             pred_str, pred_stridemp
.loop_16:
  loop16 0, src_strideq, 0, pred_str, 0, diff_strideq*2
  lea                diffq, [diffq+diff_strideq*4]
  lea                predq, [predq+pred_str*2]
  lea                 srcq, [srcq+src_strideq*2]
  sub                rowsd, 2
  jg .loop_16
  RET

%macro loop_h 0
  movh                  m0, [srcq]
  movh                  m2, [srcq+src_strideq]
  movh                  m1, [predq]
  movh                  m3, [predq+pred_str]
  punpcklbw             m0, m7
  punpcklbw             m1, m7
  punpcklbw             m2, m7
  punpcklbw             m3, m7
  psubw                 m0, m1
  psubw                 m2, m3
  mova             [diffq], m0
  mova [diffq+diff_strideq*2], m2
%endmacro

.case_8:
  mov             pred_str, pred_stridemp
.loop_8:
  loop_h
  lea                diffq, [diffq+diff_strideq*4]
  lea                 srcq, [srcq+src_strideq*2]
  lea                predq, [predq+pred_str*2]
  sub                rowsd, 2
  jg .loop_8
  RET

INIT_MMX
.case_4:
  mov             pred_str, pred_stridemp
.loop_4:
  loop_h
  lea                diffq, [diffq+diff_strideq*4]
  lea                 srcq, [srcq+src_strideq*2]
  lea                predq, [predq+pred_str*2]
  sub                rowsd, 2
  jg .loop_4
  emms
  RET
