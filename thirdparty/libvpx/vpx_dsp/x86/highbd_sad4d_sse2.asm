;
;  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

%include "third_party/x86inc/x86inc.asm"

SECTION .text

; HIGH_PROCESS_4x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro HIGH_PROCESS_4x2x4 5-6 0
  movh                  m0, [srcq +%2*2]
%if %1 == 1
  movu                  m4, [ref1q+%3*2]
  movu                  m5, [ref2q+%3*2]
  movu                  m6, [ref3q+%3*2]
  movu                  m7, [ref4q+%3*2]
  movhps                m0, [srcq +%4*2]
  movhps                m4, [ref1q+%5*2]
  movhps                m5, [ref2q+%5*2]
  movhps                m6, [ref3q+%5*2]
  movhps                m7, [ref4q+%5*2]
  mova                  m3, m0
  mova                  m2, m0
  psubusw               m3, m4
  psubusw               m2, m5
  psubusw               m4, m0
  psubusw               m5, m0
  por                   m4, m3
  por                   m5, m2
  pmaddwd               m4, m1
  pmaddwd               m5, m1
  mova                  m3, m0
  mova                  m2, m0
  psubusw               m3, m6
  psubusw               m2, m7
  psubusw               m6, m0
  psubusw               m7, m0
  por                   m6, m3
  por                   m7, m2
  pmaddwd               m6, m1
  pmaddwd               m7, m1
%else
  movu                  m2, [ref1q+%3*2]
  movhps                m0, [srcq +%4*2]
  movhps                m2, [ref1q+%5*2]
  mova                  m3, m0
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  pmaddwd               m2, m1
  paddd                 m4, m2

  movu                  m2, [ref2q+%3*2]
  mova                  m3, m0
  movhps                m2, [ref2q+%5*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  pmaddwd               m2, m1
  paddd                 m5, m2

  movu                  m2, [ref3q+%3*2]
  mova                  m3, m0
  movhps                m2, [ref3q+%5*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  pmaddwd               m2, m1
  paddd                 m6, m2

  movu                  m2, [ref4q+%3*2]
  mova                  m3, m0
  movhps                m2, [ref4q+%5*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  pmaddwd               m2, m1
  paddd                 m7, m2
%endif
%if %6 == 1
  lea                 srcq, [srcq +src_strideq*4]
  lea                ref1q, [ref1q+ref_strideq*4]
  lea                ref2q, [ref2q+ref_strideq*4]
  lea                ref3q, [ref3q+ref_strideq*4]
  lea                ref4q, [ref4q+ref_strideq*4]
%endif
%endmacro

; PROCESS_8x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro HIGH_PROCESS_8x2x4 5-6 0
  ; 1st 8 px
  mova                  m0, [srcq +%2*2]
%if %1 == 1
  movu                  m4, [ref1q+%3*2]
  movu                  m5, [ref2q+%3*2]
  movu                  m6, [ref3q+%3*2]
  movu                  m7, [ref4q+%3*2]
  mova                  m3, m0
  mova                  m2, m0
  psubusw               m3, m4
  psubusw               m2, m5
  psubusw               m4, m0
  psubusw               m5, m0
  por                   m4, m3
  por                   m5, m2
  pmaddwd               m4, m1
  pmaddwd               m5, m1
  mova                  m3, m0
  mova                  m2, m0
  psubusw               m3, m6
  psubusw               m2, m7
  psubusw               m6, m0
  psubusw               m7, m0
  por                   m6, m3
  por                   m7, m2
  pmaddwd               m6, m1
  pmaddwd               m7, m1
%else
  mova                  m3, m0
  movu                  m2, [ref1q+%3*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  mova                  m3, m0
  pmaddwd               m2, m1
  paddd                 m4, m2
  movu                  m2, [ref2q+%3*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  mova                  m3, m0
  pmaddwd               m2, m1
  paddd                 m5, m2
  movu                  m2, [ref3q+%3*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  mova                  m3, m0
  pmaddwd               m2, m1
  paddd                 m6, m2
  movu                  m2, [ref4q+%3*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  pmaddwd               m2, m1
  paddd                 m7, m2
%endif

  ; 2nd 8 px
  mova                  m0, [srcq +(%4)*2]
  mova                  m3, m0
  movu                  m2, [ref1q+(%5)*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  mova                  m3, m0
  pmaddwd               m2, m1
  paddd                 m4, m2
  movu                  m2, [ref2q+(%5)*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  mova                  m3, m0
  pmaddwd               m2, m1
  paddd                 m5, m2
  movu                  m2, [ref3q+(%5)*2]
  psubusw               m3, m2
  psubusw               m2, m0
  por                   m2, m3
  mova                  m3, m0
  pmaddwd               m2, m1
  paddd                 m6, m2
  movu                  m2, [ref4q+(%5)*2]
  psubusw               m3, m2
  psubusw               m2, m0
%if %6 == 1
  lea                 srcq, [srcq +src_strideq*4]
  lea                ref1q, [ref1q+ref_strideq*4]
  lea                ref2q, [ref2q+ref_strideq*4]
  lea                ref3q, [ref3q+ref_strideq*4]
  lea                ref4q, [ref4q+ref_strideq*4]
%endif
  por                   m2, m3
  pmaddwd               m2, m1
  paddd                 m7, m2
%endmacro

; HIGH_PROCESS_16x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro HIGH_PROCESS_16x2x4 5-6 0
  HIGH_PROCESS_8x2x4 %1, %2, %3, (%2 + 8), (%3 + 8)
  HIGH_PROCESS_8x2x4  0, %4, %5, (%4 + 8), (%5 + 8), %6
%endmacro

; HIGH_PROCESS_32x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro HIGH_PROCESS_32x2x4 5-6 0
  HIGH_PROCESS_16x2x4 %1, %2, %3, (%2 + 16), (%3 + 16)
  HIGH_PROCESS_16x2x4  0, %4, %5, (%4 + 16), (%5 + 16), %6
%endmacro

; HIGH_PROCESS_64x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro HIGH_PROCESS_64x2x4 5-6 0
  HIGH_PROCESS_32x2x4 %1, %2, %3, (%2 + 32), (%3 + 32)
  HIGH_PROCESS_32x2x4  0, %4, %5, (%4 + 32), (%5 + 32), %6
%endmacro

; void vpx_highbd_sadNxNx4d_sse2(uint8_t *src,    int src_stride,
;                         uint8_t *ref[4], int ref_stride,
;                         uint32_t res[4]);
; where NxN = 64x64, 32x32, 16x16, 16x8, 8x16 or 8x8
; Macro Arguments:
;   1: Width
;   2: Height
;   3: If 0, then normal sad, if 2, then skip every other row
%macro HIGH_SADNXN4D 2-3 0
%if %3 == 0  ; normal sad
%if UNIX64
cglobal highbd_sad%1x%2x4d, 5, 8, 8, src, src_stride, ref1, ref_stride, \
                              res, ref2, ref3, ref4
%else
cglobal highbd_sad%1x%2x4d, 4, 7, 8, src, src_stride, ref1, ref_stride, \
                              ref2, ref3, ref4
%endif
%else  ; %3 == 2, downsample
%if UNIX64
cglobal highbd_sad_skip_%1x%2x4d, 5, 8, 8, src, src_stride, ref1, ref_stride, \
                              res, ref2, ref3, ref4
%else
cglobal highbd_sad_skip_%1x%2x4d, 4, 7, 8, src, src_stride, ref1, ref_stride, \
                              ref2, ref3, ref4
%endif  ;
%endif  ; sad/avg/skip

; set m1
  push                srcq
  mov                 srcd, 0x00010001
  movd                  m1, srcd
  pshufd                m1, m1, 0x0
  pop                 srcq

%if %3 == 2  ; skip rows
  lea          src_strided, [2*src_strided]
  lea          ref_strided, [2*ref_strided]
%endif  ; skip rows
  movsxdifnidn src_strideq, src_strided
  movsxdifnidn ref_strideq, ref_strided
  mov                ref2q, [ref1q+gprsize*1]
  mov                ref3q, [ref1q+gprsize*2]
  mov                ref4q, [ref1q+gprsize*3]
  mov                ref1q, [ref1q+gprsize*0]

; convert byte pointers to short pointers
  shl                 srcq, 1
  shl                ref2q, 1
  shl                ref3q, 1
  shl                ref4q, 1
  shl                ref1q, 1

  HIGH_PROCESS_%1x2x4 1, 0, 0, src_strideq, ref_strideq, 1
%if %3 == 2  ;  Downsampling by two
%define num_rep (%2-8)/4
%else
%define num_rep (%2-4)/2
%endif
%rep num_rep
  HIGH_PROCESS_%1x2x4 0, 0, 0, src_strideq, ref_strideq, 1
%endrep
%undef rep
  HIGH_PROCESS_%1x2x4 0, 0, 0, src_strideq, ref_strideq, 0
  ; N.B. HIGH_PROCESS outputs dwords (32 bits)
  ; so in high bit depth even the smallest width (4) needs 128bits i.e. XMM
  movhlps               m0, m4
  movhlps               m1, m5
  movhlps               m2, m6
  movhlps               m3, m7
  paddd                 m4, m0
  paddd                 m5, m1
  paddd                 m6, m2
  paddd                 m7, m3
  punpckldq             m4, m5
  punpckldq             m6, m7
  movhlps               m0, m4
  movhlps               m1, m6
  paddd                 m4, m0
  paddd                 m6, m1
  punpcklqdq            m4, m6
%if %3 == 2  ; skip rows
  pslld                 m4, 1
%endif
  movifnidn             r4, r4mp
  movu                [r4], m4
  RET
%endmacro


INIT_XMM sse2
HIGH_SADNXN4D 64, 64
HIGH_SADNXN4D 64, 32
HIGH_SADNXN4D 32, 64
HIGH_SADNXN4D 32, 32
HIGH_SADNXN4D 32, 16
HIGH_SADNXN4D 16, 32
HIGH_SADNXN4D 16, 16
HIGH_SADNXN4D 16,  8
HIGH_SADNXN4D  8, 16
HIGH_SADNXN4D  8,  8
HIGH_SADNXN4D  8,  4
HIGH_SADNXN4D  4,  8
HIGH_SADNXN4D  4,  4

HIGH_SADNXN4D 64, 64, 2
HIGH_SADNXN4D 64, 32, 2
HIGH_SADNXN4D 32, 64, 2
HIGH_SADNXN4D 32, 32, 2
HIGH_SADNXN4D 32, 16, 2
HIGH_SADNXN4D 16, 32, 2
HIGH_SADNXN4D 16, 16, 2
HIGH_SADNXN4D 16,  8, 2
HIGH_SADNXN4D  8, 16, 2
HIGH_SADNXN4D  8,  8, 2
HIGH_SADNXN4D  4,  8, 2
