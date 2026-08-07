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

; PROCESS_4x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro PROCESS_4x2x4 5-6 0
  movd                  m0, [srcq +%2]
%if %1 == 1
  movd                  m6, [ref1q+%3]
  movd                  m4, [ref2q+%3]
  movd                  m7, [ref3q+%3]
  movd                  m5, [ref4q+%3]
  movd                  m1, [srcq +%4]
  movd                  m2, [ref1q+%5]
  punpckldq             m0, m1
  punpckldq             m6, m2
  movd                  m1, [ref2q+%5]
  movd                  m2, [ref3q+%5]
  movd                  m3, [ref4q+%5]
  punpckldq             m4, m1
  punpckldq             m7, m2
  punpckldq             m5, m3
  movlhps               m0, m0
  movlhps               m6, m4
  movlhps               m7, m5
  psadbw                m6, m0
  psadbw                m7, m0
%else
  movd                  m1, [ref1q+%3]
  movd                  m5, [ref1q+%5]
  movd                  m2, [ref2q+%3]
  movd                  m4, [ref2q+%5]
  punpckldq             m1, m5
  punpckldq             m2, m4
  movd                  m3, [ref3q+%3]
  movd                  m5, [ref3q+%5]
  punpckldq             m3, m5
  movd                  m4, [ref4q+%3]
  movd                  m5, [ref4q+%5]
  punpckldq             m4, m5
  movd                  m5, [srcq +%4]
  punpckldq             m0, m5
  movlhps               m0, m0
  movlhps               m1, m2
  movlhps               m3, m4
  psadbw                m1, m0
  psadbw                m3, m0
  paddd                 m6, m1
  paddd                 m7, m3
%endif
%if %6 == 1
  lea                 srcq, [srcq +src_strideq*2]
  lea                ref1q, [ref1q+ref_strideq*2]
  lea                ref2q, [ref2q+ref_strideq*2]
  lea                ref3q, [ref3q+ref_strideq*2]
  lea                ref4q, [ref4q+ref_strideq*2]
%endif
%endmacro

; PROCESS_8x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro PROCESS_8x2x4 5-6 0
  movh                  m0, [srcq +%2]
%if %1 == 1
  movh                  m4, [ref1q+%3]
  movh                  m5, [ref2q+%3]
  movh                  m6, [ref3q+%3]
  movh                  m7, [ref4q+%3]
  movhps                m0, [srcq +%4]
  movhps                m4, [ref1q+%5]
  movhps                m5, [ref2q+%5]
  movhps                m6, [ref3q+%5]
  movhps                m7, [ref4q+%5]
  psadbw                m4, m0
  psadbw                m5, m0
  psadbw                m6, m0
  psadbw                m7, m0
%else
  movh                  m1, [ref1q+%3]
  movh                  m2, [ref2q+%3]
  movh                  m3, [ref3q+%3]
  movhps                m0, [srcq +%4]
  movhps                m1, [ref1q+%5]
  movhps                m2, [ref2q+%5]
  movhps                m3, [ref3q+%5]
  psadbw                m1, m0
  psadbw                m2, m0
  psadbw                m3, m0
  paddd                 m4, m1
  movh                  m1, [ref4q+%3]
  movhps                m1, [ref4q+%5]
  paddd                 m5, m2
  paddd                 m6, m3
  psadbw                m1, m0
  paddd                 m7, m1
%endif
%if %6 == 1
  lea                 srcq, [srcq +src_strideq*2]
  lea                ref1q, [ref1q+ref_strideq*2]
  lea                ref2q, [ref2q+ref_strideq*2]
  lea                ref3q, [ref3q+ref_strideq*2]
  lea                ref4q, [ref4q+ref_strideq*2]
%endif
%endmacro

; PROCESS_16x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro PROCESS_16x2x4 5-6 0
  ; 1st 16 px
  mova                  m0, [srcq +%2]
%if %1 == 1
  movu                  m4, [ref1q+%3]
  movu                  m5, [ref2q+%3]
  movu                  m6, [ref3q+%3]
  movu                  m7, [ref4q+%3]
  psadbw                m4, m0
  psadbw                m5, m0
  psadbw                m6, m0
  psadbw                m7, m0
%else
  movu                  m1, [ref1q+%3]
  movu                  m2, [ref2q+%3]
  movu                  m3, [ref3q+%3]
  psadbw                m1, m0
  psadbw                m2, m0
  psadbw                m3, m0
  paddd                 m4, m1
  movu                  m1, [ref4q+%3]
  paddd                 m5, m2
  paddd                 m6, m3
  psadbw                m1, m0
  paddd                 m7, m1
%endif

  ; 2nd 16 px
  mova                  m0, [srcq +%4]
  movu                  m1, [ref1q+%5]
  movu                  m2, [ref2q+%5]
  movu                  m3, [ref3q+%5]
  psadbw                m1, m0
  psadbw                m2, m0
  psadbw                m3, m0
  paddd                 m4, m1
  movu                  m1, [ref4q+%5]
  paddd                 m5, m2
  paddd                 m6, m3
%if %6 == 1
  lea                 srcq, [srcq +src_strideq*2]
  lea                ref1q, [ref1q+ref_strideq*2]
  lea                ref2q, [ref2q+ref_strideq*2]
  lea                ref3q, [ref3q+ref_strideq*2]
  lea                ref4q, [ref4q+ref_strideq*2]
%endif
  psadbw                m1, m0
  paddd                 m7, m1
%endmacro

; PROCESS_32x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro PROCESS_32x2x4 5-6 0
  PROCESS_16x2x4 %1, %2, %3, %2 + 16, %3 + 16
  PROCESS_16x2x4  0, %4, %5, %4 + 16, %5 + 16, %6
%endmacro

; PROCESS_64x2x4 first, off_{first,second}_{src,ref}, advance_at_end
%macro PROCESS_64x2x4 5-6 0
  PROCESS_32x2x4 %1, %2, %3, %2 + 32, %3 + 32
  PROCESS_32x2x4  0, %4, %5, %4 + 32, %5 + 32, %6
%endmacro

; void vpx_sadNxNx4d_sse2(uint8_t *src,    int src_stride,
;                         uint8_t *ref[4], int ref_stride,
;                         uint32_t res[4]);
; where NxN = 64x64, 32x32, 16x16, 16x8, 8x16, 8x8, 8x4, 4x8 and 4x4
%macro SADNXN4D 2-3 0
%if %3 == 1  ; skip rows
%if UNIX64
cglobal sad_skip_%1x%2x4d, 5, 8, 8, src, src_stride, ref1, ref_stride, \
                              res, ref2, ref3, ref4
%else
cglobal sad_skip_%1x%2x4d, 4, 7, 8, src, src_stride, ref1, ref_stride, \
                              ref2, ref3, ref4
%endif
%else  ; normal sad
%if UNIX64
cglobal sad%1x%2x4d, 5, 8, 8, src, src_stride, ref1, ref_stride, \
                              res, ref2, ref3, ref4
%else
cglobal sad%1x%2x4d, 4, 7, 8, src, src_stride, ref1, ref_stride, \
                              ref2, ref3, ref4
%endif
%endif
%if %3 == 1
  lea          src_strided, [2*src_strided]
  lea          ref_strided, [2*ref_strided]
%endif
  movsxdifnidn src_strideq, src_strided
  movsxdifnidn ref_strideq, ref_strided
  mov                ref2q, [ref1q+gprsize*1]
  mov                ref3q, [ref1q+gprsize*2]
  mov                ref4q, [ref1q+gprsize*3]
  mov                ref1q, [ref1q+gprsize*0]

  PROCESS_%1x2x4 1, 0, 0, src_strideq, ref_strideq, 1
%if %3 == 1  ; downsample number of rows by 2
%define num_rep (%2-8)/4
%else
%define num_rep (%2-4)/2
%endif
%rep num_rep
  PROCESS_%1x2x4 0, 0, 0, src_strideq, ref_strideq, 1
%endrep
%undef num_rep
  PROCESS_%1x2x4 0, 0, 0, src_strideq, ref_strideq, 0

%if %1 > 4
  pslldq                m5, 4
  pslldq                m7, 4
  por                   m4, m5
  por                   m6, m7
  mova                  m5, m4
  mova                  m7, m6
  punpcklqdq            m4, m6
  punpckhqdq            m5, m7
  movifnidn             r4, r4mp
  paddd                 m4, m5
%if %3 == 1
  pslld                 m4, 1
%endif
  movu                [r4], m4
  RET
%else
  movifnidn             r4, r4mp
  pshufd            m6, m6, 0x08
  pshufd            m7, m7, 0x08
%if %3 == 1
  pslld                 m6, 1
  pslld                 m7, 1
%endif
  movq              [r4+0], m6
  movq              [r4+8], m7
  RET
%endif
%endmacro

INIT_XMM sse2
SADNXN4D 64, 64
SADNXN4D 64, 32
SADNXN4D 32, 64
SADNXN4D 32, 32
SADNXN4D 32, 16
SADNXN4D 16, 32
SADNXN4D 16, 16
SADNXN4D 16,  8
SADNXN4D  8, 16
SADNXN4D  8,  8
SADNXN4D  8,  4
SADNXN4D  4,  8
SADNXN4D  4,  4

SADNXN4D 64, 64, 1
SADNXN4D 64, 32, 1
SADNXN4D 32, 64, 1
SADNXN4D 32, 32, 1
SADNXN4D 32, 16, 1
SADNXN4D 16, 32, 1
SADNXN4D 16, 16, 1
SADNXN4D 16,  8, 1
SADNXN4D  8, 16, 1
SADNXN4D  8,  8, 1
SADNXN4D  4,  8, 1
