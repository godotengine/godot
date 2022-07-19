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

%macro convolve_fn 1-2
%ifidn %1, avg
%define AUX_XMM_REGS 4
%else
%define AUX_XMM_REGS 0
%endif
%ifidn %2, highbd
%define pavg pavgw
cglobal %2_convolve_%1, 4, 7, 4+AUX_XMM_REGS, src, src_stride, \
                                              dst, dst_stride, \
                                              fx, fxs, fy, fys, w, h, bd
%else
%define pavg pavgb
cglobal convolve_%1, 4, 7, 4+AUX_XMM_REGS, src, src_stride, \
                                           dst, dst_stride, \
                                           fx, fxs, fy, fys, w, h
%endif
  mov r4d, dword wm
%ifidn %2, highbd
  shl r4d, 1
  shl srcq, 1
  shl src_strideq, 1
  shl dstq, 1
  shl dst_strideq, 1
%else
  cmp r4d, 4
  je .w4
%endif
  cmp r4d, 8
  je .w8
  cmp r4d, 16
  je .w16
  cmp r4d, 32
  je .w32
%ifidn %2, highbd
  cmp r4d, 64
  je .w64

  mov                    r4d, dword hm
.loop128:
  movu                    m0, [srcq]
  movu                    m1, [srcq+16]
  movu                    m2, [srcq+32]
  movu                    m3, [srcq+48]
%ifidn %1, avg
  pavg                    m0, [dstq]
  pavg                    m1, [dstq+16]
  pavg                    m2, [dstq+32]
  pavg                    m3, [dstq+48]
%endif
  mova             [dstq   ], m0
  mova             [dstq+16], m1
  mova             [dstq+32], m2
  mova             [dstq+48], m3
  movu                    m0, [srcq+64]
  movu                    m1, [srcq+80]
  movu                    m2, [srcq+96]
  movu                    m3, [srcq+112]
  add                   srcq, src_strideq
%ifidn %1, avg
  pavg                    m0, [dstq+64]
  pavg                    m1, [dstq+80]
  pavg                    m2, [dstq+96]
  pavg                    m3, [dstq+112]
%endif
  mova             [dstq+64], m0
  mova             [dstq+80], m1
  mova             [dstq+96], m2
  mova            [dstq+112], m3
  add                   dstq, dst_strideq
  dec                    r4d
  jnz .loop128
  RET
%endif

.w64
  mov                    r4d, dword hm
.loop64:
  movu                    m0, [srcq]
  movu                    m1, [srcq+16]
  movu                    m2, [srcq+32]
  movu                    m3, [srcq+48]
  add                   srcq, src_strideq
%ifidn %1, avg
  pavg                    m0, [dstq]
  pavg                    m1, [dstq+16]
  pavg                    m2, [dstq+32]
  pavg                    m3, [dstq+48]
%endif
  mova             [dstq   ], m0
  mova             [dstq+16], m1
  mova             [dstq+32], m2
  mova             [dstq+48], m3
  add                   dstq, dst_strideq
  dec                    r4d
  jnz .loop64
  RET

.w32:
  mov                    r4d, dword hm
.loop32:
  movu                    m0, [srcq]
  movu                    m1, [srcq+16]
  movu                    m2, [srcq+src_strideq]
  movu                    m3, [srcq+src_strideq+16]
  lea                   srcq, [srcq+src_strideq*2]
%ifidn %1, avg
  pavg                    m0, [dstq]
  pavg                    m1, [dstq            +16]
  pavg                    m2, [dstq+dst_strideq]
  pavg                    m3, [dstq+dst_strideq+16]
%endif
  mova [dstq               ], m0
  mova [dstq            +16], m1
  mova [dstq+dst_strideq   ], m2
  mova [dstq+dst_strideq+16], m3
  lea                   dstq, [dstq+dst_strideq*2]
  sub                    r4d, 2
  jnz .loop32
  RET

.w16:
  mov                    r4d, dword hm
  lea                    r5q, [src_strideq*3]
  lea                    r6q, [dst_strideq*3]
.loop16:
  movu                    m0, [srcq]
  movu                    m1, [srcq+src_strideq]
  movu                    m2, [srcq+src_strideq*2]
  movu                    m3, [srcq+r5q]
  lea                   srcq, [srcq+src_strideq*4]
%ifidn %1, avg
  pavg                    m0, [dstq]
  pavg                    m1, [dstq+dst_strideq]
  pavg                    m2, [dstq+dst_strideq*2]
  pavg                    m3, [dstq+r6q]
%endif
  mova  [dstq              ], m0
  mova  [dstq+dst_strideq  ], m1
  mova  [dstq+dst_strideq*2], m2
  mova  [dstq+r6q          ], m3
  lea                   dstq, [dstq+dst_strideq*4]
  sub                    r4d, 4
  jnz .loop16
  RET

.w8:
  mov                    r4d, dword hm
  lea                    r5q, [src_strideq*3]
  lea                    r6q, [dst_strideq*3]
.loop8:
  movh                    m0, [srcq]
  movh                    m1, [srcq+src_strideq]
  movh                    m2, [srcq+src_strideq*2]
  movh                    m3, [srcq+r5q]
  lea                   srcq, [srcq+src_strideq*4]
%ifidn %1, avg
  movh                    m4, [dstq]
  movh                    m5, [dstq+dst_strideq]
  movh                    m6, [dstq+dst_strideq*2]
  movh                    m7, [dstq+r6q]
  pavg                    m0, m4
  pavg                    m1, m5
  pavg                    m2, m6
  pavg                    m3, m7
%endif
  movh  [dstq              ], m0
  movh  [dstq+dst_strideq  ], m1
  movh  [dstq+dst_strideq*2], m2
  movh  [dstq+r6q          ], m3
  lea                   dstq, [dstq+dst_strideq*4]
  sub                    r4d, 4
  jnz .loop8
  RET

%ifnidn %2, highbd
.w4:
  mov                    r4d, dword hm
  lea                    r5q, [src_strideq*3]
  lea                    r6q, [dst_strideq*3]
.loop4:
  movd                    m0, [srcq]
  movd                    m1, [srcq+src_strideq]
  movd                    m2, [srcq+src_strideq*2]
  movd                    m3, [srcq+r5q]
  lea                   srcq, [srcq+src_strideq*4]
%ifidn %1, avg
  movd                    m4, [dstq]
  movd                    m5, [dstq+dst_strideq]
  movd                    m6, [dstq+dst_strideq*2]
  movd                    m7, [dstq+r6q]
  pavg                    m0, m4
  pavg                    m1, m5
  pavg                    m2, m6
  pavg                    m3, m7
%endif
  movd  [dstq              ], m0
  movd  [dstq+dst_strideq  ], m1
  movd  [dstq+dst_strideq*2], m2
  movd  [dstq+r6q          ], m3
  lea                   dstq, [dstq+dst_strideq*4]
  sub                    r4d, 4
  jnz .loop4
  RET
%endif
%endmacro

INIT_XMM sse2
convolve_fn copy
convolve_fn avg
%if CONFIG_VP9_HIGHBITDEPTH
convolve_fn copy, highbd
convolve_fn avg, highbd
%endif
