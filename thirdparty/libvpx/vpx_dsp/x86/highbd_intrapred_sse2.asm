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

SECTION_RODATA
pw_4:  times 8 dw 4
pw_8:  times 8 dw 8
pw_16: times 4 dd 16
pw_32: times 4 dd 32

SECTION .text
INIT_XMM sse2
cglobal highbd_dc_predictor_4x4, 4, 5, 4, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  movq                  m0, [aboveq]
  movq                  m2, [leftq]
  paddw                 m0, m2
  pshuflw               m1, m0, 0xe
  paddw                 m0, m1
  pshuflw               m1, m0, 0x1
  paddw                 m0, m1
  paddw                 m0, [GLOBAL(pw_4)]
  psraw                 m0, 3
  pshuflw               m0, m0, 0x0
  movq    [dstq          ], m0
  movq    [dstq+strideq*2], m0
  lea                 dstq, [dstq+strideq*4]
  movq    [dstq          ], m0
  movq    [dstq+strideq*2], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal highbd_dc_predictor_8x8, 4, 5, 4, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [aboveq]
  mova                  m2, [leftq]
  DEFINE_ARGS dst, stride, stride3, one
  mov                 oned, 0x00010001
  lea             stride3q, [strideq*3]
  movd                  m3, oned
  pshufd                m3, m3, 0x0
  paddw                 m0, m2
  pmaddwd               m0, m3
  packssdw              m0, m1
  pmaddwd               m0, m3
  packssdw              m0, m1
  pmaddwd               m0, m3
  paddw                 m0, [GLOBAL(pw_8)]
  psrlw                 m0, 4
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
  mova   [dstq           ], m0
  mova   [dstq+strideq*2 ], m0
  mova   [dstq+strideq*4 ], m0
  mova   [dstq+stride3q*2], m0
  lea                 dstq, [dstq+strideq*8]
  mova   [dstq           ], m0
  mova   [dstq+strideq*2 ], m0
  mova   [dstq+strideq*4 ], m0
  mova   [dstq+stride3q*2], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal highbd_dc_predictor_16x16, 4, 5, 5, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [aboveq]
  mova                  m3, [aboveq+16]
  mova                  m2, [leftq]
  mova                  m4, [leftq+16]
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 4
  paddw                 m0, m2
  paddw                 m0, m3
  paddw                 m0, m4
  movhlps               m2, m0
  paddw                 m0, m2
  punpcklwd             m0, m1
  movhlps               m2, m0
  paddd                 m0, m2
  punpckldq             m0, m1
  movhlps               m2, m0
  paddd                 m0, m2
  paddd                 m0, [GLOBAL(pw_16)]
  psrad                 m0, 5
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
.loop:
  mova   [dstq              ], m0
  mova   [dstq           +16], m0
  mova   [dstq+strideq*2    ], m0
  mova   [dstq+strideq*2 +16], m0
  mova   [dstq+strideq*4    ], m0
  mova   [dstq+strideq*4 +16], m0
  mova   [dstq+stride3q*2   ], m0
  mova   [dstq+stride3q*2+16], m0
  lea                 dstq, [dstq+strideq*8]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET

INIT_XMM sse2
cglobal highbd_dc_predictor_32x32, 4, 5, 7, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  mova                  m0, [aboveq]
  mova                  m2, [aboveq+16]
  mova                  m3, [aboveq+32]
  mova                  m4, [aboveq+48]
  paddw                 m0, m2
  paddw                 m3, m4
  mova                  m2, [leftq]
  mova                  m4, [leftq+16]
  mova                  m5, [leftq+32]
  mova                  m6, [leftq+48]
  paddw                 m2, m4
  paddw                 m5, m6
  paddw                 m0, m3
  paddw                 m2, m5
  pxor                  m1, m1
  paddw                 m0, m2
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 8
  movhlps               m2, m0
  paddw                 m0, m2
  punpcklwd             m0, m1
  movhlps               m2, m0
  paddd                 m0, m2
  punpckldq             m0, m1
  movhlps               m2, m0
  paddd                 m0, m2
  paddd                 m0, [GLOBAL(pw_32)]
  psrad                 m0, 6
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
.loop:
  mova [dstq               ], m0
  mova [dstq          +16  ], m0
  mova [dstq          +32  ], m0
  mova [dstq          +48  ], m0
  mova [dstq+strideq*2     ], m0
  mova [dstq+strideq*2+16  ], m0
  mova [dstq+strideq*2+32  ], m0
  mova [dstq+strideq*2+48  ], m0
  mova [dstq+strideq*4     ], m0
  mova [dstq+strideq*4+16  ], m0
  mova [dstq+strideq*4+32  ], m0
  mova [dstq+strideq*4+48  ], m0
  mova [dstq+stride3q*2    ], m0
  mova [dstq+stride3q*2 +16], m0
  mova [dstq+stride3q*2 +32], m0
  mova [dstq+stride3q*2 +48], m0
  lea                 dstq, [dstq+strideq*8]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET

INIT_XMM sse2
cglobal highbd_v_predictor_4x4, 3, 3, 1, dst, stride, above
  movq                  m0, [aboveq]
  movq    [dstq          ], m0
  movq    [dstq+strideq*2], m0
  lea                 dstq, [dstq+strideq*4]
  movq    [dstq          ], m0
  movq    [dstq+strideq*2], m0
  RET

INIT_XMM sse2
cglobal highbd_v_predictor_8x8, 3, 3, 1, dst, stride, above
  mova                  m0, [aboveq]
  DEFINE_ARGS dst, stride, stride3
  lea             stride3q, [strideq*3]
  mova   [dstq           ], m0
  mova   [dstq+strideq*2 ], m0
  mova   [dstq+strideq*4 ], m0
  mova   [dstq+stride3q*2], m0
  lea                 dstq, [dstq+strideq*8]
  mova   [dstq           ], m0
  mova   [dstq+strideq*2 ], m0
  mova   [dstq+strideq*4 ], m0
  mova   [dstq+stride3q*2], m0
  RET

INIT_XMM sse2
cglobal highbd_v_predictor_16x16, 3, 4, 2, dst, stride, above
  mova                  m0, [aboveq]
  mova                  m1, [aboveq+16]
  DEFINE_ARGS dst, stride, stride3, nlines4
  lea             stride3q, [strideq*3]
  mov              nlines4d, 4
.loop:
  mova    [dstq              ], m0
  mova    [dstq           +16], m1
  mova    [dstq+strideq*2    ], m0
  mova    [dstq+strideq*2 +16], m1
  mova    [dstq+strideq*4    ], m0
  mova    [dstq+strideq*4 +16], m1
  mova    [dstq+stride3q*2   ], m0
  mova    [dstq+stride3q*2+16], m1
  lea                 dstq, [dstq+strideq*8]
  dec             nlines4d
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal highbd_v_predictor_32x32, 3, 4, 4, dst, stride, above
  mova                  m0, [aboveq]
  mova                  m1, [aboveq+16]
  mova                  m2, [aboveq+32]
  mova                  m3, [aboveq+48]
  DEFINE_ARGS dst, stride, stride3, nlines4
  lea             stride3q, [strideq*3]
  mov              nlines4d, 8
.loop:
  mova [dstq               ], m0
  mova [dstq            +16], m1
  mova [dstq            +32], m2
  mova [dstq            +48], m3
  mova [dstq+strideq*2     ], m0
  mova [dstq+strideq*2  +16], m1
  mova [dstq+strideq*2  +32], m2
  mova [dstq+strideq*2  +48], m3
  mova [dstq+strideq*4     ], m0
  mova [dstq+strideq*4  +16], m1
  mova [dstq+strideq*4  +32], m2
  mova [dstq+strideq*4  +48], m3
  mova [dstq+stride3q*2    ], m0
  mova [dstq+stride3q*2 +16], m1
  mova [dstq+stride3q*2 +32], m2
  mova [dstq+stride3q*2 +48], m3
  lea                 dstq, [dstq+strideq*8]
  dec             nlines4d
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal highbd_tm_predictor_4x4, 5, 5, 6, dst, stride, above, left, bd
  movd                  m1, [aboveq-2]
  movq                  m0, [aboveq]
  pshuflw               m1, m1, 0x0
  movlhps               m0, m0         ; t1 t2 t3 t4 t1 t2 t3 t4
  movlhps               m1, m1         ; tl tl tl tl tl tl tl tl
  ; Get the values to compute the maximum value at this bit depth
  pcmpeqw               m3, m3
  movd                  m4, bdd
  psubw                 m0, m1         ; t1-tl t2-tl t3-tl t4-tl
  psllw                 m3, m4
  pcmpeqw               m2, m2
  pxor                  m4, m4         ; min possible value
  pxor                  m3, m2         ; max possible value
  mova                  m1, [leftq]
  pshuflw               m2, m1, 0x0
  pshuflw               m5, m1, 0x55
  movlhps               m2, m5         ; l1 l1 l1 l1 l2 l2 l2 l2
  paddw                 m2, m0
  ;Clamp to the bit-depth
  pminsw                m2, m3
  pmaxsw                m2, m4
  ;Store the values
  movq    [dstq          ], m2
  movhpd  [dstq+strideq*2], m2
  lea                 dstq, [dstq+strideq*4]
  pshuflw               m2, m1, 0xaa
  pshuflw               m5, m1, 0xff
  movlhps               m2, m5
  paddw                 m2, m0
  ;Clamp to the bit-depth
  pminsw                m2, m3
  pmaxsw                m2, m4
  ;Store the values
  movq    [dstq          ], m2
  movhpd  [dstq+strideq*2], m2
  RET

INIT_XMM sse2
cglobal highbd_tm_predictor_8x8, 5, 6, 5, dst, stride, above, left, bd, one
  movd                  m1, [aboveq-2]
  mova                  m0, [aboveq]
  pshuflw               m1, m1, 0x0
  ; Get the values to compute the maximum value at this bit depth
  mov                 oned, 1
  pxor                  m3, m3
  pxor                  m4, m4
  pinsrw                m3, oned, 0
  pinsrw                m4, bdd, 0
  pshuflw               m3, m3, 0x0
  DEFINE_ARGS dst, stride, line, left
  punpcklqdq            m3, m3
  mov                lineq, -4
  mova                  m2, m3
  punpcklqdq            m1, m1
  psllw                 m3, m4
  add                leftq, 16
  psubw                 m3, m2 ; max possible value
  pxor                  m4, m4 ; min possible value
  psubw                 m0, m1
.loop:
  movd                  m1, [leftq+lineq*4]
  movd                  m2, [leftq+lineq*4+2]
  pshuflw               m1, m1, 0x0
  pshuflw               m2, m2, 0x0
  punpcklqdq            m1, m1
  punpcklqdq            m2, m2
  paddw                 m1, m0
  paddw                 m2, m0
  ;Clamp to the bit-depth
  pminsw                m1, m3
  pminsw                m2, m3
  pmaxsw                m1, m4
  pmaxsw                m2, m4
  ;Store the values
  mova      [dstq          ], m1
  mova      [dstq+strideq*2], m2
  lea                 dstq, [dstq+strideq*4]
  inc                lineq
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal highbd_tm_predictor_16x16, 5, 5, 8, dst, stride, above, left, bd
  movd                  m2, [aboveq-2]
  mova                  m0, [aboveq]
  mova                  m1, [aboveq+16]
  pshuflw               m2, m2, 0x0
  ; Get the values to compute the maximum value at this bit depth
  pcmpeqw               m3, m3
  movd                  m4, bdd
  punpcklqdq            m2, m2
  psllw                 m3, m4
  pcmpeqw               m5, m5
  pxor                  m4, m4         ; min possible value
  pxor                  m3, m5         ; max possible value
  DEFINE_ARGS dst, stride, line, left
  mov                lineq, -8
  psubw                 m0, m2
  psubw                 m1, m2
.loop:
  movd                  m7, [leftq]
  pshuflw               m5, m7, 0x0
  pshuflw               m2, m7, 0x55
  punpcklqdq            m5, m5         ; l1 l1 l1 l1 l1 l1 l1 l1
  punpcklqdq            m2, m2         ; l2 l2 l2 l2 l2 l2 l2 l2
  paddw                 m6, m5, m0     ; t1-tl+l1 to t4-tl+l1
  paddw                 m5, m1         ; t5-tl+l1 to t8-tl+l1
  pminsw                m6, m3
  pminsw                m5, m3
  pmaxsw                m6, m4         ; Clamp to the bit-depth
  pmaxsw                m5, m4
  mova   [dstq           ], m6
  mova   [dstq        +16], m5
  paddw                 m6, m2, m0
  paddw                 m2, m1
  pminsw                m6, m3
  pminsw                m2, m3
  pmaxsw                m6, m4
  pmaxsw                m2, m4
  mova   [dstq+strideq*2 ], m6
  mova [dstq+strideq*2+16], m2
  lea                 dstq, [dstq+strideq*4]
  inc                lineq
  lea                leftq, [leftq+4]

  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal highbd_tm_predictor_32x32, 5, 5, 8, dst, stride, above, left, bd
  movd                  m0, [aboveq-2]
  mova                  m1, [aboveq]
  mova                  m2, [aboveq+16]
  mova                  m3, [aboveq+32]
  mova                  m4, [aboveq+48]
  pshuflw               m0, m0, 0x0
  ; Get the values to compute the maximum value at this bit depth
  pcmpeqw               m5, m5
  movd                  m6, bdd
  psllw                 m5, m6
  pcmpeqw               m7, m7
  pxor                  m6, m6         ; min possible value
  pxor                  m5, m7         ; max possible value
  punpcklqdq            m0, m0
  DEFINE_ARGS dst, stride, line, left
  mov                lineq, -16
  psubw                 m1, m0
  psubw                 m2, m0
  psubw                 m3, m0
  psubw                 m4, m0
.loop:
  movd                  m7, [leftq]
  pshuflw               m7, m7, 0x0
  punpcklqdq            m7, m7         ; l1 l1 l1 l1 l1 l1 l1 l1
  paddw                 m0, m7, m1
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq           ], m0
  paddw                 m0, m7, m2
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq        +16], m0
  paddw                 m0, m7, m3
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq        +32], m0
  paddw                 m0, m7, m4
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq        +48], m0
  movd                  m7, [leftq+2]
  pshuflw               m7, m7, 0x0
  punpcklqdq            m7, m7         ; l2 l2 l2 l2 l2 l2 l2 l2
  paddw                 m0, m7, m1
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq+strideq*2 ], m0
  paddw                 m0, m7, m2
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq+strideq*2+16], m0
  paddw                 m0, m7, m3
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq+strideq*2+32], m0
  paddw                 m0, m7, m4
  pminsw                m0, m5
  pmaxsw                m0, m6
  mova   [dstq+strideq*2+48], m0
  lea                 dstq, [dstq+strideq*4]
  lea                leftq, [leftq+4]
  inc                lineq
  jnz .loop
  REP_RET
