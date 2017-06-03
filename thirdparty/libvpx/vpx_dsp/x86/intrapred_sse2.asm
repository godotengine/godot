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

SECTION_RODATA
pb_1: times 16 db 1
pw_4:  times 8 dw 4
pw_8:  times 8 dw 8
pw_16: times 8 dw 16
pw_32: times 8 dw 32
dc_128: times 16 db 128
pw2_4:  times 8 dw 2
pw2_8:  times 8 dw 4
pw2_16:  times 8 dw 8
pw2_32:  times 8 dw 16

SECTION .text

; ------------------------------------------
; input: x, y, z, result
;
; trick from pascal
; (x+2y+z+2)>>2 can be calculated as:
; result = avg(x,z)
; result -= xor(x,z) & 1
; result = avg(result,y)
; ------------------------------------------
%macro X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 4
  pavgb               %4, %1, %3
  pxor                %3, %1
  pand                %3, [GLOBAL(pb_1)]
  psubb               %4, %3
  pavgb               %4, %2
%endmacro

INIT_XMM sse2
cglobal d45_predictor_4x4, 3, 4, 4, dst, stride, above, goffset
  GET_GOT     goffsetq

  movq                 m0, [aboveq]
  DEFINE_ARGS dst, stride, temp
  psrldq               m1, m0, 1
  psrldq               m2, m0, 2
  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m1, m2, m3

  ; store 4 lines
  movd   [dstq          ], m3
  psrlq                m3, 8
  movd   [dstq+strideq  ], m3
  lea                dstq, [dstq+strideq*2]
  psrlq                m3, 8
  movd   [dstq          ], m3
  psrlq                m3, 8
  movd   [dstq+strideq  ], m3
  psrlq                m0, 56
  movd              tempq, m0
  mov    [dstq+strideq+3], tempb

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal d45_predictor_8x8, 3, 4, 4, dst, stride, above, goffset
  GET_GOT     goffsetq

  movu                m1, [aboveq]
  pslldq              m0, m1, 1
  psrldq              m2, m1, 1
  DEFINE_ARGS dst, stride, stride3
  lea           stride3q, [strideq*3]
  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m1, m2, m3
  punpckhbw           m0, m0 ; 7 7
  punpcklwd           m0, m0 ; 7 7 7 7
  punpckldq           m0, m0 ; 7 7 7 7 7 7 7 7
  punpcklqdq          m3, m0 ; -1 0 1 2 3 4 5 6 7 7 7 7 7 7 7 7

 ; store 4 lines
  psrldq                m3, 1
  movq    [dstq          ], m3
  psrldq                m3, 1
  movq    [dstq+strideq  ], m3
  psrldq                m3, 1
  movq    [dstq+strideq*2], m3
  psrldq                m3, 1
  movq    [dstq+stride3q ], m3
  lea                 dstq, [dstq+strideq*4]

  ; store next 4 lines
  psrldq                m3, 1
  movq    [dstq          ], m3
  psrldq                m3, 1
  movq    [dstq+strideq  ], m3
  psrldq                m3, 1
  movq    [dstq+strideq*2], m3
  psrldq                m3, 1
  movq    [dstq+stride3q ], m3

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal d207_predictor_4x4, 4, 4, 5, dst, stride, unused, left, goffset
  GET_GOT     goffsetq

  movd                m0, [leftq]                ; abcd [byte]
  punpcklbw           m4, m0, m0                 ; aabb ccdd
  punpcklwd           m4, m4                     ; aaaa bbbb cccc dddd
  psrldq              m4, 12                     ; dddd
  punpckldq           m0, m4                     ; abcd dddd
  psrldq              m1, m0, 1                  ; bcdd
  psrldq              m2, m0, 2                  ; cddd

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m1, m2, m3   ; a2bc b2cd c3d d
  pavgb               m1, m0                     ; ab, bc, cd, d [byte]

  punpcklbw           m1, m3             ; ab, a2bc, bc, b2cd, cd, c3d, d, d
  movd    [dstq        ], m1
  psrlq               m1, 16             ; bc, b2cd, cd, c3d, d, d
  movd    [dstq+strideq], m1

  lea               dstq, [dstq+strideq*2]
  psrlq               m1, 16             ; cd, c3d, d, d
  movd    [dstq        ], m1
  movd    [dstq+strideq], m4             ; d, d, d, d
  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_predictor_4x4, 4, 5, 3, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  movd                  m2, [leftq]
  movd                  m0, [aboveq]
  pxor                  m1, m1
  punpckldq             m0, m2
  psadbw                m0, m1
  paddw                 m0, [GLOBAL(pw_4)]
  psraw                 m0, 3
  pshuflw               m0, m0, 0x0
  packuswb              m0, m0
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0
  lea                 dstq, [dstq+strideq*2]
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_left_predictor_4x4, 2, 5, 2, dst, stride, above, left, goffset
  movifnidn          leftq, leftmp
  GET_GOT     goffsetq

  pxor                  m1, m1
  movd                  m0, [leftq]
  psadbw                m0, m1
  paddw                 m0, [GLOBAL(pw2_4)]
  psraw                 m0, 2
  pshuflw               m0, m0, 0x0
  packuswb              m0, m0
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0
  lea                 dstq, [dstq+strideq*2]
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_top_predictor_4x4, 3, 5, 2, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  movd                  m0, [aboveq]
  psadbw                m0, m1
  paddw                 m0, [GLOBAL(pw2_4)]
  psraw                 m0, 2
  pshuflw               m0, m0, 0x0
  packuswb              m0, m0
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0
  lea                 dstq, [dstq+strideq*2]
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_predictor_8x8, 4, 5, 3, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  movq                  m0, [aboveq]
  movq                  m2, [leftq]
  DEFINE_ARGS dst, stride, stride3
  lea             stride3q, [strideq*3]
  psadbw                m0, m1
  psadbw                m2, m1
  paddw                 m0, m2
  paddw                 m0, [GLOBAL(pw_8)]
  psraw                 m0, 4
  punpcklbw             m0, m0
  pshuflw               m0, m0, 0x0
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_top_predictor_8x8, 3, 5, 2, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  movq                  m0, [aboveq]
  DEFINE_ARGS dst, stride, stride3
  lea             stride3q, [strideq*3]
  psadbw                m0, m1
  paddw                 m0, [GLOBAL(pw2_8)]
  psraw                 m0, 3
  punpcklbw             m0, m0
  pshuflw               m0, m0, 0x0
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_left_predictor_8x8, 2, 5, 2, dst, stride, above, left, goffset
  movifnidn          leftq, leftmp
  GET_GOT     goffsetq

  pxor                  m1, m1
  movq                  m0, [leftq]
  DEFINE_ARGS dst, stride, stride3
  lea             stride3q, [strideq*3]
  psadbw                m0, m1
  paddw                 m0, [GLOBAL(pw2_8)]
  psraw                 m0, 3
  punpcklbw             m0, m0
  pshuflw               m0, m0, 0x0
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0

  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_128_predictor_4x4, 2, 5, 1, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  DEFINE_ARGS dst, stride, stride3
  lea             stride3q, [strideq*3]
  movd     m0,        [GLOBAL(dc_128)]
  movd    [dstq          ], m0
  movd    [dstq+strideq  ], m0
  movd    [dstq+strideq*2], m0
  movd    [dstq+stride3q ], m0
  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_128_predictor_8x8, 2, 5, 1, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  DEFINE_ARGS dst, stride, stride3
  lea             stride3q, [strideq*3]
  movq    m0,        [GLOBAL(dc_128)]
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0
  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal dc_predictor_16x16, 4, 5, 3, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [aboveq]
  mova                  m2, [leftq]
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 4
  psadbw                m0, m1
  psadbw                m2, m1
  paddw                 m0, m2
  movhlps               m2, m0
  paddw                 m0, m2
  paddw                 m0, [GLOBAL(pw_16)]
  psraw                 m0, 5
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
  packuswb              m0, m0
.loop:
  mova    [dstq          ], m0
  mova    [dstq+strideq  ], m0
  mova    [dstq+strideq*2], m0
  mova    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET


INIT_XMM sse2
cglobal dc_top_predictor_16x16, 4, 5, 3, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [aboveq]
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 4
  psadbw                m0, m1
  movhlps               m2, m0
  paddw                 m0, m2
  paddw                 m0, [GLOBAL(pw2_16)]
  psraw                 m0, 4
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
  packuswb              m0, m0
.loop:
  mova    [dstq          ], m0
  mova    [dstq+strideq  ], m0
  mova    [dstq+strideq*2], m0
  mova    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET

INIT_XMM sse2
cglobal dc_left_predictor_16x16, 4, 5, 3, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [leftq]
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 4
  psadbw                m0, m1
  movhlps               m2, m0
  paddw                 m0, m2
  paddw                 m0, [GLOBAL(pw2_16)]
  psraw                 m0, 4
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
  packuswb              m0, m0
.loop:
  mova    [dstq          ], m0
  mova    [dstq+strideq  ], m0
  mova    [dstq+strideq*2], m0
  mova    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET

INIT_XMM sse2
cglobal dc_128_predictor_16x16, 4, 5, 3, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 4
  mova    m0,        [GLOBAL(dc_128)]
.loop:
  mova    [dstq          ], m0
  mova    [dstq+strideq  ], m0
  mova    [dstq+strideq*2], m0
  mova    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop
  RESTORE_GOT
  RET


INIT_XMM sse2
cglobal dc_predictor_32x32, 4, 5, 5, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [aboveq]
  mova                  m2, [aboveq+16]
  mova                  m3, [leftq]
  mova                  m4, [leftq+16]
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 8
  psadbw                m0, m1
  psadbw                m2, m1
  psadbw                m3, m1
  psadbw                m4, m1
  paddw                 m0, m2
  paddw                 m0, m3
  paddw                 m0, m4
  movhlps               m2, m0
  paddw                 m0, m2
  paddw                 m0, [GLOBAL(pw_32)]
  psraw                 m0, 6
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
  packuswb              m0, m0
.loop:
  mova [dstq             ], m0
  mova [dstq          +16], m0
  mova [dstq+strideq     ], m0
  mova [dstq+strideq  +16], m0
  mova [dstq+strideq*2   ], m0
  mova [dstq+strideq*2+16], m0
  mova [dstq+stride3q    ], m0
  mova [dstq+stride3q +16], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET

INIT_XMM sse2
cglobal dc_top_predictor_32x32, 4, 5, 5, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [aboveq]
  mova                  m2, [aboveq+16]
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 8
  psadbw                m0, m1
  psadbw                m2, m1
  paddw                 m0, m2
  movhlps               m2, m0
  paddw                 m0, m2
  paddw                 m0, [GLOBAL(pw2_32)]
  psraw                 m0, 5
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
  packuswb              m0, m0
.loop:
  mova [dstq             ], m0
  mova [dstq          +16], m0
  mova [dstq+strideq     ], m0
  mova [dstq+strideq  +16], m0
  mova [dstq+strideq*2   ], m0
  mova [dstq+strideq*2+16], m0
  mova [dstq+stride3q    ], m0
  mova [dstq+stride3q +16], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET

INIT_XMM sse2
cglobal dc_left_predictor_32x32, 4, 5, 5, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  pxor                  m1, m1
  mova                  m0, [leftq]
  mova                  m2, [leftq+16]
  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 8
  psadbw                m0, m1
  psadbw                m2, m1
  paddw                 m0, m2
  movhlps               m2, m0
  paddw                 m0, m2
  paddw                 m0, [GLOBAL(pw2_32)]
  psraw                 m0, 5
  pshuflw               m0, m0, 0x0
  punpcklqdq            m0, m0
  packuswb              m0, m0
.loop:
  mova [dstq             ], m0
  mova [dstq          +16], m0
  mova [dstq+strideq     ], m0
  mova [dstq+strideq  +16], m0
  mova [dstq+strideq*2   ], m0
  mova [dstq+strideq*2+16], m0
  mova [dstq+stride3q    ], m0
  mova [dstq+stride3q +16], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop

  RESTORE_GOT
  REP_RET

INIT_XMM sse2
cglobal dc_128_predictor_32x32, 4, 5, 3, dst, stride, above, left, goffset
  GET_GOT     goffsetq

  DEFINE_ARGS dst, stride, stride3, lines4
  lea             stride3q, [strideq*3]
  mov              lines4d, 8
  mova    m0,        [GLOBAL(dc_128)]
.loop:
  mova [dstq             ], m0
  mova [dstq          +16], m0
  mova [dstq+strideq     ], m0
  mova [dstq+strideq  +16], m0
  mova [dstq+strideq*2   ], m0
  mova [dstq+strideq*2+16], m0
  mova [dstq+stride3q    ], m0
  mova [dstq+stride3q +16], m0
  lea                 dstq, [dstq+strideq*4]
  dec              lines4d
  jnz .loop
  RESTORE_GOT
  RET

INIT_XMM sse2
cglobal v_predictor_4x4, 3, 3, 1, dst, stride, above
  movd                  m0, [aboveq]
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0
  lea                 dstq, [dstq+strideq*2]
  movd      [dstq        ], m0
  movd      [dstq+strideq], m0
  RET

INIT_XMM sse2
cglobal v_predictor_8x8, 3, 3, 1, dst, stride, above
  movq                  m0, [aboveq]
  DEFINE_ARGS dst, stride, stride3
  lea             stride3q, [strideq*3]
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  movq    [dstq          ], m0
  movq    [dstq+strideq  ], m0
  movq    [dstq+strideq*2], m0
  movq    [dstq+stride3q ], m0
  RET

INIT_XMM sse2
cglobal v_predictor_16x16, 3, 4, 1, dst, stride, above
  mova                  m0, [aboveq]
  DEFINE_ARGS dst, stride, stride3, nlines4
  lea             stride3q, [strideq*3]
  mov              nlines4d, 4
.loop:
  mova    [dstq          ], m0
  mova    [dstq+strideq  ], m0
  mova    [dstq+strideq*2], m0
  mova    [dstq+stride3q ], m0
  lea                 dstq, [dstq+strideq*4]
  dec             nlines4d
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal v_predictor_32x32, 3, 4, 2, dst, stride, above
  mova                  m0, [aboveq]
  mova                  m1, [aboveq+16]
  DEFINE_ARGS dst, stride, stride3, nlines4
  lea             stride3q, [strideq*3]
  mov              nlines4d, 8
.loop:
  mova [dstq             ], m0
  mova [dstq          +16], m1
  mova [dstq+strideq     ], m0
  mova [dstq+strideq  +16], m1
  mova [dstq+strideq*2   ], m0
  mova [dstq+strideq*2+16], m1
  mova [dstq+stride3q    ], m0
  mova [dstq+stride3q +16], m1
  lea                 dstq, [dstq+strideq*4]
  dec             nlines4d
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal h_predictor_4x4, 2, 4, 4, dst, stride, line, left
  movifnidn          leftq, leftmp
  movd                  m0, [leftq]
  punpcklbw             m0, m0
  punpcklbw             m0, m0
  pshufd                m1, m0, 0x1
  movd      [dstq        ], m0
  movd      [dstq+strideq], m1
  pshufd                m2, m0, 0x2
  lea                 dstq, [dstq+strideq*2]
  pshufd                m3, m0, 0x3
  movd      [dstq        ], m2
  movd      [dstq+strideq], m3
  RET

INIT_XMM sse2
cglobal h_predictor_8x8, 2, 5, 3, dst, stride, line, left
  movifnidn          leftq, leftmp
  mov                lineq, -2
  DEFINE_ARGS  dst, stride, line, left, stride3
  lea             stride3q, [strideq*3]
  movq                  m0, [leftq    ]
  punpcklbw             m0, m0              ; l1 l1 l2 l2 ... l8 l8
.loop:
  pshuflw               m1, m0, 0x0         ; l1 l1 l1 l1 l1 l1 l1 l1
  pshuflw               m2, m0, 0x55        ; l2 l2 l2 l2 l2 l2 l2 l2
  movq      [dstq        ], m1
  movq      [dstq+strideq], m2
  pshuflw               m1, m0, 0xaa
  pshuflw               m2, m0, 0xff
  movq    [dstq+strideq*2], m1
  movq    [dstq+stride3q ], m2
  pshufd                m0, m0, 0xe         ; [63:0] l5 l5 l6 l6 l7 l7 l8 l8
  inc                lineq
  lea                 dstq, [dstq+strideq*4]
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal h_predictor_16x16, 2, 5, 3, dst, stride, line, left
  movifnidn          leftq, leftmp
  mov                lineq, -4
  DEFINE_ARGS dst, stride, line, left, stride3
  lea             stride3q, [strideq*3]
.loop:
  movd                  m0, [leftq]
  punpcklbw             m0, m0
  punpcklbw             m0, m0              ; l1 to l4 each repeated 4 times
  pshufd            m1, m0, 0x0             ; l1 repeated 16 times
  pshufd            m2, m0, 0x55            ; l2 repeated 16 times
  mova    [dstq          ], m1
  mova    [dstq+strideq  ], m2
  pshufd            m1, m0, 0xaa
  pshufd            m2, m0, 0xff
  mova    [dstq+strideq*2], m1
  mova    [dstq+stride3q ], m2
  inc                lineq
  lea                leftq, [leftq+4       ]
  lea                 dstq, [dstq+strideq*4]
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal h_predictor_32x32, 2, 5, 3, dst, stride, line, left
  movifnidn              leftq, leftmp
  mov                    lineq, -8
  DEFINE_ARGS dst, stride, line, left, stride3
  lea                 stride3q, [strideq*3]
.loop:
  movd                      m0, [leftq]
  punpcklbw                 m0, m0
  punpcklbw                 m0, m0              ; l1 to l4 each repeated 4 times
  pshufd                m1, m0, 0x0             ; l1 repeated 16 times
  pshufd                m2, m0, 0x55            ; l2 repeated 16 times
  mova     [dstq             ], m1
  mova     [dstq+16          ], m1
  mova     [dstq+strideq     ], m2
  mova     [dstq+strideq+16  ], m2
  pshufd                m1, m0, 0xaa
  pshufd                m2, m0, 0xff
  mova     [dstq+strideq*2   ], m1
  mova     [dstq+strideq*2+16], m1
  mova     [dstq+stride3q    ], m2
  mova     [dstq+stride3q+16 ], m2
  inc                    lineq
  lea                    leftq, [leftq+4       ]
  lea                     dstq, [dstq+strideq*4]
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal tm_predictor_4x4, 4, 4, 5, dst, stride, above, left
  pxor                  m1, m1
  movq                  m0, [aboveq-1]; [63:0] tl t1 t2 t3 t4 x x x
  punpcklbw             m0, m1
  pshuflw               m2, m0, 0x0   ; [63:0] tl tl tl tl [word]
  psrldq                m0, 2
  psubw                 m0, m2        ; [63:0] t1-tl t2-tl t3-tl t4-tl [word]
  movd                  m2, [leftq]
  punpcklbw             m2, m1
  pshuflw               m4, m2, 0x0   ; [63:0] l1 l1 l1 l1 [word]
  pshuflw               m3, m2, 0x55  ; [63:0] l2 l2 l2 l2 [word]
  paddw                 m4, m0
  paddw                 m3, m0
  packuswb              m4, m4
  packuswb              m3, m3
  movd      [dstq        ], m4
  movd      [dstq+strideq], m3
  lea                 dstq, [dstq+strideq*2]
  pshuflw               m4, m2, 0xaa
  pshuflw               m3, m2, 0xff
  paddw                 m4, m0
  paddw                 m3, m0
  packuswb              m4, m4
  packuswb              m3, m3
  movd      [dstq        ], m4
  movd      [dstq+strideq], m3
  RET

INIT_XMM sse2
cglobal tm_predictor_8x8, 4, 4, 5, dst, stride, above, left
  pxor                  m1, m1
  movd                  m2, [aboveq-1]
  movq                  m0, [aboveq]
  punpcklbw             m2, m1
  punpcklbw             m0, m1        ; t1 t2 t3 t4 t5 t6 t7 t8 [word]
  pshuflw               m2, m2, 0x0   ; [63:0] tl tl tl tl [word]
  DEFINE_ARGS dst, stride, line, left
  mov                lineq, -4
  punpcklqdq            m2, m2        ; tl tl tl tl tl tl tl tl [word]
  psubw                 m0, m2        ; t1-tl t2-tl ... t8-tl [word]
  movq                  m2, [leftq]
  punpcklbw             m2, m1        ; l1 l2 l3 l4 l5 l6 l7 l8 [word]
.loop
  pshuflw               m4, m2, 0x0   ; [63:0] l1 l1 l1 l1 [word]
  pshuflw               m3, m2, 0x55  ; [63:0] l2 l2 l2 l2 [word]
  punpcklqdq            m4, m4        ; l1 l1 l1 l1 l1 l1 l1 l1 [word]
  punpcklqdq            m3, m3        ; l2 l2 l2 l2 l2 l2 l2 l2 [word]
  paddw                 m4, m0
  paddw                 m3, m0
  packuswb              m4, m3
  movq      [dstq        ], m4
  movhps    [dstq+strideq], m4
  lea                 dstq, [dstq+strideq*2]
  psrldq                m2, 4
  inc                lineq
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal tm_predictor_16x16, 4, 5, 8, dst, stride, above, left
  pxor                  m1, m1
  mova                  m2, [aboveq-16];
  mova                  m0, [aboveq]   ; t1 t2 ... t16 [byte]
  punpckhbw             m2, m1         ; [127:112] tl [word]
  punpckhbw             m4, m0, m1
  punpcklbw             m0, m1         ; m0:m4 t1 t2 ... t16 [word]
  DEFINE_ARGS dst, stride, line, left, stride8
  mov                lineq, -8
  pshufhw               m2, m2, 0xff
  mova                  m3, [leftq]    ; l1 l2 ... l16 [byte]
  punpckhqdq            m2, m2         ; tl repeated 8 times [word]
  psubw                 m0, m2
  psubw                 m4, m2         ; m0:m4 t1-tl t2-tl ... t16-tl [word]
  punpckhbw             m5, m3, m1
  punpcklbw             m3, m1         ; m3:m5 l1 l2 ... l16 [word]
  lea             stride8q, [strideq*8]
.loop:
  pshuflw               m6, m3, 0x0
  pshuflw               m7, m5, 0x0
  punpcklqdq            m6, m6         ; l1 repeated 8 times [word]
  punpcklqdq            m7, m7         ; l8 repeated 8 times [word]
  paddw                 m1, m6, m0
  paddw                 m6, m4         ; m1:m6 ti-tl+l1 [i=1,15] [word]
  psrldq                m5, 2
  packuswb              m1, m6
  mova     [dstq         ], m1
  paddw                 m1, m7, m0
  paddw                 m7, m4         ; m1:m7 ti-tl+l8 [i=1,15] [word]
  psrldq                m3, 2
  packuswb              m1, m7
  mova     [dstq+stride8q], m1
  inc                lineq
  lea                 dstq, [dstq+strideq]
  jnz .loop
  REP_RET

INIT_XMM sse2
cglobal tm_predictor_32x32, 4, 4, 8, dst, stride, above, left
  pxor                  m1, m1
  movd                  m2, [aboveq-1]
  mova                  m0, [aboveq]
  mova                  m4, [aboveq+16]
  punpcklbw             m2, m1
  punpckhbw             m3, m0, m1
  punpckhbw             m5, m4, m1
  punpcklbw             m0, m1
  punpcklbw             m4, m1
  pshuflw               m2, m2, 0x0
  DEFINE_ARGS dst, stride, line, left
  mov                lineq, -16
  punpcklqdq            m2, m2
  add                leftq, 32
  psubw                 m0, m2
  psubw                 m3, m2
  psubw                 m4, m2
  psubw                 m5, m2
.loop:
  movd                  m2, [leftq+lineq*2]
  pxor                  m1, m1
  punpcklbw             m2, m1
  pshuflw               m7, m2, 0x55
  pshuflw               m2, m2, 0x0
  punpcklqdq            m2, m2
  punpcklqdq            m7, m7
  paddw                 m6, m2, m3
  paddw                 m1, m2, m0
  packuswb              m1, m6
  mova   [dstq           ], m1
  paddw                 m6, m2, m5
  paddw                 m1, m2, m4
  packuswb              m1, m6
  mova   [dstq+16        ], m1
  paddw                 m6, m7, m3
  paddw                 m1, m7, m0
  packuswb              m1, m6
  mova   [dstq+strideq   ], m1
  paddw                 m6, m7, m5
  paddw                 m1, m7, m4
  packuswb              m1, m6
  mova   [dstq+strideq+16], m1
  lea                 dstq, [dstq+strideq*2]
  inc                lineq
  jnz .loop
  REP_RET
