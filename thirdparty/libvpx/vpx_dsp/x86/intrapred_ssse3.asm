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
sh_b12345677: db 1, 2, 3, 4, 5, 6, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0
sh_b23456777: db 2, 3, 4, 5, 6, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0
sh_b0123456777777777: db 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7
sh_b1234567777777777: db 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
sh_b2345677777777777: db 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
sh_b123456789abcdeff: db 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15
sh_b23456789abcdefff: db 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 15
sh_b32104567: db 3, 2, 1, 0, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0
sh_b8091a2b345: db 8, 0, 9, 1, 10, 2, 11, 3, 4, 5, 0, 0, 0, 0, 0, 0
sh_b76543210: db 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
sh_b65432108: db 6, 5, 4, 3, 2, 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0
sh_b54321089: db 5, 4, 3, 2, 1, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0
sh_b89abcdef: db 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0
sh_bfedcba9876543210: db 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

SECTION .text

INIT_XMM ssse3
cglobal d45_predictor_16x16, 3, 6, 4, dst, stride, above, dst8, line, goffset
  GET_GOT     goffsetq

  mova                   m0, [aboveq]
  DEFINE_ARGS dst, stride, stride3, dst8, line
  lea              stride3q, [strideq*3]
  lea                 dst8q, [dstq+strideq*8]
  mova                   m1, [GLOBAL(sh_b123456789abcdeff)]
  pshufb                 m2, m0, [GLOBAL(sh_b23456789abcdefff)]
  pavgb                  m3, m2, m0
  pxor                   m2, m0
  pshufb                 m0, m1
  pand                   m2, [GLOBAL(pb_1)]
  psubb                  m3, m2
  pavgb                  m0, m3

  ; first 4 lines and first half of 3rd 4 lines
  mov                 lined, 2
.loop:
  mova   [dstq            ], m0
  movhps [dst8q           ], m0
  pshufb                 m0, m1
  mova   [dstq +strideq   ], m0
  movhps [dst8q+strideq   ], m0
  pshufb                 m0, m1
  mova   [dstq +strideq*2 ], m0
  movhps [dst8q+strideq*2 ], m0
  pshufb                 m0, m1
  mova   [dstq +stride3q  ], m0
  movhps [dst8q+stride3q  ], m0
  pshufb                 m0, m1
  lea                  dstq, [dstq +strideq*4]
  lea                 dst8q, [dst8q+strideq*4]
  dec                 lined
  jnz .loop

  ; bottom-right 8x8 block
  movhps [dstq          +8], m0
  movhps [dstq+strideq  +8], m0
  movhps [dstq+strideq*2+8], m0
  movhps [dstq+stride3q +8], m0
  lea                  dstq, [dstq+strideq*4]
  movhps [dstq          +8], m0
  movhps [dstq+strideq  +8], m0
  movhps [dstq+strideq*2+8], m0
  movhps [dstq+stride3q +8], m0

  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d45_predictor_32x32, 3, 6, 7, dst, stride, above, dst16, line, goffset
  GET_GOT     goffsetq

  mova                   m0, [aboveq]
  mova                   m4, [aboveq+16]
  DEFINE_ARGS dst, stride, stride3, dst16, line
  lea              stride3q, [strideq*3]
  lea                dst16q, [dstq  +strideq*8]
  lea                dst16q, [dst16q+strideq*8]
  mova                   m1, [GLOBAL(sh_b123456789abcdeff)]
  pshufb                 m2, m4, [GLOBAL(sh_b23456789abcdefff)]
  pavgb                  m3, m2, m4
  pxor                   m2, m4
  palignr                m5, m4, m0, 1
  palignr                m6, m4, m0, 2
  pshufb                 m4, m1
  pand                   m2, [GLOBAL(pb_1)]
  psubb                  m3, m2
  pavgb                  m4, m3
  pavgb                  m3, m0, m6
  pxor                   m0, m6
  pand                   m0, [GLOBAL(pb_1)]
  psubb                  m3, m0
  pavgb                  m5, m3

  ; write 4x4 lines (and the first half of the second 4x4 lines)
  mov                  lined, 4
.loop:
  mova [dstq               ], m5
  mova [dstq            +16], m4
  mova [dst16q             ], m4
  palignr                 m3, m4, m5, 1
  pshufb                  m4, m1
  mova [dstq  +strideq     ], m3
  mova [dstq  +strideq  +16], m4
  mova [dst16q+strideq     ], m4
  palignr                 m5, m4, m3, 1
  pshufb                  m4, m1
  mova [dstq  +strideq*2   ], m5
  mova [dstq  +strideq*2+16], m4
  mova [dst16q+strideq*2   ], m4
  palignr                 m3, m4, m5, 1
  pshufb                  m4, m1
  mova [dstq  +stride3q    ], m3
  mova [dstq  +stride3q +16], m4
  mova [dst16q+stride3q    ], m4
  palignr                 m5, m4, m3, 1
  pshufb                  m4, m1
  lea                  dstq, [dstq  +strideq*4]
  lea                dst16q, [dst16q+strideq*4]
  dec                 lined
  jnz .loop

  ; write second half of second 4x4 lines
  mova [dstq            +16], m4
  mova [dstq  +strideq  +16], m4
  mova [dstq  +strideq*2+16], m4
  mova [dstq  +stride3q +16], m4
  lea                  dstq, [dstq  +strideq*4]
  mova [dstq            +16], m4
  mova [dstq  +strideq  +16], m4
  mova [dstq  +strideq*2+16], m4
  mova [dstq  +stride3q +16], m4
  lea                  dstq, [dstq  +strideq*4]
  mova [dstq            +16], m4
  mova [dstq  +strideq  +16], m4
  mova [dstq  +strideq*2+16], m4
  mova [dstq  +stride3q +16], m4
  lea                  dstq, [dstq  +strideq*4]
  mova [dstq            +16], m4
  mova [dstq  +strideq  +16], m4
  mova [dstq  +strideq*2+16], m4
  mova [dstq  +stride3q +16], m4

  RESTORE_GOT
  RET

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

INIT_XMM ssse3
cglobal d63_predictor_4x4, 3, 4, 5, dst, stride, above, goffset
  GET_GOT     goffsetq

  movq                m3, [aboveq]
  pshufb              m1, m3, [GLOBAL(sh_b23456777)]
  pshufb              m2, m3, [GLOBAL(sh_b12345677)]

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m3, m2, m1, m4
  pavgb               m3, m2

  ; store 4 lines
  movd    [dstq        ], m3
  movd    [dstq+strideq], m4
  lea               dstq, [dstq+strideq*2]
  psrldq              m3, 1
  psrldq              m4, 1
  movd    [dstq        ], m3
  movd    [dstq+strideq], m4
  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d63_predictor_8x8, 3, 4, 5, dst, stride, above, goffset
  GET_GOT     goffsetq

  movq                m3, [aboveq]
  DEFINE_ARGS dst, stride, stride3
  lea           stride3q, [strideq*3]
  pshufb              m1, m3, [GLOBAL(sh_b2345677777777777)]
  pshufb              m0, m3, [GLOBAL(sh_b0123456777777777)]
  pshufb              m2, m3, [GLOBAL(sh_b1234567777777777)]
  pshufb              m3, [GLOBAL(sh_b0123456777777777)]

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m2, m1, m4
  pavgb               m3, m2

  ; store 4 lines
  movq    [dstq        ], m3
  movq    [dstq+strideq], m4
  psrldq              m3, 1
  psrldq              m4, 1
  movq  [dstq+strideq*2], m3
  movq  [dstq+stride3q ], m4
  lea               dstq, [dstq+strideq*4]
  psrldq              m3, 1
  psrldq              m4, 1

  ; store 4 lines
  movq    [dstq        ], m3
  movq    [dstq+strideq], m4
  psrldq              m3, 1
  psrldq              m4, 1
  movq  [dstq+strideq*2], m3
  movq  [dstq+stride3q ], m4
  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d63_predictor_16x16, 3, 5, 5, dst, stride, above, line, goffset
  GET_GOT     goffsetq

  mova                m0, [aboveq]
  DEFINE_ARGS dst, stride, stride3, line
  lea           stride3q, [strideq*3]
  mova                m1, [GLOBAL(sh_b123456789abcdeff)]
  pshufb              m2, m0, [GLOBAL(sh_b23456789abcdefff)]
  pshufb              m3, m0, m1

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m3, m2, m4
  pavgb               m0, m3

  mov              lined, 4
.loop:
  mova  [dstq          ], m0
  mova  [dstq+strideq  ], m4
  pshufb              m0, m1
  pshufb              m4, m1
  mova  [dstq+strideq*2], m0
  mova  [dstq+stride3q ], m4
  pshufb              m0, m1
  pshufb              m4, m1
  lea               dstq, [dstq+strideq*4]
  dec              lined
  jnz .loop
  RESTORE_GOT
  REP_RET

INIT_XMM ssse3
cglobal d63_predictor_32x32, 3, 5, 8, dst, stride, above, line, goffset
  GET_GOT     goffsetq

  mova                   m0, [aboveq]
  mova                   m7, [aboveq+16]
  DEFINE_ARGS dst, stride, stride3, line
  mova                   m1, [GLOBAL(sh_b123456789abcdeff)]
  lea              stride3q, [strideq*3]
  pshufb                 m2, m7, [GLOBAL(sh_b23456789abcdefff)]
  pshufb                 m3, m7, m1

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m7, m3, m2, m4
  palignr                m6, m7, m0, 1
  palignr                m5, m7, m0, 2
  pavgb                  m7, m3

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m6, m5, m2
  pavgb                  m0, m6

  mov                 lined, 8
.loop:
  mova  [dstq             ], m0
  mova  [dstq          +16], m7
  mova  [dstq+strideq     ], m2
  mova  [dstq+strideq  +16], m4
  palignr                m3, m7, m0, 1
  palignr                m5, m4, m2, 1
  pshufb                 m7, m1
  pshufb                 m4, m1

  mova  [dstq+strideq*2   ], m3
  mova  [dstq+strideq*2+16], m7
  mova  [dstq+stride3q    ], m5
  mova  [dstq+stride3q +16], m4
  palignr                m0, m7, m3, 1
  palignr                m2, m4, m5, 1
  pshufb                 m7, m1
  pshufb                 m4, m1
  lea                  dstq, [dstq+strideq*4]
  dec                 lined
  jnz .loop
  RESTORE_GOT
  REP_RET

INIT_XMM ssse3
cglobal d153_predictor_4x4, 4, 5, 4, dst, stride, above, left, goffset
  GET_GOT     goffsetq
  movd                m0, [leftq]               ; l1, l2, l3, l4
  movd                m1, [aboveq-1]            ; tl, t1, t2, t3
  punpckldq           m0, m1                    ; l1, l2, l3, l4, tl, t1, t2, t3
  pshufb              m0, [GLOBAL(sh_b32104567)]; l4, l3, l2, l1, tl, t1, t2, t3
  psrldq              m1, m0, 1                 ; l3, l2, l1, tl, t1, t2, t3
  psrldq              m2, m0, 2                 ; l2, l1, tl, t1, t2, t3
  ; comments below are for a predictor like this
  ; A1 B1 C1 D1
  ; A2 B2 A1 B1
  ; A3 B3 A2 B2
  ; A4 B4 A3 B3
  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m1, m2, m3  ; 3-tap avg B4 B3 B2 B1 C1 D1
  pavgb               m1, m0                    ; 2-tap avg A4 A3 A2 A1

  punpcklqdq          m3, m1                    ; B4 B3 B2 B1 C1 D1 x x A4 A3 A2 A1 ..

  DEFINE_ARGS dst, stride, stride3
  lea           stride3q, [strideq*3]
  pshufb              m3, [GLOBAL(sh_b8091a2b345)] ; A4 B4 A3 B3 A2 B2 A1 B1 C1 D1 ..
  movd  [dstq+stride3q ], m3
  psrldq              m3, 2                     ; A3 B3 A2 B2 A1 B1 C1 D1 ..
  movd  [dstq+strideq*2], m3
  psrldq              m3, 2                     ; A2 B2 A1 B1 C1 D1 ..
  movd  [dstq+strideq  ], m3
  psrldq              m3, 2                     ; A1 B1 C1 D1 ..
  movd  [dstq          ], m3
  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d153_predictor_8x8, 4, 5, 8, dst, stride, above, left, goffset
  GET_GOT     goffsetq
  movq                m0, [leftq]                     ; [0- 7] l1-8 [byte]
  movhps              m0, [aboveq-1]                  ; [8-15] tl, t1-7 [byte]
  pshufb              m1, m0, [GLOBAL(sh_b76543210)]  ; l8-1 [word]
  pshufb              m2, m0, [GLOBAL(sh_b65432108)]  ; l7-1,tl [word]
  pshufb              m3, m0, [GLOBAL(sh_b54321089)]  ; l6-1,tl,t1 [word]
  pshufb              m0, [GLOBAL(sh_b89abcdef)]      ; tl,t1-7 [word]
  psrldq              m4, m0, 1                       ; t1-7 [word]
  psrldq              m5, m0, 2                       ; t2-7 [word]
  ; comments below are for a predictor like this
  ; A1 B1 C1 D1 E1 F1 G1 H1
  ; A2 B2 A1 B1 C1 D1 E1 F1
  ; A3 B3 A2 B2 A1 B1 C1 D1
  ; A4 B4 A3 B3 A2 B2 A1 B1
  ; A5 B5 A4 B4 A3 B3 A2 B2
  ; A6 B6 A5 B5 A4 B4 A3 B3
  ; A7 B7 A6 B6 A5 B5 A4 B4
  ; A8 B8 A7 B7 A6 B6 A5 B5
  pavgb               m6, m1, m2                ; 2-tap avg A8-A1

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m4, m5, m7  ; 3-tap avg C-H1

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m1, m2, m3, m0  ; 3-tap avg B8-1

  punpcklbw           m6, m0                    ; A-B8, A-B7 ... A-B2, A-B1

  DEFINE_ARGS dst, stride, stride3
  lea           stride3q, [strideq*3]

  movhps [dstq+stride3q], m6                    ; A-B4, A-B3, A-B2, A-B1
  palignr             m0, m7, m6, 10            ; A-B3, A-B2, A-B1, C-H1
  movq  [dstq+strideq*2], m0
  psrldq              m0, 2                     ; A-B2, A-B1, C-H1
  movq  [dstq+strideq  ], m0
  psrldq              m0, 2                     ; A-H1
  movq  [dstq          ], m0
  lea               dstq, [dstq+strideq*4]
  movq  [dstq+stride3q ], m6                    ; A-B8, A-B7, A-B6, A-B5
  psrldq              m6, 2                     ; A-B7, A-B6, A-B5, A-B4
  movq  [dstq+strideq*2], m6
  psrldq              m6, 2                     ; A-B6, A-B5, A-B4, A-B3
  movq  [dstq+strideq  ], m6
  psrldq              m6, 2                     ; A-B5, A-B4, A-B3, A-B2
  movq  [dstq          ], m6
  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d153_predictor_16x16, 4, 5, 8, dst, stride, above, left, goffset
  GET_GOT     goffsetq
  mova                m0, [leftq]
  movu                m7, [aboveq-1]
  ; comments below are for a predictor like this
  ; A1 B1 C1 D1 E1 F1 G1 H1 I1 J1 K1 L1 M1 N1 O1 P1
  ; A2 B2 A1 B1 C1 D1 E1 F1 G1 H1 I1 J1 K1 L1 M1 N1
  ; A3 B3 A2 B2 A1 B1 C1 D1 E1 F1 G1 H1 I1 J1 K1 L1
  ; A4 B4 A3 B3 A2 B2 A1 B1 C1 D1 E1 F1 G1 H1 I1 J1
  ; A5 B5 A4 B4 A3 B3 A2 B2 A1 B1 C1 D1 E1 F1 G1 H1
  ; A6 B6 A5 B5 A4 B4 A3 B3 A2 B2 A1 B1 C1 D1 E1 F1
  ; A7 B7 A6 B6 A5 B5 A4 B4 A3 B3 A2 B2 A1 B1 C1 D1
  ; A8 B8 A7 B7 A6 B6 A5 B5 A4 B4 A3 B3 A2 B2 A1 B1
  ; A9 B9 A8 B8 A7 B7 A6 B6 A5 B5 A4 B4 A3 B3 A2 B2
  ; Aa Ba A9 B9 A8 B8 A7 B7 A6 B6 A5 B5 A4 B4 A3 B3
  ; Ab Bb Aa Ba A9 B9 A8 B8 A7 B7 A6 B6 A5 B5 A4 B4
  ; Ac Bc Ab Bb Aa Ba A9 B9 A8 B8 A7 B7 A6 B6 A5 B5
  ; Ad Bd Ac Bc Ab Bb Aa Ba A9 B9 A8 B8 A7 B7 A6 B6
  ; Ae Be Ad Bd Ac Bc Ab Bb Aa Ba A9 B9 A8 B8 A7 B7
  ; Af Bf Ae Be Ad Bd Ac Bc Ab Bb Aa Ba A9 B9 A8 B8
  ; Ag Bg Af Bf Ae Be Ad Bd Ac Bc Ab Bb Aa Ba A9 B9
  pshufb              m6, m7, [GLOBAL(sh_bfedcba9876543210)]
  palignr             m5, m0, m6, 15
  palignr             m3, m0, m6, 14

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m5, m3, m4          ; 3-tap avg B3-Bg
  pshufb              m1, m0, [GLOBAL(sh_b123456789abcdeff)]
  pavgb               m5, m0                            ; A1 - Ag

  punpcklbw           m0, m4, m5                        ; A-B8 ... A-B1
  punpckhbw           m4, m5                            ; A-B9 ... A-Bg

  pshufb              m3, m7, [GLOBAL(sh_b123456789abcdeff)]
  pshufb              m5, m7, [GLOBAL(sh_b23456789abcdefff)]

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m7, m3, m5, m1          ; 3-tap avg C1-P1

  pshufb              m6, m0, [GLOBAL(sh_bfedcba9876543210)]
  DEFINE_ARGS dst, stride, stride3
  lea           stride3q, [strideq*3]
  palignr             m2, m1, m6, 14
  mova  [dstq          ], m2
  palignr             m2, m1, m6, 12
  mova  [dstq+strideq  ], m2
  palignr             m2, m1, m6, 10
  mova  [dstq+strideq*2], m2
  palignr             m2, m1, m6, 8
  mova  [dstq+stride3q ], m2
  lea               dstq, [dstq+strideq*4]
  palignr             m2, m1, m6, 6
  mova  [dstq          ], m2
  palignr             m2, m1, m6, 4
  mova  [dstq+strideq  ], m2
  palignr             m2, m1, m6, 2
  mova  [dstq+strideq*2], m2
  pshufb              m4, [GLOBAL(sh_bfedcba9876543210)]
  mova  [dstq+stride3q ], m6
  lea               dstq, [dstq+strideq*4]

  palignr             m2, m6, m4, 14
  mova  [dstq          ], m2
  palignr             m2, m6, m4, 12
  mova  [dstq+strideq  ], m2
  palignr             m2, m6, m4, 10
  mova  [dstq+strideq*2], m2
  palignr             m2, m6, m4, 8
  mova  [dstq+stride3q ], m2
  lea               dstq, [dstq+strideq*4]
  palignr             m2, m6, m4, 6
  mova  [dstq          ], m2
  palignr             m2, m6, m4, 4
  mova  [dstq+strideq  ], m2
  palignr             m2, m6, m4, 2
  mova  [dstq+strideq*2], m2
  mova  [dstq+stride3q ], m4
  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d153_predictor_32x32, 4, 5, 8, dst, stride, above, left, goffset
  GET_GOT     goffsetq
  mova                  m0, [leftq]
  movu                  m7, [aboveq-1]
  movu                  m1, [aboveq+15]

  pshufb                m4, m1, [GLOBAL(sh_b123456789abcdeff)]
  pshufb                m6, m1, [GLOBAL(sh_b23456789abcdefff)]

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m1, m4, m6, m2          ; 3-tap avg above [high]

  palignr               m3, m1, m7, 1
  palignr               m5, m1, m7, 2

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m7, m3, m5, m1          ; 3-tap avg above [low]

  pshufb                m7, [GLOBAL(sh_bfedcba9876543210)]
  palignr               m5, m0, m7, 15
  palignr               m3, m0, m7, 14

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m5, m3, m4          ; 3-tap avg B3-Bg
  pavgb                 m5, m0                            ; A1 - Ag
  punpcklbw             m6, m4, m5                        ; A-B8 ... A-B1
  punpckhbw             m4, m5                            ; A-B9 ... A-Bg
  pshufb                m6, [GLOBAL(sh_bfedcba9876543210)]
  pshufb                m4, [GLOBAL(sh_bfedcba9876543210)]

  DEFINE_ARGS dst, stride, stride3, left, line
  lea             stride3q, [strideq*3]

  palignr               m5, m2, m1, 14
  palignr               m7, m1, m6, 14
  mova  [dstq            ], m7
  mova  [dstq+16         ], m5
  palignr               m5, m2, m1, 12
  palignr               m7, m1, m6, 12
  mova  [dstq+strideq    ], m7
  mova  [dstq+strideq+16 ], m5
  palignr                m5, m2, m1, 10
  palignr                m7, m1, m6, 10
  mova  [dstq+strideq*2   ], m7
  mova  [dstq+strideq*2+16], m5
  palignr                m5, m2, m1, 8
  palignr                m7, m1, m6, 8
  mova  [dstq+stride3q    ], m7
  mova  [dstq+stride3q+16 ], m5
  lea                  dstq, [dstq+strideq*4]
  palignr                m5, m2, m1, 6
  palignr                m7, m1, m6, 6
  mova  [dstq             ], m7
  mova  [dstq+16          ], m5
  palignr                m5, m2, m1, 4
  palignr                m7, m1, m6, 4
  mova  [dstq+strideq     ], m7
  mova  [dstq+strideq+16  ], m5
  palignr                m5, m2, m1, 2
  palignr                m7, m1, m6, 2
  mova  [dstq+strideq*2   ], m7
  mova  [dstq+strideq*2+16], m5
  mova  [dstq+stride3q    ], m6
  mova  [dstq+stride3q+16 ], m1
  lea                  dstq, [dstq+strideq*4]

  palignr                m5, m1, m6, 14
  palignr                m3, m6, m4, 14
  mova  [dstq             ], m3
  mova  [dstq+16          ], m5
  palignr                m5, m1, m6, 12
  palignr                m3, m6, m4, 12
  mova  [dstq+strideq     ], m3
  mova  [dstq+strideq+16  ], m5
  palignr                m5, m1, m6, 10
  palignr                m3, m6, m4, 10
  mova  [dstq+strideq*2   ], m3
  mova  [dstq+strideq*2+16], m5
  palignr                m5, m1, m6, 8
  palignr                m3, m6, m4, 8
  mova  [dstq+stride3q    ], m3
  mova  [dstq+stride3q+16 ], m5
  lea                  dstq, [dstq+strideq*4]
  palignr                m5, m1, m6, 6
  palignr                m3, m6, m4, 6
  mova  [dstq             ], m3
  mova  [dstq+16          ], m5
  palignr                m5, m1, m6, 4
  palignr                m3, m6, m4, 4
  mova  [dstq+strideq     ], m3
  mova  [dstq+strideq+16  ], m5
  palignr                m5, m1, m6, 2
  palignr                m3, m6, m4, 2
  mova  [dstq+strideq*2   ], m3
  mova  [dstq+strideq*2+16], m5
  mova  [dstq+stride3q    ], m4
  mova  [dstq+stride3q+16 ], m6
  lea               dstq, [dstq+strideq*4]

  mova                   m7, [leftq]
  mova                   m3, [leftq+16]
  palignr                m5, m3, m7, 15
  palignr                m0, m3, m7, 14

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m3, m5, m0, m2          ; 3-tap avg Bh -
  pavgb                  m5, m3                            ; Ah -
  punpcklbw              m3, m2, m5                        ; A-B8 ... A-B1
  punpckhbw              m2, m5                            ; A-B9 ... A-Bg
  pshufb                 m3, [GLOBAL(sh_bfedcba9876543210)]
  pshufb                 m2, [GLOBAL(sh_bfedcba9876543210)]

  palignr                m7, m6, m4, 14
  palignr                m0, m4, m3, 14
  mova  [dstq             ], m0
  mova  [dstq+16          ], m7
  palignr                m7, m6, m4, 12
  palignr                m0, m4, m3, 12
  mova  [dstq+strideq     ], m0
  mova  [dstq+strideq+16  ], m7
  palignr                m7, m6, m4, 10
  palignr                m0, m4, m3, 10
  mova  [dstq+strideq*2   ], m0
  mova  [dstq+strideq*2+16], m7
  palignr                m7, m6, m4, 8
  palignr                m0, m4, m3, 8
  mova  [dstq+stride3q    ], m0
  mova  [dstq+stride3q+16 ], m7
  lea                  dstq, [dstq+strideq*4]
  palignr                m7, m6, m4, 6
  palignr                m0, m4, m3, 6
  mova  [dstq             ], m0
  mova  [dstq+16          ], m7
  palignr                m7, m6, m4, 4
  palignr                m0, m4, m3, 4
  mova  [dstq+strideq     ], m0
  mova  [dstq+strideq+16  ], m7
  palignr                m7, m6, m4, 2
  palignr                m0, m4, m3, 2
  mova  [dstq+strideq*2   ], m0
  mova  [dstq+strideq*2+16], m7
  mova  [dstq+stride3q    ], m3
  mova  [dstq+stride3q+16 ], m4
  lea                  dstq, [dstq+strideq*4]

  palignr                m7, m4, m3, 14
  palignr                m0, m3, m2, 14
  mova  [dstq             ], m0
  mova  [dstq+16          ], m7
  palignr                m7, m4, m3, 12
  palignr                m0, m3, m2, 12
  mova  [dstq+strideq     ], m0
  mova  [dstq+strideq+16  ], m7
  palignr                m7, m4, m3, 10
  palignr                m0, m3, m2, 10
  mova  [dstq+strideq*2   ], m0
  mova  [dstq+strideq*2+16], m7
  palignr                m7, m4, m3, 8
  palignr                m0, m3, m2, 8
  mova  [dstq+stride3q    ], m0
  mova  [dstq+stride3q+16 ], m7
  lea                  dstq, [dstq+strideq*4]
  palignr                m7, m4, m3, 6
  palignr                m0, m3, m2, 6
  mova  [dstq             ], m0
  mova  [dstq+16          ], m7
  palignr                m7, m4, m3, 4
  palignr                m0, m3, m2, 4
  mova  [dstq+strideq     ], m0
  mova  [dstq+strideq+16  ], m7
  palignr                m7, m4, m3, 2
  palignr                m0, m3, m2, 2
  mova  [dstq+strideq*2   ], m0
  mova  [dstq+strideq*2+16], m7
  mova  [dstq+stride3q    ], m2
  mova  [dstq+stride3q+16 ], m3

  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d207_predictor_8x8, 4, 5, 4, dst, stride, stride3, left, goffset
  GET_GOT     goffsetq
  movq                m3, [leftq]            ; abcdefgh [byte]
  lea           stride3q, [strideq*3]

  pshufb              m1, m3, [GLOBAL(sh_b2345677777777777)]
  pshufb              m0, m3, [GLOBAL(sh_b0123456777777777)]
  pshufb              m2, m3, [GLOBAL(sh_b1234567777777777)]

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m2, m1, m3
  pavgb               m0, m2
  punpcklbw           m0, m3        ; interleaved output

  movq  [dstq          ], m0
  psrldq              m0, 2
  movq  [dstq+strideq  ], m0
  psrldq              m0, 2
  movq  [dstq+strideq*2], m0
  psrldq              m0, 2
  movq  [dstq+stride3q ], m0
  lea               dstq, [dstq+strideq*4]
  pshufhw             m0, m0, q0000 ; de, d2ef, ef, e2fg, fg, f2gh, gh, g3h, 8xh
  psrldq              m0, 2
  movq  [dstq          ], m0
  psrldq              m0, 2
  movq  [dstq+strideq  ], m0
  psrldq              m0, 2
  movq  [dstq+strideq*2], m0
  psrldq              m0, 2
  movq  [dstq+stride3q ], m0
  RESTORE_GOT
  RET

INIT_XMM ssse3
cglobal d207_predictor_16x16, 4, 5, 5, dst, stride, stride3, left, goffset
  GET_GOT     goffsetq
  lea           stride3q, [strideq*3]
  mova                m0, [leftq]            ; abcdefghijklmnop [byte]
  pshufb              m1, m0, [GLOBAL(sh_b123456789abcdeff)] ; bcdefghijklmnopp
  pshufb              m2, m0, [GLOBAL(sh_b23456789abcdefff)]

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m0, m1, m2, m3
  pavgb               m1, m0                 ; ab, bc, cd .. no, op, pp [byte]

  punpckhbw           m4, m1, m3    ; interleaved input
  punpcklbw           m1, m3        ; interleaved output
  mova  [dstq          ], m1
  palignr             m3, m4, m1, 2
  mova  [dstq+strideq  ], m3
  palignr             m3, m4, m1, 4
  mova  [dstq+strideq*2], m3
  palignr             m3, m4, m1, 6
  mova  [dstq+stride3q ], m3
  lea               dstq, [dstq+strideq*4]
  palignr             m3, m4, m1, 8
  mova  [dstq          ], m3
  palignr             m3, m4, m1, 10
  mova  [dstq+strideq  ], m3
  palignr             m3, m4, m1, 12
  mova  [dstq+strideq*2], m3
  palignr             m3, m4, m1, 14
  mova  [dstq+stride3q ], m3
  DEFINE_ARGS dst, stride, stride3, line
  mov              lined, 2
  mova                m0, [GLOBAL(sh_b23456789abcdefff)]
.loop:
  lea               dstq, [dstq+strideq*4]
  mova  [dstq          ], m4
  pshufb              m4, m0
  mova  [dstq+strideq  ], m4
  pshufb              m4, m0
  mova  [dstq+strideq*2], m4
  pshufb              m4, m0
  mova  [dstq+stride3q ], m4
  pshufb              m4, m0
  dec              lined
  jnz .loop
  RESTORE_GOT
  REP_RET

INIT_XMM ssse3
cglobal d207_predictor_32x32, 4, 5, 8, dst, stride, stride3, left, goffset
  GET_GOT     goffsetq
  lea           stride3q, [strideq*3]
  mova                m1, [leftq]              ;  0-15 [byte]
  mova                m2, [leftq+16]           ; 16-31 [byte]
  pshufb              m0, m2, [GLOBAL(sh_b23456789abcdefff)]
  pshufb              m4, m2, [GLOBAL(sh_b123456789abcdeff)]

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m2, m4, m0, m3
  palignr             m6, m2, m1, 1
  palignr             m5, m2, m1, 2
  pavgb               m2, m4         ; high 16px even lines

  X_PLUS_2Y_PLUS_Z_PLUS_2_RSH_2 m1, m6, m5, m0
  pavgb                   m1, m6         ; low 16px even lines

  punpckhbw               m6, m1, m0               ; interleaved output 2
  punpcklbw               m1, m0                   ; interleaved output 1

  punpckhbw               m7, m2, m3               ; interleaved output 4
  punpcklbw               m2, m3                   ; interleaved output 3

  ; output 1st 8 lines (and half of 2nd 8 lines)
  DEFINE_ARGS dst, stride, stride3, dst8
  lea                  dst8q, [dstq+strideq*8]
  mova  [dstq              ], m1
  mova  [dstq           +16], m6
  mova  [dst8q             ], m6
  palignr             m0, m6, m1, 2
  palignr             m4, m2, m6, 2
  mova  [dstq +strideq     ], m0
  mova  [dstq +strideq  +16], m4
  mova  [dst8q+strideq     ], m4
  palignr             m0, m6, m1, 4
  palignr             m4, m2, m6, 4
  mova  [dstq +strideq*2   ], m0
  mova  [dstq +strideq*2+16], m4
  mova  [dst8q+strideq*2   ], m4
  palignr             m0, m6, m1, 6
  palignr             m4, m2, m6, 6
  mova  [dstq +stride3q    ], m0
  mova  [dstq +stride3q +16], m4
  mova  [dst8q+stride3q    ], m4
  lea               dstq, [dstq +strideq*4]
  lea              dst8q, [dst8q+strideq*4]
  palignr             m0, m6, m1, 8
  palignr             m4, m2, m6, 8
  mova  [dstq              ], m0
  mova  [dstq           +16], m4
  mova  [dst8q             ], m4
  palignr             m0, m6, m1, 10
  palignr             m4, m2, m6, 10
  mova  [dstq +strideq     ], m0
  mova  [dstq +strideq  +16], m4
  mova  [dst8q+strideq     ], m4
  palignr             m0, m6, m1, 12
  palignr             m4, m2, m6, 12
  mova  [dstq +strideq*2   ], m0
  mova  [dstq +strideq*2+16], m4
  mova  [dst8q+strideq*2   ], m4
  palignr             m0, m6, m1, 14
  palignr             m4, m2, m6, 14
  mova  [dstq +stride3q    ], m0
  mova  [dstq +stride3q +16], m4
  mova  [dst8q+stride3q    ], m4
  lea               dstq, [dstq+strideq*4]
  lea              dst8q, [dst8q+strideq*4]

  ; output 2nd half of 2nd 8 lines and half of 3rd 8 lines
  mova  [dstq           +16], m2
  mova  [dst8q             ], m2
  palignr             m4, m7, m2, 2
  mova  [dstq +strideq  +16], m4
  mova  [dst8q+strideq     ], m4
  palignr             m4, m7, m2, 4
  mova  [dstq +strideq*2+16], m4
  mova  [dst8q+strideq*2   ], m4
  palignr             m4, m7, m2, 6
  mova  [dstq +stride3q +16], m4
  mova  [dst8q+stride3q    ], m4
  lea               dstq, [dstq+strideq*4]
  lea              dst8q, [dst8q+strideq*4]
  palignr             m4, m7, m2, 8
  mova  [dstq           +16], m4
  mova  [dst8q             ], m4
  palignr             m4, m7, m2, 10
  mova  [dstq +strideq  +16], m4
  mova  [dst8q+strideq     ], m4
  palignr             m4, m7, m2, 12
  mova  [dstq +strideq*2+16], m4
  mova  [dst8q+strideq*2   ], m4
  palignr             m4, m7, m2, 14
  mova  [dstq +stride3q +16], m4
  mova  [dst8q+stride3q    ], m4
  lea               dstq, [dstq+strideq*4]
  lea              dst8q, [dst8q+strideq*4]

  ; output 2nd half of 3rd 8 lines and half of 4th 8 lines
  mova                m0, [GLOBAL(sh_b23456789abcdefff)]
  mova  [dstq           +16], m7
  mova  [dst8q             ], m7
  pshufb              m7, m0
  mova  [dstq +strideq  +16], m7
  mova  [dst8q+strideq     ], m7
  pshufb              m7, m0
  mova  [dstq +strideq*2+16], m7
  mova  [dst8q+strideq*2   ], m7
  pshufb              m7, m0
  mova  [dstq +stride3q +16], m7
  mova  [dst8q+stride3q    ], m7
  pshufb              m7, m0
  lea               dstq, [dstq+strideq*4]
  lea              dst8q, [dst8q+strideq*4]
  mova  [dstq           +16], m7
  mova  [dst8q             ], m7
  pshufb              m7, m0
  mova  [dstq +strideq  +16], m7
  mova  [dst8q+strideq     ], m7
  pshufb              m7, m0
  mova  [dstq +strideq*2+16], m7
  mova  [dst8q+strideq*2   ], m7
  pshufb              m7, m0
  mova  [dstq +stride3q +16], m7
  mova  [dst8q+stride3q    ], m7
  pshufb              m7, m0
  lea               dstq, [dstq+strideq*4]

  ; output last half of 4th 8 lines
  mova  [dstq           +16], m7
  mova  [dstq +strideq  +16], m7
  mova  [dstq +strideq*2+16], m7
  mova  [dstq +stride3q +16], m7
  lea               dstq, [dstq+strideq*4]
  mova  [dstq           +16], m7
  mova  [dstq +strideq  +16], m7
  mova  [dstq +strideq*2+16], m7
  mova  [dstq +stride3q +16], m7

  ; done!
  RESTORE_GOT
  RET
