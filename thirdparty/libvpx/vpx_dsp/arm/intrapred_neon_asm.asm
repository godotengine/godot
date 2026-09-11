;
;  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vpx_v_predictor_4x4_neon|
    EXPORT  |vpx_v_predictor_8x8_neon|
    EXPORT  |vpx_v_predictor_16x16_neon|
    EXPORT  |vpx_v_predictor_32x32_neon|
    EXPORT  |vpx_h_predictor_4x4_neon|
    EXPORT  |vpx_h_predictor_8x8_neon|
    EXPORT  |vpx_h_predictor_16x16_neon|
    EXPORT  |vpx_h_predictor_32x32_neon|
    EXPORT  |vpx_tm_predictor_4x4_neon|
    EXPORT  |vpx_tm_predictor_8x8_neon|
    EXPORT  |vpx_tm_predictor_16x16_neon|
    EXPORT  |vpx_tm_predictor_32x32_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

;void vpx_v_predictor_4x4_neon(uint8_t *dst, ptrdiff_t y_stride,
;                              const uint8_t *above,
;                              const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_v_predictor_4x4_neon| PROC
    vld1.32             {d0[0]}, [r2]
    vst1.32             {d0[0]}, [r0], r1
    vst1.32             {d0[0]}, [r0], r1
    vst1.32             {d0[0]}, [r0], r1
    vst1.32             {d0[0]}, [r0], r1
    bx                  lr
    ENDP                ; |vpx_v_predictor_4x4_neon|

;void vpx_v_predictor_8x8_neon(uint8_t *dst, ptrdiff_t y_stride,
;                              const uint8_t *above,
;                              const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_v_predictor_8x8_neon| PROC
    vld1.8              {d0}, [r2]
    vst1.8              {d0}, [r0], r1
    vst1.8              {d0}, [r0], r1
    vst1.8              {d0}, [r0], r1
    vst1.8              {d0}, [r0], r1
    vst1.8              {d0}, [r0], r1
    vst1.8              {d0}, [r0], r1
    vst1.8              {d0}, [r0], r1
    vst1.8              {d0}, [r0], r1
    bx                  lr
    ENDP                ; |vpx_v_predictor_8x8_neon|

;void vpx_v_predictor_16x16_neon(uint8_t *dst, ptrdiff_t y_stride,
;                                const uint8_t *above,
;                                const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_v_predictor_16x16_neon| PROC
    vld1.8              {q0}, [r2]
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    vst1.8              {q0}, [r0], r1
    bx                  lr
    ENDP                ; |vpx_v_predictor_16x16_neon|

;void vpx_v_predictor_32x32_neon(uint8_t *dst, ptrdiff_t y_stride,
;                                const uint8_t *above,
;                                const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_v_predictor_32x32_neon| PROC
    vld1.8              {q0, q1}, [r2]
    mov                 r2, #2
loop_v
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    vst1.8              {q0, q1}, [r0], r1
    subs                r2, r2, #1
    bgt                 loop_v
    bx                  lr
    ENDP                ; |vpx_v_predictor_32x32_neon|

;void vpx_h_predictor_4x4_neon(uint8_t *dst, ptrdiff_t y_stride,
;                              const uint8_t *above,
;                              const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_h_predictor_4x4_neon| PROC
    vld1.32             {d1[0]}, [r3]
    vdup.8              d0, d1[0]
    vst1.32             {d0[0]}, [r0], r1
    vdup.8              d0, d1[1]
    vst1.32             {d0[0]}, [r0], r1
    vdup.8              d0, d1[2]
    vst1.32             {d0[0]}, [r0], r1
    vdup.8              d0, d1[3]
    vst1.32             {d0[0]}, [r0], r1
    bx                  lr
    ENDP                ; |vpx_h_predictor_4x4_neon|

;void vpx_h_predictor_8x8_neon(uint8_t *dst, ptrdiff_t y_stride,
;                              const uint8_t *above,
;                              const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_h_predictor_8x8_neon| PROC
    vld1.64             {d1}, [r3]
    vdup.8              d0, d1[0]
    vst1.64             {d0}, [r0], r1
    vdup.8              d0, d1[1]
    vst1.64             {d0}, [r0], r1
    vdup.8              d0, d1[2]
    vst1.64             {d0}, [r0], r1
    vdup.8              d0, d1[3]
    vst1.64             {d0}, [r0], r1
    vdup.8              d0, d1[4]
    vst1.64             {d0}, [r0], r1
    vdup.8              d0, d1[5]
    vst1.64             {d0}, [r0], r1
    vdup.8              d0, d1[6]
    vst1.64             {d0}, [r0], r1
    vdup.8              d0, d1[7]
    vst1.64             {d0}, [r0], r1
    bx                  lr
    ENDP                ; |vpx_h_predictor_8x8_neon|

;void vpx_h_predictor_16x16_neon(uint8_t *dst, ptrdiff_t y_stride,
;                                const uint8_t *above,
;                                const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_h_predictor_16x16_neon| PROC
    vld1.8              {q1}, [r3]
    vdup.8              q0, d2[0]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[1]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[2]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[3]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[4]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[5]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[6]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[7]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[0]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[1]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[2]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[3]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[4]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[5]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[6]
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[7]
    vst1.8              {q0}, [r0], r1
    bx                  lr
    ENDP                ; |vpx_h_predictor_16x16_neon|

;void vpx_h_predictor_32x32_neon(uint8_t *dst, ptrdiff_t y_stride,
;                                const uint8_t *above,
;                                const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_h_predictor_32x32_neon| PROC
    sub                 r1, r1, #16
    mov                 r2, #2
loop_h
    vld1.8              {q1}, [r3]!
    vdup.8              q0, d2[0]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[1]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[2]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[3]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[4]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[5]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[6]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d2[7]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[0]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[1]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[2]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[3]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[4]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[5]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[6]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    vdup.8              q0, d3[7]
    vst1.8              {q0}, [r0]!
    vst1.8              {q0}, [r0], r1
    subs                r2, r2, #1
    bgt                 loop_h
    bx                  lr
    ENDP                ; |vpx_h_predictor_32x32_neon|

;void vpx_tm_predictor_4x4_neon (uint8_t *dst, ptrdiff_t y_stride,
;                                const uint8_t *above,
;                                const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_tm_predictor_4x4_neon| PROC
    ; Load ytop_left = above[-1];
    sub                 r12, r2, #1
    vld1.u8             {d0[]}, [r12]

    ; Load above 4 pixels
    vld1.32             {d2[0]}, [r2]

    ; Compute above - ytop_left
    vsubl.u8            q3, d2, d0

    ; Load left row by row and compute left + (above - ytop_left)
    ; 1st row and 2nd row
    vld1.u8             {d2[]}, [r3]!
    vld1.u8             {d4[]}, [r3]!
    vmovl.u8            q1, d2
    vmovl.u8            q2, d4
    vadd.s16            q1, q1, q3
    vadd.s16            q2, q2, q3
    vqmovun.s16         d0, q1
    vqmovun.s16         d1, q2
    vst1.32             {d0[0]}, [r0], r1
    vst1.32             {d1[0]}, [r0], r1

    ; 3rd row and 4th row
    vld1.u8             {d2[]}, [r3]!
    vld1.u8             {d4[]}, [r3]
    vmovl.u8            q1, d2
    vmovl.u8            q2, d4
    vadd.s16            q1, q1, q3
    vadd.s16            q2, q2, q3
    vqmovun.s16         d0, q1
    vqmovun.s16         d1, q2
    vst1.32             {d0[0]}, [r0], r1
    vst1.32             {d1[0]}, [r0], r1
    bx                  lr
    ENDP                ; |vpx_tm_predictor_4x4_neon|

;void vpx_tm_predictor_8x8_neon (uint8_t *dst, ptrdiff_t y_stride,
;                                const uint8_t *above,
;                                const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_tm_predictor_8x8_neon| PROC
    ; Load ytop_left = above[-1];
    sub                 r12, r2, #1
    vld1.8              {d0[]}, [r12]

    ; preload 8 left
    vld1.8              {d30}, [r3]

    ; Load above 8 pixels
    vld1.64             {d2}, [r2]

    vmovl.u8            q10, d30

    ; Compute above - ytop_left
    vsubl.u8            q3, d2, d0

    ; Load left row by row and compute left + (above - ytop_left)
    ; 1st row and 2nd row
    vdup.16             q0, d20[0]
    vdup.16             q1, d20[1]
    vadd.s16            q0, q3, q0
    vadd.s16            q1, q3, q1

    ; 3rd row and 4th row
    vdup.16             q8, d20[2]
    vdup.16             q9, d20[3]
    vadd.s16            q8, q3, q8
    vadd.s16            q9, q3, q9

    vqmovun.s16         d0, q0
    vqmovun.s16         d1, q1
    vqmovun.s16         d2, q8
    vqmovun.s16         d3, q9

    vst1.64             {d0}, [r0], r1
    vst1.64             {d1}, [r0], r1
    vst1.64             {d2}, [r0], r1
    vst1.64             {d3}, [r0], r1

    ; 5th row and 6th row
    vdup.16             q0, d21[0]
    vdup.16             q1, d21[1]
    vadd.s16            q0, q3, q0
    vadd.s16            q1, q3, q1

    ; 7th row and 8th row
    vdup.16             q8, d21[2]
    vdup.16             q9, d21[3]
    vadd.s16            q8, q3, q8
    vadd.s16            q9, q3, q9

    vqmovun.s16         d0, q0
    vqmovun.s16         d1, q1
    vqmovun.s16         d2, q8
    vqmovun.s16         d3, q9

    vst1.64             {d0}, [r0], r1
    vst1.64             {d1}, [r0], r1
    vst1.64             {d2}, [r0], r1
    vst1.64             {d3}, [r0], r1

    bx                  lr
    ENDP                ; |vpx_tm_predictor_8x8_neon|

;void vpx_tm_predictor_16x16_neon (uint8_t *dst, ptrdiff_t y_stride,
;                                const uint8_t *above,
;                                const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_tm_predictor_16x16_neon| PROC
    ; Load ytop_left = above[-1];
    sub                 r12, r2, #1
    vld1.8              {d0[]}, [r12]

    ; Load above 8 pixels
    vld1.8              {q1}, [r2]

    ; preload 8 left into r12
    vld1.8              {d18}, [r3]!

    ; Compute above - ytop_left
    vsubl.u8            q2, d2, d0
    vsubl.u8            q3, d3, d0

    vmovl.u8            q10, d18

    ; Load left row by row and compute left + (above - ytop_left)
    ; Process 8 rows in each single loop and loop 2 times to process 16 rows.
    mov                 r2, #2

loop_16x16_neon
    ; Process two rows.
    vdup.16             q0, d20[0]
    vdup.16             q8, d20[1]
    vadd.s16            q1, q0, q2
    vadd.s16            q0, q0, q3
    vadd.s16            q11, q8, q2
    vadd.s16            q8, q8, q3
    vqmovun.s16         d2, q1
    vqmovun.s16         d3, q0
    vqmovun.s16         d22, q11
    vqmovun.s16         d23, q8
    vdup.16             q0, d20[2]                  ; proload next 2 rows data
    vdup.16             q8, d20[3]
    vst1.64             {d2,d3}, [r0], r1
    vst1.64             {d22,d23}, [r0], r1

    ; Process two rows.
    vadd.s16            q1, q0, q2
    vadd.s16            q0, q0, q3
    vadd.s16            q11, q8, q2
    vadd.s16            q8, q8, q3
    vqmovun.s16         d2, q1
    vqmovun.s16         d3, q0
    vqmovun.s16         d22, q11
    vqmovun.s16         d23, q8
    vdup.16             q0, d21[0]                  ; proload next 2 rows data
    vdup.16             q8, d21[1]
    vst1.64             {d2,d3}, [r0], r1
    vst1.64             {d22,d23}, [r0], r1

    vadd.s16            q1, q0, q2
    vadd.s16            q0, q0, q3
    vadd.s16            q11, q8, q2
    vadd.s16            q8, q8, q3
    vqmovun.s16         d2, q1
    vqmovun.s16         d3, q0
    vqmovun.s16         d22, q11
    vqmovun.s16         d23, q8
    vdup.16             q0, d21[2]                  ; proload next 2 rows data
    vdup.16             q8, d21[3]
    vst1.64             {d2,d3}, [r0], r1
    vst1.64             {d22,d23}, [r0], r1


    vadd.s16            q1, q0, q2
    vadd.s16            q0, q0, q3
    vadd.s16            q11, q8, q2
    vadd.s16            q8, q8, q3
    vqmovun.s16         d2, q1
    vqmovun.s16         d3, q0
    vqmovun.s16         d22, q11
    vqmovun.s16         d23, q8
    vld1.8              {d18}, [r3]!                  ; preload 8 left into r12
    vmovl.u8            q10, d18
    vst1.64             {d2,d3}, [r0], r1
    vst1.64             {d22,d23}, [r0], r1

    subs                r2, r2, #1
    bgt                 loop_16x16_neon

    bx                  lr
    ENDP                ; |vpx_tm_predictor_16x16_neon|

;void vpx_tm_predictor_32x32_neon (uint8_t *dst, ptrdiff_t y_stride,
;                                  const uint8_t *above,
;                                  const uint8_t *left)
; r0  uint8_t *dst
; r1  ptrdiff_t y_stride
; r2  const uint8_t *above
; r3  const uint8_t *left

|vpx_tm_predictor_32x32_neon| PROC
    ; Load ytop_left = above[-1];
    sub                 r12, r2, #1
    vld1.8              {d0[]}, [r12]

    ; Load above 32 pixels
    vld1.8              {q1}, [r2]!
    vld1.8              {q2}, [r2]

    ; preload 8 left pixels
    vld1.8              {d26}, [r3]!

    ; Compute above - ytop_left
    vsubl.u8            q8, d2, d0
    vsubl.u8            q9, d3, d0
    vsubl.u8            q10, d4, d0
    vsubl.u8            q11, d5, d0

    vmovl.u8            q3, d26

    ; Load left row by row and compute left + (above - ytop_left)
    ; Process 8 rows in each single loop and loop 4 times to process 32 rows.
    mov                 r2, #4

loop_32x32_neon
    ; Process two rows.
    vdup.16             q0, d6[0]
    vdup.16             q2, d6[1]
    vadd.s16            q12, q0, q8
    vadd.s16            q13, q0, q9
    vadd.s16            q14, q0, q10
    vadd.s16            q15, q0, q11
    vqmovun.s16         d0, q12
    vqmovun.s16         d1, q13
    vadd.s16            q12, q2, q8
    vadd.s16            q13, q2, q9
    vqmovun.s16         d2, q14
    vqmovun.s16         d3, q15
    vadd.s16            q14, q2, q10
    vadd.s16            q15, q2, q11
    vst1.64             {d0-d3}, [r0], r1
    vqmovun.s16         d24, q12
    vqmovun.s16         d25, q13
    vqmovun.s16         d26, q14
    vqmovun.s16         d27, q15
    vdup.16             q1, d6[2]
    vdup.16             q2, d6[3]
    vst1.64             {d24-d27}, [r0], r1

    ; Process two rows.
    vadd.s16            q12, q1, q8
    vadd.s16            q13, q1, q9
    vadd.s16            q14, q1, q10
    vadd.s16            q15, q1, q11
    vqmovun.s16         d0, q12
    vqmovun.s16         d1, q13
    vadd.s16            q12, q2, q8
    vadd.s16            q13, q2, q9
    vqmovun.s16         d2, q14
    vqmovun.s16         d3, q15
    vadd.s16            q14, q2, q10
    vadd.s16            q15, q2, q11
    vst1.64             {d0-d3}, [r0], r1
    vqmovun.s16         d24, q12
    vqmovun.s16         d25, q13
    vqmovun.s16         d26, q14
    vqmovun.s16         d27, q15
    vdup.16             q0, d7[0]
    vdup.16             q2, d7[1]
    vst1.64             {d24-d27}, [r0], r1

    ; Process two rows.
    vadd.s16            q12, q0, q8
    vadd.s16            q13, q0, q9
    vadd.s16            q14, q0, q10
    vadd.s16            q15, q0, q11
    vqmovun.s16         d0, q12
    vqmovun.s16         d1, q13
    vadd.s16            q12, q2, q8
    vadd.s16            q13, q2, q9
    vqmovun.s16         d2, q14
    vqmovun.s16         d3, q15
    vadd.s16            q14, q2, q10
    vadd.s16            q15, q2, q11
    vst1.64             {d0-d3}, [r0], r1
    vqmovun.s16         d24, q12
    vqmovun.s16         d25, q13
    vqmovun.s16         d26, q14
    vqmovun.s16         d27, q15
    vdup.16             q0, d7[2]
    vdup.16             q2, d7[3]
    vst1.64             {d24-d27}, [r0], r1

    ; Process two rows.
    vadd.s16            q12, q0, q8
    vadd.s16            q13, q0, q9
    vadd.s16            q14, q0, q10
    vadd.s16            q15, q0, q11
    vqmovun.s16         d0, q12
    vqmovun.s16         d1, q13
    vadd.s16            q12, q2, q8
    vadd.s16            q13, q2, q9
    vqmovun.s16         d2, q14
    vqmovun.s16         d3, q15
    vadd.s16            q14, q2, q10
    vadd.s16            q15, q2, q11
    vst1.64             {d0-d3}, [r0], r1
    vqmovun.s16         d24, q12
    vqmovun.s16         d25, q13
    vld1.8              {d0}, [r3]!                   ; preload 8 left pixels
    vqmovun.s16         d26, q14
    vqmovun.s16         d27, q15
    vmovl.u8            q3, d0
    vst1.64             {d24-d27}, [r0], r1

    subs                r2, r2, #1
    bgt                 loop_32x32_neon

    bx                  lr
    ENDP                ; |vpx_tm_predictor_32x32_neon|

    END
