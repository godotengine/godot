;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vpx_idct4x4_16_add_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

    INCLUDE vpx_dsp/arm/idct_neon.asm.S

    AREA     Block, CODE, READONLY
;void vpx_idct4x4_16_add_neon(int16_t *input, uint8_t *dest, int stride)
;
; r0  int16_t input
; r1  uint8_t *dest
; r2  int stride)

|vpx_idct4x4_16_add_neon| PROC

    ; The 2D transform is done with two passes which are actually pretty
    ; similar. We first transform the rows. This is done by transposing
    ; the inputs, doing an SIMD column transform (the columns are the
    ; transposed rows) and then transpose the results (so that it goes back
    ; in normal/row positions). Then, we transform the columns by doing
    ; another SIMD column transform.
    ; So, two passes of a transpose followed by a column transform.

    ; load the inputs into q8-q9, d16-d19
    LOAD_TRAN_LOW_TO_S16 d16, d17, d18, d19, r0

    ; generate scalar constants
    ; cospi_8_64 = 15137
    movw            r0, #0x3b21
    ; cospi_16_64 = 11585
    movw            r3, #0x2d41
    ; cospi_24_64 = 6270
    movw            r12, #0x187e

    ; transpose the input data
    ; 00 01 02 03   d16
    ; 10 11 12 13   d17
    ; 20 21 22 23   d18
    ; 30 31 32 33   d19
    vtrn.16         d16, d17
    vtrn.16         d18, d19

    ; generate constant vectors
    vdup.16         d20, r0         ; replicate cospi_8_64
    vdup.16         d21, r3         ; replicate cospi_16_64

    ; 00 10 02 12   d16
    ; 01 11 03 13   d17
    ; 20 30 22 32   d18
    ; 21 31 23 33   d19
    vtrn.32         q8, q9
    ; 00 10 20 30   d16
    ; 01 11 21 31   d17
    ; 02 12 22 32   d18
    ; 03 13 23 33   d19

    vdup.16         d22, r12        ; replicate cospi_24_64

    ; do the transform on transposed rows

    ; stage 1
    vmull.s16 q15, d17, d22         ; input[1] * cospi_24_64
    vmull.s16 q1,  d17, d20         ; input[1] * cospi_8_64

    ; (input[0] + input[2]) * cospi_16_64;
    ; (input[0] - input[2]) * cospi_16_64;
    vmull.s16 q8,  d16, d21
    vmull.s16 q14, d18, d21
    vadd.s32  q13, q8,  q14
    vsub.s32  q14, q8,  q14

    ; input[1] * cospi_24_64 - input[3] * cospi_8_64;
    ; input[1] * cospi_8_64  + input[3] * cospi_24_64;
    vmlsl.s16 q15, d19, d20
    vmlal.s16 q1,  d19, d22

    ; dct_const_round_shift
    vrshrn.s32 d26, q13, #14
    vrshrn.s32 d27, q14, #14
    vrshrn.s32 d29, q15, #14
    vrshrn.s32 d28, q1,  #14

    ; stage 2
    ; output[0] = step[0] + step[3];
    ; output[1] = step[1] + step[2];
    ; output[3] = step[0] - step[3];
    ; output[2] = step[1] - step[2];
    vadd.s16 q8,  q13, q14
    vsub.s16 q9,  q13, q14
    vswp     d18, d19

    ; transpose the results
    ; 00 01 02 03   d16
    ; 10 11 12 13   d17
    ; 20 21 22 23   d18
    ; 30 31 32 33   d19
    vtrn.16         d16, d17
    vtrn.16         d18, d19
    ; 00 10 02 12   d16
    ; 01 11 03 13   d17
    ; 20 30 22 32   d18
    ; 21 31 23 33   d19
    vtrn.32         q8, q9
    ; 00 10 20 30   d16
    ; 01 11 21 31   d17
    ; 02 12 22 32   d18
    ; 03 13 23 33   d19

    ; do the transform on columns

    ; stage 1
    vadd.s16  d23, d16, d18         ; (input[0] + input[2])
    vsub.s16  d24, d16, d18         ; (input[0] - input[2])

    vmull.s16 q15, d17, d22         ; input[1] * cospi_24_64
    vmull.s16 q1,  d17, d20         ; input[1] * cospi_8_64

    ; (input[0] + input[2]) * cospi_16_64;
    ; (input[0] - input[2]) * cospi_16_64;
    vmull.s16 q13, d23, d21
    vmull.s16 q14, d24, d21

    ; input[1] * cospi_24_64 - input[3] * cospi_8_64;
    ; input[1] * cospi_8_64  + input[3] * cospi_24_64;
    vmlsl.s16 q15, d19, d20
    vmlal.s16 q1,  d19, d22

    ; dct_const_round_shift
    vrshrn.s32 d26, q13, #14
    vrshrn.s32 d27, q14, #14
    vrshrn.s32 d29, q15, #14
    vrshrn.s32 d28, q1,  #14

    ; stage 2
    ; output[0] = step[0] + step[3];
    ; output[1] = step[1] + step[2];
    ; output[3] = step[0] - step[3];
    ; output[2] = step[1] - step[2];
    vadd.s16 q8,  q13, q14
    vsub.s16 q9,  q13, q14

    ; The results are in two registers, one of them being swapped. This will
    ; be taken care of by loading the 'dest' value in a swapped fashion and
    ; also storing them in the same swapped fashion.
    ; temp_out[0, 1] = d16, d17 = q8
    ; temp_out[2, 3] = d19, d18 = q9 swapped

    ; ROUND_POWER_OF_TWO(temp_out[j], 4)
    vrshr.s16 q8, q8, #4
    vrshr.s16 q9, q9, #4

    vld1.32 {d26[0]}, [r1], r2
    vld1.32 {d26[1]}, [r1], r2
    vld1.32 {d27[1]}, [r1], r2
    vld1.32 {d27[0]}, [r1]  ; no post-increment

    ; ROUND_POWER_OF_TWO(temp_out[j], 4) + dest[j * stride + i]
    vaddw.u8 q8, q8, d26
    vaddw.u8 q9, q9, d27

    ; clip_pixel
    vqmovun.s16 d26, q8
    vqmovun.s16 d27, q9

    ; do the stores in reverse order with negative post-increment, by changing
    ; the sign of the stride
    rsb r2, r2, #0
    vst1.32 {d27[0]}, [r1], r2
    vst1.32 {d27[1]}, [r1], r2
    vst1.32 {d26[1]}, [r1], r2
    vst1.32 {d26[0]}, [r1]  ; no post-increment
    bx              lr
    ENDP  ; |vpx_idct4x4_16_add_neon|

    END
