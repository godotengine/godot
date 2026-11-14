;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license and patent
;  grant that can be found in the LICENSE file in the root of the source
;  tree. All contributing project authors may be found in the AUTHORS
;  file in the root of the source tree.
;


    EXPORT  |vpx_idct4x4_1_add_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

;void vpx_idct4x4_1_add_neon(int16_t *input, uint8_t *dest, int stride)
;
; r0  int16_t input
; r1  uint8_t *dest
; r2  int stride)

|vpx_idct4x4_1_add_neon| PROC
    ldrsh            r0, [r0]

    ; cospi_16_64 = 11585
    movw             r12, #0x2d41

    ; out = dct_const_round_shift(input[0] * cospi_16_64)
    mul              r0, r0, r12               ; input[0] * cospi_16_64
    add              r0, r0, #0x2000           ; +(1 << ((DCT_CONST_BITS) - 1))
    asr              r0, r0, #14               ; >> DCT_CONST_BITS

    ; out = dct_const_round_shift(out * cospi_16_64)
    mul              r0, r0, r12               ; out * cospi_16_64
    mov              r12, r1                   ; save dest
    add              r0, r0, #0x2000           ; +(1 << ((DCT_CONST_BITS) - 1))
    asr              r0, r0, #14               ; >> DCT_CONST_BITS

    ; a1 = ROUND_POWER_OF_TWO(out, 4)
    add              r0, r0, #8                ; + (1 <<((4) - 1))
    asr              r0, r0, #4                ; >> 4

    vdup.s16         q0, r0                    ; duplicate a1

    vld1.32          {d2[0]}, [r1], r2
    vld1.32          {d2[1]}, [r1], r2
    vld1.32          {d4[0]}, [r1], r2
    vld1.32          {d4[1]}, [r1]

    vaddw.u8         q8, q0, d2                ; dest[x] + a1
    vaddw.u8         q9, q0, d4

    vqmovun.s16      d6, q8                    ; clip_pixel
    vqmovun.s16      d7, q9

    vst1.32          {d6[0]}, [r12], r2
    vst1.32          {d6[1]}, [r12], r2
    vst1.32          {d7[0]}, [r12], r2
    vst1.32          {d7[1]}, [r12]

    bx               lr
    ENDP             ; |vpx_idct4x4_1_add_neon|

    END
