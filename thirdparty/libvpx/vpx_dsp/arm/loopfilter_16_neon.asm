;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vpx_lpf_horizontal_16_neon|
    EXPORT  |vpx_lpf_horizontal_16_dual_neon|
    EXPORT  |vpx_lpf_vertical_16_neon|
    EXPORT  |vpx_lpf_vertical_16_dual_neon|
    ARM

    AREA ||.text||, CODE, READONLY, ALIGN=2

; void mb_lpf_horizontal_edge(uint8_t *s, int p,
;                             const uint8_t *blimit,
;                             const uint8_t *limit,
;                             const uint8_t *thresh,
;                             int count)
; r0    uint8_t *s,
; r1    int p, /* pitch */
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh,
; r12   int count
|mb_lpf_horizontal_edge| PROC
    push        {r4-r8, lr}
    vpush       {d8-d15}
    ldr         r4, [sp, #88]              ; load thresh

h_count
    vld1.8      {d16[]}, [r2]              ; load *blimit
    vld1.8      {d17[]}, [r3]              ; load *limit
    vld1.8      {d18[]}, [r4]              ; load *thresh

    sub         r8, r0, r1, lsl #3         ; move src pointer down by 8 lines

    vld1.u8     {d0}, [r8@64], r1          ; p7
    vld1.u8     {d1}, [r8@64], r1          ; p6
    vld1.u8     {d2}, [r8@64], r1          ; p5
    vld1.u8     {d3}, [r8@64], r1          ; p4
    vld1.u8     {d4}, [r8@64], r1          ; p3
    vld1.u8     {d5}, [r8@64], r1          ; p2
    vld1.u8     {d6}, [r8@64], r1          ; p1
    vld1.u8     {d7}, [r8@64], r1          ; p0
    vld1.u8     {d8}, [r8@64], r1          ; q0
    vld1.u8     {d9}, [r8@64], r1          ; q1
    vld1.u8     {d10}, [r8@64], r1         ; q2
    vld1.u8     {d11}, [r8@64], r1         ; q3
    vld1.u8     {d12}, [r8@64], r1         ; q4
    vld1.u8     {d13}, [r8@64], r1         ; q5
    vld1.u8     {d14}, [r8@64], r1         ; q6
    vld1.u8     {d15}, [r8@64], r1         ; q7

    bl          vpx_wide_mbfilter_neon

    tst         r7, #1
    beq         h_mbfilter

    ; flat && mask were not set for any of the channels. Just store the values
    ; from filter.
    sub         r8, r0, r1, lsl #1

    vst1.u8     {d25}, [r8@64], r1         ; store op1
    vst1.u8     {d24}, [r8@64], r1         ; store op0
    vst1.u8     {d23}, [r8@64], r1         ; store oq0
    vst1.u8     {d26}, [r8@64], r1         ; store oq1

    b           h_next

h_mbfilter
    tst         r7, #2
    beq         h_wide_mbfilter

    ; flat2 was not set for any of the channels. Just store the values from
    ; mbfilter.
    sub         r8, r0, r1, lsl #1
    sub         r8, r8, r1

    vst1.u8     {d18}, [r8@64], r1         ; store op2
    vst1.u8     {d19}, [r8@64], r1         ; store op1
    vst1.u8     {d20}, [r8@64], r1         ; store op0
    vst1.u8     {d21}, [r8@64], r1         ; store oq0
    vst1.u8     {d22}, [r8@64], r1         ; store oq1
    vst1.u8     {d23}, [r8@64], r1         ; store oq2

    b           h_next

h_wide_mbfilter
    sub         r8, r0, r1, lsl #3
    add         r8, r8, r1

    vst1.u8     {d16}, [r8@64], r1         ; store op6
    vst1.u8     {d24}, [r8@64], r1         ; store op5
    vst1.u8     {d25}, [r8@64], r1         ; store op4
    vst1.u8     {d26}, [r8@64], r1         ; store op3
    vst1.u8     {d27}, [r8@64], r1         ; store op2
    vst1.u8     {d18}, [r8@64], r1         ; store op1
    vst1.u8     {d19}, [r8@64], r1         ; store op0
    vst1.u8     {d20}, [r8@64], r1         ; store oq0
    vst1.u8     {d21}, [r8@64], r1         ; store oq1
    vst1.u8     {d22}, [r8@64], r1         ; store oq2
    vst1.u8     {d23}, [r8@64], r1         ; store oq3
    vst1.u8     {d1}, [r8@64], r1          ; store oq4
    vst1.u8     {d2}, [r8@64], r1          ; store oq5
    vst1.u8     {d3}, [r8@64], r1          ; store oq6

h_next
    add         r0, r0, #8
    subs        r12, r12, #1
    bne         h_count

    vpop        {d8-d15}
    pop         {r4-r8, pc}

    ENDP        ; |mb_lpf_horizontal_edge|

; void vpx_lpf_horizontal_16_neon(uint8_t *s, int pitch,
;                                     const uint8_t *blimit,
;                                     const uint8_t *limit,
;                                     const uint8_t *thresh)
; r0    uint8_t *s,
; r1    int pitch,
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh
|vpx_lpf_horizontal_16_neon| PROC
    mov r12, #1
    b mb_lpf_horizontal_edge
    ENDP        ; |vpx_lpf_horizontal_16_neon|

; void vpx_lpf_horizontal_16_dual_neon(uint8_t *s, int pitch,
;                                      const uint8_t *blimit,
;                                      const uint8_t *limit,
;                                      const uint8_t *thresh)
; r0    uint8_t *s,
; r1    int pitch,
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh
|vpx_lpf_horizontal_16_dual_neon| PROC
    mov r12, #2
    b mb_lpf_horizontal_edge
    ENDP        ; |vpx_lpf_horizontal_16_dual_neon|

; void mb_lpf_vertical_edge_w(uint8_t *s, int p, const uint8_t *blimit,
;                             const uint8_t *limit, const uint8_t *thresh,
;                             int count) {
; r0    uint8_t *s,
; r1    int p, /* pitch */
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh,
; r12   int count
|mb_lpf_vertical_edge_w| PROC
    push        {r4-r8, lr}
    vpush       {d8-d15}
    ldr         r4, [sp, #88]              ; load thresh

v_count
    vld1.8      {d16[]}, [r2]              ; load *blimit
    vld1.8      {d17[]}, [r3]              ; load *limit
    vld1.8      {d18[]}, [r4]              ; load *thresh

    sub         r8, r0, #8

    vld1.8      {d0}, [r8@64], r1
    vld1.8      {d8}, [r0@64], r1
    vld1.8      {d1}, [r8@64], r1
    vld1.8      {d9}, [r0@64], r1
    vld1.8      {d2}, [r8@64], r1
    vld1.8      {d10}, [r0@64], r1
    vld1.8      {d3}, [r8@64], r1
    vld1.8      {d11}, [r0@64], r1
    vld1.8      {d4}, [r8@64], r1
    vld1.8      {d12}, [r0@64], r1
    vld1.8      {d5}, [r8@64], r1
    vld1.8      {d13}, [r0@64], r1
    vld1.8      {d6}, [r8@64], r1
    vld1.8      {d14}, [r0@64], r1
    vld1.8      {d7}, [r8@64], r1
    vld1.8      {d15}, [r0@64], r1

    sub         r0, r0, r1, lsl #3

    vtrn.32     q0, q2
    vtrn.32     q1, q3
    vtrn.32     q4, q6
    vtrn.32     q5, q7

    vtrn.16     q0, q1
    vtrn.16     q2, q3
    vtrn.16     q4, q5
    vtrn.16     q6, q7

    vtrn.8      d0, d1
    vtrn.8      d2, d3
    vtrn.8      d4, d5
    vtrn.8      d6, d7

    vtrn.8      d8, d9
    vtrn.8      d10, d11
    vtrn.8      d12, d13
    vtrn.8      d14, d15

    bl          vpx_wide_mbfilter_neon

    tst         r7, #1
    beq         v_mbfilter

    ; flat && mask were not set for any of the channels. Just store the values
    ; from filter.
    sub         r0, #2

    vswp        d23, d25

    vst4.8      {d23[0], d24[0], d25[0], d26[0]}, [r0], r1
    vst4.8      {d23[1], d24[1], d25[1], d26[1]}, [r0], r1
    vst4.8      {d23[2], d24[2], d25[2], d26[2]}, [r0], r1
    vst4.8      {d23[3], d24[3], d25[3], d26[3]}, [r0], r1
    vst4.8      {d23[4], d24[4], d25[4], d26[4]}, [r0], r1
    vst4.8      {d23[5], d24[5], d25[5], d26[5]}, [r0], r1
    vst4.8      {d23[6], d24[6], d25[6], d26[6]}, [r0], r1
    vst4.8      {d23[7], d24[7], d25[7], d26[7]}, [r0], r1
    add         r0, #2

    b           v_next

v_mbfilter
    tst         r7, #2
    beq         v_wide_mbfilter

    ; flat2 was not set for any of the channels. Just store the values from
    ; mbfilter.
    sub         r8, r0, #3

    vst3.8      {d18[0], d19[0], d20[0]}, [r8], r1
    vst3.8      {d21[0], d22[0], d23[0]}, [r0], r1
    vst3.8      {d18[1], d19[1], d20[1]}, [r8], r1
    vst3.8      {d21[1], d22[1], d23[1]}, [r0], r1
    vst3.8      {d18[2], d19[2], d20[2]}, [r8], r1
    vst3.8      {d21[2], d22[2], d23[2]}, [r0], r1
    vst3.8      {d18[3], d19[3], d20[3]}, [r8], r1
    vst3.8      {d21[3], d22[3], d23[3]}, [r0], r1
    vst3.8      {d18[4], d19[4], d20[4]}, [r8], r1
    vst3.8      {d21[4], d22[4], d23[4]}, [r0], r1
    vst3.8      {d18[5], d19[5], d20[5]}, [r8], r1
    vst3.8      {d21[5], d22[5], d23[5]}, [r0], r1
    vst3.8      {d18[6], d19[6], d20[6]}, [r8], r1
    vst3.8      {d21[6], d22[6], d23[6]}, [r0], r1
    vst3.8      {d18[7], d19[7], d20[7]}, [r8], r1
    vst3.8      {d21[7], d22[7], d23[7]}, [r0], r1

    b           v_next

v_wide_mbfilter
    sub         r8, r0, #8

    vtrn.32     d0,  d26
    vtrn.32     d16, d27
    vtrn.32     d24, d18
    vtrn.32     d25, d19

    vtrn.16     d0,  d24
    vtrn.16     d16, d25
    vtrn.16     d26, d18
    vtrn.16     d27, d19

    vtrn.8      d0,  d16
    vtrn.8      d24, d25
    vtrn.8      d26, d27
    vtrn.8      d18, d19

    vtrn.32     d20, d1
    vtrn.32     d21, d2
    vtrn.32     d22, d3
    vtrn.32     d23, d15

    vtrn.16     d20, d22
    vtrn.16     d21, d23
    vtrn.16     d1,  d3
    vtrn.16     d2,  d15

    vtrn.8      d20, d21
    vtrn.8      d22, d23
    vtrn.8      d1,  d2
    vtrn.8      d3,  d15

    vst1.8      {d0}, [r8@64], r1
    vst1.8      {d20}, [r0@64], r1
    vst1.8      {d16}, [r8@64], r1
    vst1.8      {d21}, [r0@64], r1
    vst1.8      {d24}, [r8@64], r1
    vst1.8      {d22}, [r0@64], r1
    vst1.8      {d25}, [r8@64], r1
    vst1.8      {d23}, [r0@64], r1
    vst1.8      {d26}, [r8@64], r1
    vst1.8      {d1}, [r0@64], r1
    vst1.8      {d27}, [r8@64], r1
    vst1.8      {d2}, [r0@64], r1
    vst1.8      {d18}, [r8@64], r1
    vst1.8      {d3}, [r0@64], r1
    vst1.8      {d19}, [r8@64], r1
    vst1.8      {d15}, [r0@64], r1

v_next
    subs        r12, #1
    bne         v_count

    vpop        {d8-d15}
    pop         {r4-r8, pc}

    ENDP        ; |mb_lpf_vertical_edge_w|

; void vpx_lpf_vertical_16_neon(uint8_t *s, int p, const uint8_t *blimit,
;                               const uint8_t *limit, const uint8_t *thresh)
; r0    uint8_t *s,
; r1    int p, /* pitch */
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh
|vpx_lpf_vertical_16_neon| PROC
    mov r12, #1
    b mb_lpf_vertical_edge_w
    ENDP        ; |vpx_lpf_vertical_16_neon|

; void vpx_lpf_vertical_16_dual_neon(uint8_t *s, int p, const uint8_t *blimit,
;                                    const uint8_t *limit,
;                                    const uint8_t *thresh)
; r0    uint8_t *s,
; r1    int p, /* pitch */
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh
|vpx_lpf_vertical_16_dual_neon| PROC
    mov r12, #2
    b mb_lpf_vertical_edge_w
    ENDP        ; |vpx_lpf_vertical_16_dual_neon|

; void vpx_wide_mbfilter_neon();
; This is a helper function for the loopfilters. The invidual functions do the
; necessary load, transpose (if necessary) and store.
;
; r0-r3 PRESERVE
; d16    blimit
; d17    limit
; d18    thresh
; d0    p7
; d1    p6
; d2    p5
; d3    p4
; d4    p3
; d5    p2
; d6    p1
; d7    p0
; d8    q0
; d9    q1
; d10   q2
; d11   q3
; d12   q4
; d13   q5
; d14   q6
; d15   q7
|vpx_wide_mbfilter_neon| PROC
    mov         r7, #0

    ; filter_mask
    vabd.u8     d19, d4, d5                ; abs(p3 - p2)
    vabd.u8     d20, d5, d6                ; abs(p2 - p1)
    vabd.u8     d21, d6, d7                ; abs(p1 - p0)
    vabd.u8     d22, d9, d8                ; abs(q1 - q0)
    vabd.u8     d23, d10, d9               ; abs(q2 - q1)
    vabd.u8     d24, d11, d10              ; abs(q3 - q2)

    ; only compare the largest value to limit
    vmax.u8     d19, d19, d20              ; max(abs(p3 - p2), abs(p2 - p1))
    vmax.u8     d20, d21, d22              ; max(abs(p1 - p0), abs(q1 - q0))
    vmax.u8     d23, d23, d24              ; max(abs(q2 - q1), abs(q3 - q2))
    vmax.u8     d19, d19, d20

    vabd.u8     d24, d7, d8                ; abs(p0 - q0)

    vmax.u8     d19, d19, d23

    vabd.u8     d23, d6, d9                ; a = abs(p1 - q1)
    vqadd.u8    d24, d24, d24              ; b = abs(p0 - q0) * 2

    ; abs () > limit
    vcge.u8     d19, d17, d19

    ; flatmask4
    vabd.u8     d25, d7, d5                ; abs(p0 - p2)
    vabd.u8     d26, d8, d10               ; abs(q0 - q2)
    vabd.u8     d27, d4, d7                ; abs(p3 - p0)
    vabd.u8     d28, d11, d8               ; abs(q3 - q0)

    ; only compare the largest value to thresh
    vmax.u8     d25, d25, d26              ; max(abs(p0 - p2), abs(q0 - q2))
    vmax.u8     d26, d27, d28              ; max(abs(p3 - p0), abs(q3 - q0))
    vmax.u8     d25, d25, d26
    vmax.u8     d20, d20, d25

    vshr.u8     d23, d23, #1               ; a = a / 2
    vqadd.u8    d24, d24, d23              ; a = b + a

    vmov.u8     d30, #1
    vcge.u8     d24, d16, d24              ; (a > blimit * 2 + limit) * -1

    vcge.u8     d20, d30, d20              ; flat

    vand        d19, d19, d24              ; mask

    ; hevmask
    vcgt.u8     d21, d21, d18              ; (abs(p1 - p0) > thresh)*-1
    vcgt.u8     d22, d22, d18              ; (abs(q1 - q0) > thresh)*-1
    vorr        d21, d21, d22              ; hev

    vand        d16, d20, d19              ; flat && mask
    vmov        r5, r6, d16

    ; flatmask5(1, p7, p6, p5, p4, p0, q0, q4, q5, q6, q7)
    vabd.u8     d22, d3, d7                ; abs(p4 - p0)
    vabd.u8     d23, d12, d8               ; abs(q4 - q0)
    vabd.u8     d24, d7, d2                ; abs(p0 - p5)
    vabd.u8     d25, d8, d13               ; abs(q0 - q5)
    vabd.u8     d26, d1, d7                ; abs(p6 - p0)
    vabd.u8     d27, d14, d8               ; abs(q6 - q0)
    vabd.u8     d28, d0, d7                ; abs(p7 - p0)
    vabd.u8     d29, d15, d8               ; abs(q7 - q0)

    ; only compare the largest value to thresh
    vmax.u8     d22, d22, d23              ; max(abs(p4 - p0), abs(q4 - q0))
    vmax.u8     d23, d24, d25              ; max(abs(p0 - p5), abs(q0 - q5))
    vmax.u8     d24, d26, d27              ; max(abs(p6 - p0), abs(q6 - q0))
    vmax.u8     d25, d28, d29              ; max(abs(p7 - p0), abs(q7 - q0))

    vmax.u8     d26, d22, d23
    vmax.u8     d27, d24, d25
    vmax.u8     d23, d26, d27

    vcge.u8     d18, d30, d23              ; flat2

    vmov.u8     d22, #0x80

    orrs        r5, r5, r6                 ; Check for 0
    orreq       r7, r7, #1                 ; Only do filter branch

    vand        d17, d18, d16              ; flat2 && flat && mask
    vmov        r5, r6, d17

    ; mbfilter() function

    ; filter() function
    ; convert to signed
    veor        d23, d8, d22               ; qs0
    veor        d24, d7, d22               ; ps0
    veor        d25, d6, d22               ; ps1
    veor        d26, d9, d22               ; qs1

    vmov.u8     d27, #3

    vsub.s8     d28, d23, d24              ; ( qs0 - ps0)
    vqsub.s8    d29, d25, d26              ; filter = clamp(ps1-qs1)
    vmull.s8    q15, d28, d27              ; 3 * ( qs0 - ps0)
    vand        d29, d29, d21              ; filter &= hev
    vaddw.s8    q15, q15, d29              ; filter + 3 * (qs0 - ps0)
    vmov.u8     d29, #4

    ; filter = clamp(filter + 3 * ( qs0 - ps0))
    vqmovn.s16  d28, q15

    vand        d28, d28, d19              ; filter &= mask

    vqadd.s8    d30, d28, d27              ; filter2 = clamp(filter+3)
    vqadd.s8    d29, d28, d29              ; filter1 = clamp(filter+4)
    vshr.s8     d30, d30, #3               ; filter2 >>= 3
    vshr.s8     d29, d29, #3               ; filter1 >>= 3


    vqadd.s8    d24, d24, d30              ; op0 = clamp(ps0 + filter2)
    vqsub.s8    d23, d23, d29              ; oq0 = clamp(qs0 - filter1)

    ; outer tap adjustments: ++filter1 >> 1
    vrshr.s8    d29, d29, #1
    vbic        d29, d29, d21              ; filter &= ~hev

    vqadd.s8    d25, d25, d29              ; op1 = clamp(ps1 + filter)
    vqsub.s8    d26, d26, d29              ; oq1 = clamp(qs1 - filter)

    veor        d24, d24, d22              ; *f_op0 = u^0x80
    veor        d23, d23, d22              ; *f_oq0 = u^0x80
    veor        d25, d25, d22              ; *f_op1 = u^0x80
    veor        d26, d26, d22              ; *f_oq1 = u^0x80

    tst         r7, #1
    bxne        lr

    orrs        r5, r5, r6                 ; Check for 0
    orreq       r7, r7, #2                 ; Only do mbfilter branch

    ; mbfilter flat && mask branch
    ; TODO(fgalligan): Can I decrease the cycles shifting to consective d's
    ; and using vibt on the q's?
    vmov.u8     d29, #2
    vaddl.u8    q15, d7, d8                ; op2 = p0 + q0
    vmlal.u8    q15, d4, d27               ; op2 = p0 + q0 + p3 * 3
    vmlal.u8    q15, d5, d29               ; op2 = p0 + q0 + p3 * 3 + p2 * 2
    vaddl.u8    q10, d4, d5
    vaddw.u8    q15, d6                    ; op2=p1 + p0 + q0 + p3 * 3 + p2 *2
    vaddl.u8    q14, d6, d9
    vqrshrn.u16 d18, q15, #3               ; r_op2

    vsub.i16    q15, q10
    vaddl.u8    q10, d4, d6
    vadd.i16    q15, q14
    vaddl.u8    q14, d7, d10
    vqrshrn.u16 d19, q15, #3               ; r_op1

    vsub.i16    q15, q10
    vadd.i16    q15, q14
    vaddl.u8    q14, d8, d11
    vqrshrn.u16 d20, q15, #3               ; r_op0

    vsubw.u8    q15, d4                    ; oq0 = op0 - p3
    vsubw.u8    q15, d7                    ; oq0 -= p0
    vadd.i16    q15, q14
    vaddl.u8    q14, d9, d11
    vqrshrn.u16 d21, q15, #3               ; r_oq0

    vsubw.u8    q15, d5                    ; oq1 = oq0 - p2
    vsubw.u8    q15, d8                    ; oq1 -= q0
    vadd.i16    q15, q14
    vaddl.u8    q14, d10, d11
    vqrshrn.u16 d22, q15, #3               ; r_oq1

    vsubw.u8    q15, d6                    ; oq2 = oq0 - p1
    vsubw.u8    q15, d9                    ; oq2 -= q1
    vadd.i16    q15, q14
    vqrshrn.u16 d27, q15, #3               ; r_oq2

    ; Filter does not set op2 or oq2, so use p2 and q2.
    vbif        d18, d5, d16               ; t_op2 |= p2 & ~(flat & mask)
    vbif        d19, d25, d16              ; t_op1 |= f_op1 & ~(flat & mask)
    vbif        d20, d24, d16              ; t_op0 |= f_op0 & ~(flat & mask)
    vbif        d21, d23, d16              ; t_oq0 |= f_oq0 & ~(flat & mask)
    vbif        d22, d26, d16              ; t_oq1 |= f_oq1 & ~(flat & mask)

    vbit        d23, d27, d16              ; t_oq2 |= r_oq2 & (flat & mask)
    vbif        d23, d10, d16              ; t_oq2 |= q2 & ~(flat & mask)

    tst         r7, #2
    bxne        lr

    ; wide_mbfilter flat2 && flat && mask branch
    vmov.u8     d16, #7
    vaddl.u8    q15, d7, d8                ; op6 = p0 + q0
    vaddl.u8    q12, d2, d3
    vaddl.u8    q13, d4, d5
    vaddl.u8    q14, d1, d6
    vmlal.u8    q15, d0, d16               ; op6 += p7 * 3
    vadd.i16    q12, q13
    vadd.i16    q15, q14
    vaddl.u8    q14, d2, d9
    vadd.i16    q15, q12
    vaddl.u8    q12, d0, d1
    vaddw.u8    q15, d1
    vaddl.u8    q13, d0, d2
    vadd.i16    q14, q15, q14
    vqrshrn.u16 d16, q15, #4               ; w_op6

    vsub.i16    q15, q14, q12
    vaddl.u8    q14, d3, d10
    vqrshrn.u16 d24, q15, #4               ; w_op5

    vsub.i16    q15, q13
    vaddl.u8    q13, d0, d3
    vadd.i16    q15, q14
    vaddl.u8    q14, d4, d11
    vqrshrn.u16 d25, q15, #4               ; w_op4

    vadd.i16    q15, q14
    vaddl.u8    q14, d0, d4
    vsub.i16    q15, q13
    vsub.i16    q14, q15, q14
    vqrshrn.u16 d26, q15, #4               ; w_op3

    vaddw.u8    q15, q14, d5               ; op2 += p2
    vaddl.u8    q14, d0, d5
    vaddw.u8    q15, d12                   ; op2 += q4
    vbif        d26, d4, d17               ; op3 |= p3 & ~(f2 & f & m)
    vqrshrn.u16 d27, q15, #4               ; w_op2

    vsub.i16    q15, q14
    vaddl.u8    q14, d0, d6
    vaddw.u8    q15, d6                    ; op1 += p1
    vaddw.u8    q15, d13                   ; op1 += q5
    vbif        d27, d18, d17              ; op2 |= t_op2 & ~(f2 & f & m)
    vqrshrn.u16 d18, q15, #4               ; w_op1

    vsub.i16    q15, q14
    vaddl.u8    q14, d0, d7
    vaddw.u8    q15, d7                    ; op0 += p0
    vaddw.u8    q15, d14                   ; op0 += q6
    vbif        d18, d19, d17              ; op1 |= t_op1 & ~(f2 & f & m)
    vqrshrn.u16 d19, q15, #4               ; w_op0

    vsub.i16    q15, q14
    vaddl.u8    q14, d1, d8
    vaddw.u8    q15, d8                    ; oq0 += q0
    vaddw.u8    q15, d15                   ; oq0 += q7
    vbif        d19, d20, d17              ; op0 |= t_op0 & ~(f2 & f & m)
    vqrshrn.u16 d20, q15, #4               ; w_oq0

    vsub.i16    q15, q14
    vaddl.u8    q14, d2, d9
    vaddw.u8    q15, d9                    ; oq1 += q1
    vaddl.u8    q4, d10, d15
    vaddw.u8    q15, d15                   ; oq1 += q7
    vbif        d20, d21, d17              ; oq0 |= t_oq0 & ~(f2 & f & m)
    vqrshrn.u16 d21, q15, #4               ; w_oq1

    vsub.i16    q15, q14
    vaddl.u8    q14, d3, d10
    vadd.i16    q15, q4
    vaddl.u8    q4, d11, d15
    vbif        d21, d22, d17              ; oq1 |= t_oq1 & ~(f2 & f & m)
    vqrshrn.u16 d22, q15, #4               ; w_oq2

    vsub.i16    q15, q14
    vaddl.u8    q14, d4, d11
    vadd.i16    q15, q4
    vaddl.u8    q4, d12, d15
    vbif        d22, d23, d17              ; oq2 |= t_oq2 & ~(f2 & f & m)
    vqrshrn.u16 d23, q15, #4               ; w_oq3

    vsub.i16    q15, q14
    vaddl.u8    q14, d5, d12
    vadd.i16    q15, q4
    vaddl.u8    q4, d13, d15
    vbif        d16, d1, d17               ; op6 |= p6 & ~(f2 & f & m)
    vqrshrn.u16 d1, q15, #4                ; w_oq4

    vsub.i16    q15, q14
    vaddl.u8    q14, d6, d13
    vadd.i16    q15, q4
    vaddl.u8    q4, d14, d15
    vbif        d24, d2, d17               ; op5 |= p5 & ~(f2 & f & m)
    vqrshrn.u16 d2, q15, #4                ; w_oq5

    vsub.i16    q15, q14
    vbif        d25, d3, d17               ; op4 |= p4 & ~(f2 & f & m)
    vadd.i16    q15, q4
    vbif        d23, d11, d17              ; oq3 |= q3 & ~(f2 & f & m)
    vqrshrn.u16 d3, q15, #4                ; w_oq6
    vbif        d1, d12, d17               ; oq4 |= q4 & ~(f2 & f & m)
    vbif        d2, d13, d17               ; oq5 |= q5 & ~(f2 & f & m)
    vbif        d3, d14, d17               ; oq6 |= q6 & ~(f2 & f & m)

    bx          lr
    ENDP        ; |vpx_wide_mbfilter_neon|

    END
