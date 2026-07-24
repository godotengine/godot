;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vpx_lpf_horizontal_8_neon|
    EXPORT  |vpx_lpf_horizontal_8_dual_neon|
    EXPORT  |vpx_lpf_vertical_8_neon|
    EXPORT  |vpx_lpf_vertical_8_dual_neon|
    ARM

    AREA ||.text||, CODE, READONLY, ALIGN=2

; Currently vpx only works on iterations 8 at a time. The vp8 loop filter
; works on 16 iterations at a time.
;
; void vpx_lpf_horizontal_8_neon(uint8_t *s, int p,
;                                const uint8_t *blimit,
;                                const uint8_t *limit,
;                                const uint8_t *thresh)
; r0    uint8_t *s,
; r1    int p, /* pitch */
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh,
|vpx_lpf_horizontal_8_neon| PROC
    push        {r4-r5, lr}

    vld1.8      {d0[]}, [r2]               ; duplicate *blimit
    ldr         r2, [sp, #12]              ; load thresh
    add         r1, r1, r1                 ; double pitch

    vld1.8      {d1[]}, [r3]               ; duplicate *limit
    vld1.8      {d2[]}, [r2]               ; duplicate *thresh

    sub         r3, r0, r1, lsl #1         ; move src pointer down by 4 lines
    add         r2, r3, r1, lsr #1         ; set to 3 lines down

    vld1.u8     {d3}, [r3@64], r1          ; p3
    vld1.u8     {d4}, [r2@64], r1          ; p2
    vld1.u8     {d5}, [r3@64], r1          ; p1
    vld1.u8     {d6}, [r2@64], r1          ; p0
    vld1.u8     {d7}, [r3@64], r1          ; q0
    vld1.u8     {d16}, [r2@64], r1         ; q1
    vld1.u8     {d17}, [r3@64]             ; q2
    vld1.u8     {d18}, [r2@64], r1         ; q3

    sub         r3, r3, r1, lsl #1
    sub         r2, r2, r1, lsl #2

    bl          vpx_mbloop_filter_neon

    vst1.u8     {d0}, [r2@64], r1          ; store op2
    vst1.u8     {d1}, [r3@64], r1          ; store op1
    vst1.u8     {d2}, [r2@64], r1          ; store op0
    vst1.u8     {d3}, [r3@64], r1          ; store oq0
    vst1.u8     {d4}, [r2@64], r1          ; store oq1
    vst1.u8     {d5}, [r3@64], r1          ; store oq2

    pop         {r4-r5, pc}

    ENDP        ; |vpx_lpf_horizontal_8_neon|

;void vpx_lpf_horizontal_8_dual_neon(uint8_t *s,
;                                    int p,
;                                    const uint8_t *blimit0,
;                                    const uint8_t *limit0,
;                                    const uint8_t *thresh0,
;                                    const uint8_t *blimit1,
;                                    const uint8_t *limit1,
;                                    const uint8_t *thresh1)
; r0      uint8_t *s,
; r1      int p, /* pitch */
; r2      const uint8_t *blimit0,
; r3      const uint8_t *limit0,
; sp      const uint8_t *thresh0,
; sp + 4  const uint8_t *blimit1,
; sp + 8  const uint8_t *limit1,
; sp + 12 const uint8_t *thresh1,
|vpx_lpf_horizontal_8_dual_neon| PROC
    push        {r0-r1, lr}
    ldr         lr, [sp, #12]
    push        {lr}                       ; thresh0
    bl          vpx_lpf_horizontal_8_neon

    ldr         r2, [sp, #20]              ; blimit1
    ldr         r3, [sp, #24]              ; limit1
    ldr         lr, [sp, #28]
    str         lr, [sp, #16]              ; thresh1
    add         sp, #4
    pop         {r0-r1, lr}
    add         r0, #8                     ; s + 8
    b           vpx_lpf_horizontal_8_neon
    ENDP        ; |vpx_lpf_horizontal_8_dual_neon|

; void vpx_lpf_vertical_8_neon(uint8_t *s,
;                              int pitch,
;                              const uint8_t *blimit,
;                              const uint8_t *limit,
;                              const uint8_t *thresh)
;
; r0    uint8_t *s,
; r1    int pitch,
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh,
|vpx_lpf_vertical_8_neon| PROC
    push        {r4-r5, lr}

    vld1.8      {d0[]}, [r2]              ; duplicate *blimit
    vld1.8      {d1[]}, [r3]              ; duplicate *limit

    ldr         r3, [sp, #12]             ; load thresh
    sub         r2, r0, #4                ; move s pointer down by 4 columns

    vld1.8      {d2[]}, [r3]              ; duplicate *thresh

    vld1.u8     {d3}, [r2], r1             ; load s data
    vld1.u8     {d4}, [r2], r1
    vld1.u8     {d5}, [r2], r1
    vld1.u8     {d6}, [r2], r1
    vld1.u8     {d7}, [r2], r1
    vld1.u8     {d16}, [r2], r1
    vld1.u8     {d17}, [r2], r1
    vld1.u8     {d18}, [r2]

    ;transpose to 8x16 matrix
    vtrn.32     d3, d7
    vtrn.32     d4, d16
    vtrn.32     d5, d17
    vtrn.32     d6, d18

    vtrn.16     d3, d5
    vtrn.16     d4, d6
    vtrn.16     d7, d17
    vtrn.16     d16, d18

    vtrn.8      d3, d4
    vtrn.8      d5, d6
    vtrn.8      d7, d16
    vtrn.8      d17, d18

    sub         r2, r0, #3
    add         r3, r0, #1

    bl          vpx_mbloop_filter_neon

    ;store op2, op1, op0, oq0
    vst4.8      {d0[0], d1[0], d2[0], d3[0]}, [r2], r1
    vst4.8      {d0[1], d1[1], d2[1], d3[1]}, [r2], r1
    vst4.8      {d0[2], d1[2], d2[2], d3[2]}, [r2], r1
    vst4.8      {d0[3], d1[3], d2[3], d3[3]}, [r2], r1
    vst4.8      {d0[4], d1[4], d2[4], d3[4]}, [r2], r1
    vst4.8      {d0[5], d1[5], d2[5], d3[5]}, [r2], r1
    vst4.8      {d0[6], d1[6], d2[6], d3[6]}, [r2], r1
    vst4.8      {d0[7], d1[7], d2[7], d3[7]}, [r2]

    ;store oq1, oq2
    vst2.8      {d4[0], d5[0]}, [r3], r1
    vst2.8      {d4[1], d5[1]}, [r3], r1
    vst2.8      {d4[2], d5[2]}, [r3], r1
    vst2.8      {d4[3], d5[3]}, [r3], r1
    vst2.8      {d4[4], d5[4]}, [r3], r1
    vst2.8      {d4[5], d5[5]}, [r3], r1
    vst2.8      {d4[6], d5[6]}, [r3], r1
    vst2.8      {d4[7], d5[7]}, [r3]

    pop         {r4-r5, pc}
    ENDP        ; |vpx_lpf_vertical_8_neon|

;void vpx_lpf_vertical_8_dual_neon(uint8_t *s,
;                                  int pitch,
;                                  const uint8_t *blimit0,
;                                  const uint8_t *limit0,
;                                  const uint8_t *thresh0,
;                                  const uint8_t *blimit1,
;                                  const uint8_t *limit1,
;                                  const uint8_t *thresh1)
; r0      uint8_t *s,
; r1      int pitch
; r2      const uint8_t *blimit0,
; r3      const uint8_t *limit0,
; sp      const uint8_t *thresh0,
; sp + 4  const uint8_t *blimit1,
; sp + 8  const uint8_t *limit1,
; sp + 12 const uint8_t *thresh1,
|vpx_lpf_vertical_8_dual_neon| PROC
    push        {r0-r1, lr}
    ldr         lr, [sp, #12]
    push        {lr}                       ; thresh0
    bl          vpx_lpf_vertical_8_neon

    ldr         r2, [sp, #20]              ; blimit1
    ldr         r3, [sp, #24]              ; limit1
    ldr         lr, [sp, #28]
    str         lr, [sp, #16]              ; thresh1
    add         sp, #4
    pop         {r0-r1, lr}
    add         r0, r0, r1, lsl #3         ; s + 8 * pitch
    b           vpx_lpf_vertical_8_neon
    ENDP        ; |vpx_lpf_vertical_8_dual_neon|

; void vpx_mbloop_filter_neon();
; This is a helper function for the loopfilters. The invidual functions do the
; necessary load, transpose (if necessary) and store. The function does not use
; registers d8-d15.
;
; Inputs:
; r0-r3, r12 PRESERVE
; d0    blimit
; d1    limit
; d2    thresh
; d3    p3
; d4    p2
; d5    p1
; d6    p0
; d7    q0
; d16   q1
; d17   q2
; d18   q3
;
; Outputs:
; d0    op2
; d1    op1
; d2    op0
; d3    oq0
; d4    oq1
; d5    oq2
|vpx_mbloop_filter_neon| PROC
    ; filter_mask
    vabd.u8     d19, d3, d4                ; m1 = abs(p3 - p2)
    vabd.u8     d20, d4, d5                ; m2 = abs(p2 - p1)
    vabd.u8     d21, d5, d6                ; m3 = abs(p1 - p0)
    vabd.u8     d22, d16, d7               ; m4 = abs(q1 - q0)
    vabd.u8     d23, d17, d16              ; m5 = abs(q2 - q1)
    vabd.u8     d24, d18, d17              ; m6 = abs(q3 - q2)

    ; only compare the largest value to limit
    vmax.u8     d19, d19, d20              ; m1 = max(m1, m2)
    vmax.u8     d20, d21, d22              ; m2 = max(m3, m4)

    vabd.u8     d25, d6, d4                ; m7 = abs(p0 - p2)

    vmax.u8     d23, d23, d24              ; m3 = max(m5, m6)

    vabd.u8     d26, d7, d17               ; m8 = abs(q0 - q2)

    vmax.u8     d19, d19, d20

    vabd.u8     d24, d6, d7                ; m9 = abs(p0 - q0)
    vabd.u8     d27, d3, d6                ; m10 = abs(p3 - p0)
    vabd.u8     d28, d18, d7               ; m11 = abs(q3 - q0)

    vmax.u8     d19, d19, d23

    vabd.u8     d23, d5, d16               ; a = abs(p1 - q1)
    vqadd.u8    d24, d24, d24              ; b = abs(p0 - q0) * 2

    ; abs () > limit
    vcge.u8     d19, d1, d19

    ; only compare the largest value to thresh
    vmax.u8     d25, d25, d26              ; m4 = max(m7, m8)
    vmax.u8     d26, d27, d28              ; m5 = max(m10, m11)

    vshr.u8     d23, d23, #1               ; a = a / 2

    vmax.u8     d25, d25, d26              ; m4 = max(m4, m5)

    vqadd.u8    d24, d24, d23              ; a = b + a

    vmax.u8     d20, d20, d25              ; m2 = max(m2, m4)

    vmov.u8     d23, #1
    vcge.u8     d24, d0, d24               ; a > blimit

    vcgt.u8     d21, d21, d2               ; (abs(p1 - p0) > thresh)*-1

    vcge.u8     d20, d23, d20              ; flat

    vand        d19, d19, d24              ; mask

    vcgt.u8     d23, d22, d2               ; (abs(q1 - q0) > thresh)*-1

    vand        d20, d20, d19              ; flat & mask

    vmov.u8     d22, #0x80

    vorr        d23, d21, d23              ; hev

    ; This instruction will truncate the "flat & mask" masks down to 4 bits
    ; each to fit into one 32 bit arm register. The values are stored in
    ; q10.64[0].
    vshrn.u16   d30, q10, #4
    vmov.u32    r4, d30[0]                 ; flat & mask 4bits

    adds        r5, r4, #1                 ; Check for all 1's

    ; If mask and flat are 1's for all vectors, then we only need to execute
    ; the power branch for all vectors.
    beq         power_branch_only

    cmp         r4, #0                     ; Check for 0, set flag for later

    ; mbfilter() function
    ; filter() function
    ; convert to signed
    veor        d21, d7, d22               ; qs0
    veor        d24, d6, d22               ; ps0
    veor        d25, d5, d22               ; ps1
    veor        d26, d16, d22              ; qs1

    vmov.u8     d27, #3

    vsub.s8     d28, d21, d24              ; ( qs0 - ps0)

    vqsub.s8    d29, d25, d26              ; filter = clamp(ps1-qs1)

    vmull.s8    q15, d28, d27              ; 3 * ( qs0 - ps0)

    vand        d29, d29, d23              ; filter &= hev

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
    vqsub.s8    d21, d21, d29              ; oq0 = clamp(qs0 - filter1)

    ; outer tap adjustments: ++filter1 >> 1
    vrshr.s8    d29, d29, #1
    vbic        d29, d29, d23              ; filter &= ~hev

    vqadd.s8    d25, d25, d29              ; op1 = clamp(ps1 + filter)
    vqsub.s8    d26, d26, d29              ; oq1 = clamp(qs1 - filter)

    ; If mask and flat are 0's for all vectors, then we only need to execute
    ; the filter branch for all vectors.
    beq         filter_branch_only

    ; If mask and flat are mixed then we must perform both branches and
    ; combine the data.
    veor        d24, d24, d22              ; *f_op0 = u^0x80
    veor        d21, d21, d22              ; *f_oq0 = u^0x80
    veor        d25, d25, d22              ; *f_op1 = u^0x80
    veor        d26, d26, d22              ; *f_oq1 = u^0x80

    ; At this point we have already executed the filter branch. The filter
    ; branch does not set op2 or oq2, so use p2 and q2. Execute the power
    ; branch and combine the data.
    vmov.u8     d23, #2
    vaddl.u8    q14, d6, d7                ; r_op2 = p0 + q0
    vmlal.u8    q14, d3, d27               ; r_op2 += p3 * 3
    vmlal.u8    q14, d4, d23               ; r_op2 += p2 * 2

    vbif        d0, d4, d20                ; op2 |= p2 & ~(flat & mask)

    vaddw.u8    q14, d5                    ; r_op2 += p1

    vbif        d1, d25, d20               ; op1 |= f_op1 & ~(flat & mask)

    vqrshrn.u16 d30, q14, #3               ; r_op2

    vsubw.u8    q14, d3                    ; r_op1 = r_op2 - p3
    vsubw.u8    q14, d4                    ; r_op1 -= p2
    vaddw.u8    q14, d5                    ; r_op1 += p1
    vaddw.u8    q14, d16                   ; r_op1 += q1

    vbif        d2, d24, d20               ; op0 |= f_op0 & ~(flat & mask)

    vqrshrn.u16 d31, q14, #3               ; r_op1

    vsubw.u8    q14, d3                    ; r_op0 = r_op1 - p3
    vsubw.u8    q14, d5                    ; r_op0 -= p1
    vaddw.u8    q14, d6                    ; r_op0 += p0
    vaddw.u8    q14, d17                   ; r_op0 += q2

    vbit        d0, d30, d20               ; op2 |= r_op2 & (flat & mask)

    vqrshrn.u16 d23, q14, #3               ; r_op0

    vsubw.u8    q14, d3                    ; r_oq0 = r_op0 - p3
    vsubw.u8    q14, d6                    ; r_oq0 -= p0
    vaddw.u8    q14, d7                    ; r_oq0 += q0

    vbit        d1, d31, d20               ; op1 |= r_op1 & (flat & mask)

    vaddw.u8    q14, d18                   ; oq0 += q3

    vbit        d2, d23, d20               ; op0 |= r_op0 & (flat & mask)

    vqrshrn.u16 d22, q14, #3               ; r_oq0

    vsubw.u8    q14, d4                    ; r_oq1 = r_oq0 - p2
    vsubw.u8    q14, d7                    ; r_oq1 -= q0
    vaddw.u8    q14, d16                   ; r_oq1 += q1

    vbif        d3, d21, d20               ; oq0 |= f_oq0 & ~(flat & mask)

    vaddw.u8    q14, d18                   ; r_oq1 += q3

    vbif        d4, d26, d20               ; oq1 |= f_oq1 & ~(flat & mask)

    vqrshrn.u16 d6, q14, #3                ; r_oq1

    vsubw.u8    q14, d5                    ; r_oq2 = r_oq1 - p1
    vsubw.u8    q14, d16                   ; r_oq2 -= q1
    vaddw.u8    q14, d17                   ; r_oq2 += q2
    vaddw.u8    q14, d18                   ; r_oq2 += q3

    vbif        d5, d17, d20               ; oq2 |= q2 & ~(flat & mask)

    vqrshrn.u16 d7, q14, #3                ; r_oq2

    vbit        d3, d22, d20               ; oq0 |= r_oq0 & (flat & mask)
    vbit        d4, d6, d20                ; oq1 |= r_oq1 & (flat & mask)
    vbit        d5, d7, d20                ; oq2 |= r_oq2 & (flat & mask)

    bx          lr

power_branch_only
    vmov.u8     d27, #3
    vmov.u8     d21, #2
    vaddl.u8    q14, d6, d7                ; op2 = p0 + q0
    vmlal.u8    q14, d3, d27               ; op2 += p3 * 3
    vmlal.u8    q14, d4, d21               ; op2 += p2 * 2
    vaddw.u8    q14, d5                    ; op2 += p1
    vqrshrn.u16 d0, q14, #3                ; op2

    vsubw.u8    q14, d3                    ; op1 = op2 - p3
    vsubw.u8    q14, d4                    ; op1 -= p2
    vaddw.u8    q14, d5                    ; op1 += p1
    vaddw.u8    q14, d16                   ; op1 += q1
    vqrshrn.u16 d1, q14, #3                ; op1

    vsubw.u8    q14, d3                    ; op0 = op1 - p3
    vsubw.u8    q14, d5                    ; op0 -= p1
    vaddw.u8    q14, d6                    ; op0 += p0
    vaddw.u8    q14, d17                   ; op0 += q2
    vqrshrn.u16 d2, q14, #3                ; op0

    vsubw.u8    q14, d3                    ; oq0 = op0 - p3
    vsubw.u8    q14, d6                    ; oq0 -= p0
    vaddw.u8    q14, d7                    ; oq0 += q0
    vaddw.u8    q14, d18                   ; oq0 += q3
    vqrshrn.u16 d3, q14, #3                ; oq0

    vsubw.u8    q14, d4                    ; oq1 = oq0 - p2
    vsubw.u8    q14, d7                    ; oq1 -= q0
    vaddw.u8    q14, d16                   ; oq1 += q1
    vaddw.u8    q14, d18                   ; oq1 += q3
    vqrshrn.u16 d4, q14, #3                ; oq1

    vsubw.u8    q14, d5                    ; oq2 = oq1 - p1
    vsubw.u8    q14, d16                   ; oq2 -= q1
    vaddw.u8    q14, d17                   ; oq2 += q2
    vaddw.u8    q14, d18                   ; oq2 += q3
    vqrshrn.u16 d5, q14, #3                ; oq2

    bx          lr

filter_branch_only
    ; TODO(fgalligan): See if we can rearange registers so we do not need to
    ; do the 2 vswp.
    vswp        d0, d4                      ; op2
    vswp        d5, d17                     ; oq2
    veor        d2, d24, d22                ; *op0 = u^0x80
    veor        d3, d21, d22                ; *oq0 = u^0x80
    veor        d1, d25, d22                ; *op1 = u^0x80
    veor        d4, d26, d22                ; *oq1 = u^0x80

    bx          lr

    ENDP        ; |vpx_mbloop_filter_neon|

    END
