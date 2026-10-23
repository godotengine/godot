;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vpx_lpf_horizontal_4_neon|
    EXPORT  |vpx_lpf_vertical_4_neon|
    EXPORT  |vpx_lpf_horizontal_4_dual_neon|
    EXPORT  |vpx_lpf_vertical_4_dual_neon|
    ARM

    AREA ||.text||, CODE, READONLY, ALIGN=2

; Currently vpx only works on iterations 8 at a time. The vp8 loop filter
; works on 16 iterations at a time.
;
; void vpx_lpf_horizontal_4_neon(uint8_t *s,
;                                int p /* pitch */,
;                                const uint8_t *blimit,
;                                const uint8_t *limit,
;                                const uint8_t *thresh)
;
; r0    uint8_t *s,
; r1    int p, /* pitch */
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh,
|vpx_lpf_horizontal_4_neon| PROC
    push        {lr}

    vld1.8      {d0[]}, [r2]               ; duplicate *blimit
    ldr         r2, [sp, #4]               ; load thresh
    add         r1, r1, r1                 ; double pitch

    vld1.8      {d1[]}, [r3]               ; duplicate *limit
    vld1.8      {d2[]}, [r2]               ; duplicate *thresh

    sub         r2, r0, r1, lsl #1         ; move src pointer down by 4 lines
    add         r3, r2, r1, lsr #1         ; set to 3 lines down

    vld1.u8     {d3}, [r2@64], r1          ; p3
    vld1.u8     {d4}, [r3@64], r1          ; p2
    vld1.u8     {d5}, [r2@64], r1          ; p1
    vld1.u8     {d6}, [r3@64], r1          ; p0
    vld1.u8     {d7}, [r2@64], r1          ; q0
    vld1.u8     {d16}, [r3@64], r1         ; q1
    vld1.u8     {d17}, [r2@64]             ; q2
    vld1.u8     {d18}, [r3@64]             ; q3

    sub         r2, r2, r1, lsl #1
    sub         r3, r3, r1, lsl #1

    bl          filter4_8

    vst1.u8     {d4}, [r2@64], r1          ; store op1
    vst1.u8     {d5}, [r3@64], r1          ; store op0
    vst1.u8     {d6}, [r2@64], r1          ; store oq0
    vst1.u8     {d7}, [r3@64], r1          ; store oq1

    pop         {pc}
    ENDP        ; |vpx_lpf_horizontal_4_neon|

; Currently vpx only works on iterations 8 at a time. The vp8 loop filter
; works on 16 iterations at a time.
;
; void vpx_lpf_vertical_4_neon(uint8_t *s,
;                              int p /* pitch */,
;                              const uint8_t *blimit,
;                              const uint8_t *limit,
;                              const uint8_t *thresh)
;
; r0    uint8_t *s,
; r1    int p, /* pitch */
; r2    const uint8_t *blimit,
; r3    const uint8_t *limit,
; sp    const uint8_t *thresh,
|vpx_lpf_vertical_4_neon| PROC
    push        {lr}

    vld1.8      {d0[]}, [r2]              ; duplicate *blimit
    vld1.8      {d1[]}, [r3]              ; duplicate *limit

    ldr         r3, [sp, #4]              ; load thresh
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

    bl          filter4_8

    sub         r0, r0, #2

    ;store op1, op0, oq0, oq1
    vst4.8      {d4[0], d5[0], d6[0], d7[0]}, [r0], r1
    vst4.8      {d4[1], d5[1], d6[1], d7[1]}, [r0], r1
    vst4.8      {d4[2], d5[2], d6[2], d7[2]}, [r0], r1
    vst4.8      {d4[3], d5[3], d6[3], d7[3]}, [r0], r1
    vst4.8      {d4[4], d5[4], d6[4], d7[4]}, [r0], r1
    vst4.8      {d4[5], d5[5], d6[5], d7[5]}, [r0], r1
    vst4.8      {d4[6], d5[6], d6[6], d7[6]}, [r0], r1
    vst4.8      {d4[7], d5[7], d6[7], d7[7]}, [r0]

    pop         {pc}
    ENDP        ; |vpx_lpf_vertical_4_neon|

; void filter4_8();
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
; d4    op1
; d5    op0
; d6    oq0
; d7    oq1
|filter4_8| PROC
    ; filter_mask
    vabd.u8     d19, d3, d4                 ; m1 = abs(p3 - p2)
    vabd.u8     d20, d4, d5                 ; m2 = abs(p2 - p1)
    vabd.u8     d21, d5, d6                 ; m3 = abs(p1 - p0)
    vabd.u8     d22, d16, d7                ; m4 = abs(q1 - q0)
    vabd.u8     d3, d17, d16                ; m5 = abs(q2 - q1)
    vabd.u8     d4, d18, d17                ; m6 = abs(q3 - q2)

    ; only compare the largest value to limit
    vmax.u8     d19, d19, d20               ; m1 = max(m1, m2)
    vmax.u8     d20, d21, d22               ; m2 = max(m3, m4)

    vabd.u8     d17, d6, d7                 ; abs(p0 - q0)

    vmax.u8     d3, d3, d4                  ; m3 = max(m5, m6)

    vmov.u8     d18, #0x80

    vmax.u8     d23, d19, d20               ; m1 = max(m1, m2)

    ; hevmask
    vcgt.u8     d21, d21, d2                ; (abs(p1 - p0) > thresh)*-1
    vcgt.u8     d22, d22, d2                ; (abs(q1 - q0) > thresh)*-1
    vmax.u8     d23, d23, d3                ; m1 = max(m1, m3)

    vabd.u8     d28, d5, d16                ; a = abs(p1 - q1)
    vqadd.u8    d17, d17, d17               ; b = abs(p0 - q0) * 2

    veor        d7, d7, d18                 ; qs0

    vcge.u8     d23, d1, d23                ; abs(m1) > limit

    ; filter() function
    ; convert to signed

    vshr.u8     d28, d28, #1                ; a = a / 2
    veor        d6, d6, d18                 ; ps0

    veor        d5, d5, d18                 ; ps1
    vqadd.u8    d17, d17, d28               ; a = b + a

    veor        d16, d16, d18               ; qs1

    vmov.u8     d19, #3

    vsub.s8     d28, d7, d6                 ; ( qs0 - ps0)

    vcge.u8     d17, d0, d17                ; a > blimit

    vqsub.s8    d27, d5, d16                ; filter = clamp(ps1-qs1)
    vorr        d22, d21, d22               ; hevmask

    vmull.s8    q12, d28, d19               ; 3 * ( qs0 - ps0)

    vand        d27, d27, d22               ; filter &= hev
    vand        d23, d23, d17               ; filter_mask

    vaddw.s8    q12, q12, d27               ; filter + 3 * (qs0 - ps0)

    vmov.u8     d17, #4

    ; filter = clamp(filter + 3 * ( qs0 - ps0))
    vqmovn.s16  d27, q12

    vand        d27, d27, d23               ; filter &= mask

    vqadd.s8    d28, d27, d19               ; filter2 = clamp(filter+3)
    vqadd.s8    d27, d27, d17               ; filter1 = clamp(filter+4)
    vshr.s8     d28, d28, #3                ; filter2 >>= 3
    vshr.s8     d27, d27, #3                ; filter1 >>= 3

    vqadd.s8    d19, d6, d28                ; u = clamp(ps0 + filter2)
    vqsub.s8    d26, d7, d27                ; u = clamp(qs0 - filter1)

    ; outer tap adjustments
    vrshr.s8    d27, d27, #1                ; filter = ++filter1 >> 1

    veor        d6, d26, d18                ; *oq0 = u^0x80

    vbic        d27, d27, d22               ; filter &= ~hev

    vqadd.s8    d21, d5, d27                ; u = clamp(ps1 + filter)
    vqsub.s8    d20, d16, d27               ; u = clamp(qs1 - filter)

    veor        d5, d19, d18                ; *op0 = u^0x80
    veor        d4, d21, d18                ; *op1 = u^0x80
    veor        d7, d20, d18                ; *oq1 = u^0x80

    bx          lr
    ENDP        ; |filter4_8|

;void vpx_lpf_horizontal_4_dual_neon(uint8_t *s, int p,
;                                    const uint8_t *blimit0,
;                                    const uint8_t *limit0,
;                                    const uint8_t *thresh0,
;                                    const uint8_t *blimit1,
;                                    const uint8_t *limit1,
;                                    const uint8_t *thresh1)
; r0    uint8_t *s,
; r1    int p,
; r2    const uint8_t *blimit0,
; r3    const uint8_t *limit0,
; sp    const uint8_t *thresh0,
; sp+4  const uint8_t *blimit1,
; sp+8  const uint8_t *limit1,
; sp+12 const uint8_t *thresh1,

|vpx_lpf_horizontal_4_dual_neon| PROC
    push        {lr}

    ldr         r12, [sp, #4]              ; load thresh0
    vld1.8      {d0}, [r2]                 ; load blimit0 to first half q
    vld1.8      {d2}, [r3]                 ; load limit0 to first half q

    add         r1, r1, r1                 ; double pitch
    ldr         r2, [sp, #8]               ; load blimit1

    vld1.8      {d4}, [r12]                ; load thresh0 to first half q

    ldr         r3, [sp, #12]              ; load limit1
    ldr         r12, [sp, #16]             ; load thresh1
    vld1.8      {d1}, [r2]                 ; load blimit1 to 2nd half q

    sub         r2, r0, r1, lsl #1         ; s[-4 * p]

    vld1.8      {d3}, [r3]                 ; load limit1 to 2nd half q
    vld1.8      {d5}, [r12]                ; load thresh1 to 2nd half q

    vpush       {d8-d15}                   ; save neon registers

    add         r3, r2, r1, lsr #1         ; s[-3 * p]

    vld1.u8     {q3}, [r2@64], r1          ; p3
    vld1.u8     {q4}, [r3@64], r1          ; p2
    vld1.u8     {q5}, [r2@64], r1          ; p1
    vld1.u8     {q6}, [r3@64], r1          ; p0
    vld1.u8     {q7}, [r2@64], r1          ; q0
    vld1.u8     {q8}, [r3@64], r1          ; q1
    vld1.u8     {q9}, [r2@64]              ; q2
    vld1.u8     {q10}, [r3@64]             ; q3

    sub         r2, r2, r1, lsl #1
    sub         r3, r3, r1, lsl #1

    bl          filter4_16

    vst1.u8     {q5}, [r2@64], r1          ; store op1
    vst1.u8     {q6}, [r3@64], r1          ; store op0
    vst1.u8     {q7}, [r2@64], r1          ; store oq0
    vst1.u8     {q8}, [r3@64], r1          ; store oq1

    vpop        {d8-d15}                   ; restore neon registers

    pop         {pc}
    ENDP        ; |vpx_lpf_horizontal_4_dual_neon|

;void vpx_lpf_vertical_4_dual_neon(uint8_t *s, int p,
;                                  const uint8_t *blimit0,
;                                  const uint8_t *limit0,
;                                  const uint8_t *thresh0,
;                                  const uint8_t *blimit1,
;                                  const uint8_t *limit1,
;                                  const uint8_t *thresh1)
; r0    uint8_t *s,
; r1    int p,
; r2    const uint8_t *blimit0,
; r3    const uint8_t *limit0,
; sp    const uint8_t *thresh0,
; sp+4  const uint8_t *blimit1,
; sp+8  const uint8_t *limit1,
; sp+12 const uint8_t *thresh1,

|vpx_lpf_vertical_4_dual_neon| PROC
    push        {lr}

    ldr         r12, [sp, #4]              ; load thresh0
    vld1.8      {d0}, [r2]                 ; load blimit0 to first half q
    vld1.8      {d2}, [r3]                 ; load limit0 to first half q

    ldr         r2, [sp, #8]               ; load blimit1

    vld1.8      {d4}, [r12]                ; load thresh0 to first half q

    ldr         r3, [sp, #12]              ; load limit1
    ldr         r12, [sp, #16]             ; load thresh1
    vld1.8      {d1}, [r2]                 ; load blimit1 to 2nd half q

    sub         r2, r0, #4                 ; s[-4]

    vld1.8      {d3}, [r3]                 ; load limit1 to 2nd half q
    vld1.8      {d5}, [r12]                ; load thresh1 to 2nd half q

    vpush       {d8-d15}                   ; save neon registers

    vld1.u8     {d6}, [r2], r1             ; 00 01 02 03 04 05 06 07
    vld1.u8     {d8}, [r2], r1             ; 10 11 12 13 14 15 16 17
    vld1.u8     {d10}, [r2], r1            ; 20 21 22 23 24 25 26 27
    vld1.u8     {d12}, [r2], r1            ; 30 31 32 33 34 35 36 37
    vld1.u8     {d14}, [r2], r1            ; 40 41 42 43 44 45 46 47
    vld1.u8     {d16}, [r2], r1            ; 50 51 52 53 54 55 56 57
    vld1.u8     {d18}, [r2], r1            ; 60 61 62 63 64 65 66 67
    vld1.u8     {d20}, [r2], r1            ; 70 71 72 73 74 75 76 77
    vld1.u8     {d7}, [r2], r1             ; 80 81 82 83 84 85 86 87
    vld1.u8     {d9}, [r2], r1             ; 90 91 92 93 94 95 96 97
    vld1.u8     {d11}, [r2], r1            ; A0 A1 A2 A3 A4 A5 A6 A7
    vld1.u8     {d13}, [r2], r1            ; B0 B1 B2 B3 B4 B5 B6 B7
    vld1.u8     {d15}, [r2], r1            ; C0 C1 C2 C3 C4 C5 C6 C7
    vld1.u8     {d17}, [r2], r1            ; D0 D1 D2 D3 D4 D5 D6 D7
    vld1.u8     {d19}, [r2], r1            ; E0 E1 E2 E3 E4 E5 E6 E7
    vld1.u8     {d21}, [r2]                ; F0 F1 F2 F3 F4 F5 F6 F7

    vtrn.8      q3, q4  ; q3 : 00 10 02 12 04 14 06 16  80 90 82 92 84 94 86 96
                        ; q4 : 01 11 03 13 05 15 07 17  81 91 83 93 85 95 87 97
    vtrn.8      q5, q6  ; q5 : 20 30 22 32 24 34 26 36  A0 B0 A2 B2 A4 B4 A6 B6
                        ; q6 : 21 31 23 33 25 35 27 37  A1 B1 A3 B3 A5 B5 A7 B7
    vtrn.8      q7, q8  ; q7 : 40 50 42 52 44 54 46 56  C0 D0 C2 D2 C4 D4 C6 D6
                        ; q8 : 41 51 43 53 45 55 47 57  C1 D1 C3 D3 C5 D5 C7 D7
    vtrn.8      q9, q10 ; q9 : 60 70 62 72 64 74 66 76  E0 F0 E2 F2 E4 F4 E6 F6
                        ; q10: 61 71 63 73 65 75 67 77  E1 F1 E3 F3 E5 F5 E7 F7

    vtrn.16     q3, q5  ; q3 : 00 10 20 30 04 14 24 34  80 90 A0 B0 84 94 A4 B4
                        ; q5 : 02 12 22 32 06 16 26 36  82 92 A2 B2 86 96 A6 B6
    vtrn.16     q4, q6  ; q4 : 01 11 21 31 05 15 25 35  81 91 A1 B1 85 95 A5 B5
                        ; q6 : 03 13 23 33 07 17 27 37  83 93 A3 B3 87 97 A7 B7
    vtrn.16     q7, q9  ; q7 : 40 50 60 70 44 54 64 74  C0 D0 E0 F0 C4 D4 E4 F4
                        ; q9 : 42 52 62 72 46 56 66 76  C2 D2 E2 F2 C6 D6 E6 F6
    vtrn.16     q8, q10 ; q8 : 41 51 61 71 45 55 65 75  C1 D1 E1 F1 C5 D5 E5 F5
                        ; q10: 43 53 63 73 47 57 67 77  C3 D3 E3 F3 C7 D7 E7 F7

    vtrn.32     q3, q7  ; q3 : 00 10 20 30 40 50 60 70  80 90 A0 B0 C0 D0 E0 F0
                        ; q7 : 04 14 24 34 44 54 64 74  84 94 A4 B4 C4 D4 E4 F4
    vtrn.32     q5, q9  ; q5 : 02 12 22 32 42 52 62 72  82 92 A2 B2 C2 D2 E2 F2
                        ; q9 : 06 16 26 36 46 56 66 76  86 96 A6 B6 C6 D6 E6 F6
    vtrn.32     q4, q8  ; q4 : 01 11 21 31 41 51 61 71  81 91 A1 B1 C1 D1 E1 F1
                        ; q8 : 05 15 25 35 45 55 65 75  85 95 A5 B5 C5 D5 E5 F5
    vtrn.32     q6, q10 ; q6 : 03 13 23 33 43 53 63 73  83 93 A3 B3 C3 D3 E3 F3
                        ; q10: 07 17 27 37 47 57 67 77  87 97 A7 B7 C7 D7 E7 F7

    bl          filter4_16

    sub         r0, #2

    vmov        d0, d11
    vmov        d1, d13
    vmov        d2, d15
    vmov        d3, d17
    vmov        d11, d12
    vmov        d12, d14
    vmov        d13, d16
    vst4.8      {d10[0], d11[0], d12[0], d13[0]}, [r0], r1
    vst4.8      {d10[1], d11[1], d12[1], d13[1]}, [r0], r1
    vst4.8      {d10[2], d11[2], d12[2], d13[2]}, [r0], r1
    vst4.8      {d10[3], d11[3], d12[3], d13[3]}, [r0], r1
    vst4.8      {d10[4], d11[4], d12[4], d13[4]}, [r0], r1
    vst4.8      {d10[5], d11[5], d12[5], d13[5]}, [r0], r1
    vst4.8      {d10[6], d11[6], d12[6], d13[6]}, [r0], r1
    vst4.8      {d10[7], d11[7], d12[7], d13[7]}, [r0], r1
    vst4.8      {d0[0], d1[0], d2[0], d3[0]}, [r0], r1
    vst4.8      {d0[1], d1[1], d2[1], d3[1]}, [r0], r1
    vst4.8      {d0[2], d1[2], d2[2], d3[2]}, [r0], r1
    vst4.8      {d0[3], d1[3], d2[3], d3[3]}, [r0], r1
    vst4.8      {d0[4], d1[4], d2[4], d3[4]}, [r0], r1
    vst4.8      {d0[5], d1[5], d2[5], d3[5]}, [r0], r1
    vst4.8      {d0[6], d1[6], d2[6], d3[6]}, [r0], r1
    vst4.8      {d0[7], d1[7], d2[7], d3[7]}, [r0]

    vpop        {d8-d15}                   ; restore neon registers

    pop         {pc}
    ENDP        ; |vpx_lpf_vertical_4_dual_neon|

; void filter4_16();
; This is a helper function for the loopfilters. The invidual functions do the
; necessary load, transpose (if necessary) and store. This function uses
; registers d8-d15, so the calling function must save those registers.
;
; r0-r3, r12 PRESERVE
; q0    blimit
; q1    limit
; q2    thresh
; q3    p3
; q4    p2
; q5    p1
; q6    p0
; q7    q0
; q8    q1
; q9    q2
; q10   q3
;
; Outputs:
; q5    op1
; q6    op0
; q7    oq0
; q8    oq1
|filter4_16| PROC

    ; filter_mask
    vabd.u8     q11, q3, q4                 ; m1 = abs(p3 - p2)
    vabd.u8     q12, q4, q5                 ; m2 = abs(p2 - p1)
    vabd.u8     q13, q5, q6                 ; m3 = abs(p1 - p0)
    vabd.u8     q14, q8, q7                 ; m4 = abs(q1 - q0)
    vabd.u8     q3, q9, q8                  ; m5 = abs(q2 - q1)
    vabd.u8     q4, q10, q9                 ; m6 = abs(q3 - q2)

    ; only compare the largest value to limit
    vmax.u8     q11, q11, q12               ; m7 = max(m1, m2)
    vmax.u8     q12, q13, q14               ; m8 = max(m3, m4)

    vabd.u8     q9, q6, q7                  ; abs(p0 - q0)

    vmax.u8     q3, q3, q4                  ; m9 = max(m5, m6)

    vmov.u8     q10, #0x80

    vmax.u8     q15, q11, q12               ; m10 = max(m7, m8)

    vcgt.u8     q13, q13, q2                ; (abs(p1 - p0) > thresh)*-1
    vcgt.u8     q14, q14, q2                ; (abs(q1 - q0) > thresh)*-1
    vmax.u8     q15, q15, q3                ; m11 = max(m10, m9)

    vabd.u8     q2, q5, q8                  ; a = abs(p1 - q1)
    vqadd.u8    q9, q9, q9                  ; b = abs(p0 - q0) * 2

    veor        q7, q7, q10                 ; qs0

    vcge.u8     q15, q1, q15                ; abs(m11) > limit

    vshr.u8     q2, q2, #1                  ; a = a / 2
    veor        q6, q6, q10                 ; ps0

    veor        q5, q5, q10                 ; ps1
    vqadd.u8    q9, q9, q2                  ; a = b + a

    veor        q8, q8, q10                 ; qs1

    vmov.u16    q4, #3

    vsubl.s8    q2, d14, d12                ; ( qs0 - ps0)
    vsubl.s8    q11, d15, d13

    vcge.u8     q9, q0, q9                  ; a > blimit

    vqsub.s8    q1, q5, q8                  ; filter = clamp(ps1-qs1)
    vorr        q14, q13, q14               ; hev

    vmul.i16    q2, q2, q4                  ; 3 * ( qs0 - ps0)
    vmul.i16    q11, q11, q4

    vand        q1, q1, q14                 ; filter &= hev
    vand        q15, q15, q9                ; mask

    vmov.u8     q4, #3

    vaddw.s8    q2, q2, d2                  ; filter + 3 * (qs0 - ps0)
    vaddw.s8    q11, q11, d3

    vmov.u8     q9, #4

    ; filter = clamp(filter + 3 * ( qs0 - ps0))
    vqmovn.s16  d2, q2
    vqmovn.s16  d3, q11
    vand        q1, q1, q15                 ; filter &= mask

    vqadd.s8    q2, q1, q4                  ; filter2 = clamp(filter+3)
    vqadd.s8    q1, q1, q9                  ; filter1 = clamp(filter+4)
    vshr.s8     q2, q2, #3                  ; filter2 >>= 3
    vshr.s8     q1, q1, #3                  ; filter1 >>= 3


    vqadd.s8    q11, q6, q2                 ; u = clamp(ps0 + filter2)
    vqsub.s8    q0, q7, q1                  ; u = clamp(qs0 - filter1)

    ; outer tap adjustments
    vrshr.s8    q1, q1, #1                  ; filter = ++filter1 >> 1

    veor        q7, q0,  q10                ; *oq0 = u^0x80

    vbic        q1, q1, q14                 ; filter &= ~hev

    vqadd.s8    q13, q5, q1                 ; u = clamp(ps1 + filter)
    vqsub.s8    q12, q8, q1                 ; u = clamp(qs1 - filter)

    veor        q6, q11, q10                ; *op0 = u^0x80
    veor        q5, q13, q10                ; *op1 = u^0x80
    veor        q8, q12, q10                ; *oq1 = u^0x80

    bx          lr
    ENDP        ; |filter4_16|

    END
