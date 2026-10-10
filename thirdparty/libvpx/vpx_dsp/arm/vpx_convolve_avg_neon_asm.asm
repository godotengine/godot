;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vpx_convolve_avg_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

|vpx_convolve_avg_neon| PROC
    push                {r4-r6, lr}
    ldrd                r4, r5, [sp, #36]
    mov                 r6, r2

    cmp                 r4, #32
    bgt                 avg64
    beq                 avg32
    cmp                 r4, #8
    bgt                 avg16
    beq                 avg8
    b                   avg4

avg64
    sub                 lr, r1, #32
    sub                 r4, r3, #32
avg64_h
    pld                 [r0, r1, lsl #1]
    vld1.8              {q0-q1}, [r0]!
    vld1.8              {q2-q3}, [r0], lr
    pld                 [r2, r3]
    vld1.8              {q8-q9},   [r6@128]!
    vld1.8              {q10-q11}, [r6@128], r4
    vrhadd.u8           q0, q0, q8
    vrhadd.u8           q1, q1, q9
    vrhadd.u8           q2, q2, q10
    vrhadd.u8           q3, q3, q11
    vst1.8              {q0-q1}, [r2@128]!
    vst1.8              {q2-q3}, [r2@128], r4
    subs                r5, r5, #1
    bgt                 avg64_h
    pop                 {r4-r6, pc}

avg32
    vld1.8              {q0-q1}, [r0], r1
    vld1.8              {q2-q3}, [r0], r1
    vld1.8              {q8-q9},   [r6@128], r3
    vld1.8              {q10-q11}, [r6@128], r3
    pld                 [r0]
    vrhadd.u8           q0, q0, q8
    pld                 [r0, r1]
    vrhadd.u8           q1, q1, q9
    pld                 [r6]
    vrhadd.u8           q2, q2, q10
    pld                 [r6, r3]
    vrhadd.u8           q3, q3, q11
    vst1.8              {q0-q1}, [r2@128], r3
    vst1.8              {q2-q3}, [r2@128], r3
    subs                r5, r5, #2
    bgt                 avg32
    pop                 {r4-r6, pc}

avg16
    vld1.8              {q0}, [r0], r1
    vld1.8              {q1}, [r0], r1
    vld1.8              {q2}, [r6@128], r3
    vld1.8              {q3}, [r6@128], r3
    pld                 [r0]
    pld                 [r0, r1]
    vrhadd.u8           q0, q0, q2
    pld                 [r6]
    pld                 [r6, r3]
    vrhadd.u8           q1, q1, q3
    vst1.8              {q0}, [r2@128], r3
    vst1.8              {q1}, [r2@128], r3
    subs                r5, r5, #2
    bgt                 avg16
    pop                 {r4-r6, pc}

avg8
    vld1.8              {d0}, [r0], r1
    vld1.8              {d1}, [r0], r1
    vld1.8              {d2}, [r6@64], r3
    vld1.8              {d3}, [r6@64], r3
    pld                 [r0]
    pld                 [r0, r1]
    vrhadd.u8           q0, q0, q1
    pld                 [r6]
    pld                 [r6, r3]
    vst1.8              {d0}, [r2@64], r3
    vst1.8              {d1}, [r2@64], r3
    subs                r5, r5, #2
    bgt                 avg8
    pop                 {r4-r6, pc}

avg4
    vld1.32             {d0[0]}, [r0], r1
    vld1.32             {d0[1]}, [r0], r1
    vld1.32             {d2[0]}, [r6@32], r3
    vld1.32             {d2[1]}, [r6@32], r3
    vrhadd.u8           d0, d0, d2
    vst1.32             {d0[0]}, [r2@32], r3
    vst1.32             {d0[1]}, [r2@32], r3
    subs                r5, r5, #2
    bgt                 avg4
    pop                 {r4-r6, pc}
    ENDP

    END
