;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vpx_convolve_copy_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

|vpx_convolve_copy_neon| PROC
    push                {r4-r5, lr}
    ldrd                r4, r5, [sp, #32]

    cmp                 r4, #32
    bgt                 copy64
    beq                 copy32
    cmp                 r4, #8
    bgt                 copy16
    beq                 copy8
    b                   copy4

copy64
    sub                 lr, r1, #32
    sub                 r3, r3, #32
copy64_h
    pld                 [r0, r1, lsl #1]
    vld1.8              {q0-q1}, [r0]!
    vld1.8              {q2-q3}, [r0], lr
    vst1.8              {q0-q1}, [r2@128]!
    vst1.8              {q2-q3}, [r2@128], r3
    subs                r5, r5, #1
    bgt                 copy64_h
    pop                 {r4-r5, pc}

copy32
    pld                 [r0, r1, lsl #1]
    vld1.8              {q0-q1}, [r0], r1
    pld                 [r0, r1, lsl #1]
    vld1.8              {q2-q3}, [r0], r1
    vst1.8              {q0-q1}, [r2@128], r3
    vst1.8              {q2-q3}, [r2@128], r3
    subs                r5, r5, #2
    bgt                 copy32
    pop                 {r4-r5, pc}

copy16
    pld                 [r0, r1, lsl #1]
    vld1.8              {q0}, [r0], r1
    pld                 [r0, r1, lsl #1]
    vld1.8              {q1}, [r0], r1
    vst1.8              {q0}, [r2@128], r3
    vst1.8              {q1}, [r2@128], r3
    subs                r5, r5, #2
    bgt                 copy16
    pop                 {r4-r5, pc}

copy8
    pld                 [r0, r1, lsl #1]
    vld1.8              {d0}, [r0], r1
    pld                 [r0, r1, lsl #1]
    vld1.8              {d2}, [r0], r1
    vst1.8              {d0}, [r2@64], r3
    vst1.8              {d2}, [r2@64], r3
    subs                r5, r5, #2
    bgt                 copy8
    pop                 {r4-r5, pc}

copy4
    ldr                 r12, [r0], r1
    str                 r12, [r2], r3
    subs                r5, r5, #1
    bgt                 copy4
    pop                 {r4-r5, pc}
    ENDP

    END
