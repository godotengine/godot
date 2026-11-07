;
;  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;
;**************Variables Vs Registers***********************************
;    r0 => src
;    r1 => dst
;    r2 =>  src_stride
;    r3 =>  dst_stride
;    r4 => filter_x0
;    r8 =>  ht
;    r10 =>  wd

    EXPORT          |vpx_convolve8_horiz_filter_type2_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA  ||.text||, CODE, READONLY, ALIGN=2

|vpx_convolve8_horiz_filter_type2_neon| PROC

    stmfd           sp!,    {r4  -  r12,    r14} ;stack stores the values of
                                                 ; the arguments
    vpush           {d8  -  d15}                 ; stack offset by 64
    mov             r4,     r1
    mov             r1,     r2
    mov             r2,     r4

start_loop_count
    ldr             r4,     [sp,    #104]   ;loads pi1_coeff
    ldr             r8,     [sp,    #108]   ;loads x0_q4
    add             r4,     r4,     r8,     lsl #4 ;r4 = filter[x0_q4]
    ldr             r8,     [sp,    #128]   ;loads ht
    ldr             r10,    [sp,    #124]   ;loads wd
    vld2.8          {d0,    d1},    [r4]    ;coeff = vld1_s8(pi1_coeff)
    mov             r11,    #1
    subs            r14,    r8,     #0      ;checks for ht == 0
    vabs.s8         d2,     d0              ;vabs_s8(coeff)
    vdup.8          d24,    d2[0]           ;coeffabs_0 = vdup_lane_u8(coeffabs,
                                            ; 0)
    sub             r12,    r0,     #3      ;pu1_src - 3
    vdup.8          d25,    d2[1]           ;coeffabs_1 = vdup_lane_u8(coeffabs,
                                            ; 1)
    add             r4,     r12,    r2      ;pu1_src_tmp2_8 = pu1_src + src_strd
    vdup.8          d26,    d2[2]           ;coeffabs_2 = vdup_lane_u8(coeffabs,
                                            ; 2)
    rsb             r9,     r10,    r2,     lsl #1 ;2*src_strd - wd
    vdup.8          d27,    d2[3]           ;coeffabs_3 = vdup_lane_u8(coeffabs,
                                            ; 3)
    rsb             r8,     r10,    r3,     lsl #1 ;2*dst_strd - wd
    vdup.8          d28,    d2[4]           ;coeffabs_4 = vdup_lane_u8(coeffabs,
                                            ; 4)
    vdup.8          d29,    d2[5]           ;coeffabs_5 = vdup_lane_u8(coeffabs,
                                            ; 5)
    vdup.8          d30,    d2[6]           ;coeffabs_6 = vdup_lane_u8(coeffabs,
                                            ; 6)
    vdup.8          d31,    d2[7]           ;coeffabs_7 = vdup_lane_u8(coeffabs,
                                            ; 7)
    mov             r7,     r1
    cmp             r10,    #4
    ble             outer_loop_4

    cmp             r10,    #24
    moveq           r10,    #16
    addeq           r8,     #8
    addeq           r9,     #8
    cmp             r10,    #16
    bge             outer_loop_16

    cmp             r10,    #12
    addeq           r8,     #4
    addeq           r9,     #4
    b               outer_loop_8

outer_loop8_residual
    sub             r12,    r0,     #3      ;pu1_src - 3
    mov             r1,     r7
    mov             r14,    #32
    add             r1,     #16
    add             r12,    #16
    mov             r10,    #8
    add             r8,     #8
    add             r9,     #8

outer_loop_8
    add             r6,     r1,     r3      ;pu1_dst + dst_strd
    add             r4,     r12,    r2      ;pu1_src + src_strd
    subs            r5,     r10,    #0      ;checks wd
    ble             end_inner_loop_8

inner_loop_8
    mov             r7,     #0xc000
    vld1.u32        {d0},   [r12],  r11     ;vector load pu1_src
    vdup.16         q4,     r7
    vld1.u32        {d1},   [r12],  r11
    vdup.16         q5,     r7
    vld1.u32        {d2},   [r12],  r11
    vld1.u32        {d3},   [r12],  r11
    mov             r7,     #0x4000
    vld1.u32        {d4},   [r12],  r11
    vmlal.u8        q4,     d1,     d25     ;mul_res = vmlal_u8(src[0_1],
                                            ; coeffabs_1);
    vld1.u32        {d5},   [r12],  r11
    vmlal.u8        q4,     d3,     d27     ;mul_res = vmull_u8(src[0_3],
                                            ; coeffabs_3);
    vld1.u32        {d6},   [r12],  r11
    vmlsl.u8        q4,     d0,     d24     ;mul_res = vmlsl_u8(src[0_0],
                                            ; coeffabs_0);
    vld1.u32        {d7},   [r12],  r11
    vmlsl.u8        q4,     d2,     d26     ;mul_res = vmlsl_u8(src[0_2],
                                            ; coeffabs_2);
    vld1.u32        {d12},  [r4],   r11     ;vector load pu1_src + src_strd
    vmlal.u8        q4,     d4,     d28     ;mul_res = vmlal_u8(src[0_4],
                                            ; coeffabs_4);
    vld1.u32        {d13},  [r4],   r11
    vmlsl.u8        q4,     d5,     d29     ;mul_res = vmlsl_u8(src[0_5],
                                            ; coeffabs_5);
    vld1.u32        {d14},  [r4],   r11
    vmlal.u8        q4,     d6,     d30     ;mul_res = vmlal_u8(src[0_6],
                                            ; coeffabs_6);
    vld1.u32        {d15},  [r4],   r11
    vmlsl.u8        q4,     d7,     d31     ;mul_res = vmlsl_u8(src[0_7],
                                            ; coeffabs_7);
    vld1.u32        {d16},  [r4],   r11     ;vector load pu1_src + src_strd
    vdup.16         q11,    r7
    vmlal.u8        q5,     d15,    d27     ;mul_res = vmull_u8(src[0_3],
                                            ; coeffabs_3);
    vld1.u32        {d17},  [r4],   r11
    vmlsl.u8        q5,     d14,    d26     ;mul_res = vmlsl_u8(src[0_2],
                                            ; coeffabs_2);
    vhadd.s16       q4,     q4,     q11
    vld1.u32        {d18},  [r4],   r11
    vmlal.u8        q5,     d16,    d28     ;mul_res = vmlal_u8(src[0_4],
                                            ; coeffabs_4);
    vld1.u32        {d19},  [r4],   r11     ;vector load pu1_src + src_strd
    vmlsl.u8        q5,     d17,    d29     ;mul_res = vmlsl_u8(src[0_5],
                                            ; coeffabs_5);
    vmlal.u8        q5,     d18,    d30     ;mul_res = vmlal_u8(src[0_6],
                                            ; coeffabs_6);
    vmlsl.u8        q5,     d19,    d31     ;mul_res = vmlsl_u8(src[0_7],
                                            ; coeffabs_7);
    vqrshrun.s16    d20,    q4,     #6      ;right shift and saturating narrow
                                            ; result 1
    vmlsl.u8        q5,     d12,    d24     ;mul_res = vmlsl_u8(src[0_0],
                                            ; coeffabs_0);
    vmlal.u8        q5,     d13,    d25     ;mul_res = vmlal_u8(src[0_1],
                                            ; coeffabs_1);
    vst1.8          {d20},  [r1]!           ;store the result pu1_dst
    vhadd.s16       q5,     q5,     q11
    subs            r5,     r5,     #8      ;decrement the wd loop
    vqrshrun.s16    d8,     q5,     #6      ;right shift and saturating narrow
                                            ; result 2
    vst1.8          {d8},   [r6]!           ;store the result pu1_dst
    cmp             r5,     #4
    bgt             inner_loop_8

end_inner_loop_8
    subs            r14,    r14,    #2      ;decrement the ht loop
    add             r12,    r12,    r9      ;increment the src pointer by
                                            ; 2*src_strd-wd
    add             r1,     r1,     r8      ;increment the dst pointer by
                                            ; 2*dst_strd-wd
    bgt             outer_loop_8

    ldr             r10,    [sp,    #120]   ;loads wd
    cmp             r10,    #12
    beq             outer_loop4_residual

end_loops
    b               end_func

outer_loop_16
    str             r0,     [sp,  #-4]!
    str             r7,     [sp,  #-4]!
    add             r6,     r1,     r3      ;pu1_dst + dst_strd
    add             r4,     r12,    r2      ;pu1_src + src_strd
    and             r0,     r12,    #31
    mov             r7,     #0xc000
    sub             r5,     r10,    #0      ;checks wd
    pld             [r4,    r2,     lsl #1]
    pld             [r12,   r2,     lsl #1]
    vld1.u32        {q0},   [r12],  r11     ;vector load pu1_src
    vdup.16         q4,     r7
    vld1.u32        {q1},   [r12],  r11
    vld1.u32        {q2},   [r12],  r11
    vld1.u32        {q3},   [r12],  r11
    vmlsl.u8        q4,     d0,     d24     ;mul_res = vmlsl_u8(src[0_0],
                                            ; coeffabs_0);
    vld1.u32        {q6},   [r12],  r11
    vmlal.u8        q4,     d2,     d25     ;mul_res = vmlal_u8(src[0_1],
                                            ; coeffabs_1);
    vld1.u32        {q7},   [r12],  r11
    vmlsl.u8        q4,     d4,     d26     ;mul_res = vmlsl_u8(src[0_2],
                                            ; coeffabs_2);
    vld1.u32        {q8},   [r12],  r11
    vmlal.u8        q4,     d6,     d27     ;mul_res = vmull_u8(src[0_3],
                                            ; coeffabs_3);
    vld1.u32        {q9},   [r12],  r11
    vmlal.u8        q4,     d12,    d28     ;mul_res = vmlal_u8(src[0_4],
                                            ; coeffabs_4);
    vmlsl.u8        q4,     d14,    d29     ;mul_res = vmlsl_u8(src[0_5],
                                            ; coeffabs_5);
    vdup.16         q10,    r7
    vmlal.u8        q4,     d16,    d30     ;mul_res = vmlal_u8(src[0_6],
                                            ; coeffabs_6);
    vmlsl.u8        q4,     d18,    d31     ;mul_res = vmlsl_u8(src[0_7],
                                            ; coeffabs_7);

inner_loop_16
    vmlsl.u8        q10,    d1,     d24
    vdup.16         q5,     r7
    vmlal.u8        q10,    d3,     d25
    mov             r7,     #0x4000
    vdup.16         q11,    r7
    vmlsl.u8        q10,    d5,     d26
    vld1.u32        {q0},   [r4],   r11     ;vector load pu1_src
    vhadd.s16       q4,     q4,     q11
    vld1.u32        {q1},   [r4],   r11
    vmlal.u8        q10,    d7,     d27
    add             r12,    #8
    subs            r5,     r5,     #16
    vmlal.u8        q10,    d13,    d28
    vld1.u32        {q2},   [r4],   r11
    vmlsl.u8        q10,    d15,    d29
    vld1.u32        {q3},   [r4],   r11
    vqrshrun.s16    d8,     q4,     #6      ;right shift and saturating narrow
                                            ; result 1
    vmlal.u8        q10,    d17,    d30
    vld1.u32        {q6},   [r4],   r11
    vmlsl.u8        q10,    d19,    d31
    vld1.u32        {q7},   [r4],   r11
    vmlsl.u8        q5,     d0,     d24     ;mul_res = vmlsl_u8(src[0_0],
                                            ; coeffabs_0);
    vmlal.u8        q5,     d2,     d25     ;mul_res = vmlal_u8(src[0_1],
                                            ; coeffabs_1);
    vld1.u32        {q8},   [r4],   r11
    vhadd.s16       q10,    q10,    q11
    vld1.u32        {q9},   [r4],   r11
    vmlsl.u8        q5,     d4,     d26     ;mul_res = vmlsl_u8(src[0_2],
                                            ; coeffabs_2);
    vmlal.u8        q5,     d6,     d27     ;mul_res = vmull_u8(src[0_3],
                                            ; coeffabs_3);
    add             r4,     #8
    mov             r7,     #0xc000
    vmlal.u8        q5,     d12,    d28     ;mul_res = vmlal_u8(src[0_4],
                                            ; coeffabs_4);
    vmlsl.u8        q5,     d14,    d29     ;mul_res = vmlsl_u8(src[0_5],
                                            ; coeffabs_5);
    vqrshrun.s16    d9,     q10,    #6
    vdup.16         q11,    r7
    vmlal.u8        q5,     d16,    d30     ;mul_res = vmlal_u8(src[0_6],
                                            ; coeffabs_6);
    vmlsl.u8        q5,     d18,    d31     ;mul_res = vmlsl_u8(src[0_7],
                                            ; coeffabs_7);
    mov             r7,     #0x4000
    vmlsl.u8        q11,    d1,     d24
    vst1.8          {q4},   [r1]!           ;store the result pu1_dst
    vmlal.u8        q11,    d3,     d25
    vdup.16         q10,    r7
    vmlsl.u8        q11,    d5,     d26
    pld             [r12,   r2,     lsl #2]
    pld             [r4,    r2,     lsl #2]
    addeq           r12,    r12,    r9      ;increment the src pointer by
                                            ; 2*src_strd-wd
    addeq           r4,     r12,    r2      ;pu1_src + src_strd
    vmlal.u8        q11,    d7,     d27
    addeq           r1,     r1,     r8
    subeq           r14,    r14,    #2
    vmlal.u8        q11,    d13,    d28
    vhadd.s16       q5,     q5,     q10
    vmlsl.u8        q11,    d15,    d29
    vmlal.u8        q11,    d17,    d30
    cmp             r14,    #0
    vmlsl.u8        q11,    d19,    d31
    vqrshrun.s16    d10,    q5,     #6      ;right shift and saturating narrow
                                            ; result 2
    beq             epilog_16

    vld1.u32        {q0},   [r12],  r11     ;vector load pu1_src
    mov             r7,     #0xc000
    cmp             r5,     #0
    vld1.u32        {q1},   [r12],  r11
    vhadd.s16       q11,    q11,    q10
    vld1.u32        {q2},   [r12],  r11
    vdup.16         q4,     r7
    vld1.u32        {q3},   [r12],  r11
    vmlsl.u8        q4,     d0,     d24     ;mul_res = vmlsl_u8(src[0_0],
                                            ; coeffabs_0);
    vld1.u32        {q6},   [r12],  r11
    vld1.u32        {q7},   [r12],  r11
    vmlal.u8        q4,     d2,     d25     ;mul_res = vmlal_u8(src[0_1],
                                            ; coeffabs_1);
    vld1.u32        {q8},   [r12],  r11
    vmlsl.u8        q4,     d4,     d26     ;mul_res = vmlsl_u8(src[0_2],
                                            ; coeffabs_2);
    vld1.u32        {q9},   [r12],  r11
    vqrshrun.s16    d11,    q11,    #6
    vmlal.u8        q4,     d6,     d27     ;mul_res = vmull_u8(src[0_3],
                                            ; coeffabs_3);
    moveq           r5,     r10
    vmlal.u8        q4,     d12,    d28     ;mul_res = vmlal_u8(src[0_4],
                                            ; coeffabs_4);
    vdup.16         q10,    r7
    vmlsl.u8        q4,     d14,    d29     ;mul_res = vmlsl_u8(src[0_5],
                                            ; coeffabs_5);
    vst1.8          {q5},   [r6]!           ;store the result pu1_dst
    vmlal.u8        q4,     d16,    d30     ;mul_res = vmlal_u8(src[0_6],
                                            ; coeffabs_6);
    vmlsl.u8        q4,     d18,    d31     ;mul_res = vmlsl_u8(src[0_7],
                                            ; coeffabs_7);
    addeq           r6,     r1,     r3      ;pu1_dst + dst_strd
    b               inner_loop_16

epilog_16
    mov             r7,     #0x4000
    ldr             r0,     [sp],   #4
    ldr             r10,    [sp,    #120]
    vdup.16         q10,    r7
    vhadd.s16       q11,    q11,    q10
    vqrshrun.s16    d11,    q11,    #6
    vst1.8          {q5},   [r6]!           ;store the result pu1_dst
    ldr             r7,     [sp],   #4
    cmp             r10,    #24
    beq             outer_loop8_residual

end_loops1
    b               end_func

outer_loop4_residual
    sub             r12,    r0,     #3      ;pu1_src - 3
    mov             r1,     r7
    add             r1,     #8
    mov             r10,    #4
    add             r12,    #8
    mov             r14,    #16
    add             r8,     #4
    add             r9,     #4

outer_loop_4
    add             r6,     r1,     r3      ;pu1_dst + dst_strd
    add             r4,     r12,    r2      ;pu1_src + src_strd
    subs            r5,     r10,    #0      ;checks wd
    ble             end_inner_loop_4

inner_loop_4
    vld1.u32        {d0},   [r12],  r11     ;vector load pu1_src
    vld1.u32        {d1},   [r12],  r11
    vld1.u32        {d2},   [r12],  r11
    vld1.u32        {d3},   [r12],  r11
    vld1.u32        {d4},   [r12],  r11
    vld1.u32        {d5},   [r12],  r11
    vld1.u32        {d6},   [r12],  r11
    vld1.u32        {d7},   [r12],  r11
    sub             r12,    r12,    #4
    vld1.u32        {d12},  [r4],   r11     ;vector load pu1_src + src_strd
    vld1.u32        {d13},  [r4],   r11
    vzip.32         d0,     d12             ;vector zip the i iteration and ii
                                            ; interation in single register
    vld1.u32        {d14},  [r4],   r11
    vzip.32         d1,     d13
    vld1.u32        {d15},  [r4],   r11
    vzip.32         d2,     d14
    vld1.u32        {d16},  [r4],   r11
    vzip.32         d3,     d15
    vld1.u32        {d17},  [r4],   r11
    vzip.32         d4,     d16
    vld1.u32        {d18},  [r4],   r11
    vzip.32         d5,     d17
    vld1.u32        {d19},  [r4],   r11
    mov             r7,     #0xc000
    vdup.16         q4,     r7
    sub             r4,     r4,     #4
    vzip.32         d6,     d18
    vzip.32         d7,     d19
    vmlal.u8        q4,     d1,     d25     ;arithmetic operations for ii
                                            ; iteration in the same time
    vmlsl.u8        q4,     d0,     d24
    vmlsl.u8        q4,     d2,     d26
    vmlal.u8        q4,     d3,     d27
    vmlal.u8        q4,     d4,     d28
    vmlsl.u8        q4,     d5,     d29
    vmlal.u8        q4,     d6,     d30
    vmlsl.u8        q4,     d7,     d31
    mov             r7,     #0x4000
    vdup.16         q10,    r7
    vhadd.s16       q4,     q4,     q10
    vqrshrun.s16    d8,     q4,     #6
    vst1.32         {d8[0]},[r1]!           ;store the i iteration result which
                                            ; is in upper part of the register
    vst1.32         {d8[1]},[r6]!           ;store the ii iteration result which
                                            ; is in lower part of the register
    subs            r5,     r5,     #4      ;decrement the wd by 4
    bgt             inner_loop_4

end_inner_loop_4
    subs            r14,    r14,    #2      ;decrement the ht by 4
    add             r12,    r12,    r9      ;increment the input pointer
                                            ; 2*src_strd-wd
    add             r1,     r1,     r8      ;increment the output pointer
                                            ; 2*dst_strd-wd
    bgt             outer_loop_4

end_func
    vpop            {d8  -  d15}
    ldmfd           sp!,    {r4  -  r12,    r15} ;reload the registers from sp

    ENDP

    END
