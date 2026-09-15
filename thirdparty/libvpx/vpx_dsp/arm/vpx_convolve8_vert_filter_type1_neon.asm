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
;    r6 =>  dst_stride
;    r12 => filter_y0
;    r5 =>  ht
;    r3 =>  wd

    EXPORT          |vpx_convolve8_vert_filter_type1_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA  ||.text||, CODE, READONLY, ALIGN=2

|vpx_convolve8_vert_filter_type1_neon| PROC

    stmfd           sp!,    {r4  -  r12,    r14} ;stack stores the values of
                                                 ; the arguments
    vpush           {d8  -  d15}                 ; stack offset by 64
    mov             r4,     r1
    mov             r1,     r2
    mov             r2,     r4
    vmov.i16        q15,    #0x4000
    mov             r11,    #0xc000
    ldr             r12,    [sp,    #104]   ;load filter
    ldr             r6,     [sp,    #116]   ;load y0_q4
    add             r12,    r12,    r6,     lsl #4 ;r12 = filter[y0_q4]
    mov             r6,     r3
    ldr             r5,     [sp,    #124]   ;load wd
    vld2.8          {d0,    d1},    [r12]   ;coeff = vld1_s8(pi1_coeff)
    sub             r12,    r2,     r2,     lsl #2 ;src_ctrd & pi1_coeff
    vabs.s8         d0,     d0              ;vabs_s8(coeff)
    add             r0,     r0,     r12     ;r0->pu1_src    r12->pi1_coeff
    ldr             r3,     [sp,    #128]   ;load ht
    subs            r7,     r3,     #0      ;r3->ht
    vdup.u8         d22,    d0[0]           ;coeffabs_0 = vdup_lane_u8(coeffabs,
                                            ; 0);
    cmp             r5,     #8
    vdup.u8         d23,    d0[1]           ;coeffabs_1 = vdup_lane_u8(coeffabs,
                                            ; 1);
    vdup.u8         d24,    d0[2]           ;coeffabs_2 = vdup_lane_u8(coeffabs,
                                            ; 2);
    vdup.u8         d25,    d0[3]           ;coeffabs_3 = vdup_lane_u8(coeffabs,
                                            ; 3);
    vdup.u8         d26,    d0[4]           ;coeffabs_4 = vdup_lane_u8(coeffabs,
                                            ; 4);
    vdup.u8         d27,    d0[5]           ;coeffabs_5 = vdup_lane_u8(coeffabs,
                                            ; 5);
    vdup.u8         d28,    d0[6]           ;coeffabs_6 = vdup_lane_u8(coeffabs,
                                            ; 6);
    vdup.u8         d29,    d0[7]           ;coeffabs_7 = vdup_lane_u8(coeffabs,
                                            ; 7);
    blt             core_loop_wd_4          ;core loop wd 4 jump

    str             r0,     [sp,  #-4]!
    str             r1,     [sp,  #-4]!
    bic             r4,     r5,     #7      ;r5 ->wd
    rsb             r9,     r4,     r6,     lsl #2 ;r6->dst_strd    r5    ->wd
    rsb             r8,     r4,     r2,     lsl #2 ;r2->src_strd
    mov             r3,     r5,     lsr #3  ;divide by 8
    mul             r7,     r3              ;multiply height by width
    sub             r7,     #4              ;subtract by one for epilog

prolog
    and             r10,    r0,     #31
    add             r3,     r0,     r2      ;pu1_src_tmp += src_strd;
    vdup.16         q4,     r11
    vld1.u8         {d1},   [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vld1.u8         {d0},   [r0]!           ;src_tmp1 = vld1_u8(pu1_src_tmp);
    subs            r4,     r4,     #8
    vld1.u8         {d2},   [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q4,     d1,     d23     ;mul_res1 = vmull_u8(src_tmp2,
                                            ; coeffabs_1);
    vld1.u8         {d3},   [r3],   r2      ;src_tmp4 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q4,     d0,     d22     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp1, coeffabs_0);
    vld1.u8         {d4},   [r3],   r2      ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q4,     d2,     d24     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp3, coeffabs_2);
    vld1.u8         {d5},   [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q4,     d3,     d25     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp4, coeffabs_3);
    vld1.u8         {d6},   [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q4,     d4,     d26     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp1, coeffabs_4);
    vld1.u8         {d7},   [r3],   r2      ;src_tmp4 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q4,     d5,     d27     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp2, coeffabs_5);
    vld1.u8         {d16},  [r3],   r2      ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q4,     d6,     d28     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp3, coeffabs_6);
    vld1.u8         {d17},  [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q4,     d7,     d29     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp4, coeffabs_7);
    vdup.16         q5,     r11
    vld1.u8         {d18},  [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q5,     d2,     d23     ;mul_res2 = vmull_u8(src_tmp3,
                                            ; coeffabs_1);
    addle           r0,     r0,     r8
    vmlsl.u8        q5,     d1,     d22     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp2, coeffabs_0);
    bicle           r4,     r5,     #7      ;r5 ->wd
    vmlal.u8        q5,     d3,     d24     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp4, coeffabs_2);
    pld             [r3]
    vmlal.u8        q5,     d4,     d25     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp1, coeffabs_3);
    vhadd.s16       q4,     q4,     q15
    vdup.16         q6,     r11
    pld             [r3,    r2]
    vmlal.u8        q5,     d5,     d26     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp2, coeffabs_4);
    pld             [r3,    r2,     lsl #1]
    vmlal.u8        q5,     d6,     d27     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp3, coeffabs_5);
    add             r3,     r3,     r2
    vmlsl.u8        q5,     d7,     d28     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp4, coeffabs_6);
    pld             [r3,    r2,     lsl #1]
    vmlsl.u8        q5,     d16,    d29     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp1, coeffabs_7);
    add             r3,     r0,     r2      ;pu1_src_tmp += src_strd;
    vqrshrun.s16    d8,     q4,     #6      ;sto_res = vqmovun_s16(sto_res_tmp);
    vld1.u8         {d1},   [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q6,     d3,     d23
    vld1.u8         {d0},   [r0]!           ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q6,     d2,     d22
    vld1.u8         {d2},   [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q6,     d4,     d24
    vhadd.s16       q5,     q5,     q15
    vdup.16         q7,     r11
    vmlal.u8        q6,     d5,     d25
    vmlal.u8        q6,     d6,     d26
    vmlal.u8        q6,     d7,     d27
    vmlsl.u8        q6,     d16,    d28
    vmlsl.u8        q6,     d17,    d29
    add             r14,    r1,     r6
    vst1.8          {d8},   [r1]!           ;vst1_u8(pu1_dst,sto_res);
    vqrshrun.s16    d10,    q5,     #6      ;sto_res = vqmovun_s16(sto_res_tmp);
    addle           r1,     r1,     r9
    vmlsl.u8        q7,     d4,     d23
    subs            r7,     r7,     #4
    vmlsl.u8        q7,     d3,     d22
    vmlal.u8        q7,     d5,     d24
    vmlal.u8        q7,     d6,     d25
    vld1.u8         {d3},   [r3],   r2      ;src_tmp4 = vld1_u8(pu1_src_tmp);
    vhadd.s16       q6,     q6,     q15
    vdup.16         q4,     r11
    vmlal.u8        q7,     d7,     d26
    vld1.u8         {d4},   [r3],   r2      ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q7,     d16,    d27
    vld1.u8         {d5},   [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q7,     d17,    d28
    vld1.u8         {d6},   [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q7,     d18,    d29
    vld1.u8         {d7},   [r3],   r2      ;src_tmp4 = vld1_u8(pu1_src_tmp);
    vst1.8          {d10},  [r14],  r6      ;vst1_u8(pu1_dst_tmp,sto_res);
    vqrshrun.s16    d12,    q6,     #6
    blt             epilog_end              ;jumps to epilog_end

    beq             epilog                  ;jumps to epilog

main_loop_8
    subs            r4,     r4,     #8
    vmlsl.u8        q4,     d1,     d23     ;mul_res1 = vmull_u8(src_tmp2,
                                            ; coeffabs_1);
    addle           r0,     r0,     r8
    vmlsl.u8        q4,     d0,     d22     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp1, coeffabs_0);
    bicle           r4,     r5,     #7      ;r5 ->wd
    vmlal.u8        q4,     d2,     d24     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp3, coeffabs_2);
    vld1.u8         {d16},  [r3],   r2      ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q4,     d3,     d25     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp4, coeffabs_3);
    vhadd.s16       q7,     q7,     q15
    vdup.16         q5,     r11
    vld1.u8         {d17},  [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q4,     d4,     d26     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp1, coeffabs_4);
    vld1.u8         {d18},  [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q4,     d5,     d27     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp2, coeffabs_5);
    vmlsl.u8        q4,     d6,     d28     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp3, coeffabs_6);
    vmlsl.u8        q4,     d7,     d29     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp4, coeffabs_7);
    vst1.8          {d12},  [r14],  r6
    vqrshrun.s16    d14,    q7,     #6
    add             r3,     r0,     r2      ;pu1_src_tmp += src_strd;
    vmlsl.u8        q5,     d2,     d23     ;mul_res2 = vmull_u8(src_tmp3,
                                            ; coeffabs_1);
    vld1.u8         {d0},   [r0]!           ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q5,     d1,     d22     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp2, coeffabs_0);
    vmlal.u8        q5,     d3,     d24     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp4, coeffabs_2);
    vld1.u8         {d1},   [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q5,     d4,     d25     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp1, coeffabs_3);
    vhadd.s16       q4,     q4,     q15
    vdup.16         q6,     r11
    vst1.8          {d14},  [r14],  r6
    vmlal.u8        q5,     d5,     d26     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp2, coeffabs_4);
    add             r14,    r1,     #0
    vmlal.u8        q5,     d6,     d27     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp3, coeffabs_5);
    add             r1,     r1,     #8
    vmlsl.u8        q5,     d7,     d28     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp4, coeffabs_6);
    vmlsl.u8        q5,     d16,    d29     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp1, coeffabs_7);
    addle           r1,     r1,     r9
    vqrshrun.s16    d8,     q4,     #6      ;sto_res = vqmovun_s16(sto_res_tmp);
    vmlsl.u8        q6,     d3,     d23
    add             r10,    r3,     r2,     lsl #3 ; 10*strd - 8+2
    vmlsl.u8        q6,     d2,     d22
    add             r10,    r10,    r2      ; 11*strd
    vmlal.u8        q6,     d4,     d24
    vld1.u8         {d2},   [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q6,     d5,     d25
    vhadd.s16       q5,     q5,     q15
    vdup.16         q7,     r11
    vmlal.u8        q6,     d6,     d26
    vst1.8          {d8},   [r14],  r6      ;vst1_u8(pu1_dst,sto_res);
    pld             [r10]                   ;11+ 0
    vmlal.u8        q6,     d7,     d27
    pld             [r10,   r2]             ;11+ 1*strd
    vmlsl.u8        q6,     d16,    d28
    pld             [r10,   r2,     lsl #1] ;11+ 2*strd
    vmlsl.u8        q6,     d17,    d29
    add             r10,    r10,    r2      ;12*strd
    vqrshrun.s16    d10,    q5,     #6      ;sto_res = vqmovun_s16(sto_res_tmp);
    pld             [r10,   r2,     lsl #1] ;11+ 3*strd
    vmlsl.u8        q7,     d4,     d23
    vmlsl.u8        q7,     d3,     d22
    subs            r7,     r7,     #4
    vmlal.u8        q7,     d5,     d24
    vmlal.u8        q7,     d6,     d25
    vld1.u8         {d3},   [r3],   r2      ;src_tmp4 = vld1_u8(pu1_src_tmp);
    vhadd.s16       q6,     q6,     q15
    vdup.16         q4,     r11
    vmlal.u8        q7,     d7,     d26
    vld1.u8         {d4},   [r3],   r2      ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlal.u8        q7,     d16,    d27
    vld1.u8         {d5},   [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q7,     d17,    d28
    vld1.u8         {d6},   [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q7,     d18,    d29
    vld1.u8         {d7},   [r3],   r2      ;src_tmp4 = vld1_u8(pu1_src_tmp);
    vqrshrun.s16    d12,    q6,     #6
    vst1.8          {d10},  [r14],  r6      ;vst1_u8(pu1_dst_tmp,sto_res);
    bgt             main_loop_8             ;jumps to main_loop_8

epilog
    vmlsl.u8        q4,     d1,     d23     ;mul_res1 = vmull_u8(src_tmp2,
                                            ; coeffabs_1);
    vmlsl.u8        q4,     d0,     d22     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp1, coeffabs_0);
    vmlal.u8        q4,     d2,     d24     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp3, coeffabs_2);
    vmlal.u8        q4,     d3,     d25     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp4, coeffabs_3);
    vhadd.s16       q7,     q7,     q15
    vdup.16         q5,     r11
    vmlal.u8        q4,     d4,     d26     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp1, coeffabs_4);
    vmlal.u8        q4,     d5,     d27     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp2, coeffabs_5);
    vmlsl.u8        q4,     d6,     d28     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; src_tmp3, coeffabs_6);
    vmlsl.u8        q4,     d7,     d29     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; src_tmp4, coeffabs_7);
    vst1.8          {d12},  [r14],  r6
    vqrshrun.s16    d14,    q7,     #6
    vld1.u8         {d16},  [r3],   r2      ;src_tmp1 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q5,     d2,     d23     ;mul_res2 = vmull_u8(src_tmp3,
                                            ; coeffabs_1);
    vmlsl.u8        q5,     d1,     d22     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp2, coeffabs_0);
    vmlal.u8        q5,     d3,     d24     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp4, coeffabs_2);
    vmlal.u8        q5,     d4,     d25     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp1, coeffabs_3);
    vhadd.s16       q4,     q4,     q15
    vdup.16         q6,     r11
    vmlal.u8        q5,     d5,     d26     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp2, coeffabs_4);
    vmlal.u8        q5,     d6,     d27     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp3, coeffabs_5);
    vmlsl.u8        q5,     d7,     d28     ;mul_res2 = vmlal_u8(mul_res2,
                                            ; src_tmp4, coeffabs_6);
    vmlsl.u8        q5,     d16,    d29     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; src_tmp1, coeffabs_7);
    vst1.8          {d14},  [r14],  r6
    vqrshrun.s16    d8,     q4,     #6      ;sto_res = vqmovun_s16(sto_res_tmp);
    vld1.u8         {d17},  [r3],   r2      ;src_tmp2 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q6,     d3,     d23
    vmlsl.u8        q6,     d2,     d22
    vmlal.u8        q6,     d4,     d24
    vmlal.u8        q6,     d5,     d25
    vhadd.s16       q5,     q5,     q15
    vdup.16         q7,     r11
    vmlal.u8        q6,     d6,     d26
    vmlal.u8        q6,     d7,     d27
    vmlsl.u8        q6,     d16,    d28
    vmlsl.u8        q6,     d17,    d29
    add             r14,    r1,     r6
    vst1.8          {d8},   [r1]!           ;vst1_u8(pu1_dst,sto_res);
    vqrshrun.s16    d10,    q5,     #6      ;sto_res = vqmovun_s16(sto_res_tmp);
    vld1.u8         {d18},  [r3],   r2      ;src_tmp3 = vld1_u8(pu1_src_tmp);
    vmlsl.u8        q7,     d4,     d23
    vmlsl.u8        q7,     d3,     d22
    vmlal.u8        q7,     d5,     d24
    vmlal.u8        q7,     d6,     d25
    vhadd.s16       q6,     q6,     q15
    vmlal.u8        q7,     d7,     d26
    vmlal.u8        q7,     d16,    d27
    vmlsl.u8        q7,     d17,    d28
    vmlsl.u8        q7,     d18,    d29
    vst1.8          {d10},  [r14],  r6      ;vst1_u8(pu1_dst_tmp,sto_res);
    vqrshrun.s16    d12,    q6,     #6

epilog_end
    vst1.8          {d12},  [r14],  r6
    vhadd.s16       q7,     q7,     q15
    vqrshrun.s16    d14,    q7,     #6
    vst1.8          {d14},  [r14],  r6

end_loops
    tst             r5,     #7
    ldr             r1,     [sp],   #4
    ldr             r0,     [sp],   #4
    vpopeq          {d8  -  d15}
    ldmfdeq         sp!,    {r4  -  r12,    r15} ;reload the registers from
                                            ; sp
    mov             r5,     #4
    add             r0,     r0,     #8
    add             r1,     r1,     #8
    mov             r7,     #16

core_loop_wd_4
    rsb             r9,     r5,     r6,     lsl #2 ;r6->dst_strd    r5    ->wd
    rsb             r8,     r5,     r2,     lsl #2 ;r2->src_strd
    vmov.i8         d4,     #0

outer_loop_wd_4
    subs            r12,    r5,     #0
    ble             end_inner_loop_wd_4     ;outer loop jump

inner_loop_wd_4
    add             r3,     r0,     r2
    vld1.u32        {d4[1]},[r3],   r2      ;src_tmp1 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp1, 1);
    subs            r12,    r12,    #4
    vdup.u32        d5,     d4[1]           ;src_tmp2 = vdup_lane_u32(src_tmp1,
                                            ; 1);
    vld1.u32        {d5[1]},[r3],   r2      ;src_tmp2 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp2, 1);
    vld1.u32        {d4[0]},[r0]            ;src_tmp1 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp1, 0);
    vdup.16         q0,     r11
    vmlsl.u8        q0,     d5,     d23     ;mul_res1 =
                                            ; vmull_u8(vreinterpret_u8_u32(src_tmp2), coeffabs_1);

    vdup.u32        d6,     d5[1]           ;src_tmp3 = vdup_lane_u32(src_tmp2,
                                            ; 1);
    add             r0,     r0,     #4
    vld1.u32        {d6[1]},[r3],   r2      ;src_tmp3 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp3, 1);
    vmlsl.u8        q0,     d4,     d22     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; vreinterpret_u8_u32(src_tmp1), coeffabs_0);
    vdup.u32        d7,     d6[1]           ;src_tmp4 = vdup_lane_u32(src_tmp3,
                                            ; 1);
    vld1.u32        {d7[1]},[r3],   r2      ;src_tmp4 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp4, 1);
    vmlal.u8        q0,     d6,     d24     ;mul_res1 = vmlsl_u8(mul_res1,
                                            ; vreinterpret_u8_u32(src_tmp3), coeffabs_2);
    vdup.16         q4,     r11
    vmlsl.u8        q4,     d7,     d23
    vdup.u32        d4,     d7[1]           ;src_tmp1 = vdup_lane_u32(src_tmp4,
                                            ; 1);
    vmull.u8        q1,     d7,     d25     ;mul_res2 =
                                            ; vmull_u8(vreinterpret_u8_u32(src_tmp4), coeffabs_3);
    vld1.u32        {d4[1]},[r3],   r2      ;src_tmp1 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp1, 1);
    vmlsl.u8        q4,     d6,     d22
    vmlal.u8        q0,     d4,     d26     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; vreinterpret_u8_u32(src_tmp1), coeffabs_4);
    vdup.u32        d5,     d4[1]           ;src_tmp2 = vdup_lane_u32(src_tmp1,
                                            ; 1);
    vmlal.u8        q4,     d4,     d24
    vld1.u32        {d5[1]},[r3],   r2      ;src_tmp2 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp2, 1);
    vmlal.u8        q1,     d5,     d27     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; vreinterpret_u8_u32(src_tmp2), coeffabs_5);
    vdup.u32        d6,     d5[1]           ;src_tmp3 = vdup_lane_u32(src_tmp2,
                                            ; 1);
    vmlal.u8        q4,     d5,     d25
    vld1.u32        {d6[1]},[r3],   r2      ;src_tmp3 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp3, 1);
    vmlsl.u8        q0,     d6,     d28     ;mul_res1 = vmlal_u8(mul_res1,
                                            ; vreinterpret_u8_u32(src_tmp3), coeffabs_6);
    vdup.u32        d7,     d6[1]           ;src_tmp4 = vdup_lane_u32(src_tmp3,
                                            ; 1);
    vmlal.u8        q4,     d6,     d26
    vld1.u32        {d7[1]},[r3],   r2      ;src_tmp4 = vld1_lane_u32((uint32_t
                                            ; *)pu1_src_tmp, src_tmp4, 1);
    vmlsl.u8        q1,     d7,     d29     ;mul_res2 = vmlsl_u8(mul_res2,
                                            ; vreinterpret_u8_u32(src_tmp4), coeffabs_7);
    vdup.u32        d4,     d7[1]
    vadd.i16        q0,     q0,     q1      ;mul_res1 = vaddq_u16(mul_res1,
                                            ; mul_res2);
    vmlal.u8        q4,     d7,     d27
    vld1.u32        {d4[1]},[r3],   r2
    vmlsl.u8        q4,     d4,     d28
    vdup.u32        d5,     d4[1]
    vhadd.s16       q0,     q0,     q15
    vqrshrun.s16    d0,     q0,     #6      ;sto_res = vqmovun_s16(sto_res_tmp);
    vld1.u32        {d5[1]},[r3]
    add             r3,     r1,     r6
    vst1.32         {d0[0]},[r1]            ;vst1_lane_u32((uint32_t *)pu1_dst,
                                            ; vreinterpret_u32_u8(sto_res), 0);
    vmlsl.u8        q4,     d5,     d29
    vst1.32         {d0[1]},[r3],   r6      ;vst1_lane_u32((uint32_t
                                            ; *)pu1_dst_tmp, vreinterpret_u32_u8(sto_res), 1);
    vhadd.s16       q4,     q4,     q15
    vqrshrun.s16    d8,     q4,     #6
    vst1.32         {d8[0]},[r3],   r6
    add             r1,     r1,     #4
    vst1.32         {d8[1]},[r3]
    bgt             inner_loop_wd_4

end_inner_loop_wd_4
    subs            r7,     r7,     #4
    add             r1,     r1,     r9
    add             r0,     r0,     r8
    bgt             outer_loop_wd_4

    vpop            {d8  -  d15}
    ldmfd           sp!,    {r4  -  r12,    r15} ;reload the registers from sp

    ENDP

    END
