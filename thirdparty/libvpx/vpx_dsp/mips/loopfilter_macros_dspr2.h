/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_MIPS_LOOPFILTER_MACROS_DSPR2_H_
#define VPX_VPX_DSP_MIPS_LOOPFILTER_MACROS_DSPR2_H_

#include <stdlib.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_mem/vpx_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_DSPR2
#define STORE_F0()                                                       \
  {                                                                      \
    __asm__ __volatile__(                                                \
        "sb     %[q1_f0],    1(%[s4])           \n\t"                    \
        "sb     %[q0_f0],    0(%[s4])           \n\t"                    \
        "sb     %[p0_f0],   -1(%[s4])           \n\t"                    \
        "sb     %[p1_f0],   -2(%[s4])           \n\t"                    \
                                                                         \
        :                                                                \
        : [q1_f0] "r"(q1_f0), [q0_f0] "r"(q0_f0), [p0_f0] "r"(p0_f0),    \
          [p1_f0] "r"(p1_f0), [s4] "r"(s4));                             \
                                                                         \
    __asm__ __volatile__(                                                \
        "srl    %[q1_f0],   %[q1_f0],   8       \n\t"                    \
        "srl    %[q0_f0],   %[q0_f0],   8       \n\t"                    \
        "srl    %[p0_f0],   %[p0_f0],   8       \n\t"                    \
        "srl    %[p1_f0],   %[p1_f0],   8       \n\t"                    \
                                                                         \
        : [q1_f0] "+r"(q1_f0), [q0_f0] "+r"(q0_f0), [p0_f0] "+r"(p0_f0), \
          [p1_f0] "+r"(p1_f0)                                            \
        :);                                                              \
                                                                         \
    __asm__ __volatile__(                                                \
        "sb     %[q1_f0],    1(%[s3])           \n\t"                    \
        "sb     %[q0_f0],    0(%[s3])           \n\t"                    \
        "sb     %[p0_f0],   -1(%[s3])           \n\t"                    \
        "sb     %[p1_f0],   -2(%[s3])           \n\t"                    \
                                                                         \
        : [p1_f0] "+r"(p1_f0)                                            \
        : [q1_f0] "r"(q1_f0), [q0_f0] "r"(q0_f0), [s3] "r"(s3),          \
          [p0_f0] "r"(p0_f0));                                           \
                                                                         \
    __asm__ __volatile__(                                                \
        "srl    %[q1_f0],   %[q1_f0],   8       \n\t"                    \
        "srl    %[q0_f0],   %[q0_f0],   8       \n\t"                    \
        "srl    %[p0_f0],   %[p0_f0],   8       \n\t"                    \
        "srl    %[p1_f0],   %[p1_f0],   8       \n\t"                    \
                                                                         \
        : [q1_f0] "+r"(q1_f0), [q0_f0] "+r"(q0_f0), [p0_f0] "+r"(p0_f0), \
          [p1_f0] "+r"(p1_f0)                                            \
        :);                                                              \
                                                                         \
    __asm__ __volatile__(                                                \
        "sb     %[q1_f0],    1(%[s2])           \n\t"                    \
        "sb     %[q0_f0],    0(%[s2])           \n\t"                    \
        "sb     %[p0_f0],   -1(%[s2])           \n\t"                    \
        "sb     %[p1_f0],   -2(%[s2])           \n\t"                    \
                                                                         \
        :                                                                \
        : [q1_f0] "r"(q1_f0), [q0_f0] "r"(q0_f0), [p0_f0] "r"(p0_f0),    \
          [p1_f0] "r"(p1_f0), [s2] "r"(s2));                             \
                                                                         \
    __asm__ __volatile__(                                                \
        "srl    %[q1_f0],   %[q1_f0],   8       \n\t"                    \
        "srl    %[q0_f0],   %[q0_f0],   8       \n\t"                    \
        "srl    %[p0_f0],   %[p0_f0],   8       \n\t"                    \
        "srl    %[p1_f0],   %[p1_f0],   8       \n\t"                    \
                                                                         \
        : [q1_f0] "+r"(q1_f0), [q0_f0] "+r"(q0_f0), [p0_f0] "+r"(p0_f0), \
          [p1_f0] "+r"(p1_f0)                                            \
        :);                                                              \
                                                                         \
    __asm__ __volatile__(                                                \
        "sb     %[q1_f0],    1(%[s1])           \n\t"                    \
        "sb     %[q0_f0],    0(%[s1])           \n\t"                    \
        "sb     %[p0_f0],   -1(%[s1])           \n\t"                    \
        "sb     %[p1_f0],   -2(%[s1])           \n\t"                    \
                                                                         \
        :                                                                \
        : [q1_f0] "r"(q1_f0), [q0_f0] "r"(q0_f0), [p0_f0] "r"(p0_f0),    \
          [p1_f0] "r"(p1_f0), [s1] "r"(s1));                             \
  }

#define STORE_F1()                                                             \
  {                                                                            \
    __asm__ __volatile__(                                                      \
        "sb     %[q2_r],     2(%[s4])           \n\t"                          \
        "sb     %[q1_r],     1(%[s4])           \n\t"                          \
        "sb     %[q0_r],     0(%[s4])           \n\t"                          \
        "sb     %[p0_r],    -1(%[s4])           \n\t"                          \
        "sb     %[p1_r],    -2(%[s4])           \n\t"                          \
        "sb     %[p2_r],    -3(%[s4])           \n\t"                          \
                                                                               \
        :                                                                      \
        : [q2_r] "r"(q2_r), [q1_r] "r"(q1_r), [q0_r] "r"(q0_r),                \
          [p0_r] "r"(p0_r), [p1_r] "r"(p1_r), [p2_r] "r"(p2_r), [s4] "r"(s4)); \
                                                                               \
    __asm__ __volatile__(                                                      \
        "srl    %[q2_r],    %[q2_r],    16      \n\t"                          \
        "srl    %[q1_r],    %[q1_r],    16      \n\t"                          \
        "srl    %[q0_r],    %[q0_r],    16      \n\t"                          \
        "srl    %[p0_r],    %[p0_r],    16      \n\t"                          \
        "srl    %[p1_r],    %[p1_r],    16      \n\t"                          \
        "srl    %[p2_r],    %[p2_r],    16      \n\t"                          \
                                                                               \
        : [q2_r] "+r"(q2_r), [q1_r] "+r"(q1_r), [q0_r] "+r"(q0_r),             \
          [p0_r] "+r"(p0_r), [p1_r] "+r"(p1_r), [p2_r] "+r"(p2_r)              \
        :);                                                                    \
                                                                               \
    __asm__ __volatile__(                                                      \
        "sb     %[q2_r],     2(%[s3])           \n\t"                          \
        "sb     %[q1_r],     1(%[s3])           \n\t"                          \
        "sb     %[q0_r],     0(%[s3])           \n\t"                          \
        "sb     %[p0_r],    -1(%[s3])           \n\t"                          \
        "sb     %[p1_r],    -2(%[s3])           \n\t"                          \
        "sb     %[p2_r],    -3(%[s3])           \n\t"                          \
                                                                               \
        :                                                                      \
        : [q2_r] "r"(q2_r), [q1_r] "r"(q1_r), [q0_r] "r"(q0_r),                \
          [p0_r] "r"(p0_r), [p1_r] "r"(p1_r), [p2_r] "r"(p2_r), [s3] "r"(s3)); \
                                                                               \
    __asm__ __volatile__(                                                      \
        "sb     %[q2_l],     2(%[s2])           \n\t"                          \
        "sb     %[q1_l],     1(%[s2])           \n\t"                          \
        "sb     %[q0_l],     0(%[s2])           \n\t"                          \
        "sb     %[p0_l],    -1(%[s2])           \n\t"                          \
        "sb     %[p1_l],    -2(%[s2])           \n\t"                          \
        "sb     %[p2_l],    -3(%[s2])           \n\t"                          \
                                                                               \
        :                                                                      \
        : [q2_l] "r"(q2_l), [q1_l] "r"(q1_l), [q0_l] "r"(q0_l),                \
          [p0_l] "r"(p0_l), [p1_l] "r"(p1_l), [p2_l] "r"(p2_l), [s2] "r"(s2)); \
                                                                               \
    __asm__ __volatile__(                                                      \
        "srl    %[q2_l],    %[q2_l],    16      \n\t"                          \
        "srl    %[q1_l],    %[q1_l],    16      \n\t"                          \
        "srl    %[q0_l],    %[q0_l],    16      \n\t"                          \
        "srl    %[p0_l],    %[p0_l],    16      \n\t"                          \
        "srl    %[p1_l],    %[p1_l],    16      \n\t"                          \
        "srl    %[p2_l],    %[p2_l],    16      \n\t"                          \
                                                                               \
        : [q2_l] "+r"(q2_l), [q1_l] "+r"(q1_l), [q0_l] "+r"(q0_l),             \
          [p0_l] "+r"(p0_l), [p1_l] "+r"(p1_l), [p2_l] "+r"(p2_l)              \
        :);                                                                    \
                                                                               \
    __asm__ __volatile__(                                                      \
        "sb     %[q2_l],     2(%[s1])           \n\t"                          \
        "sb     %[q1_l],     1(%[s1])           \n\t"                          \
        "sb     %[q0_l],     0(%[s1])           \n\t"                          \
        "sb     %[p0_l],    -1(%[s1])           \n\t"                          \
        "sb     %[p1_l],    -2(%[s1])           \n\t"                          \
        "sb     %[p2_l],    -3(%[s1])           \n\t"                          \
                                                                               \
        :                                                                      \
        : [q2_l] "r"(q2_l), [q1_l] "r"(q1_l), [q0_l] "r"(q0_l),                \
          [p0_l] "r"(p0_l), [p1_l] "r"(p1_l), [p2_l] "r"(p2_l), [s1] "r"(s1)); \
  }

#define STORE_F2()                                                 \
  {                                                                \
    __asm__ __volatile__(                                          \
        "sb     %[q6_r],     6(%[s4])           \n\t"              \
        "sb     %[q5_r],     5(%[s4])           \n\t"              \
        "sb     %[q4_r],     4(%[s4])           \n\t"              \
        "sb     %[q3_r],     3(%[s4])           \n\t"              \
        "sb     %[q2_r],     2(%[s4])           \n\t"              \
        "sb     %[q1_r],     1(%[s4])           \n\t"              \
        "sb     %[q0_r],     0(%[s4])           \n\t"              \
        "sb     %[p0_r],    -1(%[s4])           \n\t"              \
        "sb     %[p1_r],    -2(%[s4])           \n\t"              \
        "sb     %[p2_r],    -3(%[s4])           \n\t"              \
        "sb     %[p3_r],    -4(%[s4])           \n\t"              \
        "sb     %[p4_r],    -5(%[s4])           \n\t"              \
        "sb     %[p5_r],    -6(%[s4])           \n\t"              \
        "sb     %[p6_r],    -7(%[s4])           \n\t"              \
                                                                   \
        :                                                          \
        : [q6_r] "r"(q6_r), [q5_r] "r"(q5_r), [q4_r] "r"(q4_r),    \
          [q3_r] "r"(q3_r), [q2_r] "r"(q2_r), [q1_r] "r"(q1_r),    \
          [q0_r] "r"(q0_r), [p0_r] "r"(p0_r), [p1_r] "r"(p1_r),    \
          [p2_r] "r"(p2_r), [p3_r] "r"(p3_r), [p4_r] "r"(p4_r),    \
          [p5_r] "r"(p5_r), [p6_r] "r"(p6_r), [s4] "r"(s4));       \
                                                                   \
    __asm__ __volatile__(                                          \
        "srl    %[q6_r],    %[q6_r],    16      \n\t"              \
        "srl    %[q5_r],    %[q5_r],    16      \n\t"              \
        "srl    %[q4_r],    %[q4_r],    16      \n\t"              \
        "srl    %[q3_r],    %[q3_r],    16      \n\t"              \
        "srl    %[q2_r],    %[q2_r],    16      \n\t"              \
        "srl    %[q1_r],    %[q1_r],    16      \n\t"              \
        "srl    %[q0_r],    %[q0_r],    16      \n\t"              \
        "srl    %[p0_r],    %[p0_r],    16      \n\t"              \
        "srl    %[p1_r],    %[p1_r],    16      \n\t"              \
        "srl    %[p2_r],    %[p2_r],    16      \n\t"              \
        "srl    %[p3_r],    %[p3_r],    16      \n\t"              \
        "srl    %[p4_r],    %[p4_r],    16      \n\t"              \
        "srl    %[p5_r],    %[p5_r],    16      \n\t"              \
        "srl    %[p6_r],    %[p6_r],    16      \n\t"              \
                                                                   \
        : [q6_r] "+r"(q6_r), [q5_r] "+r"(q5_r), [q4_r] "+r"(q4_r), \
          [q3_r] "+r"(q3_r), [q2_r] "+r"(q2_r), [q1_r] "+r"(q1_r), \
          [q0_r] "+r"(q0_r), [p0_r] "+r"(p0_r), [p1_r] "+r"(p1_r), \
          [p2_r] "+r"(p2_r), [p3_r] "+r"(p3_r), [p4_r] "+r"(p4_r), \
          [p5_r] "+r"(p5_r), [p6_r] "+r"(p6_r)                     \
        :);                                                        \
                                                                   \
    __asm__ __volatile__(                                          \
        "sb     %[q6_r],     6(%[s3])           \n\t"              \
        "sb     %[q5_r],     5(%[s3])           \n\t"              \
        "sb     %[q4_r],     4(%[s3])           \n\t"              \
        "sb     %[q3_r],     3(%[s3])           \n\t"              \
        "sb     %[q2_r],     2(%[s3])           \n\t"              \
        "sb     %[q1_r],     1(%[s3])           \n\t"              \
        "sb     %[q0_r],     0(%[s3])           \n\t"              \
        "sb     %[p0_r],    -1(%[s3])           \n\t"              \
        "sb     %[p1_r],    -2(%[s3])           \n\t"              \
        "sb     %[p2_r],    -3(%[s3])           \n\t"              \
        "sb     %[p3_r],    -4(%[s3])           \n\t"              \
        "sb     %[p4_r],    -5(%[s3])           \n\t"              \
        "sb     %[p5_r],    -6(%[s3])           \n\t"              \
        "sb     %[p6_r],    -7(%[s3])           \n\t"              \
                                                                   \
        :                                                          \
        : [q6_r] "r"(q6_r), [q5_r] "r"(q5_r), [q4_r] "r"(q4_r),    \
          [q3_r] "r"(q3_r), [q2_r] "r"(q2_r), [q1_r] "r"(q1_r),    \
          [q0_r] "r"(q0_r), [p0_r] "r"(p0_r), [p1_r] "r"(p1_r),    \
          [p2_r] "r"(p2_r), [p3_r] "r"(p3_r), [p4_r] "r"(p4_r),    \
          [p5_r] "r"(p5_r), [p6_r] "r"(p6_r), [s3] "r"(s3));       \
                                                                   \
    __asm__ __volatile__(                                          \
        "sb     %[q6_l],     6(%[s2])           \n\t"              \
        "sb     %[q5_l],     5(%[s2])           \n\t"              \
        "sb     %[q4_l],     4(%[s2])           \n\t"              \
        "sb     %[q3_l],     3(%[s2])           \n\t"              \
        "sb     %[q2_l],     2(%[s2])           \n\t"              \
        "sb     %[q1_l],     1(%[s2])           \n\t"              \
        "sb     %[q0_l],     0(%[s2])           \n\t"              \
        "sb     %[p0_l],    -1(%[s2])           \n\t"              \
        "sb     %[p1_l],    -2(%[s2])           \n\t"              \
        "sb     %[p2_l],    -3(%[s2])           \n\t"              \
        "sb     %[p3_l],    -4(%[s2])           \n\t"              \
        "sb     %[p4_l],    -5(%[s2])           \n\t"              \
        "sb     %[p5_l],    -6(%[s2])           \n\t"              \
        "sb     %[p6_l],    -7(%[s2])           \n\t"              \
                                                                   \
        :                                                          \
        : [q6_l] "r"(q6_l), [q5_l] "r"(q5_l), [q4_l] "r"(q4_l),    \
          [q3_l] "r"(q3_l), [q2_l] "r"(q2_l), [q1_l] "r"(q1_l),    \
          [q0_l] "r"(q0_l), [p0_l] "r"(p0_l), [p1_l] "r"(p1_l),    \
          [p2_l] "r"(p2_l), [p3_l] "r"(p3_l), [p4_l] "r"(p4_l),    \
          [p5_l] "r"(p5_l), [p6_l] "r"(p6_l), [s2] "r"(s2));       \
                                                                   \
    __asm__ __volatile__(                                          \
        "srl    %[q6_l],    %[q6_l],    16     \n\t"               \
        "srl    %[q5_l],    %[q5_l],    16     \n\t"               \
        "srl    %[q4_l],    %[q4_l],    16     \n\t"               \
        "srl    %[q3_l],    %[q3_l],    16     \n\t"               \
        "srl    %[q2_l],    %[q2_l],    16     \n\t"               \
        "srl    %[q1_l],    %[q1_l],    16     \n\t"               \
        "srl    %[q0_l],    %[q0_l],    16     \n\t"               \
        "srl    %[p0_l],    %[p0_l],    16     \n\t"               \
        "srl    %[p1_l],    %[p1_l],    16     \n\t"               \
        "srl    %[p2_l],    %[p2_l],    16     \n\t"               \
        "srl    %[p3_l],    %[p3_l],    16     \n\t"               \
        "srl    %[p4_l],    %[p4_l],    16     \n\t"               \
        "srl    %[p5_l],    %[p5_l],    16     \n\t"               \
        "srl    %[p6_l],    %[p6_l],    16     \n\t"               \
                                                                   \
        : [q6_l] "+r"(q6_l), [q5_l] "+r"(q5_l), [q4_l] "+r"(q4_l), \
          [q3_l] "+r"(q3_l), [q2_l] "+r"(q2_l), [q1_l] "+r"(q1_l), \
          [q0_l] "+r"(q0_l), [p0_l] "+r"(p0_l), [p1_l] "+r"(p1_l), \
          [p2_l] "+r"(p2_l), [p3_l] "+r"(p3_l), [p4_l] "+r"(p4_l), \
          [p5_l] "+r"(p5_l), [p6_l] "+r"(p6_l)                     \
        :);                                                        \
                                                                   \
    __asm__ __volatile__(                                          \
        "sb     %[q6_l],     6(%[s1])           \n\t"              \
        "sb     %[q5_l],     5(%[s1])           \n\t"              \
        "sb     %[q4_l],     4(%[s1])           \n\t"              \
        "sb     %[q3_l],     3(%[s1])           \n\t"              \
        "sb     %[q2_l],     2(%[s1])           \n\t"              \
        "sb     %[q1_l],     1(%[s1])           \n\t"              \
        "sb     %[q0_l],     0(%[s1])           \n\t"              \
        "sb     %[p0_l],    -1(%[s1])           \n\t"              \
        "sb     %[p1_l],    -2(%[s1])           \n\t"              \
        "sb     %[p2_l],    -3(%[s1])           \n\t"              \
        "sb     %[p3_l],    -4(%[s1])           \n\t"              \
        "sb     %[p4_l],    -5(%[s1])           \n\t"              \
        "sb     %[p5_l],    -6(%[s1])           \n\t"              \
        "sb     %[p6_l],    -7(%[s1])           \n\t"              \
                                                                   \
        :                                                          \
        : [q6_l] "r"(q6_l), [q5_l] "r"(q5_l), [q4_l] "r"(q4_l),    \
          [q3_l] "r"(q3_l), [q2_l] "r"(q2_l), [q1_l] "r"(q1_l),    \
          [q0_l] "r"(q0_l), [p0_l] "r"(p0_l), [p1_l] "r"(p1_l),    \
          [p2_l] "r"(p2_l), [p3_l] "r"(p3_l), [p4_l] "r"(p4_l),    \
          [p5_l] "r"(p5_l), [p6_l] "r"(p6_l), [s1] "r"(s1));       \
  }

#define PACK_LEFT_0TO3()                                              \
  {                                                                   \
    __asm__ __volatile__(                                             \
        "preceu.ph.qbl   %[p3_l],   %[p3]   \n\t"                     \
        "preceu.ph.qbl   %[p2_l],   %[p2]   \n\t"                     \
        "preceu.ph.qbl   %[p1_l],   %[p1]   \n\t"                     \
        "preceu.ph.qbl   %[p0_l],   %[p0]   \n\t"                     \
        "preceu.ph.qbl   %[q0_l],   %[q0]   \n\t"                     \
        "preceu.ph.qbl   %[q1_l],   %[q1]   \n\t"                     \
        "preceu.ph.qbl   %[q2_l],   %[q2]   \n\t"                     \
        "preceu.ph.qbl   %[q3_l],   %[q3]   \n\t"                     \
                                                                      \
        : [p3_l] "=&r"(p3_l), [p2_l] "=&r"(p2_l), [p1_l] "=&r"(p1_l), \
          [p0_l] "=&r"(p0_l), [q0_l] "=&r"(q0_l), [q1_l] "=&r"(q1_l), \
          [q2_l] "=&r"(q2_l), [q3_l] "=&r"(q3_l)                      \
        : [p3] "r"(p3), [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0),     \
          [q0] "r"(q0), [q1] "r"(q1), [q2] "r"(q2), [q3] "r"(q3));    \
  }

#define PACK_LEFT_4TO7()                                              \
  {                                                                   \
    __asm__ __volatile__(                                             \
        "preceu.ph.qbl   %[p7_l],   %[p7]   \n\t"                     \
        "preceu.ph.qbl   %[p6_l],   %[p6]   \n\t"                     \
        "preceu.ph.qbl   %[p5_l],   %[p5]   \n\t"                     \
        "preceu.ph.qbl   %[p4_l],   %[p4]   \n\t"                     \
        "preceu.ph.qbl   %[q4_l],   %[q4]   \n\t"                     \
        "preceu.ph.qbl   %[q5_l],   %[q5]   \n\t"                     \
        "preceu.ph.qbl   %[q6_l],   %[q6]   \n\t"                     \
        "preceu.ph.qbl   %[q7_l],   %[q7]   \n\t"                     \
                                                                      \
        : [p7_l] "=&r"(p7_l), [p6_l] "=&r"(p6_l), [p5_l] "=&r"(p5_l), \
          [p4_l] "=&r"(p4_l), [q4_l] "=&r"(q4_l), [q5_l] "=&r"(q5_l), \
          [q6_l] "=&r"(q6_l), [q7_l] "=&r"(q7_l)                      \
        : [p7] "r"(p7), [p6] "r"(p6), [p5] "r"(p5), [p4] "r"(p4),     \
          [q4] "r"(q4), [q5] "r"(q5), [q6] "r"(q6), [q7] "r"(q7));    \
  }

#define PACK_RIGHT_0TO3()                                             \
  {                                                                   \
    __asm__ __volatile__(                                             \
        "preceu.ph.qbr   %[p3_r],   %[p3]  \n\t"                      \
        "preceu.ph.qbr   %[p2_r],   %[p2]   \n\t"                     \
        "preceu.ph.qbr   %[p1_r],   %[p1]   \n\t"                     \
        "preceu.ph.qbr   %[p0_r],   %[p0]   \n\t"                     \
        "preceu.ph.qbr   %[q0_r],   %[q0]   \n\t"                     \
        "preceu.ph.qbr   %[q1_r],   %[q1]   \n\t"                     \
        "preceu.ph.qbr   %[q2_r],   %[q2]   \n\t"                     \
        "preceu.ph.qbr   %[q3_r],   %[q3]   \n\t"                     \
                                                                      \
        : [p3_r] "=&r"(p3_r), [p2_r] "=&r"(p2_r), [p1_r] "=&r"(p1_r), \
          [p0_r] "=&r"(p0_r), [q0_r] "=&r"(q0_r), [q1_r] "=&r"(q1_r), \
          [q2_r] "=&r"(q2_r), [q3_r] "=&r"(q3_r)                      \
        : [p3] "r"(p3), [p2] "r"(p2), [p1] "r"(p1), [p0] "r"(p0),     \
          [q0] "r"(q0), [q1] "r"(q1), [q2] "r"(q2), [q3] "r"(q3));    \
  }

#define PACK_RIGHT_4TO7()                                             \
  {                                                                   \
    __asm__ __volatile__(                                             \
        "preceu.ph.qbr   %[p7_r],   %[p7]   \n\t"                     \
        "preceu.ph.qbr   %[p6_r],   %[p6]   \n\t"                     \
        "preceu.ph.qbr   %[p5_r],   %[p5]   \n\t"                     \
        "preceu.ph.qbr   %[p4_r],   %[p4]   \n\t"                     \
        "preceu.ph.qbr   %[q4_r],   %[q4]   \n\t"                     \
        "preceu.ph.qbr   %[q5_r],   %[q5]   \n\t"                     \
        "preceu.ph.qbr   %[q6_r],   %[q6]   \n\t"                     \
        "preceu.ph.qbr   %[q7_r],   %[q7]   \n\t"                     \
                                                                      \
        : [p7_r] "=&r"(p7_r), [p6_r] "=&r"(p6_r), [p5_r] "=&r"(p5_r), \
          [p4_r] "=&r"(p4_r), [q4_r] "=&r"(q4_r), [q5_r] "=&r"(q5_r), \
          [q6_r] "=&r"(q6_r), [q7_r] "=&r"(q7_r)                      \
        : [p7] "r"(p7), [p6] "r"(p6), [p5] "r"(p5), [p4] "r"(p4),     \
          [q4] "r"(q4), [q5] "r"(q5), [q6] "r"(q6), [q7] "r"(q7));    \
  }

#define COMBINE_LEFT_RIGHT_0TO2()                                         \
  {                                                                       \
    __asm__ __volatile__(                                                 \
        "precr.qb.ph    %[p2],  %[p2_l],    %[p2_r]    \n\t"              \
        "precr.qb.ph    %[p1],  %[p1_l],    %[p1_r]    \n\t"              \
        "precr.qb.ph    %[p0],  %[p0_l],    %[p0_r]    \n\t"              \
        "precr.qb.ph    %[q0],  %[q0_l],    %[q0_r]    \n\t"              \
        "precr.qb.ph    %[q1],  %[q1_l],    %[q1_r]    \n\t"              \
        "precr.qb.ph    %[q2],  %[q2_l],    %[q2_r]    \n\t"              \
                                                                          \
        : [p2] "=&r"(p2), [p1] "=&r"(p1), [p0] "=&r"(p0), [q0] "=&r"(q0), \
          [q1] "=&r"(q1), [q2] "=&r"(q2)                                  \
        : [p2_l] "r"(p2_l), [p2_r] "r"(p2_r), [p1_l] "r"(p1_l),           \
          [p1_r] "r"(p1_r), [p0_l] "r"(p0_l), [p0_r] "r"(p0_r),           \
          [q0_l] "r"(q0_l), [q0_r] "r"(q0_r), [q1_l] "r"(q1_l),           \
          [q1_r] "r"(q1_r), [q2_l] "r"(q2_l), [q2_r] "r"(q2_r));          \
  }

#define COMBINE_LEFT_RIGHT_3TO6()                                         \
  {                                                                       \
    __asm__ __volatile__(                                                 \
        "precr.qb.ph    %[p6],  %[p6_l],    %[p6_r]    \n\t"              \
        "precr.qb.ph    %[p5],  %[p5_l],    %[p5_r]    \n\t"              \
        "precr.qb.ph    %[p4],  %[p4_l],    %[p4_r]    \n\t"              \
        "precr.qb.ph    %[p3],  %[p3_l],    %[p3_r]    \n\t"              \
        "precr.qb.ph    %[q3],  %[q3_l],    %[q3_r]    \n\t"              \
        "precr.qb.ph    %[q4],  %[q4_l],    %[q4_r]    \n\t"              \
        "precr.qb.ph    %[q5],  %[q5_l],    %[q5_r]    \n\t"              \
        "precr.qb.ph    %[q6],  %[q6_l],    %[q6_r]    \n\t"              \
                                                                          \
        : [p6] "=&r"(p6), [p5] "=&r"(p5), [p4] "=&r"(p4), [p3] "=&r"(p3), \
          [q3] "=&r"(q3), [q4] "=&r"(q4), [q5] "=&r"(q5), [q6] "=&r"(q6)  \
        : [p6_l] "r"(p6_l), [p5_l] "r"(p5_l), [p4_l] "r"(p4_l),           \
          [p3_l] "r"(p3_l), [p6_r] "r"(p6_r), [p5_r] "r"(p5_r),           \
          [p4_r] "r"(p4_r), [p3_r] "r"(p3_r), [q3_l] "r"(q3_l),           \
          [q4_l] "r"(q4_l), [q5_l] "r"(q5_l), [q6_l] "r"(q6_l),           \
          [q3_r] "r"(q3_r), [q4_r] "r"(q4_r), [q5_r] "r"(q5_r),           \
          [q6_r] "r"(q6_r));                                              \
  }

#endif  // #if HAVE_DSPR2
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_MIPS_LOOPFILTER_MACROS_DSPR2_H_
