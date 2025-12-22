/*
 *  Copyright 2024 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"
#include "libyuv/row_sve.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) && \
    defined(__aarch64__)

#define RGBTOARGB8_SVE_2X                                 \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h,  A: z19.b */ \
  "uqshrnb     z16.b, z16.h, #6     \n" /* B0 */          \
  "uqshrnb     z17.b, z17.h, #6     \n" /* G0 */          \
  "uqshrnb     z18.b, z18.h, #6     \n" /* R0 */          \
  "uqshrnt     z16.b, z20.h, #6     \n" /* B1 */          \
  "uqshrnt     z17.b, z21.h, #6     \n" /* G1 */          \
  "uqshrnt     z18.b, z22.h, #6     \n" /* R1 */

__arm_locally_streaming void I444ToARGBRow_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  // Streaming-SVE only, no use of ZA tile.
  uint64_t vl;
  asm volatile(
      "cntb     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // A
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.b                                    \n"
      "1:                                               \n"  //
      READYUV444_SVE_2X I444TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]] \n"
      "incb     %[dst_argb], all, mul #4                \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.b, wzr, %w[width]                    \n"  //
      READYUV444_SVE_2X I444TORGB_SVE_2X RGBTOARGB8_SVE_2X
      "st4b     {z16.b, z17.b, z18.b, z19.b}, p1, [%[dst_argb]] \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width),                               // %[width]
        [vl] "=&r"(vl)                                     // %[vl]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_SVE_REGS);
}

__arm_locally_streaming void I444ToRGB24Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) {
  I444ToRGB24Row_SVE_SC(src_y, src_u, src_v, dst_rgb24, yuvconstants, width);
}

__arm_locally_streaming void I400ToARGBRow_SME(
    const uint8_t* src_y,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  // Streaming-SVE only, no use of ZA tile.
  I400ToARGBRow_SVE_SC(src_y, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I422ToARGBRow_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  // Streaming-SVE only, no use of ZA tile.
  I422ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I422ToRGB24Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I422ToRGB24Row_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I422ToRGB565Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_rgb565,
    const struct YuvConstants* yuvconstants,
    int width) {
  I422ToRGB565Row_SVE_SC(src_y, src_u, src_v, dst_rgb565, yuvconstants, width);
}

__arm_locally_streaming void I422ToARGB1555Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb1555,
    const struct YuvConstants* yuvconstants,
    int width) {
  I422ToARGB1555Row_SVE_SC(src_y, src_u, src_v, dst_argb1555, yuvconstants,
                           width);
}

__arm_locally_streaming void I422ToARGB4444Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb4444,
    const struct YuvConstants* yuvconstants,
    int width) {
  I422ToARGB4444Row_SVE_SC(src_y, src_u, src_v, dst_argb4444, yuvconstants,
                           width);
}

__arm_locally_streaming void I422ToRGBARow_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I422ToRGBARow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I422ToAR30Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I422ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I422AlphaToARGBRow_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    const uint8_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I422AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

__arm_locally_streaming void I444AlphaToARGBRow_SME(
    const uint8_t* src_y,
    const uint8_t* src_u,
    const uint8_t* src_v,
    const uint8_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I444AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

__arm_locally_streaming void NV12ToARGBRow_SME(
    const uint8_t* src_y,
    const uint8_t* src_uv,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  NV12ToARGBRow_SVE_SC(src_y, src_uv, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void NV21ToARGBRow_SME(
    const uint8_t* src_y,
    const uint8_t* src_vu,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  NV21ToARGBRow_SVE_SC(src_y, src_vu, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void NV12ToRGB24Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_uv,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) {
  NV12ToRGB24Row_SVE_SC(src_y, src_uv, dst_rgb24, yuvconstants, width);
}

__arm_locally_streaming void NV21ToRGB24Row_SME(
    const uint8_t* src_y,
    const uint8_t* src_vu,
    uint8_t* dst_rgb24,
    const struct YuvConstants* yuvconstants,
    int width) {
  NV21ToRGB24Row_SVE_SC(src_y, src_vu, dst_rgb24, yuvconstants, width);
}

__arm_locally_streaming void YUY2ToARGBRow_SME(
    const uint8_t* src_yuy2,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  YUY2ToARGBRow_SVE_SC(src_yuy2, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void UYVYToARGBRow_SME(
    const uint8_t* src_uyvy,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  UYVYToARGBRow_SVE_SC(src_uyvy, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I210ToARGBRow_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I210ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I210AlphaToARGBRow_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    const uint16_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I210AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

__arm_locally_streaming void I210ToAR30Row_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    uint8_t* dst_ar30,
    const struct YuvConstants* yuvconstants,
    int width) {
  I210ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
}

__arm_locally_streaming void P210ToARGBRow_SME(
    const uint16_t* src_y,
    const uint16_t* src_uv,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  P210ToARGBRow_SVE_SC(src_y, src_uv, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void P210ToAR30Row_SME(
    const uint16_t* src_y,
    const uint16_t* src_uv,
    uint8_t* dst_ar30,
    const struct YuvConstants* yuvconstants,
    int width) {
  P210ToAR30Row_SVE_SC(src_y, src_uv, dst_ar30, yuvconstants, width);
}

__arm_locally_streaming void I410ToARGBRow_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I410ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void I410AlphaToARGBRow_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    const uint16_t* src_a,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I410AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

__arm_locally_streaming void I410ToAR30Row_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    uint8_t* dst_ar30,
    const struct YuvConstants* yuvconstants,
    int width) {
  I410ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
}

__arm_locally_streaming void P410ToARGBRow_SME(
    const uint16_t* src_y,
    const uint16_t* src_uv,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  P410ToARGBRow_SVE_SC(src_y, src_uv, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void P410ToAR30Row_SME(
    const uint16_t* src_y,
    const uint16_t* src_uv,
    uint8_t* dst_ar30,
    const struct YuvConstants* yuvconstants,
    int width) {
  P410ToAR30Row_SVE_SC(src_y, src_uv, dst_ar30, yuvconstants, width);
}

__arm_locally_streaming void I212ToAR30Row_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    uint8_t* dst_ar30,
    const struct YuvConstants* yuvconstants,
    int width) {
  I212ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
}

__arm_locally_streaming void I212ToARGBRow_SME(
    const uint16_t* src_y,
    const uint16_t* src_u,
    const uint16_t* src_v,
    uint8_t* dst_argb,
    const struct YuvConstants* yuvconstants,
    int width) {
  I212ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

__arm_locally_streaming void MultiplyRow_16_SME(const uint16_t* src_y,
                                                uint16_t* dst_y,
                                                int scale,
                                                int width) {
  // Streaming-SVE only, no use of ZA tile.
  int vl;
  asm volatile(
      "cnth    %x[vl]                                   \n"
      "mov     z0.h, %w[scale]                          \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "b.lt    2f                                       \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue   p0.h                                     \n"
      "1:                                               \n"
      "ld1h    {z1.h}, p0/z, [%[src_y]]                 \n"
      "incb    %[src_y]                                 \n"
      "mul     z1.h, z0.h, z1.h                         \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "st1h    {z1.h}, p0, [%[dst_y]]                   \n"
      "incb    %[dst_y]                                 \n"
      "b.ge    1b                                       \n"

      "2:                                               \n"
      "adds    %w[width], %w[width], %w[vl]             \n"
      "b.eq    99f                                      \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt p0.h, wzr, %w[width]                     \n"
      "ld1h    {z1.h}, p0/z, [%[src_y]]                 \n"
      "mul     z1.h, z0.h, z1.h                         \n"
      "st1h    {z1.h}, p0, [%[dst_y]]                   \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),  // %[src_y]
        [dst_y] "+r"(dst_y),  // %[dst_y]
        [width] "+r"(width),  // %[width]
        [vl] "=&r"(vl)        // %[vl]
      : [scale] "r"(scale)    // %[scale]
      : "memory", "cc", "z0", "z1", "p0");
}

__arm_locally_streaming void ARGBMultiplyRow_SME(const uint8_t* src_argb,
                                                 const uint8_t* src_argb1,
                                                 uint8_t* dst_argb,
                                                 int width) {
  // Streaming-SVE only, no use of ZA tile.
  width *= 4;
  int vl;
  asm volatile(
      "cntb    %x[vl]                                   \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "b.lt    2f                                       \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue   p0.b                                     \n"
      "1:                                               \n"
      "ld1b    {z0.b}, p0/z, [%[src_argb]]              \n"
      "ld1b    {z1.b}, p0/z, [%[src_argb1]]             \n"
      "incb    %[src_argb]                              \n"
      "incb    %[src_argb1]                             \n"
      "umullb  z2.h, z0.b, z1.b                         \n"
      "umullt  z1.h, z0.b, z1.b                         \n"
      "rshrnb  z0.b, z2.h, #8                           \n"
      "rshrnt  z0.b, z1.h, #8                           \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "st1b    {z0.b}, p0, [%[dst_argb]]                \n"
      "incb    %[dst_argb]                              \n"
      "b.ge    1b                                       \n"

      "2:                                               \n"
      "adds    %w[width], %w[width], %w[vl]             \n"
      "b.eq    99f                                      \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt p0.b, wzr, %w[width]                     \n"
      "ld1b    {z0.b}, p0/z, [%[src_argb]]              \n"
      "ld1b    {z1.b}, p0/z, [%[src_argb1]]             \n"
      "umullb  z2.h, z0.b, z1.b                         \n"
      "umullt  z1.h, z0.b, z1.b                         \n"
      "rshrnb  z0.b, z2.h, #8                           \n"
      "rshrnt  z0.b, z1.h, #8                           \n"
      "st1b    {z0.b}, p0, [%[dst_argb]]                \n"

      "99:                                              \n"
      : [src_argb] "+r"(src_argb),    // %[src_argb]
        [src_argb1] "+r"(src_argb1),  // %[src_argb1]
        [dst_argb] "+r"(dst_argb),    // %[dst_argb]
        [width] "+r"(width),          // %[width]
        [vl] "=&r"(vl)                // %[vl]
      :
      : "memory", "cc", "z0", "z1", "z2", "p0", "p1");
}

__arm_locally_streaming void MergeUVRow_SME(const uint8_t* src_u,
                                            const uint8_t* src_v,
                                            uint8_t* dst_uv,
                                            int width) {
  // Streaming-SVE only, no use of ZA tile.
  int vl;
  asm volatile(
      "cntb    %x[vl]                                   \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "b.lt    2f                                       \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue   p0.b                                     \n"
      "1:                                               \n"
      "ld1b    {z1.b}, p0/z, [%[src_u]]                 \n"
      "ld1b    {z2.b}, p0/z, [%[src_v]]                 \n"
      "incb    %[src_u]                                 \n"
      "incb    %[src_v]                                 \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "st2b    {z1.b, z2.b}, p0, [%[dst_uv]]            \n"
      "incb    %[dst_uv], all, mul #2                   \n"
      "b.ge    1b                                       \n"

      "2:                                               \n"
      "adds    %w[width], %w[width], %w[vl]             \n"
      "b.eq    99f                                      \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt p0.b, wzr, %w[width]                     \n"
      "ld1b    {z1.b}, p0/z, [%[src_u]]                 \n"
      "ld1b    {z2.b}, p0/z, [%[src_v]]                 \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "st2b    {z1.b, z2.b}, p0, [%[dst_uv]]            \n"

      "99:                                              \n"
      : [src_u] "+r"(src_u),    // %[src_u]
        [src_v] "+r"(src_v),    // %[src_v]
        [dst_uv] "+r"(dst_uv),  // %[dst_uv]
        [width] "+r"(width),    // %[width]
        [vl] "=&r"(vl)          // %[vl]
      :
      : "memory", "cc", "z0", "z1", "z2", "p0");
}

__arm_locally_streaming void MergeUVRow_16_SME(const uint16_t* src_u,
                                               const uint16_t* src_v,
                                               uint16_t* dst_uv,
                                               int depth,
                                               int width) {
  int shift = 16 - depth;
  // Streaming-SVE only, no use of ZA tile.
  int vl;
  asm volatile(
      "cnth    %x[vl]                                   \n"
      "mov     z0.h, %w[shift]                          \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "b.lt    2f                                       \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue   p0.h                                     \n"
      "1:                                               \n"
      "ld1h    {z1.h}, p0/z, [%[src_u]]                 \n"
      "ld1h    {z2.h}, p0/z, [%[src_v]]                 \n"
      "incb    %[src_u]                                 \n"
      "incb    %[src_v]                                 \n"
      "lsl     z1.h, p0/m, z1.h, z0.h                   \n"
      "lsl     z2.h, p0/m, z2.h, z0.h                   \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "st2h    {z1.h, z2.h}, p0, [%[dst_uv]]            \n"
      "incb    %[dst_uv], all, mul #2                   \n"
      "b.ge    1b                                       \n"

      "2:                                               \n"
      "adds    %w[width], %w[width], %w[vl]             \n"
      "b.eq    99f                                      \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt p0.h, wzr, %w[width]                     \n"
      "ld1h    {z1.h}, p0/z, [%[src_u]]                 \n"
      "ld1h    {z2.h}, p0/z, [%[src_v]]                 \n"
      "lsl     z1.h, p0/m, z1.h, z0.h                   \n"
      "lsl     z2.h, p0/m, z2.h, z0.h                   \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "st2h    {z1.h, z2.h}, p0, [%[dst_uv]]            \n"

      "99:                                              \n"
      : [src_u] "+r"(src_u),    // %[src_u]
        [src_v] "+r"(src_v),    // %[src_v]
        [dst_uv] "+r"(dst_uv),  // %[dst_uv]
        [width] "+r"(width),    // %[width]
        [vl] "=&r"(vl)          // %[vl]
      : [shift] "r"(shift)      // %[shift]
      : "memory", "cc", "z0", "z1", "z2", "p0");
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits = shr 1
// 16384 = 10 bits = shr 2
// 4096 = 12 bits = shr 4
// 256 = 16 bits = shr 8
__arm_locally_streaming void Convert16To8Row_SME(const uint16_t* src_y,
                                                 uint8_t* dst_y,
                                                 int scale,
                                                 int width) {
  // 15 - clz(scale), + 8 to shift result into the high half of the lane to
  // saturate, then we can just use UZP2 to narrow rather than a pair of
  // saturating narrow instructions.
  int shift = 23 - __builtin_clz((int32_t)scale);
  int vl;
  asm volatile(
      "cntb     %x[vl]                                  \n"
      "dup      z0.h, %w[shift]                         \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p0.b                                    \n"
      "1:                                               \n"
      "ld1h     {z1.h}, p0/z, [%[src_y]]                \n"
      "ld1h     {z2.h}, p0/z, [%[src_y], #1, mul vl]    \n"
      "incb     %[src_y], all, mul #2                   \n"
      "uqshl    z1.h, p0/m, z1.h, z0.h                  \n"
      "uqshl    z2.h, p0/m, z2.h, z0.h                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "uzp2     z1.b, z1.b, z2.b                        \n"
      "st1b     {z1.b}, p0, [%[dst_y]]                  \n"
      "incb     %[dst_y]                                \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      // We need separate predicates for the load and store instructions since
      // they are operating on different element sizes (.b vs .h).
      "cnth     %x[vl]                                  \n"
      "whilelt  p0.h, wzr, %w[width]                    \n"
      "whilelt  p1.h, %w[vl], %w[width]                 \n"
      "whilelt  p2.b, wzr, %w[width]                    \n"
      "ld1h     {z1.h}, p0/z, [%[src_y]]                \n"
      "ld1h     {z2.h}, p1/z, [%[src_y], #1, mul vl]    \n"
      "uqshl    z1.h, p0/m, z1.h, z0.h                  \n"
      "uqshl    z2.h, p1/m, z2.h, z0.h                  \n"
      "uzp2     z1.b, z1.b, z2.b                        \n"
      "st1b     {z1.b}, p2, [%[dst_y]]                  \n"

      "99:                                              \n"
      : [src_y] "+r"(src_y),  // %[src_y]
        [dst_y] "+r"(dst_y),  // %[dst_y]
        [width] "+r"(width),  // %[width]
        [vl] "=&r"(vl)        // %[vl]
      : [shift] "r"(shift)    // %[shift]
      : "cc", "memory", "z0", "z1", "z2", "p0", "p1", "p2");
}

__arm_locally_streaming void CopyRow_SME(const uint8_t* src,
                                         uint8_t* dst,
                                         int width) {
  // Streaming-SVE only, no use of ZA tile.
  int vl;
  asm volatile(
      "cntb    %x[vl]                                   \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "b.lt    2f                                       \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue   p0.b                                     \n"
      "1:                                               \n"
      "ld1b    {z0.b}, p0/z, [%[src]]                   \n"
      "incb    %[src]                                   \n"
      "subs    %w[width], %w[width], %w[vl]             \n"
      "st1b    {z0.b}, p0, [%[dst]]                     \n"
      "incb    %[dst]                                   \n"
      "b.ge    1b                                       \n"

      "2:                                               \n"
      "adds    %w[width], %w[width], %w[vl]             \n"
      "b.eq    99f                                      \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt p0.b, wzr, %w[width]                     \n"
      "ld1b    {z0.b}, p0/z, [%[src]]                   \n"
      "st1b    {z0.b}, p0, [%[dst]]                     \n"

      "99:                                              \n"
      : [src] "+r"(src),      // %[src]
        [dst] "+r"(dst),      // %[dst]
        [width] "+r"(width),  // %[width]
        [vl] "=&r"(vl)        // %[vl]
      :
      : "memory", "cc", "z0", "p0");
}

__arm_locally_streaming static void HalfRow_SME(uint8_t* dst_ptr,
                                                const uint8_t* src_ptr,
                                                ptrdiff_t src_stride,
                                                int width) {
  const uint8_t* src_ptr1 = src_ptr + src_stride;

  int vl;
  asm volatile(
      "cntb     %x[vl]                                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p0.b                                    \n"
      "1:                                               \n"
      "ld1b     {z2.b}, p0/z, [%[src_ptr]]              \n"
      "ld1b     {z3.b}, p0/z, [%[src_ptr1]]             \n"
      "incb     %[src_ptr]                              \n"
      "incb     %[src_ptr1]                             \n"
      "urhadd   z2.b, p0/m, z2.b, z3.b                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st1b     {z2.b}, p0, [%[dst_ptr]]                \n"
      "incb     %[dst_ptr]                              \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p0.b, wzr, %w[width]                    \n"
      "ld1b     {z2.b}, p0/z, [%[src_ptr]]              \n"
      "ld1b     {z3.b}, p0/z, [%[src_ptr1]]             \n"
      "urhadd   z2.b, p0/m, z2.b, z3.b                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st1b     {z2.b}, p0, [%[dst_ptr]]                \n"

      "99:                                              \n"
      : [src_ptr] "+r"(src_ptr),    // %[src_ptr]
        [src_ptr1] "+r"(src_ptr1),  // %[src_ptr1]
        [dst_ptr] "+r"(dst_ptr),    // %[dst_ptr]
        [width] "+r"(width),        // %[width]
        [vl] "=&r"(vl)              // %[vl]
      :
      : "cc", "memory", "z0", "z1", "z2", "z3", "p0");
}

__arm_locally_streaming void InterpolateRow_SME(uint8_t* dst_ptr,
                                                const uint8_t* src_ptr,
                                                ptrdiff_t src_stride,
                                                int width,
                                                int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint8_t* src_ptr1 = src_ptr + src_stride;

  if (y0_fraction == 0) {
    CopyRow_SME(src_ptr1, dst_ptr, width);
    return;
  }
  if (y0_fraction == 128) {
    HalfRow_SME(dst_ptr, src_ptr, src_stride, width);
    return;
  }
  if (y0_fraction == 256) {
    CopyRow_SME(src_ptr, dst_ptr, width);
    return;
  }

  int vl;
  asm volatile(
      "cntb     %x[vl]                                  \n"
      "dup      z0.b, %w[y0_fraction]                   \n"
      "dup      z1.b, %w[y1_fraction]                   \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p0.b                                    \n"
      "1:                                               \n"
      "ld1b     {z2.b}, p0/z, [%[src_ptr]]              \n"
      "ld1b     {z3.b}, p0/z, [%[src_ptr1]]             \n"
      "incb     %[src_ptr]                              \n"
      "incb     %[src_ptr1]                             \n"
      "umullb   z4.h, z2.b, z0.b                        \n"
      "umullt   z2.h, z2.b, z0.b                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "umlalb   z4.h, z3.b, z1.b                        \n"
      "umlalt   z2.h, z3.b, z1.b                        \n"
      "rshrnb   z3.b, z4.h, #8                          \n"
      "rshrnt   z3.b, z2.h, #8                          \n"
      "st1b     {z3.b}, p0, [%[dst_ptr]]                \n"
      "incb     %[dst_ptr]                              \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p0.b, wzr, %w[width]                    \n"
      "ld1b     {z2.b}, p0/z, [%[src_ptr]]              \n"
      "ld1b     {z3.b}, p0/z, [%[src_ptr1]]             \n"
      "umullb   z4.h, z2.b, z0.b                        \n"
      "umullt   z2.h, z2.b, z0.b                        \n"
      "umlalb   z4.h, z3.b, z1.b                        \n"
      "umlalt   z2.h, z3.b, z1.b                        \n"
      "rshrnb   z3.b, z4.h, #8                          \n"
      "rshrnt   z3.b, z2.h, #8                          \n"
      "st1b     {z3.b}, p0, [%[dst_ptr]]                \n"

      "99:                                              \n"
      : [src_ptr] "+r"(src_ptr),         // %[src_ptr]
        [src_ptr1] "+r"(src_ptr1),       // %[src_ptr1]
        [dst_ptr] "+r"(dst_ptr),         // %[dst_ptr]
        [width] "+r"(width),             // %[width]
        [vl] "=&r"(vl)                   // %[vl]
      : [y0_fraction] "r"(y0_fraction),  // %[y0_fraction]
        [y1_fraction] "r"(y1_fraction)   // %[y1_fraction]
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "p0");
}

__arm_locally_streaming static void HalfRow_16_SME(uint16_t* dst_ptr,
                                                   const uint16_t* src_ptr,
                                                   ptrdiff_t src_stride,
                                                   int width) {
  const uint16_t* src_ptr1 = src_ptr + src_stride;

  int vl;
  asm volatile(
      "cnth     %x[vl]                                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p0.h                                    \n"
      "1:                                               \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "incb     %[src_ptr]                              \n"
      "incb     %[src_ptr1]                             \n"
      "urhadd   z2.h, p0/m, z2.h, z3.h                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st1h     {z2.h}, p0, [%[dst_ptr]]                \n"
      "incb     %[dst_ptr]                              \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p0.h, wzr, %w[width]                    \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "urhadd   z2.h, p0/m, z2.h, z3.h                  \n"
      "st1h     {z2.h}, p0, [%[dst_ptr]]                \n"

      "99:                                              \n"
      : [src_ptr] "+r"(src_ptr),    // %[src_ptr]
        [src_ptr1] "+r"(src_ptr1),  // %[src_ptr1]
        [dst_ptr] "+r"(dst_ptr),    // %[dst_ptr]
        [width] "+r"(width),        // %[width]
        [vl] "=&r"(vl)              // %[vl]
      :
      : "cc", "memory", "z0", "z1", "z2", "z3", "p0");
}

__arm_locally_streaming void InterpolateRow_16_SME(uint16_t* dst_ptr,
                                                   const uint16_t* src_ptr,
                                                   ptrdiff_t src_stride,
                                                   int width,
                                                   int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint16_t* src_ptr1 = src_ptr + src_stride;

  if (y0_fraction == 0) {
    CopyRow_SME((const uint8_t*)src_ptr1, (uint8_t*)dst_ptr,
                width * sizeof(uint16_t));
    return;
  }
  if (y0_fraction == 128) {
    HalfRow_16_SME(dst_ptr, src_ptr, src_stride, width);
    return;
  }
  if (y0_fraction == 256) {
    CopyRow_SME((const uint8_t*)src_ptr, (uint8_t*)dst_ptr,
                width * sizeof(uint16_t));
    return;
  }

  int vl;
  asm volatile(
      "cnth     %x[vl]                                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "dup      z0.h, %w[y0_fraction]                   \n"
      "dup      z1.h, %w[y1_fraction]                   \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p0.h                                    \n"
      "1:                                               \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "incb     %[src_ptr]                              \n"
      "incb     %[src_ptr1]                             \n"
      "umullb   z4.s, z2.h, z0.h                        \n"
      "umullt   z2.s, z2.h, z0.h                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "umlalb   z4.s, z3.h, z1.h                        \n"
      "umlalt   z2.s, z3.h, z1.h                        \n"
      "rshrnb   z3.h, z4.s, #8                          \n"
      "rshrnt   z3.h, z2.s, #8                          \n"
      "st1h     {z3.h}, p0, [%[dst_ptr]]                \n"
      "incb     %[dst_ptr]                              \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p0.h, wzr, %w[width]                    \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "umullb   z4.s, z2.h, z0.h                        \n"
      "umullt   z2.s, z2.h, z0.h                        \n"
      "umlalb   z4.s, z3.h, z1.h                        \n"
      "umlalt   z2.s, z3.h, z1.h                        \n"
      "rshrnb   z3.h, z4.s, #8                          \n"
      "rshrnt   z3.h, z2.s, #8                          \n"
      "st1h     {z3.h}, p0, [%[dst_ptr]]                \n"

      "99:                                              \n"
      : [src_ptr] "+r"(src_ptr),         // %[src_ptr]
        [src_ptr1] "+r"(src_ptr1),       // %[src_ptr1]
        [dst_ptr] "+r"(dst_ptr),         // %[dst_ptr]
        [width] "+r"(width),             // %[width]
        [vl] "=&r"(vl)                   // %[vl]
      : [y0_fraction] "r"(y0_fraction),  // %[y0_fraction]
        [y1_fraction] "r"(y1_fraction)   // %[y1_fraction]
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "p0");
}

__arm_locally_streaming static void HalfRow_16To8_SME(uint8_t* dst_ptr,
                                                      const uint16_t* src_ptr,
                                                      ptrdiff_t src_stride,
                                                      int scale,
                                                      int width) {
  const uint16_t* src_ptr1 = src_ptr + src_stride;

  // 15 - clz(scale), + 8 to shift result into the high half of the lane to
  // saturate, then we can just use UZP2 to narrow rather than a pair of
  // saturating narrow instructions.
  int shift = 23 - __builtin_clz((int32_t)scale);

  int vl;
  asm volatile(
      "cnth     %x[vl]                                  \n"
      "dup      z31.h, %w[shift]                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p0.h                                    \n"
      "1:                                               \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "incb     %[src_ptr]                              \n"
      "incb     %[src_ptr1]                             \n"
      "urhadd   z2.h, p0/m, z2.h, z3.h                  \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "uqshl    z2.h, p0/m, z2.h, z31.h                 \n"
      "shrnb    z2.b, z2.h, #8                          \n"
      "st1b     {z2.h}, p0, [%[dst_ptr]]                \n"
      "inch     %[dst_ptr]                              \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p0.h, wzr, %w[width]                    \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "urhadd   z2.h, p0/m, z2.h, z3.h                  \n"
      "uqshl    z2.h, p0/m, z2.h, z31.h                 \n"
      "shrnb    z2.b, z2.h, #8                          \n"
      "st1b     {z2.h}, p0, [%[dst_ptr]]                \n"

      "99:                                              \n"
      : [src_ptr] "+r"(src_ptr),    // %[src_ptr]
        [src_ptr1] "+r"(src_ptr1),  // %[src_ptr1]
        [dst_ptr] "+r"(dst_ptr),    // %[dst_ptr]
        [width] "+r"(width),        // %[width]
        [vl] "=&r"(vl)              // %[vl]
      : [shift] "r"(shift)          // %[shift]
      : "cc", "memory", "z0", "z1", "z2", "z3", "z31", "p0");
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits
// 16384 = 10 bits
// 4096 = 12 bits
// 256 = 16 bits
// TODO(fbarchard): change scale to bits
__arm_locally_streaming void InterpolateRow_16To8_SME(uint8_t* dst_ptr,
                                                      const uint16_t* src_ptr,
                                                      ptrdiff_t src_stride,
                                                      int scale,
                                                      int width,
                                                      int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint16_t* src_ptr1 = src_ptr + src_stride;

  // y0_fraction == 0 is never called here.
  if (y0_fraction == 128) {
    HalfRow_16To8_SME(dst_ptr, src_ptr, src_stride, scale, width);
    return;
  }
  if (y0_fraction == 256) {
    Convert16To8Row_SME(src_ptr, dst_ptr, scale, width);
    return;
  }

  // 15 - clz(scale), + 8 to shift result into the high half of the lane to
  // saturate, then we can just use UZP2 to narrow rather than a pair of
  // saturating narrow instructions.
  int shift = 23 - __builtin_clz((int32_t)scale);

  int vl;
  asm volatile(
      "cnth     %x[vl]                                  \n"
      "dup      z31.h, %w[shift]                        \n"
      "dup      z0.h, %w[y0_fraction]                   \n"
      "dup      z1.h, %w[y1_fraction]                   \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p0.h                                    \n"
      "1:                                               \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "incb     %[src_ptr]                              \n"
      "incb     %[src_ptr1]                             \n"
      "umullb   z4.s, z2.h, z0.h                        \n"
      "umullt   z2.s, z2.h, z0.h                        \n"
      "subs     %w[width], %w[width], %w[vl]            \n"
      "umlalb   z4.s, z3.h, z1.h                        \n"
      "umlalt   z2.s, z3.h, z1.h                        \n"
      "rshrnb   z3.h, z4.s, #8                          \n"
      "rshrnt   z3.h, z2.s, #8                          \n"
      "uqshl    z3.h, p0/m, z3.h, z31.h                 \n"
      "shrnb    z3.b, z3.h, #8                          \n"
      "st1b     {z3.h}, p0, [%[dst_ptr]]                \n"
      "inch     %[dst_ptr]                              \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p0.h, wzr, %w[width]                    \n"
      "ld1h     {z2.h}, p0/z, [%[src_ptr]]              \n"
      "ld1h     {z3.h}, p0/z, [%[src_ptr1]]             \n"
      "umullb   z4.s, z2.h, z0.h                        \n"
      "umullt   z2.s, z2.h, z0.h                        \n"
      "umlalb   z4.s, z3.h, z1.h                        \n"
      "umlalt   z2.s, z3.h, z1.h                        \n"
      "rshrnb   z3.h, z4.s, #8                          \n"
      "rshrnt   z3.h, z2.s, #8                          \n"
      "uqshl    z3.h, p0/m, z3.h, z31.h                 \n"
      "shrnb    z3.b, z3.h, #8                          \n"
      "st1b     {z3.h}, p0, [%[dst_ptr]]                \n"

      "99:                                              \n"
      : [src_ptr] "+r"(src_ptr),         // %[src_ptr]
        [src_ptr1] "+r"(src_ptr1),       // %[src_ptr1]
        [dst_ptr] "+r"(dst_ptr),         // %[dst_ptr]
        [width] "+r"(width),             // %[width]
        [vl] "=&r"(vl)                   // %[vl]
      : [y0_fraction] "r"(y0_fraction),  // %[y0_fraction]
        [y1_fraction] "r"(y1_fraction),  // %[y1_fraction]
        [shift] "r"(shift)               // %[shift]
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "z31", "p0");
}

__arm_locally_streaming void Convert8To8Row_SME(const uint8_t* src_y,
                                                uint8_t* dst_y,
                                                int scale,
                                                int bias,
                                                int width) {
  Convert8To8Row_SVE_SC(src_y, dst_y, scale, bias, width);
}

#define CONVERT8TO16_SVE                                 \
  "ld1b        {z0.h}, p0/z, [%[src]]                \n" \
  "ld1b        {z1.h}, p1/z, [%[src], #1, mul vl]    \n" \
  "incb        %[src]                                \n" \
  "subs        %w[width], %w[width], %w[vl], lsl #1  \n" \
  "trn1        z0.b, z0.b, z0.b                      \n" \
  "trn1        z1.b, z1.b, z1.b                      \n" \
  "lsr         z0.h, p0/m, z0.h, z2.h                \n" \
  "lsr         z1.h, p1/m, z1.h, z2.h                \n" \
  "prfm        pldl1keep, [%[src], 448]              \n" \
  "st1h        {z0.h}, p0, [%[dst]]                  \n" \
  "st1h        {z1.h}, p1, [%[dst], #1, mul vl]      \n" \
  "incb        %[dst], all, mul #2                   \n"

__arm_locally_streaming void Convert8To16Row_SME(const uint8_t* src_y,
                                                 uint16_t* dst_y,
                                                 int scale,
                                                 int width) {
  // (src * 0x0101 * scale) >> 16.
  // Since scale is a power of two, compute the shift to use to avoid needing
  // to widen to int32.
  int shift = __builtin_clz(scale) - 15;

  uint64_t vl;
  asm volatile(
      "dup         z2.h, %w[shift]                      \n"
      "cnth        %[vl]                                \n"
      "subs        %w[width], %w[width], %w[vl], lsl #1 \n"
      "b.lt        2f                                   \n"

      // Run bulk of computation with all-true predicates to avoid predicate
      // generation overhead.
      "ptrue       p0.h                                 \n"
      "ptrue       p1.h                                 \n"
      "1:                                               \n"  //
      CONVERT8TO16_SVE
      "b.ge        1b                                   \n"

      "2:                                               \n"
      "adds        %w[width], %w[width], %w[vl], lsl #1 \n"
      "b.eq        99f                                  \n"

      // Calculate predicates for the final iteration to deal with the tail.
      "whilelt     p0.h, wzr, %w[width]                 \n"
      "whilelt     p1.h, %w[vl], %w[width]              \n"  //
      CONVERT8TO16_SVE

      "99:                                              \n"
      : [src] "+r"(src_y),    // %[src]
        [dst] "+r"(dst_y),    // %[dst]
        [width] "+r"(width),  // %[width]
        [vl] "=&r"(vl)        // %[vl]
      : [shift] "r"(shift)    // %[shift]
      : "cc", "memory", "z0", "z1", "z2", "p0", "p1");
}

__arm_locally_streaming void ARGBToUVRow_SME(const uint8_t* src_argb,
                                             int src_stride_argb,
                                             uint8_t* dst_u,
                                             uint8_t* dst_v,
                                             int width) {
  ARGBToUVMatrixRow_SVE_SC(src_argb, src_stride_argb, dst_u, dst_v, width,
                           kARGBToUVCoefficients);
}

__arm_locally_streaming void ARGBToUVJRow_SME(const uint8_t* src_argb,
                                              int src_stride_argb,
                                              uint8_t* dst_u,
                                              uint8_t* dst_v,
                                              int width) {
  ARGBToUVMatrixRow_SVE_SC(src_argb, src_stride_argb, dst_u, dst_v, width,
                           kARGBToUVJCoefficients);
}

__arm_locally_streaming void ABGRToUVJRow_SME(const uint8_t* src_abgr,
                                              int src_stride_abgr,
                                              uint8_t* dst_uj,
                                              uint8_t* dst_vj,
                                              int width) {
  ARGBToUVMatrixRow_SVE_SC(src_abgr, src_stride_abgr, dst_uj, dst_vj, width,
                           kABGRToUVJCoefficients);
}

__arm_locally_streaming void BGRAToUVRow_SME(const uint8_t* src_bgra,
                                             int src_stride_bgra,
                                             uint8_t* dst_u,
                                             uint8_t* dst_v,
                                             int width) {
  ARGBToUVMatrixRow_SVE_SC(src_bgra, src_stride_bgra, dst_u, dst_v, width,
                           kBGRAToUVCoefficients);
}

__arm_locally_streaming void ABGRToUVRow_SME(const uint8_t* src_abgr,
                                             int src_stride_abgr,
                                             uint8_t* dst_u,
                                             uint8_t* dst_v,
                                             int width) {
  ARGBToUVMatrixRow_SVE_SC(src_abgr, src_stride_abgr, dst_u, dst_v, width,
                           kABGRToUVCoefficients);
}

__arm_locally_streaming void RGBAToUVRow_SME(const uint8_t* src_rgba,
                                             int src_stride_rgba,
                                             uint8_t* dst_u,
                                             uint8_t* dst_v,
                                             int width) {
  ARGBToUVMatrixRow_SVE_SC(src_rgba, src_stride_rgba, dst_u, dst_v, width,
                           kRGBAToUVCoefficients);
}

#endif  // !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) &&
        // defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
