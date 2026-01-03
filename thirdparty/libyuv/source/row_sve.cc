/*
 *  Copyright 2024 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row_sve.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if !defined(LIBYUV_DISABLE_SVE) && defined(__aarch64__)

#define RGBTOARGB8_SVE_2X                                 \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h,  A: z19.b */ \
  "uqshrnb     z16.b, z16.h, #6     \n" /* B0 */          \
  "uqshrnb     z17.b, z17.h, #6     \n" /* G0 */          \
  "uqshrnb     z18.b, z18.h, #6     \n" /* R0 */          \
  "uqshrnt     z16.b, z20.h, #6     \n" /* B1 */          \
  "uqshrnt     z17.b, z21.h, #6     \n" /* G1 */          \
  "uqshrnt     z18.b, z22.h, #6     \n" /* R1 */

#define RGBTOARGB8_SVE_TOP_2X                        \
  /* Inputs: B: z16.h,  G: z17.h,  R: z18.h */       \
  "uqshl     z16.h, p0/m, z16.h, #2     \n" /* B0 */ \
  "uqshl     z17.h, p0/m, z17.h, #2     \n" /* G0 */ \
  "uqshl     z18.h, p0/m, z18.h, #2     \n" /* R0 */ \
  "uqshl     z20.h, p0/m, z20.h, #2     \n" /* B1 */ \
  "uqshl     z21.h, p0/m, z21.h, #2     \n" /* G1 */ \
  "uqshl     z22.h, p0/m, z22.h, #2     \n" /* R1 */

void I444ToARGBRow_SVE2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  uint64_t vl;
  asm volatile(
      "cnth     %[vl]                                   \n"
      "ptrue    p0.b                                    \n"  //
      YUVTORGB_SVE_SETUP
      "dup      z19.b, #255                             \n"  // Alpha
      "subs     %w[width], %w[width], %w[vl]            \n"
      "b.lt     2f                                      \n"

      // Run bulk of computation with an all-true predicate to avoid predicate
      // generation overhead.
      "ptrue    p1.h                                    \n"
      "1:                                               \n"  //
      READYUV444_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "subs     %w[width], %w[width], %w[vl]            \n"
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"
      "add      %[dst_argb], %[dst_argb], %[vl], lsl #2 \n"
      "b.ge     1b                                      \n"

      "2:                                               \n"
      "adds     %w[width], %w[width], %w[vl]            \n"
      "b.eq     99f                                     \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt  p1.h, wzr, %w[width]                    \n"  //
      READYUV444_SVE I4XXTORGB_SVE RGBTOARGB8_SVE
      "st2h     {z16.h, z17.h}, p1, [%[dst_argb]]       \n"

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

void I444ToRGB24Row_SVE2(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  I444ToRGB24Row_SVE_SC(src_y, src_u, src_v, dst_rgb24, yuvconstants, width);
}

void I400ToARGBRow_SVE2(const uint8_t* src_y,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I400ToARGBRow_SVE_SC(src_y, dst_argb, yuvconstants, width);
}

void I422ToARGBRow_SVE2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I422ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

void I422ToRGB24Row_SVE2(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_argb,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  I422ToRGB24Row_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

void I422ToRGB565Row_SVE2(const uint8_t* src_y,
                          const uint8_t* src_u,
                          const uint8_t* src_v,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  I422ToRGB565Row_SVE_SC(src_y, src_u, src_v, dst_rgb565, yuvconstants, width);
}

void I422ToARGB1555Row_SVE2(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb1555,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  I422ToARGB1555Row_SVE_SC(src_y, src_u, src_v, dst_argb1555, yuvconstants,
                           width);
}

void I422ToARGB4444Row_SVE2(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb4444,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  I422ToARGB4444Row_SVE_SC(src_y, src_u, src_v, dst_argb4444, yuvconstants,
                           width);
}

void I422ToRGBARow_SVE2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I422ToRGBARow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

void I422ToAR30Row_SVE2(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I422ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

void I422AlphaToARGBRow_SVE2(const uint8_t* src_y,
                             const uint8_t* src_u,
                             const uint8_t* src_v,
                             const uint8_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  I422AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

void I444AlphaToARGBRow_SVE2(const uint8_t* src_y,
                             const uint8_t* src_u,
                             const uint8_t* src_v,
                             const uint8_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  I444AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

void NV12ToARGBRow_SVE2(const uint8_t* src_y,
                        const uint8_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  NV12ToARGBRow_SVE_SC(src_y, src_uv, dst_argb, yuvconstants, width);
}

void NV21ToARGBRow_SVE2(const uint8_t* src_y,
                        const uint8_t* src_vu,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  NV21ToARGBRow_SVE_SC(src_y, src_vu, dst_argb, yuvconstants, width);
}

void NV12ToRGB24Row_SVE2(const uint8_t* src_y,
                         const uint8_t* src_uv,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  NV12ToRGB24Row_SVE_SC(src_y, src_uv, dst_rgb24, yuvconstants, width);
}

void NV21ToRGB24Row_SVE2(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  NV21ToRGB24Row_SVE_SC(src_y, src_vu, dst_rgb24, yuvconstants, width);
}

void ARGBToUVRow_SVE2(const uint8_t* src_argb,
                      int src_stride_argb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  ARGBToUVMatrixRow_SVE_SC(src_argb, src_stride_argb, dst_u, dst_v, width,
                           kARGBToUVCoefficients);
}

void ARGBToUVJRow_SVE2(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  ARGBToUVMatrixRow_SVE_SC(src_argb, src_stride_argb, dst_u, dst_v, width,
                           kARGBToUVJCoefficients);
}

void ABGRToUVJRow_SVE2(const uint8_t* src_abgr,
                       int src_stride_abgr,
                       uint8_t* dst_uj,
                       uint8_t* dst_vj,
                       int width) {
  ARGBToUVMatrixRow_SVE_SC(src_abgr, src_stride_abgr, dst_uj, dst_vj, width,
                           kABGRToUVJCoefficients);
}

void BGRAToUVRow_SVE2(const uint8_t* src_bgra,
                      int src_stride_bgra,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  ARGBToUVMatrixRow_SVE_SC(src_bgra, src_stride_bgra, dst_u, dst_v, width,
                           kBGRAToUVCoefficients);
}

void ABGRToUVRow_SVE2(const uint8_t* src_abgr,
                      int src_stride_abgr,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  ARGBToUVMatrixRow_SVE_SC(src_abgr, src_stride_abgr, dst_u, dst_v, width,
                           kABGRToUVCoefficients);
}

void RGBAToUVRow_SVE2(const uint8_t* src_rgba,
                      int src_stride_rgba,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  ARGBToUVMatrixRow_SVE_SC(src_rgba, src_stride_rgba, dst_u, dst_v, width,
                           kRGBAToUVCoefficients);
}

#define ARGBTORGB565_SVE                    \
  /* Inputs:                                \
   * z0: rrrrrxxxbbbbbxxx                   \
   * z1: xxxxxxxxggggggxx                   \
   * z3: 0000000000000011 (3, 0, 3, 0, ...) \
   * z4: 0000011111100000                   \
   */                                       \
  "lsr     z0.b, p0/m, z0.b, z3.b       \n" \
  "lsl     z1.h, z1.h, #3               \n" \
  "bsl     z1.d, z1.d, z0.d, z4.d       \n"

void ARGBToRGB565Row_SVE2(const uint8_t* src_argb,
                          uint8_t* dst_rgb,
                          int width) {
  unsigned bsl_mask = 0x7e0;
  uint64_t vl;
  width *= 2;
  asm volatile(
      "mov     z3.h, #3                     \n"
      "dup     z4.h, %w[bsl_mask]           \n"

      "cntb    %[vl]                        \n"
      "subs    %w[width], %w[width], %w[vl] \n"
      "b.lt    2f                           \n"

      "ptrue   p0.b                         \n"
      "1:                                   \n"
      "ld2b    {z0.b, z1.b}, p0/z, [%[src]] \n"  // BR, GA
      "incb    %[src], all, mul #2          \n"
      "subs    %w[width], %w[width], %w[vl] \n"  //
      ARGBTORGB565_SVE
      "st1b    {z1.b}, p0, [%[dst]]         \n"
      "incb    %[dst]                       \n"
      "b.ge    1b                           \n"

      "2:                                   \n"
      "adds    %w[width], %w[width], %w[vl] \n"
      "b.eq    99f                          \n"

      "whilelt p0.b, wzr, %w[width]         \n"
      "ld2b    {z0.b, z1.b}, p0/z, [%[src]] \n"  // BR, GA
      ARGBTORGB565_SVE
      "st1b    {z1.b}, p0, [%[dst]]         \n"

      "99:                                  \n"
      : [src] "+r"(src_argb),     // %[src]
        [dst] "+r"(dst_rgb),      // %[dst]
        [width] "+r"(width),      // %[width]
        [vl] "=&r"(vl)            // %[vl]
      : [bsl_mask] "r"(bsl_mask)  // %[bsl_mask]
      : "cc", "memory", "z0", "z1", "z3", "z4", "p0");
}

void ARGBToRGB565DitherRow_SVE2(const uint8_t* src_argb,
                                uint8_t* dst_rgb,
                                uint32_t dither4,
                                int width) {
  unsigned bsl_mask = 0x7e0;
  uint64_t vl;
  width *= 2;
  asm volatile(
      "mov     z3.h, #3                     \n"
      "dup     z4.h, %w[bsl_mask]           \n"
      "dup     z2.s, %w[dither4]            \n"
      "zip1    z2.b, z2.b, z2.b             \n"

      "cntb    %[vl]                        \n"
      "subs    %w[width], %w[width], %w[vl] \n"
      "b.lt    2f                           \n"

      "ptrue   p0.b                         \n"
      "1:                                   \n"
      "ld2b    {z0.b, z1.b}, p0/z, [%[src]] \n"  // BR, GA
      "incb    %[src], all, mul #2          \n"
      "uqadd   z0.b, z0.b, z2.b             \n"
      "uqadd   z1.b, z1.b, z2.b             \n"
      "subs    %w[width], %w[width], %w[vl] \n"  //
      ARGBTORGB565_SVE
      "st1b    {z1.b}, p0, [%[dst]]         \n"
      "incb    %[dst]                       \n"
      "b.ge    1b                           \n"

      "2:                                   \n"
      "adds    %w[width], %w[width], %w[vl] \n"
      "b.eq    99f                          \n"

      "whilelt p0.b, wzr, %w[width]         \n"
      "ld2b    {z0.b, z1.b}, p0/z, [%[src]] \n"  // BR, GA
      "uqadd   z0.b, z0.b, z2.b             \n"
      "uqadd   z1.b, z1.b, z2.b             \n"  //
      ARGBTORGB565_SVE
      "st1b    {z1.b}, p0, [%[dst]]         \n"

      "99:                                  \n"
      : [src] "+r"(src_argb),      // %[src]
        [dst] "+r"(dst_rgb),       // %[dst]
        [width] "+r"(width),       // %[width]
        [vl] "=&r"(vl)             // %[vl]
      : [bsl_mask] "r"(bsl_mask),  // %[bsl_mask]
        [dither4] "r"(dither4)     // %[dither4]
      : "cc", "memory", "z0", "z1", "z3", "z4", "p0");
}

#define ARGB1555TOARGB                                        \
  /* Input: z1/z3.h = arrrrrgggggbbbbb */                     \
  "lsl     z0.h, z1.h, #3          \n" /* rrrgggggbbbbb000 */ \
  "lsl     z2.h, z3.h, #3          \n" /* rrrgggggbbbbb000 */ \
  "asr     z1.h, z1.h, #7          \n" /* aaaaaaaarrrrrggg */ \
  "asr     z3.h, z3.h, #7          \n" /* aaaaaaaarrrrrggg */ \
  "lsl     z0.b, p0/m, z0.b, z4.b  \n" /* ggggg000bbbbb000 */ \
  "lsl     z2.b, p0/m, z2.b, z4.b  \n" /* ggggg000bbbbb000 */ \
  "sri     z1.b, z1.b, #5          \n" /* aaaaaaaarrrrrrrr */ \
  "sri     z3.b, z3.b, #5          \n" /* aaaaaaaarrrrrrrr */ \
  "sri     z0.b, z0.b, #5          \n" /* ggggggggbbbbbbbb */ \
  "sri     z2.b, z2.b, #5          \n" /* ggggggggbbbbbbbb */

void ARGB1555ToARGBRow_SVE2(const uint8_t* src_argb1555,
                            uint8_t* dst_argb,
                            int width) {
  uint64_t vl;
  asm volatile(
      "mov     z4.h, #0x0300                           \n"
      "ptrue   p0.b                                    \n"

      "cnth    %x[vl]                                  \n"
      "subs    %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.lt    2f                                      \n"

      "1:                                              \n"
      "ld1h    {z1.h}, p0/z, [%[src]]                  \n"
      "ld1h    {z3.h}, p0/z, [%[src], #1, mul vl]      \n"
      "incb    %[src], all, mul #2                     \n"  //
      ARGB1555TOARGB
      "subs    %w[width], %w[width], %w[vl], lsl #1    \n"
      "st2h    {z0.h, z1.h}, p0, [%[dst]]              \n"
      "st2h    {z2.h, z3.h}, p0, [%[dst], #2, mul vl]  \n"
      "incb    %[dst], all, mul #4                     \n"
      "b.ge    1b                                      \n"

      "2:                                              \n"
      "adds    %w[width], %w[width], %w[vl], lsl #1    \n"
      "b.eq    99f                                     \n"

      "whilelt p1.h, wzr, %w[width]                    \n"
      "whilelt p2.h, %w[vl], %w[width]                 \n"
      "ld1h    {z1.h}, p1/z, [%[src]]                  \n"
      "ld1h    {z3.h}, p2/z, [%[src], #1, mul vl]      \n"  //
      ARGB1555TOARGB
      "st2h    {z0.h, z1.h}, p1, [%[dst]]              \n"
      "st2h    {z2.h, z3.h}, p2, [%[dst], #2, mul vl]  \n"

      "99:                                             \n"
      : [src] "+r"(src_argb1555),  // %[src]
        [dst] "+r"(dst_argb),      // %[dst]
        [width] "+r"(width),       // %[width]
        [vl] "=&r"(vl)             // %[vl]
      :
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "p0", "p1", "p2");
}

// clang-format off
#define AYUVTOUV_SVE(zU0, zV0, zU1, zV1)                   /* e.g. */          \
  "ld2h     {z0.h, z1.h}, p0/z, [%[src0]]              \n" /* VUVU.. YAYA.. */ \
  "ld2h     {z1.h, z2.h}, p1/z, [%[src0], #2, mul vl]  \n" /* VUVU.. YAYA.. */ \
  "ld2h     {z2.h, z3.h}, p0/z, [%[src1]]              \n" /* VUVU.. YAYA.. */ \
  "ld2h     {z3.h, z4.h}, p1/z, [%[src1], #2, mul vl]  \n" /* VUVU.. YAYA.. */ \
  "incb     %[src0], all, mul #4                       \n"                     \
  "incb     %[src1], all, mul #4                       \n"                     \
  "uaddlb   z4.h, z0.b, z2.b                           \n" /* V */             \
  "uaddlt   z5.h, z0.b, z2.b                           \n" /* U */             \
  "uaddlb   z6.h, z1.b, z3.b                           \n" /* V */             \
  "uaddlt   z7.h, z1.b, z3.b                           \n" /* U */             \
  "addp   " #zU0 ".h, p0/m, " #zU0 ".h, " #zV0 ".h     \n" /* UV */            \
  "addp   " #zU1 ".h, p1/m, " #zU1 ".h, " #zV1 ".h     \n" /* UV */            \
  "subs     %w[width], %w[width], %w[vl]               \n"                     \
  "urshr  " #zU0 ".h, p0/m, " #zU0 ".h, #2             \n" /* U0V0 */          \
  "urshr  " #zU1 ".h, p1/m, " #zU1 ".h, #2             \n" /* U0V0 */          \
  "st1b     {" #zU0 ".h}, p0, [%[dst]]                 \n"                     \
  "st1b     {" #zU1 ".h}, p1, [%[dst], #1, mul vl]     \n"                     \
  "incb     %[dst]                                     \n"
// clang-format on

// Filter 2 rows of AYUV UV's (444) into UV (420).
// AYUV is VUYA in memory.  UV for NV12 is UV order in memory.
void AYUVToUVRow_SVE2(const uint8_t* src_ayuv,
                      int src_stride_ayuv,
                      uint8_t* dst_uv,
                      int width) {
  // Output a row of UV values, filtering 2x2 rows of AYUV.
  const uint8_t* src_ayuv1 = src_ayuv + src_stride_ayuv;
  int vl;
  asm volatile (
      "cntb    %x[vl]                            \n"
      "subs    %w[width], %w[width], %w[vl]      \n"
      "b.lt    2f                                \n"

      "ptrue   p0.h                              \n"
      "ptrue   p1.h                              \n"
      "1:                                        \n"
      AYUVTOUV_SVE(z5, z4, z7, z6)
      "b.ge    1b                                \n"

      "2:                                        \n"
      "adds    %w[width], %w[width], %w[vl]      \n"
      "b.eq    99f                               \n"

      "cnth    %x[vl]                            \n"
      "whilelt p0.h, wzr, %w[width]              \n" // first row
      "whilelt p1.h, %w[vl], %w[width]           \n" // second row
      AYUVTOUV_SVE(z5, z4, z7, z6)

      "99:                                       \n"
      : [src0]"+r"(src_ayuv),   // %[src0]
        [src1]"+r"(src_ayuv1),  // %[src1]
        [dst]"+r"(dst_uv),      // %[dst]
        [width]"+r"(width),     // %[width]
        [vl]"=&r"(vl)           // %[vl]
      :
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0",
        "p1");
}

// Filter 2 rows of AYUV UV's (444) into VU (420).
void AYUVToVURow_SVE2(const uint8_t* src_ayuv,
                      int src_stride_ayuv,
                      uint8_t* dst_vu,
                      int width) {
  // Output a row of VU values, filtering 2x2 rows of AYUV.
  const uint8_t* src_ayuv1 = src_ayuv + src_stride_ayuv;
  int vl;
  asm volatile (
      "cntb    %x[vl]                            \n"
      "cmp     %w[width], %w[vl]                 \n"
      "subs    %w[width], %w[width], %w[vl]      \n"
      "b.lt    2f                                \n"

      "ptrue   p0.h                              \n"
      "ptrue   p1.h                              \n"
      "1:                                        \n"
      AYUVTOUV_SVE(z4, z5, z6, z7)
      "b.ge    1b                                \n"

      "2:                                        \n"
      "adds    %w[width], %w[width], %w[vl]      \n"
      "b.eq    99f                               \n"

      "cnth    %x[vl]                            \n"
      "whilelt p0.h, wzr, %w[width]              \n" // first row
      "whilelt p1.h, %w[vl], %w[width]           \n" // second row
      AYUVTOUV_SVE(z4, z5, z6, z7)

      "99:                                       \n"
      : [src0]"+r"(src_ayuv),   // %[src0]
        [src1]"+r"(src_ayuv1),  // %[src1]
        [dst]"+r"(dst_vu),      // %[dst]
        [width]"+r"(width),     // %[width]
        [vl]"=&r"(vl)           // %[vl]
      :
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0",
        "p1");
}

void YUY2ToARGBRow_SVE2(const uint8_t* src_yuy2,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  YUY2ToARGBRow_SVE_SC(src_yuy2, dst_argb, yuvconstants, width);
}

void UYVYToARGBRow_SVE2(const uint8_t* src_uyvy,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  UYVYToARGBRow_SVE_SC(src_uyvy, dst_argb, yuvconstants, width);
}

static inline void RAWToWXYZRow_SVE2(const uint8_t* src_raw,
                                     uint8_t* dst_wxyz,
                                     int width,
                                     uint32_t idx_start,
                                     uint32_t idx_step,
                                     uint32_t alpha) {
  uint32_t vl;
  asm("cntw %x0" : "=r"(vl));
  uint32_t vl_mul3 = vl * 3;
  uint32_t rem_mul3;
  asm volatile(
      "index   z31.s, %w[idx_start], %w[idx_step]        \n"
      "dup     z30.s, %w[alpha]                          \n"
      "subs    %w[width], %w[width], %w[vl], lsl #1      \n"
      "b.lt    2f                                        \n"

      // Run bulk of computation with the same predicates to avoid predicate
      // generation overhead. We set up p1 to only load 3/4 of a vector.
      "ptrue   p0.s                                      \n"
      "whilelt p1.b, wzr, %w[vl_mul3]                    \n"
      "1:                                                \n"
      "ld1b    {z0.b}, p1/z, [%[src]]                    \n"
      "add     %[src], %[src], %x[vl_mul3]               \n"
      "ld1b    {z1.b}, p1/z, [%[src]]                    \n"
      "add     %[src], %[src], %x[vl_mul3]               \n"
      "tbl     z0.b, {z0.b}, z31.b                       \n"
      "tbl     z1.b, {z1.b}, z31.b                       \n"
      "subs    %w[width], %w[width], %w[vl], lsl #1      \n"
      "orr     z0.d, z0.d, z30.d                         \n"
      "orr     z1.d, z1.d, z30.d                         \n"
      "st1w    {z0.s}, p0, [%[dst]]                      \n"
      "st1w    {z1.s}, p0, [%[dst], #1, mul vl]          \n"
      "incb    %[dst], all, mul #2                       \n"
      "b.ge    1b                                        \n"

      "2:                                                \n"
      "adds     %w[width], %w[width], %w[vl], lsl #1     \n"
      "b.eq     99f                                      \n"

      // Calculate a pair of predicates for the final iteration to deal with
      // the tail.
      "3:                                                \n"
      "add     %w[rem_mul3], %w[width], %w[width], lsl #1 \n"
      "whilelt p0.s, wzr, %w[width]                      \n"
      "whilelt p1.b, wzr, %w[rem_mul3]                    \n"
      "ld1b    {z0.b}, p1/z, [%[src]]                    \n"
      "add     %[src], %[src], %x[vl_mul3]               \n"
      "tbl     z0.b, {z0.b}, z31.b                       \n"
      "subs    %w[width], %w[width], %w[vl]              \n"
      "orr     z0.d, z0.d, z30.d                         \n"
      "st1w    {z0.s}, p0, [%[dst]]                      \n"
      "incb    %[dst]                                    \n"
      "b.gt    3b                                        \n"

      "99:                                               \n"
      : [src] "+r"(src_raw),         // %[src]
        [dst] "+r"(dst_wxyz),        // %[dst]
        [width] "+r"(width),         // %[width]
        [vl_mul3] "+r"(vl_mul3),     // %[vl_mul3]
        [rem_mul3] "=&r"(rem_mul3)   // %[rem_mul3]
      : [idx_start] "r"(idx_start),  // %[idx_start]
        [idx_step] "r"(idx_step),    // %[idx_step]
        [alpha] "r"(alpha),          // %[alpha]
        [vl] "r"(vl)                 // %[vl]
      : "cc", "memory", "z0", "z1", "z30", "z31", "p0", "p1");
}

void RAWToARGBRow_SVE2(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  RAWToWXYZRow_SVE2(src_raw, dst_argb, width, 0xff000102U, 0x00030303U,
                    0xff000000U);
}

void RAWToRGBARow_SVE2(const uint8_t* src_raw, uint8_t* dst_rgba, int width) {
  RAWToWXYZRow_SVE2(src_raw, dst_rgba, width, 0x000102ffU, 0x03030300U,
                    0x000000ffU);
}

void RGB24ToARGBRow_SVE2(const uint8_t* src_rgb24,
                         uint8_t* dst_argb,
                         int width) {
  RAWToWXYZRow_SVE2(src_rgb24, dst_argb, width, 0xff020100U, 0x00030303U,
                    0xff000000U);
}

static const uint8_t kRAWToRGB24Indices[] = {
    2,   1,   0,   5,   4,   3,   8,   7,   6,   11,  10,  9,   14,  13,  12,
    17,  16,  15,  20,  19,  18,  23,  22,  21,  26,  25,  24,  29,  28,  27,
    32,  31,  30,  35,  34,  33,  38,  37,  36,  41,  40,  39,  44,  43,  42,
    47,  46,  45,  50,  49,  48,  53,  52,  51,  56,  55,  54,  59,  58,  57,
    62,  61,  60,  65,  64,  63,  68,  67,  66,  71,  70,  69,  74,  73,  72,
    77,  76,  75,  80,  79,  78,  83,  82,  81,  86,  85,  84,  89,  88,  87,
    92,  91,  90,  95,  94,  93,  98,  97,  96,  101, 100, 99,  104, 103, 102,
    107, 106, 105, 110, 109, 108, 113, 112, 111, 116, 115, 114, 119, 118, 117,
    122, 121, 120, 125, 124, 123, 128, 127, 126, 131, 130, 129, 134, 133, 132,
    137, 136, 135, 140, 139, 138, 143, 142, 141, 146, 145, 144, 149, 148, 147,
    152, 151, 150, 155, 154, 153, 158, 157, 156, 161, 160, 159, 164, 163, 162,
    167, 166, 165, 170, 169, 168, 173, 172, 171, 176, 175, 174, 179, 178, 177,
    182, 181, 180, 185, 184, 183, 188, 187, 186, 191, 190, 189, 194, 193, 192,
    197, 196, 195, 200, 199, 198, 203, 202, 201, 206, 205, 204, 209, 208, 207,
    212, 211, 210, 215, 214, 213, 218, 217, 216, 221, 220, 219, 224, 223, 222,
    227, 226, 225, 230, 229, 228, 233, 232, 231, 236, 235, 234, 239, 238, 237,
    242, 241, 240, 245, 244, 243, 248, 247, 246, 251, 250, 249, 254, 253, 252};

void RAWToRGB24Row_SVE2(const uint8_t* src_raw, uint8_t* dst_rgb24, int width) {
  // width is in elements, convert to bytes.
  width *= 3;
  // we use the mul3 predicate pattern throughout to use the largest multiple
  // of three number of lanes, for instance with a vector length of 16 bytes
  // only the first 15 bytes will be used for load/store instructions.
  uint32_t vl;
  asm volatile(
      "cntb    %x[vl], mul3                              \n"
      "ptrue   p0.b, mul3                                \n"
      "ld1b    {z31.b}, p0/z, [%[kIndices]]              \n"
      "subs    %w[width], %w[width], %w[vl]              \n"
      "b.lt    2f                                        \n"

      // Run bulk of computation with the same predicate to avoid predicate
      // generation overhead.
      "1:                                                \n"
      "ld1b    {z0.b}, p0/z, [%[src]]                    \n"
      "add     %[src], %[src], %x[vl]                    \n"
      "tbl     z0.b, {z0.b}, z31.b                       \n"
      "subs    %w[width], %w[width], %w[vl]              \n"
      "st1b    {z0.b}, p0, [%[dst]]                      \n"
      "add     %[dst], %[dst], %x[vl]                    \n"
      "b.ge    1b                                        \n"

      "2:                                                \n"
      "adds    %w[width], %w[width], %w[vl]              \n"
      "b.eq    99f                                       \n"

      // Calculate a predicate for the final iteration to deal with the tail.
      "whilelt p0.b, wzr, %w[width]                      \n"
      "ld1b    {z0.b}, p0/z, [%[src]]                    \n"
      "tbl     z0.b, {z0.b}, z31.b                       \n"
      "st1b    {z0.b}, p0, [%[dst]]                      \n"

      "99:                                               \n"
      : [src] "+r"(src_raw),                // %[src]
        [dst] "+r"(dst_rgb24),              // %[dst]
        [width] "+r"(width),                // %[width]
        [vl] "=&r"(vl)                      // %[vl]
      : [kIndices] "r"(kRAWToRGB24Indices)  // %[kIndices]
      : "cc", "memory", "z0", "z31", "p0");
}

static inline void ARGBToXYZRow_SVE2(const uint8_t* src_argb,
                                     uint8_t* dst_xyz,
                                     int width,
                                     const uint8_t* indices) {
  uint32_t vl;
  asm("cntw %x0" : "=r"(vl));
  uint32_t vl_mul3 = vl * 3;
  uint32_t rem_mul3;
  asm volatile(
      "whilelt p1.b, wzr, %w[vl_mul3]                     \n"
      "ld1b    {z31.b}, p1/z, [%[indices]]                \n"
      "subs    %w[width], %w[width], %w[vl], lsl #1       \n"
      "b.lt    2f                                         \n"

      // Run bulk of computation with the same predicates to avoid predicate
      // generation overhead. We set up p1 to only store 3/4 of a vector.
      "ptrue   p0.s                                       \n"
      "1:                                                 \n"
      "ld1w    {z0.s}, p0/z, [%[src]]                     \n"
      "ld1w    {z1.s}, p0/z, [%[src], #1, mul vl]         \n"
      "incb    %[src], all, mul #2                        \n"
      "tbl     z0.b, {z0.b}, z31.b                        \n"
      "tbl     z1.b, {z1.b}, z31.b                        \n"
      "subs    %w[width], %w[width], %w[vl], lsl #1       \n"
      "st1b    {z0.b}, p1, [%[dst]]                       \n"
      "add     %[dst], %[dst], %x[vl_mul3]                \n"
      "st1b    {z1.b}, p1, [%[dst]]                       \n"
      "add     %[dst], %[dst], %x[vl_mul3]                \n"
      "b.ge    1b                                         \n"

      "2:                                                 \n"
      "adds    %w[width], %w[width], %w[vl], lsl #1       \n"
      "b.eq    99f                                        \n"

      // Calculate predicates for the final iteration to deal with the tail.
      "add     %w[rem_mul3], %w[width], %w[width], lsl #1 \n"
      "whilelt p0.s, wzr, %w[width]                       \n"
      "whilelt p1.b, wzr, %w[rem_mul3]                    \n"
      "whilelt p2.s, %w[vl], %w[width]                    \n"
      "whilelt p3.b, %w[vl_mul3], %w[rem_mul3]            \n"
      "ld1w    {z0.s}, p0/z, [%[src]]                     \n"
      "ld1w    {z1.s}, p2/z, [%[src], #1, mul vl]         \n"
      "tbl     z0.b, {z0.b}, z31.b                        \n"
      "tbl     z1.b, {z1.b}, z31.b                        \n"
      "st1b    {z0.b}, p1, [%[dst]]                       \n"
      "add     %[dst], %[dst], %x[vl_mul3]                \n"
      "st1b    {z1.b}, p3, [%[dst]]                       \n"

      "99:                                                \n"
      : [src] "+r"(src_argb),       // %[src]
        [dst] "+r"(dst_xyz),        // %[dst]
        [width] "+r"(width),        // %[width]
        [rem_mul3] "=&r"(rem_mul3)  // %[rem_mul3]
      : [indices] "r"(indices),     // %[indices]
        [vl_mul3] "r"(vl_mul3),     // %[vl_mul3]
        [vl] "r"(vl)                // %[vl]
      : "cc", "memory", "z0", "z1", "z31", "p0", "p1", "p2", "p3");
}

static const uint8_t kARGBToRGB24RowIndices[] = {
    0,   1,   2,   4,   5,   6,   8,   9,   10,  12,  13,  14,  16,  17,  18,
    20,  21,  22,  24,  25,  26,  28,  29,  30,  32,  33,  34,  36,  37,  38,
    40,  41,  42,  44,  45,  46,  48,  49,  50,  52,  53,  54,  56,  57,  58,
    60,  61,  62,  64,  65,  66,  68,  69,  70,  72,  73,  74,  76,  77,  78,
    80,  81,  82,  84,  85,  86,  88,  89,  90,  92,  93,  94,  96,  97,  98,
    100, 101, 102, 104, 105, 106, 108, 109, 110, 112, 113, 114, 116, 117, 118,
    120, 121, 122, 124, 125, 126, 128, 129, 130, 132, 133, 134, 136, 137, 138,
    140, 141, 142, 144, 145, 146, 148, 149, 150, 152, 153, 154, 156, 157, 158,
    160, 161, 162, 164, 165, 166, 168, 169, 170, 172, 173, 174, 176, 177, 178,
    180, 181, 182, 184, 185, 186, 188, 189, 190, 192, 193, 194, 196, 197, 198,
    200, 201, 202, 204, 205, 206, 208, 209, 210, 212, 213, 214, 216, 217, 218,
    220, 221, 222, 224, 225, 226, 228, 229, 230, 232, 233, 234, 236, 237, 238,
    240, 241, 242, 244, 245, 246, 248, 249, 250, 252, 253, 254,
};

static const uint8_t kARGBToRAWRowIndices[] = {
    2,   1,   0,   6,   5,   4,   10,  9,   8,   14,  13,  12,  18,  17,  16,
    22,  21,  20,  26,  25,  24,  30,  29,  28,  34,  33,  32,  38,  37,  36,
    42,  41,  40,  46,  45,  44,  50,  49,  48,  54,  53,  52,  58,  57,  56,
    62,  61,  60,  66,  65,  64,  70,  69,  68,  74,  73,  72,  78,  77,  76,
    82,  81,  80,  86,  85,  84,  90,  89,  88,  94,  93,  92,  98,  97,  96,
    102, 101, 100, 106, 105, 104, 110, 109, 108, 114, 113, 112, 118, 117, 116,
    122, 121, 120, 126, 125, 124, 130, 129, 128, 134, 133, 132, 138, 137, 136,
    142, 141, 140, 146, 145, 144, 150, 149, 148, 154, 153, 152, 158, 157, 156,
    162, 161, 160, 166, 165, 164, 170, 169, 168, 174, 173, 172, 178, 177, 176,
    182, 181, 180, 186, 185, 184, 190, 189, 188, 194, 193, 192, 198, 197, 196,
    202, 201, 200, 206, 205, 204, 210, 209, 208, 214, 213, 212, 218, 217, 216,
    222, 221, 220, 226, 225, 224, 230, 229, 228, 234, 233, 232, 238, 237, 236,
    242, 241, 240, 246, 245, 244, 250, 249, 248, 254, 253, 252,
};

void ARGBToRGB24Row_SVE2(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  ARGBToXYZRow_SVE2(src_argb, dst_rgb, width, kARGBToRGB24RowIndices);
}

void ARGBToRAWRow_SVE2(const uint8_t* src_argb, uint8_t* dst_rgb, int width) {
  ARGBToXYZRow_SVE2(src_argb, dst_rgb, width, kARGBToRAWRowIndices);
}

void DivideRow_16_SVE2(const uint16_t* src_y,
                       uint16_t* dst_y,
                       int scale,
                       int width) {
  uint64_t vl;
  asm volatile(
      "cnth   %x[vl]                                     \n"
      "dup    z0.h, %w[scale]                            \n"
      "subs   %w[width], %w[width], %w[vl], lsl #1       \n"
      "b.le    2f                                        \n"

      // Run bulk of computation with the same predicates to avoid predicate
      // generation overhead.
      "ptrue   p0.h                                      \n"
      "1:                                                \n"
      "ld1h   {z1.h}, p0/z, [%[src]]                     \n"
      "ld1h   {z2.h}, p0/z, [%[src], #1, mul vl]         \n"
      "incb   %[src], all, mul #2                        \n"
      "umulh  z1.h, z1.h, z0.h                           \n"
      "umulh  z2.h, z2.h, z0.h                           \n"
      "subs   %w[width], %w[width], %w[vl], lsl #1       \n"
      "st1h   {z1.h}, p0, [%[dst]]                       \n"
      "st1h   {z2.h}, p0, [%[dst], #1, mul vl]           \n"
      "incb   %[dst], all, mul #2                        \n"
      "b.gt    1b                                        \n"

      "2:                                                \n"
      "adds     %w[width], %w[width], %w[vl], lsl #1     \n"
      "b.eq     99f                                      \n"

      // Calculate a pair of predicates for the final iteration to deal with
      // the tail.
      "whilelt p0.h, wzr, %w[width]                      \n"
      "whilelt p1.h, %w[vl], %w[width]                   \n"
      "ld1h   {z1.h}, p0/z, [%[src]]                     \n"
      "ld1h   {z2.h}, p1/z, [%[src], #1, mul vl]         \n"
      "umulh  z1.h, z1.h, z0.h                           \n"
      "umulh  z2.h, z2.h, z0.h                           \n"
      "st1h   {z1.h}, p0, [%[dst]]                       \n"
      "st1h   {z2.h}, p1, [%[dst], #1, mul vl]           \n"

      "99:                                               \n"
      : [src] "+r"(src_y),    // %[src]
        [dst] "+r"(dst_y),    // %[dst]
        [width] "+r"(width),  // %[width]
        [vl] "=&r"(vl)        // %[vl]
      : [scale] "r"(scale)    // %[scale]
      : "cc", "memory", "z0", "z1", "z2", "p0", "p1");
}

#define HALFFLOAT_SVE                                    \
  "scvtf       z0.s, p0/m, z0.s                      \n" \
  "scvtf       z1.s, p0/m, z1.s                      \n" \
  "scvtf       z2.s, p0/m, z2.s                      \n" \
  "scvtf       z3.s, p0/m, z3.s                      \n" \
  "fmul        z0.s, z0.s, z4.s                      \n" \
  "fmul        z1.s, z1.s, z4.s                      \n" \
  "fmul        z2.s, z2.s, z4.s                      \n" \
  "fmul        z3.s, z3.s, z4.s                      \n" \
  "uqshrnb     z0.h, z0.s, #13                       \n" \
  "uqshrnb     z1.h, z1.s, #13                       \n" \
  "uqshrnb     z2.h, z2.s, #13                       \n" \
  "uqshrnb     z3.h, z3.s, #13                       \n"

void HalfFloatRow_SVE2(const uint16_t* src,
                       uint16_t* dst,
                       float scale,
                       int width) {
  uint64_t vl;
  asm("cntw %x0" : "=r"(vl));
  asm volatile(
      "mov         z4.s, %s[scale]                       \n"
      "subs        %w[width], %w[width], %w[vl], lsl #2  \n"
      "b.lt        2f                                    \n"

      // Run bulk of computation with all-true predicates to avoid predicate
      // generation overhead.
      "ptrue       p0.s                                  \n"
      "1:                                                \n"
      "ld1h        {z0.s}, p0/z, [%[src]]                \n"
      "ld1h        {z1.s}, p0/z, [%[src], #1, mul vl]    \n"
      "ld1h        {z2.s}, p0/z, [%[src], #2, mul vl]    \n"
      "ld1h        {z3.s}, p0/z, [%[src], #3, mul vl]    \n"
      "incb        %[src], all, mul #2                   \n"  //
      HALFFLOAT_SVE
      "subs        %w[width], %w[width], %w[vl], lsl #2  \n"
      "st1h        {z0.s}, p0, [%[dst]]                  \n"
      "st1h        {z1.s}, p0, [%[dst], #1, mul vl]      \n"
      "st1h        {z2.s}, p0, [%[dst], #2, mul vl]      \n"
      "st1h        {z3.s}, p0, [%[dst], #3, mul vl]      \n"
      "incb        %[dst], all, mul #2                   \n"
      "b.ge        1b                                    \n"

      "2:                                                \n"
      "adds     %w[width], %w[width], %w[vl], lsl #2     \n"
      "b.eq     99f                                      \n"

      // Calculate predicates for the final iteration to deal with the tail.
      "whilelt     p0.s, wzr, %w[width]                  \n"
      "whilelt     p1.s, %w[vl], %w[width]               \n"
      "whilelt     p2.s, %w[vl2], %w[width]              \n"
      "whilelt     p3.s, %w[vl3], %w[width]              \n"
      "ld1h        {z0.s}, p0/z, [%[src]]                \n"
      "ld1h        {z1.s}, p1/z, [%[src], #1, mul vl]    \n"
      "ld1h        {z2.s}, p2/z, [%[src], #2, mul vl]    \n"
      "ld1h        {z3.s}, p3/z, [%[src], #3, mul vl]    \n"  //
      HALFFLOAT_SVE
      "st1h        {z0.s}, p0, [%[dst]]                  \n"
      "st1h        {z1.s}, p1, [%[dst], #1, mul vl]      \n"
      "st1h        {z2.s}, p2, [%[dst], #2, mul vl]      \n"
      "st1h        {z3.s}, p3, [%[dst], #3, mul vl]      \n"

      "99:                                               \n"
      : [src] "+r"(src),                        // %[src]
        [dst] "+r"(dst),                        // %[dst]
        [width] "+r"(width)                     // %[width]
      : [vl] "r"(vl),                           // %[vl]
        [vl2] "r"(vl * 2),                      // %[vl2]
        [vl3] "r"(vl * 3),                      // %[vl3]
        [scale] "w"(scale * 1.9259299444e-34f)  // %[scale]
      : "cc", "memory", "z0", "z1", "z2", "z3", "z4", "p0", "p1", "p2", "p3");
}

void HalfFloat1Row_SVE2(const uint16_t* src,
                        uint16_t* dst,
                        float scale,
                        int width) {
  uint64_t vl;
  asm volatile(
      "cnth        %x[vl]                                \n"
      "subs        %w[width], %w[width], %w[vl], lsl #1  \n"
      "b.lt        2f                                    \n"

      // Run bulk of computation with all-true predicates to avoid predicate
      // generation overhead.
      "ptrue       p0.h                                  \n"
      "1:                                                \n"
      "ld1h        {z0.h}, p0/z, [%[src]]                \n"
      "ld1h        {z1.h}, p0/z, [%[src], #1, mul vl]    \n"
      "incb        %[src], all, mul #2                   \n"
      "ucvtf       z0.h, p0/m, z0.h                      \n"
      "ucvtf       z1.h, p0/m, z1.h                      \n"
      "subs        %w[width], %w[width], %w[vl], lsl #1  \n"
      "st1h        {z0.h}, p0, [%[dst]]                  \n"
      "st1h        {z1.h}, p0, [%[dst], #1, mul vl]      \n"
      "incb        %[dst], all, mul #2                   \n"
      "b.ge        1b                                    \n"

      "2:                                                \n"
      "adds     %w[width], %w[width], %w[vl], lsl #1     \n"
      "b.eq     99f                                      \n"

      // Calculate predicates for the final iteration to deal with the tail.
      "whilelt     p0.h, wzr, %w[width]                  \n"
      "whilelt     p1.h, %w[vl], %w[width]               \n"
      "ld1h        {z0.h}, p0/z, [%[src]]                \n"
      "ld1h        {z1.h}, p1/z, [%[src], #1, mul vl]    \n"
      "ucvtf       z0.h, p0/m, z0.h                      \n"
      "ucvtf       z1.h, p0/m, z1.h                      \n"
      "st1h        {z0.h}, p0, [%[dst]]                  \n"
      "st1h        {z1.h}, p1, [%[dst], #1, mul vl]      \n"

      "99:                                               \n"
      : [src] "+r"(src),      // %[src]
        [dst] "+r"(dst),      // %[dst]
        [width] "+r"(width),  // %[width]
        [vl] "=&r"(vl)        // %[vl]
      :
      : "cc", "memory", "z0", "z1", "p0", "p1");
}

void I210ToARGBRow_SVE2(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I210ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

void I210AlphaToARGBRow_SVE2(const uint16_t* src_y,
                             const uint16_t* src_u,
                             const uint16_t* src_v,
                             const uint16_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  I210AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

void I210ToAR30Row_SVE2(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I210ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
}

void P210ToARGBRow_SVE2(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  P210ToARGBRow_SVE_SC(src_y, src_uv, dst_argb, yuvconstants, width);
}

void P210ToAR30Row_SVE2(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  P210ToAR30Row_SVE_SC(src_y, src_uv, dst_ar30, yuvconstants, width);
}

void I410ToARGBRow_SVE2(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I410ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

void I410AlphaToARGBRow_SVE2(const uint16_t* src_y,
                             const uint16_t* src_u,
                             const uint16_t* src_v,
                             const uint16_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  I410AlphaToARGBRow_SVE_SC(src_y, src_u, src_v, src_a, dst_argb, yuvconstants,
                            width);
}

void I410ToAR30Row_SVE2(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I410ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
}

void P410ToARGBRow_SVE2(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  P410ToARGBRow_SVE_SC(src_y, src_uv, dst_argb, yuvconstants, width);
}

void P410ToAR30Row_SVE2(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  P410ToAR30Row_SVE_SC(src_y, src_uv, dst_ar30, yuvconstants, width);
}

void I212ToAR30Row_SVE2(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I212ToAR30Row_SVE_SC(src_y, src_u, src_v, dst_ar30, yuvconstants, width);
}

void I212ToARGBRow_SVE2(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  I212ToARGBRow_SVE_SC(src_y, src_u, src_v, dst_argb, yuvconstants, width);
}

void Convert8To8Row_SVE2(const uint8_t* src_y,
                         uint8_t* dst_y,
                         int scale,
                         int bias,
                         int width) {
  Convert8To8Row_SVE_SC(src_y, dst_y, scale, bias, width);
}

#endif  // !defined(LIBYUV_DISABLE_SVE) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
