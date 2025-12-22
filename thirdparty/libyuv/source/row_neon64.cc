/*
 *  Copyright 2014 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Enable LIBYUV_USE_ST2, LIBYUV_USE_ST3, LIBYUV_USE_ST4 for CPUs that prefer
// STn over ZIP1+ST1
// Exynos M1, M2, M3 are slow with ST2, ST3 and ST4 instructions.

// This module is for GCC Neon armv8 64 bit.
#if !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

// v0.8h: Y
// v1.16b: 8U, 8V

// Read 8 Y, 4 U and 4 V from 422
#define READYUV422                               \
  "ldr        d0, [%[src_y]], #8             \n" \
  "ldr        s1, [%[src_u]], #4             \n" \
  "ldr        s2, [%[src_v]], #4             \n" \
  "zip1       v0.16b, v0.16b, v0.16b         \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "zip1       v1.8b, v1.8b, v1.8b            \n" \
  "zip1       v2.8b, v2.8b, v2.8b            \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n"

// Read 8 Y, 4 U and 4 V from 210
#define READYUV210                               \
  "ldr        q2, [%[src_y]], #16            \n" \
  "ldr        d1, [%[src_u]], #8             \n" \
  "ldr        d3, [%[src_v]], #8             \n" \
  "shl        v0.8h, v2.8h, #6               \n" \
  "usra       v0.8h, v2.8h, #4               \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "zip1       v2.8h, v3.8h, v3.8h            \n" \
  "zip1       v3.8h, v1.8h, v1.8h            \n" \
  "uqshrn     v1.8b, v3.8h, #2               \n" \
  "uqshrn2    v1.16b, v2.8h, #2              \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n"

// Read 8 Y, 4 U and 4 V interleaved from 210
#define READYUVP210                              \
  "ldr        q0, [%[src_y]], #16            \n" \
  "ldr        q1, [%[src_uv]], #16           \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "tbl        v1.16b, {v1.16b}, v2.16b       \n"

// Read 8 Y, 4 U and 4 V from 212
#define READYUV212                               \
  "ldr        q2, [%[src_y]], #16            \n" \
  "ldr        d1, [%[src_u]], #8             \n" \
  "ldr        d3, [%[src_v]], #8             \n" \
  "shl        v0.8h, v2.8h, #4               \n" \
  "usra       v0.8h, v2.8h, #8               \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "zip1       v2.8h, v3.8h, v3.8h            \n" \
  "zip1       v3.8h, v1.8h, v1.8h            \n" \
  "uqshrn     v1.8b, v3.8h, #4               \n" \
  "uqshrn2    v1.16b, v2.8h, #4              \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n"

// Read 8 Y, 8 U and 8 V from 410
#define READYUV410                               \
  "ldr        q1, [%[src_y]], #16            \n" \
  "ldr        q2, [%[src_u]], #16            \n" \
  "ldr        q3, [%[src_v]], #16            \n" \
  "shl        v0.8h, v1.8h, #6               \n" \
  "usra       v0.8h, v1.8h, #4               \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "uqshrn     v1.8b, v2.8h, #2               \n" \
  "uqshrn2    v1.16b, v3.8h, #2              \n" \
  "prfm       pldl1keep, [%[src_u], 128]     \n" \
  "prfm       pldl1keep, [%[src_v], 128]     \n"

// Read 8 Y, 8 U and 8 V interleaved from 410
#define READYUVP410                                \
  "ldr        q0, [%[src_y]], #16              \n" \
  "ldp        q4, q5, [%[src_uv]], #32         \n" \
  "prfm       pldl1keep, [%[src_y], 448]       \n" \
  "tbl        v1.16b, {v4.16b, v5.16b}, v2.16b \n"

// Read 8 Y, 8 U and 8 V from 444
#define READYUV444                               \
  "ldr        d0, [%[src_y]], #8             \n" \
  "ldr        d1, [%[src_u]], #8             \n" \
  "ldr        d2, [%[src_v]], #8             \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "prfm       pldl1keep, [%[src_u], 448]     \n" \
  "zip1       v0.16b, v0.16b, v0.16b         \n" \
  "prfm       pldl1keep, [%[src_v], 448]     \n"

// Read 8 Y
#define READYUV400                               \
  "ldr        d0, [%[src_y]], #8             \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "zip1       v0.16b, v0.16b, v0.16b         \n"

static const uvec8 kNV12Table = {0, 0, 2, 2, 4, 4, 6, 6,
                                 1, 1, 3, 3, 5, 5, 7, 7};
static const uvec8 kNV12InterleavedTable = {0, 0, 4, 4, 8,  8,  12, 12,
                                            2, 2, 6, 6, 10, 10, 14, 14};
static const uvec8 kNV21Table = {1, 1, 3, 3, 5, 5, 7, 7,
                                 0, 0, 2, 2, 4, 4, 6, 6};
static const uvec8 kNV21InterleavedTable = {1, 1, 5, 5, 9,  9,  13, 13,
                                            3, 3, 7, 7, 11, 11, 15, 15};

// Read 8 Y and 4 UV from NV12 or NV21
#define READNV12                                 \
  "ldr        d0, [%[src_y]], #8             \n" \
  "ldr        d1, [%[src_uv]], #8            \n" \
  "zip1       v0.16b, v0.16b, v0.16b         \n" \
  "prfm       pldl1keep, [%[src_y], 448]     \n" \
  "tbl        v1.16b, {v1.16b}, v2.16b       \n" \
  "prfm       pldl1keep, [%[src_uv], 448]    \n"

// Read 8 YUY2
#define READYUY2                                 \
  "ld1        {v3.16b}, [%[src_yuy2]], #16   \n" \
  "trn1       v0.16b, v3.16b, v3.16b         \n" \
  "prfm       pldl1keep, [%[src_yuy2], 448]  \n" \
  "tbl        v1.16b, {v3.16b}, v2.16b       \n"

// Read 8 UYVY
#define READUYVY                                 \
  "ld1        {v3.16b}, [%[src_uyvy]], #16   \n" \
  "trn2       v0.16b, v3.16b, v3.16b         \n" \
  "prfm       pldl1keep, [%[src_uyvy], 448]  \n" \
  "tbl        v1.16b, {v3.16b}, v2.16b       \n"

// UB VR UG VG
// YG BB BG BR
#define YUVTORGB_SETUP                                                \
  "ld4r       {v28.16b, v29.16b, v30.16b, v31.16b}, [%[kUVCoeff]] \n" \
  "ld4r       {v24.8h, v25.8h, v26.8h, v27.8h}, [%[kRGBCoeffBias]] \n"

// v16.8h: B
// v17.8h: G
// v18.8h: R

// Convert from YUV (NV12 or NV21) to 2.14 fixed point RGB.
// Similar to I4XXTORGB but U/V components are in the low/high halves of v1.
#define NVTORGB                                           \
  "umull2     v3.4s, v0.8h, v24.8h           \n"          \
  "umull      v6.8h, v1.8b, v30.8b           \n"          \
  "umull      v0.4s, v0.4h, v24.4h           \n"          \
  "umlal2     v6.8h, v1.16b, v31.16b         \n" /* DG */ \
  "uzp2       v0.8h, v0.8h, v3.8h            \n" /* Y */  \
  "umull      v4.8h, v1.8b, v28.8b           \n" /* DB */ \
  "umull2     v5.8h, v1.16b, v29.16b         \n" /* DR */ \
  "add        v17.8h, v0.8h, v26.8h          \n" /* G */  \
  "add        v16.8h, v0.8h, v4.8h           \n" /* B */  \
  "add        v18.8h, v0.8h, v5.8h           \n" /* R */  \
  "uqsub      v17.8h, v17.8h, v6.8h          \n" /* G */  \
  "uqsub      v16.8h, v16.8h, v25.8h         \n" /* B */  \
  "uqsub      v18.8h, v18.8h, v27.8h         \n" /* R */

// Convert from YUV (I444 or I420) to 2.14 fixed point RGB.
// Similar to NVTORGB but U/V components are in v1/v2.
#define I4XXTORGB                                         \
  "umull2     v3.4s, v0.8h, v24.8h           \n"          \
  "umull      v6.8h, v1.8b, v30.8b           \n"          \
  "umull      v0.4s, v0.4h, v24.4h           \n"          \
  "umlal      v6.8h, v2.8b, v31.8b           \n" /* DG */ \
  "uzp2       v0.8h, v0.8h, v3.8h            \n" /* Y */  \
  "umull      v4.8h, v1.8b, v28.8b           \n" /* DB */ \
  "umull      v5.8h, v2.8b, v29.8b           \n" /* DR */ \
  "add        v17.8h, v0.8h, v26.8h          \n" /* G */  \
  "add        v16.8h, v0.8h, v4.8h           \n" /* B */  \
  "add        v18.8h, v0.8h, v5.8h           \n" /* R */  \
  "uqsub      v17.8h, v17.8h, v6.8h          \n" /* G */  \
  "uqsub      v16.8h, v16.8h, v25.8h         \n" /* B */  \
  "uqsub      v18.8h, v18.8h, v27.8h         \n" /* R */

// Convert from YUV I400 to 2.14 fixed point RGB
#define I400TORGB                                        \
  "umull2     v3.4s, v0.8h, v24.8h           \n"         \
  "umull      v0.4s, v0.4h, v24.4h           \n"         \
  "uzp2       v0.8h, v0.8h, v3.8h            \n" /* Y */ \
  "add        v17.8h, v0.8h, v26.8h          \n" /* G */ \
  "add        v16.8h, v0.8h, v4.8h           \n" /* B */ \
  "add        v18.8h, v0.8h, v5.8h           \n" /* R */ \
  "uqsub      v17.8h, v17.8h, v6.8h          \n" /* G */ \
  "uqsub      v16.8h, v16.8h, v25.8h         \n" /* B */ \
  "uqsub      v18.8h, v18.8h, v27.8h         \n" /* R */

// Convert from 2.14 fixed point RGB To 8 bit RGB
#define RGBTORGB8                                \
  "uqshrn     v17.8b, v17.8h, #6             \n" \
  "uqshrn     v16.8b, v16.8h, #6             \n" \
  "uqshrn     v18.8b, v18.8h, #6             \n"

// Convert from 2.14 fixed point RGB to 8 bit RGB, placing the results in the
// top half of each lane.
#define RGBTORGB8_TOP                            \
  "uqshl      v17.8h, v17.8h, #2             \n" \
  "uqshl      v16.8h, v16.8h, #2             \n" \
  "uqshl      v18.8h, v18.8h, #2             \n"

// Store 2.14 fixed point RGB as AR30 elements
#define STOREAR30                                                         \
  /* Inputs:                                                              \
   *   v16.8h: xxbbbbbbbbbbxxxx                                           \
   *   v17.8h: xxggggggggggxxxx                                           \
   *   v18.8h: xxrrrrrrrrrrxxxx                                           \
   *   v22.8h: 0011111111110000 (umin limit)                              \
   *   v23.8h: 1100000000000000 (alpha)                                   \
   */                                                                     \
  "uqshl    v0.8h, v16.8h, #2                  \n" /* bbbbbbbbbbxxxxxx */ \
  "uqshl    v1.8h, v17.8h, #2                  \n" /* ggggggggggxxxxxx */ \
  "umin     v6.8h, v18.8h, v22.8h              \n" /* 00rrrrrrrrrrxxxx */ \
  "shl      v4.8h, v1.8h, #4                   \n" /* ggggggxxxxxx0000 */ \
  "orr      v5.16b, v6.16b, v23.16b            \n" /* 11rrrrrrrrrrxxxx */ \
  "sri      v4.8h, v0.8h, #6                   \n" /* ggggggbbbbbbbbbb */ \
  "sri      v5.8h, v1.8h, #12                  \n" /* 11rrrrrrrrrrgggg */ \
  "st2      {v4.8h, v5.8h}, [%[dst_ar30]], #32 \n"

#define YUVTORGB_REGS                                                         \
  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v24", \
      "v25", "v26", "v27", "v28", "v29", "v30", "v31"

void I444ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n" /* A */
      "1:          \n"                               //
      READYUV444
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I444ToRGB24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"  //
      READYUV444
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB RGBTORGB8
      "st3         {v16.8b,v17.8b,v18.8b}, [%[dst_rgb24]], #24 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgb24] "+r"(dst_rgb24),                       // %[dst_rgb24]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

void I210ToAR30Row_NEON(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  uint16_t limit = 0x3ff0;
  uint16_t alpha = 0xc000;
  asm volatile(YUVTORGB_SETUP
      "dup         v22.8h, %w[limit]             \n"
      "dup         v23.8h, %w[alpha]             \n"
      "1:          \n"  //
               READYUV210
      "subs        %w[width], %w[width], #8      \n" NVTORGB STOREAR30
      "b.gt        1b                            \n"
               : [src_y] "+r"(src_y),             // %[src_y]
                 [src_u] "+r"(src_u),             // %[src_u]
                 [src_v] "+r"(src_v),             // %[src_v]
                 [dst_ar30] "+r"(dst_ar30),       // %[dst_ar30]
                 [width] "+r"(width)              // %[width]
               : [kUVCoeff] "r"(uv_coeff),        // %[kUVCoeff]
                 [kRGBCoeffBias] "r"(rgb_coeff),  // %[kRGBCoeffBias]
                 [limit] "r"(limit),              // %[limit]
                 [alpha] "r"(alpha)               // %[alpha]
               : "cc", "memory", YUVTORGB_REGS, "v22", "v23");
}

void I410ToAR30Row_NEON(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  uint16_t limit = 0x3ff0;
  uint16_t alpha = 0xc000;
  asm volatile(YUVTORGB_SETUP
      "dup         v22.8h, %w[limit]             \n"
      "dup         v23.8h, %w[alpha]             \n"
      "1:          \n"  //
               READYUV410
      "subs        %w[width], %w[width], #8      \n" NVTORGB STOREAR30
      "b.gt        1b                            \n"
               : [src_y] "+r"(src_y),             // %[src_y]
                 [src_u] "+r"(src_u),             // %[src_u]
                 [src_v] "+r"(src_v),             // %[src_v]
                 [dst_ar30] "+r"(dst_ar30),       // %[dst_ar30]
                 [width] "+r"(width)              // %[width]
               : [kUVCoeff] "r"(uv_coeff),        // %[kUVCoeff]
                 [kRGBCoeffBias] "r"(rgb_coeff),  // %[kRGBCoeffBias]
                 [limit] "r"(limit),              // %[limit]
                 [alpha] "r"(alpha)               // %[alpha]
               : "cc", "memory", YUVTORGB_REGS, "v22", "v23");
}

void I212ToAR30Row_NEON(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  const uint16_t limit = 0x3ff0;
  asm volatile(YUVTORGB_SETUP
      "dup         v22.8h, %w[limit]             \n"
      "movi        v23.8h, #0xc0, lsl #8         \n"  // A
      "1:          \n"                                //
               READYUV212
      "subs        %w[width], %w[width], #8      \n" NVTORGB STOREAR30
      "b.gt        1b                            \n"
               : [src_y] "+r"(src_y),             // %[src_y]
                 [src_u] "+r"(src_u),             // %[src_u]
                 [src_v] "+r"(src_v),             // %[src_v]
                 [dst_ar30] "+r"(dst_ar30),       // %[dst_ar30]
                 [width] "+r"(width)              // %[width]
               : [kUVCoeff] "r"(uv_coeff),        // %[kUVCoeff]
                 [kRGBCoeffBias] "r"(rgb_coeff),  // %[kRGBCoeffBias]
                 [limit] "r"(limit)               // %[limit]
               : "cc", "memory", YUVTORGB_REGS, "v22", "v23");
}

void I210ToARGBRow_NEON(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "1:          \n"  //
      READYUV210
      "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I410ToARGBRow_NEON(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "1:          \n"  //
      READYUV410
      "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I212ToARGBRow_NEON(const uint16_t* src_y,
                        const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "1:          \n"  //
      READYUV212
      "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),            // %[src_y]
        [src_u] "+r"(src_u),            // %[src_u]
        [src_v] "+r"(src_v),            // %[src_v]
        [dst_argb] "+r"(dst_argb),      // %[dst_argb]
        [width] "+r"(width)             // %[width]
      : [kUVCoeff] "r"(uv_coeff),       // %[kUVCoeff]
        [kRGBCoeffBias] "r"(rgb_coeff)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I422ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n" /* A */
      "1:          \n"                               //
      READYUV422
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

uint8_t kP210LoadShuffleIndices[] = {1, 1, 5, 5, 9,  9,  13, 13,
                                     3, 3, 7, 7, 11, 11, 15, 15};

void P210ToARGBRow_NEON(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "ldr         q2, [%[kIndices]]             \n"
      "1:          \n"  //
      READYUVP210
      "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b, v17.8b, v18.8b, v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                     // %[src_y]
        [src_uv] "+r"(src_uv),                   // %[src_uv]
        [dst_argb] "+r"(dst_argb),               // %[dst_argb]
        [width] "+r"(width)                      // %[width]
      : [kUVCoeff] "r"(uv_coeff),                // %[kUVCoeff]
        [kRGBCoeffBias] "r"(rgb_coeff),          // %[kRGBCoeffBias]
        [kIndices] "r"(kP210LoadShuffleIndices)  // %[kIndices]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

uint8_t kP410LoadShuffleIndices[] = {1, 5, 9,  13, 17, 21, 25, 29,
                                     3, 7, 11, 15, 19, 23, 27, 31};

void P410ToARGBRow_NEON(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "ldr         q2, [%[kIndices]]             \n"
      "1:          \n"  //
      READYUVP410
      "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b, v17.8b, v18.8b, v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                     // %[src_y]
        [src_uv] "+r"(src_uv),                   // %[src_uv]
        [dst_argb] "+r"(dst_argb),               // %[dst_argb]
        [width] "+r"(width)                      // %[width]
      : [kUVCoeff] "r"(uv_coeff),                // %[kUVCoeff]
        [kRGBCoeffBias] "r"(rgb_coeff),          // %[kRGBCoeffBias]
        [kIndices] "r"(kP410LoadShuffleIndices)  // %[kIndices]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void P210ToAR30Row_NEON(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  const uint16_t limit = 0x3ff0;
  asm volatile(YUVTORGB_SETUP
      "dup         v22.8h, %w[limit]             \n"
      "movi        v23.8h, #0xc0, lsl #8         \n"  // A
      "ldr         q2, [%[kIndices]]             \n"
      "1:          \n"  //
               READYUVP210
      "subs        %w[width], %w[width], #8      \n" NVTORGB STOREAR30
      "b.gt        1b                            \n"
               : [src_y] "+r"(src_y),                     // %[src_y]
                 [src_uv] "+r"(src_uv),                   // %[src_uv]
                 [dst_ar30] "+r"(dst_ar30),               // %[dst_ar30]
                 [width] "+r"(width)                      // %[width]
               : [kUVCoeff] "r"(uv_coeff),                // %[kUVCoeff]
                 [kRGBCoeffBias] "r"(rgb_coeff),          // %[kRGBCoeffBias]
                 [limit] "r"(limit),                      // %[limit]
                 [kIndices] "r"(kP210LoadShuffleIndices)  // %[kIndices]
               : "cc", "memory", YUVTORGB_REGS, "v22", "v23");
}

void P410ToAR30Row_NEON(const uint16_t* src_y,
                        const uint16_t* src_uv,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  uint16_t limit = 0x3ff0;
  asm volatile(YUVTORGB_SETUP
      "dup         v22.8h, %w[limit]             \n"
      "movi        v23.8h, #0xc0, lsl #8         \n"  // A
      "ldr         q2, [%[kIndices]]             \n"
      "1:          \n"  //
               READYUVP410
      "subs        %w[width], %w[width], #8      \n" NVTORGB STOREAR30
      "b.gt        1b                            \n"
               : [src_y] "+r"(src_y),                     // %[src_y]
                 [src_uv] "+r"(src_uv),                   // %[src_uv]
                 [dst_ar30] "+r"(dst_ar30),               // %[dst_ar30]
                 [width] "+r"(width)                      // %[width]
               : [kUVCoeff] "r"(uv_coeff),                // %[kUVCoeff]
                 [kRGBCoeffBias] "r"(rgb_coeff),          // %[kRGBCoeffBias]
                 [limit] "r"(limit),                      // %[limit]
                 [kIndices] "r"(kP410LoadShuffleIndices)  // %[kIndices]
               : "cc", "memory", YUVTORGB_REGS, "v22", "v23");
}

void I422ToAR30Row_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_ar30,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  const uvec8* uv_coeff = &yuvconstants->kUVCoeff;
  const vec16* rgb_coeff = &yuvconstants->kRGBCoeffBias;
  const uint16_t limit = 0x3ff0;
  asm volatile(
      YUVTORGB_SETUP
      "dup         v22.8h, %w[limit]             \n"
      "movi        v23.8h, #0xc0, lsl #8         \n"  // A
      "1:          \n"                                //
      READYUV422
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB STOREAR30
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),             // %[src_y]
        [src_u] "+r"(src_u),             // %[src_u]
        [src_v] "+r"(src_v),             // %[src_v]
        [dst_ar30] "+r"(dst_ar30),       // %[dst_ar30]
        [width] "+r"(width)              // %[width]
      : [kUVCoeff] "r"(uv_coeff),        // %[kUVCoeff]
        [kRGBCoeffBias] "r"(rgb_coeff),  // %[kRGBCoeffBias]
        [limit] "r"(limit)               // %[limit]
      : "cc", "memory", YUVTORGB_REGS, "v22", "v23");
}

void I444AlphaToARGBRow_NEON(const uint8_t* src_y,
                             const uint8_t* src_u,
                             const uint8_t* src_v,
                             const uint8_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"
      "ld1         {v19.8b}, [%[src_a]], #8      \n" READYUV444
      "subs        %w[width], %w[width], #8      \n"
      "prfm        pldl1keep, [%[src_a], 448]    \n" I4XXTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I410AlphaToARGBRow_NEON(const uint16_t* src_y,
                             const uint16_t* src_u,
                             const uint16_t* src_v,
                             const uint16_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"
      "ld1         {v19.16b}, [%[src_a]], #16    \n" READYUV410
      "subs        %w[width], %w[width], #8      \n"
      "uqshrn      v19.8b, v19.8h, #2            \n" NVTORGB RGBTORGB8
      "st4         {v16.8b, v17.8b, v18.8b, v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I210AlphaToARGBRow_NEON(const uint16_t* src_y,
                             const uint16_t* src_u,
                             const uint16_t* src_v,
                             const uint16_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"
      "ld1         {v19.16b}, [%[src_a]], #16    \n" READYUV210
      "subs        %w[width], %w[width], #8      \n"
      "uqshrn      v19.8b, v19.8h, #2            \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I422AlphaToARGBRow_NEON(const uint8_t* src_y,
                             const uint8_t* src_u,
                             const uint8_t* src_v,
                             const uint8_t* src_a,
                             uint8_t* dst_argb,
                             const struct YuvConstants* yuvconstants,
                             int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"
      "ld1         {v19.8b}, [%[src_a]], #8      \n" READYUV422
      "subs        %w[width], %w[width], #8      \n"
      "prfm        pldl1keep, [%[src_a], 448]    \n" I4XXTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I422ToRGBARow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_rgba,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v15.8b, #255                  \n" /* A */
      "1:          \n"                               //
      READYUV422
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB RGBTORGB8
      "st4         {v15.8b,v16.8b,v17.8b,v18.8b}, [%[dst_rgba]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgba] "+r"(dst_rgba),                         // %[dst_rgba]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v15");
}

void I422ToRGB24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"  //
      READYUV422
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB RGBTORGB8
      "st3         {v16.8b,v17.8b,v18.8b}, [%[dst_rgb24]], #24 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgb24] "+r"(dst_rgb24),                       // %[dst_rgb24]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

#define ARGBTORGB565                                                \
  /* Inputs:                                                        \
   * v16: bbbbbxxx                                                  \
   * v17: ggggggxx                                                  \
   * v18: rrrrrxxx */                                               \
  "shll       v18.8h, v18.8b, #8     \n" /* rrrrrrxx00000000     */ \
  "shll       v17.8h, v17.8b, #8     \n" /* gggggxxx00000000     */ \
  "shll       v16.8h, v16.8b, #8     \n" /* bbbbbbxx00000000     */ \
  "sri        v18.8h, v17.8h, #5     \n" /* rrrrrgggggg00000     */ \
  "sri        v18.8h, v16.8h, #11    \n" /* rrrrrggggggbbbbb     */

#define ARGBTORGB565_FROM_TOP                                       \
  /* Inputs:                                                        \
   * v16: bbbbbxxxxxxxxxxx                                          \
   * v17: ggggggxxxxxxxxxx                                          \
   * v18: rrrrrxxxxxxxxxxx */                                       \
  "sri        v18.8h, v17.8h, #5     \n" /* rrrrrgggggg00000     */ \
  "sri        v18.8h, v16.8h, #11    \n" /* rrrrrggggggbbbbb     */

void I422ToRGB565Row_NEON(const uint8_t* src_y,
                          const uint8_t* src_u,
                          const uint8_t* src_v,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"  //
      READYUV422
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB RGBTORGB8_TOP
          ARGBTORGB565_FROM_TOP
      "st1         {v18.8h}, [%[dst_rgb565]], #16 \n"  // store 8 pixels RGB565.
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgb565] "+r"(dst_rgb565),                     // %[dst_rgb565]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

#define ARGBTOARGB1555                                                  \
  /* Inputs:                                                            \
   * v16: gggggxxxbbbbbxxx  v17: axxxxxxxrrrrrxxx  */                   \
  "shl        v1.8h, v16.8h, #8              \n" /* bbbbbxxx00000000 */ \
  "shl        v2.8h, v17.8h, #8              \n" /* rrrrrxxx00000000 */ \
  "sri        v17.8h, v2.8h, #1              \n" /* arrrrrxxxrrrrxxx */ \
  "sri        v17.8h, v16.8h, #6             \n" /* arrrrrgggggxxxbb */ \
  "sri        v17.8h, v1.8h, #11             \n" /* arrrrrgggggbbbbb */

#define ARGBTOARGB1555_FROM_TOP                                         \
  /* Inputs:                                                            \
   * v16: bbbbbxxxxxxxxxxx  v17: gggggxxxxxxxxxxx                       \
   * v18: rrrrrxxxxxxxxxxx  v19: axxxxxxxxxxxxxxx */                    \
  "sri        v19.8h,  v18.8h, #1            \n" /* arrrrrxxxxxxxxxx */ \
  "sri        v19.8h,  v17.8h, #6            \n" /* arrrrrgggggxxxxx */ \
  "sri        v19.8h,  v16.8h, #11           \n" /* arrrrrgggggbbbbb */

void I422ToARGB1555Row_NEON(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb1555,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8h, #0x80, lsl #8         \n"
      "1:          \n"                                           //
      READYUV422 "subs        %w[width], %w[width], #8      \n"  //
      I4XXTORGB RGBTORGB8_TOP ARGBTOARGB1555_FROM_TOP
      "st1         {v19.8h}, [%[dst_argb1555]], #16 \n"  // store 8 pixels
                                                         // RGB1555.
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb1555] "+r"(dst_argb1555),                 // %[dst_argb1555]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

#define ARGBTOARGB4444                                   \
  /* Input v16.8b<=B, v17.8b<=G, v18.8b<=R, v19.8b<=A */ \
  "sri    v17.8b, v16.8b, #4       \n" /* BG */          \
  "sri    v19.8b, v18.8b, #4       \n" /* RA */          \
  "zip1   v0.16b, v17.16b, v19.16b \n" /* BGRA */

void I422ToARGB4444Row_NEON(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb4444,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"  //
      READYUV422
      "subs        %w[width], %w[width], #8      \n" I4XXTORGB RGBTORGB8
      "movi        v19.8b, #255                  \n" ARGBTOARGB4444
      "st1         {v0.8h}, [%[dst_argb4444]], #16 \n"  // store 8
                                                        // pixels
                                                        // ARGB4444.
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb4444] "+r"(dst_argb4444),                 // %[dst_argb4444]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

void I400ToARGBRow_NEON(const uint8_t* src_y,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v1.16b, #128                  \n"
      "movi        v19.8b, #255                  \n"
      "umull       v6.8h, v1.8b, v30.8b          \n"
      "umlal2      v6.8h, v1.16b, v31.16b        \n" /* DG */
      "umull       v4.8h, v1.8b, v28.8b          \n" /* DB */
      "umull2      v5.8h, v1.16b, v29.16b        \n" /* DR */
      "1:          \n"                               //
      READYUV400 I400TORGB
      "subs        %w[width], %w[width], #8      \n" RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "v19");
}

#if defined(LIBYUV_USE_ST4)
void J400ToARGBRow_NEON(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  asm volatile(
      "movi        v23.8b, #255                  \n"
      "1:          \n"
      "ld1         {v20.8b}, [%0], #8            \n"
      "subs        %w2, %w2, #8                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v21.8b, v20.8b                \n"
      "mov         v22.8b, v20.8b                \n"
      "st4         {v20.8b,v21.8b,v22.8b,v23.8b}, [%1], #32 \n"
      "b.gt        1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v20", "v21", "v22", "v23");
}
#else
void J400ToARGBRow_NEON(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  asm volatile(
      "movi        v20.8b, #255                  \n"
      "1:          \n"
      "ldr         d16, [%0], #8                 \n"
      "subs        %w2, %w2, #8                  \n"
      "zip1        v18.16b, v16.16b, v16.16b     \n"  // YY
      "zip1        v19.16b, v16.16b, v20.16b     \n"  // YA
      "prfm        pldl1keep, [%0, 448]          \n"
      "zip1        v16.16b, v18.16b, v19.16b     \n"  // YYYA
      "zip2        v17.16b, v18.16b, v19.16b     \n"
      "stp         q16, q17, [%1], #32           \n"
      "b.gt        1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v16", "v17", "v18", "v19", "v20");
}
#endif  // LIBYUV_USE_ST4

void NV12ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "ldr         q2, [%[kNV12Table]]           \n"
      "1:          \n"  //
      READNV12 "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [kNV12Table] "r"(&kNV12Table)
      : "cc", "memory", YUVTORGB_REGS, "v2", "v19");
}

void NV21ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_vu,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "ldr         q2, [%[kNV12Table]]           \n"
      "1:          \n"  //
      READNV12 "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_vu),                              // %[src_uv]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [kNV12Table] "r"(&kNV21Table)
      : "cc", "memory", YUVTORGB_REGS, "v2", "v19");
}

void NV12ToRGB24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_uv,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  asm volatile(
      YUVTORGB_SETUP
      "ldr         q2, [%[kNV12Table]]           \n"
      "1:          \n"  //
      READNV12 "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st3         {v16.8b,v17.8b,v18.8b}, [%[dst_rgb24]], #24 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_rgb24] "+r"(dst_rgb24),                        // %[dst_rgb24]
        [width] "+r"(width)                                 // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [kNV12Table] "r"(&kNV12Table)
      : "cc", "memory", YUVTORGB_REGS, "v2");
}

void NV21ToRGB24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  asm volatile(
      YUVTORGB_SETUP
      "ldr         q2, [%[kNV12Table]]           \n"
      "1:          \n"  //
      READNV12 "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st3         {v16.8b,v17.8b,v18.8b}, [%[dst_rgb24]], #24 \n"
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_vu),                              // %[src_uv]
        [dst_rgb24] "+r"(dst_rgb24),                        // %[dst_rgb24]
        [width] "+r"(width)                                 // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [kNV12Table] "r"(&kNV21Table)
      : "cc", "memory", YUVTORGB_REGS, "v2");
}

void NV12ToRGB565Row_NEON(const uint8_t* src_y,
                          const uint8_t* src_uv,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  asm volatile(
      YUVTORGB_SETUP
      "ldr         q2, [%[kNV12Table]]           \n"
      "1:          \n"  //
      READNV12
      "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8_TOP
          ARGBTORGB565_FROM_TOP
      "st1         {v18.8h}, [%[dst_rgb565]], #16 \n"  // store 8
                                                       // pixels
                                                       // RGB565.
      "b.gt        1b                            \n"
      : [src_y] "+r"(src_y),                                // %[src_y]
        [src_uv] "+r"(src_uv),                              // %[src_uv]
        [dst_rgb565] "+r"(dst_rgb565),                      // %[dst_rgb565]
        [width] "+r"(width)                                 // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [kNV12Table] "r"(&kNV12Table)
      : "cc", "memory", YUVTORGB_REGS, "v2");
}

void YUY2ToARGBRow_NEON(const uint8_t* src_yuy2,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "ldr         q2, [%[kNV21InterleavedTable]] \n"
      "1:          \n"  //
      READYUY2 "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_yuy2] "+r"(src_yuy2),                          // %[src_yuy2]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [kNV21InterleavedTable] "r"(&kNV21InterleavedTable)
      : "cc", "memory", YUVTORGB_REGS, "v2", "v19");
}

void UYVYToARGBRow_NEON(const uint8_t* src_uyvy,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "movi        v19.8b, #255                  \n"
      "ldr         q2, [%[kNV12InterleavedTable]] \n"
      "1:          \n"  //
      READUYVY "subs        %w[width], %w[width], #8      \n" NVTORGB RGBTORGB8
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%[dst_argb]], #32 \n"
      "b.gt        1b                            \n"
      : [src_uyvy] "+r"(src_uyvy),                          // %[src_yuy2]
        [dst_argb] "+r"(dst_argb),                          // %[dst_argb]
        [width] "+r"(width)                                 // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),            // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias),  // %[kRGBCoeffBias]
        [kNV12InterleavedTable] "r"(&kNV12InterleavedTable)
      : "cc", "memory", YUVTORGB_REGS, "v2", "v19");
}

// Reads 16 pairs of UV and write even values to dst_u and odd to dst_v.
void SplitUVRow_NEON(const uint8_t* src_uv,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  asm volatile(
      "1:          \n"
      "ld2         {v0.16b,v1.16b}, [%0], #32    \n"  // load 16 pairs of UV
      "subs        %w3, %w3, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v0.16b}, [%1], #16           \n"  // store U
      "st1         {v1.16b}, [%2], #16           \n"  // store V
      "b.gt        1b                            \n"
      : "+r"(src_uv),               // %0
        "+r"(dst_u),                // %1
        "+r"(dst_v),                // %2
        "+r"(width)                 // %3  // Output registers
      :                             // Input registers
      : "cc", "memory", "v0", "v1"  // Clobber List
  );
}

// Reads 16 byte Y's from tile and writes out 16 Y's.
// MM21 Y tiles are 16x32 so src_tile_stride = 512 bytes
// MM21 UV tiles are 8x16 so src_tile_stride = 256 bytes
// width measured in bytes so 8 UV = 16.
void DetileRow_NEON(const uint8_t* src,
                    ptrdiff_t src_tile_stride,
                    uint8_t* dst,
                    int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], %3            \n"  // load 16 bytes
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 1792]         \n"  // 7 tiles of 256b ahead
      "st1         {v0.16b}, [%1], #16           \n"  // store 16 bytes
      "b.gt        1b                            \n"
      : "+r"(src),            // %0
        "+r"(dst),            // %1
        "+r"(width)           // %2
      : "r"(src_tile_stride)  // %3
      : "cc", "memory", "v0"  // Clobber List
  );
}

// Reads 16 byte Y's of 16 bits from tile and writes out 16 Y's.
void DetileRow_16_NEON(const uint16_t* src,
                       ptrdiff_t src_tile_stride,
                       uint16_t* dst,
                       int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.8h,v1.8h}, [%0], %3       \n"  // load 16 pixels
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 3584]         \n"  // 7 tiles of 512b ahead
      "st1         {v0.8h,v1.8h}, [%1], #32      \n"  // store 16 pixels
      "b.gt        1b                            \n"
      : "+r"(src),                  // %0
        "+r"(dst),                  // %1
        "+r"(width)                 // %2
      : "r"(src_tile_stride * 2)    // %3
      : "cc", "memory", "v0", "v1"  // Clobber List
  );
}

// Read 16 bytes of UV, detile, and write 8 bytes of U and 8 bytes of V.
void DetileSplitUVRow_NEON(const uint8_t* src_uv,
                           ptrdiff_t src_tile_stride,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  asm volatile(
      "1:          \n"
      "ld2         {v0.8b,v1.8b}, [%0], %4       \n"
      "subs        %w3, %w3, #16                 \n"
      "prfm        pldl1keep, [%0, 1792]         \n"
      "st1         {v0.8b}, [%1], #8             \n"
      "st1         {v1.8b}, [%2], #8             \n"
      "b.gt        1b                            \n"
      : "+r"(src_uv),               // %0
        "+r"(dst_u),                // %1
        "+r"(dst_v),                // %2
        "+r"(width)                 // %3
      : "r"(src_tile_stride)        // %4
      : "cc", "memory", "v0", "v1"  // Clobber List
  );
}

#if defined(LIBYUV_USE_ST2)
// Read 16 Y, 8 UV, and write 8 YUY2
void DetileToYUY2_NEON(const uint8_t* src_y,
                       ptrdiff_t src_y_tile_stride,
                       const uint8_t* src_uv,
                       ptrdiff_t src_uv_tile_stride,
                       uint8_t* dst_yuy2,
                       int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], %4            \n"  // load 16 Ys
      "subs        %w3, %w3, #16                 \n"  // store 8 YUY2
      "prfm        pldl1keep, [%0, 1792]         \n"
      "ld1         {v1.16b}, [%1], %5            \n"  // load 8 UVs
      "prfm        pldl1keep, [%1, 1792]         \n"
      "st2         {v0.16b,v1.16b}, [%2], #32    \n"
      "b.gt        1b                            \n"
      : "+r"(src_y),                // %0
        "+r"(src_uv),               // %1
        "+r"(dst_yuy2),             // %2
        "+r"(width)                 // %3
      : "r"(src_y_tile_stride),     // %4
        "r"(src_uv_tile_stride)     // %5
      : "cc", "memory", "v0", "v1"  // Clobber list
  );
}
#else
// Read 16 Y, 8 UV, and write 8 YUY2
void DetileToYUY2_NEON(const uint8_t* src_y,
                       ptrdiff_t src_y_tile_stride,
                       const uint8_t* src_uv,
                       ptrdiff_t src_uv_tile_stride,
                       uint8_t* dst_yuy2,
                       int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], %4            \n"  // load 16 Ys
      "ld1         {v1.16b}, [%1], %5            \n"  // load 8 UVs
      "subs        %w3, %w3, #16                 \n"
      "prfm        pldl1keep, [%0, 1792]         \n"
      "zip1        v2.16b, v0.16b, v1.16b        \n"
      "prfm        pldl1keep, [%1, 1792]         \n"
      "zip2        v3.16b, v0.16b, v1.16b        \n"
      "st1         {v2.16b,v3.16b}, [%2], #32    \n"  // store 8 YUY2
      "b.gt        1b                            \n"
      : "+r"(src_y),                            // %0
        "+r"(src_uv),                           // %1
        "+r"(dst_yuy2),                         // %2
        "+r"(width)                             // %3
      : "r"(src_y_tile_stride),                 // %4
        "r"(src_uv_tile_stride)                 // %5
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber list
  );
}
#endif

// Unpack MT2T into tiled P010 64 pixels at a time. See
// tinyurl.com/mtk-10bit-video-format for format documentation.
void UnpackMT2T_NEON(const uint8_t* src, uint16_t* dst, size_t size) {
  asm volatile(
      "1:          \n"
      "ld1         {v7.16b}, [%0], #16           \n"
      "ld1         {v0.16b-v3.16b}, [%0], #64    \n"
      "subs        %2, %2, #80                   \n"
      "shl         v4.16b, v7.16b, #6            \n"
      "shl         v5.16b, v7.16b, #4            \n"
      "shl         v6.16b, v7.16b, #2            \n"
      "zip1        v16.16b, v4.16b, v0.16b       \n"
      "zip1        v18.16b, v5.16b, v1.16b       \n"
      "zip1        v20.16b, v6.16b, v2.16b       \n"
      "zip1        v22.16b, v7.16b, v3.16b       \n"
      "zip2        v17.16b, v4.16b, v0.16b       \n"
      "zip2        v19.16b, v5.16b, v1.16b       \n"
      "zip2        v21.16b, v6.16b, v2.16b       \n"
      "zip2        v23.16b, v7.16b, v3.16b       \n"
      "sri         v16.8h, v16.8h, #10           \n"
      "sri         v17.8h, v17.8h, #10           \n"
      "sri         v18.8h, v18.8h, #10           \n"
      "sri         v19.8h, v19.8h, #10           \n"
      "st1         {v16.8h-v19.8h}, [%1], #64    \n"
      "sri         v20.8h, v20.8h, #10           \n"
      "sri         v21.8h, v21.8h, #10           \n"
      "sri         v22.8h, v22.8h, #10           \n"
      "sri         v23.8h, v23.8h, #10           \n"
      "st1         {v20.8h-v23.8h}, [%1], #64    \n"
      "b.gt        1b                            \n"
      : "+r"(src),  // %0
        "+r"(dst),  // %1
        "+r"(size)  // %2
      :
      : "cc", "memory", "w0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

#if defined(LIBYUV_USE_ST2)
// Reads 16 U's and V's and writes out 16 pairs of UV.
void MergeUVRow_NEON(const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* dst_uv,
                     int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"  // load U
      "ld1         {v1.16b}, [%1], #16           \n"  // load V
      "subs        %w3, %w3, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "st2         {v0.16b,v1.16b}, [%2], #32    \n"  // store 16 pairs of UV
      "b.gt        1b                            \n"
      : "+r"(src_u),                // %0
        "+r"(src_v),                // %1
        "+r"(dst_uv),               // %2
        "+r"(width)                 // %3  // Output registers
      :                             // Input registers
      : "cc", "memory", "v0", "v1"  // Clobber List
  );
}

void MergeUVRow_16_NEON(const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint16_t* dst_uv,
                        int depth,
                        int width) {
  int shift = 16 - depth;
  asm volatile(
      "dup         v2.8h, %w4                    \n"
      "1:          \n"
      "ld1         {v0.8h}, [%0], #16            \n"  // load 8 U
      "ld1         {v1.8h}, [%1], #16            \n"  // load 8 V
      "subs        %w3, %w3, #8                  \n"  // 8 src pixels per loop
      "ushl        v0.8h, v0.8h, v2.8h           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "ushl        v1.8h, v1.8h, v2.8h           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "st2         {v0.8h, v1.8h}, [%2], #32     \n"  // store 8 UV pixels
      "b.gt        1b                            \n"
      : "+r"(src_u),   // %0
        "+r"(src_v),   // %1
        "+r"(dst_uv),  // %2
        "+r"(width)    // %3
      : "r"(shift)     // %4
      : "cc", "memory", "v0", "v1", "v2");
}
#else
// Reads 16 U's and V's and writes out 16 pairs of UV.
void MergeUVRow_NEON(const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* dst_uv,
                     int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"  // load U
      "ld1         {v1.16b}, [%1], #16           \n"  // load V
      "subs        %w3, %w3, #16                 \n"  // 16 processed per loop
      "zip1        v2.16b, v0.16b, v1.16b        \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "zip2        v3.16b, v0.16b, v1.16b        \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "st1         {v2.16b,v3.16b}, [%2], #32    \n"  // store 16 pairs of UV
      "b.gt        1b                            \n"
      : "+r"(src_u),                            // %0
        "+r"(src_v),                            // %1
        "+r"(dst_uv),                           // %2
        "+r"(width)                             // %3  // Output registers
      :                                         // Input registers
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void MergeUVRow_16_NEON(const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint16_t* dst_uv,
                        int depth,
                        int width) {
  int shift = 16 - depth;
  asm volatile(
      "dup         v4.8h, %w4                    \n"
      "1:          \n"
      "ld1         {v0.8h}, [%0], #16            \n"  // load 8 U
      "ld1         {v1.8h}, [%1], #16            \n"  // load 8 V
      "subs        %w3, %w3, #8                  \n"  // 8 src pixels per loop
      "ushl        v0.8h, v0.8h, v4.8h           \n"
      "ushl        v1.8h, v1.8h, v4.8h           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "zip1        v2.8h, v0.8h, v1.8h           \n"
      "zip2        v3.8h, v0.8h, v1.8h           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "st1         {v2.8h, v3.8h}, [%2], #32     \n"  // store 8 UV pixels
      "b.gt        1b                            \n"
      : "+r"(src_u),   // %0
        "+r"(src_v),   // %1
        "+r"(dst_uv),  // %2
        "+r"(width)    // %3
      : "r"(shift)     // %4
      : "cc", "memory", "v0", "v1", "v2", "v1", "v2", "v3", "v4");
}
#endif  // LIBYUV_USE_ST2

// Reads 16 packed RGB and write to planar dst_r, dst_g, dst_b.
void SplitRGBRow_NEON(const uint8_t* src_rgb,
                      uint8_t* dst_r,
                      uint8_t* dst_g,
                      uint8_t* dst_b,
                      int width) {
  asm volatile(
      "1:          \n"
      "ld3         {v0.16b,v1.16b,v2.16b}, [%0], #48 \n"  // load 16 RGB
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v0.16b}, [%1], #16           \n"  // store R
      "st1         {v1.16b}, [%2], #16           \n"  // store G
      "st1         {v2.16b}, [%3], #16           \n"  // store B
      "b.gt        1b                            \n"
      : "+r"(src_rgb),                    // %0
        "+r"(dst_r),                      // %1
        "+r"(dst_g),                      // %2
        "+r"(dst_b),                      // %3
        "+r"(width)                       // %4
      :                                   // Input registers
      : "cc", "memory", "v0", "v1", "v2"  // Clobber List
  );
}

// Reads 16 planar R's, G's and B's and writes out 16 packed RGB at a time
void MergeRGBRow_NEON(const uint8_t* src_r,
                      const uint8_t* src_g,
                      const uint8_t* src_b,
                      uint8_t* dst_rgb,
                      int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"  // load R
      "ld1         {v1.16b}, [%1], #16           \n"  // load G
      "ld1         {v2.16b}, [%2], #16           \n"  // load B
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "st3         {v0.16b,v1.16b,v2.16b}, [%3], #48 \n"  // store 16 RGB
      "b.gt        1b                            \n"
      : "+r"(src_r),                      // %0
        "+r"(src_g),                      // %1
        "+r"(src_b),                      // %2
        "+r"(dst_rgb),                    // %3
        "+r"(width)                       // %4
      :                                   // Input registers
      : "cc", "memory", "v0", "v1", "v2"  // Clobber List
  );
}

// Reads 16 packed ARGB and write to planar dst_r, dst_g, dst_b, dst_a.
void SplitARGBRow_NEON(const uint8_t* src_rgba,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       uint8_t* dst_a,
                       int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 ARGB
      "subs        %w5, %w5, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v0.16b}, [%3], #16           \n"  // store B
      "st1         {v1.16b}, [%2], #16           \n"  // store G
      "st1         {v2.16b}, [%1], #16           \n"  // store R
      "st1         {v3.16b}, [%4], #16           \n"  // store A
      "b.gt        1b                            \n"
      : "+r"(src_rgba),                         // %0
        "+r"(dst_r),                            // %1
        "+r"(dst_g),                            // %2
        "+r"(dst_b),                            // %3
        "+r"(dst_a),                            // %4
        "+r"(width)                             // %5
      :                                         // Input registers
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

#if defined(LIBYUV_USE_ST4)
// Reads 16 planar R's, G's, B's and A's and writes out 16 packed ARGB at a time
void MergeARGBRow_NEON(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       const uint8_t* src_a,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%2], #16           \n"  // load B
      "ld1         {v1.16b}, [%1], #16           \n"  // load G
      "ld1         {v2.16b}, [%0], #16           \n"  // load R
      "ld1         {v3.16b}, [%3], #16           \n"  // load A
      "subs        %w5, %w5, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "prfm        pldl1keep, [%3, 448]          \n"
      "st4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%4], #64 \n"  // store 16ARGB
      "b.gt        1b                            \n"
      : "+r"(src_r),                            // %0
        "+r"(src_g),                            // %1
        "+r"(src_b),                            // %2
        "+r"(src_a),                            // %3
        "+r"(dst_argb),                         // %4
        "+r"(width)                             // %5
      :                                         // Input registers
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}
#else
// Reads 16 planar R's, G's, B's and A's and writes out 16 packed ARGB at a time
void MergeARGBRow_NEON(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       const uint8_t* src_a,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%2], #16           \n"  // load B
      "ld1         {v1.16b}, [%1], #16           \n"  // load G
      "ld1         {v2.16b}, [%0], #16           \n"  // load R
      "ld1         {v3.16b}, [%3], #16           \n"  // load A
      "subs        %w5, %w5, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%2, 448]          \n"
      "zip1        v4.16b, v0.16b, v1.16b        \n"  // BG
      "zip1        v5.16b, v2.16b, v3.16b        \n"  // RA
      "prfm        pldl1keep, [%1, 448]          \n"
      "zip2        v6.16b, v0.16b, v1.16b        \n"  // BG
      "zip2        v7.16b, v2.16b, v3.16b        \n"  // RA
      "prfm        pldl1keep, [%0, 448]          \n"
      "zip1        v0.8h, v4.8h, v5.8h           \n"  // BGRA
      "zip2        v1.8h, v4.8h, v5.8h           \n"
      "prfm        pldl1keep, [%3, 448]          \n"
      "zip1        v2.8h, v6.8h, v7.8h           \n"
      "zip2        v3.8h, v6.8h, v7.8h           \n"
      "st1         {v0.16b,v1.16b,v2.16b,v3.16b}, [%4], #64 \n"  // store 16ARGB
      "b.gt        1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_argb),  // %4
        "+r"(width)      // %5
      :                  // Input registers
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7"  // Clobber List
  );
}
#endif  // LIBYUV_USE_ST4

// Reads 16 packed ARGB and write to planar dst_r, dst_g, dst_b.
void SplitXRGBRow_NEON(const uint8_t* src_rgba,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 ARGB
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v0.16b}, [%3], #16           \n"  // store B
      "st1         {v1.16b}, [%2], #16           \n"  // store G
      "st1         {v2.16b}, [%1], #16           \n"  // store R
      "b.gt        1b                            \n"
      : "+r"(src_rgba),                         // %0
        "+r"(dst_r),                            // %1
        "+r"(dst_g),                            // %2
        "+r"(dst_b),                            // %3
        "+r"(width)                             // %4
      :                                         // Input registers
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

// Reads 16 planar R's, G's and B's and writes out 16 packed ARGB at a time
void MergeXRGBRow_NEON(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "movi        v3.16b, #255                  \n"  // load A(255)
      "1:          \n"
      "ld1         {v2.16b}, [%0], #16           \n"  // load R
      "ld1         {v1.16b}, [%1], #16           \n"  // load G
      "ld1         {v0.16b}, [%2], #16           \n"  // load B
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "st4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%3], #64 \n"  // store 16ARGB
      "b.gt        1b                            \n"
      : "+r"(src_r),                            // %0
        "+r"(src_g),                            // %1
        "+r"(src_b),                            // %2
        "+r"(dst_argb),                         // %3
        "+r"(width)                             // %4
      :                                         // Input registers
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void MergeXR30Row_NEON(const uint16_t* src_r,
                       const uint16_t* src_g,
                       const uint16_t* src_b,
                       uint8_t* dst_ar30,
                       int depth,
                       int width) {
  int shift = 10 - depth;
  asm volatile(
      "movi        v30.16b, #255                 \n"
      "ushr        v30.4s, v30.4s, #22           \n"  // 1023
      "dup         v31.4s, %w5                   \n"
      "1:          \n"
      "ldr         d2, [%2], #8                  \n"  // B
      "ldr         d1, [%1], #8                  \n"  // G
      "ldr         d0, [%0], #8                  \n"  // R
      "subs        %w4, %w4, #4                  \n"
      "ushll       v2.4s, v2.4h, #0              \n"  // B
      "ushll       v1.4s, v1.4h, #0              \n"  // G
      "ushll       v0.4s, v0.4h, #0              \n"  // R
      "ushl        v2.4s, v2.4s, v31.4s          \n"  // 000B
      "ushl        v1.4s, v1.4s, v31.4s          \n"  // G
      "ushl        v0.4s, v0.4s, v31.4s          \n"  // R
      "umin        v2.4s, v2.4s, v30.4s          \n"
      "umin        v1.4s, v1.4s, v30.4s          \n"
      "umin        v0.4s, v0.4s, v30.4s          \n"
      "sli         v2.4s, v1.4s, #10             \n"  // 00GB
      "sli         v2.4s, v0.4s, #20             \n"  // 0RGB
      "orr         v2.4s, #0xc0, lsl #24         \n"  // ARGB (AR30)
      "str         q2, [%3], #16                 \n"
      "b.gt        1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_ar30),  // %3
        "+r"(width)      // %4
      : "r"(shift)       // %5
      : "memory", "cc", "v0", "v1", "v2", "v30", "v31");
}

void MergeXR30Row_10_NEON(const uint16_t* src_r,
                          const uint16_t* src_g,
                          const uint16_t* src_b,
                          uint8_t* dst_ar30,
                          int /* depth */,
                          int width) {
  // Neon has no "shift left and accumulate/orr", so use a multiply-add to
  // perform the shift instead.
  int limit = 1023;
  asm volatile(
      "dup         v5.8h, %w[limit]              \n"
      "movi        v6.8h, #16                    \n"  // 1 << 4
      "movi        v7.8h, #4, lsl #8             \n"  // 1 << 10
      "1:          \n"
      "ldr         q0, [%0], #16                 \n"  // xxxxxxRrrrrrrrrr
      "ldr         q1, [%1], #16                 \n"  // xxxxxxGggggggggg
      "ldr         q2, [%2], #16                 \n"  // xxxxxxBbbbbbbbbb
      "subs        %w4, %w4, #8                  \n"
      "umin        v0.8h, v0.8h, v5.8h           \n"  // 000000Rrrrrrrrrr
      "umin        v1.8h, v1.8h, v5.8h           \n"  // 000000Gggggggggg
      "movi        v4.8h, #0xc0, lsl #8          \n"  // 1100000000000000
      "umin        v3.8h, v2.8h, v5.8h           \n"  // 000000Bbbbbbbbbb
      "mla         v4.8h, v0.8h, v6.8h           \n"  // 11Rrrrrrrrrr0000
      "mla         v3.8h, v1.8h, v7.8h           \n"  // ggggggBbbbbbbbbb
      "usra        v4.8h, v1.8h, #6              \n"  // 11RrrrrrrrrrGggg
      "st2         {v3.8h, v4.8h}, [%3], #32     \n"
      "b.gt        1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_ar30),  // %3
        "+r"(width)      // %4
      : [limit] "r"(limit)
      : "memory", "cc", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

void MergeAR64Row_NEON(const uint16_t* src_r,
                       const uint16_t* src_g,
                       const uint16_t* src_b,
                       const uint16_t* src_a,
                       uint16_t* dst_ar64,
                       int depth,
                       int width) {
  int shift = 16 - depth;
  int mask = (1 << depth) - 1;
  asm volatile(

      "dup         v30.8h, %w7                   \n"
      "dup         v31.8h, %w6                   \n"
      "1:          \n"
      "ldr         q2, [%0], #16                 \n"  // R
      "ldr         q1, [%1], #16                 \n"  // G
      "ldr         q0, [%2], #16                 \n"  // B
      "ldr         q3, [%3], #16                 \n"  // A
      "subs        %w5, %w5, #8                  \n"
      "umin        v2.8h, v2.8h, v30.8h          \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "umin        v1.8h, v1.8h, v30.8h          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "umin        v0.8h, v0.8h, v30.8h          \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "umin        v3.8h, v3.8h, v30.8h          \n"
      "prfm        pldl1keep, [%3, 448]          \n"
      "ushl        v2.8h, v2.8h, v31.8h          \n"
      "ushl        v1.8h, v1.8h, v31.8h          \n"
      "ushl        v0.8h, v0.8h, v31.8h          \n"
      "ushl        v3.8h, v3.8h, v31.8h          \n"
      "st4         {v0.8h, v1.8h, v2.8h, v3.8h}, [%4], #64 \n"
      "b.gt        1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_ar64),  // %4
        "+r"(width)      // %5
      : "r"(shift),      // %6
        "r"(mask)        // %7
      : "memory", "cc", "v0", "v1", "v2", "v3", "v31");
}

void MergeXR64Row_NEON(const uint16_t* src_r,
                       const uint16_t* src_g,
                       const uint16_t* src_b,
                       uint16_t* dst_ar64,
                       int depth,
                       int width) {
  int shift = 16 - depth;
  int mask = (1 << depth) - 1;
  asm volatile(

      "movi        v3.16b, #0xff                 \n"  // A (0xffff)
      "dup         v30.8h, %w6                   \n"
      "dup         v31.8h, %w5                   \n"

      "1:          \n"
      "ldr         q2, [%0], #16                 \n"  // R
      "ldr         q1, [%1], #16                 \n"  // G
      "ldr         q0, [%2], #16                 \n"  // B
      "subs        %w4, %w4, #8                  \n"
      "umin        v2.8h, v2.8h, v30.8h          \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "umin        v1.8h, v1.8h, v30.8h          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "umin        v0.8h, v0.8h, v30.8h          \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "ushl        v2.8h, v2.8h, v31.8h          \n"
      "ushl        v1.8h, v1.8h, v31.8h          \n"
      "ushl        v0.8h, v0.8h, v31.8h          \n"
      "st4         {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
      "b.gt        1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_ar64),  // %3
        "+r"(width)      // %4
      : "r"(shift),      // %5
        "r"(mask)        // %6
      : "memory", "cc", "v0", "v1", "v2", "v3", "v31");
}

void MergeARGB16To8Row_NEON(const uint16_t* src_r,
                            const uint16_t* src_g,
                            const uint16_t* src_b,
                            const uint16_t* src_a,
                            uint8_t* dst_argb,
                            int depth,
                            int width) {
  // Shift is 8 - depth, +8 so the result is in the top half of each lane.
  int shift = 16 - depth;
  asm volatile(
      "dup         v31.8h, %w6                   \n"
      "1:          \n"
      "ldr         q0, [%0], #16                 \n"  // B
      "ldr         q1, [%1], #16                 \n"  // G
      "ldr         q2, [%2], #16                 \n"  // R
      "ldr         q3, [%3], #16                 \n"  // A
      "subs        %w5, %w5, #8                  \n"
      "uqshl       v0.8h, v0.8h, v31.8h          \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "uqshl       v1.8h, v1.8h, v31.8h          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "uqshl       v2.8h, v2.8h, v31.8h          \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "uqshl       v3.8h, v3.8h, v31.8h          \n"
      "prfm        pldl1keep, [%3, 448]          \n"
      "trn2        v0.16b, v0.16b, v1.16b        \n"
      "trn2        v1.16b, v2.16b, v3.16b        \n"
      "st2         {v0.8h, v1.8h}, [%4], #32     \n"
      "b.gt        1b                            \n"
      : "+r"(src_b),     // %0
        "+r"(src_g),     // %1
        "+r"(src_r),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_argb),  // %4
        "+r"(width)      // %5
      : "r"(shift)       // %6
      : "memory", "cc", "v0", "v1", "v2", "v3", "v31");
}

void MergeXRGB16To8Row_NEON(const uint16_t* src_r,
                            const uint16_t* src_g,
                            const uint16_t* src_b,
                            uint8_t* dst_argb,
                            int depth,
                            int width) {
  // Shift is 8 - depth, +8 so the result is in the top half of each lane.
  int shift = 16 - depth;
  asm volatile(
      "dup         v31.8h, %w5                   \n"
      "movi        v3.16b, #0xff                 \n"  // A (0xff)
      "1:          \n"
      "ldr         q0, [%0], #16                 \n"  // B
      "ldr         q1, [%1], #16                 \n"  // G
      "ldr         q2, [%2], #16                 \n"  // R
      "subs        %w4, %w4, #8                  \n"
      "uqshl       v0.8h, v0.8h, v31.8h          \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "uqshl       v1.8h, v1.8h, v31.8h          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "uqshl       v2.8h, v2.8h, v31.8h          \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "trn2        v0.16b, v0.16b, v1.16b        \n"
      "trn2        v1.16b, v2.16b, v3.16b        \n"
      "st2         {v0.8h, v1.8h}, [%3], #32     \n"
      "b.gt        1b                            \n"
      : "+r"(src_b),     // %0
        "+r"(src_g),     // %1
        "+r"(src_r),     // %2
        "+r"(dst_argb),  // %3
        "+r"(width)      // %4
      : "r"(shift)       // %5
      : "memory", "cc", "v0", "v1", "v2", "v3", "v31");
}

// Copy multiple of 32.
void CopyRow_NEON(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "subs        %w2, %w2, #32                 \n"  // 32 processed per loop
      "stp         q0, q1, [%1], #32             \n"
      "b.gt        1b                            \n"
      : "+r"(src),                  // %0
        "+r"(dst),                  // %1
        "+r"(width)                 // %2  // Output registers
      :                             // Input registers
      : "cc", "memory", "v0", "v1"  // Clobber List
  );
}

// SetRow writes 'width' bytes using an 8 bit value repeated.
void SetRow_NEON(uint8_t* dst, uint8_t v8, int width) {
  asm volatile(
      "dup         v0.16b, %w2                   \n"  // duplicate 16 bytes
      "1:          \n"
      "subs        %w1, %w1, #16                 \n"  // 16 bytes per loop
      "st1         {v0.16b}, [%0], #16           \n"  // store
      "b.gt        1b                            \n"
      : "+r"(dst),   // %0
        "+r"(width)  // %1
      : "r"(v8)      // %2
      : "cc", "memory", "v0");
}

void ARGBSetRow_NEON(uint8_t* dst, uint32_t v32, int width) {
  asm volatile(
      "dup         v0.4s, %w2                    \n"  // duplicate 4 ints
      "1:          \n"
      "subs        %w1, %w1, #4                  \n"  // 4 ints per loop
      "st1         {v0.16b}, [%0], #16           \n"  // store
      "b.gt        1b                            \n"
      : "+r"(dst),   // %0
        "+r"(width)  // %1
      : "r"(v32)     // %2
      : "cc", "memory", "v0");
}

// Shuffle table for reversing the bytes.
static const uvec8 kShuffleMirror = {15u, 14u, 13u, 12u, 11u, 10u, 9u, 8u,
                                     7u,  6u,  5u,  4u,  3u,  2u,  1u, 0u};

void MirrorRow_NEON(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      // Start at end of source row.
      "ld1         {v3.16b}, [%3]                \n"  // shuffler
      "add         %0, %0, %w2, sxtw             \n"
      "sub         %0, %0, #32                   \n"
      "1:          \n"
      "ldr         q2, [%0, 16]                  \n"
      "ldr         q1, [%0], -32                 \n"  // src -= 32
      "subs        %w2, %w2, #32                 \n"  // 32 pixels per loop.
      "tbl         v0.16b, {v2.16b}, v3.16b      \n"
      "tbl         v1.16b, {v1.16b}, v3.16b      \n"
      "st1         {v0.16b, v1.16b}, [%1], #32   \n"  // store 32 pixels
      "b.gt        1b                            \n"
      : "+r"(src),            // %0
        "+r"(dst),            // %1
        "+r"(width)           // %2
      : "r"(&kShuffleMirror)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

// Shuffle table for reversing the UV.
static const uvec8 kShuffleMirrorUV = {14u, 15u, 12u, 13u, 10u, 11u, 8u, 9u,
                                       6u,  7u,  4u,  5u,  2u,  3u,  0u, 1u};

void MirrorUVRow_NEON(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  asm volatile(
      // Start at end of source row.
      "ld1         {v4.16b}, [%3]                \n"  // shuffler
      "add         %0, %0, %w2, sxtw #1          \n"
      "sub         %0, %0, #32                   \n"
      "1:          \n"
      "ldr         q1, [%0, 16]                  \n"
      "ldr         q0, [%0], -32                 \n"  // src -= 32
      "subs        %w2, %w2, #16                 \n"  // 16 pixels per loop.
      "tbl         v2.16b, {v1.16b}, v4.16b      \n"
      "tbl         v3.16b, {v0.16b}, v4.16b      \n"
      "st1         {v2.16b, v3.16b}, [%1], #32   \n"  // dst += 32
      "b.gt        1b                            \n"
      : "+r"(src_uv),           // %0
        "+r"(dst_uv),           // %1
        "+r"(width)             // %2
      : "r"(&kShuffleMirrorUV)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}

void MirrorSplitUVRow_NEON(const uint8_t* src_uv,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  asm volatile(
      // Start at end of source row.
      "ld1         {v4.16b}, [%4]                \n"  // shuffler
      "add         %0, %0, %w3, sxtw #1          \n"
      "sub         %0, %0, #32                   \n"
      "1:          \n"
      "ldr         q1, [%0, 16]                  \n"
      "ldr         q0, [%0], -32                 \n"  // src -= 32
      "subs        %w3, %w3, #16                 \n"  // 16 pixels per loop.
      "tbl         v2.16b, {v1.16b}, v4.16b      \n"
      "tbl         v3.16b, {v0.16b}, v4.16b      \n"
      "uzp1        v0.16b, v2.16b, v3.16b        \n"  // U
      "uzp2        v1.16b, v2.16b, v3.16b        \n"  // V
      "st1         {v0.16b}, [%1], #16           \n"  // dst += 16
      "st1         {v1.16b}, [%2], #16           \n"
      "b.gt        1b                            \n"
      : "+r"(src_uv),           // %0
        "+r"(dst_u),            // %1
        "+r"(dst_v),            // %2
        "+r"(width)             // %3
      : "r"(&kShuffleMirrorUV)  // %4
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}

// Shuffle table for reversing the ARGB.
static const uvec8 kShuffleMirrorARGB = {12u, 13u, 14u, 15u, 8u, 9u, 10u, 11u,
                                         4u,  5u,  6u,  7u,  0u, 1u, 2u,  3u};

void ARGBMirrorRow_NEON(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  asm volatile(
      // Start at end of source row.
      "ld1         {v4.16b}, [%3]                \n"  // shuffler
      "add         %0, %0, %w2, sxtw #2          \n"
      "sub         %0, %0, #32                   \n"
      "1:          \n"
      "ldr         q1, [%0, 16]                  \n"
      "ldr         q0, [%0], -32                 \n"  // src -= 32
      "subs        %w2, %w2, #8                  \n"  // 8 pixels per loop.
      "tbl         v2.16b, {v1.16b}, v4.16b      \n"
      "tbl         v3.16b, {v0.16b}, v4.16b      \n"
      "st1         {v2.16b, v3.16b}, [%1], #32   \n"  // dst += 32
      "b.gt        1b                            \n"
      : "+r"(src_argb),           // %0
        "+r"(dst_argb),           // %1
        "+r"(width)               // %2
      : "r"(&kShuffleMirrorARGB)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}

void RGB24MirrorRow_NEON(const uint8_t* src_rgb24,
                         uint8_t* dst_rgb24,
                         int width) {
  asm volatile(
      "ld1         {v3.16b}, [%4]                \n"  // shuffler
      "add         %0, %0, %w2, sxtw #1          \n"  // Start at end of row.
      "add         %0, %0, %w2, sxtw             \n"
      "sub         %0, %0, #48                   \n"

      "1:          \n"
      "ld3         {v0.16b, v1.16b, v2.16b}, [%0], %3 \n"  // src -= 48
      "subs        %w2, %w2, #16                 \n"  // 16 pixels per loop.
      "tbl         v0.16b, {v0.16b}, v3.16b      \n"
      "tbl         v1.16b, {v1.16b}, v3.16b      \n"
      "tbl         v2.16b, {v2.16b}, v3.16b      \n"
      "st3         {v0.16b, v1.16b, v2.16b}, [%1], #48 \n"  // dst += 48
      "b.gt        1b                            \n"
      : "+r"(src_rgb24),      // %0
        "+r"(dst_rgb24),      // %1
        "+r"(width)           // %2
      : "r"((ptrdiff_t)-48),  // %3
        "r"(&kShuffleMirror)  // %4
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

void RGB24ToARGBRow_NEON(const uint8_t* src_rgb24,
                         uint8_t* dst_argb,
                         int width) {
  asm volatile(
      "movi        v4.8b, #255                   \n"  // Alpha
      "1:          \n"
      "ld3         {v1.8b,v2.8b,v3.8b}, [%0], #24 \n"  // load 8 pixels of
                                                       // RGB24.
      "subs        %w2, %w2, #8                  \n"   // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "st4         {v1.8b,v2.8b,v3.8b,v4.8b}, [%1], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_rgb24),  // %0
        "+r"(dst_argb),   // %1
        "+r"(width)       // %2
      :
      : "cc", "memory", "v1", "v2", "v3", "v4"  // Clobber List
  );
}

void RAWToARGBRow_NEON(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  asm volatile(
      "movi        v5.8b, #255                   \n"  // Alpha
      "1:          \n"
      "ld3         {v0.8b,v1.8b,v2.8b}, [%0], #24 \n"  // read r g b
      "subs        %w2, %w2, #8                  \n"   // 8 processed per loop.
      "mov         v3.8b, v1.8b                  \n"   // move g
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v4.8b, v0.8b                  \n"         // move r
      "st4         {v2.8b,v3.8b,v4.8b,v5.8b}, [%1], #32 \n"  // store b g r a
      "b.gt        1b                            \n"
      : "+r"(src_raw),   // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5"  // Clobber List
  );
}

void RAWToRGBARow_NEON(const uint8_t* src_raw, uint8_t* dst_rgba, int width) {
  asm volatile(
      "movi        v0.8b, #255                   \n"  // Alpha
      "1:          \n"
      "ld3         {v3.8b,v4.8b,v5.8b}, [%0], #24 \n"  // read r g b
      "subs        %w2, %w2, #8                  \n"   // 8 processed per loop.
      "mov         v2.8b, v4.8b                  \n"   // move g
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v1.8b, v5.8b                  \n"         // move r
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%1], #32 \n"  // store a b g r
      "b.gt        1b                            \n"
      : "+r"(src_raw),   // %0
        "+r"(dst_rgba),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5"  // Clobber List
  );
}

void RAWToRGB24Row_NEON(const uint8_t* src_raw, uint8_t* dst_rgb24, int width) {
  asm volatile(
      "1:          \n"
      "ld3         {v0.8b,v1.8b,v2.8b}, [%0], #24 \n"  // read r g b
      "subs        %w2, %w2, #8                  \n"   // 8 processed per loop.
      "mov         v3.8b, v1.8b                  \n"   // move g
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v4.8b, v0.8b                  \n"   // move r
      "st3         {v2.8b,v3.8b,v4.8b}, [%1], #24 \n"  // store b g r
      "b.gt        1b                            \n"
      : "+r"(src_raw),    // %0
        "+r"(dst_rgb24),  // %1
        "+r"(width)       // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4"  // Clobber List
  );
}

#define RGB565TOARGB                                                      \
  /* Input: v0/v4.8h: RRRRRGGGGGGBBBBB */                                 \
  "shrn       v1.8b, v0.8h, #3               \n" /* G GGGGGGxx */         \
  "shrn2      v1.16b, v4.8h, #3              \n" /* G GGGGGGxx */         \
  "uzp2       v2.16b, v0.16b, v4.16b         \n" /* R RRRRRxxx */         \
  "uzp1       v0.16b, v0.16b, v4.16b         \n" /* B xxxBBBBB */         \
  "sri        v1.16b, v1.16b, #6             \n" /* G GGGGGGGG, fill 2 */ \
  "shl        v0.16b, v0.16b, #3             \n" /* B BBBBB000 */         \
  "sri        v2.16b, v2.16b, #5             \n" /* R RRRRRRRR, fill 3 */ \
  "sri        v0.16b, v0.16b, #5             \n" /* R BBBBBBBB, fill 3 */

void RGB565ToARGBRow_NEON(const uint8_t* src_rgb565,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      "movi        v3.16b, #255                  \n"  // Alpha
      "1:          \n"
      "ldp         q0, q4, [%0], #32             \n"  // load 16 RGB565 pixels
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n" RGB565TOARGB
      "st4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%1] \n"  // store 16 ARGB
      "add         %1, %1, #64                   \n"
      "b.gt        1b                            \n"
      : "+r"(src_rgb565),  // %0
        "+r"(dst_argb),    // %1
        "+r"(width)        // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v6"  // Clobber List
  );
}

#define ARGB1555TOARGB                                   \
  /* Input: ARRRRRGGGGGBBBBB */                          \
  "shrn       v2.8b, v0.8h, #7        \n" /* RRRRRxxx */ \
  "uzp1       v29.16b, v0.16b, v4.16b \n" /* xxxBBBBB */ \
  "shrn       v1.8b, v0.8h, #2        \n" /* GGGGGxxx */ \
  "uzp2       v3.16b, v0.16b, v4.16b  \n" /* Axxxxxxx */ \
  "shrn2      v2.16b, v4.8h, #7       \n" /* RRRRRxxx */ \
  "shl        v0.16b, v29.16b, #3     \n" /* BBBBB000 */ \
  "shrn2      v1.16b, v4.8h, #2       \n" /* GGGGGxxx */ \
  "sshr       v3.16b, v3.16b, #7      \n" /* AAAAAAAA */ \
  "sri        v2.16b, v2.16b, #5      \n" /* RRRRRRRR */ \
  "sri        v1.16b, v1.16b, #5      \n" /* GGGGGGGG */ \
  "sri        v0.16b, v0.16b, #5      \n" /* BBBBBBBB */

// RGB555TOARGB is same as ARGB1555TOARGB but ignores alpha.
#define RGB555TOARGB                                         \
  /* Input: xRRRRRGGGGGBBBBB */                              \
  "uzp1       v29.16b, v0.16b, v3.16b     \n" /* xxxBBBBB */ \
  "shrn       v2.8b, v0.8h, #7            \n" /* RRRRRxxx */ \
  "shrn       v1.8b, v0.8h, #2            \n" /* GGGGGxxx */ \
  "shl        v0.16b, v29.16b, #3         \n" /* BBBBB000 */ \
  "shrn2      v2.16b, v3.8h, #7           \n" /* RRRRRxxx */ \
  "shrn2      v1.16b, v3.8h, #2           \n" /* GGGGGxxx */ \
                                                             \
  "sri        v0.16b, v0.16b, #5          \n" /* BBBBBBBB */ \
  "sri        v2.16b, v2.16b, #5          \n" /* RRRRRRRR */ \
  "sri        v1.16b, v1.16b, #5          \n" /* GGGGGGGG */

void ARGB1555ToARGBRow_NEON(const uint8_t* src_argb1555,
                            uint8_t* dst_argb,
                            int width) {
  asm volatile(
      "1:          \n"
      "ldp         q0, q4, [%0], #32             \n"  // load 16 ARGB1555 pixels
      "prfm        pldl1keep, [%0, 448]          \n"
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      ARGB1555TOARGB
      "st4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%1] \n"  // store 16 ARGB
      "add         %1, %1, #64                   \n"
      "b.gt        1b                            \n"
      : "+r"(src_argb1555),  // %0
        "+r"(dst_argb),      // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v29"  // Clobber List
  );
}

#define ARGB4444TOARGB                                        \
  /* Input: v1.8h = AAAARRRR_GGGGBBBB */                      \
  "shl        v0.16b, v1.16b, #4  \n" /* RRRR0000_BBBB0000 */ \
  "sri        v1.16b, v1.16b, #4  \n" /* AAAAAAAA_GGGGGGGG */ \
  "sri        v0.16b, v0.16b, #4  \n" /* RRRRRRRR_BBBBBBBB */

#define ARGB4444TORGB                                    \
  /* Input: v0.8h = xxxxRRRRGGGGBBBB */                  \
  "uzp1       v1.16b, v0.16b, v3.16b  \n" /* GGGGBBBB */ \
  "shrn       v2.8b, v0.8h, #4        \n" /* RRRRxxxx */ \
  "shl        v0.16b, v1.16b, #4      \n" /* BBBB0000 */ \
  "shrn2      v2.16b, v3.8h, #4       \n" /* RRRRxxxx */ \
  "sri        v1.16b, v1.16b, #4      \n" /* GGGGGGGG */ \
  "sri        v2.16b, v2.16b, #4      \n" /* RRRRRRRR */ \
  "sri        v0.16b, v0.16b, #4      \n" /* BBBBBBBB */

void ARGB4444ToARGBRow_NEON(const uint8_t* src_argb4444,
                            uint8_t* dst_argb,
                            int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v1.16b}, [%0], #16           \n"  // load 8 ARGB4444 pixels.
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n" ARGB4444TOARGB
      "st2         {v0.16b, v1.16b}, [%1], #32   \n"  // store 8 ARGB.
      "b.gt        1b                            \n"
      : "+r"(src_argb4444),  // %0
        "+r"(dst_argb),      // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4"  // Clobber List
  );
}

static const int16_t kAR30Row_BoxShifts[] = {0, -6, 0, -6, 0, -6, 0, -6};

static const uint8_t kABGRToAR30Row_BoxIndices[] = {
    2, 2, 1, 1, 6, 6, 5, 5, 10, 10, 9,  9,  14, 14, 13, 13,
    0, 0, 3, 3, 4, 4, 7, 7, 8,  8,  11, 11, 12, 12, 15, 15};
static const uint8_t kARGBToAR30Row_BoxIndices[] = {
    0, 0, 1, 1, 4, 4, 5, 5, 8,  8,  9,  9,  12, 12, 13, 13,
    2, 2, 3, 3, 6, 6, 7, 7, 10, 10, 11, 11, 14, 14, 15, 15};

// ARGB or ABGR as input, reordering based on TBL indices parameter.
static void ABCDToAR30Row_NEON(const uint8_t* src_abcd,
                               uint8_t* dst_ar30,
                               int width,
                               const uint8_t* indices) {
  asm volatile(
      "movi        v2.4s, #0xf, msl 16           \n"  // 0xfffff
      "ldr         q3, [%[kAR30Row_BoxShifts]]   \n"
      "ldp         q4, q5, [%[indices]]          \n"
      "1:          \n"
      "ldp         q0, q20, [%[src]], #32        \n"
      "subs        %w[width], %w[width], #8      \n"
      "tbl         v1.16b, {v0.16b}, v5.16b      \n"
      "tbl         v21.16b, {v20.16b}, v5.16b    \n"
      "tbl         v0.16b, {v0.16b}, v4.16b      \n"
      "tbl         v20.16b, {v20.16b}, v4.16b    \n"
      "ushl        v0.8h, v0.8h, v3.8h           \n"
      "ushl        v20.8h, v20.8h, v3.8h         \n"
      "ushl        v1.8h, v1.8h, v3.8h           \n"
      "ushl        v21.8h, v21.8h, v3.8h         \n"
      "ushr        v0.4s, v0.4s, #6              \n"
      "ushr        v20.4s, v20.4s, #6            \n"
      "shl         v1.4s, v1.4s, #14             \n"
      "shl         v21.4s, v21.4s, #14           \n"
      "bif         v0.16b, v1.16b, v2.16b        \n"
      "bif         v20.16b, v21.16b, v2.16b      \n"
      "stp         q0, q20, [%[dst]], #32        \n"
      "b.gt        1b                            \n"
      : [src] "+r"(src_abcd),                          // %[src]
        [dst] "+r"(dst_ar30),                          // %[dst]
        [width] "+r"(width)                            // %[width]
      : [kAR30Row_BoxShifts] "r"(kAR30Row_BoxShifts),  // %[kAR30Row_BoxShifts]
        [indices] "r"(indices)                         // %[indices]
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v20", "v21");
}

void ABGRToAR30Row_NEON(const uint8_t* src_abgr, uint8_t* dst_ar30, int width) {
  ABCDToAR30Row_NEON(src_abgr, dst_ar30, width, kABGRToAR30Row_BoxIndices);
}

void ARGBToAR30Row_NEON(const uint8_t* src_argb, uint8_t* dst_ar30, int width) {
  ABCDToAR30Row_NEON(src_argb, dst_ar30, width, kARGBToAR30Row_BoxIndices);
}

void ARGBToRGB24Row_NEON(const uint8_t* src_argb,
                         uint8_t* dst_rgb24,
                         int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 ARGB
      "subs        %w2, %w2, #16                 \n"  // 16 pixels per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "st3         {v0.16b,v1.16b,v2.16b}, [%1], #48 \n"  // store 8 RGB24
      "b.gt        1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(dst_rgb24),  // %1
        "+r"(width)       // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void ARGBToRAWRow_NEON(const uint8_t* src_argb, uint8_t* dst_raw, int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v1.8b,v2.8b,v3.8b,v4.8b}, [%0], #32 \n"  // load b g r a
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "mov         v4.8b, v2.8b                  \n"  // mov g
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v5.8b, v1.8b                  \n"   // mov b
      "st3         {v3.8b,v4.8b,v5.8b}, [%1], #24 \n"  // store r g b
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_raw),   // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v1", "v2", "v3", "v4", "v5"  // Clobber List
  );
}

void YUY2ToYRow_NEON(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "ld2         {v0.16b,v1.16b}, [%0], #32    \n"  // load 16 pixels of YUY2.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v0.16b}, [%1], #16           \n"  // store 16 pixels of Y.
      "b.gt        1b                            \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1"  // Clobber List
  );
}

void UYVYToYRow_NEON(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "ld2         {v0.16b,v1.16b}, [%0], #32    \n"  // load 16 pixels of UYVY.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v1.16b}, [%1], #16           \n"  // store 16 pixels of Y.
      "b.gt        1b                            \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1"  // Clobber List
  );
}

void YUY2ToUV422Row_NEON(const uint8_t* src_yuy2,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 16 YUY2
      "subs        %w3, %w3, #16                 \n"  // 16 pixels = 8 UVs.
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v1.8b}, [%1], #8             \n"  // store 8 U.
      "st1         {v3.8b}, [%2], #8             \n"  // store 8 V.
      "b.gt        1b                            \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void UYVYToUV422Row_NEON(const uint8_t* src_uyvy,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 16 UYVY
      "subs        %w3, %w3, #16                 \n"  // 16 pixels = 8 UVs.
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v0.8b}, [%1], #8             \n"  // store 8 U.
      "st1         {v2.8b}, [%2], #8             \n"  // store 8 V.
      "b.gt        1b                            \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void YUY2ToUVRow_NEON(const uint8_t* src_yuy2,
                      int stride_yuy2,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_yuy2b = src_yuy2 + stride_yuy2;
  asm volatile(
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 16 pixels
      "subs        %w4, %w4, #16                 \n"  // 16 pixels = 8 UVs.
      "ld4         {v4.8b,v5.8b,v6.8b,v7.8b}, [%1], #32 \n"  // load next row
      "urhadd      v1.8b, v1.8b, v5.8b           \n"  // average rows of U
      "prfm        pldl1keep, [%0, 448]          \n"
      "urhadd      v3.8b, v3.8b, v7.8b           \n"  // average rows of V
      "st1         {v1.8b}, [%2], #8             \n"  // store 8 U.
      "st1         {v3.8b}, [%3], #8             \n"  // store 8 V.
      "b.gt        1b                            \n"
      : "+r"(src_yuy2),   // %0
        "+r"(src_yuy2b),  // %1
        "+r"(dst_u),      // %2
        "+r"(dst_v),      // %3
        "+r"(width)       // %4
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7"  // Clobber List
  );
}

void UYVYToUVRow_NEON(const uint8_t* src_uyvy,
                      int stride_uyvy,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_uyvyb = src_uyvy + stride_uyvy;
  asm volatile(
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 16 pixels
      "subs        %w4, %w4, #16                 \n"  // 16 pixels = 8 UVs.
      "ld4         {v4.8b,v5.8b,v6.8b,v7.8b}, [%1], #32 \n"  // load next row
      "urhadd      v0.8b, v0.8b, v4.8b           \n"  // average rows of U
      "prfm        pldl1keep, [%0, 448]          \n"
      "urhadd      v2.8b, v2.8b, v6.8b           \n"  // average rows of V
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 U.
      "st1         {v2.8b}, [%3], #8             \n"  // store 8 V.
      "b.gt        1b                            \n"
      : "+r"(src_uyvy),   // %0
        "+r"(src_uyvyb),  // %1
        "+r"(dst_u),      // %2
        "+r"(dst_v),      // %3
        "+r"(width)       // %4
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7"  // Clobber List
  );
}

void YUY2ToNVUVRow_NEON(const uint8_t* src_yuy2,
                        int stride_yuy2,
                        uint8_t* dst_uv,
                        int width) {
  const uint8_t* src_yuy2b = src_yuy2 + stride_yuy2;
  asm volatile(
      "1:          \n"
      "ld2         {v0.16b,v1.16b}, [%0], #32    \n"  // load 16 pixels
      "subs        %w3, %w3, #16                 \n"  // 16 pixels = 8 UVs.
      "ld2         {v2.16b,v3.16b}, [%1], #32    \n"  // load next row
      "urhadd      v4.16b, v1.16b, v3.16b        \n"  // average rows of UV
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v4.16b}, [%2], #16           \n"  // store 8 UV.
      "b.gt        1b                            \n"
      : "+r"(src_yuy2),   // %0
        "+r"(src_yuy2b),  // %1
        "+r"(dst_uv),     // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4"  // Clobber List
  );
}

// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
void ARGBShuffleRow_NEON(const uint8_t* src_argb,
                         uint8_t* dst_argb,
                         const uint8_t* shuffler,
                         int width) {
  asm volatile(
      "ld1         {v2.16b}, [%3]                \n"  // shuffler
      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"  // load 4 pixels.
      "subs        %w2, %w2, #4                  \n"  // 4 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "tbl         v1.16b, {v0.16b}, v2.16b      \n"  // look up 4 pixels
      "st1         {v1.16b}, [%1], #16           \n"  // store 4.
      "b.gt        1b                            \n"
      : "+r"(src_argb),                   // %0
        "+r"(dst_argb),                   // %1
        "+r"(width)                       // %2
      : "r"(shuffler)                     // %3
      : "cc", "memory", "v0", "v1", "v2"  // Clobber List
  );
}

void I422ToYUY2Row_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_yuy2,
                        int width) {
  asm volatile(
      "1:          \n"
      "ld2         {v0.8b, v1.8b}, [%0], #16     \n"  // load 16 Ys
      "subs        %w4, %w4, #16                 \n"  // 16 pixels
      "mov         v2.8b, v1.8b                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "ld1         {v1.8b}, [%1], #8             \n"         // load 8 Us
      "ld1         {v3.8b}, [%2], #8             \n"         // load 8 Vs
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%3], #32 \n"  // Store 16 pixels.
      "b.gt        1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(src_u),     // %1
        "+r"(src_v),     // %2
        "+r"(dst_yuy2),  // %3
        "+r"(width)      // %4
      :
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

void I422ToUYVYRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_uyvy,
                        int width) {
  asm volatile(
      "1:          \n"
      "ld2         {v1.8b,v2.8b}, [%0], #16      \n"  // load 16 Ys
      "subs        %w4, %w4, #16                 \n"  // 16 pixels
      "mov         v3.8b, v2.8b                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "ld1         {v0.8b}, [%1], #8             \n"         // load 8 Us
      "ld1         {v2.8b}, [%2], #8             \n"         // load 8 Vs
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%3], #32 \n"  // Store 16 pixels.
      "b.gt        1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(src_u),     // %1
        "+r"(src_v),     // %2
        "+r"(dst_uyvy),  // %3
        "+r"(width)      // %4
      :
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

void ARGBToRGB565Row_NEON(const uint8_t* src_argb,
                          uint8_t* dst_rgb565,
                          int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%0], #32 \n"  // load 8
                                                                 // pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n" ARGBTORGB565
      "st1         {v18.16b}, [%1], #16          \n"  // store 8 pixels RGB565.
      "b.gt        1b                            \n"
      : "+r"(src_argb),    // %0
        "+r"(dst_rgb565),  // %1
        "+r"(width)        // %2
      :
      : "cc", "memory", "v16", "v17", "v18", "v19");
}

void ARGBToRGB565DitherRow_NEON(const uint8_t* src_argb,
                                uint8_t* dst_rgb,
                                uint32_t dither4,
                                int width) {
  asm volatile(
      "dup         v1.4s, %w3                    \n"  // dither4
      "1:          \n"
      "ld4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%0], #32 \n"  // load 8 ARGB
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "uqadd       v16.8b, v16.8b, v1.8b         \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "uqadd       v17.8b, v17.8b, v1.8b         \n"
      "uqadd       v18.8b, v18.8b, v1.8b         \n" ARGBTORGB565
      "st1         {v18.16b}, [%1], #16          \n"  // store 8 pixels RGB565.
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_rgb),   // %1
        "+r"(width)      // %2
      : "r"(dither4)     // %3
      : "cc", "memory", "v1", "v16", "v17", "v18", "v19");
}

void ARGBToARGB1555Row_NEON(const uint8_t* src_argb,
                            uint8_t* dst_argb1555,
                            int width) {
  asm volatile(
      "1:          \n"
      "ld2         {v16.8h,v17.8h}, [%0], #32    \n"  // load 8 pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n" ARGBTOARGB1555
      "st1         {v17.16b}, [%1], #16          \n"  // store 8 pixels
      "b.gt        1b                            \n"
      : "+r"(src_argb),      // %0
        "+r"(dst_argb1555),  // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "v1", "v2", "v16", "v17");
}

void ARGBToARGB4444Row_NEON(const uint8_t* src_argb,
                            uint8_t* dst_argb4444,
                            int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%0], #32 \n"  // load 8
                                                                 // pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n" ARGBTOARGB4444
      "st1         {v0.16b}, [%1], #16           \n"  // store 8 pixels
      "b.gt        1b                            \n"
      : "+r"(src_argb),      // %0
        "+r"(dst_argb4444),  // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "v0", "v1", "v16", "v17", "v18", "v19");
}

#if defined(LIBYUV_USE_ST2)
void ARGBToAR64Row_NEON(const uint8_t* src_argb,
                        uint16_t* dst_ar64,
                        int width) {
  asm volatile(
      "1:          \n"
      "ldp         q0, q2, [%0], #32             \n"  // load 8 pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "mov         v1.16b, v0.16b                \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v3.16b, v2.16b                \n"
      "st2         {v0.16b, v1.16b}, [%1], #32   \n"  // store 4 pixels
      "st2         {v2.16b, v3.16b}, [%1], #32   \n"  // store 4 pixels
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_ar64),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

static const uvec8 kShuffleARGBToABGR = {2,  1, 0, 3,  6,  5,  4,  7,
                                         10, 9, 8, 11, 14, 13, 12, 15};

void ARGBToAB64Row_NEON(const uint8_t* src_argb,
                        uint16_t* dst_ab64,
                        int width) {
  asm volatile(
      "ldr         q4, [%3]                      \n"  // shuffler
      "1:          \n"
      "ldp         q0, q2, [%0], #32             \n"  // load 8 pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "tbl         v0.16b, {v0.16b}, v4.16b      \n"
      "tbl         v2.16b, {v2.16b}, v4.16b      \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v1.16b, v0.16b                \n"
      "mov         v3.16b, v2.16b                \n"
      "st2         {v0.16b, v1.16b}, [%1], #32   \n"  // store 4 pixels
      "st2         {v2.16b, v3.16b}, [%1], #32   \n"  // store 4 pixels
      "b.gt        1b                            \n"
      : "+r"(src_argb),           // %0
        "+r"(dst_ab64),           // %1
        "+r"(width)               // %2
      : "r"(&kShuffleARGBToABGR)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}
#else
void ARGBToAR64Row_NEON(const uint8_t* src_argb,
                        uint16_t* dst_ar64,
                        int width) {
  asm volatile(
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"  // load 8 ARGB pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "zip1        v2.16b, v0.16b, v0.16b        \n"
      "zip2        v3.16b, v0.16b, v0.16b        \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "zip1        v4.16b, v1.16b, v1.16b        \n"
      "zip2        v5.16b, v1.16b, v1.16b        \n"
      "st1         {v2.8h, v3.8h, v4.8h, v5.8h}, [%1], #64 \n"  // 8 AR64
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_ar64),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
}

static const uvec8 kShuffleARGBToAB64[2] = {
    {2, 2, 1, 1, 0, 0, 3, 3, 6, 6, 5, 5, 4, 4, 7, 7},
    {10, 10, 9, 9, 8, 8, 11, 11, 14, 14, 13, 13, 12, 12, 15, 15}};

void ARGBToAB64Row_NEON(const uint8_t* src_argb,
                        uint16_t* dst_ab64,
                        int width) {
  asm volatile(
      "ldp         q6, q7, [%3]                  \n"  // 2 shufflers
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"  // load 8 pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "tbl         v2.16b, {v0.16b}, v6.16b      \n"  // ARGB to AB64
      "tbl         v3.16b, {v0.16b}, v7.16b      \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "tbl         v4.16b, {v1.16b}, v6.16b      \n"
      "tbl         v5.16b, {v1.16b}, v7.16b      \n"
      "st1         {v2.8h, v3.8h, v4.8h, v5.8h}, [%1], #64 \n"  // 8 AR64
      "b.gt        1b                            \n"
      : "+r"(src_argb),              // %0
        "+r"(dst_ab64),              // %1
        "+r"(width)                  // %2
      : "r"(&kShuffleARGBToAB64[0])  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}
#endif  // LIBYUV_USE_ST2

static const uvec8 kShuffleAR64ToARGB = {1,  3,  5,  7,  9,  11, 13, 15,
                                         17, 19, 21, 23, 25, 27, 29, 31};

void AR64ToARGBRow_NEON(const uint16_t* src_ar64,
                        uint8_t* dst_argb,
                        int width) {
  asm volatile(
      "ldr         q4, [%3]                      \n"  // shuffler
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"  // load 4 pixels
      "ldp         q2, q3, [%0], #32             \n"  // load 4 pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "tbl         v0.16b, {v0.16b, v1.16b}, v4.16b \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "tbl         v2.16b, {v2.16b, v3.16b}, v4.16b \n"
      "stp         q0, q2, [%1], #32             \n"  // store 8 pixels
      "b.gt        1b                            \n"
      : "+r"(src_ar64),           // %0
        "+r"(dst_argb),           // %1
        "+r"(width)               // %2
      : "r"(&kShuffleAR64ToARGB)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}

static const uvec8 kShuffleAB64ToARGB = {5,  3,  1,  7,  13, 11, 9,  15,
                                         21, 19, 17, 23, 29, 27, 25, 31};

void AB64ToARGBRow_NEON(const uint16_t* src_ab64,
                        uint8_t* dst_argb,
                        int width) {
  asm volatile(
      "ldr         q4, [%3]                      \n"  // shuffler
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"  // load 4 pixels
      "ldp         q2, q3, [%0], #32             \n"  // load 4 pixels
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "tbl         v0.16b, {v0.16b, v1.16b}, v4.16b \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "tbl         v2.16b, {v2.16b, v3.16b}, v4.16b \n"
      "stp         q0, q2, [%1], #32             \n"  // store 8 pixels
      "b.gt        1b                            \n"
      : "+r"(src_ab64),           // %0
        "+r"(dst_argb),           // %1
        "+r"(width)               // %2
      : "r"(&kShuffleAB64ToARGB)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}

void ARGBExtractAlphaRow_NEON(const uint8_t* src_argb,
                              uint8_t* dst_a,
                              int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16
      "prfm        pldl1keep, [%0, 448]          \n"
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop
      "st1         {v3.16b}, [%1], #16           \n"  // store 16 A's.
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_a),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

// Coefficients expressed as negatives to allow 128
struct RgbUVConstants {
  int8_t kRGBToU[4];
  int8_t kRGBToV[4];
};

// 8x1 pixels.
static void ARGBToUV444MatrixRow_NEON(
    const uint8_t* src_argb,
    uint8_t* dst_u,
    uint8_t* dst_v,
    int width,
    const struct RgbUVConstants* rgbuvconstants) {
  asm volatile(
      "ldr         d0, [%4]                      \n"  // load rgbuvconstants
      "dup         v24.16b, v0.b[0]              \n"  // UB  0.875 coefficient
      "dup         v25.16b, v0.b[1]              \n"  // UG -0.5781 coefficient
      "dup         v26.16b, v0.b[2]              \n"  // UR -0.2969 coefficient
      "dup         v27.16b, v0.b[4]              \n"  // VB -0.1406 coefficient
      "dup         v28.16b, v0.b[5]              \n"  // VG -0.7344 coefficient
      "neg         v24.16b, v24.16b              \n"
      "movi        v29.8h, #0x80, lsl #8         \n"  // 128.0

      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 8 ARGB
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "umull       v4.8h, v0.8b, v24.8b          \n"  // B
      "umlsl       v4.8h, v1.8b, v25.8b          \n"  // G
      "umlsl       v4.8h, v2.8b, v26.8b          \n"  // R
      "prfm        pldl1keep, [%0, 448]          \n"

      "umull       v3.8h, v2.8b, v24.8b          \n"  // R
      "umlsl       v3.8h, v1.8b, v28.8b          \n"  // G
      "umlsl       v3.8h, v0.8b, v27.8b          \n"  // B

      "addhn       v0.8b, v4.8h, v29.8h          \n"  // signed -> unsigned
      "addhn       v1.8b, v3.8h, v29.8h          \n"

      "st1         {v0.8b}, [%1], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%2], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
      : "+r"(src_argb),      // %0
        "+r"(dst_u),         // %1
        "+r"(dst_v),         // %2
        "+r"(width)          // %3
      : "r"(rgbuvconstants)  // %4
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v24", "v25", "v26",
        "v27", "v28", "v29");
}

static void ARGBToUV444MatrixRow_NEON_I8MM(
    const uint8_t* src_argb,
    uint8_t* dst_u,
    uint8_t* dst_v,
    int width,
    const struct RgbUVConstants* rgbuvconstants) {
  asm volatile(
      "ld2r        {v16.4s, v17.4s}, [%[rgbuvconstants]] \n"
      "movi        v29.8h, #0x80, lsl #8         \n"  // 128.0
      "1:          \n"
      "ldp         q0, q1, [%[src]], #32         \n"
      "subs        %w[width], %w[width], #8      \n"  // 8 processed per loop.
      "movi        v2.4s, #0                     \n"
      "movi        v3.4s, #0                     \n"
      "movi        v4.4s, #0                     \n"
      "movi        v5.4s, #0                     \n"
      "usdot       v2.4s, v0.16b, v16.16b        \n"
      "usdot       v3.4s, v1.16b, v16.16b        \n"
      "usdot       v4.4s, v0.16b, v17.16b        \n"
      "usdot       v5.4s, v1.16b, v17.16b        \n"
      "prfm        pldl1keep, [%[src], 448]      \n"
      "uzp1        v0.8h, v2.8h, v3.8h           \n"
      "uzp1        v1.8h, v4.8h, v5.8h           \n"
      "subhn       v0.8b, v29.8h, v0.8h          \n"  // -signed -> unsigned
      "subhn       v1.8b, v29.8h, v1.8h          \n"
      "str         d0, [%[dst_u]], #8            \n"  // store 8 pixels U.
      "str         d1, [%[dst_v]], #8            \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
      : [src] "+r"(src_argb),                 // %[src]
        [dst_u] "+r"(dst_u),                  // %[dst_u]
        [dst_v] "+r"(dst_v),                  // %[dst_v]
        [width] "+r"(width)                   // %[width]
      : [rgbuvconstants] "r"(rgbuvconstants)  // %[rgbuvconstants]
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17",
        "v29");
}

// RGB to BT601 coefficients
// UB   0.875 coefficient = 112
// UG -0.5781 coefficient = -74
// UR -0.2969 coefficient = -38
// VB -0.1406 coefficient = -18
// VG -0.7344 coefficient = -94
// VR   0.875 coefficient = 112

static const struct RgbUVConstants kARGBI601UVConstants = {{-112, 74, 38, 0},
                                                           {18, 94, -112, 0}};

void ARGBToUV444Row_NEON(const uint8_t* src_argb,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  ARGBToUV444MatrixRow_NEON(src_argb, dst_u, dst_v, width,
                            &kARGBI601UVConstants);
}

void ARGBToUV444Row_NEON_I8MM(const uint8_t* src_argb,
                              uint8_t* dst_u,
                              uint8_t* dst_v,
                              int width) {
  ARGBToUV444MatrixRow_NEON_I8MM(src_argb, dst_u, dst_v, width,
                                 &kARGBI601UVConstants);
}

// RGB to JPEG coefficients
// UB  0.500    coefficient = 128
// UG -0.33126  coefficient = -85
// UR -0.16874  coefficient = -43
// VB -0.08131  coefficient = -21
// VG -0.41869  coefficient = -107
// VR 0.500     coefficient = 128

static const struct RgbUVConstants kARGBJPEGUVConstants = {{-128, 85, 43, 0},
                                                           {21, 107, -128, 0}};

void ARGBToUVJ444Row_NEON(const uint8_t* src_argb,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  ARGBToUV444MatrixRow_NEON(src_argb, dst_u, dst_v, width,
                            &kARGBJPEGUVConstants);
}

void ARGBToUVJ444Row_NEON_I8MM(const uint8_t* src_argb,
                               uint8_t* dst_u,
                               uint8_t* dst_v,
                               int width) {
  ARGBToUV444MatrixRow_NEON_I8MM(src_argb, dst_u, dst_v, width,
                                 &kARGBJPEGUVConstants);
}

#define RGBTOUV_SETUP_REG                                                  \
  "movi       v20.8h, #112          \n" /* UB/VR coefficient  (0.875)   */ \
  "movi       v21.8h, #74           \n" /* UG coefficient    (-0.5781)  */ \
  "movi       v22.8h, #38           \n" /* UR coefficient    (-0.2969)  */ \
  "movi       v23.8h, #18           \n" /* VB coefficient    (-0.1406)  */ \
  "movi       v24.8h, #94           \n" /* VG coefficient    (-0.7344)  */ \
  "movi       v25.8h, #0x80, lsl #8 \n" /* 128.0 (0x8000 in 16-bit) */

// 16x2 pixels -> 8x1.  width is number of argb pixels. e.g. 16.
// clang-format off
#define RGBTOUV(QB, QG, QR)                                                 \
  "mul        v3.8h, " #QB ",v20.8h          \n" /* B                    */ \
  "mul        v4.8h, " #QR ",v20.8h          \n" /* R                    */ \
  "mls        v3.8h, " #QG ",v21.8h          \n" /* G                    */ \
  "mls        v4.8h, " #QG ",v24.8h          \n" /* G                    */ \
  "mls        v3.8h, " #QR ",v22.8h          \n" /* R                    */ \
  "mls        v4.8h, " #QB ",v23.8h          \n" /* B                    */ \
  "addhn      v0.8b, v3.8h, v25.8h           \n" /* +128 -> unsigned     */ \
  "addhn      v1.8b, v4.8h, v25.8h           \n" /* +128 -> unsigned     */
// clang-format on

// TODO(fbarchard): Consider vhadd vertical, then vpaddl horizontal, avoid shr.
// TODO(fbarchard): consider ptrdiff_t for all strides.

void ARGBToUVRow_NEON(const uint8_t* src_argb,
                      int src_stride_argb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_argb_1 = src_argb + src_stride_argb;
  asm volatile (
    RGBTOUV_SETUP_REG
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v2.16b                 \n"  // R 16 bytes -> 8 shorts.

      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load next 16
      "uadalp      v0.8h, v4.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v6.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_argb),  // %0
    "+r"(src_argb_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void ARGBToUVJRow_NEON(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  const uint8_t* src_argb_1 = src_argb + src_stride_argb;
  asm volatile (
      "movi        v20.8h, #128                  \n"  // UB/VR coeff (0.500)
      "movi        v21.8h, #85                   \n"  // UG coeff (-0.33126)
      "movi        v22.8h, #43                   \n"  // UR coeff (-0.16874)
      "movi        v23.8h, #21                   \n"  // VB coeff (-0.08131)
      "movi        v24.8h, #107                  \n"  // VG coeff (-0.41869)
      "movi        v25.8h, #0x80, lsl #8         \n"  // 128.0 (0x8000 in 16-bit)
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v2.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load next 16
      "uadalp      v0.8h, v4.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v6.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_argb),  // %0
    "+r"(src_argb_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void ABGRToUVJRow_NEON(const uint8_t* src_abgr,
                       int src_stride_abgr,
                       uint8_t* dst_uj,
                       uint8_t* dst_vj,
                       int width) {
  const uint8_t* src_abgr_1 = src_abgr + src_stride_abgr;
  asm volatile (
      "movi        v20.8h, #128                  \n"  // UB/VR coeff (0.500)
      "movi        v21.8h, #85                   \n"  // UG coeff (-0.33126)
      "movi        v22.8h, #43                   \n"  // UR coeff (-0.16874)
      "movi        v23.8h, #21                   \n"  // VB coeff (-0.08131)
      "movi        v24.8h, #107                  \n"  // VG coeff (-0.41869)
      "movi        v25.8h, #0x80, lsl #8         \n"  // 128.0 (0x8000 in 16-bit)
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // R 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v2.16b                 \n"  // B 16 bytes -> 8 shorts.
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load next 16
      "uadalp      v0.8h, v4.16b                 \n"  // R 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v6.16b                 \n"  // B 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v2.8h, v1.8h, v0.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_abgr),  // %0
    "+r"(src_abgr_1),  // %1
    "+r"(dst_uj),     // %2
    "+r"(dst_vj),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void RGB24ToUVJRow_NEON(const uint8_t* src_rgb24,
                        int src_stride_rgb24,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  const uint8_t* src_rgb24_1 = src_rgb24 + src_stride_rgb24;
  asm volatile (
      "movi        v20.8h, #128                  \n"  // UB/VR coeff (0.500)
      "movi        v21.8h, #85                   \n"  // UG coeff (-0.33126)
      "movi        v22.8h, #43                   \n"  // UR coeff (-0.16874)
      "movi        v23.8h, #21                   \n"  // VB coeff (-0.08131)
      "movi        v24.8h, #107                  \n"  // VG coeff (-0.41869)
      "movi        v25.8h, #0x80, lsl #8         \n"  // 128.0 (0x8000 in 16-bit)
      "1:          \n"
      "ld3         {v0.16b,v1.16b,v2.16b}, [%0], #48 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v2.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld3         {v4.16b,v5.16b,v6.16b}, [%1], #48 \n"  // load next 16
      "uadalp      v0.8h, v4.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v6.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_rgb24),  // %0
    "+r"(src_rgb24_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void RAWToUVJRow_NEON(const uint8_t* src_raw,
                      int src_stride_raw,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_raw_1 = src_raw + src_stride_raw;
  asm volatile (
      "movi        v20.8h, #128                  \n"  // UB/VR coeff (0.500)
      "movi        v21.8h, #85                   \n"  // UG coeff (-0.33126)
      "movi        v22.8h, #43                   \n"  // UR coeff (-0.16874)
      "movi        v23.8h, #21                   \n"  // VB coeff (-0.08131)
      "movi        v24.8h, #107                  \n"  // VG coeff (-0.41869)
      "movi        v25.8h, #0x80, lsl #8         \n"  // 128.0 (0x8000 in 16-bit)
      "1:          \n"
      "ld3         {v0.16b,v1.16b,v2.16b}, [%0], #48 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v2.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld3         {v4.16b,v5.16b,v6.16b}, [%1], #48 \n"  // load next 16
      "uadalp      v0.8h, v4.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v6.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v2.8h, v1.8h, v0.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_raw),  // %0
    "+r"(src_raw_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void BGRAToUVRow_NEON(const uint8_t* src_bgra,
                      int src_stride_bgra,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_bgra_1 = src_bgra + src_stride_bgra;
  asm volatile (
    RGBTOUV_SETUP_REG
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v3.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v3.8h, v2.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v1.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load 16 more
      "uadalp      v0.8h, v7.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v3.8h, v6.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v5.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v3.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_bgra),  // %0
    "+r"(src_bgra_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void ABGRToUVRow_NEON(const uint8_t* src_abgr,
                      int src_stride_abgr,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_abgr_1 = src_abgr + src_stride_abgr;
  asm volatile (
    RGBTOUV_SETUP_REG
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v3.8h, v2.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v2.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v1.8h, v0.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load 16 more.
      "uadalp      v3.8h, v6.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v2.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v1.8h, v4.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v3.8h, #2              \n"  // average of 4
      "urshr       v2.8h, v2.8h, #2              \n"
      "urshr       v1.8h, v1.8h, #2              \n"

    RGBTOUV(v0.8h, v2.8h, v1.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_abgr),  // %0
    "+r"(src_abgr_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void RGBAToUVRow_NEON(const uint8_t* src_rgba,
                      int src_stride_rgba,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  const uint8_t* src_rgba_1 = src_rgba + src_stride_rgba;
  asm volatile (
    RGBTOUV_SETUP_REG
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v1.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v2.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v3.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load 16 more.
      "uadalp      v0.8h, v5.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v6.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v7.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_rgba),  // %0
    "+r"(src_rgba_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void RGB24ToUVRow_NEON(const uint8_t* src_rgb24,
                       int src_stride_rgb24,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  const uint8_t* src_rgb24_1 = src_rgb24 + src_stride_rgb24;
  asm volatile (
    RGBTOUV_SETUP_REG
      "1:          \n"
      "ld3         {v0.16b,v1.16b,v2.16b}, [%0], #48 \n"  // load 16 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v2.8h, v2.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld3         {v4.16b,v5.16b,v6.16b}, [%1], #48 \n"  // load 16 more.
      "uadalp      v0.8h, v4.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v2.8h, v6.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v0.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v2.8h, v2.8h, #2              \n"

    RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_rgb24),  // %0
    "+r"(src_rgb24_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

void RAWToUVRow_NEON(const uint8_t* src_raw,
                     int src_stride_raw,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  const uint8_t* src_raw_1 = src_raw + src_stride_raw;
  asm volatile (
    RGBTOUV_SETUP_REG
      "1:          \n"
      "ld3         {v0.16b,v1.16b,v2.16b}, [%0], #48 \n"  // load 16 RAW pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      "uaddlp      v2.8h, v2.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v0.8h, v0.16b                 \n"  // R 16 bytes -> 8 shorts.
      "ld3         {v4.16b,v5.16b,v6.16b}, [%1], #48 \n"  // load 8 more RAW pixels
      "uadalp      v2.8h, v6.16b                 \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v5.16b                 \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v0.8h, v4.16b                 \n"  // R 16 bytes -> 8 shorts.

      "urshr       v2.8h, v2.8h, #2              \n"  // average of 4
      "urshr       v1.8h, v1.8h, #2              \n"
      "urshr       v0.8h, v0.8h, #2              \n"

    RGBTOUV(v2.8h, v1.8h, v0.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
  : "+r"(src_raw),  // %0
    "+r"(src_raw_1),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
    "v20", "v21", "v22", "v23", "v24", "v25"
  );
}

// 16x2 pixels -> 8x1.  width is number of rgb pixels. e.g. 16.
void RGB565ToUVRow_NEON(const uint8_t* src_rgb565,
                        int src_stride_rgb565,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  const uint8_t* src_rgb565_1 = src_rgb565 + src_stride_rgb565;
  asm volatile(
      RGBTOUV_SETUP_REG
      "1:          \n"
      "ldp         q0, q4, [%0], #32             \n"  // load 16 RGB565 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      RGB565TOARGB
      "uaddlp      v16.8h, v0.16b                \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v17.8h, v1.16b                \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v18.8h, v2.16b                \n"  // R 16 bytes -> 8 shorts.

      "ldp         q0, q4, [%1], #32             \n"  // load 16 RGB565 pixels.
      RGB565TOARGB
      "uadalp      v16.8h, v0.16b                \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v17.8h, v1.16b                \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v18.8h, v2.16b                \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v16.8h, #2             \n"  // average of 4
      "urshr       v1.8h, v17.8h, #2             \n"
      "urshr       v2.8h, v18.8h, #2             \n"

      RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
      : "+r"(src_rgb565),    // %0
        "+r"(src_rgb565_1),  // %1
        "+r"(dst_u),           // %2
        "+r"(dst_v),           // %3
        "+r"(width)            // %4
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28");
}

// 16x2 pixels -> 8x1.  width is number of argb pixels. e.g. 16.
void ARGB1555ToUVRow_NEON(const uint8_t* src_argb1555,
                          int src_stride_argb1555,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  const uint8_t* src_argb1555_1 = src_argb1555 + src_stride_argb1555;
  asm volatile(
      RGBTOUV_SETUP_REG
      "1:          \n"
      "ldp         q0, q3, [%0], #32             \n"  // load 16 ARGB1555 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      RGB555TOARGB
      "uaddlp      v16.8h, v0.16b                \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v17.8h, v1.16b                \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v18.8h, v2.16b                \n"  // R 16 bytes -> 8 shorts.

      "ldp         q0, q3, [%1], #32             \n"  // load 16 ARGB1555 pixels.
      RGB555TOARGB
      "uadalp      v16.8h, v0.16b                \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v17.8h, v1.16b                \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v18.8h, v2.16b                \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v16.8h, #2             \n"  // average of 4
      "urshr       v1.8h, v17.8h, #2             \n"
      "urshr       v2.8h, v18.8h, #2             \n"

      RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
      : "+r"(src_argb1555),    // %0
        "+r"(src_argb1555_1),  // %1
        "+r"(dst_u),           // %2
        "+r"(dst_v),           // %3
        "+r"(width)            // %4
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28", "v29");
}

// 16x2 pixels -> 8x1.  width is number of argb pixels. e.g. 16.
void ARGB4444ToUVRow_NEON(const uint8_t* src_argb4444,
                          int src_stride_argb4444,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  const uint8_t* src_argb4444_1 = src_argb4444 + src_stride_argb4444;
  asm volatile(
      RGBTOUV_SETUP_REG  // sets v20-v25
      "1:          \n"
      "ldp         q0, q3, [%0], #32             \n"  // load 16 ARGB4444 pixels.
      "subs        %w4, %w4, #16                 \n"  // 16 processed per loop.
      ARGB4444TORGB
      "uaddlp      v16.8h, v0.16b                \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v17.8h, v1.16b                \n"  // G 16 bytes -> 8 shorts.
      "uaddlp      v18.8h, v2.16b                \n"  // R 16 bytes -> 8 shorts.

      "ldp         q0, q3, [%1], #32             \n"  // load 16 ARGB4444 pixels.
      ARGB4444TORGB
      "uadalp      v16.8h, v0.16b                \n"  // B 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v17.8h, v1.16b                \n"  // G 16 bytes -> 8 shorts.
      "uadalp      v18.8h, v2.16b                \n"  // R 16 bytes -> 8 shorts.

      "urshr       v0.8h, v16.8h, #2             \n"  // average of 4
      "urshr       v1.8h, v17.8h, #2             \n"
      "urshr       v2.8h, v18.8h, #2             \n"

      RGBTOUV(v0.8h, v1.8h, v2.8h)
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 pixels U.
      "st1         {v1.8b}, [%3], #8             \n"  // store 8 pixels V.
      "b.gt        1b                            \n"
      : "+r"(src_argb4444),    // %0
        "+r"(src_argb4444_1),  // %1
        "+r"(dst_u),           // %2
        "+r"(dst_v),           // %3
        "+r"(width)            // %4
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17",
        "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
        "v28"

  );
}

// Process any of ARGB, ABGR, BGRA, RGBA, by adjusting the uvconstants layout.
static void ABCDToUVMatrixRow_NEON_I8MM(const uint8_t* src,
                                        int src_stride,
                                        uint8_t* dst_u,
                                        uint8_t* dst_v,
                                        int width,
                                        const int8_t* uvconstants) {
  const uint8_t* src1 = src + src_stride;
  asm volatile(
      "movi        v23.8h, #0x80, lsl #8           \n"  // 128.0 (0x8000 in
                                                        // 16-bit)
      "ld2r        {v24.4s, v25.4s}, [%[uvconstants]] \n"

      "1:          \n"
      "ld2         {v0.4s, v1.4s}, [%[src]], #32   \n"  // load 8 pixels
      "ld2         {v2.4s, v3.4s}, [%[src]], #32   \n"  // load 8 pixels
      "subs        %w[width], %w[width], #16       \n"  // 16 processed per loop
      "uaddl       v4.8h, v0.8b, v1.8b             \n"  // ABCDABCD
      "uaddl2      v5.8h, v0.16b, v1.16b           \n"  // ABCDABCD
      "uaddl       v6.8h, v2.8b, v3.8b             \n"  // ABCDABCD
      "uaddl2      v7.8h, v2.16b, v3.16b           \n"  // ABCDABCD

      "ld2         {v0.4s, v1.4s}, [%[src1]], #32  \n"  // load 8 pixels
      "ld2         {v2.4s, v3.4s}, [%[src1]], #32  \n"  // load 8 pixels
      "uaddw       v4.8h, v4.8h, v0.8b             \n"  // ABCDABCD
      "uaddw2      v5.8h, v5.8h, v0.16b            \n"  // ABCDABCD
      "uaddw       v6.8h, v6.8h, v2.8b             \n"  // ABCDABCD
      "uaddw2      v7.8h, v7.8h, v2.16b            \n"  // ABCDABCD
      "prfm        pldl1keep, [%[src], 448]        \n"
      "uaddw       v4.8h, v4.8h, v1.8b             \n"  // ABCDABCD
      "uaddw2      v5.8h, v5.8h, v1.16b            \n"  // ABCDABCD
      "uaddw       v6.8h, v6.8h, v3.8b             \n"  // ABCDABCD
      "uaddw2      v7.8h, v7.8h, v3.16b            \n"  // ABCDABCD
      "prfm        pldl1keep, [%[src1], 448]       \n"

      "rshrn       v4.8b, v4.8h, #2                \n"  // average of 4 pixels
      "rshrn       v6.8b, v6.8h, #2                \n"  // average of 4 pixels
      "rshrn2      v4.16b, v5.8h, #2               \n"  // average of 4 pixels
      "rshrn2      v6.16b, v7.8h, #2               \n"  // average of 4 pixels

      "movi        v0.4s, #0                       \n"  // U
      "movi        v1.4s, #0                       \n"  // U
      "usdot       v0.4s, v4.16b, v24.16b          \n"
      "usdot       v1.4s, v6.16b, v24.16b          \n"

      "movi        v2.4s, #0                       \n"  // V
      "movi        v3.4s, #0                       \n"  // V
      "usdot       v2.4s, v4.16b, v25.16b          \n"
      "usdot       v3.4s, v6.16b, v25.16b          \n"

      "uzp1        v0.8h, v0.8h, v1.8h             \n"  // U
      "uzp1        v1.8h, v2.8h, v3.8h             \n"  // V

      "subhn       v0.8b, v23.8h, v0.8h            \n"  // +128 -> unsigned
      "subhn       v1.8b, v23.8h, v1.8h            \n"  // +128 -> unsigned

      "str         d0, [%[dst_u]], #8              \n"  // store 8 pixels U
      "str         d1, [%[dst_v]], #8              \n"  // store 8 pixels V
      "b.gt        1b                              \n"
      : [src] "+r"(src),                // %[src]
        [src1] "+r"(src1),              // %[src1]
        [dst_u] "+r"(dst_u),            // %[dst_u]
        [dst_v] "+r"(dst_v),            // %[dst_v]
        [width] "+r"(width)             // %[width]
      : [uvconstants] "r"(uvconstants)  // %[uvconstants]
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v23",
        "v24", "v25");
}

// RGB to BT601 coefficients
// UB   0.875 coefficient = 112
// UG -0.5781 coefficient = -74
// UR -0.2969 coefficient = -38
// VB -0.1406 coefficient = -18
// VG -0.7344 coefficient = -94
// VR   0.875 coefficient = 112
// I8MM constants are stored negated such that we can store 128 in int8_t.

static const int8_t kARGBToUVCoefficients[] = {
    // -UB, -UG, -UR, 0, -VB, -VG, -VR, 0
    -112, 74, 38, 0, 18, 94, -112, 0,
};

static const int8_t kABGRToUVCoefficients[] = {
    // -UR, -UG, -UB, 0, -VR, -VG, -VB, 0
    38, 74, -112, 0, -112, 94, 18, 0,
};

static const int8_t kBGRAToUVCoefficients[] = {
    // 0, -UR, -UG, -UB, 0, -VR, -VG, -VB
    0, 38, 74, -112, 0, -112, 94, 18,
};

static const int8_t kRGBAToUVCoefficients[] = {
    // 0, -UB, -UG, -UR, 0, -VB, -VG, -VR
    0, -112, 74, 38, 0, 18, 94, -112,
};

void ARGBToUVRow_NEON_I8MM(const uint8_t* src_argb,
                           int src_stride_argb,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  ABCDToUVMatrixRow_NEON_I8MM(src_argb, src_stride_argb, dst_u, dst_v, width,
                              kARGBToUVCoefficients);
}

void ABGRToUVRow_NEON_I8MM(const uint8_t* src_abgr,
                           int src_stride_abgr,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  ABCDToUVMatrixRow_NEON_I8MM(src_abgr, src_stride_abgr, dst_u, dst_v, width,
                              kABGRToUVCoefficients);
}

void BGRAToUVRow_NEON_I8MM(const uint8_t* src_bgra,
                           int src_stride_bgra,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  ABCDToUVMatrixRow_NEON_I8MM(src_bgra, src_stride_bgra, dst_u, dst_v, width,
                              kBGRAToUVCoefficients);
}

void RGBAToUVRow_NEON_I8MM(const uint8_t* src_rgba,
                           int src_stride_rgba,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  ABCDToUVMatrixRow_NEON_I8MM(src_rgba, src_stride_rgba, dst_u, dst_v, width,
                              kRGBAToUVCoefficients);
}

// RGB to JPEG coefficients
// UB  0.500    coefficient = 128
// UG -0.33126  coefficient = -85
// UR -0.16874  coefficient = -43
// VB -0.08131  coefficient = -21
// VG -0.41869  coefficient = -107
// VR 0.500     coefficient = 128
// I8MM constants are stored negated such that we can store 128 in int8_t.

static const int8_t kARGBToUVJCoefficients[] = {
    // -UB, -UG, -UR, 0, -VB, -VG, -VR, 0
    -128, 85, 43, 0, 21, 107, -128, 0,
};

static const int8_t kABGRToUVJCoefficients[] = {
    // -UR, -UG, -UB, 0, -VR, -VG, -VB, 0
    43, 85, -128, 0, -128, 107, 21, 0,
};

void ARGBToUVJRow_NEON_I8MM(const uint8_t* src_argb,
                            int src_stride_argb,
                            uint8_t* dst_u,
                            uint8_t* dst_v,
                            int width) {
  ABCDToUVMatrixRow_NEON_I8MM(src_argb, src_stride_argb, dst_u, dst_v, width,
                              kARGBToUVJCoefficients);
}

void ABGRToUVJRow_NEON_I8MM(const uint8_t* src_abgr,
                            int src_stride_abgr,
                            uint8_t* dst_u,
                            uint8_t* dst_v,
                            int width) {
  ABCDToUVMatrixRow_NEON_I8MM(src_abgr, src_stride_abgr, dst_u, dst_v, width,
                              kABGRToUVJCoefficients);
}

void RGB565ToYRow_NEON(const uint8_t* src_rgb565, uint8_t* dst_y, int width) {
  asm volatile(
      "movi        v24.16b, #25                  \n"  // B * 0.1016 coefficient
      "movi        v25.16b, #129                 \n"  // G * 0.5078 coefficient
      "movi        v26.16b, #66                  \n"  // R * 0.2578 coefficient
      "movi        v27.16b, #16                  \n"  // Add 16 constant
      "1:          \n"
      "ldp         q0, q4, [%0], #32             \n"  // load 16 RGB565 pixels.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      RGB565TOARGB
      "umull       v3.8h, v0.8b, v24.8b          \n"  // B
      "umull2      v4.8h, v0.16b, v24.16b        \n"  // B
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v3.8h, v1.8b, v25.8b          \n"  // G
      "umlal2      v4.8h, v1.16b, v25.16b        \n"  // G
      "umlal       v3.8h, v2.8b, v26.8b          \n"  // R
      "umlal2      v4.8h, v2.16b, v26.16b        \n"  // R
      "uqrshrn     v0.8b, v3.8h, #8              \n"  // 16 bit to 8 bit Y
      "uqrshrn     v1.8b, v4.8h, #8              \n"  // 16 bit to 8 bit Y
      "uqadd       v0.8b, v0.8b, v27.8b          \n"
      "uqadd       v1.8b, v1.8b, v27.8b          \n"
      "stp         d0, d1, [%1], #16             \n"  // store 8 pixels Y.
      "b.gt        1b                            \n"
      : "+r"(src_rgb565),  // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v6", "v24", "v25", "v26",
        "v27");
}

void ARGB1555ToYRow_NEON(const uint8_t* src_argb1555,
                         uint8_t* dst_y,
                         int width) {
  asm volatile(
      "movi        v4.16b, #25                   \n"  // B * 0.1016 coefficient
      "movi        v5.16b, #129                  \n"  // G * 0.5078 coefficient
      "movi        v6.16b, #66                   \n"  // R * 0.2578 coefficient
      "movi        v7.16b, #16                   \n"  // Add 16 constant
      "1:          \n"
      "ldp         q0, q3, [%0], #32             \n"  // load 16 ARGB1555
                                                      // pixels.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      RGB555TOARGB
      "umull       v16.8h, v0.8b, v4.8b          \n"  // B
      "umull2      v17.8h, v0.16b, v4.16b        \n"  // B
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v16.8h, v1.8b, v5.8b          \n"  // G
      "umlal2      v17.8h, v1.16b, v5.16b        \n"  // G
      "umlal       v16.8h, v2.8b, v6.8b          \n"  // R
      "umlal2      v17.8h, v2.16b, v6.16b        \n"  // R
      "uqrshrn     v0.8b, v16.8h, #8             \n"  // 16 bit to 8 bit Y
      "uqrshrn2    v0.16b, v17.8h, #8            \n"  // 16 bit to 8 bit Y
      "uqadd       v0.16b, v0.16b, v7.16b        \n"
      "str         q0, [%1], #16                 \n"  // store  pixels Y.
      "b.gt        1b                            \n"
      : "+r"(src_argb1555),  // %0
        "+r"(dst_y),         // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v29");
}

void ARGB4444ToYRow_NEON(const uint8_t* src_argb4444,
                         uint8_t* dst_y,
                         int width) {
  asm volatile(
      "movi        v24.16b, #25                  \n"  // B * 0.1016 coefficient
      "movi        v25.16b, #129                 \n"  // G * 0.5078 coefficient
      "movi        v26.16b, #66                  \n"  // R * 0.2578 coefficient
      "movi        v27.16b, #16                  \n"  // Add 16 constant
      "1:          \n"
      "ldp         q0, q3, [%0], #32             \n"  // load 16 ARGB4444
                                                      // pixels.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      ARGB4444TORGB
      "umull       v16.8h, v0.8b, v24.8b         \n"  // B
      "umull2      v17.8h, v0.16b, v24.16b       \n"  // B
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v16.8h, v1.8b, v25.8b         \n"  // G
      "umlal2      v17.8h, v1.16b, v25.16b       \n"  // G
      "umlal       v16.8h, v2.8b, v26.8b         \n"  // R
      "umlal2      v17.8h, v2.16b, v26.16b       \n"  // R
      "uqrshrn     v0.8b, v16.8h, #8             \n"  // 16 bit to 8 bit Y
      "uqrshrn2    v0.16b, v17.8h, #8            \n"  // 16 bit to 8 bit Y
      "uqadd       v0.16b, v0.16b, v27.16b       \n"
      "str         q0, [%1], #16                 \n"  // store 8 pixels Y.
      "b.gt        1b                            \n"
      : "+r"(src_argb4444),  // %0
        "+r"(dst_y),         // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v24", "v25", "v26", "v27");
}

struct RgbConstants {
  uint8_t kRGBToY[4];
  uint16_t kAddY;
};

// ARGB expects first 3 values to contain RGB and 4th value is ignored.
static void ARGBToYMatrixRow_NEON(const uint8_t* src_argb,
                                  uint8_t* dst_y,
                                  int width,
                                  const struct RgbConstants* rgbconstants) {
  asm volatile(
      "ldr         d0, [%3]                      \n"  // load rgbconstants
      "dup         v6.16b, v0.b[0]               \n"
      "dup         v7.16b, v0.b[1]               \n"
      "dup         v16.16b, v0.b[2]              \n"
      "dup         v17.8h,  v0.h[2]              \n"
      "1:          \n"
      "ld4         {v2.16b,v3.16b,v4.16b,v5.16b}, [%0], #64 \n"  // load 16
                                                                 // pixels.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      "umull       v0.8h, v2.8b, v6.8b           \n"  // B
      "umull2      v1.8h, v2.16b, v6.16b         \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v0.8h, v3.8b, v7.8b           \n"  // G
      "umlal2      v1.8h, v3.16b, v7.16b         \n"
      "umlal       v0.8h, v4.8b, v16.8b          \n"  // R
      "umlal2      v1.8h, v4.16b, v16.16b        \n"
      "addhn       v0.8b, v0.8h, v17.8h          \n"  // 16 bit to 8 bit Y
      "addhn       v1.8b, v1.8h, v17.8h          \n"
      "st1         {v0.8b, v1.8b}, [%1], #16     \n"  // store 16 pixels Y.
      "b.gt        1b                            \n"
      : "+r"(src_argb),    // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      : "r"(rgbconstants)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17");
}

static void ARGBToYMatrixRow_NEON_DotProd(
    const uint8_t* src_argb,
    uint8_t* dst_y,
    int width,
    const struct RgbConstants* rgbconstants) {
  asm volatile(
      "ldr         d0, [%3]                      \n"  // load rgbconstants
      "dup         v16.4s, v0.s[0]               \n"
      "dup         v17.8h,  v0.h[2]              \n"
      "1:          \n"
      "ld1         {v4.16b, v5.16b, v6.16b, v7.16b}, [%0], #64 \n"  // load 16
                                                                    // pixels.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      "movi        v0.16b, #0                    \n"
      "movi        v1.16b, #0                    \n"
      "movi        v2.16b, #0                    \n"
      "movi        v3.16b, #0                    \n"
      "udot        v0.4s, v4.16b, v16.16b        \n"
      "udot        v1.4s, v5.16b, v16.16b        \n"
      "udot        v2.4s, v6.16b, v16.16b        \n"
      "udot        v3.4s, v7.16b, v16.16b        \n"
      "uzp1        v0.8h, v0.8h, v1.8h           \n"
      "uzp1        v1.8h, v2.8h, v3.8h           \n"
      "addhn       v0.8b, v0.8h, v17.8h          \n"
      "addhn       v1.8b, v1.8h, v17.8h          \n"
      "st1         {v0.8b, v1.8b}, [%1], #16     \n"  // store 16 pixels Y.
      "b.gt        1b                            \n"
      : "+r"(src_argb),    // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      : "r"(rgbconstants)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17");
}

// RGB to JPeg coefficients
// B * 0.1140 coefficient = 29
// G * 0.5870 coefficient = 150
// R * 0.2990 coefficient = 77
// Add 0.5
static const struct RgbConstants kRgb24JPEGConstants = {{29, 150, 77, 0},
                                                        0x0080};
static const struct RgbConstants kRgb24JPEGDotProdConstants = {{0, 29, 150, 77},
                                                               0x0080};

static const struct RgbConstants kRawJPEGConstants = {{77, 150, 29, 0}, 0x0080};

// RGB to BT.601 coefficients
// B * 0.1016 coefficient = 25
// G * 0.5078 coefficient = 129
// R * 0.2578 coefficient = 66
// Add 16.5 = 0x1080

static const struct RgbConstants kRgb24I601Constants = {{25, 129, 66, 0},
                                                        0x1080};
static const struct RgbConstants kRgb24I601DotProdConstants = {{0, 25, 129, 66},
                                                               0x1080};

static const struct RgbConstants kRawI601Constants = {{66, 129, 25, 0}, 0x1080};
static const struct RgbConstants kRawI601DotProdConstants = {{0, 66, 129, 25},
                                                             0x1080};

void ARGBToYRow_NEON(const uint8_t* src_argb, uint8_t* dst_y, int width) {
  ARGBToYMatrixRow_NEON(src_argb, dst_y, width, &kRgb24I601Constants);
}

void ARGBToYJRow_NEON(const uint8_t* src_argb, uint8_t* dst_yj, int width) {
  ARGBToYMatrixRow_NEON(src_argb, dst_yj, width, &kRgb24JPEGConstants);
}

void ABGRToYRow_NEON(const uint8_t* src_abgr, uint8_t* dst_y, int width) {
  ARGBToYMatrixRow_NEON(src_abgr, dst_y, width, &kRawI601Constants);
}

void ABGRToYJRow_NEON(const uint8_t* src_abgr, uint8_t* dst_yj, int width) {
  ARGBToYMatrixRow_NEON(src_abgr, dst_yj, width, &kRawJPEGConstants);
}

void ARGBToYRow_NEON_DotProd(const uint8_t* src_argb,
                             uint8_t* dst_y,
                             int width) {
  ARGBToYMatrixRow_NEON_DotProd(src_argb, dst_y, width, &kRgb24I601Constants);
}

void ARGBToYJRow_NEON_DotProd(const uint8_t* src_argb,
                              uint8_t* dst_yj,
                              int width) {
  ARGBToYMatrixRow_NEON_DotProd(src_argb, dst_yj, width, &kRgb24JPEGConstants);
}

void ABGRToYRow_NEON_DotProd(const uint8_t* src_abgr,
                             uint8_t* dst_y,
                             int width) {
  ARGBToYMatrixRow_NEON_DotProd(src_abgr, dst_y, width, &kRawI601Constants);
}

void ABGRToYJRow_NEON_DotProd(const uint8_t* src_abgr,
                              uint8_t* dst_yj,
                              int width) {
  ARGBToYMatrixRow_NEON_DotProd(src_abgr, dst_yj, width, &kRawJPEGConstants);
}

// RGBA expects first value to be A and ignored, then 3 values to contain RGB.
// Same code as ARGB, except the LD4
static void RGBAToYMatrixRow_NEON(const uint8_t* src_rgba,
                                  uint8_t* dst_y,
                                  int width,
                                  const struct RgbConstants* rgbconstants) {
  asm volatile(
      "ldr         d0, [%3]                      \n"  // load rgbconstants
      "dup         v6.16b, v0.b[0]               \n"
      "dup         v7.16b, v0.b[1]               \n"
      "dup         v16.16b, v0.b[2]              \n"
      "dup         v17.8h,  v0.h[2]              \n"
      "1:          \n"
      "ld4         {v1.16b,v2.16b,v3.16b,v4.16b}, [%0], #64 \n"  // load 16
                                                                 // pixels.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      "umull       v0.8h, v2.8b, v6.8b           \n"  // B
      "umull2      v1.8h, v2.16b, v6.16b         \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v0.8h, v3.8b, v7.8b           \n"  // G
      "umlal2      v1.8h, v3.16b, v7.16b         \n"
      "umlal       v0.8h, v4.8b, v16.8b          \n"  // R
      "umlal2      v1.8h, v4.16b, v16.16b        \n"
      "addhn       v0.8b, v0.8h, v17.8h          \n"  // 16 bit to 8 bit Y
      "addhn       v1.8b, v1.8h, v17.8h          \n"
      "st1         {v0.8b, v1.8b}, [%1], #16     \n"  // store 16 pixels Y.
      "b.gt        1b                            \n"
      : "+r"(src_rgba),    // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      : "r"(rgbconstants)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17");
}

void RGBAToYRow_NEON(const uint8_t* src_rgba, uint8_t* dst_y, int width) {
  RGBAToYMatrixRow_NEON(src_rgba, dst_y, width, &kRgb24I601Constants);
}

void RGBAToYJRow_NEON(const uint8_t* src_rgba, uint8_t* dst_yj, int width) {
  RGBAToYMatrixRow_NEON(src_rgba, dst_yj, width, &kRgb24JPEGConstants);
}

void BGRAToYRow_NEON(const uint8_t* src_bgra, uint8_t* dst_y, int width) {
  RGBAToYMatrixRow_NEON(src_bgra, dst_y, width, &kRawI601Constants);
}

void RGBAToYRow_NEON_DotProd(const uint8_t* src_rgba,
                             uint8_t* dst_y,
                             int width) {
  // No need for a separate implementation for RGBA inputs, just permute the
  // RGB constants.
  ARGBToYMatrixRow_NEON_DotProd(src_rgba, dst_y, width,
                                &kRgb24I601DotProdConstants);
}

void RGBAToYJRow_NEON_DotProd(const uint8_t* src_rgba,
                              uint8_t* dst_yj,
                              int width) {
  // No need for a separate implementation for RGBA inputs, just permute the
  // RGB constants.
  ARGBToYMatrixRow_NEON_DotProd(src_rgba, dst_yj, width,
                                &kRgb24JPEGDotProdConstants);
}

void BGRAToYRow_NEON_DotProd(const uint8_t* src_bgra,
                             uint8_t* dst_y,
                             int width) {
  // No need for a separate implementation for RGBA inputs, just permute the
  // RGB constants.
  ARGBToYMatrixRow_NEON_DotProd(src_bgra, dst_y, width,
                                &kRawI601DotProdConstants);
}

static void RGBToYMatrixRow_NEON(const uint8_t* src_rgb,
                                 uint8_t* dst_y,
                                 int width,
                                 const struct RgbConstants* rgbconstants) {
  asm volatile(
      "ldr         d0, [%3]                      \n"  // load rgbconstants
      "dup         v5.16b, v0.b[0]               \n"
      "dup         v6.16b, v0.b[1]               \n"
      "dup         v7.16b, v0.b[2]               \n"
      "dup         v16.8h,  v0.h[2]              \n"
      "1:          \n"
      "ld3         {v2.16b,v3.16b,v4.16b}, [%0], #48 \n"  // load 16 pixels.
      "subs        %w2, %w2, #16                 \n"  // 16 processed per loop.
      "umull       v0.8h, v2.8b, v5.8b           \n"  // B
      "umull2      v1.8h, v2.16b, v5.16b         \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v0.8h, v3.8b, v6.8b           \n"  // G
      "umlal2      v1.8h, v3.16b, v6.16b         \n"
      "umlal       v0.8h, v4.8b, v7.8b           \n"  // R
      "umlal2      v1.8h, v4.16b, v7.16b         \n"
      "addhn       v0.8b, v0.8h, v16.8h          \n"  // 16 bit to 8 bit Y
      "addhn       v1.8b, v1.8h, v16.8h          \n"
      "st1         {v0.8b, v1.8b}, [%1], #16     \n"  // store 16 pixels Y.
      "b.gt        1b                            \n"
      : "+r"(src_rgb),     // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      : "r"(rgbconstants)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16");
}

void RGB24ToYJRow_NEON(const uint8_t* src_rgb24, uint8_t* dst_yj, int width) {
  RGBToYMatrixRow_NEON(src_rgb24, dst_yj, width, &kRgb24JPEGConstants);
}

void RAWToYJRow_NEON(const uint8_t* src_raw, uint8_t* dst_yj, int width) {
  RGBToYMatrixRow_NEON(src_raw, dst_yj, width, &kRawJPEGConstants);
}

void RGB24ToYRow_NEON(const uint8_t* src_rgb24, uint8_t* dst_y, int width) {
  RGBToYMatrixRow_NEON(src_rgb24, dst_y, width, &kRgb24I601Constants);
}

void RAWToYRow_NEON(const uint8_t* src_raw, uint8_t* dst_y, int width) {
  RGBToYMatrixRow_NEON(src_raw, dst_y, width, &kRawI601Constants);
}

// Bilinear filter 16x2 -> 16x1
void InterpolateRow_NEON(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         int dst_width,
                         int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint8_t* src_ptr1 = src_ptr + src_stride;
  asm volatile(
      "cmp         %w4, #0                       \n"
      "b.eq        100f                          \n"
      "cmp         %w4, #128                     \n"
      "b.eq        50f                           \n"

      "dup         v5.16b, %w4                   \n"
      "dup         v4.16b, %w5                   \n"
      // General purpose row blend.
      "1:          \n"
      "ld1         {v0.16b}, [%1], #16           \n"
      "ld1         {v1.16b}, [%2], #16           \n"
      "subs        %w3, %w3, #16                 \n"
      "umull       v2.8h, v0.8b,  v4.8b          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "umull2      v3.8h, v0.16b, v4.16b         \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "umlal       v2.8h, v1.8b,  v5.8b          \n"
      "umlal2      v3.8h, v1.16b, v5.16b         \n"
      "rshrn       v0.8b,  v2.8h, #8             \n"
      "rshrn2      v0.16b, v3.8h, #8             \n"
      "st1         {v0.16b}, [%0], #16           \n"
      "b.gt        1b                            \n"
      "b           99f                           \n"

      // Blend 50 / 50.
      "50:         \n"
      "ld1         {v0.16b}, [%1], #16           \n"
      "ld1         {v1.16b}, [%2], #16           \n"
      "subs        %w3, %w3, #16                 \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "urhadd      v0.16b, v0.16b, v1.16b        \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "st1         {v0.16b}, [%0], #16           \n"
      "b.gt        50b                           \n"
      "b           99f                           \n"

      // Blend 100 / 0 - Copy row unchanged.
      "100:        \n"
      "ld1         {v0.16b}, [%1], #16           \n"
      "subs        %w3, %w3, #16                 \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "st1         {v0.16b}, [%0], #16           \n"
      "b.gt        100b                          \n"

      "99:         \n"
      : "+r"(dst_ptr),      // %0
        "+r"(src_ptr),      // %1
        "+r"(src_ptr1),     // %2
        "+r"(dst_width),    // %3
        "+r"(y1_fraction),  // %4
        "+r"(y0_fraction)   // %5
      :
      : "cc", "memory", "v0", "v1", "v3", "v4", "v5");
}

// Bilinear filter 8x2 -> 8x1
void InterpolateRow_16_NEON(uint16_t* dst_ptr,
                            const uint16_t* src_ptr,
                            ptrdiff_t src_stride,
                            int dst_width,
                            int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint16_t* src_ptr1 = src_ptr + src_stride;

  asm volatile(
      "cmp         %w4, #0                       \n"
      "b.eq        100f                          \n"
      "cmp         %w4, #128                     \n"
      "b.eq        50f                           \n"

      "dup         v5.8h, %w4                    \n"
      "dup         v4.8h, %w5                    \n"
      // General purpose row blend.
      "1:          \n"
      "ld1         {v0.8h}, [%1], #16            \n"
      "ld1         {v1.8h}, [%2], #16            \n"
      "subs        %w3, %w3, #8                  \n"
      "umull       v2.4s, v0.4h, v4.4h           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "umull2      v3.4s, v0.8h, v4.8h           \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "umlal       v2.4s, v1.4h, v5.4h           \n"
      "umlal2      v3.4s, v1.8h, v5.8h           \n"
      "rshrn       v0.4h, v2.4s, #8              \n"
      "rshrn2      v0.8h, v3.4s, #8              \n"
      "st1         {v0.8h}, [%0], #16            \n"
      "b.gt        1b                            \n"
      "b           99f                           \n"

      // Blend 50 / 50.
      "50:         \n"
      "ld1         {v0.8h}, [%1], #16            \n"
      "ld1         {v1.8h}, [%2], #16            \n"
      "subs        %w3, %w3, #8                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "urhadd      v0.8h, v0.8h, v1.8h           \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "st1         {v0.8h}, [%0], #16            \n"
      "b.gt        50b                           \n"
      "b           99f                           \n"

      // Blend 100 / 0 - Copy row unchanged.
      "100:        \n"
      "ld1         {v0.8h}, [%1], #16            \n"
      "subs        %w3, %w3, #8                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "st1         {v0.8h}, [%0], #16            \n"
      "b.gt        100b                          \n"

      "99:         \n"
      : "+r"(dst_ptr),     // %0
        "+r"(src_ptr),     // %1
        "+r"(src_ptr1),    // %2
        "+r"(dst_width)    // %3
      : "r"(y1_fraction),  // %4
        "r"(y0_fraction)   // %5
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
}

// Bilinear filter 8x2 -> 8x1
// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits
// 16384 = 10 bits
// 4096 = 12 bits
// 256 = 16 bits
void InterpolateRow_16To8_NEON(uint8_t* dst_ptr,
                               const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               int scale,
                               int dst_width,
                               int source_y_fraction) {
  int y1_fraction = source_y_fraction;
  int y0_fraction = 256 - y1_fraction;
  const uint16_t* src_ptr1 = src_ptr + src_stride;
  int shift = 15 - __builtin_clz((int32_t)scale);  // Negative shl is shr

  asm volatile(
      "dup         v6.8h, %w6                    \n"
      "cmp         %w4, #0                       \n"
      "b.eq        100f                          \n"
      "cmp         %w4, #128                     \n"
      "b.eq        50f                           \n"

      "dup         v5.8h, %w4                    \n"
      "dup         v4.8h, %w5                    \n"
      // General purpose row blend.
      "1:          \n"
      "ld1         {v0.8h}, [%1], #16            \n"
      "ld1         {v1.8h}, [%2], #16            \n"
      "subs        %w3, %w3, #8                  \n"
      "umull       v2.4s, v0.4h, v4.4h           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "umull2      v3.4s, v0.8h, v4.8h           \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "umlal       v2.4s, v1.4h, v5.4h           \n"
      "umlal2      v3.4s, v1.8h, v5.8h           \n"
      "rshrn       v0.4h, v2.4s, #8              \n"
      "rshrn2      v0.8h, v3.4s, #8              \n"
      "ushl        v0.8h, v0.8h, v6.8h           \n"
      "uqxtn       v0.8b, v0.8h                  \n"
      "st1         {v0.8b}, [%0], #8             \n"
      "b.gt        1b                            \n"
      "b           99f                           \n"

      // Blend 50 / 50.
      "50:         \n"
      "ld1         {v0.8h}, [%1], #16            \n"
      "ld1         {v1.8h}, [%2], #16            \n"
      "subs        %w3, %w3, #8                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "urhadd      v0.8h, v0.8h, v1.8h           \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "ushl        v0.8h, v0.8h, v6.8h           \n"
      "uqxtn       v0.8b, v0.8h                  \n"
      "st1         {v0.8b}, [%0], #8             \n"
      "b.gt        50b                           \n"
      "b           99f                           \n"

      // Blend 100 / 0 - Copy row unchanged.
      "100:        \n"
      "ldr         q0, [%1], #16                 \n"
      "ushl        v0.8h, v0.8h, v2.8h           \n"  // shr = v2 is negative
      "prfm        pldl1keep, [%1, 448]          \n"
      "uqxtn       v0.8b, v0.8h                  \n"
      "subs        %w3, %w3, #8                  \n"  // 8 src pixels per loop
      "str         d0, [%0], #8                  \n"  // store 8 pixels
      "b.gt        100b                          \n"

      "99:         \n"
      : "+r"(dst_ptr),     // %0
        "+r"(src_ptr),     // %1
        "+r"(src_ptr1),    // %2
        "+r"(dst_width)    // %3
      : "r"(y1_fraction),  // %4
        "r"(y0_fraction),  // %5
        "r"(shift)         // %6
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
}

// dr * (256 - sa) / 256 + sr = dr - dr * sa / 256 + sr
void ARGBBlendRow_NEON(const uint8_t* src_argb,
                       const uint8_t* src_argb1,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "subs        %w3, %w3, #8                  \n"
      "b.lt        89f                           \n"
      // Blend 8 pixels.
      "8:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 8 ARGB0
      "ld4         {v4.8b,v5.8b,v6.8b,v7.8b}, [%1], #32 \n"  // load 8 ARGB1
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "umull       v16.8h, v4.8b, v3.8b          \n"  // db * a
      "prfm        pldl1keep, [%0, 448]          \n"
      "umull       v17.8h, v5.8b, v3.8b          \n"  // dg * a
      "prfm        pldl1keep, [%1, 448]          \n"
      "umull       v18.8h, v6.8b, v3.8b          \n"  // dr * a
      "uqrshrn     v16.8b, v16.8h, #8            \n"  // db >>= 8
      "uqrshrn     v17.8b, v17.8h, #8            \n"  // dg >>= 8
      "uqrshrn     v18.8b, v18.8h, #8            \n"  // dr >>= 8
      "uqsub       v4.8b, v4.8b, v16.8b          \n"  // db - (db * a / 256)
      "uqsub       v5.8b, v5.8b, v17.8b          \n"  // dg - (dg * a / 256)
      "uqsub       v6.8b, v6.8b, v18.8b          \n"  // dr - (dr * a / 256)
      "uqadd       v0.8b, v0.8b, v4.8b           \n"  // + sb
      "uqadd       v1.8b, v1.8b, v5.8b           \n"  // + sg
      "uqadd       v2.8b, v2.8b, v6.8b           \n"  // + sr
      "movi        v3.8b, #255                   \n"  // a = 255
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%2], #32 \n"  // store 8 ARGB
                                                             // pixels
      "b.ge        8b                            \n"

      "89:         \n"
      "adds        %w3, %w3, #8-1                \n"
      "b.lt        99f                           \n"

      // Blend 1 pixels.
      "1:          \n"
      "ld4         {v0.b,v1.b,v2.b,v3.b}[0], [%0], #4 \n"  // load 1 pixel
                                                           // ARGB0.
      "ld4         {v4.b,v5.b,v6.b,v7.b}[0], [%1], #4 \n"  // load 1 pixel
                                                           // ARGB1.
      "subs        %w3, %w3, #1                  \n"  // 1 processed per loop.
      "umull       v16.8h, v4.8b, v3.8b          \n"  // db * a
      "prfm        pldl1keep, [%0, 448]          \n"
      "umull       v17.8h, v5.8b, v3.8b          \n"  // dg * a
      "prfm        pldl1keep, [%1, 448]          \n"
      "umull       v18.8h, v6.8b, v3.8b          \n"  // dr * a
      "uqrshrn     v16.8b, v16.8h, #8            \n"  // db >>= 8
      "uqrshrn     v17.8b, v17.8h, #8            \n"  // dg >>= 8
      "uqrshrn     v18.8b, v18.8h, #8            \n"  // dr >>= 8
      "uqsub       v4.8b, v4.8b, v16.8b          \n"  // db - (db * a / 256)
      "uqsub       v5.8b, v5.8b, v17.8b          \n"  // dg - (dg * a / 256)
      "uqsub       v6.8b, v6.8b, v18.8b          \n"  // dr - (dr * a / 256)
      "uqadd       v0.8b, v0.8b, v4.8b           \n"  // + sb
      "uqadd       v1.8b, v1.8b, v5.8b           \n"  // + sg
      "uqadd       v2.8b, v2.8b, v6.8b           \n"  // + sr
      "movi        v3.8b, #255                   \n"  // a = 255
      "st4         {v0.b,v1.b,v2.b,v3.b}[0], [%2], #4 \n"  // store 1 pixel.
      "b.ge        1b                            \n"

      "99:         \n"

      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18");
}

// Attenuate 8 pixels at a time.
void ARGBAttenuateRow_NEON(const uint8_t* src_argb,
                           uint8_t* dst_argb,
                           int width) {
  asm volatile(
      "movi        v7.8h, #0x00ff                \n"  // 255 for rounding up

      // Attenuate 8 pixels.
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 8 ARGB
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "umull       v4.8h, v0.8b, v3.8b           \n"  // b * a
      "prfm        pldl1keep, [%0, 448]          \n"
      "umull       v5.8h, v1.8b, v3.8b           \n"         // g * a
      "umull       v6.8h, v2.8b, v3.8b           \n"         // r * a
      "addhn       v0.8b, v4.8h, v7.8h           \n"         // (b + 255) >> 8
      "addhn       v1.8b, v5.8h, v7.8h           \n"         // (g + 255) >> 8
      "addhn       v2.8b, v6.8h, v7.8h           \n"         // (r + 255) >> 8
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%1], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

// Quantize 8 ARGB pixels (32 bytes).
// dst = (dst * scale >> 16) * interval_size + interval_offset;
void ARGBQuantizeRow_NEON(uint8_t* dst_argb,
                          int scale,
                          int interval_size,
                          int interval_offset,
                          int width) {
  asm volatile(
      "dup         v4.8h, %w2                    \n"
      "ushr        v4.8h, v4.8h, #1              \n"  // scale >>= 1
      "dup         v5.8h, %w3                    \n"  // interval multiply.
      "dup         v6.8h, %w4                    \n"  // interval add

      // 8 pixel loop.
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0] \n"  // load 8  ARGB.
      "subs        %w1, %w1, #8                  \n"    // 8 processed per loop.
      "uxtl        v0.8h, v0.8b                  \n"    // b (0 .. 255)
      "prfm        pldl1keep, [%0, 448]          \n"
      "uxtl        v1.8h, v1.8b                  \n"
      "uxtl        v2.8h, v2.8b                  \n"
      "sqdmulh     v0.8h, v0.8h, v4.8h           \n"  // b * scale
      "sqdmulh     v1.8h, v1.8h, v4.8h           \n"  // g
      "sqdmulh     v2.8h, v2.8h, v4.8h           \n"  // r
      "mul         v0.8h, v0.8h, v5.8h           \n"  // b * interval_size
      "mul         v1.8h, v1.8h, v5.8h           \n"  // g
      "mul         v2.8h, v2.8h, v5.8h           \n"  // r
      "add         v0.8h, v0.8h, v6.8h           \n"  // b + interval_offset
      "add         v1.8h, v1.8h, v6.8h           \n"  // g
      "add         v2.8h, v2.8h, v6.8h           \n"  // r
      "uqxtn       v0.8b, v0.8h                  \n"
      "uqxtn       v1.8b, v1.8h                  \n"
      "uqxtn       v2.8b, v2.8h                  \n"
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(dst_argb),       // %0
        "+r"(width)           // %1
      : "r"(scale),           // %2
        "r"(interval_size),   // %3
        "r"(interval_offset)  // %4
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
}

// Shade 8 pixels at a time by specified value.
// sqrdmulh is a rounding instruction, so +1 if high bit of low half of
// multiply result is set.
void ARGBShadeRow_NEON(const uint8_t* src_argb,
                       uint8_t* dst_argb,
                       int width,
                       uint32_t value) {
  asm volatile(
      "dup         v0.4s, %w3                    \n"  // duplicate scale value.
      "zip1        v0.16b, v0.16b, v0.16b        \n"  // v0.16b
                                                      // aarrggbbaarrggbb.
      "ushr        v0.8h, v0.8h, #1              \n"  // scale / 2.

      // 8 pixel loop.
      "1:          \n"
      "ld1         {v4.8b,v5.8b,v6.8b,v7.8b}, [%0], #32 \n"  // load 8 ARGB
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "uxtl        v4.8h, v4.8b                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "uxtl        v5.8h, v5.8b                  \n"
      "uxtl        v6.8h, v6.8b                  \n"
      "uxtl        v7.8h, v7.8b                  \n"
      "sqrdmulh    v4.8h, v4.8h, v0.8h           \n"  // argb * scale * 2
      "sqrdmulh    v5.8h, v5.8h, v0.8h           \n"
      "sqrdmulh    v6.8h, v6.8h, v0.8h           \n"
      "sqrdmulh    v7.8h, v7.8h, v0.8h           \n"
      "uqxtn       v4.8b, v4.8h                  \n"
      "uqxtn       v5.8b, v5.8h                  \n"
      "uqxtn       v6.8b, v6.8h                  \n"
      "uqxtn       v7.8b, v7.8h                  \n"
      "st1         {v4.8b,v5.8b,v6.8b,v7.8b}, [%1], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      : "r"(value)       // %3
      : "cc", "memory", "v0", "v4", "v5", "v6", "v7");
}

// Convert 8 ARGB pixels (64 bytes) to 8 Gray ARGB pixels
// Similar to ARGBToYJ but stores ARGB.
// C code is (29 * b + 150 * g + 77 * r + 128) >> 8;
void ARGBGrayRow_NEON(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  asm volatile(
      "movi        v24.8b, #29                   \n"  // B * 0.1140 coefficient
      "movi        v25.8b, #150                  \n"  // G * 0.5870 coefficient
      "movi        v26.8b, #77                   \n"  // R * 0.2990 coefficient
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 8 ARGB
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "umull       v4.8h, v0.8b, v24.8b          \n"  // B
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v4.8h, v1.8b, v25.8b          \n"  // G
      "umlal       v4.8h, v2.8b, v26.8b          \n"  // R
      "uqrshrn     v0.8b, v4.8h, #8              \n"  // 16 bit to 8 bit B
      "mov         v1.8b, v0.8b                  \n"  // G
      "mov         v2.8b, v0.8b                  \n"  // R
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%1], #32 \n"  // store 8 pixels.
      "b.gt        1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v24", "v25", "v26");
}

static const uvec8 kARGBGrayRowCoeffs = {29, 150, 77, 0};
static const uvec8 kARGBGrayRowIndices = {0, 0, 0, 19, 2, 2, 2, 23,
                                          4, 4, 4, 27, 6, 6, 6, 31};

void ARGBGrayRow_NEON_DotProd(const uint8_t* src_argb,
                              uint8_t* dst_argb,
                              int width) {
  asm volatile(
      "ld1r        {v24.4s}, [%[coeffs]]         \n"
      "ldr         q25, [%[indices]]             \n"
      "1:          \n"
      "ldp         q1, q3, [%[src]], #32         \n"  // load 8 ARGB
      "subs        %w[width], %w[width], #8      \n"  // 8 processed per loop
      "movi        v0.4s, #0                     \n"
      "movi        v2.4s, #0                     \n"
      "udot        v0.4s, v1.16b, v24.16b        \n"
      "udot        v2.4s, v3.16b, v24.16b        \n"
      "prfm        pldl1keep, [%[src], 448]      \n"
      "uqrshrn     v0.8b, v0.8h, #8              \n"
      "uqrshrn     v2.8b, v2.8h, #8              \n"
      "tbl         v0.16b, {v0.16b, v1.16b}, v25.16b \n"  // merge in alpha
      "tbl         v1.16b, {v2.16b, v3.16b}, v25.16b \n"
      "stp         q0, q1, [%[dst]], #32         \n"  // store 8 pixels
      "b.gt        1b                            \n"
      : [src] "+r"(src_argb),                // %[src]
        [dst] "+r"(dst_argb),                // %[dst]
        [width] "+r"(width)                  // %[width]
      : [coeffs] "r"(&kARGBGrayRowCoeffs),   // %[coeffs]
        [indices] "r"(&kARGBGrayRowIndices)  // %[indices]
      : "cc", "memory", "v0", "v1", "v2", "v3", "v24", "v25");
}

// Convert 8 ARGB pixels (32 bytes) to 8 Sepia ARGB pixels.
//    b = (r * 35 + g * 68 + b * 17) >> 7
//    g = (r * 45 + g * 88 + b * 22) >> 7
//    r = (r * 50 + g * 98 + b * 24) >> 7

void ARGBSepiaRow_NEON(uint8_t* dst_argb, int width) {
  asm volatile(
      "movi        v20.8b, #17                   \n"  // BB coefficient
      "movi        v21.8b, #68                   \n"  // BG coefficient
      "movi        v22.8b, #35                   \n"  // BR coefficient
      "movi        v24.8b, #22                   \n"  // GB coefficient
      "movi        v25.8b, #88                   \n"  // GG coefficient
      "movi        v26.8b, #45                   \n"  // GR coefficient
      "movi        v28.8b, #24                   \n"  // BB coefficient
      "movi        v29.8b, #98                   \n"  // BG coefficient
      "movi        v30.8b, #50                   \n"  // BR coefficient
      "1:          \n"
      "ld4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0] \n"  // load 8 ARGB pixels.
      "subs        %w1, %w1, #8                  \n"    // 8 processed per loop.
      "umull       v4.8h, v0.8b, v20.8b          \n"    // B to Sepia B
      "prfm        pldl1keep, [%0, 448]          \n"
      "umlal       v4.8h, v1.8b, v21.8b          \n"  // G
      "umlal       v4.8h, v2.8b, v22.8b          \n"  // R
      "umull       v5.8h, v0.8b, v24.8b          \n"  // B to Sepia G
      "umlal       v5.8h, v1.8b, v25.8b          \n"  // G
      "umlal       v5.8h, v2.8b, v26.8b          \n"  // R
      "umull       v6.8h, v0.8b, v28.8b          \n"  // B to Sepia R
      "umlal       v6.8h, v1.8b, v29.8b          \n"  // G
      "umlal       v6.8h, v2.8b, v30.8b          \n"  // R
      "uqshrn      v0.8b, v4.8h, #7              \n"  // 16 bit to 8 bit B
      "uqshrn      v1.8b, v5.8h, #7              \n"  // 16 bit to 8 bit G
      "uqshrn      v2.8b, v6.8h, #7              \n"  // 16 bit to 8 bit R
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // store 8 pixels.
      "b.gt        1b                            \n"
      : "+r"(dst_argb),  // %0
        "+r"(width)      // %1
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20",
        "v21", "v22", "v24", "v25", "v26", "v28", "v29", "v30");
}

static const uvec8 kARGBSepiaRowCoeffs = {17, 68, 35, 0,  22, 88,
                                          45, 0,  24, 98, 50, 0};
static const uvec8 kARGBSepiaRowAlphaIndices = {3, 7, 11, 15, 19, 23, 27, 31};

void ARGBSepiaRow_NEON_DotProd(uint8_t* dst_argb, int width) {
  asm volatile(
      "ld3r        {v20.4s, v21.4s, v22.4s}, [%[coeffs]] \n"
      "ldr         d23, [%[indices]]             \n"
      "1:          \n"
      "ldp         q0, q1, [%[dst]]              \n"
      "subs        %w1, %w1, #8                  \n"
      "movi        v2.4s, #0                     \n"
      "movi        v3.4s, #0                     \n"
      "movi        v4.4s, #0                     \n"
      "movi        v5.4s, #0                     \n"
      "movi        v6.4s, #0                     \n"
      "movi        v7.4s, #0                     \n"
      "udot        v2.4s, v0.16b, v20.16b        \n"
      "udot        v3.4s, v1.16b, v20.16b        \n"
      "udot        v4.4s, v0.16b, v21.16b        \n"
      "udot        v5.4s, v1.16b, v21.16b        \n"
      "udot        v6.4s, v0.16b, v22.16b        \n"
      "udot        v7.4s, v1.16b, v22.16b        \n"
      "prfm        pldl1keep, [%[dst], 448]      \n"
      "uzp1        v6.8h, v6.8h, v7.8h           \n"
      "uzp1        v5.8h, v4.8h, v5.8h           \n"
      "uzp1        v4.8h, v2.8h, v3.8h           \n"
      "tbl         v3.16b, {v0.16b, v1.16b}, v23.16b \n"
      "uqshrn      v0.8b, v4.8h, #7              \n"
      "uqshrn      v1.8b, v5.8h, #7              \n"
      "uqshrn      v2.8b, v6.8h, #7              \n"
      "st4         {v0.8b, v1.8b, v2.8b, v3.8b}, [%[dst]], #32 \n"
      "b.gt        1b                            \n"
      : [dst] "+r"(dst_argb),                      // %[dst]
        [width] "+r"(width)                        // %[width]
      : [coeffs] "r"(&kARGBSepiaRowCoeffs),        // %[coeffs]
        [indices] "r"(&kARGBSepiaRowAlphaIndices)  // %[indices]
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20",
        "v21", "v22", "v24", "v25", "v26", "v28", "v29", "v30");
}

// Tranform 8 ARGB pixels (32 bytes) with color matrix.
// TODO(fbarchard): Was same as Sepia except matrix is provided.  This function
// needs to saturate.  Consider doing a non-saturating version.
void ARGBColorMatrixRow_NEON(const uint8_t* src_argb,
                             uint8_t* dst_argb,
                             const int8_t* matrix_argb,
                             int width) {
  asm volatile(
      "ld1         {v2.16b}, [%3]                \n"  // load 3 ARGB vectors.
      "sxtl        v0.8h, v2.8b                  \n"  // B,G coefficients s16.
      "sxtl2       v1.8h, v2.16b                 \n"  // R,A coefficients s16.

      "1:          \n"
      "ld4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%0], #32 \n"  // load 8 ARGB
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.
      "uxtl        v16.8h, v16.8b                \n"  // b (0 .. 255) 16 bit
      "prfm        pldl1keep, [%0, 448]          \n"
      "uxtl        v17.8h, v17.8b                \n"  // g
      "uxtl        v18.8h, v18.8b                \n"  // r
      "uxtl        v19.8h, v19.8b                \n"  // a
      "mul         v22.8h, v16.8h, v0.h[0]       \n"  // B = B * Matrix B
      "mul         v23.8h, v16.8h, v0.h[4]       \n"  // G = B * Matrix G
      "mul         v24.8h, v16.8h, v1.h[0]       \n"  // R = B * Matrix R
      "mul         v25.8h, v16.8h, v1.h[4]       \n"  // A = B * Matrix A
      "mul         v4.8h, v17.8h, v0.h[1]        \n"  // B += G * Matrix B
      "mul         v5.8h, v17.8h, v0.h[5]        \n"  // G += G * Matrix G
      "mul         v6.8h, v17.8h, v1.h[1]        \n"  // R += G * Matrix R
      "mul         v7.8h, v17.8h, v1.h[5]        \n"  // A += G * Matrix A
      "sqadd       v22.8h, v22.8h, v4.8h         \n"  // Accumulate B
      "sqadd       v23.8h, v23.8h, v5.8h         \n"  // Accumulate G
      "sqadd       v24.8h, v24.8h, v6.8h         \n"  // Accumulate R
      "sqadd       v25.8h, v25.8h, v7.8h         \n"  // Accumulate A
      "mul         v4.8h, v18.8h, v0.h[2]        \n"  // B += R * Matrix B
      "mul         v5.8h, v18.8h, v0.h[6]        \n"  // G += R * Matrix G
      "mul         v6.8h, v18.8h, v1.h[2]        \n"  // R += R * Matrix R
      "mul         v7.8h, v18.8h, v1.h[6]        \n"  // A += R * Matrix A
      "sqadd       v22.8h, v22.8h, v4.8h         \n"  // Accumulate B
      "sqadd       v23.8h, v23.8h, v5.8h         \n"  // Accumulate G
      "sqadd       v24.8h, v24.8h, v6.8h         \n"  // Accumulate R
      "sqadd       v25.8h, v25.8h, v7.8h         \n"  // Accumulate A
      "mul         v4.8h, v19.8h, v0.h[3]        \n"  // B += A * Matrix B
      "mul         v5.8h, v19.8h, v0.h[7]        \n"  // G += A * Matrix G
      "mul         v6.8h, v19.8h, v1.h[3]        \n"  // R += A * Matrix R
      "mul         v7.8h, v19.8h, v1.h[7]        \n"  // A += A * Matrix A
      "sqadd       v22.8h, v22.8h, v4.8h         \n"  // Accumulate B
      "sqadd       v23.8h, v23.8h, v5.8h         \n"  // Accumulate G
      "sqadd       v24.8h, v24.8h, v6.8h         \n"  // Accumulate R
      "sqadd       v25.8h, v25.8h, v7.8h         \n"  // Accumulate A
      "sqshrun     v16.8b, v22.8h, #6            \n"  // 16 bit to 8 bit B
      "sqshrun     v17.8b, v23.8h, #6            \n"  // 16 bit to 8 bit G
      "sqshrun     v18.8b, v24.8h, #6            \n"  // 16 bit to 8 bit R
      "sqshrun     v19.8b, v25.8h, #6            \n"  // 16 bit to 8 bit A
      "st4         {v16.8b,v17.8b,v18.8b,v19.8b}, [%1], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(dst_argb),   // %1
        "+r"(width)       // %2
      : "r"(matrix_argb)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16",
        "v17", "v18", "v19", "v22", "v23", "v24", "v25");
}

void ARGBColorMatrixRow_NEON_I8MM(const uint8_t* src_argb,
                                  uint8_t* dst_argb,
                                  const int8_t* matrix_argb,
                                  int width) {
  asm volatile(
      "ld1         {v31.16b}, [%[matrix_argb]]   \n"

      "1:          \n"
      "ld1         {v0.16b, v1.16b}, [%[src_argb]], #32 \n"
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop.

      "movi        v16.4s, #0                    \n"
      "movi        v17.4s, #0                    \n"
      "movi        v18.4s, #0                    \n"
      "movi        v19.4s, #0                    \n"
      "movi        v20.4s, #0                    \n"
      "movi        v21.4s, #0                    \n"
      "movi        v22.4s, #0                    \n"
      "movi        v23.4s, #0                    \n"

      "prfm        pldl1keep, [%[src_argb], 448] \n"

      "sudot       v16.4s, v31.16b, v0.4b[0]     \n"
      "sudot       v17.4s, v31.16b, v0.4b[1]     \n"
      "sudot       v18.4s, v31.16b, v0.4b[2]     \n"
      "sudot       v19.4s, v31.16b, v0.4b[3]     \n"
      "sudot       v20.4s, v31.16b, v1.4b[0]     \n"
      "sudot       v21.4s, v31.16b, v1.4b[1]     \n"
      "sudot       v22.4s, v31.16b, v1.4b[2]     \n"
      "sudot       v23.4s, v31.16b, v1.4b[3]     \n"

      "shrn        v16.4h, v16.4s, #6            \n"
      "shrn        v18.4h, v18.4s, #6            \n"
      "shrn        v20.4h, v20.4s, #6            \n"
      "shrn        v22.4h, v22.4s, #6            \n"
      "shrn2       v16.8h, v17.4s, #6            \n"
      "shrn2       v18.8h, v19.4s, #6            \n"
      "shrn2       v20.8h, v21.4s, #6            \n"
      "shrn2       v22.8h, v23.4s, #6            \n"

      "uqxtn       v16.8b, v16.8h                \n"
      "uqxtn       v18.8b, v18.8h                \n"
      "uqxtn       v20.8b, v20.8h                \n"
      "uqxtn       v22.8b, v22.8h                \n"

      "stp         d16, d18, [%[dst_argb]], #16  \n"
      "stp         d20, d22, [%[dst_argb]], #16  \n"
      "b.gt        1b                            \n"
      : [src_argb] "+r"(src_argb),      // %[src_argb]
        [dst_argb] "+r"(dst_argb),      // %[dst_argb]
        [width] "+r"(width)             // %[width]
      : [matrix_argb] "r"(matrix_argb)  // %[matrix_argb]
      : "cc", "memory", "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21",
        "v22", "v23", "v31");
}

// Multiply 2 rows of ARGB pixels together, 8 pixels at a time.
void ARGBMultiplyRow_NEON(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      // 8 pixel loop.
      "1:          \n"
      "ld1         {v0.8b,v1.8b,v2.8b,v3.8b}, [%0], #32 \n"  // load 8 ARGB
      "ld1         {v4.8b,v5.8b,v6.8b,v7.8b}, [%1], #32 \n"  // load 8 more
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "umull       v0.8h, v0.8b, v4.8b           \n"  // multiply B
      "prfm        pldl1keep, [%0, 448]          \n"
      "umull       v1.8h, v1.8b, v5.8b           \n"  // multiply G
      "prfm        pldl1keep, [%1, 448]          \n"
      "umull       v2.8h, v2.8b, v6.8b           \n"  // multiply R
      "umull       v3.8h, v3.8b, v7.8b           \n"  // multiply A
      "rshrn       v0.8b, v0.8h, #8              \n"  // 16 bit to 8 bit B
      "rshrn       v1.8b, v1.8h, #8              \n"  // 16 bit to 8 bit G
      "rshrn       v2.8b, v2.8h, #8              \n"  // 16 bit to 8 bit R
      "rshrn       v3.8b, v3.8h, #8              \n"  // 16 bit to 8 bit A
      "st1         {v0.8b,v1.8b,v2.8b,v3.8b}, [%2], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

// Add 2 rows of ARGB pixels together, 8 pixels at a time.
void ARGBAddRow_NEON(const uint8_t* src_argb,
                     const uint8_t* src_argb1,
                     uint8_t* dst_argb,
                     int width) {
  asm volatile(
      // 8 pixel loop.
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"  // load 8 ARGB
      "ldp         q4, q5, [%1], #32             \n"  // load 8 more
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "uqadd       v0.16b, v0.16b, v4.16b        \n"
      "uqadd       v1.16b, v1.16b, v5.16b        \n"
      "stp         q0, q1, [%2], #32             \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

// Subtract 2 rows of ARGB pixels, 8 pixels at a time.
void ARGBSubtractRow_NEON(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      // 8 pixel loop.
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"  // load 8 ARGB
      "ldp         q4, q5, [%1], #32             \n"  // load 8 more
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "uqsub       v0.16b, v0.16b, v4.16b        \n"
      "uqsub       v1.16b, v1.16b, v5.16b        \n"
      "stp         q0, q1, [%2], #32             \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

// Adds Sobel X and Sobel Y and stores Sobel into ARGB.
// A = 255
// R = Sobel
// G = Sobel
// B = Sobel
void SobelRow_NEON(const uint8_t* src_sobelx,
                   const uint8_t* src_sobely,
                   uint8_t* dst_argb,
                   int width) {
  asm volatile(
      "movi        v3.8b, #255                   \n"  // alpha
      // 8 pixel loop.
      "1:          \n"
      "ld1         {v0.8b}, [%0], #8             \n"  // load 8 sobelx.
      "ld1         {v1.8b}, [%1], #8             \n"  // load 8 sobely.
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "uqadd       v0.8b, v0.8b, v1.8b           \n"  // add
      "prfm        pldl1keep, [%0, 448]          \n"
      "mov         v1.8b, v0.8b                  \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "mov         v2.8b, v0.8b                  \n"
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%2], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_argb),    // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

// Adds Sobel X and Sobel Y and stores Sobel into plane.
void SobelToPlaneRow_NEON(const uint8_t* src_sobelx,
                          const uint8_t* src_sobely,
                          uint8_t* dst_y,
                          int width) {
  asm volatile(
      // 16 pixel loop.
      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"  // load 16 sobelx.
      "ld1         {v1.16b}, [%1], #16           \n"  // load 16 sobely.
      "subs        %w3, %w3, #16                 \n"  // 16 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uqadd       v0.16b, v0.16b, v1.16b        \n"  // add
      "prfm        pldl1keep, [%1, 448]          \n"
      "st1         {v0.16b}, [%2], #16           \n"  // store 16 pixels.
      "b.gt        1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_y),       // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "v0", "v1");
}

// Mixes Sobel X, Sobel Y and Sobel into ARGB.
// A = 255
// R = Sobel X
// G = Sobel
// B = Sobel Y
void SobelXYRow_NEON(const uint8_t* src_sobelx,
                     const uint8_t* src_sobely,
                     uint8_t* dst_argb,
                     int width) {
  asm volatile(
      "movi        v3.8b, #255                   \n"  // alpha
      // 8 pixel loop.
      "1:          \n"
      "ld1         {v2.8b}, [%0], #8             \n"  // load 8 sobelx.
      "ld1         {v0.8b}, [%1], #8             \n"  // load 8 sobely.
      "subs        %w3, %w3, #8                  \n"  // 8 processed per loop.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uqadd       v1.8b, v0.8b, v2.8b           \n"  // add
      "prfm        pldl1keep, [%1, 448]          \n"
      "st4         {v0.8b,v1.8b,v2.8b,v3.8b}, [%2], #32 \n"  // store 8 ARGB
      "b.gt        1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_argb),    // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

// SobelX as a matrix is
// -1  0  1
// -2  0  2
// -1  0  1
void SobelXRow_NEON(const uint8_t* src_y0,
                    const uint8_t* src_y1,
                    const uint8_t* src_y2,
                    uint8_t* dst_sobelx,
                    int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.8b}, [%0],%5              \n"  // top
      "ld1         {v1.8b}, [%0],%6              \n"
      "subs        %w4, %w4, #8                  \n"  // 8 pixels
      "usubl       v0.8h, v0.8b, v1.8b           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "ld1         {v2.8b}, [%1],%5              \n"  // center * 2
      "ld1         {v3.8b}, [%1],%6              \n"
      "usubl       v1.8h, v2.8b, v3.8b           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "add         v0.8h, v0.8h, v1.8h           \n"
      "add         v0.8h, v0.8h, v1.8h           \n"
      "ld1         {v2.8b}, [%2],%5              \n"  // bottom
      "ld1         {v3.8b}, [%2],%6              \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "usubl       v1.8h, v2.8b, v3.8b           \n"
      "add         v0.8h, v0.8h, v1.8h           \n"
      "abs         v0.8h, v0.8h                  \n"
      "uqxtn       v0.8b, v0.8h                  \n"
      "st1         {v0.8b}, [%3], #8             \n"  // store 8 sobelx
      "b.gt        1b                            \n"
      : "+r"(src_y0),                           // %0
        "+r"(src_y1),                           // %1
        "+r"(src_y2),                           // %2
        "+r"(dst_sobelx),                       // %3
        "+r"(width)                             // %4
      : "r"(2LL),                               // %5
        "r"(6LL)                                // %6
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

// SobelY as a matrix is
// -1 -2 -1
//  0  0  0
//  1  2  1
void SobelYRow_NEON(const uint8_t* src_y0,
                    const uint8_t* src_y1,
                    uint8_t* dst_sobely,
                    int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.8b}, [%0],%4              \n"  // left
      "ld1         {v1.8b}, [%1],%4              \n"
      "subs        %w3, %w3, #8                  \n"  // 8 pixels
      "usubl       v0.8h, v0.8b, v1.8b           \n"
      "ld1         {v2.8b}, [%0],%4              \n"  // center * 2
      "ld1         {v3.8b}, [%1],%4              \n"
      "usubl       v1.8h, v2.8b, v3.8b           \n"
      "add         v0.8h, v0.8h, v1.8h           \n"
      "add         v0.8h, v0.8h, v1.8h           \n"
      "ld1         {v2.8b}, [%0],%5              \n"  // right
      "ld1         {v3.8b}, [%1],%5              \n"
      "usubl       v1.8h, v2.8b, v3.8b           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "add         v0.8h, v0.8h, v1.8h           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "abs         v0.8h, v0.8h                  \n"
      "uqxtn       v0.8b, v0.8h                  \n"
      "st1         {v0.8b}, [%2], #8             \n"  // store 8 sobely
      "b.gt        1b                            \n"
      : "+r"(src_y0),                           // %0
        "+r"(src_y1),                           // %1
        "+r"(dst_sobely),                       // %2
        "+r"(width)                             // %3
      : "r"(1LL),                               // %4
        "r"(6LL)                                // %5
      : "cc", "memory", "v0", "v1", "v2", "v3"  // Clobber List
  );
}

void HalfFloatRow_NEON(const uint16_t* src,
                       uint16_t* dst,
                       float scale,
                       int width) {
  asm volatile(
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"  // load 16 shorts
      "subs        %w2, %w2, #16                 \n"  // 16 pixels per loop
      "uxtl        v2.4s, v0.4h                  \n"
      "uxtl        v4.4s, v1.4h                  \n"
      "uxtl2       v3.4s, v0.8h                  \n"
      "uxtl2       v5.4s, v1.8h                  \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "scvtf       v2.4s, v2.4s                  \n"
      "scvtf       v4.4s, v4.4s                  \n"
      "scvtf       v3.4s, v3.4s                  \n"
      "scvtf       v5.4s, v5.4s                  \n"
      "fmul        v2.4s, v2.4s, %3.s[0]         \n"  // adjust exponent
      "fmul        v4.4s, v4.4s, %3.s[0]         \n"
      "fmul        v3.4s, v3.4s, %3.s[0]         \n"
      "fmul        v5.4s, v5.4s, %3.s[0]         \n"
      "uqshrn      v0.4h, v2.4s, #13             \n"  // isolate halffloat
      "uqshrn      v1.4h, v4.4s, #13             \n"
      "uqshrn2     v0.8h, v3.4s, #13             \n"
      "uqshrn2     v1.8h, v5.4s, #13             \n"
      "stp         q0, q1, [%1], #32             \n"  // store 16 fp16
      "b.gt        1b                            \n"
      : "+r"(src),                      // %0
        "+r"(dst),                      // %1
        "+r"(width)                     // %2
      : "w"(scale * 1.9259299444e-34f)  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
}

void ByteToFloatRow_NEON(const uint8_t* src,
                         float* dst,
                         float scale,
                         int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v1.8b}, [%0], #8             \n"  // load 8 bytes
      "subs        %w2, %w2, #8                  \n"  // 8 pixels per loop
      "uxtl        v1.8h, v1.8b                  \n"  // 8 shorts
      "prfm        pldl1keep, [%0, 448]          \n"
      "uxtl        v2.4s, v1.4h                  \n"  // 8 ints
      "uxtl2       v3.4s, v1.8h                  \n"
      "scvtf       v2.4s, v2.4s                  \n"  // 8 floats
      "scvtf       v3.4s, v3.4s                  \n"
      "fmul        v2.4s, v2.4s, %3.s[0]         \n"  // scale
      "fmul        v3.4s, v3.4s, %3.s[0]         \n"
      "st1         {v2.16b, v3.16b}, [%1], #32   \n"  // store 8 floats
      "b.gt        1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      : "w"(scale)   // %3
      : "cc", "memory", "v1", "v2", "v3");
}

// Convert FP16 Half Floats to FP32 Floats
void ConvertFP16ToFP32Row_NEON(const uint16_t* src,  // fp16
                               float* dst,
                               int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v1.8h}, [%0], #16            \n"  // load 8 halffloats
      "subs        %w2, %w2, #8                  \n"  // 8 floats per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "fcvtl       v2.4s, v1.4h                  \n"  // 8 floats
      "fcvtl2      v3.4s, v1.8h                  \n"
      "stp         q2, q3, [%1], #32             \n"  // store 8 floats
      "b.gt        1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "cc", "memory", "v1", "v2", "v3");
}

// Convert FP16 Half Floats to FP32 Floats
// Read a column and write a row
void ConvertFP16ToFP32Column_NEON(const uint16_t* src,  // fp16
                                  int src_stride,       // stride in elements
                                  float* dst,
                                  int width) {
  asm volatile(
      "cmp         %w2, #8                       \n"  // Is there 8 rows?
      "b.lo        2f                            \n"
      "1:          \n"
      "ld1         {v0.h}[0], [%0], %3           \n"  // load 8 halffloats
      "ld1         {v0.h}[1], [%0], %3           \n"
      "ld1         {v0.h}[2], [%0], %3           \n"
      "ld1         {v0.h}[3], [%0], %3           \n"
      "ld1         {v1.h}[0], [%0], %3           \n"
      "ld1         {v1.h}[1], [%0], %3           \n"
      "ld1         {v1.h}[2], [%0], %3           \n"
      "ld1         {v1.h}[3], [%0], %3           \n"
      "subs        %w2, %w2, #8                  \n"  // 8 rows per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "fcvtl       v2.4s, v0.4h                  \n"  // 4 floats
      "fcvtl       v3.4s, v1.4h                  \n"  // 4 more floats
      "stp         q2, q3, [%1], #32             \n"  // store 8 floats
      "b.gt        1b                            \n"
      "cmp         %w2, #1                       \n"  // Is there 1 value?
      "b.lo        3f                            \n"
      "2:          \n"
      "ld1         {v1.h}[0], [%0], %3           \n"  // load 1 halffloats
      "subs        %w2, %w2, #1                  \n"  // 1 floats per loop
      "fcvtl       v2.4s, v1.4h                  \n"  // 1 floats
      "str         s2, [%1], #4                  \n"  // store 1 floats
      "b.gt        2b                            \n"
      "3:          \n"
      : "+r"(src),                        // %0
        "+r"(dst),                        // %1
        "+r"(width)                       // %2
      : "r"((ptrdiff_t)(src_stride * 2))  // %3
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

// Convert FP32 Floats to FP16 Half Floats
void ConvertFP32ToFP16Row_NEON(const float* src,
                               uint16_t* dst,  // fp16
                               int width) {
  asm volatile(
      "1:          \n"
      "ldp         q2, q3, [%0], #32             \n"  // load 8 floats
      "subs        %w2, %w2, #8                  \n"  // 8 floats per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "fcvtn       v1.4h, v2.4s                  \n"  // 8 fp16 halffloats
      "fcvtn2      v1.8h, v3.4s                  \n"
      "str         q1, [%1], #16                 \n"  // store 8 fp16 halffloats
      "b.gt        1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      :
      : "cc", "memory", "v1", "v2", "v3");
}

float ScaleMaxSamples_NEON(const float* src,
                           float* dst,
                           float scale,
                           int width) {
  float fmax;
  asm volatile(
      "movi        v5.4s, #0                     \n"  // max
      "movi        v6.4s, #0                     \n"

      "1:          \n"
      "ld1         {v1.4s, v2.4s}, [%0], #32     \n"  // load 8 samples
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop
      "fmul        v3.4s, v1.4s, %4.s[0]         \n"  // scale
      "prfm        pldl1keep, [%0, 448]          \n"
      "fmul        v4.4s, v2.4s, %4.s[0]         \n"  // scale
      "fmax        v5.4s, v5.4s, v1.4s           \n"  // max
      "fmax        v6.4s, v6.4s, v2.4s           \n"
      "st1         {v3.4s, v4.4s}, [%1], #32     \n"  // store 8 samples
      "b.gt        1b                            \n"
      "fmax        v5.4s, v5.4s, v6.4s           \n"  // max
      "fmaxv       %s3, v5.4s                    \n"  // signed max acculator
      : "+r"(src),                                    // %0
        "+r"(dst),                                    // %1
        "+r"(width),                                  // %2
        "=w"(fmax)                                    // %3
      : "w"(scale)                                    // %4
      : "cc", "memory", "v1", "v2", "v3", "v4", "v5", "v6");
  return fmax;
}

float ScaleSumSamples_NEON(const float* src,
                           float* dst,
                           float scale,
                           int width) {
  float fsum;
  asm volatile(
      "movi        v5.4s, #0                     \n"  // max
      "movi        v6.4s, #0                     \n"  // max

      "1:          \n"
      "ld1         {v1.4s, v2.4s}, [%0], #32     \n"  // load 8 samples
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop
      "fmul        v3.4s, v1.4s, %4.s[0]         \n"  // scale
      "prfm        pldl1keep, [%0, 448]          \n"
      "fmul        v4.4s, v2.4s, %4.s[0]         \n"
      "fmla        v5.4s, v1.4s, v1.4s           \n"  // sum of squares
      "fmla        v6.4s, v2.4s, v2.4s           \n"
      "st1         {v3.4s, v4.4s}, [%1], #32     \n"  // store 8 samples
      "b.gt        1b                            \n"
      "faddp       v5.4s, v5.4s, v6.4s           \n"
      "faddp       v5.4s, v5.4s, v5.4s           \n"
      "faddp       %3.4s, v5.4s, v5.4s           \n"  // sum
      : "+r"(src),                                    // %0
        "+r"(dst),                                    // %1
        "+r"(width),                                  // %2
        "=w"(fsum)                                    // %3
      : "w"(scale)                                    // %4
      : "cc", "memory", "v1", "v2", "v3", "v4", "v5", "v6");
  return fsum;
}

void ScaleSamples_NEON(const float* src, float* dst, float scale, int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v1.4s, v2.4s}, [%0], #32     \n"  // load 8 samples
      "subs        %w2, %w2, #8                  \n"  // 8 processed per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "fmul        v1.4s, v1.4s, %3.s[0]         \n"  // scale
      "fmul        v2.4s, v2.4s, %3.s[0]         \n"  // scale
      "st1         {v1.4s, v2.4s}, [%1], #32     \n"  // store 8 samples
      "b.gt        1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      : "w"(scale)   // %3
      : "cc", "memory", "v1", "v2");
}

// filter 5 rows with 1, 4, 6, 4, 1 coefficients to produce 1 row.
void GaussCol_NEON(const uint16_t* src0,
                   const uint16_t* src1,
                   const uint16_t* src2,
                   const uint16_t* src3,
                   const uint16_t* src4,
                   uint32_t* dst,
                   int width) {
  asm volatile(
      "movi        v6.8h, #4                     \n"  // constant 4
      "movi        v7.8h, #6                     \n"  // constant 6

      "1:          \n"
      "ld1         {v1.8h}, [%0], #16            \n"  // load 8 samples, 5 rows
      "ld1         {v2.8h}, [%4], #16            \n"
      "subs        %w6, %w6, #8                  \n"  // 8 processed per loop
      "uaddl       v0.4s, v1.4h, v2.4h           \n"  // * 1
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddl2      v1.4s, v1.8h, v2.8h           \n"  // * 1
      "ld1         {v2.8h}, [%1], #16            \n"
      "umlal       v0.4s, v2.4h, v6.4h           \n"  // * 4
      "prfm        pldl1keep, [%1, 448]          \n"
      "umlal2      v1.4s, v2.8h, v6.8h           \n"  // * 4
      "ld1         {v2.8h}, [%2], #16            \n"
      "umlal       v0.4s, v2.4h, v7.4h           \n"  // * 6
      "prfm        pldl1keep, [%2, 448]          \n"
      "umlal2      v1.4s, v2.8h, v7.8h           \n"  // * 6
      "ld1         {v2.8h}, [%3], #16            \n"
      "umlal       v0.4s, v2.4h, v6.4h           \n"  // * 4
      "prfm        pldl1keep, [%3, 448]          \n"
      "umlal2      v1.4s, v2.8h, v6.8h           \n"  // * 4
      "st1         {v0.4s,v1.4s}, [%5], #32      \n"  // store 8 samples
      "prfm        pldl1keep, [%4, 448]          \n"
      "b.gt        1b                            \n"
      : "+r"(src0),  // %0
        "+r"(src1),  // %1
        "+r"(src2),  // %2
        "+r"(src3),  // %3
        "+r"(src4),  // %4
        "+r"(dst),   // %5
        "+r"(width)  // %6
      :
      : "cc", "memory", "v0", "v1", "v2", "v6", "v7");
}

// filter 5 rows with 1, 4, 6, 4, 1 coefficients to produce 1 row.
void GaussRow_NEON(const uint32_t* src, uint16_t* dst, int width) {
  const uint32_t* src1 = src + 1;
  const uint32_t* src2 = src + 2;
  const uint32_t* src3 = src + 3;
  asm volatile(
      "movi        v6.4s, #4                     \n"  // constant 4
      "movi        v7.4s, #6                     \n"  // constant 6

      "1:          \n"
      "ld1         {v0.4s,v1.4s,v2.4s}, [%0], %6 \n"  // load 12 source samples
      "subs        %w5, %w5, #8                  \n"  // 8 processed per loop
      "add         v0.4s, v0.4s, v1.4s           \n"  // * 1
      "add         v1.4s, v1.4s, v2.4s           \n"  // * 1
      "ld1         {v2.4s,v3.4s}, [%2], #32      \n"
      "mla         v0.4s, v2.4s, v7.4s           \n"  // * 6
      "mla         v1.4s, v3.4s, v7.4s           \n"  // * 6
      "ld1         {v2.4s,v3.4s}, [%1], #32      \n"
      "ld1         {v4.4s,v5.4s}, [%3], #32      \n"
      "add         v2.4s, v2.4s, v4.4s           \n"  // add rows for * 4
      "add         v3.4s, v3.4s, v5.4s           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "mla         v0.4s, v2.4s, v6.4s           \n"  // * 4
      "mla         v1.4s, v3.4s, v6.4s           \n"  // * 4
      "uqrshrn     v0.4h, v0.4s, #8              \n"  // round and pack
      "uqrshrn2    v0.8h, v1.4s, #8              \n"
      "st1         {v0.8h}, [%4], #16            \n"  // store 8 samples
      "b.gt        1b                            \n"
      : "+r"(src),   // %0
        "+r"(src1),  // %1
        "+r"(src2),  // %2
        "+r"(src3),  // %3
        "+r"(dst),   // %4
        "+r"(width)  // %5
      : "r"(32LL)    // %6
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

static const vecf32 kGaussCoefficients = {4.0f, 6.0f, 1.0f / 256.0f, 0.0f};

// filter 5 rows with 1, 4, 6, 4, 1 coefficients to produce 1 row.
void GaussCol_F32_NEON(const float* src0,
                       const float* src1,
                       const float* src2,
                       const float* src3,
                       const float* src4,
                       float* dst,
                       int width) {
  asm volatile(
      "ld2r        {v6.4s, v7.4s}, [%7]          \n"  // constants 4 and 6

      "1:          \n"
      "ld1         {v0.4s, v1.4s}, [%0], #32     \n"  // load 8 samples, 5 rows
      "ld1         {v2.4s, v3.4s}, [%1], #32     \n"
      "subs        %w6, %w6, #8                  \n"  // 8 processed per loop
      "fmla        v0.4s, v2.4s, v6.4s           \n"  // * 4
      "ld1         {v4.4s, v5.4s}, [%2], #32     \n"
      "fmla        v1.4s, v3.4s, v6.4s           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "fmla        v0.4s, v4.4s, v7.4s           \n"  // * 6
      "ld1         {v2.4s, v3.4s}, [%3], #32     \n"
      "fmla        v1.4s, v5.4s, v7.4s           \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "fmla        v0.4s, v2.4s, v6.4s           \n"  // * 4
      "ld1         {v4.4s, v5.4s}, [%4], #32     \n"
      "fmla        v1.4s, v3.4s, v6.4s           \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "fadd        v0.4s, v0.4s, v4.4s           \n"  // * 1
      "prfm        pldl1keep, [%3, 448]          \n"
      "fadd        v1.4s, v1.4s, v5.4s           \n"
      "prfm        pldl1keep, [%4, 448]          \n"
      "st1         {v0.4s, v1.4s}, [%5], #32     \n"  // store 8 samples
      "b.gt        1b                            \n"
      : "+r"(src0),               // %0
        "+r"(src1),               // %1
        "+r"(src2),               // %2
        "+r"(src3),               // %3
        "+r"(src4),               // %4
        "+r"(dst),                // %5
        "+r"(width)               // %6
      : "r"(&kGaussCoefficients)  // %7
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

// filter 5 rows with 1, 4, 6, 4, 1 coefficients to produce 1 row.
void GaussRow_F32_NEON(const float* src, float* dst, int width) {
  asm volatile(
      "ld3r        {v6.4s, v7.4s, v8.4s}, [%3]   \n"  // constants 4, 6, 1/256

      "1:          \n"
      "ld1         {v0.4s, v1.4s, v2.4s}, [%0], %4 \n"  // load 12 samples, 5
                                                        // rows
      "subs        %w2, %w2, #8                  \n"    // 8 processed per loop
      "fadd        v0.4s, v0.4s, v1.4s           \n"    // * 1
      "ld1         {v4.4s, v5.4s}, [%0], %5      \n"
      "fadd        v1.4s, v1.4s, v2.4s           \n"
      "fmla        v0.4s, v4.4s, v7.4s           \n"  // * 6
      "ld1         {v2.4s, v3.4s}, [%0], %4      \n"
      "fmla        v1.4s, v5.4s, v7.4s           \n"
      "ld1         {v4.4s, v5.4s}, [%0], %6      \n"
      "fadd        v2.4s, v2.4s, v4.4s           \n"
      "fadd        v3.4s, v3.4s, v5.4s           \n"
      "fmla        v0.4s, v2.4s, v6.4s           \n"  // * 4
      "fmla        v1.4s, v3.4s, v6.4s           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "fmul        v0.4s, v0.4s, v8.4s           \n"  // / 256
      "fmul        v1.4s, v1.4s, v8.4s           \n"
      "st1         {v0.4s, v1.4s}, [%1], #32     \n"  // store 8 samples
      "b.gt        1b                            \n"
      : "+r"(src),                 // %0
        "+r"(dst),                 // %1
        "+r"(width)                // %2
      : "r"(&kGaussCoefficients),  // %3
        "r"(8LL),                  // %4
        "r"(-4LL),                 // %5
        "r"(20LL)                  // %6
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8");
}

#if defined(LIBYUV_USE_ST3)
// Convert biplanar NV21 to packed YUV24
void NV21ToYUV24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_yuv24,
                         int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v2.16b}, [%0], #16           \n"  // load 16 Y values
      "ld2         {v0.8b, v1.8b}, [%1], #16     \n"  // load 8 VU values
      "subs        %w3, %w3, #16                 \n"  // 16 pixels per loop
      "zip1        v0.16b, v0.16b, v0.16b        \n"  // replicate V values
      "prfm        pldl1keep, [%0, 448]          \n"
      "zip1        v1.16b, v1.16b, v1.16b        \n"  // replicate U values
      "prfm        pldl1keep, [%1, 448]          \n"
      "st3         {v0.16b,v1.16b,v2.16b}, [%2], #48 \n"  // store 16 YUV pixels
      "b.gt        1b                            \n"
      : "+r"(src_y),      // %0
        "+r"(src_vu),     // %1
        "+r"(dst_yuv24),  // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "v0", "v1", "v2");
}
#else
static const uvec8 kYUV24Shuffle[3] = {
    {16, 17, 0, 16, 17, 1, 18, 19, 2, 18, 19, 3, 20, 21, 4, 20},
    {21, 5, 22, 23, 6, 22, 23, 7, 24, 25, 8, 24, 25, 9, 26, 27},
    {10, 26, 27, 11, 28, 29, 12, 28, 29, 13, 30, 31, 14, 30, 31, 15}};

// Convert biplanar NV21 to packed YUV24
// NV21 has VU in memory for chroma.
// YUV24 is VUY in memory
void NV21ToYUV24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_yuv24,
                         int width) {
  asm volatile(
      "ld1         {v5.16b,v6.16b,v7.16b}, [%4]  \n"  // 3 shuffler constants
      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"    // load 16 Y values
      "ld1         {v1.16b}, [%1], #16           \n"    // load 8 VU values
      "subs        %w3, %w3, #16                 \n"    // 16 pixels per loop
      "tbl         v2.16b, {v0.16b,v1.16b}, v5.16b \n"  // weave into YUV24
      "prfm        pldl1keep, [%0, 448]          \n"
      "tbl         v3.16b, {v0.16b,v1.16b}, v6.16b \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "tbl         v4.16b, {v0.16b,v1.16b}, v7.16b \n"
      "st1         {v2.16b,v3.16b,v4.16b}, [%2], #48 \n"  // store 16 YUV pixels
      "b.gt        1b                            \n"
      : "+r"(src_y),            // %0
        "+r"(src_vu),           // %1
        "+r"(dst_yuv24),        // %2
        "+r"(width)             // %3
      : "r"(&kYUV24Shuffle[0])  // %4
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}
#endif  // LIBYUV_USE_ST3

// Note ST2 8b version is faster than zip+ST1

// AYUV is VUYA in memory.  UV for NV12 is UV order in memory.
void AYUVToUVRow_NEON(const uint8_t* src_ayuv,
                      int src_stride_ayuv,
                      uint8_t* dst_uv,
                      int width) {
  const uint8_t* src_ayuv_1 = src_ayuv + src_stride_ayuv;
  asm volatile(

      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 ayuv
      "subs        %w3, %w3, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // V 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // U 16 bytes -> 8 shorts.
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load next 16
      "uadalp      v0.8h, v4.16b                 \n"  // V 16 bytes -> 8 shorts.
      "uadalp      v1.8h, v5.16b                 \n"  // U 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uqrshrn     v3.8b, v0.8h, #2              \n"  // 2x2 average
      "uqrshrn     v2.8b, v1.8h, #2              \n"
      "st2         {v2.8b,v3.8b}, [%2], #16      \n"  // store 8 pixels UV.
      "b.gt        1b                            \n"
      : "+r"(src_ayuv),    // %0
        "+r"(src_ayuv_1),  // %1
        "+r"(dst_uv),      // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

void AYUVToVURow_NEON(const uint8_t* src_ayuv,
                      int src_stride_ayuv,
                      uint8_t* dst_vu,
                      int width) {
  const uint8_t* src_ayuv_1 = src_ayuv + src_stride_ayuv;
  asm volatile(

      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16 ayuv
      "subs        %w3, %w3, #16                 \n"  // 16 processed per loop.
      "uaddlp      v0.8h, v0.16b                 \n"  // V 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"  // U 16 bytes -> 8 shorts.
      "ld4         {v4.16b,v5.16b,v6.16b,v7.16b}, [%1], #64 \n"  // load next 16
      "uadalp      v0.8h, v4.16b                 \n"  // V 16 bytes -> 8 shorts.
      "uadalp      v1.8h, v5.16b                 \n"  // U 16 bytes -> 8 shorts.
      "prfm        pldl1keep, [%1, 448]          \n"
      "uqrshrn     v0.8b, v0.8h, #2              \n"  // 2x2 average
      "uqrshrn     v1.8b, v1.8h, #2              \n"
      "st2         {v0.8b,v1.8b}, [%2], #16      \n"  // store 8 pixels VU.
      "b.gt        1b                            \n"
      : "+r"(src_ayuv),    // %0
        "+r"(src_ayuv_1),  // %1
        "+r"(dst_vu),      // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

// Copy row of AYUV Y's into Y
void AYUVToYRow_NEON(const uint8_t* src_ayuv, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "ld4         {v0.16b,v1.16b,v2.16b,v3.16b}, [%0], #64 \n"  // load 16
      "subs        %w2, %w2, #16                 \n"  // 16 pixels per loop
      "prfm        pldl1keep, [%0, 448]          \n"
      "st1         {v2.16b}, [%1], #16           \n"  // store 16 Y pixels
      "b.gt        1b                            \n"
      : "+r"(src_ayuv),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

// Convert UV plane of NV12 to VU of NV21.
void SwapUVRow_NEON(const uint8_t* src_uv, uint8_t* dst_vu, int width) {
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], 16            \n"  // load 16 UV values
      "ld1         {v1.16b}, [%0], 16            \n"
      "subs        %w2, %w2, #16                 \n"  // 16 pixels per loop
      "rev16       v0.16b, v0.16b                \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "rev16       v1.16b, v1.16b                \n"
      "stp         q0, q1, [%1], 32              \n"  // store 16 VU pixels
      "b.gt        1b                            \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_vu),  // %1
        "+r"(width)    // %2
      :
      : "cc", "memory", "v0", "v1");
}

void HalfMergeUVRow_NEON(const uint8_t* src_u,
                         int src_stride_u,
                         const uint8_t* src_v,
                         int src_stride_v,
                         uint8_t* dst_uv,
                         int width) {
  const uint8_t* src_u_1 = src_u + src_stride_u;
  const uint8_t* src_v_1 = src_v + src_stride_v;
  asm volatile(
      "1:          \n"
      "ld1         {v0.16b}, [%0], #16           \n"  // load 16 U values
      "ld1         {v1.16b}, [%2], #16           \n"  // load 16 V values
      "ld1         {v2.16b}, [%1], #16           \n"
      "ld1         {v3.16b}, [%3], #16           \n"
      "subs        %w5, %w5, #16                 \n"  // 16 src pixels per loop
      "uaddlp      v0.8h, v0.16b                 \n"  // half size
      "prfm        pldl1keep, [%0, 448]          \n"
      "uaddlp      v1.8h, v1.16b                 \n"
      "prfm        pldl1keep, [%2, 448]          \n"
      "uadalp      v0.8h, v2.16b                 \n"
      "prfm        pldl1keep, [%1, 448]          \n"
      "uadalp      v1.8h, v3.16b                 \n"
      "prfm        pldl1keep, [%3, 448]          \n"
      "uqrshrn     v0.8b, v0.8h, #2              \n"
      "uqrshrn     v1.8b, v1.8h, #2              \n"
      "st2         {v0.8b, v1.8b}, [%4], #16     \n"  // store 8 UV pixels
      "b.gt        1b                            \n"
      : "+r"(src_u),    // %0
        "+r"(src_u_1),  // %1
        "+r"(src_v),    // %2
        "+r"(src_v_1),  // %3
        "+r"(dst_uv),   // %4
        "+r"(width)     // %5
      :
      : "cc", "memory", "v0", "v1", "v2", "v3");
}

void SplitUVRow_16_NEON(const uint16_t* src_uv,
                        uint16_t* dst_u,
                        uint16_t* dst_v,
                        int depth,
                        int width) {
  int shift = depth - 16;  // Negative for right shift.
  asm volatile(
      "dup         v2.8h, %w4                    \n"
      "1:          \n"
      "ld2         {v0.8h, v1.8h}, [%0], #32     \n"  // load 8 UV
      "subs        %w3, %w3, #8                  \n"  // 8 src pixels per loop
      "ushl        v0.8h, v0.8h, v2.8h           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "ushl        v1.8h, v1.8h, v2.8h           \n"
      "st1         {v0.8h}, [%1], #16            \n"  // store 8 U pixels
      "st1         {v1.8h}, [%2], #16            \n"  // store 8 V pixels
      "b.gt        1b                            \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_u),   // %1
        "+r"(dst_v),   // %2
        "+r"(width)    // %3
      : "r"(shift)     // %4
      : "cc", "memory", "v0", "v1", "v2");
}

void MultiplyRow_16_NEON(const uint16_t* src_y,
                         uint16_t* dst_y,
                         int scale,
                         int width) {
  asm volatile(
      "dup         v2.8h, %w3                    \n"
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"
      "subs        %w2, %w2, #16                 \n"  // 16 src pixels per loop
      "mul         v0.8h, v0.8h, v2.8h           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "mul         v1.8h, v1.8h, v2.8h           \n"
      "stp         q0, q1, [%1], #32             \n"  // store 16 pixels
      "b.gt        1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "cc", "memory", "v0", "v1", "v2");
}

void DivideRow_16_NEON(const uint16_t* src_y,
                       uint16_t* dst_y,
                       int scale,
                       int width) {
  asm volatile(
      "dup         v4.8h, %w3                    \n"
      "1:          \n"
      "ldp         q2, q3, [%0], #32             \n"
      "subs        %w2, %w2, #16                 \n"  // 16 src pixels per loop
      "umull       v0.4s, v2.4h, v4.4h           \n"
      "umull2      v1.4s, v2.8h, v4.8h           \n"
      "umull       v2.4s, v3.4h, v4.4h           \n"
      "umull2      v3.4s, v3.8h, v4.8h           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "uzp2        v0.8h, v0.8h, v1.8h           \n"
      "uzp2        v1.8h, v2.8h, v3.8h           \n"
      "stp         q0, q1, [%1], #32             \n"  // store 16 pixels
      "b.gt        1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits = shr 1
// 16384 = 10 bits = shr 2
// 4096 = 12 bits = shr 4
// 256 = 16 bits = shr 8
void Convert16To8Row_NEON(const uint16_t* src_y,
                          uint8_t* dst_y,
                          int scale,
                          int width) {
  // 15 - clz(scale), + 8 to shift result into the high half of the lane to
  // saturate, then we can just use UZP2 to narrow rather than a pair of
  // saturating narrow instructions.
  int shift = 23 - __builtin_clz((int32_t)scale);
  asm volatile(
      "dup         v2.8h, %w3                    \n"
      "1:          \n"
      "ldp         q0, q1, [%0], #32             \n"
      "subs        %w2, %w2, #16                 \n"  // 16 src pixels per loop
      "uqshl       v0.8h, v0.8h, v2.8h           \n"
      "uqshl       v1.8h, v1.8h, v2.8h           \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "uzp2        v0.16b, v0.16b, v1.16b        \n"
      "str         q0, [%1], #16                 \n"  // store 16 pixels
      "b.gt        1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(shift)    // %3
      : "cc", "memory", "v0", "v1", "v2");
}

// Use scale to convert J420 to I420
// scale parameter is 8.8 fixed point but limited to 0 to 255
// Function is based on DivideRow, but adds a bias
// Does not clamp
void Convert8To8Row_NEON(const uint8_t* src_y,
                         uint8_t* dst_y,
                         int scale,
                         int bias,
                         int width) {
  asm volatile(
      "dup         v4.16b, %w3                   \n"  // scale
      "dup         v5.16b, %w4                   \n"  // bias
      "1:          \n"
      "ldp         q2, q3, [%0], #32             \n"
      "subs        %w2, %w2, #32                 \n"  // 32 pixels per loop
      "umull       v0.8h, v2.8b, v4.8b           \n"
      "umull2      v1.8h, v2.16b, v4.16b         \n"
      "umull       v2.8h, v3.8b, v4.8b           \n"
      "umull2      v3.8h, v3.16b, v4.16b         \n"
      "prfm        pldl1keep, [%0, 448]          \n"
      "uzp2        v0.16b, v0.16b, v1.16b        \n"
      "uzp2        v1.16b, v2.16b, v3.16b        \n"
      "add         v0.16b, v0.16b, v5.16b        \n"  // add bias (16)
      "add         v1.16b, v1.16b, v5.16b        \n"
      "stp         q0, q1, [%1], #32             \n"  // store 32 pixels
      "b.gt        1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale),   // %3
        "r"(bias)     // %4
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 1024 = 10 bits
void Convert8To16Row_NEON(const uint8_t* src_y,
                          uint16_t* dst_y,
                          int scale,
                          int width) {
  // (src * 0x0101 * scale) >> 16.
  // Since scale is a power of two, compute the shift to use to avoid needing
  // to widen to int32.
  int shift = 15 - __builtin_clz(scale);
  asm volatile(
      "dup         v2.8h, %w[shift]                 \n"
      "1:          \n"
      "ldr         q0, [%[src]], #16                \n"
      "zip2        v1.16b, v0.16b, v0.16b           \n"
      "zip1        v0.16b, v0.16b, v0.16b           \n"
      "subs        %w[width], %w[width], #16        \n"
      "ushl        v1.8h, v1.8h, v2.8h              \n"
      "ushl        v0.8h, v0.8h, v2.8h              \n"
      "stp         q0, q1, [%[dst]], #32            \n"
      "b.ne        1b                               \n"
      : [src] "+r"(src_y),   // %[src]
        [dst] "+r"(dst_y),   // %[dst]
        [width] "+r"(width)  // %[width]
      : [shift] "r"(shift)   // %[shift]
      : "cc", "memory", "v0", "v1", "v2");
}

#endif  // !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
