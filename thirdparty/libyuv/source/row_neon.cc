/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
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

// This module is for GCC Neon
#if !defined(LIBYUV_DISABLE_NEON) && defined(__ARM_NEON__) && \
    !defined(__aarch64__)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are
// reserved.

// q0: Y uint16x8_t
// d2: U uint8x8_t
// d3: V uint8x8_t

// Read 8 Y, 4 U and 4 V from 422
#define READYUV422                               \
  "vld1.8     {d0}, [%[src_y]]!              \n" \
  "vld1.32    {d2[0]}, [%[src_u]]!           \n" \
  "vld1.32    {d2[1]}, [%[src_v]]!           \n" \
  "vmov.u8    d1, d0                         \n" \
  "vmovl.u8   q1, d2                         \n" \
  "vzip.u8    d0, d1                         \n" \
  "vsli.u16   q1, q1, #8                     \n"

// Read 8 Y, 8 U and 8 V from 444
#define READYUV444                               \
  "vld1.8     {d0}, [%[src_y]]!              \n" \
  "vld1.8     {d2}, [%[src_u]]!              \n" \
  "vmovl.u8   q0, d0                         \n" \
  "vld1.8     {d3}, [%[src_v]]!              \n" \
  "vsli.u16   q0, q0, #8                     \n"

// Read 8 Y, and set 4 U and 4 V to 128
#define READYUV400                               \
  "vld1.8     {d0}, [%[src_y]]!              \n" \
  "vmov.u8    q1, #128                       \n" \
  "vmovl.u8   q0, d0                         \n" \
  "vsli.u16   q0, q0, #8                     \n"

// Read 8 Y and 4 UV from NV12
#define READNV12                                                              \
  "vld1.8     {d0}, [%[src_y]]!              \n"                              \
  "vld1.8     {d2}, [%[src_uv]]!             \n"                              \
  "vmov.u8    d1, d0                         \n"                              \
  "vmov.u8    d3, d2                         \n"                              \
  "vzip.u8    d0, d1                         \n"                              \
  "vsli.u16   d2, d2, #8                     \n" /* Duplicate low byte (U) */ \
  "vsri.u16   d3, d3, #8                     \n" /* Duplicate high byte (V) */

// Read 8 Y and 4 VU from NV21
#define READNV21                                                               \
  "vld1.8     {d0}, [%[src_y]]!              \n"                               \
  "vld1.8     {d2}, [%[src_vu]]!             \n"                               \
  "vmov.u8    d1, d0                         \n"                               \
  "vmov.u8    d3, d2                         \n"                               \
  "vzip.u8    d0, d1                         \n"                               \
  "vsri.u16   d2, d2, #8                     \n" /* Duplicate high byte (U) */ \
  "vsli.u16   d3, d3, #8                     \n" /* Duplicate low byte (V) */

// Read 8 YUY2
#define READYUY2                                 \
  "vld2.8     {d0, d2}, [%[src_yuy2]]!       \n" \
  "vmovl.u8   q0, d0                         \n" \
  "vmov.u8    d3, d2                         \n" \
  "vsli.u16   q0, q0, #8                     \n" \
  "vsli.u16   d2, d2, #8                     \n" \
  "vsri.u16   d3, d3, #8                     \n"

// Read 8 UYVY
#define READUYVY                                 \
  "vld2.8     {d2, d3}, [%[src_uyvy]]!       \n" \
  "vmovl.u8   q0, d3                         \n" \
  "vmov.u8    d3, d2                         \n" \
  "vsli.u16   q0, q0, #8                     \n" \
  "vsli.u16   d2, d2, #8                     \n" \
  "vsri.u16   d3, d3, #8                     \n"

// TODO: Use single register for kUVCoeff and multiply by lane
#define YUVTORGB_SETUP                                        \
  "vld1.16    {d31}, [%[kRGBCoeffBias]]                   \n" \
  "vld4.8     {d26[], d27[], d28[], d29[]}, [%[kUVCoeff]] \n" \
  "vdup.u16   q10, d31[1]                                 \n" \
  "vdup.u16   q11, d31[2]                                 \n" \
  "vdup.u16   q12, d31[3]                                 \n" \
  "vdup.u16   d31, d31[0]                                 \n"

// q0: B uint16x8_t
// q1: G uint16x8_t
// q2: R uint16x8_t

// Convert from YUV to 2.14 fixed point RGB
#define YUVTORGB                                           \
  "vmull.u16  q2, d1, d31                    \n"           \
  "vmull.u8   q8, d3, d29                    \n" /* DGV */ \
  "vmull.u16  q0, d0, d31                    \n"           \
  "vmlal.u8   q8, d2, d28                    \n" /* DG */  \
  "vqshrn.u32 d0, q0, #16                    \n"           \
  "vqshrn.u32 d1, q2, #16                    \n" /* Y */   \
  "vmull.u8   q9, d2, d26                    \n" /* DB */  \
  "vmull.u8   q2, d3, d27                    \n" /* DR */  \
  "vadd.u16   q4, q0, q11                    \n" /* G */   \
  "vadd.u16   q2, q0, q2                     \n" /* R */   \
  "vadd.u16   q0, q0, q9                     \n" /* B */   \
  "vqsub.u16  q1, q4, q8                     \n" /* G */   \
  "vqsub.u16  q0, q0, q10                    \n" /* B */   \
  "vqsub.u16  q2, q2, q12                    \n" /* R */

// Convert from 2.14 fixed point RGB To 8 bit RGB
#define RGBTORGB8                                        \
  "vqshrn.u16 d4, q2, #6                     \n" /* R */ \
  "vqshrn.u16 d2, q1, #6                     \n" /* G */ \
  "vqshrn.u16 d0, q0, #6                     \n" /* B */

#define YUVTORGB_REGS \
  "q0", "q1", "q2", "q4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "d31"

#define STORERGBA                                \
  "vmov.u8    d1, d0                         \n" \
  "vmov.u8    d3, d4                         \n" \
  "vmov.u8    d0, d6                         \n" \
  "vst4.8     {d0, d1, d2, d3}, [%[dst_rgba]]! \n"

void I444ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READYUV444
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
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
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
      "vst3.8      {d0, d2, d4}, [%[dst_rgb24]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgb24] "+r"(dst_rgb24),                       // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

void I422ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READYUV422
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
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
      "1:          \n"  //
      READYUV444
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
      "vld1.8      {d6}, [%[src_a]]!             \n"
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
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
      "1:          \n"  //
      READYUV422
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
      "vld1.8      {d6}, [%[src_a]]!             \n"
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [src_a] "+r"(src_a),                               // %[src_a]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
}

void I422ToRGBARow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_rgba,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"                                //
      READYUV422                                      //
      "subs        %[width], %[width], #8        \n"  //
      YUVTORGB                                        //
          RGBTORGB8                                   //
              STORERGBA                               //
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgba] "+r"(dst_rgba),                         // %[dst_rgba]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
}

void I422ToRGB24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_u,
                         const uint8_t* src_v,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READYUV422
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
      "vst3.8      {d0, d2, d4}, [%[dst_rgb24]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgb24] "+r"(dst_rgb24),                       // %[dst_rgb24]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

#define ARGBTORGB565                                                        \
  "vshll.u8    q2, d4, #8                    \n" /* R                    */ \
  "vshll.u8    q1, d2, #8                    \n" /* G                    */ \
  "vshll.u8    q0, d0, #8                    \n" /* B                    */ \
  "vsri.16     q2, q1, #5                    \n" /* RG                   */ \
  "vsri.16     q2, q0, #11                   \n" /* RGB                  */

void I422ToRGB565Row_NEON(const uint8_t* src_y,
                          const uint8_t* src_u,
                          const uint8_t* src_v,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READYUV422
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
          ARGBTORGB565
      "vst1.8      {q2}, [%[dst_rgb565]]!        \n"  // store 8 pixels RGB565.
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_rgb565] "+r"(dst_rgb565),                     // %[dst_rgb565]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

#define ARGBTOARGB1555                                                      \
  "vshll.u8    q3, d6, #8                    \n" /* A                    */ \
  "vshll.u8    q2, d4, #8                    \n" /* R                    */ \
  "vshll.u8    q1, d2, #8                    \n" /* G                    */ \
  "vshll.u8    q0, d0, #8                    \n" /* B                    */ \
  "vsri.16     q3, q2, #1                    \n" /* AR                   */ \
  "vsri.16     q3, q1, #6                    \n" /* ARG                  */ \
  "vsri.16     q3, q0, #11                   \n" /* ARGB                 */

void I422ToARGB1555Row_NEON(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb1555,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  asm volatile(
      YUVTORGB_SETUP
      "1:          \n"  //
      READYUV422
      "subs        %[width], %[width], #8        \n" YUVTORGB RGBTORGB8
      "vmov.u8     d6, #0xff                     \n" ARGBTOARGB1555
      "vst1.8      {q3}, [%[dst_argb1555]]!      \n"  // store 8 pixels RGB1555.
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb1555] "+r"(dst_argb1555),                 // %[dst_argb1555]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "q3");
}

#define ARGBTOARGB4444                                                      \
  "vshr.u8    d0, d0, #4                     \n" /* B                    */ \
  "vbic.32    d2, d2, d7                     \n" /* G                    */ \
  "vshr.u8    d4, d4, #4                     \n" /* R                    */ \
  "vbic.32    d6, d6, d7                     \n" /* A                    */ \
  "vorr       d0, d0, d2                     \n" /* BG                   */ \
  "vorr       d1, d4, d6                     \n" /* RA                   */ \
  "vzip.u8    d0, d1                         \n" /* BGRA                 */

void I422ToARGB4444Row_NEON(const uint8_t* src_y,
                            const uint8_t* src_u,
                            const uint8_t* src_v,
                            uint8_t* dst_argb4444,
                            const struct YuvConstants* yuvconstants,
                            int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "vmov.u8     d7, #0x0f                     \n"  // vbic bits to clear
      "1:          \n"                                //
      READYUV422 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n" ARGBTOARGB4444
      "vst1.8      {q0}, [%[dst_argb4444]]!      \n"  // store 8 pixels
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_u] "+r"(src_u),                               // %[src_u]
        [src_v] "+r"(src_v),                               // %[src_v]
        [dst_argb4444] "+r"(dst_argb4444),                 // %[dst_argb4444]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "q3");
}

void I400ToARGBRow_NEON(const uint8_t* src_y,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READYUV400 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n"
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
}

void J400ToARGBRow_NEON(const uint8_t* src_y, uint8_t* dst_argb, int width) {
  asm volatile(
      "vmov.u8     d23, #255                     \n"
      "1:          \n"
      "vld1.8      {d20}, [%0]!                  \n"
      "subs        %2, %2, #8                    \n"
      "vmov        d21, d20                      \n"
      "vmov        d22, d20                      \n"
      "vst4.8      {d20, d21, d22, d23}, [%1]!   \n"
      "bgt         1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "d20", "d21", "d22", "d23");
}

void NV12ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_uv,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READNV12 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n"
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_uv] "+r"(src_uv),                             // %[src_uv]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
}

void NV21ToARGBRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_vu,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READNV21 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n"
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_vu] "+r"(src_vu),                             // %[src_vu]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
}

void NV12ToRGB24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_uv,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READNV12 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n"
      "vst3.8      {d0, d2, d4}, [%[dst_rgb24]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_uv] "+r"(src_uv),                             // %[src_uv]
        [dst_rgb24] "+r"(dst_rgb24),                       // %[dst_rgb24]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

void NV21ToRGB24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_rgb24,
                         const struct YuvConstants* yuvconstants,
                         int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READNV21 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n"
      "vst3.8      {d0, d2, d4}, [%[dst_rgb24]]! \n"
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_vu] "+r"(src_vu),                             // %[src_vu]
        [dst_rgb24] "+r"(dst_rgb24),                       // %[dst_rgb24]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

void NV12ToRGB565Row_NEON(const uint8_t* src_y,
                          const uint8_t* src_uv,
                          uint8_t* dst_rgb565,
                          const struct YuvConstants* yuvconstants,
                          int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READNV12 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n" ARGBTORGB565
      "vst1.8      {q2}, [%[dst_rgb565]]!        \n"  // store 8 pixels RGB565.
      "bgt         1b                            \n"
      : [src_y] "+r"(src_y),                               // %[src_y]
        [src_uv] "+r"(src_uv),                             // %[src_uv]
        [dst_rgb565] "+r"(dst_rgb565),                     // %[dst_rgb565]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS);
}

void YUY2ToARGBRow_NEON(const uint8_t* src_yuy2,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READYUY2 YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n"
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_yuy2] "+r"(src_yuy2),                         // %[src_yuy2]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
}

void UYVYToARGBRow_NEON(const uint8_t* src_uyvy,
                        uint8_t* dst_argb,
                        const struct YuvConstants* yuvconstants,
                        int width) {
  asm volatile(
      YUVTORGB_SETUP
      "vmov.u8     d6, #255                      \n"
      "1:          \n"  //
      READUYVY YUVTORGB RGBTORGB8
      "subs        %[width], %[width], #8        \n"
      "vst4.8      {d0, d2, d4, d6}, [%[dst_argb]]! \n"
      "bgt         1b                            \n"
      : [src_uyvy] "+r"(src_uyvy),                         // %[src_uyvy]
        [dst_argb] "+r"(dst_argb),                         // %[dst_argb]
        [width] "+r"(width)                                // %[width]
      : [kUVCoeff] "r"(&yuvconstants->kUVCoeff),           // %[kUVCoeff]
        [kRGBCoeffBias] "r"(&yuvconstants->kRGBCoeffBias)  // %[kRGBCoeffBias]
      : "cc", "memory", YUVTORGB_REGS, "d6");
}

// Reads 16 pairs of UV and write even values to dst_u and odd to dst_v.
void SplitUVRow_NEON(const uint8_t* src_uv,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  asm volatile(
      "1:          \n"
      "vld2.8      {q0, q1}, [%0]!               \n"  // load 16 pairs of UV
      "subs        %3, %3, #16                   \n"  // 16 processed per loop
      "vst1.8      {q0}, [%1]!                   \n"  // store U
      "vst1.8      {q1}, [%2]!                   \n"  // store V
      "bgt         1b                            \n"
      : "+r"(src_uv),               // %0
        "+r"(dst_u),                // %1
        "+r"(dst_v),                // %2
        "+r"(width)                 // %3  // Output registers
      :                             // Input registers
      : "cc", "memory", "q0", "q1"  // Clobber List
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
      "vld1.8      {q0}, [%0], %3                \n"  // load 16 bytes
      "subs        %2, %2, #16                   \n"  // 16 processed per loop
      "pld         [%0, #1792]                   \n"
      "vst1.8      {q0}, [%1]!                   \n"  // store 16 bytes
      "bgt         1b                            \n"
      : "+r"(src),            // %0
        "+r"(dst),            // %1
        "+r"(width)           // %2
      : "r"(src_tile_stride)  // %3
      : "cc", "memory", "q0"  // Clobber List
  );
}

// Reads 16 byte Y's of 16 bits from tile and writes out 16 Y's.
void DetileRow_16_NEON(const uint16_t* src,
                       ptrdiff_t src_tile_stride,
                       uint16_t* dst,
                       int width) {
  asm volatile(
      "1:          \n"
      "vld1.16     {q0, q1}, [%0], %3            \n"  // load 16 pixels
      "subs        %2, %2, #16                   \n"  // 16 processed per loop
      "pld         [%0, #3584]                   \n"
      "vst1.16     {q0, q1}, [%1]!               \n"  // store 16 pixels
      "bgt         1b                            \n"
      : "+r"(src),                  // %0
        "+r"(dst),                  // %1
        "+r"(width)                 // %2
      : "r"(src_tile_stride * 2)    // %3
      : "cc", "memory", "q0", "q1"  // Clobber List
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
      "vld2.8      {d0, d1}, [%0], %4            \n"
      "subs        %3, %3, #16                   \n"
      "pld         [%0, #1792]                   \n"
      "vst1.8      {d0}, [%1]!                   \n"
      "vst1.8      {d1}, [%2]!                   \n"
      "bgt         1b                            \n"
      : "+r"(src_uv),               // %0
        "+r"(dst_u),                // %1
        "+r"(dst_v),                // %2
        "+r"(width)                 // %3
      : "r"(src_tile_stride)        // %4
      : "cc", "memory", "d0", "d1"  // Clobber List
  );
}

#if defined(LIBYUV_USE_ST2)
// Read 16 Y, 8 UV, and write 8 YUYV.
void DetileToYUY2_NEON(const uint8_t* src_y,
                       ptrdiff_t src_y_tile_stride,
                       const uint8_t* src_uv,
                       ptrdiff_t src_uv_tile_stride,
                       uint8_t* dst_yuy2,
                       int width) {
  asm volatile(
      "1:          \n"
      "vld1.8      {q0}, [%0], %4                \n"  // Load 16 Y
      "pld         [%0, #1792]                   \n"
      "vld1.8      {q1}, [%1], %5                \n"  // Load 8 UV
      "pld         [%1, #1792]                   \n"
      "subs        %3, %3, #16                   \n"
      "vst2.8      {q0, q1}, [%2]!               \n"
      "bgt         1b                            \n"
      : "+r"(src_y),                            // %0
        "+r"(src_uv),                           // %1
        "+r"(dst_yuy2),                         // %2
        "+r"(width)                             // %3
      : "r"(src_y_tile_stride),                 // %4
        "r"(src_uv_tile_stride)                 // %5
      : "cc", "memory", "d0", "d1", "d2", "d3"  // Clobber list
  );
}
#else
// Read 16 Y, 8 UV, and write 8 YUYV.
void DetileToYUY2_NEON(const uint8_t* src_y,
                       ptrdiff_t src_y_tile_stride,
                       const uint8_t* src_uv,
                       ptrdiff_t src_uv_tile_stride,
                       uint8_t* dst_yuy2,
                       int width) {
  asm volatile(
      "1:          \n"
      "vld1.8      {q0}, [%0], %4                \n"  // Load 16 Y
      "vld1.8      {q1}, [%1], %5                \n"  // Load 8 UV
      "subs        %3, %3, #16                   \n"
      "pld         [%0, #1792]                   \n"
      "vzip.8      q0, q1                        \n"
      "pld         [%1, #1792]                   \n"
      "vst1.8      {q0, q1}, [%2]!               \n"
      "bgt         1b                            \n"
      : "+r"(src_y),                            // %0
        "+r"(src_uv),                           // %1
        "+r"(dst_yuy2),                         // %2
        "+r"(width)                             // %3
      : "r"(src_y_tile_stride),                 // %4
        "r"(src_uv_tile_stride)                 // %5
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber list
  );
}
#endif

void UnpackMT2T_NEON(const uint8_t* src, uint16_t* dst, size_t size) {
  asm volatile(
      "1:          \n"
      "vld1.8      {q14}, [%0]!                  \n"  // Load lower bits.
      "vld1.8      {q9}, [%0]!                   \n"  // Load upper bits row
                                                      // by row.
      "vld1.8      {q11}, [%0]!                  \n"
      "vld1.8      {q13}, [%0]!                  \n"
      "vld1.8      {q15}, [%0]!                  \n"
      "subs        %2, %2, #80                   \n"
      "vshl.u8     q8, q14, #6                   \n"  // Shift lower bit data
                                                      // appropriately.
      "vshl.u8     q10, q14, #4                  \n"
      "vshl.u8     q12, q14, #2                  \n"
      "vzip.u8     q8, q9                        \n"  // Interleave upper and
                                                      // lower bits.
      "vzip.u8     q10, q11                      \n"
      "vzip.u8     q12, q13                      \n"
      "vzip.u8     q14, q15                      \n"
      "vsri.u16    q8, q8, #10                   \n"  // Copy upper 6 bits
                                                      // into lower 6 bits for
                                                      // better accuracy in
                                                      // conversions.
      "vsri.u16    q9, q9, #10                   \n"
      "vsri.u16    q10, q10, #10                 \n"
      "vsri.u16    q11, q11, #10                 \n"
      "vsri.u16    q12, q12, #10                 \n"
      "vsri.u16    q13, q13, #10                 \n"
      "vsri.u16    q14, q14, #10                 \n"
      "vsri.u16    q15, q15, #10                 \n"
      "vstmia      %1!, {q8-q15}                 \n"  // Store pixel block (64
                                                      // pixels).
      "bgt         1b                            \n"
      : "+r"(src),  // %0
        "+r"(dst),  // %1
        "+r"(size)  // %2
      :
      : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

// Reads 16 U's and V's and writes out 16 pairs of UV.
void MergeUVRow_NEON(const uint8_t* src_u,
                     const uint8_t* src_v,
                     uint8_t* dst_uv,
                     int width) {
  asm volatile(
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load U
      "vld1.8      {q1}, [%1]!                   \n"  // load V
      "subs        %3, %3, #16                   \n"  // 16 processed per loop
      "vst2.8      {q0, q1}, [%2]!               \n"  // store 16 pairs of UV
      "bgt         1b                            \n"
      : "+r"(src_u),                // %0
        "+r"(src_v),                // %1
        "+r"(dst_uv),               // %2
        "+r"(width)                 // %3  // Output registers
      :                             // Input registers
      : "cc", "memory", "q0", "q1"  // Clobber List
  );
}

// Reads 16 packed RGB and write to planar dst_r, dst_g, dst_b.
void SplitRGBRow_NEON(const uint8_t* src_rgb,
                      uint8_t* dst_r,
                      uint8_t* dst_g,
                      uint8_t* dst_b,
                      int width) {
  asm volatile(
      "1:          \n"
      "vld3.8      {d0, d2, d4}, [%0]!           \n"  // load 8 RGB
      "vld3.8      {d1, d3, d5}, [%0]!           \n"  // next 8 RGB
      "subs        %4, %4, #16                   \n"  // 16 processed per loop
      "vst1.8      {q0}, [%1]!                   \n"  // store R
      "vst1.8      {q1}, [%2]!                   \n"  // store G
      "vst1.8      {q2}, [%3]!                   \n"  // store B
      "bgt         1b                            \n"
      : "+r"(src_rgb),                    // %0
        "+r"(dst_r),                      // %1
        "+r"(dst_g),                      // %2
        "+r"(dst_b),                      // %3
        "+r"(width)                       // %4
      :                                   // Input registers
      : "cc", "memory", "q0", "q1", "q2"  // Clobber List
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
      "vld1.8      {q0}, [%0]!                   \n"  // load R
      "vld1.8      {q1}, [%1]!                   \n"  // load G
      "vld1.8      {q2}, [%2]!                   \n"  // load B
      "subs        %4, %4, #16                   \n"  // 16 processed per loop
      "vst3.8      {d0, d2, d4}, [%3]!           \n"  // store 8 RGB
      "vst3.8      {d1, d3, d5}, [%3]!           \n"  // next 8 RGB
      "bgt         1b                            \n"
      : "+r"(src_r),                      // %0
        "+r"(src_g),                      // %1
        "+r"(src_b),                      // %2
        "+r"(dst_rgb),                    // %3
        "+r"(width)                       // %4
      :                                   // Input registers
      : "cc", "memory", "q0", "q1", "q2"  // Clobber List
  );
}

// Reads 16 packed ARGB and write to planar dst_r, dst_g, dst_b, dst_a.
void SplitARGBRow_NEON(const uint8_t* src_argb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       uint8_t* dst_a,
                       int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ARGB
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // next 8 ARGB
      "subs        %5, %5, #16                   \n"  // 16 processed per loop
      "vst1.8      {q0}, [%3]!                   \n"  // store B
      "vst1.8      {q1}, [%2]!                   \n"  // store G
      "vst1.8      {q2}, [%1]!                   \n"  // store R
      "vst1.8      {q3}, [%4]!                   \n"  // store A
      "bgt         1b                            \n"
      : "+r"(src_argb),                         // %0
        "+r"(dst_r),                            // %1
        "+r"(dst_g),                            // %2
        "+r"(dst_b),                            // %3
        "+r"(dst_a),                            // %4
        "+r"(width)                             // %5
      :                                         // Input registers
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

// Reads 16 planar R's, G's and B's and writes out 16 packed ARGB at a time
void MergeARGBRow_NEON(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       const uint8_t* src_a,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "1:          \n"
      "vld1.8      {q2}, [%0]!                   \n"  // load R
      "vld1.8      {q1}, [%1]!                   \n"  // load G
      "vld1.8      {q0}, [%2]!                   \n"  // load B
      "vld1.8      {q3}, [%3]!                   \n"  // load A
      "subs        %5, %5, #16                   \n"  // 16 processed per loop
      "vst4.8      {d0, d2, d4, d6}, [%4]!       \n"  // store 8 ARGB
      "vst4.8      {d1, d3, d5, d7}, [%4]!       \n"  // next 8 ARGB
      "bgt         1b                            \n"
      : "+r"(src_r),                            // %0
        "+r"(src_g),                            // %1
        "+r"(src_b),                            // %2
        "+r"(src_a),                            // %3
        "+r"(dst_argb),                         // %4
        "+r"(width)                             // %5
      :                                         // Input registers
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

// Reads 16 packed ARGB and write to planar dst_r, dst_g, dst_b.
void SplitXRGBRow_NEON(const uint8_t* src_argb,
                       uint8_t* dst_r,
                       uint8_t* dst_g,
                       uint8_t* dst_b,
                       int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ARGB
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // next 8 ARGB
      "subs        %4, %4, #16                   \n"  // 16 processed per loop
      "vst1.8      {q0}, [%3]!                   \n"  // store B
      "vst1.8      {q1}, [%2]!                   \n"  // store G
      "vst1.8      {q2}, [%1]!                   \n"  // store R
      "bgt         1b                            \n"
      : "+r"(src_argb),                         // %0
        "+r"(dst_r),                            // %1
        "+r"(dst_g),                            // %2
        "+r"(dst_b),                            // %3
        "+r"(width)                             // %4
      :                                         // Input registers
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

// Reads 16 planar R's, G's, B's and A's and writes out 16 packed ARGB at a time
void MergeXRGBRow_NEON(const uint8_t* src_r,
                       const uint8_t* src_g,
                       const uint8_t* src_b,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "vmov.u8     q3, #255                      \n"  // load A(255)
      "1:          \n"
      "vld1.8      {q2}, [%0]!                   \n"  // load R
      "vld1.8      {q1}, [%1]!                   \n"  // load G
      "vld1.8      {q0}, [%2]!                   \n"  // load B
      "subs        %4, %4, #16                   \n"  // 16 processed per loop
      "vst4.8      {d0, d2, d4, d6}, [%3]!       \n"  // store 8 ARGB
      "vst4.8      {d1, d3, d5, d7}, [%3]!       \n"  // next 8 ARGB
      "bgt         1b                            \n"
      : "+r"(src_r),                            // %0
        "+r"(src_g),                            // %1
        "+r"(src_b),                            // %2
        "+r"(dst_argb),                         // %3
        "+r"(width)                             // %4
      :                                         // Input registers
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
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
      "vmov.u32    q14, #1023                    \n"
      "vdup.32     q15, %5                       \n"
      "1:          \n"
      "vld1.16     {d4}, [%2]!                   \n"  // B
      "vld1.16     {d2}, [%1]!                   \n"  // G
      "vld1.16     {d0}, [%0]!                   \n"  // R
      "subs        %4, %4, #4                    \n"
      "vmovl.u16   q2, d4                        \n"  // B
      "vmovl.u16   q1, d2                        \n"  // G
      "vmovl.u16   q0, d0                        \n"  // R
      "vshl.u32    q2, q2, q15                   \n"  // 000B
      "vshl.u32    q1, q1, q15                   \n"
      "vshl.u32    q0, q0, q15                   \n"
      "vmin.u32    q2, q2, q14                   \n"
      "vmin.u32    q1, q1, q14                   \n"
      "vmin.u32    q0, q0, q14                   \n"
      "vsli.u32    q2, q1, #10                   \n"  // 00GB
      "vsli.u32    q2, q0, #20                   \n"  // 0RGB
      "vorr.u32    q2, #0xc0000000               \n"  // ARGB (AR30)
      "vst1.8      {q2}, [%3]!                   \n"
      "bgt         1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_ar30),  // %3
        "+r"(width)      // %4
      : "r"(shift)       // %5
      : "memory", "cc", "q0", "q1", "q2", "q14", "q15");
}

void MergeXR30Row_10_NEON(const uint16_t* src_r,
                          const uint16_t* src_g,
                          const uint16_t* src_b,
                          uint8_t* dst_ar30,
                          int /* depth */,
                          int width) {
  asm volatile(
      "vmov.u32    q14, #1023                    \n"
      "1:          \n"
      "vld1.16     {d4}, [%2]!                   \n"  // B
      "vld1.16     {d2}, [%1]!                   \n"  // G
      "vld1.16     {d0}, [%0]!                   \n"  // R
      "subs        %4, %4, #4                    \n"
      "vmovl.u16   q2, d4                        \n"  // 000B
      "vmovl.u16   q1, d2                        \n"  // G
      "vmovl.u16   q0, d0                        \n"  // R
      "vmin.u32    q2, q2, q14                   \n"
      "vmin.u32    q1, q1, q14                   \n"
      "vmin.u32    q0, q0, q14                   \n"
      "vsli.u32    q2, q1, #10                   \n"  // 00GB
      "vsli.u32    q2, q0, #20                   \n"  // 0RGB
      "vorr.u32    q2, #0xc0000000               \n"  // ARGB (AR30)
      "vst1.8      {q2}, [%3]!                   \n"
      "bgt         1b                            \n"
      "3:          \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_ar30),  // %3
        "+r"(width)      // %4
      :
      : "memory", "cc", "q0", "q1", "q2", "q14");
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

      "vdup.u16    q15, %6                       \n"
      "vdup.u16    q14, %7                       \n"
      "1:          \n"
      "vld1.16     {q2}, [%0]!                   \n"  // R
      "vld1.16     {q1}, [%1]!                   \n"  // G
      "vld1.16     {q0}, [%2]!                   \n"  // B
      "vld1.16     {q3}, [%3]!                   \n"  // A
      "subs        %5, %5, #8                    \n"
      "vmin.u16    q2, q2, q14                   \n"
      "vmin.u16    q1, q1, q14                   \n"
      "vmin.u16    q0, q0, q14                   \n"
      "vmin.u16    q3, q3, q14                   \n"
      "vshl.u16    q2, q2, q15                   \n"
      "vshl.u16    q1, q1, q15                   \n"
      "vshl.u16    q0, q0, q15                   \n"
      "vshl.u16    q3, q3, q15                   \n"
      "vst4.16     {d0, d2, d4, d6}, [%4]!       \n"
      "vst4.16     {d1, d3, d5, d7}, [%4]!       \n"
      "bgt         1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_ar64),  // %4
        "+r"(width)      // %5
      : "r"(shift),      // %6
        "r"(mask)        // %7
      : "memory", "cc", "q0", "q1", "q2", "q3", "q15");
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

      "vmov.u8     q3, #0xff                     \n"  // A (0xffff)
      "vdup.u16    q15, %5                       \n"
      "vdup.u16    q14, %6                       \n"
      "1:          \n"
      "vld1.16     {q2}, [%0]!                   \n"  // R
      "vld1.16     {q1}, [%1]!                   \n"  // G
      "vld1.16     {q0}, [%2]!                   \n"  // B
      "subs        %4, %4, #8                    \n"
      "vmin.u16    q2, q2, q14                   \n"
      "vmin.u16    q1, q1, q14                   \n"
      "vmin.u16    q0, q0, q14                   \n"
      "vshl.u16    q2, q2, q15                   \n"
      "vshl.u16    q1, q1, q15                   \n"
      "vshl.u16    q0, q0, q15                   \n"
      "vst4.16     {d0, d2, d4, d6}, [%3]!       \n"
      "vst4.16     {d1, d3, d5, d7}, [%3]!       \n"
      "bgt         1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_ar64),  // %3
        "+r"(width)      // %4
      : "r"(shift),      // %5
        "r"(mask)        // %6
      : "memory", "cc", "q0", "q1", "q2", "q3", "q15");
}

void MergeARGB16To8Row_NEON(const uint16_t* src_r,
                            const uint16_t* src_g,
                            const uint16_t* src_b,
                            const uint16_t* src_a,
                            uint8_t* dst_argb,
                            int depth,
                            int width) {
  int shift = 8 - depth;
  asm volatile(

      "vdup.16     q15, %6                       \n"
      "1:          \n"
      "vld1.16     {q2}, [%0]!                   \n"  // R
      "vld1.16     {q1}, [%1]!                   \n"  // G
      "vld1.16     {q0}, [%2]!                   \n"  // B
      "vld1.16     {q3}, [%3]!                   \n"  // A
      "subs        %5, %5, #8                    \n"
      "vshl.u16    q2, q2, q15                   \n"
      "vshl.u16    q1, q1, q15                   \n"
      "vshl.u16    q0, q0, q15                   \n"
      "vshl.u16    q3, q3, q15                   \n"
      "vqmovn.u16  d0, q0                        \n"
      "vqmovn.u16  d1, q1                        \n"
      "vqmovn.u16  d2, q2                        \n"
      "vqmovn.u16  d3, q3                        \n"
      "vst4.8      {d0, d1, d2, d3}, [%4]!       \n"
      "bgt         1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(src_a),     // %3
        "+r"(dst_argb),  // %4
        "+r"(width)      // %5
      : "r"(shift)       // %6
      : "memory", "cc", "q0", "q1", "q2", "q3", "q15");
}

void MergeXRGB16To8Row_NEON(const uint16_t* src_r,
                            const uint16_t* src_g,
                            const uint16_t* src_b,
                            uint8_t* dst_argb,
                            int depth,
                            int width) {
  int shift = 8 - depth;
  asm volatile(

      "vdup.16     q15, %5                       \n"
      "vmov.u8     d6, #0xff                     \n"  // A (0xff)
      "1:          \n"
      "vld1.16     {q2}, [%0]!                   \n"  // R
      "vld1.16     {q1}, [%1]!                   \n"  // G
      "vld1.16     {q0}, [%2]!                   \n"  // B
      "subs        %4, %4, #8                    \n"
      "vshl.u16    q2, q2, q15                   \n"
      "vshl.u16    q1, q1, q15                   \n"
      "vshl.u16    q0, q0, q15                   \n"
      "vqmovn.u16  d5, q2                        \n"
      "vqmovn.u16  d4, q1                        \n"
      "vqmovn.u16  d3, q0                        \n"
      "vst4.u8     {d3, d4, d5, d6}, [%3]!       \n"
      "bgt         1b                            \n"
      : "+r"(src_r),     // %0
        "+r"(src_g),     // %1
        "+r"(src_b),     // %2
        "+r"(dst_argb),  // %3
        "+r"(width)      // %4
      : "r"(shift)       // %5
      : "memory", "cc", "q0", "q1", "q2", "d6", "q15");
}

// Copy multiple of 32.  vld4.8  allow unaligned and is fastest on a15.
void CopyRow_NEON(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      "1:          \n"
      "vld1.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 32
      "subs        %2, %2, #32                   \n"  // 32 processed per loop
      "vst1.8      {d0, d1, d2, d3}, [%1]!       \n"  // store 32
      "bgt         1b                            \n"
      : "+r"(src),                  // %0
        "+r"(dst),                  // %1
        "+r"(width)                 // %2  // Output registers
      :                             // Input registers
      : "cc", "memory", "q0", "q1"  // Clobber List
  );
}

// SetRow writes 'width' bytes using an 8 bit value repeated.
void SetRow_NEON(uint8_t* dst, uint8_t v8, int width) {
  asm volatile(
      "vdup.8      q0, %2                        \n"  // duplicate 16 bytes
      "1:          \n"
      "subs        %1, %1, #16                   \n"  // 16 bytes per loop
      "vst1.8      {q0}, [%0]!                   \n"  // store
      "bgt         1b                            \n"
      : "+r"(dst),   // %0
        "+r"(width)  // %1
      : "r"(v8)      // %2
      : "cc", "memory", "q0");
}

// ARGBSetRow writes 'width' pixels using an 32 bit value repeated.
void ARGBSetRow_NEON(uint8_t* dst, uint32_t v32, int width) {
  asm volatile(
      "vdup.u32    q0, %2                        \n"  // duplicate 4 ints
      "1:          \n"
      "subs        %1, %1, #4                    \n"  // 4 pixels per loop
      "vst1.8      {q0}, [%0]!                   \n"  // store
      "bgt         1b                            \n"
      : "+r"(dst),   // %0
        "+r"(width)  // %1
      : "r"(v32)     // %2
      : "cc", "memory", "q0");
}

void MirrorRow_NEON(const uint8_t* src, uint8_t* dst, int width) {
  asm volatile(
      // Start at end of source row.
      "add         %0, %0, %2                    \n"
      "sub         %0, %0, #32                   \n"  // 32 bytes per loop

      "1:          \n"
      "vld1.8      {q1, q2}, [%0], %3            \n"  // src -= 32
      "subs        %2, #32                       \n"  // 32 pixels per loop.
      "vrev64.8    q0, q2                        \n"
      "vrev64.8    q1, q1                        \n"
      "vswp        d0, d1                        \n"
      "vswp        d2, d3                        \n"
      "vst1.8      {q0, q1}, [%1]!               \n"  // dst += 32
      "bgt         1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      : "r"(-32)     // %3
      : "cc", "memory", "q0", "q1", "q2");
}

void MirrorUVRow_NEON(const uint8_t* src_uv, uint8_t* dst_uv, int width) {
  asm volatile(
      // Start at end of source row.
      "mov         r12, #-16                     \n"
      "add         %0, %0, %2, lsl #1            \n"
      "sub         %0, #16                       \n"

      "1:          \n"
      "vld2.8      {d0, d1}, [%0], r12           \n"  // src -= 16
      "subs        %2, #8                        \n"  // 8 pixels per loop.
      "vrev64.8    q0, q0                        \n"
      "vst2.8      {d0, d1}, [%1]!               \n"  // dst += 16
      "bgt         1b                            \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_uv),  // %1
        "+r"(width)    // %2
      :
      : "cc", "memory", "r12", "q0");
}

void MirrorSplitUVRow_NEON(const uint8_t* src_uv,
                           uint8_t* dst_u,
                           uint8_t* dst_v,
                           int width) {
  asm volatile(
      // Start at end of source row.
      "mov         r12, #-16                     \n"
      "add         %0, %0, %3, lsl #1            \n"
      "sub         %0, #16                       \n"

      "1:          \n"
      "vld2.8      {d0, d1}, [%0], r12           \n"  // src -= 16
      "subs        %3, #8                        \n"  // 8 pixels per loop.
      "vrev64.8    q0, q0                        \n"
      "vst1.8      {d0}, [%1]!                   \n"  // dst += 8
      "vst1.8      {d1}, [%2]!                   \n"
      "bgt         1b                            \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_u),   // %1
        "+r"(dst_v),   // %2
        "+r"(width)    // %3
      :
      : "cc", "memory", "r12", "q0");
}

void ARGBMirrorRow_NEON(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  asm volatile(
      "add         %0, %0, %2, lsl #2            \n"
      "sub         %0, #32                       \n"

      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0], %3    \n"  // src -= 32
      "subs        %2, #8                        \n"  // 8 pixels per loop.
      "vrev64.8    d0, d0                        \n"
      "vrev64.8    d1, d1                        \n"
      "vrev64.8    d2, d2                        \n"
      "vrev64.8    d3, d3                        \n"
      "vst4.8      {d0, d1, d2, d3}, [%1]!       \n"  // dst += 32
      "bgt         1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      : "r"(-32)         // %3
      : "cc", "memory", "d0", "d1", "d2", "d3");
}

void RGB24MirrorRow_NEON(const uint8_t* src_rgb24,
                         uint8_t* dst_rgb24,
                         int width) {
  src_rgb24 += width * 3 - 24;
  asm volatile(
      "1:          \n"
      "vld3.8      {d0, d1, d2}, [%0], %3        \n"  // src -= 24
      "subs        %2, #8                        \n"  // 8 pixels per loop.
      "vrev64.8    d0, d0                        \n"
      "vrev64.8    d1, d1                        \n"
      "vrev64.8    d2, d2                        \n"
      "vst3.8      {d0, d1, d2}, [%1]!           \n"  // dst += 24
      "bgt         1b                            \n"
      : "+r"(src_rgb24),  // %0
        "+r"(dst_rgb24),  // %1
        "+r"(width)       // %2
      : "r"(-24)          // %3
      : "cc", "memory", "d0", "d1", "d2");
}

void RGB24ToARGBRow_NEON(const uint8_t* src_rgb24,
                         uint8_t* dst_argb,
                         int width) {
  asm volatile(
      "vmov.u8     d4, #255                      \n"  // Alpha
      "1:          \n"
      "vld3.8      {d1, d2, d3}, [%0]!           \n"  // load 8 pixels of RGB24.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vst4.8      {d1, d2, d3, d4}, [%1]!       \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(src_rgb24),  // %0
        "+r"(dst_argb),   // %1
        "+r"(width)       // %2
      :
      : "cc", "memory", "d1", "d2", "d3", "d4"  // Clobber List
  );
}

void RAWToARGBRow_NEON(const uint8_t* src_raw, uint8_t* dst_argb, int width) {
  asm volatile(
      "vmov.u8     d4, #255                      \n"  // Alpha
      "1:          \n"
      "vld3.8      {d1, d2, d3}, [%0]!           \n"  // load 8 pixels of RAW.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vswp.u8     d1, d3                        \n"  // swap R, B
      "vst4.8      {d1, d2, d3, d4}, [%1]!       \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(src_raw),   // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "d1", "d2", "d3", "d4"  // Clobber List
  );
}

void RAWToRGBARow_NEON(const uint8_t* src_raw, uint8_t* dst_rgba, int width) {
  asm volatile(
      "vmov.u8     d0, #255                      \n"  // Alpha
      "1:          \n"
      "vld3.8      {d1, d2, d3}, [%0]!           \n"  // load 8 pixels of RAW.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vswp.u8     d1, d3                        \n"  // swap R, B
      "vst4.8      {d0, d1, d2, d3}, [%1]!       \n"  // store 8 pixels of RGBA.
      "bgt         1b                            \n"
      : "+r"(src_raw),   // %0
        "+r"(dst_rgba),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "d0", "d1", "d2", "d3"  // Clobber List
  );
}
void RAWToRGB24Row_NEON(const uint8_t* src_raw, uint8_t* dst_rgb24, int width) {
  asm volatile(
      "1:          \n"
      "vld3.8      {d1, d2, d3}, [%0]!           \n"  // load 8 pixels of RAW.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vswp.u8     d1, d3                        \n"  // swap R, B
      "vst3.8      {d1, d2, d3}, [%1]!           \n"  // store 8 pixels of
                                                      // RGB24.
      "bgt         1b                            \n"
      : "+r"(src_raw),    // %0
        "+r"(dst_rgb24),  // %1
        "+r"(width)       // %2
      :
      : "cc", "memory", "d1", "d2", "d3"  // Clobber List
  );
}

#define RGB565TOARGB                                                        \
  "vshrn.u16  d6, q0, #5                     \n" /* G xxGGGGGG           */ \
  "vuzp.u8    d0, d1                         \n" /* d0 xxxBBBBB RRRRRxxx */ \
  "vshl.u8    d6, d6, #2                     \n" /* G GGGGGG00 upper 6   */ \
  "vshr.u8    d1, d1, #3                     \n" /* R 000RRRRR lower 5   */ \
  "vshl.u8    q0, q0, #3                     \n" /* B,R BBBBB000 upper 5 */ \
  "vshr.u8    q2, q0, #5                     \n" /* B,R 00000BBB lower 3 */ \
  "vorr.u8    d0, d0, d4                     \n" /* B                    */ \
  "vshr.u8    d4, d6, #6                     \n" /* G 000000GG lower 2   */ \
  "vorr.u8    d2, d1, d5                     \n" /* R                    */ \
  "vorr.u8    d1, d4, d6                     \n" /* G                    */

void RGB565ToARGBRow_NEON(const uint8_t* src_rgb565,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      "vmov.u8     d3, #255                      \n"  // Alpha
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 RGB565 pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      RGB565TOARGB
      "vst4.8      {d0, d1, d2, d3}, [%1]!       \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(src_rgb565),  // %0
        "+r"(dst_argb),    // %1
        "+r"(width)        // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

#define ARGB1555TOARGB                                                      \
  "vshrn.u16  d7, q0, #8                     \n" /* A Arrrrrxx           */ \
  "vshr.u8    d6, d7, #2                     \n" /* R xxxRRRRR           */ \
  "vshrn.u16  d5, q0, #5                     \n" /* G xxxGGGGG           */ \
  "vmovn.u16  d4, q0                         \n" /* B xxxBBBBB           */ \
  "vshr.u8    d7, d7, #7                     \n" /* A 0000000A           */ \
  "vneg.s8    d7, d7                         \n" /* A AAAAAAAA upper 8   */ \
  "vshl.u8    d6, d6, #3                     \n" /* R RRRRR000 upper 5   */ \
  "vshr.u8    q1, q3, #5                     \n" /* R,A 00000RRR lower 3 */ \
  "vshl.u8    q0, q2, #3                     \n" /* B,G BBBBB000 upper 5 */ \
  "vshr.u8    q2, q0, #5                     \n" /* B,G 00000BBB lower 3 */ \
  "vorr.u8    q1, q1, q3                     \n" /* R,A                  */ \
  "vorr.u8    q0, q0, q2                     \n" /* B,G                  */

// RGB555TOARGB is same as ARGB1555TOARGB but ignores alpha.
#define RGB555TOARGB                                                        \
  "vshrn.u16  d6, q0, #5                     \n" /* G xxxGGGGG           */ \
  "vuzp.u8    d0, d1                         \n" /* d0 xxxBBBBB xRRRRRxx */ \
  "vshl.u8    d6, d6, #3                     \n" /* G GGGGG000 upper 5   */ \
  "vshr.u8    d1, d1, #2                     \n" /* R 00xRRRRR lower 5   */ \
  "vshl.u8    q0, q0, #3                     \n" /* B,R BBBBB000 upper 5 */ \
  "vshr.u8    q2, q0, #5                     \n" /* B,R 00000BBB lower 3 */ \
  "vorr.u8    d0, d0, d4                     \n" /* B                    */ \
  "vshr.u8    d4, d6, #5                     \n" /* G 00000GGG lower 3   */ \
  "vorr.u8    d2, d1, d5                     \n" /* R                    */ \
  "vorr.u8    d1, d4, d6                     \n" /* G                    */

void ARGB1555ToARGBRow_NEON(const uint8_t* src_argb1555,
                            uint8_t* dst_argb,
                            int width) {
  asm volatile(
      "vmov.u8     d3, #255                      \n"  // Alpha
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 ARGB1555 pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      ARGB1555TOARGB
      "vst4.8      {d0, d1, d2, d3}, [%1]!       \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(src_argb1555),  // %0
        "+r"(dst_argb),      // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

#define ARGB4444TOARGB                                                      \
  "vuzp.u8    d0, d1                         \n" /* d0 BG, d1 RA         */ \
  "vshl.u8    q2, q0, #4                     \n" /* B,R BBBB0000         */ \
  "vshr.u8    q1, q0, #4                     \n" /* G,A 0000GGGG         */ \
  "vshr.u8    q0, q2, #4                     \n" /* B,R 0000BBBB         */ \
  "vorr.u8    q0, q0, q2                     \n" /* B,R BBBBBBBB         */ \
  "vshl.u8    q2, q1, #4                     \n" /* G,A GGGG0000         */ \
  "vorr.u8    q1, q1, q2                     \n" /* G,A GGGGGGGG         */ \
  "vswp.u8    d1, d2                         \n" /* B,R,G,A -> B,G,R,A   */

void ARGB4444ToARGBRow_NEON(const uint8_t* src_argb4444,
                            uint8_t* dst_argb,
                            int width) {
  asm volatile(
      "vmov.u8     d3, #255                      \n"  // Alpha
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 ARGB4444 pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      ARGB4444TOARGB
      "vst4.8      {d0, d1, d2, d3}, [%1]!       \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(src_argb4444),  // %0
        "+r"(dst_argb),      // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "q0", "q1", "q2"  // Clobber List
  );
}

void ARGBToRGB24Row_NEON(const uint8_t* src_argb,
                         uint8_t* dst_rgb24,
                         int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 16 pixels of ARGB.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"
      "subs        %2, %2, #16                   \n"  // 16 processed per loop.
      "vst3.8      {d0, d2, d4}, [%1]!           \n"  // store 16 RGB24 pixels.
      "vst3.8      {d1, d3, d5}, [%1]!           \n"
      "bgt         1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(dst_rgb24),  // %1
        "+r"(width)       // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
  );
}

void ARGBToRAWRow_NEON(const uint8_t* src_argb, uint8_t* dst_raw, int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d1, d2, d3, d4}, [%0]!       \n"  // load 8 pixels of ARGB.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vswp.u8     d1, d3                        \n"  // swap R, B
      "vst3.8      {d1, d2, d3}, [%1]!           \n"  // store 8 pixels of RAW.
      "bgt         1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_raw),   // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "d1", "d2", "d3", "d4"  // Clobber List
  );
}

void YUY2ToYRow_NEON(const uint8_t* src_yuy2, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "vld2.8      {q0, q1}, [%0]!               \n"  // load 16 pixels of YUY2.
      "subs        %2, %2, #16                   \n"  // 16 processed per loop.
      "vst1.8      {q0}, [%1]!                   \n"  // store 16 pixels of Y.
      "bgt         1b                            \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1"  // Clobber List
  );
}

void UYVYToYRow_NEON(const uint8_t* src_uyvy, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "vld2.8      {q0, q1}, [%0]!               \n"  // load 16 pixels of UYVY.
      "subs        %2, %2, #16                   \n"  // 16 processed per loop.
      "vst1.8      {q1}, [%1]!                   \n"  // store 16 pixels of Y.
      "bgt         1b                            \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1"  // Clobber List
  );
}

void YUY2ToUV422Row_NEON(const uint8_t* src_yuy2,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 16 pixels of YUY2.
      "subs        %3, %3, #16                   \n"  // 16 pixels = 8 UVs.
      "vst1.8      {d1}, [%1]!                   \n"  // store 8 U.
      "vst1.8      {d3}, [%2]!                   \n"  // store 8 V.
      "bgt         1b                            \n"
      : "+r"(src_yuy2),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "cc", "memory", "d0", "d1", "d2", "d3"  // Clobber List
  );
}

void UYVYToUV422Row_NEON(const uint8_t* src_uyvy,
                         uint8_t* dst_u,
                         uint8_t* dst_v,
                         int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 16 pixels of UYVY.
      "subs        %3, %3, #16                   \n"  // 16 pixels = 8 UVs.
      "vst1.8      {d0}, [%1]!                   \n"  // store 8 U.
      "vst1.8      {d2}, [%2]!                   \n"  // store 8 V.
      "bgt         1b                            \n"
      : "+r"(src_uyvy),  // %0
        "+r"(dst_u),     // %1
        "+r"(dst_v),     // %2
        "+r"(width)      // %3
      :
      : "cc", "memory", "d0", "d1", "d2", "d3"  // Clobber List
  );
}

void YUY2ToUVRow_NEON(const uint8_t* src_yuy2,
                      int stride_yuy2,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile(
      "add         %1, %0, %1                    \n"  // stride + src_yuy2
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 16 pixels of YUY2.
      "vld4.8      {d4, d5, d6, d7}, [%1]!       \n"  // load next row YUY2.
      "subs        %4, %4, #16                   \n"  // 16 pixels = 8 UVs.
      "vrhadd.u8   d1, d1, d5                    \n"  // average rows of U
      "vrhadd.u8   d3, d3, d7                    \n"  // average rows of V
      "vst1.8      {d1}, [%2]!                   \n"  // store 8 U.
      "vst1.8      {d3}, [%3]!                   \n"  // store 8 V.
      "bgt         1b                            \n"
      : "+r"(src_yuy2),     // %0
        "+r"(stride_yuy2),  // %1
        "+r"(dst_u),        // %2
        "+r"(dst_v),        // %3
        "+r"(width)         // %4
      :
      : "cc", "memory", "d0", "d1", "d2", "d3", "d4", "d5", "d6",
        "d7"  // Clobber List
  );
}

void UYVYToUVRow_NEON(const uint8_t* src_uyvy,
                      int stride_uyvy,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile(
      "add         %1, %0, %1                    \n"  // stride + src_uyvy
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 16 pixels of UYVY.
      "vld4.8      {d4, d5, d6, d7}, [%1]!       \n"  // load next row UYVY.
      "subs        %4, %4, #16                   \n"  // 16 pixels = 8 UVs.
      "vrhadd.u8   d0, d0, d4                    \n"  // average rows of U
      "vrhadd.u8   d2, d2, d6                    \n"  // average rows of V
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 U.
      "vst1.8      {d2}, [%3]!                   \n"  // store 8 V.
      "bgt         1b                            \n"
      : "+r"(src_uyvy),     // %0
        "+r"(stride_uyvy),  // %1
        "+r"(dst_u),        // %2
        "+r"(dst_v),        // %3
        "+r"(width)         // %4
      :
      : "cc", "memory", "d0", "d1", "d2", "d3", "d4", "d5", "d6",
        "d7"  // Clobber List
  );
}

void YUY2ToNVUVRow_NEON(const uint8_t* src_yuy2,
                        int stride_yuy2,
                        uint8_t* dst_uv,
                        int width) {
  asm volatile(
      "add         %1, %0, %1                    \n"  // stride + src_yuy2
      "1:          \n"
      "vld2.8      {q0, q1}, [%0]!               \n"  // load 16 pixels of YUY2.
      "subs        %3, %3, #16                   \n"  // 16 pixels = 8 UVs.
      "vld2.8      {q2, q3}, [%1]!               \n"  // load next row YUY2.
      "vrhadd.u8   q4, q1, q3                    \n"  // average rows of UV
      "vst1.8      {q4}, [%2]!                   \n"  // store 8 UV.
      "bgt         1b                            \n"
      : "+r"(src_yuy2),     // %0
        "+r"(stride_yuy2),  // %1
        "+r"(dst_uv),       // %2
        "+r"(width)         // %3
      :
      : "cc", "memory", "d0", "d1", "d2", "d3", "d4", "d5", "d6",
        "d7"  // Clobber List
  );
}

// For BGRAToARGB, ABGRToARGB, RGBAToARGB, and ARGBToRGBA.
void ARGBShuffleRow_NEON(const uint8_t* src_argb,
                         uint8_t* dst_argb,
                         const uint8_t* shuffler,
                         int width) {
  asm volatile(
      "vld1.8      {q2}, [%3]                    \n"  // shuffler
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 4 pixels.
      "subs        %2, %2, #4                    \n"  // 4 processed per loop
      "vtbl.8      d2, {d0, d1}, d4              \n"  // look up 2 first pixels
      "vtbl.8      d3, {d0, d1}, d5              \n"  // look up 2 next pixels
      "vst1.8      {q1}, [%1]!                   \n"  // store 4.
      "bgt         1b                            \n"
      : "+r"(src_argb),                   // %0
        "+r"(dst_argb),                   // %1
        "+r"(width)                       // %2
      : "r"(shuffler)                     // %3
      : "cc", "memory", "q0", "q1", "q2"  // Clobber List
  );
}

void I422ToYUY2Row_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_yuy2,
                        int width) {
  asm volatile(
      "1:          \n"
      "vld2.8      {d0, d2}, [%0]!               \n"  // load 16 Ys
      "vld1.8      {d1}, [%1]!                   \n"  // load 8 Us
      "vld1.8      {d3}, [%2]!                   \n"  // load 8 Vs
      "subs        %4, %4, #16                   \n"  // 16 pixels
      "vst4.8      {d0, d1, d2, d3}, [%3]!       \n"  // Store 8 YUY2/16 pixels.
      "bgt         1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(src_u),     // %1
        "+r"(src_v),     // %2
        "+r"(dst_yuy2),  // %3
        "+r"(width)      // %4
      :
      : "cc", "memory", "d0", "d1", "d2", "d3");
}

void I422ToUYVYRow_NEON(const uint8_t* src_y,
                        const uint8_t* src_u,
                        const uint8_t* src_v,
                        uint8_t* dst_uyvy,
                        int width) {
  asm volatile(
      "1:          \n"
      "vld2.8      {d1, d3}, [%0]!               \n"  // load 16 Ys
      "vld1.8      {d0}, [%1]!                   \n"  // load 8 Us
      "vld1.8      {d2}, [%2]!                   \n"  // load 8 Vs
      "subs        %4, %4, #16                   \n"  // 16 pixels
      "vst4.8      {d0, d1, d2, d3}, [%3]!       \n"  // Store 8 UYVY/16 pixels.
      "bgt         1b                            \n"
      : "+r"(src_y),     // %0
        "+r"(src_u),     // %1
        "+r"(src_v),     // %2
        "+r"(dst_uyvy),  // %3
        "+r"(width)      // %4
      :
      : "cc", "memory", "d0", "d1", "d2", "d3");
}

void ARGBToRGB565Row_NEON(const uint8_t* src_argb,
                          uint8_t* dst_rgb565,
                          int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 pixels of ARGB.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      ARGBTORGB565
      "vst1.8      {q2}, [%1]!                   \n"  // store 8 pixels RGB565.
      "bgt         1b                            \n"
      : "+r"(src_argb),    // %0
        "+r"(dst_rgb565),  // %1
        "+r"(width)        // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "d6");
}

void ARGBToRGB565DitherRow_NEON(const uint8_t* src_argb,
                                uint8_t* dst_rgb,
                                uint32_t dither4,
                                int width) {
  asm volatile(
      "vdup.32     d7, %2                        \n"  // dither4
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%1]!       \n"  // load 8 pixels of ARGB.
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vqadd.u8    d0, d0, d7                    \n"
      "vqadd.u8    d2, d2, d7                    \n"
      "vqadd.u8    d4, d4, d7                    \n"  // add for dither
      ARGBTORGB565
      "vst1.8      {q2}, [%0]!                   \n"  // store 8 RGB565.
      "bgt         1b                            \n"
      : "+r"(dst_rgb)   // %0
      : "r"(src_argb),  // %1
        "r"(dither4),   // %2
        "r"(width)      // %3
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

void ARGBToARGB1555Row_NEON(const uint8_t* src_argb,
                            uint8_t* dst_argb1555,
                            int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 pixels of ARGB.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      ARGBTOARGB1555
      "vst1.8      {q3}, [%1]!                   \n"  // store 8 ARGB1555.
      "bgt         1b                            \n"
      : "+r"(src_argb),      // %0
        "+r"(dst_argb1555),  // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

void ARGBToARGB4444Row_NEON(const uint8_t* src_argb,
                            uint8_t* dst_argb4444,
                            int width) {
  asm volatile(
      "vmov.u8     d7, #0x0f                     \n"  // bits to clear with
                                                      // vbic.
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 pixels of ARGB.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      ARGBTOARGB4444
      "vst1.8      {q0}, [%1]!                   \n"  // store 8 ARGB4444.
      "bgt         1b                            \n"
      : "+r"(src_argb),      // %0
        "+r"(dst_argb4444),  // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

void ARGBExtractAlphaRow_NEON(const uint8_t* src_argb,
                              uint8_t* dst_a,
                              int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ARGB pixels
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 ARGB pixels
      "subs        %2, %2, #16                   \n"  // 16 processed per loop
      "vst1.8      {q3}, [%1]!                   \n"  // store 16 A's.
      "bgt         1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_a),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3"  // Clobber List
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
      "vld1.8      {d0}, [%4]                    \n"  // load rgbuvconstants
      "vdup.u8     d24, d0[0]                    \n"  // UB  0.875  coefficient
      "vdup.u8     d25, d0[1]                    \n"  // UG -0.5781 coefficient
      "vdup.u8     d26, d0[2]                    \n"  // UR -0.2969 coefficient
      "vdup.u8     d27, d0[4]                    \n"  // VB -0.1406 coefficient
      "vdup.u8     d28, d0[5]                    \n"  // VG -0.7344 coefficient
      "vneg.s8     d24, d24                      \n"
      "vmov.u16    q15, #0x8000                  \n"  // 128.0

      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 8 ARGB pixels.
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vmull.u8    q2, d0, d24                   \n"  // B
      "vmlsl.u8    q2, d1, d25                   \n"  // G
      "vmlsl.u8    q2, d2, d26                   \n"  // R

      "vmull.u8    q3, d2, d24                   \n"  // R
      "vmlsl.u8    q3, d1, d28                   \n"  // G
      "vmlsl.u8    q3, d0, d27                   \n"  // B

      "vaddhn.u16  d0, q2, q15                   \n"  // signed -> unsigned
      "vaddhn.u16  d1, q3, q15                   \n"

      "vst1.8      {d0}, [%1]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%2]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
      : "+r"(src_argb),      // %0
        "+r"(dst_u),         // %1
        "+r"(dst_v),         // %2
        "+r"(width)          // %3
      : "r"(rgbuvconstants)  // %4
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q12", "q13", "q14",
        "q15");
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

// clang-format off
// 16x2 pixels -> 8x1.  width is number of argb pixels. e.g. 16.
#define RGBTOUV(QB, QG, QR)                                                 \
  "vmul.s16   q8, " #QB ", q10               \n" /* B                    */ \
  "vmls.s16   q8, " #QG ", q11               \n" /* G                    */ \
  "vmls.s16   q8, " #QR ", q12               \n" /* R                    */ \
  "vmul.s16   q9, " #QR ", q10               \n" /* R                    */ \
  "vmls.s16   q9, " #QG ", q14               \n" /* G                    */ \
  "vmls.s16   q9, " #QB ", q13               \n" /* B                    */ \
  "vaddhn.u16 d0, q8, q15                    \n" /* +128 -> unsigned     */ \
  "vaddhn.u16 d1, q9, q15                    \n" /* +128 -> unsigned     */
// clang-format on

// TODO(fbarchard): Consider vhadd vertical, then vpaddl horizontal, avoid shr.
void ARGBToUVRow_NEON(const uint8_t* src_argb,
                      int src_stride_argb,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_argb
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ARGB pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 ARGB pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q2                        \n"  // R 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more ARGB pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 ARGB pixels.
      "vpadal.u8   q0, q4                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q6                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q0, q1, q2)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_argb),  // %0
    "+r"(src_stride_argb),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void ARGBToUVJRow_NEON(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_argb
      "vmov.s16    q10, #128                     \n"  // UB/VR 0.500 coefficient
      "vmov.s16    q11, #85                      \n"  // UG -0.33126 coefficient
      "vmov.s16    q12, #43                      \n"  // UR -0.16874 coefficient
      "vmov.s16    q13, #21                      \n"  // VB -0.08131 coefficient
      "vmov.s16    q14, #107                     \n"  // VG -0.41869 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ARGB pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 ARGB pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q2                        \n"  // R 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more ARGB pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 ARGB pixels.
      "vpadal.u8   q0, q4                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q6                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q0, q1, q2)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_argb),  // %0
    "+r"(src_stride_argb),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void ABGRToUVJRow_NEON(const uint8_t* src_abgr,
                       int src_stride_abgr,
                       uint8_t* dst_uj,
                       uint8_t* dst_vj,
                       int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_argb
      "vmov.s16    q10, #128                     \n"  // UB/VR 0.500 coefficient
      "vmov.s16    q11, #85                      \n"  // UG -0.33126 coefficient
      "vmov.s16    q12, #43                      \n"  // UR -0.16874 coefficient
      "vmov.s16    q13, #21                      \n"  // VB -0.08131 coefficient
      "vmov.s16    q14, #107                     \n"  // VG -0.41869 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ABGR pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 ABGR pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // R 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q2                        \n"  // B 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more ABGR pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 ABGR pixels.
      "vpadal.u8   q0, q4                        \n"  // R 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q6                        \n"  // B 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q2, q1, q0)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_abgr),  // %0
    "+r"(src_stride_abgr),  // %1
    "+r"(dst_uj),     // %2
    "+r"(dst_vj),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void RGB24ToUVJRow_NEON(const uint8_t* src_rgb24,
                        int src_stride_rgb24,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_rgb24
      "vmov.s16    q10, #128                     \n"  // UB/VR 0.500 coefficient
      "vmov.s16    q11, #85                      \n"  // UG -0.33126 coefficient
      "vmov.s16    q12, #43                      \n"  // UR -0.16874 coefficient
      "vmov.s16    q13, #21                      \n"  // VB -0.08131 coefficient
      "vmov.s16    q14, #107                     \n"  // VG -0.41869 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld3.8      {d0, d2, d4}, [%0]!           \n"  // load 8 RGB24 pixels.
      "vld3.8      {d1, d3, d5}, [%0]!           \n"  // load next 8 RGB24 pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q2                        \n"  // R 16 bytes -> 8 shorts.
      "vld3.8      {d8, d10, d12}, [%1]!         \n"  // load 8 more RGB24 pixels.
      "vld3.8      {d9, d11, d13}, [%1]!         \n"  // load last 8 RGB24 pixels.
      "vpadal.u8   q0, q4                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q6                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q0, q1, q2)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_rgb24),  // %0
    "+r"(src_stride_rgb24),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void RAWToUVJRow_NEON(const uint8_t* src_raw,
                      int src_stride_raw,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_raw
      "vmov.s16    q10, #128                     \n"  // UB/VR 0.500 coefficient
      "vmov.s16    q11, #85                      \n"  // UG -0.33126 coefficient
      "vmov.s16    q12, #43                      \n"  // UR -0.16874 coefficient
      "vmov.s16    q13, #21                      \n"  // VB -0.08131 coefficient
      "vmov.s16    q14, #107                     \n"  // VG -0.41869 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld3.8      {d0, d2, d4}, [%0]!           \n"  // load 8 RAW pixels.
      "vld3.8      {d1, d3, d5}, [%0]!           \n"  // load next 8 RAW pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q2                        \n"  // R 16 bytes -> 8 shorts.
      "vld3.8      {d8, d10, d12}, [%1]!         \n"  // load 8 more RAW pixels.
      "vld3.8      {d9, d11, d13}, [%1]!         \n"  // load last 8 RAW pixels.
      "vpadal.u8   q0, q4                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q6                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q2, q1, q0)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_raw),  // %0
    "+r"(src_stride_raw),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void BGRAToUVRow_NEON(const uint8_t* src_bgra,
                      int src_stride_bgra,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_bgra
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 BGRA pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 BGRA pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q3, q3                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q2                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // R 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more BGRA pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 BGRA pixels.
      "vpadal.u8   q3, q7                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q6                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q1, q1, #2                    \n"  // average of 4
      "vrshr.u16   q2, q2, #2                    \n"
      "vrshr.u16   q3, q3, #2                    \n"

    RGBTOUV(q3, q2, q1)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_bgra),  // %0
    "+r"(src_stride_bgra),  // %1
    "+r"(dst_u),     // %2-
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void ABGRToUVRow_NEON(const uint8_t* src_abgr,
                      int src_stride_abgr,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_abgr
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ABGR pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 ABGR pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q2, q2                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q0, q0                        \n"  // R 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more ABGR pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 ABGR pixels.
      "vpadal.u8   q2, q6                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q0, q4                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q2, q1, q0)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_abgr),  // %0
    "+r"(src_stride_abgr),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void RGBAToUVRow_NEON(const uint8_t* src_rgba,
                      int src_stride_rgba,
                      uint8_t* dst_u,
                      uint8_t* dst_v,
                      int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_rgba
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 RGBA pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 RGBA pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q1                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q2                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q3                        \n"  // R 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more RGBA pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 RGBA pixels.
      "vpadal.u8   q0, q5                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q6                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q7                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q0, q1, q2)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_rgba),  // %0
    "+r"(src_stride_rgba),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void RGB24ToUVRow_NEON(const uint8_t* src_rgb24,
                       int src_stride_rgb24,
                       uint8_t* dst_u,
                       uint8_t* dst_v,
                       int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_rgb24
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld3.8      {d0, d2, d4}, [%0]!           \n"  // load 8 RGB24 pixels.
      "vld3.8      {d1, d3, d5}, [%0]!           \n"  // load next 8 RGB24 pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q2, q2                        \n"  // R 16 bytes -> 8 shorts.
      "vld3.8      {d8, d10, d12}, [%1]!         \n"  // load 8 more RGB24 pixels.
      "vld3.8      {d9, d11, d13}, [%1]!         \n"  // load last 8 RGB24 pixels.
      "vpadal.u8   q0, q4                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q2, q6                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q0, q1, q2)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_rgb24),  // %0
    "+r"(src_stride_rgb24),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

void RAWToUVRow_NEON(const uint8_t* src_raw,
                     int src_stride_raw,
                     uint8_t* dst_u,
                     uint8_t* dst_v,
                     int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_raw
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld3.8      {d0, d2, d4}, [%0]!           \n"  // load 8 RAW pixels.
      "vld3.8      {d1, d3, d5}, [%0]!           \n"  // load next 8 RAW pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q2, q2                        \n"  // B 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // G 16 bytes -> 8 shorts.
      "vpaddl.u8   q0, q0                        \n"  // R 16 bytes -> 8 shorts.
      "vld3.8      {d8, d10, d12}, [%1]!         \n"  // load 8 more RAW pixels.
      "vld3.8      {d9, d11, d13}, [%1]!         \n"  // load last 8 RAW pixels.
      "vpadal.u8   q2, q6                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vpadal.u8   q0, q4                        \n"  // R 16 bytes -> 8 shorts.

      "vrshr.u16   q0, q0, #2                    \n"  // average of 4
      "vrshr.u16   q1, q1, #2                    \n"
      "vrshr.u16   q2, q2, #2                    \n"

    RGBTOUV(q2, q1, q0)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
  : "+r"(src_raw),  // %0
    "+r"(src_stride_raw),  // %1
    "+r"(dst_u),     // %2
    "+r"(dst_v),     // %3
    "+r"(width)        // %4
  :
  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
  );
}

// 16x2 pixels -> 8x1.  width is number of argb pixels. e.g. 16.
void RGB565ToUVRow_NEON(const uint8_t* src_rgb565,
                        int src_stride_rgb565,
                        uint8_t* dst_u,
                        uint8_t* dst_v,
                        int width) {
  asm volatile(
      "add         %1, %0, %1                    \n"  // src_stride + src_argb
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 RGB565 pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      RGB565TOARGB
      "vpaddl.u8   d8, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpaddl.u8   d10, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpaddl.u8   d12, d2                       \n"  // R 8 bytes -> 4 shorts.
      "vld1.8      {q0}, [%0]!                   \n"  // next 8 RGB565 pixels.
      RGB565TOARGB
      "vpaddl.u8   d9, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpaddl.u8   d11, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpaddl.u8   d13, d2                       \n"  // R 8 bytes -> 4 shorts.

      "vld1.8      {q0}, [%1]!                   \n"  // load 8 RGB565 pixels.
      RGB565TOARGB
      "vpadal.u8   d8, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpadal.u8   d10, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpadal.u8   d12, d2                       \n"  // R 8 bytes -> 4 shorts.
      "vld1.8      {q0}, [%1]!                   \n"  // next 8 RGB565 pixels.
      RGB565TOARGB
      "vpadal.u8   d9, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpadal.u8   d11, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpadal.u8   d13, d2                       \n"  // R 8 bytes -> 4 shorts.

      "vrshr.u16   q4, q4, #2                    \n"  // average of 4
      "vrshr.u16   q5, q5, #2                    \n"
      "vrshr.u16   q6, q6, #2                    \n"

      "vmul.s16    q8, q4, q10                   \n"  // B
      "vmls.s16    q8, q5, q11                   \n"  // G
      "vmls.s16    q8, q6, q12                   \n"  // R
      "vadd.u16    q8, q8, q15                   \n"  // +128 -> unsigned
      "vmul.s16    q9, q6, q10                   \n"  // R
      "vmls.s16    q9, q5, q14                   \n"  // G
      "vmls.s16    q9, q4, q13                   \n"  // B
      "vadd.u16    q9, q9, q15                   \n"  // +128 -> unsigned
      "vqshrn.u16  d0, q8, #8                    \n"  // 16 bit to 8 bit U
      "vqshrn.u16  d1, q9, #8                    \n"  // 16 bit to 8 bit V
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
      : "+r"(src_rgb565),         // %0
        "+r"(src_stride_rgb565),  // %1
        "+r"(dst_u),              // %2
        "+r"(dst_v),              // %3
        "+r"(width)               // %4
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

// 16x2 pixels -> 8x1.  width is number of argb pixels. e.g. 16.
void ARGB1555ToUVRow_NEON(const uint8_t* src_argb1555,
                          int src_stride_argb1555,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  asm volatile(
      "add         %1, %0, %1                    \n"  // src_stride + src_argb
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 ARGB1555 pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      RGB555TOARGB
      "vpaddl.u8   d8, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpaddl.u8   d10, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpaddl.u8   d12, d2                       \n"  // R 8 bytes -> 4 shorts.
      "vld1.8      {q0}, [%0]!                   \n"  // next 8 ARGB1555 pixels.
      RGB555TOARGB
      "vpaddl.u8   d9, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpaddl.u8   d11, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpaddl.u8   d13, d2                       \n"  // R 8 bytes -> 4 shorts.

      "vld1.8      {q0}, [%1]!                   \n"  // load 8 ARGB1555 pixels.
      RGB555TOARGB
      "vpadal.u8   d8, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpadal.u8   d10, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpadal.u8   d12, d2                       \n"  // R 8 bytes -> 4 shorts.
      "vld1.8      {q0}, [%1]!                   \n"  // next 8 ARGB1555 pixels.
      RGB555TOARGB
      "vpadal.u8   d9, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpadal.u8   d11, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpadal.u8   d13, d2                       \n"  // R 8 bytes -> 4 shorts.

      "vrshr.u16   q4, q4, #2                    \n"  // average of 4
      "vrshr.u16   q5, q5, #2                    \n"
      "vrshr.u16   q6, q6, #2                    \n"

      "vmul.s16    q8, q4, q10                   \n"  // B
      "vmls.s16    q8, q5, q11                   \n"  // G
      "vmls.s16    q8, q6, q12                   \n"  // R
      "vadd.u16    q8, q8, q15                   \n"  // +128 -> unsigned
      "vmul.s16    q9, q6, q10                   \n"  // R
      "vmls.s16    q9, q5, q14                   \n"  // G
      "vmls.s16    q9, q4, q13                   \n"  // B
      "vadd.u16    q9, q9, q15                   \n"  // +128 -> unsigned
      "vqshrn.u16  d0, q8, #8                    \n"  // 16 bit to 8 bit U
      "vqshrn.u16  d1, q9, #8                    \n"  // 16 bit to 8 bit V
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
      : "+r"(src_argb1555),         // %0
        "+r"(src_stride_argb1555),  // %1
        "+r"(dst_u),                // %2
        "+r"(dst_v),                // %3
        "+r"(width)                 // %4
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

// 16x2 pixels -> 8x1.  width is number of argb pixels. e.g. 16.
void ARGB4444ToUVRow_NEON(const uint8_t* src_argb4444,
                          int src_stride_argb4444,
                          uint8_t* dst_u,
                          uint8_t* dst_v,
                          int width) {
  asm volatile (
      "add         %1, %0, %1                    \n"  // src_stride + src_argb
      "vmov.s16    q10, #112                     \n"  // UB/VR 0.875 coefficient
      "vmov.s16    q11, #74                      \n"  // UG -0.5781 coefficient
      "vmov.s16    q12, #38                      \n"  // UR -0.2969 coefficient
      "vmov.s16    q13, #18                      \n"  // VB -0.1406 coefficient
      "vmov.s16    q14, #94                      \n"  // VG -0.7344 coefficient
      "vmov.u16    q15, #0x8000                  \n"  // 128.0
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 ARGB4444 pixels.
      "subs        %4, %4, #16                   \n"  // 16 processed per loop.
      ARGB4444TOARGB
      "vpaddl.u8   d8, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpaddl.u8   d10, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpaddl.u8   d12, d2                       \n"  // R 8 bytes -> 4 shorts.
      "vld1.8      {q0}, [%0]!                   \n"  // next 8 ARGB4444 pixels.
      ARGB4444TOARGB
      "vpaddl.u8   d9, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpaddl.u8   d11, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpaddl.u8   d13, d2                       \n"  // R 8 bytes -> 4 shorts.

      "vld1.8      {q0}, [%1]!                   \n"  // load 8 ARGB4444 pixels.
      ARGB4444TOARGB
      "vpadal.u8   d8, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpadal.u8   d10, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpadal.u8   d12, d2                       \n"  // R 8 bytes -> 4 shorts.
      "vld1.8      {q0}, [%1]!                   \n"  // next 8 ARGB4444 pixels.
      ARGB4444TOARGB
      "vpadal.u8   d9, d0                        \n"  // B 8 bytes -> 4 shorts.
      "vpadal.u8   d11, d1                       \n"  // G 8 bytes -> 4 shorts.
      "vpadal.u8   d13, d2                       \n"  // R 8 bytes -> 4 shorts.

      "vrshr.u16   q0, q4, #2                    \n"  // average of 4
      "vrshr.u16   q1, q5, #2                    \n"
      "vrshr.u16   q2, q6, #2                    \n"

      RGBTOUV(q0, q1, q2)
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 pixels U.
      "vst1.8      {d1}, [%3]!                   \n"  // store 8 pixels V.
      "bgt         1b                            \n"
      : "+r"(src_argb4444),         // %0
        "+r"(src_stride_argb4444),  // %1
        "+r"(dst_u),                // %2
        "+r"(dst_v),                // %3
        "+r"(width)                 // %4
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
        "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}

void RGB565ToYRow_NEON(const uint8_t* src_rgb565, uint8_t* dst_y, int width) {
  asm volatile(
      "vmov.u8     d24, #25                      \n"  // B * 0.1016 coefficient
      "vmov.u8     d25, #129                     \n"  // G * 0.5078 coefficient
      "vmov.u8     d26, #66                      \n"  // R * 0.2578 coefficient
      "vmov.u8     d27, #16                      \n"  // Add 16 constant
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 RGB565 pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      RGB565TOARGB
      "vmull.u8    q2, d0, d24                   \n"  // B
      "vmlal.u8    q2, d1, d25                   \n"  // G
      "vmlal.u8    q2, d2, d26                   \n"  // R
      "vqrshrn.u16 d0, q2, #8                    \n"  // 16 bit to 8 bit Y
      "vqadd.u8    d0, d27                       \n"
      "vst1.8      {d0}, [%1]!                   \n"  // store 8 pixels Y.
      "bgt         1b                            \n"
      : "+r"(src_rgb565),  // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q12", "q13");
}

void ARGB1555ToYRow_NEON(const uint8_t* src_argb1555,
                         uint8_t* dst_y,
                         int width) {
  asm volatile(
      "vmov.u8     d24, #25                      \n"  // B * 0.1016 coefficient
      "vmov.u8     d25, #129                     \n"  // G * 0.5078 coefficient
      "vmov.u8     d26, #66                      \n"  // R * 0.2578 coefficient
      "vmov.u8     d27, #16                      \n"  // Add 16 constant
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 ARGB1555 pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      ARGB1555TOARGB
      "vmull.u8    q2, d0, d24                   \n"  // B
      "vmlal.u8    q2, d1, d25                   \n"  // G
      "vmlal.u8    q2, d2, d26                   \n"  // R
      "vqrshrn.u16 d0, q2, #8                    \n"  // 16 bit to 8 bit Y
      "vqadd.u8    d0, d27                       \n"
      "vst1.8      {d0}, [%1]!                   \n"  // store 8 pixels Y.
      "bgt         1b                            \n"
      : "+r"(src_argb1555),  // %0
        "+r"(dst_y),         // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q12", "q13");
}

void ARGB4444ToYRow_NEON(const uint8_t* src_argb4444,
                         uint8_t* dst_y,
                         int width) {
  asm volatile(
      "vmov.u8     d24, #25                      \n"  // B * 0.1016 coefficient
      "vmov.u8     d25, #129                     \n"  // G * 0.5078 coefficient
      "vmov.u8     d26, #66                      \n"  // R * 0.2578 coefficient
      "vmov.u8     d27, #16                      \n"  // Add 16 constant
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 8 ARGB4444 pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      ARGB4444TOARGB
      "vmull.u8    q2, d0, d24                   \n"  // B
      "vmlal.u8    q2, d1, d25                   \n"  // G
      "vmlal.u8    q2, d2, d26                   \n"  // R
      "vqrshrn.u16 d0, q2, #8                    \n"  // 16 bit to 8 bit Y
      "vqadd.u8    d0, d27                       \n"
      "vst1.8      {d0}, [%1]!                   \n"  // store 8 pixels Y.
      "bgt         1b                            \n"
      : "+r"(src_argb4444),  // %0
        "+r"(dst_y),         // %1
        "+r"(width)          // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q12", "q13");
}

void ARGBToAR64Row_NEON(const uint8_t* src_argb,
                        uint16_t* dst_ar64,
                        int width) {
  asm volatile(
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"
      "vld1.8      {q2}, [%0]!                   \n"
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vmov.u8     q1, q0                        \n"
      "vmov.u8     q3, q2                        \n"
      "vst2.8      {q0, q1}, [%1]!               \n"  // store 4 pixels
      "vst2.8      {q2, q3}, [%1]!               \n"  // store 4 pixels
      "bgt         1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_ar64),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

static const uvec8 kShuffleARGBToABGR = {2,  1, 0, 3,  6,  5,  4,  7,
                                         10, 9, 8, 11, 14, 13, 12, 15};

void ARGBToAB64Row_NEON(const uint8_t* src_argb,
                        uint16_t* dst_ab64,
                        int width) {
  asm volatile(
      "vld1.8      {q4}, [%3]                    \n"  // shuffler

      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"
      "vld1.8      {q2}, [%0]!                   \n"
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vtbl.8      d2, {d0, d1}, d8              \n"
      "vtbl.8      d3, {d0, d1}, d9              \n"
      "vtbl.8      d6, {d4, d5}, d8              \n"
      "vtbl.8      d7, {d4, d5}, d9              \n"
      "vmov.u8     q0, q1                        \n"
      "vmov.u8     q2, q3                        \n"
      "vst2.8      {q0, q1}, [%1]!               \n"  // store 4 pixels
      "vst2.8      {q2, q3}, [%1]!               \n"  // store 4 pixels
      "bgt         1b                            \n"
      : "+r"(src_argb),           // %0
        "+r"(dst_ab64),           // %1
        "+r"(width)               // %2
      : "r"(&kShuffleARGBToABGR)  // %3
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
}

void AR64ToARGBRow_NEON(const uint16_t* src_ar64,
                        uint8_t* dst_argb,
                        int width) {
  asm volatile(
      "1:          \n"
      "vld1.16     {q0}, [%0]!                   \n"
      "vld1.16     {q1}, [%0]!                   \n"
      "vld1.16     {q2}, [%0]!                   \n"
      "vld1.16     {q3}, [%0]!                   \n"
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vshrn.u16   d0, q0, #8                    \n"
      "vshrn.u16   d1, q1, #8                    \n"
      "vshrn.u16   d4, q2, #8                    \n"
      "vshrn.u16   d5, q3, #8                    \n"
      "vst1.8      {q0}, [%1]!                   \n"  // store 4 pixels
      "vst1.8      {q2}, [%1]!                   \n"  // store 4 pixels
      "bgt         1b                            \n"
      : "+r"(src_ar64),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

static const uvec8 kShuffleAB64ToARGB = {5, 3, 1, 7, 13, 11, 9, 15};

void AB64ToARGBRow_NEON(const uint16_t* src_ab64,
                        uint8_t* dst_argb,
                        int width) {
  asm volatile(
      "vld1.8      {d8}, [%3]                    \n"  // shuffler

      "1:          \n"
      "vld1.16     {q0}, [%0]!                   \n"
      "vld1.16     {q1}, [%0]!                   \n"
      "vld1.16     {q2}, [%0]!                   \n"
      "vld1.16     {q3}, [%0]!                   \n"
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vtbl.8      d0, {d0, d1}, d8              \n"
      "vtbl.8      d1, {d2, d3}, d8              \n"
      "vtbl.8      d4, {d4, d5}, d8              \n"
      "vtbl.8      d5, {d6, d7}, d8              \n"
      "vst1.8      {q0}, [%1]!                   \n"  // store 4 pixels
      "vst1.8      {q2}, [%1]!                   \n"  // store 4 pixels
      "bgt         1b                            \n"
      : "+r"(src_ab64),           // %0
        "+r"(dst_argb),           // %1
        "+r"(width)               // %2
      : "r"(&kShuffleAB64ToARGB)  // %3
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
}

struct RgbConstants {
  uint8_t kRGBToY[4];
  uint16_t kAddY;
};

// RGB to JPeg coefficients
// B * 0.1140 coefficient = 29
// G * 0.5870 coefficient = 150
// R * 0.2990 coefficient = 77
// Add 0.5
static const struct RgbConstants kRgb24JPEGConstants = {{29, 150, 77, 0},
                                                        0x0080};

static const struct RgbConstants kRawJPEGConstants = {{77, 150, 29, 0}, 0x0080};

// RGB to BT.601 coefficients
// B * 0.1016 coefficient = 25
// G * 0.5078 coefficient = 129
// R * 0.2578 coefficient = 66
// Add 16.5 = 0x1080

static const struct RgbConstants kRgb24I601Constants = {{25, 129, 66, 0},
                                                        0x1080};

static const struct RgbConstants kRawI601Constants = {{66, 129, 25, 0}, 0x1080};

// ARGB expects first 3 values to contain RGB and 4th value is ignored.
static void ARGBToYMatrixRow_NEON(const uint8_t* src_argb,
                                  uint8_t* dst_y,
                                  int width,
                                  const struct RgbConstants* rgbconstants) {
  asm volatile(
      "vld1.8      {d0}, [%3]                    \n"  // load rgbconstants
      "vdup.u8     d20, d0[0]                    \n"
      "vdup.u8     d21, d0[1]                    \n"
      "vdup.u8     d22, d0[2]                    \n"
      "vdup.u16    q12, d0[2]                    \n"
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 16 pixels of ARGB
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"
      "subs        %2, %2, #16                   \n"  // 16 processed per loop.
      "vmull.u8    q8, d0, d20                   \n"  // B
      "vmull.u8    q9, d1, d20                   \n"
      "vmlal.u8    q8, d2, d21                   \n"  // G
      "vmlal.u8    q9, d3, d21                   \n"
      "vmlal.u8    q8, d4, d22                   \n"  // R
      "vmlal.u8    q9, d5, d22                   \n"
      "vaddhn.u16  d0, q8, q12                   \n"  // 16 bit to 8 bit Y
      "vaddhn.u16  d1, q9, q12                   \n"
      "vst1.8      {d0, d1}, [%1]!               \n"  // store 16 pixels Y.
      "bgt         1b                            \n"
      : "+r"(src_argb),    // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      : "r"(rgbconstants)  // %3
      : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "d20", "d21", "d22",
        "q12");
}

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

// RGBA expects first value to be A and ignored, then 3 values to contain RGB.
// Same code as ARGB, except the LD4
static void RGBAToYMatrixRow_NEON(const uint8_t* src_rgba,
                                  uint8_t* dst_y,
                                  int width,
                                  const struct RgbConstants* rgbconstants) {
  asm volatile(
      "vld1.8      {d0}, [%3]                    \n"  // load rgbconstants
      "vdup.u8     d20, d0[0]                    \n"
      "vdup.u8     d21, d0[1]                    \n"
      "vdup.u8     d22, d0[2]                    \n"
      "vdup.u16    q12, d0[2]                    \n"
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 16 pixels of RGBA
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"
      "subs        %2, %2, #16                   \n"  // 16 processed per loop.
      "vmull.u8    q8, d2, d20                   \n"  // B
      "vmull.u8    q9, d3, d20                   \n"
      "vmlal.u8    q8, d4, d21                   \n"  // G
      "vmlal.u8    q9, d5, d21                   \n"
      "vmlal.u8    q8, d6, d22                   \n"  // R
      "vmlal.u8    q9, d7, d22                   \n"
      "vaddhn.u16  d0, q8, q12                   \n"  // 16 bit to 8 bit Y
      "vaddhn.u16  d1, q9, q12                   \n"
      "vst1.8      {d0, d1}, [%1]!               \n"  // store 16 pixels Y.
      "bgt         1b                            \n"
      : "+r"(src_rgba),    // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      : "r"(rgbconstants)  // %3
      : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "d20", "d21", "d22",
        "q12");
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

static void RGBToYMatrixRow_NEON(const uint8_t* src_rgb,
                                 uint8_t* dst_y,
                                 int width,
                                 const struct RgbConstants* rgbconstants) {
  asm volatile(
      "vld1.8      {d0}, [%3]                    \n"  // load rgbconstants
      "vdup.u8     d20, d0[0]                    \n"
      "vdup.u8     d21, d0[1]                    \n"
      "vdup.u8     d22, d0[2]                    \n"
      "vdup.u16    q12, d0[2]                    \n"
      "1:          \n"
      "vld3.8      {d2, d4, d6}, [%0]!           \n"  // load 16 pixels of
                                                      // RGB24.
      "vld3.8      {d3, d5, d7}, [%0]!           \n"
      "subs        %2, %2, #16                   \n"  // 16 processed per loop.
      "vmull.u8    q8, d2, d20                   \n"  // B
      "vmull.u8    q9, d3, d20                   \n"
      "vmlal.u8    q8, d4, d21                   \n"  // G
      "vmlal.u8    q9, d5, d21                   \n"
      "vmlal.u8    q8, d6, d22                   \n"  // R
      "vmlal.u8    q9, d7, d22                   \n"
      "vaddhn.u16  d0, q8, q12                   \n"  // 16 bit to 8 bit Y
      "vaddhn.u16  d1, q9, q12                   \n"
      "vst1.8      {d0, d1}, [%1]!               \n"  // store 16 pixels Y.
      "bgt         1b                            \n"
      : "+r"(src_rgb),     // %0
        "+r"(dst_y),       // %1
        "+r"(width)        // %2
      : "r"(rgbconstants)  // %3
      : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "d20", "d21", "d22",
        "q12");
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
  asm volatile(
      "cmp         %4, #0                        \n"
      "beq         100f                          \n"
      "add         %2, %1                        \n"
      "cmp         %4, #128                      \n"
      "beq         50f                           \n"

      "vdup.8      d5, %4                        \n"
      "rsb         %4, #256                      \n"
      "vdup.8      d4, %4                        \n"
      // General purpose row blend.
      "1:          \n"
      "vld1.8      {q0}, [%1]!                   \n"
      "vld1.8      {q1}, [%2]!                   \n"
      "subs        %3, %3, #16                   \n"
      "vmull.u8    q13, d0, d4                   \n"
      "vmull.u8    q14, d1, d4                   \n"
      "vmlal.u8    q13, d2, d5                   \n"
      "vmlal.u8    q14, d3, d5                   \n"
      "vrshrn.u16  d0, q13, #8                   \n"
      "vrshrn.u16  d1, q14, #8                   \n"
      "vst1.8      {q0}, [%0]!                   \n"
      "bgt         1b                            \n"
      "b           99f                           \n"

      // Blend 50 / 50.
      "50:         \n"
      "vld1.8      {q0}, [%1]!                   \n"
      "vld1.8      {q1}, [%2]!                   \n"
      "subs        %3, %3, #16                   \n"
      "vrhadd.u8   q0, q1                        \n"
      "vst1.8      {q0}, [%0]!                   \n"
      "bgt         50b                           \n"
      "b           99f                           \n"

      // Blend 100 / 0 - Copy row unchanged.
      "100:        \n"
      "vld1.8      {q0}, [%1]!                   \n"
      "subs        %3, %3, #16                   \n"
      "vst1.8      {q0}, [%0]!                   \n"
      "bgt         100b                          \n"

      "99:         \n"
      : "+r"(dst_ptr),     // %0
        "+r"(src_ptr),     // %1
        "+r"(src_stride),  // %2
        "+r"(dst_width),   // %3
        "+r"(y1_fraction)  // %4
      :
      : "cc", "memory", "q0", "q1", "d4", "d5", "q13", "q14");
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
      "cmp         %4, #0                        \n"
      "beq         100f                          \n"
      "cmp         %4, #128                      \n"
      "beq         50f                           \n"

      "vdup.16     d17, %4                       \n"
      "vdup.16     d16, %5                       \n"
      // General purpose row blend.
      "1:          \n"
      "vld1.16     {q0}, [%1]!                   \n"
      "vld1.16     {q1}, [%2]!                   \n"
      "subs        %3, %3, #8                    \n"
      "vmull.u16   q2, d0, d16                   \n"
      "vmull.u16   q3, d1, d16                   \n"
      "vmlal.u16   q2, d2, d17                   \n"
      "vmlal.u16   q3, d3, d17                   \n"
      "vrshrn.u32  d0, q2, #8                    \n"
      "vrshrn.u32  d1, q3, #8                    \n"
      "vst1.16     {q0}, [%0]!                   \n"
      "bgt         1b                            \n"
      "b           99f                           \n"

      // Blend 50 / 50.
      "50:         \n"
      "vld1.16     {q0}, [%1]!                   \n"
      "vld1.16     {q1}, [%2]!                   \n"
      "subs        %3, %3, #8                    \n"
      "vrhadd.u16  q0, q1                        \n"
      "vst1.16     {q0}, [%0]!                   \n"
      "bgt         50b                           \n"
      "b           99f                           \n"

      // Blend 100 / 0 - Copy row unchanged.
      "100:        \n"
      "vld1.16     {q0}, [%1]!                   \n"
      "subs        %3, %3, #8                    \n"
      "vst1.16     {q0}, [%0]!                   \n"
      "bgt         100b                          \n"

      "99:         \n"
      : "+r"(dst_ptr),     // %0
        "+r"(src_ptr),     // %1
        "+r"(src_ptr1),    // %2
        "+r"(dst_width)    // %3
      : "r"(y1_fraction),  // %4
        "r"(y0_fraction)   // %5
      : "cc", "memory", "q0", "q1", "q2", "q3", "q8");
}

// dr * (256 - sa) / 256 + sr = dr - dr * sa / 256 + sr
void ARGBBlendRow_NEON(const uint8_t* src_argb,
                       const uint8_t* src_argb1,
                       uint8_t* dst_argb,
                       int width) {
  asm volatile(
      "subs        %3, #8                        \n"
      "blt         89f                           \n"
      // Blend 8 pixels.
      "8:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 8 pixels of ARGB0.
      "vld4.8      {d4, d5, d6, d7}, [%1]!       \n"  // load 8 pixels of ARGB1.
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vmull.u8    q10, d4, d3                   \n"  // db * a
      "vmull.u8    q11, d5, d3                   \n"  // dg * a
      "vmull.u8    q12, d6, d3                   \n"  // dr * a
      "vqrshrn.u16 d20, q10, #8                  \n"  // db >>= 8
      "vqrshrn.u16 d21, q11, #8                  \n"  // dg >>= 8
      "vqrshrn.u16 d22, q12, #8                  \n"  // dr >>= 8
      "vqsub.u8    q2, q2, q10                   \n"  // dbg - dbg * a / 256
      "vqsub.u8    d6, d6, d22                   \n"  // dr - dr * a / 256
      "vqadd.u8    q0, q0, q2                    \n"  // + sbg
      "vqadd.u8    d2, d2, d6                    \n"  // + sr
      "vmov.u8     d3, #255                      \n"  // a = 255
      "vst4.8      {d0, d1, d2, d3}, [%2]!       \n"  // store 8 pixels of ARGB.
      "bge         8b                            \n"

      "89:         \n"
      "adds        %3, #8-1                      \n"
      "blt         99f                           \n"

      // Blend 1 pixels.
      "1:          \n"
      "vld4.8      {d0[0],d1[0],d2[0],d3[0]}, [%0]! \n"  // load 1 pixel ARGB0.
      "vld4.8      {d4[0],d5[0],d6[0],d7[0]}, [%1]! \n"  // load 1 pixel ARGB1.
      "subs        %3, %3, #1                    \n"  // 1 processed per loop.
      "vmull.u8    q10, d4, d3                   \n"  // db * a
      "vmull.u8    q11, d5, d3                   \n"  // dg * a
      "vmull.u8    q12, d6, d3                   \n"  // dr * a
      "vqrshrn.u16 d20, q10, #8                  \n"  // db >>= 8
      "vqrshrn.u16 d21, q11, #8                  \n"  // dg >>= 8
      "vqrshrn.u16 d22, q12, #8                  \n"  // dr >>= 8
      "vqsub.u8    q2, q2, q10                   \n"  // dbg - dbg * a / 256
      "vqsub.u8    d6, d6, d22                   \n"  // dr - dr * a / 256
      "vqadd.u8    q0, q0, q2                    \n"  // + sbg
      "vqadd.u8    d2, d2, d6                    \n"  // + sr
      "vmov.u8     d3, #255                      \n"  // a = 255
      "vst4.8      {d0[0],d1[0],d2[0],d3[0]}, [%2]! \n"  // store 1 pixel.
      "bge         1b                            \n"

      "99:         \n"

      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q11", "q12");
}

// Attenuate 8 pixels at a time.
void ARGBAttenuateRow_NEON(const uint8_t* src_argb,
                           uint8_t* dst_argb,
                           int width) {
  asm volatile(
      "vmov.u16    q15, #0x00ff                  \n"  // 255 for rounding up

      // Attenuate 8 pixels.
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 8 pixels of ARGB.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vmull.u8    q10, d0, d3                   \n"  // b * a
      "vmull.u8    q11, d1, d3                   \n"  // g * a
      "vmull.u8    q12, d2, d3                   \n"  // r * a
      "vaddhn.u16  d0, q10, q15                  \n"  // (b + 255) >> 8
      "vaddhn.u16  d1, q11, q15                  \n"  // (g + 255) >> 8
      "vaddhn.u16  d2, q12, q15                  \n"  // (r + 255) >> 8
      "vst4.8      {d0, d1, d2, d3}, [%1]!       \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1", "q10", "q11", "q12", "q15");
}

// Quantize 8 ARGB pixels (32 bytes).
// dst = (dst * scale >> 16) * interval_size + interval_offset;
void ARGBQuantizeRow_NEON(uint8_t* dst_argb,
                          int scale,
                          int interval_size,
                          int interval_offset,
                          int width) {
  asm volatile(
      "vdup.u16    q8, %2                        \n"
      "vshr.u16    q8, q8, #1                    \n"  // scale >>= 1
      "vdup.u16    q9, %3                        \n"  // interval multiply.
      "vdup.u16    q10, %4                       \n"  // interval add

      // 8 pixel loop.
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]        \n"  // load 8 pixels of ARGB.
      "subs        %1, %1, #8                    \n"  // 8 processed per loop.
      "vmovl.u8    q0, d0                        \n"  // b (0 .. 255)
      "vmovl.u8    q1, d2                        \n"
      "vmovl.u8    q2, d4                        \n"
      "vqdmulh.s16 q0, q0, q8                    \n"  // b * scale
      "vqdmulh.s16 q1, q1, q8                    \n"  // g
      "vqdmulh.s16 q2, q2, q8                    \n"  // r
      "vmul.u16    q0, q0, q9                    \n"  // b * interval_size
      "vmul.u16    q1, q1, q9                    \n"  // g
      "vmul.u16    q2, q2, q9                    \n"  // r
      "vadd.u16    q0, q0, q10                   \n"  // b + interval_offset
      "vadd.u16    q1, q1, q10                   \n"  // g
      "vadd.u16    q2, q2, q10                   \n"  // r
      "vqmovn.u16  d0, q0                        \n"
      "vqmovn.u16  d2, q1                        \n"
      "vqmovn.u16  d4, q2                        \n"
      "vst4.8      {d0, d2, d4, d6}, [%0]!       \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(dst_argb),       // %0
        "+r"(width)           // %1
      : "r"(scale),           // %2
        "r"(interval_size),   // %3
        "r"(interval_offset)  // %4
      : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
}

// Shade 8 pixels at a time by specified value.
// NOTE vqrdmulh.s16 q10, q10, d0[0] must use a scaler register from 0 to 8.
// Rounding in vqrdmulh does +1 to high if high bit of low s16 is set.
void ARGBShadeRow_NEON(const uint8_t* src_argb,
                       uint8_t* dst_argb,
                       int width,
                       uint32_t value) {
  asm volatile(
      "vdup.u32    q0, %3                        \n"  // duplicate scale value.
      "vzip.u8     d0, d1                        \n"  // d0 aarrggbb.
      "vshr.u16    q0, q0, #1                    \n"  // scale / 2.

      // 8 pixel loop.
      "1:          \n"
      "vld4.8      {d20, d22, d24, d26}, [%0]!   \n"  // load 8 pixels of ARGB.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vmovl.u8    q10, d20                      \n"  // b (0 .. 255)
      "vmovl.u8    q11, d22                      \n"
      "vmovl.u8    q12, d24                      \n"
      "vmovl.u8    q13, d26                      \n"
      "vqrdmulh.s16 q10, q10, d0[0]              \n"  // b * scale * 2
      "vqrdmulh.s16 q11, q11, d0[1]              \n"  // g
      "vqrdmulh.s16 q12, q12, d0[2]              \n"  // r
      "vqrdmulh.s16 q13, q13, d0[3]              \n"  // a
      "vqmovn.u16  d20, q10                      \n"
      "vqmovn.u16  d22, q11                      \n"
      "vqmovn.u16  d24, q12                      \n"
      "vqmovn.u16  d26, q13                      \n"
      "vst4.8      {d20, d22, d24, d26}, [%1]!   \n"  // store 8 pixels of ARGB.
      "bgt         1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      : "r"(value)       // %3
      : "cc", "memory", "q0", "q10", "q11", "q12", "q13");
}

// Convert 8 ARGB pixels (64 bytes) to 8 Gray ARGB pixels
// Similar to ARGBToYJ but stores ARGB.
// C code is (29 * b + 150 * g + 77 * r + 128) >> 8;
void ARGBGrayRow_NEON(const uint8_t* src_argb, uint8_t* dst_argb, int width) {
  asm volatile(
      "vmov.u8     d24, #29                      \n"  // B * 0.1140 coefficient
      "vmov.u8     d25, #150                     \n"  // G * 0.5870 coefficient
      "vmov.u8     d26, #77                      \n"  // R * 0.2990 coefficient
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 8 ARGB pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vmull.u8    q2, d0, d24                   \n"  // B
      "vmlal.u8    q2, d1, d25                   \n"  // G
      "vmlal.u8    q2, d2, d26                   \n"  // R
      "vqrshrn.u16 d0, q2, #8                    \n"  // 16 bit to 8 bit B
      "vmov        d1, d0                        \n"  // G
      "vmov        d2, d0                        \n"  // R
      "vst4.8      {d0, d1, d2, d3}, [%1]!       \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(src_argb),  // %0
        "+r"(dst_argb),  // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q12", "q13");
}

// Convert 8 ARGB pixels (32 bytes) to 8 Sepia ARGB pixels.
//    b = (r * 35 + g * 68 + b * 17) >> 7
//    g = (r * 45 + g * 88 + b * 22) >> 7
//    r = (r * 50 + g * 98 + b * 24) >> 7
void ARGBSepiaRow_NEON(uint8_t* dst_argb, int width) {
  asm volatile(
      "vmov.u8     d20, #17                      \n"  // BB coefficient
      "vmov.u8     d21, #68                      \n"  // BG coefficient
      "vmov.u8     d22, #35                      \n"  // BR coefficient
      "vmov.u8     d24, #22                      \n"  // GB coefficient
      "vmov.u8     d25, #88                      \n"  // GG coefficient
      "vmov.u8     d26, #45                      \n"  // GR coefficient
      "vmov.u8     d28, #24                      \n"  // BB coefficient
      "vmov.u8     d29, #98                      \n"  // BG coefficient
      "vmov.u8     d30, #50                      \n"  // BR coefficient
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]        \n"  // load 8 ARGB pixels.
      "subs        %1, %1, #8                    \n"  // 8 processed per loop.
      "vmull.u8    q2, d0, d20                   \n"  // B to Sepia B
      "vmlal.u8    q2, d1, d21                   \n"  // G
      "vmlal.u8    q2, d2, d22                   \n"  // R
      "vmull.u8    q3, d0, d24                   \n"  // B to Sepia G
      "vmlal.u8    q3, d1, d25                   \n"  // G
      "vmlal.u8    q3, d2, d26                   \n"  // R
      "vmull.u8    q8, d0, d28                   \n"  // B to Sepia R
      "vmlal.u8    q8, d1, d29                   \n"  // G
      "vmlal.u8    q8, d2, d30                   \n"  // R
      "vqshrn.u16  d0, q2, #7                    \n"  // 16 bit to 8 bit B
      "vqshrn.u16  d1, q3, #7                    \n"  // 16 bit to 8 bit G
      "vqshrn.u16  d2, q8, #7                    \n"  // 16 bit to 8 bit R
      "vst4.8      {d0, d1, d2, d3}, [%0]!       \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(dst_argb),  // %0
        "+r"(width)      // %1
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q10", "q11", "q12", "q13",
        "q14", "q15");
}

// Tranform 8 ARGB pixels (32 bytes) with color matrix.
// TODO(fbarchard): Was same as Sepia except matrix is provided.  This function
// needs to saturate.  Consider doing a non-saturating version.
void ARGBColorMatrixRow_NEON(const uint8_t* src_argb,
                             uint8_t* dst_argb,
                             const int8_t* matrix_argb,
                             int width) {
  asm volatile(
      "vld1.8      {q2}, [%3]                    \n"  // load 3 ARGB vectors.
      "vmovl.s8    q0, d4                        \n"  // B,G coefficients s16.
      "vmovl.s8    q1, d5                        \n"  // R,A coefficients s16.

      "1:          \n"
      "vld4.8      {d16, d18, d20, d22}, [%0]!   \n"  // load 8 ARGB pixels.
      "subs        %2, %2, #8                    \n"  // 8 processed per loop.
      "vmovl.u8    q8, d16                       \n"  // b (0 .. 255) 16 bit
      "vmovl.u8    q9, d18                       \n"  // g
      "vmovl.u8    q10, d20                      \n"  // r
      "vmovl.u8    q11, d22                      \n"  // a
      "vmul.s16    q12, q8, d0[0]                \n"  // B = B * Matrix B
      "vmul.s16    q13, q8, d1[0]                \n"  // G = B * Matrix G
      "vmul.s16    q14, q8, d2[0]                \n"  // R = B * Matrix R
      "vmul.s16    q15, q8, d3[0]                \n"  // A = B * Matrix A
      "vmul.s16    q4, q9, d0[1]                 \n"  // B += G * Matrix B
      "vmul.s16    q5, q9, d1[1]                 \n"  // G += G * Matrix G
      "vmul.s16    q6, q9, d2[1]                 \n"  // R += G * Matrix R
      "vmul.s16    q7, q9, d3[1]                 \n"  // A += G * Matrix A
      "vqadd.s16   q12, q12, q4                  \n"  // Accumulate B
      "vqadd.s16   q13, q13, q5                  \n"  // Accumulate G
      "vqadd.s16   q14, q14, q6                  \n"  // Accumulate R
      "vqadd.s16   q15, q15, q7                  \n"  // Accumulate A
      "vmul.s16    q4, q10, d0[2]                \n"  // B += R * Matrix B
      "vmul.s16    q5, q10, d1[2]                \n"  // G += R * Matrix G
      "vmul.s16    q6, q10, d2[2]                \n"  // R += R * Matrix R
      "vmul.s16    q7, q10, d3[2]                \n"  // A += R * Matrix A
      "vqadd.s16   q12, q12, q4                  \n"  // Accumulate B
      "vqadd.s16   q13, q13, q5                  \n"  // Accumulate G
      "vqadd.s16   q14, q14, q6                  \n"  // Accumulate R
      "vqadd.s16   q15, q15, q7                  \n"  // Accumulate A
      "vmul.s16    q4, q11, d0[3]                \n"  // B += A * Matrix B
      "vmul.s16    q5, q11, d1[3]                \n"  // G += A * Matrix G
      "vmul.s16    q6, q11, d2[3]                \n"  // R += A * Matrix R
      "vmul.s16    q7, q11, d3[3]                \n"  // A += A * Matrix A
      "vqadd.s16   q12, q12, q4                  \n"  // Accumulate B
      "vqadd.s16   q13, q13, q5                  \n"  // Accumulate G
      "vqadd.s16   q14, q14, q6                  \n"  // Accumulate R
      "vqadd.s16   q15, q15, q7                  \n"  // Accumulate A
      "vqshrun.s16 d16, q12, #6                  \n"  // 16 bit to 8 bit B
      "vqshrun.s16 d18, q13, #6                  \n"  // 16 bit to 8 bit G
      "vqshrun.s16 d20, q14, #6                  \n"  // 16 bit to 8 bit R
      "vqshrun.s16 d22, q15, #6                  \n"  // 16 bit to 8 bit A
      "vst4.8      {d16, d18, d20, d22}, [%1]!   \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(dst_argb),   // %1
        "+r"(width)       // %2
      : "r"(matrix_argb)  // %3
      : "cc", "memory", "q0", "q1", "q2", "q4", "q5", "q6", "q7", "q8", "q9",
        "q10", "q11", "q12", "q13", "q14", "q15");
}

// Multiply 2 rows of ARGB pixels together, 8 pixels at a time.
void ARGBMultiplyRow_NEON(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      // 8 pixel loop.
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 ARGB pixels.
      "vld4.8      {d1, d3, d5, d7}, [%1]!       \n"  // load 8 more ARGB
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vmull.u8    q0, d0, d1                    \n"  // multiply B
      "vmull.u8    q1, d2, d3                    \n"  // multiply G
      "vmull.u8    q2, d4, d5                    \n"  // multiply R
      "vmull.u8    q3, d6, d7                    \n"  // multiply A
      "vrshrn.u16  d0, q0, #8                    \n"  // 16 bit to 8 bit B
      "vrshrn.u16  d1, q1, #8                    \n"  // 16 bit to 8 bit G
      "vrshrn.u16  d2, q2, #8                    \n"  // 16 bit to 8 bit R
      "vrshrn.u16  d3, q3, #8                    \n"  // 16 bit to 8 bit A
      "vst4.8      {d0, d1, d2, d3}, [%2]!       \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

// Add 2 rows of ARGB pixels together, 8 pixels at a time.
void ARGBAddRow_NEON(const uint8_t* src_argb,
                     const uint8_t* src_argb1,
                     uint8_t* dst_argb,
                     int width) {
  asm volatile(
      // 8 pixel loop.
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 8 ARGB pixels.
      "vld4.8      {d4, d5, d6, d7}, [%1]!       \n"  // load 8 more ARGB
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vqadd.u8    q0, q0, q2                    \n"  // add B, G
      "vqadd.u8    q1, q1, q3                    \n"  // add R, A
      "vst4.8      {d0, d1, d2, d3}, [%2]!       \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

// Subtract 2 rows of ARGB pixels, 8 pixels at a time.
void ARGBSubtractRow_NEON(const uint8_t* src_argb,
                          const uint8_t* src_argb1,
                          uint8_t* dst_argb,
                          int width) {
  asm volatile(
      // 8 pixel loop.
      "1:          \n"
      "vld4.8      {d0, d1, d2, d3}, [%0]!       \n"  // load 8 ARGB pixels.
      "vld4.8      {d4, d5, d6, d7}, [%1]!       \n"  // load 8 more ARGB
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vqsub.u8    q0, q0, q2                    \n"  // subtract B, G
      "vqsub.u8    q1, q1, q3                    \n"  // subtract R, A
      "vst4.8      {d0, d1, d2, d3}, [%2]!       \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(src_argb),   // %0
        "+r"(src_argb1),  // %1
        "+r"(dst_argb),   // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
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
      "vmov.u8     d3, #255                      \n"  // alpha
      // 8 pixel loop.
      "1:          \n"
      "vld1.8      {d0}, [%0]!                   \n"  // load 8 sobelx.
      "vld1.8      {d1}, [%1]!                   \n"  // load 8 sobely.
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vqadd.u8    d0, d0, d1                    \n"  // add
      "vmov.u8     d1, d0                        \n"
      "vmov.u8     d2, d0                        \n"
      "vst4.8      {d0, d1, d2, d3}, [%2]!       \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_argb),    // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "q0", "q1");
}

// Adds Sobel X and Sobel Y and stores Sobel into plane.
void SobelToPlaneRow_NEON(const uint8_t* src_sobelx,
                          const uint8_t* src_sobely,
                          uint8_t* dst_y,
                          int width) {
  asm volatile(
      // 16 pixel loop.
      "1:          \n"
      "vld1.8      {q0}, [%0]!                   \n"  // load 16 sobelx.
      "vld1.8      {q1}, [%1]!                   \n"  // load 16 sobely.
      "subs        %3, %3, #16                   \n"  // 16 processed per loop.
      "vqadd.u8    q0, q0, q1                    \n"  // add
      "vst1.8      {q0}, [%2]!                   \n"  // store 16 pixels.
      "bgt         1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_y),       // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "q0", "q1");
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
      "vmov.u8     d3, #255                      \n"  // alpha
      // 8 pixel loop.
      "1:          \n"
      "vld1.8      {d2}, [%0]!                   \n"  // load 8 sobelx.
      "vld1.8      {d0}, [%1]!                   \n"  // load 8 sobely.
      "subs        %3, %3, #8                    \n"  // 8 processed per loop.
      "vqadd.u8    d1, d0, d2                    \n"  // add
      "vst4.8      {d0, d1, d2, d3}, [%2]!       \n"  // store 8 ARGB pixels.
      "bgt         1b                            \n"
      : "+r"(src_sobelx),  // %0
        "+r"(src_sobely),  // %1
        "+r"(dst_argb),    // %2
        "+r"(width)        // %3
      :
      : "cc", "memory", "q0", "q1");
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
      "vld1.8      {d0}, [%0],%5                 \n"  // top
      "vld1.8      {d1}, [%0],%6                 \n"
      "subs        %4, %4, #8                    \n"  // 8 pixels
      "vsubl.u8    q0, d0, d1                    \n"
      "vld1.8      {d2}, [%1],%5                 \n"  // center * 2
      "vld1.8      {d3}, [%1],%6                 \n"
      "vsubl.u8    q1, d2, d3                    \n"
      "vadd.s16    q0, q0, q1                    \n"
      "vadd.s16    q0, q0, q1                    \n"
      "vld1.8      {d2}, [%2],%5                 \n"  // bottom
      "vld1.8      {d3}, [%2],%6                 \n"
      "vsubl.u8    q1, d2, d3                    \n"
      "vadd.s16    q0, q0, q1                    \n"
      "vabs.s16    q0, q0                        \n"
      "vqmovn.u16  d0, q0                        \n"
      "vst1.8      {d0}, [%3]!                   \n"  // store 8 sobelx
      "bgt         1b                            \n"
      : "+r"(src_y0),               // %0
        "+r"(src_y1),               // %1
        "+r"(src_y2),               // %2
        "+r"(dst_sobelx),           // %3
        "+r"(width)                 // %4
      : "r"(2),                     // %5
        "r"(6)                      // %6
      : "cc", "memory", "q0", "q1"  // Clobber List
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
      "vld1.8      {d0}, [%0],%4                 \n"  // left
      "vld1.8      {d1}, [%1],%4                 \n"
      "subs        %3, %3, #8                    \n"  // 8 pixels
      "vsubl.u8    q0, d0, d1                    \n"
      "vld1.8      {d2}, [%0],%4                 \n"  // center * 2
      "vld1.8      {d3}, [%1],%4                 \n"
      "vsubl.u8    q1, d2, d3                    \n"
      "vadd.s16    q0, q0, q1                    \n"
      "vadd.s16    q0, q0, q1                    \n"
      "vld1.8      {d2}, [%0],%5                 \n"  // right
      "vld1.8      {d3}, [%1],%5                 \n"
      "vsubl.u8    q1, d2, d3                    \n"
      "vadd.s16    q0, q0, q1                    \n"
      "vabs.s16    q0, q0                        \n"
      "vqmovn.u16  d0, q0                        \n"
      "vst1.8      {d0}, [%2]!                   \n"  // store 8 sobely
      "bgt         1b                            \n"
      : "+r"(src_y0),               // %0
        "+r"(src_y1),               // %1
        "+r"(dst_sobely),           // %2
        "+r"(width)                 // %3
      : "r"(1),                     // %4
        "r"(6)                      // %5
      : "cc", "memory", "q0", "q1"  // Clobber List
  );
}

// %y passes a float as a scalar vector for vector * scalar multiply.
// the register must be d0 to d15 and indexed with [0] or [1] to access
// the float in the first or second float of the d-reg

void HalfFloatRow_NEON(const uint16_t* src,
                       uint16_t* dst,
                       float scale,
                       int width) {
  asm volatile(

      "1:          \n"
      "vld1.16     {q0, q1}, [%0]!               \n"  // load 16 shorts
      "subs        %2, %2, #16                   \n"  // 16 pixels per loop
      "vmovl.u16   q8, d0                        \n"
      "vmovl.u16   q9, d1                        \n"
      "vmovl.u16   q10, d2                       \n"
      "vmovl.u16   q11, d3                       \n"
      "vcvt.f32.u32 q8, q8                       \n"
      "vcvt.f32.u32 q9, q9                       \n"
      "vcvt.f32.u32 q10, q10                     \n"
      "vcvt.f32.u32 q11, q11                     \n"
      "vmul.f32    q8, q8, %y3                   \n"  // adjust exponent
      "vmul.f32    q9, q9, %y3                   \n"
      "vmul.f32    q10, q10, %y3                 \n"
      "vmul.f32    q11, q11, %y3                 \n"
      "vqshrn.u32  d0, q8, #13                   \n"  // isolate halffloat
      "vqshrn.u32  d1, q9, #13                   \n"
      "vqshrn.u32  d2, q10, #13                  \n"
      "vqshrn.u32  d3, q11, #13                  \n"
      "vst1.16     {q0, q1}, [%1]!               \n"  // store 16 fp16
      "bgt         1b                            \n"
      : "+r"(src),                      // %0
        "+r"(dst),                      // %1
        "+r"(width)                     // %2
      : "w"(scale * 1.9259299444e-34f)  // %3
      : "cc", "memory", "q0", "q1", "q8", "q9", "q10", "q11");
}

void ByteToFloatRow_NEON(const uint8_t* src,
                         float* dst,
                         float scale,
                         int width) {
  asm volatile(

      "1:          \n"
      "vld1.8      {d2}, [%0]!                   \n"  // load 8 bytes
      "subs        %2, %2, #8                    \n"  // 8 pixels per loop
      "vmovl.u8    q1, d2                        \n"  // 8 shorts
      "vmovl.u16   q2, d2                        \n"  // 8 ints
      "vmovl.u16   q3, d3                        \n"
      "vcvt.f32.u32 q2, q2                       \n"  // 8 floats
      "vcvt.f32.u32 q3, q3                       \n"
      "vmul.f32    q2, q2, %y3                   \n"  // scale
      "vmul.f32    q3, q3, %y3                   \n"
      "vst1.8      {q2, q3}, [%1]!               \n"  // store 8 floats
      "bgt         1b                            \n"
      : "+r"(src),   // %0
        "+r"(dst),   // %1
        "+r"(width)  // %2
      : "w"(scale)   // %3
      : "cc", "memory", "q1", "q2", "q3");
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
      "vmov.u16    d6, #4                        \n"  // constant 4
      "vmov.u16    d7, #6                        \n"  // constant 6

      "1:          \n"
      "vld1.16     {q1}, [%0]!                   \n"  // load 8 samples, 5 rows
      "vld1.16     {q2}, [%4]!                   \n"
      "subs        %6, %6, #8                    \n"  // 8 processed per loop
      "vaddl.u16   q0, d2, d4                    \n"  // * 1
      "vaddl.u16   q1, d3, d5                    \n"  // * 1
      "vld1.16     {q2}, [%1]!                   \n"
      "vmlal.u16   q0, d4, d6                    \n"  // * 4
      "vmlal.u16   q1, d5, d6                    \n"  // * 4
      "vld1.16     {q2}, [%2]!                   \n"
      "vmlal.u16   q0, d4, d7                    \n"  // * 6
      "vmlal.u16   q1, d5, d7                    \n"  // * 6
      "vld1.16     {q2}, [%3]!                   \n"
      "vmlal.u16   q0, d4, d6                    \n"  // * 4
      "vmlal.u16   q1, d5, d6                    \n"  // * 4
      "vst1.32     {q0, q1}, [%5]!               \n"  // store 8 samples
      "bgt         1b                            \n"
      : "+r"(src0),  // %0
        "+r"(src1),  // %1
        "+r"(src2),  // %2
        "+r"(src3),  // %3
        "+r"(src4),  // %4
        "+r"(dst),   // %5
        "+r"(width)  // %6
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

// filter 5 rows with 1, 4, 6, 4, 1 coefficients to produce 1 row.
void GaussRow_NEON(const uint32_t* src, uint16_t* dst, int width) {
  const uint32_t* src1 = src + 1;
  const uint32_t* src2 = src + 2;
  const uint32_t* src3 = src + 3;
  asm volatile(
      "vmov.u32    q10, #4                       \n"  // constant 4
      "vmov.u32    q11, #6                       \n"  // constant 6

      "1:          \n"
      "vld1.32     {q0, q1}, [%0]!               \n"  // load 12 source samples
      "vld1.32     {q2}, [%0]                    \n"
      "subs        %5, %5, #8                    \n"  // 8 processed per loop
      "vadd.u32    q0, q0, q1                    \n"  // * 1
      "vadd.u32    q1, q1, q2                    \n"  // * 1
      "vld1.32     {q2, q3}, [%2]!               \n"
      "vmla.u32    q0, q2, q11                   \n"  // * 6
      "vmla.u32    q1, q3, q11                   \n"  // * 6
      "vld1.32     {q2, q3}, [%1]!               \n"
      "vld1.32     {q8, q9}, [%3]!               \n"
      "vadd.u32    q2, q2, q8                    \n"  // add rows for * 4
      "vadd.u32    q3, q3, q9                    \n"
      "vmla.u32    q0, q2, q10                   \n"  // * 4
      "vmla.u32    q1, q3, q10                   \n"  // * 4
      "vqshrn.u32  d0, q0, #8                    \n"  // round and pack
      "vqshrn.u32  d1, q1, #8                    \n"
      "vst1.u16    {q0}, [%4]!                   \n"  // store 8 samples
      "bgt         1b                            \n"
      : "+r"(src),   // %0
        "+r"(src1),  // %1
        "+r"(src2),  // %2
        "+r"(src3),  // %3
        "+r"(dst),   // %4
        "+r"(width)  // %5
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
}

// Convert biplanar NV21 to packed YUV24
void NV21ToYUV24Row_NEON(const uint8_t* src_y,
                         const uint8_t* src_vu,
                         uint8_t* dst_yuv24,
                         int width) {
  asm volatile(
      "1:          \n"
      "vld1.8      {q2}, [%0]!                   \n"  // load 16 Y values
      "vld2.8      {d0, d2}, [%1]!               \n"  // load 8 VU values
      "subs        %3, %3, #16                   \n"  // 16 pixels per loop
      "vmov        d1, d0                        \n"
      "vzip.u8     d0, d1                        \n"  // VV
      "vmov        d3, d2                        \n"
      "vzip.u8     d2, d3                        \n"  // UU
      "vst3.8      {d0, d2, d4}, [%2]!           \n"  // store 16 YUV pixels
      "vst3.8      {d1, d3, d5}, [%2]!           \n"
      "bgt         1b                            \n"
      : "+r"(src_y),      // %0
        "+r"(src_vu),     // %1
        "+r"(dst_yuv24),  // %2
        "+r"(width)       // %3
      :
      : "cc", "memory", "q0", "q1", "q2");
}

void AYUVToUVRow_NEON(const uint8_t* src_ayuv,
                      int src_stride_ayuv,
                      uint8_t* dst_uv,
                      int width) {
  asm volatile(
      "add         %1, %0, %1                    \n"  // src_stride + src_AYUV
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 AYUV pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 AYUV
                                                      // pixels.
      "subs        %3, %3, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // V 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // U 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more AYUV
                                                      // pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 AYUV
                                                      // pixels.
      "vpadal.u8   q0, q4                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vqrshrun.s16 d1, q0, #2                   \n"  // 2x2 average
      "vqrshrun.s16 d0, q1, #2                   \n"
      "vst2.8      {d0, d1}, [%2]!               \n"  // store 8 pixels UV.
      "bgt         1b                            \n"
      : "+r"(src_ayuv),         // %0
        "+r"(src_stride_ayuv),  // %1
        "+r"(dst_uv),           // %2
        "+r"(width)             // %3
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
}

void AYUVToVURow_NEON(const uint8_t* src_ayuv,
                      int src_stride_ayuv,
                      uint8_t* dst_vu,
                      int width) {
  asm volatile(
      "add         %1, %0, %1                    \n"  // src_stride + src_AYUV
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 AYUV pixels.
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 AYUV
                                                      // pixels.
      "subs        %3, %3, #16                   \n"  // 16 processed per loop.
      "vpaddl.u8   q0, q0                        \n"  // V 16 bytes -> 8 shorts.
      "vpaddl.u8   q1, q1                        \n"  // U 16 bytes -> 8 shorts.
      "vld4.8      {d8, d10, d12, d14}, [%1]!    \n"  // load 8 more AYUV
                                                      // pixels.
      "vld4.8      {d9, d11, d13, d15}, [%1]!    \n"  // load last 8 AYUV
                                                      // pixels.
      "vpadal.u8   q0, q4                        \n"  // B 16 bytes -> 8 shorts.
      "vpadal.u8   q1, q5                        \n"  // G 16 bytes -> 8 shorts.
      "vqrshrun.s16 d0, q0, #2                   \n"  // 2x2 average
      "vqrshrun.s16 d1, q1, #2                   \n"
      "vst2.8      {d0, d1}, [%2]!               \n"  // store 8 pixels VU.
      "bgt         1b                            \n"
      : "+r"(src_ayuv),         // %0
        "+r"(src_stride_ayuv),  // %1
        "+r"(dst_vu),           // %2
        "+r"(width)             // %3
      :
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
}

// Copy row of AYUV Y's into Y.
// Similar to ARGBExtractAlphaRow_NEON
void AYUVToYRow_NEON(const uint8_t* src_ayuv, uint8_t* dst_y, int width) {
  asm volatile(
      "1:          \n"
      "vld4.8      {d0, d2, d4, d6}, [%0]!       \n"  // load 8 AYUV pixels
      "vld4.8      {d1, d3, d5, d7}, [%0]!       \n"  // load next 8 AYUV pixels
      "subs        %2, %2, #16                   \n"  // 16 processed per loop
      "vst1.8      {q2}, [%1]!                   \n"  // store 16 Y's.
      "bgt         1b                            \n"
      : "+r"(src_ayuv),  // %0
        "+r"(dst_y),     // %1
        "+r"(width)      // %2
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

// Convert UV plane of NV12 to VU of NV21.
void SwapUVRow_NEON(const uint8_t* src_uv, uint8_t* dst_vu, int width) {
  asm volatile(
      "1:          \n"
      "vld2.8      {d0, d2}, [%0]!               \n"  // load 16 UV values
      "vld2.8      {d1, d3}, [%0]!               \n"
      "subs        %2, %2, #16                   \n"  // 16 pixels per loop
      "vmov.u8     q2, q0                        \n"  // move U after V
      "vst2.8      {q1, q2}, [%1]!               \n"  // store 16 VU pixels
      "bgt         1b                            \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_vu),  // %1
        "+r"(width)    // %2
      :
      : "cc", "memory", "q0", "q1", "q2");
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
      "vld1.8      {q0}, [%0]!                   \n"  // load 16 U values
      "vld1.8      {q1}, [%2]!                   \n"  // load 16 V values
      "vld1.8      {q2}, [%1]!                   \n"
      "vld1.8      {q3}, [%3]!                   \n"
      "subs        %5, %5, #16                   \n"  // 16 src pixels per loop
      "vpaddl.u8   q0, q0                        \n"  // half size
      "vpaddl.u8   q1, q1                        \n"
      "vpadal.u8   q0, q2                        \n"
      "vpadal.u8   q1, q3                        \n"
      "vqrshrn.u16 d0, q0, #2                    \n"
      "vqrshrn.u16 d1, q1, #2                    \n"
      "vst2.8      {d0, d1}, [%4]!               \n"  // store 8 UV pixels
      "bgt         1b                            \n"
      : "+r"(src_u),    // %0
        "+r"(src_u_1),  // %1
        "+r"(src_v),    // %2
        "+r"(src_v_1),  // %3
        "+r"(dst_uv),   // %4
        "+r"(width)     // %5
      :
      : "cc", "memory", "q0", "q1", "q2", "q3");
}

void SplitUVRow_16_NEON(const uint16_t* src_uv,
                        uint16_t* dst_u,
                        uint16_t* dst_v,
                        int depth,
                        int width) {
  int shift = depth - 16;  // Negative for right shift.
  asm volatile(
      "vdup.16     q2, %4                        \n"
      "1:          \n"
      "vld2.16     {q0, q1}, [%0]!               \n"  // load 8 UV
      "subs        %3, %3, #8                    \n"  // 8 src pixels per loop
      "vshl.u16    q0, q0, q2                    \n"
      "vshl.u16    q1, q1, q2                    \n"
      "vst1.16     {q0}, [%1]!                   \n"  // store 8 U pixels
      "vst1.16     {q1}, [%2]!                   \n"  // store 8 V pixels
      "bgt         1b                            \n"
      : "+r"(src_uv),  // %0
        "+r"(dst_u),   // %1
        "+r"(dst_v),   // %2
        "+r"(width)    // %3
      : "r"(shift)     // %4
      : "cc", "memory", "q0", "q1", "q2");
}

void MergeUVRow_16_NEON(const uint16_t* src_u,
                        const uint16_t* src_v,
                        uint16_t* dst_uv,
                        int depth,
                        int width) {
  int shift = 16 - depth;
  asm volatile(
      "vdup.16     q2, %4                        \n"
      "1:          \n"
      "vld1.16     {q0}, [%0]!                   \n"  // load 8 U
      "vld1.16     {q1}, [%1]!                   \n"  // load 8 V
      "subs        %3, %3, #8                    \n"  // 8 src pixels per loop
      "vshl.u16    q0, q0, q2                    \n"
      "vshl.u16    q1, q1, q2                    \n"
      "vst2.16     {q0, q1}, [%2]!               \n"  // store 8 UV pixels
      "bgt         1b                            \n"
      : "+r"(src_u),   // %0
        "+r"(src_v),   // %1
        "+r"(dst_uv),  // %2
        "+r"(width)    // %3
      : "r"(shift)     // %4
      : "cc", "memory", "q0", "q1", "q2");
}

void MultiplyRow_16_NEON(const uint16_t* src_y,
                         uint16_t* dst_y,
                         int scale,
                         int width) {
  asm volatile(
      "vdup.16     q2, %3                        \n"
      "1:          \n"
      "vld1.16     {q0}, [%0]!                   \n"
      "vld1.16     {q1}, [%0]!                   \n"
      "subs        %2, %2, #16                   \n"  // 16 src pixels per loop
      "vmul.u16    q0, q0, q2                    \n"
      "vmul.u16    q1, q1, q2                    \n"
      "vst1.16     {q0}, [%1]!                   \n"
      "vst1.16     {q1}, [%1]!                   \n"
      "bgt         1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "cc", "memory", "q0", "q1", "q2");
}

void DivideRow_16_NEON(const uint16_t* src_y,
                       uint16_t* dst_y,
                       int scale,
                       int width) {
  asm volatile(
      "vdup.16     d8, %3                        \n"
      "1:          \n"
      "vld1.16     {q2, q3}, [%0]!               \n"
      "subs        %2, %2, #16                   \n"  // 16 src pixels per loop
      "vmull.u16   q0, d4, d8                    \n"
      "vmull.u16   q1, d5, d8                    \n"
      "vmull.u16   q2, d6, d8                    \n"
      "vmull.u16   q3, d7, d8                    \n"
      "vshrn.u32   d0, q0, #16                   \n"
      "vshrn.u32   d1, q1, #16                   \n"
      "vshrn.u32   d2, q2, #16                   \n"
      "vshrn.u32   d3, q3, #16                   \n"
      "vst1.16     {q0, q1}, [%1]!               \n"  // store 16 pixels
      "bgt         1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale)    // %3
      : "cc", "memory", "q0", "q1", "q2", "q3", "d8");
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
  int shift = 15 - __builtin_clz((int32_t)scale);  // Negative shl is shr
  asm volatile(
      "vdup.16     q2, %3                        \n"
      "1:          \n"
      "vld1.16     {q0}, [%0]!                   \n"
      "vld1.16     {q1}, [%0]!                   \n"
      "subs        %2, %2, #16                   \n"  // 16 src pixels per loop
      "vshl.u16    q0, q0, q2                    \n"  // shr = q2 is negative
      "vshl.u16    q1, q1, q2                    \n"
      "vqmovn.u16  d0, q0                        \n"
      "vqmovn.u16  d1, q1                        \n"
      "vst1.8      {q0}, [%1]!                   \n"
      "bgt         1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(shift)    // %3
      : "cc", "memory", "q0", "q1", "q2");
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
      "vdup.8      d8, %3                        \n"
      "vdup.8      q5, %4                        \n"
      "1:          \n"
      "vld1.8      {q2, q3}, [%0]!               \n"
      "subs        %2, %2, #32                   \n"  // 32 src pixels per loop
      "vmull.u8    q0, d4, d8                    \n"
      "vmull.u8    q1, d5, d8                    \n"
      "vmull.u8    q2, d6, d8                    \n"
      "vmull.u8    q3, d7, d8                    \n"
      "vshrn.u16   d0, q0, #8                    \n"
      "vshrn.u16   d1, q1, #8                    \n"
      "vshrn.u16   d2, q2, #8                    \n"
      "vshrn.u16   d3, q3, #8                    \n"
      "vadd.u8     q0, q0, q5                    \n"
      "vadd.u8     q1, q1, q5                    \n"
      "vst1.8      {q0, q1}, [%1]!               \n"  // store 32 pixels
      "bgt         1b                            \n"
      : "+r"(src_y),  // %0
        "+r"(dst_y),  // %1
        "+r"(width)   // %2
      : "r"(scale),   // %3
        "r"(bias)     // %4
      : "cc", "memory", "q0", "q1", "q2", "q3", "d8", "q5");
}

#endif  // !defined(LIBYUV_DISABLE_NEON) && defined(__ARM_NEON__)..

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
