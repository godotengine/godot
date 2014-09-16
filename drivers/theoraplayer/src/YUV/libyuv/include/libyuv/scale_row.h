/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_SCALE_ROW_H_  // NOLINT
#define INCLUDE_LIBYUV_SCALE_ROW_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if defined(__pnacl__) || defined(__CLR_VER) || defined(COVERAGE_ENABLED) || \
    defined(TARGET_IPHONE_SIMULATOR)
#define LIBYUV_DISABLE_X86
#endif

// The following are available on all x86 platforms:
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#define HAS_SCALEROWDOWN2_SSE2
#define HAS_SCALEROWDOWN4_SSE2
#define HAS_SCALEROWDOWN34_SSSE3
#define HAS_SCALEROWDOWN38_SSSE3
#define HAS_SCALEADDROWS_SSE2
#define HAS_SCALEFILTERCOLS_SSSE3
#define HAS_SCALECOLSUP2_SSE2
#define HAS_SCALEARGBROWDOWN2_SSE2
#define HAS_SCALEARGBROWDOWNEVEN_SSE2
#define HAS_SCALEARGBCOLS_SSE2
#define HAS_SCALEARGBFILTERCOLS_SSSE3
#define HAS_SCALEARGBCOLSUP2_SSE2
#define HAS_FIXEDDIV_X86
#define HAS_FIXEDDIV1_X86
#endif

// The following are available on Neon platforms:
#if !defined(LIBYUV_DISABLE_NEON) && !defined(__native_client__) && \
    (defined(__ARM_NEON__) || defined(LIBYUV_NEON))
#define HAS_SCALEROWDOWN2_NEON
#define HAS_SCALEROWDOWN4_NEON
#define HAS_SCALEROWDOWN34_NEON
#define HAS_SCALEROWDOWN38_NEON
#define HAS_SCALEARGBROWDOWNEVEN_NEON
#define HAS_SCALEARGBROWDOWN2_NEON
#endif

// The following are available on Mips platforms:
#if !defined(LIBYUV_DISABLE_MIPS) && !defined(__native_client__) && \
    defined(__mips__) && defined(__mips_dsp) && (__mips_dsp_rev >= 2)
#define HAS_SCALEROWDOWN2_MIPS_DSPR2
#define HAS_SCALEROWDOWN4_MIPS_DSPR2
#define HAS_SCALEROWDOWN34_MIPS_DSPR2
#define HAS_SCALEROWDOWN38_MIPS_DSPR2
#endif

// Scale ARGB vertically with bilinear interpolation.
void ScalePlaneVertical(int src_height,
                        int dst_width, int dst_height,
                        int src_stride, int dst_stride,
                        const uint8* src_argb, uint8* dst_argb,
                        int x, int y, int dy,
                        int bpp, enum FilterMode filtering);

// Simplify the filtering based on scale factors.
enum FilterMode ScaleFilterReduce(int src_width, int src_height,
                                  int dst_width, int dst_height,
                                  enum FilterMode filtering);

// Divide num by div and return as 16.16 fixed point result.
int FixedDiv_C(int num, int div);
int FixedDiv_X86(int num, int div);
// Divide num - 1 by div - 1 and return as 16.16 fixed point result.
int FixedDiv1_C(int num, int div);
int FixedDiv1_X86(int num, int div);
#ifdef HAS_FIXEDDIV_X86
#define FixedDiv FixedDiv_X86
#define FixedDiv1 FixedDiv1_X86
#else
#define FixedDiv FixedDiv_C
#define FixedDiv1 FixedDiv1_C
#endif

// Compute slope values for stepping.
void ScaleSlope(int src_width, int src_height,
                int dst_width, int dst_height,
                enum FilterMode filtering,
                int* x, int* y, int* dx, int* dy);

void ScaleRowDown2_C(const uint8* src_ptr, ptrdiff_t src_stride,
                     uint8* dst, int dst_width);
void ScaleRowDown2Linear_C(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst, int dst_width);
void ScaleRowDown2Box_C(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst, int dst_width);
void ScaleRowDown4_C(const uint8* src_ptr, ptrdiff_t src_stride,
                     uint8* dst, int dst_width);
void ScaleRowDown4Box_C(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst, int dst_width);
void ScaleRowDown34_C(const uint8* src_ptr, ptrdiff_t src_stride,
                      uint8* dst, int dst_width);
void ScaleRowDown34_0_Box_C(const uint8* src_ptr, ptrdiff_t src_stride,
                            uint8* d, int dst_width);
void ScaleRowDown34_1_Box_C(const uint8* src_ptr, ptrdiff_t src_stride,
                            uint8* d, int dst_width);
void ScaleCols_C(uint8* dst_ptr, const uint8* src_ptr,
                 int dst_width, int x, int dx);
void ScaleColsUp2_C(uint8* dst_ptr, const uint8* src_ptr,
                    int dst_width, int, int);
void ScaleFilterCols_C(uint8* dst_ptr, const uint8* src_ptr,
                       int dst_width, int x, int dx);
void ScaleFilterCols64_C(uint8* dst_ptr, const uint8* src_ptr,
                         int dst_width, int x, int dx);
void ScaleRowDown38_C(const uint8* src_ptr, ptrdiff_t src_stride,
                      uint8* dst, int dst_width);
void ScaleRowDown38_3_Box_C(const uint8* src_ptr,
                            ptrdiff_t src_stride,
                            uint8* dst_ptr, int dst_width);
void ScaleRowDown38_2_Box_C(const uint8* src_ptr, ptrdiff_t src_stride,
                            uint8* dst_ptr, int dst_width);
void ScaleAddRows_C(const uint8* src_ptr, ptrdiff_t src_stride,
                    uint16* dst_ptr, int src_width, int src_height);
void ScaleARGBRowDown2_C(const uint8* src_argb,
                         ptrdiff_t src_stride,
                         uint8* dst_argb, int dst_width);
void ScaleARGBRowDown2Linear_C(const uint8* src_argb,
                               ptrdiff_t src_stride,
                               uint8* dst_argb, int dst_width);
void ScaleARGBRowDown2Box_C(const uint8* src_argb, ptrdiff_t src_stride,
                            uint8* dst_argb, int dst_width);
void ScaleARGBRowDownEven_C(const uint8* src_argb, ptrdiff_t src_stride,
                            int src_stepx,
                            uint8* dst_argb, int dst_width);
void ScaleARGBRowDownEvenBox_C(const uint8* src_argb,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8* dst_argb, int dst_width);
void ScaleARGBCols_C(uint8* dst_argb, const uint8* src_argb,
                     int dst_width, int x, int dx);
void ScaleARGBCols64_C(uint8* dst_argb, const uint8* src_argb,
                       int dst_width, int x, int dx);
void ScaleARGBColsUp2_C(uint8* dst_argb, const uint8* src_argb,
                        int dst_width, int, int);
void ScaleARGBFilterCols_C(uint8* dst_argb, const uint8* src_argb,
                           int dst_width, int x, int dx);
void ScaleARGBFilterCols64_C(uint8* dst_argb, const uint8* src_argb,
                             int dst_width, int x, int dx);

void ScaleRowDown2_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width);
void ScaleRowDown2Linear_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                              uint8* dst_ptr, int dst_width);
void ScaleRowDown2Box_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width);
void ScaleRowDown2_Unaligned_SSE2(const uint8* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8* dst_ptr, int dst_width);
void ScaleRowDown2Linear_Unaligned_SSE2(const uint8* src_ptr,
                                        ptrdiff_t src_stride,
                                        uint8* dst_ptr, int dst_width);
void ScaleRowDown2Box_Unaligned_SSE2(const uint8* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8* dst_ptr, int dst_width);
void ScaleRowDown4_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width);
void ScaleRowDown4Box_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width);
void ScaleRowDown34_SSSE3(const uint8* src_ptr, ptrdiff_t src_stride,
                          uint8* dst_ptr, int dst_width);
void ScaleRowDown34_1_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width);
void ScaleRowDown34_0_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width);
void ScaleRowDown38_SSSE3(const uint8* src_ptr, ptrdiff_t src_stride,
                          uint8* dst_ptr, int dst_width);
void ScaleRowDown38_3_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width);
void ScaleRowDown38_2_Box_SSSE3(const uint8* src_ptr,
                                ptrdiff_t src_stride,
                                uint8* dst_ptr, int dst_width);
void ScaleAddRows_SSE2(const uint8* src_ptr, ptrdiff_t src_stride,
                       uint16* dst_ptr, int src_width,
                       int src_height);
void ScaleFilterCols_SSSE3(uint8* dst_ptr, const uint8* src_ptr,
                           int dst_width, int x, int dx);
void ScaleColsUp2_SSE2(uint8* dst_ptr, const uint8* src_ptr,
                       int dst_width, int x, int dx);
void ScaleARGBRowDown2_SSE2(const uint8* src_argb,
                            ptrdiff_t src_stride,
                            uint8* dst_argb, int dst_width);
void ScaleARGBRowDown2Linear_SSE2(const uint8* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8* dst_argb, int dst_width);
void ScaleARGBRowDown2Box_SSE2(const uint8* src_argb,
                               ptrdiff_t src_stride,
                               uint8* dst_argb, int dst_width);
void ScaleARGBRowDownEven_SSE2(const uint8* src_argb, ptrdiff_t src_stride,
                               int src_stepx,
                               uint8* dst_argb, int dst_width);
void ScaleARGBRowDownEvenBox_SSE2(const uint8* src_argb,
                                  ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8* dst_argb, int dst_width);
void ScaleARGBCols_SSE2(uint8* dst_argb, const uint8* src_argb,
                        int dst_width, int x, int dx);
void ScaleARGBFilterCols_SSSE3(uint8* dst_argb, const uint8* src_argb,
                               int dst_width, int x, int dx);
void ScaleARGBColsUp2_SSE2(uint8* dst_argb, const uint8* src_argb,
                           int dst_width, int x, int dx);
// Row functions.
void ScaleARGBRowDownEven_NEON(const uint8* src_argb, int src_stride,
                               int src_stepx,
                               uint8* dst_argb, int dst_width);
void ScaleARGBRowDownEvenBox_NEON(const uint8* src_argb, int src_stride,
                                  int src_stepx,
                                  uint8* dst_argb, int dst_width);
void ScaleARGBRowDown2_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                            uint8* dst, int dst_width);
void ScaleARGBRowDown2Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                               uint8* dst, int dst_width);

// ScaleRowDown2Box also used by planar functions
// NEON downscalers with interpolation.

// Note - not static due to reuse in convert for 444 to 420.
void ScaleRowDown2_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst, int dst_width);

void ScaleRowDown2Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst, int dst_width);

void ScaleRowDown4_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                        uint8* dst_ptr, int dst_width);
void ScaleRowDown4Box_NEON(const uint8* src_ptr, ptrdiff_t src_stride,
                           uint8* dst_ptr, int dst_width);

// Down scale from 4 to 3 pixels. Use the neon multilane read/write
//  to load up the every 4th pixel into a 4 different registers.
// Point samples 32 pixels to 24 pixels.
void ScaleRowDown34_NEON(const uint8* src_ptr,
                         ptrdiff_t src_stride,
                         uint8* dst_ptr, int dst_width);
void ScaleRowDown34_0_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width);
void ScaleRowDown34_1_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width);

// 32 -> 12
void ScaleRowDown38_NEON(const uint8* src_ptr,
                         ptrdiff_t src_stride,
                         uint8* dst_ptr, int dst_width);
// 32x3 -> 12x1
void ScaleRowDown38_3_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width);
// 32x2 -> 12x1
void ScaleRowDown38_2_Box_NEON(const uint8* src_ptr,
                               ptrdiff_t src_stride,
                               uint8* dst_ptr, int dst_width);

void ScaleRowDown2_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                              uint8* dst, int dst_width);
void ScaleRowDown2Box_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                                 uint8* dst, int dst_width);
void ScaleRowDown4_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                              uint8* dst, int dst_width);
void ScaleRowDown4Box_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                                 uint8* dst, int dst_width);
void ScaleRowDown34_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                               uint8* dst, int dst_width);
void ScaleRowDown34_0_Box_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                                     uint8* d, int dst_width);
void ScaleRowDown34_1_Box_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                                     uint8* d, int dst_width);
void ScaleRowDown38_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                               uint8* dst, int dst_width);
void ScaleRowDown38_2_Box_MIPS_DSPR2(const uint8* src_ptr, ptrdiff_t src_stride,
                                     uint8* dst_ptr, int dst_width);
void ScaleRowDown38_3_Box_MIPS_DSPR2(const uint8* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8* dst_ptr, int dst_width);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_SCALE_ROW_H_  NOLINT
