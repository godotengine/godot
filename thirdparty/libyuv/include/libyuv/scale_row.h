/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_SCALE_ROW_H_
#define INCLUDE_LIBYUV_SCALE_ROW_H_

#include "libyuv/basic_types.h"
#include "libyuv/cpu_support.h"
#include "libyuv/scale.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// The following are available on all x86 platforms:
#if !defined(LIBYUV_DISABLE_X86) &&                             \
    (defined(_M_IX86) ||                                        \
     (defined(__x86_64__) && !defined(LIBYUV_ENABLE_ROWWIN)) || \
     defined(__i386__))
#define HAS_FIXEDDIV1_X86
#define HAS_FIXEDDIV_X86
#define HAS_SCALEADDROW_SSE2
#define HAS_SCALEARGBCOLS_SSE2
#define HAS_SCALEARGBCOLSUP2_SSE2
#define HAS_SCALEARGBFILTERCOLS_SSSE3
#define HAS_SCALEARGBROWDOWN2_SSE2
#define HAS_SCALEARGBROWDOWNEVEN_SSE2
#define HAS_SCALECOLSUP2_SSE2
#define HAS_SCALEFILTERCOLS_SSSE3
#define HAS_SCALEROWDOWN2_SSSE3
#define HAS_SCALEROWDOWN34_SSSE3
#define HAS_SCALEROWDOWN38_SSSE3
#define HAS_SCALEROWDOWN4_SSSE3
#endif

// The following are available for gcc/clang x86 platforms:
// TODO(fbarchard): Port to Visual C
#if !defined(LIBYUV_DISABLE_X86) &&               \
    (defined(__x86_64__) || defined(__i386__)) && \
    !defined(LIBYUV_ENABLE_ROWWIN)
#define HAS_SCALEUVROWDOWN2BOX_SSSE3
#define HAS_SCALEROWUP2_LINEAR_SSE2
#define HAS_SCALEROWUP2_LINEAR_SSSE3
#define HAS_SCALEROWUP2_BILINEAR_SSE2
#define HAS_SCALEROWUP2_BILINEAR_SSSE3
#define HAS_SCALEROWUP2_LINEAR_12_SSSE3
#define HAS_SCALEROWUP2_BILINEAR_12_SSSE3
#define HAS_SCALEROWUP2_LINEAR_16_SSE2
#define HAS_SCALEROWUP2_BILINEAR_16_SSE2
#define HAS_SCALEUVROWUP2_LINEAR_SSSE3
#define HAS_SCALEUVROWUP2_BILINEAR_SSSE3
#define HAS_SCALEUVROWUP2_LINEAR_16_SSE41
#define HAS_SCALEUVROWUP2_BILINEAR_16_SSE41
#endif

// The following are available for gcc/clang x86 platforms, but
// require clang 3.4 or gcc 4.7.
// TODO(fbarchard): Port to Visual C
#if !defined(LIBYUV_DISABLE_X86) &&               \
    (defined(__x86_64__) || defined(__i386__)) && \
    (defined(CLANG_HAS_AVX2) || defined(GCC_HAS_AVX2))
#define HAS_SCALEUVROWDOWN2BOX_AVX2
#define HAS_SCALEROWUP2_LINEAR_AVX2
#define HAS_SCALEROWUP2_BILINEAR_AVX2
#define HAS_SCALEROWUP2_LINEAR_12_AVX2
#define HAS_SCALEROWUP2_BILINEAR_12_AVX2
#define HAS_SCALEROWUP2_LINEAR_16_AVX2
#define HAS_SCALEROWUP2_BILINEAR_16_AVX2
#define HAS_SCALEUVROWUP2_LINEAR_AVX2
#define HAS_SCALEUVROWUP2_BILINEAR_AVX2
#define HAS_SCALEUVROWUP2_LINEAR_16_AVX2
#define HAS_SCALEUVROWUP2_BILINEAR_16_AVX2
#endif

// The following are available on all x86 platforms, but
// require VS2012, clang 3.4 or gcc 4.7.
// The code supports NaCL but requires a new compiler and validator.
#if !defined(LIBYUV_DISABLE_X86) &&                          \
    (defined(VISUALC_HAS_AVX2) || defined(CLANG_HAS_AVX2) || \
     defined(GCC_HAS_AVX2))
#define HAS_SCALEADDROW_AVX2
#define HAS_SCALEROWDOWN2_AVX2
#define HAS_SCALEROWDOWN4_AVX2
#endif

// The following are available on Neon platforms:
#if !defined(LIBYUV_DISABLE_NEON) && \
    (defined(__ARM_NEON__) || defined(LIBYUV_NEON) || defined(__aarch64__))
#define HAS_SCALEADDROW_NEON
#define HAS_SCALEARGBCOLS_NEON
#define HAS_SCALEARGBFILTERCOLS_NEON
#define HAS_SCALEARGBROWDOWN2_NEON
#define HAS_SCALEARGBROWDOWNEVEN_NEON
#define HAS_SCALEFILTERCOLS_NEON
#define HAS_SCALEROWDOWN2_NEON
#define HAS_SCALEROWDOWN34_NEON
#define HAS_SCALEROWDOWN38_NEON
#define HAS_SCALEROWDOWN4_NEON
#define HAS_SCALEUVROWDOWN2_NEON
#define HAS_SCALEUVROWDOWN2LINEAR_NEON
#define HAS_SCALEUVROWDOWN2BOX_NEON
#define HAS_SCALEUVROWDOWNEVEN_NEON
#define HAS_SCALEROWUP2_LINEAR_NEON
#define HAS_SCALEROWUP2_BILINEAR_NEON
#define HAS_SCALEROWUP2_LINEAR_12_NEON
#define HAS_SCALEROWUP2_BILINEAR_12_NEON
#define HAS_SCALEROWUP2_LINEAR_16_NEON
#define HAS_SCALEROWUP2_BILINEAR_16_NEON
#define HAS_SCALEUVROWUP2_LINEAR_NEON
#define HAS_SCALEUVROWUP2_BILINEAR_NEON
#define HAS_SCALEUVROWUP2_LINEAR_16_NEON
#define HAS_SCALEUVROWUP2_BILINEAR_16_NEON
#endif

// The following are available on AArch64 Neon platforms:
#if !defined(LIBYUV_DISABLE_NEON) && defined(__aarch64__)
#define HAS_SCALEROWDOWN2_16_NEON
#endif

// The following are available on AArch64 SME platforms:
#if !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) && \
    defined(__aarch64__)
#define HAS_SCALEARGBROWDOWN2_SME
#define HAS_SCALEROWDOWN2_16_SME
#define HAS_SCALEROWDOWN2_SME
#define HAS_SCALEUVROWDOWN2BOX_SME
#define HAS_SCALEUVROWDOWN2LINEAR_SME
#define HAS_SCALEUVROWDOWN2_SME
#endif

#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#define HAS_SCALEADDROW_MSA
#define HAS_SCALEARGBCOLS_MSA
#define HAS_SCALEARGBFILTERCOLS_MSA
#define HAS_SCALEARGBROWDOWN2_MSA
#define HAS_SCALEARGBROWDOWNEVEN_MSA
#define HAS_SCALEFILTERCOLS_MSA
#define HAS_SCALEROWDOWN2_MSA
#define HAS_SCALEROWDOWN34_MSA
#define HAS_SCALEROWDOWN38_MSA
#define HAS_SCALEROWDOWN4_MSA
#endif

#if !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
#define HAS_SCALEARGBROWDOWN2_LSX
#define HAS_SCALEARGBROWDOWNEVEN_LSX
#define HAS_SCALEROWDOWN2_LSX
#define HAS_SCALEROWDOWN4_LSX
#define HAS_SCALEROWDOWN38_LSX
#define HAS_SCALEFILTERCOLS_LSX
#define HAS_SCALEADDROW_LSX
#define HAS_SCALEARGBCOLS_LSX
#define HAS_SCALEARGBFILTERCOLS_LSX
#define HAS_SCALEROWDOWN34_LSX
#endif

// The following are available on RVV with 64 bit elements
// TODO: Update compiler to support 64 bit
#if !defined(LIBYUV_DISABLE_RVV) && defined(__riscv_vector) && \
    defined(__riscv_zve64x)
#define HAS_SCALEUVROWDOWN4_RVV
#define HAS_SCALEARGBROWDOWN2_RVV
#endif

// The following are available on RVV 1.1
// TODO: Port to RVV 1.2
#if !defined(LIBYUV_DISABLE_RVV) && defined(__riscv_vector) && \
    defined(__riscv_v_intrinsic) && __riscv_v_intrinsic == 11000
#define HAS_SCALEROWDOWN34_0_BOX_RVV
#define HAS_SCALEROWDOWN34_1_BOX_RVV
#define HAS_SCALEROWDOWN38_2_BOX_RVV
#define HAS_SCALEROWDOWN38_3_BOX_RVV
#define HAS_SCALEUVROWUP2_BILINEAR_RVV
#define HAS_SCALEUVROWUP2_LINEAR_RVV
#define HAS_SCALEROWDOWN34_RVV
#define HAS_SCALEROWDOWN38_RVV
#define HAS_SCALEROWUP2_BILINEAR_RVV
#define HAS_SCALEROWUP2_LINEAR_RVV
#endif

// The following are available on RVV
#if !defined(LIBYUV_DISABLE_RVV) && defined(__riscv_vector) && \
    defined(__riscv_v_intrinsic) && __riscv_v_intrinsic == 11000
#define HAS_SCALEARGBFILTERCOLS_RVV
#define HAS_SCALEARGBROWDOWN2BOX_RVV
#define HAS_SCALEARGBROWDOWN2LINEAR_RVV
#define HAS_SCALEARGBROWDOWNEVENBOX_RVV
#define HAS_SCALEROWDOWN2BOX_RVV
#define HAS_SCALEADDROW_RVV
// TODO: Test ScaleARGBRowDownEven_RVV and enable it
// #define HAS_SCALEARGBROWDOWNEVEN_RVV
#define HAS_SCALEUVROWDOWNEVEN_RVV
#define HAS_SCALEROWDOWN2_RVV
#define HAS_SCALEROWDOWN2LINEAR_RVV
#define HAS_SCALEROWDOWN4_RVV
#define HAS_SCALEROWDOWN4BOX_RVV
#define HAS_SCALEUVROWDOWN2_RVV
#define HAS_SCALEUVROWDOWN2BOX_RVV
#define HAS_SCALEUVROWDOWN2LINEAR_RVV
#endif

// Scale ARGB vertically with bilinear interpolation.
void ScalePlaneVertical(int src_height,
                        int dst_width,
                        int dst_height,
                        int src_stride,
                        int dst_stride,
                        const uint8_t* src_argb,
                        uint8_t* dst_argb,
                        int x,
                        int y,
                        int dy,
                        int bpp,
                        enum FilterMode filtering);

void ScalePlaneVertical_16(int src_height,
                           int dst_width,
                           int dst_height,
                           int src_stride,
                           int dst_stride,
                           const uint16_t* src_argb,
                           uint16_t* dst_argb,
                           int x,
                           int y,
                           int dy,
                           int wpp,
                           enum FilterMode filtering);

void ScalePlaneVertical_16To8(int src_height,
                              int dst_width,
                              int dst_height,
                              int src_stride,
                              int dst_stride,
                              const uint16_t* src_argb,
                              uint8_t* dst_argb,
                              int x,
                              int y,
                              int dy,
                              int wpp,
                              int scale,
                              enum FilterMode filtering);

void ScalePlaneDown2_16To8(int src_width,
                           int src_height,
                           int dst_width,
                           int dst_height,
                           int src_stride,
                           int dst_stride,
                           const uint16_t* src_ptr,
                           uint8_t* dst_ptr,
                           int scale,
                           enum FilterMode filtering);

// Simplify the filtering based on scale factors.
enum FilterMode ScaleFilterReduce(int src_width,
                                  int src_height,
                                  int dst_width,
                                  int dst_height,
                                  enum FilterMode filtering);

// Divide num by div and return as 16.16 fixed point result.
int FixedDiv_C(int num, int div);
int FixedDiv_X86(int num, int div);
int FixedDiv_MIPS(int num, int div);
// Divide num - 1 by div - 1 and return as 16.16 fixed point result.
int FixedDiv1_C(int num, int div);
int FixedDiv1_X86(int num, int div);
int FixedDiv1_MIPS(int num, int div);
#ifdef HAS_FIXEDDIV_X86
#define FixedDiv FixedDiv_X86
#define FixedDiv1 FixedDiv1_X86
#elif defined HAS_FIXEDDIV_MIPS
#define FixedDiv FixedDiv_MIPS
#define FixedDiv1 FixedDiv1_MIPS
#else
#define FixedDiv FixedDiv_C
#define FixedDiv1 FixedDiv1_C
#endif

// Compute slope values for stepping.
void ScaleSlope(int src_width,
                int src_height,
                int dst_width,
                int dst_height,
                enum FilterMode filtering,
                int* x,
                int* y,
                int* dx,
                int* dy);

void ScaleRowDown2_C(const uint8_t* src_ptr,
                     ptrdiff_t src_stride,
                     uint8_t* dst,
                     int dst_width);
void ScaleRowDown2_16_C(const uint16_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint16_t* dst,
                        int dst_width);
void ScaleRowDown2_16To8_C(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst,
                           int dst_width,
                           int scale);
void ScaleRowDown2_16To8_Odd_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst,
                               int dst_width,
                               int scale);
void ScaleRowDown2Linear_C(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst,
                           int dst_width);
void ScaleRowDown2Linear_16_C(const uint16_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint16_t* dst,
                              int dst_width);
void ScaleRowDown2Linear_16To8_C(const uint16_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst,
                                 int dst_width,
                                 int scale);
void ScaleRowDown2Linear_16To8_Odd_C(const uint16_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8_t* dst,
                                     int dst_width,
                                     int scale);
void ScaleRowDown2Box_C(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown2Box_Odd_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst,
                            int dst_width);
void ScaleRowDown2Box_16_C(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint16_t* dst,
                           int dst_width);
void ScaleRowDown2Box_16To8_C(const uint16_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst,
                              int dst_width,
                              int scale);
void ScaleRowDown2Box_16To8_Odd_C(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst,
                                  int dst_width,
                                  int scale);
void ScaleRowDown4_C(const uint8_t* src_ptr,
                     ptrdiff_t src_stride,
                     uint8_t* dst,
                     int dst_width);
void ScaleRowDown4_16_C(const uint16_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint16_t* dst,
                        int dst_width);
void ScaleRowDown4Box_C(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown4Box_16_C(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint16_t* dst,
                           int dst_width);
void ScaleRowDown34_C(const uint8_t* src_ptr,
                      ptrdiff_t src_stride,
                      uint8_t* dst,
                      int dst_width);
void ScaleRowDown34_16_C(const uint16_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint16_t* dst,
                         int dst_width);
void ScaleRowDown34_0_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* d,
                            int dst_width);
void ScaleRowDown34_0_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* d,
                               int dst_width);
void ScaleRowDown34_1_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* d,
                            int dst_width);
void ScaleRowDown34_1_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* d,
                               int dst_width);

void ScaleRowUp2_Linear_C(const uint8_t* src_ptr,
                          uint8_t* dst_ptr,
                          int dst_width);
void ScaleRowUp2_Bilinear_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            ptrdiff_t dst_stride,
                            int dst_width);
void ScaleRowUp2_Linear_16_C(const uint16_t* src_ptr,
                             uint16_t* dst_ptr,
                             int dst_width);
void ScaleRowUp2_Bilinear_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width);
void ScaleRowUp2_Linear_Any_C(const uint8_t* src_ptr,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowUp2_Bilinear_Any_C(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                ptrdiff_t dst_stride,
                                int dst_width);
void ScaleRowUp2_Linear_16_Any_C(const uint16_t* src_ptr,
                                 uint16_t* dst_ptr,
                                 int dst_width);
void ScaleRowUp2_Bilinear_16_Any_C(const uint16_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint16_t* dst_ptr,
                                   ptrdiff_t dst_stride,
                                   int dst_width);

void ScaleCols_C(uint8_t* dst_ptr,
                 const uint8_t* src_ptr,
                 int dst_width,
                 int x,
                 int dx);
void ScaleCols_16_C(uint16_t* dst_ptr,
                    const uint16_t* src_ptr,
                    int dst_width,
                    int x,
                    int dx);
void ScaleColsUp2_C(uint8_t* dst_ptr,
                    const uint8_t* src_ptr,
                    int dst_width,
                    int,
                    int);
void ScaleColsUp2_16_C(uint16_t* dst_ptr,
                       const uint16_t* src_ptr,
                       int dst_width,
                       int,
                       int);
void ScaleFilterCols_C(uint8_t* dst_ptr,
                       const uint8_t* src_ptr,
                       int dst_width,
                       int x,
                       int dx);
void ScaleFilterCols_16_C(uint16_t* dst_ptr,
                          const uint16_t* src_ptr,
                          int dst_width,
                          int x,
                          int dx);
void ScaleFilterCols64_C(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         int dst_width,
                         int x32,
                         int dx);
void ScaleFilterCols64_16_C(uint16_t* dst_ptr,
                            const uint16_t* src_ptr,
                            int dst_width,
                            int x32,
                            int dx);
void ScaleRowDown38_C(const uint8_t* src_ptr,
                      ptrdiff_t src_stride,
                      uint8_t* dst,
                      int dst_width);
void ScaleRowDown38_16_C(const uint16_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint16_t* dst,
                         int dst_width);
void ScaleRowDown38_3_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown38_3_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* dst_ptr,
                               int dst_width);
void ScaleRowDown38_2_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown38_2_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* dst_ptr,
                               int dst_width);
void ScaleAddRow_C(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width);
void ScaleAddRow_16_C(const uint16_t* src_ptr,
                      uint32_t* dst_ptr,
                      int src_width);
void ScaleARGBRowDown2_C(const uint8_t* src_argb,
                         ptrdiff_t src_stride,
                         uint8_t* dst_argb,
                         int dst_width);
void ScaleARGBRowDown2Linear_C(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               uint8_t* dst_argb,
                               int dst_width);
void ScaleARGBRowDown2Box_C(const uint8_t* src_argb,
                            ptrdiff_t src_stride,
                            uint8_t* dst_argb,
                            int dst_width);
void ScaleARGBRowDownEven_C(const uint8_t* src_argb,
                            ptrdiff_t src_stride,
                            int src_stepx,
                            uint8_t* dst_argb,
                            int dst_width);
void ScaleARGBRowDownEvenBox_C(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8_t* dst_argb,
                               int dst_width);
void ScaleARGBCols_C(uint8_t* dst_argb,
                     const uint8_t* src_argb,
                     int dst_width,
                     int x,
                     int dx);
void ScaleARGBCols64_C(uint8_t* dst_argb,
                       const uint8_t* src_argb,
                       int dst_width,
                       int x32,
                       int dx);
void ScaleARGBColsUp2_C(uint8_t* dst_argb,
                        const uint8_t* src_argb,
                        int dst_width,
                        int,
                        int);
void ScaleARGBFilterCols_C(uint8_t* dst_argb,
                           const uint8_t* src_argb,
                           int dst_width,
                           int x,
                           int dx);
void ScaleARGBFilterCols64_C(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x32,
                             int dx);
void ScaleUVRowDown2_C(const uint8_t* src_uv,
                       ptrdiff_t src_stride,
                       uint8_t* dst_uv,
                       int dst_width);
void ScaleUVRowDown2Linear_C(const uint8_t* src_uv,
                             ptrdiff_t src_stride,
                             uint8_t* dst_uv,
                             int dst_width);
void ScaleUVRowDown2Box_C(const uint8_t* src_uv,
                          ptrdiff_t src_stride,
                          uint8_t* dst_uv,
                          int dst_width);
void ScaleUVRowDownEven_C(const uint8_t* src_uv,
                          ptrdiff_t src_stride,
                          int src_stepx,
                          uint8_t* dst_uv,
                          int dst_width);
void ScaleUVRowDownEvenBox_C(const uint8_t* src_uv,
                             ptrdiff_t src_stride,
                             int src_stepx,
                             uint8_t* dst_uv,
                             int dst_width);

void ScaleUVRowUp2_Linear_C(const uint8_t* src_ptr,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleUVRowUp2_Bilinear_C(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              ptrdiff_t dst_stride,
                              int dst_width);
void ScaleUVRowUp2_Linear_Any_C(const uint8_t* src_ptr,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleUVRowUp2_Bilinear_Any_C(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width);
void ScaleUVRowUp2_Linear_16_C(const uint16_t* src_ptr,
                               uint16_t* dst_ptr,
                               int dst_width);
void ScaleUVRowUp2_Bilinear_16_C(const uint16_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint16_t* dst_ptr,
                                 ptrdiff_t dst_stride,
                                 int dst_width);
void ScaleUVRowUp2_Linear_16_Any_C(const uint16_t* src_ptr,
                                   uint16_t* dst_ptr,
                                   int dst_width);
void ScaleUVRowUp2_Bilinear_16_Any_C(const uint16_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint16_t* dst_ptr,
                                     ptrdiff_t dst_stride,
                                     int dst_width);

void ScaleUVCols_C(uint8_t* dst_uv,
                   const uint8_t* src_uv,
                   int dst_width,
                   int x,
                   int dx);
void ScaleUVCols64_C(uint8_t* dst_uv,
                     const uint8_t* src_uv,
                     int dst_width,
                     int x32,
                     int dx);
void ScaleUVColsUp2_C(uint8_t* dst_uv,
                      const uint8_t* src_uv,
                      int dst_width,
                      int,
                      int);
void ScaleUVFilterCols_C(uint8_t* dst_uv,
                         const uint8_t* src_uv,
                         int dst_width,
                         int x,
                         int dx);
void ScaleUVFilterCols64_C(uint8_t* dst_uv,
                           const uint8_t* src_uv,
                           int dst_width,
                           int x32,
                           int dx);

// Specialized scalers for x86.
void ScaleRowDown2_SSSE3(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width);
void ScaleRowDown2Linear_SSSE3(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleRowDown2Box_SSSE3(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown2_AVX2(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width);
void ScaleRowDown2Linear_AVX2(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown2Box_AVX2(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width);
void ScaleRowDown4_SSSE3(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width);
void ScaleRowDown4Box_SSSE3(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown4_AVX2(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width);
void ScaleRowDown4Box_AVX2(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width);

void ScaleRowDown34_SSSE3(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width);
void ScaleRowDown34_1_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleRowDown34_0_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleRowDown38_SSSE3(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width);
void ScaleRowDown38_3_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleRowDown38_2_Box_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);

void ScaleRowUp2_Linear_SSE2(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             int dst_width);
void ScaleRowUp2_Bilinear_SSE2(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width);
void ScaleRowUp2_Linear_12_SSSE3(const uint16_t* src_ptr,
                                 uint16_t* dst_ptr,
                                 int dst_width);
void ScaleRowUp2_Bilinear_12_SSSE3(const uint16_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint16_t* dst_ptr,
                                   ptrdiff_t dst_stride,
                                   int dst_width);
void ScaleRowUp2_Linear_16_SSE2(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width);
void ScaleRowUp2_Bilinear_16_SSE2(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width);
void ScaleRowUp2_Linear_SSSE3(const uint8_t* src_ptr,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowUp2_Bilinear_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                ptrdiff_t dst_stride,
                                int dst_width);
void ScaleRowUp2_Linear_AVX2(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             int dst_width);
void ScaleRowUp2_Bilinear_AVX2(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width);
void ScaleRowUp2_Linear_12_AVX2(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width);
void ScaleRowUp2_Bilinear_12_AVX2(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width);
void ScaleRowUp2_Linear_16_AVX2(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width);
void ScaleRowUp2_Bilinear_16_AVX2(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width);
void ScaleRowUp2_Linear_Any_SSE2(const uint8_t* src_ptr,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleRowUp2_Bilinear_Any_SSE2(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   ptrdiff_t dst_stride,
                                   int dst_width);
void ScaleRowUp2_Linear_12_Any_SSSE3(const uint16_t* src_ptr,
                                     uint16_t* dst_ptr,
                                     int dst_width);
void ScaleRowUp2_Bilinear_12_Any_SSSE3(const uint16_t* src_ptr,
                                       ptrdiff_t src_stride,
                                       uint16_t* dst_ptr,
                                       ptrdiff_t dst_stride,
                                       int dst_width);
void ScaleRowUp2_Linear_16_Any_SSE2(const uint16_t* src_ptr,
                                    uint16_t* dst_ptr,
                                    int dst_width);
void ScaleRowUp2_Bilinear_16_Any_SSE2(const uint16_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint16_t* dst_ptr,
                                      ptrdiff_t dst_stride,
                                      int dst_width);
void ScaleRowUp2_Linear_Any_SSSE3(const uint8_t* src_ptr,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleRowUp2_Bilinear_Any_SSSE3(const uint8_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint8_t* dst_ptr,
                                    ptrdiff_t dst_stride,
                                    int dst_width);
void ScaleRowUp2_Linear_Any_AVX2(const uint8_t* src_ptr,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleRowUp2_Bilinear_Any_AVX2(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   ptrdiff_t dst_stride,
                                   int dst_width);
void ScaleRowUp2_Linear_12_Any_AVX2(const uint16_t* src_ptr,
                                    uint16_t* dst_ptr,
                                    int dst_width);
void ScaleRowUp2_Bilinear_12_Any_AVX2(const uint16_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint16_t* dst_ptr,
                                      ptrdiff_t dst_stride,
                                      int dst_width);
void ScaleRowUp2_Linear_16_Any_AVX2(const uint16_t* src_ptr,
                                    uint16_t* dst_ptr,
                                    int dst_width);
void ScaleRowUp2_Bilinear_16_Any_AVX2(const uint16_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint16_t* dst_ptr,
                                      ptrdiff_t dst_stride,
                                      int dst_width);

void ScaleRowDown2_Any_SSSE3(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst_ptr,
                             int dst_width);
void ScaleRowDown2Linear_Any_SSSE3(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleRowDown2Box_Any_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleRowDown2Box_Odd_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleRowDown2_Any_AVX2(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown2Linear_Any_AVX2(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleRowDown2Box_Any_AVX2(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleRowDown2Box_Odd_AVX2(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleRowDown4_Any_SSSE3(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst_ptr,
                             int dst_width);
void ScaleRowDown4Box_Any_SSSE3(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleRowDown4_Any_AVX2(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown4Box_Any_AVX2(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);

void ScaleRowDown34_Any_SSSE3(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown34_1_Box_Any_SSSE3(const uint8_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint8_t* dst_ptr,
                                    int dst_width);
void ScaleRowDown34_0_Box_Any_SSSE3(const uint8_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint8_t* dst_ptr,
                                    int dst_width);
void ScaleRowDown38_Any_SSSE3(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown38_3_Box_Any_SSSE3(const uint8_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint8_t* dst_ptr,
                                    int dst_width);
void ScaleRowDown38_2_Box_Any_SSSE3(const uint8_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint8_t* dst_ptr,
                                    int dst_width);

void ScaleAddRow_SSE2(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width);
void ScaleAddRow_AVX2(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width);
void ScaleAddRow_Any_SSE2(const uint8_t* src_ptr,
                          uint16_t* dst_ptr,
                          int src_width);
void ScaleAddRow_Any_AVX2(const uint8_t* src_ptr,
                          uint16_t* dst_ptr,
                          int src_width);

void ScaleFilterCols_SSSE3(uint8_t* dst_ptr,
                           const uint8_t* src_ptr,
                           int dst_width,
                           int x,
                           int dx);
void ScaleColsUp2_SSE2(uint8_t* dst_ptr,
                       const uint8_t* src_ptr,
                       int dst_width,
                       int x,
                       int dx);

// ARGB Column functions
void ScaleARGBCols_SSE2(uint8_t* dst_argb,
                        const uint8_t* src_argb,
                        int dst_width,
                        int x,
                        int dx);
void ScaleARGBFilterCols_SSSE3(uint8_t* dst_argb,
                               const uint8_t* src_argb,
                               int dst_width,
                               int x,
                               int dx);
void ScaleARGBColsUp2_SSE2(uint8_t* dst_argb,
                           const uint8_t* src_argb,
                           int dst_width,
                           int x,
                           int dx);
void ScaleARGBFilterCols_NEON(uint8_t* dst_argb,
                              const uint8_t* src_argb,
                              int dst_width,
                              int x,
                              int dx);
void ScaleARGBCols_NEON(uint8_t* dst_argb,
                        const uint8_t* src_argb,
                        int dst_width,
                        int x,
                        int dx);
void ScaleARGBFilterCols_Any_NEON(uint8_t* dst_ptr,
                                  const uint8_t* src_ptr,
                                  int dst_width,
                                  int x,
                                  int dx);
void ScaleARGBCols_Any_NEON(uint8_t* dst_ptr,
                            const uint8_t* src_ptr,
                            int dst_width,
                            int x,
                            int dx);
void ScaleARGBFilterCols_MSA(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x,
                             int dx);
void ScaleARGBCols_MSA(uint8_t* dst_argb,
                       const uint8_t* src_argb,
                       int dst_width,
                       int x,
                       int dx);
void ScaleARGBFilterCols_Any_MSA(uint8_t* dst_ptr,
                                 const uint8_t* src_ptr,
                                 int dst_width,
                                 int x,
                                 int dx);
void ScaleARGBCols_Any_MSA(uint8_t* dst_ptr,
                           const uint8_t* src_ptr,
                           int dst_width,
                           int x,
                           int dx);
void ScaleARGBFilterCols_RVV(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x,
                             int dx);

// ARGB Row functions
void ScaleARGBRowDown2_SSE2(const uint8_t* src_argb,
                            ptrdiff_t src_stride,
                            uint8_t* dst_argb,
                            int dst_width);
void ScaleARGBRowDown2Linear_SSE2(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_argb,
                                  int dst_width);
void ScaleARGBRowDown2Box_SSE2(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               uint8_t* dst_argb,
                               int dst_width);
void ScaleARGBRowDown2_NEON(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst,
                            int dst_width);
void ScaleARGBRowDown2_SME(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst,
                           int dst_width);
void ScaleARGBRowDown2Linear_NEON(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_argb,
                                  int dst_width);
void ScaleARGBRowDown2Linear_SME(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width);
void ScaleARGBRowDown2Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst,
                               int dst_width);
void ScaleARGBRowDown2Box_SME(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst,
                              int dst_width);
void ScaleARGBRowDown2_RVV(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width);
void ScaleARGBRowDown2Linear_RVV(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width);
void ScaleARGBRowDown2Box_RVV(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width);
void ScaleARGBRowDown2_MSA(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width);
void ScaleARGBRowDown2Linear_MSA(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width);
void ScaleARGBRowDown2Box_MSA(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width);
void ScaleARGBRowDown2_LSX(const uint8_t* src_argb,
                           ptrdiff_t src_stride,
                           uint8_t* dst_argb,
                           int dst_width);
void ScaleARGBRowDown2Linear_LSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_argb,
                                 int dst_width);
void ScaleARGBRowDown2Box_LSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              uint8_t* dst_argb,
                              int dst_width);
void ScaleARGBRowDown2_Any_SSE2(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleARGBRowDown2Linear_Any_SSE2(const uint8_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint8_t* dst_ptr,
                                      int dst_width);
void ScaleARGBRowDown2Box_Any_SSE2(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleARGBRowDown2_Any_NEON(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleARGBRowDown2Linear_Any_NEON(const uint8_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint8_t* dst_ptr,
                                      int dst_width);
void ScaleARGBRowDown2Box_Any_NEON(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleARGBRowDown2_Any_MSA(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleARGBRowDown2Linear_Any_MSA(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8_t* dst_ptr,
                                     int dst_width);
void ScaleARGBRowDown2Box_Any_MSA(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleARGBRowDown2_Any_LSX(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleARGBRowDown2Linear_Any_LSX(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8_t* dst_ptr,
                                     int dst_width);
void ScaleARGBRowDown2Box_Any_LSX(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleARGBRowDownEven_SSE2(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8_t* dst_argb,
                               int dst_width);
void ScaleARGBRowDownEvenBox_SSE2(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8_t* dst_argb,
                                  int dst_width);
void ScaleARGBRowDownEven_NEON(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8_t* dst_argb,
                               int dst_width);
void ScaleARGBRowDownEvenBox_NEON(const uint8_t* src_argb,
                                  ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8_t* dst_argb,
                                  int dst_width);
void ScaleARGBRowDownEven_MSA(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              int32_t src_stepx,
                              uint8_t* dst_argb,
                              int dst_width);
void ScaleARGBRowDownEvenBox_MSA(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width);
void ScaleARGBRowDownEven_LSX(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              int32_t src_stepx,
                              uint8_t* dst_argb,
                              int dst_width);
void ScaleARGBRowDownEvenBox_LSX(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width);
void ScaleARGBRowDownEven_RVV(const uint8_t* src_argb,
                              ptrdiff_t src_stride,
                              int32_t src_stepx,
                              uint8_t* dst_argb,
                              int dst_width);
void ScaleARGBRowDownEvenBox_RVV(const uint8_t* src_argb,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_argb,
                                 int dst_width);
void ScaleARGBRowDownEven_Any_SSE2(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   int src_stepx,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleARGBRowDownEvenBox_Any_SSE2(const uint8_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      int src_stepx,
                                      uint8_t* dst_ptr,
                                      int dst_width);
void ScaleARGBRowDownEven_Any_NEON(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   int src_stepx,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleARGBRowDownEvenBox_Any_NEON(const uint8_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      int src_stepx,
                                      uint8_t* dst_ptr,
                                      int dst_width);
void ScaleARGBRowDownEven_Any_MSA(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  int32_t src_stepx,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleARGBRowDownEvenBox_Any_MSA(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     int src_stepx,
                                     uint8_t* dst_ptr,
                                     int dst_width);
void ScaleARGBRowDownEven_Any_LSX(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  int32_t src_stepx,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleARGBRowDownEvenBox_Any_LSX(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     int src_stepx,
                                     uint8_t* dst_ptr,
                                     int dst_width);

// UV Row functions
void ScaleUVRowDown2_SSSE3(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_uv,
                           int dst_width);
void ScaleUVRowDown2Linear_SSSE3(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_uv,
                                 int dst_width);
void ScaleUVRowDown2Box_SSSE3(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_uv,
                              int dst_width);
void ScaleUVRowDown2Box_AVX2(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst_uv,
                             int dst_width);
void ScaleUVRowDown2_NEON(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width);
void ScaleUVRowDown2_SME(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst,
                         int dst_width);
void ScaleUVRowDown2Linear_NEON(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_uv,
                                int dst_width);
void ScaleUVRowDown2Linear_SME(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_uv,
                               int dst_width);
void ScaleUVRowDown2Box_NEON(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width);
void ScaleUVRowDown2Box_SME(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst,
                            int dst_width);
void ScaleUVRowDown2_MSA(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_uv,
                         int dst_width);
void ScaleUVRowDown2Linear_MSA(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_uv,
                               int dst_width);
void ScaleUVRowDown2Box_MSA(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_uv,
                            int dst_width);
void ScaleUVRowDown2_RVV(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_uv,
                         int dst_width);
void ScaleUVRowDown2Linear_RVV(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_uv,
                               int dst_width);
void ScaleUVRowDown2Box_RVV(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst,
                            int dst_width);
void ScaleUVRowDown2_Any_SSSE3(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleUVRowDown2Linear_Any_SSSE3(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8_t* dst_ptr,
                                     int dst_width);
void ScaleUVRowDown2Box_Any_SSSE3(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleUVRowDown2Box_Any_AVX2(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleUVRowDown2_Any_NEON(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleUVRowDown2Linear_Any_NEON(const uint8_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint8_t* dst_ptr,
                                    int dst_width);
void ScaleUVRowDown2Box_Any_NEON(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleUVRowDown2_Any_MSA(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst_ptr,
                             int dst_width);
void ScaleUVRowDown2Linear_Any_MSA(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleUVRowDown2Box_Any_MSA(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleUVRowDownEven_SSSE3(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              int src_stepx,
                              uint8_t* dst_uv,
                              int dst_width);
void ScaleUVRowDownEvenBox_SSSE3(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_uv,
                                 int dst_width);
void ScaleUVRowDownEven_NEON(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             int src_stepx,
                             uint8_t* dst_uv,
                             int dst_width);
void ScaleUVRowDownEvenBox_NEON(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                int src_stepx,
                                uint8_t* dst_uv,
                                int dst_width);
void ScaleUVRowDown4_RVV(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         int32_t src_stepx,
                         uint8_t* dst_uv,
                         int dst_width);
void ScaleUVRowDownEven_RVV(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            int32_t src_stepx,
                            uint8_t* dst_uv,
                            int dst_width);
void ScaleUVRowDownEven_MSA(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            int32_t src_stepx,
                            uint8_t* dst_uv,
                            int dst_width);
void ScaleUVRowDownEvenBox_MSA(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8_t* dst_uv,
                               int dst_width);
void ScaleUVRowDownEven_Any_SSSE3(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  int src_stepx,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleUVRowDownEvenBox_Any_SSSE3(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     int src_stepx,
                                     uint8_t* dst_ptr,
                                     int dst_width);
void ScaleUVRowDownEven_Any_NEON(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 int src_stepx,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleUVRowDownEvenBox_Any_NEON(const uint8_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    int src_stepx,
                                    uint8_t* dst_ptr,
                                    int dst_width);
void ScaleUVRowDownEven_Any_MSA(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                int32_t src_stepx,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleUVRowDownEvenBox_Any_MSA(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   int src_stepx,
                                   uint8_t* dst_ptr,
                                   int dst_width);

void ScaleUVRowUp2_Linear_SSSE3(const uint8_t* src_ptr,
                                uint8_t* dst_ptr,
                                int dst_width);
void ScaleUVRowUp2_Bilinear_SSSE3(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width);
void ScaleUVRowUp2_Linear_Any_SSSE3(const uint8_t* src_ptr,
                                    uint8_t* dst_ptr,
                                    int dst_width);
void ScaleUVRowUp2_Bilinear_Any_SSSE3(const uint8_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint8_t* dst_ptr,
                                      ptrdiff_t dst_stride,
                                      int dst_width);
void ScaleUVRowUp2_Linear_AVX2(const uint8_t* src_ptr,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleUVRowUp2_Bilinear_AVX2(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 ptrdiff_t dst_stride,
                                 int dst_width);
void ScaleUVRowUp2_Linear_Any_AVX2(const uint8_t* src_ptr,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleUVRowUp2_Bilinear_Any_AVX2(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8_t* dst_ptr,
                                     ptrdiff_t dst_stride,
                                     int dst_width);
void ScaleUVRowUp2_Linear_NEON(const uint8_t* src_ptr,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleUVRowUp2_Bilinear_NEON(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 ptrdiff_t dst_stride,
                                 int dst_width);
void ScaleUVRowUp2_Linear_Any_NEON(const uint8_t* src_ptr,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleUVRowUp2_Bilinear_Any_NEON(const uint8_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8_t* dst_ptr,
                                     ptrdiff_t dst_stride,
                                     int dst_width);
void ScaleUVRowUp2_Linear_RVV(const uint8_t* src_ptr,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleUVRowUp2_Bilinear_RVV(const uint8_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint8_t* dst_ptr,
                                ptrdiff_t dst_stride,
                                int dst_width);
void ScaleUVRowUp2_Linear_16_SSE41(const uint16_t* src_ptr,
                                   uint16_t* dst_ptr,
                                   int dst_width);
void ScaleUVRowUp2_Bilinear_16_SSE41(const uint16_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint16_t* dst_ptr,
                                     ptrdiff_t dst_stride,
                                     int dst_width);
void ScaleUVRowUp2_Linear_16_Any_SSE41(const uint16_t* src_ptr,
                                       uint16_t* dst_ptr,
                                       int dst_width);
void ScaleUVRowUp2_Bilinear_16_Any_SSE41(const uint16_t* src_ptr,
                                         ptrdiff_t src_stride,
                                         uint16_t* dst_ptr,
                                         ptrdiff_t dst_stride,
                                         int dst_width);
void ScaleUVRowUp2_Linear_16_AVX2(const uint16_t* src_ptr,
                                  uint16_t* dst_ptr,
                                  int dst_width);
void ScaleUVRowUp2_Bilinear_16_AVX2(const uint16_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint16_t* dst_ptr,
                                    ptrdiff_t dst_stride,
                                    int dst_width);
void ScaleUVRowUp2_Linear_16_Any_AVX2(const uint16_t* src_ptr,
                                      uint16_t* dst_ptr,
                                      int dst_width);
void ScaleUVRowUp2_Bilinear_16_Any_AVX2(const uint16_t* src_ptr,
                                        ptrdiff_t src_stride,
                                        uint16_t* dst_ptr,
                                        ptrdiff_t dst_stride,
                                        int dst_width);
void ScaleUVRowUp2_Linear_16_NEON(const uint16_t* src_ptr,
                                  uint16_t* dst_ptr,
                                  int dst_width);
void ScaleUVRowUp2_Bilinear_16_NEON(const uint16_t* src_ptr,
                                    ptrdiff_t src_stride,
                                    uint16_t* dst_ptr,
                                    ptrdiff_t dst_stride,
                                    int dst_width);
void ScaleUVRowUp2_Linear_16_Any_NEON(const uint16_t* src_ptr,
                                      uint16_t* dst_ptr,
                                      int dst_width);
void ScaleUVRowUp2_Bilinear_16_Any_NEON(const uint16_t* src_ptr,
                                        ptrdiff_t src_stride,
                                        uint16_t* dst_ptr,
                                        ptrdiff_t dst_stride,
                                        int dst_width);

// ScaleRowDown2Box also used by planar functions
// NEON/SME downscalers with interpolation.
void ScaleRowDown2_NEON(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown2_SME(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width);
void ScaleRowDown2_16_NEON(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint16_t* dst,
                           int dst_width);
void ScaleRowDown2_16_SME(const uint16_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint16_t* dst,
                          int dst_width);
void ScaleRowDown2Linear_NEON(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst,
                              int dst_width);
void ScaleRowDown2Linear_SME(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width);
void ScaleRowDown2Linear_16_NEON(const uint16_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint16_t* dst,
                                 int dst_width);
void ScaleRowDown2Linear_16_SME(const uint16_t* src_ptr,
                                ptrdiff_t src_stride,
                                uint16_t* dst,
                                int dst_width);
void ScaleRowDown2Box_NEON(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst,
                           int dst_width);
void ScaleRowDown2Box_SME(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width);
void ScaleRowDown2Box_16_NEON(const uint16_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint16_t* dst,
                              int dst_width);
void ScaleRowDown2Box_16_SME(const uint16_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint16_t* dst,
                             int dst_width);

void ScaleRowDown4_NEON(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width);
void ScaleRowDown4Box_NEON(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width);

// Down scale from 4 to 3 pixels. Use the neon multilane read/write
//  to load up the every 4th pixel into a 4 different registers.
// Point samples 32 pixels to 24 pixels.
void ScaleRowDown34_NEON(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width);
void ScaleRowDown34_0_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleRowDown34_1_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);

// 32 -> 12
void ScaleRowDown38_NEON(const uint8_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint8_t* dst_ptr,
                         int dst_width);
// 32x3 -> 12x1
void ScaleRowDown38_3_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
// 32x2 -> 12x1
void ScaleRowDown38_2_Box_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);

void ScaleRowDown2_Any_NEON(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown2Linear_Any_NEON(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleRowDown2Box_Any_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleRowDown2Box_Odd_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleRowDown4_Any_NEON(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown4Box_Any_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               int dst_width);
void ScaleRowDown34_Any_NEON(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst_ptr,
                             int dst_width);
void ScaleRowDown34_0_Box_Any_NEON(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);
void ScaleRowDown34_1_Box_Any_NEON(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);
// 32 -> 12
void ScaleRowDown38_Any_NEON(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst_ptr,
                             int dst_width);
// 32x3 -> 12x1
void ScaleRowDown38_3_Box_Any_NEON(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);
// 32x2 -> 12x1
void ScaleRowDown38_2_Box_Any_NEON(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   int dst_width);

void ScaleRowUp2_Linear_NEON(const uint8_t* src_ptr,
                             uint8_t* dst_ptr,
                             int dst_width);
void ScaleRowUp2_Bilinear_NEON(const uint8_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width);
void ScaleRowUp2_Linear_12_NEON(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width);
void ScaleRowUp2_Bilinear_12_NEON(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width);
void ScaleRowUp2_Linear_16_NEON(const uint16_t* src_ptr,
                                uint16_t* dst_ptr,
                                int dst_width);
void ScaleRowUp2_Bilinear_16_NEON(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint16_t* dst_ptr,
                                  ptrdiff_t dst_stride,
                                  int dst_width);
void ScaleRowUp2_Linear_Any_NEON(const uint8_t* src_ptr,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleRowUp2_Bilinear_Any_NEON(const uint8_t* src_ptr,
                                   ptrdiff_t src_stride,
                                   uint8_t* dst_ptr,
                                   ptrdiff_t dst_stride,
                                   int dst_width);
void ScaleRowUp2_Linear_12_Any_NEON(const uint16_t* src_ptr,
                                    uint16_t* dst_ptr,
                                    int dst_width);
void ScaleRowUp2_Bilinear_12_Any_NEON(const uint16_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint16_t* dst_ptr,
                                      ptrdiff_t dst_stride,
                                      int dst_width);
void ScaleRowUp2_Linear_16_Any_NEON(const uint16_t* src_ptr,
                                    uint16_t* dst_ptr,
                                    int dst_width);
void ScaleRowUp2_Bilinear_16_Any_NEON(const uint16_t* src_ptr,
                                      ptrdiff_t src_stride,
                                      uint16_t* dst_ptr,
                                      ptrdiff_t dst_stride,
                                      int dst_width);

void ScaleAddRow_NEON(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width);
void ScaleAddRow_Any_NEON(const uint8_t* src_ptr,
                          uint16_t* dst_ptr,
                          int src_width);

void ScaleFilterCols_NEON(uint8_t* dst_ptr,
                          const uint8_t* src_ptr,
                          int dst_width,
                          int x,
                          int dx);

void ScaleFilterCols_Any_NEON(uint8_t* dst_ptr,
                              const uint8_t* src_ptr,
                              int dst_width,
                              int x,
                              int dx);

void ScaleRowDown2_MSA(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width);
void ScaleRowDown2Linear_MSA(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width);
void ScaleRowDown2Box_MSA(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width);
void ScaleRowDown4_MSA(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width);
void ScaleRowDown4Box_MSA(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width);
void ScaleRowDown38_MSA(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown38_2_Box_MSA(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown38_3_Box_MSA(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleAddRow_MSA(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width);
void ScaleFilterCols_MSA(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         int dst_width,
                         int x,
                         int dx);
void ScaleRowDown34_MSA(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown34_0_Box_MSA(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* d,
                              int dst_width);
void ScaleRowDown34_1_Box_MSA(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* d,
                              int dst_width);

void ScaleRowDown2_Any_MSA(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width);
void ScaleRowDown2Linear_Any_MSA(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleRowDown2Box_Any_MSA(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown4_Any_MSA(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width);
void ScaleRowDown4Box_Any_MSA(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown38_Any_MSA(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown38_2_Box_Any_MSA(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleRowDown38_3_Box_Any_MSA(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleAddRow_Any_MSA(const uint8_t* src_ptr,
                         uint16_t* dst_ptr,
                         int src_width);
void ScaleFilterCols_Any_MSA(uint8_t* dst_ptr,
                             const uint8_t* src_ptr,
                             int dst_width,
                             int x,
                             int dx);
void ScaleRowDown34_Any_MSA(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown34_0_Box_Any_MSA(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleRowDown34_1_Box_Any_MSA(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);

void ScaleRowDown2_LSX(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width);
void ScaleRowDown2Linear_LSX(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width);
void ScaleRowDown2Box_LSX(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width);
void ScaleRowDown4_LSX(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width);
void ScaleRowDown4Box_LSX(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width);
void ScaleRowDown38_LSX(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown38_2_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown38_3_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleAddRow_LSX(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width);
void ScaleFilterCols_LSX(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         int dst_width,
                         int x,
                         int dx);
void ScaleARGBFilterCols_LSX(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x,
                             int dx);
void ScaleARGBCols_LSX(uint8_t* dst_argb,
                       const uint8_t* src_argb,
                       int dst_width,
                       int x,
                       int dx);
void ScaleRowDown34_LSX(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown34_0_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* d,
                              int dst_width);
void ScaleRowDown34_1_Box_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* d,
                              int dst_width);
void ScaleRowDown2_Any_LSX(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width);
void ScaleRowDown2Linear_Any_LSX(const uint8_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst_ptr,
                                 int dst_width);
void ScaleRowDown2Box_Any_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown4_Any_LSX(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst_ptr,
                           int dst_width);
void ScaleRowDown4Box_Any_LSX(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown38_Any_LSX(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown38_2_Box_Any_LSX(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleRowDown38_3_Box_Any_LSX(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleAddRow_Any_LSX(const uint8_t* src_ptr,
                         uint16_t* dst_ptr,
                         int src_width);
void ScaleFilterCols_Any_LSX(uint8_t* dst_ptr,
                             const uint8_t* src_ptr,
                             int dst_width,
                             int x,
                             int dx);
void ScaleARGBCols_Any_LSX(uint8_t* dst_ptr,
                           const uint8_t* src_ptr,
                           int dst_width,
                           int x,
                           int dx);
void ScaleARGBFilterCols_Any_LSX(uint8_t* dst_ptr,
                                 const uint8_t* src_ptr,
                                 int dst_width,
                                 int x,
                                 int dx);
void ScaleRowDown34_Any_LSX(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowDown34_0_Box_Any_LSX(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);
void ScaleRowDown34_1_Box_Any_LSX(const uint8_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst_ptr,
                                  int dst_width);

void ScaleAddRow_RVV(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width);
void ScaleRowDown2_RVV(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst,
                       int dst_width);
void ScaleRowDown2Linear_RVV(const uint8_t* src_ptr,
                             ptrdiff_t src_stride,
                             uint8_t* dst,
                             int dst_width);
void ScaleRowDown2Box_RVV(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst,
                          int dst_width);

void ScaleRowDown4_RVV(const uint8_t* src_ptr,
                       ptrdiff_t src_stride,
                       uint8_t* dst_ptr,
                       int dst_width);
void ScaleRowDown4Box_RVV(const uint8_t* src_ptr,
                          ptrdiff_t src_stride,
                          uint8_t* dst_ptr,
                          int dst_width);
void ScaleRowDown34_RVV(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst_ptr,
                        int dst_width);
void ScaleRowDown34_0_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown34_1_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown38_RVV(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width);
void ScaleRowDown38_3_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);
void ScaleRowDown38_2_Box_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              int dst_width);

void ScaleRowUp2_Linear_RVV(const uint8_t* src_ptr,
                            uint8_t* dst_ptr,
                            int dst_width);
void ScaleRowUp2_Bilinear_RVV(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              ptrdiff_t dst_stride,
                              int dst_width);
#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_SCALE_ROW_H_
