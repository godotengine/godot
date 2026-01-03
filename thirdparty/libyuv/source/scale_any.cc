/*
 *  Copyright 2015 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string.h>  // For memset/memcpy

#include "libyuv/scale.h"
#include "libyuv/scale_row.h"

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Fixed scale down.
// Mask may be non-power of 2, so use MOD
#define SDANY(NAMEANY, SCALEROWDOWN_SIMD, SCALEROWDOWN_C, FACTOR, BPP, MASK)   \
  void NAMEANY(const uint8_t* src_ptr, ptrdiff_t src_stride, uint8_t* dst_ptr, \
               int dst_width) {                                                \
    int r = (int)((unsigned int)dst_width % (MASK + 1)); /* NOLINT */          \
    int n = dst_width - r;                                                     \
    if (n > 0) {                                                               \
      SCALEROWDOWN_SIMD(src_ptr, src_stride, dst_ptr, n);                      \
    }                                                                          \
    SCALEROWDOWN_C(src_ptr + (n * FACTOR) * BPP, src_stride,                   \
                   dst_ptr + n * BPP, r);                                      \
  }

// Fixed scale down for odd source width.  Used by I420Blend subsampling.
// Since dst_width is (width + 1) / 2, this function scales one less pixel
// and copies the last pixel.
#define SDODD(NAMEANY, SCALEROWDOWN_SIMD, SCALEROWDOWN_C, FACTOR, BPP, MASK)   \
  void NAMEANY(const uint8_t* src_ptr, ptrdiff_t src_stride, uint8_t* dst_ptr, \
               int dst_width) {                                                \
    int r = (int)((unsigned int)(dst_width - 1) % (MASK + 1)); /* NOLINT */    \
    int n = (dst_width - 1) - r;                                               \
    if (n > 0) {                                                               \
      SCALEROWDOWN_SIMD(src_ptr, src_stride, dst_ptr, n);                      \
    }                                                                          \
    SCALEROWDOWN_C(src_ptr + (n * FACTOR) * BPP, src_stride,                   \
                   dst_ptr + n * BPP, r + 1);                                  \
  }

#ifdef HAS_SCALEROWDOWN2_SSSE3
SDANY(ScaleRowDown2_Any_SSSE3, ScaleRowDown2_SSSE3, ScaleRowDown2_C, 2, 1, 15)
SDANY(ScaleRowDown2Linear_Any_SSSE3,
      ScaleRowDown2Linear_SSSE3,
      ScaleRowDown2Linear_C,
      2,
      1,
      15)
SDANY(ScaleRowDown2Box_Any_SSSE3,
      ScaleRowDown2Box_SSSE3,
      ScaleRowDown2Box_C,
      2,
      1,
      15)
SDODD(ScaleRowDown2Box_Odd_SSSE3,
      ScaleRowDown2Box_SSSE3,
      ScaleRowDown2Box_Odd_C,
      2,
      1,
      15)
#endif
#ifdef HAS_SCALEUVROWDOWN2BOX_SSSE3
SDANY(ScaleUVRowDown2Box_Any_SSSE3,
      ScaleUVRowDown2Box_SSSE3,
      ScaleUVRowDown2Box_C,
      2,
      2,
      3)
#endif
#ifdef HAS_SCALEUVROWDOWN2BOX_AVX2
SDANY(ScaleUVRowDown2Box_Any_AVX2,
      ScaleUVRowDown2Box_AVX2,
      ScaleUVRowDown2Box_C,
      2,
      2,
      7)
#endif
#ifdef HAS_SCALEROWDOWN2_AVX2
SDANY(ScaleRowDown2_Any_AVX2, ScaleRowDown2_AVX2, ScaleRowDown2_C, 2, 1, 31)
SDANY(ScaleRowDown2Linear_Any_AVX2,
      ScaleRowDown2Linear_AVX2,
      ScaleRowDown2Linear_C,
      2,
      1,
      31)
SDANY(ScaleRowDown2Box_Any_AVX2,
      ScaleRowDown2Box_AVX2,
      ScaleRowDown2Box_C,
      2,
      1,
      31)
SDODD(ScaleRowDown2Box_Odd_AVX2,
      ScaleRowDown2Box_AVX2,
      ScaleRowDown2Box_Odd_C,
      2,
      1,
      31)
#endif
#ifdef HAS_SCALEROWDOWN2_NEON
SDANY(ScaleRowDown2_Any_NEON, ScaleRowDown2_NEON, ScaleRowDown2_C, 2, 1, 15)
SDANY(ScaleRowDown2Linear_Any_NEON,
      ScaleRowDown2Linear_NEON,
      ScaleRowDown2Linear_C,
      2,
      1,
      15)
SDANY(ScaleRowDown2Box_Any_NEON,
      ScaleRowDown2Box_NEON,
      ScaleRowDown2Box_C,
      2,
      1,
      15)
SDODD(ScaleRowDown2Box_Odd_NEON,
      ScaleRowDown2Box_NEON,
      ScaleRowDown2Box_Odd_C,
      2,
      1,
      15)
#endif
#ifdef HAS_SCALEUVROWDOWN2_NEON
SDANY(ScaleUVRowDown2_Any_NEON,
      ScaleUVRowDown2_NEON,
      ScaleUVRowDown2_C,
      2,
      2,
      7)
#endif
#ifdef HAS_SCALEUVROWDOWN2LINEAR_NEON
SDANY(ScaleUVRowDown2Linear_Any_NEON,
      ScaleUVRowDown2Linear_NEON,
      ScaleUVRowDown2Linear_C,
      2,
      2,
      7)
#endif
#ifdef HAS_SCALEUVROWDOWN2BOX_NEON
SDANY(ScaleUVRowDown2Box_Any_NEON,
      ScaleUVRowDown2Box_NEON,
      ScaleUVRowDown2Box_C,
      2,
      2,
      7)
#endif

#ifdef HAS_SCALEROWDOWN2_MSA
SDANY(ScaleRowDown2_Any_MSA, ScaleRowDown2_MSA, ScaleRowDown2_C, 2, 1, 31)
SDANY(ScaleRowDown2Linear_Any_MSA,
      ScaleRowDown2Linear_MSA,
      ScaleRowDown2Linear_C,
      2,
      1,
      31)
SDANY(ScaleRowDown2Box_Any_MSA,
      ScaleRowDown2Box_MSA,
      ScaleRowDown2Box_C,
      2,
      1,
      31)
#endif
#ifdef HAS_SCALEROWDOWN2_LSX
SDANY(ScaleRowDown2_Any_LSX, ScaleRowDown2_LSX, ScaleRowDown2_C, 2, 1, 31)
SDANY(ScaleRowDown2Linear_Any_LSX,
      ScaleRowDown2Linear_LSX,
      ScaleRowDown2Linear_C,
      2,
      1,
      31)
SDANY(ScaleRowDown2Box_Any_LSX,
      ScaleRowDown2Box_LSX,
      ScaleRowDown2Box_C,
      2,
      1,
      31)
#endif
#ifdef HAS_SCALEROWDOWN4_SSSE3
SDANY(ScaleRowDown4_Any_SSSE3, ScaleRowDown4_SSSE3, ScaleRowDown4_C, 4, 1, 7)
SDANY(ScaleRowDown4Box_Any_SSSE3,
      ScaleRowDown4Box_SSSE3,
      ScaleRowDown4Box_C,
      4,
      1,
      7)
#endif
#ifdef HAS_SCALEROWDOWN4_AVX2
SDANY(ScaleRowDown4_Any_AVX2, ScaleRowDown4_AVX2, ScaleRowDown4_C, 4, 1, 15)
SDANY(ScaleRowDown4Box_Any_AVX2,
      ScaleRowDown4Box_AVX2,
      ScaleRowDown4Box_C,
      4,
      1,
      15)
#endif
#ifdef HAS_SCALEROWDOWN4_NEON
SDANY(ScaleRowDown4_Any_NEON, ScaleRowDown4_NEON, ScaleRowDown4_C, 4, 1, 15)
SDANY(ScaleRowDown4Box_Any_NEON,
      ScaleRowDown4Box_NEON,
      ScaleRowDown4Box_C,
      4,
      1,
      7)
#endif
#ifdef HAS_SCALEROWDOWN4_MSA
SDANY(ScaleRowDown4_Any_MSA, ScaleRowDown4_MSA, ScaleRowDown4_C, 4, 1, 15)
SDANY(ScaleRowDown4Box_Any_MSA,
      ScaleRowDown4Box_MSA,
      ScaleRowDown4Box_C,
      4,
      1,
      15)
#endif
#ifdef HAS_SCALEROWDOWN4_LSX
SDANY(ScaleRowDown4_Any_LSX, ScaleRowDown4_LSX, ScaleRowDown4_C, 4, 1, 15)
SDANY(ScaleRowDown4Box_Any_LSX,
      ScaleRowDown4Box_LSX,
      ScaleRowDown4Box_C,
      4,
      1,
      15)
#endif
#ifdef HAS_SCALEROWDOWN34_SSSE3
SDANY(ScaleRowDown34_Any_SSSE3,
      ScaleRowDown34_SSSE3,
      ScaleRowDown34_C,
      4 / 3,
      1,
      23)
SDANY(ScaleRowDown34_0_Box_Any_SSSE3,
      ScaleRowDown34_0_Box_SSSE3,
      ScaleRowDown34_0_Box_C,
      4 / 3,
      1,
      23)
SDANY(ScaleRowDown34_1_Box_Any_SSSE3,
      ScaleRowDown34_1_Box_SSSE3,
      ScaleRowDown34_1_Box_C,
      4 / 3,
      1,
      23)
#endif
#ifdef HAS_SCALEROWDOWN34_NEON
#ifdef __aarch64__
SDANY(ScaleRowDown34_Any_NEON,
      ScaleRowDown34_NEON,
      ScaleRowDown34_C,
      4 / 3,
      1,
      47)
SDANY(ScaleRowDown34_0_Box_Any_NEON,
      ScaleRowDown34_0_Box_NEON,
      ScaleRowDown34_0_Box_C,
      4 / 3,
      1,
      47)
SDANY(ScaleRowDown34_1_Box_Any_NEON,
      ScaleRowDown34_1_Box_NEON,
      ScaleRowDown34_1_Box_C,
      4 / 3,
      1,
      47)
#else
SDANY(ScaleRowDown34_Any_NEON,
      ScaleRowDown34_NEON,
      ScaleRowDown34_C,
      4 / 3,
      1,
      23)
SDANY(ScaleRowDown34_0_Box_Any_NEON,
      ScaleRowDown34_0_Box_NEON,
      ScaleRowDown34_0_Box_C,
      4 / 3,
      1,
      23)
SDANY(ScaleRowDown34_1_Box_Any_NEON,
      ScaleRowDown34_1_Box_NEON,
      ScaleRowDown34_1_Box_C,
      4 / 3,
      1,
      23)
#endif
#endif
#ifdef HAS_SCALEROWDOWN34_MSA
SDANY(ScaleRowDown34_Any_MSA,
      ScaleRowDown34_MSA,
      ScaleRowDown34_C,
      4 / 3,
      1,
      47)
SDANY(ScaleRowDown34_0_Box_Any_MSA,
      ScaleRowDown34_0_Box_MSA,
      ScaleRowDown34_0_Box_C,
      4 / 3,
      1,
      47)
SDANY(ScaleRowDown34_1_Box_Any_MSA,
      ScaleRowDown34_1_Box_MSA,
      ScaleRowDown34_1_Box_C,
      4 / 3,
      1,
      47)
#endif
#ifdef HAS_SCALEROWDOWN34_LSX
SDANY(ScaleRowDown34_Any_LSX,
      ScaleRowDown34_LSX,
      ScaleRowDown34_C,
      4 / 3,
      1,
      47)
SDANY(ScaleRowDown34_0_Box_Any_LSX,
      ScaleRowDown34_0_Box_LSX,
      ScaleRowDown34_0_Box_C,
      4 / 3,
      1,
      47)
SDANY(ScaleRowDown34_1_Box_Any_LSX,
      ScaleRowDown34_1_Box_LSX,
      ScaleRowDown34_1_Box_C,
      4 / 3,
      1,
      47)
#endif
#ifdef HAS_SCALEROWDOWN38_SSSE3
SDANY(ScaleRowDown38_Any_SSSE3,
      ScaleRowDown38_SSSE3,
      ScaleRowDown38_C,
      8 / 3,
      1,
      11)
SDANY(ScaleRowDown38_3_Box_Any_SSSE3,
      ScaleRowDown38_3_Box_SSSE3,
      ScaleRowDown38_3_Box_C,
      8 / 3,
      1,
      5)
SDANY(ScaleRowDown38_2_Box_Any_SSSE3,
      ScaleRowDown38_2_Box_SSSE3,
      ScaleRowDown38_2_Box_C,
      8 / 3,
      1,
      5)
#endif
#ifdef HAS_SCALEROWDOWN38_NEON
SDANY(ScaleRowDown38_Any_NEON,
      ScaleRowDown38_NEON,
      ScaleRowDown38_C,
      8 / 3,
      1,
      11)
SDANY(ScaleRowDown38_3_Box_Any_NEON,
      ScaleRowDown38_3_Box_NEON,
      ScaleRowDown38_3_Box_C,
      8 / 3,
      1,
      11)
SDANY(ScaleRowDown38_2_Box_Any_NEON,
      ScaleRowDown38_2_Box_NEON,
      ScaleRowDown38_2_Box_C,
      8 / 3,
      1,
      11)
#endif
#ifdef HAS_SCALEROWDOWN38_MSA
SDANY(ScaleRowDown38_Any_MSA,
      ScaleRowDown38_MSA,
      ScaleRowDown38_C,
      8 / 3,
      1,
      11)
SDANY(ScaleRowDown38_3_Box_Any_MSA,
      ScaleRowDown38_3_Box_MSA,
      ScaleRowDown38_3_Box_C,
      8 / 3,
      1,
      11)
SDANY(ScaleRowDown38_2_Box_Any_MSA,
      ScaleRowDown38_2_Box_MSA,
      ScaleRowDown38_2_Box_C,
      8 / 3,
      1,
      11)
#endif
#ifdef HAS_SCALEROWDOWN38_LSX
SDANY(ScaleRowDown38_Any_LSX,
      ScaleRowDown38_LSX,
      ScaleRowDown38_C,
      8 / 3,
      1,
      11)
SDANY(ScaleRowDown38_3_Box_Any_LSX,
      ScaleRowDown38_3_Box_LSX,
      ScaleRowDown38_3_Box_C,
      8 / 3,
      1,
      11)
SDANY(ScaleRowDown38_2_Box_Any_LSX,
      ScaleRowDown38_2_Box_LSX,
      ScaleRowDown38_2_Box_C,
      8 / 3,
      1,
      11)
#endif

#ifdef HAS_SCALEARGBROWDOWN2_SSE2
SDANY(ScaleARGBRowDown2_Any_SSE2,
      ScaleARGBRowDown2_SSE2,
      ScaleARGBRowDown2_C,
      2,
      4,
      3)
SDANY(ScaleARGBRowDown2Linear_Any_SSE2,
      ScaleARGBRowDown2Linear_SSE2,
      ScaleARGBRowDown2Linear_C,
      2,
      4,
      3)
SDANY(ScaleARGBRowDown2Box_Any_SSE2,
      ScaleARGBRowDown2Box_SSE2,
      ScaleARGBRowDown2Box_C,
      2,
      4,
      3)
#endif
#ifdef HAS_SCALEARGBROWDOWN2_NEON
SDANY(ScaleARGBRowDown2_Any_NEON,
      ScaleARGBRowDown2_NEON,
      ScaleARGBRowDown2_C,
      2,
      4,
      7)
SDANY(ScaleARGBRowDown2Linear_Any_NEON,
      ScaleARGBRowDown2Linear_NEON,
      ScaleARGBRowDown2Linear_C,
      2,
      4,
      7)
SDANY(ScaleARGBRowDown2Box_Any_NEON,
      ScaleARGBRowDown2Box_NEON,
      ScaleARGBRowDown2Box_C,
      2,
      4,
      7)
#endif
#ifdef HAS_SCALEARGBROWDOWN2_MSA
SDANY(ScaleARGBRowDown2_Any_MSA,
      ScaleARGBRowDown2_MSA,
      ScaleARGBRowDown2_C,
      2,
      4,
      3)
SDANY(ScaleARGBRowDown2Linear_Any_MSA,
      ScaleARGBRowDown2Linear_MSA,
      ScaleARGBRowDown2Linear_C,
      2,
      4,
      3)
SDANY(ScaleARGBRowDown2Box_Any_MSA,
      ScaleARGBRowDown2Box_MSA,
      ScaleARGBRowDown2Box_C,
      2,
      4,
      3)
#endif
#ifdef HAS_SCALEARGBROWDOWN2_LSX
SDANY(ScaleARGBRowDown2_Any_LSX,
      ScaleARGBRowDown2_LSX,
      ScaleARGBRowDown2_C,
      2,
      4,
      3)
SDANY(ScaleARGBRowDown2Linear_Any_LSX,
      ScaleARGBRowDown2Linear_LSX,
      ScaleARGBRowDown2Linear_C,
      2,
      4,
      3)
SDANY(ScaleARGBRowDown2Box_Any_LSX,
      ScaleARGBRowDown2Box_LSX,
      ScaleARGBRowDown2Box_C,
      2,
      4,
      3)
#endif
#undef SDANY

// Scale down by even scale factor.
#define SDAANY(NAMEANY, SCALEROWDOWN_SIMD, SCALEROWDOWN_C, BPP, MASK)       \
  void NAMEANY(const uint8_t* src_ptr, ptrdiff_t src_stride, int src_stepx, \
               uint8_t* dst_ptr, int dst_width) {                           \
    int r = dst_width & MASK;                                               \
    int n = dst_width & ~MASK;                                              \
    if (n > 0) {                                                            \
      SCALEROWDOWN_SIMD(src_ptr, src_stride, src_stepx, dst_ptr, n);        \
    }                                                                       \
    SCALEROWDOWN_C(src_ptr + (n * src_stepx) * BPP, src_stride, src_stepx,  \
                   dst_ptr + n * BPP, r);                                   \
  }

#ifdef HAS_SCALEARGBROWDOWNEVEN_SSE2
SDAANY(ScaleARGBRowDownEven_Any_SSE2,
       ScaleARGBRowDownEven_SSE2,
       ScaleARGBRowDownEven_C,
       4,
       3)
SDAANY(ScaleARGBRowDownEvenBox_Any_SSE2,
       ScaleARGBRowDownEvenBox_SSE2,
       ScaleARGBRowDownEvenBox_C,
       4,
       3)
#endif
#ifdef HAS_SCALEARGBROWDOWNEVEN_NEON
SDAANY(ScaleARGBRowDownEven_Any_NEON,
       ScaleARGBRowDownEven_NEON,
       ScaleARGBRowDownEven_C,
       4,
       3)
SDAANY(ScaleARGBRowDownEvenBox_Any_NEON,
       ScaleARGBRowDownEvenBox_NEON,
       ScaleARGBRowDownEvenBox_C,
       4,
       3)
#endif
#ifdef HAS_SCALEARGBROWDOWNEVEN_MSA
SDAANY(ScaleARGBRowDownEven_Any_MSA,
       ScaleARGBRowDownEven_MSA,
       ScaleARGBRowDownEven_C,
       4,
       3)
SDAANY(ScaleARGBRowDownEvenBox_Any_MSA,
       ScaleARGBRowDownEvenBox_MSA,
       ScaleARGBRowDownEvenBox_C,
       4,
       3)
#endif
#ifdef HAS_SCALEARGBROWDOWNEVEN_LSX
SDAANY(ScaleARGBRowDownEven_Any_LSX,
       ScaleARGBRowDownEven_LSX,
       ScaleARGBRowDownEven_C,
       4,
       3)
SDAANY(ScaleARGBRowDownEvenBox_Any_LSX,
       ScaleARGBRowDownEvenBox_LSX,
       ScaleARGBRowDownEvenBox_C,
       4,
       3)
#endif
#ifdef HAS_SCALEUVROWDOWNEVEN_NEON
SDAANY(ScaleUVRowDownEven_Any_NEON,
       ScaleUVRowDownEven_NEON,
       ScaleUVRowDownEven_C,
       2,
       3)
#endif

#ifdef SASIMDONLY
// This also works and uses memcpy and SIMD instead of C, but is slower on ARM

// Add rows box filter scale down.  Using macro from row_any
#define SAROW(NAMEANY, ANY_SIMD, SBPP, BPP, MASK)                      \
  void NAMEANY(const uint8_t* src_ptr, uint16_t* dst_ptr, int width) { \
    SIMD_ALIGNED(uint16_t dst_temp[32]);                               \
    SIMD_ALIGNED(uint8_t src_temp[32]);                                \
    memset(dst_temp, 0, 32 * 2); /* for msan */                        \
    int r = width & MASK;                                              \
    int n = width & ~MASK;                                             \
    if (n > 0) {                                                       \
      ANY_SIMD(src_ptr, dst_ptr, n);                                   \
    }                                                                  \
    memcpy(src_temp, src_ptr + n * SBPP, r * SBPP);                    \
    memcpy(dst_temp, dst_ptr + n * BPP, r * BPP);                      \
    ANY_SIMD(src_temp, dst_temp, MASK + 1);                            \
    memcpy(dst_ptr + n * BPP, dst_temp, r * BPP);                      \
  }

#ifdef HAS_SCALEADDROW_SSE2
SAROW(ScaleAddRow_Any_SSE2, ScaleAddRow_SSE2, 1, 2, 15)
#endif
#ifdef HAS_SCALEADDROW_AVX2
SAROW(ScaleAddRow_Any_AVX2, ScaleAddRow_AVX2, 1, 2, 31)
#endif
#ifdef HAS_SCALEADDROW_NEON
SAROW(ScaleAddRow_Any_NEON, ScaleAddRow_NEON, 1, 2, 15)
#endif
#ifdef HAS_SCALEADDROW_MSA
SAROW(ScaleAddRow_Any_MSA, ScaleAddRow_MSA, 1, 2, 15)
#endif
#ifdef HAS_SCALEADDROW_LSX
SAROW(ScaleAddRow_Any_LSX, ScaleAddRow_LSX, 1, 2, 15)
#endif
#undef SAANY

#else

// Add rows box filter scale down.
#define SAANY(NAMEANY, SCALEADDROW_SIMD, SCALEADDROW_C, MASK)              \
  void NAMEANY(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width) { \
    int n = src_width & ~MASK;                                             \
    if (n > 0) {                                                           \
      SCALEADDROW_SIMD(src_ptr, dst_ptr, n);                               \
    }                                                                      \
    SCALEADDROW_C(src_ptr + n, dst_ptr + n, src_width & MASK);             \
  }

#ifdef HAS_SCALEADDROW_SSE2
SAANY(ScaleAddRow_Any_SSE2, ScaleAddRow_SSE2, ScaleAddRow_C, 15)
#endif
#ifdef HAS_SCALEADDROW_AVX2
SAANY(ScaleAddRow_Any_AVX2, ScaleAddRow_AVX2, ScaleAddRow_C, 31)
#endif
#ifdef HAS_SCALEADDROW_NEON
SAANY(ScaleAddRow_Any_NEON, ScaleAddRow_NEON, ScaleAddRow_C, 15)
#endif
#ifdef HAS_SCALEADDROW_MSA
SAANY(ScaleAddRow_Any_MSA, ScaleAddRow_MSA, ScaleAddRow_C, 15)
#endif
#ifdef HAS_SCALEADDROW_LSX
SAANY(ScaleAddRow_Any_LSX, ScaleAddRow_LSX, ScaleAddRow_C, 15)
#endif
#undef SAANY

#endif  // SASIMDONLY

// Definition for ScaleFilterCols, ScaleARGBCols and ScaleARGBFilterCols
#define CANY(NAMEANY, TERP_SIMD, TERP_C, BPP, MASK)                            \
  void NAMEANY(uint8_t* dst_ptr, const uint8_t* src_ptr, int dst_width, int x, \
               int dx) {                                                       \
    int r = dst_width & MASK;                                                  \
    int n = dst_width & ~MASK;                                                 \
    if (n > 0) {                                                               \
      TERP_SIMD(dst_ptr, src_ptr, n, x, dx);                                   \
    }                                                                          \
    TERP_C(dst_ptr + n * BPP, src_ptr, r, x + n * dx, dx);                     \
  }

#ifdef HAS_SCALEFILTERCOLS_NEON
CANY(ScaleFilterCols_Any_NEON, ScaleFilterCols_NEON, ScaleFilterCols_C, 1, 7)
#endif
#ifdef HAS_SCALEFILTERCOLS_MSA
CANY(ScaleFilterCols_Any_MSA, ScaleFilterCols_MSA, ScaleFilterCols_C, 1, 15)
#endif
#ifdef HAS_SCALEFILTERCOLS_LSX
CANY(ScaleFilterCols_Any_LSX, ScaleFilterCols_LSX, ScaleFilterCols_C, 1, 15)
#endif
#ifdef HAS_SCALEARGBCOLS_NEON
CANY(ScaleARGBCols_Any_NEON, ScaleARGBCols_NEON, ScaleARGBCols_C, 4, 7)
#endif
#ifdef HAS_SCALEARGBCOLS_MSA
CANY(ScaleARGBCols_Any_MSA, ScaleARGBCols_MSA, ScaleARGBCols_C, 4, 3)
#endif
#ifdef HAS_SCALEARGBCOLS_LSX
CANY(ScaleARGBCols_Any_LSX, ScaleARGBCols_LSX, ScaleARGBCols_C, 4, 3)
#endif
#ifdef HAS_SCALEARGBFILTERCOLS_NEON
CANY(ScaleARGBFilterCols_Any_NEON,
     ScaleARGBFilterCols_NEON,
     ScaleARGBFilterCols_C,
     4,
     3)
#endif
#ifdef HAS_SCALEARGBFILTERCOLS_MSA
CANY(ScaleARGBFilterCols_Any_MSA,
     ScaleARGBFilterCols_MSA,
     ScaleARGBFilterCols_C,
     4,
     7)
#endif
#ifdef HAS_SCALEARGBFILTERCOLS_LSX
CANY(ScaleARGBFilterCols_Any_LSX,
     ScaleARGBFilterCols_LSX,
     ScaleARGBFilterCols_C,
     4,
     7)
#endif
#undef CANY

// Scale up horizontally 2 times using linear filter.
#define SUH2LANY(NAME, SIMD, C, MASK, PTYPE)                       \
  void NAME(const PTYPE* src_ptr, PTYPE* dst_ptr, int dst_width) { \
    int work_width = (dst_width - 1) & ~1;                         \
    int r = work_width & MASK;                                     \
    int n = work_width & ~MASK;                                    \
    dst_ptr[0] = src_ptr[0];                                       \
    if (work_width > 0) {                                          \
      if (n != 0) {                                                \
        SIMD(src_ptr, dst_ptr + 1, n);                             \
      }                                                            \
      C(src_ptr + (n / 2), dst_ptr + n + 1, r);                    \
    }                                                              \
    dst_ptr[dst_width - 1] = src_ptr[(dst_width - 1) / 2];         \
  }

// Even the C versions need to be wrapped, because boundary pixels have to
// be handled differently

SUH2LANY(ScaleRowUp2_Linear_Any_C,
         ScaleRowUp2_Linear_C,
         ScaleRowUp2_Linear_C,
         0,
         uint8_t)

SUH2LANY(ScaleRowUp2_Linear_16_Any_C,
         ScaleRowUp2_Linear_16_C,
         ScaleRowUp2_Linear_16_C,
         0,
         uint16_t)

#ifdef HAS_SCALEROWUP2_LINEAR_SSE2
SUH2LANY(ScaleRowUp2_Linear_Any_SSE2,
         ScaleRowUp2_Linear_SSE2,
         ScaleRowUp2_Linear_C,
         15,
         uint8_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_SSSE3
SUH2LANY(ScaleRowUp2_Linear_Any_SSSE3,
         ScaleRowUp2_Linear_SSSE3,
         ScaleRowUp2_Linear_C,
         15,
         uint8_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_12_SSSE3
SUH2LANY(ScaleRowUp2_Linear_12_Any_SSSE3,
         ScaleRowUp2_Linear_12_SSSE3,
         ScaleRowUp2_Linear_16_C,
         15,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_16_SSE2
SUH2LANY(ScaleRowUp2_Linear_16_Any_SSE2,
         ScaleRowUp2_Linear_16_SSE2,
         ScaleRowUp2_Linear_16_C,
         7,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_AVX2
SUH2LANY(ScaleRowUp2_Linear_Any_AVX2,
         ScaleRowUp2_Linear_AVX2,
         ScaleRowUp2_Linear_C,
         31,
         uint8_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_12_AVX2
SUH2LANY(ScaleRowUp2_Linear_12_Any_AVX2,
         ScaleRowUp2_Linear_12_AVX2,
         ScaleRowUp2_Linear_16_C,
         31,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_16_AVX2
SUH2LANY(ScaleRowUp2_Linear_16_Any_AVX2,
         ScaleRowUp2_Linear_16_AVX2,
         ScaleRowUp2_Linear_16_C,
         15,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_NEON
#ifdef __aarch64__
SUH2LANY(ScaleRowUp2_Linear_Any_NEON,
         ScaleRowUp2_Linear_NEON,
         ScaleRowUp2_Linear_C,
         31,
         uint8_t)
#else
SUH2LANY(ScaleRowUp2_Linear_Any_NEON,
         ScaleRowUp2_Linear_NEON,
         ScaleRowUp2_Linear_C,
         15,
         uint8_t)
#endif
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_12_NEON
SUH2LANY(ScaleRowUp2_Linear_12_Any_NEON,
         ScaleRowUp2_Linear_12_NEON,
         ScaleRowUp2_Linear_16_C,
         15,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_LINEAR_16_NEON
SUH2LANY(ScaleRowUp2_Linear_16_Any_NEON,
         ScaleRowUp2_Linear_16_NEON,
         ScaleRowUp2_Linear_16_C,
         15,
         uint16_t)
#endif

#undef SUH2LANY

// Scale up 2 times using bilinear filter.
// This function produces 2 rows at a time.
#define SU2BLANY(NAME, SIMD, C, MASK, PTYPE)                              \
  void NAME(const PTYPE* src_ptr, ptrdiff_t src_stride, PTYPE* dst_ptr,   \
            ptrdiff_t dst_stride, int dst_width) {                        \
    int work_width = (dst_width - 1) & ~1;                                \
    int r = work_width & MASK;                                            \
    int n = work_width & ~MASK;                                           \
    const PTYPE* sa = src_ptr;                                            \
    const PTYPE* sb = src_ptr + src_stride;                               \
    PTYPE* da = dst_ptr;                                                  \
    PTYPE* db = dst_ptr + dst_stride;                                     \
    da[0] = (3 * sa[0] + sb[0] + 2) >> 2;                                 \
    db[0] = (sa[0] + 3 * sb[0] + 2) >> 2;                                 \
    if (work_width > 0) {                                                 \
      if (n != 0) {                                                       \
        SIMD(sa, sb - sa, da + 1, db - da, n);                            \
      }                                                                   \
      C(sa + (n / 2), sb - sa, da + n + 1, db - da, r);                   \
    }                                                                     \
    da[dst_width - 1] =                                                   \
        (3 * sa[(dst_width - 1) / 2] + sb[(dst_width - 1) / 2] + 2) >> 2; \
    db[dst_width - 1] =                                                   \
        (sa[(dst_width - 1) / 2] + 3 * sb[(dst_width - 1) / 2] + 2) >> 2; \
  }

SU2BLANY(ScaleRowUp2_Bilinear_Any_C,
         ScaleRowUp2_Bilinear_C,
         ScaleRowUp2_Bilinear_C,
         0,
         uint8_t)

SU2BLANY(ScaleRowUp2_Bilinear_16_Any_C,
         ScaleRowUp2_Bilinear_16_C,
         ScaleRowUp2_Bilinear_16_C,
         0,
         uint16_t)

#ifdef HAS_SCALEROWUP2_BILINEAR_SSE2
SU2BLANY(ScaleRowUp2_Bilinear_Any_SSE2,
         ScaleRowUp2_Bilinear_SSE2,
         ScaleRowUp2_Bilinear_C,
         15,
         uint8_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_12_SSSE3
SU2BLANY(ScaleRowUp2_Bilinear_12_Any_SSSE3,
         ScaleRowUp2_Bilinear_12_SSSE3,
         ScaleRowUp2_Bilinear_16_C,
         15,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_16_SSE2
SU2BLANY(ScaleRowUp2_Bilinear_16_Any_SSE2,
         ScaleRowUp2_Bilinear_16_SSE2,
         ScaleRowUp2_Bilinear_16_C,
         7,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_SSSE3
SU2BLANY(ScaleRowUp2_Bilinear_Any_SSSE3,
         ScaleRowUp2_Bilinear_SSSE3,
         ScaleRowUp2_Bilinear_C,
         15,
         uint8_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_AVX2
SU2BLANY(ScaleRowUp2_Bilinear_Any_AVX2,
         ScaleRowUp2_Bilinear_AVX2,
         ScaleRowUp2_Bilinear_C,
         31,
         uint8_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_12_AVX2
SU2BLANY(ScaleRowUp2_Bilinear_12_Any_AVX2,
         ScaleRowUp2_Bilinear_12_AVX2,
         ScaleRowUp2_Bilinear_16_C,
         15,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_16_AVX2
SU2BLANY(ScaleRowUp2_Bilinear_16_Any_AVX2,
         ScaleRowUp2_Bilinear_16_AVX2,
         ScaleRowUp2_Bilinear_16_C,
         15,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_NEON
SU2BLANY(ScaleRowUp2_Bilinear_Any_NEON,
         ScaleRowUp2_Bilinear_NEON,
         ScaleRowUp2_Bilinear_C,
         15,
         uint8_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_12_NEON
SU2BLANY(ScaleRowUp2_Bilinear_12_Any_NEON,
         ScaleRowUp2_Bilinear_12_NEON,
         ScaleRowUp2_Bilinear_16_C,
         15,
         uint16_t)
#endif

#ifdef HAS_SCALEROWUP2_BILINEAR_16_NEON
SU2BLANY(ScaleRowUp2_Bilinear_16_Any_NEON,
         ScaleRowUp2_Bilinear_16_NEON,
         ScaleRowUp2_Bilinear_16_C,
         7,
         uint16_t)
#endif

#undef SU2BLANY

// Scale bi-planar plane up horizontally 2 times using linear filter.
#define SBUH2LANY(NAME, SIMD, C, MASK, PTYPE)                         \
  void NAME(const PTYPE* src_ptr, PTYPE* dst_ptr, int dst_width) {    \
    int work_width = (dst_width - 1) & ~1;                            \
    int r = work_width & MASK;                                        \
    int n = work_width & ~MASK;                                       \
    dst_ptr[0] = src_ptr[0];                                          \
    dst_ptr[1] = src_ptr[1];                                          \
    if (work_width > 0) {                                             \
      if (n != 0) {                                                   \
        SIMD(src_ptr, dst_ptr + 2, n);                                \
      }                                                               \
      C(src_ptr + n, dst_ptr + 2 * n + 2, r);                         \
    }                                                                 \
    dst_ptr[2 * dst_width - 2] = src_ptr[((dst_width + 1) & ~1) - 2]; \
    dst_ptr[2 * dst_width - 1] = src_ptr[((dst_width + 1) & ~1) - 1]; \
  }

SBUH2LANY(ScaleUVRowUp2_Linear_Any_C,
          ScaleUVRowUp2_Linear_C,
          ScaleUVRowUp2_Linear_C,
          0,
          uint8_t)

SBUH2LANY(ScaleUVRowUp2_Linear_16_Any_C,
          ScaleUVRowUp2_Linear_16_C,
          ScaleUVRowUp2_Linear_16_C,
          0,
          uint16_t)

#ifdef HAS_SCALEUVROWUP2_LINEAR_SSSE3
SBUH2LANY(ScaleUVRowUp2_Linear_Any_SSSE3,
          ScaleUVRowUp2_Linear_SSSE3,
          ScaleUVRowUp2_Linear_C,
          7,
          uint8_t)
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_AVX2
SBUH2LANY(ScaleUVRowUp2_Linear_Any_AVX2,
          ScaleUVRowUp2_Linear_AVX2,
          ScaleUVRowUp2_Linear_C,
          15,
          uint8_t)
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_SSE41
SBUH2LANY(ScaleUVRowUp2_Linear_16_Any_SSE41,
          ScaleUVRowUp2_Linear_16_SSE41,
          ScaleUVRowUp2_Linear_16_C,
          3,
          uint16_t)
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_AVX2
SBUH2LANY(ScaleUVRowUp2_Linear_16_Any_AVX2,
          ScaleUVRowUp2_Linear_16_AVX2,
          ScaleUVRowUp2_Linear_16_C,
          7,
          uint16_t)
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_NEON
SBUH2LANY(ScaleUVRowUp2_Linear_Any_NEON,
          ScaleUVRowUp2_Linear_NEON,
          ScaleUVRowUp2_Linear_C,
          15,
          uint8_t)
#endif

#ifdef HAS_SCALEUVROWUP2_LINEAR_16_NEON
SBUH2LANY(ScaleUVRowUp2_Linear_16_Any_NEON,
          ScaleUVRowUp2_Linear_16_NEON,
          ScaleUVRowUp2_Linear_16_C,
          15,
          uint16_t)
#endif

#undef SBUH2LANY

// Scale bi-planar plane up 2 times using bilinear filter.
// This function produces 2 rows at a time.
#define SBU2BLANY(NAME, SIMD, C, MASK, PTYPE)                           \
  void NAME(const PTYPE* src_ptr, ptrdiff_t src_stride, PTYPE* dst_ptr, \
            ptrdiff_t dst_stride, int dst_width) {                      \
    int work_width = (dst_width - 1) & ~1;                              \
    int r = work_width & MASK;                                          \
    int n = work_width & ~MASK;                                         \
    const PTYPE* sa = src_ptr;                                          \
    const PTYPE* sb = src_ptr + src_stride;                             \
    PTYPE* da = dst_ptr;                                                \
    PTYPE* db = dst_ptr + dst_stride;                                   \
    da[0] = (3 * sa[0] + sb[0] + 2) >> 2;                               \
    db[0] = (sa[0] + 3 * sb[0] + 2) >> 2;                               \
    da[1] = (3 * sa[1] + sb[1] + 2) >> 2;                               \
    db[1] = (sa[1] + 3 * sb[1] + 2) >> 2;                               \
    if (work_width > 0) {                                               \
      if (n != 0) {                                                     \
        SIMD(sa, sb - sa, da + 2, db - da, n);                          \
      }                                                                 \
      C(sa + n, sb - sa, da + 2 * n + 2, db - da, r);                   \
    }                                                                   \
    da[2 * dst_width - 2] = (3 * sa[((dst_width + 1) & ~1) - 2] +       \
                             sb[((dst_width + 1) & ~1) - 2] + 2) >>     \
                            2;                                          \
    db[2 * dst_width - 2] = (sa[((dst_width + 1) & ~1) - 2] +           \
                             3 * sb[((dst_width + 1) & ~1) - 2] + 2) >> \
                            2;                                          \
    da[2 * dst_width - 1] = (3 * sa[((dst_width + 1) & ~1) - 1] +       \
                             sb[((dst_width + 1) & ~1) - 1] + 2) >>     \
                            2;                                          \
    db[2 * dst_width - 1] = (sa[((dst_width + 1) & ~1) - 1] +           \
                             3 * sb[((dst_width + 1) & ~1) - 1] + 2) >> \
                            2;                                          \
  }

SBU2BLANY(ScaleUVRowUp2_Bilinear_Any_C,
          ScaleUVRowUp2_Bilinear_C,
          ScaleUVRowUp2_Bilinear_C,
          0,
          uint8_t)

SBU2BLANY(ScaleUVRowUp2_Bilinear_16_Any_C,
          ScaleUVRowUp2_Bilinear_16_C,
          ScaleUVRowUp2_Bilinear_16_C,
          0,
          uint16_t)

#ifdef HAS_SCALEUVROWUP2_BILINEAR_SSSE3
SBU2BLANY(ScaleUVRowUp2_Bilinear_Any_SSSE3,
          ScaleUVRowUp2_Bilinear_SSSE3,
          ScaleUVRowUp2_Bilinear_C,
          7,
          uint8_t)
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_AVX2
SBU2BLANY(ScaleUVRowUp2_Bilinear_Any_AVX2,
          ScaleUVRowUp2_Bilinear_AVX2,
          ScaleUVRowUp2_Bilinear_C,
          15,
          uint8_t)
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_SSE41
SBU2BLANY(ScaleUVRowUp2_Bilinear_16_Any_SSE41,
          ScaleUVRowUp2_Bilinear_16_SSE41,
          ScaleUVRowUp2_Bilinear_16_C,
          7,
          uint16_t)
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_AVX2
SBU2BLANY(ScaleUVRowUp2_Bilinear_16_Any_AVX2,
          ScaleUVRowUp2_Bilinear_16_AVX2,
          ScaleUVRowUp2_Bilinear_16_C,
          7,
          uint16_t)
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_NEON
SBU2BLANY(ScaleUVRowUp2_Bilinear_Any_NEON,
          ScaleUVRowUp2_Bilinear_NEON,
          ScaleUVRowUp2_Bilinear_C,
          7,
          uint8_t)
#endif

#ifdef HAS_SCALEUVROWUP2_BILINEAR_16_NEON
SBU2BLANY(ScaleUVRowUp2_Bilinear_16_Any_NEON,
          ScaleUVRowUp2_Bilinear_16_NEON,
          ScaleUVRowUp2_Bilinear_16_C,
          7,
          uint16_t)
#endif

#undef SBU2BLANY

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
