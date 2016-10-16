/*
 *  Copyright 2015 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/scale.h"
#include "libyuv/scale_row.h"

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Definition for ScaleFilterCols, ScaleARGBCols and ScaleARGBFilterCols
#define CANY(NAMEANY, TERP_SIMD, TERP_C, BPP, MASK)                            \
    void NAMEANY(uint8* dst_ptr, const uint8* src_ptr,                         \
                 int dst_width, int x, int dx) {                               \
      int n = dst_width & ~MASK;                                               \
      if (n > 0) {                                                             \
        TERP_SIMD(dst_ptr, src_ptr, n, x, dx);                                 \
      }                                                                        \
      TERP_C(dst_ptr + n * BPP, src_ptr,                                       \
             dst_width & MASK, x + n * dx, dx);                                \
    }

#ifdef HAS_SCALEFILTERCOLS_NEON
CANY(ScaleFilterCols_Any_NEON, ScaleFilterCols_NEON, ScaleFilterCols_C, 1, 7)
#endif
#ifdef HAS_SCALEARGBCOLS_NEON
CANY(ScaleARGBCols_Any_NEON, ScaleARGBCols_NEON, ScaleARGBCols_C, 4, 7)
#endif
#ifdef HAS_SCALEARGBFILTERCOLS_NEON
CANY(ScaleARGBFilterCols_Any_NEON, ScaleARGBFilterCols_NEON,
     ScaleARGBFilterCols_C, 4, 3)
#endif
#undef CANY

// Fixed scale down.
#define SDANY(NAMEANY, SCALEROWDOWN_SIMD, SCALEROWDOWN_C, FACTOR, BPP, MASK)   \
    void NAMEANY(const uint8* src_ptr, ptrdiff_t src_stride,                   \
                 uint8* dst_ptr, int dst_width) {                              \
      int r = (int)((unsigned int)dst_width % (MASK + 1));                     \
      int n = dst_width - r;                                                   \
      if (n > 0) {                                                             \
        SCALEROWDOWN_SIMD(src_ptr, src_stride, dst_ptr, n);                    \
      }                                                                        \
      SCALEROWDOWN_C(src_ptr + (n * FACTOR) * BPP, src_stride,                 \
                     dst_ptr + n * BPP, r);                                    \
    }

// Fixed scale down for odd source width.  Used by I420Blend subsampling.
// Since dst_width is (width + 1) / 2, this function scales one less pixel
// and copies the last pixel.
#define SDODD(NAMEANY, SCALEROWDOWN_SIMD, SCALEROWDOWN_C, FACTOR, BPP, MASK)   \
    void NAMEANY(const uint8* src_ptr, ptrdiff_t src_stride,                   \
                 uint8* dst_ptr, int dst_width) {                              \
      int r = (int)((unsigned int)(dst_width - 1) % (MASK + 1));               \
      int n = dst_width - r;                                                   \
      if (n > 0) {                                                             \
        SCALEROWDOWN_SIMD(src_ptr, src_stride, dst_ptr, n);                    \
      }                                                                        \
      SCALEROWDOWN_C(src_ptr + (n * FACTOR) * BPP, src_stride,                 \
                     dst_ptr + n * BPP, r);                                    \
    }

#ifdef HAS_SCALEROWDOWN2_SSSE3
SDANY(ScaleRowDown2_Any_SSSE3, ScaleRowDown2_SSSE3, ScaleRowDown2_C, 2, 1, 15)
SDANY(ScaleRowDown2Linear_Any_SSSE3, ScaleRowDown2Linear_SSSE3,
      ScaleRowDown2Linear_C, 2, 1, 15)
SDANY(ScaleRowDown2Box_Any_SSSE3, ScaleRowDown2Box_SSSE3, ScaleRowDown2Box_C,
      2, 1, 15)
SDODD(ScaleRowDown2Box_Odd_SSSE3, ScaleRowDown2Box_SSSE3,
      ScaleRowDown2Box_Odd_C, 2, 1, 15)
#endif
#ifdef HAS_SCALEROWDOWN2_AVX2
SDANY(ScaleRowDown2_Any_AVX2, ScaleRowDown2_AVX2, ScaleRowDown2_C, 2, 1, 31)
SDANY(ScaleRowDown2Linear_Any_AVX2, ScaleRowDown2Linear_AVX2,
      ScaleRowDown2Linear_C, 2, 1, 31)
SDANY(ScaleRowDown2Box_Any_AVX2, ScaleRowDown2Box_AVX2, ScaleRowDown2Box_C,
      2, 1, 31)
SDODD(ScaleRowDown2Box_Odd_AVX2, ScaleRowDown2Box_AVX2, ScaleRowDown2Box_Odd_C,
      2, 1, 31)
#endif
#ifdef HAS_SCALEROWDOWN2_NEON
SDANY(ScaleRowDown2_Any_NEON, ScaleRowDown2_NEON, ScaleRowDown2_C, 2, 1, 15)
SDANY(ScaleRowDown2Linear_Any_NEON, ScaleRowDown2Linear_NEON,
      ScaleRowDown2Linear_C, 2, 1, 15)
SDANY(ScaleRowDown2Box_Any_NEON, ScaleRowDown2Box_NEON,
      ScaleRowDown2Box_C, 2, 1, 15)
SDODD(ScaleRowDown2Box_Odd_NEON, ScaleRowDown2Box_NEON,
      ScaleRowDown2Box_Odd_C, 2, 1, 15)
#endif
#ifdef HAS_SCALEROWDOWN4_SSSE3
SDANY(ScaleRowDown4_Any_SSSE3, ScaleRowDown4_SSSE3, ScaleRowDown4_C, 4, 1, 7)
SDANY(ScaleRowDown4Box_Any_SSSE3, ScaleRowDown4Box_SSSE3, ScaleRowDown4Box_C,
      4, 1, 7)
#endif
#ifdef HAS_SCALEROWDOWN4_AVX2
SDANY(ScaleRowDown4_Any_AVX2, ScaleRowDown4_AVX2, ScaleRowDown4_C, 4, 1, 15)
SDANY(ScaleRowDown4Box_Any_AVX2, ScaleRowDown4Box_AVX2, ScaleRowDown4Box_C,
      4, 1, 15)
#endif
#ifdef HAS_SCALEROWDOWN4_NEON
SDANY(ScaleRowDown4_Any_NEON, ScaleRowDown4_NEON, ScaleRowDown4_C, 4, 1, 7)
SDANY(ScaleRowDown4Box_Any_NEON, ScaleRowDown4Box_NEON, ScaleRowDown4Box_C,
      4, 1, 7)
#endif
#ifdef HAS_SCALEROWDOWN34_SSSE3
SDANY(ScaleRowDown34_Any_SSSE3, ScaleRowDown34_SSSE3,
      ScaleRowDown34_C, 4 / 3, 1, 23)
SDANY(ScaleRowDown34_0_Box_Any_SSSE3, ScaleRowDown34_0_Box_SSSE3,
      ScaleRowDown34_0_Box_C, 4 / 3, 1, 23)
SDANY(ScaleRowDown34_1_Box_Any_SSSE3, ScaleRowDown34_1_Box_SSSE3,
      ScaleRowDown34_1_Box_C, 4 / 3, 1, 23)
#endif
#ifdef HAS_SCALEROWDOWN34_NEON
SDANY(ScaleRowDown34_Any_NEON, ScaleRowDown34_NEON,
      ScaleRowDown34_C, 4 / 3, 1, 23)
SDANY(ScaleRowDown34_0_Box_Any_NEON, ScaleRowDown34_0_Box_NEON,
      ScaleRowDown34_0_Box_C, 4 / 3, 1, 23)
SDANY(ScaleRowDown34_1_Box_Any_NEON, ScaleRowDown34_1_Box_NEON,
      ScaleRowDown34_1_Box_C, 4 / 3, 1, 23)
#endif
#ifdef HAS_SCALEROWDOWN38_SSSE3
SDANY(ScaleRowDown38_Any_SSSE3, ScaleRowDown38_SSSE3,
      ScaleRowDown38_C, 8 / 3, 1, 11)
SDANY(ScaleRowDown38_3_Box_Any_SSSE3, ScaleRowDown38_3_Box_SSSE3,
      ScaleRowDown38_3_Box_C, 8 / 3, 1, 5)
SDANY(ScaleRowDown38_2_Box_Any_SSSE3, ScaleRowDown38_2_Box_SSSE3,
      ScaleRowDown38_2_Box_C, 8 / 3, 1, 5)
#endif
#ifdef HAS_SCALEROWDOWN38_NEON
SDANY(ScaleRowDown38_Any_NEON, ScaleRowDown38_NEON,
      ScaleRowDown38_C, 8 / 3, 1, 11)
SDANY(ScaleRowDown38_3_Box_Any_NEON, ScaleRowDown38_3_Box_NEON,
      ScaleRowDown38_3_Box_C, 8 / 3, 1, 11)
SDANY(ScaleRowDown38_2_Box_Any_NEON, ScaleRowDown38_2_Box_NEON,
      ScaleRowDown38_2_Box_C, 8 / 3, 1, 11)
#endif

#ifdef HAS_SCALEARGBROWDOWN2_SSE2
SDANY(ScaleARGBRowDown2_Any_SSE2, ScaleARGBRowDown2_SSE2,
      ScaleARGBRowDown2_C, 2, 4, 3)
SDANY(ScaleARGBRowDown2Linear_Any_SSE2, ScaleARGBRowDown2Linear_SSE2,
      ScaleARGBRowDown2Linear_C, 2, 4, 3)
SDANY(ScaleARGBRowDown2Box_Any_SSE2, ScaleARGBRowDown2Box_SSE2,
      ScaleARGBRowDown2Box_C, 2, 4, 3)
#endif
#ifdef HAS_SCALEARGBROWDOWN2_NEON
SDANY(ScaleARGBRowDown2_Any_NEON, ScaleARGBRowDown2_NEON,
      ScaleARGBRowDown2_C, 2, 4, 7)
SDANY(ScaleARGBRowDown2Linear_Any_NEON, ScaleARGBRowDown2Linear_NEON,
      ScaleARGBRowDown2Linear_C, 2, 4, 7)
SDANY(ScaleARGBRowDown2Box_Any_NEON, ScaleARGBRowDown2Box_NEON,
      ScaleARGBRowDown2Box_C, 2, 4, 7)
#endif
#undef SDANY

// Scale down by even scale factor.
#define SDAANY(NAMEANY, SCALEROWDOWN_SIMD, SCALEROWDOWN_C, BPP, MASK)          \
    void NAMEANY(const uint8* src_ptr, ptrdiff_t src_stride, int src_stepx,    \
                 uint8* dst_ptr, int dst_width) {                              \
      int r = (int)((unsigned int)dst_width % (MASK + 1));                     \
      int n = dst_width - r;                                                   \
      if (n > 0) {                                                             \
        SCALEROWDOWN_SIMD(src_ptr, src_stride, src_stepx, dst_ptr, n);         \
      }                                                                        \
      SCALEROWDOWN_C(src_ptr + (n * src_stepx) * BPP, src_stride,              \
                     src_stepx, dst_ptr + n * BPP, r);                         \
    }

#ifdef HAS_SCALEARGBROWDOWNEVEN_SSE2
SDAANY(ScaleARGBRowDownEven_Any_SSE2, ScaleARGBRowDownEven_SSE2,
       ScaleARGBRowDownEven_C, 4, 3)
SDAANY(ScaleARGBRowDownEvenBox_Any_SSE2, ScaleARGBRowDownEvenBox_SSE2,
       ScaleARGBRowDownEvenBox_C, 4, 3)
#endif
#ifdef HAS_SCALEARGBROWDOWNEVEN_NEON
SDAANY(ScaleARGBRowDownEven_Any_NEON, ScaleARGBRowDownEven_NEON,
       ScaleARGBRowDownEven_C, 4, 3)
SDAANY(ScaleARGBRowDownEvenBox_Any_NEON, ScaleARGBRowDownEvenBox_NEON,
       ScaleARGBRowDownEvenBox_C, 4, 3)
#endif

// Add rows box filter scale down.
#define SAANY(NAMEANY, SCALEADDROW_SIMD, SCALEADDROW_C, MASK)                  \
  void NAMEANY(const uint8* src_ptr, uint16* dst_ptr, int src_width) {         \
      int n = src_width & ~MASK;                                               \
      if (n > 0) {                                                             \
        SCALEADDROW_SIMD(src_ptr, dst_ptr, n);                                 \
      }                                                                        \
      SCALEADDROW_C(src_ptr + n, dst_ptr + n, src_width & MASK);               \
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
#undef SAANY

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif





