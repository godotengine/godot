/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/row.h"

#include <string.h>  // For memset.

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Subsampled source needs to be increase by 1 of not even.
#define SS(width, shift) (((width) + (1 << (shift)) - 1) >> (shift))

// Any 4 planes to 1 with yuvconstants
#define ANY41C(NAMEANY, ANY_SIMD, UVSHIFT, DUVSHIFT, BPP, MASK)                \
    void NAMEANY(const uint8* y_buf, const uint8* u_buf, const uint8* v_buf,   \
                 const uint8* a_buf, uint8* dst_ptr,                           \
                 const struct YuvConstants* yuvconstants,  int width) {        \
      SIMD_ALIGNED(uint8 temp[64 * 5]);                                        \
      memset(temp, 0, 64 * 4);  /* for msan */                                 \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(y_buf, u_buf, v_buf, a_buf, dst_ptr, yuvconstants, n);        \
      }                                                                        \
      memcpy(temp, y_buf + n, r);                                              \
      memcpy(temp + 64, u_buf + (n >> UVSHIFT), SS(r, UVSHIFT));               \
      memcpy(temp + 128, v_buf + (n >> UVSHIFT), SS(r, UVSHIFT));              \
      memcpy(temp + 192, a_buf + n, r);                                        \
      ANY_SIMD(temp, temp + 64, temp + 128, temp + 192, temp + 256,            \
               yuvconstants, MASK + 1);                                        \
      memcpy(dst_ptr + (n >> DUVSHIFT) * BPP, temp + 256,                      \
             SS(r, DUVSHIFT) * BPP);                                           \
    }

#ifdef HAS_I422ALPHATOARGBROW_SSSE3
ANY41C(I422AlphaToARGBRow_Any_SSSE3, I422AlphaToARGBRow_SSSE3, 1, 0, 4, 7)
#endif
#ifdef HAS_I422ALPHATOARGBROW_AVX2
ANY41C(I422AlphaToARGBRow_Any_AVX2, I422AlphaToARGBRow_AVX2, 1, 0, 4, 15)
#endif
#ifdef HAS_I422ALPHATOARGBROW_NEON
ANY41C(I422AlphaToARGBRow_Any_NEON, I422AlphaToARGBRow_NEON, 1, 0, 4, 7)
#endif
#undef ANY41C

// Any 3 planes to 1.
#define ANY31(NAMEANY, ANY_SIMD, UVSHIFT, DUVSHIFT, BPP, MASK)                 \
    void NAMEANY(const uint8* y_buf, const uint8* u_buf, const uint8* v_buf,   \
                 uint8* dst_ptr, int width) {                                  \
      SIMD_ALIGNED(uint8 temp[64 * 4]);                                        \
      memset(temp, 0, 64 * 3);  /* for YUY2 and msan */                        \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(y_buf, u_buf, v_buf, dst_ptr, n);                             \
      }                                                                        \
      memcpy(temp, y_buf + n, r);                                              \
      memcpy(temp + 64, u_buf + (n >> UVSHIFT), SS(r, UVSHIFT));               \
      memcpy(temp + 128, v_buf + (n >> UVSHIFT), SS(r, UVSHIFT));              \
      ANY_SIMD(temp, temp + 64, temp + 128, temp + 192, MASK + 1);             \
      memcpy(dst_ptr + (n >> DUVSHIFT) * BPP, temp + 192,                      \
             SS(r, DUVSHIFT) * BPP);                                           \
    }
#ifdef HAS_I422TOYUY2ROW_SSE2
ANY31(I422ToYUY2Row_Any_SSE2, I422ToYUY2Row_SSE2, 1, 1, 4, 15)
ANY31(I422ToUYVYRow_Any_SSE2, I422ToUYVYRow_SSE2, 1, 1, 4, 15)
#endif
#ifdef HAS_I422TOYUY2ROW_NEON
ANY31(I422ToYUY2Row_Any_NEON, I422ToYUY2Row_NEON, 1, 1, 4, 15)
#endif
#ifdef HAS_I422TOUYVYROW_NEON
ANY31(I422ToUYVYRow_Any_NEON, I422ToUYVYRow_NEON, 1, 1, 4, 15)
#endif
#ifdef HAS_BLENDPLANEROW_AVX2
ANY31(BlendPlaneRow_Any_AVX2, BlendPlaneRow_AVX2, 0, 0, 1, 31)
#endif
#ifdef HAS_BLENDPLANEROW_SSSE3
ANY31(BlendPlaneRow_Any_SSSE3, BlendPlaneRow_SSSE3, 0, 0, 1, 7)
#endif
#undef ANY31

// Note that odd width replication includes 444 due to implementation
// on arm that subsamples 444 to 422 internally.
// Any 3 planes to 1 with yuvconstants
#define ANY31C(NAMEANY, ANY_SIMD, UVSHIFT, DUVSHIFT, BPP, MASK)                \
    void NAMEANY(const uint8* y_buf, const uint8* u_buf, const uint8* v_buf,   \
                 uint8* dst_ptr, const struct YuvConstants* yuvconstants,      \
                 int width) {                                                  \
      SIMD_ALIGNED(uint8 temp[64 * 4]);                                        \
      memset(temp, 0, 64 * 3);  /* for YUY2 and msan */                        \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(y_buf, u_buf, v_buf, dst_ptr, yuvconstants, n);               \
      }                                                                        \
      memcpy(temp, y_buf + n, r);                                              \
      memcpy(temp + 64, u_buf + (n >> UVSHIFT), SS(r, UVSHIFT));               \
      memcpy(temp + 128, v_buf + (n >> UVSHIFT), SS(r, UVSHIFT));              \
      if (width & 1) {                                                         \
        temp[64 + SS(r, UVSHIFT)] = temp[64 + SS(r, UVSHIFT) - 1];             \
        temp[128 + SS(r, UVSHIFT)] = temp[128 + SS(r, UVSHIFT) - 1];           \
      }                                                                        \
      ANY_SIMD(temp, temp + 64, temp + 128, temp + 192,                        \
               yuvconstants, MASK + 1);                                        \
      memcpy(dst_ptr + (n >> DUVSHIFT) * BPP, temp + 192,                      \
             SS(r, DUVSHIFT) * BPP);                                           \
    }

#ifdef HAS_I422TOARGBROW_SSSE3
ANY31C(I422ToARGBRow_Any_SSSE3, I422ToARGBRow_SSSE3, 1, 0, 4, 7)
#endif
#ifdef HAS_I411TOARGBROW_SSSE3
ANY31C(I411ToARGBRow_Any_SSSE3, I411ToARGBRow_SSSE3, 2, 0, 4, 7)
#endif
#ifdef HAS_I444TOARGBROW_SSSE3
ANY31C(I444ToARGBRow_Any_SSSE3, I444ToARGBRow_SSSE3, 0, 0, 4, 7)
ANY31C(I422ToRGBARow_Any_SSSE3, I422ToRGBARow_SSSE3, 1, 0, 4, 7)
ANY31C(I422ToARGB4444Row_Any_SSSE3, I422ToARGB4444Row_SSSE3, 1, 0, 2, 7)
ANY31C(I422ToARGB1555Row_Any_SSSE3, I422ToARGB1555Row_SSSE3, 1, 0, 2, 7)
ANY31C(I422ToRGB565Row_Any_SSSE3, I422ToRGB565Row_SSSE3, 1, 0, 2, 7)
ANY31C(I422ToRGB24Row_Any_SSSE3, I422ToRGB24Row_SSSE3, 1, 0, 3, 7)
#endif  // HAS_I444TOARGBROW_SSSE3
#ifdef HAS_I422TORGB24ROW_AVX2
ANY31C(I422ToRGB24Row_Any_AVX2, I422ToRGB24Row_AVX2, 1, 0, 3, 15)
#endif
#ifdef HAS_I422TOARGBROW_AVX2
ANY31C(I422ToARGBRow_Any_AVX2, I422ToARGBRow_AVX2, 1, 0, 4, 15)
#endif
#ifdef HAS_I422TORGBAROW_AVX2
ANY31C(I422ToRGBARow_Any_AVX2, I422ToRGBARow_AVX2, 1, 0, 4, 15)
#endif
#ifdef HAS_I444TOARGBROW_AVX2
ANY31C(I444ToARGBRow_Any_AVX2, I444ToARGBRow_AVX2, 0, 0, 4, 15)
#endif
#ifdef HAS_I411TOARGBROW_AVX2
ANY31C(I411ToARGBRow_Any_AVX2, I411ToARGBRow_AVX2, 2, 0, 4, 15)
#endif
#ifdef HAS_I422TOARGB4444ROW_AVX2
ANY31C(I422ToARGB4444Row_Any_AVX2, I422ToARGB4444Row_AVX2, 1, 0, 2, 7)
#endif
#ifdef HAS_I422TOARGB1555ROW_AVX2
ANY31C(I422ToARGB1555Row_Any_AVX2, I422ToARGB1555Row_AVX2, 1, 0, 2, 7)
#endif
#ifdef HAS_I422TORGB565ROW_AVX2
ANY31C(I422ToRGB565Row_Any_AVX2, I422ToRGB565Row_AVX2, 1, 0, 2, 7)
#endif
#ifdef HAS_I422TOARGBROW_NEON
ANY31C(I444ToARGBRow_Any_NEON, I444ToARGBRow_NEON, 0, 0, 4, 7)
ANY31C(I422ToARGBRow_Any_NEON, I422ToARGBRow_NEON, 1, 0, 4, 7)
ANY31C(I411ToARGBRow_Any_NEON, I411ToARGBRow_NEON, 2, 0, 4, 7)
ANY31C(I422ToRGBARow_Any_NEON, I422ToRGBARow_NEON, 1, 0, 4, 7)
ANY31C(I422ToRGB24Row_Any_NEON, I422ToRGB24Row_NEON, 1, 0, 3, 7)
ANY31C(I422ToARGB4444Row_Any_NEON, I422ToARGB4444Row_NEON, 1, 0, 2, 7)
ANY31C(I422ToARGB1555Row_Any_NEON, I422ToARGB1555Row_NEON, 1, 0, 2, 7)
ANY31C(I422ToRGB565Row_Any_NEON, I422ToRGB565Row_NEON, 1, 0, 2, 7)
#endif
#undef ANY31C

// Any 2 planes to 1.
#define ANY21(NAMEANY, ANY_SIMD, UVSHIFT, SBPP, SBPP2, BPP, MASK)              \
    void NAMEANY(const uint8* y_buf, const uint8* uv_buf,                      \
                 uint8* dst_ptr, int width) {                                  \
      SIMD_ALIGNED(uint8 temp[64 * 3]);                                        \
      memset(temp, 0, 64 * 2);  /* for msan */                                 \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(y_buf, uv_buf, dst_ptr, n);                                   \
      }                                                                        \
      memcpy(temp, y_buf + n * SBPP, r * SBPP);                                \
      memcpy(temp + 64, uv_buf + (n >> UVSHIFT) * SBPP2,                       \
             SS(r, UVSHIFT) * SBPP2);                                          \
      ANY_SIMD(temp, temp + 64, temp + 128, MASK + 1);                         \
      memcpy(dst_ptr + n * BPP, temp + 128, r * BPP);                          \
    }

// Merge functions.
#ifdef HAS_MERGEUVROW_SSE2
ANY21(MergeUVRow_Any_SSE2, MergeUVRow_SSE2, 0, 1, 1, 2, 15)
#endif
#ifdef HAS_MERGEUVROW_AVX2
ANY21(MergeUVRow_Any_AVX2, MergeUVRow_AVX2, 0, 1, 1, 2, 31)
#endif
#ifdef HAS_MERGEUVROW_NEON
ANY21(MergeUVRow_Any_NEON, MergeUVRow_NEON, 0, 1, 1, 2, 15)
#endif

// Math functions.
#ifdef HAS_ARGBMULTIPLYROW_SSE2
ANY21(ARGBMultiplyRow_Any_SSE2, ARGBMultiplyRow_SSE2, 0, 4, 4, 4, 3)
#endif
#ifdef HAS_ARGBADDROW_SSE2
ANY21(ARGBAddRow_Any_SSE2, ARGBAddRow_SSE2, 0, 4, 4, 4, 3)
#endif
#ifdef HAS_ARGBSUBTRACTROW_SSE2
ANY21(ARGBSubtractRow_Any_SSE2, ARGBSubtractRow_SSE2, 0, 4, 4, 4, 3)
#endif
#ifdef HAS_ARGBMULTIPLYROW_AVX2
ANY21(ARGBMultiplyRow_Any_AVX2, ARGBMultiplyRow_AVX2, 0, 4, 4, 4, 7)
#endif
#ifdef HAS_ARGBADDROW_AVX2
ANY21(ARGBAddRow_Any_AVX2, ARGBAddRow_AVX2, 0, 4, 4, 4, 7)
#endif
#ifdef HAS_ARGBSUBTRACTROW_AVX2
ANY21(ARGBSubtractRow_Any_AVX2, ARGBSubtractRow_AVX2, 0, 4, 4, 4, 7)
#endif
#ifdef HAS_ARGBMULTIPLYROW_NEON
ANY21(ARGBMultiplyRow_Any_NEON, ARGBMultiplyRow_NEON, 0, 4, 4, 4, 7)
#endif
#ifdef HAS_ARGBADDROW_NEON
ANY21(ARGBAddRow_Any_NEON, ARGBAddRow_NEON, 0, 4, 4, 4, 7)
#endif
#ifdef HAS_ARGBSUBTRACTROW_NEON
ANY21(ARGBSubtractRow_Any_NEON, ARGBSubtractRow_NEON, 0, 4, 4, 4, 7)
#endif
#ifdef HAS_SOBELROW_SSE2
ANY21(SobelRow_Any_SSE2, SobelRow_SSE2, 0, 1, 1, 4, 15)
#endif
#ifdef HAS_SOBELROW_NEON
ANY21(SobelRow_Any_NEON, SobelRow_NEON, 0, 1, 1, 4, 7)
#endif
#ifdef HAS_SOBELTOPLANEROW_SSE2
ANY21(SobelToPlaneRow_Any_SSE2, SobelToPlaneRow_SSE2, 0, 1, 1, 1, 15)
#endif
#ifdef HAS_SOBELTOPLANEROW_NEON
ANY21(SobelToPlaneRow_Any_NEON, SobelToPlaneRow_NEON, 0, 1, 1, 1, 15)
#endif
#ifdef HAS_SOBELXYROW_SSE2
ANY21(SobelXYRow_Any_SSE2, SobelXYRow_SSE2, 0, 1, 1, 4, 15)
#endif
#ifdef HAS_SOBELXYROW_NEON
ANY21(SobelXYRow_Any_NEON, SobelXYRow_NEON, 0, 1, 1, 4, 7)
#endif
#undef ANY21

// Any 2 planes to 1 with yuvconstants
#define ANY21C(NAMEANY, ANY_SIMD, UVSHIFT, SBPP, SBPP2, BPP, MASK)             \
    void NAMEANY(const uint8* y_buf, const uint8* uv_buf,                      \
                 uint8* dst_ptr, const struct YuvConstants* yuvconstants,      \
                 int width) {                                                  \
      SIMD_ALIGNED(uint8 temp[64 * 3]);                                        \
      memset(temp, 0, 64 * 2);  /* for msan */                                 \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(y_buf, uv_buf, dst_ptr, yuvconstants, n);                     \
      }                                                                        \
      memcpy(temp, y_buf + n * SBPP, r * SBPP);                                \
      memcpy(temp + 64, uv_buf + (n >> UVSHIFT) * SBPP2,                       \
             SS(r, UVSHIFT) * SBPP2);                                          \
      ANY_SIMD(temp, temp + 64, temp + 128, yuvconstants, MASK + 1);           \
      memcpy(dst_ptr + n * BPP, temp + 128, r * BPP);                          \
    }

// Biplanar to RGB.
#ifdef HAS_NV12TOARGBROW_SSSE3
ANY21C(NV12ToARGBRow_Any_SSSE3, NV12ToARGBRow_SSSE3, 1, 1, 2, 4, 7)
#endif
#ifdef HAS_NV12TOARGBROW_AVX2
ANY21C(NV12ToARGBRow_Any_AVX2, NV12ToARGBRow_AVX2, 1, 1, 2, 4, 15)
#endif
#ifdef HAS_NV12TOARGBROW_NEON
ANY21C(NV12ToARGBRow_Any_NEON, NV12ToARGBRow_NEON, 1, 1, 2, 4, 7)
#endif
#ifdef HAS_NV21TOARGBROW_SSSE3
ANY21C(NV21ToARGBRow_Any_SSSE3, NV21ToARGBRow_SSSE3, 1, 1, 2, 4, 7)
#endif
#ifdef HAS_NV21TOARGBROW_AVX2
ANY21C(NV21ToARGBRow_Any_AVX2, NV21ToARGBRow_AVX2, 1, 1, 2, 4, 15)
#endif
#ifdef HAS_NV21TOARGBROW_NEON
ANY21C(NV21ToARGBRow_Any_NEON, NV21ToARGBRow_NEON, 1, 1, 2, 4, 7)
#endif
#ifdef HAS_NV12TORGB565ROW_SSSE3
ANY21C(NV12ToRGB565Row_Any_SSSE3, NV12ToRGB565Row_SSSE3, 1, 1, 2, 2, 7)
#endif
#ifdef HAS_NV12TORGB565ROW_AVX2
ANY21C(NV12ToRGB565Row_Any_AVX2, NV12ToRGB565Row_AVX2, 1, 1, 2, 2, 15)
#endif
#ifdef HAS_NV12TORGB565ROW_NEON
ANY21C(NV12ToRGB565Row_Any_NEON, NV12ToRGB565Row_NEON, 1, 1, 2, 2, 7)
#endif
#undef ANY21C

// Any 1 to 1.
#define ANY11(NAMEANY, ANY_SIMD, UVSHIFT, SBPP, BPP, MASK)                     \
    void NAMEANY(const uint8* src_ptr, uint8* dst_ptr, int width) {            \
      SIMD_ALIGNED(uint8 temp[128 * 2]);                                       \
      memset(temp, 0, 128);  /* for YUY2 and msan */                           \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(src_ptr, dst_ptr, n);                                         \
      }                                                                        \
      memcpy(temp, src_ptr + (n >> UVSHIFT) * SBPP, SS(r, UVSHIFT) * SBPP);    \
      ANY_SIMD(temp, temp + 128, MASK + 1);                                    \
      memcpy(dst_ptr + n * BPP, temp + 128, r * BPP);                          \
    }

#ifdef HAS_COPYROW_AVX
ANY11(CopyRow_Any_AVX, CopyRow_AVX, 0, 1, 1, 63)
#endif
#ifdef HAS_COPYROW_SSE2
ANY11(CopyRow_Any_SSE2, CopyRow_SSE2, 0, 1, 1, 31)
#endif
#ifdef HAS_COPYROW_NEON
ANY11(CopyRow_Any_NEON, CopyRow_NEON, 0, 1, 1, 31)
#endif
#if defined(HAS_ARGBTORGB24ROW_SSSE3)
ANY11(ARGBToRGB24Row_Any_SSSE3, ARGBToRGB24Row_SSSE3, 0, 4, 3, 15)
ANY11(ARGBToRAWRow_Any_SSSE3, ARGBToRAWRow_SSSE3, 0, 4, 3, 15)
ANY11(ARGBToRGB565Row_Any_SSE2, ARGBToRGB565Row_SSE2, 0, 4, 2, 3)
ANY11(ARGBToARGB1555Row_Any_SSE2, ARGBToARGB1555Row_SSE2, 0, 4, 2, 3)
ANY11(ARGBToARGB4444Row_Any_SSE2, ARGBToARGB4444Row_SSE2, 0, 4, 2, 3)
#endif
#if defined(HAS_ARGBTORGB565ROW_AVX2)
ANY11(ARGBToRGB565Row_Any_AVX2, ARGBToRGB565Row_AVX2, 0, 4, 2, 7)
#endif
#if defined(HAS_ARGBTOARGB4444ROW_AVX2)
ANY11(ARGBToARGB1555Row_Any_AVX2, ARGBToARGB1555Row_AVX2, 0, 4, 2, 7)
ANY11(ARGBToARGB4444Row_Any_AVX2, ARGBToARGB4444Row_AVX2, 0, 4, 2, 7)
#endif
#if defined(HAS_J400TOARGBROW_SSE2)
ANY11(J400ToARGBRow_Any_SSE2, J400ToARGBRow_SSE2, 0, 1, 4, 7)
#endif
#if defined(HAS_J400TOARGBROW_AVX2)
ANY11(J400ToARGBRow_Any_AVX2, J400ToARGBRow_AVX2, 0, 1, 4, 15)
#endif
#if defined(HAS_I400TOARGBROW_SSE2)
ANY11(I400ToARGBRow_Any_SSE2, I400ToARGBRow_SSE2, 0, 1, 4, 7)
#endif
#if defined(HAS_I400TOARGBROW_AVX2)
ANY11(I400ToARGBRow_Any_AVX2, I400ToARGBRow_AVX2, 0, 1, 4, 15)
#endif
#if defined(HAS_RGB24TOARGBROW_SSSE3)
ANY11(RGB24ToARGBRow_Any_SSSE3, RGB24ToARGBRow_SSSE3, 0, 3, 4, 15)
ANY11(RAWToARGBRow_Any_SSSE3, RAWToARGBRow_SSSE3, 0, 3, 4, 15)
ANY11(RGB565ToARGBRow_Any_SSE2, RGB565ToARGBRow_SSE2, 0, 2, 4, 7)
ANY11(ARGB1555ToARGBRow_Any_SSE2, ARGB1555ToARGBRow_SSE2, 0, 2, 4, 7)
ANY11(ARGB4444ToARGBRow_Any_SSE2, ARGB4444ToARGBRow_SSE2, 0, 2, 4, 7)
#endif
#if defined(HAS_RAWTORGB24ROW_SSSE3)
ANY11(RAWToRGB24Row_Any_SSSE3, RAWToRGB24Row_SSSE3, 0, 3, 3, 7)
#endif
#if defined(HAS_RGB565TOARGBROW_AVX2)
ANY11(RGB565ToARGBRow_Any_AVX2, RGB565ToARGBRow_AVX2, 0, 2, 4, 15)
#endif
#if defined(HAS_ARGB1555TOARGBROW_AVX2)
ANY11(ARGB1555ToARGBRow_Any_AVX2, ARGB1555ToARGBRow_AVX2, 0, 2, 4, 15)
#endif
#if defined(HAS_ARGB4444TOARGBROW_AVX2)
ANY11(ARGB4444ToARGBRow_Any_AVX2, ARGB4444ToARGBRow_AVX2, 0, 2, 4, 15)
#endif
#if defined(HAS_ARGBTORGB24ROW_NEON)
ANY11(ARGBToRGB24Row_Any_NEON, ARGBToRGB24Row_NEON, 0, 4, 3, 7)
ANY11(ARGBToRAWRow_Any_NEON, ARGBToRAWRow_NEON, 0, 4, 3, 7)
ANY11(ARGBToRGB565Row_Any_NEON, ARGBToRGB565Row_NEON, 0, 4, 2, 7)
ANY11(ARGBToARGB1555Row_Any_NEON, ARGBToARGB1555Row_NEON, 0, 4, 2, 7)
ANY11(ARGBToARGB4444Row_Any_NEON, ARGBToARGB4444Row_NEON, 0, 4, 2, 7)
ANY11(J400ToARGBRow_Any_NEON, J400ToARGBRow_NEON, 0, 1, 4, 7)
ANY11(I400ToARGBRow_Any_NEON, I400ToARGBRow_NEON, 0, 1, 4, 7)
#endif
#if defined(HAS_RAWTORGB24ROW_NEON)
ANY11(RAWToRGB24Row_Any_NEON, RAWToRGB24Row_NEON, 0, 3, 3, 7)
#endif
#ifdef HAS_ARGBTOYROW_AVX2
ANY11(ARGBToYRow_Any_AVX2, ARGBToYRow_AVX2, 0, 4, 1, 31)
#endif
#ifdef HAS_ARGBTOYJROW_AVX2
ANY11(ARGBToYJRow_Any_AVX2, ARGBToYJRow_AVX2, 0, 4, 1, 31)
#endif
#ifdef HAS_UYVYTOYROW_AVX2
ANY11(UYVYToYRow_Any_AVX2, UYVYToYRow_AVX2, 0, 2, 1, 31)
#endif
#ifdef HAS_YUY2TOYROW_AVX2
ANY11(YUY2ToYRow_Any_AVX2, YUY2ToYRow_AVX2, 1, 4, 1, 31)
#endif
#ifdef HAS_ARGBTOYROW_SSSE3
ANY11(ARGBToYRow_Any_SSSE3, ARGBToYRow_SSSE3, 0, 4, 1, 15)
#endif
#ifdef HAS_BGRATOYROW_SSSE3
ANY11(BGRAToYRow_Any_SSSE3, BGRAToYRow_SSSE3, 0, 4, 1, 15)
ANY11(ABGRToYRow_Any_SSSE3, ABGRToYRow_SSSE3, 0, 4, 1, 15)
ANY11(RGBAToYRow_Any_SSSE3, RGBAToYRow_SSSE3, 0, 4, 1, 15)
ANY11(YUY2ToYRow_Any_SSE2, YUY2ToYRow_SSE2, 1, 4, 1, 15)
ANY11(UYVYToYRow_Any_SSE2, UYVYToYRow_SSE2, 1, 4, 1, 15)
#endif
#ifdef HAS_ARGBTOYJROW_SSSE3
ANY11(ARGBToYJRow_Any_SSSE3, ARGBToYJRow_SSSE3, 0, 4, 1, 15)
#endif
#ifdef HAS_ARGBTOYROW_NEON
ANY11(ARGBToYRow_Any_NEON, ARGBToYRow_NEON, 0, 4, 1, 7)
#endif
#ifdef HAS_ARGBTOYJROW_NEON
ANY11(ARGBToYJRow_Any_NEON, ARGBToYJRow_NEON, 0, 4, 1, 7)
#endif
#ifdef HAS_BGRATOYROW_NEON
ANY11(BGRAToYRow_Any_NEON, BGRAToYRow_NEON, 0, 4, 1, 7)
#endif
#ifdef HAS_ABGRTOYROW_NEON
ANY11(ABGRToYRow_Any_NEON, ABGRToYRow_NEON, 0, 4, 1, 7)
#endif
#ifdef HAS_RGBATOYROW_NEON
ANY11(RGBAToYRow_Any_NEON, RGBAToYRow_NEON, 0, 4, 1, 7)
#endif
#ifdef HAS_RGB24TOYROW_NEON
ANY11(RGB24ToYRow_Any_NEON, RGB24ToYRow_NEON, 0, 3, 1, 7)
#endif
#ifdef HAS_RAWTOYROW_NEON
ANY11(RAWToYRow_Any_NEON, RAWToYRow_NEON, 0, 3, 1, 7)
#endif
#ifdef HAS_RGB565TOYROW_NEON
ANY11(RGB565ToYRow_Any_NEON, RGB565ToYRow_NEON, 0, 2, 1, 7)
#endif
#ifdef HAS_ARGB1555TOYROW_NEON
ANY11(ARGB1555ToYRow_Any_NEON, ARGB1555ToYRow_NEON, 0, 2, 1, 7)
#endif
#ifdef HAS_ARGB4444TOYROW_NEON
ANY11(ARGB4444ToYRow_Any_NEON, ARGB4444ToYRow_NEON, 0, 2, 1, 7)
#endif
#ifdef HAS_YUY2TOYROW_NEON
ANY11(YUY2ToYRow_Any_NEON, YUY2ToYRow_NEON, 1, 4, 1, 15)
#endif
#ifdef HAS_UYVYTOYROW_NEON
ANY11(UYVYToYRow_Any_NEON, UYVYToYRow_NEON, 0, 2, 1, 15)
#endif
#ifdef HAS_RGB24TOARGBROW_NEON
ANY11(RGB24ToARGBRow_Any_NEON, RGB24ToARGBRow_NEON, 0, 3, 4, 7)
#endif
#ifdef HAS_RAWTOARGBROW_NEON
ANY11(RAWToARGBRow_Any_NEON, RAWToARGBRow_NEON, 0, 3, 4, 7)
#endif
#ifdef HAS_RGB565TOARGBROW_NEON
ANY11(RGB565ToARGBRow_Any_NEON, RGB565ToARGBRow_NEON, 0, 2, 4, 7)
#endif
#ifdef HAS_ARGB1555TOARGBROW_NEON
ANY11(ARGB1555ToARGBRow_Any_NEON, ARGB1555ToARGBRow_NEON, 0, 2, 4, 7)
#endif
#ifdef HAS_ARGB4444TOARGBROW_NEON
ANY11(ARGB4444ToARGBRow_Any_NEON, ARGB4444ToARGBRow_NEON, 0, 2, 4, 7)
#endif
#ifdef HAS_ARGBATTENUATEROW_SSSE3
ANY11(ARGBAttenuateRow_Any_SSSE3, ARGBAttenuateRow_SSSE3, 0, 4, 4, 3)
#endif
#ifdef HAS_ARGBUNATTENUATEROW_SSE2
ANY11(ARGBUnattenuateRow_Any_SSE2, ARGBUnattenuateRow_SSE2, 0, 4, 4, 3)
#endif
#ifdef HAS_ARGBATTENUATEROW_AVX2
ANY11(ARGBAttenuateRow_Any_AVX2, ARGBAttenuateRow_AVX2, 0, 4, 4, 7)
#endif
#ifdef HAS_ARGBUNATTENUATEROW_AVX2
ANY11(ARGBUnattenuateRow_Any_AVX2, ARGBUnattenuateRow_AVX2, 0, 4, 4, 7)
#endif
#ifdef HAS_ARGBATTENUATEROW_NEON
ANY11(ARGBAttenuateRow_Any_NEON, ARGBAttenuateRow_NEON, 0, 4, 4, 7)
#endif
#ifdef HAS_ARGBEXTRACTALPHAROW_SSE2
ANY11(ARGBExtractAlphaRow_Any_SSE2, ARGBExtractAlphaRow_SSE2, 0, 4, 1, 7)
#endif
#ifdef HAS_ARGBEXTRACTALPHAROW_NEON
ANY11(ARGBExtractAlphaRow_Any_NEON, ARGBExtractAlphaRow_NEON, 0, 4, 1, 15)
#endif
#undef ANY11

// Any 1 to 1 blended.  Destination is read, modify, write.
#define ANY11B(NAMEANY, ANY_SIMD, UVSHIFT, SBPP, BPP, MASK)                    \
    void NAMEANY(const uint8* src_ptr, uint8* dst_ptr, int width) {            \
      SIMD_ALIGNED(uint8 temp[128 * 2]);                                       \
      memset(temp, 0, 128 * 2);  /* for YUY2 and msan */                       \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(src_ptr, dst_ptr, n);                                         \
      }                                                                        \
      memcpy(temp, src_ptr + (n >> UVSHIFT) * SBPP, SS(r, UVSHIFT) * SBPP);    \
      memcpy(temp + 128, dst_ptr + n * BPP, r * BPP);                          \
      ANY_SIMD(temp, temp + 128, MASK + 1);                                    \
      memcpy(dst_ptr + n * BPP, temp + 128, r * BPP);                          \
    }

#ifdef HAS_ARGBCOPYALPHAROW_AVX2
ANY11B(ARGBCopyAlphaRow_Any_AVX2, ARGBCopyAlphaRow_AVX2, 0, 4, 4, 15)
#endif
#ifdef HAS_ARGBCOPYALPHAROW_SSE2
ANY11B(ARGBCopyAlphaRow_Any_SSE2, ARGBCopyAlphaRow_SSE2, 0, 4, 4, 7)
#endif
#ifdef HAS_ARGBCOPYYTOALPHAROW_AVX2
ANY11B(ARGBCopyYToAlphaRow_Any_AVX2, ARGBCopyYToAlphaRow_AVX2, 0, 1, 4, 15)
#endif
#ifdef HAS_ARGBCOPYYTOALPHAROW_SSE2
ANY11B(ARGBCopyYToAlphaRow_Any_SSE2, ARGBCopyYToAlphaRow_SSE2, 0, 1, 4, 7)
#endif
#undef ANY11B

// Any 1 to 1 with parameter.
#define ANY11P(NAMEANY, ANY_SIMD, T, SBPP, BPP, MASK)                          \
    void NAMEANY(const uint8* src_ptr, uint8* dst_ptr,                         \
                 T shuffler, int width) {                                      \
      SIMD_ALIGNED(uint8 temp[64 * 2]);                                        \
      memset(temp, 0, 64);  /* for msan */                                     \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(src_ptr, dst_ptr, shuffler, n);                               \
      }                                                                        \
      memcpy(temp, src_ptr + n * SBPP, r * SBPP);                              \
      ANY_SIMD(temp, temp + 64, shuffler, MASK + 1);                           \
      memcpy(dst_ptr + n * BPP, temp + 64, r * BPP);                           \
    }

#if defined(HAS_ARGBTORGB565DITHERROW_SSE2)
ANY11P(ARGBToRGB565DitherRow_Any_SSE2, ARGBToRGB565DitherRow_SSE2,
       const uint32, 4, 2, 3)
#endif
#if defined(HAS_ARGBTORGB565DITHERROW_AVX2)
ANY11P(ARGBToRGB565DitherRow_Any_AVX2, ARGBToRGB565DitherRow_AVX2,
       const uint32, 4, 2, 7)
#endif
#if defined(HAS_ARGBTORGB565DITHERROW_NEON)
ANY11P(ARGBToRGB565DitherRow_Any_NEON, ARGBToRGB565DitherRow_NEON,
       const uint32, 4, 2, 7)
#endif
#ifdef HAS_ARGBSHUFFLEROW_SSE2
ANY11P(ARGBShuffleRow_Any_SSE2, ARGBShuffleRow_SSE2, const uint8*, 4, 4, 3)
#endif
#ifdef HAS_ARGBSHUFFLEROW_SSSE3
ANY11P(ARGBShuffleRow_Any_SSSE3, ARGBShuffleRow_SSSE3, const uint8*, 4, 4, 7)
#endif
#ifdef HAS_ARGBSHUFFLEROW_AVX2
ANY11P(ARGBShuffleRow_Any_AVX2, ARGBShuffleRow_AVX2, const uint8*, 4, 4, 15)
#endif
#ifdef HAS_ARGBSHUFFLEROW_NEON
ANY11P(ARGBShuffleRow_Any_NEON, ARGBShuffleRow_NEON, const uint8*, 4, 4, 3)
#endif
#undef ANY11P

// Any 1 to 1 with yuvconstants
#define ANY11C(NAMEANY, ANY_SIMD, UVSHIFT, SBPP, BPP, MASK)                    \
    void NAMEANY(const uint8* src_ptr, uint8* dst_ptr,                         \
                 const struct YuvConstants* yuvconstants, int width) {         \
      SIMD_ALIGNED(uint8 temp[128 * 2]);                                       \
      memset(temp, 0, 128);  /* for YUY2 and msan */                           \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(src_ptr, dst_ptr, yuvconstants, n);                           \
      }                                                                        \
      memcpy(temp, src_ptr + (n >> UVSHIFT) * SBPP, SS(r, UVSHIFT) * SBPP);    \
      ANY_SIMD(temp, temp + 128, yuvconstants, MASK + 1);                      \
      memcpy(dst_ptr + n * BPP, temp + 128, r * BPP);                          \
    }
#if defined(HAS_YUY2TOARGBROW_SSSE3)
ANY11C(YUY2ToARGBRow_Any_SSSE3, YUY2ToARGBRow_SSSE3, 1, 4, 4, 15)
ANY11C(UYVYToARGBRow_Any_SSSE3, UYVYToARGBRow_SSSE3, 1, 4, 4, 15)
#endif
#if defined(HAS_YUY2TOARGBROW_AVX2)
ANY11C(YUY2ToARGBRow_Any_AVX2, YUY2ToARGBRow_AVX2, 1, 4, 4, 31)
ANY11C(UYVYToARGBRow_Any_AVX2, UYVYToARGBRow_AVX2, 1, 4, 4, 31)
#endif
#if defined(HAS_YUY2TOARGBROW_NEON)
ANY11C(YUY2ToARGBRow_Any_NEON, YUY2ToARGBRow_NEON, 1, 4, 4, 7)
ANY11C(UYVYToARGBRow_Any_NEON, UYVYToARGBRow_NEON, 1, 4, 4, 7)
#endif
#undef ANY11C

// Any 1 to 1 interpolate.  Takes 2 rows of source via stride.
#define ANY11T(NAMEANY, ANY_SIMD, SBPP, BPP, MASK)                             \
    void NAMEANY(uint8* dst_ptr, const uint8* src_ptr,                         \
                 ptrdiff_t src_stride_ptr, int width,                          \
                 int source_y_fraction) {                                      \
      SIMD_ALIGNED(uint8 temp[64 * 3]);                                        \
      memset(temp, 0, 64 * 2);  /* for msan */                                 \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(dst_ptr, src_ptr, src_stride_ptr, n, source_y_fraction);      \
      }                                                                        \
      memcpy(temp, src_ptr + n * SBPP, r * SBPP);                              \
      memcpy(temp + 64, src_ptr + src_stride_ptr + n * SBPP, r * SBPP);        \
      ANY_SIMD(temp + 128, temp, 64, MASK + 1, source_y_fraction);             \
      memcpy(dst_ptr + n * BPP, temp + 128, r * BPP);                          \
    }

#ifdef HAS_INTERPOLATEROW_AVX2
ANY11T(InterpolateRow_Any_AVX2, InterpolateRow_AVX2, 1, 1, 31)
#endif
#ifdef HAS_INTERPOLATEROW_SSSE3
ANY11T(InterpolateRow_Any_SSSE3, InterpolateRow_SSSE3, 1, 1, 15)
#endif
#ifdef HAS_INTERPOLATEROW_NEON
ANY11T(InterpolateRow_Any_NEON, InterpolateRow_NEON, 1, 1, 15)
#endif
#ifdef HAS_INTERPOLATEROW_DSPR2
ANY11T(InterpolateRow_Any_DSPR2, InterpolateRow_DSPR2, 1, 1, 3)
#endif
#undef ANY11T

// Any 1 to 1 mirror.
#define ANY11M(NAMEANY, ANY_SIMD, BPP, MASK)                                   \
    void NAMEANY(const uint8* src_ptr, uint8* dst_ptr, int width) {            \
      SIMD_ALIGNED(uint8 temp[64 * 2]);                                        \
      memset(temp, 0, 64);  /* for msan */                                     \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(src_ptr + r * BPP, dst_ptr, n);                               \
      }                                                                        \
      memcpy(temp, src_ptr, r * BPP);                                          \
      ANY_SIMD(temp, temp + 64, MASK + 1);                                     \
      memcpy(dst_ptr + n * BPP, temp + 64 + (MASK + 1 - r) * BPP, r * BPP);    \
    }

#ifdef HAS_MIRRORROW_AVX2
ANY11M(MirrorRow_Any_AVX2, MirrorRow_AVX2, 1, 31)
#endif
#ifdef HAS_MIRRORROW_SSSE3
ANY11M(MirrorRow_Any_SSSE3, MirrorRow_SSSE3, 1, 15)
#endif
#ifdef HAS_MIRRORROW_NEON
ANY11M(MirrorRow_Any_NEON, MirrorRow_NEON, 1, 15)
#endif
#ifdef HAS_MIRRORROW_MSA
ANY11M(MirrorRow_Any_MSA, MirrorRow_MSA, 1, 63)
#endif
#ifdef HAS_ARGBMIRRORROW_AVX2
ANY11M(ARGBMirrorRow_Any_AVX2, ARGBMirrorRow_AVX2, 4, 7)
#endif
#ifdef HAS_ARGBMIRRORROW_SSE2
ANY11M(ARGBMirrorRow_Any_SSE2, ARGBMirrorRow_SSE2, 4, 3)
#endif
#ifdef HAS_ARGBMIRRORROW_NEON
ANY11M(ARGBMirrorRow_Any_NEON, ARGBMirrorRow_NEON, 4, 3)
#endif
#undef ANY11M

// Any 1 plane. (memset)
#define ANY1(NAMEANY, ANY_SIMD, T, BPP, MASK)                                  \
    void NAMEANY(uint8* dst_ptr, T v32, int width) {                           \
      SIMD_ALIGNED(uint8 temp[64]);                                            \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(dst_ptr, v32, n);                                             \
      }                                                                        \
      ANY_SIMD(temp, v32, MASK + 1);                                           \
      memcpy(dst_ptr + n * BPP, temp, r * BPP);                                \
    }

#ifdef HAS_SETROW_X86
ANY1(SetRow_Any_X86, SetRow_X86, uint8, 1, 3)
#endif
#ifdef HAS_SETROW_NEON
ANY1(SetRow_Any_NEON, SetRow_NEON, uint8, 1, 15)
#endif
#ifdef HAS_ARGBSETROW_NEON
ANY1(ARGBSetRow_Any_NEON, ARGBSetRow_NEON, uint32, 4, 3)
#endif
#undef ANY1

// Any 1 to 2.  Outputs UV planes.
#define ANY12(NAMEANY, ANY_SIMD, UVSHIFT, BPP, DUVSHIFT, MASK)                 \
    void NAMEANY(const uint8* src_ptr, uint8* dst_u, uint8* dst_v, int width) {\
      SIMD_ALIGNED(uint8 temp[128 * 3]);                                       \
      memset(temp, 0, 128);  /* for msan */                                    \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(src_ptr, dst_u, dst_v, n);                                    \
      }                                                                        \
      memcpy(temp, src_ptr  + (n >> UVSHIFT) * BPP, SS(r, UVSHIFT) * BPP);     \
      /* repeat last 4 bytes for 422 subsampler */                             \
      if ((width & 1) && BPP == 4 && DUVSHIFT == 1) {                          \
        memcpy(temp + SS(r, UVSHIFT) * BPP,                                    \
               temp + SS(r, UVSHIFT) * BPP - BPP, BPP);                        \
      }                                                                        \
      /* repeat last 4 - 12 bytes for 411 subsampler */                        \
      if (((width & 3) == 1) && BPP == 4 && DUVSHIFT == 2) {                   \
        memcpy(temp + SS(r, UVSHIFT) * BPP,                                    \
               temp + SS(r, UVSHIFT) * BPP - BPP, BPP);                        \
        memcpy(temp + SS(r, UVSHIFT) * BPP + BPP,                              \
               temp + SS(r, UVSHIFT) * BPP - BPP, BPP * 2);                    \
      }                                                                        \
      if (((width & 3) == 2) && BPP == 4 && DUVSHIFT == 2) {                   \
        memcpy(temp + SS(r, UVSHIFT) * BPP,                                    \
               temp + SS(r, UVSHIFT) * BPP - BPP * 2, BPP * 2);                \
      }                                                                        \
      if (((width & 3) == 3) && BPP == 4 && DUVSHIFT == 2) {                   \
        memcpy(temp + SS(r, UVSHIFT) * BPP,                                    \
               temp + SS(r, UVSHIFT) * BPP - BPP, BPP);                        \
      }                                                                        \
      ANY_SIMD(temp, temp + 128, temp + 256, MASK + 1);                        \
      memcpy(dst_u + (n >> DUVSHIFT), temp + 128, SS(r, DUVSHIFT));            \
      memcpy(dst_v + (n >> DUVSHIFT), temp + 256, SS(r, DUVSHIFT));            \
    }

#ifdef HAS_SPLITUVROW_SSE2
ANY12(SplitUVRow_Any_SSE2, SplitUVRow_SSE2, 0, 2, 0, 15)
#endif
#ifdef HAS_SPLITUVROW_AVX2
ANY12(SplitUVRow_Any_AVX2, SplitUVRow_AVX2, 0, 2, 0, 31)
#endif
#ifdef HAS_SPLITUVROW_NEON
ANY12(SplitUVRow_Any_NEON, SplitUVRow_NEON, 0, 2, 0, 15)
#endif
#ifdef HAS_SPLITUVROW_DSPR2
ANY12(SplitUVRow_Any_DSPR2, SplitUVRow_DSPR2, 0, 2, 0, 15)
#endif
#ifdef HAS_ARGBTOUV444ROW_SSSE3
ANY12(ARGBToUV444Row_Any_SSSE3, ARGBToUV444Row_SSSE3, 0, 4, 0, 15)
#endif
#ifdef HAS_YUY2TOUV422ROW_AVX2
ANY12(YUY2ToUV422Row_Any_AVX2, YUY2ToUV422Row_AVX2, 1, 4, 1, 31)
ANY12(UYVYToUV422Row_Any_AVX2, UYVYToUV422Row_AVX2, 1, 4, 1, 31)
#endif
#ifdef HAS_YUY2TOUV422ROW_SSE2
ANY12(YUY2ToUV422Row_Any_SSE2, YUY2ToUV422Row_SSE2, 1, 4, 1, 15)
ANY12(UYVYToUV422Row_Any_SSE2, UYVYToUV422Row_SSE2, 1, 4, 1, 15)
#endif
#ifdef HAS_YUY2TOUV422ROW_NEON
ANY12(ARGBToUV444Row_Any_NEON, ARGBToUV444Row_NEON, 0, 4, 0, 7)
ANY12(ARGBToUV411Row_Any_NEON, ARGBToUV411Row_NEON, 0, 4, 2, 31)
ANY12(YUY2ToUV422Row_Any_NEON, YUY2ToUV422Row_NEON, 1, 4, 1, 15)
ANY12(UYVYToUV422Row_Any_NEON, UYVYToUV422Row_NEON, 1, 4, 1, 15)
#endif
#undef ANY12

// Any 1 to 2 with source stride (2 rows of source).  Outputs UV planes.
// 128 byte row allows for 32 avx ARGB pixels.
#define ANY12S(NAMEANY, ANY_SIMD, UVSHIFT, BPP, MASK)                          \
    void NAMEANY(const uint8* src_ptr, int src_stride_ptr,                     \
                 uint8* dst_u, uint8* dst_v, int width) {                      \
      SIMD_ALIGNED(uint8 temp[128 * 4]);                                       \
      memset(temp, 0, 128 * 2);  /* for msan */                                \
      int r = width & MASK;                                                    \
      int n = width & ~MASK;                                                   \
      if (n > 0) {                                                             \
        ANY_SIMD(src_ptr, src_stride_ptr, dst_u, dst_v, n);                    \
      }                                                                        \
      memcpy(temp, src_ptr  + (n >> UVSHIFT) * BPP, SS(r, UVSHIFT) * BPP);     \
      memcpy(temp + 128, src_ptr  + src_stride_ptr + (n >> UVSHIFT) * BPP,     \
             SS(r, UVSHIFT) * BPP);                                            \
      if ((width & 1) && UVSHIFT == 0) {  /* repeat last pixel for subsample */\
        memcpy(temp + SS(r, UVSHIFT) * BPP,                                    \
               temp + SS(r, UVSHIFT) * BPP - BPP, BPP);                        \
        memcpy(temp + 128 + SS(r, UVSHIFT) * BPP,                              \
               temp + 128 + SS(r, UVSHIFT) * BPP - BPP, BPP);                  \
      }                                                                        \
      ANY_SIMD(temp, 128, temp + 256, temp + 384, MASK + 1);                   \
      memcpy(dst_u + (n >> 1), temp + 256, SS(r, 1));                          \
      memcpy(dst_v + (n >> 1), temp + 384, SS(r, 1));                          \
    }

#ifdef HAS_ARGBTOUVROW_AVX2
ANY12S(ARGBToUVRow_Any_AVX2, ARGBToUVRow_AVX2, 0, 4, 31)
#endif
#ifdef HAS_ARGBTOUVJROW_AVX2
ANY12S(ARGBToUVJRow_Any_AVX2, ARGBToUVJRow_AVX2, 0, 4, 31)
#endif
#ifdef HAS_ARGBTOUVROW_SSSE3
ANY12S(ARGBToUVRow_Any_SSSE3, ARGBToUVRow_SSSE3, 0, 4, 15)
ANY12S(ARGBToUVJRow_Any_SSSE3, ARGBToUVJRow_SSSE3, 0, 4, 15)
ANY12S(BGRAToUVRow_Any_SSSE3, BGRAToUVRow_SSSE3, 0, 4, 15)
ANY12S(ABGRToUVRow_Any_SSSE3, ABGRToUVRow_SSSE3, 0, 4, 15)
ANY12S(RGBAToUVRow_Any_SSSE3, RGBAToUVRow_SSSE3, 0, 4, 15)
#endif
#ifdef HAS_YUY2TOUVROW_AVX2
ANY12S(YUY2ToUVRow_Any_AVX2, YUY2ToUVRow_AVX2, 1, 4, 31)
ANY12S(UYVYToUVRow_Any_AVX2, UYVYToUVRow_AVX2, 1, 4, 31)
#endif
#ifdef HAS_YUY2TOUVROW_SSE2
ANY12S(YUY2ToUVRow_Any_SSE2, YUY2ToUVRow_SSE2, 1, 4, 15)
ANY12S(UYVYToUVRow_Any_SSE2, UYVYToUVRow_SSE2, 1, 4, 15)
#endif
#ifdef HAS_ARGBTOUVROW_NEON
ANY12S(ARGBToUVRow_Any_NEON, ARGBToUVRow_NEON, 0, 4, 15)
#endif
#ifdef HAS_ARGBTOUVJROW_NEON
ANY12S(ARGBToUVJRow_Any_NEON, ARGBToUVJRow_NEON, 0, 4, 15)
#endif
#ifdef HAS_BGRATOUVROW_NEON
ANY12S(BGRAToUVRow_Any_NEON, BGRAToUVRow_NEON, 0, 4, 15)
#endif
#ifdef HAS_ABGRTOUVROW_NEON
ANY12S(ABGRToUVRow_Any_NEON, ABGRToUVRow_NEON, 0, 4, 15)
#endif
#ifdef HAS_RGBATOUVROW_NEON
ANY12S(RGBAToUVRow_Any_NEON, RGBAToUVRow_NEON, 0, 4, 15)
#endif
#ifdef HAS_RGB24TOUVROW_NEON
ANY12S(RGB24ToUVRow_Any_NEON, RGB24ToUVRow_NEON, 0, 3, 15)
#endif
#ifdef HAS_RAWTOUVROW_NEON
ANY12S(RAWToUVRow_Any_NEON, RAWToUVRow_NEON, 0, 3, 15)
#endif
#ifdef HAS_RGB565TOUVROW_NEON
ANY12S(RGB565ToUVRow_Any_NEON, RGB565ToUVRow_NEON, 0, 2, 15)
#endif
#ifdef HAS_ARGB1555TOUVROW_NEON
ANY12S(ARGB1555ToUVRow_Any_NEON, ARGB1555ToUVRow_NEON, 0, 2, 15)
#endif
#ifdef HAS_ARGB4444TOUVROW_NEON
ANY12S(ARGB4444ToUVRow_Any_NEON, ARGB4444ToUVRow_NEON, 0, 2, 15)
#endif
#ifdef HAS_YUY2TOUVROW_NEON
ANY12S(YUY2ToUVRow_Any_NEON, YUY2ToUVRow_NEON, 1, 4, 15)
#endif
#ifdef HAS_UYVYTOUVROW_NEON
ANY12S(UYVYToUVRow_Any_NEON, UYVYToUVRow_NEON, 1, 4, 15)
#endif
#undef ANY12S

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
