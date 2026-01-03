/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_CONVERT_ARGB_H_
#define INCLUDE_LIBYUV_CONVERT_ARGB_H_

#include "libyuv/basic_types.h"

#include "libyuv/rotate.h"  // For enum RotationMode.
#include "libyuv/scale.h"   // For enum FilterMode.

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Conversion matrix for YUV to RGB
LIBYUV_API extern const struct YuvConstants kYuvI601Constants;   // BT.601
LIBYUV_API extern const struct YuvConstants kYuvJPEGConstants;   // BT.601 full
LIBYUV_API extern const struct YuvConstants kYuvH709Constants;   // BT.709
LIBYUV_API extern const struct YuvConstants kYuvF709Constants;   // BT.709 full
LIBYUV_API extern const struct YuvConstants kYuv2020Constants;   // BT.2020
LIBYUV_API extern const struct YuvConstants kYuvV2020Constants;  // BT.2020 full

// Conversion matrix for YVU to BGR
LIBYUV_API extern const struct YuvConstants kYvuI601Constants;   // BT.601
LIBYUV_API extern const struct YuvConstants kYvuJPEGConstants;   // BT.601 full
LIBYUV_API extern const struct YuvConstants kYvuH709Constants;   // BT.709
LIBYUV_API extern const struct YuvConstants kYvuF709Constants;   // BT.709 full
LIBYUV_API extern const struct YuvConstants kYvu2020Constants;   // BT.2020
LIBYUV_API extern const struct YuvConstants kYvuV2020Constants;  // BT.2020 full

// Macros for end swapped destination Matrix conversions.
// Swap UV and pass mirrored kYvuJPEGConstants matrix.
// TODO(fbarchard): Add macro for each Matrix function.
#define kYuvI601ConstantsVU kYvuI601Constants
#define kYuvJPEGConstantsVU kYvuJPEGConstants
#define kYuvH709ConstantsVU kYvuH709Constants
#define kYuvF709ConstantsVU kYvuF709Constants
#define kYuv2020ConstantsVU kYvu2020Constants
#define kYuvV2020ConstantsVU kYvuV2020Constants

#define NV12ToABGRMatrix(a, b, c, d, e, f, g, h, i) \
  NV21ToARGBMatrix(a, b, c, d, e, f, g##VU, h, i)
#define NV21ToABGRMatrix(a, b, c, d, e, f, g, h, i) \
  NV12ToARGBMatrix(a, b, c, d, e, f, g##VU, h, i)
#define NV12ToRAWMatrix(a, b, c, d, e, f, g, h, i) \
  NV21ToRGB24Matrix(a, b, c, d, e, f, g##VU, h, i)
#define NV21ToRAWMatrix(a, b, c, d, e, f, g, h, i) \
  NV12ToRGB24Matrix(a, b, c, d, e, f, g##VU, h, i)
#define I010ToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k) \
  I010ToARGBMatrix(a, b, e, f, c, d, g, h, i##VU, j, k)
#define I210ToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k) \
  I210ToARGBMatrix(a, b, e, f, c, d, g, h, i##VU, j, k)
#define I410ToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k) \
  I410ToARGBMatrix(a, b, e, f, c, d, g, h, i##VU, j, k)
#define I010ToAB30Matrix(a, b, c, d, e, f, g, h, i, j, k) \
  I010ToAR30Matrix(a, b, e, f, c, d, g, h, i##VU, j, k)
#define I210ToAB30Matrix(a, b, c, d, e, f, g, h, i, j, k) \
  I210ToAR30Matrix(a, b, e, f, c, d, g, h, i##VU, j, k)
#define I410ToAB30Matrix(a, b, c, d, e, f, g, h, i, j, k) \
  I410ToAR30Matrix(a, b, e, f, c, d, g, h, i##VU, j, k)
#define I012ToAB30Matrix(a, b, c, d, e, f, g, h, i, j, k) \
  I012ToAR30Matrix(a, b, e, f, c, d, g, h, i##VU, j, k)
#define I420AlphaToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
  I420AlphaToARGBMatrix(a, b, e, f, c, d, g, h, i, j, k##VU, l, m, n)
#define I422AlphaToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
  I422AlphaToARGBMatrix(a, b, e, f, c, d, g, h, i, j, k##VU, l, m, n)
#define I444AlphaToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
  I444AlphaToARGBMatrix(a, b, e, f, c, d, g, h, i, j, k##VU, l, m, n)
#define I010AlphaToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
  I010AlphaToARGBMatrix(a, b, e, f, c, d, g, h, i, j, k##VU, l, m, n)
#define I210AlphaToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
  I210AlphaToARGBMatrix(a, b, e, f, c, d, g, h, i, j, k##VU, l, m, n)
#define I410AlphaToABGRMatrix(a, b, c, d, e, f, g, h, i, j, k, l, m, n) \
  I410AlphaToARGBMatrix(a, b, e, f, c, d, g, h, i, j, k##VU, l, m, n)

// Alias.
#define ARGBToARGB ARGBCopy

// Copy ARGB to ARGB.
LIBYUV_API
int ARGBCopy(const uint8_t* src_argb,
             int src_stride_argb,
             uint8_t* dst_argb,
             int dst_stride_argb,
             int width,
             int height);

// Convert I420 to ARGB.
LIBYUV_API
int I420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert I420 to ABGR.
LIBYUV_API
int I420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert J420 to ARGB.
LIBYUV_API
int J420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert J420 to ABGR.
LIBYUV_API
int J420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert H420 to ARGB.
LIBYUV_API
int H420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert H420 to ABGR.
LIBYUV_API
int H420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert U420 to ARGB.
LIBYUV_API
int U420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert U420 to ABGR.
LIBYUV_API
int U420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert I422 to ARGB.
LIBYUV_API
int I422ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert I422 to ABGR.
LIBYUV_API
int I422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert J422 to ARGB.
LIBYUV_API
int J422ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert J422 to ABGR.
LIBYUV_API
int J422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert H422 to ARGB.
LIBYUV_API
int H422ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert H422 to ABGR.
LIBYUV_API
int H422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert U422 to ARGB.
LIBYUV_API
int U422ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert U422 to ABGR.
LIBYUV_API
int U422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert I444 to ARGB.
LIBYUV_API
int I444ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert I444 to ABGR.
LIBYUV_API
int I444ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert J444 to ARGB.
LIBYUV_API
int J444ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert J444 to ABGR.
LIBYUV_API
int J444ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert H444 to ARGB.
LIBYUV_API
int H444ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert H444 to ABGR.
LIBYUV_API
int H444ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert U444 to ARGB.
LIBYUV_API
int U444ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert U444 to ABGR.
LIBYUV_API
int U444ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert I444 to RGB24.
LIBYUV_API
int I444ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

// Convert I444 to RAW.
LIBYUV_API
int I444ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

// Convert I010 to ARGB.
LIBYUV_API
int I010ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert I010 to ABGR.
LIBYUV_API
int I010ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert H010 to ARGB.
LIBYUV_API
int H010ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert H010 to ABGR.
LIBYUV_API
int H010ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert U010 to ARGB.
LIBYUV_API
int U010ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert U010 to ABGR.
LIBYUV_API
int U010ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert I210 to ARGB.
LIBYUV_API
int I210ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert I210 to ABGR.
LIBYUV_API
int I210ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert H210 to ARGB.
LIBYUV_API
int H210ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert H210 to ABGR.
LIBYUV_API
int H210ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert U210 to ARGB.
LIBYUV_API
int U210ToARGB(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert U210 to ABGR.
LIBYUV_API
int U210ToABGR(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert I420 with Alpha to preattenuated ARGB.
LIBYUV_API
int I420AlphaToARGB(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height,
                    int attenuate);

// Convert I420 with Alpha to preattenuated ABGR.
LIBYUV_API
int I420AlphaToABGR(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_abgr,
                    int dst_stride_abgr,
                    int width,
                    int height,
                    int attenuate);

// Convert I422 with Alpha to preattenuated ARGB.
LIBYUV_API
int I422AlphaToARGB(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height,
                    int attenuate);

// Convert I422 with Alpha to preattenuated ABGR.
LIBYUV_API
int I422AlphaToABGR(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_abgr,
                    int dst_stride_abgr,
                    int width,
                    int height,
                    int attenuate);

// Convert I444 with Alpha to preattenuated ARGB.
LIBYUV_API
int I444AlphaToARGB(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height,
                    int attenuate);

// Convert I444 with Alpha to preattenuated ABGR.
LIBYUV_API
int I444AlphaToABGR(const uint8_t* src_y,
                    int src_stride_y,
                    const uint8_t* src_u,
                    int src_stride_u,
                    const uint8_t* src_v,
                    int src_stride_v,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_abgr,
                    int dst_stride_abgr,
                    int width,
                    int height,
                    int attenuate);

// Convert I400 (grey) to ARGB.  Reverse of ARGBToI400.
LIBYUV_API
int I400ToARGB(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert J400 (jpeg grey) to ARGB.
LIBYUV_API
int J400ToARGB(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Alias.
#define YToARGB I400ToARGB

// Convert NV12 to ARGB.
LIBYUV_API
int NV12ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert NV21 to ARGB.
LIBYUV_API
int NV21ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert NV12 to ABGR.
LIBYUV_API
int NV12ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert NV21 to ABGR.
LIBYUV_API
int NV21ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert NV12 to RGB24.
LIBYUV_API
int NV12ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_uv,
                int src_stride_uv,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

// Convert NV21 to RGB24.
LIBYUV_API
int NV21ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_vu,
                int src_stride_vu,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

// Convert NV21 to YUV24.
LIBYUV_API
int NV21ToYUV24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_vu,
                int src_stride_vu,
                uint8_t* dst_yuv24,
                int dst_stride_yuv24,
                int width,
                int height);

// Convert NV12 to RAW.
LIBYUV_API
int NV12ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_uv,
              int src_stride_uv,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

// Convert NV21 to RAW.
LIBYUV_API
int NV21ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_vu,
              int src_stride_vu,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

// Convert YUY2 to ARGB.
LIBYUV_API
int YUY2ToARGB(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert UYVY to ARGB.
LIBYUV_API
int UYVYToARGB(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert I010 to AR30.
LIBYUV_API
int I010ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert H010 to AR30.
LIBYUV_API
int H010ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert I010 to AB30.
LIBYUV_API
int I010ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert H010 to AB30.
LIBYUV_API
int H010ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert U010 to AR30.
LIBYUV_API
int U010ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert U010 to AB30.
LIBYUV_API
int U010ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert I210 to AR30.
LIBYUV_API
int I210ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert I210 to AB30.
LIBYUV_API
int I210ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert H210 to AR30.
LIBYUV_API
int H210ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert H210 to AB30.
LIBYUV_API
int H210ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert U210 to AR30.
LIBYUV_API
int U210ToAR30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert U210 to AB30.
LIBYUV_API
int U210ToAB30(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// BGRA little endian (argb in memory) to ARGB.
LIBYUV_API
int BGRAToARGB(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// ABGR little endian (rgba in memory) to ARGB.
LIBYUV_API
int ABGRToARGB(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// RGBA little endian (abgr in memory) to ARGB.
LIBYUV_API
int RGBAToARGB(const uint8_t* src_rgba,
               int src_stride_rgba,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Deprecated function name.
#define BG24ToARGB RGB24ToARGB

// RGB little endian (bgr in memory) to ARGB.
LIBYUV_API
int RGB24ToARGB(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_argb,
                int dst_stride_argb,
                int width,
                int height);

// RGB big endian (rgb in memory) to ARGB.
LIBYUV_API
int RAWToARGB(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height);

// RGB big endian (rgb in memory) to RGBA.
LIBYUV_API
int RAWToRGBA(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_rgba,
              int dst_stride_rgba,
              int width,
              int height);

// RGB16 (RGBP fourcc) little endian to ARGB.
LIBYUV_API
int RGB565ToARGB(const uint8_t* src_rgb565,
                 int src_stride_rgb565,
                 uint8_t* dst_argb,
                 int dst_stride_argb,
                 int width,
                 int height);

// RGB15 (RGBO fourcc) little endian to ARGB.
LIBYUV_API
int ARGB1555ToARGB(const uint8_t* src_argb1555,
                   int src_stride_argb1555,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   int width,
                   int height);

// RGB12 (R444 fourcc) little endian to ARGB.
LIBYUV_API
int ARGB4444ToARGB(const uint8_t* src_argb4444,
                   int src_stride_argb4444,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   int width,
                   int height);

// Aliases
#define AB30ToARGB AR30ToABGR
#define AB30ToABGR AR30ToARGB
#define AB30ToAR30 AR30ToAB30

// Convert AR30 To ARGB.
LIBYUV_API
int AR30ToARGB(const uint8_t* src_ar30,
               int src_stride_ar30,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert AR30 To ABGR.
LIBYUV_API
int AR30ToABGR(const uint8_t* src_ar30,
               int src_stride_ar30,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert AR30 To AB30.
LIBYUV_API
int AR30ToAB30(const uint8_t* src_ar30,
               int src_stride_ar30,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert AR64 to ARGB.
LIBYUV_API
int AR64ToARGB(const uint16_t* src_ar64,
               int src_stride_ar64,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert AB64 to ABGR.
#define AB64ToABGR AR64ToARGB

// Convert AB64 to ARGB.
LIBYUV_API
int AB64ToARGB(const uint16_t* src_ab64,
               int src_stride_ab64,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Convert AR64 to ABGR.
#define AR64ToABGR AB64ToARGB

// Convert AR64 To AB64.
LIBYUV_API
int AR64ToAB64(const uint16_t* src_ar64,
               int src_stride_ar64,
               uint16_t* dst_ab64,
               int dst_stride_ab64,
               int width,
               int height);

// Convert AB64 To AR64.
#define AB64ToAR64 AR64ToAB64

// src_width/height provided by capture
// dst_width/height for clipping determine final size.
LIBYUV_API
int MJPGToARGB(const uint8_t* sample,
               size_t sample_size,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height);

// Convert Android420 to ARGB.
LIBYUV_API
int Android420ToARGB(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     int src_pixel_stride_uv,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     int width,
                     int height);

// Convert Android420 to ABGR.
LIBYUV_API
int Android420ToABGR(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     int src_pixel_stride_uv,
                     uint8_t* dst_abgr,
                     int dst_stride_abgr,
                     int width,
                     int height);

// Convert NV12 to RGB565.
LIBYUV_API
int NV12ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_uv,
                 int src_stride_uv,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height);

// Convert I422 to BGRA.
LIBYUV_API
int I422ToBGRA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_bgra,
               int dst_stride_bgra,
               int width,
               int height);

// Convert I422 to ABGR.
LIBYUV_API
int I422ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

// Convert I422 to RGBA.
LIBYUV_API
int I422ToRGBA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_rgba,
               int dst_stride_rgba,
               int width,
               int height);

LIBYUV_API
int I420ToARGB(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

LIBYUV_API
int I420ToBGRA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_bgra,
               int dst_stride_bgra,
               int width,
               int height);

LIBYUV_API
int I420ToABGR(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_abgr,
               int dst_stride_abgr,
               int width,
               int height);

LIBYUV_API
int I420ToRGBA(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_rgba,
               int dst_stride_rgba,
               int width,
               int height);

LIBYUV_API
int I420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

LIBYUV_API
int I420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

LIBYUV_API
int H420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

LIBYUV_API
int H420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

LIBYUV_API
int J420ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

LIBYUV_API
int J420ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

// Convert I422 to RGB24.
LIBYUV_API
int I422ToRGB24(const uint8_t* src_y,
                int src_stride_y,
                const uint8_t* src_u,
                int src_stride_u,
                const uint8_t* src_v,
                int src_stride_v,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

// Convert I422 to RAW.
LIBYUV_API
int I422ToRAW(const uint8_t* src_y,
              int src_stride_y,
              const uint8_t* src_u,
              int src_stride_u,
              const uint8_t* src_v,
              int src_stride_v,
              uint8_t* dst_raw,
              int dst_stride_raw,
              int width,
              int height);

LIBYUV_API
int I420ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height);

LIBYUV_API
int J420ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height);

LIBYUV_API
int H420ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height);

LIBYUV_API
int I422ToRGB565(const uint8_t* src_y,
                 int src_stride_y,
                 const uint8_t* src_u,
                 int src_stride_u,
                 const uint8_t* src_v,
                 int src_stride_v,
                 uint8_t* dst_rgb565,
                 int dst_stride_rgb565,
                 int width,
                 int height);

// Convert I420 To RGB565 with 4x4 dither matrix (16 bytes).
// Values in dither matrix from 0 to 7 recommended.
// The order of the dither matrix is first byte is upper left.

LIBYUV_API
int I420ToRGB565Dither(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const uint8_t* dither4x4,
                       int width,
                       int height);

LIBYUV_API
int I420ToARGB1555(const uint8_t* src_y,
                   int src_stride_y,
                   const uint8_t* src_u,
                   int src_stride_u,
                   const uint8_t* src_v,
                   int src_stride_v,
                   uint8_t* dst_argb1555,
                   int dst_stride_argb1555,
                   int width,
                   int height);

LIBYUV_API
int I420ToARGB4444(const uint8_t* src_y,
                   int src_stride_y,
                   const uint8_t* src_u,
                   int src_stride_u,
                   const uint8_t* src_v,
                   int src_stride_v,
                   uint8_t* dst_argb4444,
                   int dst_stride_argb4444,
                   int width,
                   int height);

// Convert I420 to AR30.
LIBYUV_API
int I420ToAR30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert I420 to AB30.
LIBYUV_API
int I420ToAB30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert H420 to AR30.
LIBYUV_API
int H420ToAR30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ar30,
               int dst_stride_ar30,
               int width,
               int height);

// Convert H420 to AB30.
LIBYUV_API
int H420ToAB30(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_ab30,
               int dst_stride_ab30,
               int width,
               int height);

// Convert I420 to ARGB with matrix.
LIBYUV_API
int I420ToARGBMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert I422 to ARGB with matrix.
LIBYUV_API
int I422ToARGBMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert I444 to ARGB with matrix.
LIBYUV_API
int I444ToARGBMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert I444 to RGB24 with matrix.
LIBYUV_API
int I444ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height);

// Convert 10 bit 420 YUV to ARGB with matrix.
LIBYUV_API
int I010ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert 10 bit 420 YUV to ARGB with matrix.
LIBYUV_API
int I210ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert 10 bit 444 YUV to ARGB with matrix.
LIBYUV_API
int I410ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert 10 bit YUV to ARGB with matrix.
LIBYUV_API
int I010ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// multiply 12 bit yuv into high bits to allow any number of bits.
LIBYUV_API
int I012ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert 12 bit YUV to ARGB with matrix.
LIBYUV_API
int I012ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert 10 bit 422 YUV to ARGB with matrix.
LIBYUV_API
int I210ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert 10 bit 444 YUV to ARGB with matrix.
LIBYUV_API
int I410ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert P010 to ARGB with matrix.
LIBYUV_API
int P010ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert P210 to ARGB with matrix.
LIBYUV_API
int P210ToARGBMatrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert P010 to AR30 with matrix.
LIBYUV_API
int P010ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert P210 to AR30 with matrix.
LIBYUV_API
int P210ToAR30Matrix(const uint16_t* src_y,
                     int src_stride_y,
                     const uint16_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// P012 and P010 use most significant bits so the conversion is the same.
// Convert P012 to ARGB with matrix.
#define P012ToARGBMatrix P010ToARGBMatrix
// Convert P012 to AR30 with matrix.
#define P012ToAR30Matrix P010ToAR30Matrix
// Convert P212 to ARGB with matrix.
#define P212ToARGBMatrix P210ToARGBMatrix
// Convert P212 to AR30 with matrix.
#define P212ToAR30Matrix P210ToAR30Matrix

// Convert P016 to ARGB with matrix.
#define P016ToARGBMatrix P010ToARGBMatrix
// Convert P016 to AR30 with matrix.
#define P016ToAR30Matrix P010ToAR30Matrix
// Convert P216 to ARGB with matrix.
#define P216ToARGBMatrix P210ToARGBMatrix
// Convert P216 to AR30 with matrix.
#define P216ToAR30Matrix P210ToAR30Matrix

// Convert I420 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I420AlphaToARGBMatrix(const uint8_t* src_y,
                          int src_stride_y,
                          const uint8_t* src_u,
                          int src_stride_u,
                          const uint8_t* src_v,
                          int src_stride_v,
                          const uint8_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate);

// Convert I422 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I422AlphaToARGBMatrix(const uint8_t* src_y,
                          int src_stride_y,
                          const uint8_t* src_u,
                          int src_stride_u,
                          const uint8_t* src_v,
                          int src_stride_v,
                          const uint8_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate);

// Convert I444 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I444AlphaToARGBMatrix(const uint8_t* src_y,
                          int src_stride_y,
                          const uint8_t* src_u,
                          int src_stride_u,
                          const uint8_t* src_v,
                          int src_stride_v,
                          const uint8_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate);

// Convert I010 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I010AlphaToARGBMatrix(const uint16_t* src_y,
                          int src_stride_y,
                          const uint16_t* src_u,
                          int src_stride_u,
                          const uint16_t* src_v,
                          int src_stride_v,
                          const uint16_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate);

// Convert I210 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I210AlphaToARGBMatrix(const uint16_t* src_y,
                          int src_stride_y,
                          const uint16_t* src_u,
                          int src_stride_u,
                          const uint16_t* src_v,
                          int src_stride_v,
                          const uint16_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate);

// Convert I410 with Alpha to preattenuated ARGB with matrix.
LIBYUV_API
int I410AlphaToARGBMatrix(const uint16_t* src_y,
                          int src_stride_y,
                          const uint16_t* src_u,
                          int src_stride_u,
                          const uint16_t* src_v,
                          int src_stride_v,
                          const uint16_t* src_a,
                          int src_stride_a,
                          uint8_t* dst_argb,
                          int dst_stride_argb,
                          const struct YuvConstants* yuvconstants,
                          int width,
                          int height,
                          int attenuate);

// Convert NV12 to ARGB with matrix.
LIBYUV_API
int NV12ToARGBMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_uv,
                     int src_stride_uv,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert NV21 to ARGB with matrix.
LIBYUV_API
int NV21ToARGBMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_vu,
                     int src_stride_vu,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert NV12 to RGB565 with matrix.
LIBYUV_API
int NV12ToRGB565Matrix(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_uv,
                       int src_stride_uv,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width,
                       int height);

// Convert NV12 to RGB24 with matrix.
LIBYUV_API
int NV12ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_uv,
                      int src_stride_uv,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height);

// Convert NV21 to RGB24 with matrix.
LIBYUV_API
int NV21ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_vu,
                      int src_stride_vu,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height);

// Convert YUY2 to ARGB with matrix.
LIBYUV_API
int YUY2ToARGBMatrix(const uint8_t* src_yuy2,
                     int src_stride_yuy2,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert UYVY to ARGB with matrix.
LIBYUV_API
int UYVYToARGBMatrix(const uint8_t* src_uyvy,
                     int src_stride_uyvy,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert Android420 to ARGB with matrix.
LIBYUV_API
int Android420ToARGBMatrix(const uint8_t* src_y,
                           int src_stride_y,
                           const uint8_t* src_u,
                           int src_stride_u,
                           const uint8_t* src_v,
                           int src_stride_v,
                           int src_pixel_stride_uv,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height);

// Convert I422 to RGBA with matrix.
LIBYUV_API
int I422ToRGBAMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_rgba,
                     int dst_stride_rgba,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert I420 to RGBA with matrix.
LIBYUV_API
int I420ToRGBAMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_rgba,
                     int dst_stride_rgba,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert I420 to RGB24 with matrix.
LIBYUV_API
int I420ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height);

// Convert I422 to RGB24 with matrix.
LIBYUV_API
int I422ToRGB24Matrix(const uint8_t* src_y,
                      int src_stride_y,
                      const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_rgb24,
                      int dst_stride_rgb24,
                      const struct YuvConstants* yuvconstants,
                      int width,
                      int height);

// Convert I420 to RGB565 with specified color matrix.
LIBYUV_API
int I420ToRGB565Matrix(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width,
                       int height);

// Convert I422 to RGB565 with specified color matrix.
LIBYUV_API
int I422ToRGB565Matrix(const uint8_t* src_y,
                       int src_stride_y,
                       const uint8_t* src_u,
                       int src_stride_u,
                       const uint8_t* src_v,
                       int src_stride_v,
                       uint8_t* dst_rgb565,
                       int dst_stride_rgb565,
                       const struct YuvConstants* yuvconstants,
                       int width,
                       int height);

// Convert I420 to AR30 with matrix.
LIBYUV_API
int I420ToAR30Matrix(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     uint8_t* dst_ar30,
                     int dst_stride_ar30,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert I400 (grey) to ARGB.  Reverse of ARGBToI400.
LIBYUV_API
int I400ToARGBMatrix(const uint8_t* src_y,
                     int src_stride_y,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     const struct YuvConstants* yuvconstants,
                     int width,
                     int height);

// Convert I420 to ARGB with matrix and UV filter mode.
LIBYUV_API
int I420ToARGBMatrixFilter(const uint8_t* src_y,
                           int src_stride_y,
                           const uint8_t* src_u,
                           int src_stride_u,
                           const uint8_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert I422 to ARGB with matrix and UV filter mode.
LIBYUV_API
int I422ToARGBMatrixFilter(const uint8_t* src_y,
                           int src_stride_y,
                           const uint8_t* src_u,
                           int src_stride_u,
                           const uint8_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert I422 to RGB24 with matrix and UV filter mode.
LIBYUV_API
int I422ToRGB24MatrixFilter(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_rgb24,
                            int dst_stride_rgb24,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height,
                            enum FilterMode filter);

// Convert I420 to RGB24 with matrix and UV filter mode.
LIBYUV_API
int I420ToRGB24MatrixFilter(const uint8_t* src_y,
                            int src_stride_y,
                            const uint8_t* src_u,
                            int src_stride_u,
                            const uint8_t* src_v,
                            int src_stride_v,
                            uint8_t* dst_rgb24,
                            int dst_stride_rgb24,
                            const struct YuvConstants* yuvconstants,
                            int width,
                            int height,
                            enum FilterMode filter);

// Convert I010 to AR30 with matrix and UV filter mode.
LIBYUV_API
int I010ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert I210 to AR30 with matrix and UV filter mode.
LIBYUV_API
int I210ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert I010 to ARGB with matrix and UV filter mode.
LIBYUV_API
int I010ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert I210 to ARGB with matrix and UV filter mode.
LIBYUV_API
int I210ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_u,
                           int src_stride_u,
                           const uint16_t* src_v,
                           int src_stride_v,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert I420 with Alpha to attenuated ARGB with matrix and UV filter mode.
LIBYUV_API
int I420AlphaToARGBMatrixFilter(const uint8_t* src_y,
                                int src_stride_y,
                                const uint8_t* src_u,
                                int src_stride_u,
                                const uint8_t* src_v,
                                int src_stride_v,
                                const uint8_t* src_a,
                                int src_stride_a,
                                uint8_t* dst_argb,
                                int dst_stride_argb,
                                const struct YuvConstants* yuvconstants,
                                int width,
                                int height,
                                int attenuate,
                                enum FilterMode filter);

// Convert I422 with Alpha to attenuated ARGB with matrix and UV filter mode.
LIBYUV_API
int I422AlphaToARGBMatrixFilter(const uint8_t* src_y,
                                int src_stride_y,
                                const uint8_t* src_u,
                                int src_stride_u,
                                const uint8_t* src_v,
                                int src_stride_v,
                                const uint8_t* src_a,
                                int src_stride_a,
                                uint8_t* dst_argb,
                                int dst_stride_argb,
                                const struct YuvConstants* yuvconstants,
                                int width,
                                int height,
                                int attenuate,
                                enum FilterMode filter);

// Convert I010 with Alpha to attenuated ARGB with matrix and UV filter mode.
LIBYUV_API
int I010AlphaToARGBMatrixFilter(const uint16_t* src_y,
                                int src_stride_y,
                                const uint16_t* src_u,
                                int src_stride_u,
                                const uint16_t* src_v,
                                int src_stride_v,
                                const uint16_t* src_a,
                                int src_stride_a,
                                uint8_t* dst_argb,
                                int dst_stride_argb,
                                const struct YuvConstants* yuvconstants,
                                int width,
                                int height,
                                int attenuate,
                                enum FilterMode filter);

// Convert I210 with Alpha to attenuated ARGB with matrix and UV filter mode.
LIBYUV_API
int I210AlphaToARGBMatrixFilter(const uint16_t* src_y,
                                int src_stride_y,
                                const uint16_t* src_u,
                                int src_stride_u,
                                const uint16_t* src_v,
                                int src_stride_v,
                                const uint16_t* src_a,
                                int src_stride_a,
                                uint8_t* dst_argb,
                                int dst_stride_argb,
                                const struct YuvConstants* yuvconstants,
                                int width,
                                int height,
                                int attenuate,
                                enum FilterMode filter);

// Convert P010 to ARGB with matrix and UV filter mode.
LIBYUV_API
int P010ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert P210 to ARGB with matrix and UV filter mode.
LIBYUV_API
int P210ToARGBMatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_argb,
                           int dst_stride_argb,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert P010 to AR30 with matrix and UV filter mode.
LIBYUV_API
int P010ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert P210 to AR30 with matrix and UV filter mode.
LIBYUV_API
int P210ToAR30MatrixFilter(const uint16_t* src_y,
                           int src_stride_y,
                           const uint16_t* src_uv,
                           int src_stride_uv,
                           uint8_t* dst_ar30,
                           int dst_stride_ar30,
                           const struct YuvConstants* yuvconstants,
                           int width,
                           int height,
                           enum FilterMode filter);

// Convert camera sample to ARGB with cropping, rotation and vertical flip.
// "sample_size" is needed to parse MJPG.
// "dst_stride_argb" number of bytes in a row of the dst_argb plane.
//   Normally this would be the same as dst_width, with recommended alignment
//   to 16 bytes for better efficiency.
//   If rotation of 90 or 270 is used, stride is affected. The caller should
//   allocate the I420 buffer according to rotation.
// "dst_stride_u" number of bytes in a row of the dst_u plane.
//   Normally this would be the same as (dst_width + 1) / 2, with
//   recommended alignment to 16 bytes for better efficiency.
//   If rotation of 90 or 270 is used, stride is affected.
// "crop_x" and "crop_y" are starting position for cropping.
//   To center, crop_x = (src_width - dst_width) / 2
//              crop_y = (src_height - dst_height) / 2
// "src_width" / "src_height" is size of src_frame in pixels.
//   "src_height" can be negative indicating a vertically flipped image source.
// "crop_width" / "crop_height" is the size to crop the src to.
//    Must be less than or equal to src_width/src_height
//    Cropping parameters are pre-rotation.
// "rotation" can be 0, 90, 180 or 270.
// "fourcc" is a fourcc. ie 'I420', 'YUY2'
// Returns 0 for successful; -1 for invalid parameter. Non-zero for failure.
LIBYUV_API
int ConvertToARGB(const uint8_t* sample,
                  size_t sample_size,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
                  int crop_x,
                  int crop_y,
                  int src_width,
                  int src_height,
                  int crop_width,
                  int crop_height,
                  enum RotationMode rotation,
                  uint32_t fourcc);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_CONVERT_ARGB_H_
