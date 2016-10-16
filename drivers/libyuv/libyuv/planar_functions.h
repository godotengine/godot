/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_PLANAR_FUNCTIONS_H_  // NOLINT
#define INCLUDE_LIBYUV_PLANAR_FUNCTIONS_H_

#include "libyuv/basic_types.h"

// TODO(fbarchard): Remove the following headers includes.
#include "libyuv/convert.h"
#include "libyuv/convert_argb.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Copy a plane of data.
LIBYUV_API
void CopyPlane(const uint8* src_y, int src_stride_y,
               uint8* dst_y, int dst_stride_y,
               int width, int height);

LIBYUV_API
void CopyPlane_16(const uint16* src_y, int src_stride_y,
                  uint16* dst_y, int dst_stride_y,
                  int width, int height);

// Set a plane of data to a 32 bit value.
LIBYUV_API
void SetPlane(uint8* dst_y, int dst_stride_y,
              int width, int height,
              uint32 value);

// Split interleaved UV plane into separate U and V planes.
LIBYUV_API
void SplitUVPlane(const uint8* src_uv, int src_stride_uv,
                  uint8* dst_u, int dst_stride_u,
                  uint8* dst_v, int dst_stride_v,
                  int width, int height);

// Merge separate U and V planes into one interleaved UV plane.
LIBYUV_API
void MergeUVPlane(const uint8* src_u, int src_stride_u,
                  const uint8* src_v, int src_stride_v,
                  uint8* dst_uv, int dst_stride_uv,
                  int width, int height);

// Copy I400.  Supports inverting.
LIBYUV_API
int I400ToI400(const uint8* src_y, int src_stride_y,
               uint8* dst_y, int dst_stride_y,
               int width, int height);

#define J400ToJ400 I400ToI400

// Copy I422 to I422.
#define I422ToI422 I422Copy
LIBYUV_API
int I422Copy(const uint8* src_y, int src_stride_y,
             const uint8* src_u, int src_stride_u,
             const uint8* src_v, int src_stride_v,
             uint8* dst_y, int dst_stride_y,
             uint8* dst_u, int dst_stride_u,
             uint8* dst_v, int dst_stride_v,
             int width, int height);

// Copy I444 to I444.
#define I444ToI444 I444Copy
LIBYUV_API
int I444Copy(const uint8* src_y, int src_stride_y,
             const uint8* src_u, int src_stride_u,
             const uint8* src_v, int src_stride_v,
             uint8* dst_y, int dst_stride_y,
             uint8* dst_u, int dst_stride_u,
             uint8* dst_v, int dst_stride_v,
             int width, int height);

// Convert YUY2 to I422.
LIBYUV_API
int YUY2ToI422(const uint8* src_yuy2, int src_stride_yuy2,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_u, int dst_stride_u,
               uint8* dst_v, int dst_stride_v,
               int width, int height);

// Convert UYVY to I422.
LIBYUV_API
int UYVYToI422(const uint8* src_uyvy, int src_stride_uyvy,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_u, int dst_stride_u,
               uint8* dst_v, int dst_stride_v,
               int width, int height);

LIBYUV_API
int YUY2ToNV12(const uint8* src_yuy2, int src_stride_yuy2,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_uv, int dst_stride_uv,
               int width, int height);

LIBYUV_API
int UYVYToNV12(const uint8* src_uyvy, int src_stride_uyvy,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_uv, int dst_stride_uv,
               int width, int height);

// Convert I420 to I400. (calls CopyPlane ignoring u/v).
LIBYUV_API
int I420ToI400(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_y, int dst_stride_y,
               int width, int height);

// Alias
#define J420ToJ400 I420ToI400
#define I420ToI420Mirror I420Mirror

// I420 mirror.
LIBYUV_API
int I420Mirror(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_y, int dst_stride_y,
               uint8* dst_u, int dst_stride_u,
               uint8* dst_v, int dst_stride_v,
               int width, int height);

// Alias
#define I400ToI400Mirror I400Mirror

// I400 mirror.  A single plane is mirrored horizontally.
// Pass negative height to achieve 180 degree rotation.
LIBYUV_API
int I400Mirror(const uint8* src_y, int src_stride_y,
               uint8* dst_y, int dst_stride_y,
               int width, int height);

// Alias
#define ARGBToARGBMirror ARGBMirror

// ARGB mirror.
LIBYUV_API
int ARGBMirror(const uint8* src_argb, int src_stride_argb,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Convert NV12 to RGB565.
LIBYUV_API
int NV12ToRGB565(const uint8* src_y, int src_stride_y,
                 const uint8* src_uv, int src_stride_uv,
                 uint8* dst_rgb565, int dst_stride_rgb565,
                 int width, int height);

// I422ToARGB is in convert_argb.h
// Convert I422 to BGRA.
LIBYUV_API
int I422ToBGRA(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_bgra, int dst_stride_bgra,
               int width, int height);

// Convert I422 to ABGR.
LIBYUV_API
int I422ToABGR(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_abgr, int dst_stride_abgr,
               int width, int height);

// Convert I422 to RGBA.
LIBYUV_API
int I422ToRGBA(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_rgba, int dst_stride_rgba,
               int width, int height);

// Alias
#define RGB24ToRAW RAWToRGB24

LIBYUV_API
int RAWToRGB24(const uint8* src_raw, int src_stride_raw,
               uint8* dst_rgb24, int dst_stride_rgb24,
               int width, int height);

// Draw a rectangle into I420.
LIBYUV_API
int I420Rect(uint8* dst_y, int dst_stride_y,
             uint8* dst_u, int dst_stride_u,
             uint8* dst_v, int dst_stride_v,
             int x, int y, int width, int height,
             int value_y, int value_u, int value_v);

// Draw a rectangle into ARGB.
LIBYUV_API
int ARGBRect(uint8* dst_argb, int dst_stride_argb,
             int x, int y, int width, int height, uint32 value);

// Convert ARGB to gray scale ARGB.
LIBYUV_API
int ARGBGrayTo(const uint8* src_argb, int src_stride_argb,
               uint8* dst_argb, int dst_stride_argb,
               int width, int height);

// Make a rectangle of ARGB gray scale.
LIBYUV_API
int ARGBGray(uint8* dst_argb, int dst_stride_argb,
             int x, int y, int width, int height);

// Make a rectangle of ARGB Sepia tone.
LIBYUV_API
int ARGBSepia(uint8* dst_argb, int dst_stride_argb,
              int x, int y, int width, int height);

// Apply a matrix rotation to each ARGB pixel.
// matrix_argb is 4 signed ARGB values. -128 to 127 representing -2 to 2.
// The first 4 coefficients apply to B, G, R, A and produce B of the output.
// The next 4 coefficients apply to B, G, R, A and produce G of the output.
// The next 4 coefficients apply to B, G, R, A and produce R of the output.
// The last 4 coefficients apply to B, G, R, A and produce A of the output.
LIBYUV_API
int ARGBColorMatrix(const uint8* src_argb, int src_stride_argb,
                    uint8* dst_argb, int dst_stride_argb,
                    const int8* matrix_argb,
                    int width, int height);

// Deprecated. Use ARGBColorMatrix instead.
// Apply a matrix rotation to each ARGB pixel.
// matrix_argb is 3 signed ARGB values. -128 to 127 representing -1 to 1.
// The first 4 coefficients apply to B, G, R, A and produce B of the output.
// The next 4 coefficients apply to B, G, R, A and produce G of the output.
// The last 4 coefficients apply to B, G, R, A and produce R of the output.
LIBYUV_API
int RGBColorMatrix(uint8* dst_argb, int dst_stride_argb,
                   const int8* matrix_rgb,
                   int x, int y, int width, int height);

// Apply a color table each ARGB pixel.
// Table contains 256 ARGB values.
LIBYUV_API
int ARGBColorTable(uint8* dst_argb, int dst_stride_argb,
                   const uint8* table_argb,
                   int x, int y, int width, int height);

// Apply a color table each ARGB pixel but preserve destination alpha.
// Table contains 256 ARGB values.
LIBYUV_API
int RGBColorTable(uint8* dst_argb, int dst_stride_argb,
                  const uint8* table_argb,
                  int x, int y, int width, int height);

// Apply a luma/color table each ARGB pixel but preserve destination alpha.
// Table contains 32768 values indexed by [Y][C] where 7 it 7 bit luma from
// RGB (YJ style) and C is an 8 bit color component (R, G or B).
LIBYUV_API
int ARGBLumaColorTable(const uint8* src_argb, int src_stride_argb,
                       uint8* dst_argb, int dst_stride_argb,
                       const uint8* luma_rgb_table,
                       int width, int height);

// Apply a 3 term polynomial to ARGB values.
// poly points to a 4x4 matrix.  The first row is constants.  The 2nd row is
// coefficients for b, g, r and a.  The 3rd row is coefficients for b squared,
// g squared, r squared and a squared.  The 4rd row is coefficients for b to
// the 3, g to the 3, r to the 3 and a to the 3.  The values are summed and
// result clamped to 0 to 255.
// A polynomial approximation can be dirived using software such as 'R'.

LIBYUV_API
int ARGBPolynomial(const uint8* src_argb, int src_stride_argb,
                   uint8* dst_argb, int dst_stride_argb,
                   const float* poly,
                   int width, int height);

// Quantize a rectangle of ARGB. Alpha unaffected.
// scale is a 16 bit fractional fixed point scaler between 0 and 65535.
// interval_size should be a value between 1 and 255.
// interval_offset should be a value between 0 and 255.
LIBYUV_API
int ARGBQuantize(uint8* dst_argb, int dst_stride_argb,
                 int scale, int interval_size, int interval_offset,
                 int x, int y, int width, int height);

// Copy ARGB to ARGB.
LIBYUV_API
int ARGBCopy(const uint8* src_argb, int src_stride_argb,
             uint8* dst_argb, int dst_stride_argb,
             int width, int height);

// Copy Alpha channel of ARGB to alpha of ARGB.
LIBYUV_API
int ARGBCopyAlpha(const uint8* src_argb, int src_stride_argb,
                  uint8* dst_argb, int dst_stride_argb,
                  int width, int height);

// Extract the alpha channel from ARGB.
LIBYUV_API
int ARGBExtractAlpha(const uint8* src_argb, int src_stride_argb,
                     uint8* dst_a, int dst_stride_a,
                     int width, int height);

// Copy Y channel to Alpha of ARGB.
LIBYUV_API
int ARGBCopyYToAlpha(const uint8* src_y, int src_stride_y,
                     uint8* dst_argb, int dst_stride_argb,
                     int width, int height);

typedef void (*ARGBBlendRow)(const uint8* src_argb0, const uint8* src_argb1,
                             uint8* dst_argb, int width);

// Get function to Alpha Blend ARGB pixels and store to destination.
LIBYUV_API
ARGBBlendRow GetARGBBlend();

// Alpha Blend ARGB images and store to destination.
// Source is pre-multiplied by alpha using ARGBAttenuate.
// Alpha of destination is set to 255.
LIBYUV_API
int ARGBBlend(const uint8* src_argb0, int src_stride_argb0,
              const uint8* src_argb1, int src_stride_argb1,
              uint8* dst_argb, int dst_stride_argb,
              int width, int height);

// Alpha Blend plane and store to destination.
// Source is not pre-multiplied by alpha.
LIBYUV_API
int BlendPlane(const uint8* src_y0, int src_stride_y0,
               const uint8* src_y1, int src_stride_y1,
               const uint8* alpha, int alpha_stride,
               uint8* dst_y, int dst_stride_y,
               int width, int height);

// Alpha Blend YUV images and store to destination.
// Source is not pre-multiplied by alpha.
// Alpha is full width x height and subsampled to half size to apply to UV.
LIBYUV_API
int I420Blend(const uint8* src_y0, int src_stride_y0,
              const uint8* src_u0, int src_stride_u0,
              const uint8* src_v0, int src_stride_v0,
              const uint8* src_y1, int src_stride_y1,
              const uint8* src_u1, int src_stride_u1,
              const uint8* src_v1, int src_stride_v1,
              const uint8* alpha, int alpha_stride,
              uint8* dst_y, int dst_stride_y,
              uint8* dst_u, int dst_stride_u,
              uint8* dst_v, int dst_stride_v,
              int width, int height);

// Multiply ARGB image by ARGB image. Shifted down by 8. Saturates to 255.
LIBYUV_API
int ARGBMultiply(const uint8* src_argb0, int src_stride_argb0,
                 const uint8* src_argb1, int src_stride_argb1,
                 uint8* dst_argb, int dst_stride_argb,
                 int width, int height);

// Add ARGB image with ARGB image. Saturates to 255.
LIBYUV_API
int ARGBAdd(const uint8* src_argb0, int src_stride_argb0,
            const uint8* src_argb1, int src_stride_argb1,
            uint8* dst_argb, int dst_stride_argb,
            int width, int height);

// Subtract ARGB image (argb1) from ARGB image (argb0). Saturates to 0.
LIBYUV_API
int ARGBSubtract(const uint8* src_argb0, int src_stride_argb0,
                 const uint8* src_argb1, int src_stride_argb1,
                 uint8* dst_argb, int dst_stride_argb,
                 int width, int height);

// Convert I422 to YUY2.
LIBYUV_API
int I422ToYUY2(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_frame, int dst_stride_frame,
               int width, int height);

// Convert I422 to UYVY.
LIBYUV_API
int I422ToUYVY(const uint8* src_y, int src_stride_y,
               const uint8* src_u, int src_stride_u,
               const uint8* src_v, int src_stride_v,
               uint8* dst_frame, int dst_stride_frame,
               int width, int height);

// Convert unattentuated ARGB to preattenuated ARGB.
LIBYUV_API
int ARGBAttenuate(const uint8* src_argb, int src_stride_argb,
                  uint8* dst_argb, int dst_stride_argb,
                  int width, int height);

// Convert preattentuated ARGB to unattenuated ARGB.
LIBYUV_API
int ARGBUnattenuate(const uint8* src_argb, int src_stride_argb,
                    uint8* dst_argb, int dst_stride_argb,
                    int width, int height);

// Internal function - do not call directly.
// Computes table of cumulative sum for image where the value is the sum
// of all values above and to the left of the entry. Used by ARGBBlur.
LIBYUV_API
int ARGBComputeCumulativeSum(const uint8* src_argb, int src_stride_argb,
                             int32* dst_cumsum, int dst_stride32_cumsum,
                             int width, int height);

// Blur ARGB image.
// dst_cumsum table of width * (height + 1) * 16 bytes aligned to
//   16 byte boundary.
// dst_stride32_cumsum is number of ints in a row (width * 4).
// radius is number of pixels around the center.  e.g. 1 = 3x3. 2=5x5.
// Blur is optimized for radius of 5 (11x11) or less.
LIBYUV_API
int ARGBBlur(const uint8* src_argb, int src_stride_argb,
             uint8* dst_argb, int dst_stride_argb,
             int32* dst_cumsum, int dst_stride32_cumsum,
             int width, int height, int radius);

// Multiply ARGB image by ARGB value.
LIBYUV_API
int ARGBShade(const uint8* src_argb, int src_stride_argb,
              uint8* dst_argb, int dst_stride_argb,
              int width, int height, uint32 value);

// Interpolate between two images using specified amount of interpolation
// (0 to 255) and store to destination.
// 'interpolation' is specified as 8 bit fraction where 0 means 100% src0
// and 255 means 1% src0 and 99% src1.
LIBYUV_API
int InterpolatePlane(const uint8* src0, int src_stride0,
                     const uint8* src1, int src_stride1,
                     uint8* dst, int dst_stride,
                     int width, int height, int interpolation);

// Interpolate between two ARGB images using specified amount of interpolation
// Internally calls InterpolatePlane with width * 4 (bpp).
LIBYUV_API
int ARGBInterpolate(const uint8* src_argb0, int src_stride_argb0,
                    const uint8* src_argb1, int src_stride_argb1,
                    uint8* dst_argb, int dst_stride_argb,
                    int width, int height, int interpolation);

// Interpolate between two YUV images using specified amount of interpolation
// Internally calls InterpolatePlane on each plane where the U and V planes
// are half width and half height.
LIBYUV_API
int I420Interpolate(const uint8* src0_y, int src0_stride_y,
                    const uint8* src0_u, int src0_stride_u,
                    const uint8* src0_v, int src0_stride_v,
                    const uint8* src1_y, int src1_stride_y,
                    const uint8* src1_u, int src1_stride_u,
                    const uint8* src1_v, int src1_stride_v,
                    uint8* dst_y, int dst_stride_y,
                    uint8* dst_u, int dst_stride_u,
                    uint8* dst_v, int dst_stride_v,
                    int width, int height, int interpolation);

#if defined(__pnacl__) || defined(__CLR_VER) || \
    (defined(__i386__) && !defined(__SSE2__))
#define LIBYUV_DISABLE_X86
#endif
// MemorySanitizer does not support assembly code yet. http://crbug.com/344505
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define LIBYUV_DISABLE_X86
#endif
#endif
// The following are available on all x86 platforms:
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#define HAS_ARGBAFFINEROW_SSE2
#endif

// Row function for copying pixels from a source with a slope to a row
// of destination. Useful for scaling, rotation, mirror, texture mapping.
LIBYUV_API
void ARGBAffineRow_C(const uint8* src_argb, int src_argb_stride,
                     uint8* dst_argb, const float* uv_dudv, int width);
LIBYUV_API
void ARGBAffineRow_SSE2(const uint8* src_argb, int src_argb_stride,
                        uint8* dst_argb, const float* uv_dudv, int width);

// Shuffle ARGB channel order.  e.g. BGRA to ARGB.
// shuffler is 16 bytes and must be aligned.
LIBYUV_API
int ARGBShuffle(const uint8* src_bgra, int src_stride_bgra,
                uint8* dst_argb, int dst_stride_argb,
                const uint8* shuffler, int width, int height);

// Sobel ARGB effect with planar output.
LIBYUV_API
int ARGBSobelToPlane(const uint8* src_argb, int src_stride_argb,
                     uint8* dst_y, int dst_stride_y,
                     int width, int height);

// Sobel ARGB effect.
LIBYUV_API
int ARGBSobel(const uint8* src_argb, int src_stride_argb,
              uint8* dst_argb, int dst_stride_argb,
              int width, int height);

// Sobel ARGB effect w/ Sobel X, Sobel, Sobel Y in ARGB.
LIBYUV_API
int ARGBSobelXY(const uint8* src_argb, int src_stride_argb,
                uint8* dst_argb, int dst_stride_argb,
                int width, int height);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_PLANAR_FUNCTIONS_H_  NOLINT
