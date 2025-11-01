/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_PLANAR_FUNCTIONS_H_
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
void CopyPlane(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height);

LIBYUV_API
void CopyPlane_16(const uint16_t* src_y,
                  int src_stride_y,
                  uint16_t* dst_y,
                  int dst_stride_y,
                  int width,
                  int height);

LIBYUV_API
void Convert16To8Plane(const uint16_t* src_y,
                       int src_stride_y,
                       uint8_t* dst_y,
                       int dst_stride_y,
                       int scale,  // 16384 for 10 bits
                       int width,
                       int height);

LIBYUV_API
void Convert8To16Plane(const uint8_t* src_y,
                       int src_stride_y,
                       uint16_t* dst_y,
                       int dst_stride_y,
                       int scale,  // 1024 for 10 bits
                       int width,
                       int height);

LIBYUV_API
void Convert8To8Plane(const uint8_t* src_y,
                      int src_stride_y,
                      uint8_t* dst_y,
                      int dst_stride_y,
                      int scale,  // 220 for Y, 225 for U,V
                      int bias,   // 16
                      int width,
                      int height);

// Set a plane of data to a 32 bit value.
LIBYUV_API
void SetPlane(uint8_t* dst_y,
              int dst_stride_y,
              int width,
              int height,
              uint32_t value);

// Convert a plane of tiles of 16 x H to linear.
LIBYUV_API
int DetilePlane(const uint8_t* src_y,
                int src_stride_y,
                uint8_t* dst_y,
                int dst_stride_y,
                int width,
                int height,
                int tile_height);

// Convert a plane of 16 bit tiles of 16 x H to linear.
LIBYUV_API
int DetilePlane_16(const uint16_t* src_y,
                   int src_stride_y,
                   uint16_t* dst_y,
                   int dst_stride_y,
                   int width,
                   int height,
                   int tile_height);

// Convert a UV plane of tiles of 16 x H into linear U and V planes.
LIBYUV_API
void DetileSplitUVPlane(const uint8_t* src_uv,
                        int src_stride_uv,
                        uint8_t* dst_u,
                        int dst_stride_u,
                        uint8_t* dst_v,
                        int dst_stride_v,
                        int width,
                        int height,
                        int tile_height);

// Convert a Y and UV plane of tiles into interlaced YUY2.
LIBYUV_API
void DetileToYUY2(const uint8_t* src_y,
                  int src_stride_y,
                  const uint8_t* src_uv,
                  int src_stride_uv,
                  uint8_t* dst_yuy2,
                  int dst_stride_yuy2,
                  int width,
                  int height,
                  int tile_height);

// Split interleaved UV plane into separate U and V planes.
LIBYUV_API
void SplitUVPlane(const uint8_t* src_uv,
                  int src_stride_uv,
                  uint8_t* dst_u,
                  int dst_stride_u,
                  uint8_t* dst_v,
                  int dst_stride_v,
                  int width,
                  int height);

// Merge separate U and V planes into one interleaved UV plane.
LIBYUV_API
void MergeUVPlane(const uint8_t* src_u,
                  int src_stride_u,
                  const uint8_t* src_v,
                  int src_stride_v,
                  uint8_t* dst_uv,
                  int dst_stride_uv,
                  int width,
                  int height);

// Split interleaved msb UV plane into separate lsb U and V planes.
LIBYUV_API
void SplitUVPlane_16(const uint16_t* src_uv,
                     int src_stride_uv,
                     uint16_t* dst_u,
                     int dst_stride_u,
                     uint16_t* dst_v,
                     int dst_stride_v,
                     int width,
                     int height,
                     int depth);

// Merge separate lsb U and V planes into one interleaved msb UV plane.
LIBYUV_API
void MergeUVPlane_16(const uint16_t* src_u,
                     int src_stride_u,
                     const uint16_t* src_v,
                     int src_stride_v,
                     uint16_t* dst_uv,
                     int dst_stride_uv,
                     int width,
                     int height,
                     int depth);

// Convert lsb plane to msb plane
LIBYUV_API
void ConvertToMSBPlane_16(const uint16_t* src_y,
                          int src_stride_y,
                          uint16_t* dst_y,
                          int dst_stride_y,
                          int width,
                          int height,
                          int depth);

// Convert msb plane to lsb plane
LIBYUV_API
void ConvertToLSBPlane_16(const uint16_t* src_y,
                          int src_stride_y,
                          uint16_t* dst_y,
                          int dst_stride_y,
                          int width,
                          int height,
                          int depth);

// Scale U and V to half width and height and merge into interleaved UV plane.
// width and height are source size, allowing odd sizes.
// Use for converting I444 or I422 to NV12.
LIBYUV_API
void HalfMergeUVPlane(const uint8_t* src_u,
                      int src_stride_u,
                      const uint8_t* src_v,
                      int src_stride_v,
                      uint8_t* dst_uv,
                      int dst_stride_uv,
                      int width,
                      int height);

// Swap U and V channels in interleaved UV plane.
LIBYUV_API
void SwapUVPlane(const uint8_t* src_uv,
                 int src_stride_uv,
                 uint8_t* dst_vu,
                 int dst_stride_vu,
                 int width,
                 int height);

// Split interleaved RGB plane into separate R, G and B planes.
LIBYUV_API
void SplitRGBPlane(const uint8_t* src_rgb,
                   int src_stride_rgb,
                   uint8_t* dst_r,
                   int dst_stride_r,
                   uint8_t* dst_g,
                   int dst_stride_g,
                   uint8_t* dst_b,
                   int dst_stride_b,
                   int width,
                   int height);

// Merge separate R, G and B planes into one interleaved RGB plane.
LIBYUV_API
void MergeRGBPlane(const uint8_t* src_r,
                   int src_stride_r,
                   const uint8_t* src_g,
                   int src_stride_g,
                   const uint8_t* src_b,
                   int src_stride_b,
                   uint8_t* dst_rgb,
                   int dst_stride_rgb,
                   int width,
                   int height);

// Split interleaved ARGB plane into separate R, G, B and A planes.
// dst_a can be NULL to discard alpha plane.
LIBYUV_API
void SplitARGBPlane(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_r,
                    int dst_stride_r,
                    uint8_t* dst_g,
                    int dst_stride_g,
                    uint8_t* dst_b,
                    int dst_stride_b,
                    uint8_t* dst_a,
                    int dst_stride_a,
                    int width,
                    int height);

// Merge separate R, G, B and A planes into one interleaved ARGB plane.
// src_a can be NULL to fill opaque value to alpha.
LIBYUV_API
void MergeARGBPlane(const uint8_t* src_r,
                    int src_stride_r,
                    const uint8_t* src_g,
                    int src_stride_g,
                    const uint8_t* src_b,
                    int src_stride_b,
                    const uint8_t* src_a,
                    int src_stride_a,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height);

// Merge separate 'depth' bit R, G and B planes stored in lsb
// into one interleaved XR30 plane.
// depth should in range [10, 16]
LIBYUV_API
void MergeXR30Plane(const uint16_t* src_r,
                    int src_stride_r,
                    const uint16_t* src_g,
                    int src_stride_g,
                    const uint16_t* src_b,
                    int src_stride_b,
                    uint8_t* dst_ar30,
                    int dst_stride_ar30,
                    int width,
                    int height,
                    int depth);

// Merge separate 'depth' bit R, G, B and A planes stored in lsb
// into one interleaved AR64 plane.
// src_a can be NULL to fill opaque value to alpha.
// depth should in range [1, 16]
LIBYUV_API
void MergeAR64Plane(const uint16_t* src_r,
                    int src_stride_r,
                    const uint16_t* src_g,
                    int src_stride_g,
                    const uint16_t* src_b,
                    int src_stride_b,
                    const uint16_t* src_a,
                    int src_stride_a,
                    uint16_t* dst_ar64,
                    int dst_stride_ar64,
                    int width,
                    int height,
                    int depth);

// Merge separate 'depth' bit R, G, B and A planes stored in lsb
// into one interleaved ARGB plane.
// src_a can be NULL to fill opaque value to alpha.
// depth should in range [8, 16]
LIBYUV_API
void MergeARGB16To8Plane(const uint16_t* src_r,
                         int src_stride_r,
                         const uint16_t* src_g,
                         int src_stride_g,
                         const uint16_t* src_b,
                         int src_stride_b,
                         const uint16_t* src_a,
                         int src_stride_a,
                         uint8_t* dst_argb,
                         int dst_stride_argb,
                         int width,
                         int height,
                         int depth);

// Copy I400.  Supports inverting.
LIBYUV_API
int I400ToI400(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height);

#define J400ToJ400 I400ToI400

// Copy I422 to I422.
#define I422ToI422 I422Copy
LIBYUV_API
int I422Copy(const uint8_t* src_y,
             int src_stride_y,
             const uint8_t* src_u,
             int src_stride_u,
             const uint8_t* src_v,
             int src_stride_v,
             uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_u,
             int dst_stride_u,
             uint8_t* dst_v,
             int dst_stride_v,
             int width,
             int height);

// Copy I444 to I444.
#define I444ToI444 I444Copy
LIBYUV_API
int I444Copy(const uint8_t* src_y,
             int src_stride_y,
             const uint8_t* src_u,
             int src_stride_u,
             const uint8_t* src_v,
             int src_stride_v,
             uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_u,
             int dst_stride_u,
             uint8_t* dst_v,
             int dst_stride_v,
             int width,
             int height);

// Copy I210 to I210.
#define I210ToI210 I210Copy
LIBYUV_API
int I210Copy(const uint16_t* src_y,
             int src_stride_y,
             const uint16_t* src_u,
             int src_stride_u,
             const uint16_t* src_v,
             int src_stride_v,
             uint16_t* dst_y,
             int dst_stride_y,
             uint16_t* dst_u,
             int dst_stride_u,
             uint16_t* dst_v,
             int dst_stride_v,
             int width,
             int height);

// Copy I410 to I410.
#define I410ToI410 I410Copy
LIBYUV_API
int I410Copy(const uint16_t* src_y,
             int src_stride_y,
             const uint16_t* src_u,
             int src_stride_u,
             const uint16_t* src_v,
             int src_stride_v,
             uint16_t* dst_y,
             int dst_stride_y,
             uint16_t* dst_u,
             int dst_stride_u,
             uint16_t* dst_v,
             int dst_stride_v,
             int width,
             int height);

// Copy NV12. Supports inverting.
LIBYUV_API
int NV12Copy(const uint8_t* src_y,
             int src_stride_y,
             const uint8_t* src_uv,
             int src_stride_uv,
             uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_uv,
             int dst_stride_uv,
             int width,
             int height);

// Copy NV21. Supports inverting.
LIBYUV_API
int NV21Copy(const uint8_t* src_y,
             int src_stride_y,
             const uint8_t* src_vu,
             int src_stride_vu,
             uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_vu,
             int dst_stride_vu,
             int width,
             int height);

// Convert YUY2 to I422.
LIBYUV_API
int YUY2ToI422(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert UYVY to I422.
LIBYUV_API
int UYVYToI422(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

LIBYUV_API
int YUY2ToNV12(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

LIBYUV_API
int UYVYToNV12(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert NV21 to NV12.
LIBYUV_API
int NV21ToNV12(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

LIBYUV_API
int YUY2ToY(const uint8_t* src_yuy2,
            int src_stride_yuy2,
            uint8_t* dst_y,
            int dst_stride_y,
            int width,
            int height);

LIBYUV_API
int UYVYToY(const uint8_t* src_uyvy,
            int src_stride_uyvy,
            uint8_t* dst_y,
            int dst_stride_y,
            int width,
            int height);

// Convert I420 to I400. (calls CopyPlane ignoring u/v).
LIBYUV_API
int I420ToI400(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height);

// Alias
#define J420ToJ400 I420ToI400
#define I420ToI420Mirror I420Mirror

// I420 mirror.
LIBYUV_API
int I420Mirror(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Alias
#define I400ToI400Mirror I400Mirror

// I400 mirror.  A single plane is mirrored horizontally.
// Pass negative height to achieve 180 degree rotation.
LIBYUV_API
int I400Mirror(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height);

// Alias
#define NV12ToNV12Mirror NV12Mirror

// NV12 mirror.
LIBYUV_API
int NV12Mirror(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Alias
#define ARGBToARGBMirror ARGBMirror

// ARGB mirror.
LIBYUV_API
int ARGBMirror(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Alias
#define RGB24ToRGB24Mirror RGB24Mirror

// RGB24 mirror.
LIBYUV_API
int RGB24Mirror(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_rgb24,
                int dst_stride_rgb24,
                int width,
                int height);

// Mirror a plane of data.
LIBYUV_API
void MirrorPlane(const uint8_t* src_y,
                 int src_stride_y,
                 uint8_t* dst_y,
                 int dst_stride_y,
                 int width,
                 int height);

// Mirror a plane of UV data.
LIBYUV_API
void MirrorUVPlane(const uint8_t* src_uv,
                   int src_stride_uv,
                   uint8_t* dst_uv,
                   int dst_stride_uv,
                   int width,
                   int height);

// Alias
#define RGB24ToRAW RAWToRGB24

LIBYUV_API
int RAWToRGB24(const uint8_t* src_raw,
               int src_stride_raw,
               uint8_t* dst_rgb24,
               int dst_stride_rgb24,
               int width,
               int height);

// Draw a rectangle into I420.
LIBYUV_API
int I420Rect(uint8_t* dst_y,
             int dst_stride_y,
             uint8_t* dst_u,
             int dst_stride_u,
             uint8_t* dst_v,
             int dst_stride_v,
             int x,
             int y,
             int width,
             int height,
             int value_y,
             int value_u,
             int value_v);

// Draw a rectangle into ARGB.
LIBYUV_API
int ARGBRect(uint8_t* dst_argb,
             int dst_stride_argb,
             int dst_x,
             int dst_y,
             int width,
             int height,
             uint32_t value);

// Convert ARGB to gray scale ARGB.
LIBYUV_API
int ARGBGrayTo(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_argb,
               int dst_stride_argb,
               int width,
               int height);

// Make a rectangle of ARGB gray scale.
LIBYUV_API
int ARGBGray(uint8_t* dst_argb,
             int dst_stride_argb,
             int dst_x,
             int dst_y,
             int width,
             int height);

// Make a rectangle of ARGB Sepia tone.
LIBYUV_API
int ARGBSepia(uint8_t* dst_argb,
              int dst_stride_argb,
              int dst_x,
              int dst_y,
              int width,
              int height);

// Apply a matrix rotation to each ARGB pixel.
// matrix_argb is 4 signed ARGB values. -128 to 127 representing -2 to 2.
// The first 4 coefficients apply to B, G, R, A and produce B of the output.
// The next 4 coefficients apply to B, G, R, A and produce G of the output.
// The next 4 coefficients apply to B, G, R, A and produce R of the output.
// The last 4 coefficients apply to B, G, R, A and produce A of the output.
LIBYUV_API
int ARGBColorMatrix(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    const int8_t* matrix_argb,
                    int width,
                    int height);

// Deprecated. Use ARGBColorMatrix instead.
// Apply a matrix rotation to each ARGB pixel.
// matrix_argb is 3 signed ARGB values. -128 to 127 representing -1 to 1.
// The first 4 coefficients apply to B, G, R, A and produce B of the output.
// The next 4 coefficients apply to B, G, R, A and produce G of the output.
// The last 4 coefficients apply to B, G, R, A and produce R of the output.
LIBYUV_API
int RGBColorMatrix(uint8_t* dst_argb,
                   int dst_stride_argb,
                   const int8_t* matrix_rgb,
                   int dst_x,
                   int dst_y,
                   int width,
                   int height);

// Apply a color table each ARGB pixel.
// Table contains 256 ARGB values.
LIBYUV_API
int ARGBColorTable(uint8_t* dst_argb,
                   int dst_stride_argb,
                   const uint8_t* table_argb,
                   int dst_x,
                   int dst_y,
                   int width,
                   int height);

// Apply a color table each ARGB pixel but preserve destination alpha.
// Table contains 256 ARGB values.
LIBYUV_API
int RGBColorTable(uint8_t* dst_argb,
                  int dst_stride_argb,
                  const uint8_t* table_argb,
                  int dst_x,
                  int dst_y,
                  int width,
                  int height);

// Apply a luma/color table each ARGB pixel but preserve destination alpha.
// Table contains 32768 values indexed by [Y][C] where 7 it 7 bit luma from
// RGB (YJ style) and C is an 8 bit color component (R, G or B).
LIBYUV_API
int ARGBLumaColorTable(const uint8_t* src_argb,
                       int src_stride_argb,
                       uint8_t* dst_argb,
                       int dst_stride_argb,
                       const uint8_t* luma,
                       int width,
                       int height);

// Apply a 3 term polynomial to ARGB values.
// poly points to a 4x4 matrix.  The first row is constants.  The 2nd row is
// coefficients for b, g, r and a.  The 3rd row is coefficients for b squared,
// g squared, r squared and a squared.  The 4rd row is coefficients for b to
// the 3, g to the 3, r to the 3 and a to the 3.  The values are summed and
// result clamped to 0 to 255.
// A polynomial approximation can be dirived using software such as 'R'.

LIBYUV_API
int ARGBPolynomial(const uint8_t* src_argb,
                   int src_stride_argb,
                   uint8_t* dst_argb,
                   int dst_stride_argb,
                   const float* poly,
                   int width,
                   int height);

// Convert plane of 16 bit shorts to half floats.
// Source values are multiplied by scale before storing as half float.
//
// Note: Unlike other libyuv functions that operate on uint16_t buffers, the
// src_stride_y and dst_stride_y parameters of HalfFloatPlane() are in bytes,
// not in units of uint16_t.
LIBYUV_API
int HalfFloatPlane(const uint16_t* src_y,
                   int src_stride_y,
                   uint16_t* dst_y,
                   int dst_stride_y,
                   float scale,
                   int width,
                   int height);

// Convert a buffer of bytes to floats, scale the values and store as floats.
LIBYUV_API
int ByteToFloat(const uint8_t* src_y, float* dst_y, float scale, int width);

// Quantize a rectangle of ARGB. Alpha unaffected.
// scale is a 16 bit fractional fixed point scaler between 0 and 65535.
// interval_size should be a value between 1 and 255.
// interval_offset should be a value between 0 and 255.
LIBYUV_API
int ARGBQuantize(uint8_t* dst_argb,
                 int dst_stride_argb,
                 int scale,
                 int interval_size,
                 int interval_offset,
                 int dst_x,
                 int dst_y,
                 int width,
                 int height);

// Copy ARGB to ARGB.
LIBYUV_API
int ARGBCopy(const uint8_t* src_argb,
             int src_stride_argb,
             uint8_t* dst_argb,
             int dst_stride_argb,
             int width,
             int height);

// Copy Alpha channel of ARGB to alpha of ARGB.
LIBYUV_API
int ARGBCopyAlpha(const uint8_t* src_argb,
                  int src_stride_argb,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
                  int width,
                  int height);

// Extract the alpha channel from ARGB.
LIBYUV_API
int ARGBExtractAlpha(const uint8_t* src_argb,
                     int src_stride_argb,
                     uint8_t* dst_a,
                     int dst_stride_a,
                     int width,
                     int height);

// Copy Y channel to Alpha of ARGB.
LIBYUV_API
int ARGBCopyYToAlpha(const uint8_t* src_y,
                     int src_stride_y,
                     uint8_t* dst_argb,
                     int dst_stride_argb,
                     int width,
                     int height);

// Alpha Blend ARGB images and store to destination.
// Source is pre-multiplied by alpha using ARGBAttenuate.
// Alpha of destination is set to 255.
LIBYUV_API
int ARGBBlend(const uint8_t* src_argb0,
              int src_stride_argb0,
              const uint8_t* src_argb1,
              int src_stride_argb1,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height);

// Alpha Blend plane and store to destination.
// Source is not pre-multiplied by alpha.
LIBYUV_API
int BlendPlane(const uint8_t* src_y0,
               int src_stride_y0,
               const uint8_t* src_y1,
               int src_stride_y1,
               const uint8_t* alpha,
               int alpha_stride,
               uint8_t* dst_y,
               int dst_stride_y,
               int width,
               int height);

// Alpha Blend YUV images and store to destination.
// Source is not pre-multiplied by alpha.
// Alpha is full width x height and subsampled to half size to apply to UV.
LIBYUV_API
int I420Blend(const uint8_t* src_y0,
              int src_stride_y0,
              const uint8_t* src_u0,
              int src_stride_u0,
              const uint8_t* src_v0,
              int src_stride_v0,
              const uint8_t* src_y1,
              int src_stride_y1,
              const uint8_t* src_u1,
              int src_stride_u1,
              const uint8_t* src_v1,
              int src_stride_v1,
              const uint8_t* alpha,
              int alpha_stride,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height);

// Multiply ARGB image by ARGB image. Shifted down by 8. Saturates to 255.
LIBYUV_API
int ARGBMultiply(const uint8_t* src_argb0,
                 int src_stride_argb0,
                 const uint8_t* src_argb1,
                 int src_stride_argb1,
                 uint8_t* dst_argb,
                 int dst_stride_argb,
                 int width,
                 int height);

// Add ARGB image with ARGB image. Saturates to 255.
LIBYUV_API
int ARGBAdd(const uint8_t* src_argb0,
            int src_stride_argb0,
            const uint8_t* src_argb1,
            int src_stride_argb1,
            uint8_t* dst_argb,
            int dst_stride_argb,
            int width,
            int height);

// Subtract ARGB image (argb1) from ARGB image (argb0). Saturates to 0.
LIBYUV_API
int ARGBSubtract(const uint8_t* src_argb0,
                 int src_stride_argb0,
                 const uint8_t* src_argb1,
                 int src_stride_argb1,
                 uint8_t* dst_argb,
                 int dst_stride_argb,
                 int width,
                 int height);

// Convert I422 to YUY2.
LIBYUV_API
int I422ToYUY2(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height);

// Convert I422 to UYVY.
LIBYUV_API
int I422ToUYVY(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_uyvy,
               int dst_stride_uyvy,
               int width,
               int height);

// Convert unattentuated ARGB to preattenuated ARGB.
LIBYUV_API
int ARGBAttenuate(const uint8_t* src_argb,
                  int src_stride_argb,
                  uint8_t* dst_argb,
                  int dst_stride_argb,
                  int width,
                  int height);

// Convert preattentuated ARGB to unattenuated ARGB.
LIBYUV_API
int ARGBUnattenuate(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height);

// Internal function - do not call directly.
// Computes table of cumulative sum for image where the value is the sum
// of all values above and to the left of the entry. Used by ARGBBlur.
LIBYUV_API
int ARGBComputeCumulativeSum(const uint8_t* src_argb,
                             int src_stride_argb,
                             int32_t* dst_cumsum,
                             int dst_stride32_cumsum,
                             int width,
                             int height);

// Blur ARGB image.
// dst_cumsum table of width * (height + 1) * 16 bytes aligned to
//   16 byte boundary.
// dst_stride32_cumsum is number of ints in a row (width * 4).
// radius is number of pixels around the center.  e.g. 1 = 3x3. 2=5x5.
// Blur is optimized for radius of 5 (11x11) or less.
LIBYUV_API
int ARGBBlur(const uint8_t* src_argb,
             int src_stride_argb,
             uint8_t* dst_argb,
             int dst_stride_argb,
             int32_t* dst_cumsum,
             int dst_stride32_cumsum,
             int width,
             int height,
             int radius);

// Gaussian 5x5 blur a float plane.
// Coefficients of 1, 4, 6, 4, 1.
// Each destination pixel is a blur of the 5x5
// pixels from the source.
// Source edges are clamped.
LIBYUV_API
int GaussPlane_F32(const float* src,
                   int src_stride,
                   float* dst,
                   int dst_stride,
                   int width,
                   int height);

// Multiply ARGB image by ARGB value.
LIBYUV_API
int ARGBShade(const uint8_t* src_argb,
              int src_stride_argb,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height,
              uint32_t value);

// Interpolate between two images using specified amount of interpolation
// (0 to 255) and store to destination.
// 'interpolation' is specified as 8 bit fraction where 0 means 100% src0
// and 255 means 1% src0 and 99% src1.
LIBYUV_API
int InterpolatePlane(const uint8_t* src0,
                     int src_stride0,
                     const uint8_t* src1,
                     int src_stride1,
                     uint8_t* dst,
                     int dst_stride,
                     int width,
                     int height,
                     int interpolation);

// Interpolate between two images using specified amount of interpolation
// (0 to 255) and store to destination.
// 'interpolation' is specified as 8 bit fraction where 0 means 100% src0
// and 255 means 1% src0 and 99% src1.
LIBYUV_API
int InterpolatePlane_16(const uint16_t* src0,
                        int src_stride0,  // measured in 16 bit pixels
                        const uint16_t* src1,
                        int src_stride1,
                        uint16_t* dst,
                        int dst_stride,
                        int width,
                        int height,
                        int interpolation);

// Interpolate between two ARGB images using specified amount of interpolation
// Internally calls InterpolatePlane with width * 4 (bpp).
LIBYUV_API
int ARGBInterpolate(const uint8_t* src_argb0,
                    int src_stride_argb0,
                    const uint8_t* src_argb1,
                    int src_stride_argb1,
                    uint8_t* dst_argb,
                    int dst_stride_argb,
                    int width,
                    int height,
                    int interpolation);

// Interpolate between two YUV images using specified amount of interpolation
// Internally calls InterpolatePlane on each plane where the U and V planes
// are half width and half height.
LIBYUV_API
int I420Interpolate(const uint8_t* src0_y,
                    int src0_stride_y,
                    const uint8_t* src0_u,
                    int src0_stride_u,
                    const uint8_t* src0_v,
                    int src0_stride_v,
                    const uint8_t* src1_y,
                    int src1_stride_y,
                    const uint8_t* src1_u,
                    int src1_stride_u,
                    const uint8_t* src1_v,
                    int src1_stride_v,
                    uint8_t* dst_y,
                    int dst_stride_y,
                    uint8_t* dst_u,
                    int dst_stride_u,
                    uint8_t* dst_v,
                    int dst_stride_v,
                    int width,
                    int height,
                    int interpolation);

// Shuffle ARGB channel order.  e.g. BGRA to ARGB.
// shuffler is 16 bytes.
LIBYUV_API
int ARGBShuffle(const uint8_t* src_bgra,
                int src_stride_bgra,
                uint8_t* dst_argb,
                int dst_stride_argb,
                const uint8_t* shuffler,
                int width,
                int height);

// Shuffle AR64 channel order.  e.g. AR64 to AB64.
// shuffler is 16 bytes.
LIBYUV_API
int AR64Shuffle(const uint16_t* src_ar64,
                int src_stride_ar64,
                uint16_t* dst_ar64,
                int dst_stride_ar64,
                const uint8_t* shuffler,
                int width,
                int height);

// Sobel ARGB effect with planar output.
LIBYUV_API
int ARGBSobelToPlane(const uint8_t* src_argb,
                     int src_stride_argb,
                     uint8_t* dst_y,
                     int dst_stride_y,
                     int width,
                     int height);

// Sobel ARGB effect.
LIBYUV_API
int ARGBSobel(const uint8_t* src_argb,
              int src_stride_argb,
              uint8_t* dst_argb,
              int dst_stride_argb,
              int width,
              int height);

// Sobel ARGB effect w/ Sobel X, Sobel, Sobel Y in ARGB.
LIBYUV_API
int ARGBSobelXY(const uint8_t* src_argb,
                int src_stride_argb,
                uint8_t* dst_argb,
                int dst_stride_argb,
                int width,
                int height);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_PLANAR_FUNCTIONS_H_
