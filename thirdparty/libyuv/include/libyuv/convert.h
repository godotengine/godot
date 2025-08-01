/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_CONVERT_H_
#define INCLUDE_LIBYUV_CONVERT_H_

#include "libyuv/basic_types.h"

#include "libyuv/rotate.h"  // For enum RotationMode.

// TODO(fbarchard): fix WebRTC source to include following libyuv headers:
#include "libyuv/convert_argb.h"      // For WebRTC I420ToARGB. b/620
#include "libyuv/convert_from.h"      // For WebRTC ConvertFromI420. b/620
#include "libyuv/planar_functions.h"  // For WebRTC I420Rect, CopyPlane. b/618

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Convert I444 to I420.
LIBYUV_API
int I444ToI420(const uint8_t* src_y,
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

// Convert I444 to NV12.
LIBYUV_API
int I444ToNV12(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert I444 to NV21.
LIBYUV_API
int I444ToNV21(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height);

// Convert I422 to I420.
LIBYUV_API
int I422ToI420(const uint8_t* src_y,
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

// Convert I422 to I444.
LIBYUV_API
int I422ToI444(const uint8_t* src_y,
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

// Convert I422 to I210.
LIBYUV_API
int I422ToI210(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert MM21 to NV12.
LIBYUV_API
int MM21ToNV12(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert MM21 to I420.
LIBYUV_API
int MM21ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert MM21 to YUY2
LIBYUV_API
int MM21ToYUY2(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_yuy2,
               int dst_stride_yuy2,
               int width,
               int height);

// Convert MT2T to P010
// Note that src_y and src_uv point to packed 10-bit values, so the Y plane will
// be 10 / 8 times the dimensions of the image. Also for this reason,
// src_stride_y and src_stride_uv are given in bytes.
LIBYUV_API
int MT2TToP010(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert I422 to NV21.
LIBYUV_API
int I422ToNV21(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_u,
               int src_stride_u,
               const uint8_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height);

// Copy I420 to I420.
#define I420ToI420 I420Copy
LIBYUV_API
int I420Copy(const uint8_t* src_y,
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

// Convert I420 to I444.
LIBYUV_API
int I420ToI444(const uint8_t* src_y,
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

// Copy I010 to I010
#define I010ToI010 I010Copy
#define H010ToH010 I010Copy
LIBYUV_API
int I010Copy(const uint16_t* src_y,
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

// Convert 10 bit YUV to 8 bit
#define H010ToH420 I010ToI420
LIBYUV_API
int I010ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H210ToH420 I210ToI420
LIBYUV_API
int I210ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H210ToH422 I210ToI422
LIBYUV_API
int I210ToI422(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H410ToH420 I410ToI420
LIBYUV_API
int I410ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H410ToH444 I410ToI444
LIBYUV_API
int I410ToI444(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H012ToH420 I012ToI420
LIBYUV_API
int I012ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H212ToH422 I212ToI422
LIBYUV_API
int I212ToI422(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H212ToH420 I212ToI420
LIBYUV_API
int I212ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H412ToH444 I412ToI444
LIBYUV_API
int I412ToI444(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

#define H412ToH420 I412ToI420
LIBYUV_API
int I412ToI420(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert 10 bit P010 to 8 bit NV12.
// dst_y can be NULL
LIBYUV_API
int P010ToNV12(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

#define I412ToI012 I410ToI010
#define H410ToH010 I410ToI010
#define H412ToH012 I410ToI010
LIBYUV_API
int I410ToI010(const uint16_t* src_y,
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

#define I212ToI012 I210ToI010
#define H210ToH010 I210ToI010
#define H212ToH012 I210ToI010
LIBYUV_API
int I210ToI010(const uint16_t* src_y,
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

// Convert I010 to I410
LIBYUV_API
int I010ToI410(const uint16_t* src_y,
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

// Convert I012 to I412
#define I012ToI412 I010ToI410

// Convert I210 to I410
LIBYUV_API
int I210ToI410(const uint16_t* src_y,
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

// Convert I212 to I412
#define I212ToI412 I210ToI410

// Convert I010 to P010
LIBYUV_API
int I010ToP010(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert 10 bit YUV I010 to NV12
LIBYUV_API
int I010ToNV12(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert I210 to P210
LIBYUV_API
int I210ToP210(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert I012 to P012
LIBYUV_API
int I012ToP012(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert I212 to P212
LIBYUV_API
int I212ToP212(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_u,
               int src_stride_u,
               const uint16_t* src_v,
               int src_stride_v,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert I400 (grey) to I420.
LIBYUV_API
int I400ToI420(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert J420 to I420.
LIBYUV_API
int J420ToI420(const uint8_t* src_y,
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

// Convert I400 (grey) to NV21.
LIBYUV_API
int I400ToNV21(const uint8_t* src_y,
               int src_stride_y,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height);

#define J400ToJ420 I400ToI420

// Convert NV12 to I420.
LIBYUV_API
int NV12ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert NV21 to I420.
LIBYUV_API
int NV21ToI420(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_vu,
               int src_stride_vu,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert NV12 to NV24.
LIBYUV_API
int NV12ToNV24(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert NV16 to NV24.
LIBYUV_API
int NV16ToNV24(const uint8_t* src_y,
               int src_stride_y,
               const uint8_t* src_uv,
               int src_stride_uv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert P010 to I010.
LIBYUV_API
int P010ToI010(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert P012 to I012.
LIBYUV_API
int P012ToI012(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_u,
               int dst_stride_u,
               uint16_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert P010 to P410.
LIBYUV_API
int P010ToP410(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert P012 to P412.
#define P012ToP412 P010ToP410

// Convert P016 to P416.
#define P016ToP416 P010ToP410

// Convert P210 to P410.
LIBYUV_API
int P210ToP410(const uint16_t* src_y,
               int src_stride_y,
               const uint16_t* src_uv,
               int src_stride_uv,
               uint16_t* dst_y,
               int dst_stride_y,
               uint16_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert P212 to P412.
#define P212ToP412 P210ToP410

// Convert P216 to P416.
#define P216ToP416 P210ToP410

// Convert YUY2 to I420.
LIBYUV_API
int YUY2ToI420(const uint8_t* src_yuy2,
               int src_stride_yuy2,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert UYVY to I420.
LIBYUV_API
int UYVYToI420(const uint8_t* src_uyvy,
               int src_stride_uyvy,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert AYUV to NV12.
LIBYUV_API
int AYUVToNV12(const uint8_t* src_ayuv,
               int src_stride_ayuv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int width,
               int height);

// Convert AYUV to NV21.
LIBYUV_API
int AYUVToNV21(const uint8_t* src_ayuv,
               int src_stride_ayuv,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int width,
               int height);

// Convert Android420 to I420.
LIBYUV_API
int Android420ToI420(const uint8_t* src_y,
                     int src_stride_y,
                     const uint8_t* src_u,
                     int src_stride_u,
                     const uint8_t* src_v,
                     int src_stride_v,
                     int src_pixel_stride_uv,
                     uint8_t* dst_y,
                     int dst_stride_y,
                     uint8_t* dst_u,
                     int dst_stride_u,
                     uint8_t* dst_v,
                     int dst_stride_v,
                     int width,
                     int height);

// ARGB little endian (bgra in memory) to I420.
LIBYUV_API
int ARGBToI420(const uint8_t* src_argb,
               int src_stride_argb,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// Convert ARGB to I420 with Alpha
LIBYUV_API
int ARGBToI420Alpha(const uint8_t* src_argb,
                    int src_stride_argb,
                    uint8_t* dst_y,
                    int dst_stride_y,
                    uint8_t* dst_u,
                    int dst_stride_u,
                    uint8_t* dst_v,
                    int dst_stride_v,
                    uint8_t* dst_a,
                    int dst_stride_a,
                    int width,
                    int height);

// BGRA little endian (argb in memory) to I420.
LIBYUV_API
int BGRAToI420(const uint8_t* src_bgra,
               int src_stride_bgra,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// ABGR little endian (rgba in memory) to I420.
LIBYUV_API
int ABGRToI420(const uint8_t* src_abgr,
               int src_stride_abgr,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// RGBA little endian (abgr in memory) to I420.
LIBYUV_API
int RGBAToI420(const uint8_t* src_rgba,
               int src_stride_rgba,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int width,
               int height);

// RGB little endian (bgr in memory) to I420.
LIBYUV_API
int RGB24ToI420(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_y,
                int dst_stride_y,
                uint8_t* dst_u,
                int dst_stride_u,
                uint8_t* dst_v,
                int dst_stride_v,
                int width,
                int height);

// RGB little endian (bgr in memory) to J420.
LIBYUV_API
int RGB24ToJ420(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_y,
                int dst_stride_y,
                uint8_t* dst_u,
                int dst_stride_u,
                uint8_t* dst_v,
                int dst_stride_v,
                int width,
                int height);

// RGB big endian (rgb in memory) to I420.
LIBYUV_API
int RAWToI420(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height);

// RGB big endian (rgb in memory) to I444.
LIBYUV_API
int RAWToI444(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height);

// RGB big endian (rgb in memory) to J420.
LIBYUV_API
int RAWToJ420(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height);

// RGB big endian (rgb in memory) to J444.
LIBYUV_API
int RAWToJ444(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_y,
              int dst_stride_y,
              uint8_t* dst_u,
              int dst_stride_u,
              uint8_t* dst_v,
              int dst_stride_v,
              int width,
              int height);

// RGB16 (RGBP fourcc) little endian to I420.
LIBYUV_API
int RGB565ToI420(const uint8_t* src_rgb565,
                 int src_stride_rgb565,
                 uint8_t* dst_y,
                 int dst_stride_y,
                 uint8_t* dst_u,
                 int dst_stride_u,
                 uint8_t* dst_v,
                 int dst_stride_v,
                 int width,
                 int height);

// RGB15 (RGBO fourcc) little endian to I420.
LIBYUV_API
int ARGB1555ToI420(const uint8_t* src_argb1555,
                   int src_stride_argb1555,
                   uint8_t* dst_y,
                   int dst_stride_y,
                   uint8_t* dst_u,
                   int dst_stride_u,
                   uint8_t* dst_v,
                   int dst_stride_v,
                   int width,
                   int height);

// RGB12 (R444 fourcc) little endian to I420.
LIBYUV_API
int ARGB4444ToI420(const uint8_t* src_argb4444,
                   int src_stride_argb4444,
                   uint8_t* dst_y,
                   int dst_stride_y,
                   uint8_t* dst_u,
                   int dst_stride_u,
                   uint8_t* dst_v,
                   int dst_stride_v,
                   int width,
                   int height);

// RGB little endian (bgr in memory) to J400.
LIBYUV_API
int RGB24ToJ400(const uint8_t* src_rgb24,
                int src_stride_rgb24,
                uint8_t* dst_yj,
                int dst_stride_yj,
                int width,
                int height);

// RGB big endian (rgb in memory) to J400.
LIBYUV_API
int RAWToJ400(const uint8_t* src_raw,
              int src_stride_raw,
              uint8_t* dst_yj,
              int dst_stride_yj,
              int width,
              int height);

// src_width/height provided by capture.
// dst_width/height for clipping determine final size.
LIBYUV_API
int MJPGToI420(const uint8_t* sample,
               size_t sample_size,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_u,
               int dst_stride_u,
               uint8_t* dst_v,
               int dst_stride_v,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height);

// JPEG to NV21
LIBYUV_API
int MJPGToNV21(const uint8_t* sample,
               size_t sample_size,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_vu,
               int dst_stride_vu,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height);

// JPEG to NV12
LIBYUV_API
int MJPGToNV12(const uint8_t* sample,
               size_t sample_size,
               uint8_t* dst_y,
               int dst_stride_y,
               uint8_t* dst_uv,
               int dst_stride_uv,
               int src_width,
               int src_height,
               int dst_width,
               int dst_height);

// Query size of MJPG in pixels.
LIBYUV_API
int MJPGSize(const uint8_t* sample,
             size_t sample_size,
             int* width,
             int* height);

// Convert camera sample to I420 with cropping, rotation and vertical flip.
// "src_size" is needed to parse MJPG.
// "dst_stride_y" number of bytes in a row of the dst_y plane.
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
int ConvertToI420(const uint8_t* sample,
                  size_t sample_size,
                  uint8_t* dst_y,
                  int dst_stride_y,
                  uint8_t* dst_u,
                  int dst_stride_u,
                  uint8_t* dst_v,
                  int dst_stride_v,
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

#endif  // INCLUDE_LIBYUV_CONVERT_H_
