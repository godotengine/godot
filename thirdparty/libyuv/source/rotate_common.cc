/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/rotate_row.h"
#include "libyuv/row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

void TransposeWx8_C(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width) {
  int i;
  for (i = 0; i < width; ++i) {
    dst[0] = src[0 * src_stride];
    dst[1] = src[1 * src_stride];
    dst[2] = src[2 * src_stride];
    dst[3] = src[3 * src_stride];
    dst[4] = src[4 * src_stride];
    dst[5] = src[5 * src_stride];
    dst[6] = src[6 * src_stride];
    dst[7] = src[7 * src_stride];
    ++src;
    dst += dst_stride;
  }
}

void TransposeWx16_C(const uint8_t* src,
                     int src_stride,
                     uint8_t* dst,
                     int dst_stride,
                     int width) {
  TransposeWx8_C(src, src_stride, dst, dst_stride, width);
  TransposeWx8_C((src + 8 * src_stride), src_stride, (dst + 8), dst_stride,
                 width);
}

void TransposeUVWx8_C(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width) {
  int i;
  for (i = 0; i < width; ++i) {
    dst_a[0] = src[0 * src_stride + 0];
    dst_b[0] = src[0 * src_stride + 1];
    dst_a[1] = src[1 * src_stride + 0];
    dst_b[1] = src[1 * src_stride + 1];
    dst_a[2] = src[2 * src_stride + 0];
    dst_b[2] = src[2 * src_stride + 1];
    dst_a[3] = src[3 * src_stride + 0];
    dst_b[3] = src[3 * src_stride + 1];
    dst_a[4] = src[4 * src_stride + 0];
    dst_b[4] = src[4 * src_stride + 1];
    dst_a[5] = src[5 * src_stride + 0];
    dst_b[5] = src[5 * src_stride + 1];
    dst_a[6] = src[6 * src_stride + 0];
    dst_b[6] = src[6 * src_stride + 1];
    dst_a[7] = src[7 * src_stride + 0];
    dst_b[7] = src[7 * src_stride + 1];
    src += 2;
    dst_a += dst_stride_a;
    dst_b += dst_stride_b;
  }
}

void TransposeWxH_C(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height) {
  int i;
  for (i = 0; i < width; ++i) {
    int j;
    for (j = 0; j < height; ++j) {
      dst[i * dst_stride + j] = src[j * src_stride + i];
    }
  }
}

void TransposeUVWxH_C(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height) {
  int i;
  for (i = 0; i < width * 2; i += 2) {
    int j;
    for (j = 0; j < height; ++j) {
      dst_a[((i >> 1) * dst_stride_a) + j] = src[i + (j * src_stride)];
      dst_b[((i >> 1) * dst_stride_b) + j] = src[i + (j * src_stride) + 1];
    }
  }
}

void TransposeWx8_16_C(const uint16_t* src,
                       int src_stride,
                       uint16_t* dst,
                       int dst_stride,
                       int width) {
  int i;
  for (i = 0; i < width; ++i) {
    dst[0] = src[0 * src_stride];
    dst[1] = src[1 * src_stride];
    dst[2] = src[2 * src_stride];
    dst[3] = src[3 * src_stride];
    dst[4] = src[4 * src_stride];
    dst[5] = src[5 * src_stride];
    dst[6] = src[6 * src_stride];
    dst[7] = src[7 * src_stride];
    ++src;
    dst += dst_stride;
  }
}

void TransposeWxH_16_C(const uint16_t* src,
                       int src_stride,
                       uint16_t* dst,
                       int dst_stride,
                       int width,
                       int height) {
  int i;
  for (i = 0; i < width; ++i) {
    int j;
    for (j = 0; j < height; ++j) {
      dst[i * dst_stride + j] = src[j * src_stride + i];
    }
  }
}

// Transpose 32 bit values (ARGB)
void Transpose4x4_32_C(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width) {
  const uint8_t* src1 = src + src_stride;
  const uint8_t* src2 = src1 + src_stride;
  const uint8_t* src3 = src2 + src_stride;
  uint8_t* dst1 = dst + dst_stride;
  uint8_t* dst2 = dst1 + dst_stride;
  uint8_t* dst3 = dst2 + dst_stride;
  int i;
  for (i = 0; i < width; i += 4) {
    uint32_t p00 = ((uint32_t*)(src))[0];
    uint32_t p10 = ((uint32_t*)(src))[1];
    uint32_t p20 = ((uint32_t*)(src))[2];
    uint32_t p30 = ((uint32_t*)(src))[3];
    uint32_t p01 = ((uint32_t*)(src1))[0];
    uint32_t p11 = ((uint32_t*)(src1))[1];
    uint32_t p21 = ((uint32_t*)(src1))[2];
    uint32_t p31 = ((uint32_t*)(src1))[3];
    uint32_t p02 = ((uint32_t*)(src2))[0];
    uint32_t p12 = ((uint32_t*)(src2))[1];
    uint32_t p22 = ((uint32_t*)(src2))[2];
    uint32_t p32 = ((uint32_t*)(src2))[3];
    uint32_t p03 = ((uint32_t*)(src3))[0];
    uint32_t p13 = ((uint32_t*)(src3))[1];
    uint32_t p23 = ((uint32_t*)(src3))[2];
    uint32_t p33 = ((uint32_t*)(src3))[3];
    ((uint32_t*)(dst))[0] = p00;
    ((uint32_t*)(dst))[1] = p01;
    ((uint32_t*)(dst))[2] = p02;
    ((uint32_t*)(dst))[3] = p03;
    ((uint32_t*)(dst1))[0] = p10;
    ((uint32_t*)(dst1))[1] = p11;
    ((uint32_t*)(dst1))[2] = p12;
    ((uint32_t*)(dst1))[3] = p13;
    ((uint32_t*)(dst2))[0] = p20;
    ((uint32_t*)(dst2))[1] = p21;
    ((uint32_t*)(dst2))[2] = p22;
    ((uint32_t*)(dst2))[3] = p23;
    ((uint32_t*)(dst3))[0] = p30;
    ((uint32_t*)(dst3))[1] = p31;
    ((uint32_t*)(dst3))[2] = p32;
    ((uint32_t*)(dst3))[3] = p33;
    src += src_stride * 4;  // advance 4 rows
    src1 += src_stride * 4;
    src2 += src_stride * 4;
    src3 += src_stride * 4;
    dst += 4 * 4;  // advance 4 columns
    dst1 += 4 * 4;
    dst2 += 4 * 4;
    dst3 += 4 * 4;
  }
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
