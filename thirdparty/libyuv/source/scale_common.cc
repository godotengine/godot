/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/scale.h"

#include <assert.h>
#include <string.h>

#include "libyuv/cpu_id.h"
#include "libyuv/planar_functions.h"  // For CopyARGB
#include "libyuv/row.h"
#include "libyuv/scale_row.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#ifdef __cplusplus
#define STATIC_CAST(type, expr) static_cast<type>(expr)
#else
#define STATIC_CAST(type, expr) (type)(expr)
#endif

// TODO(fbarchard): make clamp255 preserve negative values.
static __inline int32_t clamp255(int32_t v) {
  return (-(v >= 255) | v) & 255;
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits
// 16384 = 10 bits
// 4096 = 12 bits
// 256 = 16 bits
// TODO(fbarchard): change scale to bits
#define C16TO8(v, scale) clamp255(((v) * (scale)) >> 16)

static __inline int Abs(int v) {
  return v >= 0 ? v : -v;
}

// CPU agnostic row functions
void ScaleRowDown2_C(const uint8_t* src_ptr,
                     ptrdiff_t src_stride,
                     uint8_t* dst,
                     int dst_width) {
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = src_ptr[1];
    dst[1] = src_ptr[3];
    dst += 2;
    src_ptr += 4;
  }
  if (dst_width & 1) {
    dst[0] = src_ptr[1];
  }
}

void ScaleRowDown2_16_C(const uint16_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint16_t* dst,
                        int dst_width) {
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = src_ptr[1];
    dst[1] = src_ptr[3];
    dst += 2;
    src_ptr += 4;
  }
  if (dst_width & 1) {
    dst[0] = src_ptr[1];
  }
}

void ScaleRowDown2_16To8_C(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst,
                           int dst_width,
                           int scale) {
  int x;
  (void)src_stride;
  assert(scale >= 256);
  assert(scale <= 32768);
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8(src_ptr[1], scale));
    dst[1] = STATIC_CAST(uint8_t, C16TO8(src_ptr[3], scale));
    dst += 2;
    src_ptr += 4;
  }
  if (dst_width & 1) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8(src_ptr[1], scale));
  }
}

void ScaleRowDown2_16To8_Odd_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint8_t* dst,
                               int dst_width,
                               int scale) {
  int x;
  (void)src_stride;
  assert(scale >= 256);
  assert(scale <= 32768);
  dst_width -= 1;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8(src_ptr[1], scale));
    dst[1] = STATIC_CAST(uint8_t, C16TO8(src_ptr[3], scale));
    dst += 2;
    src_ptr += 4;
  }
  if (dst_width & 1) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8(src_ptr[1], scale));
    dst += 1;
    src_ptr += 2;
  }
  dst[0] = STATIC_CAST(uint8_t, C16TO8(src_ptr[0], scale));
}

void ScaleRowDown2Linear_C(const uint8_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint8_t* dst,
                           int dst_width) {
  const uint8_t* s = src_ptr;
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = (s[0] + s[1] + 1) >> 1;
    dst[1] = (s[2] + s[3] + 1) >> 1;
    dst += 2;
    s += 4;
  }
  if (dst_width & 1) {
    dst[0] = (s[0] + s[1] + 1) >> 1;
  }
}

void ScaleRowDown2Linear_16_C(const uint16_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint16_t* dst,
                              int dst_width) {
  const uint16_t* s = src_ptr;
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = (s[0] + s[1] + 1) >> 1;
    dst[1] = (s[2] + s[3] + 1) >> 1;
    dst += 2;
    s += 4;
  }
  if (dst_width & 1) {
    dst[0] = (s[0] + s[1] + 1) >> 1;
  }
}

void ScaleRowDown2Linear_16To8_C(const uint16_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint8_t* dst,
                                 int dst_width,
                                 int scale) {
  const uint16_t* s = src_ptr;
  int x;
  (void)src_stride;
  assert(scale >= 256);
  assert(scale <= 32768);
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8((s[0] + s[1] + 1) >> 1, scale));
    dst[1] = STATIC_CAST(uint8_t, C16TO8((s[2] + s[3] + 1) >> 1, scale));
    dst += 2;
    s += 4;
  }
  if (dst_width & 1) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8((s[0] + s[1] + 1) >> 1, scale));
  }
}

void ScaleRowDown2Linear_16To8_Odd_C(const uint16_t* src_ptr,
                                     ptrdiff_t src_stride,
                                     uint8_t* dst,
                                     int dst_width,
                                     int scale) {
  const uint16_t* s = src_ptr;
  int x;
  (void)src_stride;
  assert(scale >= 256);
  assert(scale <= 32768);
  dst_width -= 1;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8((s[0] + s[1] + 1) >> 1, scale));
    dst[1] = STATIC_CAST(uint8_t, C16TO8((s[2] + s[3] + 1) >> 1, scale));
    dst += 2;
    s += 4;
  }
  if (dst_width & 1) {
    dst[0] = STATIC_CAST(uint8_t, C16TO8((s[0] + s[1] + 1) >> 1, scale));
    dst += 1;
    s += 2;
  }
  dst[0] = STATIC_CAST(uint8_t, C16TO8(s[0], scale));
}

void ScaleRowDown2Box_C(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  int x;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = (s[0] + s[1] + t[0] + t[1] + 2) >> 2;
    dst[1] = (s[2] + s[3] + t[2] + t[3] + 2) >> 2;
    dst += 2;
    s += 4;
    t += 4;
  }
  if (dst_width & 1) {
    dst[0] = (s[0] + s[1] + t[0] + t[1] + 2) >> 2;
  }
}

void ScaleRowDown2Box_Odd_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst,
                            int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  int x;
  dst_width -= 1;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = (s[0] + s[1] + t[0] + t[1] + 2) >> 2;
    dst[1] = (s[2] + s[3] + t[2] + t[3] + 2) >> 2;
    dst += 2;
    s += 4;
    t += 4;
  }
  if (dst_width & 1) {
    dst[0] = (s[0] + s[1] + t[0] + t[1] + 2) >> 2;
    dst += 1;
    s += 2;
    t += 2;
  }
  dst[0] = (s[0] + t[0] + 1) >> 1;
}

void ScaleRowDown2Box_16_C(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint16_t* dst,
                           int dst_width) {
  const uint16_t* s = src_ptr;
  const uint16_t* t = src_ptr + src_stride;
  int x;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = (s[0] + s[1] + t[0] + t[1] + 2) >> 2;
    dst[1] = (s[2] + s[3] + t[2] + t[3] + 2) >> 2;
    dst += 2;
    s += 4;
    t += 4;
  }
  if (dst_width & 1) {
    dst[0] = (s[0] + s[1] + t[0] + t[1] + 2) >> 2;
  }
}

void ScaleRowDown2Box_16To8_C(const uint16_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst,
                              int dst_width,
                              int scale) {
  const uint16_t* s = src_ptr;
  const uint16_t* t = src_ptr + src_stride;
  int x;
  assert(scale >= 256);
  assert(scale <= 32768);
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = STATIC_CAST(uint8_t,
                         C16TO8((s[0] + s[1] + t[0] + t[1] + 2) >> 2, scale));
    dst[1] = STATIC_CAST(uint8_t,
                         C16TO8((s[2] + s[3] + t[2] + t[3] + 2) >> 2, scale));
    dst += 2;
    s += 4;
    t += 4;
  }
  if (dst_width & 1) {
    dst[0] = STATIC_CAST(uint8_t,
                         C16TO8((s[0] + s[1] + t[0] + t[1] + 2) >> 2, scale));
  }
}

void ScaleRowDown2Box_16To8_Odd_C(const uint16_t* src_ptr,
                                  ptrdiff_t src_stride,
                                  uint8_t* dst,
                                  int dst_width,
                                  int scale) {
  const uint16_t* s = src_ptr;
  const uint16_t* t = src_ptr + src_stride;
  int x;
  assert(scale >= 256);
  assert(scale <= 32768);
  dst_width -= 1;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = STATIC_CAST(uint8_t,
                         C16TO8((s[0] + s[1] + t[0] + t[1] + 2) >> 2, scale));
    dst[1] = STATIC_CAST(uint8_t,
                         C16TO8((s[2] + s[3] + t[2] + t[3] + 2) >> 2, scale));
    dst += 2;
    s += 4;
    t += 4;
  }
  if (dst_width & 1) {
    dst[0] = STATIC_CAST(uint8_t,
                         C16TO8((s[0] + s[1] + t[0] + t[1] + 2) >> 2, scale));
    dst += 1;
    s += 2;
    t += 2;
  }
  dst[0] = STATIC_CAST(uint8_t, C16TO8((s[0] + t[0] + 1) >> 1, scale));
}

void ScaleRowDown4_C(const uint8_t* src_ptr,
                     ptrdiff_t src_stride,
                     uint8_t* dst,
                     int dst_width) {
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = src_ptr[2];
    dst[1] = src_ptr[6];
    dst += 2;
    src_ptr += 8;
  }
  if (dst_width & 1) {
    dst[0] = src_ptr[2];
  }
}

void ScaleRowDown4_16_C(const uint16_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint16_t* dst,
                        int dst_width) {
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = src_ptr[2];
    dst[1] = src_ptr[6];
    dst += 2;
    src_ptr += 8;
  }
  if (dst_width & 1) {
    dst[0] = src_ptr[2];
  }
}

void ScaleRowDown4Box_C(const uint8_t* src_ptr,
                        ptrdiff_t src_stride,
                        uint8_t* dst,
                        int dst_width) {
  intptr_t stride = src_stride;
  int x;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[3] +
              src_ptr[stride + 0] + src_ptr[stride + 1] + src_ptr[stride + 2] +
              src_ptr[stride + 3] + src_ptr[stride * 2 + 0] +
              src_ptr[stride * 2 + 1] + src_ptr[stride * 2 + 2] +
              src_ptr[stride * 2 + 3] + src_ptr[stride * 3 + 0] +
              src_ptr[stride * 3 + 1] + src_ptr[stride * 3 + 2] +
              src_ptr[stride * 3 + 3] + 8) >>
             4;
    dst[1] = (src_ptr[4] + src_ptr[5] + src_ptr[6] + src_ptr[7] +
              src_ptr[stride + 4] + src_ptr[stride + 5] + src_ptr[stride + 6] +
              src_ptr[stride + 7] + src_ptr[stride * 2 + 4] +
              src_ptr[stride * 2 + 5] + src_ptr[stride * 2 + 6] +
              src_ptr[stride * 2 + 7] + src_ptr[stride * 3 + 4] +
              src_ptr[stride * 3 + 5] + src_ptr[stride * 3 + 6] +
              src_ptr[stride * 3 + 7] + 8) >>
             4;
    dst += 2;
    src_ptr += 8;
  }
  if (dst_width & 1) {
    dst[0] = (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[3] +
              src_ptr[stride + 0] + src_ptr[stride + 1] + src_ptr[stride + 2] +
              src_ptr[stride + 3] + src_ptr[stride * 2 + 0] +
              src_ptr[stride * 2 + 1] + src_ptr[stride * 2 + 2] +
              src_ptr[stride * 2 + 3] + src_ptr[stride * 3 + 0] +
              src_ptr[stride * 3 + 1] + src_ptr[stride * 3 + 2] +
              src_ptr[stride * 3 + 3] + 8) >>
             4;
  }
}

void ScaleRowDown4Box_16_C(const uint16_t* src_ptr,
                           ptrdiff_t src_stride,
                           uint16_t* dst,
                           int dst_width) {
  intptr_t stride = src_stride;
  int x;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[3] +
              src_ptr[stride + 0] + src_ptr[stride + 1] + src_ptr[stride + 2] +
              src_ptr[stride + 3] + src_ptr[stride * 2 + 0] +
              src_ptr[stride * 2 + 1] + src_ptr[stride * 2 + 2] +
              src_ptr[stride * 2 + 3] + src_ptr[stride * 3 + 0] +
              src_ptr[stride * 3 + 1] + src_ptr[stride * 3 + 2] +
              src_ptr[stride * 3 + 3] + 8) >>
             4;
    dst[1] = (src_ptr[4] + src_ptr[5] + src_ptr[6] + src_ptr[7] +
              src_ptr[stride + 4] + src_ptr[stride + 5] + src_ptr[stride + 6] +
              src_ptr[stride + 7] + src_ptr[stride * 2 + 4] +
              src_ptr[stride * 2 + 5] + src_ptr[stride * 2 + 6] +
              src_ptr[stride * 2 + 7] + src_ptr[stride * 3 + 4] +
              src_ptr[stride * 3 + 5] + src_ptr[stride * 3 + 6] +
              src_ptr[stride * 3 + 7] + 8) >>
             4;
    dst += 2;
    src_ptr += 8;
  }
  if (dst_width & 1) {
    dst[0] = (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[3] +
              src_ptr[stride + 0] + src_ptr[stride + 1] + src_ptr[stride + 2] +
              src_ptr[stride + 3] + src_ptr[stride * 2 + 0] +
              src_ptr[stride * 2 + 1] + src_ptr[stride * 2 + 2] +
              src_ptr[stride * 2 + 3] + src_ptr[stride * 3 + 0] +
              src_ptr[stride * 3 + 1] + src_ptr[stride * 3 + 2] +
              src_ptr[stride * 3 + 3] + 8) >>
             4;
  }
}

void ScaleRowDown34_C(const uint8_t* src_ptr,
                      ptrdiff_t src_stride,
                      uint8_t* dst,
                      int dst_width) {
  int x;
  (void)src_stride;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (x = 0; x < dst_width; x += 3) {
    dst[0] = src_ptr[0];
    dst[1] = src_ptr[1];
    dst[2] = src_ptr[3];
    dst += 3;
    src_ptr += 4;
  }
}

void ScaleRowDown34_16_C(const uint16_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint16_t* dst,
                         int dst_width) {
  int x;
  (void)src_stride;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (x = 0; x < dst_width; x += 3) {
    dst[0] = src_ptr[0];
    dst[1] = src_ptr[1];
    dst[2] = src_ptr[3];
    dst += 3;
    src_ptr += 4;
  }
}

// Filter rows 0 and 1 together, 3 : 1
void ScaleRowDown34_0_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* d,
                            int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  int x;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (x = 0; x < dst_width; x += 3) {
    uint8_t a0 = (s[0] * 3 + s[1] * 1 + 2) >> 2;
    uint8_t a1 = (s[1] * 1 + s[2] * 1 + 1) >> 1;
    uint8_t a2 = (s[2] * 1 + s[3] * 3 + 2) >> 2;
    uint8_t b0 = (t[0] * 3 + t[1] * 1 + 2) >> 2;
    uint8_t b1 = (t[1] * 1 + t[2] * 1 + 1) >> 1;
    uint8_t b2 = (t[2] * 1 + t[3] * 3 + 2) >> 2;
    d[0] = (a0 * 3 + b0 + 2) >> 2;
    d[1] = (a1 * 3 + b1 + 2) >> 2;
    d[2] = (a2 * 3 + b2 + 2) >> 2;
    d += 3;
    s += 4;
    t += 4;
  }
}

void ScaleRowDown34_0_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* d,
                               int dst_width) {
  const uint16_t* s = src_ptr;
  const uint16_t* t = src_ptr + src_stride;
  int x;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (x = 0; x < dst_width; x += 3) {
    uint16_t a0 = (s[0] * 3 + s[1] * 1 + 2) >> 2;
    uint16_t a1 = (s[1] * 1 + s[2] * 1 + 1) >> 1;
    uint16_t a2 = (s[2] * 1 + s[3] * 3 + 2) >> 2;
    uint16_t b0 = (t[0] * 3 + t[1] * 1 + 2) >> 2;
    uint16_t b1 = (t[1] * 1 + t[2] * 1 + 1) >> 1;
    uint16_t b2 = (t[2] * 1 + t[3] * 3 + 2) >> 2;
    d[0] = (a0 * 3 + b0 + 2) >> 2;
    d[1] = (a1 * 3 + b1 + 2) >> 2;
    d[2] = (a2 * 3 + b2 + 2) >> 2;
    d += 3;
    s += 4;
    t += 4;
  }
}

// Filter rows 1 and 2 together, 1 : 1
void ScaleRowDown34_1_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* d,
                            int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  int x;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (x = 0; x < dst_width; x += 3) {
    uint8_t a0 = (s[0] * 3 + s[1] * 1 + 2) >> 2;
    uint8_t a1 = (s[1] * 1 + s[2] * 1 + 1) >> 1;
    uint8_t a2 = (s[2] * 1 + s[3] * 3 + 2) >> 2;
    uint8_t b0 = (t[0] * 3 + t[1] * 1 + 2) >> 2;
    uint8_t b1 = (t[1] * 1 + t[2] * 1 + 1) >> 1;
    uint8_t b2 = (t[2] * 1 + t[3] * 3 + 2) >> 2;
    d[0] = (a0 + b0 + 1) >> 1;
    d[1] = (a1 + b1 + 1) >> 1;
    d[2] = (a2 + b2 + 1) >> 1;
    d += 3;
    s += 4;
    t += 4;
  }
}

void ScaleRowDown34_1_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* d,
                               int dst_width) {
  const uint16_t* s = src_ptr;
  const uint16_t* t = src_ptr + src_stride;
  int x;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (x = 0; x < dst_width; x += 3) {
    uint16_t a0 = (s[0] * 3 + s[1] * 1 + 2) >> 2;
    uint16_t a1 = (s[1] * 1 + s[2] * 1 + 1) >> 1;
    uint16_t a2 = (s[2] * 1 + s[3] * 3 + 2) >> 2;
    uint16_t b0 = (t[0] * 3 + t[1] * 1 + 2) >> 2;
    uint16_t b1 = (t[1] * 1 + t[2] * 1 + 1) >> 1;
    uint16_t b2 = (t[2] * 1 + t[3] * 3 + 2) >> 2;
    d[0] = (a0 + b0 + 1) >> 1;
    d[1] = (a1 + b1 + 1) >> 1;
    d[2] = (a2 + b2 + 1) >> 1;
    d += 3;
    s += 4;
    t += 4;
  }
}

// Sample position: (O is src sample position, X is dst sample position)
//
//      v dst_ptr at here           v stop at here
//  X O X   X O X   X O X   X O X   X O X
//    ^ src_ptr at here
void ScaleRowUp2_Linear_C(const uint8_t* src_ptr,
                          uint8_t* dst_ptr,
                          int dst_width) {
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    dst_ptr[2 * x + 0] = (src_ptr[x + 0] * 3 + src_ptr[x + 1] * 1 + 2) >> 2;
    dst_ptr[2 * x + 1] = (src_ptr[x + 0] * 1 + src_ptr[x + 1] * 3 + 2) >> 2;
  }
}

// Sample position: (O is src sample position, X is dst sample position)
//
//    src_ptr at here
//  X v X   X   X   X   X   X   X   X   X
//    O       O       O       O       O
//  X   X   X   X   X   X   X   X   X   X
//      ^ dst_ptr at here           ^ stop at here
//  X   X   X   X   X   X   X   X   X   X
//    O       O       O       O       O
//  X   X   X   X   X   X   X   X   X   X
void ScaleRowUp2_Bilinear_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            ptrdiff_t dst_stride,
                            int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  uint8_t* d = dst_ptr;
  uint8_t* e = dst_ptr + dst_stride;
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    d[2 * x + 0] =
        (s[x + 0] * 9 + s[x + 1] * 3 + t[x + 0] * 3 + t[x + 1] * 1 + 8) >> 4;
    d[2 * x + 1] =
        (s[x + 0] * 3 + s[x + 1] * 9 + t[x + 0] * 1 + t[x + 1] * 3 + 8) >> 4;
    e[2 * x + 0] =
        (s[x + 0] * 3 + s[x + 1] * 1 + t[x + 0] * 9 + t[x + 1] * 3 + 8) >> 4;
    e[2 * x + 1] =
        (s[x + 0] * 1 + s[x + 1] * 3 + t[x + 0] * 3 + t[x + 1] * 9 + 8) >> 4;
  }
}

// Only suitable for at most 14 bit range.
void ScaleRowUp2_Linear_16_C(const uint16_t* src_ptr,
                             uint16_t* dst_ptr,
                             int dst_width) {
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    dst_ptr[2 * x + 0] = (src_ptr[x + 0] * 3 + src_ptr[x + 1] * 1 + 2) >> 2;
    dst_ptr[2 * x + 1] = (src_ptr[x + 0] * 1 + src_ptr[x + 1] * 3 + 2) >> 2;
  }
}

// Only suitable for at most 12bit range.
void ScaleRowUp2_Bilinear_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* dst_ptr,
                               ptrdiff_t dst_stride,
                               int dst_width) {
  const uint16_t* s = src_ptr;
  const uint16_t* t = src_ptr + src_stride;
  uint16_t* d = dst_ptr;
  uint16_t* e = dst_ptr + dst_stride;
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    d[2 * x + 0] =
        (s[x + 0] * 9 + s[x + 1] * 3 + t[x + 0] * 3 + t[x + 1] * 1 + 8) >> 4;
    d[2 * x + 1] =
        (s[x + 0] * 3 + s[x + 1] * 9 + t[x + 0] * 1 + t[x + 1] * 3 + 8) >> 4;
    e[2 * x + 0] =
        (s[x + 0] * 3 + s[x + 1] * 1 + t[x + 0] * 9 + t[x + 1] * 3 + 8) >> 4;
    e[2 * x + 1] =
        (s[x + 0] * 1 + s[x + 1] * 3 + t[x + 0] * 3 + t[x + 1] * 9 + 8) >> 4;
  }
}

// Scales a single row of pixels using point sampling.
void ScaleCols_C(uint8_t* dst_ptr,
                 const uint8_t* src_ptr,
                 int dst_width,
                 int x,
                 int dx) {
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst_ptr[0] = src_ptr[x >> 16];
    x += dx;
    dst_ptr[1] = src_ptr[x >> 16];
    x += dx;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    dst_ptr[0] = src_ptr[x >> 16];
  }
}

void ScaleCols_16_C(uint16_t* dst_ptr,
                    const uint16_t* src_ptr,
                    int dst_width,
                    int x,
                    int dx) {
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst_ptr[0] = src_ptr[x >> 16];
    x += dx;
    dst_ptr[1] = src_ptr[x >> 16];
    x += dx;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    dst_ptr[0] = src_ptr[x >> 16];
  }
}

// Scales a single row of pixels up by 2x using point sampling.
void ScaleColsUp2_C(uint8_t* dst_ptr,
                    const uint8_t* src_ptr,
                    int dst_width,
                    int x,
                    int dx) {
  int j;
  (void)x;
  (void)dx;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst_ptr[1] = dst_ptr[0] = src_ptr[0];
    src_ptr += 1;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    dst_ptr[0] = src_ptr[0];
  }
}

void ScaleColsUp2_16_C(uint16_t* dst_ptr,
                       const uint16_t* src_ptr,
                       int dst_width,
                       int x,
                       int dx) {
  int j;
  (void)x;
  (void)dx;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst_ptr[1] = dst_ptr[0] = src_ptr[0];
    src_ptr += 1;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    dst_ptr[0] = src_ptr[0];
  }
}

// (1-f)a + fb can be replaced with a + f(b-a)
#if defined(__arm__) || defined(__aarch64__)
#define BLENDER(a, b, f) \
  (uint8_t)((int)(a) + ((((int)((f)) * ((int)(b) - (int)(a))) + 0x8000) >> 16))
#else
// Intel uses 7 bit math with rounding.
#define BLENDER(a, b, f) \
  (uint8_t)((int)(a) + (((int)((f) >> 9) * ((int)(b) - (int)(a)) + 0x40) >> 7))
#endif

void ScaleFilterCols_C(uint8_t* dst_ptr,
                       const uint8_t* src_ptr,
                       int dst_width,
                       int x,
                       int dx) {
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
    x += dx;
    xi = x >> 16;
    a = src_ptr[xi];
    b = src_ptr[xi + 1];
    dst_ptr[1] = BLENDER(a, b, x & 0xffff);
    x += dx;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    int xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
  }
}

void ScaleFilterCols64_C(uint8_t* dst_ptr,
                         const uint8_t* src_ptr,
                         int dst_width,
                         int x32,
                         int dx) {
  int64_t x = (int64_t)(x32);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int64_t xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
    x += dx;
    xi = x >> 16;
    a = src_ptr[xi];
    b = src_ptr[xi + 1];
    dst_ptr[1] = BLENDER(a, b, x & 0xffff);
    x += dx;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    int64_t xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
  }
}
#undef BLENDER

// Same as 8 bit arm blender but return is cast to uint16_t
#define BLENDER(a, b, f) \
  (uint16_t)(            \
      (int)(a) +         \
      (int)((((int64_t)((f)) * ((int64_t)(b) - (int)(a))) + 0x8000) >> 16))

void ScaleFilterCols_16_C(uint16_t* dst_ptr,
                          const uint16_t* src_ptr,
                          int dst_width,
                          int x,
                          int dx) {
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
    x += dx;
    xi = x >> 16;
    a = src_ptr[xi];
    b = src_ptr[xi + 1];
    dst_ptr[1] = BLENDER(a, b, x & 0xffff);
    x += dx;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    int xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
  }
}

void ScaleFilterCols64_16_C(uint16_t* dst_ptr,
                            const uint16_t* src_ptr,
                            int dst_width,
                            int x32,
                            int dx) {
  int64_t x = (int64_t)(x32);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int64_t xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
    x += dx;
    xi = x >> 16;
    a = src_ptr[xi];
    b = src_ptr[xi + 1];
    dst_ptr[1] = BLENDER(a, b, x & 0xffff);
    x += dx;
    dst_ptr += 2;
  }
  if (dst_width & 1) {
    int64_t xi = x >> 16;
    int a = src_ptr[xi];
    int b = src_ptr[xi + 1];
    dst_ptr[0] = BLENDER(a, b, x & 0xffff);
  }
}
#undef BLENDER

void ScaleRowDown38_C(const uint8_t* src_ptr,
                      ptrdiff_t src_stride,
                      uint8_t* dst,
                      int dst_width) {
  int x;
  (void)src_stride;
  assert(dst_width % 3 == 0);
  for (x = 0; x < dst_width; x += 3) {
    dst[0] = src_ptr[0];
    dst[1] = src_ptr[3];
    dst[2] = src_ptr[6];
    dst += 3;
    src_ptr += 8;
  }
}

void ScaleRowDown38_16_C(const uint16_t* src_ptr,
                         ptrdiff_t src_stride,
                         uint16_t* dst,
                         int dst_width) {
  int x;
  (void)src_stride;
  assert(dst_width % 3 == 0);
  for (x = 0; x < dst_width; x += 3) {
    dst[0] = src_ptr[0];
    dst[1] = src_ptr[3];
    dst[2] = src_ptr[6];
    dst += 3;
    src_ptr += 8;
  }
}

// 8x3 -> 3x1
void ScaleRowDown38_3_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width) {
  intptr_t stride = src_stride;
  int i;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (i = 0; i < dst_width; i += 3) {
    dst_ptr[0] =
        (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[stride + 0] +
         src_ptr[stride + 1] + src_ptr[stride + 2] + src_ptr[stride * 2 + 0] +
         src_ptr[stride * 2 + 1] + src_ptr[stride * 2 + 2]) *
            (65536 / 9) >>
        16;
    dst_ptr[1] =
        (src_ptr[3] + src_ptr[4] + src_ptr[5] + src_ptr[stride + 3] +
         src_ptr[stride + 4] + src_ptr[stride + 5] + src_ptr[stride * 2 + 3] +
         src_ptr[stride * 2 + 4] + src_ptr[stride * 2 + 5]) *
            (65536 / 9) >>
        16;
    dst_ptr[2] =
        (src_ptr[6] + src_ptr[7] + src_ptr[stride + 6] + src_ptr[stride + 7] +
         src_ptr[stride * 2 + 6] + src_ptr[stride * 2 + 7]) *
            (65536 / 6) >>
        16;
    src_ptr += 8;
    dst_ptr += 3;
  }
}

void ScaleRowDown38_3_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* dst_ptr,
                               int dst_width) {
  intptr_t stride = src_stride;
  int i;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (i = 0; i < dst_width; i += 3) {
    dst_ptr[0] =
        (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[stride + 0] +
         src_ptr[stride + 1] + src_ptr[stride + 2] + src_ptr[stride * 2 + 0] +
         src_ptr[stride * 2 + 1] + src_ptr[stride * 2 + 2]) *
            (65536u / 9u) >>
        16;
    dst_ptr[1] =
        (src_ptr[3] + src_ptr[4] + src_ptr[5] + src_ptr[stride + 3] +
         src_ptr[stride + 4] + src_ptr[stride + 5] + src_ptr[stride * 2 + 3] +
         src_ptr[stride * 2 + 4] + src_ptr[stride * 2 + 5]) *
            (65536u / 9u) >>
        16;
    dst_ptr[2] =
        (src_ptr[6] + src_ptr[7] + src_ptr[stride + 6] + src_ptr[stride + 7] +
         src_ptr[stride * 2 + 6] + src_ptr[stride * 2 + 7]) *
            (65536u / 6u) >>
        16;
    src_ptr += 8;
    dst_ptr += 3;
  }
}

// 8x2 -> 3x1
void ScaleRowDown38_2_Box_C(const uint8_t* src_ptr,
                            ptrdiff_t src_stride,
                            uint8_t* dst_ptr,
                            int dst_width) {
  intptr_t stride = src_stride;
  int i;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (i = 0; i < dst_width; i += 3) {
    dst_ptr[0] = (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[stride + 0] +
                  src_ptr[stride + 1] + src_ptr[stride + 2]) *
                     (65536 / 6) >>
                 16;
    dst_ptr[1] = (src_ptr[3] + src_ptr[4] + src_ptr[5] + src_ptr[stride + 3] +
                  src_ptr[stride + 4] + src_ptr[stride + 5]) *
                     (65536 / 6) >>
                 16;
    dst_ptr[2] =
        (src_ptr[6] + src_ptr[7] + src_ptr[stride + 6] + src_ptr[stride + 7]) *
            (65536 / 4) >>
        16;
    src_ptr += 8;
    dst_ptr += 3;
  }
}

void ScaleRowDown38_2_Box_16_C(const uint16_t* src_ptr,
                               ptrdiff_t src_stride,
                               uint16_t* dst_ptr,
                               int dst_width) {
  intptr_t stride = src_stride;
  int i;
  assert((dst_width % 3 == 0) && (dst_width > 0));
  for (i = 0; i < dst_width; i += 3) {
    dst_ptr[0] = (src_ptr[0] + src_ptr[1] + src_ptr[2] + src_ptr[stride + 0] +
                  src_ptr[stride + 1] + src_ptr[stride + 2]) *
                     (65536u / 6u) >>
                 16;
    dst_ptr[1] = (src_ptr[3] + src_ptr[4] + src_ptr[5] + src_ptr[stride + 3] +
                  src_ptr[stride + 4] + src_ptr[stride + 5]) *
                     (65536u / 6u) >>
                 16;
    dst_ptr[2] =
        (src_ptr[6] + src_ptr[7] + src_ptr[stride + 6] + src_ptr[stride + 7]) *
            (65536u / 4u) >>
        16;
    src_ptr += 8;
    dst_ptr += 3;
  }
}

void ScaleAddRow_C(const uint8_t* src_ptr, uint16_t* dst_ptr, int src_width) {
  int x;
  assert(src_width > 0);
  for (x = 0; x < src_width - 1; x += 2) {
    dst_ptr[0] += src_ptr[0];
    dst_ptr[1] += src_ptr[1];
    src_ptr += 2;
    dst_ptr += 2;
  }
  if (src_width & 1) {
    dst_ptr[0] += src_ptr[0];
  }
}

void ScaleAddRow_16_C(const uint16_t* src_ptr,
                      uint32_t* dst_ptr,
                      int src_width) {
  int x;
  assert(src_width > 0);
  for (x = 0; x < src_width - 1; x += 2) {
    dst_ptr[0] += src_ptr[0];
    dst_ptr[1] += src_ptr[1];
    src_ptr += 2;
    dst_ptr += 2;
  }
  if (src_width & 1) {
    dst_ptr[0] += src_ptr[0];
  }
}

// ARGB scale row functions

void ScaleARGBRowDown2_C(const uint8_t* src_argb,
                         ptrdiff_t src_stride,
                         uint8_t* dst_argb,
                         int dst_width) {
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = src[1];
    dst[1] = src[3];
    src += 4;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[1];
  }
}

void ScaleARGBRowDown2Linear_C(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               uint8_t* dst_argb,
                               int dst_width) {
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width; ++x) {
    dst_argb[0] = (src_argb[0] + src_argb[4] + 1) >> 1;
    dst_argb[1] = (src_argb[1] + src_argb[5] + 1) >> 1;
    dst_argb[2] = (src_argb[2] + src_argb[6] + 1) >> 1;
    dst_argb[3] = (src_argb[3] + src_argb[7] + 1) >> 1;
    src_argb += 8;
    dst_argb += 4;
  }
}

void ScaleARGBRowDown2Box_C(const uint8_t* src_argb,
                            ptrdiff_t src_stride,
                            uint8_t* dst_argb,
                            int dst_width) {
  int x;
  for (x = 0; x < dst_width; ++x) {
    dst_argb[0] = (src_argb[0] + src_argb[4] + src_argb[src_stride] +
                   src_argb[src_stride + 4] + 2) >>
                  2;
    dst_argb[1] = (src_argb[1] + src_argb[5] + src_argb[src_stride + 1] +
                   src_argb[src_stride + 5] + 2) >>
                  2;
    dst_argb[2] = (src_argb[2] + src_argb[6] + src_argb[src_stride + 2] +
                   src_argb[src_stride + 6] + 2) >>
                  2;
    dst_argb[3] = (src_argb[3] + src_argb[7] + src_argb[src_stride + 3] +
                   src_argb[src_stride + 7] + 2) >>
                  2;
    src_argb += 8;
    dst_argb += 4;
  }
}

void ScaleARGBRowDownEven_C(const uint8_t* src_argb,
                            ptrdiff_t src_stride,
                            int src_stepx,
                            uint8_t* dst_argb,
                            int dst_width) {
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  (void)src_stride;
  int x;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = src[0];
    dst[1] = src[src_stepx];
    src += src_stepx * 2;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[0];
  }
}

void ScaleARGBRowDownEvenBox_C(const uint8_t* src_argb,
                               ptrdiff_t src_stride,
                               int src_stepx,
                               uint8_t* dst_argb,
                               int dst_width) {
  int x;
  for (x = 0; x < dst_width; ++x) {
    dst_argb[0] = (src_argb[0] + src_argb[4] + src_argb[src_stride] +
                   src_argb[src_stride + 4] + 2) >>
                  2;
    dst_argb[1] = (src_argb[1] + src_argb[5] + src_argb[src_stride + 1] +
                   src_argb[src_stride + 5] + 2) >>
                  2;
    dst_argb[2] = (src_argb[2] + src_argb[6] + src_argb[src_stride + 2] +
                   src_argb[src_stride + 6] + 2) >>
                  2;
    dst_argb[3] = (src_argb[3] + src_argb[7] + src_argb[src_stride + 3] +
                   src_argb[src_stride + 7] + 2) >>
                  2;
    src_argb += src_stepx * 4;
    dst_argb += 4;
  }
}

// Scales a single row of pixels using point sampling.
void ScaleARGBCols_C(uint8_t* dst_argb,
                     const uint8_t* src_argb,
                     int dst_width,
                     int x,
                     int dx) {
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst[0] = src[x >> 16];
    x += dx;
    dst[1] = src[x >> 16];
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[x >> 16];
  }
}

void ScaleARGBCols64_C(uint8_t* dst_argb,
                       const uint8_t* src_argb,
                       int dst_width,
                       int x32,
                       int dx) {
  int64_t x = (int64_t)(x32);
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst[0] = src[x >> 16];
    x += dx;
    dst[1] = src[x >> 16];
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[x >> 16];
  }
}

// Scales a single row of pixels up by 2x using point sampling.
void ScaleARGBColsUp2_C(uint8_t* dst_argb,
                        const uint8_t* src_argb,
                        int dst_width,
                        int x,
                        int dx) {
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  int j;
  (void)x;
  (void)dx;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst[1] = dst[0] = src[0];
    src += 1;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[0];
  }
}

// TODO(fbarchard): Replace 0x7f ^ f with 128-f.  bug=607.
// Mimics SSSE3 blender
#define BLENDER1(a, b, f) ((a) * (0x7f ^ f) + (b)*f) >> 7
#define BLENDERC(a, b, f, s) \
  (uint32_t)(BLENDER1(((a) >> s) & 255, ((b) >> s) & 255, f) << s)
#define BLENDER(a, b, f)                                                 \
  BLENDERC(a, b, f, 24) | BLENDERC(a, b, f, 16) | BLENDERC(a, b, f, 8) | \
      BLENDERC(a, b, f, 0)

void ScaleARGBFilterCols_C(uint8_t* dst_argb,
                           const uint8_t* src_argb,
                           int dst_width,
                           int x,
                           int dx) {
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint32_t a = src[xi];
    uint32_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
    x += dx;
    xi = x >> 16;
    xf = (x >> 9) & 0x7f;
    a = src[xi];
    b = src[xi + 1];
    dst[1] = BLENDER(a, b, xf);
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    int xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint32_t a = src[xi];
    uint32_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
  }
}

void ScaleARGBFilterCols64_C(uint8_t* dst_argb,
                             const uint8_t* src_argb,
                             int dst_width,
                             int x32,
                             int dx) {
  int64_t x = (int64_t)(x32);
  const uint32_t* src = (const uint32_t*)(src_argb);
  uint32_t* dst = (uint32_t*)(dst_argb);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int64_t xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint32_t a = src[xi];
    uint32_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
    x += dx;
    xi = x >> 16;
    xf = (x >> 9) & 0x7f;
    a = src[xi];
    b = src[xi + 1];
    dst[1] = BLENDER(a, b, xf);
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    int64_t xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint32_t a = src[xi];
    uint32_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
  }
}
#undef BLENDER1
#undef BLENDERC
#undef BLENDER

// UV scale row functions
// same as ARGB but 2 channels

void ScaleUVRowDown2_C(const uint8_t* src_uv,
                       ptrdiff_t src_stride,
                       uint8_t* dst_uv,
                       int dst_width) {
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width; ++x) {
    dst_uv[0] = src_uv[2];  // Store the 2nd UV
    dst_uv[1] = src_uv[3];
    src_uv += 4;
    dst_uv += 2;
  }
}

void ScaleUVRowDown2Linear_C(const uint8_t* src_uv,
                             ptrdiff_t src_stride,
                             uint8_t* dst_uv,
                             int dst_width) {
  int x;
  (void)src_stride;
  for (x = 0; x < dst_width; ++x) {
    dst_uv[0] = (src_uv[0] + src_uv[2] + 1) >> 1;
    dst_uv[1] = (src_uv[1] + src_uv[3] + 1) >> 1;
    src_uv += 4;
    dst_uv += 2;
  }
}

void ScaleUVRowDown2Box_C(const uint8_t* src_uv,
                          ptrdiff_t src_stride,
                          uint8_t* dst_uv,
                          int dst_width) {
  int x;
  for (x = 0; x < dst_width; ++x) {
    dst_uv[0] = (src_uv[0] + src_uv[2] + src_uv[src_stride] +
                 src_uv[src_stride + 2] + 2) >>
                2;
    dst_uv[1] = (src_uv[1] + src_uv[3] + src_uv[src_stride + 1] +
                 src_uv[src_stride + 3] + 2) >>
                2;
    src_uv += 4;
    dst_uv += 2;
  }
}

void ScaleUVRowDownEven_C(const uint8_t* src_uv,
                          ptrdiff_t src_stride,
                          int src_stepx,
                          uint8_t* dst_uv,
                          int dst_width) {
  const uint16_t* src = (const uint16_t*)(src_uv);
  uint16_t* dst = (uint16_t*)(dst_uv);
  (void)src_stride;
  int x;
  for (x = 0; x < dst_width - 1; x += 2) {
    dst[0] = src[0];
    dst[1] = src[src_stepx];
    src += src_stepx * 2;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[0];
  }
}

void ScaleUVRowDownEvenBox_C(const uint8_t* src_uv,
                             ptrdiff_t src_stride,
                             int src_stepx,
                             uint8_t* dst_uv,
                             int dst_width) {
  int x;
  for (x = 0; x < dst_width; ++x) {
    dst_uv[0] = (src_uv[0] + src_uv[2] + src_uv[src_stride] +
                 src_uv[src_stride + 2] + 2) >>
                2;
    dst_uv[1] = (src_uv[1] + src_uv[3] + src_uv[src_stride + 1] +
                 src_uv[src_stride + 3] + 2) >>
                2;
    src_uv += src_stepx * 2;
    dst_uv += 2;
  }
}

void ScaleUVRowUp2_Linear_C(const uint8_t* src_ptr,
                            uint8_t* dst_ptr,
                            int dst_width) {
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    dst_ptr[4 * x + 0] =
        (src_ptr[2 * x + 0] * 3 + src_ptr[2 * x + 2] * 1 + 2) >> 2;
    dst_ptr[4 * x + 1] =
        (src_ptr[2 * x + 1] * 3 + src_ptr[2 * x + 3] * 1 + 2) >> 2;
    dst_ptr[4 * x + 2] =
        (src_ptr[2 * x + 0] * 1 + src_ptr[2 * x + 2] * 3 + 2) >> 2;
    dst_ptr[4 * x + 3] =
        (src_ptr[2 * x + 1] * 1 + src_ptr[2 * x + 3] * 3 + 2) >> 2;
  }
}

void ScaleUVRowUp2_Bilinear_C(const uint8_t* src_ptr,
                              ptrdiff_t src_stride,
                              uint8_t* dst_ptr,
                              ptrdiff_t dst_stride,
                              int dst_width) {
  const uint8_t* s = src_ptr;
  const uint8_t* t = src_ptr + src_stride;
  uint8_t* d = dst_ptr;
  uint8_t* e = dst_ptr + dst_stride;
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    d[4 * x + 0] = (s[2 * x + 0] * 9 + s[2 * x + 2] * 3 + t[2 * x + 0] * 3 +
                    t[2 * x + 2] * 1 + 8) >>
                   4;
    d[4 * x + 1] = (s[2 * x + 1] * 9 + s[2 * x + 3] * 3 + t[2 * x + 1] * 3 +
                    t[2 * x + 3] * 1 + 8) >>
                   4;
    d[4 * x + 2] = (s[2 * x + 0] * 3 + s[2 * x + 2] * 9 + t[2 * x + 0] * 1 +
                    t[2 * x + 2] * 3 + 8) >>
                   4;
    d[4 * x + 3] = (s[2 * x + 1] * 3 + s[2 * x + 3] * 9 + t[2 * x + 1] * 1 +
                    t[2 * x + 3] * 3 + 8) >>
                   4;
    e[4 * x + 0] = (s[2 * x + 0] * 3 + s[2 * x + 2] * 1 + t[2 * x + 0] * 9 +
                    t[2 * x + 2] * 3 + 8) >>
                   4;
    e[4 * x + 1] = (s[2 * x + 1] * 3 + s[2 * x + 3] * 1 + t[2 * x + 1] * 9 +
                    t[2 * x + 3] * 3 + 8) >>
                   4;
    e[4 * x + 2] = (s[2 * x + 0] * 1 + s[2 * x + 2] * 3 + t[2 * x + 0] * 3 +
                    t[2 * x + 2] * 9 + 8) >>
                   4;
    e[4 * x + 3] = (s[2 * x + 1] * 1 + s[2 * x + 3] * 3 + t[2 * x + 1] * 3 +
                    t[2 * x + 3] * 9 + 8) >>
                   4;
  }
}

void ScaleUVRowUp2_Linear_16_C(const uint16_t* src_ptr,
                               uint16_t* dst_ptr,
                               int dst_width) {
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    dst_ptr[4 * x + 0] =
        (src_ptr[2 * x + 0] * 3 + src_ptr[2 * x + 2] * 1 + 2) >> 2;
    dst_ptr[4 * x + 1] =
        (src_ptr[2 * x + 1] * 3 + src_ptr[2 * x + 3] * 1 + 2) >> 2;
    dst_ptr[4 * x + 2] =
        (src_ptr[2 * x + 0] * 1 + src_ptr[2 * x + 2] * 3 + 2) >> 2;
    dst_ptr[4 * x + 3] =
        (src_ptr[2 * x + 1] * 1 + src_ptr[2 * x + 3] * 3 + 2) >> 2;
  }
}

void ScaleUVRowUp2_Bilinear_16_C(const uint16_t* src_ptr,
                                 ptrdiff_t src_stride,
                                 uint16_t* dst_ptr,
                                 ptrdiff_t dst_stride,
                                 int dst_width) {
  const uint16_t* s = src_ptr;
  const uint16_t* t = src_ptr + src_stride;
  uint16_t* d = dst_ptr;
  uint16_t* e = dst_ptr + dst_stride;
  int src_width = dst_width >> 1;
  int x;
  assert((dst_width % 2 == 0) && (dst_width >= 0));
  for (x = 0; x < src_width; ++x) {
    d[4 * x + 0] = (s[2 * x + 0] * 9 + s[2 * x + 2] * 3 + t[2 * x + 0] * 3 +
                    t[2 * x + 2] * 1 + 8) >>
                   4;
    d[4 * x + 1] = (s[2 * x + 1] * 9 + s[2 * x + 3] * 3 + t[2 * x + 1] * 3 +
                    t[2 * x + 3] * 1 + 8) >>
                   4;
    d[4 * x + 2] = (s[2 * x + 0] * 3 + s[2 * x + 2] * 9 + t[2 * x + 0] * 1 +
                    t[2 * x + 2] * 3 + 8) >>
                   4;
    d[4 * x + 3] = (s[2 * x + 1] * 3 + s[2 * x + 3] * 9 + t[2 * x + 1] * 1 +
                    t[2 * x + 3] * 3 + 8) >>
                   4;
    e[4 * x + 0] = (s[2 * x + 0] * 3 + s[2 * x + 2] * 1 + t[2 * x + 0] * 9 +
                    t[2 * x + 2] * 3 + 8) >>
                   4;
    e[4 * x + 1] = (s[2 * x + 1] * 3 + s[2 * x + 3] * 1 + t[2 * x + 1] * 9 +
                    t[2 * x + 3] * 3 + 8) >>
                   4;
    e[4 * x + 2] = (s[2 * x + 0] * 1 + s[2 * x + 2] * 3 + t[2 * x + 0] * 3 +
                    t[2 * x + 2] * 9 + 8) >>
                   4;
    e[4 * x + 3] = (s[2 * x + 1] * 1 + s[2 * x + 3] * 3 + t[2 * x + 1] * 3 +
                    t[2 * x + 3] * 9 + 8) >>
                   4;
  }
}

// Scales a single row of pixels using point sampling.
void ScaleUVCols_C(uint8_t* dst_uv,
                   const uint8_t* src_uv,
                   int dst_width,
                   int x,
                   int dx) {
  const uint16_t* src = (const uint16_t*)(src_uv);
  uint16_t* dst = (uint16_t*)(dst_uv);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst[0] = src[x >> 16];
    x += dx;
    dst[1] = src[x >> 16];
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[x >> 16];
  }
}

void ScaleUVCols64_C(uint8_t* dst_uv,
                     const uint8_t* src_uv,
                     int dst_width,
                     int x32,
                     int dx) {
  int64_t x = (int64_t)(x32);
  const uint16_t* src = (const uint16_t*)(src_uv);
  uint16_t* dst = (uint16_t*)(dst_uv);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst[0] = src[x >> 16];
    x += dx;
    dst[1] = src[x >> 16];
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[x >> 16];
  }
}

// Scales a single row of pixels up by 2x using point sampling.
void ScaleUVColsUp2_C(uint8_t* dst_uv,
                      const uint8_t* src_uv,
                      int dst_width,
                      int x,
                      int dx) {
  const uint16_t* src = (const uint16_t*)(src_uv);
  uint16_t* dst = (uint16_t*)(dst_uv);
  int j;
  (void)x;
  (void)dx;
  for (j = 0; j < dst_width - 1; j += 2) {
    dst[1] = dst[0] = src[0];
    src += 1;
    dst += 2;
  }
  if (dst_width & 1) {
    dst[0] = src[0];
  }
}

// TODO(fbarchard): Replace 0x7f ^ f with 128-f.  bug=607.
// Mimics SSSE3 blender
#define BLENDER1(a, b, f) ((a) * (0x7f ^ f) + (b)*f) >> 7
#define BLENDERC(a, b, f, s) \
  (uint16_t)(BLENDER1(((a) >> s) & 255, ((b) >> s) & 255, f) << s)
#define BLENDER(a, b, f) BLENDERC(a, b, f, 8) | BLENDERC(a, b, f, 0)

void ScaleUVFilterCols_C(uint8_t* dst_uv,
                         const uint8_t* src_uv,
                         int dst_width,
                         int x,
                         int dx) {
  const uint16_t* src = (const uint16_t*)(src_uv);
  uint16_t* dst = (uint16_t*)(dst_uv);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint16_t a = src[xi];
    uint16_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
    x += dx;
    xi = x >> 16;
    xf = (x >> 9) & 0x7f;
    a = src[xi];
    b = src[xi + 1];
    dst[1] = BLENDER(a, b, xf);
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    int xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint16_t a = src[xi];
    uint16_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
  }
}

void ScaleUVFilterCols64_C(uint8_t* dst_uv,
                           const uint8_t* src_uv,
                           int dst_width,
                           int x32,
                           int dx) {
  int64_t x = (int64_t)(x32);
  const uint16_t* src = (const uint16_t*)(src_uv);
  uint16_t* dst = (uint16_t*)(dst_uv);
  int j;
  for (j = 0; j < dst_width - 1; j += 2) {
    int64_t xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint16_t a = src[xi];
    uint16_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
    x += dx;
    xi = x >> 16;
    xf = (x >> 9) & 0x7f;
    a = src[xi];
    b = src[xi + 1];
    dst[1] = BLENDER(a, b, xf);
    x += dx;
    dst += 2;
  }
  if (dst_width & 1) {
    int64_t xi = x >> 16;
    int xf = (x >> 9) & 0x7f;
    uint16_t a = src[xi];
    uint16_t b = src[xi + 1];
    dst[0] = BLENDER(a, b, xf);
  }
}
#undef BLENDER1
#undef BLENDERC
#undef BLENDER

// Scale plane vertically with bilinear interpolation.
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
                        int bpp,  // bytes per pixel. 4 for ARGB.
                        enum FilterMode filtering) {
  // TODO(fbarchard): Allow higher bpp.
  int dst_width_bytes = dst_width * bpp;
  void (*InterpolateRow)(uint8_t* dst_argb, const uint8_t* src_argb,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_C;
  const int max_y = (src_height > 1) ? ((src_height - 1) << 16) - 1 : 0;
  int j;
  assert(bpp >= 1 && bpp <= 4);
  assert(src_height != 0);
  assert(dst_width > 0);
  assert(dst_height > 0);
  src_argb += (x >> 16) * bpp;
#if defined(HAS_INTERPOLATEROW_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_Any_SSSE3;
    if (IS_ALIGNED(dst_width_bytes, 16)) {
      InterpolateRow = InterpolateRow_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_Any_AVX2;
    if (IS_ALIGNED(dst_width_bytes, 32)) {
      InterpolateRow = InterpolateRow_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_Any_NEON;
    if (IS_ALIGNED(dst_width_bytes, 16)) {
      InterpolateRow = InterpolateRow_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    InterpolateRow = InterpolateRow_SME;
  }
#endif
#if defined(HAS_INTERPOLATEROW_MSA)
  if (TestCpuFlag(kCpuHasMSA)) {
    InterpolateRow = InterpolateRow_Any_MSA;
    if (IS_ALIGNED(dst_width_bytes, 32)) {
      InterpolateRow = InterpolateRow_MSA;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_LSX)
  if (TestCpuFlag(kCpuHasLSX)) {
    InterpolateRow = InterpolateRow_Any_LSX;
    if (IS_ALIGNED(dst_width_bytes, 32)) {
      InterpolateRow = InterpolateRow_LSX;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_RVV)
  if (TestCpuFlag(kCpuHasRVV)) {
    InterpolateRow = InterpolateRow_RVV;
  }
#endif

  for (j = 0; j < dst_height; ++j) {
    int yi;
    int yf;
    if (y > max_y) {
      y = max_y;
    }
    yi = y >> 16;
    yf = filtering ? ((y >> 8) & 255) : 0;
    InterpolateRow(dst_argb, src_argb + yi * src_stride, src_stride,
                   dst_width_bytes, yf);
    dst_argb += dst_stride;
    y += dy;
  }
}

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
                           int wpp, /* words per pixel. normally 1 */
                           enum FilterMode filtering) {
  // TODO(fbarchard): Allow higher wpp.
  int dst_width_words = dst_width * wpp;
  void (*InterpolateRow)(uint16_t* dst_argb, const uint16_t* src_argb,
                         ptrdiff_t src_stride, int dst_width,
                         int source_y_fraction) = InterpolateRow_16_C;
  const int max_y = (src_height > 1) ? ((src_height - 1) << 16) - 1 : 0;
  int j;
  assert(wpp >= 1 && wpp <= 2);
  assert(src_height != 0);
  assert(dst_width > 0);
  assert(dst_height > 0);
  src_argb += (x >> 16) * wpp;
#if defined(HAS_INTERPOLATEROW_16_SSE2)
  if (TestCpuFlag(kCpuHasSSE2)) {
    InterpolateRow = InterpolateRow_16_Any_SSE2;
    if (IS_ALIGNED(dst_width_words, 16)) {
      InterpolateRow = InterpolateRow_16_SSE2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_SSSE3)
  if (TestCpuFlag(kCpuHasSSSE3)) {
    InterpolateRow = InterpolateRow_16_Any_SSSE3;
    if (IS_ALIGNED(dst_width_words, 16)) {
      InterpolateRow = InterpolateRow_16_SSSE3;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow = InterpolateRow_16_Any_AVX2;
    if (IS_ALIGNED(dst_width_words, 32)) {
      InterpolateRow = InterpolateRow_16_AVX2;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow = InterpolateRow_16_Any_NEON;
    if (IS_ALIGNED(dst_width_words, 8)) {
      InterpolateRow = InterpolateRow_16_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    InterpolateRow = InterpolateRow_16_SME;
  }
#endif
  for (j = 0; j < dst_height; ++j) {
    int yi;
    int yf;
    if (y > max_y) {
      y = max_y;
    }
    yi = y >> 16;
    yf = filtering ? ((y >> 8) & 255) : 0;
    InterpolateRow(dst_argb, src_argb + yi * src_stride, src_stride,
                   dst_width_words, yf);
    dst_argb += dst_stride;
    y += dy;
  }
}

// Use scale to convert lsb formats to msb, depending how many bits there are:
// 32768 = 9 bits
// 16384 = 10 bits
// 4096 = 12 bits
// 256 = 16 bits
// TODO(fbarchard): change scale to bits
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
                              int wpp, /* words per pixel. normally 1 */
                              int scale,
                              enum FilterMode filtering) {
  // TODO(fbarchard): Allow higher wpp.
  int dst_width_words = dst_width * wpp;
  // TODO(https://crbug.com/libyuv/931): Add NEON 32 bit and AVX2 versions.
  void (*InterpolateRow_16To8)(uint8_t* dst_argb, const uint16_t* src_argb,
                               ptrdiff_t src_stride, int scale, int dst_width,
                               int source_y_fraction) = InterpolateRow_16To8_C;
  const int max_y = (src_height > 1) ? ((src_height - 1) << 16) - 1 : 0;
  int j;
  assert(wpp >= 1 && wpp <= 2);
  assert(src_height != 0);
  assert(dst_width > 0);
  assert(dst_height > 0);
  src_argb += (x >> 16) * wpp;

#if defined(HAS_INTERPOLATEROW_16TO8_NEON)
  if (TestCpuFlag(kCpuHasNEON)) {
    InterpolateRow_16To8 = InterpolateRow_16To8_Any_NEON;
    if (IS_ALIGNED(dst_width, 8)) {
      InterpolateRow_16To8 = InterpolateRow_16To8_NEON;
    }
  }
#endif
#if defined(HAS_INTERPOLATEROW_16TO8_SME)
  if (TestCpuFlag(kCpuHasSME)) {
    InterpolateRow_16To8 = InterpolateRow_16To8_SME;
  }
#endif
#if defined(HAS_INTERPOLATEROW_16TO8_AVX2)
  if (TestCpuFlag(kCpuHasAVX2)) {
    InterpolateRow_16To8 = InterpolateRow_16To8_Any_AVX2;
    if (IS_ALIGNED(dst_width, 32)) {
      InterpolateRow_16To8 = InterpolateRow_16To8_AVX2;
    }
  }
#endif
  for (j = 0; j < dst_height; ++j) {
    int yi;
    int yf;
    if (y > max_y) {
      y = max_y;
    }
    yi = y >> 16;
    yf = filtering ? ((y >> 8) & 255) : 0;
    InterpolateRow_16To8(dst_argb, src_argb + yi * src_stride, src_stride,
                         scale, dst_width_words, yf);
    dst_argb += dst_stride;
    y += dy;
  }
}

// Simplify the filtering based on scale factors.
enum FilterMode ScaleFilterReduce(int src_width,
                                  int src_height,
                                  int dst_width,
                                  int dst_height,
                                  enum FilterMode filtering) {
  if (src_width < 0) {
    src_width = -src_width;
  }
  if (src_height < 0) {
    src_height = -src_height;
  }
  if (filtering == kFilterBox) {
    // If scaling either axis to 0.5 or larger, switch from Box to Bilinear.
    if (dst_width * 2 >= src_width || dst_height * 2 >= src_height) {
      filtering = kFilterBilinear;
    }
  }
  if (filtering == kFilterBilinear) {
    if (src_height == 1) {
      filtering = kFilterLinear;
    }
    // TODO(fbarchard): Detect any odd scale factor and reduce to Linear.
    if (dst_height == src_height || dst_height * 3 == src_height) {
      filtering = kFilterLinear;
    }
    // TODO(fbarchard): Remove 1 pixel wide filter restriction, which is to
    // avoid reading 2 pixels horizontally that causes memory exception.
    if (src_width == 1) {
      filtering = kFilterNone;
    }
  }
  if (filtering == kFilterLinear) {
    if (src_width == 1) {
      filtering = kFilterNone;
    }
    // TODO(fbarchard): Detect any odd scale factor and reduce to None.
    if (dst_width == src_width || dst_width * 3 == src_width) {
      filtering = kFilterNone;
    }
  }
  return filtering;
}

// Divide num by div and return as 16.16 fixed point result.
int FixedDiv_C(int num, int div) {
  return (int)(((int64_t)(num) << 16) / div);
}

// Divide num - 1 by div - 1 and return as 16.16 fixed point result.
int FixedDiv1_C(int num, int div) {
  return (int)((((int64_t)(num) << 16) - 0x00010001) / (div - 1));
}

#define CENTERSTART(dx, s) (dx < 0) ? -((-dx >> 1) + s) : ((dx >> 1) + s)

// Compute slope values for stepping.
void ScaleSlope(int src_width,
                int src_height,
                int dst_width,
                int dst_height,
                enum FilterMode filtering,
                int* x,
                int* y,
                int* dx,
                int* dy) {
  assert(x != NULL);
  assert(y != NULL);
  assert(dx != NULL);
  assert(dy != NULL);
  assert(src_width != 0);
  assert(src_height != 0);
  assert(dst_width > 0);
  assert(dst_height > 0);
  // Check for 1 pixel and avoid FixedDiv overflow.
  if (dst_width == 1 && src_width >= 32768) {
    dst_width = src_width;
  }
  if (dst_height == 1 && src_height >= 32768) {
    dst_height = src_height;
  }
  if (filtering == kFilterBox) {
    // Scale step for point sampling duplicates all pixels equally.
    *dx = FixedDiv(Abs(src_width), dst_width);
    *dy = FixedDiv(src_height, dst_height);
    *x = 0;
    *y = 0;
  } else if (filtering == kFilterBilinear) {
    // Scale step for bilinear sampling renders last pixel once for upsample.
    if (dst_width <= Abs(src_width)) {
      *dx = FixedDiv(Abs(src_width), dst_width);
      *x = CENTERSTART(*dx, -32768);  // Subtract 0.5 (32768) to center filter.
    } else if (src_width > 1 && dst_width > 1) {
      *dx = FixedDiv1(Abs(src_width), dst_width);
      *x = 0;
    }
    if (dst_height <= src_height) {
      *dy = FixedDiv(src_height, dst_height);
      *y = CENTERSTART(*dy, -32768);  // Subtract 0.5 (32768) to center filter.
    } else if (src_height > 1 && dst_height > 1) {
      *dy = FixedDiv1(src_height, dst_height);
      *y = 0;
    }
  } else if (filtering == kFilterLinear) {
    // Scale step for bilinear sampling renders last pixel once for upsample.
    if (dst_width <= Abs(src_width)) {
      *dx = FixedDiv(Abs(src_width), dst_width);
      *x = CENTERSTART(*dx, -32768);  // Subtract 0.5 (32768) to center filter.
    } else if (src_width > 1 && dst_width > 1) {
      *dx = FixedDiv1(Abs(src_width), dst_width);
      *x = 0;
    }
    *dy = FixedDiv(src_height, dst_height);
    *y = *dy >> 1;
  } else {
    // Scale step for point sampling duplicates all pixels equally.
    *dx = FixedDiv(Abs(src_width), dst_width);
    *dy = FixedDiv(src_height, dst_height);
    *x = CENTERSTART(*dx, 0);
    *y = CENTERSTART(*dy, 0);
  }
  // Negative src_width means horizontally mirror.
  if (src_width < 0) {
    *x += (dst_width - 1) * *dx;
    *dx = -*dx;
    // src_width = -src_width;   // Caller must do this.
  }
}
#undef CENTERSTART

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
