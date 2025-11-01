/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

// Common definitions for video, including fourcc and VideoFormat.

#ifndef INCLUDE_LIBYUV_VIDEO_COMMON_H_
#define INCLUDE_LIBYUV_VIDEO_COMMON_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

//////////////////////////////////////////////////////////////////////////////
// Definition of FourCC codes
//////////////////////////////////////////////////////////////////////////////

// Convert four characters to a FourCC code.
// Needs to be a macro otherwise the OS X compiler complains when the kFormat*
// constants are used in a switch.
#ifdef __cplusplus
#define FOURCC(a, b, c, d)                                        \
  ((static_cast<uint32_t>(a)) | (static_cast<uint32_t>(b) << 8) | \
   (static_cast<uint32_t>(c) << 16) | /* NOLINT */                \
   (static_cast<uint32_t>(d) << 24))  /* NOLINT */
#else
#define FOURCC(a, b, c, d)                                     \
  (((uint32_t)(a)) | ((uint32_t)(b) << 8) |       /* NOLINT */ \
   ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24)) /* NOLINT */
#endif

// Some pages discussing FourCC codes:
//   http://www.fourcc.org/yuv.php
//   http://v4l2spec.bytesex.org/spec/book1.htm
//   http://developer.apple.com/quicktime/icefloe/dispatch020.html
//   http://msdn.microsoft.com/library/windows/desktop/dd206750.aspx#nv12
//   http://people.xiph.org/~xiphmont/containers/nut/nut4cc.txt

// FourCC codes grouped according to implementation efficiency.
// Primary formats should convert in 1 efficient step.
// Secondary formats are converted in 2 steps.
// Auxilliary formats call primary converters.
enum FourCC {
  // 10 Primary YUV formats: 5 planar, 2 biplanar, 2 packed.
  FOURCC_I420 = FOURCC('I', '4', '2', '0'),
  FOURCC_I422 = FOURCC('I', '4', '2', '2'),
  FOURCC_I444 = FOURCC('I', '4', '4', '4'),
  FOURCC_I400 = FOURCC('I', '4', '0', '0'),
  FOURCC_NV21 = FOURCC('N', 'V', '2', '1'),
  FOURCC_NV12 = FOURCC('N', 'V', '1', '2'),
  FOURCC_YUY2 = FOURCC('Y', 'U', 'Y', '2'),
  FOURCC_UYVY = FOURCC('U', 'Y', 'V', 'Y'),
  FOURCC_I010 = FOURCC('I', '0', '1', '0'),  // bt.601 10 bit 420
  FOURCC_I210 = FOURCC('I', '2', '1', '0'),  // bt.601 10 bit 422

  // 1 Secondary YUV format: row biplanar.  deprecated.
  FOURCC_M420 = FOURCC('M', '4', '2', '0'),

  // 13 Primary RGB formats: 4 32 bpp, 2 24 bpp, 3 16 bpp, 1 10 bpc 2 64 bpp
  FOURCC_ARGB = FOURCC('A', 'R', 'G', 'B'),
  FOURCC_BGRA = FOURCC('B', 'G', 'R', 'A'),
  FOURCC_ABGR = FOURCC('A', 'B', 'G', 'R'),
  FOURCC_AR30 = FOURCC('A', 'R', '3', '0'),  // 10 bit per channel. 2101010.
  FOURCC_AB30 = FOURCC('A', 'B', '3', '0'),  // ABGR version of 10 bit
  FOURCC_AR64 = FOURCC('A', 'R', '6', '4'),  // 16 bit per channel.
  FOURCC_AB64 = FOURCC('A', 'B', '6', '4'),  // ABGR version of 16 bit
  FOURCC_24BG = FOURCC('2', '4', 'B', 'G'),
  FOURCC_RAW = FOURCC('r', 'a', 'w', ' '),
  FOURCC_RGBA = FOURCC('R', 'G', 'B', 'A'),
  FOURCC_RGBP = FOURCC('R', 'G', 'B', 'P'),  // rgb565 LE.
  FOURCC_RGBO = FOURCC('R', 'G', 'B', 'O'),  // argb1555 LE.
  FOURCC_R444 = FOURCC('R', '4', '4', '4'),  // argb4444 LE.

  // 1 Primary Compressed YUV format.
  FOURCC_MJPG = FOURCC('M', 'J', 'P', 'G'),

  // 14 Auxiliary YUV variations: 3 with U and V planes are swapped, 1 Alias.
  FOURCC_YV12 = FOURCC('Y', 'V', '1', '2'),
  FOURCC_YV16 = FOURCC('Y', 'V', '1', '6'),
  FOURCC_YV24 = FOURCC('Y', 'V', '2', '4'),
  FOURCC_YU12 = FOURCC('Y', 'U', '1', '2'),  // Linux version of I420.
  FOURCC_J420 =
      FOURCC('J', '4', '2', '0'),  // jpeg (bt.601 full), unofficial fourcc
  FOURCC_J422 =
      FOURCC('J', '4', '2', '2'),  // jpeg (bt.601 full), unofficial fourcc
  FOURCC_J444 =
      FOURCC('J', '4', '4', '4'),  // jpeg (bt.601 full), unofficial fourcc
  FOURCC_J400 =
      FOURCC('J', '4', '0', '0'),  // jpeg (bt.601 full), unofficial fourcc
  FOURCC_F420 = FOURCC('F', '4', '2', '0'),  // bt.709 full, unofficial fourcc
  FOURCC_F422 = FOURCC('F', '4', '2', '2'),  // bt.709 full, unofficial fourcc
  FOURCC_F444 = FOURCC('F', '4', '4', '4'),  // bt.709 full, unofficial fourcc
  FOURCC_H420 = FOURCC('H', '4', '2', '0'),  // bt.709, unofficial fourcc
  FOURCC_H422 = FOURCC('H', '4', '2', '2'),  // bt.709, unofficial fourcc
  FOURCC_H444 = FOURCC('H', '4', '4', '4'),  // bt.709, unofficial fourcc
  FOURCC_U420 = FOURCC('U', '4', '2', '0'),  // bt.2020, unofficial fourcc
  FOURCC_U422 = FOURCC('U', '4', '2', '2'),  // bt.2020, unofficial fourcc
  FOURCC_U444 = FOURCC('U', '4', '4', '4'),  // bt.2020, unofficial fourcc
  FOURCC_F010 = FOURCC('F', '0', '1', '0'),  // bt.709 full range 10 bit 420
  FOURCC_H010 = FOURCC('H', '0', '1', '0'),  // bt.709 10 bit 420
  FOURCC_U010 = FOURCC('U', '0', '1', '0'),  // bt.2020 10 bit 420
  FOURCC_F210 = FOURCC('F', '2', '1', '0'),  // bt.709 full range 10 bit 422
  FOURCC_H210 = FOURCC('H', '2', '1', '0'),  // bt.709 10 bit 422
  FOURCC_U210 = FOURCC('U', '2', '1', '0'),  // bt.2020 10 bit 422
  FOURCC_P010 = FOURCC('P', '0', '1', '0'),
  FOURCC_P210 = FOURCC('P', '2', '1', '0'),

  // 14 Auxiliary aliases.  CanonicalFourCC() maps these to canonical fourcc.
  FOURCC_IYUV = FOURCC('I', 'Y', 'U', 'V'),  // Alias for I420.
  FOURCC_YU16 = FOURCC('Y', 'U', '1', '6'),  // Alias for I422.
  FOURCC_YU24 = FOURCC('Y', 'U', '2', '4'),  // Alias for I444.
  FOURCC_YUYV = FOURCC('Y', 'U', 'Y', 'V'),  // Alias for YUY2.
  FOURCC_YUVS = FOURCC('y', 'u', 'v', 's'),  // Alias for YUY2 on Mac.
  FOURCC_HDYC = FOURCC('H', 'D', 'Y', 'C'),  // Alias for UYVY.
  FOURCC_2VUY = FOURCC('2', 'v', 'u', 'y'),  // Alias for UYVY on Mac.
  FOURCC_JPEG = FOURCC('J', 'P', 'E', 'G'),  // Alias for MJPG.
  FOURCC_DMB1 = FOURCC('d', 'm', 'b', '1'),  // Alias for MJPG on Mac.
  FOURCC_BA81 = FOURCC('B', 'A', '8', '1'),  // Alias for BGGR.
  FOURCC_RGB3 = FOURCC('R', 'G', 'B', '3'),  // Alias for RAW.
  FOURCC_BGR3 = FOURCC('B', 'G', 'R', '3'),  // Alias for 24BG.
  FOURCC_CM32 = FOURCC(0, 0, 0, 32),  // Alias for BGRA kCMPixelFormat_32ARGB
  FOURCC_CM24 = FOURCC(0, 0, 0, 24),  // Alias for RAW kCMPixelFormat_24RGB
  FOURCC_L555 = FOURCC('L', '5', '5', '5'),  // Alias for RGBO.
  FOURCC_L565 = FOURCC('L', '5', '6', '5'),  // Alias for RGBP.
  FOURCC_5551 = FOURCC('5', '5', '5', '1'),  // Alias for RGBO.

  // deprecated formats.  Not supported, but defined for backward compatibility.
  FOURCC_I411 = FOURCC('I', '4', '1', '1'),
  FOURCC_Q420 = FOURCC('Q', '4', '2', '0'),
  FOURCC_RGGB = FOURCC('R', 'G', 'G', 'B'),
  FOURCC_BGGR = FOURCC('B', 'G', 'G', 'R'),
  FOURCC_GRBG = FOURCC('G', 'R', 'B', 'G'),
  FOURCC_GBRG = FOURCC('G', 'B', 'R', 'G'),
  FOURCC_H264 = FOURCC('H', '2', '6', '4'),

  // Match any fourcc.
  FOURCC_ANY = -1,
};

enum FourCCBpp {
  // Canonical fourcc codes used in our code.
  FOURCC_BPP_I420 = 12,
  FOURCC_BPP_I422 = 16,
  FOURCC_BPP_I444 = 24,
  FOURCC_BPP_I411 = 12,
  FOURCC_BPP_I400 = 8,
  FOURCC_BPP_NV21 = 12,
  FOURCC_BPP_NV12 = 12,
  FOURCC_BPP_YUY2 = 16,
  FOURCC_BPP_UYVY = 16,
  FOURCC_BPP_M420 = 12,  // deprecated
  FOURCC_BPP_Q420 = 12,
  FOURCC_BPP_ARGB = 32,
  FOURCC_BPP_BGRA = 32,
  FOURCC_BPP_ABGR = 32,
  FOURCC_BPP_RGBA = 32,
  FOURCC_BPP_AR30 = 32,
  FOURCC_BPP_AB30 = 32,
  FOURCC_BPP_AR64 = 64,
  FOURCC_BPP_AB64 = 64,
  FOURCC_BPP_24BG = 24,
  FOURCC_BPP_RAW = 24,
  FOURCC_BPP_RGBP = 16,
  FOURCC_BPP_RGBO = 16,
  FOURCC_BPP_R444 = 16,
  FOURCC_BPP_RGGB = 8,
  FOURCC_BPP_BGGR = 8,
  FOURCC_BPP_GRBG = 8,
  FOURCC_BPP_GBRG = 8,
  FOURCC_BPP_YV12 = 12,
  FOURCC_BPP_YV16 = 16,
  FOURCC_BPP_YV24 = 24,
  FOURCC_BPP_YU12 = 12,
  FOURCC_BPP_J420 = 12,
  FOURCC_BPP_J400 = 8,
  FOURCC_BPP_H420 = 12,
  FOURCC_BPP_H422 = 16,
  FOURCC_BPP_I010 = 15,
  FOURCC_BPP_I210 = 20,
  FOURCC_BPP_H010 = 15,
  FOURCC_BPP_H210 = 20,
  FOURCC_BPP_P010 = 15,
  FOURCC_BPP_P210 = 20,
  FOURCC_BPP_MJPG = 0,  // 0 means unknown.
  FOURCC_BPP_H264 = 0,
  FOURCC_BPP_IYUV = 12,
  FOURCC_BPP_YU16 = 16,
  FOURCC_BPP_YU24 = 24,
  FOURCC_BPP_YUYV = 16,
  FOURCC_BPP_YUVS = 16,
  FOURCC_BPP_HDYC = 16,
  FOURCC_BPP_2VUY = 16,
  FOURCC_BPP_JPEG = 1,
  FOURCC_BPP_DMB1 = 1,
  FOURCC_BPP_BA81 = 8,
  FOURCC_BPP_RGB3 = 24,
  FOURCC_BPP_BGR3 = 24,
  FOURCC_BPP_CM32 = 32,
  FOURCC_BPP_CM24 = 24,

  // Match any fourcc.
  FOURCC_BPP_ANY = 0,  // 0 means unknown.
};

// Converts fourcc aliases into canonical ones.
LIBYUV_API uint32_t CanonicalFourCC(uint32_t fourcc);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_VIDEO_COMMON_H_
