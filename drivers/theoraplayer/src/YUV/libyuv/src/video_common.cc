/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "libyuv/video_common.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#define ARRAY_SIZE(x) (int)(sizeof(x) / sizeof(x[0]))

struct FourCCAliasEntry {
  uint32 alias;
  uint32 canonical;
};

static const struct FourCCAliasEntry kFourCCAliases[] = {
  {FOURCC_IYUV, FOURCC_I420},
  {FOURCC_YU16, FOURCC_I422},
  {FOURCC_YU24, FOURCC_I444},
  {FOURCC_YUYV, FOURCC_YUY2},
  {FOURCC_YUVS, FOURCC_YUY2},  // kCMPixelFormat_422YpCbCr8_yuvs
  {FOURCC_HDYC, FOURCC_UYVY},
  {FOURCC_2VUY, FOURCC_UYVY},  // kCMPixelFormat_422YpCbCr8
  {FOURCC_JPEG, FOURCC_MJPG},  // Note: JPEG has DHT while MJPG does not.
  {FOURCC_DMB1, FOURCC_MJPG},
  {FOURCC_BA81, FOURCC_BGGR},
  {FOURCC_RGB3, FOURCC_RAW },
  {FOURCC_BGR3, FOURCC_24BG},
  {FOURCC_CM32, FOURCC_BGRA},  // kCMPixelFormat_32ARGB
  {FOURCC_CM24, FOURCC_RAW },  // kCMPixelFormat_24RGB
  {FOURCC_L555, FOURCC_RGBO},  // kCMPixelFormat_16LE555
  {FOURCC_L565, FOURCC_RGBP},  // kCMPixelFormat_16LE565
  {FOURCC_5551, FOURCC_RGBO},  // kCMPixelFormat_16LE5551
};
// TODO(fbarchard): Consider mapping kCMPixelFormat_32BGRA to FOURCC_ARGB.
//  {FOURCC_BGRA, FOURCC_ARGB},  // kCMPixelFormat_32BGRA

LIBYUV_API
uint32 CanonicalFourCC(uint32 fourcc) {
  int i;
  for (i = 0; i < ARRAY_SIZE(kFourCCAliases); ++i) {
    if (kFourCCAliases[i].alias == fourcc) {
      return kFourCCAliases[i].canonical;
    }
  }
  // Not an alias, so return it as-is.
  return fourcc;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

