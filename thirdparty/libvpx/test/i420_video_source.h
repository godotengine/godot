/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TEST_I420_VIDEO_SOURCE_H_
#define VPX_TEST_I420_VIDEO_SOURCE_H_
#include <cstdio>
#include <cstdlib>
#include <string>

#include "test/yuv_video_source.h"

namespace libvpx_test {

// This class extends VideoSource to allow parsing of raw yv12
// so that we can do actual file encodes.
class I420VideoSource : public YUVVideoSource {
 public:
  I420VideoSource(const std::string &file_name, unsigned int width,
                  unsigned int height, int rate_numerator, int rate_denominator,
                  unsigned int start, int limit)
      : YUVVideoSource(file_name, VPX_IMG_FMT_I420, width, height,
                       rate_numerator, rate_denominator, start, limit) {}
};

}  // namespace libvpx_test

#endif  // VPX_TEST_I420_VIDEO_SOURCE_H_
