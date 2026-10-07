// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "common/video_frame.h"

#include <cstdio>

namespace libwebm {

bool VideoFrame::Buffer::Init(std::size_t new_length) {
  capacity = 0;
  length = 0;
  data.reset(new std::uint8_t[new_length]);

  if (data == nullptr) {
    fprintf(stderr, "VideoFrame: Out of memory.");
    return false;
  }

  capacity = new_length;
  length = 0;
  return true;
}

bool VideoFrame::Init(std::size_t length) { return buffer_.Init(length); }

bool VideoFrame::Init(std::size_t length, std::int64_t nano_pts, Codec codec) {
  nanosecond_pts_ = nano_pts;
  codec_ = codec;
  return Init(length);
}

bool VideoFrame::SetBufferLength(std::size_t length) {
  if (length > buffer_.capacity || buffer_.data.get() == nullptr)
    return false;

  buffer_.length = length;
  return true;
}

}  // namespace libwebm
