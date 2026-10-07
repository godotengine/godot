// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef LIBWEBM_COMMON_VIDEO_FRAME_H_
#define LIBWEBM_COMMON_VIDEO_FRAME_H_

#include <cstdint>
#include <memory>

namespace libwebm {

// VideoFrame is a storage class for compressed video frames.
class VideoFrame {
 public:
  enum Codec { kVP8, kVP9 };
  struct Buffer {
    Buffer() = default;
    ~Buffer() = default;

    // Resets |data| to be of size |new_length| bytes, sets |capacity| to
    // |new_length|, sets |length| to 0 (aka empty). Returns true for success.
    bool Init(std::size_t new_length);

    std::unique_ptr<std::uint8_t[]> data;
    std::size_t length = 0;
    std::size_t capacity = 0;
  };

  VideoFrame() = default;
  ~VideoFrame() = default;
  VideoFrame(std::int64_t pts_in_nanoseconds, Codec vpx_codec)
      : nanosecond_pts_(pts_in_nanoseconds), codec_(vpx_codec) {}
  VideoFrame(bool keyframe, std::int64_t pts_in_nanoseconds, Codec vpx_codec)
      : keyframe_(keyframe),
        nanosecond_pts_(pts_in_nanoseconds),
        codec_(vpx_codec) {}
  bool Init(std::size_t length);
  bool Init(std::size_t length, std::int64_t nano_pts, Codec codec);

  // Updates actual length of data stored in |buffer_.data| when it's been
  // written via the raw pointer returned from buffer_.data.get().
  // Returns false when buffer_.data.get() return nullptr and/or when
  // |length| > |buffer_.length|. Returns true otherwise.
  bool SetBufferLength(std::size_t length);

  // Accessors.
  const Buffer& buffer() const { return buffer_; }
  bool keyframe() const { return keyframe_; }
  std::int64_t nanosecond_pts() const { return nanosecond_pts_; }
  Codec codec() const { return codec_; }

  // Mutators.
  void set_nanosecond_pts(std::int64_t nano_pts) { nanosecond_pts_ = nano_pts; }

 private:
  Buffer buffer_;
  bool keyframe_ = false;
  std::int64_t nanosecond_pts_ = 0;
  Codec codec_ = kVP9;
};

}  // namespace libwebm

#endif  // LIBWEBM_COMMON_VIDEO_FRAME_H_