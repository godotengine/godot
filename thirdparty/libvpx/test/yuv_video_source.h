/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TEST_YUV_VIDEO_SOURCE_H_
#define VPX_TEST_YUV_VIDEO_SOURCE_H_

#include <cstdio>
#include <cstdlib>
#include <string>

#include "test/video_source.h"
#include "vpx/vpx_image.h"

namespace libvpx_test {

// This class extends VideoSource to allow parsing of raw YUV
// formats of various color sampling and bit-depths so that we can
// do actual file encodes.
class YUVVideoSource : public VideoSource {
 public:
  YUVVideoSource(const std::string &file_name, vpx_img_fmt format,
                 unsigned int width, unsigned int height, int rate_numerator,
                 int rate_denominator, unsigned int start, int limit)
      : file_name_(file_name), input_file_(nullptr), img_(nullptr),
        start_(start), limit_(limit), frame_(0), width_(0), height_(0),
        format_(VPX_IMG_FMT_NONE), framerate_numerator_(rate_numerator),
        framerate_denominator_(rate_denominator) {
    // This initializes format_, raw_size_, width_, height_ and allocates img.
    SetSize(width, height, format);
  }

  ~YUVVideoSource() override {
    vpx_img_free(img_);
    if (input_file_) fclose(input_file_);
  }

  void Begin() override {
    if (input_file_) fclose(input_file_);
    input_file_ = OpenTestDataFile(file_name_);
    ASSERT_NE(input_file_, nullptr)
        << "Input file open failed. Filename: " << file_name_;
    if (start_) {
      fseek(input_file_, static_cast<unsigned>(raw_size_) * start_, SEEK_SET);
    }

    frame_ = start_;
    FillFrame();
  }

  void Next() override {
    ++frame_;
    FillFrame();
  }

  vpx_image_t *img() const override {
    return (frame_ < limit_) ? img_ : nullptr;
  }

  // Models a stream where Timebase = 1/FPS, so pts == frame.
  vpx_codec_pts_t pts() const override { return frame_; }

  unsigned long duration() const override { return 1; }

  vpx_rational_t timebase() const override {
    const vpx_rational_t t = { framerate_denominator_, framerate_numerator_ };
    return t;
  }

  unsigned int frame() const override { return frame_; }

  unsigned int limit() const override { return limit_; }

  virtual void SetSize(unsigned int width, unsigned int height,
                       vpx_img_fmt format) {
    if (width != width_ || height != height_ || format != format_) {
      vpx_img_free(img_);
      img_ = vpx_img_alloc(nullptr, format, width, height, 1);
      ASSERT_NE(img_, nullptr);
      width_ = width;
      height_ = height;
      format_ = format;
      switch (format) {
        case VPX_IMG_FMT_NV12:
        case VPX_IMG_FMT_I420: raw_size_ = width * height * 3 / 2; break;
        case VPX_IMG_FMT_I422: raw_size_ = width * height * 2; break;
        case VPX_IMG_FMT_I440: raw_size_ = width * height * 2; break;
        case VPX_IMG_FMT_I444: raw_size_ = width * height * 3; break;
        case VPX_IMG_FMT_I42016: raw_size_ = width * height * 3; break;
        case VPX_IMG_FMT_I42216: raw_size_ = width * height * 4; break;
        case VPX_IMG_FMT_I44016: raw_size_ = width * height * 4; break;
        case VPX_IMG_FMT_I44416: raw_size_ = width * height * 6; break;
        default: ASSERT_TRUE(0);
      }
    }
  }

  virtual void FillFrame() {
    ASSERT_NE(input_file_, nullptr);
    // Read a frame from input_file.
    if (fread(img_->img_data, raw_size_, 1, input_file_) == 0) {
      limit_ = frame_;
    }
  }

 protected:
  std::string file_name_;
  FILE *input_file_;
  vpx_image_t *img_;
  size_t raw_size_;
  unsigned int start_;
  unsigned int limit_;
  unsigned int frame_;
  unsigned int width_;
  unsigned int height_;
  vpx_img_fmt format_;
  int framerate_numerator_;
  int framerate_denominator_;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_YUV_VIDEO_SOURCE_H_
