/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TEST_Y4M_VIDEO_SOURCE_H_
#define VPX_TEST_Y4M_VIDEO_SOURCE_H_
#include <algorithm>
#include <memory>
#include <string>

#include "test/video_source.h"
#include "./y4minput.h"

namespace libvpx_test {

// This class extends VideoSource to allow parsing of raw yv12
// so that we can do actual file encodes.
class Y4mVideoSource : public VideoSource {
 public:
  Y4mVideoSource(const std::string &file_name, unsigned int start, int limit)
      : file_name_(file_name), input_file_(nullptr), img_(new vpx_image_t()),
        start_(start), limit_(limit), frame_(0), framerate_numerator_(0),
        framerate_denominator_(0), y4m_() {}

  ~Y4mVideoSource() override {
    vpx_img_free(img_.get());
    CloseSource();
  }

  virtual void OpenSource() {
    CloseSource();
    input_file_ = OpenTestDataFile(file_name_);
    ASSERT_NE(input_file_, nullptr)
        << "Input file open failed. Filename: " << file_name_;
  }

  virtual void ReadSourceToStart() {
    ASSERT_NE(input_file_, nullptr);
    ASSERT_FALSE(y4m_input_open(&y4m_, input_file_, nullptr, 0, 0));
    framerate_numerator_ = y4m_.fps_n;
    framerate_denominator_ = y4m_.fps_d;
    frame_ = 0;
    for (unsigned int i = 0; i < start_; i++) {
      Next();
    }
    FillFrame();
  }

  void Begin() override {
    OpenSource();
    ReadSourceToStart();
  }

  void Next() override {
    ++frame_;
    FillFrame();
  }

  vpx_image_t *img() const override {
    return (frame_ < limit_) ? img_.get() : nullptr;
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

  virtual void FillFrame() {
    ASSERT_NE(input_file_, nullptr);
    // Read a frame from input_file.
    y4m_input_fetch_frame(&y4m_, input_file_, img_.get());
  }

  // Swap buffers with another y4m source. This allows reading a new frame
  // while keeping the old frame around. A whole Y4mSource is required and
  // not just a vpx_image_t because of how the y4m reader manipulates
  // vpx_image_t internals,
  void SwapBuffers(Y4mVideoSource *other) {
    std::swap(other->y4m_.dst_buf, y4m_.dst_buf);
    vpx_image_t *tmp;
    tmp = other->img_.release();
    other->img_.reset(img_.release());
    img_.reset(tmp);
  }

 protected:
  void CloseSource() {
    y4m_input_close(&y4m_);
    y4m_ = y4m_input();
    if (input_file_ != nullptr) {
      fclose(input_file_);
      input_file_ = nullptr;
    }
  }

  std::string file_name_;
  FILE *input_file_;
  std::unique_ptr<vpx_image_t> img_;
  unsigned int start_;
  unsigned int limit_;
  unsigned int frame_;
  int framerate_numerator_;
  int framerate_denominator_;
  y4m_input y4m_;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_Y4M_VIDEO_SOURCE_H_
