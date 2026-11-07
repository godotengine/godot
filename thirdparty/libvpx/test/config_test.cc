/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/video_source.h"

namespace {

class ConfigTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  ConfigTest()
      : EncoderTest(GET_PARAM(0)), frame_count_in_(0), frame_count_out_(0),
        frame_count_max_(0) {}
  ~ConfigTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    frame_count_in_ = 0;
    frame_count_out_ = 0;
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource * /*video*/) override {
    ++frame_count_in_;
    abort_ |= (frame_count_in_ >= frame_count_max_);
  }

  void FramePktHook(const vpx_codec_cx_pkt_t * /*pkt*/) override {
    ++frame_count_out_;
  }

  unsigned int frame_count_in_;
  unsigned int frame_count_out_;
  unsigned int frame_count_max_;
};

TEST_P(ConfigTest, LagIsDisabled) {
  frame_count_max_ = 2;
  cfg_.g_lag_in_frames = 15;

  libvpx_test::DummyVideoSource video;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  EXPECT_EQ(frame_count_in_, frame_count_out_);
}

VP8_INSTANTIATE_TEST_SUITE(ConfigTest, ONE_PASS_TEST_MODES);
}  // namespace
