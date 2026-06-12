/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
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
#include "vpx_config.h"

namespace {

const int kVideoSourceWidth = 320;
const int kVideoSourceHeight = 240;
const int kFramesToEncode = 3;

// A video source that exposes functions to set the timebase, framerate and
// starting pts.
class DummyTimebaseVideoSource : public ::libvpx_test::DummyVideoSource {
 public:
  // Parameters num and den set the timebase for the video source.
  DummyTimebaseVideoSource(int num, int den)
      : timebase_({ num, den }), framerate_numerator_(30),
        framerate_denominator_(1), starting_pts_(0) {
    SetSize(kVideoSourceWidth, kVideoSourceHeight);
    set_limit(kFramesToEncode);
  }

  void SetFramerate(int numerator, int denominator) {
    framerate_numerator_ = numerator;
    framerate_denominator_ = denominator;
  }

  // Returns one frames duration in timebase units as a double.
  double FrameDuration() const {
    return (static_cast<double>(timebase_.den) / timebase_.num) /
           (static_cast<double>(framerate_numerator_) / framerate_denominator_);
  }

  vpx_codec_pts_t pts() const override {
    return static_cast<vpx_codec_pts_t>(frame_ * FrameDuration() +
                                        starting_pts_ + 0.5);
  }

  unsigned long duration() const override {
    return static_cast<unsigned long>(FrameDuration() + 0.5);
  }

  vpx_rational_t timebase() const override { return timebase_; }

  void set_starting_pts(int64_t starting_pts) { starting_pts_ = starting_pts; }

 private:
  vpx_rational_t timebase_;
  int framerate_numerator_;
  int framerate_denominator_;
  int64_t starting_pts_;
};

class TimestampTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  TimestampTest() : EncoderTest(GET_PARAM(0)) {}
  ~TimestampTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
  }
};

// Tests encoding in millisecond timebase.
TEST_P(TimestampTest, EncodeFrames) {
  DummyTimebaseVideoSource video(1, 1000);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(TimestampTest, TestMicrosecondTimebase) {
  // Set the timebase to microseconds.
  DummyTimebaseVideoSource video(1, 1000000);
  video.set_limit(1);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

TEST_P(TimestampTest, TestVpxRollover) {
  DummyTimebaseVideoSource video(1, 1000);
  video.set_starting_pts(922337170351ll);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
}

#if CONFIG_REALTIME_ONLY
VP8_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kRealTime));
VP9_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kRealTime));
#else
VP8_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kTwoPassGood));
VP9_INSTANTIATE_TEST_SUITE(TimestampTest,
                           ::testing::Values(::libvpx_test::kTwoPassGood));
#endif
}  // namespace
