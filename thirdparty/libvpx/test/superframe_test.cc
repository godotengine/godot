/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <climits>
#include <tuple>

#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"

namespace {

const int kTestMode = 0;

using SuperframeTestParam = std::tuple<libvpx_test::TestMode, int>;

class SuperframeTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<SuperframeTestParam> {
 protected:
  SuperframeTest()
      : EncoderTest(GET_PARAM(0)), modified_buf_(nullptr), last_sf_pts_(0) {}
  ~SuperframeTest() override = default;

  void SetUp() override {
    InitializeConfig();
    const SuperframeTestParam input = GET_PARAM(1);
    const libvpx_test::TestMode mode = std::get<kTestMode>(input);
    SetMode(mode);
    sf_count_ = 0;
    sf_count_max_ = INT_MAX;
  }

  void TearDown() override { delete[] modified_buf_; }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video,
                          libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
    }
  }

  const vpx_codec_cx_pkt_t *MutateEncoderOutputHook(
      const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) return pkt;

    const uint8_t *buffer = reinterpret_cast<uint8_t *>(pkt->data.frame.buf);
    const uint8_t marker = buffer[pkt->data.frame.sz - 1];
    const int frames = (marker & 0x7) + 1;
    const int mag = ((marker >> 3) & 3) + 1;
    const unsigned int index_sz = 2 + mag * frames;
    if ((marker & 0xe0) == 0xc0 && pkt->data.frame.sz >= index_sz &&
        buffer[pkt->data.frame.sz - index_sz] == marker) {
      // frame is a superframe. strip off the index.
      if (modified_buf_) delete[] modified_buf_;
      modified_buf_ = new uint8_t[pkt->data.frame.sz - index_sz];
      memcpy(modified_buf_, pkt->data.frame.buf, pkt->data.frame.sz - index_sz);
      modified_pkt_ = *pkt;
      modified_pkt_.data.frame.buf = modified_buf_;
      modified_pkt_.data.frame.sz -= index_sz;

      sf_count_++;
      last_sf_pts_ = pkt->data.frame.pts;
      return &modified_pkt_;
    }

    // Make sure we do a few frames after the last SF
    abort_ |=
        sf_count_ > sf_count_max_ && pkt->data.frame.pts - last_sf_pts_ >= 5;
    return pkt;
  }

  int sf_count_;
  int sf_count_max_;
  vpx_codec_cx_pkt_t modified_pkt_;
  uint8_t *modified_buf_;
  vpx_codec_pts_t last_sf_pts_;
};

TEST_P(SuperframeTest, TestSuperframeIndexIsOptional) {
  sf_count_max_ = 0;  // early exit on successful test.
  cfg_.g_lag_in_frames = 25;

  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 40);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  EXPECT_EQ(sf_count_, 1);
}

VP9_INSTANTIATE_TEST_SUITE(
    SuperframeTest,
    ::testing::Combine(::testing::Values(::libvpx_test::kTwoPassGood),
                       ::testing::Values(0)));
}  // namespace
