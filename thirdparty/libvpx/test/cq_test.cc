/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <cmath>
#include <map>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "vpx_config.h"

namespace {

// CQ level range: [kCQLevelMin, kCQLevelMax).
const int kCQLevelMin = 4;
const int kCQLevelMax = 63;
const int kCQLevelStep = 8;
const unsigned int kCQTargetBitrate = 2000;

class CQTest : public ::libvpx_test::EncoderTest,
               public ::libvpx_test::CodecTestWithParam<int> {
 public:
  // maps the cqlevel to the bitrate produced.
  using BitrateMap = std::map<int, uint32_t>;

  static void SetUpTestSuite() { bitrates_.clear(); }

  static void TearDownTestSuite() {
    ASSERT_TRUE(!HasFailure())
        << "skipping bitrate validation due to earlier failure.";
    uint32_t prev_actual_bitrate = kCQTargetBitrate;
    for (BitrateMap::const_iterator iter = bitrates_.begin();
         iter != bitrates_.end(); ++iter) {
      const uint32_t cq_actual_bitrate = iter->second;
      EXPECT_LE(cq_actual_bitrate, prev_actual_bitrate)
          << "cq_level: " << iter->first
          << ", bitrate should decrease with increase in CQ level.";
      prev_actual_bitrate = cq_actual_bitrate;
    }
  }

 protected:
  CQTest() : EncoderTest(GET_PARAM(0)), cq_level_(GET_PARAM(1)) {
    init_flags_ = VPX_CODEC_USE_PSNR;
  }

  ~CQTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(libvpx_test::kTwoPassGood);
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    file_size_ = 0;
    psnr_ = 0.0;
    n_frames_ = 0;
  }

  void PreEncodeFrameHook(libvpx_test::VideoSource *video,
                          libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      if (cfg_.rc_end_usage == VPX_CQ) {
        encoder->Control(VP8E_SET_CQ_LEVEL, cq_level_);
      }
      encoder->Control(VP8E_SET_CPUUSED, 3);
    }
  }

  void PSNRPktHook(const vpx_codec_cx_pkt_t *pkt) override {
    psnr_ += pow(10.0, pkt->data.psnr.psnr[0] / 10.0);
    n_frames_++;
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    file_size_ += pkt->data.frame.sz;
  }

  double GetLinearPSNROverBitrate() const {
    double avg_psnr = log10(psnr_ / n_frames_) * 10.0;
    return pow(10.0, avg_psnr / 10.0) / file_size_;
  }

  int cq_level() const { return cq_level_; }
  size_t file_size() const { return file_size_; }
  int n_frames() const { return n_frames_; }

  static BitrateMap bitrates_;

 private:
  int cq_level_;
  size_t file_size_;
  double psnr_;
  int n_frames_;
};

CQTest::BitrateMap CQTest::bitrates_;

TEST_P(CQTest, LinearPSNRIsHigherForCQLevel) {
  const vpx_rational timebase = { 33333333, 1000000000 };
#if CONFIG_REALTIME_ONlY
  GTEST_SKIP()
      << "Non-zero g_lag_in_frames is unsupported with CONFIG_REALTIME_ONLY";
#else
  cfg_.g_timebase = timebase;
  cfg_.rc_target_bitrate = kCQTargetBitrate;
  cfg_.g_lag_in_frames = 25;

  cfg_.rc_end_usage = VPX_CQ;
  libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                     timebase.den, timebase.num, 0, 30);
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const double cq_psnr_lin = GetLinearPSNROverBitrate();
  const unsigned int cq_actual_bitrate =
      static_cast<unsigned int>(file_size()) * 8 * 30 / (n_frames() * 1000);
  EXPECT_LE(cq_actual_bitrate, kCQTargetBitrate);
  bitrates_[cq_level()] = cq_actual_bitrate;

  // try targeting the approximate same bitrate with VBR mode
  cfg_.rc_end_usage = VPX_VBR;
  cfg_.rc_target_bitrate = cq_actual_bitrate;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const double vbr_psnr_lin = GetLinearPSNROverBitrate();
  EXPECT_GE(cq_psnr_lin, vbr_psnr_lin);
#endif  // CONFIG_REALTIME_ONLY
}

VP8_INSTANTIATE_TEST_SUITE(CQTest, ::testing::Range(kCQLevelMin, kCQLevelMax,
                                                    kCQLevelStep));
}  // namespace
