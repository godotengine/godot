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
#include <cstring>
#include <vector>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "./vpx_config.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_encoder.h"
#include "vpx/vpx_image.h"

namespace {

class KeyframeTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  KeyframeTest() : EncoderTest(GET_PARAM(0)) {}
  ~KeyframeTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(GET_PARAM(1));
    kf_count_ = 0;
    kf_count_max_ = INT_MAX;
    kf_do_force_kf_ = false;
    set_cpu_used_ = 0;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (kf_do_force_kf_) {
      frame_flags_ = (video->frame() % 3) ? 0 : VPX_EFLAG_FORCE_KF;
    }
    if (set_cpu_used_ && video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, set_cpu_used_);
    }
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->data.frame.flags & VPX_FRAME_IS_KEY) {
      kf_pts_list_.push_back(pkt->data.frame.pts);
      kf_count_++;
      abort_ |= kf_count_ > kf_count_max_;
    }
  }

  bool kf_do_force_kf_;
  int kf_count_;
  int kf_count_max_;
  std::vector<vpx_codec_pts_t> kf_pts_list_;
  int set_cpu_used_;
};

TEST_P(KeyframeTest, TestRandomVideoSource) {
  // Validate that encoding the RandomVideoSource produces multiple keyframes.
  // This validates the results of the TestDisableKeyframes test.
  kf_count_max_ = 2;  // early exit successful tests.

  ::libvpx_test::RandomVideoSource video;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // In realtime mode - auto placed keyframes are exceedingly rare,  don't
  // bother with this check   if(GetParam() > 0)
  if (GET_PARAM(1) > 0) {
    EXPECT_GT(kf_count_, 1);
  }
}

TEST_P(KeyframeTest, TestDisableKeyframes) {
  cfg_.kf_mode = VPX_KF_DISABLED;
  kf_count_max_ = 1;  // early exit failed tests.

  ::libvpx_test::RandomVideoSource video;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  EXPECT_EQ(1, kf_count_);
}

TEST_P(KeyframeTest, TestForceKeyframe) {
  cfg_.kf_mode = VPX_KF_DISABLED;
  kf_do_force_kf_ = true;

  ::libvpx_test::DummyVideoSource video;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // verify that every third frame is a keyframe.
  for (std::vector<vpx_codec_pts_t>::const_iterator iter = kf_pts_list_.begin();
       iter != kf_pts_list_.end(); ++iter) {
    ASSERT_EQ(0, *iter % 3) << "Unexpected keyframe at frame " << *iter;
  }
}

TEST_P(KeyframeTest, TestKeyframeMaxDistance) {
  cfg_.kf_max_dist = 25;

  ::libvpx_test::DummyVideoSource video;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // verify that keyframe interval matches kf_max_dist
  for (std::vector<vpx_codec_pts_t>::const_iterator iter = kf_pts_list_.begin();
       iter != kf_pts_list_.end(); ++iter) {
    ASSERT_EQ(0, *iter % 25) << "Unexpected keyframe at frame " << *iter;
  }
}

TEST_P(KeyframeTest, TestAutoKeyframe) {
  cfg_.kf_mode = VPX_KF_AUTO;
  kf_do_force_kf_ = false;

  // Force a deterministic speed step in Real Time mode, as the faster modes
  // may not produce a keyframe like we expect. This is necessary when running
  // on very slow environments (like Valgrind). The step -11 was determined
  // experimentally as the fastest mode that still throws the keyframe.
  if (deadline_ == VPX_DL_REALTIME) set_cpu_used_ = -11;

  // This clip has a cut scene every 30 frames -> Frame 0, 30, 60, 90, 120.
  // I check only the first 40 frames to make sure there's a keyframe at frame
  // 0 and 30.
  ::libvpx_test::I420VideoSource video("hantro_collage_w352h288.yuv", 352, 288,
                                       30, 1, 0, 40);

  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // In realtime mode - auto placed keyframes are exceedingly rare,  don't
  // bother with this check
  if (GET_PARAM(1) > 0) {
    EXPECT_EQ(2u, kf_pts_list_.size()) << " Not the right number of keyframes ";
  }

  // Verify that keyframes match the file keyframes in the file.
  for (std::vector<vpx_codec_pts_t>::const_iterator iter = kf_pts_list_.begin();
       iter != kf_pts_list_.end(); ++iter) {
    if (deadline_ == VPX_DL_REALTIME && *iter > 0)
      EXPECT_EQ(0, (*iter - 1) % 30)
          << "Unexpected keyframe at frame " << *iter;
    else
      EXPECT_EQ(0, *iter % 30) << "Unexpected keyframe at frame " << *iter;
  }
}

VP8_INSTANTIATE_TEST_SUITE(KeyframeTest, ALL_TEST_MODES);

bool IsVP9(vpx_codec_iface_t *iface) {
  static const char kVP9Name[] = "WebM Project VP9";
  return strncmp(kVP9Name, vpx_codec_iface_name(iface), sizeof(kVP9Name) - 1) ==
         0;
}

vpx_image_t *CreateGrayImage(vpx_img_fmt_t fmt, unsigned int w,
                             unsigned int h) {
  vpx_image_t *const image = vpx_img_alloc(nullptr, fmt, w, h, 1);
  if (!image) return image;

  for (unsigned int i = 0; i < image->d_h; ++i) {
    memset(image->planes[0] + i * image->stride[0], 128, image->d_w);
  }
  const unsigned int uv_h = (image->d_h + 1) / 2;
  const unsigned int uv_w = (image->d_w + 1) / 2;
  for (unsigned int i = 0; i < uv_h; ++i) {
    memset(image->planes[1] + i * image->stride[1], 128, uv_w);
    memset(image->planes[2] + i * image->stride[2], 128, uv_w);
  }
  return image;
}

// Tests kf_max_dist in one-pass encoding with zero lag.
void TestKeyframeMaximumInterval(vpx_codec_iface_t *iface,
                                 vpx_enc_deadline_t deadline,
                                 unsigned int kf_max_dist) {
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, /*usage=*/0),
            VPX_CODEC_OK);
  cfg.g_w = 320;
  cfg.g_h = 240;
  cfg.g_pass = VPX_RC_ONE_PASS;
  cfg.g_lag_in_frames = 0;
  cfg.kf_mode = VPX_KF_AUTO;
  cfg.kf_min_dist = 0;
  cfg.kf_max_dist = kf_max_dist;

  vpx_codec_ctx_t enc;
  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  const int speed = IsVP9(iface) ? 9 : -12;
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_CPUUSED, speed), VPX_CODEC_OK);

  vpx_image_t *image = CreateGrayImage(VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  // Encode frames.
  const vpx_codec_cx_pkt_t *pkt;
  const unsigned int num_frames = kf_max_dist == 0 ? 4 : 3 * kf_max_dist + 1;
  for (unsigned int i = 0; i < num_frames; ++i) {
    ASSERT_EQ(vpx_codec_encode(&enc, image, i, 1, 0, deadline), VPX_CODEC_OK);
    vpx_codec_iter_t iter = nullptr;
    while ((pkt = vpx_codec_get_cx_data(&enc, &iter)) != nullptr) {
      ASSERT_EQ(pkt->kind, VPX_CODEC_CX_FRAME_PKT);
      if (kf_max_dist == 0 || i % kf_max_dist == 0) {
        ASSERT_EQ(pkt->data.frame.flags & VPX_FRAME_IS_KEY, VPX_FRAME_IS_KEY);
      } else {
        ASSERT_EQ(pkt->data.frame.flags & VPX_FRAME_IS_KEY, 0u);
      }
    }
  }

  // Flush the encoder.
  bool got_data;
  do {
    ASSERT_EQ(vpx_codec_encode(&enc, nullptr, 0, 1, 0, deadline), VPX_CODEC_OK);
    got_data = false;
    vpx_codec_iter_t iter = nullptr;
    while ((pkt = vpx_codec_get_cx_data(&enc, &iter)) != nullptr) {
      ASSERT_EQ(pkt->kind, VPX_CODEC_CX_FRAME_PKT);
      got_data = true;
    }
  } while (got_data);

  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(KeyframeIntervalTest, KeyframeMaximumInterval) {
  std::vector<vpx_codec_iface_t *> ifaces;
#if CONFIG_VP8_ENCODER
  ifaces.push_back(vpx_codec_vp8_cx());
#endif
#if CONFIG_VP9_ENCODER
  ifaces.push_back(vpx_codec_vp9_cx());
#endif
  for (vpx_codec_iface_t *iface : ifaces) {
    for (vpx_enc_deadline_t deadline :
         { VPX_DL_REALTIME, VPX_DL_GOOD_QUALITY, VPX_DL_BEST_QUALITY }) {
      // Test 0 and 1 (both mean all intra), some powers of 2, some multiples
      // of 10, and some prime numbers.
      for (unsigned int kf_max_dist :
           { 0, 1, 2, 3, 4, 7, 10, 13, 16, 20, 23, 29, 32 }) {
        TestKeyframeMaximumInterval(iface, deadline, kf_max_dist);
      }
    }
  }
}

}  // namespace
