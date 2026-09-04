/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <memory>

#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/register_state_check.h"
#include "test/video_source.h"
#include "vpx_config.h"

namespace {

class EncoderWithExpectedError : public ::libvpx_test::Encoder {
 public:
  EncoderWithExpectedError(vpx_codec_enc_cfg_t cfg, vpx_enc_deadline_t deadline,
                           const unsigned long init_flags,  // NOLINT
                           ::libvpx_test::TwopassStatsStore *stats)
      : ::libvpx_test::Encoder(cfg, deadline, init_flags, stats) {}
  // This overrides with expected error code.
  void EncodeFrame(::libvpx_test::VideoSource *video,
                   const unsigned long frame_flags,  // NOLINT
                   const vpx_codec_err_t expected_err) {
    if (video->img()) {
      EncodeFrameInternal(*video, frame_flags, expected_err);
    } else {
      Flush();
    }

    // Handle twopass stats
    ::libvpx_test::CxDataIterator iter = GetCxData();

    while (const vpx_codec_cx_pkt_t *pkt = iter.Next()) {
      if (pkt->kind != VPX_CODEC_STATS_PKT) continue;

      stats_->Append(*pkt);
    }
  }

 protected:
  void EncodeFrameInternal(const ::libvpx_test::VideoSource &video,
                           const unsigned long frame_flags,  // NOLINT
                           const vpx_codec_err_t expected_err) {
    vpx_codec_err_t res;
    const vpx_image_t *img = video.img();

    // Handle frame resizing
    if (cfg_.g_w != img->d_w || cfg_.g_h != img->d_h) {
      cfg_.g_w = img->d_w;
      cfg_.g_h = img->d_h;
      res = vpx_codec_enc_config_set(&encoder_, &cfg_);
      ASSERT_EQ(res, VPX_CODEC_OK) << EncoderError();
    }

    // Encode the frame
    API_REGISTER_STATE_CHECK(res = vpx_codec_encode(&encoder_, img, video.pts(),
                                                    video.duration(),
                                                    frame_flags, deadline_));
    ASSERT_EQ(expected_err, res) << EncoderError();
  }

  vpx_codec_iface_t *CodecInterface() const override {
#if CONFIG_VP9_ENCODER
    return &vpx_codec_vp9_cx_algo;
#else
    return nullptr;
#endif
  }
};

class VP9FrameSizeTestsLarge : public ::libvpx_test::EncoderTest,
                               public ::testing::Test {
 protected:
  VP9FrameSizeTestsLarge()
      : EncoderTest(&::libvpx_test::kVP9), expected_res_(VPX_CODEC_OK) {}
  ~VP9FrameSizeTestsLarge() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kRealTime);
  }

  bool HandleDecodeResult(const vpx_codec_err_t res_dec,
                          const libvpx_test::VideoSource & /*video*/,
                          libvpx_test::Decoder *decoder) override {
    EXPECT_EQ(expected_res_, res_dec) << decoder->DecodeError();
    return !::testing::Test::HasFailure();
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP8E_SET_CPUUSED, 7);
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
      encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
      encoder->Control(VP8E_SET_ARNR_TYPE, 3);
    }
  }

  using ::libvpx_test::EncoderTest::RunLoop;
  virtual void RunLoop(::libvpx_test::VideoSource *video,
                       const vpx_codec_err_t expected_err) {
    stats_.Reset();

    ASSERT_TRUE(passes_ == 1 || passes_ == 2);
    for (unsigned int pass = 0; pass < passes_; pass++) {
      vpx_codec_pts_t last_pts = 0;

      if (passes_ == 1) {
        cfg_.g_pass = VPX_RC_ONE_PASS;
      } else if (pass == 0) {
        cfg_.g_pass = VPX_RC_FIRST_PASS;
      } else {
        cfg_.g_pass = VPX_RC_LAST_PASS;
      }

      BeginPassHook(pass);
      std::unique_ptr<EncoderWithExpectedError> encoder(
          new EncoderWithExpectedError(cfg_, deadline_, init_flags_, &stats_));
      ASSERT_NE(encoder.get(), nullptr);

      ASSERT_NO_FATAL_FAILURE(video->Begin());
      encoder->InitEncoder(video);
      ASSERT_FALSE(::testing::Test::HasFatalFailure());
      for (bool again = true; again; video->Next()) {
        again = (video->img() != nullptr);

        PreEncodeFrameHook(video, encoder.get());
        encoder->EncodeFrame(video, frame_flags_, expected_err);

        PostEncodeFrameHook(encoder.get());

        ::libvpx_test::CxDataIterator iter = encoder->GetCxData();

        while (const vpx_codec_cx_pkt_t *pkt = iter.Next()) {
          pkt = MutateEncoderOutputHook(pkt);
          again = true;
          switch (pkt->kind) {
            case VPX_CODEC_CX_FRAME_PKT:
              ASSERT_GE(pkt->data.frame.pts, last_pts);
              last_pts = pkt->data.frame.pts;
              FramePktHook(pkt);
              break;

            case VPX_CODEC_PSNR_PKT: PSNRPktHook(pkt); break;
            case VPX_CODEC_STATS_PKT: StatsPktHook(pkt); break;
            default: break;
          }
        }

        if (!Continue()) break;
      }

      EndPassHook();

      if (!Continue()) break;
    }
  }

  vpx_codec_err_t expected_res_;
};

TEST_F(VP9FrameSizeTestsLarge, TestInvalidSizes) {
#ifdef CHROMIUM
  GTEST_SKIP() << "16K framebuffers are not supported by Chromium's allocator.";
#else
  ::libvpx_test::RandomVideoSource video;

#if CONFIG_SIZE_LIMIT
  video.SetSize(DECODE_WIDTH_LIMIT + 16, DECODE_HEIGHT_LIMIT + 16);
  video.set_limit(2);
  expected_res_ = VPX_CODEC_MEM_ERROR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video, expected_res_));
#endif

#endif
}

TEST_F(VP9FrameSizeTestsLarge, ValidSizes) {
#ifdef CHROMIUM
  GTEST_SKIP()
      << "Under Chromium's configuration the allocator is unable to provide"
         "the space required for a single frame at the maximum resolution.";
#else
  ::libvpx_test::RandomVideoSource video;

#if CONFIG_SIZE_LIMIT
  video.SetSize(DECODE_WIDTH_LIMIT, DECODE_HEIGHT_LIMIT);
  video.set_limit(2);
  expected_res_ = VPX_CODEC_OK;
  ASSERT_NO_FATAL_FAILURE(::libvpx_test::EncoderTest::RunLoop(&video));
#else
// This test produces a pretty large single frame allocation,  (roughly
// 25 megabits). The encoder allocates a good number of these frames
// one for each lag in frames (for 2 pass), and then one for each possible
// reference buffer (8) - we can end up with up to 30 buffers of roughly this
// size or almost 1 gig of memory.
// In total the allocations will exceed 2GiB which may cause a failure with
// mingw + wine, use a smaller size in that case.
#if defined(_WIN32) && !defined(_WIN64)
  video.SetSize(4096, 3072);
#else
  video.SetSize(4096, 4096);
#endif
  video.set_limit(2);
  expected_res_ = VPX_CODEC_OK;
  ASSERT_NO_FATAL_FAILURE(::libvpx_test::EncoderTest::RunLoop(&video));
#endif

#endif  // defined(CHROMIUM)
}

TEST_F(VP9FrameSizeTestsLarge, OneByOneVideo) {
  ::libvpx_test::RandomVideoSource video;

  video.SetSize(1, 1);
  video.set_limit(2);
  expected_res_ = VPX_CODEC_OK;
  ASSERT_NO_FATAL_FAILURE(::libvpx_test::EncoderTest::RunLoop(&video));
}
}  // namespace
