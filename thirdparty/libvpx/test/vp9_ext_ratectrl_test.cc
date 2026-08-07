/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cstdint>
#include <new>
#include <memory>

#include "./vpx_config.h"

#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/yuv_video_source.h"
#if CONFIG_VP9_DECODER
#include "vpx/vp8dx.h"
#endif
#include "vpx/vpx_codec.h"
#include "vpx/vpx_encoder.h"
#include "vpx/vpx_ext_ratectrl.h"
#include "vpx/vpx_image.h"
#include "vpx/vpx_tpl.h"
#include "vpx_dsp/vpx_dsp_common.h"

namespace {

constexpr int kShowFrameCount = 10;
constexpr int kKeyframeQp = 10;
constexpr int kLeafQp = 40;
constexpr int kArfQp = 15;

// Simple external rate controller for testing.
class RateControllerForTest {
 public:
  RateControllerForTest() : current_gop_(-1) {}
  ~RateControllerForTest() {}

  void StartNextGop() { ++current_gop_; }

  vpx_rc_gop_decision_t GetCurrentGop() const {
    vpx_rc_gop_decision_t gop_decision;
    if (current_gop_ == 0) {
      gop_decision.use_key_frame = 1;
      gop_decision.use_alt_ref = 1;
      gop_decision.gop_coding_frames =
          kShowFrameCount - 1 + gop_decision.use_alt_ref;
      // key frame
      gop_decision.update_type[0] = VPX_RC_KF_UPDATE;
      gop_decision.update_ref_index[0] = 0;
      gop_decision.ref_frame_list[0] = get_kf_ref_frame();
      // arf
      gop_decision.update_type[1] = VPX_RC_ARF_UPDATE;
      gop_decision.update_ref_index[1] = 1;
      gop_decision.ref_frame_list[1] = get_arf_ref_frame();
      // leafs
      for (int i = 2; i < gop_decision.gop_coding_frames; ++i) {
        gop_decision.update_type[i] = VPX_RC_LF_UPDATE;
        gop_decision.update_ref_index[i] = 2;
        gop_decision.ref_frame_list[i] = get_leaf_ref_frame(i);
      }
    } else {
      // Pad a overlay-only GOP as the last GOP.
      EXPECT_EQ(current_gop_, 1);
      gop_decision.use_key_frame = 0;
      gop_decision.use_alt_ref = 0;
      gop_decision.gop_coding_frames = 1;

      gop_decision.update_type[0] = VPX_RC_OVERLAY_UPDATE;
      gop_decision.update_ref_index[0] = 1;
      gop_decision.ref_frame_list[0] = get_ovl_ref_frame();
    }
    return gop_decision;
  }

  int CalculateFrameDecision(int frame_index) {
    if (current_gop_ == 0 && frame_index == 0) {
      // Key frame, first frame in the first GOP.
      return kKeyframeQp;
    } else if (frame_index == 1) {
      // ARF, we always use ARF for this test.
      return kArfQp;
    } else {
      return kLeafQp;
    }
  }

 private:
  vpx_rc_ref_frame_t get_kf_ref_frame() const {
    vpx_rc_ref_frame_t ref_frame;
    ref_frame.index[0] = -1;
    ref_frame.index[1] = -1;
    ref_frame.index[2] = -1;
    ref_frame.name[0] = VPX_RC_INVALID_REF_FRAME;
    ref_frame.name[1] = VPX_RC_INVALID_REF_FRAME;
    ref_frame.name[2] = VPX_RC_INVALID_REF_FRAME;
    return ref_frame;
  }
  vpx_rc_ref_frame_t get_arf_ref_frame() const {
    vpx_rc_ref_frame_t ref_frame;
    ref_frame.index[0] = 0;
    ref_frame.index[1] = -1;
    ref_frame.index[2] = -1;
    ref_frame.name[0] = VPX_RC_GOLDEN_FRAME;
    ref_frame.name[1] = VPX_RC_INVALID_REF_FRAME;
    ref_frame.name[2] = VPX_RC_INVALID_REF_FRAME;
    return ref_frame;
  }
  vpx_rc_ref_frame_t get_leaf_ref_frame(int count) const {
    vpx_rc_ref_frame_t ref_frame;
    ref_frame.index[0] = 0;
    ref_frame.index[1] = 1;
    ref_frame.index[2] = count > 2 ? 2 : -1;
    ref_frame.name[0] = VPX_RC_GOLDEN_FRAME;
    ref_frame.name[1] = VPX_RC_ALTREF_FRAME;
    ref_frame.name[2] =
        count > 2 ? VPX_RC_LAST_FRAME : VPX_RC_INVALID_REF_FRAME;
    return ref_frame;
  }
  vpx_rc_ref_frame_t get_ovl_ref_frame() const {
    vpx_rc_ref_frame_t ref_frame;
    ref_frame.index[0] = 1;
    ref_frame.index[1] = -1;
    ref_frame.index[2] = -1;
    ref_frame.name[0] = VPX_RC_ALTREF_FRAME;
    ref_frame.name[1] = VPX_RC_INVALID_REF_FRAME;
    ref_frame.name[2] = VPX_RC_INVALID_REF_FRAME;
    return ref_frame;
  }

  int current_gop_;
};

// Callbacks used in this test.
vpx_rc_status_t rc_test_create_model(
    void * /*priv*/, const vpx_rc_config_t * /*ratectrl_config*/,
    vpx_rc_model_t *rate_ctrl_model_ptr) {
  std::unique_ptr<RateControllerForTest> test_controller(
      new RateControllerForTest());
  *rate_ctrl_model_ptr = test_controller.release();
  return VPX_RC_OK;
}

vpx_rc_status_t rc_test_send_firstpass_stats(
    vpx_rc_model_t /*rate_ctrl_model*/,
    const vpx_rc_firstpass_stats_t *first_pass_stats) {
  EXPECT_EQ(first_pass_stats->num_frames, kShowFrameCount);
  for (int i = 0; i < first_pass_stats->num_frames; ++i) {
    EXPECT_DOUBLE_EQ(first_pass_stats->frame_stats[i].frame, i);
  }
  return VPX_RC_OK;
}

vpx_rc_status_t rc_test_send_tpl_gop_stats(
    vpx_rc_model_t /*rate_ctrl_model*/, const VpxTplGopStats *tpl_gop_stats) {
  EXPECT_GT(tpl_gop_stats->size, 0);

  for (int i = 0; i < tpl_gop_stats->size; ++i) {
    EXPECT_GT(tpl_gop_stats->frame_stats_list[i].num_blocks, 0);
  }
  return VPX_RC_OK;
}

vpx_rc_status_t rc_test_get_encodeframe_decision(
    vpx_rc_model_t rate_ctrl_model, const int frame_gop_index,
    vpx_rc_encodeframe_decision_t *frame_decision) {
  RateControllerForTest *test_controller =
      static_cast<RateControllerForTest *>(rate_ctrl_model);
  frame_decision->q_index =
      test_controller->CalculateFrameDecision(frame_gop_index);
  frame_decision->rdmult =
      frame_decision->q_index * frame_decision->q_index / 2;
  frame_decision->delta_q_uv = 0;
  return VPX_RC_OK;
}

vpx_rc_status_t rc_test_get_gop_decision(vpx_rc_model_t rate_ctrl_model,
                                         vpx_rc_gop_decision_t *gop_decision) {
  RateControllerForTest *test_controller =
      static_cast<RateControllerForTest *>(rate_ctrl_model);
  test_controller->StartNextGop();
  *gop_decision = test_controller->GetCurrentGop();
  return VPX_RC_OK;
}

vpx_rc_status_t rc_delete_model(vpx_rc_model_t rate_ctrl_model) {
  RateControllerForTest *test_controller =
      static_cast<RateControllerForTest *>(rate_ctrl_model);
  delete test_controller;
  return VPX_RC_OK;
}

class ExtRateCtrlTest : public ::libvpx_test::EncoderTest,
                        public ::testing::Test {
 protected:
  ExtRateCtrlTest()
      : EncoderTest(&::libvpx_test::kVP9), received_show_frame_count_(0),
        current_frame_qp_(0) {}

  ~ExtRateCtrlTest() override = default;

  void SetUp() override {
    InitializeConfig();
#if CONFIG_REALTIME_ONLY
    SetMode(::libvpx_test::kRealTime);
#else
    SetMode(::libvpx_test::kTwoPassGood);
#endif
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      vpx_rc_funcs_t rc_funcs = {};
      rc_funcs.rc_type = VPX_RC_GOP_QP;
      rc_funcs.create_model = rc_test_create_model;
      rc_funcs.send_firstpass_stats = rc_test_send_firstpass_stats;
      rc_funcs.send_tpl_gop_stats = rc_test_send_tpl_gop_stats;
      rc_funcs.get_gop_decision = rc_test_get_gop_decision;
      rc_funcs.get_encodeframe_decision = rc_test_get_encodeframe_decision;
      rc_funcs.delete_model = rc_delete_model;
      encoder->Control(VP9E_SET_EXTERNAL_RATE_CONTROL, &rc_funcs);
    }
  }

#if CONFIG_VP9_DECODER
  bool HandleDecodeResult(const vpx_codec_err_t res_dec,
                          const ::libvpx_test::VideoSource & /*video*/,
                          ::libvpx_test::Decoder *decoder) override {
    EXPECT_EQ(VPX_CODEC_OK, res_dec) << decoder->DecodeError();
    decoder->Control(VPXD_GET_LAST_QUANTIZER, &current_frame_qp_);
    return VPX_CODEC_OK == res_dec;
  }

  void FramePktHook(const vpx_codec_cx_pkt_t *pkt) override {
    // We are not comparing current_frame_qp_ here because the encoder will
    // pack ARF and the next show frame into one pkt. Therefore, we might
    // receive two frames in one pkt. However, one thing we are sure is that
    // each pkt will have just one show frame. Therefore, we can check if the
    // received show frame count match the actual show frame count.
    if (pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
      ++received_show_frame_count_;
    }
  }
#endif  // CONFIG_VP9_DECODER

  int received_show_frame_count_;
  int current_frame_qp_;
};

TEST_F(ExtRateCtrlTest, EncodeTest) {
  cfg_.rc_target_bitrate = 4000;
  cfg_.g_lag_in_frames = 25;

  std::unique_ptr<libvpx_test::VideoSource> video;
  video.reset(new (std::nothrow) libvpx_test::YUVVideoSource(
      "bus_352x288_420_f20_b8.yuv", VPX_IMG_FMT_I420, 352, 288, 30, 1, 0,
      kShowFrameCount));

  ASSERT_NE(video, nullptr);
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  EXPECT_EQ(received_show_frame_count_, kShowFrameCount);
}

}  // namespace
