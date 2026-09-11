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
#include "test/encode_test_driver.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vp9/vp9_dx_iface.h"

namespace {

const int kCpuUsed = 2;

struct EncodePerfTestVideo {
  const char *name;
  uint32_t width;
  uint32_t height;
  uint32_t bitrate;
  int frames;
};

const EncodePerfTestVideo kVP9EncodePerfTestVectors[] = {
  { "niklas_1280_720_30.y4m", 1280, 720, 600, 10 },
};

struct EncodeParameters {
  int32_t tile_rows;
  int32_t tile_cols;
  int32_t lossless;
  int32_t error_resilient;
  int32_t frame_parallel;
  vpx_color_range_t color_range;
  vpx_color_space_t cs;
  int render_size[2];
  // TODO(JBB): quantizers / bitrate
};

const EncodeParameters kVP9EncodeParameterSet[] = {
  { 0, 0, 0, 1, 0, VPX_CR_STUDIO_RANGE, VPX_CS_BT_601, { 0, 0 } },
  { 0, 0, 0, 0, 0, VPX_CR_FULL_RANGE, VPX_CS_BT_709, { 0, 0 } },
  { 0, 0, 1, 0, 0, VPX_CR_FULL_RANGE, VPX_CS_BT_2020, { 0, 0 } },
  { 0, 2, 0, 0, 1, VPX_CR_STUDIO_RANGE, VPX_CS_UNKNOWN, { 640, 480 } },
  // TODO(JBB): Test profiles (requires more work).
};

class VpxEncoderParmsGetToDecoder
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<EncodeParameters,
                                                 EncodePerfTestVideo> {
 protected:
  VpxEncoderParmsGetToDecoder()
      : EncoderTest(GET_PARAM(0)), encode_parms(GET_PARAM(1)) {}

  ~VpxEncoderParmsGetToDecoder() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(::libvpx_test::kTwoPassGood);
    cfg_.g_lag_in_frames = 25;
    cfg_.g_error_resilient = encode_parms.error_resilient;
    dec_cfg_.threads = 4;
    test_video_ = GET_PARAM(2);
    cfg_.rc_target_bitrate = test_video_.bitrate;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      encoder->Control(VP9E_SET_COLOR_SPACE, encode_parms.cs);
      encoder->Control(VP9E_SET_COLOR_RANGE, encode_parms.color_range);
      encoder->Control(VP9E_SET_LOSSLESS, encode_parms.lossless);
      encoder->Control(VP9E_SET_FRAME_PARALLEL_DECODING,
                       encode_parms.frame_parallel);
      encoder->Control(VP9E_SET_TILE_ROWS, encode_parms.tile_rows);
      encoder->Control(VP9E_SET_TILE_COLUMNS, encode_parms.tile_cols);
      encoder->Control(VP8E_SET_CPUUSED, kCpuUsed);
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
      encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
      encoder->Control(VP8E_SET_ARNR_TYPE, 3);
      if (encode_parms.render_size[0] > 0 && encode_parms.render_size[1] > 0) {
        encoder->Control(VP9E_SET_RENDER_SIZE, encode_parms.render_size);
      }
    }
  }

  bool HandleDecodeResult(const vpx_codec_err_t res_dec,
                          const libvpx_test::VideoSource & /*video*/,
                          libvpx_test::Decoder *decoder) override {
    vpx_codec_ctx_t *const vp9_decoder = decoder->GetDecoder();
    vpx_codec_alg_priv_t *const priv =
        reinterpret_cast<vpx_codec_alg_priv_t *>(vp9_decoder->priv);
    VP9_COMMON *const common = &priv->pbi->common;

    if (encode_parms.lossless) {
      EXPECT_EQ(0, common->base_qindex);
      EXPECT_EQ(0, common->y_dc_delta_q);
      EXPECT_EQ(0, common->uv_dc_delta_q);
      EXPECT_EQ(0, common->uv_ac_delta_q);
      EXPECT_EQ(ONLY_4X4, common->tx_mode);
    }
    EXPECT_EQ(encode_parms.error_resilient, common->error_resilient_mode);
    if (encode_parms.error_resilient) {
      EXPECT_EQ(1, common->frame_parallel_decoding_mode);
      EXPECT_EQ(0, common->use_prev_frame_mvs);
    } else {
      EXPECT_EQ(encode_parms.frame_parallel,
                common->frame_parallel_decoding_mode);
    }
    EXPECT_EQ(encode_parms.color_range, common->color_range);
    EXPECT_EQ(encode_parms.cs, common->color_space);
    if (encode_parms.render_size[0] > 0 && encode_parms.render_size[1] > 0) {
      EXPECT_EQ(encode_parms.render_size[0], common->render_width);
      EXPECT_EQ(encode_parms.render_size[1], common->render_height);
    }
    EXPECT_EQ(encode_parms.tile_cols, common->log2_tile_cols);
    EXPECT_EQ(encode_parms.tile_rows, common->log2_tile_rows);

    EXPECT_EQ(VPX_CODEC_OK, res_dec) << decoder->DecodeError();
    return VPX_CODEC_OK == res_dec;
  }

  EncodePerfTestVideo test_video_;

 private:
  EncodeParameters encode_parms;
};

TEST_P(VpxEncoderParmsGetToDecoder, BitstreamParms) {
  init_flags_ = VPX_CODEC_USE_PSNR;

  std::unique_ptr<libvpx_test::VideoSource> video(
      new libvpx_test::Y4mVideoSource(test_video_.name, 0, test_video_.frames));
  ASSERT_NE(video.get(), nullptr);

  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
}

VP9_INSTANTIATE_TEST_SUITE(VpxEncoderParmsGetToDecoder,
                           ::testing::ValuesIn(kVP9EncodeParameterSet),
                           ::testing::ValuesIn(kVP9EncodePerfTestVectors));
}  // namespace
