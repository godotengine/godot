/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/encode_test_driver.h"
#include "test/register_state_check.h"
#include "test/video_source.h"

namespace libvpx_test {
void Encoder::InitEncoder(VideoSource *video) {
  vpx_codec_err_t res;
  const vpx_image_t *img = video->img();

  if (video->img() && !encoder_.priv) {
    cfg_.g_w = img->d_w;
    cfg_.g_h = img->d_h;
    cfg_.g_timebase = video->timebase();
    cfg_.rc_twopass_stats_in = stats_->buf();

    res = vpx_codec_enc_init(&encoder_, CodecInterface(), &cfg_, init_flags_);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();

#if CONFIG_VP9_ENCODER
    if (CodecInterface() == &vpx_codec_vp9_cx_algo) {
      // Default to 1 tile column for VP9.
      const int log2_tile_columns = 0;
      res = vpx_codec_control_(&encoder_, VP9E_SET_TILE_COLUMNS,
                               log2_tile_columns);
      ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
    } else
#endif
    {
#if CONFIG_VP8_ENCODER
      ASSERT_EQ(&vpx_codec_vp8_cx_algo, CodecInterface())
          << "Unknown Codec Interface";
#endif
    }
  }
}

void Encoder::EncodeFrame(VideoSource *video,
                          const vpx_enc_frame_flags_t frame_flags) {
  if (video->img()) {
    EncodeFrameInternal(*video, frame_flags);
  } else {
    Flush();
  }

  // Handle twopass stats
  CxDataIterator iter = GetCxData();

  while (const vpx_codec_cx_pkt_t *pkt = iter.Next()) {
    if (pkt->kind != VPX_CODEC_STATS_PKT) continue;

    stats_->Append(*pkt);
  }
}

void Encoder::EncodeFrameInternal(const VideoSource &video,
                                  const vpx_enc_frame_flags_t frame_flags) {
  vpx_codec_err_t res;
  const vpx_image_t *img = video.img();

  // Handle frame resizing
  if (cfg_.g_w != img->d_w || cfg_.g_h != img->d_h) {
    cfg_.g_w = img->d_w;
    cfg_.g_h = img->d_h;
    res = vpx_codec_enc_config_set(&encoder_, &cfg_);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  // Encode the frame
  API_REGISTER_STATE_CHECK(res = vpx_codec_encode(&encoder_, img, video.pts(),
                                                  video.duration(), frame_flags,
                                                  deadline_));
  ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
}

void Encoder::Flush() {
  const vpx_codec_err_t res =
      vpx_codec_encode(&encoder_, nullptr, 0, 0, 0, deadline_);
  if (!encoder_.priv)
    ASSERT_EQ(VPX_CODEC_ERROR, res) << EncoderError();
  else
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
}

void EncoderTest::InitializeConfig() {
  const vpx_codec_err_t res = codec_->DefaultEncoderConfig(&cfg_, 0);
  dec_cfg_ = vpx_codec_dec_cfg_t();
  ASSERT_EQ(VPX_CODEC_OK, res);
}

void EncoderTest::SetMode(TestMode mode) {
  switch (mode) {
    case kRealTime: deadline_ = VPX_DL_REALTIME; break;

    case kOnePassGood:
    case kTwoPassGood: deadline_ = VPX_DL_GOOD_QUALITY; break;

    case kOnePassBest:
    case kTwoPassBest: deadline_ = VPX_DL_BEST_QUALITY; break;

    default: ASSERT_TRUE(false) << "Unexpected mode " << mode;
  }

  if (mode == kTwoPassGood || mode == kTwoPassBest) {
    passes_ = 2;
  } else {
    passes_ = 1;
  }
}
// The function should return "true" most of the time, therefore no early
// break-out is implemented within the match checking process.
static bool compare_img(const vpx_image_t *img1, const vpx_image_t *img2) {
  bool match = (img1->fmt == img2->fmt) && (img1->cs == img2->cs) &&
               (img1->d_w == img2->d_w) && (img1->d_h == img2->d_h);

  if (!match) return false;

  const unsigned int width_y = img1->d_w;
  const unsigned int height_y = img1->d_h;
  unsigned int i;
  for (i = 0; i < height_y; ++i) {
    match = (memcmp(img1->planes[VPX_PLANE_Y] + i * img1->stride[VPX_PLANE_Y],
                    img2->planes[VPX_PLANE_Y] + i * img2->stride[VPX_PLANE_Y],
                    width_y) == 0) &&
            match;
  }
  const unsigned int width_uv = (img1->d_w + 1) >> 1;
  const unsigned int height_uv = (img1->d_h + 1) >> 1;
  for (i = 0; i < height_uv; ++i) {
    match = (memcmp(img1->planes[VPX_PLANE_U] + i * img1->stride[VPX_PLANE_U],
                    img2->planes[VPX_PLANE_U] + i * img2->stride[VPX_PLANE_U],
                    width_uv) == 0) &&
            match;
  }
  for (i = 0; i < height_uv; ++i) {
    match = (memcmp(img1->planes[VPX_PLANE_V] + i * img1->stride[VPX_PLANE_V],
                    img2->planes[VPX_PLANE_V] + i * img2->stride[VPX_PLANE_V],
                    width_uv) == 0) &&
            match;
  }
  return match;
}

void EncoderTest::MismatchHook(const vpx_image_t * /*img1*/,
                               const vpx_image_t * /*img2*/) {
  ASSERT_TRUE(0) << "Encode/Decode mismatch found";
}

void EncoderTest::RunLoop(VideoSource *video) {
  vpx_codec_dec_cfg_t dec_cfg = vpx_codec_dec_cfg_t();

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
    std::unique_ptr<Encoder> encoder(
        codec_->CreateEncoder(cfg_, deadline_, init_flags_, &stats_));
    ASSERT_NE(encoder.get(), nullptr);

    ASSERT_NO_FATAL_FAILURE(video->Begin());
    encoder->InitEncoder(video);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());

    unsigned long dec_init_flags = 0;  // NOLINT
    // Use fragment decoder if encoder outputs partitions.
    // NOTE: fragment decoder and partition encoder are only supported by VP8.
    if (init_flags_ & VPX_CODEC_USE_OUTPUT_PARTITION) {
      dec_init_flags |= VPX_CODEC_USE_INPUT_FRAGMENTS;
    }
    std::unique_ptr<Decoder> decoder(
        codec_->CreateDecoder(dec_cfg, dec_init_flags));
    bool again;
    for (again = true; again; video->Next()) {
      again = (video->img() != nullptr);

      PreEncodeFrameHook(video);
      PreEncodeFrameHook(video, encoder.get());
      encoder->EncodeFrame(video, frame_flags_);

      PostEncodeFrameHook(encoder.get());

      CxDataIterator iter = encoder->GetCxData();

      bool has_cxdata = false;
      bool has_dxdata = false;
      while (const vpx_codec_cx_pkt_t *pkt = iter.Next()) {
        pkt = MutateEncoderOutputHook(pkt);
        again = true;
        switch (pkt->kind) {
          case VPX_CODEC_CX_FRAME_PKT:
            has_cxdata = true;
            if (decoder != nullptr && DoDecode()) {
              PreDecodeFrameHook(video, decoder.get());
              vpx_codec_err_t res_dec = decoder->DecodeFrame(
                  (const uint8_t *)pkt->data.frame.buf, pkt->data.frame.sz);

              if (!HandleDecodeResult(res_dec, *video, decoder.get())) break;

              has_dxdata = true;
            }
            ASSERT_GE(pkt->data.frame.pts, last_pts);
            last_pts = pkt->data.frame.pts;
            FramePktHook(pkt);
            break;

          case VPX_CODEC_PSNR_PKT: PSNRPktHook(pkt); break;

          case VPX_CODEC_STATS_PKT: StatsPktHook(pkt); break;

          default: break;
        }
      }

      // Flush the decoder when there are no more fragments.
      if ((init_flags_ & VPX_CODEC_USE_OUTPUT_PARTITION) && has_dxdata) {
        const vpx_codec_err_t res_dec = decoder->DecodeFrame(nullptr, 0);
        if (!HandleDecodeResult(res_dec, *video, decoder.get())) break;
      }

      if (has_dxdata && has_cxdata) {
        const vpx_image_t *img_enc = encoder->GetPreviewFrame();
        DxDataIterator dec_iter = decoder->GetDxData();
        const vpx_image_t *img_dec = dec_iter.Next();
        if (img_enc && img_dec) {
          const bool res = compare_img(img_enc, img_dec);
          if (!res) {  // Mismatch
            MismatchHook(img_enc, img_dec);
          }
        }
        if (img_dec) DecompressedFrameHook(*img_dec, video->pts());
      }
      if (!Continue()) break;
    }

    EndPassHook();

    if (!Continue()) break;
  }
}

}  // namespace libvpx_test
