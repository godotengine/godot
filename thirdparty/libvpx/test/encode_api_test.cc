/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "gtest/gtest.h"
#include "test/acm_random.h"
#include "test/video_source.h"
#include "test/y4m_video_source.h"

#include "./vpx_config.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_encoder.h"
#include "vpx/vpx_image.h"

namespace {

vpx_codec_iface_t *kCodecIfaces[] = {
#if CONFIG_VP8_ENCODER
  &vpx_codec_vp8_cx_algo,
#endif
#if CONFIG_VP9_ENCODER
  &vpx_codec_vp9_cx_algo,
#endif
};

bool IsVP9(vpx_codec_iface_t *iface) {
  static const char kVP9Name[] = "WebM Project VP9";
  return strncmp(kVP9Name, vpx_codec_iface_name(iface), sizeof(kVP9Name) - 1) ==
         0;
}

void *Memset16(void *dest, int val, size_t length) {
  uint16_t *dest16 = reinterpret_cast<uint16_t *>(dest);
  for (size_t i = 0; i < length; i++) {
    *dest16++ = val;
  }
  return dest;
}

vpx_image_t *CreateImage(vpx_bit_depth_t bit_depth, vpx_img_fmt_t fmt,
                         unsigned int width, unsigned int height) {
  assert(fmt != VPX_IMG_FMT_NV12);
  if (bit_depth > VPX_BITS_8) {
    fmt = static_cast<vpx_img_fmt_t>(fmt | VPX_IMG_FMT_HIGHBITDEPTH);
  }
  vpx_image_t *image = vpx_img_alloc(nullptr, fmt, width, height, 1);
  if (!image) return image;

  const int val = 1 << (bit_depth - 1);
  const unsigned int uv_h =
      (image->d_h + image->y_chroma_shift) >> image->y_chroma_shift;
  const unsigned int uv_w =
      (image->d_w + image->x_chroma_shift) >> image->x_chroma_shift;
  if (bit_depth > VPX_BITS_8) {
    for (unsigned int i = 0; i < image->d_h; ++i) {
      Memset16(image->planes[0] + i * image->stride[0], val, image->d_w);
    }
    for (unsigned int i = 0; i < uv_h; ++i) {
      Memset16(image->planes[1] + i * image->stride[1], val, uv_w);
      Memset16(image->planes[2] + i * image->stride[2], val, uv_w);
    }
  } else {
    for (unsigned int i = 0; i < image->d_h; ++i) {
      memset(image->planes[0] + i * image->stride[0], val, image->d_w);
    }
    for (unsigned int i = 0; i < uv_h; ++i) {
      memset(image->planes[1] + i * image->stride[1], val, uv_w);
      memset(image->planes[2] + i * image->stride[2], val, uv_w);
    }
  }

  return image;
}

void InitCodec(vpx_codec_iface_t &iface, int width, int height,
               vpx_codec_ctx_t *enc, vpx_codec_enc_cfg_t *cfg) {
  cfg->g_w = width;
  cfg->g_h = height;
  cfg->g_lag_in_frames = 0;
  cfg->g_pass = VPX_RC_ONE_PASS;
  ASSERT_EQ(vpx_codec_enc_init(enc, &iface, cfg, 0), VPX_CODEC_OK);

  ASSERT_EQ(vpx_codec_control_(enc, VP8E_SET_CPUUSED, 2), VPX_CODEC_OK);
}

// Encodes 1 frame of size |cfg.g_w| x |cfg.g_h| setting |enc|'s configuration
// to |cfg|.
void EncodeWithConfig(const vpx_codec_enc_cfg_t &cfg, vpx_codec_ctx_t *enc) {
  libvpx_test::DummyVideoSource video;
  video.SetSize(cfg.g_w, cfg.g_h);
  video.Begin();
  EXPECT_EQ(vpx_codec_enc_config_set(enc, &cfg), VPX_CODEC_OK)
      << vpx_codec_error_detail(enc);

  EXPECT_EQ(vpx_codec_encode(enc, video.img(), video.pts(), video.duration(),
                             /*flags=*/0, VPX_DL_GOOD_QUALITY),
            VPX_CODEC_OK)
      << vpx_codec_error_detail(enc);
}

TEST(EncodeAPI, InvalidParams) {
  uint8_t buf[1] = { 0 };
  vpx_image_t img;
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  EXPECT_EQ(&img, vpx_img_wrap(&img, VPX_IMG_FMT_I420, 1, 1, 1, buf));

  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_enc_init(nullptr, nullptr, nullptr, 0));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_enc_init(&enc, nullptr, nullptr, 0));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_encode(nullptr, nullptr, 0, 0, 0, 0));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_encode(nullptr, &img, 0, 0, 0, 0));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM, vpx_codec_destroy(nullptr));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_enc_config_default(nullptr, nullptr, 0));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_enc_config_default(nullptr, &cfg, 0));
  EXPECT_NE(vpx_codec_error(nullptr), nullptr);

  for (const auto *iface : kCodecIfaces) {
    SCOPED_TRACE(vpx_codec_iface_name(iface));
    EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
              vpx_codec_enc_init(nullptr, iface, nullptr, 0));
    EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
              vpx_codec_enc_init(&enc, iface, nullptr, 0));
    EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
              vpx_codec_enc_config_default(iface, &cfg, 1));

    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_enc_config_default(iface, &cfg, 0));
    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_enc_init(&enc, iface, &cfg, 0));
    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_encode(&enc, nullptr, 0, 0, 0, 0));

    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_destroy(&enc));
  }
}

TEST(EncodeAPI, HighBitDepthCapability) {
// VP8 should not claim VP9 HBD as a capability.
#if CONFIG_VP8_ENCODER
  const vpx_codec_caps_t vp8_caps = vpx_codec_get_caps(&vpx_codec_vp8_cx_algo);
  EXPECT_EQ(vp8_caps & VPX_CODEC_CAP_HIGHBITDEPTH, 0);
#endif

#if CONFIG_VP9_ENCODER
  const vpx_codec_caps_t vp9_caps = vpx_codec_get_caps(&vpx_codec_vp9_cx_algo);
#if CONFIG_VP9_HIGHBITDEPTH
  EXPECT_EQ(vp9_caps & VPX_CODEC_CAP_HIGHBITDEPTH, VPX_CODEC_CAP_HIGHBITDEPTH);
#else
  EXPECT_EQ(vp9_caps & VPX_CODEC_CAP_HIGHBITDEPTH, 0);
#endif
#endif
}

#if CONFIG_VP8_ENCODER
TEST(EncodeAPI, ImageSizeSetting) {
  const int width = 711;
  const int height = 360;
  const int bps = 12;
  vpx_image_t img;
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;
  uint8_t *img_buf = reinterpret_cast<uint8_t *>(
      calloc(width * height * bps / 8, sizeof(*img_buf)));
  vpx_codec_enc_config_default(vpx_codec_vp8_cx(), &cfg, 0);

  cfg.g_w = width;
  cfg.g_h = height;

  vpx_img_wrap(&img, VPX_IMG_FMT_I420, width, height, 1, img_buf);

  vpx_codec_enc_init(&enc, vpx_codec_vp8_cx(), &cfg, 0);

  EXPECT_EQ(VPX_CODEC_OK, vpx_codec_encode(&enc, &img, 0, 1, 0, 0));

  free(img_buf);

  vpx_codec_destroy(&enc);
}

// Verifies the fix for a float-cast-overflow in vp8_change_config().
//
// Causes cpi->framerate to become the largest possible value (10,000,000) in
// VP8 by setting cfg.g_timebase to 1/10000000 and passing a duration of 1 to
// vpx_codec_encode().
TEST(EncodeAPI, HugeFramerateVp8) {
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
  cfg.g_w = 271;
  cfg.g_h = 1080;
  cfg.g_timebase.num = 1;
  // Largest value (VP8's TICKS_PER_SEC) such that frame duration is nonzero (1
  // tick).
  cfg.g_timebase.den = 10000000;
  cfg.g_pass = VPX_RC_ONE_PASS;
  cfg.g_lag_in_frames = 0;
  cfg.rc_end_usage = VPX_CBR;

  vpx_codec_ctx_t enc;
  // Before we encode the first frame, cpi->framerate is set to a guess (the
  // reciprocal of cfg.g_timebase). If this guess doesn't seem reasonable
  // (> 180), cpi->framerate is set to 30.
  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_CPUUSED, -12), VPX_CODEC_OK);

  vpx_image_t *const image =
      vpx_img_alloc(nullptr, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h, 1);
  ASSERT_NE(image, nullptr);

  for (unsigned int i = 0; i < image->d_h; ++i) {
    memset(image->planes[0] + i * image->stride[0], 128, image->d_w);
  }
  const unsigned int uv_h = (image->d_h + 1) / 2;
  const unsigned int uv_w = (image->d_w + 1) / 2;
  for (unsigned int i = 0; i < uv_h; ++i) {
    memset(image->planes[1] + i * image->stride[1], 128, uv_w);
    memset(image->planes[2] + i * image->stride[2], 128, uv_w);
  }

  // Encode a frame.
  // Up to this point cpi->framerate is 30. Now pass a duration of only 1. This
  // causes cpi->framerate to become 10,000,000.
  ASSERT_EQ(vpx_codec_encode(&enc, image, 0, 1, 0, VPX_DL_REALTIME),
            VPX_CODEC_OK);

  // Change to the same config. Since cpi->framerate is now huge, when it is
  // used to calculate raw_target_rate (bit rate of uncompressed frames), the
  // result is likely to overflow an unsigned int.
  ASSERT_EQ(vpx_codec_enc_config_set(&enc, &cfg), VPX_CODEC_OK);

  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

// A test that reproduces https://crbug.com/webm/1831.
TEST(EncodeAPI, RandomPixelsVp8) {
  // Initialize libvpx encoder
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.rc_target_bitrate = 2000;
  cfg.g_w = 1280;
  cfg.g_h = 720;

  vpx_codec_ctx_t enc;
  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  // Generate random frame data and encode
  libvpx_test::RandomVideoSource video;
  video.SetSize(cfg.g_w, cfg.g_h);
  video.SetImageFormat(VPX_IMG_FMT_I420);
  video.Begin();
  ASSERT_EQ(vpx_codec_encode(&enc, video.img(), video.pts(), video.duration(),
                             /*flags=*/0, VPX_DL_BEST_QUALITY),
            VPX_CODEC_OK);

  // Destroy libvpx encoder
  vpx_codec_destroy(&enc);
}

TEST(EncodeAPI, ChangeToL1T3AndSetBitrateVp8) {
  // Initialize libvpx encoder
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_threads = 1;
  cfg.g_profile = 0;
  cfg.g_w = 1;
  cfg.g_h = 64;
  cfg.g_bit_depth = VPX_BITS_8;
  cfg.g_input_bit_depth = 8;
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = 1000000;
  cfg.g_pass = VPX_RC_ONE_PASS;
  cfg.g_lag_in_frames = 0;
  cfg.rc_dropframe_thresh = 0;  // Don't drop frames
  cfg.rc_resize_allowed = 0;
  cfg.rc_end_usage = VPX_VBR;
  cfg.rc_target_bitrate = 10;
  cfg.rc_min_quantizer = 2;
  cfg.rc_max_quantizer = 58;
  cfg.kf_mode = VPX_KF_AUTO;
  cfg.kf_min_dist = 0;
  cfg.kf_max_dist = 10000;

  vpx_codec_ctx_t enc;
  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_CPUUSED, -6), VPX_CODEC_OK);

  // Generate random frame data and encode
  uint8_t img[1 * 64 * 3 / 2];
  libvpx_test::ACMRandom rng;
  for (size_t i = 0; i < sizeof(img); ++i) {
    img[i] = rng.Rand8();
  }
  vpx_image_t img_wrapper;
  ASSERT_EQ(
      vpx_img_wrap(&img_wrapper, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h, 1, img),
      &img_wrapper);
  vpx_enc_frame_flags_t flags = VPX_EFLAG_FORCE_KF;
  ASSERT_EQ(
      vpx_codec_encode(&enc, &img_wrapper, 0, 500000, flags, VPX_DL_REALTIME),
      VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_encode(&enc, nullptr, -1, 0, 0, 0), VPX_CODEC_OK);

  cfg.rc_target_bitrate = 4294967;
  // Set the scalability mode to L1T3.
  cfg.ts_number_layers = 3;
  cfg.ts_periodicity = 4;
  cfg.ts_layer_id[0] = 0;
  cfg.ts_layer_id[1] = 2;
  cfg.ts_layer_id[2] = 1;
  cfg.ts_layer_id[3] = 2;
  cfg.ts_rate_decimator[0] = 4;
  cfg.ts_rate_decimator[1] = 2;
  cfg.ts_rate_decimator[2] = 1;
  // Bitrate allocation L0: 50% L1: 20% L2: 30%
  cfg.layer_target_bitrate[0] = cfg.ts_target_bitrate[0] =
      50 * cfg.rc_target_bitrate / 100;
  cfg.layer_target_bitrate[1] = cfg.ts_target_bitrate[1] =
      70 * cfg.rc_target_bitrate / 100;
  cfg.layer_target_bitrate[2] = cfg.ts_target_bitrate[2] =
      cfg.rc_target_bitrate;
  cfg.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_0212;
  cfg.g_error_resilient = VPX_ERROR_RESILIENT_DEFAULT;
  ASSERT_EQ(vpx_codec_enc_config_set(&enc, &cfg), VPX_CODEC_OK);

  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_TEMPORAL_LAYER_ID, 2),
            VPX_CODEC_OK);

  constexpr vpx_enc_frame_flags_t VP8_UPDATE_NOTHING =
      VP8_EFLAG_NO_UPD_ARF | VP8_EFLAG_NO_UPD_GF | VP8_EFLAG_NO_UPD_LAST;
  // Layer 2: only reference last frame, no updates
  // It only depends on layer 0
  flags = VP8_UPDATE_NOTHING | VP8_EFLAG_NO_REF_ARF | VP8_EFLAG_NO_REF_GF;
  ASSERT_EQ(
      vpx_codec_encode(&enc, &img_wrapper, 0, 500000, flags, VPX_DL_REALTIME),
      VPX_CODEC_OK);

  // Destroy libvpx encoder
  vpx_codec_destroy(&enc);
}

// Emulates the WebCodecs VideoEncoder interface.
class VP8Encoder {
 public:
  explicit VP8Encoder(int speed) : speed_(speed) {}
  ~VP8Encoder();

  void Configure(unsigned int threads, unsigned int width, unsigned int height,
                 vpx_rc_mode end_usage, vpx_enc_deadline_t deadline);
  void Encode(bool key_frame);

 private:
  const int speed_;
  bool initialized_ = false;
  vpx_codec_enc_cfg_t cfg_;
  vpx_codec_ctx_t enc_;
  int frame_index_ = 0;
  vpx_enc_deadline_t deadline_ = 0;
};

VP8Encoder::~VP8Encoder() {
  if (initialized_) {
    EXPECT_EQ(vpx_codec_destroy(&enc_), VPX_CODEC_OK);
  }
}

void VP8Encoder::Configure(unsigned int threads, unsigned int width,
                           unsigned int height, vpx_rc_mode end_usage,
                           vpx_enc_deadline_t deadline) {
  deadline_ = deadline;

  if (!initialized_) {
    vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
    ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg_, /*usage=*/0),
              VPX_CODEC_OK);
    cfg_.g_threads = threads;
    cfg_.g_w = width;
    cfg_.g_h = height;
    cfg_.g_timebase.num = 1;
    cfg_.g_timebase.den = 1000 * 1000;  // microseconds
    cfg_.g_pass = VPX_RC_ONE_PASS;
    cfg_.g_lag_in_frames = 0;
    cfg_.rc_end_usage = end_usage;
    cfg_.rc_min_quantizer = 2;
    cfg_.rc_max_quantizer = 58;
    ASSERT_EQ(vpx_codec_enc_init(&enc_, iface, &cfg_, 0), VPX_CODEC_OK);
    ASSERT_EQ(vpx_codec_control(&enc_, VP8E_SET_CPUUSED, speed_), VPX_CODEC_OK);
    initialized_ = true;
    return;
  }

  cfg_.g_threads = threads;
  cfg_.g_w = width;
  cfg_.g_h = height;
  cfg_.rc_end_usage = end_usage;
  ASSERT_EQ(vpx_codec_enc_config_set(&enc_, &cfg_), VPX_CODEC_OK)
      << vpx_codec_error_detail(&enc_);
}

void VP8Encoder::Encode(bool key_frame) {
  assert(initialized_);
  const vpx_codec_cx_pkt_t *pkt;
  vpx_image_t *image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg_.g_w, cfg_.g_h);
  ASSERT_NE(image, nullptr);
  const vpx_enc_frame_flags_t flags = key_frame ? VPX_EFLAG_FORCE_KF : 0;
  ASSERT_EQ(vpx_codec_encode(&enc_, image, frame_index_, 1, flags, deadline_),
            VPX_CODEC_OK);
  ++frame_index_;
  vpx_codec_iter_t iter = nullptr;
  while ((pkt = vpx_codec_get_cx_data(&enc_, &iter)) != nullptr) {
    ASSERT_EQ(pkt->kind, VPX_CODEC_CX_FRAME_PKT);
    if (key_frame) {
      ASSERT_EQ(pkt->data.frame.flags & VPX_FRAME_IS_KEY, VPX_FRAME_IS_KEY);
    }
  }
  vpx_img_free(image);
}

// This is the reproducer testcase for crbug.com/324459561. However,
// just running this test is not enough to reproduce the bug. We also
// need to send signals to the test.
TEST(EncodeAPI, Chromium324459561) {
  VP8Encoder encoder(-12);

  encoder.Configure(11, 1685, 652, VPX_CBR, VPX_DL_REALTIME);

  encoder.Encode(true);
  encoder.Encode(true);
  encoder.Encode(true);

  encoder.Configure(0, 1685, 1, VPX_VBR, VPX_DL_REALTIME);
}

TEST(EncodeAPI, VP8GlobalHeaders) {
  constexpr int kWidth = 320;
  constexpr int kHeight = 240;

  vpx_codec_enc_cfg_t cfg = {};
  struct Encoder {
    ~Encoder() { EXPECT_EQ(vpx_codec_destroy(&ctx), VPX_CODEC_OK); }
    vpx_codec_ctx_t ctx = {};
  } enc;

  ASSERT_EQ(vpx_codec_enc_config_default(vpx_codec_vp8_cx(), &cfg, 0),
            VPX_CODEC_OK);
  ASSERT_NO_FATAL_FAILURE(
      InitCodec(*vpx_codec_vp8_cx(), kWidth, kHeight, &enc.ctx, &cfg));
  EXPECT_EQ(vpx_codec_get_global_headers(&enc.ctx), nullptr);
  EXPECT_NO_FATAL_FAILURE(EncodeWithConfig(cfg, &enc.ctx));
  EXPECT_EQ(vpx_codec_get_global_headers(&enc.ctx), nullptr);
}

// Encode a few frames for 2 temporal layers realtime mode.
// Set duration to be very large on first frame, much smaller
// on second frames, with the timestamp (pts) parameter very
// inconsistent with the duration (i.e, pts != prev_pts + duration).
// This reproduces the issue found in the bug: 431520320.
TEST(EncodeAPI, Vp8ChromiumIssue431520320) {
  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 320;
  cfg.g_h = 240;
  cfg.g_lag_in_frames = 0;
  cfg.rc_target_bitrate = 500;

  // 2-layers, 2-frame period.
  int ids[2] = { 0, 1 };
  cfg.ts_periodicity = 2;
  cfg.ts_number_layers = 2;
  cfg.ts_rate_decimator[0] = 2;
  cfg.ts_rate_decimator[1] = 1;
  cfg.ts_target_bitrate[0] = 300;
  cfg.ts_target_bitrate[1] = 500;
  memcpy(cfg.ts_layer_id, ids, sizeof(ids));

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  // Create input image.
  vpx_image_t *const image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  // Encode first frame.
  ASSERT_EQ(
      vpx_codec_encode(&enc, image, 0, /*duration=*/800000, 0, VPX_DL_REALTIME),
      VPX_CODEC_OK);

  // Encode second frame.
  ASSERT_EQ(vpx_codec_encode(&enc, image, 40000, /*duration=*/40000, 0,
                             VPX_DL_REALTIME),
            VPX_CODEC_OK);

  // Encode third frame.
  ASSERT_EQ(vpx_codec_encode(&enc, image, 80000, /*duration=*/40000, 0,
                             VPX_DL_REALTIME),
            VPX_CODEC_OK);

  // Free resources.
  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, AomediaIssue3509VbrMinSection2PercentVP8) {
  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 1920;
  cfg.g_h = 1080;
  cfg.g_lag_in_frames = 0;
  cfg.rc_target_bitrate = 1000000;
  // Set this to more than 1 percent to cause a signed integer overflow in the
  // multiplication cpi->av_per_frame_bandwidth *
  // cpi->oxcf.two_pass_vbrmin_section in vp8_new_framerate() if the
  // multiplication is done in the `int` type.
  cfg.rc_2pass_vbr_minsection_pct = 2;

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  // Create input image.
  vpx_image_t *const image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  // Encode frame.
  // `duration` can go as high as 300, but the UBSan error is gone if
  // `duration` is 301 or higher.
  ASSERT_EQ(
      vpx_codec_encode(&enc, image, 0, /*duration=*/300, 0, VPX_DL_REALTIME),
      VPX_CODEC_OK);

  // Free resources.
  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, AomediaIssue3509VbrMinSection101PercentVP8) {
  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 1920;
  cfg.g_h = 1080;
  cfg.g_lag_in_frames = 0;
  cfg.rc_target_bitrate = 1000000;
  // Set this to more than 100 percent to cause an error when vbr_min_bits is
  // cast to `int` in vp8_new_framerate() if vbr_min_bits is not clamped to
  // INT_MAX.
  cfg.rc_2pass_vbr_minsection_pct = 101;

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  // Create input image.
  vpx_image_t *const image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  // Encode frame.
  // `duration` can go as high as 300, but the UBSan error is gone if
  // `duration` is 301 or higher.
  ASSERT_EQ(
      vpx_codec_encode(&enc, image, 0, /*duration=*/300, 0, VPX_DL_REALTIME),
      VPX_CODEC_OK);

  // Free resources.
  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, OssFuzz69100) {
  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 64;
  cfg.g_h = 64;
  cfg.g_lag_in_frames = 25;
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = 6240592;
  cfg.rc_target_bitrate = 1202607620;
  cfg.kf_max_dist = 24377;

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_CPUUSED, 1), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_ARNR_MAXFRAMES, 0), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_ARNR_STRENGTH, 3), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control_(&enc, VP8E_SET_ARNR_TYPE, 3),
            VPX_CODEC_OK);  // deprecated
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_NOISE_SENSITIVITY, 0),
            VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_TOKEN_PARTITIONS, 0),
            VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_STATIC_THRESHOLD, 0),
            VPX_CODEC_OK);

  libvpx_test::RandomVideoSource video;
  video.set_limit(30);
  video.SetSize(cfg.g_w, cfg.g_h);
  video.SetImageFormat(VPX_IMG_FMT_I420);
  video.Begin();
  do {
    ASSERT_EQ(vpx_codec_encode(&enc, video.img(), video.pts(), video.duration(),
                               /*flags=*/0, VPX_DL_GOOD_QUALITY),
              VPX_CODEC_OK);
    video.Next();
  } while (video.img() != nullptr);

  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

void EncodeOssFuzz69906(int cpu_used, vpx_enc_deadline_t deadline) {
  char str[80];
  snprintf(str, sizeof(str), "cpu_used: %d deadline: %d", cpu_used,
           static_cast<int>(deadline));
  SCOPED_TRACE(str);

  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp8_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 4097;
  cfg.g_h = 16;
  cfg.rc_target_bitrate = 1237084865;
  cfg.kf_max_dist = 4336;

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_CPUUSED, cpu_used), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_ARNR_MAXFRAMES, 0), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_ARNR_STRENGTH, 3), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control_(&enc, VP8E_SET_ARNR_TYPE, 3),
            VPX_CODEC_OK);  // deprecated
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_NOISE_SENSITIVITY, 0),
            VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_TOKEN_PARTITIONS, 0),
            VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_STATIC_THRESHOLD, 0),
            VPX_CODEC_OK);

  libvpx_test::Y4mVideoSource video("repro-oss-fuzz-69906.y4m", /*start=*/0,
                                    /*limit=*/3);
  video.Begin();
  do {
    ASSERT_EQ(vpx_codec_encode(&enc, video.img(), video.pts(), video.duration(),
                               /*flags=*/0, deadline),
              VPX_CODEC_OK);
    video.Next();
  } while (video.img() != nullptr);

  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, OssFuzz69906) {
  // Note the original bug report was for speed 1, good quality. The remainder
  // of the settings are for added coverage.
  for (int cpu_used = 0; cpu_used <= 5; ++cpu_used) {
    EncodeOssFuzz69906(cpu_used, VPX_DL_GOOD_QUALITY);
  }

  for (int cpu_used = -16; cpu_used <= -5; ++cpu_used) {
    EncodeOssFuzz69906(cpu_used, VPX_DL_REALTIME);
  }
}
#endif  // CONFIG_VP8_ENCODER

// Set up 2 spatial streams with 2 temporal layers per stream, and generate
// invalid configuration by setting the temporal layer rate allocation
// (ts_target_bitrate[]) to 0 for both layers. This should fail independent of
// CONFIG_MULTI_RES_ENCODING.
TEST(EncodeAPI, MultiResEncode) {
  const int width = 1280;
  const int height = 720;
  const int width_down = width / 2;
  const int height_down = height / 2;
  const int target_bitrate = 1000;
  const int framerate = 30;

  for (const auto *iface : kCodecIfaces) {
    vpx_codec_ctx_t enc[2];
    vpx_codec_enc_cfg_t cfg[2];
    vpx_rational_t dsf[2] = { { 2, 1 }, { 2, 1 } };

    memset(enc, 0, sizeof(enc));

    for (int i = 0; i < 2; i++) {
      vpx_codec_enc_config_default(iface, &cfg[i], 0);
    }

    /* Highest-resolution encoder settings */
    cfg[0].g_w = width;
    cfg[0].g_h = height;
    cfg[0].rc_dropframe_thresh = 0;
    cfg[0].rc_end_usage = VPX_CBR;
    cfg[0].rc_resize_allowed = 0;
    cfg[0].rc_min_quantizer = 2;
    cfg[0].rc_max_quantizer = 56;
    cfg[0].rc_undershoot_pct = 100;
    cfg[0].rc_overshoot_pct = 15;
    cfg[0].rc_buf_initial_sz = 500;
    cfg[0].rc_buf_optimal_sz = 600;
    cfg[0].rc_buf_sz = 1000;
    cfg[0].g_error_resilient = 1; /* Enable error resilient mode */
    cfg[0].g_lag_in_frames = 0;

    cfg[0].kf_mode = VPX_KF_AUTO;
    cfg[0].kf_min_dist = 3000;
    cfg[0].kf_max_dist = 3000;

    cfg[0].rc_target_bitrate = target_bitrate; /* Set target bitrate */
    cfg[0].g_timebase.num = 1;                 /* Set fps */
    cfg[0].g_timebase.den = framerate;

    cfg[1] = cfg[0];
    cfg[1].rc_target_bitrate = 500;
    cfg[1].g_w = width_down;
    cfg[1].g_h = height_down;

    for (int i = 0; i < 2; i++) {
      cfg[i].ts_number_layers = 2;
      cfg[i].ts_periodicity = 2;
      cfg[i].ts_rate_decimator[0] = 2;
      cfg[i].ts_rate_decimator[1] = 1;
      cfg[i].ts_layer_id[0] = 0;
      cfg[i].ts_layer_id[1] = 1;
      // Invalid parameters.
      cfg[i].ts_target_bitrate[0] = 0;
      cfg[i].ts_target_bitrate[1] = 0;
    }

    // VP9 should report incapable, VP8 invalid for all configurations.
    EXPECT_EQ(IsVP9(iface) ? VPX_CODEC_INCAPABLE : VPX_CODEC_INVALID_PARAM,
              vpx_codec_enc_init_multi(&enc[0], iface, &cfg[0], 2, 0, &dsf[0]));

    for (int i = 0; i < 2; i++) {
      vpx_codec_destroy(&enc[i]);
    }
  }
}

TEST(EncodeAPI, SetRoi) {
  static struct {
    vpx_codec_iface_t *iface;
    int ctrl_id;
  } kCodecs[] = {
#if CONFIG_VP8_ENCODER
    { &vpx_codec_vp8_cx_algo, VP8E_SET_ROI_MAP },
#endif
#if CONFIG_VP9_ENCODER
    { &vpx_codec_vp9_cx_algo, VP9E_SET_ROI_MAP },
#endif
  };
  constexpr int kWidth = 64;
  constexpr int kHeight = 64;

  for (const auto &codec : kCodecs) {
    SCOPED_TRACE(vpx_codec_iface_name(codec.iface));
    vpx_codec_ctx_t enc;
    vpx_codec_enc_cfg_t cfg;

    EXPECT_EQ(vpx_codec_enc_config_default(codec.iface, &cfg, 0), VPX_CODEC_OK);
    cfg.g_w = kWidth;
    cfg.g_h = kHeight;
    EXPECT_EQ(vpx_codec_enc_init(&enc, codec.iface, &cfg, 0), VPX_CODEC_OK);

    vpx_roi_map_t roi = {};
    uint8_t roi_map[kWidth * kHeight] = {};
    if (IsVP9(codec.iface)) {
      roi.rows = (cfg.g_w + 7) >> 3;
      roi.cols = (cfg.g_h + 7) >> 3;
    } else {
      roi.rows = (cfg.g_w + 15) >> 4;
      roi.cols = (cfg.g_h + 15) >> 4;
    }
    EXPECT_EQ(vpx_codec_control_(&enc, codec.ctrl_id, &roi), VPX_CODEC_OK);

    roi.roi_map = roi_map;
    // VP8 only. This value isn't range checked.
    roi.static_threshold[1] = 1000;
    roi.static_threshold[2] = UINT_MAX / 2 + 1;
    roi.static_threshold[3] = UINT_MAX;

    for (const auto delta : { -63, -1, 0, 1, 63 }) {
      for (int i = 0; i < 8; ++i) {
        roi.delta_q[i] = delta;
        roi.delta_lf[i] = delta;
        // VP9 only.
        roi.skip[i] ^= 1;
        roi.ref_frame[i] = (roi.ref_frame[i] + 1) % 4;
        EXPECT_EQ(vpx_codec_control_(&enc, codec.ctrl_id, &roi), VPX_CODEC_OK);
      }
    }

    vpx_codec_err_t expected_error;
    for (const auto delta : { -64, 64, INT_MIN, INT_MAX }) {
      expected_error = VPX_CODEC_INVALID_PARAM;
      for (int i = 0; i < 8; ++i) {
        roi.delta_q[i] = delta;
        // The max segment count for VP8 is 4, the remainder of the entries are
        // ignored.
        if (i >= 4 && !IsVP9(codec.iface)) expected_error = VPX_CODEC_OK;

        EXPECT_EQ(vpx_codec_control_(&enc, codec.ctrl_id, &roi), expected_error)
            << "delta_q[" << i << "]: " << delta;
        roi.delta_q[i] = 0;

        roi.delta_lf[i] = delta;
        EXPECT_EQ(vpx_codec_control_(&enc, codec.ctrl_id, &roi), expected_error)
            << "delta_lf[" << i << "]: " << delta;
        roi.delta_lf[i] = 0;
      }
    }

    // VP8 should ignore skip[] and ref_frame[] values.
    expected_error =
        IsVP9(codec.iface) ? VPX_CODEC_INVALID_PARAM : VPX_CODEC_OK;
    for (const auto skip : { -2, 2, INT_MIN, INT_MAX }) {
      for (int i = 0; i < 8; ++i) {
        roi.skip[i] = skip;
        EXPECT_EQ(vpx_codec_control_(&enc, codec.ctrl_id, &roi), expected_error)
            << "skip[" << i << "]: " << skip;
        roi.skip[i] = 0;
      }
    }

    // VP9 allows negative values to be used to disable segmentation.
    for (int ref_frame = -3; ref_frame < 0; ++ref_frame) {
      for (int i = 0; i < 8; ++i) {
        roi.ref_frame[i] = ref_frame;
        EXPECT_EQ(vpx_codec_control_(&enc, codec.ctrl_id, &roi), VPX_CODEC_OK)
            << "ref_frame[" << i << "]: " << ref_frame;
        roi.ref_frame[i] = 0;
      }
    }

    for (const auto ref_frame : { 4, INT_MIN, INT_MAX }) {
      for (int i = 0; i < 8; ++i) {
        roi.ref_frame[i] = ref_frame;
        EXPECT_EQ(vpx_codec_control_(&enc, codec.ctrl_id, &roi), expected_error)
            << "ref_frame[" << i << "]: " << ref_frame;
        roi.ref_frame[i] = 0;
      }
    }

    EXPECT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
  }
}

TEST(EncodeAPI, ConfigChangeThreadCount) {
  constexpr int kWidth = 1920;
  constexpr int kHeight = 1080;

  for (const auto *iface : kCodecIfaces) {
    SCOPED_TRACE(vpx_codec_iface_name(iface));
    for (int i = 0; i < (IsVP9(iface) ? 2 : 1); ++i) {
      vpx_codec_enc_cfg_t cfg = {};
      struct Encoder {
        ~Encoder() { EXPECT_EQ(vpx_codec_destroy(&ctx), VPX_CODEC_OK); }
        vpx_codec_ctx_t ctx = {};
      } enc;

      ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
      EXPECT_NO_FATAL_FAILURE(
          InitCodec(*iface, kWidth, kHeight, &enc.ctx, &cfg));
      if (IsVP9(iface)) {
        EXPECT_EQ(vpx_codec_control_(&enc.ctx, VP9E_SET_TILE_COLUMNS, 6),
                  VPX_CODEC_OK);
        EXPECT_EQ(vpx_codec_control_(&enc.ctx, VP9E_SET_ROW_MT, i),
                  VPX_CODEC_OK);
      }

      for (const auto threads : { 1, 4, 8, 6, 2, 1 }) {
        cfg.g_threads = threads;
        EXPECT_NO_FATAL_FAILURE(EncodeWithConfig(cfg, &enc.ctx))
            << "iteration: " << i << " threads: " << threads;
      }
    }
  }
}

TEST(EncodeAPI, ConfigResizeChangeThreadCount) {
  constexpr int kInitWidth = 1024;
  constexpr int kInitHeight = 1024;

  for (const auto *iface : kCodecIfaces) {
    SCOPED_TRACE(vpx_codec_iface_name(iface));
    for (int i = 0; i < (IsVP9(iface) ? 2 : 1); ++i) {
      vpx_codec_enc_cfg_t cfg = {};
      struct Encoder {
        ~Encoder() { EXPECT_EQ(vpx_codec_destroy(&ctx), VPX_CODEC_OK); }
        vpx_codec_ctx_t ctx = {};
      } enc;

      ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
      // Start in threaded mode to ensure resolution and thread related
      // allocations are updated correctly across changes in resolution and
      // thread counts. See https://crbug.com/1486441.
      cfg.g_threads = 4;
      EXPECT_NO_FATAL_FAILURE(
          InitCodec(*iface, kInitWidth, kInitHeight, &enc.ctx, &cfg));
      if (IsVP9(iface)) {
        EXPECT_EQ(vpx_codec_control_(&enc.ctx, VP9E_SET_TILE_COLUMNS, 6),
                  VPX_CODEC_OK);
        EXPECT_EQ(vpx_codec_control_(&enc.ctx, VP9E_SET_ROW_MT, i),
                  VPX_CODEC_OK);
      }

      cfg.g_w = 1000;
      cfg.g_h = 608;
      EXPECT_EQ(vpx_codec_enc_config_set(&enc.ctx, &cfg), VPX_CODEC_OK)
          << vpx_codec_error_detail(&enc.ctx);

      cfg.g_w = 1000;
      cfg.g_h = 720;

      for (const auto threads : { 1, 4, 8, 6, 2, 1 }) {
        cfg.g_threads = threads;
        EXPECT_NO_FATAL_FAILURE(EncodeWithConfig(cfg, &enc.ctx))
            << "iteration: " << i << " threads: " << threads;
      }
    }
  }
}

TEST(EncodeAPI, ConfigResizeBiggerAfterInit) {
  for (const auto *iface : kCodecIfaces) {
    SCOPED_TRACE(vpx_codec_iface_name(iface));
    vpx_codec_enc_cfg_t cfg;
    vpx_codec_ctx_t enc;

    ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
    EXPECT_NO_FATAL_FAILURE(InitCodec(*iface, 1, 1, &enc, &cfg));

    cfg.g_w = 1920;
    cfg.g_h = 1;
    EXPECT_EQ(vpx_codec_enc_config_set(&enc, &cfg),
              IsVP9(iface) ? VPX_CODEC_OK : VPX_CODEC_INVALID_PARAM);

    EXPECT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
  }
}

TEST(EncodeAPI, ConfigResizeBiggerAfterEncode) {
  for (const auto *iface : kCodecIfaces) {
    SCOPED_TRACE(vpx_codec_iface_name(iface));
    vpx_codec_enc_cfg_t cfg;
    vpx_codec_ctx_t enc;

    ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
    EXPECT_NO_FATAL_FAILURE(InitCodec(*iface, 1, 1, &enc, &cfg));
    EXPECT_NO_FATAL_FAILURE(EncodeWithConfig(cfg, &enc));

    cfg.g_w = 1920;
    cfg.g_h = 1;
    EXPECT_EQ(vpx_codec_enc_config_set(&enc, &cfg),
              IsVP9(iface) ? VPX_CODEC_OK : VPX_CODEC_INVALID_PARAM);

    cfg.g_w = 1920;
    cfg.g_h = 1080;
    EXPECT_EQ(vpx_codec_enc_config_set(&enc, &cfg),
              IsVP9(iface) ? VPX_CODEC_OK : VPX_CODEC_INVALID_PARAM);

    EXPECT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
  }
}

TEST(EncodeAPI, PtsSmallerThanInitialPts) {
  for (const auto *iface : kCodecIfaces) {
    // Initialize libvpx encoder.
    vpx_codec_ctx_t enc;
    vpx_codec_enc_cfg_t cfg;

    ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

    ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

    // Create input image.
    vpx_image_t *const image =
        CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
    ASSERT_NE(image, nullptr);

    // Encode frame.
    ASSERT_EQ(vpx_codec_encode(&enc, image, 12, 1, 0, VPX_DL_BEST_QUALITY),
              VPX_CODEC_OK);
    ASSERT_EQ(vpx_codec_encode(&enc, image, 13, 1, 0, VPX_DL_BEST_QUALITY),
              VPX_CODEC_OK);
    // pts (10) is smaller than the initial pts (12).
    ASSERT_EQ(vpx_codec_encode(&enc, image, 10, 1, 0, VPX_DL_BEST_QUALITY),
              VPX_CODEC_INVALID_PARAM);

    // Free resources.
    vpx_img_free(image);
    ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
  }
}

TEST(EncodeAPI, PtsOrDurationTooBig) {
  for (const auto *iface : kCodecIfaces) {
    // Initialize libvpx encoder.
    vpx_codec_ctx_t enc;
    vpx_codec_enc_cfg_t cfg;

    ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

    ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

    // Create input image.
    vpx_image_t *const image =
        CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
    ASSERT_NE(image, nullptr);

    // Encode frame.
    ASSERT_EQ(vpx_codec_encode(&enc, image, 0, 1, 0, VPX_DL_BEST_QUALITY),
              VPX_CODEC_OK);
#if ULONG_MAX > INT64_MAX
    // duration is too big.
    ASSERT_EQ(vpx_codec_encode(&enc, image, 0, (1ul << 63), 0, 2),
              VPX_CODEC_INVALID_PARAM);
#endif
    // pts, when converted to ticks, is too big.
    ASSERT_EQ(vpx_codec_encode(&enc, image, INT64_MAX / 1000000 + 1, 1, 0,
                               VPX_DL_BEST_QUALITY),
              VPX_CODEC_INVALID_PARAM);
#if ULONG_MAX > INT64_MAX
    // duration is too big.
    ASSERT_EQ(
        vpx_codec_encode(&enc, image, 0, (1ul << 63), 0, VPX_DL_BEST_QUALITY),
        VPX_CODEC_INVALID_PARAM);
    // pts + duration is too big.
    ASSERT_EQ(
        vpx_codec_encode(&enc, image, 1, INT64_MAX, 0, VPX_DL_BEST_QUALITY),
        VPX_CODEC_INVALID_PARAM);
#endif
    // pts + duration, when converted to ticks, is too big.
#if ULONG_MAX > INT64_MAX
    ASSERT_EQ(vpx_codec_encode(&enc, image, 0, 0xbd6b566b15c7, 0,
                               VPX_DL_BEST_QUALITY),
              VPX_CODEC_INVALID_PARAM);
#endif
    ASSERT_EQ(vpx_codec_encode(&enc, image, INT64_MAX / 1000000, 1, 0,
                               VPX_DL_BEST_QUALITY),
              VPX_CODEC_INVALID_PARAM);

    // Free resources.
    vpx_img_free(image);
    ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
  }
}

TEST(EncodeAPI, PerFramePsnr) {
  for (const auto *iface : kCodecIfaces) {
    SCOPED_TRACE(vpx_codec_iface_name(iface));
    vpx_codec_enc_cfg_t cfg;
    ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
    cfg.g_lag_in_frames = 0;

    vpx_codec_ctx_t enc;
    ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

    vpx_image_t *const image =
        CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
    ASSERT_NE(image, nullptr);

    vpx_enc_frame_flags_t psnr_flags = VPX_EFLAG_CALCULATE_PSNR;
    ASSERT_EQ(vpx_codec_encode(&enc, image, /*pts=*/0, /*duration=*/1,
                               psnr_flags, VPX_DL_REALTIME),
              VPX_CODEC_OK);

    const vpx_codec_cx_pkt_t *pkt;
    vpx_codec_iter_t iter = nullptr;
    bool had_psnr = false;
    while ((pkt = vpx_codec_get_cx_data(&enc, &iter)) != nullptr) {
      if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) {
        ASSERT_EQ(pkt->kind, VPX_CODEC_PSNR_PKT);
        had_psnr = true;
      }
    }
    EXPECT_TRUE(had_psnr);

    vpx_enc_frame_flags_t no_psnr_flags = 0;
    ASSERT_EQ(vpx_codec_encode(&enc, image, /*pts=*/1, /*duration=*/1,
                               no_psnr_flags, VPX_DL_REALTIME),
              VPX_CODEC_OK);

    iter = nullptr;
    had_psnr = false;
    while ((pkt = vpx_codec_get_cx_data(&enc, &iter)) != nullptr) {
      if (pkt->kind != VPX_CODEC_CX_FRAME_PKT) {
        ASSERT_EQ(pkt->kind, VPX_CODEC_PSNR_PKT);
        had_psnr = true;
      }
    }
#if CONFIG_INTERNAL_STATS
    // CONFIG_INTERNAL_STATS unconditionally generates PSNR.
    EXPECT_TRUE(had_psnr);
#else
    EXPECT_FALSE(had_psnr);
#endif  // CONFIG_INTERNAL_STATS

    // Free resources.
    vpx_img_free(image);
    ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
  }
}

#if CONFIG_VP9_ENCODER
// Frame size needed to trigger the overflow exceeds the max buffer allowed on
// 32-bit systems defined by VPX_MAX_ALLOCABLE_MEMORY
#if VPX_ARCH_X86_64 || VPX_ARCH_AARCH64
TEST(EncodeAPI, ConfigLargeTargetBitrateVp9) {
#ifdef CHROMIUM
  GTEST_SKIP() << "Under Chromium's configuration the allocator is unable"
                  "to provide the space required for the frames below.";
#else
  constexpr int kWidth = 12383;
  constexpr int kHeight = 8192;
  constexpr auto *iface = &vpx_codec_vp9_cx_algo;
  SCOPED_TRACE(vpx_codec_iface_name(iface));
  vpx_codec_enc_cfg_t cfg = {};
  struct Encoder {
    ~Encoder() { EXPECT_EQ(vpx_codec_destroy(&ctx), VPX_CODEC_OK); }
    vpx_codec_ctx_t ctx = {};
  } enc;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
  // The following setting will cause avg_frame_bandwidth in rate control to be
  // larger than INT_MAX
  cfg.rc_target_bitrate = INT_MAX;
  // Framerate 0.1 (equivalent to timebase 10) is the smallest framerate allowed
  // by libvpx
  cfg.g_timebase.den = 1;
  cfg.g_timebase.num = 10;
  EXPECT_NO_FATAL_FAILURE(InitCodec(*iface, kWidth, kHeight, &enc.ctx, &cfg))
      << "target bitrate: " << cfg.rc_target_bitrate << " framerate: "
      << static_cast<double>(cfg.g_timebase.den) / cfg.g_timebase.num;
#endif  // defined(CHROMIUM)
}
#endif  // VPX_ARCH_X86_64 || VPX_ARCH_AARCH64

// Emulates the WebCodecs VideoEncoder interface.
class VP9Encoder {
 public:
  explicit VP9Encoder(int speed)
      : speed_(speed), row_mt_(0), bit_depth_(VPX_BITS_8),
        fmt_(VPX_IMG_FMT_I420) {}
  // The image format `fmt` must not have the VPX_IMG_FMT_HIGHBITDEPTH bit set.
  // If bit_depth > 8, we will set the VPX_IMG_FMT_HIGHBITDEPTH bit before
  // passing the image format to vpx_img_alloc().
  VP9Encoder(int speed, unsigned int row_mt, vpx_bit_depth_t bit_depth,
             vpx_img_fmt_t fmt)
      : speed_(speed), row_mt_(row_mt), bit_depth_(bit_depth), fmt_(fmt) {}
  ~VP9Encoder();

  void Configure(unsigned int threads, unsigned int width, unsigned int height,
                 vpx_rc_mode end_usage, vpx_enc_deadline_t deadline);
  void Encode(bool key_frame);

 private:
  const int speed_;
  const unsigned int row_mt_;
  const vpx_bit_depth_t bit_depth_;
  const vpx_img_fmt_t fmt_;
  bool initialized_ = false;
  vpx_codec_enc_cfg_t cfg_;
  vpx_codec_ctx_t enc_;
  int frame_index_ = 0;
  vpx_enc_deadline_t deadline_ = 0;
};

VP9Encoder::~VP9Encoder() {
  if (initialized_) {
    EXPECT_EQ(vpx_codec_destroy(&enc_), VPX_CODEC_OK);
  }
}

void VP9Encoder::Configure(unsigned int threads, unsigned int width,
                           unsigned int height, vpx_rc_mode end_usage,
                           vpx_enc_deadline_t deadline) {
  deadline_ = deadline;

  if (!initialized_) {
    ASSERT_EQ(fmt_ & VPX_IMG_FMT_HIGHBITDEPTH, 0);
    const bool high_bit_depth = bit_depth_ > VPX_BITS_8;
    const bool is_420 = fmt_ == VPX_IMG_FMT_I420;
    vpx_codec_iface_t *const iface = vpx_codec_vp9_cx();
    ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg_, /*usage=*/0),
              VPX_CODEC_OK);
    cfg_.g_threads = threads;
    // In profiles 0 and 2, only 4:2:0 format is allowed. In profiles 1 and 3,
    // all other subsampling formats are allowed. In profiles 0 and 1, only bit
    // depth 8 is allowed. In profiles 2 and 3, only bit depths 10 and 12 are
    // allowed.
    cfg_.g_profile = 2 * high_bit_depth + !is_420;
    cfg_.g_w = width;
    cfg_.g_h = height;
    cfg_.g_bit_depth = bit_depth_;
    cfg_.g_input_bit_depth = bit_depth_;
    cfg_.g_timebase.num = 1;
    cfg_.g_timebase.den = 1000 * 1000;  // microseconds
    cfg_.g_pass = VPX_RC_ONE_PASS;
    cfg_.g_lag_in_frames = 0;
    cfg_.rc_end_usage = end_usage;
    cfg_.rc_min_quantizer = 2;
    cfg_.rc_max_quantizer = 58;
    ASSERT_EQ(
        vpx_codec_enc_init(&enc_, iface, &cfg_,
                           high_bit_depth ? VPX_CODEC_USE_HIGHBITDEPTH : 0),
        VPX_CODEC_OK);
    ASSERT_EQ(vpx_codec_control(&enc_, VP8E_SET_CPUUSED, speed_), VPX_CODEC_OK);
    ASSERT_EQ(vpx_codec_control(&enc_, VP9E_SET_ROW_MT, row_mt_), VPX_CODEC_OK);
    initialized_ = true;
    return;
  }

  cfg_.g_threads = threads;
  cfg_.g_w = width;
  cfg_.g_h = height;
  cfg_.rc_end_usage = end_usage;
  ASSERT_EQ(vpx_codec_enc_config_set(&enc_, &cfg_), VPX_CODEC_OK)
      << vpx_codec_error_detail(&enc_);
}

void VP9Encoder::Encode(bool key_frame) {
  assert(initialized_);
  const vpx_codec_cx_pkt_t *pkt;
  vpx_image_t *image = CreateImage(bit_depth_, fmt_, cfg_.g_w, cfg_.g_h);
  ASSERT_NE(image, nullptr);
  const vpx_enc_frame_flags_t frame_flags = key_frame ? VPX_EFLAG_FORCE_KF : 0;
  ASSERT_EQ(
      vpx_codec_encode(&enc_, image, frame_index_, 1, frame_flags, deadline_),
      VPX_CODEC_OK);
  ++frame_index_;
  vpx_codec_iter_t iter = nullptr;
  while ((pkt = vpx_codec_get_cx_data(&enc_, &iter)) != nullptr) {
    ASSERT_EQ(pkt->kind, VPX_CODEC_CX_FRAME_PKT);
  }
  vpx_img_free(image);
}

// This is a test case from clusterfuzz.
TEST(EncodeAPI, PrevMiCheckNullptr) {
  VP9Encoder encoder(0);
  encoder.Configure(0, 1554, 644, VPX_VBR, VPX_DL_REALTIME);

  // First step: encode, without forcing KF.
  encoder.Encode(false);
  // Second step: change config
  encoder.Configure(0, 1131, 644, VPX_CBR, VPX_DL_GOOD_QUALITY);
  // Third step: encode, without forcing KF
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/310477034.
// Encode a few frames with multiple change config calls
// with different frame sizes.
TEST(EncodeAPI, MultipleChangeConfigResize) {
  VP9Encoder encoder(3);

  // Set initial config.
  encoder.Configure(3, 41, 1, VPX_VBR, VPX_DL_REALTIME);

  // Encode first frame.
  encoder.Encode(true);

  // Change config.
  encoder.Configure(16, 31, 1, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Change config again.
  encoder.Configure(0, 17, 1, VPX_CBR, VPX_DL_REALTIME);

  // Encode 2nd frame with new config, set delta frame.
  encoder.Encode(false);

  // Encode 3rd frame with same config, set delta frame.
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/310663186.
// Encode set of frames while varying the deadline on the fly from
// good to realtime to best and back to realtime.
TEST(EncodeAPI, DynamicDeadlineChange) {
  // Use realtime speed: 5 to 9.
  VP9Encoder encoder(5);

  // Set initial config, in particular set deadline to GOOD mode.
  encoder.Configure(0, 1, 1, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 1st frame.
  encoder.Encode(true);

  // Encode 2nd frame, delta frame.
  encoder.Encode(false);

  // Change config: change deadline to REALTIME.
  encoder.Configure(0, 1, 1, VPX_VBR, VPX_DL_REALTIME);

  // Encode 3rd frame with new config, set key frame.
  encoder.Encode(true);

  // Encode 4th frame with same config, delta frame.
  encoder.Encode(false);

  // Encode 5th frame with same config, key frame.
  encoder.Encode(true);

  // Change config: change deadline to BEST.
  encoder.Configure(0, 1, 1, VPX_VBR, VPX_DL_BEST_QUALITY);

  // Encode 6th frame with new config, set delta frame.
  encoder.Encode(false);

  // Change config: change deadline to REALTIME.
  encoder.Configure(0, 1, 1, VPX_VBR, VPX_DL_REALTIME);

  // Encode 7th frame with new config, set delta frame.
  encoder.Encode(false);

  // Encode 8th frame with new config, set key frame.
  encoder.Encode(true);

  // Encode 9th frame with new config, set delta frame.
  encoder.Encode(false);
}

TEST(EncodeAPI, Buganizer310340241) {
  VP9Encoder encoder(-6);

  // Set initial config, in particular set deadline to GOOD mode.
  encoder.Configure(0, 1, 1, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 1st frame.
  encoder.Encode(true);

  // Encode 2nd frame, delta frame.
  encoder.Encode(false);

  // Change config: change deadline to REALTIME.
  encoder.Configure(0, 1, 1, VPX_VBR, VPX_DL_REALTIME);

  // Encode 3rd frame with new config, set key frame.
  encoder.Encode(true);
}

// This is a test case from clusterfuzz: based on b/312517065.
TEST(EncodeAPI, Buganizer312517065) {
  VP9Encoder encoder(4);
  encoder.Configure(0, 1060, 437, VPX_CBR, VPX_DL_REALTIME);
  encoder.Encode(true);
  encoder.Configure(10, 33, 437, VPX_VBR, VPX_DL_GOOD_QUALITY);
  encoder.Encode(false);
  encoder.Configure(6, 327, 269, VPX_VBR, VPX_DL_GOOD_QUALITY);
  encoder.Configure(15, 1060, 437, VPX_CBR, VPX_DL_REALTIME);
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/311489136.
// Encode a few frames with multiple change config calls
// with different frame sizes.
TEST(EncodeAPI, Buganizer311489136) {
  VP9Encoder encoder(1);

  // Set initial config.
  encoder.Configure(12, 1678, 620, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode first frame.
  encoder.Encode(true);

  // Change config.
  encoder.Configure(3, 1678, 202, VPX_CBR, VPX_DL_GOOD_QUALITY);

  // Encode 2nd frame with new config, set delta frame.
  encoder.Encode(false);

  // Change config again.
  encoder.Configure(8, 1037, 476, VPX_CBR, VPX_DL_REALTIME);

  // Encode 3rd frame with new config, set delta frame.
  encoder.Encode(false);

  // Change config again.
  encoder.Configure(0, 580, 620, VPX_CBR, VPX_DL_GOOD_QUALITY);

  // Encode 4th frame with same config, set delta frame.
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/312656387.
// Encode a few frames with multiple change config calls
// with different frame sizes.
TEST(EncodeAPI, Buganizer312656387) {
  VP9Encoder encoder(1);

  // Set initial config.
  encoder.Configure(16, 1, 1024, VPX_CBR, VPX_DL_REALTIME);

  // Change config.
  encoder.Configure(15, 1, 1024, VPX_VBR, VPX_DL_REALTIME);

  // Encode first frame.
  encoder.Encode(true);

  // Change config again.
  encoder.Configure(14, 1, 595, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 2nd frame with new config.
  encoder.Encode(true);

  // Change config again.
  encoder.Configure(2, 1, 1024, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 3rd frame with new config, set delta frame.
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/310329177.
// Encode a few frames with multiple change config calls
// with different frame sizes.
TEST(EncodeAPI, Buganizer310329177) {
  VP9Encoder encoder(6);

  // Set initial config.
  encoder.Configure(10, 41, 1, VPX_VBR, VPX_DL_REALTIME);

  // Encode first frame.
  encoder.Encode(true);

  // Change config.
  encoder.Configure(16, 1, 1, VPX_VBR, VPX_DL_REALTIME);

  // Encode 2nd frame with new config, set delta frame.
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/311394513.
// Encode a few frames with multiple change config calls
// with different frame sizes.
TEST(EncodeAPI, Buganizer311394513) {
  VP9Encoder encoder(-7);

  // Set initial config.
  encoder.Configure(0, 5, 9, VPX_VBR, VPX_DL_REALTIME);

  // Encode first frame.
  encoder.Encode(false);

  // Change config.
  encoder.Configure(5, 2, 1, VPX_VBR, VPX_DL_REALTIME);

  // Encode 2nd frame with new config.
  encoder.Encode(true);
}

TEST(EncodeAPI, Buganizer311985118) {
  VP9Encoder encoder(0);

  // Set initial config, in particular set deadline to GOOD mode.
  encoder.Configure(12, 1678, 620, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 1st frame.
  encoder.Encode(false);

  // Change config: change threads and width.
  encoder.Configure(0, 1574, 620, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Change config: change threads, width and height.
  encoder.Configure(16, 837, 432, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 2nd frame.
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/314857577.
// Encode a few frames with multiple change config calls
// with different frame sizes.
TEST(EncodeAPI, Buganizer314857577) {
  VP9Encoder encoder(4);

  // Set initial config.
  encoder.Configure(12, 1060, 437, VPX_VBR, VPX_DL_REALTIME);

  // Encode first frame.
  encoder.Encode(false);

  // Change config.
  encoder.Configure(16, 1060, 1, VPX_CBR, VPX_DL_REALTIME);

  // Encode 2nd frame with new config.
  encoder.Encode(false);

  // Encode 3rd frame with new config.
  encoder.Encode(true);

  // Change config.
  encoder.Configure(15, 33, 437, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 4th frame with new config.
  encoder.Encode(true);

  // Encode 5th frame with new config.
  encoder.Encode(false);

  // Change config.
  encoder.Configure(5, 327, 269, VPX_VBR, VPX_DL_REALTIME);

  // Change config.
  encoder.Configure(15, 1060, 437, VPX_CBR, VPX_DL_REALTIME);

  // Encode 6th frame with new config.
  encoder.Encode(false);

  // Encode 7th frame with new config.
  encoder.Encode(false);

  // Change config.
  encoder.Configure(4, 1060, 437, VPX_VBR, VPX_DL_REALTIME);

  // Encode 8th frame with new config.
  encoder.Encode(false);
}

TEST(EncodeAPI, Buganizer312875957PredBufferStride) {
  VP9Encoder encoder(-1);

  encoder.Configure(12, 1678, 620, VPX_VBR, VPX_DL_REALTIME);
  encoder.Encode(true);
  encoder.Encode(false);
  encoder.Configure(0, 456, 486, VPX_VBR, VPX_DL_REALTIME);
  encoder.Encode(true);
  encoder.Configure(0, 1678, 620, VPX_CBR, 1000000);
  encoder.Encode(false);
  encoder.Encode(false);
}

// This is a test case from clusterfuzz: based on b/311294795
// Encode a few frames with multiple change config calls
// with different frame sizes.
TEST(EncodeAPI, Buganizer311294795) {
  VP9Encoder encoder(1);

  // Set initial config.
  encoder.Configure(12, 1678, 620, VPX_VBR, VPX_DL_REALTIME);

  // Encode first frame.
  encoder.Encode(false);

  // Change config.
  encoder.Configure(16, 632, 620, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 2nd frame with new config
  encoder.Encode(true);

  // Change config.
  encoder.Configure(16, 1678, 342, VPX_VBR, VPX_DL_GOOD_QUALITY);

  // Encode 3rd frame with new config.
  encoder.Encode(false);

  // Change config.
  encoder.Configure(0, 1574, 618, VPX_VBR, VPX_DL_REALTIME);
  // Encode more frames with new config.
  encoder.Encode(false);
  encoder.Encode(false);
}

// Test case to capture assert issue triggered in
// vp9_bitstream.c for good_quality, speed 1, lossless;
// See comment#22 in issue:433941753.
TEST(EncodeAPI, AssertIssueGoodQualitySpeed1Lossless) {
  vpx_codec_iface_t *const iface = vpx_codec_vp9_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
  cfg.g_w = 1540;
  cfg.g_h = 838;
  cfg.g_profile = 0;
  cfg.g_bit_depth = VPX_BITS_8;
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = 10000;
  cfg.g_pass = VPX_RC_ONE_PASS;
  cfg.g_lag_in_frames = 0;
  cfg.rc_end_usage = VPX_VBR;
  cfg.g_threads = 1;
  cfg.rc_target_bitrate = 10000;
  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_LOSSLESS, 1), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_CPUUSED, 1), VPX_CODEC_OK);
  libvpx_test::RandomVideoSource video;
  video.SetSize(cfg.g_w, cfg.g_h);
  video.SetImageFormat(VPX_IMG_FMT_I420);
  video.set_limit(20);
  video.Begin();
  do {
    ASSERT_EQ(vpx_codec_encode(&enc, video.img(), video.pts(), video.duration(),
                               0, VPX_DL_GOOD_QUALITY),
              VPX_CODEC_OK);
    video.Next();
  } while (video.img() != nullptr);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, Buganizer317105128) {
  VP9Encoder encoder(-9);
  encoder.Configure(0, 1, 1, VPX_CBR, VPX_DL_GOOD_QUALITY);
  encoder.Configure(16, 1920, 1, VPX_CBR, VPX_DL_REALTIME);
}

TEST(EncodeAPI, Buganizer319964497) {
  VP9Encoder encoder(7);
  encoder.Configure(/*threads=*/1, /*width=*/320, /*height=*/240, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Encode(/*key_frame=*/true);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/1, /*width=*/1, /*height=*/1, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/1, /*width=*/2, /*height=*/2, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
}

TEST(EncodeAPI, Buganizer329088759RowMT0) {
  VP9Encoder encoder(8, 0, VPX_BITS_8, VPX_IMG_FMT_I444);
  encoder.Configure(/*threads=*/8, /*width=*/1686, /*height=*/398, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/0, /*width=*/1686, /*height=*/1, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/0, /*width=*/1482, /*height=*/113, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/0, /*width=*/881, /*height=*/59, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Configure(/*threads=*/13, /*width=*/1271, /*height=*/385, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/2, /*width=*/1, /*height=*/62, VPX_VBR,
                    VPX_DL_REALTIME);
}

TEST(EncodeAPI, Buganizer329088759RowMT1) {
  VP9Encoder encoder(8, 1, VPX_BITS_8, VPX_IMG_FMT_I444);
  encoder.Configure(/*threads=*/8, /*width=*/1686, /*height=*/398, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Encode(/*key_frame=*/false);
  // Needs to set threads to non-zero to repro the issue.
  encoder.Configure(/*threads=*/2, /*width=*/1686, /*height=*/1, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/2, /*width=*/1482, /*height=*/113, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/2, /*width=*/881, /*height=*/59, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Configure(/*threads=*/13, /*width=*/1271, /*height=*/385, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/2, /*width=*/1, /*height=*/62, VPX_VBR,
                    VPX_DL_REALTIME);
}

TEST(EncodeAPI, Buganizer331086799) {
  VP9Encoder encoder(6, 1, VPX_BITS_8, VPX_IMG_FMT_I420);
  encoder.Configure(0, 1385, 1, VPX_CBR, VPX_DL_REALTIME);
  encoder.Configure(0, 1, 1, VPX_VBR, VPX_DL_REALTIME);
  encoder.Encode(false);
  encoder.Configure(16, 1385, 1, VPX_VBR, VPX_DL_GOOD_QUALITY);
  encoder.Encode(false);
  encoder.Encode(false);
  encoder.Configure(0, 1, 1, VPX_CBR, VPX_DL_REALTIME);
  encoder.Encode(true);
}

TEST(EncodeAPI, Buganizer331108729) {
  VP9Encoder encoder(1, 1, VPX_BITS_8, VPX_IMG_FMT_I422);
  encoder.Configure(0, 1919, 260, VPX_VBR, VPX_DL_REALTIME);
  encoder.Configure(9, 440, 1, VPX_CBR, VPX_DL_GOOD_QUALITY);
  encoder.Encode(true);
  encoder.Configure(8, 1919, 260, VPX_VBR, VPX_DL_REALTIME);
  encoder.Encode(false);
}

TEST(EncodeAPI, Buganizer331108922BitDepth8) {
  VP9Encoder encoder(9, 1, VPX_BITS_8, VPX_IMG_FMT_I420);
  encoder.Configure(/*threads=*/1, /*width=*/1, /*height=*/1080, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/0, /*width=*/1, /*height=*/1080, VPX_CBR,
                    VPX_DL_GOOD_QUALITY);
  encoder.Configure(/*threads=*/16, /*width=*/1, /*height=*/394, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/16, /*width=*/1, /*height=*/798, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
}

// Encode some frames, flip from BEST_QUALITY to REALTIME after 2 frames.
// This test is taken from the code snippet in issue:441668134.
TEST(EncodeAPI, Buganizer441668134) {
  // Get VP9 encoder interface.
  vpx_codec_iface_t *iface = vpx_codec_vp9_cx();
  // Initialize encoder configuration with default values.
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
  cfg.g_lag_in_frames = 0;
  cfg.rc_max_quantizer = 0;
  unsigned long init_flags = 0;
  vpx_codec_ctx_t ctx;
  ASSERT_EQ(vpx_codec_enc_init(&ctx, iface, &cfg, init_flags), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control_(&ctx, VP8E_SET_CPUUSED, 9), 0);
  ASSERT_EQ(vpx_codec_control_(&ctx, VP9E_SET_DELTA_Q_UV, -15), 0);
  // Image allocation.
  vpx_img_fmt_t img_fmt = VPX_IMG_FMT_I420;
  vpx_image_t *img = vpx_img_alloc(NULL, img_fmt, cfg.g_w, cfg.g_h, 32);
  for (unsigned int y = 0; y < img->d_h; y++) {
    for (unsigned int x = 0; x < img->d_w; x++) {
      img->planes[0][y * img->stride[0] + x] = ((x ^ y) * 127) & 0xFF;
    }
  }
  const unsigned int uv_height = (img->d_h + 1) >> 1;
  for (int i : { VPX_PLANE_U, VPX_PLANE_V }) {
    memset(img->planes[i], 0, img->stride[i] * uv_height);
  }
  // Encode some frames.
  int num_frames = 6;
  static constexpr int kChoices[6] = { 1, 1, 0, 0, 0, 0 };
  for (int frame = 0; frame < num_frames; frame++) {
    vpx_enc_deadline_t deadline = VPX_DL_REALTIME;
    uint8_t dl_choice = kChoices[frame];
    if (dl_choice == 1) deadline = VPX_DL_BEST_QUALITY;
    // Encode frame.
    ASSERT_EQ(vpx_codec_encode(&ctx, img, frame, 1, 0, deadline), VPX_CODEC_OK);
  }
  vpx_img_free(img);
  vpx_codec_destroy(&ctx);
}

// Encode a few frames, with realtime mode and tile_rows set to 1,
// with row-mt enabled. This triggers an assertion in vp9_bitstream.c (in
// function write_modes()), as in the issue:442105459. In this test it happens
// on very first encoded frame since lag_in_frames = 0. Issue is due to enabling
// TILE_ROWS, with number of tile_rows more than the number of superblocks.
// This test sets 2 tile_rows with height corresponding to 1 superblock (sb).
TEST(EncodeAPI, Buganizer442105459_2RowTiles) {
  // Initialize VP9 encoder interface
  vpx_codec_iface_t *iface = vpx_codec_vp9_cx();
  // Get default encoder configuration
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
  // Configure encoder
  cfg.g_w = 946u;
  cfg.g_h = 64u;  // 1 sb row, 2 tile_rows set below.
  cfg.g_threads = 1;
  cfg.g_profile = 0;
  cfg.g_bit_depth = VPX_BITS_8;
  // Rate control targeting deeper encoding paths
  cfg.rc_target_bitrate = 100;
  cfg.rc_min_quantizer = 0;
  cfg.rc_max_quantizer = 0;
  cfg.rc_end_usage = VPX_VBR;
  cfg.ss_number_layers = 1;
  cfg.g_lag_in_frames = 0;
  // Initialize encoder context
  vpx_codec_ctx_t ctx;
  ASSERT_EQ(vpx_codec_enc_init(&ctx, iface, &cfg, 0), VPX_CODEC_OK);
  // Set control parameters
  vpx_codec_control_(&ctx, VP8E_SET_CPUUSED, -5);
  vpx_codec_control_(&ctx, VP9E_SET_TILE_ROWS, 1);
  vpx_codec_control_(&ctx, VP9E_SET_TILE_COLUMNS, 1);
  vpx_codec_control_(&ctx, VP9E_SET_ROW_MT, 1);
  // Image format selection
  vpx_img_fmt_t img_fmt = VPX_IMG_FMT_I420;
  // Allocate image with varied alignment
  vpx_image_t *img = vpx_img_alloc(nullptr, img_fmt, cfg.g_w, cfg.g_h, 1);
  for (unsigned int y = 0; y < img->d_h; y++) {
    for (unsigned int x = 0; x < img->d_w; x++) {
      img->planes[0][y * img->stride[0] + x] = ((x ^ y) * 127) & 0xFF;
    }
  }
  const unsigned int uv_height = (img->d_h + 1) >> 1;
  for (int i : { VPX_PLANE_U, VPX_PLANE_V }) {
    memset(img->planes[i], 0, img->stride[i] * uv_height);
  }
  // Encode with dynamic configuration changes
  int num_frames = 2;
  // Per-frame constants captured from the original run (indices consumed per
  // frame)
  const vpx_codec_pts_t frame_pts_mul[] = { 33333UL, 33333UL };
  const unsigned long frame_durations[] = { 33333UL, 33333UL };
  const vpx_enc_deadline_t frame_deadlines[] = { VPX_DL_REALTIME,
                                                 VPX_DL_REALTIME };
  for (int frame = 0; frame < num_frames; frame++) {
    // Encode frame
    vpx_codec_pts_t pts = frame * frame_pts_mul[frame];
    unsigned long duration = frame_durations[frame];
    vpx_enc_deadline_t deadline = frame_deadlines[frame];
    ASSERT_EQ(vpx_codec_encode(&ctx, img, pts, duration, /*flags*/ 0, deadline),
              VPX_CODEC_OK);
  }
  // Flush encoder.
  ASSERT_EQ(vpx_codec_encode(&ctx, nullptr, 0, 0, 0, VPX_DL_REALTIME), 0);
  // Get remaining data
  vpx_codec_iter_t iter = nullptr;
  while (vpx_codec_get_cx_data(&ctx, &iter) != nullptr) {
    // Process remaining packets
  }
  vpx_img_free(img);
  vpx_codec_destroy(&ctx);
}

// Encode a few frames, with realtime mode and tile_rows set to 1,
// with row-mt enabled. This triggers an assertion in vp9_bitstream.c (in
// function write_modes()), as in the issue:442105459. In this test it happens
// on very first encoded frame since lag_in_frames = 0. Issue is due to enabling
// TILE_ROWS, with number of tile_rows more than the number of superblocks.
// This test sets 4 tile_rows with height corresponding to 3 superblocks.
TEST(EncodeAPI, Buganizer442105459_4RowTiles) {
  // Initialize VP9 encoder interface
  vpx_codec_iface_t *iface = vpx_codec_vp9_cx();
  // Get default encoder configuration
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
  // Configure encoder
  cfg.g_w = 946u;
  cfg.g_h = 192u;  // 3 sb rows, 4 tile_rows set below.
  cfg.g_threads = 1;
  cfg.g_profile = 0;
  cfg.g_bit_depth = VPX_BITS_8;
  // Rate control targeting deeper encoding paths
  cfg.rc_target_bitrate = 100;
  cfg.rc_min_quantizer = 0;
  cfg.rc_max_quantizer = 0;
  cfg.rc_end_usage = VPX_VBR;
  cfg.ss_number_layers = 1;
  cfg.g_lag_in_frames = 0;
  // Initialize encoder context
  vpx_codec_ctx_t ctx;
  ASSERT_EQ(vpx_codec_enc_init(&ctx, iface, &cfg, 0), VPX_CODEC_OK);
  // Set control parameters
  vpx_codec_control_(&ctx, VP8E_SET_CPUUSED, -5);
  vpx_codec_control_(&ctx, VP9E_SET_TILE_ROWS, 2);
  vpx_codec_control_(&ctx, VP9E_SET_TILE_COLUMNS, 1);
  vpx_codec_control_(&ctx, VP9E_SET_ROW_MT, 1);
  // Image format selection
  vpx_img_fmt_t img_fmt = VPX_IMG_FMT_I420;
  // Allocate image with varied alignment
  vpx_image_t *img = vpx_img_alloc(nullptr, img_fmt, cfg.g_w, cfg.g_h, 1);
  for (unsigned int y = 0; y < img->d_h; y++) {
    for (unsigned int x = 0; x < img->d_w; x++) {
      img->planes[0][y * img->stride[0] + x] = ((x ^ y) * 127) & 0xFF;
    }
  }
  const unsigned int uv_height = (img->d_h + 1) >> 1;
  for (int i : { VPX_PLANE_U, VPX_PLANE_V }) {
    memset(img->planes[i], 0, img->stride[i] * uv_height);
  }
  // Encode with dynamic configuration changes
  int num_frames = 2;
  // Per-frame constants captured from the original run (indices consumed per
  // frame)
  const vpx_codec_pts_t frame_pts_mul[] = { 33333UL, 33333UL };
  const unsigned long frame_durations[] = { 33333UL, 33333UL };
  const vpx_enc_deadline_t frame_deadlines[] = { VPX_DL_REALTIME,
                                                 VPX_DL_REALTIME };
  for (int frame = 0; frame < num_frames; frame++) {
    // Encode frame
    vpx_codec_pts_t pts = frame * frame_pts_mul[frame];
    unsigned long duration = frame_durations[frame];
    vpx_enc_deadline_t deadline = frame_deadlines[frame];
    ASSERT_EQ(vpx_codec_encode(&ctx, img, pts, duration, /*flags*/ 0, deadline),
              VPX_CODEC_OK);
  }
  // Flush encoder.
  ASSERT_EQ(vpx_codec_encode(&ctx, nullptr, 0, 0, 0, VPX_DL_REALTIME), 0);
  // Get remaining data
  vpx_codec_iter_t iter = nullptr;
  while (vpx_codec_get_cx_data(&ctx, &iter) != nullptr) {
    // Process remaining packets
  }
  vpx_img_free(img);
  vpx_codec_destroy(&ctx);
}

#if CONFIG_VP9_HIGHBITDEPTH
TEST(EncodeAPI, Buganizer329674887RowMT0BitDepth12) {
  VP9Encoder encoder(8, 0, VPX_BITS_12, VPX_IMG_FMT_I444);
  encoder.Configure(/*threads=*/2, /*width=*/1030, /*height=*/583, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/0, /*width=*/1030, /*height=*/1, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/0, /*width=*/548, /*height=*/322, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/16, /*width=*/24, /*height=*/583, VPX_CBR,
                    VPX_DL_GOOD_QUALITY);
}

TEST(EncodeAPI, Buganizer329179808RowMT0BitDepth10) {
  VP9Encoder encoder(4, 0, VPX_BITS_10, VPX_IMG_FMT_I444);
  encoder.Configure(/*threads=*/16, /*width=*/1488, /*height=*/5, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/16, /*width=*/839, /*height=*/1, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/11, /*width=*/657, /*height=*/5, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
}

TEST(EncodeAPI, Buganizer329179808RowMT1BitDepth10) {
  VP9Encoder encoder(4, 1, VPX_BITS_10, VPX_IMG_FMT_I444);
  encoder.Configure(/*threads=*/16, /*width=*/1488, /*height=*/5, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/16, /*width=*/839, /*height=*/1, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/11, /*width=*/657, /*height=*/5, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
}

TEST(EncodeAPI, Buganizer331108922BitDepth12) {
  VP9Encoder encoder(9, 1, VPX_BITS_12, VPX_IMG_FMT_I444);
  encoder.Configure(/*threads=*/1, /*width=*/1, /*height=*/1080, VPX_VBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Configure(/*threads=*/0, /*width=*/1, /*height=*/1080, VPX_CBR,
                    VPX_DL_GOOD_QUALITY);
  encoder.Configure(/*threads=*/16, /*width=*/1, /*height=*/394, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
  encoder.Encode(/*key_frame=*/true);
  encoder.Configure(/*threads=*/16, /*width=*/1, /*height=*/798, VPX_CBR,
                    VPX_DL_REALTIME);
  encoder.Encode(/*key_frame=*/false);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

TEST(EncodeAPI, VP9GlobalHeaders) {
  constexpr int kWidth = 320;
  constexpr int kHeight = 240;

  libvpx_test::DummyVideoSource video;
  video.SetSize(kWidth, kHeight);

#if CONFIG_VP9_HIGHBITDEPTH
  const int profiles[] = { 0, 1, 2, 3 };
#else
  const int profiles[] = { 0, 1 };
#endif
  char str[80];
  for (const int profile : profiles) {
    std::vector<vpx_bit_depth_t> bitdepths;
    std::vector<vpx_img_fmt_t> formats;
    switch (profile) {
      case 0:
        bitdepths = { VPX_BITS_8 };
        formats = { VPX_IMG_FMT_I420 };
        break;
      case 1:
        bitdepths = { VPX_BITS_8 };
        formats = { VPX_IMG_FMT_I422, VPX_IMG_FMT_I444 };
        break;
#if CONFIG_VP9_HIGHBITDEPTH
      case 2:
        bitdepths = { VPX_BITS_10, VPX_BITS_12 };
        formats = { VPX_IMG_FMT_I42016 };
        break;
      case 3:
        bitdepths = { VPX_BITS_10, VPX_BITS_12 };
        formats = { VPX_IMG_FMT_I42216, VPX_IMG_FMT_I44416 };
        break;
#endif
    }

    for (const auto format : formats) {
      for (const auto bitdepth : bitdepths) {
        snprintf(str, sizeof(str), "profile: %d bitdepth: %d format: %d",
                 profile, bitdepth, format);
        SCOPED_TRACE(str);

        vpx_codec_enc_cfg_t cfg = {};
        struct Encoder {
          ~Encoder() { EXPECT_EQ(vpx_codec_destroy(&ctx), VPX_CODEC_OK); }
          vpx_codec_ctx_t ctx = {};
        } enc;
        vpx_codec_ctx_t *const ctx = &enc.ctx;

        ASSERT_EQ(vpx_codec_enc_config_default(vpx_codec_vp9_cx(), &cfg, 0),
                  VPX_CODEC_OK);
        cfg.g_w = kWidth;
        cfg.g_h = kHeight;
        cfg.g_lag_in_frames = 0;
        cfg.g_pass = VPX_RC_ONE_PASS;
        cfg.g_profile = profile;
        cfg.g_bit_depth = bitdepth;
        ASSERT_EQ(
            vpx_codec_enc_init(ctx, vpx_codec_vp9_cx(), &cfg,
                               bitdepth == 8 ? 0 : VPX_CODEC_USE_HIGHBITDEPTH),
            VPX_CODEC_OK);
        ASSERT_EQ(vpx_codec_control_(ctx, VP8E_SET_CPUUSED, 2), VPX_CODEC_OK);
        ASSERT_EQ(vpx_codec_control_(ctx, VP9E_SET_TARGET_LEVEL, 62),
                  VPX_CODEC_OK);

        vpx_fixed_buf_t *global_headers = vpx_codec_get_global_headers(ctx);
        EXPECT_NE(global_headers, nullptr);
        EXPECT_EQ(global_headers->sz, size_t{ 9 });

        video.SetImageFormat(format);
        video.Begin();
        EXPECT_EQ(
            vpx_codec_encode(ctx, video.img(), video.pts(), video.duration(),
                             /*flags=*/0, VPX_DL_GOOD_QUALITY),
            VPX_CODEC_OK)
            << vpx_codec_error_detail(ctx);

        global_headers = vpx_codec_get_global_headers(ctx);
        EXPECT_NE(global_headers, nullptr);
        EXPECT_EQ(global_headers->sz, size_t{ 12 });
        uint8_t chroma_subsampling;
        if ((format & VPX_IMG_FMT_I420) == VPX_IMG_FMT_I420) {
          chroma_subsampling = 1;
        } else if ((format & VPX_IMG_FMT_I422) == VPX_IMG_FMT_I422) {
          chroma_subsampling = 2;
        } else {  // VPX_IMG_FMT_I444
          chroma_subsampling = 3;
        }
        const uint8_t expected_headers[] = { 1,
                                             1,
                                             static_cast<uint8_t>(profile),
                                             2,
                                             1,
                                             /*level,*/ 3,
                                             1,
                                             static_cast<uint8_t>(bitdepth),
                                             4,
                                             1,
                                             chroma_subsampling };
        const uint8_t *actual_headers =
            reinterpret_cast<const uint8_t *>(global_headers->buf);
        for (int i = 0; i < 5; ++i) {
          EXPECT_EQ(expected_headers[i], actual_headers[i]) << "index: " << i;
        }
        EXPECT_NE(actual_headers[6], 0);  // level
        for (int i = 5; i < 11; ++i) {
          EXPECT_EQ(expected_headers[i], actual_headers[i + 1])
              << "index: " << i + 1;
        }
      }
    }
  }
}

TEST(EncodeAPI, AomediaIssue3509VbrMinSection2PercentVP9) {
  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp9_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 1920;
  cfg.g_h = 1080;
  cfg.g_lag_in_frames = 0;
  cfg.rc_target_bitrate = 1000000;
  // Set this to more than 1 percent to cause a signed integer overflow in the
  // multiplication rc->avg_frame_bandwidth * oxcf->rc_cfg.vbrmin_section in
  // vp9_rc_update_framerate() if the multiplication is done in the `int` type.
  cfg.rc_2pass_vbr_minsection_pct = 2;

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  // Create input image.
  vpx_image_t *const image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  // Encode frame.
  // `duration` can go as high as 300, but the UBSan error is gone if
  // `duration` is 301 or higher.
  ASSERT_EQ(
      vpx_codec_encode(&enc, image, 0, /*duration=*/300, 0, VPX_DL_REALTIME),
      VPX_CODEC_OK);

  // Free resources.
  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, AomediaIssue3509VbrMinSection101PercentVP9) {
  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp9_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 1920;
  cfg.g_h = 1080;
  cfg.g_lag_in_frames = 0;
  cfg.rc_target_bitrate = 1000000;
  // Set this to more than 100 percent to cause an error when vbr_min_bits is
  // cast to `int` in vp9_rc_update_framerate() if vbr_min_bits is not clamped
  // to INT_MAX.
  cfg.rc_2pass_vbr_minsection_pct = 101;

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  // Create input image.
  vpx_image_t *const image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  // Encode frame.
  // `duration` can go as high as 300, but the UBSan error is gone if
  // `duration` is 301 or higher.
  ASSERT_EQ(
      vpx_codec_encode(&enc, image, 0, /*duration=*/300, 0, VPX_DL_REALTIME),
      VPX_CODEC_OK);

  // Free resources.
  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, Chromium352414650) {
  // Initialize libvpx encoder.
  vpx_codec_iface_t *const iface = vpx_codec_vp9_cx();
  vpx_codec_ctx_t enc;
  vpx_codec_enc_cfg_t cfg;

  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);

  cfg.g_w = 1024;
  cfg.g_h = 1024;
  cfg.g_profile = 0;
  cfg.g_pass = VPX_RC_ONE_PASS;
  cfg.g_lag_in_frames = 0;
  cfg.rc_max_quantizer = 58;
  cfg.rc_min_quantizer = 2;
  cfg.g_threads = 4;
  cfg.rc_resize_allowed = 0;
  cfg.rc_dropframe_thresh = 0;
  cfg.g_timebase.num = 1;
  cfg.g_timebase.den = 1000000;
  cfg.kf_min_dist = 0;
  cfg.kf_max_dist = 10000;
  cfg.rc_end_usage = VPX_CBR;
  cfg.rc_target_bitrate = 754974;
  cfg.ts_number_layers = 3;
  cfg.ts_periodicity = 4;
  cfg.ts_layer_id[0] = 0;
  cfg.ts_layer_id[1] = 2;
  cfg.ts_layer_id[2] = 1;
  cfg.ts_layer_id[3] = 2;
  cfg.ts_rate_decimator[0] = 4;
  cfg.ts_rate_decimator[1] = 2;
  cfg.ts_rate_decimator[2] = 1;
  cfg.layer_target_bitrate[0] = 2147483;
  cfg.layer_target_bitrate[1] = 3006476;
  cfg.layer_target_bitrate[2] = 4294967;
  cfg.temporal_layering_mode = VP9E_TEMPORAL_LAYERING_MODE_0212;
  cfg.g_error_resilient = VPX_ERROR_RESILIENT_DEFAULT;

  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_CPUUSED, 7), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_TILE_COLUMNS, 2), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_ROW_MT, 1), VPX_CODEC_OK);

  vpx_svc_extra_cfg_t svc_cfg = {};
  svc_cfg.max_quantizers[0] = svc_cfg.max_quantizers[1] =
      svc_cfg.max_quantizers[2] = 58;
  svc_cfg.min_quantizers[0] = svc_cfg.min_quantizers[1] =
      svc_cfg.min_quantizers[2] = 2;
  svc_cfg.scaling_factor_num[0] = svc_cfg.scaling_factor_num[1] =
      svc_cfg.scaling_factor_num[2] = 1;
  svc_cfg.scaling_factor_den[0] = svc_cfg.scaling_factor_den[1] =
      svc_cfg.scaling_factor_den[2] = 1;
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_SVC_PARAMETERS, &svc_cfg),
            VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_SVC, 1), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_AQ_MODE, 3), VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP8E_SET_STATIC_THRESHOLD, 1),
            VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_COLOR_SPACE, VPX_CS_SMPTE_170),
            VPX_CODEC_OK);
  ASSERT_EQ(vpx_codec_control(&enc, VP9E_SET_COLOR_RANGE, VPX_CR_STUDIO_RANGE),
            VPX_CODEC_OK);

  // Create input image.
  vpx_image_t *const image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  // Encode frame.
  ASSERT_EQ(vpx_codec_encode(&enc, image, 0, /*duration=*/500000,
                             VPX_EFLAG_FORCE_KF, VPX_DL_REALTIME),
            VPX_CODEC_OK);

  // Free resources.
  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}

TEST(EncodeAPI, PerFramePsnrNotSupportedWithLagInFrames) {
  vpx_codec_iface_t *const iface = vpx_codec_vp9_cx();
  vpx_codec_enc_cfg_t cfg;
  ASSERT_EQ(vpx_codec_enc_config_default(iface, &cfg, 0), VPX_CODEC_OK);
  ASSERT_NE(cfg.g_lag_in_frames, 0u);

  vpx_codec_ctx_t enc;
  ASSERT_EQ(vpx_codec_enc_init(&enc, iface, &cfg, 0), VPX_CODEC_OK);

  vpx_image_t *const image =
      CreateImage(VPX_BITS_8, VPX_IMG_FMT_I420, cfg.g_w, cfg.g_h);
  ASSERT_NE(image, nullptr);

  vpx_enc_frame_flags_t psnr_flags = VPX_EFLAG_CALCULATE_PSNR;
  ASSERT_EQ(vpx_codec_encode(&enc, image, /*pts=*/0, /*duration=*/1, psnr_flags,
                             VPX_DL_REALTIME),
            VPX_CODEC_INCAPABLE);

  // Free resources.
  vpx_img_free(image);
  ASSERT_EQ(vpx_codec_destroy(&enc), VPX_CODEC_OK);
}
#endif  // CONFIG_VP9_ENCODER

}  // namespace
