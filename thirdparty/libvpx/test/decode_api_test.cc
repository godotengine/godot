/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "gtest/gtest.h"

#include "./vpx_config.h"
#include "test/ivf_video_source.h"
#include "vpx/vp8dx.h"
#include "vpx/vpx_decoder.h"

namespace {

#define NELEMENTS(x) static_cast<int>(sizeof(x) / sizeof(x[0]))

TEST(DecodeAPI, InvalidParams) {
  static vpx_codec_iface_t *kCodecs[] = {
#if CONFIG_VP8_DECODER
    &vpx_codec_vp8_dx_algo,
#endif
#if CONFIG_VP9_DECODER
    &vpx_codec_vp9_dx_algo,
#endif
  };
  uint8_t buf[1] = { 0 };
  vpx_codec_ctx_t dec;

  EXPECT_EQ(vpx_codec_dec_init(nullptr, nullptr, nullptr, 0),
            VPX_CODEC_INVALID_PARAM);
  EXPECT_EQ(vpx_codec_dec_init(&dec, nullptr, nullptr, 0),
            VPX_CODEC_INVALID_PARAM);
  EXPECT_EQ(vpx_codec_decode(nullptr, nullptr, 0, nullptr, 0),
            VPX_CODEC_INVALID_PARAM);
  EXPECT_EQ(vpx_codec_decode(nullptr, buf, 0, nullptr, 0),
            VPX_CODEC_INVALID_PARAM);
  EXPECT_EQ(vpx_codec_decode(nullptr, buf, NELEMENTS(buf), nullptr, 0),
            VPX_CODEC_INVALID_PARAM);
  EXPECT_EQ(vpx_codec_decode(nullptr, nullptr, NELEMENTS(buf), nullptr, 0),
            VPX_CODEC_INVALID_PARAM);
  EXPECT_EQ(vpx_codec_destroy(nullptr), VPX_CODEC_INVALID_PARAM);
  EXPECT_NE(vpx_codec_error(nullptr), nullptr);
  EXPECT_EQ(vpx_codec_error_detail(nullptr), nullptr);

  for (int i = 0; i < NELEMENTS(kCodecs); ++i) {
    EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
              vpx_codec_dec_init(nullptr, kCodecs[i], nullptr, 0));

    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_dec_init(&dec, kCodecs[i], nullptr, 0));
    EXPECT_EQ(VPX_CODEC_UNSUP_BITSTREAM,
              vpx_codec_decode(&dec, buf, NELEMENTS(buf), nullptr, 0));
    EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
              vpx_codec_decode(&dec, nullptr, NELEMENTS(buf), nullptr, 0));
    EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
              vpx_codec_decode(&dec, buf, 0, nullptr, 0));

    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_destroy(&dec));
  }
}

#if CONFIG_VP8_DECODER
TEST(DecodeAPI, OptionalParams) {
  vpx_codec_ctx_t dec;

#if CONFIG_ERROR_CONCEALMENT
  EXPECT_EQ(VPX_CODEC_OK,
            vpx_codec_dec_init(&dec, &vpx_codec_vp8_dx_algo, nullptr,
                               VPX_CODEC_USE_ERROR_CONCEALMENT));
#else
  EXPECT_EQ(VPX_CODEC_INCAPABLE,
            vpx_codec_dec_init(&dec, &vpx_codec_vp8_dx_algo, nullptr,
                               VPX_CODEC_USE_ERROR_CONCEALMENT));
#endif  // CONFIG_ERROR_CONCEALMENT
}
#endif  // CONFIG_VP8_DECODER

#if CONFIG_VP9_DECODER
// Test VP9 codec controls after a decode error to ensure the code doesn't
// misbehave.
void TestVp9Controls(vpx_codec_ctx_t *dec) {
  static const int kControls[] = { VP8D_GET_LAST_REF_UPDATES,
                                   VP8D_GET_FRAME_CORRUPTED,
                                   VP9D_GET_DISPLAY_SIZE, VP9D_GET_FRAME_SIZE };
  int val[2];

  for (int i = 0; i < NELEMENTS(kControls); ++i) {
    const vpx_codec_err_t res = vpx_codec_control_(dec, kControls[i], val);
    switch (kControls[i]) {
      case VP8D_GET_FRAME_CORRUPTED:
        EXPECT_EQ(VPX_CODEC_ERROR, res) << kControls[i];
        break;
      default: EXPECT_EQ(VPX_CODEC_OK, res) << kControls[i]; break;
    }
    EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
              vpx_codec_control_(dec, kControls[i], nullptr));
  }

  vp9_ref_frame_t ref;
  ref.idx = 0;
  EXPECT_EQ(VPX_CODEC_ERROR, vpx_codec_control(dec, VP9_GET_REFERENCE, &ref));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_control(dec, VP9_GET_REFERENCE, nullptr));

  vpx_ref_frame_t ref_copy;
  const int width = 352;
  const int height = 288;
  EXPECT_NE(vpx_img_alloc(&ref_copy.img, VPX_IMG_FMT_I420, width, height, 1),
            nullptr);
  ref_copy.frame_type = VP8_LAST_FRAME;
  EXPECT_EQ(VPX_CODEC_ERROR,
            vpx_codec_control(dec, VP8_COPY_REFERENCE, &ref_copy));
  EXPECT_EQ(VPX_CODEC_INVALID_PARAM,
            vpx_codec_control(dec, VP8_COPY_REFERENCE, nullptr));
  vpx_img_free(&ref_copy.img);
}

TEST(DecodeAPI, Vp9InvalidDecode) {
  vpx_codec_iface_t *const codec = &vpx_codec_vp9_dx_algo;
  const char filename[] =
      "invalid-vp90-2-00-quantizer-00.webm.ivf.s5861_r01-05_b6-.v2.ivf";
  libvpx_test::IVFVideoSource video(filename);
  video.Init();
  video.Begin();
  ASSERT_TRUE(!HasFailure());

  vpx_codec_ctx_t dec;
  EXPECT_EQ(VPX_CODEC_OK, vpx_codec_dec_init(&dec, codec, nullptr, 0));
  const uint32_t frame_size = static_cast<uint32_t>(video.frame_size());
#if CONFIG_VP9_HIGHBITDEPTH
  EXPECT_EQ(VPX_CODEC_MEM_ERROR,
            vpx_codec_decode(&dec, video.cxdata(), frame_size, nullptr, 0));
#else
  EXPECT_EQ(VPX_CODEC_UNSUP_BITSTREAM,
            vpx_codec_decode(&dec, video.cxdata(), frame_size, nullptr, 0));
#endif
  vpx_codec_iter_t iter = nullptr;
  EXPECT_EQ(nullptr, vpx_codec_get_frame(&dec, &iter));

  TestVp9Controls(&dec);
  EXPECT_EQ(VPX_CODEC_OK, vpx_codec_destroy(&dec));
}

void TestPeekInfo(const uint8_t *const data, uint32_t data_sz,
                  uint32_t peek_size) {
  vpx_codec_iface_t *const codec = &vpx_codec_vp9_dx_algo;
  // Verify behavior of vpx_codec_decode. vpx_codec_decode doesn't even get
  // to decoder_peek_si_internal on frames of size < 8.
  if (data_sz >= 8) {
    vpx_codec_ctx_t dec;
    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_dec_init(&dec, codec, nullptr, 0));
    EXPECT_EQ((data_sz < peek_size) ? VPX_CODEC_UNSUP_BITSTREAM
                                    : VPX_CODEC_CORRUPT_FRAME,
              vpx_codec_decode(&dec, data, data_sz, nullptr, 0));
    vpx_codec_iter_t iter = nullptr;
    EXPECT_EQ(nullptr, vpx_codec_get_frame(&dec, &iter));
    EXPECT_EQ(VPX_CODEC_OK, vpx_codec_destroy(&dec));
  }

  // Verify behavior of vpx_codec_peek_stream_info.
  vpx_codec_stream_info_t si;
  si.sz = sizeof(si);
  EXPECT_EQ((data_sz < peek_size) ? VPX_CODEC_UNSUP_BITSTREAM : VPX_CODEC_OK,
            vpx_codec_peek_stream_info(codec, data, data_sz, &si));
}

TEST(DecodeAPI, Vp9PeekStreamInfo) {
  // The first 9 bytes are valid and the rest of the bytes are made up. Until
  // size 10, this should return VPX_CODEC_UNSUP_BITSTREAM and after that it
  // should return VPX_CODEC_CORRUPT_FRAME.
  const uint8_t data[32] = {
    0x85, 0xa4, 0xc1, 0xa1, 0x38, 0x81, 0xa3, 0x49, 0x83, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
  };

  for (uint32_t data_sz = 1; data_sz <= 32; ++data_sz) {
    TestPeekInfo(data, data_sz, 10);
  }
}

TEST(DecodeAPI, Vp9PeekStreamInfoTruncated) {
  // This profile 1 header requires 10.25 bytes, ensure
  // vpx_codec_peek_stream_info doesn't over read.
  const uint8_t profile1_data[10] = { 0xa4, 0xe9, 0x30, 0x68, 0x53,
                                      0xe9, 0x30, 0x68, 0x53, 0x04 };

  for (uint32_t data_sz = 1; data_sz <= 10; ++data_sz) {
    TestPeekInfo(profile1_data, data_sz, 11);
  }
}
#endif  // CONFIG_VP9_DECODER

TEST(DecodeAPI, HighBitDepthCapability) {
// VP8 should not claim VP9 HBD as a capability.
#if CONFIG_VP8_DECODER
  const vpx_codec_caps_t vp8_caps = vpx_codec_get_caps(&vpx_codec_vp8_dx_algo);
  EXPECT_EQ(vp8_caps & VPX_CODEC_CAP_HIGHBITDEPTH, 0);
#endif

#if CONFIG_VP9_DECODER
  const vpx_codec_caps_t vp9_caps = vpx_codec_get_caps(&vpx_codec_vp9_dx_algo);
#if CONFIG_VP9_HIGHBITDEPTH
  EXPECT_EQ(vp9_caps & VPX_CODEC_CAP_HIGHBITDEPTH, VPX_CODEC_CAP_HIGHBITDEPTH);
#else
  EXPECT_EQ(vp9_caps & VPX_CODEC_CAP_HIGHBITDEPTH, 0);
#endif
#endif
}

}  // namespace
