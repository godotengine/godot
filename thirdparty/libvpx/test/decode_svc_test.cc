/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <memory>
#include <string>

#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/ivf_video_source.h"
#include "test/test_vectors.h"
#include "test/util.h"

namespace {

const unsigned int kNumFrames = 19;

class DecodeSvcTest : public ::libvpx_test::DecoderTest,
                      public ::libvpx_test::CodecTestWithParam<const char *> {
 protected:
  DecodeSvcTest() : DecoderTest(GET_PARAM(::libvpx_test::kCodecFactoryParam)) {}
  ~DecodeSvcTest() override = default;

  void PreDecodeFrameHook(const libvpx_test::CompressedVideoSource &video,
                          libvpx_test::Decoder *decoder) override {
    if (video.frame_number() == 0)
      decoder->Control(VP9_DECODE_SVC_SPATIAL_LAYER, spatial_layer_);
  }

  void DecompressedFrameHook(const vpx_image_t &img,
                             const unsigned int frame_number) override {
    ASSERT_EQ(img.d_w, width_);
    ASSERT_EQ(img.d_h, height_);
    total_frames_ = frame_number;
  }

  int spatial_layer_;
  unsigned int width_;
  unsigned int height_;
  unsigned int total_frames_;
};

// SVC test vector is 1280x720, with 3 spatial layers, and 20 frames.

// Decode the SVC test vector, which has 3 spatial layers, and decode up to
// spatial layer 0. Verify the resolution of each decoded frame and the total
// number of frames decoded. This results in 1/4x1/4 resolution (320x180).
TEST_P(DecodeSvcTest, DecodeSvcTestUpToSpatialLayer0) {
  const std::string filename = GET_PARAM(1);
  std::unique_ptr<libvpx_test::CompressedVideoSource> video;
  video.reset(new libvpx_test::IVFVideoSource(filename));
  ASSERT_NE(video.get(), nullptr);
  video->Init();
  total_frames_ = 0;
  spatial_layer_ = 0;
  width_ = 320;
  height_ = 180;
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  ASSERT_EQ(total_frames_, kNumFrames);
}

// Decode the SVC test vector, which has 3 spatial layers, and decode up to
// spatial layer 1. Verify the resolution of each decoded frame and the total
// number of frames decoded. This results in 1/2x1/2 resolution (640x360).
TEST_P(DecodeSvcTest, DecodeSvcTestUpToSpatialLayer1) {
  const std::string filename = GET_PARAM(1);
  std::unique_ptr<libvpx_test::CompressedVideoSource> video;
  video.reset(new libvpx_test::IVFVideoSource(filename));
  ASSERT_NE(video.get(), nullptr);
  video->Init();
  total_frames_ = 0;
  spatial_layer_ = 1;
  width_ = 640;
  height_ = 360;
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  ASSERT_EQ(total_frames_, kNumFrames);
}

// Decode the SVC test vector, which has 3 spatial layers, and decode up to
// spatial layer 2. Verify the resolution of each decoded frame and the total
// number of frames decoded. This results in the full resolution (1280x720).
TEST_P(DecodeSvcTest, DecodeSvcTestUpToSpatialLayer2) {
  const std::string filename = GET_PARAM(1);
  std::unique_ptr<libvpx_test::CompressedVideoSource> video;
  video.reset(new libvpx_test::IVFVideoSource(filename));
  ASSERT_NE(video.get(), nullptr);
  video->Init();
  total_frames_ = 0;
  spatial_layer_ = 2;
  width_ = 1280;
  height_ = 720;
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  ASSERT_EQ(total_frames_, kNumFrames);
}

// Decode the SVC test vector, which has 3 spatial layers, and decode up to
// spatial layer 10. Verify the resolution of each decoded frame and the total
// number of frames decoded. This is beyond the number of spatial layers, so
// the decoding should result in the full resolution (1280x720).
TEST_P(DecodeSvcTest, DecodeSvcTestUpToSpatialLayer10) {
  const std::string filename = GET_PARAM(1);
  std::unique_ptr<libvpx_test::CompressedVideoSource> video;
  video.reset(new libvpx_test::IVFVideoSource(filename));
  ASSERT_NE(video.get(), nullptr);
  video->Init();
  total_frames_ = 0;
  spatial_layer_ = 10;
  width_ = 1280;
  height_ = 720;
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get()));
  ASSERT_EQ(total_frames_, kNumFrames);
}

VP9_INSTANTIATE_TEST_SUITE(
    DecodeSvcTest, ::testing::ValuesIn(libvpx_test::kVP9TestVectorsSvc,
                                       libvpx_test::kVP9TestVectorsSvc +
                                           libvpx_test::kNumVP9TestVectorsSvc));
}  // namespace
