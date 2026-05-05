/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "../tools_common.h"
#include "./vpx_config.h"
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/ivf_video_source.h"
#include "test/md5_helper.h"
#include "test/test_vectors.h"
#include "test/util.h"
#if CONFIG_WEBM_IO
#include "test/webm_video_source.h"
#endif
#include "vpx_mem/vpx_mem.h"

namespace {

const int kThreads = 0;
const int kMtMode = 1;
const int kFileName = 2;

using DecodeParam = std::tuple<int, int, const char *>;

class TestVectorTest : public ::libvpx_test::DecoderTest,
                       public ::libvpx_test::CodecTestWithParam<DecodeParam> {
 protected:
  TestVectorTest() : DecoderTest(GET_PARAM(0)), md5_file_(nullptr) {
#if CONFIG_VP9_DECODER
    resize_clips_.insert(::libvpx_test::kVP9TestVectorsResize,
                         ::libvpx_test::kVP9TestVectorsResize +
                             ::libvpx_test::kNumVP9TestVectorsResize);
#endif
  }

  ~TestVectorTest() override {
    if (md5_file_) fclose(md5_file_);
  }

  void OpenMD5File(const std::string &md5_file_name_) {
    md5_file_ = libvpx_test::OpenTestDataFile(md5_file_name_);
    ASSERT_NE(md5_file_, nullptr)
        << "Md5 file open failed. Filename: " << md5_file_name_;
  }

#if CONFIG_VP9_DECODER
  void PreDecodeFrameHook(const libvpx_test::CompressedVideoSource &video,
                          libvpx_test::Decoder *decoder) override {
    if (video.frame_number() == 0 && mt_mode_ >= 0) {
      if (mt_mode_ == 1) {
        decoder->Control(VP9D_SET_LOOP_FILTER_OPT, 1);
        decoder->Control(VP9D_SET_ROW_MT, 0);
      } else if (mt_mode_ == 2) {
        decoder->Control(VP9D_SET_LOOP_FILTER_OPT, 0);
        decoder->Control(VP9D_SET_ROW_MT, 1);
      } else {
        decoder->Control(VP9D_SET_LOOP_FILTER_OPT, 0);
        decoder->Control(VP9D_SET_ROW_MT, 0);
      }
    }
  }
#endif

  void DecompressedFrameHook(const vpx_image_t &img,
                             const unsigned int frame_number) override {
    ASSERT_NE(md5_file_, nullptr);
    char expected_md5[33];
    char junk[128];

    // Read correct md5 checksums.
    const int res = fscanf(md5_file_, "%s  %s", expected_md5, junk);
    ASSERT_NE(res, EOF) << "Read md5 data failed";
    expected_md5[32] = '\0';

    ::libvpx_test::MD5 md5_res;
    md5_res.Add(&img);
    const char *actual_md5 = md5_res.Get();

    // Check md5 match.
    ASSERT_STREQ(expected_md5, actual_md5)
        << "Md5 checksums don't match: frame number = " << frame_number;
  }

#if CONFIG_VP9_DECODER
  std::set<std::string> resize_clips_;
#endif
  int mt_mode_;

 private:
  FILE *md5_file_;
};

// This test runs through the whole set of test vectors, and decodes them.
// The md5 checksums are computed for each frame in the video file. If md5
// checksums match the correct md5 data, then the test is passed. Otherwise,
// the test failed.
TEST_P(TestVectorTest, MD5Match) {
  const DecodeParam input = GET_PARAM(1);
  const std::string filename = std::get<kFileName>(input);
  vpx_codec_flags_t flags = 0;
  vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
  char str[256];

  cfg.threads = std::get<kThreads>(input);
  mt_mode_ = std::get<kMtMode>(input);
  snprintf(str, sizeof(str) / sizeof(str[0]) - 1,
           "file: %s threads: %d MT mode: %d", filename.c_str(), cfg.threads,
           mt_mode_);
  SCOPED_TRACE(str);

  // Open compressed video file.
  std::unique_ptr<libvpx_test::CompressedVideoSource> video;
  if (filename.substr(filename.length() - 3, 3) == "ivf") {
    video.reset(new libvpx_test::IVFVideoSource(filename));
  } else if (filename.substr(filename.length() - 4, 4) == "webm") {
#if CONFIG_WEBM_IO
    video.reset(new libvpx_test::WebMVideoSource(filename));
#else
    fprintf(stderr, "WebM IO is disabled, skipping test vector %s\n",
            filename.c_str());
    return;
#endif
  }
  ASSERT_NE(video.get(), nullptr);
  video->Init();

  // Construct md5 file name.
  const std::string md5_filename = filename + ".md5";
  OpenMD5File(md5_filename);

  // Set decode config and flags.
  set_cfg(cfg);
  set_flags(flags);

  // Decode frame, and check the md5 matching.
  ASSERT_NO_FATAL_FAILURE(RunLoop(video.get(), cfg));
}

#if CONFIG_VP8_DECODER
VP8_INSTANTIATE_TEST_SUITE(
    TestVectorTest,
    ::testing::Combine(
        ::testing::Values(1),   // Single thread.
        ::testing::Values(-1),  // LPF opt and Row MT is not applicable
        ::testing::ValuesIn(libvpx_test::kVP8TestVectors,
                            libvpx_test::kVP8TestVectors +
                                libvpx_test::kNumVP8TestVectors)));

// Test VP8 decode in with different numbers of threads.
INSTANTIATE_TEST_SUITE_P(
    VP8MultiThreaded, TestVectorTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP8)),
        ::testing::Combine(
            ::testing::Range(2, 9),  // With 2 ~ 8 threads.
            ::testing::Values(-1),   // LPF opt and Row MT is not applicable
            ::testing::ValuesIn(libvpx_test::kVP8TestVectors,
                                libvpx_test::kVP8TestVectors +
                                    libvpx_test::kNumVP8TestVectors))));

#endif  // CONFIG_VP8_DECODER

#if CONFIG_VP9_DECODER
VP9_INSTANTIATE_TEST_SUITE(
    TestVectorTest,
    ::testing::Combine(
        ::testing::Values(1),   // Single thread.
        ::testing::Values(-1),  // LPF opt and Row MT is not applicable
        ::testing::ValuesIn(libvpx_test::kVP9TestVectors,
                            libvpx_test::kVP9TestVectors +
                                libvpx_test::kNumVP9TestVectors)));

INSTANTIATE_TEST_SUITE_P(
    VP9MultiThreaded, TestVectorTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)),
        ::testing::Combine(
            ::testing::Range(2, 9),  // With 2 ~ 8 threads.
            ::testing::Range(0, 3),  // With multi threads modes 0 ~ 2
                                     // 0: LPF opt and Row MT disabled
                                     // 1: LPF opt enabled
                                     // 2: Row MT enabled
            ::testing::ValuesIn(libvpx_test::kVP9TestVectors,
                                libvpx_test::kVP9TestVectors +
                                    libvpx_test::kNumVP9TestVectors))));
#endif
}  // namespace
