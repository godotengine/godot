/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
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
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "./vpx_config.h"
#include "test/codec_factory.h"
#include "test/decode_test_driver.h"
#include "test/ivf_video_source.h"
#include "test/util.h"
#if CONFIG_WEBM_IO
#include "test/webm_video_source.h"
#endif
#include "vpx_mem/vpx_mem.h"

namespace {

struct DecodeParam {
  int threads;
  const char *filename;
};

std::ostream &operator<<(std::ostream &os, const DecodeParam &dp) {
  return os << "threads: " << dp.threads << " file: " << dp.filename;
}

class InvalidFileTest : public ::libvpx_test::DecoderTest,
                        public ::libvpx_test::CodecTestWithParam<DecodeParam> {
 protected:
  InvalidFileTest() : DecoderTest(GET_PARAM(0)), res_file_(nullptr) {}

  ~InvalidFileTest() override {
    if (res_file_ != nullptr) fclose(res_file_);
  }

  void OpenResFile(const std::string &res_file_name_) {
    res_file_ = libvpx_test::OpenTestDataFile(res_file_name_);
    ASSERT_NE(res_file_, nullptr)
        << "Result file open failed. Filename: " << res_file_name_;
  }

  bool HandleDecodeResult(const vpx_codec_err_t res_dec,
                          const libvpx_test::CompressedVideoSource &video,
                          libvpx_test::Decoder *decoder) override {
    EXPECT_NE(res_file_, nullptr);
    int expected_res_dec;

    // Read integer result.
    const int res = fscanf(res_file_, "%d", &expected_res_dec);
    EXPECT_NE(res, EOF) << "Read result data failed";

    // Check results match.
    const DecodeParam input = GET_PARAM(1);
    if (input.threads > 1) {
      // The serial decode check is too strict for tile-threaded decoding as
      // there is no guarantee on the decode order nor which specific error
      // will take precedence. Currently a tile-level error is not forwarded so
      // the frame will simply be marked corrupt.
      EXPECT_TRUE(res_dec == expected_res_dec ||
                  res_dec == VPX_CODEC_CORRUPT_FRAME)
          << "Results don't match: frame number = " << video.frame_number()
          << ". (" << decoder->DecodeError()
          << "). Expected: " << expected_res_dec << " or "
          << VPX_CODEC_CORRUPT_FRAME;
    } else {
      EXPECT_EQ(expected_res_dec, res_dec)
          << "Results don't match: frame number = " << video.frame_number()
          << ". (" << decoder->DecodeError() << ")";
    }

    return !HasFailure();
  }

  void RunTest() {
    const DecodeParam input = GET_PARAM(1);
    vpx_codec_dec_cfg_t cfg = vpx_codec_dec_cfg_t();
    cfg.threads = input.threads;
    const std::string filename = input.filename;

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

    // Construct result file name. The file holds a list of expected integer
    // results, one for each decoded frame.  Any result that doesn't match
    // the files list will cause a test failure.
    const std::string res_filename = filename + ".res";
    OpenResFile(res_filename);

    // Decode frame, and check the md5 matching.
    ASSERT_NO_FATAL_FAILURE(RunLoop(video.get(), cfg));
  }

 private:
  FILE *res_file_;
};

TEST_P(InvalidFileTest, ReturnCode) { RunTest(); }

#if CONFIG_VP8_DECODER
const DecodeParam kVP8InvalidFileTests[] = {
  { 1, "invalid-bug-1443.ivf" },
  { 1, "invalid-bug-148271109.ivf" },
  { 1, "invalid-token-partition.ivf" },
  { 1, "invalid-vp80-00-comprehensive-s17661_r01-05_b6-.ivf" },
};

VP8_INSTANTIATE_TEST_SUITE(InvalidFileTest,
                           ::testing::ValuesIn(kVP8InvalidFileTests));
#endif  // CONFIG_VP8_DECODER

#if CONFIG_VP9_DECODER
const DecodeParam kVP9InvalidFileTests[] = {
  { 1, "invalid-vp90-02-v2.webm" },
#if CONFIG_VP9_HIGHBITDEPTH
  { 1, "invalid-vp90-2-00-quantizer-00.webm.ivf.s5861_r01-05_b6-.v2.ivf" },
  { 1,
    "invalid-vp90-2-21-resize_inter_320x180_5_3-4.webm.ivf.s45551_r01-05_b6-."
    "ivf" },
#endif
  { 1, "invalid-vp90-03-v3.webm" },
  { 1, "invalid-vp90-2-00-quantizer-11.webm.ivf.s52984_r01-05_b6-.ivf" },
  { 1, "invalid-vp90-2-00-quantizer-11.webm.ivf.s52984_r01-05_b6-z.ivf" },
// This file will cause a large allocation which is expected to fail in 32-bit
// environments. Test x86 for coverage purposes as the allocation failure will
// be in platform agnostic code.
#if VPX_ARCH_X86
  { 1, "invalid-vp90-2-00-quantizer-63.ivf.kf_65527x61446.ivf" },
#endif
  { 1, "invalid-vp90-2-12-droppable_1.ivf.s3676_r01-05_b6-.ivf" },
  { 1, "invalid-vp90-2-05-resize.ivf.s59293_r01-05_b6-.ivf" },
  { 1, "invalid-vp90-2-09-subpixel-00.ivf.s20492_r01-05_b6-.v2.ivf" },
  { 1, "invalid-vp91-2-mixedrefcsp-444to420.ivf" },
  { 1, "invalid-vp90-2-12-droppable_1.ivf.s73804_r01-05_b6-.ivf" },
  { 1, "invalid-vp90-2-03-size-224x196.webm.ivf.s44156_r01-05_b6-.ivf" },
  { 1, "invalid-vp90-2-03-size-202x210.webm.ivf.s113306_r01-05_b6-.ivf" },
  { 1,
    "invalid-vp90-2-10-show-existing-frame.webm.ivf.s180315_r01-05_b6-.ivf" },
  { 1, "invalid-crbug-667044.webm" },
};

VP9_INSTANTIATE_TEST_SUITE(InvalidFileTest,
                           ::testing::ValuesIn(kVP9InvalidFileTests));
#endif  // CONFIG_VP9_DECODER

// This class will include test vectors that are expected to fail
// peek. However they are still expected to have no fatal failures.
class InvalidFileInvalidPeekTest : public InvalidFileTest {
 protected:
  InvalidFileInvalidPeekTest() : InvalidFileTest() {}
  void HandlePeekResult(libvpx_test::Decoder *const /*decoder*/,
                        libvpx_test::CompressedVideoSource * /*video*/,
                        const vpx_codec_err_t /*res_peek*/) override {}
};

TEST_P(InvalidFileInvalidPeekTest, ReturnCode) { RunTest(); }

#if CONFIG_VP8_DECODER
const DecodeParam kVP8InvalidPeekTests[] = {
  { 1, "invalid-vp80-00-comprehensive-018.ivf.2kf_0x6.ivf" },
};

VP8_INSTANTIATE_TEST_SUITE(InvalidFileInvalidPeekTest,
                           ::testing::ValuesIn(kVP8InvalidPeekTests));
#endif  // CONFIG_VP8_DECODER

#if CONFIG_VP9_DECODER
const DecodeParam kVP9InvalidFileInvalidPeekTests[] = {
  { 1, "invalid-vp90-01-v3.webm" },
};

VP9_INSTANTIATE_TEST_SUITE(
    InvalidFileInvalidPeekTest,
    ::testing::ValuesIn(kVP9InvalidFileInvalidPeekTests));

const DecodeParam kMultiThreadedVP9InvalidFileTests[] = {
  { 4, "invalid-vp90-2-08-tile_1x4_frame_parallel_all_key.webm" },
  { 4,
    "invalid-"
    "vp90-2-08-tile_1x2_frame_parallel.webm.ivf.s47039_r01-05_b6-.ivf" },
  { 4,
    "invalid-vp90-2-08-tile_1x8_frame_parallel.webm.ivf.s288_r01-05_b6-.ivf" },
  { 2, "invalid-vp90-2-09-aq2.webm.ivf.s3984_r01-05_b6-.v2.ivf" },
  { 4, "invalid-vp90-2-09-subpixel-00.ivf.s19552_r01-05_b6-.v2.ivf" },
  { 2, "invalid-crbug-629481.webm" },
  { 3, "invalid-crbug-1558.ivf" },
  { 4, "invalid-crbug-1562.ivf" },
};

INSTANTIATE_TEST_SUITE_P(
    VP9MultiThreaded, InvalidFileTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)),
        ::testing::ValuesIn(kMultiThreadedVP9InvalidFileTests)));
#endif  // CONFIG_VP9_DECODER
}  // namespace
