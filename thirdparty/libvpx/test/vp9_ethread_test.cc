/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/md5_helper.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vp9/encoder/vp9_firstpass.h"
#include "vpx_config.h"

namespace {
// FIRSTPASS_STATS struct:
// {
//   26 double members;
//   1 int64_t member;
// }
// Whenever FIRSTPASS_STATS struct is modified, the following constants need to
// be revisited.
const int kDbl = 26;
const int kInt = 1;
const size_t kFirstPassStatsSz = kDbl * sizeof(double) + kInt * sizeof(int64_t);

class VPxFirstPassEncoderThreadTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith2Params<libvpx_test::TestMode, int> {
 protected:
  VPxFirstPassEncoderThreadTest()
      : EncoderTest(GET_PARAM(0)), encoder_initialized_(false), tiles_(0),
        encoding_mode_(GET_PARAM(1)), set_cpu_used_(GET_PARAM(2)) {
    init_flags_ = VPX_CODEC_USE_PSNR;

    row_mt_mode_ = 1;
    first_pass_only_ = true;
    firstpass_stats_.buf = nullptr;
    firstpass_stats_.sz = 0;
  }
  ~VPxFirstPassEncoderThreadTest() override { free(firstpass_stats_.buf); }

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);

    cfg_.rc_end_usage = VPX_VBR;
    cfg_.rc_2pass_vbr_minsection_pct = 5;
    cfg_.rc_2pass_vbr_maxsection_pct = 2000;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_min_quantizer = 0;
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    encoder_initialized_ = false;
    abort_ = false;
  }

  void EndPassHook() override {
    // For first pass stats test, only run first pass encoder.
    if (first_pass_only_ && cfg_.g_pass == VPX_RC_FIRST_PASS)
      abort_ |= first_pass_only_;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource * /*video*/,
                          ::libvpx_test::Encoder *encoder) override {
    if (!encoder_initialized_) {
      // Encode in 2-pass mode.
      encoder->Control(VP9E_SET_TILE_COLUMNS, tiles_);
      encoder->Control(VP8E_SET_CPUUSED, set_cpu_used_);
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
      encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
      encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
      encoder->Control(VP8E_SET_ARNR_TYPE, 3);
      encoder->Control(VP9E_SET_FRAME_PARALLEL_DECODING, 0);

      if (encoding_mode_ == ::libvpx_test::kTwoPassGood)
        encoder->Control(VP9E_SET_ROW_MT, row_mt_mode_);

      encoder_initialized_ = true;
    }
  }

  void StatsPktHook(const vpx_codec_cx_pkt_t *pkt) override {
    const uint8_t *const pkt_buf =
        reinterpret_cast<uint8_t *>(pkt->data.twopass_stats.buf);
    const size_t pkt_size = pkt->data.twopass_stats.sz;

    // First pass stats size equals sizeof(FIRSTPASS_STATS)
    EXPECT_EQ(pkt_size, kFirstPassStatsSz)
        << "Error: First pass stats size doesn't equal kFirstPassStatsSz";

    firstpass_stats_.buf =
        realloc(firstpass_stats_.buf, firstpass_stats_.sz + pkt_size);
    ASSERT_NE(firstpass_stats_.buf, nullptr);
    memcpy((uint8_t *)firstpass_stats_.buf + firstpass_stats_.sz, pkt_buf,
           pkt_size);
    firstpass_stats_.sz += pkt_size;
  }

  bool encoder_initialized_;
  int tiles_;
  ::libvpx_test::TestMode encoding_mode_;
  int set_cpu_used_;
  int row_mt_mode_;
  bool first_pass_only_;
  vpx_fixed_buf_t firstpass_stats_;
};

#if !CONFIG_REALTIME_ONLY
static void compare_fp_stats(vpx_fixed_buf_t *fp_stats, double factor) {
  // fp_stats consists of 2 set of first pass encoding stats. These 2 set of
  // stats are compared to check if the stats match or at least are very close.
  FIRSTPASS_STATS *stats1 = reinterpret_cast<FIRSTPASS_STATS *>(fp_stats->buf);
  int nframes_ = (int)(fp_stats->sz / sizeof(FIRSTPASS_STATS));
  FIRSTPASS_STATS *stats2 = stats1 + nframes_ / 2;
  int i, j;

  // The total stats are also output and included in the first pass stats. Here
  // ignore that in the comparison.
  for (i = 0; i < (nframes_ / 2 - 1); ++i) {
    const double *frame_stats1 = reinterpret_cast<double *>(stats1);
    const double *frame_stats2 = reinterpret_cast<double *>(stats2);

    for (j = 0; j < kDbl; ++j) {
      ASSERT_LE(fabs(*frame_stats1 - *frame_stats2),
                fabs(*frame_stats1) / factor)
          << "First failure @ frame #" << i << " stat #" << j << " ("
          << *frame_stats1 << " vs. " << *frame_stats2 << ")";
      frame_stats1++;
      frame_stats2++;
    }

    stats1++;
    stats2++;
  }

  // Reset firstpass_stats_ to 0.
  memset((uint8_t *)fp_stats->buf, 0, fp_stats->sz);
  fp_stats->sz = 0;
}

static void compare_fp_stats_md5(vpx_fixed_buf_t *fp_stats) {
  // fp_stats consists of 2 set of first pass encoding stats. These 2 set of
  // stats are compared to check if the stats match.
  uint8_t *stats1 = reinterpret_cast<uint8_t *>(fp_stats->buf);
  uint8_t *stats2 = stats1 + fp_stats->sz / 2;
  ::libvpx_test::MD5 md5_row_mt_0, md5_row_mt_1;

  md5_row_mt_0.Add(stats1, fp_stats->sz / 2);
  const char *md5_row_mt_0_str = md5_row_mt_0.Get();

  md5_row_mt_1.Add(stats2, fp_stats->sz / 2);
  const char *md5_row_mt_1_str = md5_row_mt_1.Get();

  // Check md5 match.
  ASSERT_STREQ(md5_row_mt_0_str, md5_row_mt_1_str)
      << "MD5 checksums don't match";

  // Reset firstpass_stats_ to 0.
  memset((uint8_t *)fp_stats->buf, 0, fp_stats->sz);
  fp_stats->sz = 0;
}
#endif  // !CONFIG_REALTIME_ONLY

TEST_P(VPxFirstPassEncoderThreadTest, FirstPassStatsTest) {
#if CONFIG_REALTIME_ONLY
  GTEST_SKIP();
#else
  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 0, 60);

  first_pass_only_ = true;
  cfg_.rc_target_bitrate = 1000;

  // Test row_mt_mode: 0 vs 1 at single thread case(threads = 1, tiles_ = 0)
  tiles_ = 0;
  cfg_.g_threads = 1;

  row_mt_mode_ = 0;
  init_flags_ = VPX_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  row_mt_mode_ = 1;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // Compare to check if using or not using row-mt generates close stats.
  ASSERT_NO_FATAL_FAILURE(compare_fp_stats(&firstpass_stats_, 400.0));

  // Test single thread vs multiple threads
  row_mt_mode_ = 1;
  tiles_ = 0;

  cfg_.g_threads = 1;
  init_flags_ = VPX_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  cfg_.g_threads = 4;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // Compare to check if single-thread and multi-thread stats are close enough.
  ASSERT_NO_FATAL_FAILURE(compare_fp_stats(&firstpass_stats_, 400.0));

  // Bit exact test in row_mt mode.
  // When row_mt_mode_=1 and using >1 threads, the encoder generates bit exact
  // result.
  row_mt_mode_ = 1;
  tiles_ = 2;

  cfg_.g_threads = 2;
  init_flags_ = VPX_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  cfg_.g_threads = 8;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

  // Compare to check if stats match with row-mt=0/1.
  compare_fp_stats_md5(&firstpass_stats_);
#endif  // CONFIG_REALTIME_ONLY
}

class VPxEncoderThreadTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWith4Params<libvpx_test::TestMode, int,
                                                 int, int> {
 protected:
  VPxEncoderThreadTest()
      : EncoderTest(GET_PARAM(0)), encoder_initialized_(false),
        tiles_(GET_PARAM(3)), threads_(GET_PARAM(4)),
        encoding_mode_(GET_PARAM(1)), set_cpu_used_(GET_PARAM(2)) {
    init_flags_ = VPX_CODEC_USE_PSNR;
    md5_.clear();
    row_mt_mode_ = 1;
    psnr_ = 0.0;
    nframes_ = 0;
  }
  ~VPxEncoderThreadTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);

    if (encoding_mode_ != ::libvpx_test::kRealTime) {
      cfg_.rc_end_usage = VPX_VBR;
      cfg_.rc_2pass_vbr_minsection_pct = 5;
      cfg_.rc_2pass_vbr_maxsection_pct = 2000;
    } else {
      cfg_.g_lag_in_frames = 0;
      cfg_.rc_end_usage = VPX_CBR;
      cfg_.g_error_resilient = 1;
    }
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_min_quantizer = 0;
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    encoder_initialized_ = false;
    psnr_ = 0.0;
    nframes_ = 0;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource * /*video*/,
                          ::libvpx_test::Encoder *encoder) override {
    if (!encoder_initialized_) {
      // Encode 4 column tiles.
      encoder->Control(VP9E_SET_TILE_COLUMNS, tiles_);
      encoder->Control(VP8E_SET_CPUUSED, set_cpu_used_);
      if (encoding_mode_ != ::libvpx_test::kRealTime) {
        encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 1);
        encoder->Control(VP8E_SET_ARNR_MAXFRAMES, 7);
        encoder->Control(VP8E_SET_ARNR_STRENGTH, 5);
        encoder->Control(VP8E_SET_ARNR_TYPE, 3);
        encoder->Control(VP9E_SET_FRAME_PARALLEL_DECODING, 0);
      } else {
        encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 0);
        encoder->Control(VP9E_SET_AQ_MODE, 3);
      }
      encoder->Control(VP9E_SET_ROW_MT, row_mt_mode_);

      encoder_initialized_ = true;
    }
  }

  void PSNRPktHook(const vpx_codec_cx_pkt_t *pkt) override {
    psnr_ += pkt->data.psnr.psnr[0];
    nframes_++;
  }

  void DecompressedFrameHook(const vpx_image_t &img,
                             vpx_codec_pts_t /*pts*/) override {
    ::libvpx_test::MD5 md5_res;
    md5_res.Add(&img);
    md5_.push_back(md5_res.Get());
  }

  bool HandleDecodeResult(const vpx_codec_err_t res,
                          const libvpx_test::VideoSource & /*video*/,
                          libvpx_test::Decoder * /*decoder*/) override {
    if (res != VPX_CODEC_OK) {
      EXPECT_EQ(VPX_CODEC_OK, res);
      return false;
    }

    return true;
  }

  double GetAveragePsnr() const { return nframes_ ? (psnr_ / nframes_) : 0.0; }

  bool encoder_initialized_;
  int tiles_;
  int threads_;
  ::libvpx_test::TestMode encoding_mode_;
  int set_cpu_used_;
  int row_mt_mode_;
  double psnr_;
  unsigned int nframes_;
  std::vector<std::string> md5_;
};

TEST_P(VPxEncoderThreadTest, EncoderResultTest) {
  ::libvpx_test::Y4mVideoSource video("niklas_1280_720_30.y4m", 15, 20);
  cfg_.rc_target_bitrate = 1000;

  // Part 1: Bit exact test for row_mt_mode_ = 0.
  // This part keeps original unit tests done before row-mt code is checked in.
  row_mt_mode_ = 0;

  // Encode using single thread.
  cfg_.g_threads = 1;
  init_flags_ = VPX_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const std::vector<std::string> single_thr_md5 = md5_;
  md5_.clear();

  // Encode using multiple threads.
  cfg_.g_threads = threads_;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const std::vector<std::string> multi_thr_md5 = md5_;
  md5_.clear();

  // Compare to check if two vectors are equal.
  ASSERT_EQ(single_thr_md5, multi_thr_md5);

  // Part 2: row_mt_mode_ = 0 vs row_mt_mode_ = 1 single thread bit exact test.
  row_mt_mode_ = 1;

  // Encode using single thread
  cfg_.g_threads = 1;
  init_flags_ = VPX_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  std::vector<std::string> row_mt_single_thr_md5 = md5_;
  md5_.clear();

  ASSERT_EQ(single_thr_md5, row_mt_single_thr_md5);

  // Part 3: Bit exact test with row-mt on
  // When row_mt_mode_=1 and using >1 threads, the encoder generates bit exact
  // result.
  row_mt_mode_ = 1;
  row_mt_single_thr_md5.clear();

  // Encode using 2 threads.
  cfg_.g_threads = 2;
  init_flags_ = VPX_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  row_mt_single_thr_md5 = md5_;
  md5_.clear();

  // Encode using multiple threads.
  cfg_.g_threads = threads_;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const std::vector<std::string> row_mt_multi_thr_md5 = md5_;
  md5_.clear();

  // Compare to check if two vectors are equal.
  ASSERT_EQ(row_mt_single_thr_md5, row_mt_multi_thr_md5);

  // Part 4: PSNR test with bit_match_mode_ = 0
  row_mt_mode_ = 1;

  // Encode using single thread.
  cfg_.g_threads = 1;
  init_flags_ = VPX_CODEC_USE_PSNR;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const double single_thr_psnr = GetAveragePsnr();

  // Encode using multiple threads.
  cfg_.g_threads = threads_;
  ASSERT_NO_FATAL_FAILURE(RunLoop(&video));
  const double multi_thr_psnr = GetAveragePsnr();

  EXPECT_NEAR(single_thr_psnr, multi_thr_psnr, 0.2);
}

INSTANTIATE_TEST_SUITE_P(
    VP9, VPxFirstPassEncoderThreadTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)),
        ::testing::Values(::libvpx_test::kTwoPassGood),
        ::testing::Range(0, 4)));  // cpu_used

constexpr libvpx_test::TestMode kOnePassTestModes[] = {
  libvpx_test::kRealTime,
#if !CONFIG_REALTIME_ONLY
  libvpx_test::kOnePassGood,
#endif
};

// Split this into multiple instantiations so that we can distinguish
// between very slow runs ( i.e., cpu_speed 0 ) vs ones that can be
// run nightly by adding Large to the title.
INSTANTIATE_TEST_SUITE_P(
    VP9, VPxEncoderThreadTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)),
        ::testing::ValuesIn(kOnePassTestModes),
        ::testing::Range(3, 10),   // cpu_used
        ::testing::Range(0, 3),    // tile_columns
        ::testing::Range(2, 5)));  // threads

INSTANTIATE_TEST_SUITE_P(
    VP9Large, VPxEncoderThreadTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)),
        ::testing::ValuesIn(kOnePassTestModes),
        ::testing::Range(0, 3),    // cpu_used
        ::testing::Range(0, 3),    // tile_columns
        ::testing::Range(2, 5)));  // threads

#if !CONFIG_REALTIME_ONLY
INSTANTIATE_TEST_SUITE_P(
    VP9LargeBest, VPxEncoderThreadTest,
    ::testing::Combine(
        ::testing::Values(
            static_cast<const libvpx_test::CodecFactory *>(&libvpx_test::kVP9)),
        ::testing::Values(libvpx_test::kOnePassBest),
        ::testing::Range(0, 10),   // cpu_used
        ::testing::Range(0, 3),    // tile_columns
        ::testing::Range(2, 5)));  // threads
#endif

}  // namespace
