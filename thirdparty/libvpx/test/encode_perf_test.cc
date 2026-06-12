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
#include <string>
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vpx/vpx_codec.h"
#include "vpx_ports/vpx_timer.h"

namespace {

const int kMaxPsnr = 100;
const double kUsecsInSec = 1000000.0;

struct EncodePerfTestVideo {
  EncodePerfTestVideo(const char *name_, uint32_t width_, uint32_t height_,
                      uint32_t bitrate_, int frames_)
      : name(name_), width(width_), height(height_), bitrate(bitrate_),
        frames(frames_) {}
  const char *name;
  uint32_t width;
  uint32_t height;
  uint32_t bitrate;
  int frames;
};

const EncodePerfTestVideo kVP9EncodePerfTestVectors[] = {
  EncodePerfTestVideo("desktop_640_360_30.yuv", 640, 360, 200, 2484),
  EncodePerfTestVideo("kirland_640_480_30.yuv", 640, 480, 200, 300),
  EncodePerfTestVideo("macmarcomoving_640_480_30.yuv", 640, 480, 200, 987),
  EncodePerfTestVideo("macmarcostationary_640_480_30.yuv", 640, 480, 200, 718),
  EncodePerfTestVideo("niklas_640_480_30.yuv", 640, 480, 200, 471),
  EncodePerfTestVideo("tacomanarrows_640_480_30.yuv", 640, 480, 200, 300),
  EncodePerfTestVideo("tacomasmallcameramovement_640_480_30.yuv", 640, 480, 200,
                      300),
  EncodePerfTestVideo("thaloundeskmtg_640_480_30.yuv", 640, 480, 200, 300),
  EncodePerfTestVideo("niklas_1280_720_30.yuv", 1280, 720, 600, 470),
};

const int kEncodePerfTestSpeeds[] = { 5, 6, 7, 8, 9 };
const int kEncodePerfTestThreads[] = { 1, 2, 4 };

#define NELEMENTS(x) (sizeof((x)) / sizeof((x)[0]))

class VP9EncodePerfTest
    : public ::libvpx_test::EncoderTest,
      public ::libvpx_test::CodecTestWithParam<libvpx_test::TestMode> {
 protected:
  VP9EncodePerfTest()
      : EncoderTest(GET_PARAM(0)), min_psnr_(kMaxPsnr), nframes_(0),
        encoding_mode_(GET_PARAM(1)), speed_(0), threads_(1) {}

  ~VP9EncodePerfTest() override = default;

  void SetUp() override {
    InitializeConfig();
    SetMode(encoding_mode_);

    cfg_.g_lag_in_frames = 0;
    cfg_.rc_min_quantizer = 2;
    cfg_.rc_max_quantizer = 56;
    cfg_.rc_dropframe_thresh = 0;
    cfg_.rc_undershoot_pct = 50;
    cfg_.rc_overshoot_pct = 50;
    cfg_.rc_buf_sz = 1000;
    cfg_.rc_buf_initial_sz = 500;
    cfg_.rc_buf_optimal_sz = 600;
    cfg_.rc_resize_allowed = 0;
    cfg_.rc_end_usage = VPX_CBR;
    cfg_.g_error_resilient = 1;
    cfg_.g_threads = threads_;
  }

  void PreEncodeFrameHook(::libvpx_test::VideoSource *video,
                          ::libvpx_test::Encoder *encoder) override {
    if (video->frame() == 0) {
      const int log2_tile_columns = 3;
      encoder->Control(VP8E_SET_CPUUSED, speed_);
      encoder->Control(VP9E_SET_TILE_COLUMNS, log2_tile_columns);
      encoder->Control(VP9E_SET_FRAME_PARALLEL_DECODING, 1);
      encoder->Control(VP8E_SET_ENABLEAUTOALTREF, 0);
    }
  }

  void BeginPassHook(unsigned int /*pass*/) override {
    min_psnr_ = kMaxPsnr;
    nframes_ = 0;
  }

  void PSNRPktHook(const vpx_codec_cx_pkt_t *pkt) override {
    if (pkt->data.psnr.psnr[0] < min_psnr_) {
      min_psnr_ = pkt->data.psnr.psnr[0];
    }
  }

  // for performance reasons don't decode
  bool DoDecode() const override { return false; }

  double min_psnr() const { return min_psnr_; }

  void set_speed(unsigned int speed) { speed_ = speed; }

  void set_threads(unsigned int threads) { threads_ = threads; }

 private:
  double min_psnr_;
  unsigned int nframes_;
  libvpx_test::TestMode encoding_mode_;
  unsigned speed_;
  unsigned int threads_;
};

TEST_P(VP9EncodePerfTest, PerfTest) {
  for (size_t i = 0; i < NELEMENTS(kVP9EncodePerfTestVectors); ++i) {
    for (size_t j = 0; j < NELEMENTS(kEncodePerfTestSpeeds); ++j) {
      for (size_t k = 0; k < NELEMENTS(kEncodePerfTestThreads); ++k) {
        if (kVP9EncodePerfTestVectors[i].width < 512 &&
            kEncodePerfTestThreads[k] > 1) {
          continue;
        } else if (kVP9EncodePerfTestVectors[i].width < 1024 &&
                   kEncodePerfTestThreads[k] > 2) {
          continue;
        }

        set_threads(kEncodePerfTestThreads[k]);
        SetUp();

        const vpx_rational timebase = { 33333333, 1000000000 };
        cfg_.g_timebase = timebase;
        cfg_.rc_target_bitrate = kVP9EncodePerfTestVectors[i].bitrate;

        init_flags_ = VPX_CODEC_USE_PSNR;

        const unsigned frames = kVP9EncodePerfTestVectors[i].frames;
        const char *video_name = kVP9EncodePerfTestVectors[i].name;
        libvpx_test::I420VideoSource video(
            video_name, kVP9EncodePerfTestVectors[i].width,
            kVP9EncodePerfTestVectors[i].height, timebase.den, timebase.num, 0,
            kVP9EncodePerfTestVectors[i].frames);
        set_speed(kEncodePerfTestSpeeds[j]);

        vpx_usec_timer t;
        vpx_usec_timer_start(&t);

        ASSERT_NO_FATAL_FAILURE(RunLoop(&video));

        vpx_usec_timer_mark(&t);
        const double elapsed_secs = vpx_usec_timer_elapsed(&t) / kUsecsInSec;
        const double fps = frames / elapsed_secs;
        const double minimum_psnr = min_psnr();
        std::string display_name(video_name);
        if (kEncodePerfTestThreads[k] > 1) {
          char thread_count[32];
          snprintf(thread_count, sizeof(thread_count), "_t-%d",
                   kEncodePerfTestThreads[k]);
          display_name += thread_count;
        }

        printf("{\n");
        printf("\t\"type\" : \"encode_perf_test\",\n");
        printf("\t\"version\" : \"%s\",\n", vpx_codec_version_str());
        printf("\t\"videoName\" : \"%s\",\n", display_name.c_str());
        printf("\t\"encodeTimeSecs\" : %f,\n", elapsed_secs);
        printf("\t\"totalFrames\" : %u,\n", frames);
        printf("\t\"framesPerSecond\" : %f,\n", fps);
        printf("\t\"minPsnr\" : %f,\n", minimum_psnr);
        printf("\t\"speed\" : %d,\n", kEncodePerfTestSpeeds[j]);
        printf("\t\"threads\" : %d\n", kEncodePerfTestThreads[k]);
        printf("}\n");
      }
    }
  }
}

VP9_INSTANTIATE_TEST_SUITE(VP9EncodePerfTest,
                           ::testing::Values(::libvpx_test::kRealTime));
}  // namespace
