/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TEST_ENCODE_TEST_DRIVER_H_
#define VPX_TEST_ENCODE_TEST_DRIVER_H_

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "./vpx_config.h"
#if CONFIG_VP8_ENCODER || CONFIG_VP9_ENCODER
#include "vpx/vp8cx.h"
#endif
#include "vpx/vpx_tpl.h"

namespace libvpx_test {

class CodecFactory;
class VideoSource;

enum TestMode {
  kRealTime,
  kOnePassGood,
  kOnePassBest,
  kTwoPassGood,
  kTwoPassBest
};

#if CONFIG_REALTIME_ONLY
#define ALL_TEST_MODES ::testing::Values(::libvpx_test::kRealTime)
#define ONE_PASS_TEST_MODES ::testing::Values(::libvpx_test::kRealTime)
#define ONE_OR_TWO_PASS_TEST_MODES ::testing::Values(::libvpx_test::kRealTime)
#else
#define ALL_TEST_MODES                                                        \
  ::testing::Values(::libvpx_test::kRealTime, ::libvpx_test::kOnePassGood,    \
                    ::libvpx_test::kOnePassBest, ::libvpx_test::kTwoPassGood, \
                    ::libvpx_test::kTwoPassBest)
#define ONE_PASS_TEST_MODES                                                \
  ::testing::Values(::libvpx_test::kRealTime, ::libvpx_test::kOnePassGood, \
                    ::libvpx_test::kOnePassBest)

#define ONE_OR_TWO_PASS_TEST_MODES \
  ::testing::Values(::libvpx_test::kOnePassGood, ::libvpx_test::kTwoPassGood)
#endif

#define TWO_PASS_TEST_MODES \
  ::testing::Values(::libvpx_test::kTwoPassGood, ::libvpx_test::kTwoPassBest)

// Provides an object to handle the libvpx get_cx_data() iteration pattern
class CxDataIterator {
 public:
  explicit CxDataIterator(vpx_codec_ctx_t *encoder)
      : encoder_(encoder), iter_(nullptr) {}

  const vpx_codec_cx_pkt_t *Next() {
    return vpx_codec_get_cx_data(encoder_, &iter_);
  }

 private:
  vpx_codec_ctx_t *encoder_;
  vpx_codec_iter_t iter_;
};

// Implements an in-memory store for libvpx twopass statistics
class TwopassStatsStore {
 public:
  void Append(const vpx_codec_cx_pkt_t &pkt) {
    buffer_.append(reinterpret_cast<char *>(pkt.data.twopass_stats.buf),
                   pkt.data.twopass_stats.sz);
  }

  vpx_fixed_buf_t buf() {
    const vpx_fixed_buf_t buf = { &buffer_[0], buffer_.size() };
    return buf;
  }

  void Reset() { buffer_.clear(); }

 protected:
  std::string buffer_;
};

// Provides a simplified interface to manage one video encoding pass, given
// a configuration and video source.
//
// TODO(jkoleszar): The exact services it provides and the appropriate
// level of abstraction will be fleshed out as more tests are written.
class Encoder {
 public:
  Encoder(vpx_codec_enc_cfg_t cfg, vpx_enc_deadline_t deadline,
          const unsigned long init_flags, TwopassStatsStore *stats)
      : cfg_(cfg), deadline_(deadline), init_flags_(init_flags), stats_(stats) {
    memset(&encoder_, 0, sizeof(encoder_));
  }

  virtual ~Encoder() { vpx_codec_destroy(&encoder_); }

  CxDataIterator GetCxData() { return CxDataIterator(&encoder_); }

  void InitEncoder(VideoSource *video);

  const vpx_image_t *GetPreviewFrame() {
    return vpx_codec_get_preview_frame(&encoder_);
  }
  // This is a thin wrapper around vpx_codec_encode(), so refer to
  // vpx_encoder.h for its semantics.
  void EncodeFrame(VideoSource *video, vpx_enc_frame_flags_t frame_flags);

  // Convenience wrapper for EncodeFrame()
  void EncodeFrame(VideoSource *video) { EncodeFrame(video, 0); }

  void Control(int ctrl_id, int arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, int *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, struct vpx_scaling_mode *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, struct vpx_svc_layer_id *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, struct vpx_svc_ref_frame_config *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, struct vpx_svc_parameters *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, struct vpx_svc_frame_drop *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, struct vpx_svc_spatial_layer_sync *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

#if CONFIG_VP9_ENCODER
  void Control(int ctrl_id, vpx_rc_funcs_t *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, VpxTplGopStats *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }
#endif  // CONFIG_VP9_ENCODER

#if CONFIG_VP8_ENCODER || CONFIG_VP9_ENCODER
  void Control(int ctrl_id, vpx_active_map_t *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }

  void Control(int ctrl_id, vpx_roi_map_t *arg) {
    const vpx_codec_err_t res = vpx_codec_control_(&encoder_, ctrl_id, arg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
  }
#endif
  void Config(const vpx_codec_enc_cfg_t *cfg) {
    const vpx_codec_err_t res = vpx_codec_enc_config_set(&encoder_, cfg);
    ASSERT_EQ(VPX_CODEC_OK, res) << EncoderError();
    cfg_ = *cfg;
  }

  void set_deadline(vpx_enc_deadline_t deadline) { deadline_ = deadline; }

 protected:
  virtual vpx_codec_iface_t *CodecInterface() const = 0;

  const char *EncoderError() {
    const char *detail = vpx_codec_error_detail(&encoder_);
    return detail ? detail : vpx_codec_error(&encoder_);
  }

  // Encode an image
  void EncodeFrameInternal(const VideoSource &video,
                           vpx_enc_frame_flags_t frame_flags);

  // Flush the encoder on EOS
  void Flush();

  vpx_codec_ctx_t encoder_;
  vpx_codec_enc_cfg_t cfg_;
  vpx_enc_deadline_t deadline_;
  unsigned long init_flags_;
  TwopassStatsStore *stats_;
};

// Common test functionality for all Encoder tests.
//
// This class is a mixin which provides the main loop common to all
// encoder tests. It provides hooks which can be overridden by subclasses
// to implement each test's specific behavior, while centralizing the bulk
// of the boilerplate. Note that it doesn't inherit the gtest testing
// classes directly, so that tests can be parameterized differently.
class EncoderTest {
 protected:
  explicit EncoderTest(const CodecFactory *codec)
      : codec_(codec), abort_(false), init_flags_(0), frame_flags_(0) {
    // Default to 1 thread.
    cfg_.g_threads = 1;
  }

  virtual ~EncoderTest() {}

  // Initialize the cfg_ member with the default configuration.
  void InitializeConfig();

  // Map the TestMode enum to the deadline_ and passes_ variables.
  void SetMode(TestMode mode);

  // Set encoder flag.
  void set_init_flags(unsigned long flag) {  // NOLINT(runtime/int)
    init_flags_ = flag;
  }

  // Main loop
  virtual void RunLoop(VideoSource *video);

  // Hook to be called at the beginning of a pass.
  virtual void BeginPassHook(unsigned int /*pass*/) {}

  // Hook to be called at the end of a pass.
  virtual void EndPassHook() {}

  // Hook to be called before encoding a frame.
  virtual void PreEncodeFrameHook(VideoSource * /*video*/) {}
  virtual void PreEncodeFrameHook(VideoSource * /*video*/,
                                  Encoder * /*encoder*/) {}

  virtual void PreDecodeFrameHook(VideoSource * /*video*/,
                                  Decoder * /*decoder*/) {}

  virtual void PostEncodeFrameHook(Encoder * /*encoder*/) {}

  // Hook to be called on every compressed data packet.
  virtual void FramePktHook(const vpx_codec_cx_pkt_t * /*pkt*/) {}

  // Hook to be called on every PSNR packet.
  virtual void PSNRPktHook(const vpx_codec_cx_pkt_t * /*pkt*/) {}

  // Hook to be called on every first pass stats packet.
  virtual void StatsPktHook(const vpx_codec_cx_pkt_t * /*pkt*/) {}

  // Hook to determine whether the encode loop should continue.
  virtual bool Continue() const {
    return !(::testing::Test::HasFatalFailure() || abort_);
  }

  const CodecFactory *codec_;
  // Hook to determine whether to decode frame after encoding
  virtual bool DoDecode() const { return true; }

  // Hook to handle encode/decode mismatch
  virtual void MismatchHook(const vpx_image_t *img1, const vpx_image_t *img2);

  // Hook to be called on every decompressed frame.
  virtual void DecompressedFrameHook(const vpx_image_t & /*img*/,
                                     vpx_codec_pts_t /*pts*/) {}

  // Hook to be called to handle decode result. Return true to continue.
  virtual bool HandleDecodeResult(const vpx_codec_err_t res_dec,
                                  const VideoSource & /*video*/,
                                  Decoder *decoder) {
    EXPECT_EQ(VPX_CODEC_OK, res_dec) << decoder->DecodeError();
    return VPX_CODEC_OK == res_dec;
  }

  // Hook that can modify the encoder's output data
  virtual const vpx_codec_cx_pkt_t *MutateEncoderOutputHook(
      const vpx_codec_cx_pkt_t *pkt) {
    return pkt;
  }

  bool abort_;
  vpx_codec_enc_cfg_t cfg_;
  vpx_codec_dec_cfg_t dec_cfg_;
  unsigned int passes_;
  vpx_enc_deadline_t deadline_;
  TwopassStatsStore stats_;
  unsigned long init_flags_;
  vpx_enc_frame_flags_t frame_flags_;
};

}  // namespace libvpx_test

#endif  // VPX_TEST_ENCODE_TEST_DRIVER_H_
