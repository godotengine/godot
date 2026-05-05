/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_TEST_CODEC_FACTORY_H_
#define VPX_TEST_CODEC_FACTORY_H_

#include <tuple>

#include "./vpx_config.h"
#include "vpx/vpx_decoder.h"
#include "vpx/vpx_encoder.h"
#if CONFIG_VP8_ENCODER || CONFIG_VP9_ENCODER
#include "vpx/vp8cx.h"
#endif
#if CONFIG_VP8_DECODER || CONFIG_VP9_DECODER
#include "vpx/vp8dx.h"
#endif

#include "test/decode_test_driver.h"
#include "test/encode_test_driver.h"
namespace libvpx_test {

const int kCodecFactoryParam = 0;

class CodecFactory {
 public:
  CodecFactory() {}

  virtual ~CodecFactory() {}

  virtual Decoder *CreateDecoder(vpx_codec_dec_cfg_t cfg) const = 0;

  virtual Decoder *CreateDecoder(vpx_codec_dec_cfg_t cfg,
                                 const vpx_codec_flags_t flags) const = 0;

  virtual Encoder *CreateEncoder(vpx_codec_enc_cfg_t cfg,
                                 vpx_enc_deadline_t deadline,
                                 const unsigned long init_flags,
                                 TwopassStatsStore *stats) const = 0;

  virtual vpx_codec_err_t DefaultEncoderConfig(vpx_codec_enc_cfg_t *cfg,
                                               int usage) const = 0;
};

/* Provide CodecTestWith<n>Params classes for a variable number of parameters
 * to avoid having to include a pointer to the CodecFactory in every test
 * definition.
 */
template <class T1>
class CodecTestWithParam
    : public ::testing::TestWithParam<
          std::tuple<const libvpx_test::CodecFactory *, T1> > {};

template <class T1, class T2>
class CodecTestWith2Params
    : public ::testing::TestWithParam<
          std::tuple<const libvpx_test::CodecFactory *, T1, T2> > {};

template <class T1, class T2, class T3>
class CodecTestWith3Params
    : public ::testing::TestWithParam<
          std::tuple<const libvpx_test::CodecFactory *, T1, T2, T3> > {};

template <class T1, class T2, class T3, class T4>
class CodecTestWith4Params
    : public ::testing::TestWithParam<
          std::tuple<const libvpx_test::CodecFactory *, T1, T2, T3, T4> > {};

/*
 * VP8 Codec Definitions
 */
#if CONFIG_VP8
class VP8Decoder : public Decoder {
 public:
  explicit VP8Decoder(vpx_codec_dec_cfg_t cfg) : Decoder(cfg) {}

  VP8Decoder(vpx_codec_dec_cfg_t cfg, const vpx_codec_flags_t flag)
      : Decoder(cfg, flag) {}

 protected:
  vpx_codec_iface_t *CodecInterface() const override {
#if CONFIG_VP8_DECODER
    return &vpx_codec_vp8_dx_algo;
#else
    return nullptr;
#endif
  }
};

class VP8Encoder : public Encoder {
 public:
  VP8Encoder(vpx_codec_enc_cfg_t cfg, vpx_enc_deadline_t deadline,
             const unsigned long init_flags, TwopassStatsStore *stats)
      : Encoder(cfg, deadline, init_flags, stats) {}

 protected:
  vpx_codec_iface_t *CodecInterface() const override {
#if CONFIG_VP8_ENCODER
    return &vpx_codec_vp8_cx_algo;
#else
    return nullptr;
#endif
  }
};

class VP8CodecFactory : public CodecFactory {
 public:
  VP8CodecFactory() : CodecFactory() {}

  Decoder *CreateDecoder(vpx_codec_dec_cfg_t cfg) const override {
    return CreateDecoder(cfg, 0);
  }

  Decoder *CreateDecoder(vpx_codec_dec_cfg_t cfg,
                         const vpx_codec_flags_t flags) const override {
#if CONFIG_VP8_DECODER
    return new VP8Decoder(cfg, flags);
#else
    (void)cfg;
    (void)flags;
    return nullptr;
#endif
  }

  Encoder *CreateEncoder(vpx_codec_enc_cfg_t cfg, vpx_enc_deadline_t deadline,
                         const unsigned long init_flags,
                         TwopassStatsStore *stats) const override {
#if CONFIG_VP8_ENCODER
    return new VP8Encoder(cfg, deadline, init_flags, stats);
#else
    (void)cfg;
    (void)deadline;
    (void)init_flags;
    (void)stats;
    return nullptr;
#endif
  }

  vpx_codec_err_t DefaultEncoderConfig(vpx_codec_enc_cfg_t *cfg,
                                       int usage) const override {
#if CONFIG_VP8_ENCODER
    return vpx_codec_enc_config_default(&vpx_codec_vp8_cx_algo, cfg, usage);
#else
    (void)cfg;
    (void)usage;
    return VPX_CODEC_INCAPABLE;
#endif
  }
};

const libvpx_test::VP8CodecFactory kVP8;

#define VP8_INSTANTIATE_TEST_SUITE(test, ...)                               \
  INSTANTIATE_TEST_SUITE_P(                                                 \
      VP8, test,                                                            \
      ::testing::Combine(                                                   \
          ::testing::Values(static_cast<const libvpx_test::CodecFactory *>( \
              &libvpx_test::kVP8)),                                         \
          __VA_ARGS__))
#else
// static_assert() is used to avoid warnings about an extra ';' outside of a
// function.
#define VP8_INSTANTIATE_TEST_SUITE(test, ...) static_assert(CONFIG_VP8 == 0, "")
#endif  // CONFIG_VP8

/*
 * VP9 Codec Definitions
 */
#if CONFIG_VP9
class VP9Decoder : public Decoder {
 public:
  explicit VP9Decoder(vpx_codec_dec_cfg_t cfg) : Decoder(cfg) {}

  VP9Decoder(vpx_codec_dec_cfg_t cfg, const vpx_codec_flags_t flag)
      : Decoder(cfg, flag) {}

 protected:
  vpx_codec_iface_t *CodecInterface() const override {
#if CONFIG_VP9_DECODER
    return &vpx_codec_vp9_dx_algo;
#else
    return nullptr;
#endif
  }
};

class VP9Encoder : public Encoder {
 public:
  VP9Encoder(vpx_codec_enc_cfg_t cfg, vpx_enc_deadline_t deadline,
             const unsigned long init_flags, TwopassStatsStore *stats)
      : Encoder(cfg, deadline, init_flags, stats) {}

 protected:
  vpx_codec_iface_t *CodecInterface() const override {
#if CONFIG_VP9_ENCODER
    return &vpx_codec_vp9_cx_algo;
#else
    return nullptr;
#endif
  }
};

class VP9CodecFactory : public CodecFactory {
 public:
  VP9CodecFactory() : CodecFactory() {}

  Decoder *CreateDecoder(vpx_codec_dec_cfg_t cfg) const override {
    return CreateDecoder(cfg, 0);
  }

  Decoder *CreateDecoder(vpx_codec_dec_cfg_t cfg,
                         const vpx_codec_flags_t flags) const override {
#if CONFIG_VP9_DECODER
    return new VP9Decoder(cfg, flags);
#else
    (void)cfg;
    (void)flags;
    return nullptr;
#endif
  }

  Encoder *CreateEncoder(vpx_codec_enc_cfg_t cfg, vpx_enc_deadline_t deadline,
                         const unsigned long init_flags,
                         TwopassStatsStore *stats) const override {
#if CONFIG_VP9_ENCODER
    return new VP9Encoder(cfg, deadline, init_flags, stats);
#else
    (void)cfg;
    (void)deadline;
    (void)init_flags;
    (void)stats;
    return nullptr;
#endif
  }

  vpx_codec_err_t DefaultEncoderConfig(vpx_codec_enc_cfg_t *cfg,
                                       int usage) const override {
#if CONFIG_VP9_ENCODER
    return vpx_codec_enc_config_default(&vpx_codec_vp9_cx_algo, cfg, usage);
#else
    (void)cfg;
    (void)usage;
    return VPX_CODEC_INCAPABLE;
#endif
  }
};

const libvpx_test::VP9CodecFactory kVP9;

#define VP9_INSTANTIATE_TEST_SUITE(test, ...)                               \
  INSTANTIATE_TEST_SUITE_P(                                                 \
      VP9, test,                                                            \
      ::testing::Combine(                                                   \
          ::testing::Values(static_cast<const libvpx_test::CodecFactory *>( \
              &libvpx_test::kVP9)),                                         \
          __VA_ARGS__))
#else
// static_assert() is used to avoid warnings about an extra ';' outside of a
// function.
#define VP9_INSTANTIATE_TEST_SUITE(test, ...) static_assert(CONFIG_VP9 == 0, "")
#endif  // CONFIG_VP9

}  // namespace libvpx_test
#endif  // VPX_TEST_CODEC_FACTORY_H_
