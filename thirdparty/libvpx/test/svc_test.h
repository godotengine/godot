/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_TEST_SVC_TEST_H_
#define VPX_TEST_SVC_TEST_H_

#include "./vpx_config.h"
#include "gtest/gtest.h"
#include "test/codec_factory.h"
#include "test/encode_test_driver.h"
#include "test/i420_video_source.h"
#include "test/util.h"
#include "test/y4m_video_source.h"
#include "vpx/vpx_codec.h"
#include "vpx_ports/bitops.h"

namespace svc_test {
class OnePassCbrSvc : public ::libvpx_test::EncoderTest {
 public:
  explicit OnePassCbrSvc(const ::libvpx_test::CodecFactory *codec)
      : EncoderTest(codec), base_speed_setting_(0), speed_setting_(0),
        superframe_count_(0), temporal_layer_id_(0), number_temporal_layers_(0),
        number_spatial_layers_(0) {
    memset(&svc_params_, 0, sizeof(svc_params_));
    memset(bits_in_buffer_model_, 0,
           sizeof(bits_in_buffer_model_[0]) * VPX_MAX_LAYERS);
    memset(layer_target_avg_bandwidth_, 0,
           sizeof(layer_target_avg_bandwidth_[0]) * VPX_MAX_LAYERS);
  }

 protected:
  ~OnePassCbrSvc() override {}

  virtual void SetConfig(const int num_temporal_layer) = 0;

  virtual void SetSvcConfig(const int num_spatial_layer,
                            const int num_temporal_layer);

  virtual void PreEncodeFrameHookSetup(::libvpx_test::VideoSource *video,
                                       ::libvpx_test::Encoder *encoder);

  void PostEncodeFrameHook(::libvpx_test::Encoder *encoder) override;

  virtual void AssignLayerBitrates();

  void MismatchHook(const vpx_image_t *, const vpx_image_t *) override {}

  vpx_svc_extra_cfg_t svc_params_;
  int64_t bits_in_buffer_model_[VPX_MAX_LAYERS];
  int layer_target_avg_bandwidth_[VPX_MAX_LAYERS];
  int base_speed_setting_;
  int speed_setting_;
  int superframe_count_;
  int temporal_layer_id_;
  int number_temporal_layers_;
  int number_spatial_layers_;
};
}  // namespace svc_test

#endif  // VPX_TEST_SVC_TEST_H_
