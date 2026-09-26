/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_TEST_TEST_VECTORS_H_
#define VPX_TEST_TEST_VECTORS_H_

#include "./vpx_config.h"

namespace libvpx_test {

#if CONFIG_VP8_DECODER
extern const int kNumVP8TestVectors;
extern const char *const kVP8TestVectors[];
#endif

#if CONFIG_VP9_DECODER
extern const int kNumVP9TestVectors;
extern const char *const kVP9TestVectors[];
extern const int kNumVP9TestVectorsSvc;
extern const char *const kVP9TestVectorsSvc[];
extern const int kNumVP9TestVectorsResize;
extern const char *const kVP9TestVectorsResize[];
#endif  // CONFIG_VP9_DECODER

}  // namespace libvpx_test

#endif  // VPX_TEST_TEST_VECTORS_H_
