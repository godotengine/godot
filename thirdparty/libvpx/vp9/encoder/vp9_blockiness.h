/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_BLOCKINESS_H_
#define VPX_VP9_ENCODER_VP9_BLOCKINESS_H_

#ifdef __cplusplus
extern "C" {
#endif

double vp9_get_blockiness(const uint8_t *img1, int img1_pitch,
                          const uint8_t *img2, int img2_pitch, int width,
                          int height);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_BLOCKINESS_H_
