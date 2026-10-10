/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_MFQE_H_
#define VPX_VP9_COMMON_VP9_MFQE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Multiframe Quality Enhancement.
// The aim for MFQE is to replace pixel blocks in the current frame with
// the correlated pixel blocks (with higher quality) in the last frame.
// The replacement can only be taken in stationary blocks by checking
// the motion of the blocks and other conditions such as the SAD of
// the current block and correlated block, the variance of the block
// difference, etc.
void vp9_mfqe(struct VP9Common *cm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_MFQE_H_
