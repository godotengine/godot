/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_FIRSTPASS_H_
#define VPX_VP8_ENCODER_FIRSTPASS_H_

#ifdef __cplusplus
extern "C" {
#endif

extern void vp8_init_first_pass(VP8_COMP *cpi);
extern void vp8_first_pass(VP8_COMP *cpi);
extern void vp8_end_first_pass(VP8_COMP *cpi);

extern void vp8_init_second_pass(VP8_COMP *cpi);
extern void vp8_second_pass(VP8_COMP *cpi);
extern void vp8_end_second_pass(VP8_COMP *cpi);

extern size_t vp8_firstpass_stats_sz(unsigned int mb_count);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_FIRSTPASS_H_
