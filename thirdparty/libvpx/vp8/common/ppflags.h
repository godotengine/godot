/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_PPFLAGS_H_
#define VPX_VP8_COMMON_PPFLAGS_H_

#ifdef __cplusplus
extern "C" {
#endif
enum {
  VP8D_NOFILTERING = 0,
  VP8D_DEBLOCK = 1 << 0,
  VP8D_DEMACROBLOCK = 1 << 1,
  VP8D_ADDNOISE = 1 << 2,
  VP8D_MFQE = 1 << 3
};

typedef struct {
  int post_proc_flag;
  int deblocking_level;
  int noise_level;
  int display_ref_frame_flag;
  int display_mb_modes_flag;
  int display_b_modes_flag;
  int display_mv_flag;
} vp8_ppflags_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_PPFLAGS_H_
