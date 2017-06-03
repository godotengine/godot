/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_COMMON_VP9_PPFLAGS_H_
#define VP9_COMMON_VP9_PPFLAGS_H_

#ifdef __cplusplus
extern "C" {
#endif

enum {
  VP9D_NOFILTERING            = 0,
  VP9D_DEBLOCK                = 1 << 0,
  VP9D_DEMACROBLOCK           = 1 << 1,
  VP9D_ADDNOISE               = 1 << 2,
  VP9D_DEBUG_TXT_FRAME_INFO   = 1 << 3,
  VP9D_DEBUG_TXT_MBLK_MODES   = 1 << 4,
  VP9D_DEBUG_TXT_DC_DIFF      = 1 << 5,
  VP9D_DEBUG_TXT_RATE_INFO    = 1 << 6,
  VP9D_DEBUG_DRAW_MV          = 1 << 7,
  VP9D_DEBUG_CLR_BLK_MODES    = 1 << 8,
  VP9D_DEBUG_CLR_FRM_REF_BLKS = 1 << 9,
  VP9D_MFQE                   = 1 << 10
};

typedef struct {
  int post_proc_flag;
  int deblocking_level;
  int noise_level;
} vp9_ppflags_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_COMMON_VP9_PPFLAGS_H_
