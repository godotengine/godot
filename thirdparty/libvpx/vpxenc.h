/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_VPXENC_H_
#define VPX_VPXENC_H_

#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

enum TestDecodeFatality {
  TEST_DECODE_OFF,
  TEST_DECODE_FATAL,
  TEST_DECODE_WARN,
};

typedef enum {
  I420,  // 4:2:0 8+ bit-depth
  I422,  // 4:2:2 8+ bit-depth
  I444,  // 4:4:4 8+ bit-depth
  I440,  // 4:4:0 8+ bit-depth
  YV12,  // 4:2:0 with uv flipped, only 8-bit depth
  NV12,  // 4:2:0 with uv interleaved
} ColorInputType;

struct VpxInterface;

/* Configuration elements common to all streams. */
struct VpxEncoderConfig {
  const struct VpxInterface *codec;
  int passes;
  int pass;
  int usage;
  int deadline;
  ColorInputType color_type;
  int quiet;
  int verbose;
  int limit;
  int skip_frames;
  int show_psnr;
  enum TestDecodeFatality test_decode;
  int have_framerate;
  struct vpx_rational framerate;
  int out_part;
  int debug;
  int show_q_hist_buckets;
  int show_rate_hist_buckets;
  int disable_warnings;
  int disable_warning_prompt;
  int experimental_bitstream;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPXENC_H_
