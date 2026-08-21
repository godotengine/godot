/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_RATE_HIST_H_
#define VPX_RATE_HIST_H_

#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rate_hist;

struct rate_hist *init_rate_histogram(const vpx_codec_enc_cfg_t *cfg,
                                      const vpx_rational_t *fps);

void destroy_rate_histogram(struct rate_hist *hist);

void update_rate_histogram(struct rate_hist *hist,
                           const vpx_codec_enc_cfg_t *cfg,
                           const vpx_codec_cx_pkt_t *pkt);

void show_q_histogram(const int counts[64], int max_buckets);

void show_rate_histogram(struct rate_hist *hist, const vpx_codec_enc_cfg_t *cfg,
                         int max_buckets);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_RATE_HIST_H_
