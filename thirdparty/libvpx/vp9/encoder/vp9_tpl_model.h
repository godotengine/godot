/*
 *  Copyright (c) 2023 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_TPL_MODEL_H_
#define VPX_VP9_ENCODER_VP9_TPL_MODEL_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_LOG2_E
#define M_LOG2_E 0.693147180559945309417
#endif
#define log2f(x) (log(x) / (float)M_LOG2_E)

#define TPL_DEP_COST_SCALE_LOG2 4

typedef struct GF_PICTURE {
  YV12_BUFFER_CONFIG *frame;
  int ref_frame[3];
  FRAME_UPDATE_TYPE update_type;
} GF_PICTURE;

void vp9_init_tpl_buffer(VP9_COMP *cpi);
void vp9_setup_tpl_stats(VP9_COMP *cpi);
void vp9_free_tpl_buffer(VP9_COMP *cpi);
void vp9_estimate_tpl_qp_gop(VP9_COMP *cpi);

void vp9_wht_fwd_txfm(int16_t *src_diff, int bw, tran_low_t *coeff,
                      TX_SIZE tx_size);
#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_wht_fwd_txfm(int16_t *src_diff, int bw, tran_low_t *coeff,
                             TX_SIZE tx_size);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_TPL_MODEL_H_
