/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_ENCODER_VP9_EXT_RATECTRL_H_
#define VPX_VP9_ENCODER_VP9_EXT_RATECTRL_H_

#include "vpx/vpx_ext_ratectrl.h"
#include "vpx/vpx_tpl.h"
#include "vp9/encoder/vp9_firstpass.h"

typedef struct EXT_RATECTRL {
  int ready;
  int ext_rdmult;
  vpx_rc_model_t model;
  vpx_rc_funcs_t funcs;
  vpx_rc_config_t ratectrl_config;
  vpx_rc_firstpass_stats_t rc_firstpass_stats;
  FILE *log_file;
} EXT_RATECTRL;

vpx_codec_err_t vp9_extrc_init(EXT_RATECTRL *ext_ratectrl);

vpx_codec_err_t vp9_extrc_create(vpx_rc_funcs_t funcs,
                                 vpx_rc_config_t ratectrl_config,
                                 EXT_RATECTRL *ext_ratectrl);

vpx_codec_err_t vp9_extrc_delete(EXT_RATECTRL *ext_ratectrl);

vpx_codec_err_t vp9_extrc_send_firstpass_stats(
    EXT_RATECTRL *ext_ratectrl, const FIRST_PASS_INFO *first_pass_info);

vpx_codec_err_t vp9_extrc_send_tpl_stats(EXT_RATECTRL *ext_ratectrl,
                                         const VpxTplGopStats *tpl_gop_stats);

vpx_codec_err_t vp9_extrc_get_encodeframe_decision(
    EXT_RATECTRL *ext_ratectrl, int gop_index,
    vpx_rc_encodeframe_decision_t *encode_frame_decision);

vpx_codec_err_t vp9_extrc_update_encodeframe_result(EXT_RATECTRL *ext_ratectrl,
                                                    int64_t bit_count,
                                                    int actual_encoding_qindex);

vpx_codec_err_t vp9_extrc_get_key_frame_decision(
    EXT_RATECTRL *ext_ratectrl,
    vpx_rc_key_frame_decision_t *key_frame_decision);

vpx_codec_err_t vp9_extrc_get_gop_decision(EXT_RATECTRL *ext_ratectrl,
                                           vpx_rc_gop_decision_t *gop_decision);

vpx_codec_err_t vp9_extrc_get_frame_rdmult(
    EXT_RATECTRL *ext_ratectrl, int show_index, int coding_index, int gop_index,
    FRAME_UPDATE_TYPE update_type, int gop_size, int use_alt_ref,
    RefCntBuffer *ref_frame_bufs[MAX_INTER_REF_FRAMES], int ref_frame_flags,
    int *rdmult);

#endif  // VPX_VP9_ENCODER_VP9_EXT_RATECTRL_H_
