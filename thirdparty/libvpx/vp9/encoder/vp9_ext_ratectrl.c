/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stddef.h>
#include <stdio.h>

#include "vp9/encoder/vp9_ext_ratectrl.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/common/vp9_common.h"
#include "vpx_dsp/psnr.h"
#include "vpx/vpx_codec.h"
#include "vpx/vpx_ext_ratectrl.h"
#include "vpx/vpx_tpl.h"

vpx_codec_err_t vp9_extrc_init(EXT_RATECTRL *ext_ratectrl) {
  if (ext_ratectrl == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  vp9_zero(*ext_ratectrl);
  return VPX_CODEC_OK;
}

vpx_codec_err_t vp9_extrc_create(vpx_rc_funcs_t funcs,
                                 vpx_rc_config_t ratectrl_config,
                                 EXT_RATECTRL *ext_ratectrl) {
  vpx_rc_status_t rc_status;
  vpx_rc_firstpass_stats_t *rc_firstpass_stats;
  if (ext_ratectrl == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  vp9_extrc_delete(ext_ratectrl);
  ext_ratectrl->funcs = funcs;
  ext_ratectrl->ratectrl_config = ratectrl_config;
  rc_status = ext_ratectrl->funcs.create_model(ext_ratectrl->funcs.priv,
                                               &ext_ratectrl->ratectrl_config,
                                               &ext_ratectrl->model);
  if (rc_status == VPX_RC_ERROR) {
    return VPX_CODEC_ERROR;
  }
  rc_firstpass_stats = &ext_ratectrl->rc_firstpass_stats;
  rc_firstpass_stats->num_frames = ratectrl_config.show_frame_count;
  rc_firstpass_stats->frame_stats =
      vpx_malloc(sizeof(*rc_firstpass_stats->frame_stats) *
                 rc_firstpass_stats->num_frames);
  if (rc_firstpass_stats->frame_stats == NULL) {
    return VPX_CODEC_MEM_ERROR;
  }
  if (funcs.rate_ctrl_log_path != NULL) {
    ext_ratectrl->log_file = fopen(funcs.rate_ctrl_log_path, "w");
    if (!ext_ratectrl->log_file) {
      return VPX_CODEC_ERROR;
    }
  } else {
    ext_ratectrl->log_file = NULL;
  }
  ext_ratectrl->ready = 1;
  return VPX_CODEC_OK;
}

vpx_codec_err_t vp9_extrc_delete(EXT_RATECTRL *ext_ratectrl) {
  if (ext_ratectrl == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  if (ext_ratectrl->ready) {
    if (ext_ratectrl->log_file) {
      fclose(ext_ratectrl->log_file);
    }
    vpx_rc_status_t rc_status =
        ext_ratectrl->funcs.delete_model(ext_ratectrl->model);
    if (rc_status == VPX_RC_ERROR) {
      return VPX_CODEC_ERROR;
    }
    vpx_free(ext_ratectrl->rc_firstpass_stats.frame_stats);
  }
  return vp9_extrc_init(ext_ratectrl);
}

static void gen_rc_firstpass_stats(const FIRSTPASS_STATS *stats,
                                   vpx_rc_frame_stats_t *rc_frame_stats) {
  rc_frame_stats->frame = stats->frame;
  rc_frame_stats->weight = stats->weight;
  rc_frame_stats->intra_error = stats->intra_error;
  rc_frame_stats->coded_error = stats->coded_error;
  rc_frame_stats->sr_coded_error = stats->sr_coded_error;
  rc_frame_stats->frame_noise_energy = stats->frame_noise_energy;
  rc_frame_stats->pcnt_inter = stats->pcnt_inter;
  rc_frame_stats->pcnt_motion = stats->pcnt_motion;
  rc_frame_stats->pcnt_second_ref = stats->pcnt_second_ref;
  rc_frame_stats->pcnt_neutral = stats->pcnt_neutral;
  rc_frame_stats->pcnt_intra_low = stats->pcnt_intra_low;
  rc_frame_stats->pcnt_intra_high = stats->pcnt_intra_high;
  rc_frame_stats->intra_skip_pct = stats->intra_skip_pct;
  rc_frame_stats->intra_smooth_pct = stats->intra_smooth_pct;
  rc_frame_stats->inactive_zone_rows = stats->inactive_zone_rows;
  rc_frame_stats->inactive_zone_cols = stats->inactive_zone_cols;
  rc_frame_stats->MVr = stats->MVr;
  rc_frame_stats->mvr_abs = stats->mvr_abs;
  rc_frame_stats->MVc = stats->MVc;
  rc_frame_stats->mvc_abs = stats->mvc_abs;
  rc_frame_stats->MVrv = stats->MVrv;
  rc_frame_stats->MVcv = stats->MVcv;
  rc_frame_stats->mv_in_out_count = stats->mv_in_out_count;
  rc_frame_stats->duration = stats->duration;
  rc_frame_stats->count = stats->count;
  rc_frame_stats->new_mv_count = stats->new_mv_count;
}

vpx_codec_err_t vp9_extrc_send_firstpass_stats(
    EXT_RATECTRL *ext_ratectrl, const FIRST_PASS_INFO *first_pass_info) {
  if (ext_ratectrl == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  if (ext_ratectrl->ready) {
    vpx_rc_status_t rc_status;
    vpx_rc_firstpass_stats_t *rc_firstpass_stats =
        &ext_ratectrl->rc_firstpass_stats;
    int i;
    assert(rc_firstpass_stats->num_frames == first_pass_info->num_frames);
    for (i = 0; i < rc_firstpass_stats->num_frames; ++i) {
      gen_rc_firstpass_stats(&first_pass_info->stats[i],
                             &rc_firstpass_stats->frame_stats[i]);
    }
    rc_status = ext_ratectrl->funcs.send_firstpass_stats(ext_ratectrl->model,
                                                         rc_firstpass_stats);
    if (rc_status == VPX_RC_ERROR) {
      return VPX_CODEC_ERROR;
    }
  }
  return VPX_CODEC_OK;
}

vpx_codec_err_t vp9_extrc_send_tpl_stats(EXT_RATECTRL *ext_ratectrl,
                                         const VpxTplGopStats *tpl_gop_stats) {
  if (ext_ratectrl == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  if (ext_ratectrl->ready && ext_ratectrl->funcs.send_tpl_gop_stats != NULL) {
    vpx_rc_status_t rc_status = ext_ratectrl->funcs.send_tpl_gop_stats(
        ext_ratectrl->model, tpl_gop_stats);
    if (rc_status == VPX_RC_ERROR) {
      return VPX_CODEC_ERROR;
    }
  }
  return VPX_CODEC_OK;
}

static int extrc_get_frame_type(FRAME_UPDATE_TYPE update_type) {
  // TODO(angiebird): Add unit test to make sure this function behaves like
  // get_frame_type_from_update_type()
  // TODO(angiebird): Merge this function with get_frame_type_from_update_type()
  switch (update_type) {
    case KF_UPDATE: return 0;       // kFrameTypeKey;
    case ARF_UPDATE: return 2;      // kFrameTypeAltRef;
    case GF_UPDATE: return 4;       // kFrameTypeGolden;
    case OVERLAY_UPDATE: return 3;  // kFrameTypeOverlay;
    case LF_UPDATE: return 1;       // kFrameTypeInter;
    default:
      fprintf(stderr, "Unsupported update_type %d\n", update_type);
      abort();
  }
}

vpx_codec_err_t vp9_extrc_get_encodeframe_decision(
    EXT_RATECTRL *ext_ratectrl, int gop_index,
    vpx_rc_encodeframe_decision_t *encode_frame_decision) {
  assert(ext_ratectrl != NULL);
  assert(ext_ratectrl->ready && (ext_ratectrl->funcs.rc_type & VPX_RC_QP) != 0);

  vpx_rc_status_t rc_status = ext_ratectrl->funcs.get_encodeframe_decision(
      ext_ratectrl->model, gop_index, encode_frame_decision);
  if (rc_status == VPX_RC_ERROR) {
    return VPX_CODEC_ERROR;
  }
  return VPX_CODEC_OK;
}

vpx_codec_err_t vp9_extrc_update_encodeframe_result(
    EXT_RATECTRL *ext_ratectrl, int64_t bit_count, int actual_encoding_qindex) {
  if (ext_ratectrl == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  if (ext_ratectrl->ready) {
    vpx_rc_status_t rc_status;
    vpx_rc_encodeframe_result_t encode_frame_result;
    encode_frame_result.bit_count = bit_count;
    encode_frame_result.actual_encoding_qindex = actual_encoding_qindex;
    rc_status = ext_ratectrl->funcs.update_encodeframe_result(
        ext_ratectrl->model, &encode_frame_result);
    if (rc_status == VPX_RC_ERROR) {
      return VPX_CODEC_ERROR;
    }
  }
  return VPX_CODEC_OK;
}

vpx_codec_err_t vp9_extrc_get_gop_decision(
    EXT_RATECTRL *ext_ratectrl, vpx_rc_gop_decision_t *gop_decision) {
  vpx_rc_status_t rc_status;
  if (ext_ratectrl == NULL || !ext_ratectrl->ready ||
      (ext_ratectrl->funcs.rc_type & VPX_RC_GOP) == 0) {
    return VPX_CODEC_INVALID_PARAM;
  }
  rc_status =
      ext_ratectrl->funcs.get_gop_decision(ext_ratectrl->model, gop_decision);
  if (rc_status == VPX_RC_ERROR) {
    return VPX_CODEC_ERROR;
  }
  return VPX_CODEC_OK;
}

vpx_codec_err_t vp9_extrc_get_key_frame_decision(
    EXT_RATECTRL *ext_ratectrl,
    vpx_rc_key_frame_decision_t *key_frame_decision) {
  if (ext_ratectrl == NULL || !ext_ratectrl->ready ||
      (ext_ratectrl->funcs.rc_type & VPX_RC_GOP) == 0) {
    return VPX_CODEC_INVALID_PARAM;
  }
  vpx_rc_status_t rc_status = ext_ratectrl->funcs.get_key_frame_decision(
      ext_ratectrl->model, key_frame_decision);
  return rc_status == VPX_RC_OK ? VPX_CODEC_OK : VPX_CODEC_ERROR;
}

vpx_codec_err_t vp9_extrc_get_frame_rdmult(
    EXT_RATECTRL *ext_ratectrl, int show_index, int coding_index, int gop_index,
    FRAME_UPDATE_TYPE update_type, int gop_size, int use_alt_ref,
    RefCntBuffer *ref_frame_bufs[MAX_INTER_REF_FRAMES], int ref_frame_flags,
    int *rdmult) {
  vpx_rc_status_t rc_status;
  vpx_rc_encodeframe_info_t encode_frame_info;
  if (ext_ratectrl == NULL || !ext_ratectrl->ready ||
      (ext_ratectrl->funcs.rc_type & VPX_RC_RDMULT) == 0) {
    return VPX_CODEC_INVALID_PARAM;
  }
  encode_frame_info.show_index = show_index;
  encode_frame_info.coding_index = coding_index;
  encode_frame_info.gop_index = gop_index;
  encode_frame_info.frame_type = extrc_get_frame_type(update_type);
  encode_frame_info.gop_size = gop_size;
  encode_frame_info.use_alt_ref = use_alt_ref;

  vp9_get_ref_frame_info(update_type, ref_frame_flags, ref_frame_bufs,
                         encode_frame_info.ref_frame_coding_indexes,
                         encode_frame_info.ref_frame_valid_list);
  rc_status = ext_ratectrl->funcs.get_frame_rdmult(ext_ratectrl->model,
                                                   &encode_frame_info, rdmult);
  if (rc_status == VPX_RC_ERROR) {
    return VPX_CODEC_ERROR;
  }
  return VPX_CODEC_OK;
}
