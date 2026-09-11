/*
 *  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include "vp9/ratectrl_rtc.h"

#include <new>

#include "vp9/common/vp9_common.h"
#include "vp9/encoder/vp9_aq_cyclicrefresh.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_picklpf.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_codec.h"
#include "vpx_mem/vpx_mem.h"

namespace libvpx {

std::unique_ptr<VP9RateControlRTC> VP9RateControlRTC::Create(
    const VP9RateControlRtcConfig &cfg) {
  std::unique_ptr<VP9RateControlRTC> rc_api(new (std::nothrow)
                                                VP9RateControlRTC());
  if (!rc_api) return nullptr;
  rc_api->cpi_ = static_cast<VP9_COMP *>(vpx_memalign(32, sizeof(*cpi_)));
  if (!rc_api->cpi_) return nullptr;
  vp9_zero(*rc_api->cpi_);

  if (!rc_api->InitRateControl(cfg)) return nullptr;
  if (cfg.aq_mode) {
    VP9_COMP *const cpi = rc_api->cpi_;
    cpi->segmentation_map = static_cast<uint8_t *>(
        vpx_calloc(cpi->common.mi_rows * cpi->common.mi_cols,
                   sizeof(*cpi->segmentation_map)));
    if (!cpi->segmentation_map) return nullptr;
    cpi->cyclic_refresh =
        vp9_cyclic_refresh_alloc(cpi->common.mi_rows, cpi->common.mi_cols);
    cpi->cyclic_refresh->content_mode = 0;
  }
  return rc_api;
}

VP9RateControlRTC::~VP9RateControlRTC() {
  if (cpi_) {
    if (cpi_->svc.number_spatial_layers > 1 ||
        cpi_->svc.number_temporal_layers > 1) {
      for (int sl = 0; sl < cpi_->svc.number_spatial_layers; sl++) {
        for (int tl = 0; tl < cpi_->svc.number_temporal_layers; tl++) {
          int layer = LAYER_IDS_TO_IDX(sl, tl, cpi_->oxcf.ts_number_layers);
          LAYER_CONTEXT *const lc = &cpi_->svc.layer_context[layer];
          vpx_free(lc->map);
          vpx_free(lc->last_coded_q_map);
          vpx_free(lc->consec_zero_mv);
        }
      }
    }
    if (cpi_->oxcf.aq_mode == CYCLIC_REFRESH_AQ) {
      vpx_free(cpi_->segmentation_map);
      cpi_->segmentation_map = NULL;
      vp9_cyclic_refresh_free(cpi_->cyclic_refresh);
    }
    vpx_free(cpi_);
  }
}

bool VP9RateControlRTC::InitRateControl(const VP9RateControlRtcConfig &rc_cfg) {
  VP9_COMMON *cm = &cpi_->common;
  VP9EncoderConfig *oxcf = &cpi_->oxcf;
  RATE_CONTROL *const rc = &cpi_->rc;
  cm->profile = PROFILE_0;
  cm->bit_depth = VPX_BITS_8;
  cm->show_frame = 1;
  oxcf->profile = cm->profile;
  oxcf->bit_depth = cm->bit_depth;
  oxcf->rc_mode = rc_cfg.rc_mode;
  oxcf->pass = 0;
  oxcf->aq_mode = rc_cfg.aq_mode ? CYCLIC_REFRESH_AQ : NO_AQ;
  oxcf->content = VP9E_CONTENT_DEFAULT;
  oxcf->drop_frames_water_mark = 0;
  cm->current_video_frame = 0;
  rc->kf_boost = DEFAULT_KF_BOOST;

  if (!UpdateRateControl(rc_cfg)) return false;
  vp9_set_mb_mi(cm, cm->width, cm->height);

  cpi_->use_svc = (cpi_->svc.number_spatial_layers > 1 ||
                   cpi_->svc.number_temporal_layers > 1)
                      ? 1
                      : 0;

  rc->rc_1_frame = 0;
  rc->rc_2_frame = 0;
  vp9_rc_init_minq_luts();
  vp9_rc_init(oxcf, 0, rc);
  rc->constrain_gf_key_freq_onepass_vbr = 0;
  cpi_->sf.use_nonrd_pick_mode = 1;
  return true;
}

bool VP9RateControlRTC::UpdateRateControl(
    const VP9RateControlRtcConfig &rc_cfg) {
  // Since VPX_MAX_LAYERS (12) is less than the product of VPX_SS_MAX_LAYERS (5)
  // and VPX_TS_MAX_LAYERS (5), check all three.
  if (rc_cfg.ss_number_layers < 1 ||
      rc_cfg.ss_number_layers > VPX_SS_MAX_LAYERS ||
      rc_cfg.ts_number_layers < 1 ||
      rc_cfg.ts_number_layers > VPX_TS_MAX_LAYERS ||
      rc_cfg.ss_number_layers * rc_cfg.ts_number_layers > VPX_MAX_LAYERS) {
    return false;
  }

  VP9_COMMON *cm = &cpi_->common;
  VP9EncoderConfig *oxcf = &cpi_->oxcf;
  RATE_CONTROL *const rc = &cpi_->rc;

  cm->width = rc_cfg.width;
  cm->height = rc_cfg.height;
  oxcf->width = rc_cfg.width;
  oxcf->height = rc_cfg.height;
  oxcf->worst_allowed_q = vp9_quantizer_to_qindex(rc_cfg.max_quantizer);
  oxcf->best_allowed_q = vp9_quantizer_to_qindex(rc_cfg.min_quantizer);
  rc->worst_quality = oxcf->worst_allowed_q;
  rc->best_quality = oxcf->best_allowed_q;
  oxcf->init_framerate = rc_cfg.framerate;
  oxcf->target_bandwidth = 1000 * rc_cfg.target_bandwidth;
  oxcf->starting_buffer_level_ms = rc_cfg.buf_initial_sz;
  oxcf->optimal_buffer_level_ms = rc_cfg.buf_optimal_sz;
  oxcf->maximum_buffer_size_ms = rc_cfg.buf_sz;
  oxcf->under_shoot_pct = rc_cfg.undershoot_pct;
  oxcf->over_shoot_pct = rc_cfg.overshoot_pct;
  oxcf->drop_frames_water_mark = rc_cfg.frame_drop_thresh;
  oxcf->content = rc_cfg.is_screen ? VP9E_CONTENT_SCREEN : VP9E_CONTENT_DEFAULT;
  oxcf->ss_number_layers = rc_cfg.ss_number_layers;
  oxcf->ts_number_layers = rc_cfg.ts_number_layers;
  oxcf->temporal_layering_mode =
      (VP9E_TEMPORAL_LAYERING_MODE)((rc_cfg.ts_number_layers > 1)
                                        ? rc_cfg.ts_number_layers
                                        : 0);

  cpi_->oxcf.rc_max_intra_bitrate_pct = rc_cfg.max_intra_bitrate_pct;
  cpi_->oxcf.rc_max_inter_bitrate_pct = rc_cfg.max_inter_bitrate_pct;
  cpi_->framerate = rc_cfg.framerate;
  cpi_->svc.number_spatial_layers = rc_cfg.ss_number_layers;
  cpi_->svc.number_temporal_layers = rc_cfg.ts_number_layers;

  vp9_set_mb_mi(cm, cm->width, cm->height);

  if (setjmp(cpi_->common.error.jmp)) {
    cpi_->common.error.setjmp = 0;
    vpx_clear_system_state();
    return false;
  }
  cpi_->common.error.setjmp = 1;

  for (int tl = 0; tl < cpi_->svc.number_temporal_layers; ++tl) {
    oxcf->ts_rate_decimator[tl] = rc_cfg.ts_rate_decimator[tl];
  }
  for (int sl = 0; sl < cpi_->svc.number_spatial_layers; ++sl) {
    for (int tl = 0; tl < cpi_->svc.number_temporal_layers; ++tl) {
      const int layer =
          LAYER_IDS_TO_IDX(sl, tl, cpi_->svc.number_temporal_layers);
      LAYER_CONTEXT *lc = &cpi_->svc.layer_context[layer];
      RATE_CONTROL *const lrc = &lc->rc;
      oxcf->layer_target_bitrate[layer] =
          1000 * rc_cfg.layer_target_bitrate[layer];
      lrc->worst_quality =
          vp9_quantizer_to_qindex(rc_cfg.max_quantizers[layer]);
      lrc->best_quality = vp9_quantizer_to_qindex(rc_cfg.min_quantizers[layer]);
      lc->scaling_factor_num = rc_cfg.scaling_factor_num[sl];
      lc->scaling_factor_den = rc_cfg.scaling_factor_den[sl];
    }
  }
  vp9_set_rc_buffer_sizes(cpi_);
  vp9_new_framerate(cpi_, cpi_->framerate);
  if (cpi_->svc.number_temporal_layers > 1 ||
      cpi_->svc.number_spatial_layers > 1) {
    if (cm->current_video_frame == 0) {
      vp9_init_layer_context(cpi_);
      // svc->framedrop_mode is not currently exposed, so only allow for
      // full superframe drop for now.
      cpi_->svc.framedrop_mode = FULL_SUPERFRAME_DROP;
    }
    vp9_update_layer_context_change_config(cpi_,
                                           (int)cpi_->oxcf.target_bandwidth);
    cpi_->svc.max_consec_drop = rc_cfg.max_consec_drop;
  }
  vp9_check_reset_rc_flag(cpi_);

  cpi_->common.error.setjmp = 0;
  return true;
}

// Compute the QP for the frame. If the frame is dropped this function
// returns kDrop, and no QP is computed. If the frame is encoded (not dropped)
// the QP is computed and kOk is returned.
FrameDropDecision VP9RateControlRTC::ComputeQP(
    const VP9FrameParamsQpRTC &frame_params) {
  VP9_COMMON *const cm = &cpi_->common;
  int width, height;
  cpi_->svc.spatial_layer_id = frame_params.spatial_layer_id;
  cpi_->svc.temporal_layer_id = frame_params.temporal_layer_id;
  if (cpi_->svc.number_spatial_layers > 1) {
    const int layer = LAYER_IDS_TO_IDX(cpi_->svc.spatial_layer_id,
                                       cpi_->svc.temporal_layer_id,
                                       cpi_->svc.number_temporal_layers);
    LAYER_CONTEXT *lc = &cpi_->svc.layer_context[layer];
    get_layer_resolution(cpi_->oxcf.width, cpi_->oxcf.height,
                         lc->scaling_factor_num, lc->scaling_factor_den, &width,
                         &height);
    cm->width = width;
    cm->height = height;
  }
  vp9_set_mb_mi(cm, cm->width, cm->height);
  cm->frame_type = static_cast<FRAME_TYPE>(frame_params.frame_type);
  // This is needed to ensure key frame does not get unset in rc_get_svc_params.
  cpi_->frame_flags = (cm->frame_type == KEY_FRAME) ? FRAMEFLAGS_KEY : 0;
  cpi_->refresh_golden_frame = (cm->frame_type == KEY_FRAME) ? 1 : 0;
  cpi_->sf.use_nonrd_pick_mode = 1;
  if (cpi_->svc.number_spatial_layers == 1 &&
      cpi_->svc.number_temporal_layers == 1) {
    int target = 0;
    if (cpi_->oxcf.rc_mode == VPX_CBR) {
      if (cpi_->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
        vp9_cyclic_refresh_update_parameters(cpi_);
      if (frame_is_intra_only(cm))
        target = vp9_calc_iframe_target_size_one_pass_cbr(cpi_);
      else
        target = vp9_calc_pframe_target_size_one_pass_cbr(cpi_);
    } else if (cpi_->oxcf.rc_mode == VPX_VBR) {
      if (cm->frame_type == KEY_FRAME) {
        cpi_->rc.this_key_frame_forced = cm->current_video_frame != 0;
        cpi_->rc.frames_to_key = cpi_->oxcf.key_freq;
      }
      vp9_set_gf_update_one_pass_vbr(cpi_);
      if (cpi_->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
        vp9_cyclic_refresh_update_parameters(cpi_);
      if (frame_is_intra_only(cm))
        target = vp9_calc_iframe_target_size_one_pass_vbr(cpi_);
      else
        target = vp9_calc_pframe_target_size_one_pass_vbr(cpi_);
    }
    vp9_rc_set_frame_target(cpi_, target);
    vp9_update_buffer_level_preencode(cpi_);
  } else {
    vp9_update_temporal_layer_framerate(cpi_);
    vp9_restore_layer_context(cpi_);
    vp9_rc_get_svc_params(cpi_);
  }
  if (cpi_->svc.spatial_layer_id == 0) vp9_zero(cpi_->svc.drop_spatial_layer);
  // SVC: check for skip encoding of enhancement layer if the
  // layer target bandwidth = 0.
  if (vp9_svc_check_skip_enhancement_layer(cpi_))
    return FrameDropDecision::kDrop;
  // Check for dropping this frame based on buffer level.
  // Never drop on key frame, or if base layer is key for svc,
  if (!frame_is_intra_only(cm) &&
      (!cpi_->use_svc ||
       !cpi_->svc.layer_context[cpi_->svc.temporal_layer_id].is_key_frame)) {
    if (vp9_rc_drop_frame(cpi_)) {
      // For FULL_SUPERFRAME_DROP mode (the only mode considered here):
      // if the superframe drop is decided we need to save the layer context for
      // all spatial layers, and call update_buffer_level and postencode_drop
      // for all spatial layers.
      if (cpi_->svc.number_spatial_layers > 1 ||
          cpi_->svc.number_temporal_layers > 1) {
        vp9_save_layer_context(cpi_);
        for (int sl = 1; sl < cpi_->svc.number_spatial_layers; sl++) {
          cpi_->svc.spatial_layer_id = sl;
          vp9_restore_layer_context(cpi_);
          vp9_update_buffer_level_svc_preencode(cpi_);
          vp9_rc_postencode_update_drop_frame(cpi_);
          vp9_save_layer_context(cpi_);
        }
      }
      return FrameDropDecision::kDrop;
    }
  }
  // Compute the QP for the frame.
  int bottom_index, top_index;
  cpi_->common.base_qindex =
      vp9_rc_pick_q_and_bounds(cpi_, &bottom_index, &top_index);

  if (cpi_->oxcf.aq_mode == CYCLIC_REFRESH_AQ) vp9_cyclic_refresh_setup(cpi_);
  if (cpi_->svc.number_spatial_layers > 1 ||
      cpi_->svc.number_temporal_layers > 1)
    vp9_save_layer_context(cpi_);

  cpi_->last_frame_dropped = 0;
  cpi_->svc.last_layer_dropped[cpi_->svc.spatial_layer_id] = 0;
  if (cpi_->svc.spatial_layer_id == cpi_->svc.number_spatial_layers - 1)
    cpi_->svc.num_encoded_top_layer++;

  return FrameDropDecision::kOk;
}

int VP9RateControlRTC::GetQP() const { return cpi_->common.base_qindex; }

int VP9RateControlRTC::GetLoopfilterLevel() const {
  struct loopfilter *const lf = &cpi_->common.lf;
  vp9_pick_filter_level(nullptr, cpi_, LPF_PICK_FROM_Q);
  return lf->filter_level;
}

bool VP9RateControlRTC::GetSegmentationData(
    VP9SegmentationData *segmentation_data) const {
  if (!cpi_->cyclic_refresh || !cpi_->cyclic_refresh->apply_cyclic_refresh) {
    return false;
  }

  segmentation_data->segmentation_map = cpi_->segmentation_map;
  segmentation_data->segmentation_map_size =
      cpi_->common.mi_cols * cpi_->common.mi_rows;
  segmentation_data->delta_q = cpi_->cyclic_refresh->qindex_delta;
  segmentation_data->delta_q_size = 3u;
  return true;
}

void VP9RateControlRTC::PostEncodeUpdate(
    uint64_t encoded_frame_size, const VP9FrameParamsQpRTC &frame_params) {
  cpi_->common.frame_type = static_cast<FRAME_TYPE>(frame_params.frame_type);
  cpi_->svc.spatial_layer_id = frame_params.spatial_layer_id;
  cpi_->svc.temporal_layer_id = frame_params.temporal_layer_id;
  if (cpi_->svc.number_spatial_layers > 1 ||
      cpi_->svc.number_temporal_layers > 1) {
    vp9_restore_layer_context(cpi_);
    const int layer = LAYER_IDS_TO_IDX(cpi_->svc.spatial_layer_id,
                                       cpi_->svc.temporal_layer_id,
                                       cpi_->svc.number_temporal_layers);
    LAYER_CONTEXT *lc = &cpi_->svc.layer_context[layer];
    cpi_->common.base_qindex = lc->frame_qp;
    cpi_->common.MBs = lc->MBs;
    // For spatial-svc, allow cyclic-refresh to be applied on the spatial
    // layers, for the base temporal layer.
    if (cpi_->oxcf.aq_mode == CYCLIC_REFRESH_AQ &&
        cpi_->svc.number_spatial_layers > 1 &&
        cpi_->svc.temporal_layer_id == 0) {
      CYCLIC_REFRESH *const cr = cpi_->cyclic_refresh;
      cr->qindex_delta[0] = lc->qindex_delta[0];
      cr->qindex_delta[1] = lc->qindex_delta[1];
      cr->qindex_delta[2] = lc->qindex_delta[2];
    }
  }
  vp9_rc_postencode_update(cpi_, encoded_frame_size);
  if (cpi_->svc.number_spatial_layers > 1 ||
      cpi_->svc.number_temporal_layers > 1)
    vp9_save_layer_context(cpi_);
  cpi_->common.current_video_frame++;
}

}  // namespace libvpx
