/*
 *  Copyright (c) 2021 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/vp8_ratectrl_rtc.h"

#include <math.h>

#include <new>

#include "vp8/common/common.h"
#include "vp8/encoder/onyx_int.h"
#include "vp8/encoder/ratectrl.h"
#include "vpx_ports/system_state.h"

namespace libvpx {
/* Quant MOD */
static const int kQTrans[] = {
  0,  1,  2,  3,  4,  5,  7,   8,   9,   10,  12,  13,  15,  17,  18,  19,
  20, 21, 23, 24, 25, 26, 27,  28,  29,  30,  31,  33,  35,  37,  39,  41,
  43, 45, 47, 49, 51, 53, 55,  57,  59,  61,  64,  67,  70,  73,  76,  79,
  82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127,
};

static const unsigned char kf_high_motion_minq[QINDEX_RANGE] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  5,
  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  8,  8,  8,  8,  9,  9,  10, 10,
  10, 10, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 15, 15, 15, 15, 16,
  16, 16, 16, 17, 17, 18, 18, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21,
  22, 22, 23, 23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 30
};

static const unsigned char inter_minq[QINDEX_RANGE] = {
  0,  0,  1,  1,  2,  3,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9,  9,  10, 11,
  11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 22, 23, 24,
  24, 25, 26, 27, 27, 28, 29, 30, 30, 31, 32, 33, 33, 34, 35, 36, 36, 37, 38,
  39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 50, 51, 52, 53,
  54, 55, 55, 56, 57, 58, 59, 60, 60, 61, 62, 63, 64, 65, 66, 67, 67, 68, 69,
  70, 71, 72, 73, 74, 75, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 86,
  87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
};

static int rescale(int val, int num, int denom) {
  int64_t llnum = num;
  int64_t llden = denom;
  int64_t llval = val;

  return (int)(llval * llnum / llden);
}

std::unique_ptr<VP8RateControlRTC> VP8RateControlRTC::Create(
    const VP8RateControlRtcConfig &cfg) {
  std::unique_ptr<VP8RateControlRTC> rc_api(new (std::nothrow)
                                                VP8RateControlRTC());
  if (!rc_api) return nullptr;
  rc_api->cpi_ = static_cast<VP8_COMP *>(vpx_memalign(32, sizeof(*cpi_)));
  if (!rc_api->cpi_) return nullptr;
  vp8_zero(*rc_api->cpi_);

  if (!rc_api->InitRateControl(cfg)) return nullptr;

  return rc_api;
}

VP8RateControlRTC::~VP8RateControlRTC() {
  if (cpi_) {
    vpx_free(cpi_->gf_active_flags);
    vpx_free(cpi_);
  }
}

bool VP8RateControlRTC::InitRateControl(const VP8RateControlRtcConfig &rc_cfg) {
  VP8_COMMON *cm = &cpi_->common;
  VP8_CONFIG *oxcf = &cpi_->oxcf;
  oxcf->end_usage = USAGE_STREAM_FROM_SERVER;
  cpi_->pass = 0;
  cm->show_frame = 1;
  oxcf->drop_frames_water_mark = 0;
  cm->current_video_frame = 0;
  cpi_->auto_gold = 1;
  cpi_->key_frame_count = 1;
  cpi_->rate_correction_factor = 1.0;
  cpi_->key_frame_rate_correction_factor = 1.0;
  cpi_->cyclic_refresh_mode_enabled = 0;
  cpi_->auto_worst_q = 1;
  cpi_->kf_overspend_bits = 0;
  cpi_->kf_bitrate_adjustment = 0;
  cpi_->gf_overspend_bits = 0;
  cpi_->non_gf_bitrate_adjustment = 0;
  if (!UpdateRateControl(rc_cfg)) return false;
  cpi_->buffer_level = oxcf->starting_buffer_level;
  cpi_->bits_off_target = oxcf->starting_buffer_level;
  return true;
}

bool VP8RateControlRTC::UpdateRateControl(
    const VP8RateControlRtcConfig &rc_cfg) {
  if (rc_cfg.ts_number_layers < 1 ||
      rc_cfg.ts_number_layers > VPX_TS_MAX_LAYERS) {
    return false;
  }

  VP8_COMMON *cm = &cpi_->common;
  VP8_CONFIG *oxcf = &cpi_->oxcf;
  const unsigned int prev_number_of_layers = oxcf->number_of_layers;
  vpx_clear_system_state();
  cm->Width = rc_cfg.width;
  cm->Height = rc_cfg.height;
  oxcf->Width = rc_cfg.width;
  oxcf->Height = rc_cfg.height;
  oxcf->worst_allowed_q = kQTrans[rc_cfg.max_quantizer];
  oxcf->best_allowed_q = kQTrans[rc_cfg.min_quantizer];
  cpi_->worst_quality = oxcf->worst_allowed_q;
  cpi_->best_quality = oxcf->best_allowed_q;
  cpi_->output_framerate = rc_cfg.framerate;
  oxcf->target_bandwidth =
      static_cast<unsigned int>(1000 * rc_cfg.target_bandwidth);
  cpi_->ref_framerate = cpi_->output_framerate;
  oxcf->fixed_q = -1;
  oxcf->error_resilient_mode = 1;
  oxcf->starting_buffer_level_in_ms = rc_cfg.buf_initial_sz;
  oxcf->optimal_buffer_level_in_ms = rc_cfg.buf_optimal_sz;
  oxcf->maximum_buffer_size_in_ms = rc_cfg.buf_sz;
  oxcf->starting_buffer_level = rc_cfg.buf_initial_sz;
  oxcf->optimal_buffer_level = rc_cfg.buf_optimal_sz;
  oxcf->maximum_buffer_size = rc_cfg.buf_sz;
  oxcf->number_of_layers = rc_cfg.ts_number_layers;
  cpi_->buffered_mode = oxcf->optimal_buffer_level > 0;
  oxcf->under_shoot_pct = rc_cfg.undershoot_pct;
  oxcf->over_shoot_pct = rc_cfg.overshoot_pct;
  oxcf->drop_frames_water_mark = rc_cfg.frame_drop_thresh;
  if (oxcf->drop_frames_water_mark > 0) cpi_->drop_frames_allowed = 1;
  cpi_->oxcf.rc_max_intra_bitrate_pct = rc_cfg.max_intra_bitrate_pct;
  cpi_->framerate = rc_cfg.framerate;
  for (int i = 0; i < KEY_FRAME_CONTEXT; ++i) {
    cpi_->prior_key_frame_distance[i] =
        static_cast<int>(cpi_->output_framerate);
  }
  oxcf->screen_content_mode = rc_cfg.is_screen;
  if (oxcf->number_of_layers > 1 || prev_number_of_layers > 1) {
    memcpy(oxcf->target_bitrate, rc_cfg.layer_target_bitrate,
           sizeof(rc_cfg.layer_target_bitrate));
    memcpy(oxcf->rate_decimator, rc_cfg.ts_rate_decimator,
           sizeof(rc_cfg.ts_rate_decimator));
    if (cm->current_video_frame == 0) {
      double prev_layer_framerate = 0;
      for (unsigned int i = 0; i < oxcf->number_of_layers; ++i) {
        vp8_init_temporal_layer_context(cpi_, oxcf, i, prev_layer_framerate);
        prev_layer_framerate = cpi_->output_framerate / oxcf->rate_decimator[i];
      }
    } else if (oxcf->number_of_layers != prev_number_of_layers) {
      // The number of temporal layers has changed, so reset/initialize the
      // temporal layer context for the new layer configuration: this means
      // calling vp8_reset_temporal_layer_change() below.

      // Start at the base of the pattern cycle, so set the layer id to 0 and
      // reset the temporal pattern counter.
      // TODO(marpan/jianj): don't think lines 148-151 are needed (user controls
      // the layer_id) so remove.
      if (cpi_->temporal_layer_id > 0) {
        cpi_->temporal_layer_id = 0;
      }
      cpi_->temporal_pattern_counter = 0;

      vp8_reset_temporal_layer_change(cpi_, oxcf,
                                      static_cast<int>(prev_number_of_layers));
    }
  }

  cpi_->total_actual_bits = 0;
  cpi_->total_target_vs_actual = 0;

  cm->mb_rows = cm->Height >> 4;
  cm->mb_cols = cm->Width >> 4;
  cm->MBs = cm->mb_rows * cm->mb_cols;
  cm->mode_info_stride = cm->mb_cols + 1;

  // For temporal layers: starting/maximum/optimal_buffer_level is already set
  // via vp8_init_temporal_layer_context() or vp8_reset_temporal_layer_change().
  if (oxcf->number_of_layers <= 1 && prev_number_of_layers <= 1) {
    oxcf->starting_buffer_level =
        rescale((int)oxcf->starting_buffer_level, oxcf->target_bandwidth, 1000);
    /* Set or reset optimal and maximum buffer levels. */
    if (oxcf->optimal_buffer_level == 0) {
      oxcf->optimal_buffer_level = oxcf->target_bandwidth / 8;
    } else {
      oxcf->optimal_buffer_level = rescale((int)oxcf->optimal_buffer_level,
                                           oxcf->target_bandwidth, 1000);
    }
    if (oxcf->maximum_buffer_size == 0) {
      oxcf->maximum_buffer_size = oxcf->target_bandwidth / 8;
    } else {
      oxcf->maximum_buffer_size =
          rescale((int)oxcf->maximum_buffer_size, oxcf->target_bandwidth, 1000);
    }
  }

  if (cpi_->bits_off_target > oxcf->maximum_buffer_size) {
    cpi_->bits_off_target = oxcf->maximum_buffer_size;
    cpi_->buffer_level = cpi_->bits_off_target;
  }

  vp8_new_framerate(cpi_, cpi_->framerate);
  vpx_clear_system_state();
  return true;
}

FrameDropDecision VP8RateControlRTC::ComputeQP(
    const VP8FrameParamsQpRTC &frame_params) {
  VP8_COMMON *const cm = &cpi_->common;
  vpx_clear_system_state();
  if (cpi_->oxcf.number_of_layers > 1) {
    cpi_->temporal_layer_id = frame_params.temporal_layer_id;
    const int layer = frame_params.temporal_layer_id;
    vp8_update_layer_contexts(cpi_);
    /* Restore layer specific context & set frame rate */
    vp8_restore_layer_context(cpi_, layer);
    vp8_new_framerate(cpi_, cpi_->layer_context[layer].framerate);
  }
  cm->frame_type = static_cast<FRAME_TYPE>(frame_params.frame_type);
  cm->refresh_golden_frame = (cm->frame_type == KEY_FRAME) ? 1 : 0;
  cm->refresh_alt_ref_frame = (cm->frame_type == KEY_FRAME) ? 1 : 0;
  if (cm->frame_type == KEY_FRAME && cpi_->common.current_video_frame > 0) {
    cpi_->common.frame_flags |= FRAMEFLAGS_KEY;
  }

  cpi_->per_frame_bandwidth = static_cast<int>(
      round(cpi_->oxcf.target_bandwidth / cpi_->output_framerate));
  if (vp8_check_drop_buffer(cpi_)) {
    if (cpi_->oxcf.number_of_layers > 1) vp8_save_layer_context(cpi_);
    return FrameDropDecision::kDrop;
  }

  if (!vp8_pick_frame_size(cpi_)) {
    cm->current_video_frame++;
    cpi_->frames_since_key++;
    cpi_->ext_refresh_frame_flags_pending = 0;
    if (cpi_->oxcf.number_of_layers > 1) vp8_save_layer_context(cpi_);
    return FrameDropDecision::kDrop;
  }

  if (cpi_->buffer_level >= cpi_->oxcf.optimal_buffer_level &&
      cpi_->buffered_mode) {
    /* Max adjustment is 1/4 */
    int Adjustment = cpi_->active_worst_quality / 4;
    if (Adjustment) {
      int buff_lvl_step;
      if (cpi_->buffer_level < cpi_->oxcf.maximum_buffer_size) {
        buff_lvl_step = (int)((cpi_->oxcf.maximum_buffer_size -
                               cpi_->oxcf.optimal_buffer_level) /
                              Adjustment);
        if (buff_lvl_step) {
          Adjustment =
              (int)((cpi_->buffer_level - cpi_->oxcf.optimal_buffer_level) /
                    buff_lvl_step);
        } else {
          Adjustment = 0;
        }
      }
      cpi_->active_worst_quality -= Adjustment;
      if (cpi_->active_worst_quality < cpi_->active_best_quality) {
        cpi_->active_worst_quality = cpi_->active_best_quality;
      }
    }
  }

  if (cpi_->ni_frames > 150) {
    int q = cpi_->active_worst_quality;
    if (cm->frame_type == KEY_FRAME) {
      cpi_->active_best_quality = kf_high_motion_minq[q];
    } else {
      cpi_->active_best_quality = inter_minq[q];
    }

    if (cpi_->buffer_level >= cpi_->oxcf.maximum_buffer_size) {
      cpi_->active_best_quality = cpi_->best_quality;

    } else if (cpi_->buffer_level > cpi_->oxcf.optimal_buffer_level) {
      int Fraction =
          (int)(((cpi_->buffer_level - cpi_->oxcf.optimal_buffer_level) * 128) /
                (cpi_->oxcf.maximum_buffer_size -
                 cpi_->oxcf.optimal_buffer_level));
      int min_qadjustment =
          ((cpi_->active_best_quality - cpi_->best_quality) * Fraction) / 128;

      cpi_->active_best_quality -= min_qadjustment;
    }
  }

  /* Clip the active best and worst quality values to limits */
  if (cpi_->active_worst_quality > cpi_->worst_quality) {
    cpi_->active_worst_quality = cpi_->worst_quality;
  }
  if (cpi_->active_best_quality < cpi_->best_quality) {
    cpi_->active_best_quality = cpi_->best_quality;
  }
  if (cpi_->active_worst_quality < cpi_->active_best_quality) {
    cpi_->active_worst_quality = cpi_->active_best_quality;
  }

  q_ = vp8_regulate_q(cpi_, cpi_->this_frame_target);
  vp8_set_quantizer(cpi_, q_);
  vpx_clear_system_state();
  return FrameDropDecision::kOk;
}

int VP8RateControlRTC::GetQP() const { return q_; }

UVDeltaQP VP8RateControlRTC::GetUVDeltaQP() const {
  VP8_COMMON *cm = &cpi_->common;
  UVDeltaQP uv_delta_q;
  uv_delta_q.uvdc_delta_q = cm->uvdc_delta_q;
  uv_delta_q.uvac_delta_q = cm->uvac_delta_q;
  return uv_delta_q;
}

int VP8RateControlRTC::GetLoopfilterLevel() const {
  VP8_COMMON *cm = &cpi_->common;
  const double qp = q_;

  // This model is from linear regression
  if (cm->Width * cm->Height <= 320 * 240) {
    cm->filter_level = static_cast<int>(0.352685 * qp + 2.957774);
  } else if (cm->Width * cm->Height <= 640 * 480) {
    cm->filter_level = static_cast<int>(0.485069 * qp - 0.534462);
  } else {
    cm->filter_level = static_cast<int>(0.314875 * qp + 7.959003);
  }

  int min_filter_level = 0;
  // This logic is from get_min_filter_level() in picklpf.c
  if (q_ > 6 && q_ <= 16) {
    min_filter_level = 1;
  } else {
    min_filter_level = (q_ / 8);
  }

  const int max_filter_level = 63;
  if (cm->filter_level < min_filter_level) cm->filter_level = min_filter_level;
  if (cm->filter_level > max_filter_level) cm->filter_level = max_filter_level;

  return cm->filter_level;
}

void VP8RateControlRTC::PostEncodeUpdate(uint64_t encoded_frame_size) {
  VP8_COMMON *const cm = &cpi_->common;
  vpx_clear_system_state();
  cpi_->total_byte_count += encoded_frame_size;
  cpi_->projected_frame_size = static_cast<int>(encoded_frame_size << 3);
  if (cpi_->oxcf.number_of_layers > 1) {
    for (unsigned int i = cpi_->current_layer + 1;
         i < cpi_->oxcf.number_of_layers; ++i) {
      cpi_->layer_context[i].total_byte_count += encoded_frame_size;
    }
  }

  vp8_update_rate_correction_factors(cpi_, 2);

  cpi_->last_q[cm->frame_type] = cm->base_qindex;

  if (cm->frame_type == KEY_FRAME) {
    vp8_adjust_key_frame_context(cpi_);
  }

  /* Keep a record of ambient average Q. */
  if (cm->frame_type != KEY_FRAME) {
    cpi_->avg_frame_qindex =
        (2 + 3 * cpi_->avg_frame_qindex + cm->base_qindex) >> 2;
  }
  /* Keep a record from which we can calculate the average Q excluding
   * key frames.
   */
  if (cm->frame_type != KEY_FRAME) {
    cpi_->ni_frames++;
    /* Damp value for first few frames */
    if (cpi_->ni_frames > 150) {
      cpi_->ni_tot_qi += q_;
      cpi_->ni_av_qi = (cpi_->ni_tot_qi / cpi_->ni_frames);
    } else {
      cpi_->ni_tot_qi += q_;
      cpi_->ni_av_qi =
          ((cpi_->ni_tot_qi / cpi_->ni_frames) + cpi_->worst_quality + 1) / 2;
    }

    /* If the average Q is higher than what was used in the last
     * frame (after going through the recode loop to keep the frame
     * size within range) then use the last frame value - 1. The -1
     * is designed to stop Q and hence the data rate, from
     * progressively falling away during difficult sections, but at
     * the same time reduce the number of itterations around the
     * recode loop.
     */
    if (q_ > cpi_->ni_av_qi) cpi_->ni_av_qi = q_ - 1;
  }

  cpi_->bits_off_target +=
      cpi_->av_per_frame_bandwidth - cpi_->projected_frame_size;
  if (cpi_->bits_off_target > cpi_->oxcf.maximum_buffer_size) {
    cpi_->bits_off_target = cpi_->oxcf.maximum_buffer_size;
  }

  cpi_->total_actual_bits += cpi_->projected_frame_size;
  cpi_->buffer_level = cpi_->bits_off_target;

  /* Propagate values to higher temporal layers */
  if (cpi_->oxcf.number_of_layers > 1) {
    for (unsigned int i = cpi_->current_layer + 1;
         i < cpi_->oxcf.number_of_layers; ++i) {
      LAYER_CONTEXT *lc = &cpi_->layer_context[i];
      int bits_off_for_this_layer = (int)round(
          lc->target_bandwidth / lc->framerate - cpi_->projected_frame_size);

      lc->bits_off_target += bits_off_for_this_layer;

      /* Clip buffer level to maximum buffer size for the layer */
      if (lc->bits_off_target > lc->maximum_buffer_size) {
        lc->bits_off_target = lc->maximum_buffer_size;
      }

      lc->total_actual_bits += cpi_->projected_frame_size;
      lc->total_target_vs_actual += bits_off_for_this_layer;
      lc->buffer_level = lc->bits_off_target;
    }
  }

  cpi_->common.current_video_frame++;
  cpi_->frames_since_key++;

  if (cpi_->oxcf.number_of_layers > 1) vp8_save_layer_context(cpi_);
  vpx_clear_system_state();
}
}  // namespace libvpx
