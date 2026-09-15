/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"
#include "vpx_ports/system_state.h"

#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_quant_common.h"
#include "vp9/common/vp9_seg_common.h"

#include "vp9/encoder/vp9_aq_cyclicrefresh.h"
#include "vp9/encoder/vp9_encodemv.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_ext_ratectrl.h"
#include "vp9/encoder/vp9_firstpass.h"
#include "vp9/encoder/vp9_ratectrl.h"
#include "vp9/encoder/vp9_svc_layercontext.h"

#include "vpx/vpx_codec.h"
#include "vpx/vpx_ext_ratectrl.h"
#include "vpx/internal/vpx_codec_internal.h"

// Max rate per frame for 1080P and below encodes if no level requirement given.
// For larger formats limit to MAX_MB_RATE bits per MB
// 4Mbits is derived from the level requirement for level 4 (1080P 30) which
// requires that HW can sustain a rate of 16Mbits over a 4 frame group.
// If a lower level requirement is specified then this may over ride this value.
#define MAX_MB_RATE 250
#define MAXRATE_1080P 4000000

#define LIMIT_QRANGE_FOR_ALTREF_AND_KEY 1

#define MIN_BPB_FACTOR 0.005
#define MAX_BPB_FACTOR 50

#if CONFIG_VP9_HIGHBITDEPTH
#define ASSIGN_MINQ_TABLE(bit_depth, name)       \
  do {                                           \
    switch (bit_depth) {                         \
      case VPX_BITS_8: name = name##_8; break;   \
      case VPX_BITS_10: name = name##_10; break; \
      default:                                   \
        assert(bit_depth == VPX_BITS_12);        \
        name = name##_12;                        \
        break;                                   \
    }                                            \
  } while (0)
#else
#define ASSIGN_MINQ_TABLE(bit_depth, name) \
  do {                                     \
    (void)bit_depth;                       \
    name = name##_8;                       \
  } while (0)
#endif

// Tables relating active max Q to active min Q
static int kf_low_motion_minq_8[QINDEX_RANGE];
static int kf_high_motion_minq_8[QINDEX_RANGE];
static int arfgf_low_motion_minq_8[QINDEX_RANGE];
static int arfgf_high_motion_minq_8[QINDEX_RANGE];
static int inter_minq_8[QINDEX_RANGE];
static int rtc_minq_8[QINDEX_RANGE];

#if CONFIG_VP9_HIGHBITDEPTH
static int kf_low_motion_minq_10[QINDEX_RANGE];
static int kf_high_motion_minq_10[QINDEX_RANGE];
static int arfgf_low_motion_minq_10[QINDEX_RANGE];
static int arfgf_high_motion_minq_10[QINDEX_RANGE];
static int inter_minq_10[QINDEX_RANGE];
static int rtc_minq_10[QINDEX_RANGE];
static int kf_low_motion_minq_12[QINDEX_RANGE];
static int kf_high_motion_minq_12[QINDEX_RANGE];
static int arfgf_low_motion_minq_12[QINDEX_RANGE];
static int arfgf_high_motion_minq_12[QINDEX_RANGE];
static int inter_minq_12[QINDEX_RANGE];
static int rtc_minq_12[QINDEX_RANGE];
#endif

#ifdef AGGRESSIVE_VBR
static int gf_high = 2400;
static int gf_low = 400;
static int kf_high = 4000;
static int kf_low = 400;
#else
static int gf_high = 2000;
static int gf_low = 400;
static int kf_high = 4800;
static int kf_low = 300;
#endif

// Functions to compute the active minq lookup table entries based on a
// formulaic approach to facilitate easier adjustment of the Q tables.
// The formulae were derived from computing a 3rd order polynomial best
// fit to the original data (after plotting real maxq vs minq (not q index))
static int get_minq_index(double maxq, double x3, double x2, double x1,
                          vpx_bit_depth_t bit_depth) {
  int i;
  const double minqtarget = VPXMIN(((x3 * maxq + x2) * maxq + x1) * maxq, maxq);

  // Special case handling to deal with the step from q2.0
  // down to lossless mode represented by q 1.0.
  if (minqtarget <= 2.0) return 0;

  for (i = 0; i < QINDEX_RANGE; i++) {
    if (minqtarget <= vp9_convert_qindex_to_q(i, bit_depth)) return i;
  }

  return QINDEX_RANGE - 1;
}

static void init_minq_luts(int *kf_low_m, int *kf_high_m, int *arfgf_low,
                           int *arfgf_high, int *inter, int *rtc,
                           vpx_bit_depth_t bit_depth) {
  int i;
  for (i = 0; i < QINDEX_RANGE; i++) {
    const double maxq = vp9_convert_qindex_to_q(i, bit_depth);
    kf_low_m[i] = get_minq_index(maxq, 0.000001, -0.0004, 0.150, bit_depth);
    kf_high_m[i] = get_minq_index(maxq, 0.0000021, -0.00125, 0.45, bit_depth);
#ifdef AGGRESSIVE_VBR
    arfgf_low[i] = get_minq_index(maxq, 0.0000015, -0.0009, 0.275, bit_depth);
    inter[i] = get_minq_index(maxq, 0.00000271, -0.00113, 0.80, bit_depth);
#else
    arfgf_low[i] = get_minq_index(maxq, 0.0000015, -0.0009, 0.30, bit_depth);
    inter[i] = get_minq_index(maxq, 0.00000271, -0.00113, 0.70, bit_depth);
#endif
    arfgf_high[i] = get_minq_index(maxq, 0.0000021, -0.00125, 0.55, bit_depth);
    rtc[i] = get_minq_index(maxq, 0.00000271, -0.00113, 0.70, bit_depth);
  }
}

void vp9_rc_init_minq_luts(void) {
  init_minq_luts(kf_low_motion_minq_8, kf_high_motion_minq_8,
                 arfgf_low_motion_minq_8, arfgf_high_motion_minq_8,
                 inter_minq_8, rtc_minq_8, VPX_BITS_8);
#if CONFIG_VP9_HIGHBITDEPTH
  init_minq_luts(kf_low_motion_minq_10, kf_high_motion_minq_10,
                 arfgf_low_motion_minq_10, arfgf_high_motion_minq_10,
                 inter_minq_10, rtc_minq_10, VPX_BITS_10);
  init_minq_luts(kf_low_motion_minq_12, kf_high_motion_minq_12,
                 arfgf_low_motion_minq_12, arfgf_high_motion_minq_12,
                 inter_minq_12, rtc_minq_12, VPX_BITS_12);
#endif
}

// These functions use formulaic calculations to make playing with the
// quantizer tables easier. If necessary they can be replaced by lookup
// tables if and when things settle down in the experimental bitstream
double vp9_convert_qindex_to_q(int qindex, vpx_bit_depth_t bit_depth) {
// Convert the index to a real Q value (scaled down to match old Q values)
#if CONFIG_VP9_HIGHBITDEPTH
  switch (bit_depth) {
    case VPX_BITS_8: return vp9_ac_quant(qindex, 0, bit_depth) / 4.0;
    case VPX_BITS_10: return vp9_ac_quant(qindex, 0, bit_depth) / 16.0;
    default:
      assert(bit_depth == VPX_BITS_12);
      return vp9_ac_quant(qindex, 0, bit_depth) / 64.0;
  }
#else
  return vp9_ac_quant(qindex, 0, bit_depth) / 4.0;
#endif
}

int vp9_convert_q_to_qindex(double q_val, vpx_bit_depth_t bit_depth) {
  int i;

  for (i = 0; i < QINDEX_RANGE; ++i)
    if (vp9_convert_qindex_to_q(i, bit_depth) >= q_val) break;

  if (i == QINDEX_RANGE) i--;

  return i;
}

int vp9_rc_bits_per_mb(FRAME_TYPE frame_type, int qindex,
                       double correction_factor, vpx_bit_depth_t bit_depth) {
  const double q = vp9_convert_qindex_to_q(qindex, bit_depth);
  int enumerator = frame_type == KEY_FRAME ? 2700000 : 1800000;

  assert(correction_factor <= MAX_BPB_FACTOR &&
         correction_factor >= MIN_BPB_FACTOR);

  // q based adjustment to baseline enumerator
  enumerator += (int)(enumerator * q) >> 12;
  return (int)(enumerator * correction_factor / q);
}

int vp9_estimate_bits_at_q(FRAME_TYPE frame_type, int q, int mbs,
                           double correction_factor,
                           vpx_bit_depth_t bit_depth) {
  const int bpm =
      (int)(vp9_rc_bits_per_mb(frame_type, q, correction_factor, bit_depth));
  return VPXMAX(FRAME_OVERHEAD_BITS,
                (int)(((uint64_t)bpm * mbs) >> BPER_MB_NORMBITS));
}

int vp9_rc_clamp_pframe_target_size(const VP9_COMP *const cpi, int target) {
  const RATE_CONTROL *rc = &cpi->rc;
  const VP9EncoderConfig *oxcf = &cpi->oxcf;

  const int min_frame_target =
      VPXMAX(rc->min_frame_bandwidth, rc->avg_frame_bandwidth >> 5);
  if (target < min_frame_target) target = min_frame_target;
  if (cpi->refresh_golden_frame && rc->is_src_frame_alt_ref) {
    // If there is an active ARF at this location use the minimum
    // bits on this frame even if it is a constructed arf.
    // The active maximum quantizer insures that an appropriate
    // number of bits will be spent if needed for constructed ARFs.
    target = min_frame_target;
  }

  // Clip the frame target to the maximum allowed value.
  if (target > rc->max_frame_bandwidth) target = rc->max_frame_bandwidth;

  if (oxcf->rc_max_inter_bitrate_pct) {
    const int64_t max_rate =
        (int64_t)rc->avg_frame_bandwidth * oxcf->rc_max_inter_bitrate_pct / 100;
    // target is of type int and VPXMIN cannot evaluate to larger than target
    target = (int)VPXMIN(target, max_rate);
  }
  return target;
}

int vp9_rc_clamp_iframe_target_size(const VP9_COMP *const cpi, int target) {
  const RATE_CONTROL *rc = &cpi->rc;
  const VP9EncoderConfig *oxcf = &cpi->oxcf;
  if (oxcf->rc_max_intra_bitrate_pct) {
    const int64_t max_rate =
        (int64_t)rc->avg_frame_bandwidth * oxcf->rc_max_intra_bitrate_pct / 100;
    target = (int)VPXMIN(target, max_rate);
  }
  if (target > rc->max_frame_bandwidth) target = rc->max_frame_bandwidth;
  return target;
}

// TODO(marpan/jianj): bits_off_target and buffer_level are used in the same
// way for CBR mode, for the buffering updates below. Look into removing one
// of these (i.e., bits_off_target).
// Update the buffer level before encoding with the per-frame-bandwidth,
void vp9_update_buffer_level_preencode(VP9_COMP *cpi) {
  RATE_CONTROL *const rc = &cpi->rc;
  rc->bits_off_target += rc->avg_frame_bandwidth;
  // Clip the buffer level to the maximum specified buffer size.
  rc->bits_off_target = VPXMIN(rc->bits_off_target, rc->maximum_buffer_size);
  rc->buffer_level = rc->bits_off_target;
}

// Update the buffer level before encoding with the per-frame-bandwidth
// for SVC. The current and all upper temporal layers are updated, needed
// for the layered rate control which involves cumulative buffer levels for
// the temporal layers. Allow for using the timestamp(pts) delta for the
// framerate when the set_ref_frame_config is used.
void vp9_update_buffer_level_svc_preencode(VP9_COMP *cpi) {
  SVC *const svc = &cpi->svc;
  int i;
  // Set this to 1 to use timestamp delta for "framerate" under
  // ref_frame_config usage.
  int use_timestamp = 1;
  const int64_t ts_delta =
      svc->time_stamp_superframe - svc->time_stamp_prev[svc->spatial_layer_id];
  for (i = svc->temporal_layer_id; i < svc->number_temporal_layers; ++i) {
    const int layer =
        LAYER_IDS_TO_IDX(svc->spatial_layer_id, i, svc->number_temporal_layers);
    LAYER_CONTEXT *const lc = &svc->layer_context[layer];
    RATE_CONTROL *const lrc = &lc->rc;
    if (use_timestamp && cpi->svc.use_set_ref_frame_config &&
        svc->number_temporal_layers == 1 && ts_delta > 0 &&
        svc->current_superframe > 0) {
      // TODO(marpan): This may need to be modified for temporal layers.
      const double framerate_pts = 10000000.0 / ts_delta;
      lrc->bits_off_target += saturate_cast_double_to_int(
          round(lc->target_bandwidth / framerate_pts));
    } else {
      lrc->bits_off_target += saturate_cast_double_to_int(
          round(lc->target_bandwidth / lc->framerate));
    }
    // Clip buffer level to maximum buffer size for the layer.
    lrc->bits_off_target =
        VPXMIN(lrc->bits_off_target, lrc->maximum_buffer_size);
    lrc->buffer_level = lrc->bits_off_target;
    if (i == svc->temporal_layer_id) {
      cpi->rc.bits_off_target = lrc->bits_off_target;
      cpi->rc.buffer_level = lrc->buffer_level;
    }
  }
}

// Update the buffer level for higher temporal layers, given the encoded current
// temporal layer.
static void update_layer_buffer_level_postencode(SVC *svc,
                                                 int encoded_frame_size) {
  int i = 0;
  const int current_temporal_layer = svc->temporal_layer_id;
  for (i = current_temporal_layer + 1; i < svc->number_temporal_layers; ++i) {
    const int layer =
        LAYER_IDS_TO_IDX(svc->spatial_layer_id, i, svc->number_temporal_layers);
    LAYER_CONTEXT *lc = &svc->layer_context[layer];
    RATE_CONTROL *lrc = &lc->rc;
    lrc->bits_off_target -= encoded_frame_size;
    // Clip buffer level to maximum buffer size for the layer.
    lrc->bits_off_target =
        VPXMIN(lrc->bits_off_target, lrc->maximum_buffer_size);
    lrc->buffer_level = lrc->bits_off_target;
  }
}

// Update the buffer level after encoding with encoded frame size.
static void update_buffer_level_postencode(VP9_COMP *cpi,
                                           int encoded_frame_size) {
  RATE_CONTROL *const rc = &cpi->rc;
  rc->bits_off_target -= encoded_frame_size;
  // Clip the buffer level to the maximum specified buffer size.
  rc->bits_off_target = VPXMIN(rc->bits_off_target, rc->maximum_buffer_size);
  // For screen-content mode, and if frame-dropper is off, don't let buffer
  // level go below threshold, given here as -rc->maximum_ buffer_size.
  if (cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
      cpi->oxcf.drop_frames_water_mark == 0)
    rc->bits_off_target = VPXMAX(rc->bits_off_target, -rc->maximum_buffer_size);

  rc->buffer_level = rc->bits_off_target;

  if (is_one_pass_svc(cpi)) {
    update_layer_buffer_level_postencode(&cpi->svc, encoded_frame_size);
  }
}

int vp9_rc_get_default_min_gf_interval(int width, int height,
                                       double framerate) {
  // Assume we do not need any constraint lower than 4K 20 fps
  static const double factor_safe = 3840 * 2160 * 20.0;
  const double factor = width * height * framerate;
  const int default_interval =
      clamp((int)round(framerate * 0.125), MIN_GF_INTERVAL, MAX_GF_INTERVAL);

  if (factor <= factor_safe)
    return default_interval;
  else
    return VPXMAX(default_interval,
                  (int)round(MIN_GF_INTERVAL * factor / factor_safe));
  // Note this logic makes:
  // 4K24: 5
  // 4K30: 6
  // 4K60: 12
}

int vp9_rc_get_default_max_gf_interval(double framerate, int min_gf_interval) {
  int interval = VPXMIN(MAX_GF_INTERVAL, (int)round(framerate * 0.75));
  interval += (interval & 0x01);  // Round to even value
  return VPXMAX(interval, min_gf_interval);
}

void vp9_rc_init(const VP9EncoderConfig *oxcf, int pass, RATE_CONTROL *rc) {
  int i;

  if (pass == 0 && oxcf->rc_mode == VPX_CBR) {
    rc->avg_frame_qindex[KEY_FRAME] = oxcf->worst_allowed_q;
    rc->avg_frame_qindex[INTER_FRAME] = oxcf->worst_allowed_q;
  } else {
    rc->avg_frame_qindex[KEY_FRAME] =
        (oxcf->worst_allowed_q + oxcf->best_allowed_q) / 2;
    rc->avg_frame_qindex[INTER_FRAME] =
        (oxcf->worst_allowed_q + oxcf->best_allowed_q) / 2;
  }

  rc->last_q[KEY_FRAME] = oxcf->best_allowed_q;
  rc->last_q[INTER_FRAME] = oxcf->worst_allowed_q;

  rc->buffer_level = rc->starting_buffer_level;
  rc->bits_off_target = rc->starting_buffer_level;

  rc->rolling_target_bits = rc->avg_frame_bandwidth;
  rc->rolling_actual_bits = rc->avg_frame_bandwidth;
  rc->long_rolling_target_bits = rc->avg_frame_bandwidth;
  rc->long_rolling_actual_bits = rc->avg_frame_bandwidth;

  rc->total_actual_bits = 0;
  rc->total_target_bits = 0;
  rc->total_target_vs_actual = 0;
  rc->avg_frame_low_motion = 0;
  rc->count_last_scene_change = 0;
  rc->af_ratio_onepass_vbr = 10;
  rc->prev_avg_source_sad_lag = 0;
  rc->high_source_sad = 0;
  rc->reset_high_source_sad = 0;
  rc->high_source_sad_lagindex = -1;
  rc->high_num_blocks_with_motion = 0;
  rc->hybrid_intra_scene_change = 0;
  rc->re_encode_maxq_scene_change = 0;
  rc->alt_ref_gf_group = 0;
  rc->last_frame_is_src_altref = 0;
  rc->fac_active_worst_inter = 150;
  rc->fac_active_worst_gf = 100;
  rc->force_qpmin = 0;
  for (i = 0; i < MAX_LAG_BUFFERS; ++i) rc->avg_source_sad[i] = 0;
  rc->frames_to_key = 0;
  rc->frames_since_key = 8;  // Sensible default for first frame.
  rc->this_key_frame_forced = 0;
  rc->next_key_frame_forced = 0;
  rc->source_alt_ref_pending = 0;
  rc->source_alt_ref_active = 0;

  rc->frames_till_gf_update_due = 0;
  rc->constrain_gf_key_freq_onepass_vbr = 1;
  rc->ni_av_qi = oxcf->worst_allowed_q;
  rc->ni_tot_qi = 0;
  rc->ni_frames = 0;

  rc->tot_q = 0.0;
  rc->avg_q = vp9_convert_qindex_to_q(oxcf->worst_allowed_q, oxcf->bit_depth);

  for (i = 0; i < RATE_FACTOR_LEVELS; ++i) {
    rc->rate_correction_factors[i] = 1.0;
    rc->damped_adjustment[i] = 0;
  }

  rc->min_gf_interval = oxcf->min_gf_interval;
  rc->max_gf_interval = oxcf->max_gf_interval;
  if (rc->min_gf_interval == 0)
    rc->min_gf_interval = vp9_rc_get_default_min_gf_interval(
        oxcf->width, oxcf->height, oxcf->init_framerate);
  if (rc->max_gf_interval == 0)
    rc->max_gf_interval = vp9_rc_get_default_max_gf_interval(
        oxcf->init_framerate, rc->min_gf_interval);
  rc->baseline_gf_interval = (rc->min_gf_interval + rc->max_gf_interval) / 2;
  if ((oxcf->pass == 0) && (oxcf->rc_mode == VPX_Q)) {
    rc->static_scene_max_gf_interval = FIXED_GF_INTERVAL;
  } else {
    rc->static_scene_max_gf_interval = MAX_STATIC_GF_GROUP_LENGTH;
  }

  rc->force_max_q = 0;
  rc->last_post_encode_dropped_scene_change = 0;
  rc->use_post_encode_drop = 0;
  rc->ext_use_post_encode_drop = 0;
  rc->disable_overshoot_maxq_cbr = 0;
  rc->arf_active_best_quality_adjustment_factor = 1.0;
  rc->arf_increase_active_best_quality = 0;
  rc->preserve_arf_as_gld = 0;
  rc->preserve_next_arf_as_gld = 0;
  rc->show_arf_as_gld = 0;
}

static int check_buffer_above_thresh(VP9_COMP *cpi, int drop_mark) {
  SVC *svc = &cpi->svc;
  if (!cpi->use_svc || cpi->svc.framedrop_mode != FULL_SUPERFRAME_DROP) {
    RATE_CONTROL *const rc = &cpi->rc;
    return (rc->buffer_level > drop_mark);
  } else {
    int i;
    // For SVC in the FULL_SUPERFRAME_DROP): the condition on
    // buffer (if its above threshold, so no drop) is checked on current and
    // upper spatial layers. If any spatial layer is not above threshold then
    // we return 0.
    for (i = svc->spatial_layer_id; i < svc->number_spatial_layers; ++i) {
      const int layer = LAYER_IDS_TO_IDX(i, svc->temporal_layer_id,
                                         svc->number_temporal_layers);
      LAYER_CONTEXT *lc = &svc->layer_context[layer];
      RATE_CONTROL *lrc = &lc->rc;
      // Exclude check for layer whose bitrate is 0.
      if (lc->target_bandwidth > 0) {
        const int drop_mark_layer = (int)(cpi->svc.framedrop_thresh[i] *
                                          lrc->optimal_buffer_level / 100);
        if (!(lrc->buffer_level > drop_mark_layer)) return 0;
      }
    }
    return 1;
  }
}

static int check_buffer_below_thresh(VP9_COMP *cpi, int drop_mark) {
  SVC *svc = &cpi->svc;
  if (!cpi->use_svc || cpi->svc.framedrop_mode == LAYER_DROP) {
    RATE_CONTROL *const rc = &cpi->rc;
    return (rc->buffer_level <= drop_mark);
  } else {
    int i;
    // For SVC in the constrained framedrop mode (svc->framedrop_mode =
    // CONSTRAINED_LAYER_DROP or FULL_SUPERFRAME_DROP): the condition on
    // buffer (if its below threshold, so drop frame) is checked on current
    // and upper spatial layers. For FULL_SUPERFRAME_DROP mode if any
    // spatial layer is <= threshold, then we return 1 (drop).
    for (i = svc->spatial_layer_id; i < svc->number_spatial_layers; ++i) {
      const int layer = LAYER_IDS_TO_IDX(i, svc->temporal_layer_id,
                                         svc->number_temporal_layers);
      LAYER_CONTEXT *lc = &svc->layer_context[layer];
      RATE_CONTROL *lrc = &lc->rc;
      // Exclude check for layer whose bitrate is 0.
      if (lc->target_bandwidth > 0) {
        const int drop_mark_layer = (int)(cpi->svc.framedrop_thresh[i] *
                                          lrc->optimal_buffer_level / 100);
        if (cpi->svc.framedrop_mode == FULL_SUPERFRAME_DROP) {
          if (lrc->buffer_level <= drop_mark_layer) return 1;
        } else {
          if (!(lrc->buffer_level <= drop_mark_layer)) return 0;
        }
      }
    }
    if (cpi->svc.framedrop_mode == FULL_SUPERFRAME_DROP)
      return 0;
    else
      return 1;
  }
}

int vp9_test_drop(VP9_COMP *cpi) {
  const VP9EncoderConfig *oxcf = &cpi->oxcf;
  RATE_CONTROL *const rc = &cpi->rc;
  SVC *svc = &cpi->svc;
  int drop_frames_water_mark = oxcf->drop_frames_water_mark;
  if (cpi->use_svc) {
    // If we have dropped max_consec_drop frames, then we don't
    // drop this spatial layer, and reset counter to 0.
    if (svc->drop_count[svc->spatial_layer_id] == svc->max_consec_drop) {
      svc->drop_count[svc->spatial_layer_id] = 0;
      return 0;
    } else {
      drop_frames_water_mark = svc->framedrop_thresh[svc->spatial_layer_id];
    }
  }
  if (!drop_frames_water_mark ||
      (svc->spatial_layer_id > 0 &&
       svc->framedrop_mode == FULL_SUPERFRAME_DROP)) {
    return 0;
  } else {
    if ((rc->buffer_level < 0 && svc->framedrop_mode != FULL_SUPERFRAME_DROP) ||
        (check_buffer_below_thresh(cpi, -1) &&
         svc->framedrop_mode == FULL_SUPERFRAME_DROP)) {
      // Always drop if buffer is below 0.
      return 1;
    } else {
      // If buffer is below drop_mark, for now just drop every other frame
      // (starting with the next frame) until it increases back over drop_mark.
      int drop_mark =
          (int)(drop_frames_water_mark * rc->optimal_buffer_level / 100);
      if (check_buffer_above_thresh(cpi, drop_mark) &&
          (rc->decimation_factor > 0)) {
        --rc->decimation_factor;
      } else if (check_buffer_below_thresh(cpi, drop_mark) &&
                 rc->decimation_factor == 0) {
        rc->decimation_factor = 1;
      }
      if (rc->decimation_factor > 0) {
        if (rc->decimation_count > 0) {
          --rc->decimation_count;
          return 1;
        } else {
          rc->decimation_count = rc->decimation_factor;
          return 0;
        }
      } else {
        rc->decimation_count = 0;
        return 0;
      }
    }
  }
}

int post_encode_drop_cbr(VP9_COMP *cpi, size_t *size) {
  size_t frame_size = *size << 3;
  int64_t new_buffer_level =
      cpi->rc.buffer_level + cpi->rc.avg_frame_bandwidth - (int64_t)frame_size;

  // For now we drop if new buffer level (given the encoded frame size) goes
  // below 0.
  if (new_buffer_level < 0) {
    *size = 0;
    vp9_rc_postencode_update_drop_frame(cpi);
    // Update flag to use for next frame.
    if (cpi->rc.high_source_sad ||
        (cpi->use_svc && cpi->svc.high_source_sad_superframe))
      cpi->rc.last_post_encode_dropped_scene_change = 1;
    // Force max_q on next fame.
    cpi->rc.force_max_q = 1;
    cpi->rc.avg_frame_qindex[INTER_FRAME] = cpi->rc.worst_quality;
    cpi->last_frame_dropped = 1;
    cpi->ext_refresh_frame_flags_pending = 0;
    if (cpi->use_svc) {
      SVC *svc = &cpi->svc;
      int sl = 0;
      int tl = 0;
      svc->last_layer_dropped[svc->spatial_layer_id] = 1;
      svc->drop_spatial_layer[svc->spatial_layer_id] = 1;
      svc->drop_count[svc->spatial_layer_id]++;
      svc->skip_enhancement_layer = 1;
      // Postencode drop is only checked on base spatial layer,
      // for now if max-q is set on base we force it on all layers.
      for (sl = 0; sl < svc->number_spatial_layers; ++sl) {
        for (tl = 0; tl < svc->number_temporal_layers; ++tl) {
          const int layer =
              LAYER_IDS_TO_IDX(sl, tl, svc->number_temporal_layers);
          LAYER_CONTEXT *lc = &svc->layer_context[layer];
          RATE_CONTROL *lrc = &lc->rc;
          lrc->force_max_q = 1;
          lrc->avg_frame_qindex[INTER_FRAME] = cpi->rc.worst_quality;
        }
      }
    }
    return 1;
  }

  cpi->rc.force_max_q = 0;
  cpi->rc.last_post_encode_dropped_scene_change = 0;
  return 0;
}

int vp9_rc_drop_frame(VP9_COMP *cpi) {
  SVC *svc = &cpi->svc;
  int svc_prev_layer_dropped = 0;
  // In the constrained or full_superframe framedrop mode for svc
  // (framedrop_mode != (LAYER_DROP && CONSTRAINED_FROM_ABOVE)),
  // if the previous spatial layer was dropped, drop the current spatial layer.
  if (cpi->use_svc && svc->spatial_layer_id > 0 &&
      svc->drop_spatial_layer[svc->spatial_layer_id - 1])
    svc_prev_layer_dropped = 1;
  if ((svc_prev_layer_dropped && svc->framedrop_mode != LAYER_DROP &&
       svc->framedrop_mode != CONSTRAINED_FROM_ABOVE_DROP) ||
      svc->force_drop_constrained_from_above[svc->spatial_layer_id] ||
      vp9_test_drop(cpi)) {
    vp9_rc_postencode_update_drop_frame(cpi);
    cpi->ext_refresh_frame_flags_pending = 0;
    cpi->last_frame_dropped = 1;
    if (cpi->use_svc) {
      svc->last_layer_dropped[svc->spatial_layer_id] = 1;
      svc->drop_spatial_layer[svc->spatial_layer_id] = 1;
      svc->drop_count[svc->spatial_layer_id]++;
      svc->skip_enhancement_layer = 1;
      if (svc->framedrop_mode == LAYER_DROP ||
          (svc->framedrop_mode == CONSTRAINED_FROM_ABOVE_DROP &&
           svc->force_drop_constrained_from_above[svc->number_spatial_layers -
                                                  1] == 0) ||
          svc->drop_spatial_layer[0] == 0) {
        // For the case of constrained drop mode where full superframe is
        // dropped, we don't increment the svc frame counters.
        // In particular temporal layer counter (which is incremented in
        // vp9_inc_frame_in_layer()) won't be incremented, so on a dropped
        // frame we try the same temporal_layer_id on next incoming frame.
        // This is to avoid an issue with temporal alignment with full
        // superframe dropping.
        vp9_inc_frame_in_layer(cpi);
      }
      if (svc->spatial_layer_id == svc->number_spatial_layers - 1) {
        int i;
        int all_layers_drop = 1;
        for (i = 0; i < svc->spatial_layer_id; i++) {
          if (svc->drop_spatial_layer[i] == 0) {
            all_layers_drop = 0;
            break;
          }
        }
        if (all_layers_drop == 1) svc->skip_enhancement_layer = 0;
      }
    }
    return 1;
  }
  return 0;
}

static int adjust_q_cbr(const VP9_COMP *cpi, int q) {
  // This makes sure q is between oscillating Qs to prevent resonance.
  if (!cpi->rc.reset_high_source_sad &&
      (!cpi->oxcf.gf_cbr_boost_pct ||
       !(cpi->refresh_alt_ref_frame || cpi->refresh_golden_frame)) &&
      (cpi->rc.rc_1_frame * cpi->rc.rc_2_frame == -1) &&
      cpi->rc.q_1_frame != cpi->rc.q_2_frame) {
    int qclamp = clamp(q, VPXMIN(cpi->rc.q_1_frame, cpi->rc.q_2_frame),
                       VPXMAX(cpi->rc.q_1_frame, cpi->rc.q_2_frame));
    // If the previous frame had overshoot and the current q needs to increase
    // above the clamped value, reduce the clamp for faster reaction to
    // overshoot.
    if (cpi->rc.rc_1_frame == -1 && q > qclamp)
      q = (q + qclamp) >> 1;
    else
      q = qclamp;
  }
  if (cpi->oxcf.content == VP9E_CONTENT_SCREEN &&
      cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
    vp9_cyclic_refresh_limit_q(cpi, &q);
  return VPXMAX(VPXMIN(q, cpi->rc.worst_quality), cpi->rc.best_quality);
}

static double get_rate_correction_factor(const VP9_COMP *cpi) {
  const RATE_CONTROL *const rc = &cpi->rc;
  const VP9_COMMON *const cm = &cpi->common;
  double rcf;

  if (frame_is_intra_only(cm)) {
    rcf = rc->rate_correction_factors[KF_STD];
  } else if (cpi->oxcf.pass == 2) {
    RATE_FACTOR_LEVEL rf_lvl =
        cpi->twopass.gf_group.rf_level[cpi->twopass.gf_group.index];
    rcf = rc->rate_correction_factors[rf_lvl];
  } else {
    if ((cpi->refresh_alt_ref_frame || cpi->refresh_golden_frame) &&
        !rc->is_src_frame_alt_ref && !cpi->use_svc &&
        (cpi->oxcf.rc_mode != VPX_CBR || cpi->oxcf.gf_cbr_boost_pct > 100))
      rcf = rc->rate_correction_factors[GF_ARF_STD];
    else
      rcf = rc->rate_correction_factors[INTER_NORMAL];
  }
  rcf *= rcf_mult[rc->frame_size_selector];
  return fclamp(rcf, MIN_BPB_FACTOR, MAX_BPB_FACTOR);
}

static void set_rate_correction_factor(VP9_COMP *cpi, double factor) {
  RATE_CONTROL *const rc = &cpi->rc;
  const VP9_COMMON *const cm = &cpi->common;

  // Normalize RCF to account for the size-dependent scaling factor.
  factor /= rcf_mult[cpi->rc.frame_size_selector];

  factor = fclamp(factor, MIN_BPB_FACTOR, MAX_BPB_FACTOR);

  if (frame_is_intra_only(cm)) {
    rc->rate_correction_factors[KF_STD] = factor;
  } else if (cpi->oxcf.pass == 2) {
    RATE_FACTOR_LEVEL rf_lvl =
        cpi->twopass.gf_group.rf_level[cpi->twopass.gf_group.index];
    rc->rate_correction_factors[rf_lvl] = factor;
  } else {
    if ((cpi->refresh_alt_ref_frame || cpi->refresh_golden_frame) &&
        !rc->is_src_frame_alt_ref && !cpi->use_svc &&
        (cpi->oxcf.rc_mode != VPX_CBR || cpi->oxcf.gf_cbr_boost_pct > 100))
      rc->rate_correction_factors[GF_ARF_STD] = factor;
    else
      rc->rate_correction_factors[INTER_NORMAL] = factor;
  }
}

void vp9_rc_update_rate_correction_factors(VP9_COMP *cpi) {
  const VP9_COMMON *const cm = &cpi->common;
  int correction_factor = 100;
  double rate_correction_factor = get_rate_correction_factor(cpi);
  double adjustment_limit;
  RATE_FACTOR_LEVEL rf_lvl =
      cpi->twopass.gf_group.rf_level[cpi->twopass.gf_group.index];

  int projected_size_based_on_q = 0;

  // Do not update the rate factors for arf overlay frames.
  if (cpi->rc.is_src_frame_alt_ref) return;

  // Clear down mmx registers to allow floating point in what follows
  vpx_clear_system_state();

  // Work out how big we would have expected the frame to be at this Q given
  // the current correction factor.
  // Stay in double to avoid int overflow when values are large
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cpi->common.seg.enabled) {
    projected_size_based_on_q =
        vp9_cyclic_refresh_estimate_bits_at_q(cpi, rate_correction_factor);
  } else {
    FRAME_TYPE frame_type = cm->intra_only ? KEY_FRAME : cm->frame_type;
    projected_size_based_on_q =
        vp9_estimate_bits_at_q(frame_type, cm->base_qindex, cm->MBs,
                               rate_correction_factor, cm->bit_depth);
  }
  // Work out a size correction factor.
  if (projected_size_based_on_q > FRAME_OVERHEAD_BITS)
    correction_factor = (int)((100 * (int64_t)cpi->rc.projected_frame_size) /
                              projected_size_based_on_q);

  // Do not use damped adjustment for the first frame of each frame type
  if (!cpi->rc.damped_adjustment[rf_lvl]) {
    adjustment_limit = 1.0;
    cpi->rc.damped_adjustment[rf_lvl] = 1;
  } else {
    // More heavily damped adjustment used if we have been oscillating either
    // side of target.
    adjustment_limit =
        0.25 + 0.5 * VPXMIN(1, fabs(log10(0.01 * correction_factor)));
  }

  cpi->rc.q_2_frame = cpi->rc.q_1_frame;
  cpi->rc.q_1_frame = cm->base_qindex;
  cpi->rc.rc_2_frame = cpi->rc.rc_1_frame;
  if (correction_factor > 110)
    cpi->rc.rc_1_frame = -1;
  else if (correction_factor < 90)
    cpi->rc.rc_1_frame = 1;
  else
    cpi->rc.rc_1_frame = 0;

  // Turn off oscilation detection in the case of massive overshoot.
  if (cpi->rc.rc_1_frame == -1 && cpi->rc.rc_2_frame == 1 &&
      correction_factor > 1000) {
    cpi->rc.rc_2_frame = 0;
  }

  if (correction_factor > 102) {
    // We are not already at the worst allowable quality
    correction_factor =
        (int)(100 + ((correction_factor - 100) * adjustment_limit));
    rate_correction_factor = (rate_correction_factor * correction_factor) / 100;
    // Keep rate_correction_factor within limits
    if (rate_correction_factor > MAX_BPB_FACTOR)
      rate_correction_factor = MAX_BPB_FACTOR;
  } else if (correction_factor < 99) {
    // We are not already at the best allowable quality
    correction_factor =
        (int)(100 - ((100 - correction_factor) * adjustment_limit));
    rate_correction_factor = (rate_correction_factor * correction_factor) / 100;

    // Keep rate_correction_factor within limits
    if (rate_correction_factor < MIN_BPB_FACTOR)
      rate_correction_factor = MIN_BPB_FACTOR;
  }

  set_rate_correction_factor(cpi, rate_correction_factor);
}

int vp9_rc_regulate_q(const VP9_COMP *cpi, int target_bits_per_frame,
                      int active_best_quality, int active_worst_quality) {
  const VP9_COMMON *const cm = &cpi->common;
  CYCLIC_REFRESH *const cr = cpi->cyclic_refresh;
  int q = active_worst_quality;
  int last_error = INT_MAX;
  int i, target_bits_per_mb, bits_per_mb_at_this_q;
  const double correction_factor = get_rate_correction_factor(cpi);

  // Calculate required scaling factor based on target frame size and size of
  // frame produced using previous Q.
  target_bits_per_mb =
      (int)(((uint64_t)target_bits_per_frame << BPER_MB_NORMBITS) / cm->MBs);

  i = active_best_quality;

  do {
    if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cr->apply_cyclic_refresh &&
        (!cpi->oxcf.gf_cbr_boost_pct || !cpi->refresh_golden_frame)) {
      bits_per_mb_at_this_q =
          (int)vp9_cyclic_refresh_rc_bits_per_mb(cpi, i, correction_factor);
    } else {
      FRAME_TYPE frame_type = cm->intra_only ? KEY_FRAME : cm->frame_type;
      bits_per_mb_at_this_q = (int)vp9_rc_bits_per_mb(
          frame_type, i, correction_factor, cm->bit_depth);
    }

    int diff_bits = (int)VPXMIN(
        VPXMAX(((int64_t)target_bits_per_mb - (int64_t)bits_per_mb_at_this_q),
               -INT_MAX),
        INT_MAX);
    if (bits_per_mb_at_this_q <= target_bits_per_mb) {
      if (diff_bits <= last_error)
        q = i;
      else
        q = i - 1;

      break;
    } else {
      last_error = -diff_bits;
    }
  } while (++i <= active_worst_quality);

  // Adjustment to q for CBR mode.
  if (cpi->oxcf.rc_mode == VPX_CBR) return adjust_q_cbr(cpi, q);

  return q;
}

static int get_active_quality(int q, int gfu_boost, int low, int high,
                              int *low_motion_minq, int *high_motion_minq) {
  if (gfu_boost > high) {
    return low_motion_minq[q];
  } else if (gfu_boost < low) {
    return high_motion_minq[q];
  } else {
    const int gap = high - low;
    const int offset = high - gfu_boost;
    const int qdiff = high_motion_minq[q] - low_motion_minq[q];
    const int adjustment = ((offset * qdiff) + (gap >> 1)) / gap;
    return low_motion_minq[q] + adjustment;
  }
}

static int get_kf_active_quality(const RATE_CONTROL *const rc, int q,
                                 vpx_bit_depth_t bit_depth) {
  int *kf_low_motion_minq;
  int *kf_high_motion_minq;
  ASSIGN_MINQ_TABLE(bit_depth, kf_low_motion_minq);
  ASSIGN_MINQ_TABLE(bit_depth, kf_high_motion_minq);
  return get_active_quality(q, rc->kf_boost, kf_low, kf_high,
                            kf_low_motion_minq, kf_high_motion_minq);
}

static int get_gf_active_quality(const VP9_COMP *const cpi, int q,
                                 vpx_bit_depth_t bit_depth) {
  const GF_GROUP *const gf_group = &cpi->twopass.gf_group;
  const RATE_CONTROL *const rc = &cpi->rc;

  int *arfgf_low_motion_minq;
  int *arfgf_high_motion_minq;
  const int gfu_boost = cpi->multi_layer_arf
                            ? gf_group->gfu_boost[gf_group->index]
                            : rc->gfu_boost;
  ASSIGN_MINQ_TABLE(bit_depth, arfgf_low_motion_minq);
  ASSIGN_MINQ_TABLE(bit_depth, arfgf_high_motion_minq);
  return get_active_quality(q, gfu_boost, gf_low, gf_high,
                            arfgf_low_motion_minq, arfgf_high_motion_minq);
}

static int calc_active_worst_quality_one_pass_vbr(const VP9_COMP *cpi) {
  const RATE_CONTROL *const rc = &cpi->rc;
  const unsigned int curr_frame = cpi->common.current_video_frame;
  int active_worst_quality;

  if (cpi->common.frame_type == KEY_FRAME) {
    active_worst_quality =
        curr_frame == 0 ? rc->worst_quality : rc->last_q[KEY_FRAME] << 1;
  } else {
    if (!rc->is_src_frame_alt_ref && !cpi->use_svc &&
        (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame)) {
      active_worst_quality =
          curr_frame == 1
              ? rc->last_q[KEY_FRAME] * 5 >> 2
              : rc->last_q[INTER_FRAME] * rc->fac_active_worst_gf / 100;
    } else {
      active_worst_quality = curr_frame == 1
                                 ? rc->last_q[KEY_FRAME] << 1
                                 : rc->avg_frame_qindex[INTER_FRAME] *
                                       rc->fac_active_worst_inter / 100;
    }
  }
  return VPXMIN(active_worst_quality, rc->worst_quality);
}

// Adjust active_worst_quality level based on buffer level.
static int calc_active_worst_quality_one_pass_cbr(const VP9_COMP *cpi) {
  // Adjust active_worst_quality: If buffer is above the optimal/target level,
  // bring active_worst_quality down depending on fullness of buffer.
  // If buffer is below the optimal level, let the active_worst_quality go from
  // ambient Q (at buffer = optimal level) to worst_quality level
  // (at buffer = critical level).
  const VP9_COMMON *const cm = &cpi->common;
  const RATE_CONTROL *rc = &cpi->rc;
  // Buffer level below which we push active_worst to worst_quality.
  int64_t critical_level = rc->optimal_buffer_level >> 3;
  int64_t buff_lvl_step = 0;
  int adjustment = 0;
  int active_worst_quality;
  int ambient_qp;
  unsigned int num_frames_weight_key = 5 * cpi->svc.number_temporal_layers;
  if (frame_is_intra_only(cm) || rc->reset_high_source_sad || rc->force_max_q)
    return rc->worst_quality;
  // For ambient_qp we use minimum of avg_frame_qindex[KEY_FRAME/INTER_FRAME]
  // for the first few frames following key frame. These are both initialized
  // to worst_quality and updated with (3/4, 1/4) average in postencode_update.
  // So for first few frames following key, the qp of that key frame is weighted
  // into the active_worst_quality setting.
  ambient_qp = (cm->current_video_frame < num_frames_weight_key)
                   ? VPXMIN(rc->avg_frame_qindex[INTER_FRAME],
                            rc->avg_frame_qindex[KEY_FRAME])
                   : rc->avg_frame_qindex[INTER_FRAME];
  active_worst_quality = VPXMIN(rc->worst_quality, (ambient_qp * 5) >> 2);
  // For SVC if the current base spatial layer was key frame, use the QP from
  // that base layer for ambient_qp.
  if (cpi->use_svc && cpi->svc.spatial_layer_id > 0) {
    int layer = LAYER_IDS_TO_IDX(0, cpi->svc.temporal_layer_id,
                                 cpi->svc.number_temporal_layers);
    const LAYER_CONTEXT *lc = &cpi->svc.layer_context[layer];
    if (lc->is_key_frame) {
      const RATE_CONTROL *lrc = &lc->rc;
      ambient_qp = VPXMIN(ambient_qp, lrc->last_q[KEY_FRAME]);
      active_worst_quality = VPXMIN(rc->worst_quality, (ambient_qp * 9) >> 3);
    }
  }
  if (rc->buffer_level > rc->optimal_buffer_level) {
    // Adjust down.
    // Maximum limit for down adjustment ~30%; make it lower for screen content.
    int max_adjustment_down = active_worst_quality / 3;
    if (cpi->oxcf.content == VP9E_CONTENT_SCREEN)
      max_adjustment_down = active_worst_quality >> 3;
    if (max_adjustment_down) {
      buff_lvl_step = ((rc->maximum_buffer_size - rc->optimal_buffer_level) /
                       max_adjustment_down);
      if (buff_lvl_step)
        adjustment = (int)((rc->buffer_level - rc->optimal_buffer_level) /
                           buff_lvl_step);
      active_worst_quality -= adjustment;
    }
  } else if (rc->buffer_level > critical_level) {
    // Adjust up from ambient Q.
    if (critical_level) {
      buff_lvl_step = (rc->optimal_buffer_level - critical_level);
      if (buff_lvl_step) {
        adjustment = (int)((rc->worst_quality - ambient_qp) *
                           (rc->optimal_buffer_level - rc->buffer_level) /
                           buff_lvl_step);
      }
      active_worst_quality = ambient_qp + adjustment;
    }
  } else {
    // Set to worst_quality if buffer is below critical level.
    active_worst_quality = rc->worst_quality;
  }
  return active_worst_quality;
}

static int rc_pick_q_and_bounds_one_pass_cbr(const VP9_COMP *cpi,
                                             int *bottom_index,
                                             int *top_index) {
  const VP9_COMMON *const cm = &cpi->common;
  const RATE_CONTROL *const rc = &cpi->rc;
  int active_best_quality;
  int active_worst_quality = calc_active_worst_quality_one_pass_cbr(cpi);
  int q;
  int *rtc_minq;
  ASSIGN_MINQ_TABLE(cm->bit_depth, rtc_minq);

  if (frame_is_intra_only(cm)) {
    active_best_quality = rc->best_quality;
    // Handle the special case for key frames forced when we have reached
    // the maximum key frame interval. Here force the Q to a range
    // based on the ambient Q to reduce the risk of popping.
    if (rc->this_key_frame_forced) {
      int qindex = rc->last_boosted_qindex;
      double last_boosted_q = vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      int delta_qindex = vp9_compute_qdelta(
          rc, last_boosted_q, (last_boosted_q * 0.75), cm->bit_depth);
      active_best_quality = VPXMAX(qindex + delta_qindex, rc->best_quality);
    } else if (cm->current_video_frame > 0) {
      // not first frame of one pass and kf_boost is set
      double q_adj_factor = 1.0;
      double q_val;

      active_best_quality = get_kf_active_quality(
          rc, rc->avg_frame_qindex[KEY_FRAME], cm->bit_depth);

      // Allow somewhat lower kf minq with small image formats.
      if ((cm->width * cm->height) <= (352 * 288)) {
        q_adj_factor -= 0.25;
      }

      // Convert the adjustment factor to a qindex delta
      // on active_best_quality.
      q_val = vp9_convert_qindex_to_q(active_best_quality, cm->bit_depth);
      active_best_quality +=
          vp9_compute_qdelta(rc, q_val, q_val * q_adj_factor, cm->bit_depth);
    }
  } else if (!rc->is_src_frame_alt_ref && !cpi->use_svc &&
             cpi->oxcf.gf_cbr_boost_pct &&
             (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame)) {
    // Use the lower of active_worst_quality and recent
    // average Q as basis for GF/ARF best Q limit unless last frame was
    // a key frame.
    if (rc->frames_since_key > 1 &&
        rc->avg_frame_qindex[INTER_FRAME] < active_worst_quality) {
      q = rc->avg_frame_qindex[INTER_FRAME];
    } else {
      q = active_worst_quality;
    }
    active_best_quality = get_gf_active_quality(cpi, q, cm->bit_depth);
  } else {
    // Use the lower of active_worst_quality and recent/average Q.
    if (cm->current_video_frame > 1) {
      if (rc->avg_frame_qindex[INTER_FRAME] < active_worst_quality)
        active_best_quality = rtc_minq[rc->avg_frame_qindex[INTER_FRAME]];
      else
        active_best_quality = rtc_minq[active_worst_quality];
    } else {
      if (rc->avg_frame_qindex[KEY_FRAME] < active_worst_quality)
        active_best_quality = rtc_minq[rc->avg_frame_qindex[KEY_FRAME]];
      else
        active_best_quality = rtc_minq[active_worst_quality];
    }
  }

  // Clip the active best and worst quality values to limits
  active_best_quality =
      clamp(active_best_quality, rc->best_quality, rc->worst_quality);
  active_worst_quality =
      clamp(active_worst_quality, active_best_quality, rc->worst_quality);

  *top_index = active_worst_quality;
  *bottom_index = active_best_quality;

  // Special case code to try and match quality with forced key frames
  if (frame_is_intra_only(cm) && rc->this_key_frame_forced) {
    q = rc->last_boosted_qindex;
  } else {
    q = vp9_rc_regulate_q(cpi, rc->this_frame_target, active_best_quality,
                          active_worst_quality);
    if (q > *top_index) {
      // Special case when we are targeting the max allowed rate
      if (rc->this_frame_target >= rc->max_frame_bandwidth)
        *top_index = q;
      else
        q = *top_index;
    }
  }

  assert(*top_index <= rc->worst_quality && *top_index >= rc->best_quality);
  assert(*bottom_index <= rc->worst_quality &&
         *bottom_index >= rc->best_quality);
  assert(q <= rc->worst_quality && q >= rc->best_quality);
  return q;
}

static int get_active_cq_level_one_pass(const RATE_CONTROL *rc,
                                        const VP9EncoderConfig *const oxcf) {
  static const double cq_adjust_threshold = 0.1;
  int active_cq_level = oxcf->cq_level;
  if (oxcf->rc_mode == VPX_CQ && rc->total_target_bits > 0) {
    const double x = (double)rc->total_actual_bits / rc->total_target_bits;
    if (x < cq_adjust_threshold) {
      active_cq_level = (int)(active_cq_level * x / cq_adjust_threshold);
    }
  }
  return active_cq_level;
}

#define SMOOTH_PCT_MIN 0.1
#define SMOOTH_PCT_DIV 0.05
static int get_active_cq_level_two_pass(const TWO_PASS *twopass,
                                        const RATE_CONTROL *rc,
                                        const VP9EncoderConfig *const oxcf) {
  static const double cq_adjust_threshold = 0.1;
  int active_cq_level = oxcf->cq_level;
  if (oxcf->rc_mode == VPX_CQ) {
    if (twopass->mb_smooth_pct > SMOOTH_PCT_MIN) {
      active_cq_level -=
          (int)((twopass->mb_smooth_pct - SMOOTH_PCT_MIN) / SMOOTH_PCT_DIV);
      active_cq_level = VPXMAX(active_cq_level, 0);
    }
    if (rc->total_target_bits > 0) {
      const double x = (double)rc->total_actual_bits / rc->total_target_bits;
      if (x < cq_adjust_threshold) {
        active_cq_level = (int)(active_cq_level * x / cq_adjust_threshold);
      }
    }
  }
  return active_cq_level;
}

static int rc_pick_q_and_bounds_one_pass_vbr(const VP9_COMP *cpi,
                                             int *bottom_index,
                                             int *top_index) {
  const VP9_COMMON *const cm = &cpi->common;
  const RATE_CONTROL *const rc = &cpi->rc;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  const int cq_level = get_active_cq_level_one_pass(rc, oxcf);
  int active_best_quality;
  int active_worst_quality = calc_active_worst_quality_one_pass_vbr(cpi);
  int q;
  int *inter_minq;
  ASSIGN_MINQ_TABLE(cm->bit_depth, inter_minq);

  if (frame_is_intra_only(cm)) {
    if (oxcf->rc_mode == VPX_Q) {
      int qindex = cq_level;
      double qstart = vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      int delta_qindex =
          vp9_compute_qdelta(rc, qstart, qstart * 0.25, cm->bit_depth);
      active_best_quality = VPXMAX(qindex + delta_qindex, rc->best_quality);
    } else if (rc->this_key_frame_forced) {
      // Handle the special case for key frames forced when we have reached
      // the maximum key frame interval. Here force the Q to a range
      // based on the ambient Q to reduce the risk of popping.
      int qindex = rc->last_boosted_qindex;
      double last_boosted_q = vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      int delta_qindex = vp9_compute_qdelta(
          rc, last_boosted_q, last_boosted_q * 0.75, cm->bit_depth);
      active_best_quality = VPXMAX(qindex + delta_qindex, rc->best_quality);
    } else {
      // not first frame of one pass and kf_boost is set
      double q_adj_factor = 1.0;
      double q_val;

      active_best_quality = get_kf_active_quality(
          rc, rc->avg_frame_qindex[KEY_FRAME], cm->bit_depth);

      // Allow somewhat lower kf minq with small image formats.
      if ((cm->width * cm->height) <= (352 * 288)) {
        q_adj_factor -= 0.25;
      }

      // Convert the adjustment factor to a qindex delta
      // on active_best_quality.
      q_val = vp9_convert_qindex_to_q(active_best_quality, cm->bit_depth);
      active_best_quality +=
          vp9_compute_qdelta(rc, q_val, q_val * q_adj_factor, cm->bit_depth);
    }
  } else if (!rc->is_src_frame_alt_ref &&
             (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame)) {
    // Use the lower of active_worst_quality and recent
    // average Q as basis for GF/ARF best Q limit unless last frame was
    // a key frame.
    if (rc->frames_since_key > 1) {
      if (rc->avg_frame_qindex[INTER_FRAME] < active_worst_quality) {
        q = rc->avg_frame_qindex[INTER_FRAME];
      } else {
        q = active_worst_quality;
      }
    } else {
      q = rc->avg_frame_qindex[KEY_FRAME];
    }
    // For constrained quality don't allow Q less than the cq level
    if (oxcf->rc_mode == VPX_CQ) {
      if (q < cq_level) q = cq_level;

      active_best_quality = get_gf_active_quality(cpi, q, cm->bit_depth);

      // Constrained quality use slightly lower active best.
      active_best_quality = active_best_quality * 15 / 16;

    } else if (oxcf->rc_mode == VPX_Q) {
      int qindex = cq_level;
      double qstart = vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      int delta_qindex;
      if (cpi->refresh_alt_ref_frame)
        delta_qindex =
            vp9_compute_qdelta(rc, qstart, qstart * 0.40, cm->bit_depth);
      else
        delta_qindex =
            vp9_compute_qdelta(rc, qstart, qstart * 0.50, cm->bit_depth);
      active_best_quality = VPXMAX(qindex + delta_qindex, rc->best_quality);
    } else {
      active_best_quality = get_gf_active_quality(cpi, q, cm->bit_depth);
    }
  } else {
    if (oxcf->rc_mode == VPX_Q) {
      int qindex = cq_level;
      double qstart = vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      double delta_rate[FIXED_GF_INTERVAL] = { 0.50, 1.0, 0.85, 1.0,
                                               0.70, 1.0, 0.85, 1.0 };
      int delta_qindex = vp9_compute_qdelta(
          rc, qstart,
          qstart * delta_rate[cm->current_video_frame % FIXED_GF_INTERVAL],
          cm->bit_depth);
      active_best_quality = VPXMAX(qindex + delta_qindex, rc->best_quality);
    } else {
      // Use the min of the average Q and active_worst_quality as basis for
      // active_best.
      if (cm->current_video_frame > 1) {
        q = VPXMIN(rc->avg_frame_qindex[INTER_FRAME], active_worst_quality);
        active_best_quality = inter_minq[q];
      } else {
        active_best_quality = inter_minq[rc->avg_frame_qindex[KEY_FRAME]];
      }
      // For the constrained quality mode we don't want
      // q to fall below the cq level.
      if ((oxcf->rc_mode == VPX_CQ) && (active_best_quality < cq_level)) {
        active_best_quality = cq_level;
      }
    }
  }

  // Clip the active best and worst quality values to limits
  active_best_quality =
      clamp(active_best_quality, rc->best_quality, rc->worst_quality);
  active_worst_quality =
      clamp(active_worst_quality, active_best_quality, rc->worst_quality);

  *top_index = active_worst_quality;
  *bottom_index = active_best_quality;

#if LIMIT_QRANGE_FOR_ALTREF_AND_KEY
  {
    int qdelta = 0;
    vpx_clear_system_state();

    // Limit Q range for the adaptive loop.
    if (cm->frame_type == KEY_FRAME && !rc->this_key_frame_forced &&
        !(cm->current_video_frame == 0)) {
      qdelta = vp9_compute_qdelta_by_rate(
          &cpi->rc, cm->frame_type, active_worst_quality, 2.0, cm->bit_depth);
    } else if (!rc->is_src_frame_alt_ref &&
               (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame)) {
      qdelta = vp9_compute_qdelta_by_rate(
          &cpi->rc, cm->frame_type, active_worst_quality, 1.75, cm->bit_depth);
    }
    if (rc->high_source_sad && cpi->sf.use_altref_onepass) qdelta = 0;
    *top_index = active_worst_quality + qdelta;
    *top_index = (*top_index > *bottom_index) ? *top_index : *bottom_index;
  }
#endif

  if (oxcf->rc_mode == VPX_Q) {
    q = active_best_quality;
    // Special case code to try and match quality with forced key frames
  } else if ((cm->frame_type == KEY_FRAME) && rc->this_key_frame_forced) {
    q = rc->last_boosted_qindex;
  } else {
    q = vp9_rc_regulate_q(cpi, rc->this_frame_target, active_best_quality,
                          active_worst_quality);

    // For no lookahead: if buffer_level indicates overshoot, then avoid going
    // to very low QP. This reduces overshoot observed in Issue: 376707227.
    // Note the buffer_level is updated for every encoded frame as:
    // buffer_level - starting_buffer_level += (avg_frame_bandwidth -
    // encoded_frame_size). So normalizing this with framerate and #encoded
    // frames (current_video_frame) gives the difference/error between target
    // and encoding bitrate. The additional avg_frame_bandwidth term is to
    // compensate for the pre-encoded buffer update (in
    // vp9_rc_get_one_pass_vbr_params).
    const int qp_thresh = 32;
    const int64_t bitrate_err =
        (int64_t)(cpi->framerate *
                  (rc->buffer_level - rc->starting_buffer_level -
                   rc->avg_frame_bandwidth) /
                  (cm->current_video_frame + 1));
    // Threshold may be tuned, but for now condition this on low QP.
    if (cpi->oxcf.lag_in_frames == 0 && bitrate_err / 1000 < -10 &&
        qp_thresh < rc->worst_quality &&
        (q < qp_thresh || *top_index < qp_thresh)) {
      q = qp_thresh;
      *top_index = VPXMAX(*top_index, q);
    }

    if (q > *top_index) {
      // Special case when we are targeting the max allowed rate
      if (rc->this_frame_target >= rc->max_frame_bandwidth)
        *top_index = q;
      else
        q = *top_index;
    }
  }

  assert(*top_index <= rc->worst_quality && *top_index >= rc->best_quality);
  assert(*bottom_index <= rc->worst_quality &&
         *bottom_index >= rc->best_quality);
  assert(q <= rc->worst_quality && q >= rc->best_quality);
  return q;
}

int vp9_frame_type_qdelta(const VP9_COMP *cpi, int rf_level, int q) {
  static const double rate_factor_deltas[RATE_FACTOR_LEVELS] = {
    1.00,  // INTER_NORMAL
    1.00,  // INTER_HIGH
    1.50,  // GF_ARF_LOW
    1.75,  // GF_ARF_STD
    2.00,  // KF_STD
  };
  const VP9_COMMON *const cm = &cpi->common;

  int qdelta = vp9_compute_qdelta_by_rate(
      &cpi->rc, cm->frame_type, q, rate_factor_deltas[rf_level], cm->bit_depth);
  return qdelta;
}

#define STATIC_MOTION_THRESH 95

static void pick_kf_q_bound_two_pass(const VP9_COMP *cpi, int *bottom_index,
                                     int *top_index) {
  const VP9_COMMON *const cm = &cpi->common;
  const RATE_CONTROL *const rc = &cpi->rc;
  int active_best_quality;
  int active_worst_quality = cpi->twopass.active_worst_quality;

  if (rc->this_key_frame_forced) {
    // Handle the special case for key frames forced when we have reached
    // the maximum key frame interval. Here force the Q to a range
    // based on the ambient Q to reduce the risk of popping.
    double last_boosted_q;
    int delta_qindex;
    int qindex;

    if (cpi->twopass.last_kfgroup_zeromotion_pct >= STATIC_MOTION_THRESH) {
      qindex = VPXMIN(rc->last_kf_qindex, rc->last_boosted_qindex);
      active_best_quality = qindex;
      last_boosted_q = vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      delta_qindex = vp9_compute_qdelta(rc, last_boosted_q,
                                        last_boosted_q * 1.25, cm->bit_depth);
      active_worst_quality =
          VPXMIN(qindex + delta_qindex, active_worst_quality);
    } else {
      qindex = rc->last_boosted_qindex;
      last_boosted_q = vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      delta_qindex = vp9_compute_qdelta(rc, last_boosted_q,
                                        last_boosted_q * 0.75, cm->bit_depth);
      active_best_quality = VPXMAX(qindex + delta_qindex, rc->best_quality);
    }
  } else {
    // Not forced keyframe.
    double q_adj_factor = 1.0;
    double q_val;
    // Baseline value derived from cpi->active_worst_quality and kf boost.
    active_best_quality =
        get_kf_active_quality(rc, active_worst_quality, cm->bit_depth);
    if (cpi->twopass.kf_zeromotion_pct >= STATIC_KF_GROUP_THRESH) {
      active_best_quality /= 4;
    }

    // Don't allow the active min to be lossless (q0) unlesss the max q
    // already indicates lossless.
    active_best_quality =
        VPXMIN(active_worst_quality, VPXMAX(1, active_best_quality));

    // Allow somewhat lower kf minq with small image formats.
    if ((cm->width * cm->height) <= (352 * 288)) {
      q_adj_factor -= 0.25;
    }

    // Make a further adjustment based on the kf zero motion measure.
    q_adj_factor += 0.05 - (0.001 * (double)cpi->twopass.kf_zeromotion_pct);

    // Convert the adjustment factor to a qindex delta
    // on active_best_quality.
    q_val = vp9_convert_qindex_to_q(active_best_quality, cm->bit_depth);
    active_best_quality +=
        vp9_compute_qdelta(rc, q_val, q_val * q_adj_factor, cm->bit_depth);
  }
  *top_index = active_worst_quality;
  *bottom_index = active_best_quality;
}

static int rc_constant_q(const VP9_COMP *cpi, int *bottom_index, int *top_index,
                         int gf_group_index) {
  const VP9_COMMON *const cm = &cpi->common;
  const RATE_CONTROL *const rc = &cpi->rc;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  const GF_GROUP *gf_group = &cpi->twopass.gf_group;
  const int is_intra_frame = frame_is_intra_only(cm);

  const int cq_level = get_active_cq_level_two_pass(&cpi->twopass, rc, oxcf);

  int q = cq_level;
  int active_best_quality = cq_level;
  int active_worst_quality = cq_level;

  // Key frame qp decision
  if (is_intra_frame && rc->frames_to_key > 1)
    pick_kf_q_bound_two_pass(cpi, &active_best_quality, &active_worst_quality);

  // ARF / GF qp decision
  if (!is_intra_frame && !rc->is_src_frame_alt_ref &&
      cpi->refresh_alt_ref_frame) {
    active_best_quality = get_gf_active_quality(cpi, q, cm->bit_depth);

    // Modify best quality for second level arfs. For mode VPX_Q this
    // becomes the baseline frame q.
    if (gf_group->rf_level[gf_group_index] == GF_ARF_LOW) {
      const int layer_depth = gf_group->layer_depth[gf_group_index];
      // linearly fit the frame q depending on the layer depth index from
      // the base layer ARF.
      active_best_quality = ((layer_depth - 1) * cq_level +
                             active_best_quality + layer_depth / 2) /
                            layer_depth;
    }
  }

  q = active_best_quality;
  *top_index = active_worst_quality;
  *bottom_index = active_best_quality;
  return q;
}

int vp9_rc_pick_q_and_bounds_two_pass(const VP9_COMP *cpi, int *bottom_index,
                                      int *top_index, int gf_group_index) {
  const VP9_COMMON *const cm = &cpi->common;
  const RATE_CONTROL *const rc = &cpi->rc;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  const GF_GROUP *gf_group = &cpi->twopass.gf_group;
  const int cq_level = get_active_cq_level_two_pass(&cpi->twopass, rc, oxcf);
  int active_best_quality;
  int active_worst_quality = cpi->twopass.active_worst_quality;
  int q;
  int *inter_minq;
  int arf_active_best_quality_hl;
  int *arfgf_high_motion_minq, *arfgf_low_motion_minq;
  const int boost_frame =
      !rc->is_src_frame_alt_ref &&
      (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame);

  ASSIGN_MINQ_TABLE(cm->bit_depth, inter_minq);

  if (oxcf->rc_mode == VPX_Q)
    return rc_constant_q(cpi, bottom_index, top_index, gf_group_index);

  if (frame_is_intra_only(cm)) {
    pick_kf_q_bound_two_pass(cpi, &active_best_quality, &active_worst_quality);
  } else if (boost_frame) {
    // Use the lower of active_worst_quality and recent
    // average Q as basis for GF/ARF best Q limit unless last frame was
    // a key frame.
    if (rc->frames_since_key > 1 &&
        rc->avg_frame_qindex[INTER_FRAME] < active_worst_quality) {
      q = rc->avg_frame_qindex[INTER_FRAME];
    } else {
      q = active_worst_quality;
    }
    // For constrained quality don't allow Q less than the cq level
    if (oxcf->rc_mode == VPX_CQ) {
      if (q < cq_level) q = cq_level;
    }
    active_best_quality = get_gf_active_quality(cpi, q, cm->bit_depth);
    arf_active_best_quality_hl = active_best_quality;

    if (rc->arf_increase_active_best_quality == 1) {
      ASSIGN_MINQ_TABLE(cm->bit_depth, arfgf_high_motion_minq);
      arf_active_best_quality_hl = arfgf_high_motion_minq[q];
    } else if (rc->arf_increase_active_best_quality == -1) {
      ASSIGN_MINQ_TABLE(cm->bit_depth, arfgf_low_motion_minq);
      arf_active_best_quality_hl = arfgf_low_motion_minq[q];
    }
    active_best_quality =
        (int)((double)active_best_quality *
                  rc->arf_active_best_quality_adjustment_factor +
              (double)arf_active_best_quality_hl *
                  (1.0 - rc->arf_active_best_quality_adjustment_factor));

    // Modify best quality for second level arfs. For mode VPX_Q this
    // becomes the baseline frame q.
    if (gf_group->rf_level[gf_group_index] == GF_ARF_LOW) {
      const int layer_depth = gf_group->layer_depth[gf_group_index];
      // linearly fit the frame q depending on the layer depth index from
      // the base layer ARF.
      active_best_quality =
          ((layer_depth - 1) * q + active_best_quality + layer_depth / 2) /
          layer_depth;
    }
  } else {
    active_best_quality = inter_minq[active_worst_quality];

    // For the constrained quality mode we don't want
    // q to fall below the cq level.
    if ((oxcf->rc_mode == VPX_CQ) && (active_best_quality < cq_level)) {
      active_best_quality = cq_level;
    }
  }

  // Extension to max or min Q if undershoot or overshoot is outside
  // the permitted range.
  if (frame_is_intra_only(cm) || boost_frame) {
    const int layer_depth = gf_group->layer_depth[gf_group_index];
    active_best_quality -=
        (cpi->twopass.extend_minq + cpi->twopass.extend_minq_fast);
    active_worst_quality += (cpi->twopass.extend_maxq / 2);

    if (gf_group->rf_level[gf_group_index] == GF_ARF_LOW) {
      assert(layer_depth > 1);
      active_best_quality =
          VPXMAX(active_best_quality,
                 cpi->twopass.last_qindex_of_arf_layer[layer_depth - 1]);
    }
  } else {
    const int max_layer_depth = gf_group->max_layer_depth;
    assert(max_layer_depth > 0);

    active_best_quality -=
        (cpi->twopass.extend_minq + cpi->twopass.extend_minq_fast) / 2;
    active_worst_quality += cpi->twopass.extend_maxq;

    // For normal frames do not allow an active minq lower than the q used for
    // the last boosted frame.
    active_best_quality =
        VPXMAX(active_best_quality,
               cpi->twopass.last_qindex_of_arf_layer[max_layer_depth - 1]);
  }

#if LIMIT_QRANGE_FOR_ALTREF_AND_KEY
  vpx_clear_system_state();
  // Static forced key frames Q restrictions dealt with elsewhere.
  if (!frame_is_intra_only(cm) || !rc->this_key_frame_forced ||
      cpi->twopass.last_kfgroup_zeromotion_pct < STATIC_MOTION_THRESH) {
    int qdelta = vp9_frame_type_qdelta(cpi, gf_group->rf_level[gf_group_index],
                                       active_worst_quality);
    active_worst_quality =
        VPXMAX(active_worst_quality + qdelta, active_best_quality);
  }
#endif

  // Modify active_best_quality for downscaled normal frames.
  if (rc->frame_size_selector != UNSCALED && !frame_is_kf_gf_arf(cpi)) {
    int qdelta = vp9_compute_qdelta_by_rate(
        rc, cm->frame_type, active_best_quality, 2.0, cm->bit_depth);
    active_best_quality =
        VPXMAX(active_best_quality + qdelta, rc->best_quality);
  }

  active_best_quality =
      clamp(active_best_quality, rc->best_quality, rc->worst_quality);
  active_worst_quality =
      clamp(active_worst_quality, active_best_quality, rc->worst_quality);

  if (frame_is_intra_only(cm) && rc->this_key_frame_forced) {
    // If static since last kf use better of last boosted and last kf q.
    if (cpi->twopass.last_kfgroup_zeromotion_pct >= STATIC_MOTION_THRESH) {
      q = VPXMIN(rc->last_kf_qindex, rc->last_boosted_qindex);
    } else {
      q = rc->last_boosted_qindex;
    }
  } else if (frame_is_intra_only(cm) && !rc->this_key_frame_forced) {
    q = active_best_quality;
  } else {
    q = vp9_rc_regulate_q(cpi, rc->this_frame_target, active_best_quality,
                          active_worst_quality);
    if (q > active_worst_quality) {
      // Special case when we are targeting the max allowed rate.
      if (rc->this_frame_target >= rc->max_frame_bandwidth)
        active_worst_quality = q;
      else
        q = active_worst_quality;
    }
  }

  *top_index = active_worst_quality;
  *bottom_index = active_best_quality;

  assert(*top_index <= rc->worst_quality && *top_index >= rc->best_quality);
  assert(*bottom_index <= rc->worst_quality &&
         *bottom_index >= rc->best_quality);
  assert(q <= rc->worst_quality && q >= rc->best_quality);
  return q;
}

int vp9_rc_pick_q_and_bounds(const VP9_COMP *cpi, int *bottom_index,
                             int *top_index) {
  int q;
  const int gf_group_index = cpi->twopass.gf_group.index;
  if (cpi->oxcf.pass == 0) {
    if (cpi->oxcf.rc_mode == VPX_CBR)
      q = rc_pick_q_and_bounds_one_pass_cbr(cpi, bottom_index, top_index);
    else
      q = rc_pick_q_and_bounds_one_pass_vbr(cpi, bottom_index, top_index);
  } else {
    q = vp9_rc_pick_q_and_bounds_two_pass(cpi, bottom_index, top_index,
                                          gf_group_index);
  }
  if (cpi->sf.use_nonrd_pick_mode) {
    if (cpi->sf.force_frame_boost == 1) q -= cpi->sf.max_delta_qindex;

    if (q < *bottom_index)
      *bottom_index = q;
    else if (q > *top_index)
      *top_index = q;
  }
  return q;
}

void vp9_configure_buffer_updates(VP9_COMP *cpi, int gf_group_index) {
  VP9_COMMON *cm = &cpi->common;
  TWO_PASS *const twopass = &cpi->twopass;

  cpi->rc.is_src_frame_alt_ref = 0;
  cm->show_existing_frame = 0;
  cpi->rc.show_arf_as_gld = 0;
  switch (twopass->gf_group.update_type[gf_group_index]) {
    case KF_UPDATE:
      cpi->refresh_last_frame = 1;
      cpi->refresh_golden_frame = 1;
      cpi->refresh_alt_ref_frame = 1;
      break;
    case LF_UPDATE:
      cpi->refresh_last_frame = 1;
      cpi->refresh_golden_frame = 0;
      cpi->refresh_alt_ref_frame = 0;
      break;
    case GF_UPDATE:
      cpi->refresh_last_frame = 1;
      cpi->refresh_golden_frame = 1;
      cpi->refresh_alt_ref_frame = 0;
      break;
    case OVERLAY_UPDATE:
      cpi->refresh_last_frame = 0;
      cpi->refresh_golden_frame = 1;
      cpi->refresh_alt_ref_frame = 0;
      cpi->rc.is_src_frame_alt_ref = 1;
      if (cpi->rc.preserve_arf_as_gld) {
        cpi->rc.show_arf_as_gld = 1;
        cpi->refresh_golden_frame = 0;
        cm->show_existing_frame = 1;
        cm->refresh_frame_context = 0;
      }
      break;
    case MID_OVERLAY_UPDATE:
      cpi->refresh_last_frame = 1;
      cpi->refresh_golden_frame = 0;
      cpi->refresh_alt_ref_frame = 0;
      cpi->rc.is_src_frame_alt_ref = 1;
      break;
    case USE_BUF_FRAME:
      cpi->refresh_last_frame = 0;
      cpi->refresh_golden_frame = 0;
      cpi->refresh_alt_ref_frame = 0;
      cpi->rc.is_src_frame_alt_ref = 1;
      cm->show_existing_frame = 1;
      cm->refresh_frame_context = 0;
      break;
    default:
      assert(twopass->gf_group.update_type[gf_group_index] == ARF_UPDATE);
      cpi->refresh_last_frame = 0;
      cpi->refresh_golden_frame = 0;
      cpi->refresh_alt_ref_frame = 1;
      break;
  }
}

void vp9_rc_compute_frame_size_bounds(const VP9_COMP *cpi, int frame_target,
                                      int *frame_under_shoot_limit,
                                      int *frame_over_shoot_limit) {
  if (cpi->oxcf.rc_mode == VPX_Q) {
    *frame_under_shoot_limit = 0;
    *frame_over_shoot_limit = INT_MAX;
  } else {
    // For very small rate targets where the fractional adjustment
    // may be tiny make sure there is at least a minimum range.
    const int tol_low =
        (int)(((int64_t)cpi->sf.recode_tolerance_low * frame_target) / 100);
    const int tol_high =
        (int)(((int64_t)cpi->sf.recode_tolerance_high * frame_target) / 100);
    *frame_under_shoot_limit = VPXMAX(frame_target - tol_low - 100, 0);
    *frame_over_shoot_limit =
        VPXMIN(frame_target + tol_high + 100, cpi->rc.max_frame_bandwidth);
  }
}

void vp9_rc_set_frame_target(VP9_COMP *cpi, int target) {
  const VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;

  rc->this_frame_target = target;

  // Modify frame size target when down-scaling.
  if (cpi->oxcf.resize_mode == RESIZE_DYNAMIC &&
      rc->frame_size_selector != UNSCALED) {
    rc->this_frame_target = (int)(rc->this_frame_target *
                                  rate_thresh_mult[rc->frame_size_selector]);
  }

  // Target rate per SB64 (including partial SB64s.
  const int64_t sb64_target_rate =
      ((int64_t)rc->this_frame_target * 64 * 64) / (cm->width * cm->height);
  rc->sb64_target_rate = (int)VPXMIN(sb64_target_rate, INT_MAX);
}

static void update_alt_ref_frame_stats(VP9_COMP *cpi) {
  // this frame refreshes means next frames don't unless specified by user
  RATE_CONTROL *const rc = &cpi->rc;
  rc->frames_since_golden = 0;

  // Mark the alt ref as done (setting to 0 means no further alt refs pending).
  rc->source_alt_ref_pending = 0;

  // Set the alternate reference frame active flag
  rc->source_alt_ref_active = 1;
}

static void update_golden_frame_stats(VP9_COMP *cpi) {
  RATE_CONTROL *const rc = &cpi->rc;

  // Update the Golden frame usage counts.
  if (cpi->refresh_golden_frame) {
    // this frame refreshes means next frames don't unless specified by user
    rc->frames_since_golden = 0;

    // If we are not using alt ref in the up and coming group clear the arf
    // active flag. In multi arf group case, if the index is not 0 then
    // we are overlaying a mid group arf so should not reset the flag.
    if (cpi->oxcf.pass == 2) {
      if (!rc->source_alt_ref_pending && (cpi->twopass.gf_group.index == 0))
        rc->source_alt_ref_active = 0;
    } else if (!rc->source_alt_ref_pending) {
      rc->source_alt_ref_active = 0;
    }

    // Decrement count down till next gf
    if (rc->frames_till_gf_update_due > 0) rc->frames_till_gf_update_due--;

  } else if (!cpi->refresh_alt_ref_frame) {
    // Decrement count down till next gf
    if (rc->frames_till_gf_update_due > 0) rc->frames_till_gf_update_due--;

    rc->frames_since_golden++;

    if (rc->show_arf_as_gld) {
      rc->frames_since_golden = 0;
      // If we are not using alt ref in the up and coming group clear the arf
      // active flag. In multi arf group case, if the index is not 0 then
      // we are overlaying a mid group arf so should not reset the flag.
      if (!rc->source_alt_ref_pending && (cpi->twopass.gf_group.index == 0))
        rc->source_alt_ref_active = 0;
    }
  }
}

static void update_altref_usage(VP9_COMP *const cpi) {
  VP9_COMMON *const cm = &cpi->common;
  int sum_ref_frame_usage = 0;
  int arf_frame_usage = 0;
  int mi_row, mi_col;
  if (cpi->rc.alt_ref_gf_group && !cpi->rc.is_src_frame_alt_ref &&
      !cpi->refresh_golden_frame && !cpi->refresh_alt_ref_frame)
    for (mi_row = 0; mi_row < cm->mi_rows; mi_row += 8) {
      for (mi_col = 0; mi_col < cm->mi_cols; mi_col += 8) {
        int sboffset = ((cm->mi_cols + 7) >> 3) * (mi_row >> 3) + (mi_col >> 3);
        sum_ref_frame_usage += cpi->count_arf_frame_usage[sboffset] +
                               cpi->count_lastgolden_frame_usage[sboffset];
        arf_frame_usage += cpi->count_arf_frame_usage[sboffset];
      }
    }
  if (sum_ref_frame_usage > 0) {
    double altref_count = 100.0 * arf_frame_usage / sum_ref_frame_usage;
    cpi->rc.perc_arf_usage =
        0.75 * cpi->rc.perc_arf_usage + 0.25 * altref_count;
  }
}

void vp9_compute_frame_low_motion(VP9_COMP *const cpi) {
  VP9_COMMON *const cm = &cpi->common;
  SVC *const svc = &cpi->svc;
  int mi_row, mi_col;
  MODE_INFO **mi = cm->mi_grid_visible;
  RATE_CONTROL *const rc = &cpi->rc;
  const int rows = cm->mi_rows, cols = cm->mi_cols;
  int cnt_zeromv = 0;
  for (mi_row = 0; mi_row < rows; mi_row++) {
    for (mi_col = 0; mi_col < cols; mi_col++) {
      if (mi[0]->ref_frame[0] == LAST_FRAME &&
          abs(mi[0]->mv[0].as_mv.row) < 16 && abs(mi[0]->mv[0].as_mv.col) < 16)
        cnt_zeromv++;
      mi++;
    }
    mi += 8;
  }
  cnt_zeromv = 100 * cnt_zeromv / (rows * cols);
  rc->avg_frame_low_motion = (3 * rc->avg_frame_low_motion + cnt_zeromv) >> 2;

  // For SVC: set avg_frame_low_motion (only computed on top spatial layer)
  // to all lower spatial layers.
  if (cpi->use_svc && svc->spatial_layer_id == svc->number_spatial_layers - 1) {
    int i;
    for (i = 0; i < svc->number_spatial_layers - 1; ++i) {
      const int layer = LAYER_IDS_TO_IDX(i, svc->temporal_layer_id,
                                         svc->number_temporal_layers);
      LAYER_CONTEXT *const lc = &svc->layer_context[layer];
      RATE_CONTROL *const lrc = &lc->rc;
      lrc->avg_frame_low_motion = rc->avg_frame_low_motion;
    }
  }
}

void vp9_rc_postencode_update(VP9_COMP *cpi, uint64_t bytes_used) {
  const VP9_COMMON *const cm = &cpi->common;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  RATE_CONTROL *const rc = &cpi->rc;
  SVC *const svc = &cpi->svc;
  const int qindex = cm->base_qindex;
  const GF_GROUP *gf_group = &cpi->twopass.gf_group;
  const int gf_group_index = cpi->twopass.gf_group.index;
  const int layer_depth = gf_group->layer_depth[gf_group_index];

  // Update rate control heuristics
  rc->projected_frame_size = (int)(bytes_used << 3);

  // Post encode loop adjustment of Q prediction.
  vp9_rc_update_rate_correction_factors(cpi);

  // Keep a record of last Q and ambient average Q.
  if (frame_is_intra_only(cm)) {
    rc->last_q[KEY_FRAME] = qindex;
    rc->avg_frame_qindex[KEY_FRAME] =
        ROUND_POWER_OF_TWO(3 * rc->avg_frame_qindex[KEY_FRAME] + qindex, 2);
    if (cpi->use_svc) {
      int i;
      for (i = 0; i < svc->number_temporal_layers; ++i) {
        const int layer = LAYER_IDS_TO_IDX(svc->spatial_layer_id, i,
                                           svc->number_temporal_layers);
        LAYER_CONTEXT *lc = &svc->layer_context[layer];
        RATE_CONTROL *lrc = &lc->rc;
        lrc->last_q[KEY_FRAME] = rc->last_q[KEY_FRAME];
        lrc->avg_frame_qindex[KEY_FRAME] = rc->avg_frame_qindex[KEY_FRAME];
      }
    }
  } else {
    if ((cpi->use_svc) ||
        (!rc->is_src_frame_alt_ref &&
         !(cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame))) {
      rc->last_q[INTER_FRAME] = qindex;
      rc->avg_frame_qindex[INTER_FRAME] =
          ROUND_POWER_OF_TWO(3 * rc->avg_frame_qindex[INTER_FRAME] + qindex, 2);
      rc->ni_frames++;
      rc->tot_q += vp9_convert_qindex_to_q(qindex, cm->bit_depth);
      rc->avg_q = rc->tot_q / rc->ni_frames;
      // Calculate the average Q for normal inter frames (not key or GFU
      // frames).
      rc->ni_tot_qi += qindex;
      rc->ni_av_qi = rc->ni_tot_qi / rc->ni_frames;
    }
  }

  if (cpi->use_svc) vp9_svc_adjust_avg_frame_qindex(cpi);

  // Keep record of last boosted (KF/KF/ARF) Q value.
  // If the current frame is coded at a lower Q then we also update it.
  // If all mbs in this group are skipped only update if the Q value is
  // better than that already stored.
  // This is used to help set quality in forced key frames to reduce popping
  if ((qindex < rc->last_boosted_qindex) || (cm->frame_type == KEY_FRAME) ||
      (!rc->constrained_gf_group &&
       (cpi->refresh_alt_ref_frame ||
        (cpi->refresh_golden_frame && !rc->is_src_frame_alt_ref)))) {
    rc->last_boosted_qindex = qindex;
  }

  if ((qindex < cpi->twopass.last_qindex_of_arf_layer[layer_depth]) ||
      (cm->frame_type == KEY_FRAME) ||
      (!rc->constrained_gf_group &&
       (cpi->refresh_alt_ref_frame ||
        (cpi->refresh_golden_frame && !rc->is_src_frame_alt_ref)))) {
    cpi->twopass.last_qindex_of_arf_layer[layer_depth] = qindex;
  }

  if (frame_is_intra_only(cm)) rc->last_kf_qindex = qindex;

  update_buffer_level_postencode(cpi, rc->projected_frame_size);

  // Rolling monitors of whether we are over or underspending used to help
  // regulate min and Max Q in two pass.
  if (!frame_is_intra_only(cm)) {
    rc->rolling_target_bits = (int)ROUND64_POWER_OF_TWO(
        (int64_t)rc->rolling_target_bits * 3 + rc->this_frame_target, 2);
    rc->rolling_actual_bits = (int)ROUND64_POWER_OF_TWO(
        (int64_t)rc->rolling_actual_bits * 3 + rc->projected_frame_size, 2);
    rc->long_rolling_target_bits = (int)ROUND64_POWER_OF_TWO(
        (int64_t)rc->long_rolling_target_bits * 31 + rc->this_frame_target, 5);
    rc->long_rolling_actual_bits = (int)ROUND64_POWER_OF_TWO(
        (int64_t)rc->long_rolling_actual_bits * 31 + rc->projected_frame_size,
        5);
  }

  // Actual bits spent
  rc->total_actual_bits += rc->projected_frame_size;
  rc->total_target_bits += cm->show_frame ? rc->avg_frame_bandwidth : 0;

  rc->total_target_vs_actual = rc->total_actual_bits - rc->total_target_bits;

  if (!cpi->use_svc) {
    if (is_altref_enabled(cpi) && cpi->refresh_alt_ref_frame &&
        (!frame_is_intra_only(cm)))
      // Update the alternate reference frame stats as appropriate.
      update_alt_ref_frame_stats(cpi);
    else
      // Update the Golden frame stats as appropriate.
      update_golden_frame_stats(cpi);
  }

  // If second (long term) temporal reference is used for SVC,
  // update the golden frame counter, only for base temporal layer.
  if (cpi->use_svc && svc->use_gf_temporal_ref_current_layer &&
      svc->temporal_layer_id == 0) {
    int i = 0;
    if (cpi->refresh_golden_frame)
      rc->frames_since_golden = 0;
    else
      rc->frames_since_golden++;
    // Decrement count down till next gf
    if (rc->frames_till_gf_update_due > 0) rc->frames_till_gf_update_due--;
    // Update the frames_since_golden for all upper temporal layers.
    for (i = 1; i < svc->number_temporal_layers; ++i) {
      const int layer = LAYER_IDS_TO_IDX(svc->spatial_layer_id, i,
                                         svc->number_temporal_layers);
      LAYER_CONTEXT *const lc = &svc->layer_context[layer];
      RATE_CONTROL *const lrc = &lc->rc;
      lrc->frames_since_golden = rc->frames_since_golden;
    }
  }

  if (frame_is_intra_only(cm)) rc->frames_since_key = 0;
  if (cm->show_frame) {
    rc->frames_since_key++;
    rc->frames_to_key--;
  }

  // Trigger the resizing of the next frame if it is scaled.
  if (oxcf->pass != 0) {
    cpi->resize_pending =
        rc->next_frame_size_selector != rc->frame_size_selector;
    rc->frame_size_selector = rc->next_frame_size_selector;
  }

  if (oxcf->pass == 0) {
    if (!frame_is_intra_only(cm))
      if (cpi->sf.use_altref_onepass) update_altref_usage(cpi);
    cpi->rc.last_frame_is_src_altref = cpi->rc.is_src_frame_alt_ref;
  }

  if (!frame_is_intra_only(cm)) rc->reset_high_source_sad = 0;

  rc->last_avg_frame_bandwidth = rc->avg_frame_bandwidth;
  if (cpi->use_svc && svc->spatial_layer_id < svc->number_spatial_layers - 1)
    svc->lower_layer_qindex = cm->base_qindex;
  cpi->deadline_mode_previous_frame = cpi->oxcf.mode;
}

void vp9_rc_postencode_update_drop_frame(VP9_COMP *cpi) {
  cpi->common.current_video_frame++;
  cpi->rc.frames_since_key++;
  cpi->rc.frames_to_key--;
  cpi->rc.rc_2_frame = 0;
  cpi->rc.rc_1_frame = 0;
  cpi->rc.last_avg_frame_bandwidth = cpi->rc.avg_frame_bandwidth;
  cpi->rc.last_q[INTER_FRAME] = cpi->common.base_qindex;
  // For SVC on dropped frame when framedrop_mode != LAYER_DROP:
  // in this mode the whole superframe may be dropped if only a single layer
  // has buffer underflow (below threshold). Since this can then lead to
  // increasing buffer levels/overflow for certain layers even though whole
  // superframe is dropped, we cap buffer level if its already stable.
  if (cpi->use_svc && cpi->svc.framedrop_mode != LAYER_DROP &&
      cpi->rc.buffer_level > cpi->rc.optimal_buffer_level) {
    cpi->rc.buffer_level = cpi->rc.optimal_buffer_level;
    cpi->rc.bits_off_target = cpi->rc.optimal_buffer_level;
  }
  cpi->deadline_mode_previous_frame = cpi->oxcf.mode;
}

int vp9_calc_pframe_target_size_one_pass_vbr(const VP9_COMP *cpi) {
  const RATE_CONTROL *const rc = &cpi->rc;
  const int af_ratio = rc->af_ratio_onepass_vbr;
  int64_t target =
      (!rc->is_src_frame_alt_ref &&
       (cpi->refresh_golden_frame || cpi->refresh_alt_ref_frame))
          ? ((int64_t)rc->avg_frame_bandwidth * rc->baseline_gf_interval *
             af_ratio) /
                (rc->baseline_gf_interval + af_ratio - 1)
          : ((int64_t)rc->avg_frame_bandwidth * rc->baseline_gf_interval) /
                (rc->baseline_gf_interval + af_ratio - 1);
  // For SVC: refresh flags are used to define the pattern, so we can't
  // use that for boosting the target size here.
  // TODO(marpan): Consider adding internal boost on TL0 for VBR-SVC.
  // For now just use the CBR logic for setting target size.
  if (cpi->use_svc) target = vp9_calc_pframe_target_size_one_pass_cbr(cpi);
  if (target > INT_MAX) target = INT_MAX;
  return vp9_rc_clamp_pframe_target_size(cpi, (int)target);
}

int vp9_calc_iframe_target_size_one_pass_vbr(const VP9_COMP *cpi) {
  static const int kf_ratio = 25;
  const RATE_CONTROL *rc = &cpi->rc;
  int target = rc->avg_frame_bandwidth;
  if (target > INT_MAX / kf_ratio)
    target = INT_MAX;
  else
    target = rc->avg_frame_bandwidth * kf_ratio;
  return vp9_rc_clamp_iframe_target_size(cpi, target);
}

static void adjust_gfint_frame_constraint(VP9_COMP *cpi, int frame_constraint) {
  RATE_CONTROL *const rc = &cpi->rc;
  rc->constrained_gf_group = 0;
  // Reset gf interval to make more equal spacing for frame_constraint.
  if ((frame_constraint <= 7 * rc->baseline_gf_interval >> 2) &&
      (frame_constraint > rc->baseline_gf_interval)) {
    rc->baseline_gf_interval = frame_constraint >> 1;
    if (rc->baseline_gf_interval < 5)
      rc->baseline_gf_interval = frame_constraint;
    rc->constrained_gf_group = 1;
  } else {
    // Reset to keep gf_interval <= frame_constraint.
    if (rc->baseline_gf_interval > frame_constraint) {
      rc->baseline_gf_interval = frame_constraint;
      rc->constrained_gf_group = 1;
    }
  }
}

void vp9_set_gf_update_one_pass_vbr(VP9_COMP *const cpi) {
  RATE_CONTROL *const rc = &cpi->rc;
  VP9_COMMON *const cm = &cpi->common;
  if (rc->frames_till_gf_update_due == 0) {
    double rate_err = 1.0;
    rc->gfu_boost = DEFAULT_GF_BOOST;
    if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cpi->oxcf.pass == 0) {
      vp9_cyclic_refresh_set_golden_update(cpi);
    } else {
      rc->baseline_gf_interval = VPXMIN(
          20, VPXMAX(10, (rc->min_gf_interval + rc->max_gf_interval) / 2));
    }
    rc->af_ratio_onepass_vbr = 10;
    if (rc->rolling_target_bits > 0)
      rate_err =
          (double)rc->rolling_actual_bits / (double)rc->rolling_target_bits;
    if (cm->current_video_frame > 30) {
      if (rc->avg_frame_qindex[INTER_FRAME] > (7 * rc->worst_quality) >> 3 &&
          rate_err > 3.5) {
        rc->baseline_gf_interval =
            VPXMIN(15, (3 * rc->baseline_gf_interval) >> 1);
      } else if (rc->avg_frame_low_motion > 0 &&
                 rc->avg_frame_low_motion < 20) {
        // Decrease gf interval for high motion case.
        rc->baseline_gf_interval = VPXMAX(6, rc->baseline_gf_interval >> 1);
      }
      // Adjust boost and af_ratio based on avg_frame_low_motion, which
      // varies between 0 and 100 (stationary, 100% zero/small motion).
      if (rc->avg_frame_low_motion > 0)
        rc->gfu_boost =
            VPXMAX(500, DEFAULT_GF_BOOST * (rc->avg_frame_low_motion << 1) /
                            (rc->avg_frame_low_motion + 100));
      else if (rc->avg_frame_low_motion == 0 && rate_err > 1.0)
        rc->gfu_boost = DEFAULT_GF_BOOST >> 1;
      rc->af_ratio_onepass_vbr = VPXMIN(15, VPXMAX(5, 3 * rc->gfu_boost / 400));
    }
    if (rc->constrain_gf_key_freq_onepass_vbr)
      adjust_gfint_frame_constraint(cpi, rc->frames_to_key);
    rc->frames_till_gf_update_due = rc->baseline_gf_interval;
    cpi->refresh_golden_frame = 1;
    rc->source_alt_ref_pending = 0;
    rc->alt_ref_gf_group = 0;
    if (cpi->sf.use_altref_onepass && cpi->oxcf.enable_auto_arf) {
      rc->source_alt_ref_pending = 1;
      rc->alt_ref_gf_group = 1;
    }
  }
}

void vp9_rc_get_one_pass_vbr_params(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  int target;
  if (!cpi->refresh_alt_ref_frame &&
      (cm->current_video_frame == 0 || (cpi->frame_flags & FRAMEFLAGS_KEY) ||
       rc->frames_to_key == 0 ||
       (cpi->oxcf.mode != cpi->deadline_mode_previous_frame))) {
    cm->frame_type = KEY_FRAME;
    rc->this_key_frame_forced =
        cm->current_video_frame != 0 && rc->frames_to_key == 0;
    rc->frames_to_key = cpi->oxcf.key_freq;
    rc->kf_boost = DEFAULT_KF_BOOST;
    rc->source_alt_ref_active = 0;
  } else {
    cm->frame_type = INTER_FRAME;
  }
  vp9_set_gf_update_one_pass_vbr(cpi);
  if (cm->frame_type == KEY_FRAME)
    target = vp9_calc_iframe_target_size_one_pass_vbr(cpi);
  else
    target = vp9_calc_pframe_target_size_one_pass_vbr(cpi);
  vp9_rc_set_frame_target(cpi, target);
  if (cm->show_frame) vp9_update_buffer_level_preencode(cpi);
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cpi->oxcf.pass == 0)
    vp9_cyclic_refresh_update_parameters(cpi);
}

int vp9_calc_pframe_target_size_one_pass_cbr(const VP9_COMP *cpi) {
  const VP9EncoderConfig *oxcf = &cpi->oxcf;
  const RATE_CONTROL *rc = &cpi->rc;
  const SVC *const svc = &cpi->svc;
  const int64_t diff = rc->optimal_buffer_level - rc->buffer_level;
  const int64_t one_pct_bits = 1 + rc->optimal_buffer_level / 100;
  int min_frame_target =
      VPXMAX(rc->avg_frame_bandwidth >> 4, FRAME_OVERHEAD_BITS);
  int64_t target;

  if (oxcf->gf_cbr_boost_pct) {
    const int af_ratio_pct = oxcf->gf_cbr_boost_pct + 100;
    target = cpi->refresh_golden_frame
                 ? ((int64_t)rc->avg_frame_bandwidth *
                    rc->baseline_gf_interval * af_ratio_pct) /
                       (rc->baseline_gf_interval * 100 + af_ratio_pct - 100)
                 : ((int64_t)rc->avg_frame_bandwidth *
                    rc->baseline_gf_interval * 100) /
                       (rc->baseline_gf_interval * 100 + af_ratio_pct - 100);
  } else {
    target = rc->avg_frame_bandwidth;
  }
  if (is_one_pass_svc(cpi)) {
    // Note that for layers, avg_frame_bandwidth is the cumulative
    // per-frame-bandwidth. For the target size of this frame, use the
    // layer average frame size (i.e., non-cumulative per-frame-bw).
    int layer = LAYER_IDS_TO_IDX(svc->spatial_layer_id, svc->temporal_layer_id,
                                 svc->number_temporal_layers);
    const LAYER_CONTEXT *lc = &svc->layer_context[layer];
    target = lc->avg_frame_size;
    min_frame_target = VPXMAX(lc->avg_frame_size >> 4, FRAME_OVERHEAD_BITS);
  }
  if (diff > 0) {
    // Lower the target bandwidth for this frame.
    const int pct_low = (int)VPXMIN(diff / one_pct_bits, oxcf->under_shoot_pct);
    target -= (target * pct_low) / 200;
  } else if (diff < 0) {
    // Increase the target bandwidth for this frame.
    const int pct_high =
        (int)VPXMIN(-diff / one_pct_bits, oxcf->over_shoot_pct);
    target += (target * pct_high) / 200;
  }
  if (oxcf->rc_max_inter_bitrate_pct) {
    const int64_t max_rate =
        (int64_t)rc->avg_frame_bandwidth * oxcf->rc_max_inter_bitrate_pct / 100;
    target = VPXMIN(target, max_rate);
  }
  if (target > INT_MAX) target = INT_MAX;
  return VPXMAX(min_frame_target, (int)target);
}

int vp9_calc_iframe_target_size_one_pass_cbr(const VP9_COMP *cpi) {
  const RATE_CONTROL *rc = &cpi->rc;
  const VP9EncoderConfig *oxcf = &cpi->oxcf;
  const SVC *const svc = &cpi->svc;
  int64_t target;
  if (cpi->common.current_video_frame == 0) {
    target = rc->starting_buffer_level / 2;
  } else {
    int kf_boost = 32;
    double framerate = cpi->framerate;
    if (svc->number_temporal_layers > 1 && oxcf->rc_mode == VPX_CBR) {
      // Use the layer framerate for temporal layers CBR mode.
      const int layer =
          LAYER_IDS_TO_IDX(svc->spatial_layer_id, svc->temporal_layer_id,
                           svc->number_temporal_layers);
      const LAYER_CONTEXT *lc = &svc->layer_context[layer];
      framerate = lc->framerate;
    }
    kf_boost = VPXMAX(kf_boost, (int)round(2 * framerate - 16));
    if (rc->frames_since_key < framerate / 2) {
      kf_boost = (int)round(kf_boost * rc->frames_since_key / (framerate / 2));
    }

    target = ((int64_t)(16 + kf_boost) * rc->avg_frame_bandwidth) >> 4;
  }
  target = VPXMIN(INT_MAX, target);
  return vp9_rc_clamp_iframe_target_size(cpi, (int)target);
}

static void set_intra_only_frame(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  SVC *const svc = &cpi->svc;
  // Don't allow intra_only frame for bypass/flexible SVC mode, or if number
  // of spatial layers is 1 or if number of spatial or temporal layers > 3.
  // Also if intra-only is inserted on very first frame, don't allow if
  // if number of temporal layers > 1. This is because on intra-only frame
  // only 3 reference buffers can be updated, but for temporal layers > 1
  // we generally need to use buffer slots 4 and 5.
  if ((cm->current_video_frame == 0 && svc->number_temporal_layers > 1) ||
      svc->number_spatial_layers > 3 || svc->number_temporal_layers > 3 ||
      svc->number_spatial_layers == 1)
    return;
  cm->show_frame = 0;
  cm->intra_only = 1;
  cm->frame_type = INTER_FRAME;
  cpi->ext_refresh_frame_flags_pending = 1;
  cpi->ext_refresh_last_frame = 1;
  cpi->ext_refresh_golden_frame = 1;
  cpi->ext_refresh_alt_ref_frame = 1;
  if (cm->current_video_frame == 0) {
    cpi->lst_fb_idx = 0;
    cpi->gld_fb_idx = 1;
    cpi->alt_fb_idx = 2;
  } else {
    int i;
    int count = 0;
    cpi->lst_fb_idx = -1;
    cpi->gld_fb_idx = -1;
    cpi->alt_fb_idx = -1;
    svc->update_buffer_slot[0] = 0;
    // For intra-only frame we need to refresh all slots that were
    // being used for the base layer (fb_idx_base[i] == 1).
    // Start with assigning last first, then golden and then alt.
    for (i = 0; i < REF_FRAMES; ++i) {
      if (svc->fb_idx_base[i] == 1) {
        svc->update_buffer_slot[0] |= 1 << i;
        count++;
      }
      if (count == 1 && cpi->lst_fb_idx == -1) cpi->lst_fb_idx = i;
      if (count == 2 && cpi->gld_fb_idx == -1) cpi->gld_fb_idx = i;
      if (count == 3 && cpi->alt_fb_idx == -1) cpi->alt_fb_idx = i;
    }
    // If golden or alt is not being used for base layer, then set them
    // to the lst_fb_idx.
    if (cpi->gld_fb_idx == -1) cpi->gld_fb_idx = cpi->lst_fb_idx;
    if (cpi->alt_fb_idx == -1) cpi->alt_fb_idx = cpi->lst_fb_idx;
    if (svc->temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_BYPASS) {
      cpi->ext_refresh_last_frame = 0;
      cpi->ext_refresh_golden_frame = 0;
      cpi->ext_refresh_alt_ref_frame = 0;
      cpi->ref_frame_flags = 0;
    }
  }
}

void vp9_rc_get_svc_params(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  SVC *const svc = &cpi->svc;
  int target = rc->avg_frame_bandwidth;
  int layer = LAYER_IDS_TO_IDX(svc->spatial_layer_id, svc->temporal_layer_id,
                               svc->number_temporal_layers);
  if (svc->first_spatial_layer_to_encode)
    svc->layer_context[svc->temporal_layer_id].is_key_frame = 0;
  // Periodic key frames is based on the super-frame counter
  // (svc.current_superframe), also only base spatial layer is key frame.
  // Key frame is set for any of the following: very first frame, frame flags
  // indicates key, superframe counter hits key frequency,(non-intra) sync
  // flag is set for spatial layer 0, or deadline mode changes.
  if ((cm->current_video_frame == 0 && !svc->previous_frame_is_intra_only) ||
      (cpi->frame_flags & FRAMEFLAGS_KEY) ||
      (cpi->oxcf.auto_key &&
       (svc->current_superframe % cpi->oxcf.key_freq == 0) &&
       !svc->previous_frame_is_intra_only && svc->spatial_layer_id == 0) ||
      (svc->spatial_layer_sync[0] == 1 && svc->spatial_layer_id == 0) ||
      (cpi->oxcf.mode != cpi->deadline_mode_previous_frame)) {
    cm->frame_type = KEY_FRAME;
    rc->source_alt_ref_active = 0;
    if (is_one_pass_svc(cpi)) {
      if (cm->current_video_frame > 0) vp9_svc_reset_temporal_layers(cpi, 1);
      layer = LAYER_IDS_TO_IDX(svc->spatial_layer_id, svc->temporal_layer_id,
                               svc->number_temporal_layers);
      svc->layer_context[layer].is_key_frame = 1;
      cpi->ref_frame_flags &= (~VP9_LAST_FLAG & ~VP9_GOLD_FLAG & ~VP9_ALT_FLAG);
      // Assumption here is that LAST_FRAME is being updated for a keyframe.
      // Thus no change in update flags.
      if (cpi->oxcf.rc_mode == VPX_CBR)
        target = vp9_calc_iframe_target_size_one_pass_cbr(cpi);
      else
        target = vp9_calc_iframe_target_size_one_pass_vbr(cpi);
    }
  } else {
    cm->frame_type = INTER_FRAME;
    if (is_one_pass_svc(cpi)) {
      LAYER_CONTEXT *lc = &svc->layer_context[layer];
      // Add condition current_video_frame > 0 for the case where first frame
      // is intra only followed by overlay/copy frame. In this case we don't
      // want to reset is_key_frame to 0 on overlay/copy frame.
      lc->is_key_frame =
          (svc->spatial_layer_id == 0 && cm->current_video_frame > 0)
              ? 0
              : svc->layer_context[svc->temporal_layer_id].is_key_frame;
      if (cpi->oxcf.rc_mode == VPX_CBR) {
        target = vp9_calc_pframe_target_size_one_pass_cbr(cpi);
      } else {
        double rate_err = 0.0;
        rc->fac_active_worst_inter = 140;
        rc->fac_active_worst_gf = 100;
        if (rc->rolling_target_bits > 0) {
          rate_err =
              (double)rc->rolling_actual_bits / (double)rc->rolling_target_bits;
          if (rate_err < 1.0)
            rc->fac_active_worst_inter = 120;
          else if (rate_err > 2.0)
            // Increase active_worst faster if rate fluctuation is high.
            rc->fac_active_worst_inter = 160;
        }
        target = vp9_calc_pframe_target_size_one_pass_vbr(cpi);
      }
    }
  }

  if (svc->simulcast_mode) {
    if (svc->spatial_layer_id > 0 &&
        svc->layer_context[layer].is_key_frame == 1) {
      cm->frame_type = KEY_FRAME;
      cpi->ref_frame_flags &= (~VP9_LAST_FLAG & ~VP9_GOLD_FLAG & ~VP9_ALT_FLAG);
      if (cpi->oxcf.rc_mode == VPX_CBR)
        target = vp9_calc_iframe_target_size_one_pass_cbr(cpi);
      else
        target = vp9_calc_iframe_target_size_one_pass_vbr(cpi);
    }
    // Set the buffer idx and refresh flags for key frames in simulcast mode.
    // Note the buffer slot for long-term reference is set below (line 2255),
    // and alt_ref is used for that on key frame. So use last and golden for
    // the other two normal slots.
    if (cm->frame_type == KEY_FRAME) {
      if (svc->number_spatial_layers == 2) {
        if (svc->spatial_layer_id == 0) {
          cpi->lst_fb_idx = 0;
          cpi->gld_fb_idx = 2;
          cpi->alt_fb_idx = 6;
        } else if (svc->spatial_layer_id == 1) {
          cpi->lst_fb_idx = 1;
          cpi->gld_fb_idx = 3;
          cpi->alt_fb_idx = 6;
        }
      } else if (svc->number_spatial_layers == 3) {
        if (svc->spatial_layer_id == 0) {
          cpi->lst_fb_idx = 0;
          cpi->gld_fb_idx = 3;
          cpi->alt_fb_idx = 6;
        } else if (svc->spatial_layer_id == 1) {
          cpi->lst_fb_idx = 1;
          cpi->gld_fb_idx = 4;
          cpi->alt_fb_idx = 6;
        } else if (svc->spatial_layer_id == 2) {
          cpi->lst_fb_idx = 2;
          cpi->gld_fb_idx = 5;
          cpi->alt_fb_idx = 7;
        }
      }
      cpi->ext_refresh_last_frame = 1;
      cpi->ext_refresh_golden_frame = 1;
      cpi->ext_refresh_alt_ref_frame = 1;
    }
  }

  // Check if superframe contains a sync layer request.
  vp9_svc_check_spatial_layer_sync(cpi);

  // If long term termporal feature is enabled, set the period of the update.
  // The update/refresh of this reference frame is always on base temporal
  // layer frame.
  if (svc->use_gf_temporal_ref_current_layer) {
    // Only use gf long-term prediction on non-key superframes.
    if (!svc->layer_context[svc->temporal_layer_id].is_key_frame) {
      // Use golden for this reference, which will be used for prediction.
      int index = svc->spatial_layer_id;
      if (svc->number_spatial_layers == 3) index = svc->spatial_layer_id - 1;
      assert(index >= 0);
      cpi->gld_fb_idx = svc->buffer_gf_temporal_ref[index].idx;
      // Enable prediction off LAST (last reference) and golden (which will
      // generally be further behind/long-term reference).
      cpi->ref_frame_flags = VP9_LAST_FLAG | VP9_GOLD_FLAG;
    }
    // Check for update/refresh of reference: only refresh on base temporal
    // layer.
    if (svc->temporal_layer_id == 0) {
      if (svc->layer_context[svc->temporal_layer_id].is_key_frame) {
        // On key frame we update the buffer index used for long term reference.
        // Use the alt_ref since it is not used or updated on key frames.
        int index = svc->spatial_layer_id;
        if (svc->number_spatial_layers == 3) index = svc->spatial_layer_id - 1;
        assert(index >= 0);
        cpi->alt_fb_idx = svc->buffer_gf_temporal_ref[index].idx;
        cpi->ext_refresh_alt_ref_frame = 1;
      } else if (rc->frames_till_gf_update_due == 0) {
        // Set perdiod of next update. Make it a multiple of 10, as the cyclic
        // refresh is typically ~10%, and we'd like the update to happen after
        // a few cylces of the refresh (so it better quality frame). Note the
        // cyclic refresh for SVC only operates on base temporal layer frames.
        // Choose 20 as perdiod for now (2 cycles).
        rc->baseline_gf_interval = 20;
        rc->frames_till_gf_update_due = rc->baseline_gf_interval;
        cpi->ext_refresh_golden_frame = 1;
        rc->gfu_boost = DEFAULT_GF_BOOST;
      }
    }
  } else if (!svc->use_gf_temporal_ref) {
    rc->frames_till_gf_update_due = INT_MAX;
    rc->baseline_gf_interval = INT_MAX;
  }
  if (svc->set_intra_only_frame) {
    set_intra_only_frame(cpi);
    if (cpi->oxcf.rc_mode == VPX_CBR)
      target = vp9_calc_iframe_target_size_one_pass_cbr(cpi);
    else
      target = vp9_calc_iframe_target_size_one_pass_vbr(cpi);
  }
  // Overlay frame predicts from LAST (intra-only)
  if (svc->previous_frame_is_intra_only) cpi->ref_frame_flags |= VP9_LAST_FLAG;

  // Any update/change of global cyclic refresh parameters (amount/delta-qp)
  // should be done here, before the frame qp is selected.
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
    vp9_cyclic_refresh_update_parameters(cpi);

  vp9_rc_set_frame_target(cpi, target);
  if (cm->show_frame) vp9_update_buffer_level_svc_preencode(cpi);

  if (cpi->oxcf.resize_mode == RESIZE_DYNAMIC && svc->single_layer_svc == 1 &&
      svc->spatial_layer_id == svc->first_spatial_layer_to_encode &&
      svc->temporal_layer_id == 0) {
    LAYER_CONTEXT *lc = NULL;
    cpi->resize_pending = vp9_resize_one_pass_cbr(cpi);
    if (cpi->resize_pending) {
      int tl, width, height;
      // Apply the same scale to all temporal layers.
      for (tl = 0; tl < svc->number_temporal_layers; tl++) {
        lc = &svc->layer_context[svc->spatial_layer_id *
                                     svc->number_temporal_layers +
                                 tl];
        lc->scaling_factor_num_resize =
            cpi->resize_scale_num * lc->scaling_factor_num;
        lc->scaling_factor_den_resize =
            cpi->resize_scale_den * lc->scaling_factor_den;
        // Reset rate control for all temporal layers.
        lc->rc.buffer_level = lc->rc.optimal_buffer_level;
        lc->rc.bits_off_target = lc->rc.optimal_buffer_level;
        lc->rc.rate_correction_factors[INTER_FRAME] =
            rc->rate_correction_factors[INTER_FRAME];
      }
      // Set the size for this current temporal layer.
      lc = &svc->layer_context[svc->spatial_layer_id *
                                   svc->number_temporal_layers +
                               svc->temporal_layer_id];
      get_layer_resolution(cpi->oxcf.width, cpi->oxcf.height,
                           lc->scaling_factor_num_resize,
                           lc->scaling_factor_den_resize, &width, &height);
      vp9_set_size_literal(cpi, width, height);
      svc->resize_set = 1;
    }
  } else {
    cpi->resize_pending = 0;
    svc->resize_set = 0;
  }
}

void vp9_rc_get_one_pass_cbr_params(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  int target;
  if ((cm->current_video_frame == 0) || (cpi->frame_flags & FRAMEFLAGS_KEY) ||
      (cpi->oxcf.auto_key && rc->frames_to_key == 0) ||
      (cpi->oxcf.mode != cpi->deadline_mode_previous_frame)) {
    cm->frame_type = KEY_FRAME;
    rc->frames_to_key = cpi->oxcf.key_freq;
    rc->kf_boost = DEFAULT_KF_BOOST;
    rc->source_alt_ref_active = 0;
  } else {
    cm->frame_type = INTER_FRAME;
  }
  if (rc->frames_till_gf_update_due == 0) {
    if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
      vp9_cyclic_refresh_set_golden_update(cpi);
    else
      rc->baseline_gf_interval =
          (rc->min_gf_interval + rc->max_gf_interval) / 2;
    rc->frames_till_gf_update_due = rc->baseline_gf_interval;
    // NOTE: frames_till_gf_update_due must be <= frames_to_key.
    if (rc->frames_till_gf_update_due > rc->frames_to_key)
      rc->frames_till_gf_update_due = rc->frames_to_key;
    cpi->refresh_golden_frame = 1;
    rc->gfu_boost = DEFAULT_GF_BOOST;
  }

  // Any update/change of global cyclic refresh parameters (amount/delta-qp)
  // should be done here, before the frame qp is selected.
  if (cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ)
    vp9_cyclic_refresh_update_parameters(cpi);

  if (frame_is_intra_only(cm))
    target = vp9_calc_iframe_target_size_one_pass_cbr(cpi);
  else
    target = vp9_calc_pframe_target_size_one_pass_cbr(cpi);

  vp9_rc_set_frame_target(cpi, target);

  if (cm->show_frame) vp9_update_buffer_level_preencode(cpi);

  if (cpi->oxcf.resize_mode == RESIZE_DYNAMIC)
    cpi->resize_pending = vp9_resize_one_pass_cbr(cpi);
  else
    cpi->resize_pending = 0;
}

int vp9_compute_qdelta(const RATE_CONTROL *rc, double qstart, double qtarget,
                       vpx_bit_depth_t bit_depth) {
  int start_index = rc->worst_quality;
  int target_index = rc->worst_quality;
  int i;

  // Convert the average q value to an index.
  for (i = rc->best_quality; i < rc->worst_quality; ++i) {
    start_index = i;
    if (vp9_convert_qindex_to_q(i, bit_depth) >= qstart) break;
  }

  // Convert the q target to an index
  for (i = rc->best_quality; i < rc->worst_quality; ++i) {
    target_index = i;
    if (vp9_convert_qindex_to_q(i, bit_depth) >= qtarget) break;
  }

  return target_index - start_index;
}

int vp9_compute_qdelta_by_rate(const RATE_CONTROL *rc, FRAME_TYPE frame_type,
                               int qindex, double rate_target_ratio,
                               vpx_bit_depth_t bit_depth) {
  int target_index = rc->worst_quality;
  int i;

  // Look up the current projected bits per block for the base index
  const int base_bits_per_mb =
      vp9_rc_bits_per_mb(frame_type, qindex, 1.0, bit_depth);

  // Find the target bits per mb based on the base value and given ratio.
  const int target_bits_per_mb = (int)(rate_target_ratio * base_bits_per_mb);

  // Convert the q target to an index
  for (i = rc->best_quality; i < rc->worst_quality; ++i) {
    if (vp9_rc_bits_per_mb(frame_type, i, 1.0, bit_depth) <=
        target_bits_per_mb) {
      target_index = i;
      break;
    }
  }
  return target_index - qindex;
}

void vp9_rc_set_gf_interval_range(const VP9_COMP *const cpi,
                                  RATE_CONTROL *const rc) {
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;

  // Special case code for 1 pass fixed Q mode tests
  if ((oxcf->pass == 0) && (oxcf->rc_mode == VPX_Q)) {
    rc->max_gf_interval = FIXED_GF_INTERVAL;
    rc->min_gf_interval = FIXED_GF_INTERVAL;
    rc->static_scene_max_gf_interval = FIXED_GF_INTERVAL;
  } else {
    double framerate = cpi->framerate;
    // Set Maximum gf/arf interval
    rc->max_gf_interval = oxcf->max_gf_interval;
    rc->min_gf_interval = oxcf->min_gf_interval;
    if (rc->min_gf_interval == 0) {
      rc->min_gf_interval = vp9_rc_get_default_min_gf_interval(
          oxcf->width, oxcf->height, framerate);
    }
    if (rc->max_gf_interval == 0) {
      rc->max_gf_interval =
          vp9_rc_get_default_max_gf_interval(framerate, rc->min_gf_interval);
    }

    // Extended max interval for genuinely static scenes like slide shows.
    rc->static_scene_max_gf_interval = MAX_STATIC_GF_GROUP_LENGTH;

    if (rc->max_gf_interval > rc->static_scene_max_gf_interval)
      rc->max_gf_interval = rc->static_scene_max_gf_interval;

    // Clamp min to max
    rc->min_gf_interval = VPXMIN(rc->min_gf_interval, rc->max_gf_interval);

    if (oxcf->target_level == LEVEL_AUTO) {
      const uint32_t pic_size = cpi->common.width * cpi->common.height;
      const uint32_t pic_breadth =
          VPXMAX(cpi->common.width, cpi->common.height);
      int i;
      for (i = 0; i < VP9_LEVELS; ++i) {
        if (vp9_level_defs[i].max_luma_picture_size >= pic_size &&
            vp9_level_defs[i].max_luma_picture_breadth >= pic_breadth) {
          if (rc->min_gf_interval <=
              (int)vp9_level_defs[i].min_altref_distance) {
            rc->min_gf_interval = (int)vp9_level_defs[i].min_altref_distance;
            rc->max_gf_interval =
                VPXMAX(rc->max_gf_interval, rc->min_gf_interval);
          }
          break;
        }
      }
    }
  }
}

void vp9_rc_update_framerate(VP9_COMP *cpi) {
  const VP9_COMMON *const cm = &cpi->common;
  const VP9EncoderConfig *const oxcf = &cpi->oxcf;
  RATE_CONTROL *const rc = &cpi->rc;

  rc->avg_frame_bandwidth = saturate_cast_double_to_int(
      round(oxcf->target_bandwidth / cpi->framerate));

  int64_t vbr_min_bits =
      (int64_t)rc->avg_frame_bandwidth * oxcf->two_pass_vbrmin_section / 100;
  vbr_min_bits = VPXMIN(vbr_min_bits, INT_MAX);

  rc->min_frame_bandwidth = VPXMAX((int)vbr_min_bits, FRAME_OVERHEAD_BITS);

  // A maximum bitrate for a frame is defined.
  // However this limit is extended if a very high rate is given on the command
  // line or the rate can not be achieved because of a user specified max q
  // (e.g. when the user specifies lossless encode).
  //
  // If a level is specified that requires a lower maximum rate then the level
  // value take precedence.
  int64_t vbr_max_bits =
      (int64_t)rc->avg_frame_bandwidth * oxcf->two_pass_vbrmax_section / 100;
  vbr_max_bits = VPXMIN(vbr_max_bits, INT_MAX);

  rc->max_frame_bandwidth =
      VPXMAX(VPXMAX((cm->MBs * MAX_MB_RATE), MAXRATE_1080P), (int)vbr_max_bits);

  vp9_rc_set_gf_interval_range(cpi, rc);
}

#define VBR_PCT_ADJUSTMENT_LIMIT 50
// For VBR...adjustment to the frame target based on error from previous frames
static void vbr_rate_correction(VP9_COMP *cpi, int *this_frame_target) {
  RATE_CONTROL *const rc = &cpi->rc;
  int64_t vbr_bits_off_target = rc->vbr_bits_off_target;
  int64_t frame_target = *this_frame_target;
  int frame_window = (int)VPXMIN(
      16, cpi->twopass.total_stats.count - cpi->common.current_video_frame);

  // Calcluate the adjustment to rate for this frame.
  if (frame_window > 0) {
    int64_t max_delta = (vbr_bits_off_target > 0)
                            ? (vbr_bits_off_target / frame_window)
                            : (-vbr_bits_off_target / frame_window);

    max_delta =
        VPXMIN(max_delta, ((frame_target * VBR_PCT_ADJUSTMENT_LIMIT) / 100));

    // vbr_bits_off_target > 0 means we have extra bits to spend
    if (vbr_bits_off_target > 0) {
      frame_target += VPXMIN(vbr_bits_off_target, max_delta);
    } else {
      frame_target -= VPXMIN(-vbr_bits_off_target, max_delta);
    }
  }

  // Fast redistribution of bits arising from massive local undershoot.
  // Don't do it for kf,arf,gf or overlay frames.
  if (!frame_is_kf_gf_arf(cpi) && !rc->is_src_frame_alt_ref &&
      rc->vbr_bits_off_target_fast) {
    int64_t one_frame_bits = VPXMAX(rc->avg_frame_bandwidth, frame_target);
    int64_t fast_extra_bits =
        VPXMIN(rc->vbr_bits_off_target_fast, one_frame_bits);
    fast_extra_bits =
        VPXMIN(fast_extra_bits,
               VPXMAX(one_frame_bits / 8, rc->vbr_bits_off_target_fast / 8));
    frame_target += fast_extra_bits;
    rc->vbr_bits_off_target_fast -= fast_extra_bits;
  }

  // Clamp the target for the frame to the maximum allowed for one frame.
  *this_frame_target = (int)VPXMIN(frame_target, INT_MAX);
}

void vp9_set_target_rate(VP9_COMP *cpi) {
  RATE_CONTROL *const rc = &cpi->rc;
  int target_rate = rc->base_frame_target;

  if (cpi->common.frame_type == KEY_FRAME)
    target_rate = vp9_rc_clamp_iframe_target_size(cpi, target_rate);
  else
    target_rate = vp9_rc_clamp_pframe_target_size(cpi, target_rate);

  if (!cpi->oxcf.vbr_corpus_complexity) {
    // Correction to rate target based on prior over or under shoot.
    if (cpi->oxcf.rc_mode == VPX_VBR || cpi->oxcf.rc_mode == VPX_CQ)
      vbr_rate_correction(cpi, &target_rate);
  }
  vp9_rc_set_frame_target(cpi, target_rate);
}

// Check if we should resize, based on average QP from past x frames.
// Only allow for resize at most one scale down for now, scaling factor is 2.
int vp9_resize_one_pass_cbr(VP9_COMP *cpi) {
  const VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  RESIZE_ACTION resize_action = NO_RESIZE;
  int avg_qp_thr1 = 70;
  int avg_qp_thr2 = 50;
  // Don't allow for resized frame to go below 320x180, resize in steps of 3/4.
  int min_width = (320 * 4) / 3;
  int min_height = (180 * 4) / 3;
  int down_size_on = 1;
  int force_downsize_rate = 0;
  cpi->resize_scale_num = 1;
  cpi->resize_scale_den = 1;
  // Don't resize on key frame; reset the counters on key frame.
  if (cm->frame_type == KEY_FRAME) {
    cpi->resize_avg_qp = 0;
    cpi->resize_count = 0;
    return 0;
  }

  // No resizing down if frame size is below some limit.
  if ((cm->width * cm->height) < min_width * min_height) down_size_on = 0;

#if CONFIG_VP9_TEMPORAL_DENOISING
  // If denoiser is on, apply a smaller qp threshold.
  if (cpi->oxcf.noise_sensitivity > 0) {
    avg_qp_thr1 = 60;
    avg_qp_thr2 = 40;
  }
#endif

  // Force downsize based on per-frame-bandwidth, for extreme case,
  // for HD input.
  if (cpi->resize_state == ORIG && cm->width * cm->height >= 1280 * 720) {
    if (rc->avg_frame_bandwidth < 300000 / 30) {
      resize_action = DOWN_ONEHALF;
      cpi->resize_state = ONE_HALF;
      force_downsize_rate = 1;
    } else if (rc->avg_frame_bandwidth < 400000 / 30) {
      resize_action = ONEHALFONLY_RESIZE ? DOWN_ONEHALF : DOWN_THREEFOUR;
      cpi->resize_state = ONEHALFONLY_RESIZE ? ONE_HALF : THREE_QUARTER;
      force_downsize_rate = 1;
    }
  } else if (cpi->resize_state == THREE_QUARTER &&
             cm->width * cm->height >= 960 * 540) {
    if (rc->avg_frame_bandwidth < 300000 / 30) {
      resize_action = DOWN_ONEHALF;
      cpi->resize_state = ONE_HALF;
      force_downsize_rate = 1;
    }
  }

  // Resize based on average buffer underflow and QP over some window.
  // Ignore samples close to key frame, since QP is usually high after key.
  if (!force_downsize_rate && cpi->rc.frames_since_key > cpi->framerate) {
    const int window = VPXMIN(30, (int)round(2 * cpi->framerate));
    cpi->resize_avg_qp += rc->last_q[INTER_FRAME];
    if (cpi->rc.buffer_level < (int)(30 * rc->optimal_buffer_level / 100))
      ++cpi->resize_buffer_underflow;
    ++cpi->resize_count;
    // Check for resize action every "window" frames.
    if (cpi->resize_count >= window) {
      int avg_qp = cpi->resize_avg_qp / cpi->resize_count;
      // Resize down if buffer level has underflowed sufficient amount in past
      // window, and we are at original or 3/4 of original resolution.
      // Resize back up if average QP is low, and we are currently in a resized
      // down state, i.e. 1/2 or 3/4 of original resolution.
      // Currently, use a flag to turn 3/4 resizing feature on/off.
      if (cpi->resize_buffer_underflow > (cpi->resize_count >> 2) &&
          down_size_on) {
        if (cpi->resize_state == THREE_QUARTER) {
          resize_action = DOWN_ONEHALF;
          cpi->resize_state = ONE_HALF;
        } else if (cpi->resize_state == ORIG) {
          resize_action = ONEHALFONLY_RESIZE ? DOWN_ONEHALF : DOWN_THREEFOUR;
          cpi->resize_state = ONEHALFONLY_RESIZE ? ONE_HALF : THREE_QUARTER;
        }
      } else if (cpi->resize_state != ORIG &&
                 avg_qp < avg_qp_thr1 * cpi->rc.worst_quality / 100) {
        if (cpi->resize_state == THREE_QUARTER ||
            avg_qp < avg_qp_thr2 * cpi->rc.worst_quality / 100 ||
            ONEHALFONLY_RESIZE) {
          resize_action = UP_ORIG;
          cpi->resize_state = ORIG;
        } else if (cpi->resize_state == ONE_HALF) {
          resize_action = UP_THREEFOUR;
          cpi->resize_state = THREE_QUARTER;
        }
      }
      // Reset for next window measurement.
      cpi->resize_avg_qp = 0;
      cpi->resize_count = 0;
      cpi->resize_buffer_underflow = 0;
    }
  }
  // If decision is to resize, reset some quantities, and check is we should
  // reduce rate correction factor,
  if (resize_action != NO_RESIZE) {
    int target_bits_per_frame;
    int active_worst_quality;
    int qindex;
    int tot_scale_change;
    if (resize_action == DOWN_THREEFOUR || resize_action == UP_THREEFOUR) {
      cpi->resize_scale_num = 3;
      cpi->resize_scale_den = 4;
    } else if (resize_action == DOWN_ONEHALF) {
      cpi->resize_scale_num = 1;
      cpi->resize_scale_den = 2;
    } else {  // UP_ORIG or anything else
      cpi->resize_scale_num = 1;
      cpi->resize_scale_den = 1;
    }
    tot_scale_change = (cpi->resize_scale_den * cpi->resize_scale_den) /
                       (cpi->resize_scale_num * cpi->resize_scale_num);
    // Reset buffer level to optimal, update target size.
    rc->buffer_level = rc->optimal_buffer_level;
    rc->bits_off_target = rc->optimal_buffer_level;
    rc->this_frame_target = vp9_calc_pframe_target_size_one_pass_cbr(cpi);
    // Get the projected qindex, based on the scaled target frame size (scaled
    // so target_bits_per_mb in vp9_rc_regulate_q will be correct target).
    target_bits_per_frame = (resize_action >= 0)
                                ? rc->this_frame_target * tot_scale_change
                                : rc->this_frame_target / tot_scale_change;
    active_worst_quality = calc_active_worst_quality_one_pass_cbr(cpi);
    qindex = vp9_rc_regulate_q(cpi, target_bits_per_frame, rc->best_quality,
                               active_worst_quality);
    // If resize is down, check if projected q index is close to worst_quality,
    // and if so, reduce the rate correction factor (since likely can afford
    // lower q for resized frame).
    if (resize_action > 0 && qindex > 90 * cpi->rc.worst_quality / 100) {
      rc->rate_correction_factors[INTER_NORMAL] *= 0.85;
    }
    // If resize is back up, check if projected q index is too much above the
    // current base_qindex, and if so, reduce the rate correction factor
    // (since prefer to keep q for resized frame at least close to previous q).
    if (resize_action < 0 && qindex > 130 * cm->base_qindex / 100) {
      rc->rate_correction_factors[INTER_NORMAL] *= 0.9;
    }
  }
  return resize_action;
}

static void adjust_gf_boost_lag_one_pass_vbr(VP9_COMP *cpi,
                                             uint64_t avg_sad_current) {
  VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  int target;
  int found = 0;
  int found2 = 0;
  int frame;
  int i;
  uint64_t avg_source_sad_lag = avg_sad_current;
  int high_source_sad_lagindex = -1;
  int steady_sad_lagindex = -1;
  uint32_t sad_thresh1 = 70000;
  uint32_t sad_thresh2 = 120000;
  int low_content = 0;
  int high_content = 0;
  double rate_err = 1.0;
  // Get measure of complexity over the future frames, and get the first
  // future frame with high_source_sad/scene-change.
  int tot_frames = (int)vp9_lookahead_depth(cpi->lookahead) - 1;
  for (frame = tot_frames; frame >= 1; --frame) {
    const int lagframe_idx = tot_frames - frame + 1;
    uint64_t reference_sad = rc->avg_source_sad[0];
    for (i = 1; i < lagframe_idx; ++i) {
      if (rc->avg_source_sad[i] > 0)
        reference_sad = (3 * reference_sad + rc->avg_source_sad[i]) >> 2;
    }
    // Detect up-coming scene change.
    if (!found &&
        (rc->avg_source_sad[lagframe_idx] >
             VPXMAX(sad_thresh1, (unsigned int)(reference_sad << 1)) ||
         rc->avg_source_sad[lagframe_idx] >
             VPXMAX(3 * sad_thresh1 >> 2,
                    (unsigned int)(reference_sad << 2)))) {
      high_source_sad_lagindex = lagframe_idx;
      found = 1;
    }
    // Detect change from motion to steady.
    if (!found2 && lagframe_idx > 1 && lagframe_idx < tot_frames &&
        rc->avg_source_sad[lagframe_idx - 1] > (sad_thresh1 >> 2)) {
      found2 = 1;
      for (i = lagframe_idx; i < tot_frames; ++i) {
        if (!(rc->avg_source_sad[i] > 0 &&
              rc->avg_source_sad[i] < (sad_thresh1 >> 2) &&
              rc->avg_source_sad[i] <
                  (rc->avg_source_sad[lagframe_idx - 1] >> 1))) {
          found2 = 0;
          i = tot_frames;
        }
      }
      if (found2) steady_sad_lagindex = lagframe_idx;
    }
    avg_source_sad_lag += rc->avg_source_sad[lagframe_idx];
  }
  if (tot_frames > 0) avg_source_sad_lag = avg_source_sad_lag / tot_frames;
  // Constrain distance between detected scene cuts.
  if (high_source_sad_lagindex != -1 &&
      high_source_sad_lagindex != rc->high_source_sad_lagindex - 1 &&
      abs(high_source_sad_lagindex - rc->high_source_sad_lagindex) < 4)
    rc->high_source_sad_lagindex = -1;
  else
    rc->high_source_sad_lagindex = high_source_sad_lagindex;
  // Adjust some factors for the next GF group, ignore initial key frame,
  // and only for lag_in_frames not too small.
  if (cpi->refresh_golden_frame == 1 && cm->current_video_frame > 30 &&
      cpi->oxcf.lag_in_frames > 8) {
    int frame_constraint;
    if (rc->rolling_target_bits > 0)
      rate_err =
          (double)rc->rolling_actual_bits / (double)rc->rolling_target_bits;
    high_content = high_source_sad_lagindex != -1 ||
                   avg_source_sad_lag > (rc->prev_avg_source_sad_lag << 1) ||
                   avg_source_sad_lag > sad_thresh2;
    low_content = high_source_sad_lagindex == -1 &&
                  ((avg_source_sad_lag < (rc->prev_avg_source_sad_lag >> 1)) ||
                   (avg_source_sad_lag < sad_thresh1));
    if (low_content) {
      rc->gfu_boost = DEFAULT_GF_BOOST;
      rc->baseline_gf_interval =
          VPXMIN(15, (3 * rc->baseline_gf_interval) >> 1);
    } else if (high_content) {
      rc->gfu_boost = DEFAULT_GF_BOOST >> 1;
      rc->baseline_gf_interval = (rate_err > 3.0)
                                     ? VPXMAX(10, rc->baseline_gf_interval >> 1)
                                     : VPXMAX(6, rc->baseline_gf_interval >> 1);
    }
    if (rc->baseline_gf_interval > cpi->oxcf.lag_in_frames - 1)
      rc->baseline_gf_interval = cpi->oxcf.lag_in_frames - 1;
    // Check for constraining gf_interval for up-coming scene/content changes,
    // or for up-coming key frame, whichever is closer.
    frame_constraint = rc->frames_to_key;
    if (rc->high_source_sad_lagindex > 0 &&
        frame_constraint > rc->high_source_sad_lagindex)
      frame_constraint = rc->high_source_sad_lagindex;
    if (steady_sad_lagindex > 3 && frame_constraint > steady_sad_lagindex)
      frame_constraint = steady_sad_lagindex;
    adjust_gfint_frame_constraint(cpi, frame_constraint);
    rc->frames_till_gf_update_due = rc->baseline_gf_interval;
    // Adjust factors for active_worst setting & af_ratio for next gf interval.
    rc->fac_active_worst_inter = 150;  // corresponds to 3/2 (= 150 /100).
    rc->fac_active_worst_gf = 100;
    if (rate_err < 2.0 && !high_content) {
      rc->fac_active_worst_inter = 120;
      rc->fac_active_worst_gf = 90;
    } else if (rate_err > 8.0 && rc->avg_frame_qindex[INTER_FRAME] < 16) {
      // Increase active_worst faster at low Q if rate fluctuation is high.
      rc->fac_active_worst_inter = 200;
      if (rc->avg_frame_qindex[INTER_FRAME] < 8)
        rc->fac_active_worst_inter = 400;
    }
    if (low_content && rc->avg_frame_low_motion > 80) {
      rc->af_ratio_onepass_vbr = 15;
    } else if (high_content || rc->avg_frame_low_motion < 30) {
      rc->af_ratio_onepass_vbr = 5;
      rc->gfu_boost = DEFAULT_GF_BOOST >> 2;
    }
    if (cpi->sf.use_altref_onepass && cpi->oxcf.enable_auto_arf) {
      // Flag to disable usage of ARF based on past usage, only allow this
      // disabling if current frame/group does not start with key frame or
      // scene cut. Note perc_arf_usage is only computed for speed >= 5.
      int arf_usage_low =
          (cm->frame_type != KEY_FRAME && !rc->high_source_sad &&
           cpi->rc.perc_arf_usage < 15 && cpi->oxcf.speed >= 5);
      // Don't use alt-ref for this group under certain conditions.
      if (arf_usage_low ||
          (rc->high_source_sad_lagindex > 0 &&
           rc->high_source_sad_lagindex <= rc->frames_till_gf_update_due) ||
          (avg_source_sad_lag > 3 * sad_thresh1 >> 3)) {
        rc->source_alt_ref_pending = 0;
        rc->alt_ref_gf_group = 0;
      } else {
        rc->source_alt_ref_pending = 1;
        rc->alt_ref_gf_group = 1;
        // If alt-ref is used for this gf group, limit the interval.
        if (rc->baseline_gf_interval > 12) {
          rc->baseline_gf_interval = 12;
          rc->frames_till_gf_update_due = rc->baseline_gf_interval;
        }
      }
    }
    target = vp9_calc_pframe_target_size_one_pass_vbr(cpi);
    vp9_rc_set_frame_target(cpi, target);
  }
  rc->prev_avg_source_sad_lag = avg_source_sad_lag;
}

// Compute average source sad (temporal sad: between current source and
// previous source) over a subset of superblocks. Use this is detect big changes
// in content and allow rate control to react.
// This function also handles special case of lag_in_frames, to measure content
// level in #future frames set by the lag_in_frames.
void vp9_scene_detection_onepass(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  YV12_BUFFER_CONFIG const *unscaled_src = cpi->un_scaled_source;
  YV12_BUFFER_CONFIG const *unscaled_last_src = cpi->unscaled_last_source;
  uint8_t *src_y;
  int src_ystride;
  int src_width;
  int src_height;
  uint8_t *last_src_y;
  int last_src_ystride;
  int last_src_width;
  int last_src_height;
  if (cpi->un_scaled_source == NULL || cpi->unscaled_last_source == NULL ||
      (cpi->use_svc && cpi->svc.current_superframe == 0))
    return;
  src_y = unscaled_src->y_buffer;
  src_ystride = unscaled_src->y_stride;
  src_width = unscaled_src->y_width;
  src_height = unscaled_src->y_height;
  last_src_y = unscaled_last_src->y_buffer;
  last_src_ystride = unscaled_last_src->y_stride;
  last_src_width = unscaled_last_src->y_width;
  last_src_height = unscaled_last_src->y_height;
#if CONFIG_VP9_HIGHBITDEPTH
  if (cm->use_highbitdepth) return;
#endif
  rc->high_source_sad = 0;
  rc->high_num_blocks_with_motion = 0;
  // For SVC: scene detection is only checked on first spatial layer of
  // the superframe using the original/unscaled resolutions.
  if (cpi->svc.spatial_layer_id == cpi->svc.first_spatial_layer_to_encode &&
      src_width == last_src_width && src_height == last_src_height) {
    YV12_BUFFER_CONFIG *frames[MAX_LAG_BUFFERS] = { NULL };
    int num_mi_cols = cm->mi_cols;
    int num_mi_rows = cm->mi_rows;
    int start_frame = 0;
    int frames_to_buffer = 1;
    int frame = 0;
    int scene_cut_force_key_frame = 0;
    int num_zero_temp_sad = 0;
    uint64_t avg_sad_current = 0;
    uint32_t min_thresh = 20000;  // ~5 * 64 * 64
    float thresh = 8.0f;
    uint32_t thresh_key = 140000;
    if (cpi->oxcf.speed <= 5) thresh_key = 240000;
    if (cpi->oxcf.content != VP9E_CONTENT_SCREEN) min_thresh = 65000;
    if (cpi->oxcf.rc_mode == VPX_VBR) thresh = 2.1f;
    if (cpi->use_svc && cpi->svc.number_spatial_layers > 1) {
      const int aligned_width = ALIGN_POWER_OF_TWO(src_width, MI_SIZE_LOG2);
      const int aligned_height = ALIGN_POWER_OF_TWO(src_height, MI_SIZE_LOG2);
      num_mi_cols = aligned_width >> MI_SIZE_LOG2;
      num_mi_rows = aligned_height >> MI_SIZE_LOG2;
    }
    if (cpi->oxcf.lag_in_frames > 0) {
      frames_to_buffer = (cm->current_video_frame == 1)
                             ? (int)vp9_lookahead_depth(cpi->lookahead) - 1
                             : 2;
      start_frame = (int)vp9_lookahead_depth(cpi->lookahead) - 1;
      for (frame = 0; frame < frames_to_buffer; ++frame) {
        const int lagframe_idx = start_frame - frame;
        if (lagframe_idx >= 0) {
          struct lookahead_entry *buf =
              vp9_lookahead_peek(cpi->lookahead, lagframe_idx);
          frames[frame] = &buf->img;
        }
      }
      // The avg_sad for this current frame is the value of frame#1
      // (first future frame) from previous frame.
      avg_sad_current = rc->avg_source_sad[1];
      if (avg_sad_current >
              VPXMAX(min_thresh,
                     (unsigned int)(rc->avg_source_sad[0] * thresh)) &&
          cm->current_video_frame > (unsigned int)cpi->oxcf.lag_in_frames)
        rc->high_source_sad = 1;
      else
        rc->high_source_sad = 0;
      if (rc->high_source_sad && avg_sad_current > thresh_key)
        scene_cut_force_key_frame = 1;
      // Update recursive average for current frame.
      if (avg_sad_current > 0)
        rc->avg_source_sad[0] =
            (3 * rc->avg_source_sad[0] + avg_sad_current) >> 2;
      // Shift back data, starting at frame#1.
      for (frame = 1; frame < cpi->oxcf.lag_in_frames - 1; ++frame)
        rc->avg_source_sad[frame] = rc->avg_source_sad[frame + 1];
    }
    for (frame = 0; frame < frames_to_buffer; ++frame) {
      if (cpi->oxcf.lag_in_frames == 0 ||
          (frames[frame] != NULL && frames[frame + 1] != NULL &&
           frames[frame]->y_width == frames[frame + 1]->y_width &&
           frames[frame]->y_height == frames[frame + 1]->y_height)) {
        int sbi_row, sbi_col;
        const int lagframe_idx =
            (cpi->oxcf.lag_in_frames == 0) ? 0 : start_frame - frame + 1;
        const BLOCK_SIZE bsize = BLOCK_64X64;
        // Loop over sub-sample of frame, compute average sad over 64x64 blocks.
        uint64_t avg_sad = 0;
        uint64_t tmp_sad = 0;
        int num_samples = 0;
        int sb_cols = (num_mi_cols + MI_BLOCK_SIZE - 1) / MI_BLOCK_SIZE;
        int sb_rows = (num_mi_rows + MI_BLOCK_SIZE - 1) / MI_BLOCK_SIZE;
        if (cpi->oxcf.lag_in_frames > 0) {
          src_y = frames[frame]->y_buffer;
          src_ystride = frames[frame]->y_stride;
          last_src_y = frames[frame + 1]->y_buffer;
          last_src_ystride = frames[frame + 1]->y_stride;
        }
        num_zero_temp_sad = 0;
        for (sbi_row = 0; sbi_row < sb_rows; ++sbi_row) {
          for (sbi_col = 0; sbi_col < sb_cols; ++sbi_col) {
            // Checker-board pattern, ignore boundary.
            if (((sbi_row > 0 && sbi_col > 0) &&
                 (sbi_row < sb_rows - 1 && sbi_col < sb_cols - 1) &&
                 ((sbi_row % 2 == 0 && sbi_col % 2 == 0) ||
                  (sbi_row % 2 != 0 && sbi_col % 2 != 0)))) {
              tmp_sad = cpi->fn_ptr[bsize].sdf(src_y, src_ystride, last_src_y,
                                               last_src_ystride);
              avg_sad += tmp_sad;
              num_samples++;
              if (tmp_sad == 0) num_zero_temp_sad++;
            }
            src_y += 64;
            last_src_y += 64;
          }
          src_y += (src_ystride << 6) - (sb_cols << 6);
          last_src_y += (last_src_ystride << 6) - (sb_cols << 6);
        }
        if (num_samples > 0) avg_sad = avg_sad / num_samples;
        // Set high_source_sad flag if we detect very high increase in avg_sad
        // between current and previous frame value(s). Use minimum threshold
        // for cases where there is small change from content that is completely
        // static.
        if (lagframe_idx == 0) {
          if (avg_sad >
                  VPXMAX(min_thresh,
                         (unsigned int)(rc->avg_source_sad[0] * thresh)) &&
              rc->frames_since_key > 1 + cpi->svc.number_spatial_layers &&
              num_zero_temp_sad < 3 * (num_samples >> 2))
            rc->high_source_sad = 1;
          else
            rc->high_source_sad = 0;
          if (rc->high_source_sad && avg_sad > thresh_key)
            scene_cut_force_key_frame = 1;
          if (avg_sad > 0 || cpi->oxcf.rc_mode == VPX_CBR)
            rc->avg_source_sad[0] = (3 * rc->avg_source_sad[0] + avg_sad) >> 2;
        } else {
          rc->avg_source_sad[lagframe_idx] = avg_sad;
        }
        if (num_zero_temp_sad < (3 * num_samples >> 2))
          rc->high_num_blocks_with_motion = 1;
      }
    }
    // For CBR non-screen content mode, check if we should reset the rate
    // control. Reset is done if high_source_sad is detected and the rate
    // control is at very low QP with rate correction factor at min level.
    if (cpi->oxcf.rc_mode == VPX_CBR &&
        cpi->oxcf.content != VP9E_CONTENT_SCREEN && !cpi->use_svc) {
      if (rc->high_source_sad && rc->last_q[INTER_FRAME] == rc->best_quality &&
          rc->avg_frame_qindex[INTER_FRAME] < (rc->best_quality << 1) &&
          rc->rate_correction_factors[INTER_NORMAL] == MIN_BPB_FACTOR) {
        rc->rate_correction_factors[INTER_NORMAL] = 0.5;
        rc->avg_frame_qindex[INTER_FRAME] = rc->worst_quality;
        rc->buffer_level = rc->optimal_buffer_level;
        rc->bits_off_target = rc->optimal_buffer_level;
        rc->reset_high_source_sad = 1;
      }
      if (cm->frame_type != KEY_FRAME && rc->reset_high_source_sad)
        rc->this_frame_target = rc->avg_frame_bandwidth;
    }
    // For SVC the new (updated) avg_source_sad[0] for the current superframe
    // updates the setting for all layers.
    if (cpi->use_svc) {
      int sl, tl;
      SVC *const svc = &cpi->svc;
      for (sl = 0; sl < svc->number_spatial_layers; ++sl)
        for (tl = 0; tl < svc->number_temporal_layers; ++tl) {
          int layer = LAYER_IDS_TO_IDX(sl, tl, svc->number_temporal_layers);
          LAYER_CONTEXT *const lc = &svc->layer_context[layer];
          RATE_CONTROL *const lrc = &lc->rc;
          lrc->avg_source_sad[0] = rc->avg_source_sad[0];
        }
    }
    // For VBR, under scene change/high content change, force golden refresh.
    if (cpi->oxcf.rc_mode == VPX_VBR && cm->frame_type != KEY_FRAME &&
        rc->high_source_sad && rc->frames_to_key > 3 &&
        rc->count_last_scene_change > 4 &&
        cpi->ext_refresh_frame_flags_pending == 0) {
      int target;
      cpi->refresh_golden_frame = 1;
      if (scene_cut_force_key_frame) cm->frame_type = KEY_FRAME;
      rc->source_alt_ref_pending = 0;
      if (cpi->sf.use_altref_onepass && cpi->oxcf.enable_auto_arf)
        rc->source_alt_ref_pending = 1;
      rc->gfu_boost = DEFAULT_GF_BOOST >> 1;
      rc->baseline_gf_interval =
          VPXMIN(20, VPXMAX(10, rc->baseline_gf_interval));
      adjust_gfint_frame_constraint(cpi, rc->frames_to_key);
      rc->frames_till_gf_update_due = rc->baseline_gf_interval;
      target = vp9_calc_pframe_target_size_one_pass_vbr(cpi);
      vp9_rc_set_frame_target(cpi, target);
      rc->count_last_scene_change = 0;
    } else {
      rc->count_last_scene_change++;
    }
    // If lag_in_frame is used, set the gf boost and interval.
    if (cpi->oxcf.lag_in_frames > 0)
      adjust_gf_boost_lag_one_pass_vbr(cpi, avg_sad_current);
  }
}

// Test if encoded frame will significantly overshoot the target bitrate, and
// if so, set the QP, reset/adjust some rate control parameters, and return 1.
// frame_size = -1 means frame has not been encoded.
int vp9_encodedframe_overshoot(VP9_COMP *cpi, int frame_size, int *q) {
  VP9_COMMON *const cm = &cpi->common;
  RATE_CONTROL *const rc = &cpi->rc;
  SPEED_FEATURES *const sf = &cpi->sf;
  int thresh_qp = 7 * (rc->worst_quality >> 3);
  int thresh_rate = rc->avg_frame_bandwidth << 3;
  // Lower thresh_qp for video (more overshoot at lower Q) to be
  // more conservative for video.
  if (cpi->oxcf.content != VP9E_CONTENT_SCREEN)
    thresh_qp = 3 * (rc->worst_quality >> 2);
  // If this decision is not based on an encoded frame size but just on
  // scene/slide change detection (i.e., re_encode_overshoot_cbr_rt ==
  // FAST_DETECTION_MAXQ), for now skip the (frame_size > thresh_rate)
  // condition in this case.
  // TODO(marpan): Use a better size/rate condition for this case and
  // adjust thresholds.
  if ((sf->overshoot_detection_cbr_rt == FAST_DETECTION_MAXQ ||
       frame_size > thresh_rate) &&
      cm->base_qindex < thresh_qp) {
    double rate_correction_factor =
        cpi->rc.rate_correction_factors[INTER_NORMAL];
    const int target_size = cpi->rc.avg_frame_bandwidth;
    const uint64_t sad_thr = 64 * 64 * 32;
    int force_maxqp = 1;
    double new_correction_factor;
    int target_bits_per_mb;
    double q2;
    int enumerator;
    // Set a larger QP.
    if (cpi->oxcf.content != VP9E_CONTENT_SCREEN &&
        (rc->buffer_level > (3 * rc->optimal_buffer_level) >> 2) &&
        (cpi->rc.avg_source_sad[0] < sad_thr)) {
      *q = (*q + cpi->rc.worst_quality) >> 1;
      force_maxqp = 0;
    } else {
      *q = cpi->rc.worst_quality;
    }
    cpi->cyclic_refresh->counter_encode_maxq_scene_change = 0;
    cpi->rc.re_encode_maxq_scene_change = 1;
    // If the frame_size is much larger than the threshold (big content change)
    // and the encoded frame used alot of Intra modes, then force hybrid_intra
    // encoding for the re-encode on this scene change. hybrid_intra will
    // use rd-based intra mode selection for small blocks.
    if (sf->overshoot_detection_cbr_rt == RE_ENCODE_MAXQ &&
        frame_size > (thresh_rate << 1) && cpi->svc.spatial_layer_id == 0) {
      MODE_INFO **mi = cm->mi_grid_visible;
      int sum_intra_usage = 0;
      int mi_row, mi_col;
      for (mi_row = 0; mi_row < cm->mi_rows; mi_row++) {
        for (mi_col = 0; mi_col < cm->mi_cols; mi_col++) {
          if (mi[0]->ref_frame[0] == INTRA_FRAME) sum_intra_usage++;
          mi++;
        }
        mi += 8;
      }
      sum_intra_usage = 100 * sum_intra_usage / (cm->mi_rows * cm->mi_cols);
      if (sum_intra_usage > 60) cpi->rc.hybrid_intra_scene_change = 1;
    }
    // Adjust avg_frame_qindex, buffer_level, and rate correction factors, as
    // these parameters will affect QP selection for subsequent frames. If they
    // have settled down to a very different (low QP) state, then not adjusting
    // them may cause next frame to select low QP and overshoot again.
    cpi->rc.avg_frame_qindex[INTER_FRAME] = *q;
    rc->buffer_level = rc->optimal_buffer_level;
    rc->bits_off_target = rc->optimal_buffer_level;
    // Reset rate under/over-shoot flags.
    cpi->rc.rc_1_frame = 0;
    cpi->rc.rc_2_frame = 0;
    // Adjust rate correction factor.
    target_bits_per_mb =
        (int)(((uint64_t)target_size << BPER_MB_NORMBITS) / cm->MBs);
    // Rate correction factor based on target_bits_per_mb and qp (==max_QP).
    // This comes from the inverse computation of vp9_rc_bits_per_mb().
    q2 = vp9_convert_qindex_to_q(*q, cm->bit_depth);
    enumerator = 1800000;  // Factor for inter frame.
    enumerator += (int)(enumerator * q2) >> 12;
    new_correction_factor = (double)target_bits_per_mb * q2 / enumerator;
    if (new_correction_factor > rate_correction_factor) {
      rate_correction_factor =
          VPXMIN(2.0 * rate_correction_factor, new_correction_factor);
      if (rate_correction_factor > MAX_BPB_FACTOR)
        rate_correction_factor = MAX_BPB_FACTOR;
      cpi->rc.rate_correction_factors[INTER_NORMAL] = rate_correction_factor;
    }
    // For temporal layers, reset the rate control parametes across all
    // temporal layers.
    // If the first_spatial_layer_to_encode > 0, then this superframe has
    // skipped lower base layers. So in this case we should also reset and
    // force max-q for spatial layers < first_spatial_layer_to_encode.
    // For the case of no inter-layer prediction on delta frames: reset and
    // force max-q for all spatial layers, to avoid excessive frame drops.
    if (cpi->use_svc) {
      int tl = 0;
      int sl = 0;
      SVC *svc = &cpi->svc;
      int num_spatial_layers = VPXMAX(1, svc->first_spatial_layer_to_encode);
      if (svc->disable_inter_layer_pred != INTER_LAYER_PRED_ON)
        num_spatial_layers = svc->number_spatial_layers;
      for (sl = 0; sl < num_spatial_layers; ++sl) {
        for (tl = 0; tl < svc->number_temporal_layers; ++tl) {
          const int layer =
              LAYER_IDS_TO_IDX(sl, tl, svc->number_temporal_layers);
          LAYER_CONTEXT *lc = &svc->layer_context[layer];
          RATE_CONTROL *lrc = &lc->rc;
          lrc->avg_frame_qindex[INTER_FRAME] = *q;
          lrc->buffer_level = lrc->optimal_buffer_level;
          lrc->bits_off_target = lrc->optimal_buffer_level;
          lrc->rc_1_frame = 0;
          lrc->rc_2_frame = 0;
          lrc->rate_correction_factors[INTER_NORMAL] = rate_correction_factor;
          lrc->force_max_q = force_maxqp;
        }
      }
    }
    return 1;
  } else {
    return 0;
  }
}
