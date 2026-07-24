/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "./vpx_config.h"
#include "vpx/vpx_encoder.h"
#include "vpx/vpx_ext_ratectrl.h"
#include "vpx_dsp/psnr.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/static_assert.h"
#include "vpx_ports/system_state.h"
#include "vpx_util/vpx_timestamp.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "./vpx_version.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_ethread.h"
#include "vpx/vp8cx.h"
#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_scale.h"
#include "vp9/vp9_cx_iface.h"
#include "vp9/encoder/vp9_firstpass.h"
#include "vp9/encoder/vp9_lookahead.h"
#include "vp9/vp9_cx_iface.h"
#include "vp9/vp9_iface_common.h"

#include "vpx/vpx_tpl.h"

typedef struct vp9_extracfg {
  int cpu_used;  // available cpu percentage in 1/16
  unsigned int enable_auto_alt_ref;
  unsigned int noise_sensitivity;
  unsigned int sharpness;
  unsigned int static_thresh;
  unsigned int tile_columns;
  unsigned int tile_rows;
  unsigned int enable_tpl_model;
  unsigned int enable_keyframe_filtering;
  unsigned int arnr_max_frames;
  unsigned int arnr_strength;
  unsigned int min_gf_interval;
  unsigned int max_gf_interval;
  vp8e_tuning tuning;
  unsigned int cq_level;  // constrained quality level
  unsigned int rc_max_intra_bitrate_pct;
  unsigned int rc_max_inter_bitrate_pct;
  unsigned int gf_cbr_boost_pct;
  unsigned int lossless;
  unsigned int target_level;
  unsigned int frame_parallel_decoding_mode;
  AQ_MODE aq_mode;
  int alt_ref_aq;
  unsigned int frame_periodic_boost;
  vpx_bit_depth_t bit_depth;
  vp9e_tune_content content;
  vpx_color_space_t color_space;
  vpx_color_range_t color_range;
  int render_width;
  int render_height;
  unsigned int row_mt;
  unsigned int motion_vector_unit_test;
  int delta_q_uv;
} vp9_extracfg;

static struct vp9_extracfg default_extra_cfg = {
#if CONFIG_REALTIME_ONLY
  5,  // cpu_used
#else
  0,  // cpu_used
#endif
  1,                     // enable_auto_alt_ref
  0,                     // noise_sensitivity
  0,                     // sharpness
  0,                     // static_thresh
  6,                     // tile_columns
  0,                     // tile_rows
  1,                     // enable_tpl_model
  0,                     // enable_keyframe_filtering
  7,                     // arnr_max_frames
  5,                     // arnr_strength
  0,                     // min_gf_interval; 0 -> default decision
  0,                     // max_gf_interval; 0 -> default decision
  VP8_TUNE_PSNR,         // tuning
  10,                    // cq_level
  0,                     // rc_max_intra_bitrate_pct
  0,                     // rc_max_inter_bitrate_pct
  0,                     // gf_cbr_boost_pct
  0,                     // lossless
  255,                   // target_level
  1,                     // frame_parallel_decoding_mode
  NO_AQ,                 // aq_mode
  0,                     // alt_ref_aq
  0,                     // frame_periodic_delta_q
  VPX_BITS_8,            // Bit depth
  VP9E_CONTENT_DEFAULT,  // content
  VPX_CS_UNKNOWN,        // color space
  0,                     // color range
  0,                     // render width
  0,                     // render height
  0,                     // row_mt
  0,                     // motion_vector_unit_test
  0,                     // delta_q_uv
};

struct vpx_codec_alg_priv {
  vpx_codec_priv_t base;
  vpx_codec_enc_cfg_t cfg;
  struct vp9_extracfg extra_cfg;
  vpx_codec_pts_t pts_offset;
  unsigned char pts_offset_initialized;
  VP9EncoderConfig oxcf;
  VP9_COMP *cpi;
  unsigned char *cx_data;
  size_t cx_data_sz;
  unsigned char *pending_cx_data;
  size_t pending_cx_data_sz;
  int pending_frame_count;
  size_t pending_frame_sizes[8];
  size_t pending_frame_magnitude;
  vpx_image_t preview_img;
  vpx_enc_frame_flags_t next_frame_flags;
  vp8_postproc_cfg_t preview_ppcfg;
  vpx_codec_pkt_list_decl(256) pkt_list;
  unsigned int fixed_kf_cntr;
  vpx_codec_priv_output_cx_pkt_cb_pair_t output_cx_pkt_cb;
  // BufferPool that holds all reference frames.
  BufferPool *buffer_pool;
  vpx_fixed_buf_t global_headers;
  int global_header_subsampling;
};

// Called by encoder_set_config() and encoder_encode() only. Must not be called
// by encoder_init() because the `error` paramerer (cpi->common.error) will be
// destroyed by vpx_codec_enc_init_ver() after encoder_init() returns an error.
// See the "IMPORTANT" comment in vpx_codec_enc_init_ver().
static vpx_codec_err_t update_error_state(
    vpx_codec_alg_priv_t *ctx, const struct vpx_internal_error_info *error) {
  const vpx_codec_err_t res = error->error_code;

  if (res != VPX_CODEC_OK)
    ctx->base.err_detail = error->has_detail ? error->detail : NULL;

  return res;
}

#undef ERROR
#define ERROR(str)                  \
  do {                              \
    ctx->base.err_detail = str;     \
    return VPX_CODEC_INVALID_PARAM; \
  } while (0)

#define RANGE_CHECK(p, memb, lo, hi)                                     \
  do {                                                                   \
    if (!(((p)->memb == (lo) || (p)->memb > (lo)) && (p)->memb <= (hi))) \
      ERROR(#memb " out of range [" #lo ".." #hi "]");                   \
  } while (0)

#define RANGE_CHECK_HI(p, memb, hi)                                     \
  do {                                                                  \
    if (!((p)->memb <= (hi))) ERROR(#memb " out of range [.." #hi "]"); \
  } while (0)

#define RANGE_CHECK_LO(p, memb, lo)                                     \
  do {                                                                  \
    if (!((p)->memb >= (lo))) ERROR(#memb " out of range [" #lo "..]"); \
  } while (0)

#define RANGE_CHECK_BOOL(p, memb)                                     \
  do {                                                                \
    if (!!((p)->memb) != (p)->memb) ERROR(#memb " expected boolean"); \
  } while (0)

static vpx_codec_err_t validate_config(vpx_codec_alg_priv_t *ctx,
                                       const vpx_codec_enc_cfg_t *cfg,
                                       const struct vp9_extracfg *extra_cfg) {
  RANGE_CHECK(cfg, g_w, 1, 65536);  // 16 bits available
  RANGE_CHECK(cfg, g_h, 1, 65536);  // 16 bits available
  RANGE_CHECK(cfg, g_timebase.den, 1, 1000000000);
  RANGE_CHECK(cfg, g_timebase.num, 1, 1000000000);
  RANGE_CHECK_HI(cfg, g_profile, 3);

  RANGE_CHECK_HI(cfg, rc_max_quantizer, 63);
  RANGE_CHECK_HI(cfg, rc_min_quantizer, cfg->rc_max_quantizer);
  RANGE_CHECK_BOOL(extra_cfg, lossless);
  RANGE_CHECK_BOOL(extra_cfg, frame_parallel_decoding_mode);
  RANGE_CHECK(extra_cfg, aq_mode, 0, AQ_MODE_COUNT - 2);
  RANGE_CHECK(extra_cfg, alt_ref_aq, 0, 1);
  RANGE_CHECK(extra_cfg, frame_periodic_boost, 0, 1);
  RANGE_CHECK_HI(cfg, g_threads, MAX_NUM_THREADS);
  RANGE_CHECK_HI(cfg, g_lag_in_frames, MAX_LAG_BUFFERS);
  RANGE_CHECK(cfg, rc_end_usage, VPX_VBR, VPX_Q);
  RANGE_CHECK_HI(cfg, rc_undershoot_pct, 100);
  RANGE_CHECK_HI(cfg, rc_overshoot_pct, 100);
  RANGE_CHECK_HI(cfg, rc_2pass_vbr_bias_pct, 100);
  RANGE_CHECK(cfg, rc_2pass_vbr_corpus_complexity, 0, 10000);
  RANGE_CHECK(cfg, kf_mode, VPX_KF_DISABLED, VPX_KF_AUTO);
  RANGE_CHECK_BOOL(cfg, rc_resize_allowed);
  RANGE_CHECK_HI(cfg, rc_dropframe_thresh, 100);
  RANGE_CHECK_HI(cfg, rc_resize_up_thresh, 100);
  RANGE_CHECK_HI(cfg, rc_resize_down_thresh, 100);
#if CONFIG_REALTIME_ONLY
  RANGE_CHECK(cfg, g_pass, VPX_RC_ONE_PASS, VPX_RC_ONE_PASS);
#else
  RANGE_CHECK(cfg, g_pass, VPX_RC_ONE_PASS, VPX_RC_LAST_PASS);
#endif
  RANGE_CHECK(extra_cfg, min_gf_interval, 0, (MAX_LAG_BUFFERS - 1));
  RANGE_CHECK(extra_cfg, max_gf_interval, 0, (MAX_LAG_BUFFERS - 1));
  if (extra_cfg->max_gf_interval > 0) {
    RANGE_CHECK(extra_cfg, max_gf_interval, 2, (MAX_LAG_BUFFERS - 1));
  }
  if (extra_cfg->min_gf_interval > 0 && extra_cfg->max_gf_interval > 0) {
    RANGE_CHECK(extra_cfg, max_gf_interval, extra_cfg->min_gf_interval,
                (MAX_LAG_BUFFERS - 1));
  }

  // For formation of valid ARF groups lag_in _frames should be 0 or greater
  // than the max_gf_interval + 2
  if (cfg->g_lag_in_frames > 0 && extra_cfg->max_gf_interval > 0 &&
      cfg->g_lag_in_frames < extra_cfg->max_gf_interval + 2) {
    ERROR("Set lag in frames to 0 (low delay) or >= (max-gf-interval + 2)");
  }

  if (cfg->rc_resize_allowed == 1) {
    RANGE_CHECK(cfg, rc_scaled_width, 0, cfg->g_w);
    RANGE_CHECK(cfg, rc_scaled_height, 0, cfg->g_h);
  }

  RANGE_CHECK(cfg, ss_number_layers, 1, VPX_SS_MAX_LAYERS);
  RANGE_CHECK(cfg, ts_number_layers, 1, VPX_TS_MAX_LAYERS);

  {
    unsigned int level = extra_cfg->target_level;
    if (level != LEVEL_1 && level != LEVEL_1_1 && level != LEVEL_2 &&
        level != LEVEL_2_1 && level != LEVEL_3 && level != LEVEL_3_1 &&
        level != LEVEL_4 && level != LEVEL_4_1 && level != LEVEL_5 &&
        level != LEVEL_5_1 && level != LEVEL_5_2 && level != LEVEL_6 &&
        level != LEVEL_6_1 && level != LEVEL_6_2 && level != LEVEL_UNKNOWN &&
        level != LEVEL_AUTO && level != LEVEL_MAX)
      ERROR("target_level is invalid");
  }

  if (cfg->ss_number_layers * cfg->ts_number_layers > VPX_MAX_LAYERS)
    ERROR("ss_number_layers * ts_number_layers is out of range");
  if (cfg->ts_number_layers > 1) {
    unsigned int sl, tl;
    for (sl = 1; sl < cfg->ss_number_layers; ++sl) {
      for (tl = 1; tl < cfg->ts_number_layers; ++tl) {
        const int layer = LAYER_IDS_TO_IDX(sl, tl, cfg->ts_number_layers);
        if (cfg->layer_target_bitrate[layer] <
            cfg->layer_target_bitrate[layer - 1])
          ERROR("ts_target_bitrate entries are not increasing");
      }
    }

    RANGE_CHECK(cfg, ts_rate_decimator[cfg->ts_number_layers - 1], 1, 1);
    for (tl = cfg->ts_number_layers - 2; tl > 0; --tl)
      if (cfg->ts_rate_decimator[tl - 1] != 2 * cfg->ts_rate_decimator[tl])
        ERROR("ts_rate_decimator factors are not powers of 2");
  }

  // VP9 does not support a lower bound on the keyframe interval in
  // automatic keyframe placement mode.
  if (cfg->kf_mode != VPX_KF_DISABLED && cfg->kf_min_dist != cfg->kf_max_dist &&
      cfg->kf_min_dist > 0)
    ERROR(
        "kf_min_dist not supported in auto mode, use 0 "
        "or kf_max_dist instead.");

  RANGE_CHECK(extra_cfg, row_mt, 0, 1);
  RANGE_CHECK(extra_cfg, motion_vector_unit_test, 0, 2);
  RANGE_CHECK(extra_cfg, enable_auto_alt_ref, 0, MAX_ARF_LAYERS);
  RANGE_CHECK(extra_cfg, cpu_used, -9, 9);
  RANGE_CHECK_HI(extra_cfg, noise_sensitivity, 6);
  RANGE_CHECK(extra_cfg, tile_columns, 0, 6);
  RANGE_CHECK(extra_cfg, tile_rows, 0, 2);
  RANGE_CHECK_HI(extra_cfg, sharpness, 7);
  RANGE_CHECK(extra_cfg, arnr_max_frames, 0, 15);
  RANGE_CHECK_HI(extra_cfg, arnr_strength, 6);
  RANGE_CHECK(extra_cfg, cq_level, 0, 63);
  RANGE_CHECK(cfg, g_bit_depth, VPX_BITS_8, VPX_BITS_12);
  RANGE_CHECK(cfg, g_input_bit_depth, 8, 12);
  RANGE_CHECK(extra_cfg, content, VP9E_CONTENT_DEFAULT,
              VP9E_CONTENT_INVALID - 1);

#if !CONFIG_REALTIME_ONLY
  if (cfg->g_pass == VPX_RC_LAST_PASS) {
    const size_t packet_sz = sizeof(FIRSTPASS_STATS);
    const int n_packets = (int)(cfg->rc_twopass_stats_in.sz / packet_sz);
    const FIRSTPASS_STATS *stats;

    if (cfg->rc_twopass_stats_in.buf == NULL)
      ERROR("rc_twopass_stats_in.buf not set.");

    if (cfg->rc_twopass_stats_in.sz % packet_sz)
      ERROR("rc_twopass_stats_in.sz indicates truncated packet.");

    if (cfg->ss_number_layers > 1 || cfg->ts_number_layers > 1) {
      int i;
      unsigned int n_packets_per_layer[VPX_SS_MAX_LAYERS] = { 0 };

      stats = cfg->rc_twopass_stats_in.buf;
      for (i = 0; i < n_packets; ++i) {
        const int layer_id = (int)stats[i].spatial_layer_id;
        if (layer_id >= 0 && layer_id < (int)cfg->ss_number_layers) {
          ++n_packets_per_layer[layer_id];
        }
      }

      for (i = 0; i < (int)cfg->ss_number_layers; ++i) {
        unsigned int layer_id;
        if (n_packets_per_layer[i] < 2) {
          ERROR(
              "rc_twopass_stats_in requires at least two packets for each "
              "layer.");
        }

        stats = (const FIRSTPASS_STATS *)cfg->rc_twopass_stats_in.buf +
                n_packets - cfg->ss_number_layers + i;
        layer_id = (int)stats->spatial_layer_id;

        if (layer_id >= cfg->ss_number_layers ||
            (unsigned int)(stats->count + 0.5) !=
                n_packets_per_layer[layer_id] - 1)
          ERROR("rc_twopass_stats_in missing EOS stats packet");
      }
    } else {
      if (cfg->rc_twopass_stats_in.sz < 2 * packet_sz)
        ERROR("rc_twopass_stats_in requires at least two packets.");

      stats =
          (const FIRSTPASS_STATS *)cfg->rc_twopass_stats_in.buf + n_packets - 1;

      if ((int)(stats->count + 0.5) != n_packets - 1)
        ERROR("rc_twopass_stats_in missing EOS stats packet");
    }
  }
#endif  // !CONFIG_REALTIME_ONLY

#if !CONFIG_VP9_HIGHBITDEPTH
  if (cfg->g_profile > (unsigned int)PROFILE_1) {
    ERROR("Profile > 1 not supported in this build configuration");
  }
#endif
  if (cfg->g_profile <= (unsigned int)PROFILE_1 &&
      cfg->g_bit_depth > VPX_BITS_8) {
    ERROR("Codec high bit-depth not supported in profile < 2");
  }
  if (cfg->g_profile <= (unsigned int)PROFILE_1 && cfg->g_input_bit_depth > 8) {
    ERROR("Source high bit-depth not supported in profile < 2");
  }
  if (cfg->g_profile > (unsigned int)PROFILE_1 &&
      cfg->g_bit_depth == VPX_BITS_8) {
    ERROR("Codec bit-depth 8 not supported in profile > 1");
  }
  RANGE_CHECK(extra_cfg, color_space, VPX_CS_UNKNOWN, VPX_CS_SRGB);
  RANGE_CHECK(extra_cfg, color_range, VPX_CR_STUDIO_RANGE, VPX_CR_FULL_RANGE);

  // The range below shall be further tuned.
  RANGE_CHECK(cfg, use_vizier_rc_params, 0, 1);
  RANGE_CHECK(cfg, active_wq_factor.den, 1, 1000);
  RANGE_CHECK(cfg, err_per_mb_factor.den, 1, 1000);
  RANGE_CHECK(cfg, sr_default_decay_limit.den, 1, 1000);
  RANGE_CHECK(cfg, sr_diff_factor.den, 1, 1000);
  RANGE_CHECK(cfg, kf_err_per_mb_factor.den, 1, 1000);
  RANGE_CHECK(cfg, kf_frame_min_boost_factor.den, 1, 1000);
  RANGE_CHECK(cfg, kf_frame_max_boost_subs_factor.den, 1, 1000);
  RANGE_CHECK(cfg, kf_max_total_boost_factor.den, 1, 1000);
  RANGE_CHECK(cfg, gf_max_total_boost_factor.den, 1, 1000);
  RANGE_CHECK(cfg, gf_frame_max_boost_factor.den, 1, 1000);
  RANGE_CHECK(cfg, zm_factor.den, 1, 1000);
  RANGE_CHECK(cfg, rd_mult_inter_qp_fac.den, 1, 1000);
  RANGE_CHECK(cfg, rd_mult_arf_qp_fac.den, 1, 1000);
  RANGE_CHECK(cfg, rd_mult_key_qp_fac.den, 1, 1000);

  return VPX_CODEC_OK;
}

static vpx_codec_err_t validate_img(vpx_codec_alg_priv_t *ctx,
                                    const vpx_image_t *img) {
  switch (img->fmt) {
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_I420:
    case VPX_IMG_FMT_I42016:
    case VPX_IMG_FMT_NV12: break;
    case VPX_IMG_FMT_I422:
    case VPX_IMG_FMT_I444:
    case VPX_IMG_FMT_I440:
      if (ctx->cfg.g_profile != (unsigned int)PROFILE_1) {
        ERROR(
            "Invalid image format. I422, I444, I440 images are not supported "
            "in profile.");
      }
      break;
    case VPX_IMG_FMT_I42216:
    case VPX_IMG_FMT_I44416:
    case VPX_IMG_FMT_I44016:
      if (ctx->cfg.g_profile != (unsigned int)PROFILE_1 &&
          ctx->cfg.g_profile != (unsigned int)PROFILE_3) {
        ERROR(
            "Invalid image format. 16-bit I422, I444, I440 images are "
            "not supported in profile.");
      }
      break;
    default:
      ERROR(
          "Invalid image format. Only YV12, I420, I422, I444, I440, NV12 "
          "images are supported.");
      break;
  }

  if (img->d_w != ctx->cfg.g_w || img->d_h != ctx->cfg.g_h)
    ERROR("Image size must match encoder init configuration size");

  return VPX_CODEC_OK;
}

static int get_image_bps(const vpx_image_t *img) {
  switch (img->fmt) {
    case VPX_IMG_FMT_YV12:
    case VPX_IMG_FMT_NV12:
    case VPX_IMG_FMT_I420: return 12;
    case VPX_IMG_FMT_I422: return 16;
    case VPX_IMG_FMT_I444: return 24;
    case VPX_IMG_FMT_I440: return 16;
    case VPX_IMG_FMT_I42016: return 24;
    case VPX_IMG_FMT_I42216: return 32;
    case VPX_IMG_FMT_I44416: return 48;
    case VPX_IMG_FMT_I44016: return 32;
    default: assert(0 && "Invalid image format"); break;
  }
  return 0;
}

// Modify the encoder config for the target level.
static void config_target_level(VP9EncoderConfig *oxcf) {
  double max_average_bitrate;  // in bits per second
  int max_over_shoot_pct;
  const int target_level_index = get_level_index(oxcf->target_level);

  vpx_clear_system_state();
  assert(target_level_index >= 0);
  assert(target_level_index < VP9_LEVELS);

  // Maximum target bit-rate is level_limit * 80%.
  max_average_bitrate =
      vp9_level_defs[target_level_index].average_bitrate * 800.0;
  if ((double)oxcf->target_bandwidth > max_average_bitrate)
    oxcf->target_bandwidth = (int64_t)(max_average_bitrate);
  if (oxcf->ss_number_layers == 1 && oxcf->pass != 0)
    oxcf->ss_target_bitrate[0] = (int)oxcf->target_bandwidth;

  // Adjust max over-shoot percentage.
  max_over_shoot_pct =
      (int)((max_average_bitrate * 1.10 - (double)oxcf->target_bandwidth) *
            100 / (double)(oxcf->target_bandwidth));
  if (oxcf->over_shoot_pct > max_over_shoot_pct)
    oxcf->over_shoot_pct = max_over_shoot_pct;

  // Adjust worst allowed quantizer.
  oxcf->worst_allowed_q = vp9_quantizer_to_qindex(63);

  // Adjust minimum art-ref distance.
  // min_gf_interval should be no less than min_altref_distance + 1,
  // as the encoder may produce bitstream with alt-ref distance being
  // min_gf_interval - 1.
  if (oxcf->min_gf_interval <=
      (int)vp9_level_defs[target_level_index].min_altref_distance) {
    oxcf->min_gf_interval =
        (int)vp9_level_defs[target_level_index].min_altref_distance + 1;
    // If oxcf->max_gf_interval == 0, it will be assigned with a default value
    // in vp9_rc_set_gf_interval_range().
    if (oxcf->max_gf_interval != 0) {
      oxcf->max_gf_interval =
          VPXMAX(oxcf->max_gf_interval, oxcf->min_gf_interval);
    }
  }

  // Adjust maximum column tiles.
  if (vp9_level_defs[target_level_index].max_col_tiles <
      (1 << oxcf->tile_columns)) {
    while (oxcf->tile_columns > 0 &&
           vp9_level_defs[target_level_index].max_col_tiles <
               (1 << oxcf->tile_columns))
      --oxcf->tile_columns;
  }
}

static vpx_rational64_t get_g_timebase_in_ts(vpx_rational_t g_timebase) {
  vpx_rational64_t g_timebase_in_ts;
  g_timebase_in_ts.den = g_timebase.den;
  g_timebase_in_ts.num = g_timebase.num;
  g_timebase_in_ts.num *= TICKS_PER_SEC;
  reduce_ratio(&g_timebase_in_ts);
  return g_timebase_in_ts;
}

static vpx_codec_err_t set_encoder_config(
    VP9EncoderConfig *oxcf, vpx_codec_enc_cfg_t *cfg,
    const struct vp9_extracfg *extra_cfg) {
  int sl, tl;
  unsigned int raw_target_rate;
  oxcf->profile = cfg->g_profile;
  oxcf->max_threads = (int)cfg->g_threads;
  oxcf->width = cfg->g_w;
  oxcf->height = cfg->g_h;
  oxcf->bit_depth = cfg->g_bit_depth;
  oxcf->input_bit_depth = cfg->g_input_bit_depth;
  // TODO(angiebird): Figure out if we can just use g_timebase to indicate the
  // inverse of framerate
  // guess a frame rate if out of whack, use 30
  oxcf->init_framerate = (double)cfg->g_timebase.den / cfg->g_timebase.num;
  if (oxcf->init_framerate > 180) oxcf->init_framerate = 30;
  oxcf->g_timebase = cfg->g_timebase;
  oxcf->g_timebase_in_ts = get_g_timebase_in_ts(oxcf->g_timebase);

  oxcf->mode = GOOD;

  switch (cfg->g_pass) {
    case VPX_RC_ONE_PASS: oxcf->pass = 0; break;
    case VPX_RC_FIRST_PASS: oxcf->pass = 1; break;
    case VPX_RC_LAST_PASS: oxcf->pass = 2; break;
  }

  oxcf->lag_in_frames =
      cfg->g_pass == VPX_RC_FIRST_PASS ? 0 : cfg->g_lag_in_frames;
  oxcf->rc_mode = cfg->rc_end_usage;

  raw_target_rate =
      (unsigned int)((int64_t)oxcf->width * oxcf->height * oxcf->bit_depth * 3 *
                     oxcf->init_framerate / 1000);
  // Cap target bitrate to raw rate or 1000Mbps, whichever is less
  cfg->rc_target_bitrate =
      VPXMIN(VPXMIN(raw_target_rate, cfg->rc_target_bitrate), 1000000);

  // Convert target bandwidth from Kbit/s to Bit/s
  oxcf->target_bandwidth = 1000 * (int64_t)cfg->rc_target_bitrate;
  oxcf->rc_max_intra_bitrate_pct = extra_cfg->rc_max_intra_bitrate_pct;
  oxcf->rc_max_inter_bitrate_pct = extra_cfg->rc_max_inter_bitrate_pct;
  oxcf->gf_cbr_boost_pct = extra_cfg->gf_cbr_boost_pct;

  oxcf->best_allowed_q =
      extra_cfg->lossless ? 0 : vp9_quantizer_to_qindex(cfg->rc_min_quantizer);
  oxcf->worst_allowed_q =
      extra_cfg->lossless ? 0 : vp9_quantizer_to_qindex(cfg->rc_max_quantizer);
  oxcf->cq_level = vp9_quantizer_to_qindex(extra_cfg->cq_level);
  oxcf->fixed_q = -1;

  oxcf->under_shoot_pct = cfg->rc_undershoot_pct;
  oxcf->over_shoot_pct = cfg->rc_overshoot_pct;

  oxcf->scaled_frame_width = cfg->rc_scaled_width;
  oxcf->scaled_frame_height = cfg->rc_scaled_height;
  if (cfg->rc_resize_allowed == 1) {
    oxcf->resize_mode =
        (oxcf->scaled_frame_width == 0 || oxcf->scaled_frame_height == 0)
            ? RESIZE_DYNAMIC
            : RESIZE_FIXED;
  } else {
    oxcf->resize_mode = RESIZE_NONE;
  }

  oxcf->maximum_buffer_size_ms = cfg->rc_buf_sz;
  oxcf->starting_buffer_level_ms = cfg->rc_buf_initial_sz;
  oxcf->optimal_buffer_level_ms = cfg->rc_buf_optimal_sz;

  oxcf->drop_frames_water_mark = cfg->rc_dropframe_thresh;

  oxcf->two_pass_vbrbias = cfg->rc_2pass_vbr_bias_pct;
  oxcf->two_pass_vbrmin_section = cfg->rc_2pass_vbr_minsection_pct;
  oxcf->two_pass_vbrmax_section = cfg->rc_2pass_vbr_maxsection_pct;
  oxcf->vbr_corpus_complexity = cfg->rc_2pass_vbr_corpus_complexity;

  oxcf->auto_key =
      cfg->kf_mode == VPX_KF_AUTO && cfg->kf_min_dist != cfg->kf_max_dist;

  oxcf->key_freq = cfg->kf_max_dist;

  oxcf->speed = abs(extra_cfg->cpu_used);
  oxcf->encode_breakout = extra_cfg->static_thresh;
  oxcf->enable_auto_arf = extra_cfg->enable_auto_alt_ref;
  if (oxcf->bit_depth == VPX_BITS_8) {
    oxcf->noise_sensitivity = extra_cfg->noise_sensitivity;
  } else {
    // Disable denoiser for high bitdepth since vp9_denoiser_filter only works
    // for 8 bits.
    oxcf->noise_sensitivity = 0;
  }
  oxcf->sharpness = extra_cfg->sharpness;

  vp9_set_first_pass_stats(oxcf, &cfg->rc_twopass_stats_in);

  oxcf->color_space = extra_cfg->color_space;
  oxcf->color_range = extra_cfg->color_range;
  oxcf->render_width = extra_cfg->render_width;
  oxcf->render_height = extra_cfg->render_height;
  oxcf->arnr_max_frames = extra_cfg->arnr_max_frames;
  oxcf->arnr_strength = extra_cfg->arnr_strength;
  oxcf->min_gf_interval = extra_cfg->min_gf_interval;
  oxcf->max_gf_interval = extra_cfg->max_gf_interval;

  oxcf->tuning = extra_cfg->tuning;
  oxcf->content = extra_cfg->content;

  oxcf->tile_columns = extra_cfg->tile_columns;

  oxcf->enable_tpl_model = extra_cfg->enable_tpl_model;

  oxcf->enable_keyframe_filtering = extra_cfg->enable_keyframe_filtering;

  // TODO(yunqing): The dependencies between row tiles cause error in multi-
  // threaded encoding. For now, tile_rows is forced to be 0 in this case.
  // The further fix can be done by adding synchronizations after a tile row
  // is encoded. But this will hurt multi-threaded encoder performance. So,
  // it is recommended to use tile-rows=0 while encoding with threads > 1.
  if (oxcf->max_threads > 1 && oxcf->tile_columns > 0)
    oxcf->tile_rows = 0;
  else
    oxcf->tile_rows = extra_cfg->tile_rows;

  oxcf->error_resilient_mode = cfg->g_error_resilient;
  oxcf->frame_parallel_decoding_mode = extra_cfg->frame_parallel_decoding_mode;

  oxcf->aq_mode = extra_cfg->aq_mode;
  oxcf->alt_ref_aq = extra_cfg->alt_ref_aq;

  oxcf->frame_periodic_boost = extra_cfg->frame_periodic_boost;

  oxcf->ss_number_layers = cfg->ss_number_layers;
  oxcf->ts_number_layers = cfg->ts_number_layers;
  oxcf->temporal_layering_mode =
      (enum vp9e_temporal_layering_mode)cfg->temporal_layering_mode;

  oxcf->target_level = extra_cfg->target_level;

  oxcf->row_mt = extra_cfg->row_mt;
  oxcf->motion_vector_unit_test = extra_cfg->motion_vector_unit_test;

  oxcf->delta_q_uv = extra_cfg->delta_q_uv;

  for (sl = 0; sl < oxcf->ss_number_layers; ++sl) {
    for (tl = 0; tl < oxcf->ts_number_layers; ++tl) {
      const int layer = sl * oxcf->ts_number_layers + tl;
      if (cfg->layer_target_bitrate[layer] > INT_MAX / 1000)
        oxcf->layer_target_bitrate[layer] = INT_MAX;
      else
        oxcf->layer_target_bitrate[layer] =
            1000 * cfg->layer_target_bitrate[layer];
    }
  }
  if (oxcf->ss_number_layers == 1 && oxcf->pass != 0) {
    oxcf->ss_target_bitrate[0] = (int)oxcf->target_bandwidth;
  }
  if (oxcf->ts_number_layers > 1) {
    for (tl = 0; tl < VPX_TS_MAX_LAYERS; ++tl) {
      oxcf->ts_rate_decimator[tl] =
          cfg->ts_rate_decimator[tl] ? cfg->ts_rate_decimator[tl] : 1;
    }
  } else if (oxcf->ts_number_layers == 1) {
    oxcf->ts_rate_decimator[0] = 1;
  }

  if (get_level_index(oxcf->target_level) >= 0) config_target_level(oxcf);
  // vp9_dump_encoder_config(oxcf, stderr);
  return VPX_CODEC_OK;
}

static vpx_codec_err_t set_twopass_params_from_config(
    const vpx_codec_enc_cfg_t *const cfg, struct VP9_COMP *cpi) {
  if (!cfg->use_vizier_rc_params) return VPX_CODEC_OK;
  if (cpi == NULL) return VPX_CODEC_ERROR;

  cpi->twopass.use_vizier_rc_params = cfg->use_vizier_rc_params;

  // The values set here are factors that will be applied to default values
  // to get the final value used in the two pass code. Hence 1.0 will
  // match the default behaviour when not using passed in values.
  // We also apply limits here to prevent the user from applying settings
  // that make no sense.
  cpi->twopass.active_wq_factor =
      (double)cfg->active_wq_factor.num / (double)cfg->active_wq_factor.den;
  if (cpi->twopass.active_wq_factor < 0.25)
    cpi->twopass.active_wq_factor = 0.25;
  else if (cpi->twopass.active_wq_factor > 16.0)
    cpi->twopass.active_wq_factor = 16.0;

  cpi->twopass.err_per_mb =
      (double)cfg->err_per_mb_factor.num / (double)cfg->err_per_mb_factor.den;
  if (cpi->twopass.err_per_mb < 0.25)
    cpi->twopass.err_per_mb = 0.25;
  else if (cpi->twopass.err_per_mb > 4.0)
    cpi->twopass.err_per_mb = 4.0;

  cpi->twopass.sr_default_decay_limit =
      (double)cfg->sr_default_decay_limit.num /
      (double)cfg->sr_default_decay_limit.den;
  if (cpi->twopass.sr_default_decay_limit < 0.25)
    cpi->twopass.sr_default_decay_limit = 0.25;
  // If the default changes this will need to change.
  else if (cpi->twopass.sr_default_decay_limit > 1.33)
    cpi->twopass.sr_default_decay_limit = 1.33;

  cpi->twopass.sr_diff_factor =
      (double)cfg->sr_diff_factor.num / (double)cfg->sr_diff_factor.den;
  if (cpi->twopass.sr_diff_factor < 0.25)
    cpi->twopass.sr_diff_factor = 0.25;
  else if (cpi->twopass.sr_diff_factor > 4.0)
    cpi->twopass.sr_diff_factor = 4.0;

  cpi->twopass.kf_err_per_mb = (double)cfg->kf_err_per_mb_factor.num /
                               (double)cfg->kf_err_per_mb_factor.den;
  if (cpi->twopass.kf_err_per_mb < 0.25)
    cpi->twopass.kf_err_per_mb = 0.25;
  else if (cpi->twopass.kf_err_per_mb > 4.0)
    cpi->twopass.kf_err_per_mb = 4.0;

  cpi->twopass.kf_frame_min_boost = (double)cfg->kf_frame_min_boost_factor.num /
                                    (double)cfg->kf_frame_min_boost_factor.den;
  if (cpi->twopass.kf_frame_min_boost < 0.25)
    cpi->twopass.kf_frame_min_boost = 0.25;
  else if (cpi->twopass.kf_frame_min_boost > 4.0)
    cpi->twopass.kf_frame_min_boost = 4.0;

  cpi->twopass.kf_frame_max_boost_first =
      (double)cfg->kf_frame_max_boost_first_factor.num /
      (double)cfg->kf_frame_max_boost_first_factor.den;
  if (cpi->twopass.kf_frame_max_boost_first < 0.25)
    cpi->twopass.kf_frame_max_boost_first = 0.25;
  else if (cpi->twopass.kf_frame_max_boost_first > 4.0)
    cpi->twopass.kf_frame_max_boost_first = 4.0;

  cpi->twopass.kf_frame_max_boost_subs =
      (double)cfg->kf_frame_max_boost_subs_factor.num /
      (double)cfg->kf_frame_max_boost_subs_factor.den;
  if (cpi->twopass.kf_frame_max_boost_subs < 0.25)
    cpi->twopass.kf_frame_max_boost_subs = 0.25;
  else if (cpi->twopass.kf_frame_max_boost_subs > 4.0)
    cpi->twopass.kf_frame_max_boost_subs = 4.0;

  cpi->twopass.kf_max_total_boost = (double)cfg->kf_max_total_boost_factor.num /
                                    (double)cfg->kf_max_total_boost_factor.den;
  if (cpi->twopass.kf_max_total_boost < 0.25)
    cpi->twopass.kf_max_total_boost = 0.25;
  else if (cpi->twopass.kf_max_total_boost > 4.0)
    cpi->twopass.kf_max_total_boost = 4.0;

  cpi->twopass.gf_max_total_boost = (double)cfg->gf_max_total_boost_factor.num /
                                    (double)cfg->gf_max_total_boost_factor.den;
  if (cpi->twopass.gf_max_total_boost < 0.25)
    cpi->twopass.gf_max_total_boost = 0.25;
  else if (cpi->twopass.gf_max_total_boost > 4.0)
    cpi->twopass.gf_max_total_boost = 4.0;

  cpi->twopass.gf_frame_max_boost = (double)cfg->gf_frame_max_boost_factor.num /
                                    (double)cfg->gf_frame_max_boost_factor.den;
  if (cpi->twopass.gf_frame_max_boost < 0.25)
    cpi->twopass.gf_frame_max_boost = 0.25;
  else if (cpi->twopass.gf_frame_max_boost > 4.0)
    cpi->twopass.gf_frame_max_boost = 4.0;

  cpi->twopass.zm_factor =
      (double)cfg->zm_factor.num / (double)cfg->zm_factor.den;
  if (cpi->twopass.zm_factor < 0.25)
    cpi->twopass.zm_factor = 0.25;
  else if (cpi->twopass.zm_factor > 2.0)
    cpi->twopass.zm_factor = 2.0;

  cpi->rd_ctrl.rd_mult_inter_qp_fac = (double)cfg->rd_mult_inter_qp_fac.num /
                                      (double)cfg->rd_mult_inter_qp_fac.den;
  if (cpi->rd_ctrl.rd_mult_inter_qp_fac < 0.25)
    cpi->rd_ctrl.rd_mult_inter_qp_fac = 0.25;
  else if (cpi->rd_ctrl.rd_mult_inter_qp_fac > 4.0)
    cpi->rd_ctrl.rd_mult_inter_qp_fac = 4.0;

  cpi->rd_ctrl.rd_mult_arf_qp_fac =
      (double)cfg->rd_mult_arf_qp_fac.num / (double)cfg->rd_mult_arf_qp_fac.den;
  if (cpi->rd_ctrl.rd_mult_arf_qp_fac < 0.25)
    cpi->rd_ctrl.rd_mult_arf_qp_fac = 0.25;
  else if (cpi->rd_ctrl.rd_mult_arf_qp_fac > 4.0)
    cpi->rd_ctrl.rd_mult_arf_qp_fac = 4.0;

  cpi->rd_ctrl.rd_mult_key_qp_fac =
      (double)cfg->rd_mult_key_qp_fac.num / (double)cfg->rd_mult_key_qp_fac.den;
  if (cpi->rd_ctrl.rd_mult_key_qp_fac < 0.25)
    cpi->rd_ctrl.rd_mult_key_qp_fac = 0.25;
  else if (cpi->rd_ctrl.rd_mult_key_qp_fac > 4.0)
    cpi->rd_ctrl.rd_mult_key_qp_fac = 4.0;

  return VPX_CODEC_OK;
}

static vpx_codec_err_t encoder_set_config(vpx_codec_alg_priv_t *ctx,
                                          const vpx_codec_enc_cfg_t *cfg) {
  vpx_codec_err_t res;
  volatile int force_key = 0;

  if (cfg->g_w != ctx->cfg.g_w || cfg->g_h != ctx->cfg.g_h) {
    if (cfg->g_lag_in_frames > 1 || cfg->g_pass != VPX_RC_ONE_PASS)
      ERROR("Cannot change width or height after initialization");
    // Note: function encoder_set_config() is allowed to be called multiple
    // times. However, when the original frame width or height is less than two
    // times of the new frame width or height, a forced key frame should be
    // used (for the case of single spatial layer, since otherwise a previous
    //  encoded frame at a lower layer may be the desired reference). To make
    //  sure the correct detection of a forced key frame, we need
    // to update the frame width and height only when the actual encoding is
    // performed. cpi->last_coded_width and cpi->last_coded_height are used to
    // track the actual coded frame size.
    if ((ctx->cpi->last_coded_width && ctx->cpi->last_coded_height &&
         (!valid_ref_frame_size(ctx->cpi->last_coded_width,
                                ctx->cpi->last_coded_height, cfg->g_w,
                                cfg->g_h) &&
          ctx->cpi->svc.number_spatial_layers == 1)) ||
        (ctx->cpi->initial_width && (int)cfg->g_w > ctx->cpi->initial_width) ||
        (ctx->cpi->initial_height &&
         (int)cfg->g_h > ctx->cpi->initial_height)) {
      force_key = 1;
    }
  }

  // Prevent increasing lag_in_frames. This check is stricter than it needs
  // to be -- the limit is not increasing past the first lag_in_frames
  // value, but we don't track the initial config, only the last successful
  // config.
  if (cfg->g_lag_in_frames > ctx->cfg.g_lag_in_frames)
    ERROR("Cannot increase lag_in_frames");

  res = validate_config(ctx, cfg, &ctx->extra_cfg);
  if (res != VPX_CODEC_OK) return res;

  if (setjmp(ctx->cpi->common.error.jmp)) {
    const vpx_codec_err_t codec_err =
        update_error_state(ctx, &ctx->cpi->common.error);
    ctx->cpi->common.error.setjmp = 0;
    vpx_clear_system_state();
    assert(codec_err != VPX_CODEC_OK);
    return codec_err;
  }
  ctx->cpi->common.error.setjmp = 1;

  ctx->cfg = *cfg;
  set_encoder_config(&ctx->oxcf, &ctx->cfg, &ctx->extra_cfg);
  set_twopass_params_from_config(&ctx->cfg, ctx->cpi);
  // On profile change, request a key frame
  force_key |= ctx->cpi->common.profile != ctx->oxcf.profile;
  vp9_change_config(ctx->cpi, &ctx->oxcf);

  if (force_key) ctx->next_frame_flags |= VPX_EFLAG_FORCE_KF;

  ctx->cpi->common.error.setjmp = 0;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_get_quantizer(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
  int *const arg = va_arg(args, int *);
  if (arg == NULL) return VPX_CODEC_INVALID_PARAM;
  *arg = vp9_get_quantizer(ctx->cpi);
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_get_quantizer64(vpx_codec_alg_priv_t *ctx,
                                            va_list args) {
  int *const arg = va_arg(args, int *);
  if (arg == NULL) return VPX_CODEC_INVALID_PARAM;
  *arg = vp9_qindex_to_quantizer(vp9_get_quantizer(ctx->cpi));
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_get_quantizer_svc_layers(vpx_codec_alg_priv_t *ctx,
                                                     va_list args) {
  int *const arg = va_arg(args, int *);
  int i;
  if (arg == NULL) return VPX_CODEC_INVALID_PARAM;
  for (i = 0; i < VPX_SS_MAX_LAYERS; i++) {
    arg[i] = ctx->cpi->svc.base_qindex[i];
  }
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_get_loopfilter_level(vpx_codec_alg_priv_t *ctx,
                                                 va_list args) {
  int *const arg = va_arg(args, int *);
  if (arg == NULL) return VPX_CODEC_INVALID_PARAM;
  *arg = ctx->cpi->common.lf.filter_level;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t update_extra_cfg(vpx_codec_alg_priv_t *ctx,
                                        const struct vp9_extracfg *extra_cfg) {
  const vpx_codec_err_t res = validate_config(ctx, &ctx->cfg, extra_cfg);
  if (res == VPX_CODEC_OK) {
    ctx->extra_cfg = *extra_cfg;
    set_encoder_config(&ctx->oxcf, &ctx->cfg, &ctx->extra_cfg);
    set_twopass_params_from_config(&ctx->cfg, ctx->cpi);
    vp9_change_config(ctx->cpi, &ctx->oxcf);
  }
  return res;
}

static vpx_codec_err_t ctrl_set_cpuused(vpx_codec_alg_priv_t *ctx,
                                        va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  // Use fastest speed setting (speed 9 or -9) if it's set beyond the range.
  extra_cfg.cpu_used = CAST(VP8E_SET_CPUUSED, args);
  extra_cfg.cpu_used = clamp(extra_cfg.cpu_used, -9, 9);
#if CONFIG_REALTIME_ONLY
  if (extra_cfg.cpu_used > -5 && extra_cfg.cpu_used < 5)
    extra_cfg.cpu_used = (extra_cfg.cpu_used > 0) ? 5 : -5;
#endif
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_enable_auto_alt_ref(vpx_codec_alg_priv_t *ctx,
                                                    va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.enable_auto_alt_ref = CAST(VP8E_SET_ENABLEAUTOALTREF, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_noise_sensitivity(vpx_codec_alg_priv_t *ctx,
                                                  va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.noise_sensitivity = CAST(VP9E_SET_NOISE_SENSITIVITY, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_sharpness(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.sharpness = CAST(VP8E_SET_SHARPNESS, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_static_thresh(vpx_codec_alg_priv_t *ctx,
                                              va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.static_thresh = CAST(VP8E_SET_STATIC_THRESHOLD, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_tile_columns(vpx_codec_alg_priv_t *ctx,
                                             va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.tile_columns = CAST(VP9E_SET_TILE_COLUMNS, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_tile_rows(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.tile_rows = CAST(VP9E_SET_TILE_ROWS, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_tpl_model(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.enable_tpl_model = CAST(VP9E_SET_TPL, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_keyframe_filtering(vpx_codec_alg_priv_t *ctx,
                                                   va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.enable_keyframe_filtering =
      CAST(VP9E_SET_KEY_FRAME_FILTERING, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_arnr_max_frames(vpx_codec_alg_priv_t *ctx,
                                                va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.arnr_max_frames = CAST(VP8E_SET_ARNR_MAXFRAMES, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_arnr_strength(vpx_codec_alg_priv_t *ctx,
                                              va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.arnr_strength = CAST(VP8E_SET_ARNR_STRENGTH, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_arnr_type(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
  (void)ctx;
  (void)args;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_tuning(vpx_codec_alg_priv_t *ctx,
                                       va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.tuning = CAST(VP8E_SET_TUNING, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_cq_level(vpx_codec_alg_priv_t *ctx,
                                         va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.cq_level = CAST(VP8E_SET_CQ_LEVEL, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_rc_max_intra_bitrate_pct(
    vpx_codec_alg_priv_t *ctx, va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.rc_max_intra_bitrate_pct =
      CAST(VP8E_SET_MAX_INTRA_BITRATE_PCT, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_rc_max_inter_bitrate_pct(
    vpx_codec_alg_priv_t *ctx, va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.rc_max_inter_bitrate_pct =
      CAST(VP9E_SET_MAX_INTER_BITRATE_PCT, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_rc_gf_cbr_boost_pct(vpx_codec_alg_priv_t *ctx,
                                                    va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.gf_cbr_boost_pct = CAST(VP9E_SET_GF_CBR_BOOST_PCT, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_lossless(vpx_codec_alg_priv_t *ctx,
                                         va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.lossless = CAST(VP9E_SET_LOSSLESS, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_frame_parallel_decoding_mode(
    vpx_codec_alg_priv_t *ctx, va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.frame_parallel_decoding_mode =
      CAST(VP9E_SET_FRAME_PARALLEL_DECODING, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_aq_mode(vpx_codec_alg_priv_t *ctx,
                                        va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.aq_mode = CAST(VP9E_SET_AQ_MODE, args);
  if (ctx->cpi->fixed_qp_onepass) extra_cfg.aq_mode = 0;
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_alt_ref_aq(vpx_codec_alg_priv_t *ctx,
                                           va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.alt_ref_aq = CAST(VP9E_SET_ALT_REF_AQ, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_min_gf_interval(vpx_codec_alg_priv_t *ctx,
                                                va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.min_gf_interval = CAST(VP9E_SET_MIN_GF_INTERVAL, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_max_gf_interval(vpx_codec_alg_priv_t *ctx,
                                                va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.max_gf_interval = CAST(VP9E_SET_MAX_GF_INTERVAL, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_frame_periodic_boost(vpx_codec_alg_priv_t *ctx,
                                                     va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.frame_periodic_boost = CAST(VP9E_SET_FRAME_PERIODIC_BOOST, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_target_level(vpx_codec_alg_priv_t *ctx,
                                             va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.target_level = CAST(VP9E_SET_TARGET_LEVEL, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_row_mt(vpx_codec_alg_priv_t *ctx,
                                       va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.row_mt = CAST(VP9E_SET_ROW_MT, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_rtc_external_ratectrl(vpx_codec_alg_priv_t *ctx,
                                                      va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  const unsigned int data = va_arg(args, unsigned int);
  if (data) {
    cpi->compute_frame_low_motion_onepass = 0;
    cpi->rc.constrain_gf_key_freq_onepass_vbr = 0;
    cpi->cyclic_refresh->content_mode = 0;
    cpi->disable_scene_detection_rtc_ratectrl = 1;
  }
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_enable_motion_vector_unit_test(
    vpx_codec_alg_priv_t *ctx, va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.motion_vector_unit_test =
      CAST(VP9E_ENABLE_MOTION_VECTOR_UNIT_TEST, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_get_level(vpx_codec_alg_priv_t *ctx, va_list args) {
  int *const arg = va_arg(args, int *);
  if (arg == NULL) return VPX_CODEC_INVALID_PARAM;
  *arg = (int)vp9_get_level(&ctx->cpi->level_info.level_spec);
  return VPX_CODEC_OK;
}

static vpx_codec_err_t encoder_init(vpx_codec_ctx_t *ctx,
                                    vpx_codec_priv_enc_mr_cfg_t *data) {
  vpx_codec_err_t res = VPX_CODEC_OK;
  (void)data;

  if (ctx->priv == NULL) {
    vpx_codec_alg_priv_t *const priv = vpx_calloc(1, sizeof(*priv));
    if (priv == NULL) return VPX_CODEC_MEM_ERROR;

    ctx->priv = (vpx_codec_priv_t *)priv;
    ctx->priv->init_flags = ctx->init_flags;
    ctx->priv->enc.total_encoders = 1;
    priv->buffer_pool = (BufferPool *)vpx_calloc(1, sizeof(BufferPool));
    if (priv->buffer_pool == NULL) return VPX_CODEC_MEM_ERROR;

    if (ctx->config.enc) {
      // Update the reference to the config structure to an internal copy.
      priv->cfg = *ctx->config.enc;
      ctx->config.enc = &priv->cfg;
    }

    priv->extra_cfg = default_extra_cfg;
    vp9_initialize_enc();

    res = validate_config(priv, &priv->cfg, &priv->extra_cfg);

    if (res == VPX_CODEC_OK) {
      priv->pts_offset_initialized = 0;
      priv->global_header_subsampling = -1;
      set_encoder_config(&priv->oxcf, &priv->cfg, &priv->extra_cfg);
#if CONFIG_VP9_HIGHBITDEPTH
      priv->oxcf.use_highbitdepth =
          (ctx->init_flags & VPX_CODEC_USE_HIGHBITDEPTH) ? 1 : 0;
#endif
      priv->cpi = vp9_create_compressor(&priv->oxcf, priv->buffer_pool);
      if (priv->cpi == NULL) res = VPX_CODEC_MEM_ERROR;
      set_twopass_params_from_config(&priv->cfg, priv->cpi);
    }
  }

  return res;
}

static vpx_codec_err_t encoder_destroy(vpx_codec_alg_priv_t *ctx) {
  free(ctx->cx_data);
  free(ctx->global_headers.buf);
  vp9_remove_compressor(ctx->cpi);
  vpx_free(ctx->buffer_pool);
  vpx_free(ctx);
  return VPX_CODEC_OK;
}

static vpx_codec_err_t pick_quickcompress_mode(vpx_codec_alg_priv_t *ctx,
                                               unsigned long duration,
                                               vpx_enc_deadline_t deadline) {
  MODE new_mode = BEST;

#if CONFIG_REALTIME_ONLY
  (void)duration;
  deadline = VPX_DL_REALTIME;
#else
  switch (ctx->cfg.g_pass) {
    case VPX_RC_ONE_PASS:
      if (deadline > 0) {
        // Convert duration parameter from stream timebase to microseconds.
        VPX_STATIC_ASSERT(TICKS_PER_SEC > 1000000 &&
                          (TICKS_PER_SEC % 1000000) == 0);

        if (duration > UINT64_MAX / (uint64_t)ctx->oxcf.g_timebase_in_ts.num) {
          ERROR("duration is too big");
        }
        uint64_t duration_us = duration *
                               (uint64_t)ctx->oxcf.g_timebase_in_ts.num /
                               ((uint64_t)ctx->oxcf.g_timebase_in_ts.den *
                                (TICKS_PER_SEC / 1000000));

        // If the deadline is more that the duration this frame is to be shown,
        // use good quality mode. Otherwise use realtime mode.
        new_mode = (deadline > duration_us) ? GOOD : REALTIME;
      } else {
        new_mode = BEST;
      }
      break;
    case VPX_RC_FIRST_PASS: break;
    case VPX_RC_LAST_PASS: new_mode = deadline > 0 ? GOOD : BEST; break;
  }
#endif  // CONFIG_REALTIME_ONLY

  if (deadline == VPX_DL_REALTIME) {
    ctx->oxcf.pass = 0;
    new_mode = REALTIME;
  }

  if (ctx->oxcf.mode != new_mode) {
    ctx->oxcf.mode = new_mode;
    vp9_change_config(ctx->cpi, &ctx->oxcf);
  }
  return VPX_CODEC_OK;
}

// Turn on to test if supplemental superframe data breaks decoding
// #define TEST_SUPPLEMENTAL_SUPERFRAME_DATA
static int write_superframe_index(vpx_codec_alg_priv_t *ctx) {
  uint8_t marker = 0xc0;
  unsigned int mask;
  int mag, index_sz;

  assert(ctx->pending_frame_count);
  assert(ctx->pending_frame_count <= 8);

  // Add the number of frames to the marker byte
  marker |= ctx->pending_frame_count - 1;

  // Choose the magnitude
  for (mag = 0, mask = 0xff; mag < 4; mag++) {
    if (ctx->pending_frame_magnitude < mask) break;
    mask <<= 8;
    mask |= 0xff;
  }
  marker |= mag << 3;

  // Write the index
  index_sz = 2 + (mag + 1) * ctx->pending_frame_count;
  if (ctx->pending_cx_data_sz + index_sz < ctx->cx_data_sz) {
    uint8_t *x = ctx->pending_cx_data + ctx->pending_cx_data_sz;
    int i, j;
#ifdef TEST_SUPPLEMENTAL_SUPERFRAME_DATA
    uint8_t marker_test = 0xc0;
    int mag_test = 2;     // 1 - 4
    int frames_test = 4;  // 1 - 8
    int index_sz_test = 2 + mag_test * frames_test;
    marker_test |= frames_test - 1;
    marker_test |= (mag_test - 1) << 3;
    *x++ = marker_test;
    for (i = 0; i < mag_test * frames_test; ++i)
      *x++ = 0;  // fill up with arbitrary data
    *x++ = marker_test;
    ctx->pending_cx_data_sz += index_sz_test;
    printf("Added supplemental superframe data\n");
#endif

    *x++ = marker;
    for (i = 0; i < ctx->pending_frame_count; i++) {
      unsigned int this_sz = (unsigned int)ctx->pending_frame_sizes[i];

      for (j = 0; j <= mag; j++) {
        *x++ = this_sz & 0xff;
        this_sz >>= 8;
      }
    }
    *x++ = marker;
    ctx->pending_cx_data_sz += index_sz;
#ifdef TEST_SUPPLEMENTAL_SUPERFRAME_DATA
    index_sz += index_sz_test;
#endif
  }
  return index_sz;
}

static vpx_codec_frame_flags_t get_frame_pkt_flags(const VP9_COMP *cpi,
                                                   unsigned int lib_flags) {
  vpx_codec_frame_flags_t flags = lib_flags << 16;

  if (lib_flags & FRAMEFLAGS_KEY ||
      (cpi->use_svc && cpi->svc
                           .layer_context[cpi->svc.spatial_layer_id *
                                              cpi->svc.number_temporal_layers +
                                          cpi->svc.temporal_layer_id]
                           .is_key_frame))
    flags |= VPX_FRAME_IS_KEY;

  if (!cpi->common.show_frame) {
    flags |= VPX_FRAME_IS_INVISIBLE;
  }

  if (cpi->droppable) flags |= VPX_FRAME_IS_DROPPABLE;

  return flags;
}

static INLINE vpx_codec_cx_pkt_t get_psnr_pkt(const PSNR_STATS *psnr) {
  vpx_codec_cx_pkt_t pkt;
  pkt.kind = VPX_CODEC_PSNR_PKT;
  pkt.data.psnr = *psnr;
  return pkt;
}

#if !CONFIG_REALTIME_ONLY
static INLINE vpx_codec_cx_pkt_t
get_first_pass_stats_pkt(FIRSTPASS_STATS *stats) {
  // WARNNING: This function assumes that stats will
  // exist and not be changed until the packet is processed
  // TODO(angiebird): Refactor the code to avoid using the assumption.
  vpx_codec_cx_pkt_t pkt;
  pkt.kind = VPX_CODEC_STATS_PKT;
  pkt.data.twopass_stats.buf = stats;
  pkt.data.twopass_stats.sz = sizeof(*stats);
  return pkt;
}
#endif

const size_t kMinCompressedSize = 8192;
static vpx_codec_err_t encoder_encode(vpx_codec_alg_priv_t *ctx,
                                      const vpx_image_t *img,
                                      vpx_codec_pts_t pts_val,
                                      unsigned long duration,
                                      vpx_enc_frame_flags_t enc_flags,
                                      vpx_enc_deadline_t deadline) {
  volatile vpx_codec_err_t res = VPX_CODEC_OK;
  volatile vpx_enc_frame_flags_t flags = enc_flags;
  volatile vpx_codec_pts_t pts = pts_val;
  VP9_COMP *const cpi = ctx->cpi;
  const vpx_rational64_t *const timebase_in_ts = &ctx->oxcf.g_timebase_in_ts;
  size_t data_sz;
  vpx_codec_cx_pkt_t pkt;
  memset(&pkt, 0, sizeof(pkt));

  if (cpi == NULL) return VPX_CODEC_INVALID_PARAM;

  cpi->last_coded_width = ctx->oxcf.width;
  cpi->last_coded_height = ctx->oxcf.height;

  if (img != NULL) {
    res = validate_img(ctx, img);
    if (res == VPX_CODEC_OK) {
      // There's no codec control for multiple alt-refs so check the encoder
      // instance for its status to determine the compressed data size.
      data_sz = ctx->cfg.g_w * ctx->cfg.g_h * get_image_bps(img) / 8 *
                (cpi->multi_layer_arf ? 8 : 2);
      if (data_sz < kMinCompressedSize) data_sz = kMinCompressedSize;
      if (ctx->cx_data == NULL || ctx->cx_data_sz < data_sz) {
        ctx->cx_data_sz = data_sz;
        free(ctx->cx_data);
        ctx->cx_data = (unsigned char *)malloc(ctx->cx_data_sz);
        if (ctx->cx_data == NULL) {
          return VPX_CODEC_MEM_ERROR;
        }
      }

      int chroma_subsampling = -1;
      if ((img->fmt & VPX_IMG_FMT_I420) == VPX_IMG_FMT_I420 ||
          (img->fmt & VPX_IMG_FMT_NV12) == VPX_IMG_FMT_NV12 ||
          (img->fmt & VPX_IMG_FMT_YV12) == VPX_IMG_FMT_YV12) {
        chroma_subsampling = 1;  // matches default for Codec Parameter String
      } else if ((img->fmt & VPX_IMG_FMT_I422) == VPX_IMG_FMT_I422) {
        chroma_subsampling = 2;
      } else if ((img->fmt & VPX_IMG_FMT_I444) == VPX_IMG_FMT_I444) {
        chroma_subsampling = 3;
      }
      if (chroma_subsampling > ctx->global_header_subsampling) {
        ctx->global_header_subsampling = chroma_subsampling;
      }
    }
  }

  res = pick_quickcompress_mode(ctx, duration, deadline);
  if (res != VPX_CODEC_OK) {
    return res;
  }
  vpx_codec_pkt_list_init(&ctx->pkt_list);

  // Handle Flags
  if (((flags & VP8_EFLAG_NO_UPD_GF) && (flags & VP8_EFLAG_FORCE_GF)) ||
      ((flags & VP8_EFLAG_NO_UPD_ARF) && (flags & VP8_EFLAG_FORCE_ARF))) {
    ctx->base.err_detail = "Conflicting flags.";
    return VPX_CODEC_INVALID_PARAM;
  }

  if (setjmp(cpi->common.error.jmp)) {
    cpi->common.error.setjmp = 0;
    res = update_error_state(ctx, &cpi->common.error);
    vpx_clear_system_state();
    return res;
  }
  cpi->common.error.setjmp = 1;

  if (res == VPX_CODEC_OK) vp9_apply_encoding_flags(cpi, flags);

  // Handle fixed keyframe intervals
  if (ctx->cfg.kf_mode == VPX_KF_AUTO &&
      ctx->cfg.kf_min_dist == ctx->cfg.kf_max_dist) {
    if (++ctx->fixed_kf_cntr > ctx->cfg.kf_min_dist) {
      flags |= VPX_EFLAG_FORCE_KF;
      ctx->fixed_kf_cntr = 1;
    }
  }

  if (res == VPX_CODEC_OK) {
    unsigned int lib_flags = 0;
    size_t size, cx_data_sz;
    unsigned char *cx_data;

    // Per-frame PSNR is not supported when g_lag_in_frames is greater than 0.
    if ((flags & VPX_EFLAG_CALCULATE_PSNR) && ctx->cfg.g_lag_in_frames != 0) {
      vpx_internal_error(
          &ctx->cpi->common.error, VPX_CODEC_INCAPABLE,
          "Cannot calculate per-frame PSNR when g_lag_in_frames is nonzero");
    }
    // Set up internal flags
#if CONFIG_INTERNAL_STATS
    assert(cpi->b_calculate_psnr == 1);
#else
    cpi->b_calculate_psnr = (ctx->base.init_flags & VPX_CODEC_USE_PSNR) ||
                            (flags & VPX_EFLAG_CALCULATE_PSNR);
#endif

    if (img != NULL) {
      YV12_BUFFER_CONFIG sd;

      if (!ctx->pts_offset_initialized) {
        ctx->pts_offset = pts;
        ctx->pts_offset_initialized = 1;
      }
      if (pts < ctx->pts_offset) {
        vpx_internal_error(&cpi->common.error, VPX_CODEC_INVALID_PARAM,
                           "pts is smaller than initial pts");
      }
      pts -= ctx->pts_offset;
      if (pts > INT64_MAX / timebase_in_ts->num) {
        vpx_internal_error(
            &cpi->common.error, VPX_CODEC_INVALID_PARAM,
            "conversion of relative pts to ticks would overflow");
      }
      const int64_t dst_time_stamp =
          timebase_units_to_ticks(timebase_in_ts, pts);

      cpi->svc.timebase_fac = timebase_units_to_ticks(timebase_in_ts, 1);
      cpi->svc.time_stamp_superframe = dst_time_stamp;

#if ULONG_MAX > INT64_MAX
      if (duration > INT64_MAX) {
        vpx_internal_error(&cpi->common.error, VPX_CODEC_INVALID_PARAM,
                           "duration is too big");
      }
#endif
      if (pts > INT64_MAX - (int64_t)duration) {
        vpx_internal_error(&cpi->common.error, VPX_CODEC_INVALID_PARAM,
                           "relative pts + duration is too big");
      }
      vpx_codec_pts_t pts_end = pts + (int64_t)duration;
      if (pts_end > INT64_MAX / timebase_in_ts->num) {
        vpx_internal_error(
            &cpi->common.error, VPX_CODEC_INVALID_PARAM,
            "conversion of relative pts + duration to ticks would overflow");
      }
      const int64_t dst_end_time_stamp =
          timebase_units_to_ticks(timebase_in_ts, pts_end);
      res = image2yuvconfig(img, &sd);

      // Store the original flags in to the frame buffer. Will extract the
      // key frame flag when we actually encode this frame.
      if (vp9_receive_raw_frame(cpi, flags | ctx->next_frame_flags, &sd,
                                dst_time_stamp, dst_end_time_stamp)) {
        res = update_error_state(ctx, &cpi->common.error);
      }
      ctx->next_frame_flags = 0;
    }

    cx_data = ctx->cx_data;
    cx_data_sz = ctx->cx_data_sz;

    /* Any pending invisible frames? */
    if (ctx->pending_cx_data) {
      assert(cx_data_sz >= ctx->pending_cx_data_sz);
      memmove(cx_data, ctx->pending_cx_data, ctx->pending_cx_data_sz);
      ctx->pending_cx_data = cx_data;
      cx_data += ctx->pending_cx_data_sz;
      cx_data_sz -= ctx->pending_cx_data_sz;

      /* TODO(webm:1844): this is a minimal check, the underlying codec doesn't
       * respect the buffer size anyway.
       */
      if (cx_data_sz < ctx->cx_data_sz / 2) {
        vpx_internal_error(&cpi->common.error, VPX_CODEC_ERROR,
                           "Compressed data buffer too small");
      }
    }

    if (cpi->oxcf.pass == 1 && !cpi->use_svc) {
#if !CONFIG_REALTIME_ONLY
      // compute first pass stats
      if (img) {
        int ret;
        int64_t dst_time_stamp;
        int64_t dst_end_time_stamp;
        vpx_codec_cx_pkt_t fps_pkt;
        ENCODE_FRAME_RESULT encode_frame_result;
        vp9_init_encode_frame_result(&encode_frame_result);
        // TODO(angiebird): Call vp9_first_pass directly
        ret = vp9_get_compressed_data(
            cpi, &lib_flags, &size, cx_data, cx_data_sz, &dst_time_stamp,
            &dst_end_time_stamp, !img, &encode_frame_result);
        assert(size == 0);  // There is no compressed data in the first pass
        (void)ret;
        assert(ret == 0);
        fps_pkt = get_first_pass_stats_pkt(&cpi->twopass.this_frame_stats);
        vpx_codec_pkt_list_add(&ctx->pkt_list.head, &fps_pkt);
      } else {
        if (!cpi->twopass.first_pass_done) {
          vpx_codec_cx_pkt_t fps_pkt;
          vp9_end_first_pass(cpi);
          fps_pkt = get_first_pass_stats_pkt(&cpi->twopass.total_stats);
          vpx_codec_pkt_list_add(&ctx->pkt_list.head, &fps_pkt);
        }
      }
#else   // !CONFIG_REALTIME_ONLY
      assert(0);
#endif  // !CONFIG_REALTIME_ONLY
    } else {
      ENCODE_FRAME_RESULT encode_frame_result;
      int64_t dst_time_stamp;
      int64_t dst_end_time_stamp;
      vp9_init_encode_frame_result(&encode_frame_result);
      while (cx_data_sz >= ctx->cx_data_sz / 2 &&
             -1 != vp9_get_compressed_data(cpi, &lib_flags, &size, cx_data,
                                           cx_data_sz, &dst_time_stamp,
                                           &dst_end_time_stamp, !img,
                                           &encode_frame_result)) {
        // Pack psnr pkt.
        if (size > 0) {
          PSNR_STATS psnr;
          if (vp9_get_psnr(cpi, &psnr)) {
            vpx_codec_cx_pkt_t psnr_pkt = get_psnr_pkt(&psnr);
            vpx_codec_pkt_list_add(&ctx->pkt_list.head, &psnr_pkt);
          }
        }

        if (size || (cpi->use_svc && cpi->svc.skip_enhancement_layer)) {
          // Pack invisible frames with the next visible frame
          if (!cpi->common.show_frame ||
              (cpi->use_svc && cpi->svc.spatial_layer_id <
                                   cpi->svc.number_spatial_layers - 1)) {
            if (ctx->pending_cx_data == NULL) ctx->pending_cx_data = cx_data;
            ctx->pending_cx_data_sz += size;
            if (size)
              ctx->pending_frame_sizes[ctx->pending_frame_count++] = size;
            ctx->pending_frame_magnitude |= size;
            cx_data += size;
            cx_data_sz -= size;
            pkt.data.frame.width[cpi->svc.spatial_layer_id] = cpi->common.width;
            pkt.data.frame.height[cpi->svc.spatial_layer_id] =
                cpi->common.height;
            pkt.data.frame.spatial_layer_encoded[cpi->svc.spatial_layer_id] =
                1 - cpi->svc.drop_spatial_layer[cpi->svc.spatial_layer_id];

            if (ctx->output_cx_pkt_cb.output_cx_pkt) {
              pkt.kind = VPX_CODEC_CX_FRAME_PKT;
              pkt.data.frame.pts =
                  ticks_to_timebase_units(timebase_in_ts, dst_time_stamp) +
                  ctx->pts_offset;
              pkt.data.frame.duration = (unsigned long)ticks_to_timebase_units(
                  timebase_in_ts, dst_end_time_stamp - dst_time_stamp);
              pkt.data.frame.flags = get_frame_pkt_flags(cpi, lib_flags);
              pkt.data.frame.buf = ctx->pending_cx_data;
              pkt.data.frame.sz = size;
              ctx->pending_cx_data = NULL;
              ctx->pending_cx_data_sz = 0;
              ctx->pending_frame_count = 0;
              ctx->pending_frame_magnitude = 0;
              ctx->output_cx_pkt_cb.output_cx_pkt(
                  &pkt, ctx->output_cx_pkt_cb.user_priv);
            }
            continue;
          }

          // Add the frame packet to the list of returned packets.
          pkt.kind = VPX_CODEC_CX_FRAME_PKT;
          pkt.data.frame.pts =
              ticks_to_timebase_units(timebase_in_ts, dst_time_stamp) +
              ctx->pts_offset;
          pkt.data.frame.duration = (unsigned long)ticks_to_timebase_units(
              timebase_in_ts, dst_end_time_stamp - dst_time_stamp);
          pkt.data.frame.flags = get_frame_pkt_flags(cpi, lib_flags);
          pkt.data.frame.width[cpi->svc.spatial_layer_id] = cpi->common.width;
          pkt.data.frame.height[cpi->svc.spatial_layer_id] = cpi->common.height;
          pkt.data.frame.spatial_layer_encoded[cpi->svc.spatial_layer_id] =
              1 - cpi->svc.drop_spatial_layer[cpi->svc.spatial_layer_id];

          if (ctx->pending_cx_data) {
            if (size)
              ctx->pending_frame_sizes[ctx->pending_frame_count++] = size;
            ctx->pending_frame_magnitude |= size;
            ctx->pending_cx_data_sz += size;
            // write the superframe only for the case when
            if (!ctx->output_cx_pkt_cb.output_cx_pkt)
              size += write_superframe_index(ctx);
            pkt.data.frame.buf = ctx->pending_cx_data;
            pkt.data.frame.sz = ctx->pending_cx_data_sz;
            ctx->pending_cx_data = NULL;
            ctx->pending_cx_data_sz = 0;
            ctx->pending_frame_count = 0;
            ctx->pending_frame_magnitude = 0;
          } else {
            pkt.data.frame.buf = cx_data;
            pkt.data.frame.sz = size;
          }
          pkt.data.frame.partition_id = -1;

          if (ctx->output_cx_pkt_cb.output_cx_pkt)
            ctx->output_cx_pkt_cb.output_cx_pkt(
                &pkt, ctx->output_cx_pkt_cb.user_priv);
          else
            vpx_codec_pkt_list_add(&ctx->pkt_list.head, &pkt);

          cx_data += size;
          cx_data_sz -= size;
          if (is_one_pass_svc(cpi) && (cpi->svc.spatial_layer_id ==
                                       cpi->svc.number_spatial_layers - 1)) {
            // Encoded all spatial layers; exit loop.
            break;
          }
        }
      }
    }
  }

  cpi->common.error.setjmp = 0;
  return res;
}

static const vpx_codec_cx_pkt_t *encoder_get_cxdata(vpx_codec_alg_priv_t *ctx,
                                                    vpx_codec_iter_t *iter) {
  return vpx_codec_pkt_list_get(&ctx->pkt_list.head, iter);
}

static vpx_codec_err_t ctrl_set_reference(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
  vpx_ref_frame_t *const frame = va_arg(args, vpx_ref_frame_t *);

  if (frame != NULL) {
    YV12_BUFFER_CONFIG sd;

    image2yuvconfig(&frame->img, &sd);
    vp9_set_reference_enc(ctx->cpi, ref_frame_to_vp9_reframe(frame->frame_type),
                          &sd);
    return VPX_CODEC_OK;
  }
  return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t ctrl_copy_reference(vpx_codec_alg_priv_t *ctx,
                                           va_list args) {
  vpx_ref_frame_t *const frame = va_arg(args, vpx_ref_frame_t *);

  if (frame != NULL) {
    YV12_BUFFER_CONFIG sd;

    image2yuvconfig(&frame->img, &sd);
    vp9_copy_reference_enc(ctx->cpi,
                           ref_frame_to_vp9_reframe(frame->frame_type), &sd);
    return VPX_CODEC_OK;
  }
  return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t ctrl_get_reference(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
  vp9_ref_frame_t *const frame = va_arg(args, vp9_ref_frame_t *);

  if (frame != NULL) {
    const int fb_idx = ctx->cpi->common.cur_show_frame_fb_idx;
    YV12_BUFFER_CONFIG *fb = get_buf_frame(&ctx->cpi->common, fb_idx);
    if (fb == NULL) return VPX_CODEC_ERROR;
    yuvconfig2image(&frame->img, fb, NULL);
    return VPX_CODEC_OK;
  }
  return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t ctrl_set_previewpp(vpx_codec_alg_priv_t *ctx,
                                          va_list args) {
#if CONFIG_VP9_POSTPROC
  vp8_postproc_cfg_t *config = va_arg(args, vp8_postproc_cfg_t *);
  if (config != NULL) {
    ctx->preview_ppcfg = *config;
    return VPX_CODEC_OK;
  }
  return VPX_CODEC_INVALID_PARAM;
#else
  (void)ctx;
  (void)args;
  return VPX_CODEC_INCAPABLE;
#endif
}

// Returns the contents of CodecPrivate described in:
// https://www.webmproject.org/docs/container/#vp9-codec-feature-metadata-codecprivate
// This includes Profile, Level, Bit depth and Chroma subsampling. Each entry
// is 3 bytes. 1 byte ID, 1 byte length (= 1) and 1 byte value.
static vpx_fixed_buf_t *encoder_get_global_headers(vpx_codec_alg_priv_t *ctx) {
  if (!ctx->cpi) return NULL;

  const unsigned int profile = ctx->cfg.g_profile;
  const VP9_LEVEL level = vp9_get_level(&ctx->cpi->level_info.level_spec);
  const vpx_bit_depth_t bit_depth = ctx->cfg.g_bit_depth;
  const int subsampling = ctx->global_header_subsampling;
  const uint8_t buf[12] = {
    1, 1, (uint8_t)profile,   2, 1, (uint8_t)level,
    3, 1, (uint8_t)bit_depth, 4, 1, (uint8_t)subsampling
  };

  if (ctx->global_headers.buf) free(ctx->global_headers.buf);
  ctx->global_headers.buf = malloc(sizeof(buf));
  if (!ctx->global_headers.buf) return NULL;

  ctx->global_headers.sz = sizeof(buf);
  // No data or I440, which isn't mapped.
  if (ctx->global_header_subsampling == -1) ctx->global_headers.sz -= 3;
  memcpy(ctx->global_headers.buf, buf, ctx->global_headers.sz);

  return &ctx->global_headers;
}

static vpx_image_t *encoder_get_preview(vpx_codec_alg_priv_t *ctx) {
  YV12_BUFFER_CONFIG sd;
  vp9_ppflags_t flags;
  vp9_zero(flags);

  if (ctx->preview_ppcfg.post_proc_flag) {
    flags.post_proc_flag = ctx->preview_ppcfg.post_proc_flag;
    flags.deblocking_level = ctx->preview_ppcfg.deblocking_level;
    flags.noise_level = ctx->preview_ppcfg.noise_level;
  }

  if (vp9_get_preview_raw_frame(ctx->cpi, &sd, &flags) == 0) {
    yuvconfig2image(&ctx->preview_img, &sd, NULL);
    return &ctx->preview_img;
  }
  return NULL;
}

static vpx_codec_err_t ctrl_set_roi_map(vpx_codec_alg_priv_t *ctx,
                                        va_list args) {
  vpx_roi_map_t *data = va_arg(args, vpx_roi_map_t *);

  if (data) {
    vpx_roi_map_t *roi = (vpx_roi_map_t *)data;
    return vp9_set_roi_map(ctx->cpi, roi->roi_map, roi->rows, roi->cols,
                           roi->delta_q, roi->delta_lf, roi->skip,
                           roi->ref_frame);
  }
  return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t ctrl_set_active_map(vpx_codec_alg_priv_t *ctx,
                                           va_list args) {
  vpx_active_map_t *const map = va_arg(args, vpx_active_map_t *);

  if (map) {
    if (!vp9_set_active_map(ctx->cpi, map->active_map, (int)map->rows,
                            (int)map->cols))
      return VPX_CODEC_OK;

    return VPX_CODEC_INVALID_PARAM;
  }
  return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t ctrl_get_active_map(vpx_codec_alg_priv_t *ctx,
                                           va_list args) {
  vpx_active_map_t *const map = va_arg(args, vpx_active_map_t *);

  if (map) {
    if (!vp9_get_active_map(ctx->cpi, map->active_map, (int)map->rows,
                            (int)map->cols))
      return VPX_CODEC_OK;

    return VPX_CODEC_INVALID_PARAM;
  }
  return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t ctrl_set_scale_mode(vpx_codec_alg_priv_t *ctx,
                                           va_list args) {
  vpx_scaling_mode_t *const mode = va_arg(args, vpx_scaling_mode_t *);

  if (mode) {
    const int res = vp9_set_internal_size(ctx->cpi, mode->h_scaling_mode,
                                          mode->v_scaling_mode);
    return (res == 0) ? VPX_CODEC_OK : VPX_CODEC_INVALID_PARAM;
  }
  return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t ctrl_set_svc(vpx_codec_alg_priv_t *ctx, va_list args) {
  int data = va_arg(args, int);
  const vpx_codec_enc_cfg_t *cfg = &ctx->cfg;
  // Both one-pass and two-pass RC are supported now.
  // User setting this has to make sure of the following.
  // In two-pass setting: either (but not both)
  //      cfg->ss_number_layers > 1, or cfg->ts_number_layers > 1
  // In one-pass setting:
  //      either or both cfg->ss_number_layers > 1, or cfg->ts_number_layers > 1

  vp9_set_svc(ctx->cpi, data);

  if (data == 1 &&
      (cfg->g_pass == VPX_RC_FIRST_PASS || cfg->g_pass == VPX_RC_LAST_PASS) &&
      cfg->ss_number_layers > 1 && cfg->ts_number_layers > 1) {
    return VPX_CODEC_INVALID_PARAM;
  }

  vp9_set_row_mt(ctx->cpi);

  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_svc_layer_id(vpx_codec_alg_priv_t *ctx,
                                             va_list args) {
  vpx_svc_layer_id_t *const data = va_arg(args, vpx_svc_layer_id_t *);
  VP9_COMP *const cpi = (VP9_COMP *)ctx->cpi;
  SVC *const svc = &cpi->svc;
  int sl;

  svc->spatial_layer_to_encode = data->spatial_layer_id;
  svc->first_spatial_layer_to_encode = data->spatial_layer_id;
  // TODO(jianj): Deprecated to be removed.
  svc->temporal_layer_id = data->temporal_layer_id;
  // Allow for setting temporal layer per spatial layer for superframe.
  for (sl = 0; sl < cpi->svc.number_spatial_layers; ++sl) {
    svc->temporal_layer_id_per_spatial[sl] =
        data->temporal_layer_id_per_spatial[sl];
  }
  // Checks on valid layer_id input.
  if (svc->temporal_layer_id < 0 ||
      svc->temporal_layer_id >= (int)ctx->cfg.ts_number_layers) {
    return VPX_CODEC_INVALID_PARAM;
  }

  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_get_svc_layer_id(vpx_codec_alg_priv_t *ctx,
                                             va_list args) {
  vpx_svc_layer_id_t *data = va_arg(args, vpx_svc_layer_id_t *);
  VP9_COMP *const cpi = (VP9_COMP *)ctx->cpi;
  SVC *const svc = &cpi->svc;

  data->spatial_layer_id = svc->spatial_layer_id;
  data->temporal_layer_id = svc->temporal_layer_id;

  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_svc_parameters(vpx_codec_alg_priv_t *ctx,
                                               va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  vpx_svc_extra_cfg_t *const params = va_arg(args, vpx_svc_extra_cfg_t *);
  int sl, tl;

  // Number of temporal layers and number of spatial layers have to be set
  // properly before calling this control function.
  for (sl = 0; sl < cpi->svc.number_spatial_layers; ++sl) {
    for (tl = 0; tl < cpi->svc.number_temporal_layers; ++tl) {
      const int layer =
          LAYER_IDS_TO_IDX(sl, tl, cpi->svc.number_temporal_layers);
      LAYER_CONTEXT *lc = &cpi->svc.layer_context[layer];
      lc->max_q = params->max_quantizers[layer];
      lc->min_q = params->min_quantizers[layer];
      // Checks on valid scale factors.
      if (params->scaling_factor_num[sl] < 1 ||
          params->scaling_factor_den[sl] < 1 ||
          (params->scaling_factor_num[sl] > params->scaling_factor_den[sl])) {
        return VPX_CODEC_INVALID_PARAM;
      }
      lc->scaling_factor_num = params->scaling_factor_num[sl];
      lc->scaling_factor_den = params->scaling_factor_den[sl];
      lc->speed = params->speed_per_layer[sl];
      lc->loopfilter_ctrl = params->loopfilter_ctrl[sl];
    }
  }

  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_get_svc_ref_frame_config(vpx_codec_alg_priv_t *ctx,
                                                     va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  vpx_svc_ref_frame_config_t *data = va_arg(args, vpx_svc_ref_frame_config_t *);
  int sl;
  for (sl = 0; sl <= cpi->svc.spatial_layer_id; sl++) {
    data->update_buffer_slot[sl] = cpi->svc.update_buffer_slot[sl];
    data->reference_last[sl] = cpi->svc.reference_last[sl];
    data->reference_golden[sl] = cpi->svc.reference_golden[sl];
    data->reference_alt_ref[sl] = cpi->svc.reference_altref[sl];
    data->lst_fb_idx[sl] = cpi->svc.lst_fb_idx[sl];
    data->gld_fb_idx[sl] = cpi->svc.gld_fb_idx[sl];
    data->alt_fb_idx[sl] = cpi->svc.alt_fb_idx[sl];
    // TODO(jianj): Remove these 3, deprecated.
    data->update_last[sl] = cpi->svc.update_last[sl];
    data->update_golden[sl] = cpi->svc.update_golden[sl];
    data->update_alt_ref[sl] = cpi->svc.update_altref[sl];
  }
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_svc_ref_frame_config(vpx_codec_alg_priv_t *ctx,
                                                     va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  vpx_svc_ref_frame_config_t *data = va_arg(args, vpx_svc_ref_frame_config_t *);
  int sl;
  cpi->svc.use_set_ref_frame_config = 1;
  for (sl = 0; sl < cpi->svc.number_spatial_layers; ++sl) {
    cpi->svc.update_buffer_slot[sl] = data->update_buffer_slot[sl];
    cpi->svc.reference_last[sl] = data->reference_last[sl];
    cpi->svc.reference_golden[sl] = data->reference_golden[sl];
    cpi->svc.reference_altref[sl] = data->reference_alt_ref[sl];
    cpi->svc.lst_fb_idx[sl] = data->lst_fb_idx[sl];
    cpi->svc.gld_fb_idx[sl] = data->gld_fb_idx[sl];
    cpi->svc.alt_fb_idx[sl] = data->alt_fb_idx[sl];
    cpi->svc.duration[sl] = data->duration[sl];
  }
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_svc_inter_layer_pred(vpx_codec_alg_priv_t *ctx,
                                                     va_list args) {
  const int data = va_arg(args, int);
  VP9_COMP *const cpi = ctx->cpi;
  cpi->svc.disable_inter_layer_pred = data;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_svc_frame_drop_layer(vpx_codec_alg_priv_t *ctx,
                                                     va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  vpx_svc_frame_drop_t *data = va_arg(args, vpx_svc_frame_drop_t *);
  int sl;
  cpi->svc.framedrop_mode = data->framedrop_mode;
  for (sl = 0; sl < cpi->svc.number_spatial_layers; ++sl)
    cpi->svc.framedrop_thresh[sl] = data->framedrop_thresh[sl];
  // Don't allow max_consec_drop values below 1.
  cpi->svc.max_consec_drop = VPXMAX(1, data->max_consec_drop);
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_svc_gf_temporal_ref(vpx_codec_alg_priv_t *ctx,
                                                    va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  const unsigned int data = va_arg(args, unsigned int);
  cpi->svc.use_gf_temporal_ref = data;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_svc_spatial_layer_sync(
    vpx_codec_alg_priv_t *ctx, va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  vpx_svc_spatial_layer_sync_t *data =
      va_arg(args, vpx_svc_spatial_layer_sync_t *);
  int sl;
  for (sl = 0; sl < cpi->svc.number_spatial_layers; ++sl)
    cpi->svc.spatial_layer_sync[sl] = data->spatial_layer_sync[sl];
  cpi->svc.set_intra_only_frame = data->base_layer_intra_only;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_delta_q_uv(vpx_codec_alg_priv_t *ctx,
                                           va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  int data = va_arg(args, int);
  data = clamp(data, -15, 15);
  extra_cfg.delta_q_uv = data;
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_register_cx_callback(vpx_codec_alg_priv_t *ctx,
                                                 va_list args) {
  vpx_codec_priv_output_cx_pkt_cb_pair_t *cbp =
      (vpx_codec_priv_output_cx_pkt_cb_pair_t *)va_arg(args, void *);
  ctx->output_cx_pkt_cb.output_cx_pkt = cbp->output_cx_pkt;
  ctx->output_cx_pkt_cb.user_priv = cbp->user_priv;

  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_tune_content(vpx_codec_alg_priv_t *ctx,
                                             va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.content = CAST(VP9E_SET_TUNE_CONTENT, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_color_space(vpx_codec_alg_priv_t *ctx,
                                            va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.color_space = CAST(VP9E_SET_COLOR_SPACE, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_color_range(vpx_codec_alg_priv_t *ctx,
                                            va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  extra_cfg.color_range = CAST(VP9E_SET_COLOR_RANGE, args);
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_render_size(vpx_codec_alg_priv_t *ctx,
                                            va_list args) {
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  int *const render_size = va_arg(args, int *);
  extra_cfg.render_width = render_size[0];
  extra_cfg.render_height = render_size[1];
  return update_extra_cfg(ctx, &extra_cfg);
}

static vpx_codec_err_t ctrl_set_postencode_drop(vpx_codec_alg_priv_t *ctx,
                                                va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  const unsigned int data = va_arg(args, unsigned int);
  cpi->rc.ext_use_post_encode_drop = data;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_disable_overshoot_maxq_cbr(
    vpx_codec_alg_priv_t *ctx, va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  const unsigned int data = va_arg(args, unsigned int);
  cpi->rc.disable_overshoot_maxq_cbr = data;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_disable_loopfilter(vpx_codec_alg_priv_t *ctx,
                                                   va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  const unsigned int data = va_arg(args, unsigned int);
  cpi->loopfilter_ctrl = data;
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_external_rate_control(vpx_codec_alg_priv_t *ctx,
                                                      va_list args) {
  vpx_rc_funcs_t funcs = *CAST(VP9E_SET_EXTERNAL_RATE_CONTROL, args);
  VP9_COMP *cpi = ctx->cpi;
  EXT_RATECTRL *ext_ratectrl = &cpi->ext_ratectrl;
  const VP9EncoderConfig *oxcf = &cpi->oxcf;
  if (oxcf->pass == 2) {
    const FRAME_INFO *frame_info = &cpi->frame_info;
    vpx_rc_config_t ratectrl_config;
    vpx_codec_err_t codec_status;
    memset(&ratectrl_config, 0, sizeof(ratectrl_config));

    ratectrl_config.frame_width = frame_info->frame_width;
    ratectrl_config.frame_height = frame_info->frame_height;
    ratectrl_config.show_frame_count = cpi->twopass.first_pass_info.num_frames;
    ratectrl_config.max_gf_interval = oxcf->max_gf_interval;
    ratectrl_config.min_gf_interval = oxcf->min_gf_interval;
    // TODO(angiebird): Double check whether this is the proper way to set up
    // target_bitrate and frame_rate.
    ratectrl_config.target_bitrate_kbps = (int)(oxcf->target_bandwidth / 1000);
    ratectrl_config.frame_rate_num = oxcf->g_timebase.den;
    ratectrl_config.frame_rate_den = oxcf->g_timebase.num;
    ratectrl_config.overshoot_percent = oxcf->over_shoot_pct;
    ratectrl_config.undershoot_percent = oxcf->under_shoot_pct;
    ratectrl_config.min_base_q_index = oxcf->best_allowed_q;
    ratectrl_config.max_base_q_index = oxcf->worst_allowed_q;
    ratectrl_config.base_qp = oxcf->cq_level;

    if (oxcf->rc_mode == VPX_VBR) {
      ratectrl_config.rc_mode = VPX_RC_VBR;
    } else if (oxcf->rc_mode == VPX_Q) {
      ratectrl_config.rc_mode = VPX_RC_QMODE;
    } else if (oxcf->rc_mode == VPX_CQ) {
      ratectrl_config.rc_mode = VPX_RC_CQ;
    }

    codec_status = vp9_extrc_create(funcs, ratectrl_config, ext_ratectrl);
    if (codec_status != VPX_CODEC_OK) {
      return codec_status;
    }
  }
  return VPX_CODEC_OK;
}

static vpx_codec_err_t ctrl_set_quantizer_one_pass(vpx_codec_alg_priv_t *ctx,
                                                   va_list args) {
  VP9_COMP *const cpi = ctx->cpi;
  const int qp = va_arg(args, int);
  vpx_codec_enc_cfg_t *cfg = &ctx->cfg;
  struct vp9_extracfg extra_cfg = ctx->extra_cfg;
  vpx_codec_err_t res;

  if (qp < 0 || qp > 63) return VPX_CODEC_INVALID_PARAM;

  cfg->rc_min_quantizer = cfg->rc_max_quantizer = qp;
  extra_cfg.aq_mode = 0;
  cpi->fixed_qp_onepass = 1;

  res = update_extra_cfg(ctx, &extra_cfg);
  return res;
}

static vpx_codec_ctrl_fn_map_t encoder_ctrl_maps[] = {
  { VP8_COPY_REFERENCE, ctrl_copy_reference },

  // Setters
  { VP8_SET_REFERENCE, ctrl_set_reference },
  { VP8_SET_POSTPROC, ctrl_set_previewpp },
  { VP9E_SET_ROI_MAP, ctrl_set_roi_map },
  { VP8E_SET_ACTIVEMAP, ctrl_set_active_map },
  { VP8E_SET_SCALEMODE, ctrl_set_scale_mode },
  { VP8E_SET_CPUUSED, ctrl_set_cpuused },
  { VP8E_SET_ENABLEAUTOALTREF, ctrl_set_enable_auto_alt_ref },
  { VP8E_SET_SHARPNESS, ctrl_set_sharpness },
  { VP8E_SET_STATIC_THRESHOLD, ctrl_set_static_thresh },
  { VP9E_SET_TILE_COLUMNS, ctrl_set_tile_columns },
  { VP9E_SET_TILE_ROWS, ctrl_set_tile_rows },
  { VP9E_SET_TPL, ctrl_set_tpl_model },
  { VP9E_SET_KEY_FRAME_FILTERING, ctrl_set_keyframe_filtering },
  { VP8E_SET_ARNR_MAXFRAMES, ctrl_set_arnr_max_frames },
  { VP8E_SET_ARNR_STRENGTH, ctrl_set_arnr_strength },
  { VP8E_SET_ARNR_TYPE, ctrl_set_arnr_type },
  { VP8E_SET_TUNING, ctrl_set_tuning },
  { VP8E_SET_CQ_LEVEL, ctrl_set_cq_level },
  { VP8E_SET_MAX_INTRA_BITRATE_PCT, ctrl_set_rc_max_intra_bitrate_pct },
  { VP9E_SET_MAX_INTER_BITRATE_PCT, ctrl_set_rc_max_inter_bitrate_pct },
  { VP9E_SET_GF_CBR_BOOST_PCT, ctrl_set_rc_gf_cbr_boost_pct },
  { VP9E_SET_LOSSLESS, ctrl_set_lossless },
  { VP9E_SET_FRAME_PARALLEL_DECODING, ctrl_set_frame_parallel_decoding_mode },
  { VP9E_SET_AQ_MODE, ctrl_set_aq_mode },
  { VP9E_SET_ALT_REF_AQ, ctrl_set_alt_ref_aq },
  { VP9E_SET_FRAME_PERIODIC_BOOST, ctrl_set_frame_periodic_boost },
  { VP9E_SET_SVC, ctrl_set_svc },
  { VP9E_SET_SVC_PARAMETERS, ctrl_set_svc_parameters },
  { VP9E_REGISTER_CX_CALLBACK, ctrl_register_cx_callback },
  { VP9E_SET_SVC_LAYER_ID, ctrl_set_svc_layer_id },
  { VP9E_SET_TUNE_CONTENT, ctrl_set_tune_content },
  { VP9E_SET_COLOR_SPACE, ctrl_set_color_space },
  { VP9E_SET_COLOR_RANGE, ctrl_set_color_range },
  { VP9E_SET_NOISE_SENSITIVITY, ctrl_set_noise_sensitivity },
  { VP9E_SET_MIN_GF_INTERVAL, ctrl_set_min_gf_interval },
  { VP9E_SET_MAX_GF_INTERVAL, ctrl_set_max_gf_interval },
  { VP9E_SET_SVC_REF_FRAME_CONFIG, ctrl_set_svc_ref_frame_config },
  { VP9E_SET_RENDER_SIZE, ctrl_set_render_size },
  { VP9E_SET_TARGET_LEVEL, ctrl_set_target_level },
  { VP9E_SET_ROW_MT, ctrl_set_row_mt },
  { VP9E_SET_POSTENCODE_DROP, ctrl_set_postencode_drop },
  { VP9E_SET_DISABLE_OVERSHOOT_MAXQ_CBR, ctrl_set_disable_overshoot_maxq_cbr },
  { VP9E_ENABLE_MOTION_VECTOR_UNIT_TEST, ctrl_enable_motion_vector_unit_test },
  { VP9E_SET_SVC_INTER_LAYER_PRED, ctrl_set_svc_inter_layer_pred },
  { VP9E_SET_SVC_FRAME_DROP_LAYER, ctrl_set_svc_frame_drop_layer },
  { VP9E_SET_SVC_GF_TEMPORAL_REF, ctrl_set_svc_gf_temporal_ref },
  { VP9E_SET_SVC_SPATIAL_LAYER_SYNC, ctrl_set_svc_spatial_layer_sync },
  { VP9E_SET_DELTA_Q_UV, ctrl_set_delta_q_uv },
  { VP9E_SET_DISABLE_LOOPFILTER, ctrl_set_disable_loopfilter },
  { VP9E_SET_RTC_EXTERNAL_RATECTRL, ctrl_set_rtc_external_ratectrl },
  { VP9E_SET_EXTERNAL_RATE_CONTROL, ctrl_set_external_rate_control },
  { VP9E_SET_QUANTIZER_ONE_PASS, ctrl_set_quantizer_one_pass },

  // Getters
  { VP8E_GET_LAST_QUANTIZER, ctrl_get_quantizer },
  { VP8E_GET_LAST_QUANTIZER_64, ctrl_get_quantizer64 },
  { VP9E_GET_LAST_QUANTIZER_SVC_LAYERS, ctrl_get_quantizer_svc_layers },
  { VP9E_GET_LOOPFILTER_LEVEL, ctrl_get_loopfilter_level },
  { VP9_GET_REFERENCE, ctrl_get_reference },
  { VP9E_GET_SVC_LAYER_ID, ctrl_get_svc_layer_id },
  { VP9E_GET_ACTIVEMAP, ctrl_get_active_map },
  { VP9E_GET_LEVEL, ctrl_get_level },
  { VP9E_GET_SVC_REF_FRAME_CONFIG, ctrl_get_svc_ref_frame_config },

  { -1, NULL },
};

static vpx_codec_enc_cfg_map_t encoder_usage_cfg_map[] = {
  { 0,
    {
        // NOLINT
        0,  // g_usage (unused)
        8,  // g_threads
        0,  // g_profile

        320,         // g_width
        240,         // g_height
        VPX_BITS_8,  // g_bit_depth
        8,           // g_input_bit_depth

        { 1, 30 },  // g_timebase

        0,  // g_error_resilient

        VPX_RC_ONE_PASS,  // g_pass

        25,  // g_lag_in_frames

        0,   // rc_dropframe_thresh
        0,   // rc_resize_allowed
        0,   // rc_scaled_width
        0,   // rc_scaled_height
        60,  // rc_resize_down_thresh
        30,  // rc_resize_up_thresh

        VPX_VBR,      // rc_end_usage
        { NULL, 0 },  // rc_twopass_stats_in
        { NULL, 0 },  // rc_firstpass_mb_stats_in
        256,          // rc_target_bitrate
        0,            // rc_min_quantizer
        63,           // rc_max_quantizer
        25,           // rc_undershoot_pct
        25,           // rc_overshoot_pct

        6000,  // rc_max_buffer_size
        4000,  // rc_buffer_initial_size
        5000,  // rc_buffer_optimal_size

        50,    // rc_two_pass_vbrbias
        0,     // rc_two_pass_vbrmin_section
        2000,  // rc_two_pass_vbrmax_section
        0,     // rc_2pass_vbr_corpus_complexity (non 0 for corpus vbr)

        // keyframing settings (kf)
        VPX_KF_AUTO,  // g_kfmode
        0,            // kf_min_dist
        128,          // kf_max_dist

        VPX_SS_DEFAULT_LAYERS,  // ss_number_layers
        { 0 },
        { 0 },     // ss_target_bitrate
        1,         // ts_number_layers
        { 0 },     // ts_target_bitrate
        { 0 },     // ts_rate_decimator
        0,         // ts_periodicity
        { 0 },     // ts_layer_id
        { 0 },     // layer_target_bitrate
        0,         // temporal_layering_mode
        0,         // use_vizier_rc_params
        { 1, 1 },  // active_wq_factor
        { 1, 1 },  // err_per_mb_factor
        { 1, 1 },  // sr_default_decay_limit
        { 1, 1 },  // sr_diff_factor
        { 1, 1 },  // kf_err_per_mb_factor
        { 1, 1 },  // kf_frame_min_boost_factor
        { 1, 1 },  // kf_frame_max_boost_first_factor
        { 1, 1 },  // kf_frame_max_boost_subs_factor
        { 1, 1 },  // kf_max_total_boost_factor
        { 1, 1 },  // gf_max_total_boost_factor
        { 1, 1 },  // gf_frame_max_boost_factor
        { 1, 1 },  // zm_factor
        { 1, 1 },  // rd_mult_inter_qp_fac
        { 1, 1 },  // rd_mult_arf_qp_fac
        { 1, 1 },  // rd_mult_key_qp_fac
    } },
};

#ifndef VERSION_STRING
#define VERSION_STRING
#endif
CODEC_INTERFACE(vpx_codec_vp9_cx) = {
  "WebM Project VP9 Encoder" VERSION_STRING,
  VPX_CODEC_INTERNAL_ABI_VERSION,
#if CONFIG_VP9_HIGHBITDEPTH
  VPX_CODEC_CAP_HIGHBITDEPTH |
#endif
      VPX_CODEC_CAP_ENCODER | VPX_CODEC_CAP_PSNR,  // vpx_codec_caps_t
  encoder_init,                                    // vpx_codec_init_fn_t
  encoder_destroy,                                 // vpx_codec_destroy_fn_t
  encoder_ctrl_maps,                               // vpx_codec_ctrl_fn_map_t
  {
      // NOLINT
      NULL,  // vpx_codec_peek_si_fn_t
      NULL,  // vpx_codec_get_si_fn_t
      NULL,  // vpx_codec_decode_fn_t
      NULL,  // vpx_codec_frame_get_fn_t
      NULL   // vpx_codec_set_fb_fn_t
  },
  {
      // NOLINT
      1,                           // 1 cfg map
      encoder_usage_cfg_map,       // vpx_codec_enc_cfg_map_t
      encoder_encode,              // vpx_codec_encode_fn_t
      encoder_get_cxdata,          // vpx_codec_get_cx_data_fn_t
      encoder_set_config,          // vpx_codec_enc_config_set_fn_t
      encoder_get_global_headers,  // vpx_codec_get_global_headers_fn_t
      encoder_get_preview,         // vpx_codec_get_preview_frame_fn_t
      NULL,                        // vpx_codec_enc_mr_get_mem_loc_fn_t
      NULL                         // vpx_codec_enc_mr_free_mem_loc_fn_t
  }
};

static vpx_codec_enc_cfg_t get_enc_cfg(int frame_width, int frame_height,
                                       vpx_rational_t frame_rate,
                                       int target_bitrate,
                                       vpx_enc_pass enc_pass) {
  vpx_codec_enc_cfg_t enc_cfg = encoder_usage_cfg_map[0].cfg;
  enc_cfg.g_w = frame_width;
  enc_cfg.g_h = frame_height;
  enc_cfg.rc_target_bitrate = target_bitrate;
  enc_cfg.g_pass = enc_pass;
  // g_timebase is the inverse of frame_rate
  enc_cfg.g_timebase.num = frame_rate.den;
  enc_cfg.g_timebase.den = frame_rate.num;
  return enc_cfg;
}

static vp9_extracfg get_extra_cfg(void) {
  vp9_extracfg extra_cfg = default_extra_cfg;
  return extra_cfg;
}

VP9EncoderConfig vp9_get_encoder_config(int frame_width, int frame_height,
                                        vpx_rational_t frame_rate,
                                        int target_bitrate, int encode_speed,
                                        int target_level,
                                        vpx_enc_pass enc_pass) {
  /* This function will generate the same VP9EncoderConfig used by the
   * vpxenc command given below.
   * The configs in the vpxenc command corresponds to parameters of
   * vp9_get_encoder_config() as follows.
   *
   * WIDTH:   frame_width
   * HEIGHT:  frame_height
   * FPS:     frame_rate
   * BITRATE: target_bitrate
   * CPU_USED:encode_speed
   * TARGET_LEVEL: target_level
   *
   * INPUT, OUTPUT, LIMIT will not affect VP9EncoderConfig
   *
   * vpxenc command:
   * INPUT=bus_cif.y4m
   * OUTPUT=output.webm
   * WIDTH=352
   * HEIGHT=288
   * BITRATE=600
   * FPS=30/1
   * LIMIT=150
   * CPU_USED=0
   * TARGET_LEVEL=0
   * ./vpxenc --limit=$LIMIT --width=$WIDTH --height=$HEIGHT --fps=$FPS
   * --lag-in-frames=25 \
   *  --codec=vp9 --good --cpu-used=CPU_USED --threads=0 --profile=0 \
   *  --min-q=0 --max-q=63 --auto-alt-ref=1 --passes=2 --kf-max-dist=150 \
   *  --kf-min-dist=0 --drop-frame=0 --static-thresh=0 --bias-pct=50 \
   *  --minsection-pct=0 --maxsection-pct=150 --arnr-maxframes=7 --psnr \
   *  --arnr-strength=5 --sharpness=0 --undershoot-pct=100 --overshoot-pct=100 \
   *  --frame-parallel=0 --tile-columns=0 --cpu-used=0 --end-usage=vbr \
   *  --target-bitrate=$BITRATE --target-level=0 -o $OUTPUT $INPUT
   */

  VP9EncoderConfig oxcf;
  vp9_extracfg extra_cfg = get_extra_cfg();
  vpx_codec_enc_cfg_t enc_cfg = get_enc_cfg(
      frame_width, frame_height, frame_rate, target_bitrate, enc_pass);
  set_encoder_config(&oxcf, &enc_cfg, &extra_cfg);

  // These settings are made to match the settings of the vpxenc command.
  oxcf.key_freq = 150;
  oxcf.under_shoot_pct = 100;
  oxcf.over_shoot_pct = 100;
  oxcf.max_threads = 0;
  oxcf.tile_columns = 0;
  oxcf.frame_parallel_decoding_mode = 0;
  oxcf.two_pass_vbrmax_section = 150;
  oxcf.speed = abs(encode_speed);
  oxcf.target_level = target_level;
  return oxcf;
}

#define DUMP_STRUCT_VALUE(fp, structure, value) \
  fprintf(fp, #value " %" PRId64 "\n", (int64_t)(structure)->value)

void vp9_dump_encoder_config(const VP9EncoderConfig *oxcf, FILE *fp) {
  DUMP_STRUCT_VALUE(fp, oxcf, profile);
  DUMP_STRUCT_VALUE(fp, oxcf, bit_depth);
  DUMP_STRUCT_VALUE(fp, oxcf, width);
  DUMP_STRUCT_VALUE(fp, oxcf, height);
  DUMP_STRUCT_VALUE(fp, oxcf, input_bit_depth);
  DUMP_STRUCT_VALUE(fp, oxcf, init_framerate);
  // TODO(angiebird): dump g_timebase
  // TODO(angiebird): dump g_timebase_in_ts

  DUMP_STRUCT_VALUE(fp, oxcf, target_bandwidth);

  DUMP_STRUCT_VALUE(fp, oxcf, noise_sensitivity);
  DUMP_STRUCT_VALUE(fp, oxcf, sharpness);
  DUMP_STRUCT_VALUE(fp, oxcf, speed);
  DUMP_STRUCT_VALUE(fp, oxcf, rc_max_intra_bitrate_pct);
  DUMP_STRUCT_VALUE(fp, oxcf, rc_max_inter_bitrate_pct);
  DUMP_STRUCT_VALUE(fp, oxcf, gf_cbr_boost_pct);

  DUMP_STRUCT_VALUE(fp, oxcf, mode);
  DUMP_STRUCT_VALUE(fp, oxcf, pass);

  // Key Framing Operations
  DUMP_STRUCT_VALUE(fp, oxcf, auto_key);
  DUMP_STRUCT_VALUE(fp, oxcf, key_freq);

  DUMP_STRUCT_VALUE(fp, oxcf, lag_in_frames);

  // ----------------------------------------------------------------
  // DATARATE CONTROL OPTIONS

  // vbr, cbr, constrained quality or constant quality
  DUMP_STRUCT_VALUE(fp, oxcf, rc_mode);

  // buffer targeting aggressiveness
  DUMP_STRUCT_VALUE(fp, oxcf, under_shoot_pct);
  DUMP_STRUCT_VALUE(fp, oxcf, over_shoot_pct);

  // buffering parameters
  // TODO(angiebird): dump tarting_buffer_level_ms
  // TODO(angiebird): dump ptimal_buffer_level_ms
  // TODO(angiebird): dump maximum_buffer_size_ms

  // Frame drop threshold.
  DUMP_STRUCT_VALUE(fp, oxcf, drop_frames_water_mark);

  // controlling quality
  DUMP_STRUCT_VALUE(fp, oxcf, fixed_q);
  DUMP_STRUCT_VALUE(fp, oxcf, worst_allowed_q);
  DUMP_STRUCT_VALUE(fp, oxcf, best_allowed_q);
  DUMP_STRUCT_VALUE(fp, oxcf, cq_level);
  DUMP_STRUCT_VALUE(fp, oxcf, aq_mode);

  // Special handling of Adaptive Quantization for AltRef frames
  DUMP_STRUCT_VALUE(fp, oxcf, alt_ref_aq);

  // Internal frame size scaling.
  DUMP_STRUCT_VALUE(fp, oxcf, resize_mode);
  DUMP_STRUCT_VALUE(fp, oxcf, scaled_frame_width);
  DUMP_STRUCT_VALUE(fp, oxcf, scaled_frame_height);

  // Enable feature to reduce the frame quantization every x frames.
  DUMP_STRUCT_VALUE(fp, oxcf, frame_periodic_boost);

  // two pass datarate control
  DUMP_STRUCT_VALUE(fp, oxcf, two_pass_vbrbias);
  DUMP_STRUCT_VALUE(fp, oxcf, two_pass_vbrmin_section);
  DUMP_STRUCT_VALUE(fp, oxcf, two_pass_vbrmax_section);
  DUMP_STRUCT_VALUE(fp, oxcf, vbr_corpus_complexity);
  // END DATARATE CONTROL OPTIONS
  // ----------------------------------------------------------------

  // Spatial and temporal scalability.
  DUMP_STRUCT_VALUE(fp, oxcf, ss_number_layers);
  DUMP_STRUCT_VALUE(fp, oxcf, ts_number_layers);

  // Bitrate allocation for spatial layers.
  // TODO(angiebird): dump layer_target_bitrate[VPX_MAX_LAYERS]
  // TODO(angiebird): dump ss_target_bitrate[VPX_SS_MAX_LAYERS]
  // TODO(angiebird): dump ss_enable_auto_arf[VPX_SS_MAX_LAYERS]
  // TODO(angiebird): dump ts_rate_decimator[VPX_TS_MAX_LAYERS]

  DUMP_STRUCT_VALUE(fp, oxcf, enable_auto_arf);
  DUMP_STRUCT_VALUE(fp, oxcf, encode_breakout);
  DUMP_STRUCT_VALUE(fp, oxcf, error_resilient_mode);
  DUMP_STRUCT_VALUE(fp, oxcf, frame_parallel_decoding_mode);

  DUMP_STRUCT_VALUE(fp, oxcf, arnr_max_frames);
  DUMP_STRUCT_VALUE(fp, oxcf, arnr_strength);

  DUMP_STRUCT_VALUE(fp, oxcf, min_gf_interval);
  DUMP_STRUCT_VALUE(fp, oxcf, max_gf_interval);

  DUMP_STRUCT_VALUE(fp, oxcf, tile_columns);
  DUMP_STRUCT_VALUE(fp, oxcf, tile_rows);

  DUMP_STRUCT_VALUE(fp, oxcf, enable_tpl_model);

  DUMP_STRUCT_VALUE(fp, oxcf, enable_keyframe_filtering);

  DUMP_STRUCT_VALUE(fp, oxcf, max_threads);

  DUMP_STRUCT_VALUE(fp, oxcf, target_level);

  // TODO(angiebird): dump two_pass_stats_in
  DUMP_STRUCT_VALUE(fp, oxcf, tuning);
  DUMP_STRUCT_VALUE(fp, oxcf, content);
#if CONFIG_VP9_HIGHBITDEPTH
  DUMP_STRUCT_VALUE(fp, oxcf, use_highbitdepth);
#endif
  DUMP_STRUCT_VALUE(fp, oxcf, color_space);
  DUMP_STRUCT_VALUE(fp, oxcf, color_range);
  DUMP_STRUCT_VALUE(fp, oxcf, render_width);
  DUMP_STRUCT_VALUE(fp, oxcf, render_height);
  DUMP_STRUCT_VALUE(fp, oxcf, temporal_layering_mode);

  DUMP_STRUCT_VALUE(fp, oxcf, row_mt);
  DUMP_STRUCT_VALUE(fp, oxcf, motion_vector_unit_test);
  DUMP_STRUCT_VALUE(fp, oxcf, delta_q_uv);
}

FRAME_INFO vp9_get_frame_info(const VP9EncoderConfig *oxcf) {
  FRAME_INFO frame_info;
  int dummy;
  frame_info.frame_width = oxcf->width;
  frame_info.frame_height = oxcf->height;
  frame_info.render_frame_width = oxcf->width;
  frame_info.render_frame_height = oxcf->height;
  frame_info.bit_depth = oxcf->bit_depth;
  vp9_set_mi_size(&frame_info.mi_rows, &frame_info.mi_cols, &dummy,
                  frame_info.frame_width, frame_info.frame_height);
  vp9_set_mb_size(&frame_info.mb_rows, &frame_info.mb_cols, &frame_info.num_mbs,
                  frame_info.mi_rows, frame_info.mi_cols);
  // TODO(angiebird): Figure out how to get subsampling_x/y here
  return frame_info;
}

void vp9_set_first_pass_stats(VP9EncoderConfig *oxcf,
                              const vpx_fixed_buf_t *stats) {
  oxcf->two_pass_stats_in = *stats;
}
