/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/**
 * @file
 * VP9 SVC encoding support via libvpx
 */

#include <assert.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VPX_DISABLE_CTRL_TYPECHECKS 1
#include "../tools_common.h"
#include "./vpx_config.h"
#include "./svc_context.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_encoder.h"
#include "vpx_mem/vpx_mem.h"
#include "vp9/common/vp9_onyxc_int.h"

#ifdef __MINGW32__
#define strtok_r strtok_s
#ifndef MINGW_HAS_SECURE_API
// proto from /usr/x86_64-w64-mingw32/include/sec_api/string_s.h
_CRTIMP char *__cdecl strtok_s(char *str, const char *delim, char **context);
#endif /* MINGW_HAS_SECURE_API */
#endif /* __MINGW32__ */

#ifdef _MSC_VER
#define strdup _strdup
#define strtok_r strtok_s
#endif

#define SVC_REFERENCE_FRAMES 8
#define SUPERFRAME_SLOTS (8)
#define SUPERFRAME_BUFFER_SIZE (SUPERFRAME_SLOTS * sizeof(uint32_t) + 2)

#define MAX_QUANTIZER 63

static const int DEFAULT_SCALE_FACTORS_NUM[VPX_SS_MAX_LAYERS] = { 4, 5, 7, 11,
                                                                  16 };

static const int DEFAULT_SCALE_FACTORS_DEN[VPX_SS_MAX_LAYERS] = { 16, 16, 16,
                                                                  16, 16 };

static const int DEFAULT_SCALE_FACTORS_NUM_2x[VPX_SS_MAX_LAYERS] = { 1, 2, 4 };

static const int DEFAULT_SCALE_FACTORS_DEN_2x[VPX_SS_MAX_LAYERS] = { 4, 4, 4 };

typedef enum {
  QUANTIZER = 0,
  BITRATE,
  SCALE_FACTOR,
  AUTO_ALT_REF,
  ALL_OPTION_TYPES
} LAYER_OPTION_TYPE;

static const int option_max_values[ALL_OPTION_TYPES] = { 63, INT_MAX, INT_MAX,
                                                         1 };

static const int option_min_values[ALL_OPTION_TYPES] = { 0, 0, 1, 0 };

// One encoded frame
typedef struct FrameData {
  void *buf;                     // compressed data buffer
  size_t size;                   // length of compressed data
  vpx_codec_frame_flags_t flags; /**< flags for this frame */
  struct FrameData *next;
} FrameData;

static SvcInternal_t *get_svc_internal(SvcContext *svc_ctx) {
  if (svc_ctx == NULL) return NULL;
  if (svc_ctx->internal == NULL) {
    SvcInternal_t *const si = (SvcInternal_t *)malloc(sizeof(*si));
    if (si != NULL) {
      memset(si, 0, sizeof(*si));
    }
    svc_ctx->internal = si;
  }
  return (SvcInternal_t *)svc_ctx->internal;
}

static const SvcInternal_t *get_const_svc_internal(const SvcContext *svc_ctx) {
  if (svc_ctx == NULL) return NULL;
  return (const SvcInternal_t *)svc_ctx->internal;
}

static VPX_TOOLS_FORMAT_PRINTF(3, 4) int svc_log(SvcContext *svc_ctx,
                                                 SVC_LOG_LEVEL level,
                                                 const char *fmt, ...) {
  char buf[512];
  int retval = 0;
  va_list ap;

  if (level > svc_ctx->log_level) {
    return retval;
  }

  va_start(ap, fmt);
  retval = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  printf("%s", buf);

  return retval;
}

static vpx_codec_err_t extract_option(LAYER_OPTION_TYPE type, char *input,
                                      int *value0, int *value1) {
  if (type == SCALE_FACTOR) {
    *value0 = (int)strtol(input, &input, 10);
    if (*input++ != '/') return VPX_CODEC_INVALID_PARAM;
    *value1 = (int)strtol(input, &input, 10);

    if (*value0 < option_min_values[SCALE_FACTOR] ||
        *value1 < option_min_values[SCALE_FACTOR] ||
        *value0 > option_max_values[SCALE_FACTOR] ||
        *value1 > option_max_values[SCALE_FACTOR] ||
        *value0 > *value1)  // num shouldn't be greater than den
      return VPX_CODEC_INVALID_PARAM;
  } else {
    *value0 = atoi(input);
    if (*value0 < option_min_values[type] || *value0 > option_max_values[type])
      return VPX_CODEC_INVALID_PARAM;
  }
  return VPX_CODEC_OK;
}

static vpx_codec_err_t parse_layer_options_from_string(SvcContext *svc_ctx,
                                                       LAYER_OPTION_TYPE type,
                                                       const char *input,
                                                       int *option0,
                                                       int *option1) {
  int i;
  vpx_codec_err_t res = VPX_CODEC_OK;
  char *input_string;
  char *token;
  const char *delim = ",";
  char *save_ptr;
  int num_layers = svc_ctx->spatial_layers;
  if (type == BITRATE)
    num_layers = svc_ctx->spatial_layers * svc_ctx->temporal_layers;

  if (input == NULL || option0 == NULL ||
      (option1 == NULL && type == SCALE_FACTOR))
    return VPX_CODEC_INVALID_PARAM;

  input_string = strdup(input);
  if (input_string == NULL) return VPX_CODEC_MEM_ERROR;
  token = strtok_r(input_string, delim, &save_ptr);
  for (i = 0; i < num_layers; ++i) {
    if (token != NULL) {
      res = extract_option(type, token, option0 + i, option1 + i);
      if (res != VPX_CODEC_OK) break;
      token = strtok_r(NULL, delim, &save_ptr);
    } else {
      break;
    }
  }
  if (res == VPX_CODEC_OK && i != num_layers) {
    svc_log(svc_ctx, SVC_LOG_ERROR,
            "svc: layer params type: %d    %d values required, "
            "but only %d specified\n",
            type, num_layers, i);
    res = VPX_CODEC_INVALID_PARAM;
  }
  free(input_string);
  return res;
}

/**
 * Parse SVC encoding options
 * Format: encoding-mode=<svc_mode>,layers=<layer_count>
 *         scale-factors=<n1>/<d1>,<n2>/<d2>,...
 *         quantizers=<q1>,<q2>,...
 * svc_mode = [i|ip|alt_ip|gf]
 */
static vpx_codec_err_t parse_options(SvcContext *svc_ctx, const char *options) {
  char *input_string;
  char *option_name;
  char *option_value;
  char *input_ptr = NULL;
  SvcInternal_t *const si = get_svc_internal(svc_ctx);
  vpx_codec_err_t res = VPX_CODEC_OK;
  int i, alt_ref_enabled = 0;

  if (options == NULL) return VPX_CODEC_OK;
  input_string = strdup(options);
  if (input_string == NULL) return VPX_CODEC_MEM_ERROR;

  // parse option name
  option_name = strtok_r(input_string, "=", &input_ptr);
  while (option_name != NULL) {
    // parse option value
    option_value = strtok_r(NULL, " ", &input_ptr);
    if (option_value == NULL) {
      svc_log(svc_ctx, SVC_LOG_ERROR, "option missing value: %s\n",
              option_name);
      res = VPX_CODEC_INVALID_PARAM;
      break;
    }
    if (strcmp("spatial-layers", option_name) == 0) {
      svc_ctx->spatial_layers = atoi(option_value);
    } else if (strcmp("temporal-layers", option_name) == 0) {
      svc_ctx->temporal_layers = atoi(option_value);
    } else if (strcmp("scale-factors", option_name) == 0) {
      res = parse_layer_options_from_string(svc_ctx, SCALE_FACTOR, option_value,
                                            si->svc_params.scaling_factor_num,
                                            si->svc_params.scaling_factor_den);
      if (res != VPX_CODEC_OK) break;
    } else if (strcmp("max-quantizers", option_name) == 0) {
      res =
          parse_layer_options_from_string(svc_ctx, QUANTIZER, option_value,
                                          si->svc_params.max_quantizers, NULL);
      if (res != VPX_CODEC_OK) break;
    } else if (strcmp("min-quantizers", option_name) == 0) {
      res =
          parse_layer_options_from_string(svc_ctx, QUANTIZER, option_value,
                                          si->svc_params.min_quantizers, NULL);
      if (res != VPX_CODEC_OK) break;
    } else if (strcmp("auto-alt-refs", option_name) == 0) {
      res = parse_layer_options_from_string(svc_ctx, AUTO_ALT_REF, option_value,
                                            si->enable_auto_alt_ref, NULL);
      if (res != VPX_CODEC_OK) break;
    } else if (strcmp("bitrates", option_name) == 0) {
      res = parse_layer_options_from_string(svc_ctx, BITRATE, option_value,
                                            si->bitrates, NULL);
      if (res != VPX_CODEC_OK) break;
    } else if (strcmp("multi-frame-contexts", option_name) == 0) {
      si->use_multiple_frame_contexts = atoi(option_value);
    } else {
      svc_log(svc_ctx, SVC_LOG_ERROR, "invalid option: %s\n", option_name);
      res = VPX_CODEC_INVALID_PARAM;
      break;
    }
    option_name = strtok_r(NULL, "=", &input_ptr);
  }
  free(input_string);

  for (i = 0; i < svc_ctx->spatial_layers; ++i) {
    if (si->svc_params.max_quantizers[i] > MAX_QUANTIZER ||
        si->svc_params.max_quantizers[i] < 0 ||
        si->svc_params.min_quantizers[i] > si->svc_params.max_quantizers[i] ||
        si->svc_params.min_quantizers[i] < 0)
      res = VPX_CODEC_INVALID_PARAM;
  }

  if (si->use_multiple_frame_contexts &&
      (svc_ctx->spatial_layers > 3 ||
       svc_ctx->spatial_layers * svc_ctx->temporal_layers > 4))
    res = VPX_CODEC_INVALID_PARAM;

  for (i = 0; i < svc_ctx->spatial_layers; ++i)
    alt_ref_enabled += si->enable_auto_alt_ref[i];
  if (alt_ref_enabled > REF_FRAMES - svc_ctx->spatial_layers) {
    svc_log(svc_ctx, SVC_LOG_ERROR,
            "svc: auto alt ref: Maxinum %d(REF_FRAMES - layers) layers could"
            "enabled auto alt reference frame, but %d layers are enabled\n",
            REF_FRAMES - svc_ctx->spatial_layers, alt_ref_enabled);
    res = VPX_CODEC_INVALID_PARAM;
  }

  return res;
}

vpx_codec_err_t vpx_svc_set_options(SvcContext *svc_ctx, const char *options) {
  SvcInternal_t *const si = get_svc_internal(svc_ctx);
  if (svc_ctx == NULL || options == NULL || si == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  strncpy(si->options, options, sizeof(si->options) - 1);
  si->options[sizeof(si->options) - 1] = '\0';
  return VPX_CODEC_OK;
}

static vpx_codec_err_t assign_layer_bitrates(
    const SvcContext *svc_ctx, vpx_codec_enc_cfg_t *const enc_cfg) {
  int i;
  const SvcInternal_t *const si = get_const_svc_internal(svc_ctx);
  int sl, tl, spatial_layer_target;

  if (svc_ctx->temporal_layering_mode != 0) {
    if (si->bitrates[0] != 0) {
      unsigned int total_bitrate = 0;
      for (sl = 0; sl < svc_ctx->spatial_layers; ++sl) {
        total_bitrate += si->bitrates[sl * svc_ctx->temporal_layers +
                                      svc_ctx->temporal_layers - 1];
        for (tl = 0; tl < svc_ctx->temporal_layers; ++tl) {
          enc_cfg->ss_target_bitrate[sl * svc_ctx->temporal_layers] +=
              (unsigned int)si->bitrates[sl * svc_ctx->temporal_layers + tl];
          enc_cfg->layer_target_bitrate[sl * svc_ctx->temporal_layers + tl] =
              si->bitrates[sl * svc_ctx->temporal_layers + tl];
          if (tl > 0 && (si->bitrates[sl * svc_ctx->temporal_layers + tl] <=
                         si->bitrates[sl * svc_ctx->temporal_layers + tl - 1]))
            return VPX_CODEC_INVALID_PARAM;
        }
      }
      if (total_bitrate != enc_cfg->rc_target_bitrate)
        return VPX_CODEC_INVALID_PARAM;
    } else {
      float total = 0;
      float alloc_ratio[VPX_MAX_LAYERS] = { 0 };

      for (sl = 0; sl < svc_ctx->spatial_layers; ++sl) {
        if (si->svc_params.scaling_factor_den[sl] > 0) {
          alloc_ratio[sl] = (float)(pow(2, sl));
          total += alloc_ratio[sl];
        }
      }

      for (sl = 0; sl < svc_ctx->spatial_layers; ++sl) {
        enc_cfg->ss_target_bitrate[sl] = spatial_layer_target =
            (unsigned int)(enc_cfg->rc_target_bitrate * alloc_ratio[sl] /
                           total);
        if (svc_ctx->temporal_layering_mode == 3) {
          enc_cfg->layer_target_bitrate[sl * svc_ctx->temporal_layers] =
              (spatial_layer_target * 6) / 10;  // 60%
          enc_cfg->layer_target_bitrate[sl * svc_ctx->temporal_layers + 1] =
              (spatial_layer_target * 8) / 10;  // 80%
          enc_cfg->layer_target_bitrate[sl * svc_ctx->temporal_layers + 2] =
              spatial_layer_target;
        } else if (svc_ctx->temporal_layering_mode == 2 ||
                   svc_ctx->temporal_layering_mode == 1) {
          enc_cfg->layer_target_bitrate[sl * svc_ctx->temporal_layers] =
              spatial_layer_target * 2 / 3;
          enc_cfg->layer_target_bitrate[sl * svc_ctx->temporal_layers + 1] =
              spatial_layer_target;
        } else {
          // User should explicitly assign bitrates in this case.
          assert(0);
        }
      }
    }
  } else {
    if (si->bitrates[0] != 0) {
      unsigned int total_bitrate = 0;
      for (i = 0; i < svc_ctx->spatial_layers; ++i) {
        enc_cfg->ss_target_bitrate[i] = (unsigned int)si->bitrates[i];
        enc_cfg->layer_target_bitrate[i] = (unsigned int)si->bitrates[i];
        total_bitrate += si->bitrates[i];
      }
      if (total_bitrate != enc_cfg->rc_target_bitrate)
        return VPX_CODEC_INVALID_PARAM;
    } else {
      float total = 0;
      float alloc_ratio[VPX_MAX_LAYERS] = { 0 };

      for (i = 0; i < svc_ctx->spatial_layers; ++i) {
        if (si->svc_params.scaling_factor_den[i] > 0) {
          alloc_ratio[i] = (float)(si->svc_params.scaling_factor_num[i] * 1.0 /
                                   si->svc_params.scaling_factor_den[i]);

          alloc_ratio[i] *= alloc_ratio[i];
          total += alloc_ratio[i];
        }
      }
      for (i = 0; i < VPX_SS_MAX_LAYERS; ++i) {
        if (total > 0) {
          enc_cfg->layer_target_bitrate[i] =
              (unsigned int)(enc_cfg->rc_target_bitrate * alloc_ratio[i] /
                             total);
        }
      }
    }
  }
  return VPX_CODEC_OK;
}

vpx_codec_err_t vpx_svc_init(SvcContext *svc_ctx, vpx_codec_ctx_t *codec_ctx,
                             vpx_codec_iface_t *iface,
                             vpx_codec_enc_cfg_t *enc_cfg) {
  vpx_codec_err_t res;
  int sl, tl;
  SvcInternal_t *const si = get_svc_internal(svc_ctx);
  if (svc_ctx == NULL || codec_ctx == NULL || iface == NULL ||
      enc_cfg == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }
  if (si == NULL) return VPX_CODEC_MEM_ERROR;

  si->codec_ctx = codec_ctx;

  si->width = enc_cfg->g_w;
  si->height = enc_cfg->g_h;

  si->kf_dist = enc_cfg->kf_max_dist;

  if (svc_ctx->spatial_layers == 0)
    svc_ctx->spatial_layers = VPX_SS_DEFAULT_LAYERS;
  if (svc_ctx->spatial_layers < 1 ||
      svc_ctx->spatial_layers > VPX_SS_MAX_LAYERS) {
    svc_log(svc_ctx, SVC_LOG_ERROR, "spatial layers: invalid value: %d\n",
            svc_ctx->spatial_layers);
    return VPX_CODEC_INVALID_PARAM;
  }

  // Note: temporal_layering_mode only applies to one-pass CBR
  // si->svc_params.temporal_layering_mode = svc_ctx->temporal_layering_mode;
  if (svc_ctx->temporal_layering_mode == 3) {
    svc_ctx->temporal_layers = 3;
  } else if (svc_ctx->temporal_layering_mode == 2 ||
             svc_ctx->temporal_layering_mode == 1) {
    svc_ctx->temporal_layers = 2;
  }

  for (sl = 0; sl < VPX_SS_MAX_LAYERS; ++sl) {
    si->svc_params.scaling_factor_num[sl] = DEFAULT_SCALE_FACTORS_NUM[sl];
    si->svc_params.scaling_factor_den[sl] = DEFAULT_SCALE_FACTORS_DEN[sl];
    si->svc_params.speed_per_layer[sl] = svc_ctx->speed;
  }
  if (enc_cfg->rc_end_usage == VPX_CBR && enc_cfg->g_pass == VPX_RC_ONE_PASS &&
      svc_ctx->spatial_layers <= 3) {
    for (sl = 0; sl < svc_ctx->spatial_layers; ++sl) {
      int sl2 = (svc_ctx->spatial_layers == 2) ? sl + 1 : sl;
      si->svc_params.scaling_factor_num[sl] = DEFAULT_SCALE_FACTORS_NUM_2x[sl2];
      si->svc_params.scaling_factor_den[sl] = DEFAULT_SCALE_FACTORS_DEN_2x[sl2];
    }
    if (svc_ctx->spatial_layers == 1) {
      si->svc_params.scaling_factor_num[0] = 1;
      si->svc_params.scaling_factor_den[0] = 1;
    }
  }
  for (tl = 0; tl < svc_ctx->temporal_layers; ++tl) {
    for (sl = 0; sl < svc_ctx->spatial_layers; ++sl) {
      const int i = sl * svc_ctx->temporal_layers + tl;
      si->svc_params.max_quantizers[i] = MAX_QUANTIZER;
      si->svc_params.min_quantizers[i] = 0;
      if (enc_cfg->rc_end_usage == VPX_CBR &&
          enc_cfg->g_pass == VPX_RC_ONE_PASS) {
        si->svc_params.max_quantizers[i] = 56;
        si->svc_params.min_quantizers[i] = 2;
      }
    }
  }

  // Parse aggregate command line options. Options must start with
  // "layers=xx" then followed by other options
  res = parse_options(svc_ctx, si->options);
  if (res != VPX_CODEC_OK) return res;

  if (svc_ctx->spatial_layers < 1) svc_ctx->spatial_layers = 1;
  if (svc_ctx->spatial_layers > VPX_SS_MAX_LAYERS)
    svc_ctx->spatial_layers = VPX_SS_MAX_LAYERS;

  if (svc_ctx->temporal_layers < 1) svc_ctx->temporal_layers = 1;
  if (svc_ctx->temporal_layers > VPX_TS_MAX_LAYERS)
    svc_ctx->temporal_layers = VPX_TS_MAX_LAYERS;

  if (svc_ctx->temporal_layers * svc_ctx->spatial_layers > VPX_MAX_LAYERS) {
    svc_log(
        svc_ctx, SVC_LOG_ERROR,
        "spatial layers * temporal layers (%d) exceeds the maximum number of "
        "allowed layers of %d\n",
        svc_ctx->spatial_layers * svc_ctx->temporal_layers, VPX_MAX_LAYERS);
    return VPX_CODEC_INVALID_PARAM;
  }
  res = assign_layer_bitrates(svc_ctx, enc_cfg);
  if (res != VPX_CODEC_OK) {
    svc_log(svc_ctx, SVC_LOG_ERROR,
            "layer bitrates incorrect: \n"
            "1) spatial layer bitrates should sum up to target \n"
            "2) temporal layer bitrates should be increasing within \n"
            "a spatial layer \n");
    return VPX_CODEC_INVALID_PARAM;
  }

  if (svc_ctx->temporal_layers > 1) {
    int i;
    for (i = 0; i < svc_ctx->temporal_layers; ++i) {
      enc_cfg->ts_target_bitrate[i] =
          enc_cfg->rc_target_bitrate / svc_ctx->temporal_layers;
      enc_cfg->ts_rate_decimator[i] = 1 << (svc_ctx->temporal_layers - 1 - i);
    }
  }

  if (svc_ctx->threads) enc_cfg->g_threads = svc_ctx->threads;

  // Modify encoder configuration
  enc_cfg->ss_number_layers = svc_ctx->spatial_layers;
  enc_cfg->ts_number_layers = svc_ctx->temporal_layers;

  if (enc_cfg->rc_end_usage == VPX_CBR) {
    enc_cfg->rc_resize_allowed = 0;
    enc_cfg->rc_min_quantizer = 2;
    enc_cfg->rc_max_quantizer = 56;
    enc_cfg->rc_undershoot_pct = 50;
    enc_cfg->rc_overshoot_pct = 50;
    enc_cfg->rc_buf_initial_sz = 500;
    enc_cfg->rc_buf_optimal_sz = 600;
    enc_cfg->rc_buf_sz = 1000;
  }

  for (tl = 0; tl < svc_ctx->temporal_layers; ++tl) {
    for (sl = 0; sl < svc_ctx->spatial_layers; ++sl) {
      const int i = sl * svc_ctx->temporal_layers + tl;
      if (enc_cfg->rc_end_usage == VPX_CBR &&
          enc_cfg->g_pass == VPX_RC_ONE_PASS) {
        si->svc_params.max_quantizers[i] = enc_cfg->rc_max_quantizer;
        si->svc_params.min_quantizers[i] = enc_cfg->rc_min_quantizer;
      }
    }
  }

  if (enc_cfg->g_error_resilient == 0 && si->use_multiple_frame_contexts == 0)
    enc_cfg->g_error_resilient = 1;

  // Initialize codec
  vpx_codec_flags_t flags = 0;
  if (svc_ctx->use_psnr) {
    flags |= VPX_CODEC_USE_PSNR;
  }
  res = vpx_codec_enc_init(codec_ctx, iface, enc_cfg, flags);
  if (res != VPX_CODEC_OK) {
    svc_log(svc_ctx, SVC_LOG_ERROR, "svc_enc_init error\n");
    return res;
  }
  if (svc_ctx->spatial_layers > 1 || svc_ctx->temporal_layers > 1) {
    vpx_codec_control(codec_ctx, VP9E_SET_SVC, 1);
    vpx_codec_control(codec_ctx, VP9E_SET_SVC_PARAMETERS, &si->svc_params);
  }
  return VPX_CODEC_OK;
}

/**
 * Encode a frame into multiple layers
 * Create a superframe containing the individual layers
 */
vpx_codec_err_t vpx_svc_encode(SvcContext *svc_ctx, vpx_codec_ctx_t *codec_ctx,
                               struct vpx_image *rawimg, vpx_codec_pts_t pts,
                               int64_t duration, int deadline) {
  vpx_codec_err_t res;
  SvcInternal_t *const si = get_svc_internal(svc_ctx);
  if (svc_ctx == NULL || codec_ctx == NULL || si == NULL) {
    return VPX_CODEC_INVALID_PARAM;
  }

  res =
      vpx_codec_encode(codec_ctx, rawimg, pts, (uint32_t)duration, 0, deadline);
  if (res != VPX_CODEC_OK) {
    return res;
  }

  return VPX_CODEC_OK;
}

static double calc_psnr(double d) {
  if (d == 0) return 100;
  return -10.0 * log(d) / log(10.0);
}

// dump accumulated statistics and reset accumulated values
void vpx_svc_dump_statistics(SvcContext *svc_ctx) {
  int i, j;
  uint32_t bytes_total = 0;
  double scale[COMPONENTS];
  double psnr[COMPONENTS];
  double mse[COMPONENTS];
  double y_scale;

  SvcInternal_t *const si = get_svc_internal(svc_ctx);
  if (svc_ctx == NULL || si == NULL) return;

  svc_log(svc_ctx, SVC_LOG_INFO, "\n");
  for (i = 0; i < svc_ctx->spatial_layers; ++i) {
    svc_log(svc_ctx, SVC_LOG_INFO,
            "Layer %d Average PSNR=[%2.3f, %2.3f, %2.3f, %2.3f], Bytes=[%u], "
            "Number_of_frames %d \n",
            i, si->psnr_sum[i][0] / si->number_of_frames[i],
            si->psnr_sum[i][1] / si->number_of_frames[i],
            si->psnr_sum[i][2] / si->number_of_frames[i],
            si->psnr_sum[i][3] / si->number_of_frames[i], si->bytes_sum[i],
            si->number_of_frames[i]);
    // the following psnr calculation is deduced from ffmpeg.c#print_report
    y_scale = si->width * si->height * 255.0 * 255.0 * si->number_of_frames[i];
    scale[1] = y_scale;
    scale[2] = scale[3] = y_scale / 4;  // U or V
    scale[0] = y_scale * 1.5;           // total

    for (j = 0; j < COMPONENTS; j++) {
      psnr[j] = calc_psnr(si->sse_sum[i][j] / scale[j]);
      mse[j] = si->sse_sum[i][j] * 255.0 * 255.0 / scale[j];
    }
    svc_log(svc_ctx, SVC_LOG_INFO,
            "Layer %d Overall PSNR=[%2.3f, %2.3f, %2.3f, %2.3f]\n", i, psnr[0],
            psnr[1], psnr[2], psnr[3]);
    svc_log(svc_ctx, SVC_LOG_INFO,
            "Layer %d Overall MSE=[%2.3f, %2.3f, %2.3f, %2.3f]\n", i, mse[0],
            mse[1], mse[2], mse[3]);

    bytes_total += si->bytes_sum[i];
  }
  // Clear sums for next time.
  for (i = 0; i < svc_ctx->spatial_layers; ++i) {
    si->bytes_sum[i] = 0;
    si->number_of_frames[i] = 0;
    for (j = 0; j < COMPONENTS; ++j) {
      si->psnr_sum[i][j] = 0;
      si->sse_sum[i][j] = 0;
    }
  }

  svc_log(svc_ctx, SVC_LOG_INFO, "Total Bytes=[%u]\n", bytes_total);
}

void vpx_svc_release(SvcContext *svc_ctx) {
  SvcInternal_t *si;
  if (svc_ctx == NULL) return;
  // do not use get_svc_internal as it will unnecessarily allocate an
  // SvcInternal_t if it was not already allocated
  si = (SvcInternal_t *)svc_ctx->internal;
  if (si != NULL) {
    free(si);
    svc_ctx->internal = NULL;
  }
}
