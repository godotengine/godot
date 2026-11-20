/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*
 * This is an example demonstrating how to implement a multi-layer
 * VP9 encoding scheme based on spatial scalability for video applications
 * that benefit from a scalable bitstream.
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../args.h"
#include "../tools_common.h"
#include "../video_writer.h"

#include "../vpx_ports/bitops.h"
#include "../vpx_ports/vpx_timer.h"
#include "./svc_context.h"
#include "vpx/vp8cx.h"
#include "vpx/vpx_decoder.h"
#include "vpx/vpx_encoder.h"
#include "../vpxstats.h"
#include "./y4minput.h"

#define OUTPUT_FRAME_STATS 0
#define OUTPUT_RC_STATS 1

#define SIMULCAST_MODE 0

static const arg_def_t outputfile =
    ARG_DEF("o", "output", 1, "Output filename");
static const arg_def_t skip_frames_arg =
    ARG_DEF("s", "skip-frames", 1, "input frames to skip");
static const arg_def_t frames_arg =
    ARG_DEF("f", "frames", 1, "number of frames to encode");
static const arg_def_t threads_arg =
    ARG_DEF("th", "threads", 1, "number of threads to use");
#if OUTPUT_RC_STATS
static const arg_def_t output_rc_stats_arg =
    ARG_DEF("rcstat", "output_rc_stats", 1, "output rc stats");
#endif
static const arg_def_t width_arg = ARG_DEF("w", "width", 1, "source width");
static const arg_def_t height_arg = ARG_DEF("h", "height", 1, "source height");
static const arg_def_t timebase_arg =
    ARG_DEF("t", "timebase", 1, "timebase (num/den)");
static const arg_def_t bitrate_arg = ARG_DEF(
    "b", "target-bitrate", 1, "encoding bitrate, in kilobits per second");
static const arg_def_t spatial_layers_arg =
    ARG_DEF("sl", "spatial-layers", 1, "number of spatial SVC layers");
static const arg_def_t temporal_layers_arg =
    ARG_DEF("tl", "temporal-layers", 1, "number of temporal SVC layers");
static const arg_def_t temporal_layering_mode_arg =
    ARG_DEF("tlm", "temporal-layering-mode", 1,
            "temporal layering scheme."
            "VP9E_TEMPORAL_LAYERING_MODE");
static const arg_def_t kf_dist_arg =
    ARG_DEF("k", "kf-dist", 1, "number of frames between keyframes");
static const arg_def_t scale_factors_arg =
    ARG_DEF("r", "scale-factors", 1, "scale factors (lowest to highest layer)");
static const arg_def_t min_q_arg =
    ARG_DEF(NULL, "min-q", 1, "Minimum quantizer");
static const arg_def_t max_q_arg =
    ARG_DEF(NULL, "max-q", 1, "Maximum quantizer");
static const arg_def_t min_bitrate_arg =
    ARG_DEF(NULL, "min-bitrate", 1, "Minimum bitrate");
static const arg_def_t max_bitrate_arg =
    ARG_DEF(NULL, "max-bitrate", 1, "Maximum bitrate");
static const arg_def_t lag_in_frame_arg =
    ARG_DEF(NULL, "lag-in-frames", 1,
            "Number of frame to input before "
            "generating any outputs");
static const arg_def_t rc_end_usage_arg =
    ARG_DEF(NULL, "rc-end-usage", 1, "0 - 3: VBR, CBR, CQ, Q");
static const arg_def_t speed_arg =
    ARG_DEF("sp", "speed", 1, "speed configuration");
static const arg_def_t aqmode_arg =
    ARG_DEF("aq", "aqmode", 1, "aq-mode off/on");
static const arg_def_t bitrates_arg =
    ARG_DEF("bl", "bitrates", 1, "bitrates[sl * num_tl + tl]");
static const arg_def_t dropframe_thresh_arg =
    ARG_DEF(NULL, "drop-frame", 1, "Temporal resampling threshold (buf %)");
static const arg_def_t psnr_arg =
    ARG_DEF(NULL, "psnr", 1, "Enable PSNR computation and statistics");
static const struct arg_enum_list tune_content_enum[] = {
  { "default", VP9E_CONTENT_DEFAULT },
  { "screen", VP9E_CONTENT_SCREEN },
  { "film", VP9E_CONTENT_FILM },
  { NULL, 0 }
};

static const arg_def_t tune_content_arg = ARG_DEF_ENUM(
    NULL, "tune-content", 1, "Tune content type", tune_content_enum);
static const arg_def_t inter_layer_pred_arg = ARG_DEF(
    NULL, "inter-layer-pred", 1, "0 - 3: On, Off, Key-frames, Constrained");

#if CONFIG_VP9_HIGHBITDEPTH
static const struct arg_enum_list bitdepth_enum[] = {
  { "8", VPX_BITS_8 }, { "10", VPX_BITS_10 }, { "12", VPX_BITS_12 }, { NULL, 0 }
};

static const arg_def_t bitdepth_arg = ARG_DEF_ENUM(
    "d", "bit-depth", 1, "Bit depth for codec 8, 10 or 12. ", bitdepth_enum);
#endif  // CONFIG_VP9_HIGHBITDEPTH

static const arg_def_t *svc_args[] = { &frames_arg,
                                       &outputfile,
                                       &width_arg,
                                       &height_arg,
                                       &timebase_arg,
                                       &bitrate_arg,
                                       &skip_frames_arg,
                                       &spatial_layers_arg,
                                       &kf_dist_arg,
                                       &scale_factors_arg,
                                       &min_q_arg,
                                       &max_q_arg,
                                       &min_bitrate_arg,
                                       &max_bitrate_arg,
                                       &temporal_layers_arg,
                                       &temporal_layering_mode_arg,
                                       &lag_in_frame_arg,
                                       &threads_arg,
                                       &aqmode_arg,
#if OUTPUT_RC_STATS
                                       &output_rc_stats_arg,
#endif

#if CONFIG_VP9_HIGHBITDEPTH
                                       &bitdepth_arg,
#endif
                                       &speed_arg,
                                       &rc_end_usage_arg,
                                       &bitrates_arg,
                                       &dropframe_thresh_arg,
                                       &tune_content_arg,
                                       &inter_layer_pred_arg,
                                       &psnr_arg,
                                       NULL };

static const uint32_t default_frames_to_skip = 0;
static const uint32_t default_frames_to_code = 60 * 60;
static const uint32_t default_width = 1920;
static const uint32_t default_height = 1080;
static const uint32_t default_timebase_num = 1;
static const uint32_t default_timebase_den = 60;
static const uint32_t default_bitrate = 1000;
static const uint32_t default_spatial_layers = 5;
static const uint32_t default_temporal_layers = 1;
static const uint32_t default_kf_dist = 100;
static const uint32_t default_temporal_layering_mode = 0;
static const uint32_t default_output_rc_stats = 0;
static const int32_t default_speed = -1;    // -1 means use library default.
static const uint32_t default_threads = 0;  // zero means use library default.

typedef struct {
  const char *output_filename;
  uint32_t frames_to_code;
  uint32_t frames_to_skip;
  struct VpxInputContext input_ctx;
  stats_io_t rc_stats;
  int tune_content;
  int inter_layer_pred;
} AppInput;

static const char *exec_name;

void usage_exit(void) {
  fprintf(stderr, "Usage: %s <options> input_filename -o output_filename\n",
          exec_name);
  fprintf(stderr, "Options:\n");
  arg_show_usage(stderr, svc_args);
  exit(EXIT_FAILURE);
}

static void parse_command_line(int argc, const char **argv_,
                               AppInput *app_input, SvcContext *svc_ctx,
                               vpx_codec_enc_cfg_t *enc_cfg) {
  struct arg arg;
  char **argv = NULL;
  char **argi = NULL;
  char **argj = NULL;
  vpx_codec_err_t res;
  unsigned int min_bitrate = 0;
  unsigned int max_bitrate = 0;
  char string_options[1024] = { 0 };

  // initialize SvcContext with parameters that will be passed to vpx_svc_init
  svc_ctx->log_level = SVC_LOG_DEBUG;
  svc_ctx->spatial_layers = default_spatial_layers;
  svc_ctx->temporal_layers = default_temporal_layers;
  svc_ctx->temporal_layering_mode = default_temporal_layering_mode;
#if OUTPUT_RC_STATS
  svc_ctx->output_rc_stat = default_output_rc_stats;
#endif
  svc_ctx->speed = default_speed;
  svc_ctx->threads = default_threads;
  svc_ctx->use_psnr = 0;

  // start with default encoder configuration
  res = vpx_codec_enc_config_default(vpx_codec_vp9_cx(), enc_cfg, 0);
  if (res) {
    die("Failed to get config: %s\n", vpx_codec_err_to_string(res));
  }
  // update enc_cfg with app default values
  enc_cfg->g_w = default_width;
  enc_cfg->g_h = default_height;
  enc_cfg->g_timebase.num = default_timebase_num;
  enc_cfg->g_timebase.den = default_timebase_den;
  enc_cfg->rc_target_bitrate = default_bitrate;
  enc_cfg->kf_min_dist = default_kf_dist;
  enc_cfg->kf_max_dist = default_kf_dist;
  enc_cfg->rc_end_usage = VPX_CQ;

  // initialize AppInput with default values
  app_input->frames_to_code = default_frames_to_code;
  app_input->frames_to_skip = default_frames_to_skip;

  // process command line options
  argv = argv_dup(argc - 1, argv_ + 1);
  if (!argv) {
    fprintf(stderr, "Error allocating argument list\n");
    exit(EXIT_FAILURE);
  }
  for (argi = argj = argv; (*argj = *argi); argi += arg.argv_step) {
    arg.argv_step = 1;

    if (arg_match(&arg, &frames_arg, argi)) {
      app_input->frames_to_code = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &outputfile, argi)) {
      app_input->output_filename = arg.val;
    } else if (arg_match(&arg, &width_arg, argi)) {
      enc_cfg->g_w = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &height_arg, argi)) {
      enc_cfg->g_h = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &timebase_arg, argi)) {
      enc_cfg->g_timebase = arg_parse_rational(&arg);
    } else if (arg_match(&arg, &bitrate_arg, argi)) {
      enc_cfg->rc_target_bitrate = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &skip_frames_arg, argi)) {
      app_input->frames_to_skip = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &spatial_layers_arg, argi)) {
      svc_ctx->spatial_layers = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &temporal_layers_arg, argi)) {
      svc_ctx->temporal_layers = arg_parse_uint(&arg);
#if OUTPUT_RC_STATS
    } else if (arg_match(&arg, &output_rc_stats_arg, argi)) {
      svc_ctx->output_rc_stat = arg_parse_uint(&arg);
#endif
    } else if (arg_match(&arg, &speed_arg, argi)) {
      svc_ctx->speed = arg_parse_uint(&arg);
      if (svc_ctx->speed > 9) {
        warn("Mapping speed %d to speed 9.\n", svc_ctx->speed);
      }
    } else if (arg_match(&arg, &aqmode_arg, argi)) {
      svc_ctx->aqmode = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &threads_arg, argi)) {
      svc_ctx->threads = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &temporal_layering_mode_arg, argi)) {
      svc_ctx->temporal_layering_mode = enc_cfg->temporal_layering_mode =
          arg_parse_int(&arg);
      if (svc_ctx->temporal_layering_mode) {
        enc_cfg->g_error_resilient = 1;
      }
    } else if (arg_match(&arg, &kf_dist_arg, argi)) {
      enc_cfg->kf_min_dist = arg_parse_uint(&arg);
      enc_cfg->kf_max_dist = enc_cfg->kf_min_dist;
    } else if (arg_match(&arg, &scale_factors_arg, argi)) {
      strncat(string_options, " scale-factors=",
              sizeof(string_options) - strlen(string_options) - 1);
      strncat(string_options, arg.val,
              sizeof(string_options) - strlen(string_options) - 1);
    } else if (arg_match(&arg, &bitrates_arg, argi)) {
      strncat(string_options, " bitrates=",
              sizeof(string_options) - strlen(string_options) - 1);
      strncat(string_options, arg.val,
              sizeof(string_options) - strlen(string_options) - 1);
    } else if (arg_match(&arg, &min_q_arg, argi)) {
      strncat(string_options, " min-quantizers=",
              sizeof(string_options) - strlen(string_options) - 1);
      strncat(string_options, arg.val,
              sizeof(string_options) - strlen(string_options) - 1);
    } else if (arg_match(&arg, &max_q_arg, argi)) {
      strncat(string_options, " max-quantizers=",
              sizeof(string_options) - strlen(string_options) - 1);
      strncat(string_options, arg.val,
              sizeof(string_options) - strlen(string_options) - 1);
    } else if (arg_match(&arg, &min_bitrate_arg, argi)) {
      min_bitrate = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &max_bitrate_arg, argi)) {
      max_bitrate = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &lag_in_frame_arg, argi)) {
      enc_cfg->g_lag_in_frames = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &rc_end_usage_arg, argi)) {
      enc_cfg->rc_end_usage = arg_parse_uint(&arg);
#if CONFIG_VP9_HIGHBITDEPTH
    } else if (arg_match(&arg, &bitdepth_arg, argi)) {
      enc_cfg->g_bit_depth = arg_parse_enum_or_int(&arg);
      switch (enc_cfg->g_bit_depth) {
        case VPX_BITS_8:
          enc_cfg->g_input_bit_depth = 8;
          enc_cfg->g_profile = 0;
          break;
        case VPX_BITS_10:
          enc_cfg->g_input_bit_depth = 10;
          enc_cfg->g_profile = 2;
          break;
        case VPX_BITS_12:
          enc_cfg->g_input_bit_depth = 12;
          enc_cfg->g_profile = 2;
          break;
        default:
          die("Error: Invalid bit depth selected (%d)\n", enc_cfg->g_bit_depth);
      }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    } else if (arg_match(&arg, &dropframe_thresh_arg, argi)) {
      enc_cfg->rc_dropframe_thresh = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &tune_content_arg, argi)) {
      app_input->tune_content = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &inter_layer_pred_arg, argi)) {
      app_input->inter_layer_pred = arg_parse_uint(&arg);
    } else if (arg_match(&arg, &psnr_arg, argi)) {
      svc_ctx->use_psnr = arg_parse_uint(&arg);
    } else {
      ++argj;
    }
  }

  // There will be a space in front of the string options
  if (strlen(string_options) > 0)
    vpx_svc_set_options(svc_ctx, string_options + 1);

  enc_cfg->g_pass = VPX_RC_ONE_PASS;

  if (enc_cfg->rc_target_bitrate > 0) {
    if (min_bitrate > 0) {
      enc_cfg->rc_2pass_vbr_minsection_pct =
          min_bitrate * 100 / enc_cfg->rc_target_bitrate;
    }
    if (max_bitrate > 0) {
      enc_cfg->rc_2pass_vbr_maxsection_pct =
          max_bitrate * 100 / enc_cfg->rc_target_bitrate;
    }
  }

  // Check for unrecognized options
  for (argi = argv; *argi; ++argi)
    if (argi[0][0] == '-' && strlen(argi[0]) > 1)
      die("Error: Unrecognized option %s\n", *argi);

  if (argv[0] == NULL) {
    usage_exit();
  }
  app_input->input_ctx.filename = argv[0];
  free(argv);

  open_input_file(&app_input->input_ctx);
  if (app_input->input_ctx.file_type == FILE_TYPE_Y4M) {
    enc_cfg->g_w = app_input->input_ctx.width;
    enc_cfg->g_h = app_input->input_ctx.height;
    enc_cfg->g_timebase.den = app_input->input_ctx.framerate.numerator;
    enc_cfg->g_timebase.num = app_input->input_ctx.framerate.denominator;
  }

  if (enc_cfg->g_w < 16 || enc_cfg->g_w % 2 || enc_cfg->g_h < 16 ||
      enc_cfg->g_h % 2)
    die("Invalid resolution: %d x %d\n", enc_cfg->g_w, enc_cfg->g_h);

  printf(
      "Codec %s\nframes: %d, skip: %d\n"
      "layers: %d\n"
      "width %d, height: %d,\n"
      "num: %d, den: %d, bitrate: %d,\n"
      "gop size: %d\n",
      vpx_codec_iface_name(vpx_codec_vp9_cx()), app_input->frames_to_code,
      app_input->frames_to_skip, svc_ctx->spatial_layers, enc_cfg->g_w,
      enc_cfg->g_h, enc_cfg->g_timebase.num, enc_cfg->g_timebase.den,
      enc_cfg->rc_target_bitrate, enc_cfg->kf_max_dist);
}

#if OUTPUT_RC_STATS
// For rate control encoding stats.
struct RateControlStats {
  // Number of input frames per layer.
  int layer_input_frames[VPX_MAX_LAYERS];
  // Total (cumulative) number of encoded frames per layer.
  int layer_tot_enc_frames[VPX_MAX_LAYERS];
  // Number of encoded non-key frames per layer.
  int layer_enc_frames[VPX_MAX_LAYERS];
  // Framerate per layer (cumulative).
  double layer_framerate[VPX_MAX_LAYERS];
  // Target average frame size per layer (per-frame-bandwidth per layer).
  double layer_pfb[VPX_MAX_LAYERS];
  // Actual average frame size per layer.
  double layer_avg_frame_size[VPX_MAX_LAYERS];
  // Average rate mismatch per layer (|target - actual| / target).
  double layer_avg_rate_mismatch[VPX_MAX_LAYERS];
  // Actual encoding bitrate per layer (cumulative).
  double layer_encoding_bitrate[VPX_MAX_LAYERS];
  // Average of the short-time encoder actual bitrate.
  // TODO(marpan): Should we add these short-time stats for each layer?
  double avg_st_encoding_bitrate;
  // Variance of the short-time encoder actual bitrate.
  double variance_st_encoding_bitrate;
  // Window (number of frames) for computing short-time encoding bitrate.
  int window_size;
  // Number of window measurements.
  int window_count;
};

// Note: these rate control stats assume only 1 key frame in the
// sequence (i.e., first frame only).
static void set_rate_control_stats(struct RateControlStats *rc,
                                   vpx_codec_enc_cfg_t *cfg) {
  unsigned int sl, tl;
  // Set the layer (cumulative) framerate and the target layer (non-cumulative)
  // per-frame-bandwidth, for the rate control encoding stats below.
  const double framerate = cfg->g_timebase.den / cfg->g_timebase.num;

  for (sl = 0; sl < cfg->ss_number_layers; ++sl) {
    for (tl = 0; tl < cfg->ts_number_layers; ++tl) {
      const int layer = sl * cfg->ts_number_layers + tl;
      if (cfg->ts_number_layers == 1)
        rc->layer_framerate[layer] = framerate;
      else
        rc->layer_framerate[layer] = framerate / cfg->ts_rate_decimator[tl];
      if (tl > 0) {
        rc->layer_pfb[layer] =
            1000.0 *
            (cfg->layer_target_bitrate[layer] -
             cfg->layer_target_bitrate[layer - 1]) /
            (rc->layer_framerate[layer] - rc->layer_framerate[layer - 1]);
      } else {
        rc->layer_pfb[layer] = 1000.0 * cfg->layer_target_bitrate[layer] /
                               rc->layer_framerate[layer];
      }
      rc->layer_input_frames[layer] = 0;
      rc->layer_enc_frames[layer] = 0;
      rc->layer_tot_enc_frames[layer] = 0;
      rc->layer_encoding_bitrate[layer] = 0.0;
      rc->layer_avg_frame_size[layer] = 0.0;
      rc->layer_avg_rate_mismatch[layer] = 0.0;
    }
  }
  rc->window_count = 0;
  rc->window_size = 15;
  rc->avg_st_encoding_bitrate = 0.0;
  rc->variance_st_encoding_bitrate = 0.0;
}

static void printout_rate_control_summary(struct RateControlStats *rc,
                                          vpx_codec_enc_cfg_t *cfg,
                                          int frame_cnt) {
  unsigned int sl, tl;
  double perc_fluctuation = 0.0;
  int tot_num_frames = 0;
  printf("Total number of processed frames: %d\n\n", frame_cnt - 1);
  printf("Rate control layer stats for sl%d tl%d layer(s):\n\n",
         cfg->ss_number_layers, cfg->ts_number_layers);
  for (sl = 0; sl < cfg->ss_number_layers; ++sl) {
    tot_num_frames = 0;
    for (tl = 0; tl < cfg->ts_number_layers; ++tl) {
      const int layer = sl * cfg->ts_number_layers + tl;
      const int num_dropped =
          (tl > 0)
              ? (rc->layer_input_frames[layer] - rc->layer_enc_frames[layer])
              : (rc->layer_input_frames[layer] - rc->layer_enc_frames[layer] -
                 1);
      tot_num_frames += rc->layer_input_frames[layer];
      rc->layer_encoding_bitrate[layer] = 0.001 * rc->layer_framerate[layer] *
                                          rc->layer_encoding_bitrate[layer] /
                                          tot_num_frames;
      rc->layer_avg_frame_size[layer] =
          rc->layer_avg_frame_size[layer] / rc->layer_enc_frames[layer];
      rc->layer_avg_rate_mismatch[layer] = 100.0 *
                                           rc->layer_avg_rate_mismatch[layer] /
                                           rc->layer_enc_frames[layer];
      printf("For layer#: sl%d tl%d \n", sl, tl);
      printf("Bitrate (target vs actual): %d %f.0 kbps\n",
             cfg->layer_target_bitrate[layer],
             rc->layer_encoding_bitrate[layer]);
      printf("Average frame size (target vs actual): %f %f bits\n",
             rc->layer_pfb[layer], rc->layer_avg_frame_size[layer]);
      printf("Average rate_mismatch: %f\n", rc->layer_avg_rate_mismatch[layer]);
      printf(
          "Number of input frames, encoded (non-key) frames, "
          "and percent dropped frames: %d %d %f.0 \n",
          rc->layer_input_frames[layer], rc->layer_enc_frames[layer],
          100.0 * num_dropped / rc->layer_input_frames[layer]);
      printf("\n");
    }
  }
  rc->avg_st_encoding_bitrate = rc->avg_st_encoding_bitrate / rc->window_count;
  rc->variance_st_encoding_bitrate =
      rc->variance_st_encoding_bitrate / rc->window_count -
      (rc->avg_st_encoding_bitrate * rc->avg_st_encoding_bitrate);
  perc_fluctuation = 100.0 * sqrt(rc->variance_st_encoding_bitrate) /
                     rc->avg_st_encoding_bitrate;
  printf("Short-time stats, for window of %d frames: \n", rc->window_size);
  printf("Average, rms-variance, and percent-fluct: %f %f %f \n",
         rc->avg_st_encoding_bitrate, sqrt(rc->variance_st_encoding_bitrate),
         perc_fluctuation);
  printf("Num of input, num of encoded (super) frames: %d %d \n", frame_cnt,
         tot_num_frames);
}

static vpx_codec_err_t parse_superframe_index(const uint8_t *data,
                                              size_t data_sz, uint64_t sizes[8],
                                              int *count) {
  // A chunk ending with a byte matching 0xc0 is an invalid chunk unless
  // it is a super frame index. If the last byte of real video compression
  // data is 0xc0 the encoder must add a 0 byte. If we have the marker but
  // not the associated matching marker byte at the front of the index we have
  // an invalid bitstream and need to return an error.

  uint8_t marker;

  marker = *(data + data_sz - 1);
  *count = 0;

  if ((marker & 0xe0) == 0xc0) {
    const uint32_t frames = (marker & 0x7) + 1;
    const uint32_t mag = ((marker >> 3) & 0x3) + 1;
    const size_t index_sz = 2 + mag * frames;

    // This chunk is marked as having a superframe index but doesn't have
    // enough data for it, thus it's an invalid superframe index.
    if (data_sz < index_sz) return VPX_CODEC_CORRUPT_FRAME;

    {
      const uint8_t marker2 = *(data + data_sz - index_sz);

      // This chunk is marked as having a superframe index but doesn't have
      // the matching marker byte at the front of the index therefore it's an
      // invalid chunk.
      if (marker != marker2) return VPX_CODEC_CORRUPT_FRAME;
    }

    {
      // Found a valid superframe index.
      uint32_t i, j;
      const uint8_t *x = &data[data_sz - index_sz + 1];

      for (i = 0; i < frames; ++i) {
        uint32_t this_sz = 0;

        for (j = 0; j < mag; ++j) this_sz |= (*x++) << (j * 8);
        sizes[i] = this_sz;
      }
      *count = frames;
    }
  }
  return VPX_CODEC_OK;
}
#endif

// Example pattern for spatial layers and 2 temporal layers used in the
// bypass/flexible mode. The pattern corresponds to the pattern
// VP9E_TEMPORAL_LAYERING_MODE_0101 (temporal_layering_mode == 2) used in
// non-flexible mode.
static void set_frame_flags_bypass_mode_ex0(
    int tl, int num_spatial_layers, int is_key_frame,
    vpx_svc_ref_frame_config_t *ref_frame_config) {
  int sl;
  for (sl = 0; sl < num_spatial_layers; ++sl)
    ref_frame_config->update_buffer_slot[sl] = 0;

  for (sl = 0; sl < num_spatial_layers; ++sl) {
    // Set the buffer idx.
    if (tl == 0) {
      ref_frame_config->lst_fb_idx[sl] = sl;
      if (sl) {
        if (is_key_frame) {
          ref_frame_config->lst_fb_idx[sl] = sl - 1;
          ref_frame_config->gld_fb_idx[sl] = sl;
        } else {
          ref_frame_config->gld_fb_idx[sl] = sl - 1;
        }
      } else {
        ref_frame_config->gld_fb_idx[sl] = 0;
      }
      ref_frame_config->alt_fb_idx[sl] = 0;
    } else if (tl == 1) {
      ref_frame_config->lst_fb_idx[sl] = sl;
      ref_frame_config->gld_fb_idx[sl] =
          (sl == 0) ? 0 : num_spatial_layers + sl - 1;
      ref_frame_config->alt_fb_idx[sl] = num_spatial_layers + sl;
    }
    // Set the reference and update flags.
    if (!tl) {
      if (!sl) {
        // Base spatial and base temporal (sl = 0, tl = 0)
        ref_frame_config->reference_last[sl] = 1;
        ref_frame_config->reference_golden[sl] = 0;
        ref_frame_config->reference_alt_ref[sl] = 0;
        ref_frame_config->update_buffer_slot[sl] |=
            1 << ref_frame_config->lst_fb_idx[sl];
      } else {
        if (is_key_frame) {
          ref_frame_config->reference_last[sl] = 1;
          ref_frame_config->reference_golden[sl] = 0;
          ref_frame_config->reference_alt_ref[sl] = 0;
          ref_frame_config->update_buffer_slot[sl] |=
              1 << ref_frame_config->gld_fb_idx[sl];
        } else {
          // Non-zero spatiall layer.
          ref_frame_config->reference_last[sl] = 1;
          ref_frame_config->reference_golden[sl] = 1;
          ref_frame_config->reference_alt_ref[sl] = 1;
          ref_frame_config->update_buffer_slot[sl] |=
              1 << ref_frame_config->lst_fb_idx[sl];
        }
      }
    } else if (tl == 1) {
      if (!sl) {
        // Base spatial and top temporal (tl = 1)
        ref_frame_config->reference_last[sl] = 1;
        ref_frame_config->reference_golden[sl] = 0;
        ref_frame_config->reference_alt_ref[sl] = 0;
        ref_frame_config->update_buffer_slot[sl] |=
            1 << ref_frame_config->alt_fb_idx[sl];
      } else {
        // Non-zero spatial.
        if (sl < num_spatial_layers - 1) {
          ref_frame_config->reference_last[sl] = 1;
          ref_frame_config->reference_golden[sl] = 1;
          ref_frame_config->reference_alt_ref[sl] = 0;
          ref_frame_config->update_buffer_slot[sl] |=
              1 << ref_frame_config->alt_fb_idx[sl];
        } else if (sl == num_spatial_layers - 1) {
          // Top spatial and top temporal (non-reference -- doesn't update any
          // reference buffers)
          ref_frame_config->reference_last[sl] = 1;
          ref_frame_config->reference_golden[sl] = 1;
          ref_frame_config->reference_alt_ref[sl] = 0;
        }
      }
    }
  }
}

// Example pattern for 2 spatial layers and 2 temporal layers used in the
// bypass/flexible mode, except only 1 spatial layer when temporal_layer_id = 1.
static void set_frame_flags_bypass_mode_ex1(
    int tl, int num_spatial_layers, int is_key_frame,
    vpx_svc_ref_frame_config_t *ref_frame_config) {
  int sl;
  for (sl = 0; sl < num_spatial_layers; ++sl)
    ref_frame_config->update_buffer_slot[sl] = 0;

  if (tl == 0) {
    if (is_key_frame) {
      ref_frame_config->lst_fb_idx[1] = 0;
      ref_frame_config->gld_fb_idx[1] = 1;
    } else {
      ref_frame_config->lst_fb_idx[1] = 1;
      ref_frame_config->gld_fb_idx[1] = 0;
    }
    ref_frame_config->alt_fb_idx[1] = 0;

    ref_frame_config->lst_fb_idx[0] = 0;
    ref_frame_config->gld_fb_idx[0] = 0;
    ref_frame_config->alt_fb_idx[0] = 0;
  }
  if (tl == 1) {
    ref_frame_config->lst_fb_idx[0] = 0;
    ref_frame_config->gld_fb_idx[0] = 1;
    ref_frame_config->alt_fb_idx[0] = 2;

    ref_frame_config->lst_fb_idx[1] = 1;
    ref_frame_config->gld_fb_idx[1] = 2;
    ref_frame_config->alt_fb_idx[1] = 3;
  }
  // Set the reference and update flags.
  if (tl == 0) {
    // Base spatial and base temporal (sl = 0, tl = 0)
    ref_frame_config->reference_last[0] = 1;
    ref_frame_config->reference_golden[0] = 0;
    ref_frame_config->reference_alt_ref[0] = 0;
    ref_frame_config->update_buffer_slot[0] |=
        1 << ref_frame_config->lst_fb_idx[0];

    if (is_key_frame) {
      ref_frame_config->reference_last[1] = 1;
      ref_frame_config->reference_golden[1] = 0;
      ref_frame_config->reference_alt_ref[1] = 0;
      ref_frame_config->update_buffer_slot[1] |=
          1 << ref_frame_config->gld_fb_idx[1];
    } else {
      // Non-zero spatiall layer.
      ref_frame_config->reference_last[1] = 1;
      ref_frame_config->reference_golden[1] = 1;
      ref_frame_config->reference_alt_ref[1] = 1;
      ref_frame_config->update_buffer_slot[1] |=
          1 << ref_frame_config->lst_fb_idx[1];
    }
  }
  if (tl == 1) {
    // Top spatial and top temporal (non-reference -- doesn't update any
    // reference buffers)
    ref_frame_config->reference_last[1] = 1;
    ref_frame_config->reference_golden[1] = 0;
    ref_frame_config->reference_alt_ref[1] = 0;
  }
}

#if CONFIG_VP9_DECODER && !SIMULCAST_MODE
static void test_decode(vpx_codec_ctx_t *encoder, vpx_codec_ctx_t *decoder,
                        const int frames_out, int *mismatch_seen) {
  vpx_image_t enc_img, dec_img;
  struct vp9_ref_frame ref_enc, ref_dec;
  if (*mismatch_seen) return;
  /* Get the internal reference frame */
  ref_enc.idx = 0;
  ref_dec.idx = 0;
  vpx_codec_control(encoder, VP9_GET_REFERENCE, &ref_enc);
  enc_img = ref_enc.img;
  vpx_codec_control(decoder, VP9_GET_REFERENCE, &ref_dec);
  dec_img = ref_dec.img;
#if CONFIG_VP9_HIGHBITDEPTH
  if ((enc_img.fmt & VPX_IMG_FMT_HIGHBITDEPTH) !=
      (dec_img.fmt & VPX_IMG_FMT_HIGHBITDEPTH)) {
    if (enc_img.fmt & VPX_IMG_FMT_HIGHBITDEPTH) {
      vpx_img_alloc(&enc_img, enc_img.fmt - VPX_IMG_FMT_HIGHBITDEPTH,
                    enc_img.d_w, enc_img.d_h, 16);
      vpx_img_truncate_16_to_8(&enc_img, &ref_enc.img);
    }
    if (dec_img.fmt & VPX_IMG_FMT_HIGHBITDEPTH) {
      vpx_img_alloc(&dec_img, dec_img.fmt - VPX_IMG_FMT_HIGHBITDEPTH,
                    dec_img.d_w, dec_img.d_h, 16);
      vpx_img_truncate_16_to_8(&dec_img, &ref_dec.img);
    }
  }
#endif

  if (!compare_img(&enc_img, &dec_img)) {
    int y[4], u[4], v[4];
#if CONFIG_VP9_HIGHBITDEPTH
    if (enc_img.fmt & VPX_IMG_FMT_HIGHBITDEPTH) {
      find_mismatch_high(&enc_img, &dec_img, y, u, v);
    } else {
      find_mismatch(&enc_img, &dec_img, y, u, v);
    }
#else
    find_mismatch(&enc_img, &dec_img, y, u, v);
#endif
    decoder->err = 1;
    printf(
        "Encode/decode mismatch on frame %d at"
        " Y[%d, %d] {%d/%d},"
        " U[%d, %d] {%d/%d},"
        " V[%d, %d] {%d/%d}\n",
        frames_out, y[0], y[1], y[2], y[3], u[0], u[1], u[2], u[3], v[0], v[1],
        v[2], v[3]);
    *mismatch_seen = frames_out;
  }

  vpx_img_free(&enc_img);
  vpx_img_free(&dec_img);
}
#endif

#if OUTPUT_RC_STATS
static void svc_output_rc_stats(
    vpx_codec_ctx_t *codec, vpx_codec_enc_cfg_t *enc_cfg,
    vpx_svc_layer_id_t *layer_id, const vpx_codec_cx_pkt_t *cx_pkt,
    struct RateControlStats *rc, VpxVideoWriter **outfile,
    const uint32_t frame_cnt, const double framerate) {
  int num_layers_encoded = 0;
  unsigned int sl, tl;
  uint64_t sizes[8];
  uint64_t sizes_parsed[8];
  int count = 0;
  double sum_bitrate = 0.0;
  double sum_bitrate2 = 0.0;
  memset(sizes, 0, sizeof(sizes));
  memset(sizes_parsed, 0, sizeof(sizes_parsed));
  vpx_codec_control(codec, VP9E_GET_SVC_LAYER_ID, layer_id);
  parse_superframe_index(cx_pkt->data.frame.buf, cx_pkt->data.frame.sz,
                         sizes_parsed, &count);
  if (enc_cfg->ss_number_layers == 1) {
    sizes[0] = cx_pkt->data.frame.sz;
  } else {
    for (sl = 0; sl < enc_cfg->ss_number_layers; ++sl) {
      sizes[sl] = 0;
      if (cx_pkt->data.frame.spatial_layer_encoded[sl]) {
        sizes[sl] = sizes_parsed[num_layers_encoded];
        num_layers_encoded++;
      }
    }
  }
  for (sl = 0; sl < enc_cfg->ss_number_layers; ++sl) {
    unsigned int sl2;
    uint64_t tot_size = 0;
#if SIMULCAST_MODE
    for (sl2 = 0; sl2 < sl; ++sl2) {
      if (cx_pkt->data.frame.spatial_layer_encoded[sl2]) tot_size += sizes[sl2];
    }
    vpx_video_writer_write_frame(outfile[sl],
                                 (uint8_t *)(cx_pkt->data.frame.buf) + tot_size,
                                 (size_t)(sizes[sl]), cx_pkt->data.frame.pts);
#else
    for (sl2 = 0; sl2 <= sl; ++sl2) {
      if (cx_pkt->data.frame.spatial_layer_encoded[sl2]) tot_size += sizes[sl2];
    }
    if (tot_size > 0)
      vpx_video_writer_write_frame(outfile[sl], cx_pkt->data.frame.buf,
                                   (size_t)(tot_size), cx_pkt->data.frame.pts);
#endif  // SIMULCAST_MODE
  }
  for (sl = 0; sl < enc_cfg->ss_number_layers; ++sl) {
    if (cx_pkt->data.frame.spatial_layer_encoded[sl]) {
      for (tl = layer_id->temporal_layer_id; tl < enc_cfg->ts_number_layers;
           ++tl) {
        const int layer = sl * enc_cfg->ts_number_layers + tl;
        ++rc->layer_tot_enc_frames[layer];
        rc->layer_encoding_bitrate[layer] += 8.0 * sizes[sl];
        // Keep count of rate control stats per layer, for non-key
        // frames.
        if (tl == (unsigned int)layer_id->temporal_layer_id &&
            !(cx_pkt->data.frame.flags & VPX_FRAME_IS_KEY)) {
          rc->layer_avg_frame_size[layer] += 8.0 * sizes[sl];
          rc->layer_avg_rate_mismatch[layer] +=
              fabs(8.0 * sizes[sl] - rc->layer_pfb[layer]) /
              rc->layer_pfb[layer];
          ++rc->layer_enc_frames[layer];
        }
      }
    }
  }

  // Update for short-time encoding bitrate states, for moving
  // window of size rc->window, shifted by rc->window / 2.
  // Ignore first window segment, due to key frame.
  if (frame_cnt > (unsigned int)rc->window_size) {
    for (sl = 0; sl < enc_cfg->ss_number_layers; ++sl) {
      if (cx_pkt->data.frame.spatial_layer_encoded[sl])
        sum_bitrate += 0.001 * 8.0 * sizes[sl] * framerate;
    }
    if (frame_cnt % rc->window_size == 0) {
      rc->window_count += 1;
      rc->avg_st_encoding_bitrate += sum_bitrate / rc->window_size;
      rc->variance_st_encoding_bitrate +=
          (sum_bitrate / rc->window_size) * (sum_bitrate / rc->window_size);
    }
  }

  // Second shifted window.
  if (frame_cnt > (unsigned int)(rc->window_size + rc->window_size / 2)) {
    for (sl = 0; sl < enc_cfg->ss_number_layers; ++sl) {
      sum_bitrate2 += 0.001 * 8.0 * sizes[sl] * framerate;
    }

    if (frame_cnt > (unsigned int)(2 * rc->window_size) &&
        frame_cnt % rc->window_size == 0) {
      rc->window_count += 1;
      rc->avg_st_encoding_bitrate += sum_bitrate2 / rc->window_size;
      rc->variance_st_encoding_bitrate +=
          (sum_bitrate2 / rc->window_size) * (sum_bitrate2 / rc->window_size);
    }
  }
}
#endif

int main(int argc, const char **argv) {
  AppInput app_input;
  VpxVideoWriter *writer = NULL;
  VpxVideoInfo info;
  vpx_codec_ctx_t encoder;
  vpx_codec_enc_cfg_t enc_cfg;
  SvcContext svc_ctx;
  vpx_svc_frame_drop_t svc_drop_frame;
  uint32_t i;
  uint32_t frame_cnt = 0;
  vpx_image_t raw;
  vpx_codec_err_t res;
  int pts = 0;            /* PTS starts at 0 */
  int frame_duration = 1; /* 1 timebase tick per frame */
  int end_of_stream = 0;
#if OUTPUT_FRAME_STATS
  int frames_received = 0;
#endif
#if OUTPUT_RC_STATS
  VpxVideoWriter *outfile[VPX_SS_MAX_LAYERS] = { NULL };
  struct RateControlStats rc;
  vpx_svc_layer_id_t layer_id;
  vpx_svc_ref_frame_config_t ref_frame_config;
  unsigned int sl;
  double framerate = 30.0;
#endif
  struct vpx_usec_timer timer;
  int64_t cx_time = 0;
#if CONFIG_INTERNAL_STATS
  FILE *f = fopen("opsnr.stt", "a");
#endif
#if CONFIG_VP9_DECODER && !SIMULCAST_MODE
  int mismatch_seen = 0;
  vpx_codec_ctx_t decoder;
#endif
  memset(&svc_ctx, 0, sizeof(svc_ctx));
  memset(&app_input, 0, sizeof(AppInput));
  memset(&info, 0, sizeof(VpxVideoInfo));
  memset(&layer_id, 0, sizeof(vpx_svc_layer_id_t));
  memset(&rc, 0, sizeof(struct RateControlStats));
  exec_name = argv[0];

  /* Setup default input stream settings */
  app_input.input_ctx.framerate.numerator = 30;
  app_input.input_ctx.framerate.denominator = 1;
  app_input.input_ctx.only_i420 = 1;
  app_input.input_ctx.bit_depth = 0;

  parse_command_line(argc, argv, &app_input, &svc_ctx, &enc_cfg);

  // Y4M reader handles its own allocation.
  if (app_input.input_ctx.file_type != FILE_TYPE_Y4M) {
// Allocate image buffer
#if CONFIG_VP9_HIGHBITDEPTH
    if (!vpx_img_alloc(&raw,
                       enc_cfg.g_input_bit_depth == 8 ? VPX_IMG_FMT_I420
                                                      : VPX_IMG_FMT_I42016,
                       enc_cfg.g_w, enc_cfg.g_h, 32)) {
      die("Failed to allocate image %dx%d\n", enc_cfg.g_w, enc_cfg.g_h);
    }
#else
    if (!vpx_img_alloc(&raw, VPX_IMG_FMT_I420, enc_cfg.g_w, enc_cfg.g_h, 32)) {
      die("Failed to allocate image %dx%d\n", enc_cfg.g_w, enc_cfg.g_h);
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
  }

  // Initialize codec
  if (vpx_svc_init(&svc_ctx, &encoder, vpx_codec_vp9_cx(), &enc_cfg) !=
      VPX_CODEC_OK)
    die("Failed to initialize encoder\n");
#if CONFIG_VP9_DECODER && !SIMULCAST_MODE
  if (vpx_codec_dec_init(
          &decoder, get_vpx_decoder_by_name("vp9")->codec_interface(), NULL, 0))
    die("Failed to initialize decoder\n");
#endif

#if OUTPUT_RC_STATS
  rc.window_count = 1;
  rc.window_size = 15;  // Silence a static analysis warning.
  rc.avg_st_encoding_bitrate = 0.0;
  rc.variance_st_encoding_bitrate = 0.0;
  if (svc_ctx.output_rc_stat) {
    set_rate_control_stats(&rc, &enc_cfg);
    framerate = enc_cfg.g_timebase.den / enc_cfg.g_timebase.num;
  }
#endif

  info.codec_fourcc = VP9_FOURCC;
  info.frame_width = enc_cfg.g_w;
  info.frame_height = enc_cfg.g_h;
  info.time_base.numerator = enc_cfg.g_timebase.num;
  info.time_base.denominator = enc_cfg.g_timebase.den;

  writer =
      vpx_video_writer_open(app_input.output_filename, kContainerIVF, &info);
  if (!writer)
    die("Failed to open %s for writing\n", app_input.output_filename);

#if OUTPUT_RC_STATS
  // Write out spatial layer stream.
  // TODO(marpan/jianj): allow for writing each spatial and temporal stream.
  if (svc_ctx.output_rc_stat) {
    for (sl = 0; sl < enc_cfg.ss_number_layers; ++sl) {
      char file_name[PATH_MAX];

      snprintf(file_name, sizeof(file_name), "%s_s%d.ivf",
               app_input.output_filename, sl);
      outfile[sl] = vpx_video_writer_open(file_name, kContainerIVF, &info);
      if (!outfile[sl]) die("Failed to open %s for writing", file_name);
    }
  }
#endif

  // skip initial frames
  for (i = 0; i < app_input.frames_to_skip; ++i)
    read_frame(&app_input.input_ctx, &raw);

  if (svc_ctx.speed != -1)
    vpx_codec_control(&encoder, VP8E_SET_CPUUSED, svc_ctx.speed);
  if (svc_ctx.threads) {
    vpx_codec_control(&encoder, VP9E_SET_TILE_COLUMNS,
                      get_msb(svc_ctx.threads));
    if (svc_ctx.threads > 1)
      vpx_codec_control(&encoder, VP9E_SET_ROW_MT, 1);
    else
      vpx_codec_control(&encoder, VP9E_SET_ROW_MT, 0);
  }
  if (svc_ctx.speed >= 5 && svc_ctx.aqmode == 1)
    vpx_codec_control(&encoder, VP9E_SET_AQ_MODE, 3);
  if (svc_ctx.speed >= 5)
    vpx_codec_control(&encoder, VP8E_SET_STATIC_THRESHOLD, 1);
  vpx_codec_control(&encoder, VP8E_SET_MAX_INTRA_BITRATE_PCT, 900);

  vpx_codec_control(&encoder, VP9E_SET_SVC_INTER_LAYER_PRED,
                    app_input.inter_layer_pred);

  vpx_codec_control(&encoder, VP9E_SET_NOISE_SENSITIVITY, 0);

  vpx_codec_control(&encoder, VP9E_SET_TUNE_CONTENT, app_input.tune_content);

  vpx_codec_control(&encoder, VP9E_SET_DISABLE_OVERSHOOT_MAXQ_CBR, 0);
  vpx_codec_control(&encoder, VP9E_SET_DISABLE_LOOPFILTER, 0);

  svc_drop_frame.framedrop_mode = FULL_SUPERFRAME_DROP;
  for (sl = 0; sl < (unsigned int)svc_ctx.spatial_layers; ++sl)
    svc_drop_frame.framedrop_thresh[sl] = enc_cfg.rc_dropframe_thresh;
  svc_drop_frame.max_consec_drop = INT_MAX;
  vpx_codec_control(&encoder, VP9E_SET_SVC_FRAME_DROP_LAYER, &svc_drop_frame);

  // Encode frames
  while (!end_of_stream) {
    vpx_codec_iter_t iter = NULL;
    const vpx_codec_cx_pkt_t *cx_pkt;
    // Example patterns for bypass/flexible mode:
    // example_pattern = 0: 2 temporal layers, and spatial_layers = 1,2,3. Exact
    // to fixed SVC patterns. example_pattern = 1: 2 spatial and 2 temporal
    // layers, with SL0 only has TL0, and SL1 has both TL0 and TL1. This example
    // uses the extended API.
    int example_pattern = 0;
    if (frame_cnt >= app_input.frames_to_code ||
        !read_frame(&app_input.input_ctx, &raw)) {
      // We need one extra vpx_svc_encode call at end of stream to flush
      // encoder and get remaining data
      end_of_stream = 1;
    }

    // For BYPASS/FLEXIBLE mode, set the frame flags (reference and updates)
    // and the buffer indices for each spatial layer of the current
    // (super)frame to be encoded. The spatial and temporal layer_id for the
    // current frame also needs to be set.
    // TODO(marpan): Should rename the "VP9E_TEMPORAL_LAYERING_MODE_BYPASS"
    // mode to "VP9E_LAYERING_MODE_BYPASS".
    if (svc_ctx.temporal_layering_mode == VP9E_TEMPORAL_LAYERING_MODE_BYPASS) {
      layer_id.spatial_layer_id = 0;
      // Example for 2 temporal layers.
      if (frame_cnt % 2 == 0) {
        layer_id.temporal_layer_id = 0;
        for (i = 0; i < VPX_SS_MAX_LAYERS; i++)
          layer_id.temporal_layer_id_per_spatial[i] = 0;
      } else {
        layer_id.temporal_layer_id = 1;
        for (i = 0; i < VPX_SS_MAX_LAYERS; i++)
          layer_id.temporal_layer_id_per_spatial[i] = 1;
      }
      if (example_pattern == 1) {
        // example_pattern 1 is hard-coded for 2 spatial and 2 temporal layers.
        assert(svc_ctx.spatial_layers == 2);
        assert(svc_ctx.temporal_layers == 2);
        if (frame_cnt % 2 == 0) {
          // Spatial layer 0 and 1 are encoded.
          layer_id.temporal_layer_id_per_spatial[0] = 0;
          layer_id.temporal_layer_id_per_spatial[1] = 0;
          layer_id.spatial_layer_id = 0;
        } else {
          // Only spatial layer 1 is encoded here.
          layer_id.temporal_layer_id_per_spatial[1] = 1;
          layer_id.spatial_layer_id = 1;
        }
      }
      vpx_codec_control(&encoder, VP9E_SET_SVC_LAYER_ID, &layer_id);
      // TODO(jianj): Fix the parameter passing for "is_key_frame" in
      // set_frame_flags_bypass_model() for case of periodic key frames.
      if (example_pattern == 0) {
        set_frame_flags_bypass_mode_ex0(layer_id.temporal_layer_id,
                                        svc_ctx.spatial_layers, frame_cnt == 0,
                                        &ref_frame_config);
      } else if (example_pattern == 1) {
        set_frame_flags_bypass_mode_ex1(layer_id.temporal_layer_id,
                                        svc_ctx.spatial_layers, frame_cnt == 0,
                                        &ref_frame_config);
      }
      ref_frame_config.duration[0] = frame_duration * 1;
      ref_frame_config.duration[1] = frame_duration * 1;

      vpx_codec_control(&encoder, VP9E_SET_SVC_REF_FRAME_CONFIG,
                        &ref_frame_config);
      // Keep track of input frames, to account for frame drops in rate control
      // stats/metrics.
      for (sl = 0; sl < enc_cfg.ss_number_layers; ++sl) {
        ++rc.layer_input_frames[sl * enc_cfg.ts_number_layers +
                                layer_id.temporal_layer_id];
      }
    } else {
      // For the fixed pattern SVC, temporal layer is given by superframe count.
      unsigned int tl = 0;
      if (enc_cfg.ts_number_layers == 2)
        tl = (frame_cnt % 2 != 0);
      else if (enc_cfg.ts_number_layers == 3) {
        if (frame_cnt % 2 != 0) tl = 2;
        if ((frame_cnt > 1) && ((frame_cnt - 2) % 4 == 0)) tl = 1;
      }
      for (sl = 0; sl < enc_cfg.ss_number_layers; ++sl)
        ++rc.layer_input_frames[sl * enc_cfg.ts_number_layers + tl];
    }

    vpx_usec_timer_start(&timer);
    res = vpx_svc_encode(
        &svc_ctx, &encoder, (end_of_stream ? NULL : &raw), pts, frame_duration,
        svc_ctx.speed >= 5 ? VPX_DL_REALTIME : VPX_DL_GOOD_QUALITY);
    vpx_usec_timer_mark(&timer);
    cx_time += vpx_usec_timer_elapsed(&timer);

    fflush(stdout);
    if (res != VPX_CODEC_OK) {
      die_codec(&encoder, "Failed to encode frame");
    }

    while ((cx_pkt = vpx_codec_get_cx_data(&encoder, &iter)) != NULL) {
      switch (cx_pkt->kind) {
        case VPX_CODEC_CX_FRAME_PKT: {
          SvcInternal_t *const si = (SvcInternal_t *)svc_ctx.internal;
          if (cx_pkt->data.frame.sz > 0) {
            vpx_video_writer_write_frame(writer, cx_pkt->data.frame.buf,
                                         cx_pkt->data.frame.sz,
                                         cx_pkt->data.frame.pts);
#if OUTPUT_RC_STATS
            if (svc_ctx.output_rc_stat) {
              svc_output_rc_stats(&encoder, &enc_cfg, &layer_id, cx_pkt, &rc,
                                  outfile, frame_cnt, framerate);
            }
#endif
          }
#if OUTPUT_FRAME_STATS
          printf("SVC frame: %d, kf: %d, size: %d, pts: %d\n", frames_received,
                 !!(cx_pkt->data.frame.flags & VPX_FRAME_IS_KEY),
                 (int)cx_pkt->data.frame.sz, (int)cx_pkt->data.frame.pts);
          ++frames_received;
#endif
          if (enc_cfg.ss_number_layers > 1) {
            uint64_t sizes[8] = { 0 };
            int count = 0;
            int num_layers_encoded = 0;
            parse_superframe_index(cx_pkt->data.frame.buf,
                                   cx_pkt->data.frame.sz, sizes, &count);
            for (sl = 0; sl < enc_cfg.ss_number_layers; ++sl) {
              if (cx_pkt->data.frame.spatial_layer_encoded[sl]) {
                si->bytes_sum[sl] += (int)sizes[num_layers_encoded];
                num_layers_encoded++;
              }
            }
          } else {
            si->bytes_sum[0] += (int)cx_pkt->data.frame.sz;
          }
#if CONFIG_VP9_DECODER && !SIMULCAST_MODE
          if (vpx_codec_decode(&decoder, cx_pkt->data.frame.buf,
                               (unsigned int)cx_pkt->data.frame.sz, NULL, 0))
            die_codec(&decoder, "Failed to decode frame.");
          vpx_codec_control(&encoder, VP9E_GET_SVC_LAYER_ID, &layer_id);
          // Don't look for mismatch on top spatial and top temporal layers as
          // they are non reference frames. Don't look at frames whose top
          // spatial layer is dropped.
          if ((enc_cfg.ss_number_layers > 1 || enc_cfg.ts_number_layers > 1) &&
              cx_pkt->data.frame
                  .spatial_layer_encoded[enc_cfg.ss_number_layers - 1] &&
              !(layer_id.temporal_layer_id > 0 &&
                layer_id.temporal_layer_id ==
                    (int)enc_cfg.ts_number_layers - 1)) {
            test_decode(&encoder, &decoder, frame_cnt, &mismatch_seen);
          }
#endif
          break;
        }
        case VPX_CODEC_PSNR_PKT: {
          SvcInternal_t *const si = (SvcInternal_t *)svc_ctx.internal;
          sl = cx_pkt->data.psnr.spatial_layer_id;
          si->number_of_frames[sl]++;
          for (int j = 0; j < 4; ++j) {
            si->psnr_sum[sl][j] += cx_pkt->data.psnr.psnr[j];
            si->sse_sum[sl][j] += cx_pkt->data.psnr.sse[j];
          }
          break;
        }
        case VPX_CODEC_STATS_PKT: {
          stats_write(&app_input.rc_stats, cx_pkt->data.twopass_stats.buf,
                      cx_pkt->data.twopass_stats.sz);
          break;
        }
        default: {
          break;
        }
      }
    }

    if (!end_of_stream) {
      ++frame_cnt;
      pts += frame_duration;
    }
  }

  printf("Processed %d frames\n", frame_cnt);

  close_input_file(&app_input.input_ctx);

#if OUTPUT_RC_STATS
  if (svc_ctx.output_rc_stat) {
    printout_rate_control_summary(&rc, &enc_cfg, frame_cnt);
    printf("\n");
  }
#endif
  if (vpx_codec_destroy(&encoder))
    die_codec(&encoder, "Failed to destroy codec");
  if (writer) {
    vpx_video_writer_close(writer);
  }
#if OUTPUT_RC_STATS
  if (svc_ctx.output_rc_stat) {
    for (sl = 0; sl < enc_cfg.ss_number_layers; ++sl) {
      vpx_video_writer_close(outfile[sl]);
    }
  }
#endif
#if CONFIG_INTERNAL_STATS
  if (mismatch_seen) {
    fprintf(f, "First mismatch occurred in frame %d\n", mismatch_seen);
  } else {
    fprintf(f, "No mismatch detected in recon buffers\n");
  }
  fclose(f);
#endif
  printf("Frame cnt and encoding time/FPS stats for encoding: %d %f %f \n",
         frame_cnt, 1000 * (float)cx_time / (double)(frame_cnt * 1000000),
         1000000 * (double)frame_cnt / (double)cx_time);
  if (app_input.input_ctx.file_type != FILE_TYPE_Y4M) {
    vpx_img_free(&raw);
  }
  // display average size, psnr
  vpx_svc_dump_statistics(&svc_ctx);
  vpx_svc_release(&svc_ctx);
  return EXIT_SUCCESS;
}
