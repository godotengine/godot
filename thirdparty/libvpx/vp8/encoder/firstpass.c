/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include "./vpx_dsp_rtcd.h"
#include "./vpx_scale_rtcd.h"
#include "block.h"
#include "onyx_int.h"
#include "vpx_dsp/variance.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "encodeintra.h"
#include "vp8/common/common.h"
#include "vp8/common/setupintrarecon.h"
#include "vp8/common/systemdependent.h"
#include "mcomp.h"
#include "firstpass.h"
#include "vpx_scale/vpx_scale.h"
#include "encodemb.h"
#include "vp8/common/extend.h"
#include "vpx_ports/system_state.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8/common/swapyv12buffer.h"
#include "rdopt.h"
#include "vp8/common/quant_common.h"
#include "encodemv.h"
#include "encodeframe.h"

#define OUTPUT_FPF 0

extern void vp8cx_frame_init_quantizer(VP8_COMP *cpi);

#define GFQ_ADJUSTMENT vp8_gf_boost_qadjustment[Q]
extern int vp8_kf_boost_qadjustment[QINDEX_RANGE];

extern const int vp8_gf_boost_qadjustment[QINDEX_RANGE];

#define IIFACTOR 1.5
#define IIKFACTOR1 1.40
#define IIKFACTOR2 1.5
#define RMAX 14.0
#define GF_RMAX 48.0

#define KF_MB_INTRA_MIN 300
#define GF_MB_INTRA_MIN 200

#define DOUBLE_DIVIDE_CHECK(X) ((X) < 0 ? (X) - .000001 : (X) + .000001)

#define POW1 (double)cpi->oxcf.two_pass_vbrbias / 100.0
#define POW2 (double)cpi->oxcf.two_pass_vbrbias / 100.0

#define NEW_BOOST 1

static int vscale_lookup[7] = { 0, 1, 1, 2, 2, 3, 3 };
static int hscale_lookup[7] = { 0, 0, 1, 1, 2, 2, 3 };

static const int cq_level[QINDEX_RANGE] = {
  0,  0,  1,  1,  2,  3,  3,  4,  4,  5,  6,  6,  7,  8,  8,  9,  9,  10, 11,
  11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 22, 23, 24,
  24, 25, 26, 27, 27, 28, 29, 30, 30, 31, 32, 33, 33, 34, 35, 36, 36, 37, 38,
  39, 39, 40, 41, 42, 42, 43, 44, 45, 46, 46, 47, 48, 49, 50, 50, 51, 52, 53,
  54, 55, 55, 56, 57, 58, 59, 60, 60, 61, 62, 63, 64, 65, 66, 67, 67, 68, 69,
  70, 71, 72, 73, 74, 75, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 86,
  87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
};

static void find_next_key_frame(VP8_COMP *cpi, FIRSTPASS_STATS *this_frame);

/* Resets the first pass file to the given position using a relative seek
 * from the current position
 */
static void reset_fpf_position(VP8_COMP *cpi, FIRSTPASS_STATS *Position) {
  cpi->twopass.stats_in = Position;
}

static int lookup_next_frame_stats(VP8_COMP *cpi, FIRSTPASS_STATS *next_frame) {
  if (cpi->twopass.stats_in >= cpi->twopass.stats_in_end) return EOF;

  *next_frame = *cpi->twopass.stats_in;
  return 1;
}

/* Read frame stats at an offset from the current position */
static int read_frame_stats(VP8_COMP *cpi, FIRSTPASS_STATS *frame_stats,
                            int offset) {
  FIRSTPASS_STATS *fps_ptr = cpi->twopass.stats_in;

  /* Check legality of offset */
  if (offset >= 0) {
    if (&fps_ptr[offset] >= cpi->twopass.stats_in_end) return EOF;
  } else if (offset < 0) {
    if (&fps_ptr[offset] < cpi->twopass.stats_in_start) return EOF;
  }

  *frame_stats = fps_ptr[offset];
  return 1;
}

static int input_stats(VP8_COMP *cpi, FIRSTPASS_STATS *fps) {
  if (cpi->twopass.stats_in >= cpi->twopass.stats_in_end) return EOF;

  *fps = *cpi->twopass.stats_in;
  cpi->twopass.stats_in =
      (void *)((char *)cpi->twopass.stats_in + sizeof(FIRSTPASS_STATS));
  return 1;
}

static void output_stats(struct vpx_codec_pkt_list *pktlist,
                         FIRSTPASS_STATS *stats) {
  struct vpx_codec_cx_pkt pkt;
  pkt.kind = VPX_CODEC_STATS_PKT;
  pkt.data.twopass_stats.buf = stats;
  pkt.data.twopass_stats.sz = sizeof(FIRSTPASS_STATS);
  vpx_codec_pkt_list_add(pktlist, &pkt);

/* TEMP debug code */
#if OUTPUT_FPF

  {
    FILE *fpfile;
    fpfile = fopen("firstpass.stt", "a");

    fprintf(fpfile,
            "%12.0f %12.0f %12.0f %12.4f %12.4f %12.4f %12.4f"
            " %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f"
            " %12.0f %12.0f %12.4f\n",
            stats->frame, stats->intra_error, stats->coded_error,
            stats->ssim_weighted_pred_err, stats->pcnt_inter,
            stats->pcnt_motion, stats->pcnt_second_ref, stats->pcnt_neutral,
            stats->MVr, stats->mvr_abs, stats->MVc, stats->mvc_abs, stats->MVrv,
            stats->MVcv, stats->mv_in_out_count, stats->new_mv_count,
            stats->count, stats->duration);
    fclose(fpfile);
  }
#endif
}

static void zero_stats(FIRSTPASS_STATS *section) {
  section->frame = 0.0;
  section->intra_error = 0.0;
  section->coded_error = 0.0;
  section->ssim_weighted_pred_err = 0.0;
  section->pcnt_inter = 0.0;
  section->pcnt_motion = 0.0;
  section->pcnt_second_ref = 0.0;
  section->pcnt_neutral = 0.0;
  section->MVr = 0.0;
  section->mvr_abs = 0.0;
  section->MVc = 0.0;
  section->mvc_abs = 0.0;
  section->MVrv = 0.0;
  section->MVcv = 0.0;
  section->mv_in_out_count = 0.0;
  section->new_mv_count = 0.0;
  section->count = 0.0;
  section->duration = 1.0;
}

static void accumulate_stats(FIRSTPASS_STATS *section, FIRSTPASS_STATS *frame) {
  section->frame += frame->frame;
  section->intra_error += frame->intra_error;
  section->coded_error += frame->coded_error;
  section->ssim_weighted_pred_err += frame->ssim_weighted_pred_err;
  section->pcnt_inter += frame->pcnt_inter;
  section->pcnt_motion += frame->pcnt_motion;
  section->pcnt_second_ref += frame->pcnt_second_ref;
  section->pcnt_neutral += frame->pcnt_neutral;
  section->MVr += frame->MVr;
  section->mvr_abs += frame->mvr_abs;
  section->MVc += frame->MVc;
  section->mvc_abs += frame->mvc_abs;
  section->MVrv += frame->MVrv;
  section->MVcv += frame->MVcv;
  section->mv_in_out_count += frame->mv_in_out_count;
  section->new_mv_count += frame->new_mv_count;
  section->count += frame->count;
  section->duration += frame->duration;
}

static void subtract_stats(FIRSTPASS_STATS *section, FIRSTPASS_STATS *frame) {
  section->frame -= frame->frame;
  section->intra_error -= frame->intra_error;
  section->coded_error -= frame->coded_error;
  section->ssim_weighted_pred_err -= frame->ssim_weighted_pred_err;
  section->pcnt_inter -= frame->pcnt_inter;
  section->pcnt_motion -= frame->pcnt_motion;
  section->pcnt_second_ref -= frame->pcnt_second_ref;
  section->pcnt_neutral -= frame->pcnt_neutral;
  section->MVr -= frame->MVr;
  section->mvr_abs -= frame->mvr_abs;
  section->MVc -= frame->MVc;
  section->mvc_abs -= frame->mvc_abs;
  section->MVrv -= frame->MVrv;
  section->MVcv -= frame->MVcv;
  section->mv_in_out_count -= frame->mv_in_out_count;
  section->new_mv_count -= frame->new_mv_count;
  section->count -= frame->count;
  section->duration -= frame->duration;
}

static void avg_stats(FIRSTPASS_STATS *section) {
  if (section->count < 1.0) return;

  section->intra_error /= section->count;
  section->coded_error /= section->count;
  section->ssim_weighted_pred_err /= section->count;
  section->pcnt_inter /= section->count;
  section->pcnt_second_ref /= section->count;
  section->pcnt_neutral /= section->count;
  section->pcnt_motion /= section->count;
  section->MVr /= section->count;
  section->mvr_abs /= section->count;
  section->MVc /= section->count;
  section->mvc_abs /= section->count;
  section->MVrv /= section->count;
  section->MVcv /= section->count;
  section->mv_in_out_count /= section->count;
  section->duration /= section->count;
}

/* Calculate a modified Error used in distributing bits between easier
 * and harder frames
 */
static double calculate_modified_err(VP8_COMP *cpi,
                                     FIRSTPASS_STATS *this_frame) {
  double av_err = (cpi->twopass.total_stats.ssim_weighted_pred_err /
                   cpi->twopass.total_stats.count);
  double this_err = this_frame->ssim_weighted_pred_err;
  double modified_err;

  if (this_err > av_err) {
    modified_err = av_err * pow((this_err / DOUBLE_DIVIDE_CHECK(av_err)), POW1);
  } else {
    modified_err = av_err * pow((this_err / DOUBLE_DIVIDE_CHECK(av_err)), POW2);
  }

  return modified_err;
}

static const double weight_table[256] = {
  0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
  0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
  0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
  0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.020000,
  0.020000, 0.020000, 0.020000, 0.020000, 0.020000, 0.031250, 0.062500,
  0.093750, 0.125000, 0.156250, 0.187500, 0.218750, 0.250000, 0.281250,
  0.312500, 0.343750, 0.375000, 0.406250, 0.437500, 0.468750, 0.500000,
  0.531250, 0.562500, 0.593750, 0.625000, 0.656250, 0.687500, 0.718750,
  0.750000, 0.781250, 0.812500, 0.843750, 0.875000, 0.906250, 0.937500,
  0.968750, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
  1.000000, 1.000000, 1.000000, 1.000000
};

static double simple_weight(YV12_BUFFER_CONFIG *source) {
  int i, j;

  unsigned char *src = source->y_buffer;
  double sum_weights = 0.0;

  /* Loop throught the Y plane raw examining levels and creating a weight
   * for the image
   */
  i = source->y_height;
  do {
    j = source->y_width;
    do {
      sum_weights += weight_table[*src];
      src++;
    } while (--j);
    src -= source->y_width;
    src += source->y_stride;
  } while (--i);

  sum_weights /= (source->y_height * source->y_width);

  return sum_weights;
}

/* This function returns the current per frame maximum bitrate target */
static int frame_max_bits(VP8_COMP *cpi) {
  /* Max allocation for a single frame based on the max section guidelines
   * passed in and how many bits are left
   */
  int max_bits;

  /* For CBR we need to also consider buffer fullness.
   * If we are running below the optimal level then we need to gradually
   * tighten up on max_bits.
   */
  if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
    double buffer_fullness_ratio =
        (double)cpi->buffer_level /
        DOUBLE_DIVIDE_CHECK((double)cpi->oxcf.optimal_buffer_level);

    /* For CBR base this on the target average bits per frame plus the
     * maximum sedction rate passed in by the user
     */
    max_bits = (int)(cpi->av_per_frame_bandwidth *
                     ((double)cpi->oxcf.two_pass_vbrmax_section / 100.0));

    /* If our buffer is below the optimum level */
    if (buffer_fullness_ratio < 1.0) {
      /* The lower of max_bits / 4 or cpi->av_per_frame_bandwidth / 4. */
      int min_max_bits = ((cpi->av_per_frame_bandwidth >> 2) < (max_bits >> 2))
                             ? cpi->av_per_frame_bandwidth >> 2
                             : max_bits >> 2;

      max_bits = (int)(max_bits * buffer_fullness_ratio);

      /* Lowest value we will set ... which should allow the buffer to
       * refill.
       */
      if (max_bits < min_max_bits) max_bits = min_max_bits;
    }
  }
  /* VBR */
  else {
    /* For VBR base this on the bits and frames left plus the
     * two_pass_vbrmax_section rate passed in by the user
     */
    max_bits = saturate_cast_double_to_int(
        ((double)cpi->twopass.bits_left /
         (cpi->twopass.total_stats.count -
          (double)cpi->common.current_video_frame)) *
        ((double)cpi->oxcf.two_pass_vbrmax_section / 100.0));
  }

  /* Trap case where we are out of bits */
  if (max_bits < 0) max_bits = 0;

  return max_bits;
}

void vp8_init_first_pass(VP8_COMP *cpi) {
  zero_stats(&cpi->twopass.total_stats);
}

void vp8_end_first_pass(VP8_COMP *cpi) {
  output_stats(cpi->output_pkt_list, &cpi->twopass.total_stats);
}

static void zz_motion_search(MACROBLOCK *x, YV12_BUFFER_CONFIG *raw_buffer,
                             int *raw_motion_err,
                             YV12_BUFFER_CONFIG *recon_buffer,
                             int *best_motion_err, int recon_yoffset) {
  MACROBLOCKD *const xd = &x->e_mbd;
  BLOCK *b = &x->block[0];
  BLOCKD *d = &x->e_mbd.block[0];

  unsigned char *src_ptr = (*(b->base_src) + b->src);
  int src_stride = b->src_stride;
  unsigned char *raw_ptr;
  int raw_stride = raw_buffer->y_stride;
  unsigned char *ref_ptr;
  int ref_stride = x->e_mbd.pre.y_stride;

  /* Set up pointers for this macro block raw buffer */
  raw_ptr = (unsigned char *)(raw_buffer->y_buffer + recon_yoffset + d->offset);
  vpx_mse16x16(src_ptr, src_stride, raw_ptr, raw_stride,
               (unsigned int *)(raw_motion_err));

  /* Set up pointers for this macro block recon buffer */
  xd->pre.y_buffer = recon_buffer->y_buffer + recon_yoffset;
  ref_ptr = (unsigned char *)(xd->pre.y_buffer + d->offset);
  vpx_mse16x16(src_ptr, src_stride, ref_ptr, ref_stride,
               (unsigned int *)(best_motion_err));
}

static void first_pass_motion_search(VP8_COMP *cpi, MACROBLOCK *x,
                                     int_mv *ref_mv, MV *best_mv,
                                     YV12_BUFFER_CONFIG *recon_buffer,
                                     int *best_motion_err, int recon_yoffset) {
  MACROBLOCKD *const xd = &x->e_mbd;
  BLOCK *b = &x->block[0];
  BLOCKD *d = &x->e_mbd.block[0];
  int num00;

  int_mv tmp_mv;
  int_mv ref_mv_full;

  int tmp_err;
  int step_param = 3; /* Don't search over full range for first pass */
  int further_steps = (MAX_MVSEARCH_STEPS - 1) - step_param;
  int n;
  vp8_variance_fn_ptr_t v_fn_ptr = cpi->fn_ptr[BLOCK_16X16];
  int new_mv_mode_penalty = 256;

  /* override the default variance function to use MSE */
  v_fn_ptr.vf = vpx_mse16x16;

  /* Set up pointers for this macro block recon buffer */
  xd->pre.y_buffer = recon_buffer->y_buffer + recon_yoffset;

  /* Initial step/diamond search centred on best mv */
  tmp_mv.as_int = 0;
  ref_mv_full.as_mv.col = ref_mv->as_mv.col >> 3;
  ref_mv_full.as_mv.row = ref_mv->as_mv.row >> 3;
  tmp_err = cpi->diamond_search_sad(x, b, d, &ref_mv_full, &tmp_mv, step_param,
                                    x->sadperbit16, &num00, &v_fn_ptr,
                                    x->mvcost, ref_mv);
  if (tmp_err < INT_MAX - new_mv_mode_penalty) tmp_err += new_mv_mode_penalty;

  if (tmp_err < *best_motion_err) {
    *best_motion_err = tmp_err;
    best_mv->row = tmp_mv.as_mv.row;
    best_mv->col = tmp_mv.as_mv.col;
  }

  /* Further step/diamond searches as necessary */
  n = num00;
  num00 = 0;

  while (n < further_steps) {
    n++;

    if (num00) {
      num00--;
    } else {
      tmp_err = cpi->diamond_search_sad(x, b, d, &ref_mv_full, &tmp_mv,
                                        step_param + n, x->sadperbit16, &num00,
                                        &v_fn_ptr, x->mvcost, ref_mv);
      if (tmp_err < INT_MAX - new_mv_mode_penalty) {
        tmp_err += new_mv_mode_penalty;
      }

      if (tmp_err < *best_motion_err) {
        *best_motion_err = tmp_err;
        best_mv->row = tmp_mv.as_mv.row;
        best_mv->col = tmp_mv.as_mv.col;
      }
    }
  }
}

void vp8_first_pass(VP8_COMP *cpi) {
  int mb_row, mb_col;
  MACROBLOCK *const x = &cpi->mb;
  VP8_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;

  int recon_yoffset, recon_uvoffset;
  YV12_BUFFER_CONFIG *lst_yv12 = &cm->yv12_fb[cm->lst_fb_idx];
  YV12_BUFFER_CONFIG *new_yv12 = &cm->yv12_fb[cm->new_fb_idx];
  YV12_BUFFER_CONFIG *gld_yv12 = &cm->yv12_fb[cm->gld_fb_idx];
  int recon_y_stride = lst_yv12->y_stride;
  int recon_uv_stride = lst_yv12->uv_stride;
  int64_t intra_error = 0;
  int64_t coded_error = 0;

  int sum_mvr = 0, sum_mvc = 0;
  int sum_mvr_abs = 0, sum_mvc_abs = 0;
  int sum_mvrs = 0, sum_mvcs = 0;
  int mvcount = 0;
  int intercount = 0;
  int second_ref_count = 0;
  int intrapenalty = 256;
  int neutral_count = 0;
  int new_mv_count = 0;
  int sum_in_vectors = 0;
  uint32_t lastmv_as_int = 0;

  int_mv zero_ref_mv;

  zero_ref_mv.as_int = 0;

  vpx_clear_system_state();

  x->src = *cpi->Source;
  xd->pre = *lst_yv12;
  xd->dst = *new_yv12;

  x->partition_info = x->pi;

  xd->mode_info_context = cm->mi;

  if (!cm->use_bilinear_mc_filter) {
    xd->subpixel_predict = vp8_sixtap_predict4x4;
    xd->subpixel_predict8x4 = vp8_sixtap_predict8x4;
    xd->subpixel_predict8x8 = vp8_sixtap_predict8x8;
    xd->subpixel_predict16x16 = vp8_sixtap_predict16x16;
  } else {
    xd->subpixel_predict = vp8_bilinear_predict4x4;
    xd->subpixel_predict8x4 = vp8_bilinear_predict8x4;
    xd->subpixel_predict8x8 = vp8_bilinear_predict8x8;
    xd->subpixel_predict16x16 = vp8_bilinear_predict16x16;
  }

  vp8_build_block_offsets(x);

  /* set up frame new frame for intra coded blocks */
  vp8_setup_intra_recon(new_yv12);
  vp8cx_frame_init_quantizer(cpi);

  /* Initialise the MV cost table to the defaults */
  {
    int flag[2] = { 1, 1 };
    vp8_initialize_rd_consts(cpi, x,
                             vp8_dc_quant(cm->base_qindex, cm->y1dc_delta_q));
    memcpy(cm->fc.mvc, vp8_default_mv_context, sizeof(vp8_default_mv_context));
    vp8_build_component_cost_table(cpi->mb.mvcost,
                                   (const MV_CONTEXT *)cm->fc.mvc, flag);
  }

  /* for each macroblock row in image */
  for (mb_row = 0; mb_row < cm->mb_rows; ++mb_row) {
    int_mv best_ref_mv;

    best_ref_mv.as_int = 0;

    /* reset above block coeffs */
    xd->up_available = (mb_row != 0);
    recon_yoffset = (mb_row * recon_y_stride * 16);
    recon_uvoffset = (mb_row * recon_uv_stride * 8);

    /* Set up limit values for motion vectors to prevent them extending
     * outside the UMV borders
     */
    x->mv_row_min = -((mb_row * 16) + (VP8BORDERINPIXELS - 16));
    x->mv_row_max =
        ((cm->mb_rows - 1 - mb_row) * 16) + (VP8BORDERINPIXELS - 16);

    /* for each macroblock col in image */
    for (mb_col = 0; mb_col < cm->mb_cols; ++mb_col) {
      int this_error;
      int gf_motion_error = INT_MAX;
      int use_dc_pred = (mb_col || mb_row) && (!mb_col || !mb_row);

      xd->dst.y_buffer = new_yv12->y_buffer + recon_yoffset;
      xd->dst.u_buffer = new_yv12->u_buffer + recon_uvoffset;
      xd->dst.v_buffer = new_yv12->v_buffer + recon_uvoffset;
      xd->left_available = (mb_col != 0);

      /* Copy current mb to a buffer */
      vp8_copy_mem16x16(x->src.y_buffer, x->src.y_stride, x->thismb, 16);

      /* do intra 16x16 prediction */
      this_error = vp8_encode_intra(x, use_dc_pred);

      /* "intrapenalty" below deals with situations where the intra
       * and inter error scores are very low (eg a plain black frame)
       * We do not have special cases in first pass for 0,0 and
       * nearest etc so all inter modes carry an overhead cost
       * estimate fot the mv. When the error score is very low this
       * causes us to pick all or lots of INTRA modes and throw lots
       * of key frames. This penalty adds a cost matching that of a
       * 0,0 mv to the intra case.
       */
      this_error += intrapenalty;

      /* Cumulative intra error total */
      intra_error += (int64_t)this_error;

      /* Set up limit values for motion vectors to prevent them
       * extending outside the UMV borders
       */
      x->mv_col_min = -((mb_col * 16) + (VP8BORDERINPIXELS - 16));
      x->mv_col_max =
          ((cm->mb_cols - 1 - mb_col) * 16) + (VP8BORDERINPIXELS - 16);

      /* Other than for the first frame do a motion search */
      if (cm->current_video_frame > 0) {
        BLOCKD *d = &x->e_mbd.block[0];
        MV tmp_mv = { 0, 0 };
        int tmp_err;
        int motion_error = INT_MAX;
        int raw_motion_error = INT_MAX;

        /* Simple 0,0 motion with no mv overhead */
        zz_motion_search(x, cpi->last_frame_unscaled_source, &raw_motion_error,
                         lst_yv12, &motion_error, recon_yoffset);
        d->bmi.mv.as_mv.row = 0;
        d->bmi.mv.as_mv.col = 0;

        if (raw_motion_error < cpi->oxcf.encode_breakout) {
          goto skip_motion_search;
        }

        /* Test last reference frame using the previous best mv as the
         * starting point (best reference) for the search
         */
        first_pass_motion_search(cpi, x, &best_ref_mv, &d->bmi.mv.as_mv,
                                 lst_yv12, &motion_error, recon_yoffset);

        /* If the current best reference mv is not centred on 0,0
         * then do a 0,0 based search as well
         */
        if (best_ref_mv.as_int) {
          tmp_err = INT_MAX;
          first_pass_motion_search(cpi, x, &zero_ref_mv, &tmp_mv, lst_yv12,
                                   &tmp_err, recon_yoffset);

          if (tmp_err < motion_error) {
            motion_error = tmp_err;
            d->bmi.mv.as_mv.row = tmp_mv.row;
            d->bmi.mv.as_mv.col = tmp_mv.col;
          }
        }

        /* Experimental search in a second reference frame ((0,0)
         * based only)
         */
        if (cm->current_video_frame > 1) {
          first_pass_motion_search(cpi, x, &zero_ref_mv, &tmp_mv, gld_yv12,
                                   &gf_motion_error, recon_yoffset);

          if ((gf_motion_error < motion_error) &&
              (gf_motion_error < this_error)) {
            second_ref_count++;
          }

          /* Reset to last frame as reference buffer */
          xd->pre.y_buffer = lst_yv12->y_buffer + recon_yoffset;
          xd->pre.u_buffer = lst_yv12->u_buffer + recon_uvoffset;
          xd->pre.v_buffer = lst_yv12->v_buffer + recon_uvoffset;
        }

      skip_motion_search:
        /* Intra assumed best */
        best_ref_mv.as_int = 0;

        if (motion_error <= this_error) {
          /* Keep a count of cases where the inter and intra were
           * very close and very low. This helps with scene cut
           * detection for example in cropped clips with black bars
           * at the sides or top and bottom.
           */
          if ((((this_error - intrapenalty) * 9) <= (motion_error * 10)) &&
              (this_error < (2 * intrapenalty))) {
            neutral_count++;
          }

          d->bmi.mv.as_mv.row *= 8;
          d->bmi.mv.as_mv.col *= 8;
          this_error = motion_error;
          vp8_set_mbmode_and_mvs(x, NEWMV, &d->bmi.mv);
          vp8_encode_inter16x16y(x);
          sum_mvr += d->bmi.mv.as_mv.row;
          sum_mvr_abs += abs(d->bmi.mv.as_mv.row);
          sum_mvc += d->bmi.mv.as_mv.col;
          sum_mvc_abs += abs(d->bmi.mv.as_mv.col);
          sum_mvrs += d->bmi.mv.as_mv.row * d->bmi.mv.as_mv.row;
          sum_mvcs += d->bmi.mv.as_mv.col * d->bmi.mv.as_mv.col;
          intercount++;

          best_ref_mv.as_int = d->bmi.mv.as_int;

          /* Was the vector non-zero */
          if (d->bmi.mv.as_int) {
            mvcount++;

            /* Was it different from the last non zero vector */
            if (d->bmi.mv.as_int != lastmv_as_int) new_mv_count++;
            lastmv_as_int = d->bmi.mv.as_int;

            /* Does the Row vector point inwards or outwards */
            if (mb_row < cm->mb_rows / 2) {
              if (d->bmi.mv.as_mv.row > 0) {
                sum_in_vectors--;
              } else if (d->bmi.mv.as_mv.row < 0) {
                sum_in_vectors++;
              }
            } else if (mb_row > cm->mb_rows / 2) {
              if (d->bmi.mv.as_mv.row > 0) {
                sum_in_vectors++;
              } else if (d->bmi.mv.as_mv.row < 0) {
                sum_in_vectors--;
              }
            }

            /* Does the Row vector point inwards or outwards */
            if (mb_col < cm->mb_cols / 2) {
              if (d->bmi.mv.as_mv.col > 0) {
                sum_in_vectors--;
              } else if (d->bmi.mv.as_mv.col < 0) {
                sum_in_vectors++;
              }
            } else if (mb_col > cm->mb_cols / 2) {
              if (d->bmi.mv.as_mv.col > 0) {
                sum_in_vectors++;
              } else if (d->bmi.mv.as_mv.col < 0) {
                sum_in_vectors--;
              }
            }
          }
        }
      }

      coded_error += (int64_t)this_error;

      /* adjust to the next column of macroblocks */
      x->src.y_buffer += 16;
      x->src.u_buffer += 8;
      x->src.v_buffer += 8;

      recon_yoffset += 16;
      recon_uvoffset += 8;
    }

    /* adjust to the next row of mbs */
    x->src.y_buffer += 16 * x->src.y_stride - 16 * cm->mb_cols;
    x->src.u_buffer += 8 * x->src.uv_stride - 8 * cm->mb_cols;
    x->src.v_buffer += 8 * x->src.uv_stride - 8 * cm->mb_cols;

    /* extend the recon for intra prediction */
    vp8_extend_mb_row(new_yv12, xd->dst.y_buffer + 16, xd->dst.u_buffer + 8,
                      xd->dst.v_buffer + 8);
    vpx_clear_system_state();
  }

  vpx_clear_system_state();
  {
    double weight = 0.0;

    FIRSTPASS_STATS fps;

    fps.frame = cm->current_video_frame;
    fps.intra_error = (double)(intra_error >> 8);
    fps.coded_error = (double)(coded_error >> 8);
    weight = simple_weight(cpi->Source);

    if (weight < 0.1) weight = 0.1;

    fps.ssim_weighted_pred_err = fps.coded_error * weight;

    fps.pcnt_inter = 0.0;
    fps.pcnt_motion = 0.0;
    fps.MVr = 0.0;
    fps.mvr_abs = 0.0;
    fps.MVc = 0.0;
    fps.mvc_abs = 0.0;
    fps.MVrv = 0.0;
    fps.MVcv = 0.0;
    fps.mv_in_out_count = 0.0;
    fps.new_mv_count = 0.0;
    fps.count = 1.0;

    fps.pcnt_inter = 1.0 * (double)intercount / cm->MBs;
    fps.pcnt_second_ref = 1.0 * (double)second_ref_count / cm->MBs;
    fps.pcnt_neutral = 1.0 * (double)neutral_count / cm->MBs;

    if (mvcount > 0) {
      fps.MVr = (double)sum_mvr / (double)mvcount;
      fps.mvr_abs = (double)sum_mvr_abs / (double)mvcount;
      fps.MVc = (double)sum_mvc / (double)mvcount;
      fps.mvc_abs = (double)sum_mvc_abs / (double)mvcount;
      fps.MVrv = ((double)sum_mvrs - (fps.MVr * fps.MVr / (double)mvcount)) /
                 (double)mvcount;
      fps.MVcv = ((double)sum_mvcs - (fps.MVc * fps.MVc / (double)mvcount)) /
                 (double)mvcount;
      fps.mv_in_out_count = (double)sum_in_vectors / (double)(mvcount * 2);
      fps.new_mv_count = new_mv_count;

      fps.pcnt_motion = 1.0 * (double)mvcount / cpi->common.MBs;
    }

    /* TODO:  handle the case when duration is set to 0, or something less
     * than the full time between subsequent cpi->source_time_stamps
     */
    fps.duration = (double)(cpi->source->ts_end - cpi->source->ts_start);

    /* don't want to do output stats with a stack variable! */
    cpi->twopass.this_frame_stats = fps;
    output_stats(cpi->output_pkt_list, &cpi->twopass.this_frame_stats);
    accumulate_stats(&cpi->twopass.total_stats, &fps);
  }

  /* Copy the previous Last Frame into the GF buffer if specific
   * conditions for doing so are met
   */
  if ((cm->current_video_frame > 0) &&
      (cpi->twopass.this_frame_stats.pcnt_inter > 0.20) &&
      ((cpi->twopass.this_frame_stats.intra_error /
        DOUBLE_DIVIDE_CHECK(cpi->twopass.this_frame_stats.coded_error)) >
       2.0)) {
    vp8_yv12_copy_frame(lst_yv12, gld_yv12);
  }

  /* swap frame pointers so last frame refers to the frame we just
   * compressed
   */
  vp8_swap_yv12_buffer(lst_yv12, new_yv12);
  vp8_yv12_extend_frame_borders(lst_yv12);

  /* Special case for the first frame. Copy into the GF buffer as a
   * second reference.
   */
  if (cm->current_video_frame == 0) {
    vp8_yv12_copy_frame(lst_yv12, gld_yv12);
  }

  cm->current_video_frame++;
}
extern const int vp8_bits_per_mb[2][QINDEX_RANGE];

/* Estimate a cost per mb attributable to overheads such as the coding of
 * modes and motion vectors.
 * Currently simplistic in its assumptions for testing.
 */

static double bitcost(double prob) {
  if (prob > 0.000122) {
    return -log(prob) / log(2.0);
  } else {
    return 13.0;
  }
}
static int64_t estimate_modemvcost(VP8_COMP *cpi, FIRSTPASS_STATS *fpstats) {
  int mv_cost;
  int64_t mode_cost;

  double av_pct_inter = fpstats->pcnt_inter / fpstats->count;
  double av_pct_motion = fpstats->pcnt_motion / fpstats->count;
  double av_intra = (1.0 - av_pct_inter);

  double zz_cost;
  double motion_cost;
  double intra_cost;

  zz_cost = bitcost(av_pct_inter - av_pct_motion);
  motion_cost = bitcost(av_pct_motion);
  intra_cost = bitcost(av_intra);

  /* Estimate of extra bits per mv overhead for mbs
   * << 9 is the normalization to the (bits * 512) used in vp8_bits_per_mb
   */
  mv_cost = ((int)(fpstats->new_mv_count / fpstats->count) * 8) << 9;

  /* Crude estimate of overhead cost from modes
   * << 9 is the normalization to (bits * 512) used in vp8_bits_per_mb
   */
  mode_cost =
      (int64_t)((((av_pct_inter - av_pct_motion) * zz_cost) +
                 (av_pct_motion * motion_cost) + (av_intra * intra_cost)) *
                cpi->common.MBs) *
      512;

  return mv_cost + mode_cost;
}

static double calc_correction_factor(double err_per_mb, double err_devisor,
                                     double pt_low, double pt_high, int Q) {
  double power_term;
  double error_term = err_per_mb / err_devisor;
  double correction_factor;

  /* Adjustment based on Q to power term. */
  power_term = pt_low + (Q * 0.01);
  power_term = (power_term > pt_high) ? pt_high : power_term;

  /* Adjustments to error term */
  /* TBD */

  /* Calculate correction factor */
  correction_factor = pow(error_term, power_term);

  /* Clip range */
  correction_factor = (correction_factor < 0.05)  ? 0.05
                      : (correction_factor > 5.0) ? 5.0
                                                  : correction_factor;

  return correction_factor;
}

static int estimate_max_q(VP8_COMP *cpi, FIRSTPASS_STATS *fpstats,
                          int section_target_bandwitdh, int overhead_bits) {
  int Q;
  int num_mbs = cpi->common.MBs;
  int target_norm_bits_per_mb;

  double section_err = (fpstats->coded_error / fpstats->count);
  double err_per_mb = section_err / num_mbs;
  double err_correction_factor;
  double speed_correction = 1.0;
  int overhead_bits_per_mb;

  if (section_target_bandwitdh <= 0) {
    return cpi->twopass.maxq_max_limit; /* Highest value allowed */
  }

  target_norm_bits_per_mb = (section_target_bandwitdh < (1 << 20))
                                ? (512 * section_target_bandwitdh) / num_mbs
                                : 512 * (section_target_bandwitdh / num_mbs);

  /* Calculate a corrective factor based on a rolling ratio of bits spent
   * vs target bits
   */
  if ((cpi->rolling_target_bits > 0) &&
      (cpi->active_worst_quality < cpi->worst_quality)) {
    double rolling_ratio;

    rolling_ratio =
        (double)cpi->rolling_actual_bits / (double)cpi->rolling_target_bits;

    if (rolling_ratio < 0.95) {
      cpi->twopass.est_max_qcorrection_factor -= 0.005;
    } else if (rolling_ratio > 1.05) {
      cpi->twopass.est_max_qcorrection_factor += 0.005;
    }

    cpi->twopass.est_max_qcorrection_factor =
        (cpi->twopass.est_max_qcorrection_factor < 0.1) ? 0.1
        : (cpi->twopass.est_max_qcorrection_factor > 10.0)
            ? 10.0
            : cpi->twopass.est_max_qcorrection_factor;
  }

  /* Corrections for higher compression speed settings
   * (reduced compression expected)
   */
  if ((cpi->compressor_speed == 3) || (cpi->compressor_speed == 1)) {
    if (cpi->oxcf.cpu_used <= 5) {
      speed_correction = 1.04 + (cpi->oxcf.cpu_used * 0.04);
    } else {
      speed_correction = 1.25;
    }
  }

  /* Estimate of overhead bits per mb */
  /* Correction to overhead bits for min allowed Q. */
  overhead_bits_per_mb = overhead_bits / num_mbs;
  overhead_bits_per_mb = (int)(overhead_bits_per_mb *
                               pow(0.98, (double)cpi->twopass.maxq_min_limit));

  /* Try and pick a max Q that will be high enough to encode the
   * content at the given rate.
   */
  for (Q = cpi->twopass.maxq_min_limit; Q < cpi->twopass.maxq_max_limit; ++Q) {
    int bits_per_mb_at_this_q;

    /* Error per MB based correction factor */
    err_correction_factor =
        calc_correction_factor(err_per_mb, 150.0, 0.40, 0.90, Q);

    bits_per_mb_at_this_q =
        vp8_bits_per_mb[INTER_FRAME][Q] + overhead_bits_per_mb;

    bits_per_mb_at_this_q =
        (int)(.5 + err_correction_factor * speed_correction *
                       cpi->twopass.est_max_qcorrection_factor *
                       cpi->twopass.section_max_qfactor *
                       (double)bits_per_mb_at_this_q);

    /* Mode and motion overhead */
    /* As Q rises in real encode loop rd code will force overhead down
     * We make a crude adjustment for this here as *.98 per Q step.
     */
    overhead_bits_per_mb = (int)((double)overhead_bits_per_mb * 0.98);

    if (bits_per_mb_at_this_q <= target_norm_bits_per_mb) break;
  }

  /* Restriction on active max q for constrained quality mode. */
  if ((cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) &&
      (Q < cpi->cq_target_quality)) {
    Q = cpi->cq_target_quality;
  }

  /* Adjust maxq_min_limit and maxq_max_limit limits based on
   * average q observed in clip for non kf/gf.arf frames
   * Give average a chance to settle though.
   */
  if ((cpi->ni_frames > ((int)cpi->twopass.total_stats.count >> 8)) &&
      (cpi->ni_frames > 150)) {
    cpi->twopass.maxq_max_limit = ((cpi->ni_av_qi + 32) < cpi->worst_quality)
                                      ? (cpi->ni_av_qi + 32)
                                      : cpi->worst_quality;
    cpi->twopass.maxq_min_limit = ((cpi->ni_av_qi - 32) > cpi->best_quality)
                                      ? (cpi->ni_av_qi - 32)
                                      : cpi->best_quality;
  }

  return Q;
}

/* For cq mode estimate a cq level that matches the observed
 * complexity and data rate.
 */
static int estimate_cq(VP8_COMP *cpi, FIRSTPASS_STATS *fpstats,
                       int section_target_bandwitdh, int overhead_bits) {
  int Q;
  int num_mbs = cpi->common.MBs;
  int target_norm_bits_per_mb;

  double section_err = (fpstats->coded_error / fpstats->count);
  double err_per_mb = section_err / num_mbs;
  double err_correction_factor;
  double speed_correction = 1.0;
  double clip_iiratio;
  double clip_iifactor;
  int overhead_bits_per_mb;

  target_norm_bits_per_mb = (section_target_bandwitdh < (1 << 20))
                                ? (512 * section_target_bandwitdh) / num_mbs
                                : 512 * (section_target_bandwitdh / num_mbs);

  /* Estimate of overhead bits per mb */
  overhead_bits_per_mb = overhead_bits / num_mbs;

  /* Corrections for higher compression speed settings
   * (reduced compression expected)
   */
  if ((cpi->compressor_speed == 3) || (cpi->compressor_speed == 1)) {
    if (cpi->oxcf.cpu_used <= 5) {
      speed_correction = 1.04 + (cpi->oxcf.cpu_used * 0.04);
    } else {
      speed_correction = 1.25;
    }
  }

  /* II ratio correction factor for clip as a whole */
  clip_iiratio = cpi->twopass.total_stats.intra_error /
                 DOUBLE_DIVIDE_CHECK(cpi->twopass.total_stats.coded_error);
  clip_iifactor = 1.0 - ((clip_iiratio - 10.0) * 0.025);
  if (clip_iifactor < 0.80) clip_iifactor = 0.80;

  /* Try and pick a Q that can encode the content at the given rate. */
  for (Q = 0; Q < MAXQ; ++Q) {
    int bits_per_mb_at_this_q;

    /* Error per MB based correction factor */
    err_correction_factor =
        calc_correction_factor(err_per_mb, 100.0, 0.40, 0.90, Q);

    bits_per_mb_at_this_q =
        vp8_bits_per_mb[INTER_FRAME][Q] + overhead_bits_per_mb;

    bits_per_mb_at_this_q =
        (int)(.5 + err_correction_factor * speed_correction * clip_iifactor *
                       (double)bits_per_mb_at_this_q);

    /* Mode and motion overhead */
    /* As Q rises in real encode loop rd code will force overhead down
     * We make a crude adjustment for this here as *.98 per Q step.
     */
    overhead_bits_per_mb = (int)((double)overhead_bits_per_mb * 0.98);

    if (bits_per_mb_at_this_q <= target_norm_bits_per_mb) break;
  }

  /* Clip value to range "best allowed to (worst allowed - 1)" */
  Q = cq_level[Q];
  if (Q >= cpi->worst_quality) Q = cpi->worst_quality - 1;
  if (Q < cpi->best_quality) Q = cpi->best_quality;

  return Q;
}

static int estimate_q(VP8_COMP *cpi, double section_err,
                      int section_target_bandwitdh) {
  int Q;
  int num_mbs = cpi->common.MBs;
  int target_norm_bits_per_mb;

  double err_per_mb = section_err / num_mbs;
  double err_correction_factor;
  double speed_correction = 1.0;

  target_norm_bits_per_mb = (section_target_bandwitdh < (1 << 20))
                                ? (512 * section_target_bandwitdh) / num_mbs
                                : 512 * (section_target_bandwitdh / num_mbs);

  /* Corrections for higher compression speed settings
   * (reduced compression expected)
   */
  if ((cpi->compressor_speed == 3) || (cpi->compressor_speed == 1)) {
    if (cpi->oxcf.cpu_used <= 5) {
      speed_correction = 1.04 + (cpi->oxcf.cpu_used * 0.04);
    } else {
      speed_correction = 1.25;
    }
  }

  /* Try and pick a Q that can encode the content at the given rate. */
  for (Q = 0; Q < MAXQ; ++Q) {
    int bits_per_mb_at_this_q;

    /* Error per MB based correction factor */
    err_correction_factor =
        calc_correction_factor(err_per_mb, 150.0, 0.40, 0.90, Q);

    bits_per_mb_at_this_q =
        (int)(.5 + (err_correction_factor * speed_correction *
                    cpi->twopass.est_max_qcorrection_factor *
                    (double)vp8_bits_per_mb[INTER_FRAME][Q] / 1.0));

    if (bits_per_mb_at_this_q <= target_norm_bits_per_mb) break;
  }

  return Q;
}

/* Estimate a worst case Q for a KF group */
static int estimate_kf_group_q(VP8_COMP *cpi, double section_err,
                               int section_target_bandwitdh,
                               double group_iiratio) {
  int Q;
  int num_mbs = cpi->common.MBs;
  int target_norm_bits_per_mb = (512 * section_target_bandwitdh) / num_mbs;
  int bits_per_mb_at_this_q;

  double err_per_mb = section_err / num_mbs;
  double err_correction_factor;
  double speed_correction = 1.0;
  double current_spend_ratio = 1.0;

  double pow_highq = (POW1 < 0.6) ? POW1 + 0.3 : 0.90;
  double pow_lowq = (POW1 < 0.7) ? POW1 + 0.1 : 0.80;

  double iiratio_correction_factor = 1.0;

  double combined_correction_factor;

  /* Trap special case where the target is <= 0 */
  if (target_norm_bits_per_mb <= 0) return MAXQ * 2;

  /* Calculate a corrective factor based on a rolling ratio of bits spent
   *  vs target bits
   * This is clamped to the range 0.1 to 10.0
   */
  if (cpi->long_rolling_target_bits <= 0) {
    current_spend_ratio = 10.0;
  } else {
    current_spend_ratio = (double)cpi->long_rolling_actual_bits /
                          (double)cpi->long_rolling_target_bits;
    current_spend_ratio = (current_spend_ratio > 10.0)  ? 10.0
                          : (current_spend_ratio < 0.1) ? 0.1
                                                        : current_spend_ratio;
  }

  /* Calculate a correction factor based on the quality of prediction in
   * the sequence as indicated by intra_inter error score ratio (IIRatio)
   * The idea here is to favour subsampling in the hardest sections vs
   * the easyest.
   */
  iiratio_correction_factor = 1.0 - ((group_iiratio - 6.0) * 0.1);

  if (iiratio_correction_factor < 0.5) iiratio_correction_factor = 0.5;

  /* Corrections for higher compression speed settings
   * (reduced compression expected)
   */
  if ((cpi->compressor_speed == 3) || (cpi->compressor_speed == 1)) {
    if (cpi->oxcf.cpu_used <= 5) {
      speed_correction = 1.04 + (cpi->oxcf.cpu_used * 0.04);
    } else {
      speed_correction = 1.25;
    }
  }

  /* Combine the various factors calculated above */
  combined_correction_factor =
      speed_correction * iiratio_correction_factor * current_spend_ratio;

  /* Try and pick a Q that should be high enough to encode the content at
   * the given rate.
   */
  for (Q = 0; Q < MAXQ; ++Q) {
    /* Error per MB based correction factor */
    err_correction_factor =
        calc_correction_factor(err_per_mb, 150.0, pow_lowq, pow_highq, Q);

    bits_per_mb_at_this_q =
        (int)(.5 + (err_correction_factor * combined_correction_factor *
                    (double)vp8_bits_per_mb[INTER_FRAME][Q]));

    if (bits_per_mb_at_this_q <= target_norm_bits_per_mb) break;
  }

  /* If we could not hit the target even at Max Q then estimate what Q
   * would have been required
   */
  while ((bits_per_mb_at_this_q > target_norm_bits_per_mb) &&
         (Q < (MAXQ * 2))) {
    bits_per_mb_at_this_q = (int)(0.96 * bits_per_mb_at_this_q);
    Q++;
  }

  return Q;
}

void vp8_init_second_pass(VP8_COMP *cpi) {
  FIRSTPASS_STATS this_frame;
  FIRSTPASS_STATS *start_pos;

  double two_pass_min_rate = (double)(cpi->oxcf.target_bandwidth *
                                      cpi->oxcf.two_pass_vbrmin_section / 100);

  zero_stats(&cpi->twopass.total_stats);
  zero_stats(&cpi->twopass.total_left_stats);

  if (!cpi->twopass.stats_in_end) return;

  cpi->twopass.total_stats = *cpi->twopass.stats_in_end;
  cpi->twopass.total_left_stats = cpi->twopass.total_stats;

  /* each frame can have a different duration, as the frame rate in the
   * source isn't guaranteed to be constant.   The frame rate prior to
   * the first frame encoded in the second pass is a guess.  However the
   * sum duration is not. Its calculated based on the actual durations of
   * all frames from the first pass.
   */
  vp8_new_framerate(cpi, 10000000.0 * cpi->twopass.total_stats.count /
                             cpi->twopass.total_stats.duration);

  cpi->twopass.bits_left = (int64_t)(cpi->twopass.total_stats.duration *
                                     cpi->oxcf.target_bandwidth / 10000000.0);
  cpi->twopass.bits_left -= (int64_t)(cpi->twopass.total_stats.duration *
                                      two_pass_min_rate / 10000000.0);

  /* Calculate a minimum intra value to be used in determining the IIratio
   * scores used in the second pass. We have this minimum to make sure
   * that clips that are static but "low complexity" in the intra domain
   * are still boosted appropriately for KF/GF/ARF
   */
  cpi->twopass.kf_intra_err_min = KF_MB_INTRA_MIN * cpi->common.MBs;
  cpi->twopass.gf_intra_err_min = GF_MB_INTRA_MIN * cpi->common.MBs;

  /* Scan the first pass file and calculate an average Intra / Inter error
   * score ratio for the sequence
   */
  {
    double sum_iiratio = 0.0;
    double IIRatio;

    start_pos = cpi->twopass.stats_in; /* Note starting "file" position */

    while (input_stats(cpi, &this_frame) != EOF) {
      IIRatio =
          this_frame.intra_error / DOUBLE_DIVIDE_CHECK(this_frame.coded_error);
      IIRatio = (IIRatio < 1.0) ? 1.0 : (IIRatio > 20.0) ? 20.0 : IIRatio;
      sum_iiratio += IIRatio;
    }

    cpi->twopass.avg_iiratio =
        sum_iiratio /
        DOUBLE_DIVIDE_CHECK((double)cpi->twopass.total_stats.count);

    /* Reset file position */
    reset_fpf_position(cpi, start_pos);
  }

  /* Scan the first pass file and calculate a modified total error based
   * upon the bias/power function used to allocate bits
   */
  {
    start_pos = cpi->twopass.stats_in; /* Note starting "file" position */

    cpi->twopass.modified_error_total = 0.0;
    cpi->twopass.modified_error_used = 0.0;

    while (input_stats(cpi, &this_frame) != EOF) {
      cpi->twopass.modified_error_total +=
          calculate_modified_err(cpi, &this_frame);
    }
    cpi->twopass.modified_error_left = cpi->twopass.modified_error_total;

    reset_fpf_position(cpi, start_pos); /* Reset file position */
  }
}

void vp8_end_second_pass(VP8_COMP *cpi) { (void)cpi; }

/* This function gives and estimate of how badly we believe the prediction
 * quality is decaying from frame to frame.
 */
static double get_prediction_decay_rate(FIRSTPASS_STATS *next_frame) {
  double prediction_decay_rate;
  double motion_decay;
  double motion_pct = next_frame->pcnt_motion;

  /* Initial basis is the % mbs inter coded */
  prediction_decay_rate = next_frame->pcnt_inter;

  /* High % motion -> somewhat higher decay rate */
  motion_decay = (1.0 - (motion_pct / 20.0));
  if (motion_decay < prediction_decay_rate) {
    prediction_decay_rate = motion_decay;
  }

  /* Adjustment to decay rate based on speed of motion */
  {
    double this_mv_rabs;
    double this_mv_cabs;
    double distance_factor;

    this_mv_rabs = fabs(next_frame->mvr_abs * motion_pct);
    this_mv_cabs = fabs(next_frame->mvc_abs * motion_pct);

    distance_factor =
        sqrt((this_mv_rabs * this_mv_rabs) + (this_mv_cabs * this_mv_cabs)) /
        250.0;
    distance_factor = ((distance_factor > 1.0) ? 0.0 : (1.0 - distance_factor));
    if (distance_factor < prediction_decay_rate) {
      prediction_decay_rate = distance_factor;
    }
  }

  return prediction_decay_rate;
}

/* Function to test for a condition where a complex transition is followed
 * by a static section. For example in slide shows where there is a fade
 * between slides. This is to help with more optimal kf and gf positioning.
 */
static int detect_transition_to_still(VP8_COMP *cpi, int frame_interval,
                                      int still_interval,
                                      double loop_decay_rate,
                                      double decay_accumulator) {
  int trans_to_still = 0;

  /* Break clause to detect very still sections after motion
   * For example a static image after a fade or other transition
   * instead of a clean scene cut.
   */
  if ((frame_interval > MIN_GF_INTERVAL) && (loop_decay_rate >= 0.999) &&
      (decay_accumulator < 0.9)) {
    int j;
    FIRSTPASS_STATS *position = cpi->twopass.stats_in;
    FIRSTPASS_STATS tmp_next_frame;
    double decay_rate;

    /* Look ahead a few frames to see if static condition persists... */
    for (j = 0; j < still_interval; ++j) {
      if (EOF == input_stats(cpi, &tmp_next_frame)) break;

      decay_rate = get_prediction_decay_rate(&tmp_next_frame);
      if (decay_rate < 0.999) break;
    }
    /* Reset file position */
    reset_fpf_position(cpi, position);

    /* Only if it does do we signal a transition to still */
    if (j == still_interval) trans_to_still = 1;
  }

  return trans_to_still;
}

/* This function detects a flash through the high relative pcnt_second_ref
 * score in the frame following a flash frame. The offset passed in should
 * reflect this
 */
static int detect_flash(VP8_COMP *cpi, int offset) {
  FIRSTPASS_STATS next_frame;

  int flash_detected = 0;

  /* Read the frame data. */
  /* The return is 0 (no flash detected) if not a valid frame */
  if (read_frame_stats(cpi, &next_frame, offset) != EOF) {
    /* What we are looking for here is a situation where there is a
     * brief break in prediction (such as a flash) but subsequent frames
     * are reasonably well predicted by an earlier (pre flash) frame.
     * The recovery after a flash is indicated by a high pcnt_second_ref
     * comapred to pcnt_inter.
     */
    if ((next_frame.pcnt_second_ref > next_frame.pcnt_inter) &&
        (next_frame.pcnt_second_ref >= 0.5)) {
      flash_detected = 1;

      /*if (1)
      {
          FILE *f = fopen("flash.stt", "a");
          fprintf(f, "%8.0f %6.2f %6.2f\n",
              next_frame.frame,
              next_frame.pcnt_inter,
              next_frame.pcnt_second_ref);
          fclose(f);
      }*/
    }
  }

  return flash_detected;
}

/* Update the motion related elements to the GF arf boost calculation */
static void accumulate_frame_motion_stats(FIRSTPASS_STATS *this_frame,
                                          double *this_frame_mv_in_out,
                                          double *mv_in_out_accumulator,
                                          double *abs_mv_in_out_accumulator,
                                          double *mv_ratio_accumulator) {
  double this_frame_mvr_ratio;
  double this_frame_mvc_ratio;
  double motion_pct;

  /* Accumulate motion stats. */
  motion_pct = this_frame->pcnt_motion;

  /* Accumulate Motion In/Out of frame stats */
  *this_frame_mv_in_out = this_frame->mv_in_out_count * motion_pct;
  *mv_in_out_accumulator += this_frame->mv_in_out_count * motion_pct;
  *abs_mv_in_out_accumulator += fabs(this_frame->mv_in_out_count * motion_pct);

  /* Accumulate a measure of how uniform (or conversely how random)
   * the motion field is. (A ratio of absmv / mv)
   */
  if (motion_pct > 0.05) {
    this_frame_mvr_ratio =
        fabs(this_frame->mvr_abs) / DOUBLE_DIVIDE_CHECK(fabs(this_frame->MVr));

    this_frame_mvc_ratio =
        fabs(this_frame->mvc_abs) / DOUBLE_DIVIDE_CHECK(fabs(this_frame->MVc));

    *mv_ratio_accumulator += (this_frame_mvr_ratio < this_frame->mvr_abs)
                                 ? (this_frame_mvr_ratio * motion_pct)
                                 : this_frame->mvr_abs * motion_pct;

    *mv_ratio_accumulator += (this_frame_mvc_ratio < this_frame->mvc_abs)
                                 ? (this_frame_mvc_ratio * motion_pct)
                                 : this_frame->mvc_abs * motion_pct;
  }
}

/* Calculate a baseline boost number for the current frame. */
static double calc_frame_boost(VP8_COMP *cpi, FIRSTPASS_STATS *this_frame,
                               double this_frame_mv_in_out) {
  double frame_boost;

  /* Underlying boost factor is based on inter intra error ratio */
  if (this_frame->intra_error > cpi->twopass.gf_intra_err_min) {
    frame_boost = (IIFACTOR * this_frame->intra_error /
                   DOUBLE_DIVIDE_CHECK(this_frame->coded_error));
  } else {
    frame_boost = (IIFACTOR * cpi->twopass.gf_intra_err_min /
                   DOUBLE_DIVIDE_CHECK(this_frame->coded_error));
  }

  /* Increase boost for frames where new data coming into frame
   * (eg zoom out). Slightly reduce boost if there is a net balance
   * of motion out of the frame (zoom in).
   * The range for this_frame_mv_in_out is -1.0 to +1.0
   */
  if (this_frame_mv_in_out > 0.0) {
    frame_boost += frame_boost * (this_frame_mv_in_out * 2.0);
    /* In extreme case boost is halved */
  } else {
    frame_boost += frame_boost * (this_frame_mv_in_out / 2.0);
  }

  /* Clip to maximum */
  if (frame_boost > GF_RMAX) frame_boost = GF_RMAX;

  return frame_boost;
}

#if NEW_BOOST
static int calc_arf_boost(VP8_COMP *cpi, int offset, int f_frames, int b_frames,
                          int *f_boost, int *b_boost) {
  FIRSTPASS_STATS this_frame;

  int i;
  double boost_score = 0.0;
  double mv_ratio_accumulator = 0.0;
  double decay_accumulator = 1.0;
  double this_frame_mv_in_out = 0.0;
  double mv_in_out_accumulator = 0.0;
  double abs_mv_in_out_accumulator = 0.0;
  double r;
  int flash_detected = 0;

  /* Search forward from the proposed arf/next gf position */
  for (i = 0; i < f_frames; ++i) {
    if (read_frame_stats(cpi, &this_frame, (i + offset)) == EOF) break;

    /* Update the motion related elements to the boost calculation */
    accumulate_frame_motion_stats(
        &this_frame, &this_frame_mv_in_out, &mv_in_out_accumulator,
        &abs_mv_in_out_accumulator, &mv_ratio_accumulator);

    /* Calculate the baseline boost number for this frame */
    r = calc_frame_boost(cpi, &this_frame, this_frame_mv_in_out);

    /* We want to discount the flash frame itself and the recovery
     * frame that follows as both will have poor scores.
     */
    flash_detected =
        detect_flash(cpi, (i + offset)) || detect_flash(cpi, (i + offset + 1));

    /* Cumulative effect of prediction quality decay */
    if (!flash_detected) {
      decay_accumulator =
          decay_accumulator * get_prediction_decay_rate(&this_frame);
      decay_accumulator = decay_accumulator < 0.1 ? 0.1 : decay_accumulator;
    }
    boost_score += (decay_accumulator * r);

    /* Break out conditions. */
    if ((!flash_detected) &&
        ((mv_ratio_accumulator > 100.0) || (abs_mv_in_out_accumulator > 3.0) ||
         (mv_in_out_accumulator < -2.0))) {
      break;
    }
  }

  *f_boost = (int)(boost_score * 100.0) >> 4;

  /* Reset for backward looking loop */
  boost_score = 0.0;
  mv_ratio_accumulator = 0.0;
  decay_accumulator = 1.0;
  this_frame_mv_in_out = 0.0;
  mv_in_out_accumulator = 0.0;
  abs_mv_in_out_accumulator = 0.0;

  /* Search forward from the proposed arf/next gf position */
  for (i = -1; i >= -b_frames; i--) {
    if (read_frame_stats(cpi, &this_frame, (i + offset)) == EOF) break;

    /* Update the motion related elements to the boost calculation */
    accumulate_frame_motion_stats(
        &this_frame, &this_frame_mv_in_out, &mv_in_out_accumulator,
        &abs_mv_in_out_accumulator, &mv_ratio_accumulator);

    /* Calculate the baseline boost number for this frame */
    r = calc_frame_boost(cpi, &this_frame, this_frame_mv_in_out);

    /* We want to discount the flash frame itself and the recovery
     * frame that follows as both will have poor scores.
     */
    flash_detected =
        detect_flash(cpi, (i + offset)) || detect_flash(cpi, (i + offset + 1));

    /* Cumulative effect of prediction quality decay */
    if (!flash_detected) {
      decay_accumulator =
          decay_accumulator * get_prediction_decay_rate(&this_frame);
      decay_accumulator = decay_accumulator < 0.1 ? 0.1 : decay_accumulator;
    }

    boost_score += (decay_accumulator * r);

    /* Break out conditions. */
    if ((!flash_detected) &&
        ((mv_ratio_accumulator > 100.0) || (abs_mv_in_out_accumulator > 3.0) ||
         (mv_in_out_accumulator < -2.0))) {
      break;
    }
  }
  *b_boost = (int)(boost_score * 100.0) >> 4;

  return (*f_boost + *b_boost);
}
#endif

/* Analyse and define a gf/arf group . */
static void define_gf_group(VP8_COMP *cpi, FIRSTPASS_STATS *this_frame) {
  FIRSTPASS_STATS next_frame;
  FIRSTPASS_STATS *start_pos;
  int i;
  double r;
  double boost_score = 0.0;
  double old_boost_score = 0.0;
  double gf_group_err = 0.0;
  double gf_first_frame_err = 0.0;
  double mod_frame_err = 0.0;

  double mv_ratio_accumulator = 0.0;
  double decay_accumulator = 1.0;

  double loop_decay_rate = 1.00; /* Starting decay rate */

  double this_frame_mv_in_out = 0.0;
  double mv_in_out_accumulator = 0.0;
  double abs_mv_in_out_accumulator = 0.0;

  int max_bits = frame_max_bits(cpi); /* Max for a single frame */

  unsigned int allow_alt_ref =
      cpi->oxcf.play_alternate && cpi->oxcf.lag_in_frames;

  int alt_boost = 0;
  int f_boost = 0;
  int b_boost = 0;
  int flash_detected;

  cpi->twopass.gf_group_bits = 0;
  cpi->twopass.gf_decay_rate = 0;

  vpx_clear_system_state();

  start_pos = cpi->twopass.stats_in;

  memset(&next_frame, 0, sizeof(next_frame)); /* assure clean */

  /* Load stats for the current frame. */
  mod_frame_err = calculate_modified_err(cpi, this_frame);

  /* Note the error of the frame at the start of the group (this will be
   * the GF frame error if we code a normal gf
   */
  gf_first_frame_err = mod_frame_err;

  /* Special treatment if the current frame is a key frame (which is also
   * a gf). If it is then its error score (and hence bit allocation) need
   * to be subtracted out from the calculation for the GF group
   */
  if (cpi->common.frame_type == KEY_FRAME) gf_group_err -= gf_first_frame_err;

  /* Scan forward to try and work out how many frames the next gf group
   * should contain and what level of boost is appropriate for the GF
   * or ARF that will be coded with the group
   */
  i = 0;

  while (((i < cpi->twopass.static_scene_max_gf_interval) ||
          ((cpi->twopass.frames_to_key - i) < MIN_GF_INTERVAL)) &&
         (i < cpi->twopass.frames_to_key)) {
    i++;

    /* Accumulate error score of frames in this gf group */
    mod_frame_err = calculate_modified_err(cpi, this_frame);

    gf_group_err += mod_frame_err;

    if (EOF == input_stats(cpi, &next_frame)) break;

    /* Test for the case where there is a brief flash but the prediction
     * quality back to an earlier frame is then restored.
     */
    flash_detected = detect_flash(cpi, 0);

    /* Update the motion related elements to the boost calculation */
    accumulate_frame_motion_stats(
        &next_frame, &this_frame_mv_in_out, &mv_in_out_accumulator,
        &abs_mv_in_out_accumulator, &mv_ratio_accumulator);

    /* Calculate a baseline boost number for this frame */
    r = calc_frame_boost(cpi, &next_frame, this_frame_mv_in_out);

    /* Cumulative effect of prediction quality decay */
    if (!flash_detected) {
      loop_decay_rate = get_prediction_decay_rate(&next_frame);
      decay_accumulator = decay_accumulator * loop_decay_rate;
      decay_accumulator = decay_accumulator < 0.1 ? 0.1 : decay_accumulator;
    }
    boost_score += (decay_accumulator * r);

    /* Break clause to detect very still sections after motion
     * For example a staic image after a fade or other transition.
     */
    if (detect_transition_to_still(cpi, i, 5, loop_decay_rate,
                                   decay_accumulator)) {
      allow_alt_ref = 0;
      boost_score = old_boost_score;
      break;
    }

    /* Break out conditions. */
    if (
        /* Break at cpi->max_gf_interval unless almost totally static */
        (i >= cpi->max_gf_interval && (decay_accumulator < 0.995)) ||
        (
            /* Don't break out with a very short interval */
            (i > MIN_GF_INTERVAL) &&
            /* Don't break out very close to a key frame */
            ((cpi->twopass.frames_to_key - i) >= MIN_GF_INTERVAL) &&
            ((boost_score > 20.0) || (next_frame.pcnt_inter < 0.75)) &&
            (!flash_detected) &&
            ((mv_ratio_accumulator > 100.0) ||
             (abs_mv_in_out_accumulator > 3.0) ||
             (mv_in_out_accumulator < -2.0) ||
             ((boost_score - old_boost_score) < 2.0)))) {
      boost_score = old_boost_score;
      break;
    }

    *this_frame = next_frame;

    old_boost_score = boost_score;
  }

  cpi->twopass.gf_decay_rate =
      (i > 0) ? (int)(100.0 * (1.0 - decay_accumulator)) / i : 0;

  /* When using CBR apply additional buffer related upper limits */
  if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
    double max_boost;

    /* For cbr apply buffer related limits */
    if (cpi->drop_frames_allowed) {
      int64_t df_buffer_level = cpi->oxcf.drop_frames_water_mark *
                                (cpi->oxcf.optimal_buffer_level / 100);

      if (cpi->buffer_level > df_buffer_level) {
        max_boost =
            ((double)((cpi->buffer_level - df_buffer_level) * 2 / 3) * 16.0) /
            DOUBLE_DIVIDE_CHECK((double)cpi->av_per_frame_bandwidth);
      } else {
        max_boost = 0.0;
      }
    } else if (cpi->buffer_level > 0) {
      max_boost = ((double)(cpi->buffer_level * 2 / 3) * 16.0) /
                  DOUBLE_DIVIDE_CHECK((double)cpi->av_per_frame_bandwidth);
    } else {
      max_boost = 0.0;
    }

    if (boost_score > max_boost) boost_score = max_boost;
  }

  /* Don't allow conventional gf too near the next kf */
  if ((cpi->twopass.frames_to_key - i) < MIN_GF_INTERVAL) {
    while (i < cpi->twopass.frames_to_key) {
      i++;

      if (EOF == input_stats(cpi, this_frame)) break;

      if (i < cpi->twopass.frames_to_key) {
        mod_frame_err = calculate_modified_err(cpi, this_frame);
        gf_group_err += mod_frame_err;
      }
    }
  }

  cpi->gfu_boost = (int)(boost_score * 100.0) >> 4;

#if NEW_BOOST
  /* Alterrnative boost calculation for alt ref */
  alt_boost = calc_arf_boost(cpi, 0, (i - 1), (i - 1), &f_boost, &b_boost);
#endif

  /* Should we use the alternate reference frame */
  if (allow_alt_ref && (i >= MIN_GF_INTERVAL) &&
      /* don't use ARF very near next kf */
      (i <= (cpi->twopass.frames_to_key - MIN_GF_INTERVAL)) &&
#if NEW_BOOST
      ((next_frame.pcnt_inter > 0.75) || (next_frame.pcnt_second_ref > 0.5)) &&
      ((mv_in_out_accumulator / (double)i > -0.2) ||
       (mv_in_out_accumulator > -2.0)) &&
      (b_boost > 100) && (f_boost > 100))
#else
      (next_frame.pcnt_inter > 0.75) &&
      ((mv_in_out_accumulator / (double)i > -0.2) ||
       (mv_in_out_accumulator > -2.0)) &&
      (cpi->gfu_boost > 100) &&
      (cpi->twopass.gf_decay_rate <=
       (ARF_DECAY_THRESH + (cpi->gfu_boost / 200))))
#endif
  {
    int Boost;
    int allocation_chunks;
    int Q =
        (cpi->oxcf.fixed_q < 0) ? cpi->last_q[INTER_FRAME] : cpi->oxcf.fixed_q;
    int tmp_q;
    int arf_frame_bits = 0;
    int group_bits;

#if NEW_BOOST
    cpi->gfu_boost = alt_boost;
#endif

    /* Estimate the bits to be allocated to the group as a whole */
    if ((cpi->twopass.kf_group_bits > 0) &&
        (cpi->twopass.kf_group_error_left > 0)) {
      group_bits =
          (int)((double)cpi->twopass.kf_group_bits *
                (gf_group_err / (double)cpi->twopass.kf_group_error_left));
    } else {
      group_bits = 0;
    }

/* Boost for arf frame */
#if NEW_BOOST
    Boost = (alt_boost * GFQ_ADJUSTMENT) / 100;
#else
    Boost = (cpi->gfu_boost * 3 * GFQ_ADJUSTMENT) / (2 * 100);
#endif
    Boost += (i * 50);

    /* Set max and minimum boost and hence minimum allocation */
    if (Boost > ((cpi->baseline_gf_interval + 1) * 200)) {
      Boost = ((cpi->baseline_gf_interval + 1) * 200);
    } else if (Boost < 125) {
      Boost = 125;
    }

    allocation_chunks = (i * 100) + Boost;

    /* Normalize Altboost and allocations chunck down to prevent overflow */
    while (Boost > 1000) {
      Boost /= 2;
      allocation_chunks /= 2;
    }

    /* Calculate the number of bits to be spent on the arf based on the
     * boost number
     */
    arf_frame_bits =
        (int)((double)Boost * (group_bits / (double)allocation_chunks));

    /* Estimate if there are enough bits available to make worthwhile use
     * of an arf.
     */
    tmp_q = estimate_q(cpi, mod_frame_err, (int)arf_frame_bits);

    /* Only use an arf if it is likely we will be able to code
     * it at a lower Q than the surrounding frames.
     */
    if (tmp_q < cpi->worst_quality) {
      int half_gf_int;
      int frames_after_arf;
      int frames_bwd = cpi->oxcf.arnr_max_frames - 1;
      int frames_fwd = cpi->oxcf.arnr_max_frames - 1;

      cpi->source_alt_ref_pending = 1;

      /*
       * For alt ref frames the error score for the end frame of the
       * group (the alt ref frame) should not contribute to the group
       * total and hence the number of bit allocated to the group.
       * Rather it forms part of the next group (it is the GF at the
       * start of the next group)
       * gf_group_err -= mod_frame_err;
       *
       * For alt ref frames alt ref frame is technically part of the
       * GF frame for the next group but we always base the error
       * calculation and bit allocation on the current group of frames.
       *
       * Set the interval till the next gf or arf.
       * For ARFs this is the number of frames to be coded before the
       * future frame that is coded as an ARF.
       * The future frame itself is part of the next group
       */
      cpi->baseline_gf_interval = i;

      /*
       * Define the arnr filter width for this group of frames:
       * We only filter frames that lie within a distance of half
       * the GF interval from the ARF frame. We also have to trap
       * cases where the filter extends beyond the end of clip.
       * Note: this_frame->frame has been updated in the loop
       * so it now points at the ARF frame.
       */
      half_gf_int = cpi->baseline_gf_interval >> 1;
      frames_after_arf =
          (int)(cpi->twopass.total_stats.count - this_frame->frame - 1);

      switch (cpi->oxcf.arnr_type) {
        case 1: /* Backward filter */
          frames_fwd = 0;
          if (frames_bwd > half_gf_int) frames_bwd = half_gf_int;
          break;

        case 2: /* Forward filter */
          if (frames_fwd > half_gf_int) frames_fwd = half_gf_int;
          if (frames_fwd > frames_after_arf) frames_fwd = frames_after_arf;
          frames_bwd = 0;
          break;

        case 3: /* Centered filter */
        default:
          frames_fwd >>= 1;
          if (frames_fwd > frames_after_arf) frames_fwd = frames_after_arf;
          if (frames_fwd > half_gf_int) frames_fwd = half_gf_int;

          frames_bwd = frames_fwd;

          /* For even length filter there is one more frame backward
           * than forward: e.g. len=6 ==> bbbAff, len=7 ==> bbbAfff.
           */
          if (frames_bwd < half_gf_int) {
            frames_bwd += (cpi->oxcf.arnr_max_frames + 1) & 0x1;
          }
          break;
      }

      cpi->active_arnr_frames = frames_bwd + 1 + frames_fwd;
    } else {
      cpi->source_alt_ref_pending = 0;
      cpi->baseline_gf_interval = i;
    }
  } else {
    cpi->source_alt_ref_pending = 0;
    cpi->baseline_gf_interval = i;
  }

  /*
   * Now decide how many bits should be allocated to the GF group as  a
   * proportion of those remaining in the kf group.
   * The final key frame group in the clip is treated as a special case
   * where cpi->twopass.kf_group_bits is tied to cpi->twopass.bits_left.
   * This is also important for short clips where there may only be one
   * key frame.
   */
  if (cpi->twopass.frames_to_key >=
      (int)(cpi->twopass.total_stats.count - cpi->common.current_video_frame)) {
    cpi->twopass.kf_group_bits =
        (cpi->twopass.bits_left > 0) ? cpi->twopass.bits_left : 0;
  }

  /* Calculate the bits to be allocated to the group as a whole */
  if ((cpi->twopass.kf_group_bits > 0) &&
      (cpi->twopass.kf_group_error_left > 0)) {
    cpi->twopass.gf_group_bits =
        (int64_t)(cpi->twopass.kf_group_bits *
                  (gf_group_err / cpi->twopass.kf_group_error_left));
  } else {
    cpi->twopass.gf_group_bits = 0;
  }

  cpi->twopass.gf_group_bits =
      (cpi->twopass.gf_group_bits < 0) ? 0
      : (cpi->twopass.gf_group_bits > cpi->twopass.kf_group_bits)
          ? cpi->twopass.kf_group_bits
          : cpi->twopass.gf_group_bits;

  /* Clip cpi->twopass.gf_group_bits based on user supplied data rate
   * variability limit (cpi->oxcf.two_pass_vbrmax_section)
   */
  if (cpi->twopass.gf_group_bits >
      (int64_t)max_bits * cpi->baseline_gf_interval) {
    cpi->twopass.gf_group_bits = (int64_t)max_bits * cpi->baseline_gf_interval;
  }

  /* Reset the file position */
  reset_fpf_position(cpi, start_pos);

  /* Update the record of error used so far (only done once per gf group) */
  cpi->twopass.modified_error_used += gf_group_err;

  /* Assign  bits to the arf or gf. */
  for (i = 0; i <= (cpi->source_alt_ref_pending &&
                    cpi->common.frame_type != KEY_FRAME);
       i++) {
    int Boost;
    int allocation_chunks;
    int Q =
        (cpi->oxcf.fixed_q < 0) ? cpi->last_q[INTER_FRAME] : cpi->oxcf.fixed_q;
    int gf_bits;

    /* For ARF frames */
    if (cpi->source_alt_ref_pending && i == 0) {
#if NEW_BOOST
      Boost = (alt_boost * GFQ_ADJUSTMENT) / 100;
#else
      Boost = (cpi->gfu_boost * 3 * GFQ_ADJUSTMENT) / (2 * 100);
#endif
      Boost += (cpi->baseline_gf_interval * 50);

      /* Set max and minimum boost and hence minimum allocation */
      if (Boost > ((cpi->baseline_gf_interval + 1) * 200)) {
        Boost = ((cpi->baseline_gf_interval + 1) * 200);
      } else if (Boost < 125) {
        Boost = 125;
      }

      allocation_chunks = ((cpi->baseline_gf_interval + 1) * 100) + Boost;
    }
    /* Else for standard golden frames */
    else {
      /* boost based on inter / intra ratio of subsequent frames */
      Boost = (cpi->gfu_boost * GFQ_ADJUSTMENT) / 100;

      /* Set max and minimum boost and hence minimum allocation */
      if (Boost > (cpi->baseline_gf_interval * 150)) {
        Boost = (cpi->baseline_gf_interval * 150);
      } else if (Boost < 125) {
        Boost = 125;
      }

      allocation_chunks = (cpi->baseline_gf_interval * 100) + (Boost - 100);
    }

    /* Normalize Altboost and allocations chunck down to prevent overflow */
    while (Boost > 1000) {
      Boost /= 2;
      allocation_chunks /= 2;
    }

    /* Calculate the number of bits to be spent on the gf or arf based on
     * the boost number
     */
    gf_bits = saturate_cast_double_to_int(
        (double)Boost *
        (cpi->twopass.gf_group_bits / (double)allocation_chunks));

    /* If the frame that is to be boosted is simpler than the average for
     * the gf/arf group then use an alternative calculation
     * based on the error score of the frame itself
     */
    if (mod_frame_err < gf_group_err / (double)cpi->baseline_gf_interval) {
      double alt_gf_grp_bits;
      int alt_gf_bits;

      alt_gf_grp_bits =
          (double)cpi->twopass.kf_group_bits *
          (mod_frame_err * (double)cpi->baseline_gf_interval) /
          DOUBLE_DIVIDE_CHECK((double)cpi->twopass.kf_group_error_left);

      alt_gf_bits =
          (int)((double)Boost * (alt_gf_grp_bits / (double)allocation_chunks));

      if (gf_bits > alt_gf_bits) {
        gf_bits = alt_gf_bits;
      }
    }
    /* Else if it is harder than other frames in the group make sure it at
     * least receives an allocation in keeping with its relative error
     * score, otherwise it may be worse off than an "un-boosted" frame
     */
    else {
      // Avoid division by 0 by clamping cpi->twopass.kf_group_error_left to 1
      int alt_gf_bits = saturate_cast_double_to_int(
          (double)cpi->twopass.kf_group_bits * mod_frame_err /
          (double)VPXMAX(cpi->twopass.kf_group_error_left, 1));

      if (alt_gf_bits > gf_bits) {
        gf_bits = alt_gf_bits;
      }
    }

    /* Apply an additional limit for CBR */
    if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
      if (cpi->twopass.gf_bits > (int)(cpi->buffer_level >> 1)) {
        cpi->twopass.gf_bits = (int)(cpi->buffer_level >> 1);
      }
    }

    /* Don't allow a negative value for gf_bits */
    if (gf_bits < 0) gf_bits = 0;

    /* Add in minimum for a frame */
    gf_bits += cpi->min_frame_bandwidth;

    if (i == 0) {
      cpi->twopass.gf_bits = gf_bits;
    }
    if (i == 1 || (!cpi->source_alt_ref_pending &&
                   (cpi->common.frame_type != KEY_FRAME))) {
      /* Per frame bit target for this frame */
      cpi->per_frame_bandwidth = gf_bits;
    }
  }

  {
    /* Adjust KF group bits and error remainin */
    cpi->twopass.kf_group_error_left -= (int64_t)gf_group_err;
    cpi->twopass.kf_group_bits -= cpi->twopass.gf_group_bits;

    if (cpi->twopass.kf_group_bits < 0) cpi->twopass.kf_group_bits = 0;

    /* Note the error score left in the remaining frames of the group.
     * For normal GFs we want to remove the error score for the first
     * frame of the group (except in Key frame case where this has
     * already happened)
     */
    if (!cpi->source_alt_ref_pending && cpi->common.frame_type != KEY_FRAME) {
      cpi->twopass.gf_group_error_left =
          (int)(gf_group_err - gf_first_frame_err);
    } else {
      cpi->twopass.gf_group_error_left = (int)gf_group_err;
    }

    cpi->twopass.gf_group_bits -=
        cpi->twopass.gf_bits - cpi->min_frame_bandwidth;

    if (cpi->twopass.gf_group_bits < 0) cpi->twopass.gf_group_bits = 0;

    /* This condition could fail if there are two kfs very close together
     * despite (MIN_GF_INTERVAL) and would cause a divide by 0 in the
     * calculation of cpi->twopass.alt_extra_bits.
     */
    if (cpi->baseline_gf_interval >= 3) {
#if NEW_BOOST
      int boost = (cpi->source_alt_ref_pending) ? b_boost : cpi->gfu_boost;
#else
      int boost = cpi->gfu_boost;
#endif
      if (boost >= 150) {
        int pct_extra;

        pct_extra = (boost - 100) / 50;
        pct_extra = (pct_extra > 20) ? 20 : pct_extra;

        cpi->twopass.alt_extra_bits =
            (int)(cpi->twopass.gf_group_bits * pct_extra) / 100;
        cpi->twopass.gf_group_bits -= cpi->twopass.alt_extra_bits;
        cpi->twopass.alt_extra_bits /= ((cpi->baseline_gf_interval - 1) >> 1);
      } else {
        cpi->twopass.alt_extra_bits = 0;
      }
    } else {
      cpi->twopass.alt_extra_bits = 0;
    }
  }

  /* Adjustments based on a measure of complexity of the section */
  if (cpi->common.frame_type != KEY_FRAME) {
    FIRSTPASS_STATS sectionstats;
    double Ratio;

    zero_stats(&sectionstats);
    reset_fpf_position(cpi, start_pos);

    for (i = 0; i < cpi->baseline_gf_interval; ++i) {
      input_stats(cpi, &next_frame);
      accumulate_stats(&sectionstats, &next_frame);
    }

    avg_stats(&sectionstats);

    cpi->twopass.section_intra_rating =
        (unsigned int)(sectionstats.intra_error /
                       DOUBLE_DIVIDE_CHECK(sectionstats.coded_error));

    Ratio = sectionstats.intra_error /
            DOUBLE_DIVIDE_CHECK(sectionstats.coded_error);
    cpi->twopass.section_max_qfactor = 1.0 - ((Ratio - 10.0) * 0.025);

    if (cpi->twopass.section_max_qfactor < 0.80) {
      cpi->twopass.section_max_qfactor = 0.80;
    }

    reset_fpf_position(cpi, start_pos);
  }
}

/* Allocate bits to a normal frame that is neither a gf an arf or a key frame.
 */
static void assign_std_frame_bits(VP8_COMP *cpi, FIRSTPASS_STATS *this_frame) {
  int target_frame_size;

  double modified_err;
  double err_fraction;

  int max_bits = frame_max_bits(cpi); /* Max for a single frame */

  /* Calculate modified prediction error used in bit allocation */
  modified_err = calculate_modified_err(cpi, this_frame);

  /* What portion of the remaining GF group error is used by this frame */
  if (cpi->twopass.gf_group_error_left > 0) {
    err_fraction = modified_err / cpi->twopass.gf_group_error_left;
  } else {
    err_fraction = 0.0;
  }

  /* How many of those bits available for allocation should we give it? */
  target_frame_size = saturate_cast_double_to_int(
      (double)cpi->twopass.gf_group_bits * err_fraction);

  /* Clip to target size to 0 - max_bits (or cpi->twopass.gf_group_bits)
   * at the top end.
   */
  if (target_frame_size < 0) {
    target_frame_size = 0;
  } else {
    if (target_frame_size > max_bits) target_frame_size = max_bits;

    if (target_frame_size > cpi->twopass.gf_group_bits) {
      target_frame_size = (int)cpi->twopass.gf_group_bits;
    }
  }

  /* Adjust error and bits remaining */
  cpi->twopass.gf_group_error_left -= (int)modified_err;
  cpi->twopass.gf_group_bits -= target_frame_size;

  if (cpi->twopass.gf_group_bits < 0) cpi->twopass.gf_group_bits = 0;

  /* Add in the minimum number of bits that is set aside for every frame. */
  target_frame_size += cpi->min_frame_bandwidth;

  /* Every other frame gets a few extra bits */
  if ((cpi->frames_since_golden & 0x01) &&
      (cpi->frames_till_gf_update_due > 0)) {
    target_frame_size += cpi->twopass.alt_extra_bits;
  }

  /* Per frame bit target for this frame */
  cpi->per_frame_bandwidth = target_frame_size;
}

void vp8_second_pass(VP8_COMP *cpi) {
  int tmp_q;
  int frames_left =
      (int)(cpi->twopass.total_stats.count - cpi->common.current_video_frame);

  FIRSTPASS_STATS this_frame;
  FIRSTPASS_STATS this_frame_copy;

  double this_frame_intra_error;
  double this_frame_coded_error;

  int overhead_bits;

  vp8_zero(this_frame);

  if (!cpi->twopass.stats_in) {
    return;
  }

  vpx_clear_system_state();

  if (EOF == input_stats(cpi, &this_frame)) return;

  this_frame_intra_error = this_frame.intra_error;
  this_frame_coded_error = this_frame.coded_error;

  /* keyframe and section processing ! */
  if (cpi->twopass.frames_to_key == 0) {
    /* Define next KF group and assign bits to it */
    this_frame_copy = this_frame;
    find_next_key_frame(cpi, &this_frame_copy);

    /* Special case: Error error_resilient_mode mode does not make much
     * sense for two pass but with its current meaning this code is
     * designed to stop outlandish behaviour if someone does set it when
     * using two pass. It effectively disables GF groups. This is
     * temporary code until we decide what should really happen in this
     * case.
     */
    if (cpi->oxcf.error_resilient_mode) {
      cpi->twopass.gf_group_bits = cpi->twopass.kf_group_bits;
      cpi->twopass.gf_group_error_left = (int)cpi->twopass.kf_group_error_left;
      cpi->baseline_gf_interval = cpi->twopass.frames_to_key;
      cpi->frames_till_gf_update_due = cpi->baseline_gf_interval;
      cpi->source_alt_ref_pending = 0;
    }
  }

  /* Is this a GF / ARF (Note that a KF is always also a GF) */
  if (cpi->frames_till_gf_update_due == 0) {
    /* Define next gf group and assign bits to it */
    this_frame_copy = this_frame;
    define_gf_group(cpi, &this_frame_copy);

    /* If we are going to code an altref frame at the end of the group
     * and the current frame is not a key frame.... If the previous
     * group used an arf this frame has already benefited from that arf
     * boost and it should not be given extra bits If the previous
     * group was NOT coded using arf we may want to apply some boost to
     * this GF as well
     */
    if (cpi->source_alt_ref_pending && (cpi->common.frame_type != KEY_FRAME)) {
      /* Assign a standard frames worth of bits from those allocated
       * to the GF group
       */
      int bak = cpi->per_frame_bandwidth;
      this_frame_copy = this_frame;
      assign_std_frame_bits(cpi, &this_frame_copy);
      cpi->per_frame_bandwidth = bak;
    }
  }

  /* Otherwise this is an ordinary frame */
  else {
    /* Special case: Error error_resilient_mode mode does not make much
     * sense for two pass but with its current meaning but this code is
     * designed to stop outlandish behaviour if someone does set it
     * when using two pass. It effectively disables GF groups. This is
     * temporary code till we decide what should really happen in this
     * case.
     */
    if (cpi->oxcf.error_resilient_mode) {
      cpi->frames_till_gf_update_due = cpi->twopass.frames_to_key;

      if (cpi->common.frame_type != KEY_FRAME) {
        /* Assign bits from those allocated to the GF group */
        this_frame_copy = this_frame;
        assign_std_frame_bits(cpi, &this_frame_copy);
      }
    } else {
      /* Assign bits from those allocated to the GF group */
      this_frame_copy = this_frame;
      assign_std_frame_bits(cpi, &this_frame_copy);
    }
  }

  /* Keep a globally available copy of this and the next frame's iiratio. */
  cpi->twopass.this_iiratio =
      (unsigned int)(this_frame_intra_error /
                     DOUBLE_DIVIDE_CHECK(this_frame_coded_error));
  {
    FIRSTPASS_STATS next_frame;
    if (lookup_next_frame_stats(cpi, &next_frame) != EOF) {
      cpi->twopass.next_iiratio =
          (unsigned int)(next_frame.intra_error /
                         DOUBLE_DIVIDE_CHECK(next_frame.coded_error));
    }
  }

  /* Set nominal per second bandwidth for this frame */
  cpi->target_bandwidth =
      (int)(cpi->per_frame_bandwidth * cpi->output_framerate);
  if (cpi->target_bandwidth < 0) cpi->target_bandwidth = 0;

  /* Account for mv, mode and other overheads. */
  overhead_bits = (int)estimate_modemvcost(cpi, &cpi->twopass.total_left_stats);

  /* Special case code for first frame. */
  if (cpi->common.current_video_frame == 0) {
    cpi->twopass.est_max_qcorrection_factor = 1.0;

    int64_t section_target_bandwidth = cpi->twopass.bits_left / frames_left;
    section_target_bandwidth = VPXMIN(section_target_bandwidth, INT_MAX);

    /* Set a cq_level in constrained quality mode. */
    if (cpi->oxcf.end_usage == USAGE_CONSTRAINED_QUALITY) {
      int est_cq;

      est_cq = estimate_cq(cpi, &cpi->twopass.total_left_stats,
                           (int)section_target_bandwidth, overhead_bits);

      cpi->cq_target_quality = cpi->oxcf.cq_level;
      if (est_cq > cpi->cq_target_quality) cpi->cq_target_quality = est_cq;
    }

    /* guess at maxq needed in 2nd pass */
    cpi->twopass.maxq_max_limit = cpi->worst_quality;
    cpi->twopass.maxq_min_limit = cpi->best_quality;

    tmp_q = estimate_max_q(cpi, &cpi->twopass.total_left_stats,
                           (int)section_target_bandwidth, overhead_bits);

    /* Limit the maxq value returned subsequently.
     * This increases the risk of overspend or underspend if the initial
     * estimate for the clip is bad, but helps prevent excessive
     * variation in Q, especially near the end of a clip
     * where for example a small overspend may cause Q to crash
     */
    cpi->twopass.maxq_max_limit =
        ((tmp_q + 32) < cpi->worst_quality) ? (tmp_q + 32) : cpi->worst_quality;
    cpi->twopass.maxq_min_limit =
        ((tmp_q - 32) > cpi->best_quality) ? (tmp_q - 32) : cpi->best_quality;

    cpi->active_worst_quality = tmp_q;
    cpi->ni_av_qi = tmp_q;
  }

  /* The last few frames of a clip almost always have to few or too many
   * bits and for the sake of over exact rate control we don't want to make
   * radical adjustments to the allowed quantizer range just to use up a
   * few surplus bits or get beneath the target rate.
   */
  else if ((cpi->common.current_video_frame <
            (((unsigned int)cpi->twopass.total_stats.count * 255) >> 8)) &&
           ((cpi->common.current_video_frame + cpi->baseline_gf_interval) <
            (unsigned int)cpi->twopass.total_stats.count)) {
    if (frames_left < 1) frames_left = 1;

    int64_t section_target_bandwidth = cpi->twopass.bits_left / frames_left;
    section_target_bandwidth = VPXMIN(section_target_bandwidth, INT_MAX);

    tmp_q = estimate_max_q(cpi, &cpi->twopass.total_left_stats,
                           (int)section_target_bandwidth, overhead_bits);

    /* Move active_worst_quality but in a damped way */
    if (tmp_q > cpi->active_worst_quality) {
      cpi->active_worst_quality++;
    } else if (tmp_q < cpi->active_worst_quality) {
      cpi->active_worst_quality--;
    }

    cpi->active_worst_quality =
        ((cpi->active_worst_quality * 3) + tmp_q + 2) / 4;
  }

  cpi->twopass.frames_to_key--;

  /* Update the total stats remaining sturcture */
  subtract_stats(&cpi->twopass.total_left_stats, &this_frame);
}

static int test_candidate_kf(VP8_COMP *cpi, FIRSTPASS_STATS *last_frame,
                             FIRSTPASS_STATS *this_frame,
                             FIRSTPASS_STATS *next_frame) {
  int is_viable_kf = 0;

  /* Does the frame satisfy the primary criteria of a key frame
   *      If so, then examine how well it predicts subsequent frames
   */
  if ((this_frame->pcnt_second_ref < 0.10) &&
      (next_frame->pcnt_second_ref < 0.10) &&
      ((this_frame->pcnt_inter < 0.05) ||
       (((this_frame->pcnt_inter - this_frame->pcnt_neutral) < .25) &&
        ((this_frame->intra_error /
          DOUBLE_DIVIDE_CHECK(this_frame->coded_error)) < 2.5) &&
        ((fabs(last_frame->coded_error - this_frame->coded_error) /
              DOUBLE_DIVIDE_CHECK(this_frame->coded_error) >
          .40) ||
         (fabs(last_frame->intra_error - this_frame->intra_error) /
              DOUBLE_DIVIDE_CHECK(this_frame->intra_error) >
          .40) ||
         ((next_frame->intra_error /
           DOUBLE_DIVIDE_CHECK(next_frame->coded_error)) > 3.5))))) {
    int i;
    FIRSTPASS_STATS *start_pos;

    FIRSTPASS_STATS local_next_frame;

    double boost_score = 0.0;
    double old_boost_score = 0.0;
    double decay_accumulator = 1.0;
    double next_iiratio;

    local_next_frame = *next_frame;

    /* Note the starting file position so we can reset to it */
    start_pos = cpi->twopass.stats_in;

    /* Examine how well the key frame predicts subsequent frames */
    for (i = 0; i < 16; ++i) {
      next_iiratio = (IIKFACTOR1 * local_next_frame.intra_error /
                      DOUBLE_DIVIDE_CHECK(local_next_frame.coded_error));

      if (next_iiratio > RMAX) next_iiratio = RMAX;

      /* Cumulative effect of decay in prediction quality */
      if (local_next_frame.pcnt_inter > 0.85) {
        decay_accumulator = decay_accumulator * local_next_frame.pcnt_inter;
      } else {
        decay_accumulator =
            decay_accumulator * ((0.85 + local_next_frame.pcnt_inter) / 2.0);
      }

      /* Keep a running total */
      boost_score += (decay_accumulator * next_iiratio);

      /* Test various breakout clauses */
      if ((local_next_frame.pcnt_inter < 0.05) || (next_iiratio < 1.5) ||
          (((local_next_frame.pcnt_inter - local_next_frame.pcnt_neutral) <
            0.20) &&
           (next_iiratio < 3.0)) ||
          ((boost_score - old_boost_score) < 0.5) ||
          (local_next_frame.intra_error < 200)) {
        break;
      }

      old_boost_score = boost_score;

      /* Get the next frame details */
      if (EOF == input_stats(cpi, &local_next_frame)) break;
    }

    /* If there is tolerable prediction for at least the next 3 frames
     * then break out else discard this pottential key frame and move on
     */
    if (boost_score > 5.0 && (i > 3)) {
      is_viable_kf = 1;
    } else {
      /* Reset the file position */
      reset_fpf_position(cpi, start_pos);

      is_viable_kf = 0;
    }
  }

  return is_viable_kf;
}
static void find_next_key_frame(VP8_COMP *cpi, FIRSTPASS_STATS *this_frame) {
  int i, j;
  FIRSTPASS_STATS last_frame;
  FIRSTPASS_STATS first_frame;
  FIRSTPASS_STATS next_frame;
  FIRSTPASS_STATS *start_position;

  double decay_accumulator = 1.0;
  double boost_score = 0;
  double old_boost_score = 0.0;
  double loop_decay_rate;

  double kf_mod_err = 0.0;
  double kf_group_err = 0.0;
  double kf_group_intra_err = 0.0;
  double kf_group_coded_err = 0.0;
  double recent_loop_decay[8] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

  memset(&next_frame, 0, sizeof(next_frame));

  vpx_clear_system_state();
  start_position = cpi->twopass.stats_in;

  cpi->common.frame_type = KEY_FRAME;

  /* is this a forced key frame by interval */
  cpi->this_key_frame_forced = cpi->next_key_frame_forced;

  /* Clear the alt ref active flag as this can never be active on a key
   * frame
   */
  cpi->source_alt_ref_active = 0;

  /* Kf is always a gf so clear frames till next gf counter */
  cpi->frames_till_gf_update_due = 0;

  cpi->twopass.frames_to_key = 1;

  /* Take a copy of the initial frame details */
  first_frame = *this_frame;

  cpi->twopass.kf_group_bits = 0;
  cpi->twopass.kf_group_error_left = 0;

  kf_mod_err = calculate_modified_err(cpi, this_frame);

  /* find the next keyframe */
  i = 0;
  while (cpi->twopass.stats_in < cpi->twopass.stats_in_end) {
    /* Accumulate kf group error */
    kf_group_err += calculate_modified_err(cpi, this_frame);

    /* These figures keep intra and coded error counts for all frames
     * including key frames in the group. The effect of the key frame
     * itself can be subtracted out using the first_frame data
     * collected above
     */
    kf_group_intra_err += this_frame->intra_error;
    kf_group_coded_err += this_frame->coded_error;

    /* Load the next frame's stats. */
    last_frame = *this_frame;
    input_stats(cpi, this_frame);

    /* Provided that we are not at the end of the file... */
    if (cpi->oxcf.auto_key &&
        lookup_next_frame_stats(cpi, &next_frame) != EOF) {
      /* Normal scene cut check */
      if ((i >= MIN_GF_INTERVAL) &&
          test_candidate_kf(cpi, &last_frame, this_frame, &next_frame)) {
        break;
      }

      /* How fast is prediction quality decaying */
      loop_decay_rate = get_prediction_decay_rate(&next_frame);

      /* We want to know something about the recent past... rather than
       * as used elsewhere where we are concened with decay in prediction
       * quality since the last GF or KF.
       */
      recent_loop_decay[i % 8] = loop_decay_rate;
      decay_accumulator = 1.0;
      for (j = 0; j < 8; ++j) {
        decay_accumulator = decay_accumulator * recent_loop_decay[j];
      }

      /* Special check for transition or high motion followed by a
       * static scene.
       */
      if (detect_transition_to_still(cpi, i,
                                     ((int)(cpi->key_frame_frequency) - (int)i),
                                     loop_decay_rate, decay_accumulator)) {
        break;
      }

      /* Step on to the next frame */
      cpi->twopass.frames_to_key++;

      /* If we don't have a real key frame within the next two
       * forcekeyframeevery intervals then break out of the loop.
       */
      if (cpi->twopass.frames_to_key >= 2 * (int)cpi->key_frame_frequency) {
        break;
      }
    } else {
      cpi->twopass.frames_to_key++;
    }

    i++;
  }

  /* If there is a max kf interval set by the user we must obey it.
   * We already breakout of the loop above at 2x max.
   * This code centers the extra kf if the actual natural
   * interval is between 1x and 2x
   */
  if (cpi->oxcf.auto_key &&
      cpi->twopass.frames_to_key > (int)cpi->key_frame_frequency) {
    FIRSTPASS_STATS *current_pos = cpi->twopass.stats_in;
    FIRSTPASS_STATS tmp_frame;

    cpi->twopass.frames_to_key /= 2;

    /* Copy first frame details */
    tmp_frame = first_frame;

    /* Reset to the start of the group */
    reset_fpf_position(cpi, start_position);

    kf_group_err = 0;
    kf_group_intra_err = 0;
    kf_group_coded_err = 0;

    /* Rescan to get the correct error data for the forced kf group */
    for (i = 0; i < cpi->twopass.frames_to_key; ++i) {
      /* Accumulate kf group errors */
      kf_group_err += calculate_modified_err(cpi, &tmp_frame);
      kf_group_intra_err += tmp_frame.intra_error;
      kf_group_coded_err += tmp_frame.coded_error;

      /* Load a the next frame's stats */
      input_stats(cpi, &tmp_frame);
    }

    /* Reset to the start of the group */
    reset_fpf_position(cpi, current_pos);

    cpi->next_key_frame_forced = 1;
  } else {
    cpi->next_key_frame_forced = 0;
  }

  /* Special case for the last frame of the file */
  if (cpi->twopass.stats_in >= cpi->twopass.stats_in_end) {
    /* Accumulate kf group error */
    kf_group_err += calculate_modified_err(cpi, this_frame);

    /* These figures keep intra and coded error counts for all frames
     * including key frames in the group. The effect of the key frame
     * itself can be subtracted out using the first_frame data
     * collected above
     */
    kf_group_intra_err += this_frame->intra_error;
    kf_group_coded_err += this_frame->coded_error;
  }

  /* Calculate the number of bits that should be assigned to the kf group. */
  if ((cpi->twopass.bits_left > 0) &&
      (cpi->twopass.modified_error_left > 0.0)) {
    /* Max for a single normal frame (not key frame) */
    int max_bits = frame_max_bits(cpi);

    /* Maximum bits for the kf group */
    int64_t max_grp_bits;

    /* Default allocation based on bits left and relative
     * complexity of the section
     */
    cpi->twopass.kf_group_bits =
        (int64_t)(cpi->twopass.bits_left *
                  (kf_group_err / cpi->twopass.modified_error_left));

    /* Clip based on maximum per frame rate defined by the user. */
    max_grp_bits = (int64_t)max_bits * (int64_t)cpi->twopass.frames_to_key;
    if (cpi->twopass.kf_group_bits > max_grp_bits) {
      cpi->twopass.kf_group_bits = max_grp_bits;
    }

    /* Additional special case for CBR if buffer is getting full. */
    if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
      int64_t opt_buffer_lvl = cpi->oxcf.optimal_buffer_level;
      int64_t buffer_lvl = cpi->buffer_level;

      /* If the buffer is near or above the optimal and this kf group is
       * not being allocated much then increase the allocation a bit.
       */
      if (buffer_lvl >= opt_buffer_lvl) {
        int64_t high_water_mark =
            (opt_buffer_lvl + cpi->oxcf.maximum_buffer_size) >> 1;

        int64_t av_group_bits;

        /* Av bits per frame * number of frames */
        av_group_bits = (int64_t)cpi->av_per_frame_bandwidth *
                        (int64_t)cpi->twopass.frames_to_key;

        /* We are at or above the maximum. */
        if (cpi->buffer_level >= high_water_mark) {
          int64_t min_group_bits;

          min_group_bits =
              av_group_bits + (int64_t)(buffer_lvl - high_water_mark);

          if (cpi->twopass.kf_group_bits < min_group_bits) {
            cpi->twopass.kf_group_bits = min_group_bits;
          }
        }
        /* We are above optimal but below the maximum */
        else if (cpi->twopass.kf_group_bits < av_group_bits) {
          int64_t bits_below_av = av_group_bits - cpi->twopass.kf_group_bits;

          cpi->twopass.kf_group_bits +=
              (int64_t)((double)bits_below_av *
                        (double)(buffer_lvl - opt_buffer_lvl) /
                        (double)(high_water_mark - opt_buffer_lvl));
        }
      }
    }
  } else {
    cpi->twopass.kf_group_bits = 0;
  }

  /* Reset the first pass file position */
  reset_fpf_position(cpi, start_position);

  /* determine how big to make this keyframe based on how well the
   * subsequent frames use inter blocks
   */
  decay_accumulator = 1.0;
  boost_score = 0.0;

  for (i = 0; i < cpi->twopass.frames_to_key; ++i) {
    double r;

    if (EOF == input_stats(cpi, &next_frame)) break;

    if (next_frame.intra_error > cpi->twopass.kf_intra_err_min) {
      r = (IIKFACTOR2 * next_frame.intra_error /
           DOUBLE_DIVIDE_CHECK(next_frame.coded_error));
    } else {
      r = (IIKFACTOR2 * cpi->twopass.kf_intra_err_min /
           DOUBLE_DIVIDE_CHECK(next_frame.coded_error));
    }

    if (r > RMAX) r = RMAX;

    /* How fast is prediction quality decaying */
    loop_decay_rate = get_prediction_decay_rate(&next_frame);

    decay_accumulator = decay_accumulator * loop_decay_rate;
    decay_accumulator = decay_accumulator < 0.1 ? 0.1 : decay_accumulator;

    boost_score += (decay_accumulator * r);

    if ((i > MIN_GF_INTERVAL) && ((boost_score - old_boost_score) < 1.0)) {
      break;
    }

    old_boost_score = boost_score;
  }

  if (1) {
    FIRSTPASS_STATS sectionstats;
    double Ratio;

    zero_stats(&sectionstats);
    reset_fpf_position(cpi, start_position);

    for (i = 0; i < cpi->twopass.frames_to_key; ++i) {
      input_stats(cpi, &next_frame);
      accumulate_stats(&sectionstats, &next_frame);
    }

    avg_stats(&sectionstats);

    cpi->twopass.section_intra_rating =
        (unsigned int)(sectionstats.intra_error /
                       DOUBLE_DIVIDE_CHECK(sectionstats.coded_error));

    Ratio = sectionstats.intra_error /
            DOUBLE_DIVIDE_CHECK(sectionstats.coded_error);
    cpi->twopass.section_max_qfactor = 1.0 - ((Ratio - 10.0) * 0.025);

    if (cpi->twopass.section_max_qfactor < 0.80) {
      cpi->twopass.section_max_qfactor = 0.80;
    }
  }

  /* When using CBR apply additional buffer fullness related upper limits */
  if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
    double max_boost;

    if (cpi->drop_frames_allowed) {
      int df_buffer_level = (int)(cpi->oxcf.drop_frames_water_mark *
                                  (cpi->oxcf.optimal_buffer_level / 100));

      if (cpi->buffer_level > df_buffer_level) {
        max_boost =
            ((double)((cpi->buffer_level - df_buffer_level) * 2 / 3) * 16.0) /
            DOUBLE_DIVIDE_CHECK((double)cpi->av_per_frame_bandwidth);
      } else {
        max_boost = 0.0;
      }
    } else if (cpi->buffer_level > 0) {
      max_boost = ((double)(cpi->buffer_level * 2 / 3) * 16.0) /
                  DOUBLE_DIVIDE_CHECK((double)cpi->av_per_frame_bandwidth);
    } else {
      max_boost = 0.0;
    }

    if (boost_score > max_boost) boost_score = max_boost;
  }

  /* Reset the first pass file position */
  reset_fpf_position(cpi, start_position);

  /* Work out how many bits to allocate for the key frame itself */
  if (1) {
    int kf_boost = (int)boost_score;
    int allocation_chunks;
    int Counter = cpi->twopass.frames_to_key;
    int alt_kf_bits;
    YV12_BUFFER_CONFIG *lst_yv12 = &cpi->common.yv12_fb[cpi->common.lst_fb_idx];
/* Min boost based on kf interval */
#if 0

        while ((kf_boost < 48) && (Counter > 0))
        {
            Counter -= 2;
            kf_boost ++;
        }

#endif

    if (kf_boost < 48) {
      kf_boost += ((Counter + 1) >> 1);

      if (kf_boost > 48) kf_boost = 48;
    }

    /* bigger frame sizes need larger kf boosts, smaller frames smaller
     * boosts...
     */
    if ((lst_yv12->y_width * lst_yv12->y_height) > (320 * 240)) {
      kf_boost += 2 * (lst_yv12->y_width * lst_yv12->y_height) / (320 * 240);
    } else if ((lst_yv12->y_width * lst_yv12->y_height) < (320 * 240)) {
      kf_boost -= 4 * (320 * 240) / (lst_yv12->y_width * lst_yv12->y_height);
    }

    /* Min KF boost */
    kf_boost = (int)((double)kf_boost * 100.0) >> 4; /* Scale 16 to 100 */
    if (kf_boost < 250) kf_boost = 250;

    /*
     * We do three calculations for kf size.
     * The first is based on the error score for the whole kf group.
     * The second (optionaly) on the key frames own error if this is
     * smaller than the average for the group.
     * The final one insures that the frame receives at least the
     * allocation it would have received based on its own error score vs
     * the error score remaining
     * Special case if the sequence appears almost totaly static
     * as measured by the decay accumulator. In this case we want to
     * spend almost all of the bits on the key frame.
     * cpi->twopass.frames_to_key-1 because key frame itself is taken
     * care of by kf_boost.
     */
    if (decay_accumulator >= 0.99) {
      allocation_chunks = ((cpi->twopass.frames_to_key - 1) * 10) + kf_boost;
    } else {
      allocation_chunks = ((cpi->twopass.frames_to_key - 1) * 100) + kf_boost;
    }

    /* Normalize Altboost and allocations chunck down to prevent overflow */
    while (kf_boost > 1000) {
      kf_boost /= 2;
      allocation_chunks /= 2;
    }

    cpi->twopass.kf_group_bits =
        (cpi->twopass.kf_group_bits < 0) ? 0 : cpi->twopass.kf_group_bits;

    /* Calculate the number of bits to be spent on the key frame */
    cpi->twopass.kf_bits =
        (int)((double)kf_boost *
              ((double)cpi->twopass.kf_group_bits / (double)allocation_chunks));

    /* Apply an additional limit for CBR */
    if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
      if (cpi->twopass.kf_bits > (int)((3 * cpi->buffer_level) >> 2)) {
        cpi->twopass.kf_bits = (int)((3 * cpi->buffer_level) >> 2);
      }
    }

    /* If the key frame is actually easier than the average for the
     * kf group (which does sometimes happen... eg a blank intro frame)
     * Then use an alternate calculation based on the kf error score
     * which should give a smaller key frame.
     */
    if (kf_mod_err < kf_group_err / cpi->twopass.frames_to_key) {
      double alt_kf_grp_bits =
          ((double)cpi->twopass.bits_left *
           (kf_mod_err * (double)cpi->twopass.frames_to_key) /
           DOUBLE_DIVIDE_CHECK(cpi->twopass.modified_error_left));

      alt_kf_bits = (int)((double)kf_boost *
                          (alt_kf_grp_bits / (double)allocation_chunks));

      if (cpi->twopass.kf_bits > alt_kf_bits) {
        cpi->twopass.kf_bits = alt_kf_bits;
      }
    }
    /* Else if it is much harder than other frames in the group make sure
     * it at least receives an allocation in keeping with its relative
     * error score
     */
    else {
      alt_kf_bits = (int)((double)cpi->twopass.bits_left *
                          (kf_mod_err / DOUBLE_DIVIDE_CHECK(
                                            cpi->twopass.modified_error_left)));

      if (alt_kf_bits > cpi->twopass.kf_bits) {
        cpi->twopass.kf_bits = alt_kf_bits;
      }
    }

    cpi->twopass.kf_group_bits -= cpi->twopass.kf_bits;
    /* Add in the minimum frame allowance */
    cpi->twopass.kf_bits += cpi->min_frame_bandwidth;

    /* Peer frame bit target for this frame */
    cpi->per_frame_bandwidth = cpi->twopass.kf_bits;

    /* Convert to a per second bitrate */
    cpi->target_bandwidth = (int)(cpi->twopass.kf_bits * cpi->output_framerate);
  }

  /* Note the total error score of the kf group minus the key frame itself */
  cpi->twopass.kf_group_error_left = (int)(kf_group_err - kf_mod_err);

  /* Adjust the count of total modified error left. The count of bits left
   * is adjusted elsewhere based on real coded frame sizes
   */
  cpi->twopass.modified_error_left -= kf_group_err;

  if (cpi->oxcf.allow_spatial_resampling) {
    int resample_trigger = 0;
    int last_kf_resampled = 0;
    int kf_q;
    int scale_val = 0;
    int hr, hs, vr, vs;
    int new_width = cpi->oxcf.Width;
    int new_height = cpi->oxcf.Height;

    int projected_buffer_level;
    int tmp_q;

    double projected_bits_perframe;
    double group_iiratio = (kf_group_intra_err - first_frame.intra_error) /
                           (kf_group_coded_err - first_frame.coded_error);
    double err_per_frame = kf_group_err / cpi->twopass.frames_to_key;
    double bits_per_frame;
    double av_bits_per_frame;
    double effective_size_ratio;

    if ((cpi->common.Width != cpi->oxcf.Width) ||
        (cpi->common.Height != cpi->oxcf.Height)) {
      last_kf_resampled = 1;
    }

    /* Set back to unscaled by defaults */
    cpi->common.horiz_scale = VP8E_NORMAL;
    cpi->common.vert_scale = VP8E_NORMAL;

    /* Calculate Average bits per frame. */
    av_bits_per_frame =
        cpi->oxcf.target_bandwidth / DOUBLE_DIVIDE_CHECK(cpi->framerate);

    /* CBR... Use the clip average as the target for deciding resample */
    if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
      bits_per_frame = av_bits_per_frame;
    }

    /* In VBR we want to avoid downsampling in easy section unless we
     * are under extreme pressure So use the larger of target bitrate
     * for this section or average bitrate for sequence
     */
    else {
      /* This accounts for how hard the section is... */
      bits_per_frame =
          (double)(cpi->twopass.kf_group_bits / cpi->twopass.frames_to_key);

      /* Don't turn to resampling in easy sections just because they
       * have been assigned a small number of bits
       */
      if (bits_per_frame < av_bits_per_frame) {
        bits_per_frame = av_bits_per_frame;
      }
    }

    /* bits_per_frame should comply with our minimum */
    if (bits_per_frame < (cpi->oxcf.target_bandwidth *
                          cpi->oxcf.two_pass_vbrmin_section / 100)) {
      bits_per_frame = (cpi->oxcf.target_bandwidth *
                        cpi->oxcf.two_pass_vbrmin_section / 100);
    }

    /* Work out if spatial resampling is necessary */
    kf_q = estimate_kf_group_q(cpi, err_per_frame, (int)bits_per_frame,
                               group_iiratio);

    /* If we project a required Q higher than the maximum allowed Q then
     * make a guess at the actual size of frames in this section
     */
    projected_bits_perframe = bits_per_frame;
    tmp_q = kf_q;

    while (tmp_q > cpi->worst_quality) {
      projected_bits_perframe *= 1.04;
      tmp_q--;
    }

    /* Guess at buffer level at the end of the section */
    projected_buffer_level =
        (int)(cpi->buffer_level -
              (int)((projected_bits_perframe - av_bits_per_frame) *
                    cpi->twopass.frames_to_key));

    /* The trigger for spatial resampling depends on the various
     * parameters such as whether we are streaming (CBR) or VBR.
     */
    if (cpi->oxcf.end_usage == USAGE_STREAM_FROM_SERVER) {
      /* Trigger resample if we are projected to fall below down
       * sample level or resampled last time and are projected to
       * remain below the up sample level
       */
      if ((projected_buffer_level < (cpi->oxcf.resample_down_water_mark *
                                     cpi->oxcf.optimal_buffer_level / 100)) ||
          (last_kf_resampled &&
           (projected_buffer_level < (cpi->oxcf.resample_up_water_mark *
                                      cpi->oxcf.optimal_buffer_level / 100)))) {
        resample_trigger = 1;
      } else {
        resample_trigger = 0;
      }
    } else {
      int64_t clip_bits = (int64_t)(cpi->twopass.total_stats.count *
                                    cpi->oxcf.target_bandwidth /
                                    DOUBLE_DIVIDE_CHECK(cpi->framerate));
      int64_t over_spend = cpi->oxcf.starting_buffer_level - cpi->buffer_level;

      /* If triggered last time the threshold for triggering again is
       * reduced:
       *
       * Projected Q higher than allowed and Overspend > 5% of total
       * bits
       */
      if ((last_kf_resampled && (kf_q > cpi->worst_quality)) ||
          ((kf_q > cpi->worst_quality) && (over_spend > clip_bits / 20))) {
        resample_trigger = 1;
      } else {
        resample_trigger = 0;
      }
    }

    if (resample_trigger) {
      while ((kf_q >= cpi->worst_quality) && (scale_val < 6)) {
        scale_val++;

        cpi->common.vert_scale = vscale_lookup[scale_val];
        cpi->common.horiz_scale = hscale_lookup[scale_val];

        Scale2Ratio(cpi->common.horiz_scale, &hr, &hs);
        Scale2Ratio(cpi->common.vert_scale, &vr, &vs);

        new_width = ((hs - 1) + (cpi->oxcf.Width * hr)) / hs;
        new_height = ((vs - 1) + (cpi->oxcf.Height * vr)) / vs;

        /* Reducing the area to 1/4 does not reduce the complexity
         * (err_per_frame) to 1/4... effective_sizeratio attempts
         * to provide a crude correction for this
         */
        effective_size_ratio = (double)(new_width * new_height) /
                               (double)(cpi->oxcf.Width * cpi->oxcf.Height);
        effective_size_ratio = (1.0 + (3.0 * effective_size_ratio)) / 4.0;

        /* Now try again and see what Q we get with the smaller
         * image size
         */
        kf_q = estimate_kf_group_q(cpi, err_per_frame * effective_size_ratio,
                                   (int)bits_per_frame, group_iiratio);
      }
    }

    if ((cpi->common.Width != new_width) ||
        (cpi->common.Height != new_height)) {
      cpi->common.Width = new_width;
      cpi->common.Height = new_height;
      vp8_alloc_compressor_data(cpi);
    }
  }
}
