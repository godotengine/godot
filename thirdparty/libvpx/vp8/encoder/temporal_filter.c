/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/common/onyxc_int.h"
#include "onyx_int.h"
#include "vp8/common/systemdependent.h"
#include "vp8/encoder/quantize.h"
#include "vp8/common/alloccommon.h"
#include "mcomp.h"
#include "firstpass.h"
#include "vpx_scale/vpx_scale.h"
#include "vp8/common/extend.h"
#include "ratectrl.h"
#include "vp8/common/quant_common.h"
#include "segmentation.h"
#include "temporal_filter.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8/common/swapyv12buffer.h"
#include "vp8/common/threading.h"
#include "vpx_ports/vpx_timer.h"

#include <math.h>
#include <limits.h>

#define ALT_REF_MC_ENABLED 1     /* toggle MC in AltRef filtering */
#define ALT_REF_SUBPEL_ENABLED 1 /* toggle subpel in MC AltRef filtering */

#if VP8_TEMPORAL_ALT_REF

static void vp8_temporal_filter_predictors_mb_c(
    MACROBLOCKD *x, unsigned char *y_mb_ptr, unsigned char *u_mb_ptr,
    unsigned char *v_mb_ptr, int stride, int mv_row, int mv_col,
    unsigned char *pred) {
  int offset;
  unsigned char *yptr, *uptr, *vptr;

  /* Y */
  yptr = y_mb_ptr + (mv_row >> 3) * stride + (mv_col >> 3);

  if ((mv_row | mv_col) & 7) {
    x->subpixel_predict16x16(yptr, stride, mv_col & 7, mv_row & 7, &pred[0],
                             16);
  } else {
    vp8_copy_mem16x16(yptr, stride, &pred[0], 16);
  }

  /* U & V */
  mv_row >>= 1;
  mv_col >>= 1;
  stride = (stride + 1) >> 1;
  offset = (mv_row >> 3) * stride + (mv_col >> 3);
  uptr = u_mb_ptr + offset;
  vptr = v_mb_ptr + offset;

  if ((mv_row | mv_col) & 7) {
    x->subpixel_predict8x8(uptr, stride, mv_col & 7, mv_row & 7, &pred[256], 8);
    x->subpixel_predict8x8(vptr, stride, mv_col & 7, mv_row & 7, &pred[320], 8);
  } else {
    vp8_copy_mem8x8(uptr, stride, &pred[256], 8);
    vp8_copy_mem8x8(vptr, stride, &pred[320], 8);
  }
}
void vp8_temporal_filter_apply_c(unsigned char *frame1, unsigned int stride,
                                 unsigned char *frame2, unsigned int block_size,
                                 int strength, int filter_weight,
                                 unsigned int *accumulator,
                                 unsigned short *count) {
  unsigned int i, j, k;
  int modifier;
  int byte = 0;
  const int rounding = strength > 0 ? 1 << (strength - 1) : 0;

  for (i = 0, k = 0; i < block_size; ++i) {
    for (j = 0; j < block_size; j++, k++) {
      int src_byte = frame1[byte];
      int pixel_value = *frame2++;

      modifier = src_byte - pixel_value;
      /* This is an integer approximation of:
       * float coeff = (3.0 * modifer * modifier) / pow(2, strength);
       * modifier =  (int)roundf(coeff > 16 ? 0 : 16-coeff);
       */
      modifier *= modifier;
      modifier *= 3;
      modifier += rounding;
      modifier >>= strength;

      if (modifier > 16) modifier = 16;

      modifier = 16 - modifier;
      modifier *= filter_weight;

      count[k] += modifier;
      accumulator[k] += modifier * pixel_value;

      byte++;
    }

    byte += stride - block_size;
  }
}

#if ALT_REF_MC_ENABLED

static int vp8_temporal_filter_find_matching_mb_c(VP8_COMP *cpi,
                                                  YV12_BUFFER_CONFIG *arf_frame,
                                                  YV12_BUFFER_CONFIG *frame_ptr,
                                                  int mb_offset,
                                                  int error_thresh) {
  MACROBLOCK *x = &cpi->mb;
  int step_param;
  int sadpb = x->sadperbit16;
  int bestsme = INT_MAX;

  BLOCK *b = &x->block[0];
  BLOCKD *d = &x->e_mbd.block[0];
  int_mv best_ref_mv1;
  int_mv best_ref_mv1_full; /* full-pixel value of best_ref_mv1 */

  /* Save input state */
  unsigned char **base_src = b->base_src;
  int src = b->src;
  int src_stride = b->src_stride;
  unsigned char *base_pre = x->e_mbd.pre.y_buffer;
  int pre = d->offset;
  int pre_stride = x->e_mbd.pre.y_stride;

  (void)error_thresh;

  best_ref_mv1.as_int = 0;
  best_ref_mv1_full.as_mv.col = best_ref_mv1.as_mv.col >> 3;
  best_ref_mv1_full.as_mv.row = best_ref_mv1.as_mv.row >> 3;

  /* Setup frame pointers */
  b->base_src = &arf_frame->y_buffer;
  b->src_stride = arf_frame->y_stride;
  b->src = mb_offset;

  x->e_mbd.pre.y_buffer = frame_ptr->y_buffer;
  x->e_mbd.pre.y_stride = frame_ptr->y_stride;
  d->offset = mb_offset;

  /* Further step/diamond searches as necessary */
  if (cpi->Speed < 8) {
    step_param = cpi->sf.first_step + (cpi->Speed > 5);
  } else {
    step_param = cpi->sf.first_step + 2;
  }

  /* TODO Check that the 16x16 vf & sdf are selected here */
  /* Ignore mv costing by sending NULL cost arrays */
  bestsme =
      vp8_hex_search(x, b, d, &best_ref_mv1_full, &d->bmi.mv, step_param, sadpb,
                     &cpi->fn_ptr[BLOCK_16X16], NULL, &best_ref_mv1);
  (void)bestsme;  // Ignore unused return value.

#if ALT_REF_SUBPEL_ENABLED
  /* Try sub-pixel MC? */
  {
    int distortion;
    unsigned int sse;
    /* Ignore mv costing by sending NULL cost array */
    bestsme = cpi->find_fractional_mv_step(
        x, b, d, &d->bmi.mv, &best_ref_mv1, x->errorperbit,
        &cpi->fn_ptr[BLOCK_16X16], NULL, &distortion, &sse);
  }
#endif

  /* Save input state */
  b->base_src = base_src;
  b->src = src;
  b->src_stride = src_stride;
  x->e_mbd.pre.y_buffer = base_pre;
  d->offset = pre;
  x->e_mbd.pre.y_stride = pre_stride;

  return bestsme;
}
#endif

static void vp8_temporal_filter_iterate_c(VP8_COMP *cpi, int frame_count,
                                          int alt_ref_index, int strength) {
  int byte;
  int frame;
  int mb_col, mb_row;
  unsigned int filter_weight;
  int mb_cols = cpi->common.mb_cols;
  int mb_rows = cpi->common.mb_rows;
  int mb_y_offset = 0;
  int mb_uv_offset = 0;
  DECLARE_ALIGNED(16, unsigned int, accumulator[16 * 16 + 8 * 8 + 8 * 8]);
  DECLARE_ALIGNED(16, unsigned short, count[16 * 16 + 8 * 8 + 8 * 8]);
  MACROBLOCKD *mbd = &cpi->mb.e_mbd;
  YV12_BUFFER_CONFIG *f = cpi->frames[alt_ref_index];
  unsigned char *dst1, *dst2;
  DECLARE_ALIGNED(16, unsigned char, predictor[16 * 16 + 8 * 8 + 8 * 8]);

  /* Save input state */
  unsigned char *y_buffer = mbd->pre.y_buffer;
  unsigned char *u_buffer = mbd->pre.u_buffer;
  unsigned char *v_buffer = mbd->pre.v_buffer;

  for (mb_row = 0; mb_row < mb_rows; ++mb_row) {
#if ALT_REF_MC_ENABLED
    /* Source frames are extended to 16 pixels.  This is different than
     *  L/A/G reference frames that have a border of 32 (VP8BORDERINPIXELS)
     * A 6 tap filter is used for motion search.  This requires 2 pixels
     *  before and 3 pixels after.  So the largest Y mv on a border would
     *  then be 16 - 3.  The UV blocks are half the size of the Y and
     *  therefore only extended by 8.  The largest mv that a UV block
     *  can support is 8 - 3.  A UV mv is half of a Y mv.
     *  (16 - 3) >> 1 == 6 which is greater than 8 - 3.
     * To keep the mv in play for both Y and UV planes the max that it
     *  can be on a border is therefore 16 - 5.
     */
    cpi->mb.mv_row_min = -((mb_row * 16) + (16 - 5));
    cpi->mb.mv_row_max = ((cpi->common.mb_rows - 1 - mb_row) * 16) + (16 - 5);
#endif

    for (mb_col = 0; mb_col < mb_cols; ++mb_col) {
      int i, j, k;
      int stride;

      memset(accumulator, 0, 384 * sizeof(unsigned int));
      memset(count, 0, 384 * sizeof(unsigned short));

#if ALT_REF_MC_ENABLED
      cpi->mb.mv_col_min = -((mb_col * 16) + (16 - 5));
      cpi->mb.mv_col_max = ((cpi->common.mb_cols - 1 - mb_col) * 16) + (16 - 5);
#endif

      for (frame = 0; frame < frame_count; ++frame) {
        if (cpi->frames[frame] == NULL) continue;

        mbd->block[0].bmi.mv.as_mv.row = 0;
        mbd->block[0].bmi.mv.as_mv.col = 0;

        if (frame == alt_ref_index) {
          filter_weight = 2;
        } else {
          int err = 0;
#if ALT_REF_MC_ENABLED
#define THRESH_LOW 10000
#define THRESH_HIGH 20000
          /* Find best match in this frame by MC */
          err = vp8_temporal_filter_find_matching_mb_c(
              cpi, cpi->frames[alt_ref_index], cpi->frames[frame], mb_y_offset,
              THRESH_LOW);
#endif
          /* Assign higher weight to matching MB if it's error
           * score is lower. If not applying MC default behavior
           * is to weight all MBs equal.
           */
          filter_weight = err < THRESH_LOW ? 2 : err < THRESH_HIGH ? 1 : 0;
        }

        if (filter_weight != 0) {
          /* Construct the predictors */
          vp8_temporal_filter_predictors_mb_c(
              mbd, cpi->frames[frame]->y_buffer + mb_y_offset,
              cpi->frames[frame]->u_buffer + mb_uv_offset,
              cpi->frames[frame]->v_buffer + mb_uv_offset,
              cpi->frames[frame]->y_stride, mbd->block[0].bmi.mv.as_mv.row,
              mbd->block[0].bmi.mv.as_mv.col, predictor);

          /* Apply the filter (YUV) */
          vp8_temporal_filter_apply(f->y_buffer + mb_y_offset, f->y_stride,
                                    predictor, 16, strength, filter_weight,
                                    accumulator, count);

          vp8_temporal_filter_apply(f->u_buffer + mb_uv_offset, f->uv_stride,
                                    predictor + 256, 8, strength, filter_weight,
                                    accumulator + 256, count + 256);

          vp8_temporal_filter_apply(f->v_buffer + mb_uv_offset, f->uv_stride,
                                    predictor + 320, 8, strength, filter_weight,
                                    accumulator + 320, count + 320);
        }
      }

      /* Normalize filter output to produce AltRef frame */
      dst1 = cpi->alt_ref_buffer.y_buffer;
      stride = cpi->alt_ref_buffer.y_stride;
      byte = mb_y_offset;
      for (i = 0, k = 0; i < 16; ++i) {
        for (j = 0; j < 16; j++, k++) {
          unsigned int pval = accumulator[k] + (count[k] >> 1);
          pval *= cpi->fixed_divide[count[k]];
          pval >>= 19;

          dst1[byte] = (unsigned char)pval;

          /* move to next pixel */
          byte++;
        }

        byte += stride - 16;
      }

      dst1 = cpi->alt_ref_buffer.u_buffer;
      dst2 = cpi->alt_ref_buffer.v_buffer;
      stride = cpi->alt_ref_buffer.uv_stride;
      byte = mb_uv_offset;
      for (i = 0, k = 256; i < 8; ++i) {
        for (j = 0; j < 8; j++, k++) {
          int m = k + 64;

          /* U */
          unsigned int pval = accumulator[k] + (count[k] >> 1);
          pval *= cpi->fixed_divide[count[k]];
          pval >>= 19;
          dst1[byte] = (unsigned char)pval;

          /* V */
          pval = accumulator[m] + (count[m] >> 1);
          pval *= cpi->fixed_divide[count[m]];
          pval >>= 19;
          dst2[byte] = (unsigned char)pval;

          /* move to next pixel */
          byte++;
        }

        byte += stride - 8;
      }

      mb_y_offset += 16;
      mb_uv_offset += 8;
    }

    mb_y_offset += 16 * (f->y_stride - mb_cols);
    mb_uv_offset += 8 * (f->uv_stride - mb_cols);
  }

  /* Restore input state */
  mbd->pre.y_buffer = y_buffer;
  mbd->pre.u_buffer = u_buffer;
  mbd->pre.v_buffer = v_buffer;
}

void vp8_temporal_filter_prepare_c(VP8_COMP *cpi, int distance) {
  int frame = 0;

  int num_frames_backward = 0;
  int num_frames_forward = 0;
  int frames_to_blur_backward = 0;
  int frames_to_blur_forward = 0;
  int frames_to_blur = 0;
  int start_frame = 0;

  int strength = cpi->oxcf.arnr_strength;

  int blur_type = cpi->oxcf.arnr_type;

  int max_frames = cpi->active_arnr_frames;

  num_frames_backward = distance;
  num_frames_forward =
      vp8_lookahead_depth(cpi->lookahead) - (num_frames_backward + 1);

  switch (blur_type) {
    case 1:
      /* Backward Blur */

      frames_to_blur_backward = num_frames_backward;

      if (frames_to_blur_backward >= max_frames) {
        frames_to_blur_backward = max_frames - 1;
      }

      frames_to_blur = frames_to_blur_backward + 1;
      break;

    case 2:
      /* Forward Blur */

      frames_to_blur_forward = num_frames_forward;

      if (frames_to_blur_forward >= max_frames) {
        frames_to_blur_forward = max_frames - 1;
      }

      frames_to_blur = frames_to_blur_forward + 1;
      break;

    case 3:
    default:
      /* Center Blur */
      frames_to_blur_forward = num_frames_forward;
      frames_to_blur_backward = num_frames_backward;

      if (frames_to_blur_forward > frames_to_blur_backward) {
        frames_to_blur_forward = frames_to_blur_backward;
      }

      if (frames_to_blur_backward > frames_to_blur_forward) {
        frames_to_blur_backward = frames_to_blur_forward;
      }

      /* When max_frames is even we have 1 more frame backward than forward */
      if (frames_to_blur_forward > (max_frames - 1) / 2) {
        frames_to_blur_forward = ((max_frames - 1) / 2);
      }

      if (frames_to_blur_backward > (max_frames / 2)) {
        frames_to_blur_backward = (max_frames / 2);
      }

      frames_to_blur = frames_to_blur_backward + frames_to_blur_forward + 1;
      break;
  }

  start_frame = distance + frames_to_blur_forward;

  /* Setup frame pointers, NULL indicates frame not included in filter */
  memset(cpi->frames, 0, max_frames * sizeof(YV12_BUFFER_CONFIG *));
  for (frame = 0; frame < frames_to_blur; ++frame) {
    int which_buffer = start_frame - frame;
    struct lookahead_entry *buf =
        vp8_lookahead_peek(cpi->lookahead, which_buffer, PEEK_FORWARD);
    cpi->frames[frames_to_blur - 1 - frame] = &buf->img;
  }

  vp8_temporal_filter_iterate_c(cpi, frames_to_blur, frames_to_blur_backward,
                                strength);
}
#endif
