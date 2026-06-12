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

#include "./vpx_scale_rtcd.h"
#include "vpx_dsp/psnr.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

#include "vp9/common/vp9_loopfilter.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_quant_common.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_picklpf.h"
#include "vp9/encoder/vp9_quantize.h"

static unsigned int get_section_intra_rating(const VP9_COMP *cpi) {
  unsigned int section_intra_rating;

  section_intra_rating = (cpi->common.frame_type == KEY_FRAME)
                             ? cpi->twopass.key_frame_section_intra_rating
                             : cpi->twopass.section_intra_rating;

  return section_intra_rating;
}

static int get_max_filter_level(const VP9_COMP *cpi) {
  if (cpi->oxcf.pass == 2) {
    unsigned int section_intra_rating = get_section_intra_rating(cpi);
    return section_intra_rating > 8 ? MAX_LOOP_FILTER * 3 / 4 : MAX_LOOP_FILTER;
  } else {
    return MAX_LOOP_FILTER;
  }
}

static int64_t try_filter_frame(const YV12_BUFFER_CONFIG *sd,
                                VP9_COMP *const cpi, int filt_level,
                                int partial_frame) {
  VP9_COMMON *const cm = &cpi->common;
  int64_t filt_err;

  vp9_build_mask_frame(cm, filt_level, partial_frame);

  if (cpi->num_workers > 1)
    vp9_loop_filter_frame_mt(cm->frame_to_show, cm, cpi->td.mb.e_mbd.plane,
                             filt_level, 1, partial_frame, cpi->workers,
                             cpi->num_workers, &cpi->lf_row_sync);
  else
    vp9_loop_filter_frame(cm->frame_to_show, cm, &cpi->td.mb.e_mbd, filt_level,
                          1, partial_frame);

#if CONFIG_VP9_HIGHBITDEPTH
  if (cm->use_highbitdepth) {
    filt_err = vpx_highbd_get_y_sse(sd, cm->frame_to_show);
  } else {
    filt_err = vpx_get_y_sse(sd, cm->frame_to_show);
  }
#else
  filt_err = vpx_get_y_sse(sd, cm->frame_to_show);
#endif  // CONFIG_VP9_HIGHBITDEPTH

  // Re-instate the unfiltered frame
  vpx_yv12_copy_y(&cpi->last_frame_uf, cm->frame_to_show);

  return filt_err;
}

static int search_filter_level(const YV12_BUFFER_CONFIG *sd, VP9_COMP *cpi,
                               int partial_frame) {
  const VP9_COMMON *const cm = &cpi->common;
  const struct loopfilter *const lf = &cm->lf;
  const int min_filter_level = 0;
  const int max_filter_level = get_max_filter_level(cpi);
  int filt_direction = 0;
  int64_t best_err;
  int filt_best;

  // Start the search at the previous frame filter level unless it is now out of
  // range.
  int filt_mid = clamp(lf->last_filt_level, min_filter_level, max_filter_level);
  int filter_step = filt_mid < 16 ? 4 : filt_mid / 4;
  // Sum squared error at each filter level
  int64_t ss_err[MAX_LOOP_FILTER + 1];
  unsigned int section_intra_rating = get_section_intra_rating(cpi);

  // Set each entry to -1
  memset(ss_err, 0xFF, sizeof(ss_err));

  //  Make a copy of the unfiltered / processed recon buffer
  vpx_yv12_copy_y(cm->frame_to_show, &cpi->last_frame_uf);

  best_err = try_filter_frame(sd, cpi, filt_mid, partial_frame);
  filt_best = filt_mid;
  ss_err[filt_mid] = best_err;

  while (filter_step > 0) {
    const int filt_high = VPXMIN(filt_mid + filter_step, max_filter_level);
    const int filt_low = VPXMAX(filt_mid - filter_step, min_filter_level);

    // Bias against raising loop filter in favor of lowering it.
    int64_t bias = (best_err >> (15 - (filt_mid / 8))) * filter_step;

    if ((cpi->oxcf.pass == 2) && (section_intra_rating < 20))
      bias = (bias * section_intra_rating) / 20;

    // yx, bias less for large block size
    if (cm->tx_mode != ONLY_4X4) bias >>= 1;

    if (filt_direction <= 0 && filt_low != filt_mid) {
      // Get Low filter error score
      if (ss_err[filt_low] < 0) {
        ss_err[filt_low] = try_filter_frame(sd, cpi, filt_low, partial_frame);
      }
      // If value is close to the best so far then bias towards a lower loop
      // filter value.
      if ((ss_err[filt_low] - bias) < best_err) {
        // Was it actually better than the previous best?
        if (ss_err[filt_low] < best_err) best_err = ss_err[filt_low];

        filt_best = filt_low;
      }
    }

    // Now look at filt_high
    if (filt_direction >= 0 && filt_high != filt_mid) {
      if (ss_err[filt_high] < 0) {
        ss_err[filt_high] = try_filter_frame(sd, cpi, filt_high, partial_frame);
      }
      // Was it better than the previous best?
      if (ss_err[filt_high] < (best_err - bias)) {
        best_err = ss_err[filt_high];
        filt_best = filt_high;
      }
    }

    // Half the step distance if the best filter value was the same as last time
    if (filt_best == filt_mid) {
      filter_step /= 2;
      filt_direction = 0;
    } else {
      filt_direction = (filt_best < filt_mid) ? -1 : 1;
      filt_mid = filt_best;
    }
  }

  return filt_best;
}

void vp9_pick_filter_level(const YV12_BUFFER_CONFIG *sd, VP9_COMP *cpi,
                           LPF_PICK_METHOD method) {
  VP9_COMMON *const cm = &cpi->common;
  struct loopfilter *const lf = &cm->lf;

  lf->sharpness_level = 0;

  if (method == LPF_PICK_MINIMAL_LPF && lf->filter_level) {
    lf->filter_level = 0;
  } else if (method >= LPF_PICK_FROM_Q) {
    const int min_filter_level = 0;
    const int max_filter_level = get_max_filter_level(cpi);
    const int q = vp9_ac_quant(cm->base_qindex, 0, cm->bit_depth);
// These values were determined by linear fitting the result of the
// searched level, filt_guess = q * 0.316206 + 3.87252
#if CONFIG_VP9_HIGHBITDEPTH
    int filt_guess;
    switch (cm->bit_depth) {
      case VPX_BITS_8:
        filt_guess = ROUND_POWER_OF_TWO(q * 20723 + 1015158, 18);
        break;
      case VPX_BITS_10:
        filt_guess = ROUND_POWER_OF_TWO(q * 20723 + 4060632, 20);
        break;
      default:
        assert(cm->bit_depth == VPX_BITS_12);
        filt_guess = ROUND_POWER_OF_TWO(q * 20723 + 16242526, 22);
        break;
    }
#else
    int filt_guess = ROUND_POWER_OF_TWO(q * 20723 + 1015158, 18);
#endif  // CONFIG_VP9_HIGHBITDEPTH
    if (cpi->oxcf.pass == 0 && cpi->oxcf.rc_mode == VPX_CBR &&
        cpi->oxcf.aq_mode == CYCLIC_REFRESH_AQ && cm->seg.enabled &&
        (cm->base_qindex < 200 || cm->width * cm->height > 320 * 240) &&
        cpi->oxcf.content != VP9E_CONTENT_SCREEN && cm->frame_type != KEY_FRAME)
      filt_guess = 5 * filt_guess >> 3;

    if (cm->frame_type == KEY_FRAME) filt_guess -= 4;
    lf->filter_level = clamp(filt_guess, min_filter_level, max_filter_level);
  } else {
    lf->filter_level =
        search_filter_level(sd, cpi, method == LPF_PICK_FROM_SUBIMAGE);
  }
}
