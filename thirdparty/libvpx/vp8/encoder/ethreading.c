/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <stddef.h>

#include "onyx_int.h"
#include "vpx_util/vpx_pthread.h"
#include "vp8/common/threading.h"
#include "vp8/common/common.h"
#include "vp8/common/extend.h"
#include "bitstream.h"
#include "encodeframe.h"
#include "ethreading.h"

#if CONFIG_MULTITHREAD

extern void vp8cx_mb_init_quantizer(VP8_COMP *cpi, MACROBLOCK *x,
                                    int ok_to_skip);

static THREADFN thread_loopfilter(void *p_data) {
  VP8_COMP *cpi = (VP8_COMP *)(((LPFTHREAD_DATA *)p_data)->ptr1);
  VP8_COMMON *cm = &cpi->common;

  while (1) {
    if (vpx_atomic_load_acquire(&cpi->b_multi_threaded) == 0) break;

    if (vp8_sem_wait(&cpi->h_event_start_lpf) == 0) {
      /* we're shutting down */
      if (vpx_atomic_load_acquire(&cpi->b_multi_threaded) == 0) break;

      vp8_loopfilter_frame(cpi, cm);

      vp8_sem_post(&cpi->h_event_end_lpf);
    }
  }

  return THREAD_EXIT_SUCCESS;
}

static THREADFN thread_encoding_proc(void *p_data) {
  int ithread = ((ENCODETHREAD_DATA *)p_data)->ithread;
  VP8_COMP *cpi = (VP8_COMP *)(((ENCODETHREAD_DATA *)p_data)->ptr1);
  MB_ROW_COMP *mbri = (MB_ROW_COMP *)(((ENCODETHREAD_DATA *)p_data)->ptr2);
  ENTROPY_CONTEXT_PLANES mb_row_left_context;

  while (1) {
    if (vpx_atomic_load_acquire(&cpi->b_multi_threaded) == 0) break;

    if (vp8_sem_wait(&cpi->h_event_start_encoding[ithread]) == 0) {
      const int nsync = cpi->mt_sync_range;
      VP8_COMMON *cm = &cpi->common;
      int mb_row;
      MACROBLOCK *x = &mbri->mb;
      MACROBLOCKD *xd = &x->e_mbd;
      TOKENEXTRA *tp;
#if CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING
      TOKENEXTRA *tp_start = cpi->tok + (1 + ithread) * (16 * 24);
      const int num_part = (1 << cm->multi_token_partition);
#endif

      int *segment_counts = mbri->segment_counts;
      int *totalrate = &mbri->totalrate;

      /* we're shutting down */
      if (vpx_atomic_load_acquire(&cpi->b_multi_threaded) == 0) break;

      xd->mode_info_context = cm->mi + cm->mode_info_stride * (ithread + 1);
      xd->mode_info_stride = cm->mode_info_stride;

      for (mb_row = ithread + 1; mb_row < cm->mb_rows;
           mb_row += (cpi->encoding_thread_count + 1)) {
        int recon_yoffset, recon_uvoffset;
        int mb_col;
        int ref_fb_idx = cm->lst_fb_idx;
        int dst_fb_idx = cm->new_fb_idx;
        int recon_y_stride = cm->yv12_fb[ref_fb_idx].y_stride;
        int recon_uv_stride = cm->yv12_fb[ref_fb_idx].uv_stride;
        int map_index = (mb_row * cm->mb_cols);
        const vpx_atomic_int *last_row_current_mb_col;
        vpx_atomic_int *current_mb_col = &cpi->mt_current_mb_col[mb_row];

#if (CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING)
        vp8_writer *w = &cpi->bc[1 + (mb_row % num_part)];
#else
        tp = cpi->tok + (mb_row * (cm->mb_cols * 16 * 24));
        cpi->tplist[mb_row].start = tp;
#endif

        last_row_current_mb_col = &cpi->mt_current_mb_col[mb_row - 1];

        /* reset above block coeffs */
        xd->above_context = cm->above_context;
        xd->left_context = &mb_row_left_context;

        vp8_zero(mb_row_left_context);

        xd->up_available = (mb_row != 0);
        recon_yoffset = (mb_row * recon_y_stride * 16);
        recon_uvoffset = (mb_row * recon_uv_stride * 8);

        /* Set the mb activity pointer to the start of the row. */
        x->mb_activity_ptr = &cpi->mb_activity_map[map_index];

        /* for each macroblock col in image */
        for (mb_col = 0; mb_col < cm->mb_cols; ++mb_col) {
          if (((mb_col - 1) % nsync) == 0) {
            vpx_atomic_store_release(current_mb_col, mb_col - 1);
          }

          if (mb_row && !(mb_col & (nsync - 1))) {
            vp8_atomic_spin_wait(mb_col, last_row_current_mb_col, nsync);
          }

#if CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING
          tp = tp_start;
#endif

          /* Distance of Mb to the various image edges.
           * These specified to 8th pel as they are always compared
           * to values that are in 1/8th pel units
           */
          xd->mb_to_left_edge = -((mb_col * 16) << 3);
          xd->mb_to_right_edge = ((cm->mb_cols - 1 - mb_col) * 16) << 3;
          xd->mb_to_top_edge = -((mb_row * 16) << 3);
          xd->mb_to_bottom_edge = ((cm->mb_rows - 1 - mb_row) * 16) << 3;

          /* Set up limit values for motion vectors used to prevent
           * them extending outside the UMV borders
           */
          x->mv_col_min = -((mb_col * 16) + (VP8BORDERINPIXELS - 16));
          x->mv_col_max =
              ((cm->mb_cols - 1 - mb_col) * 16) + (VP8BORDERINPIXELS - 16);
          x->mv_row_min = -((mb_row * 16) + (VP8BORDERINPIXELS - 16));
          x->mv_row_max =
              ((cm->mb_rows - 1 - mb_row) * 16) + (VP8BORDERINPIXELS - 16);

          xd->dst.y_buffer = cm->yv12_fb[dst_fb_idx].y_buffer + recon_yoffset;
          xd->dst.u_buffer = cm->yv12_fb[dst_fb_idx].u_buffer + recon_uvoffset;
          xd->dst.v_buffer = cm->yv12_fb[dst_fb_idx].v_buffer + recon_uvoffset;
          xd->left_available = (mb_col != 0);

          x->rddiv = cpi->RDDIV;
          x->rdmult = cpi->RDMULT;

          /* Copy current mb to a buffer */
          vp8_copy_mem16x16(x->src.y_buffer, x->src.y_stride, x->thismb, 16);

          if (cpi->oxcf.tuning == VP8_TUNE_SSIM) vp8_activity_masking(cpi, x);

          /* Is segmentation enabled */
          /* MB level adjustment to quantizer */
          if (xd->segmentation_enabled) {
            /* Code to set segment id in xd->mbmi.segment_id for
             * current MB (with range checking)
             */
            if (cpi->segmentation_map[map_index + mb_col] <= 3) {
              xd->mode_info_context->mbmi.segment_id =
                  cpi->segmentation_map[map_index + mb_col];
            } else {
              xd->mode_info_context->mbmi.segment_id = 0;
            }

            vp8cx_mb_init_quantizer(cpi, x, 1);
          } else {
            /* Set to Segment 0 by default */
            xd->mode_info_context->mbmi.segment_id = 0;
          }

          x->active_ptr = cpi->active_map + map_index + mb_col;

          if (cm->frame_type == KEY_FRAME) {
            *totalrate += vp8cx_encode_intra_macroblock(cpi, x, &tp);
#ifdef MODE_STATS
            y_modes[xd->mbmi.mode]++;
#endif
          } else {
            *totalrate += vp8cx_encode_inter_macroblock(
                cpi, x, &tp, recon_yoffset, recon_uvoffset, mb_row, mb_col);

#ifdef MODE_STATS
            inter_y_modes[xd->mbmi.mode]++;

            if (xd->mbmi.mode == SPLITMV) {
              int b;

              for (b = 0; b < xd->mbmi.partition_count; ++b) {
                inter_b_modes[x->partition->bmi[b].mode]++;
              }
            }

#endif
            // Keep track of how many (consecutive) times a  block
            // is coded as ZEROMV_LASTREF, for base layer frames.
            // Reset to 0 if its coded as anything else.
            if (cpi->current_layer == 0) {
              if (xd->mode_info_context->mbmi.mode == ZEROMV &&
                  xd->mode_info_context->mbmi.ref_frame == LAST_FRAME) {
                // Increment, check for wrap-around.
                if (cpi->consec_zero_last[map_index + mb_col] < 255) {
                  cpi->consec_zero_last[map_index + mb_col] += 1;
                }
                if (cpi->consec_zero_last_mvbias[map_index + mb_col] < 255) {
                  cpi->consec_zero_last_mvbias[map_index + mb_col] += 1;
                }
              } else {
                cpi->consec_zero_last[map_index + mb_col] = 0;
                cpi->consec_zero_last_mvbias[map_index + mb_col] = 0;
              }
              if (x->zero_last_dot_suppress) {
                cpi->consec_zero_last_mvbias[map_index + mb_col] = 0;
              }
            }

            /* Special case code for cyclic refresh
             * If cyclic update enabled then copy
             * xd->mbmi.segment_id; (which may have been updated
             * based on mode during
             * vp8cx_encode_inter_macroblock()) back into the
             * global segmentation map
             */
            if ((cpi->current_layer == 0) &&
                (cpi->cyclic_refresh_mode_enabled &&
                 xd->segmentation_enabled)) {
              const MB_MODE_INFO *mbmi = &xd->mode_info_context->mbmi;
              cpi->segmentation_map[map_index + mb_col] = mbmi->segment_id;

              /* If the block has been refreshed mark it as clean
               * (the magnitude of the -ve influences how long it
               * will be before we consider another refresh):
               * Else if it was coded (last frame 0,0) and has
               * not already been refreshed then mark it as a
               * candidate for cleanup next time (marked 0) else
               * mark it as dirty (1).
               */
              if (mbmi->segment_id) {
                cpi->cyclic_refresh_map[map_index + mb_col] = -1;
              } else if ((mbmi->mode == ZEROMV) &&
                         (mbmi->ref_frame == LAST_FRAME)) {
                if (cpi->cyclic_refresh_map[map_index + mb_col] == 1) {
                  cpi->cyclic_refresh_map[map_index + mb_col] = 0;
                }
              } else {
                cpi->cyclic_refresh_map[map_index + mb_col] = 1;
              }
            }
          }

#if CONFIG_REALTIME_ONLY & CONFIG_ONTHEFLY_BITPACKING
          /* pack tokens for this MB */
          {
            int tok_count = tp - tp_start;
            vp8_pack_tokens(w, tp_start, tok_count);
          }
#else
          cpi->tplist[mb_row].stop = tp;
#endif
          /* Increment pointer into gf usage flags structure. */
          x->gf_active_ptr++;

          /* Increment the activity mask pointers. */
          x->mb_activity_ptr++;

          /* adjust to the next column of macroblocks */
          x->src.y_buffer += 16;
          x->src.u_buffer += 8;
          x->src.v_buffer += 8;

          recon_yoffset += 16;
          recon_uvoffset += 8;

          /* Keep track of segment usage */
          segment_counts[xd->mode_info_context->mbmi.segment_id]++;

          /* skip to next mb */
          xd->mode_info_context++;
          x->partition_info++;
          xd->above_context++;
        }

        vp8_extend_mb_row(&cm->yv12_fb[dst_fb_idx], xd->dst.y_buffer + 16,
                          xd->dst.u_buffer + 8, xd->dst.v_buffer + 8);

        vpx_atomic_store_release(current_mb_col, mb_col + nsync);

        /* this is to account for the border */
        xd->mode_info_context++;
        x->partition_info++;

        x->src.y_buffer +=
            16 * x->src.y_stride * (cpi->encoding_thread_count + 1) -
            16 * cm->mb_cols;
        x->src.u_buffer +=
            8 * x->src.uv_stride * (cpi->encoding_thread_count + 1) -
            8 * cm->mb_cols;
        x->src.v_buffer +=
            8 * x->src.uv_stride * (cpi->encoding_thread_count + 1) -
            8 * cm->mb_cols;

        xd->mode_info_context +=
            xd->mode_info_stride * cpi->encoding_thread_count;
        x->partition_info += xd->mode_info_stride * cpi->encoding_thread_count;
        x->gf_active_ptr += cm->mb_cols * cpi->encoding_thread_count;
      }
      /* Signal that this thread has completed processing its rows. */
      vp8_sem_post(&cpi->h_event_end_encoding[ithread]);
    }
  }

  /* printf("exit thread %d\n", ithread); */
  return THREAD_EXIT_SUCCESS;
}

static void setup_mbby_copy(MACROBLOCK *mbdst, MACROBLOCK *mbsrc) {
  MACROBLOCK *x = mbsrc;
  MACROBLOCK *z = mbdst;
  int i;

  z->ss = x->ss;
  z->ss_count = x->ss_count;
  z->searches_per_step = x->searches_per_step;
  z->errorperbit = x->errorperbit;

  z->sadperbit16 = x->sadperbit16;
  z->sadperbit4 = x->sadperbit4;

  /*
  z->mv_col_min    = x->mv_col_min;
  z->mv_col_max    = x->mv_col_max;
  z->mv_row_min    = x->mv_row_min;
  z->mv_row_max    = x->mv_row_max;
  */

  z->short_fdct4x4 = x->short_fdct4x4;
  z->short_fdct8x4 = x->short_fdct8x4;
  z->short_walsh4x4 = x->short_walsh4x4;
  z->quantize_b = x->quantize_b;
  z->optimize = x->optimize;

  /*
  z->mvc              = x->mvc;
  z->src.y_buffer      = x->src.y_buffer;
  z->src.u_buffer      = x->src.u_buffer;
  z->src.v_buffer      = x->src.v_buffer;
  */

  z->mvcost[0] = x->mvcost[0];
  z->mvcost[1] = x->mvcost[1];
  z->mvsadcost[0] = x->mvsadcost[0];
  z->mvsadcost[1] = x->mvsadcost[1];

  z->token_costs = x->token_costs;
  z->inter_bmode_costs = x->inter_bmode_costs;
  z->mbmode_cost = x->mbmode_cost;
  z->intra_uv_mode_cost = x->intra_uv_mode_cost;
  z->bmode_costs = x->bmode_costs;

  for (i = 0; i < 25; ++i) {
    z->block[i].quant = x->block[i].quant;
    z->block[i].quant_fast = x->block[i].quant_fast;
    z->block[i].quant_shift = x->block[i].quant_shift;
    z->block[i].zbin = x->block[i].zbin;
    z->block[i].zrun_zbin_boost = x->block[i].zrun_zbin_boost;
    z->block[i].round = x->block[i].round;
    z->block[i].src_stride = x->block[i].src_stride;
  }

  z->q_index = x->q_index;
  z->act_zbin_adj = x->act_zbin_adj;
  z->last_act_zbin_adj = x->last_act_zbin_adj;

  {
    MACROBLOCKD *xd = &x->e_mbd;
    MACROBLOCKD *zd = &z->e_mbd;

    /*
    zd->mode_info_context = xd->mode_info_context;
    zd->mode_info        = xd->mode_info;

    zd->mode_info_stride  = xd->mode_info_stride;
    zd->frame_type       = xd->frame_type;
    zd->up_available     = xd->up_available   ;
    zd->left_available   = xd->left_available;
    zd->left_context     = xd->left_context;
    zd->last_frame_dc     = xd->last_frame_dc;
    zd->last_frame_dccons = xd->last_frame_dccons;
    zd->gold_frame_dc     = xd->gold_frame_dc;
    zd->gold_frame_dccons = xd->gold_frame_dccons;
    zd->mb_to_left_edge    = xd->mb_to_left_edge;
    zd->mb_to_right_edge   = xd->mb_to_right_edge;
    zd->mb_to_top_edge     = xd->mb_to_top_edge   ;
    zd->mb_to_bottom_edge  = xd->mb_to_bottom_edge;
    zd->gf_active_ptr     = xd->gf_active_ptr;
    zd->frames_since_golden       = xd->frames_since_golden;
    zd->frames_till_alt_ref_frame   = xd->frames_till_alt_ref_frame;
    */
    zd->subpixel_predict = xd->subpixel_predict;
    zd->subpixel_predict8x4 = xd->subpixel_predict8x4;
    zd->subpixel_predict8x8 = xd->subpixel_predict8x8;
    zd->subpixel_predict16x16 = xd->subpixel_predict16x16;
    zd->segmentation_enabled = xd->segmentation_enabled;
    zd->mb_segment_abs_delta = xd->mb_segment_abs_delta;
    memcpy(zd->segment_feature_data, xd->segment_feature_data,
           sizeof(xd->segment_feature_data));

    memcpy(zd->dequant_y1_dc, xd->dequant_y1_dc, sizeof(xd->dequant_y1_dc));
    memcpy(zd->dequant_y1, xd->dequant_y1, sizeof(xd->dequant_y1));
    memcpy(zd->dequant_y2, xd->dequant_y2, sizeof(xd->dequant_y2));
    memcpy(zd->dequant_uv, xd->dequant_uv, sizeof(xd->dequant_uv));

#if 1
    /*TODO:  Remove dequant from BLOCKD.  This is a temporary solution until
     * the quantizer code uses a passed in pointer to the dequant constants.
     * This will also require modifications to the x86 and neon assembly.
     * */
    for (i = 0; i < 16; ++i) zd->block[i].dequant = zd->dequant_y1;
    for (i = 16; i < 24; ++i) zd->block[i].dequant = zd->dequant_uv;
    zd->block[24].dequant = zd->dequant_y2;
#endif

    memcpy(z->rd_threshes, x->rd_threshes, sizeof(x->rd_threshes));
    memcpy(z->rd_thresh_mult, x->rd_thresh_mult, sizeof(x->rd_thresh_mult));

    z->zbin_over_quant = x->zbin_over_quant;
    z->zbin_mode_boost_enabled = x->zbin_mode_boost_enabled;
    z->zbin_mode_boost = x->zbin_mode_boost;

    memset(z->error_bins, 0, sizeof(z->error_bins));
  }
}

void vp8cx_init_mbrthread_data(VP8_COMP *cpi, MACROBLOCK *x,
                               MB_ROW_COMP *mbr_ei, int count) {
  VP8_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  int i;

  for (i = 0; i < count; ++i) {
    MACROBLOCK *mb = &mbr_ei[i].mb;
    MACROBLOCKD *mbd = &mb->e_mbd;

    mbd->subpixel_predict = xd->subpixel_predict;
    mbd->subpixel_predict8x4 = xd->subpixel_predict8x4;
    mbd->subpixel_predict8x8 = xd->subpixel_predict8x8;
    mbd->subpixel_predict16x16 = xd->subpixel_predict16x16;
    mb->gf_active_ptr = x->gf_active_ptr;

    memset(mbr_ei[i].segment_counts, 0, sizeof(mbr_ei[i].segment_counts));
    mbr_ei[i].totalrate = 0;

    mb->partition_info = x->pi + x->e_mbd.mode_info_stride * (i + 1);

    mbd->frame_type = cm->frame_type;

    mb->src = *cpi->Source;
    mbd->pre = cm->yv12_fb[cm->lst_fb_idx];
    mbd->dst = cm->yv12_fb[cm->new_fb_idx];

    mb->src.y_buffer += 16 * x->src.y_stride * (i + 1);
    mb->src.u_buffer += 8 * x->src.uv_stride * (i + 1);
    mb->src.v_buffer += 8 * x->src.uv_stride * (i + 1);

    vp8_build_block_offsets(mb);

    mbd->left_context = &cm->left_context;
    mb->mvc = cm->fc.mvc;

    setup_mbby_copy(&mbr_ei[i].mb, x);

    mbd->fullpixel_mask = ~0;
    if (cm->full_pixel) mbd->fullpixel_mask = ~7;

    vp8_zero(mb->coef_counts);
    vp8_zero(x->ymode_count);
    mb->skip_true_count = 0;
    vp8_zero(mb->MVcount);
    mb->prediction_error = 0;
    mb->intra_error = 0;
    vp8_zero(mb->count_mb_ref_frame_usage);
    mb->mbs_tested_so_far = 0;
    mb->mbs_zero_last_dot_suppress = 0;
  }
}

int vp8cx_create_encoder_threads(VP8_COMP *cpi) {
  const VP8_COMMON *cm = &cpi->common;
  int th_count = 0;

  if (cm->processor_core_count > 1 && cpi->oxcf.multi_threaded > 1) {
    th_count = cpi->oxcf.multi_threaded - 1;

    /* don't allocate more threads than cores available */
    if (cpi->oxcf.multi_threaded > cm->processor_core_count) {
      th_count = cm->processor_core_count - 1;
    }

    /* we have th_count + 1 (main) threads processing one row each */
    /* no point to have more threads than the sync range allows */
    if (th_count > ((cm->mb_cols / cpi->mt_sync_range) - 1)) {
      th_count = (cm->mb_cols / cpi->mt_sync_range) - 1;
    }
  }
  if (th_count == cpi->encoding_thread_count) return 0;

  vp8cx_remove_encoder_threads(cpi);
  if (th_count != 0) {
    int ithread;
    int rc = 0;

    CHECK_MEM_ERROR(&cpi->common.error, cpi->h_encoding_thread,
                    vpx_malloc(sizeof(pthread_t) * th_count));
    CHECK_MEM_ERROR(&cpi->common.error, cpi->h_event_start_encoding,
                    vpx_malloc(sizeof(vp8_sem_t) * th_count));
    CHECK_MEM_ERROR(&cpi->common.error, cpi->h_event_end_encoding,
                    vpx_malloc(sizeof(vp8_sem_t) * th_count));
    CHECK_MEM_ERROR(&cpi->common.error, cpi->mb_row_ei,
                    vpx_memalign(32, sizeof(MB_ROW_COMP) * th_count));
    memset(cpi->mb_row_ei, 0, sizeof(MB_ROW_COMP) * th_count);
    CHECK_MEM_ERROR(&cpi->common.error, cpi->en_thread_data,
                    vpx_malloc(sizeof(ENCODETHREAD_DATA) * th_count));

    vpx_atomic_store_release(&cpi->b_multi_threaded, 1);
    cpi->encoding_thread_count = th_count;

    /*
    printf("[VP8:] multi_threaded encoding is enabled with %d threads\n\n",
           (cpi->encoding_thread_count +1));
    */

    for (ithread = 0; ithread < th_count; ++ithread) {
      ENCODETHREAD_DATA *ethd = &cpi->en_thread_data[ithread];

      /* Setup block ptrs and offsets */
      vp8_setup_block_ptrs(&cpi->mb_row_ei[ithread].mb);
      vp8_setup_block_dptrs(&cpi->mb_row_ei[ithread].mb.e_mbd);

      vp8_sem_init(&cpi->h_event_start_encoding[ithread], 0, 0);
      vp8_sem_init(&cpi->h_event_end_encoding[ithread], 0, 0);

      ethd->ithread = ithread;
      ethd->ptr1 = (void *)cpi;
      ethd->ptr2 = (void *)&cpi->mb_row_ei[ithread];

      rc = pthread_create(&cpi->h_encoding_thread[ithread], 0,
                          thread_encoding_proc, ethd);
      if (rc) break;
    }

    if (rc) {
      /* shutdown other threads */
      vpx_atomic_store_release(&cpi->b_multi_threaded, 0);
      for (--ithread; ithread >= 0; ithread--) {
        vp8_sem_post(&cpi->h_event_start_encoding[ithread]);
        vp8_sem_post(&cpi->h_event_end_encoding[ithread]);
        pthread_join(cpi->h_encoding_thread[ithread], 0);
        vp8_sem_destroy(&cpi->h_event_start_encoding[ithread]);
        vp8_sem_destroy(&cpi->h_event_end_encoding[ithread]);
      }

      /* free thread related resources */
      vpx_free(cpi->h_event_start_encoding);
      cpi->h_event_start_encoding = NULL;
      vpx_free(cpi->h_event_end_encoding);
      cpi->h_event_end_encoding = NULL;
      vpx_free(cpi->h_encoding_thread);
      cpi->h_encoding_thread = NULL;
      vpx_free(cpi->mb_row_ei);
      cpi->mb_row_ei = NULL;
      vpx_free(cpi->en_thread_data);
      cpi->en_thread_data = NULL;
      cpi->encoding_thread_count = 0;

      return -1;
    }

    {
      LPFTHREAD_DATA *lpfthd = &cpi->lpf_thread_data;

      vp8_sem_init(&cpi->h_event_start_lpf, 0, 0);
      vp8_sem_init(&cpi->h_event_end_lpf, 0, 0);

      lpfthd->ptr1 = (void *)cpi;
      rc = pthread_create(&cpi->h_filter_thread, 0, thread_loopfilter, lpfthd);

      if (rc) {
        /* shutdown other threads */
        vpx_atomic_store_release(&cpi->b_multi_threaded, 0);
        for (--ithread; ithread >= 0; ithread--) {
          vp8_sem_post(&cpi->h_event_start_encoding[ithread]);
          vp8_sem_post(&cpi->h_event_end_encoding[ithread]);
          pthread_join(cpi->h_encoding_thread[ithread], 0);
          vp8_sem_destroy(&cpi->h_event_start_encoding[ithread]);
          vp8_sem_destroy(&cpi->h_event_end_encoding[ithread]);
        }
        vp8_sem_destroy(&cpi->h_event_end_lpf);
        vp8_sem_destroy(&cpi->h_event_start_lpf);

        /* free thread related resources */
        vpx_free(cpi->h_event_start_encoding);
        cpi->h_event_start_encoding = NULL;
        vpx_free(cpi->h_event_end_encoding);
        cpi->h_event_end_encoding = NULL;
        vpx_free(cpi->h_encoding_thread);
        cpi->h_encoding_thread = NULL;
        vpx_free(cpi->mb_row_ei);
        cpi->mb_row_ei = NULL;
        vpx_free(cpi->en_thread_data);
        cpi->en_thread_data = NULL;
        cpi->encoding_thread_count = 0;

        return -2;
      }
    }
  }
  return 0;
}

void vp8cx_remove_encoder_threads(VP8_COMP *cpi) {
  if (vpx_atomic_load_acquire(&cpi->b_multi_threaded)) {
    /* shutdown other threads */
    vpx_atomic_store_release(&cpi->b_multi_threaded, 0);
    {
      int i;

      for (i = 0; i < cpi->encoding_thread_count; ++i) {
        vp8_sem_post(&cpi->h_event_start_encoding[i]);
        vp8_sem_post(&cpi->h_event_end_encoding[i]);

        pthread_join(cpi->h_encoding_thread[i], 0);

        vp8_sem_destroy(&cpi->h_event_start_encoding[i]);
        vp8_sem_destroy(&cpi->h_event_end_encoding[i]);
      }

      vp8_sem_post(&cpi->h_event_start_lpf);
      pthread_join(cpi->h_filter_thread, 0);
    }

    vp8_sem_destroy(&cpi->h_event_end_lpf);
    vp8_sem_destroy(&cpi->h_event_start_lpf);
    cpi->b_lpf_running = 0;

    /* free thread related resources */
    vpx_free(cpi->mt_current_mb_col);
    cpi->mt_current_mb_col = NULL;
    cpi->mt_current_mb_col_size = 0;
    vpx_free(cpi->h_event_start_encoding);
    cpi->h_event_start_encoding = NULL;
    vpx_free(cpi->h_event_end_encoding);
    cpi->h_event_end_encoding = NULL;
    vpx_free(cpi->h_encoding_thread);
    cpi->h_encoding_thread = NULL;
    vpx_free(cpi->mb_row_ei);
    cpi->mb_row_ei = NULL;
    vpx_free(cpi->en_thread_data);
    cpi->en_thread_data = NULL;
    cpi->encoding_thread_count = 0;
  }
}
#endif
