/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "vpx_config.h"
#include "vp8_rtcd.h"
#if !defined(WIN32) && CONFIG_OS_SUPPORT == 1
# include <unistd.h>
#endif
#include "onyxd_int.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8/common/threading.h"

#include "vp8/common/loopfilter.h"
#include "vp8/common/extend.h"
#include "vpx_ports/vpx_timer.h"
#include "detokenize.h"
#include "vp8/common/reconintra4x4.h"
#include "vp8/common/reconinter.h"
#include "vp8/common/reconintra.h"
#include "vp8/common/setupintrarecon.h"
#if CONFIG_ERROR_CONCEALMENT
#include "error_concealment.h"
#endif

#define CALLOC_ARRAY(p, n) CHECK_MEM_ERROR((p), vpx_calloc(sizeof(*(p)), (n)))
#define CALLOC_ARRAY_ALIGNED(p, n, algn) do {                      \
  CHECK_MEM_ERROR((p), vpx_memalign((algn), sizeof(*(p)) * (n)));  \
  memset((p), 0, (n) * sizeof(*(p)));                              \
} while (0)


void vp8_mb_init_dequantizer(VP8D_COMP *pbi, MACROBLOCKD *xd);

static void setup_decoding_thread_data(VP8D_COMP *pbi, MACROBLOCKD *xd, MB_ROW_DEC *mbrd, int count)
{
    VP8_COMMON *const pc = & pbi->common;
    int i;

    for (i = 0; i < count; i++)
    {
        MACROBLOCKD *mbd = &mbrd[i].mbd;
        mbd->subpixel_predict        = xd->subpixel_predict;
        mbd->subpixel_predict8x4     = xd->subpixel_predict8x4;
        mbd->subpixel_predict8x8     = xd->subpixel_predict8x8;
        mbd->subpixel_predict16x16   = xd->subpixel_predict16x16;

        mbd->frame_type = pc->frame_type;
        mbd->pre = xd->pre;
        mbd->dst = xd->dst;

        mbd->segmentation_enabled    = xd->segmentation_enabled;
        mbd->mb_segement_abs_delta     = xd->mb_segement_abs_delta;
        memcpy(mbd->segment_feature_data, xd->segment_feature_data, sizeof(xd->segment_feature_data));

        /*signed char ref_lf_deltas[MAX_REF_LF_DELTAS];*/
        memcpy(mbd->ref_lf_deltas, xd->ref_lf_deltas, sizeof(xd->ref_lf_deltas));
        /*signed char mode_lf_deltas[MAX_MODE_LF_DELTAS];*/
        memcpy(mbd->mode_lf_deltas, xd->mode_lf_deltas, sizeof(xd->mode_lf_deltas));
        /*unsigned char mode_ref_lf_delta_enabled;
        unsigned char mode_ref_lf_delta_update;*/
        mbd->mode_ref_lf_delta_enabled    = xd->mode_ref_lf_delta_enabled;
        mbd->mode_ref_lf_delta_update    = xd->mode_ref_lf_delta_update;

        mbd->current_bc = &pbi->mbc[0];

        memcpy(mbd->dequant_y1_dc, xd->dequant_y1_dc, sizeof(xd->dequant_y1_dc));
        memcpy(mbd->dequant_y1, xd->dequant_y1, sizeof(xd->dequant_y1));
        memcpy(mbd->dequant_y2, xd->dequant_y2, sizeof(xd->dequant_y2));
        memcpy(mbd->dequant_uv, xd->dequant_uv, sizeof(xd->dequant_uv));

        mbd->fullpixel_mask = 0xffffffff;

        if (pc->full_pixel)
            mbd->fullpixel_mask = 0xfffffff8;

    }

    for (i = 0; i < pc->mb_rows; i++)
        pbi->mt_current_mb_col[i] = -1;
}

static void mt_decode_macroblock(VP8D_COMP *pbi, MACROBLOCKD *xd,
                                 unsigned int mb_idx)
{
    MB_PREDICTION_MODE mode;
    int i;
#if CONFIG_ERROR_CONCEALMENT
    int corruption_detected = 0;
#else
    (void)mb_idx;
#endif

    if (xd->mode_info_context->mbmi.mb_skip_coeff)
    {
        vp8_reset_mb_tokens_context(xd);
    }
    else if (!vp8dx_bool_error(xd->current_bc))
    {
        int eobtotal;
        eobtotal = vp8_decode_mb_tokens(pbi, xd);

        /* Special case:  Force the loopfilter to skip when eobtotal is zero */
        xd->mode_info_context->mbmi.mb_skip_coeff = (eobtotal==0);
    }

    mode = xd->mode_info_context->mbmi.mode;

    if (xd->segmentation_enabled)
        vp8_mb_init_dequantizer(pbi, xd);


#if CONFIG_ERROR_CONCEALMENT

    if(pbi->ec_active)
    {
        int throw_residual;
        /* When we have independent partitions we can apply residual even
         * though other partitions within the frame are corrupt.
         */
        throw_residual = (!pbi->independent_partitions &&
                          pbi->frame_corrupt_residual);
        throw_residual = (throw_residual || vp8dx_bool_error(xd->current_bc));

        if ((mb_idx >= pbi->mvs_corrupt_from_mb || throw_residual))
        {
            /* MB with corrupt residuals or corrupt mode/motion vectors.
             * Better to use the predictor as reconstruction.
             */
            pbi->frame_corrupt_residual = 1;
            memset(xd->qcoeff, 0, sizeof(xd->qcoeff));

            corruption_detected = 1;

            /* force idct to be skipped for B_PRED and use the
             * prediction only for reconstruction
             * */
            memset(xd->eobs, 0, 25);
        }
    }
#endif

    /* do prediction */
    if (xd->mode_info_context->mbmi.ref_frame == INTRA_FRAME)
    {
        vp8_build_intra_predictors_mbuv_s(xd,
                                          xd->recon_above[1],
                                          xd->recon_above[2],
                                          xd->recon_left[1],
                                          xd->recon_left[2],
                                          xd->recon_left_stride[1],
                                          xd->dst.u_buffer, xd->dst.v_buffer,
                                          xd->dst.uv_stride);

        if (mode != B_PRED)
        {
            vp8_build_intra_predictors_mby_s(xd,
                                                 xd->recon_above[0],
                                                 xd->recon_left[0],
                                                 xd->recon_left_stride[0],
                                                 xd->dst.y_buffer,
                                                 xd->dst.y_stride);
        }
        else
        {
            short *DQC = xd->dequant_y1;
            int dst_stride = xd->dst.y_stride;

            /* clear out residual eob info */
            if(xd->mode_info_context->mbmi.mb_skip_coeff)
                memset(xd->eobs, 0, 25);

            intra_prediction_down_copy(xd, xd->recon_above[0] + 16);

            for (i = 0; i < 16; i++)
            {
                BLOCKD *b = &xd->block[i];
                unsigned char *dst = xd->dst.y_buffer + b->offset;
                B_PREDICTION_MODE b_mode =
                    xd->mode_info_context->bmi[i].as_mode;
                unsigned char *Above;
                unsigned char *yleft;
                int left_stride;
                unsigned char top_left;

                /*Caution: For some b_mode, it needs 8 pixels (4 above + 4 above-right).*/
                if (i < 4 && pbi->common.filter_level)
                    Above = xd->recon_above[0] + b->offset;
                else
                    Above = dst - dst_stride;

                if (i%4==0 && pbi->common.filter_level)
                {
                    yleft = xd->recon_left[0] + i;
                    left_stride = 1;
                }
                else
                {
                    yleft = dst - 1;
                    left_stride = dst_stride;
                }

                if ((i==4 || i==8 || i==12) && pbi->common.filter_level)
                    top_left = *(xd->recon_left[0] + i - 1);
                else
                    top_left = Above[-1];

                vp8_intra4x4_predict(Above, yleft, left_stride,
                                     b_mode, dst, dst_stride, top_left);

                if (xd->eobs[i] )
                {
                    if (xd->eobs[i] > 1)
                    {
                        vp8_dequant_idct_add(b->qcoeff, DQC, dst, dst_stride);
                    }
                    else
                    {
                        vp8_dc_only_idct_add(b->qcoeff[0] * DQC[0],
                                             dst, dst_stride, dst, dst_stride);
                        memset(b->qcoeff, 0, 2 * sizeof(b->qcoeff[0]));
                    }
                }
            }
        }
    }
    else
    {
        vp8_build_inter_predictors_mb(xd);
    }


#if CONFIG_ERROR_CONCEALMENT
    if (corruption_detected)
    {
        return;
    }
#endif

    if(!xd->mode_info_context->mbmi.mb_skip_coeff)
    {
        /* dequantization and idct */
        if (mode != B_PRED)
        {
            short *DQC = xd->dequant_y1;

            if (mode != SPLITMV)
            {
                BLOCKD *b = &xd->block[24];

                /* do 2nd order transform on the dc block */
                if (xd->eobs[24] > 1)
                {
                    vp8_dequantize_b(b, xd->dequant_y2);

                    vp8_short_inv_walsh4x4(&b->dqcoeff[0],
                        xd->qcoeff);
                    memset(b->qcoeff, 0, 16 * sizeof(b->qcoeff[0]));
                }
                else
                {
                    b->dqcoeff[0] = b->qcoeff[0] * xd->dequant_y2[0];
                    vp8_short_inv_walsh4x4_1(&b->dqcoeff[0],
                        xd->qcoeff);
                    memset(b->qcoeff, 0, 2 * sizeof(b->qcoeff[0]));
                }

                /* override the dc dequant constant in order to preserve the
                 * dc components
                 */
                DQC = xd->dequant_y1_dc;
            }

            vp8_dequant_idct_add_y_block
                            (xd->qcoeff, DQC,
                             xd->dst.y_buffer,
                             xd->dst.y_stride, xd->eobs);
        }

        vp8_dequant_idct_add_uv_block
                        (xd->qcoeff+16*16, xd->dequant_uv,
                         xd->dst.u_buffer, xd->dst.v_buffer,
                         xd->dst.uv_stride, xd->eobs+16);
    }
}

static void mt_decode_mb_rows(VP8D_COMP *pbi, MACROBLOCKD *xd, int start_mb_row)
{
    const int *last_row_current_mb_col;
    int *current_mb_col;
    int mb_row;
    VP8_COMMON *pc = &pbi->common;
    const int nsync = pbi->sync_range;
    const int first_row_no_sync_above = pc->mb_cols + nsync;
    int num_part = 1 << pbi->common.multi_token_partition;
    int last_mb_row = start_mb_row;

    YV12_BUFFER_CONFIG *yv12_fb_new = pbi->dec_fb_ref[INTRA_FRAME];
    YV12_BUFFER_CONFIG *yv12_fb_lst = pbi->dec_fb_ref[LAST_FRAME];

    int recon_y_stride = yv12_fb_new->y_stride;
    int recon_uv_stride = yv12_fb_new->uv_stride;

    unsigned char *ref_buffer[MAX_REF_FRAMES][3];
    unsigned char *dst_buffer[3];
    int i;
    int ref_fb_corrupted[MAX_REF_FRAMES];

    ref_fb_corrupted[INTRA_FRAME] = 0;

    for(i = 1; i < MAX_REF_FRAMES; i++)
    {
        YV12_BUFFER_CONFIG *this_fb = pbi->dec_fb_ref[i];

        ref_buffer[i][0] = this_fb->y_buffer;
        ref_buffer[i][1] = this_fb->u_buffer;
        ref_buffer[i][2] = this_fb->v_buffer;

        ref_fb_corrupted[i] = this_fb->corrupted;
    }

    dst_buffer[0] = yv12_fb_new->y_buffer;
    dst_buffer[1] = yv12_fb_new->u_buffer;
    dst_buffer[2] = yv12_fb_new->v_buffer;

    xd->up_available = (start_mb_row != 0);

    xd->mode_info_context = pc->mi + pc->mode_info_stride * start_mb_row;
    xd->mode_info_stride = pc->mode_info_stride;

    for (mb_row = start_mb_row; mb_row < pc->mb_rows; mb_row += (pbi->decoding_thread_count + 1))
    {
       int recon_yoffset, recon_uvoffset;
       int mb_col;
       int filter_level;
       loop_filter_info_n *lfi_n = &pc->lf_info;

       /* save last row processed by this thread */
       last_mb_row = mb_row;
       /* select bool coder for current partition */
       xd->current_bc =  &pbi->mbc[mb_row%num_part];

       if (mb_row > 0)
           last_row_current_mb_col = &pbi->mt_current_mb_col[mb_row -1];
       else
           last_row_current_mb_col = &first_row_no_sync_above;

       current_mb_col = &pbi->mt_current_mb_col[mb_row];

       recon_yoffset = mb_row * recon_y_stride * 16;
       recon_uvoffset = mb_row * recon_uv_stride * 8;

       /* reset contexts */
       xd->above_context = pc->above_context;
       memset(xd->left_context, 0, sizeof(ENTROPY_CONTEXT_PLANES));

       xd->left_available = 0;

       xd->mb_to_top_edge = -((mb_row * 16)) << 3;
       xd->mb_to_bottom_edge = ((pc->mb_rows - 1 - mb_row) * 16) << 3;

       if (pbi->common.filter_level)
       {
          xd->recon_above[0] = pbi->mt_yabove_row[mb_row] + 0*16 +32;
          xd->recon_above[1] = pbi->mt_uabove_row[mb_row] + 0*8 +16;
          xd->recon_above[2] = pbi->mt_vabove_row[mb_row] + 0*8 +16;

          xd->recon_left[0] = pbi->mt_yleft_col[mb_row];
          xd->recon_left[1] = pbi->mt_uleft_col[mb_row];
          xd->recon_left[2] = pbi->mt_vleft_col[mb_row];

          /* TODO: move to outside row loop */
          xd->recon_left_stride[0] = 1;
          xd->recon_left_stride[1] = 1;
       }
       else
       {
          xd->recon_above[0] = dst_buffer[0] + recon_yoffset;
          xd->recon_above[1] = dst_buffer[1] + recon_uvoffset;
          xd->recon_above[2] = dst_buffer[2] + recon_uvoffset;

          xd->recon_left[0] = xd->recon_above[0] - 1;
          xd->recon_left[1] = xd->recon_above[1] - 1;
          xd->recon_left[2] = xd->recon_above[2] - 1;

          xd->recon_above[0] -= xd->dst.y_stride;
          xd->recon_above[1] -= xd->dst.uv_stride;
          xd->recon_above[2] -= xd->dst.uv_stride;

          /* TODO: move to outside row loop */
          xd->recon_left_stride[0] = xd->dst.y_stride;
          xd->recon_left_stride[1] = xd->dst.uv_stride;

          setup_intra_recon_left(xd->recon_left[0], xd->recon_left[1],
                                 xd->recon_left[2], xd->dst.y_stride,
                                 xd->dst.uv_stride);
       }

       for (mb_col = 0; mb_col < pc->mb_cols; mb_col++) {
           if (((mb_col - 1) % nsync) == 0) {
               pthread_mutex_t *mutex = &pbi->pmutex[mb_row];
               protected_write(mutex, current_mb_col, mb_col - 1);
           }

           if (mb_row && !(mb_col & (nsync - 1))) {
               pthread_mutex_t *mutex = &pbi->pmutex[mb_row-1];
               sync_read(mutex, mb_col, last_row_current_mb_col, nsync);
           }

           /* Distance of MB to the various image edges.
            * These are specified to 8th pel as they are always
            * compared to values that are in 1/8th pel units.
            */
           xd->mb_to_left_edge = -((mb_col * 16) << 3);
           xd->mb_to_right_edge = ((pc->mb_cols - 1 - mb_col) * 16) << 3;

    #if CONFIG_ERROR_CONCEALMENT
           {
               int corrupt_residual =
                           (!pbi->independent_partitions &&
                           pbi->frame_corrupt_residual) ||
                           vp8dx_bool_error(xd->current_bc);
               if (pbi->ec_active &&
                   (xd->mode_info_context->mbmi.ref_frame ==
                                                    INTRA_FRAME) &&
                   corrupt_residual)
               {
                   /* We have an intra block with corrupt
                    * coefficients, better to conceal with an inter
                    * block.
                    * Interpolate MVs from neighboring MBs
                    *
                    * Note that for the first mb with corrupt
                    * residual in a frame, we might not discover
                    * that before decoding the residual. That
                    * happens after this check, and therefore no
                    * inter concealment will be done.
                    */
                   vp8_interpolate_motion(xd,
                                          mb_row, mb_col,
                                          pc->mb_rows, pc->mb_cols);
               }
           }
    #endif


           xd->dst.y_buffer = dst_buffer[0] + recon_yoffset;
           xd->dst.u_buffer = dst_buffer[1] + recon_uvoffset;
           xd->dst.v_buffer = dst_buffer[2] + recon_uvoffset;

           xd->pre.y_buffer = ref_buffer[xd->mode_info_context->mbmi.ref_frame][0] + recon_yoffset;
           xd->pre.u_buffer = ref_buffer[xd->mode_info_context->mbmi.ref_frame][1] + recon_uvoffset;
           xd->pre.v_buffer = ref_buffer[xd->mode_info_context->mbmi.ref_frame][2] + recon_uvoffset;

           /* propagate errors from reference frames */
           xd->corrupted |= ref_fb_corrupted[xd->mode_info_context->mbmi.ref_frame];

           mt_decode_macroblock(pbi, xd, 0);

           xd->left_available = 1;

           /* check if the boolean decoder has suffered an error */
           xd->corrupted |= vp8dx_bool_error(xd->current_bc);

           xd->recon_above[0] += 16;
           xd->recon_above[1] += 8;
           xd->recon_above[2] += 8;

           if (!pbi->common.filter_level)
           {
              xd->recon_left[0] += 16;
              xd->recon_left[1] += 8;
              xd->recon_left[2] += 8;
           }

           if (pbi->common.filter_level)
           {
               int skip_lf = (xd->mode_info_context->mbmi.mode != B_PRED &&
                               xd->mode_info_context->mbmi.mode != SPLITMV &&
                               xd->mode_info_context->mbmi.mb_skip_coeff);

               const int mode_index = lfi_n->mode_lf_lut[xd->mode_info_context->mbmi.mode];
               const int seg = xd->mode_info_context->mbmi.segment_id;
               const int ref_frame = xd->mode_info_context->mbmi.ref_frame;

               filter_level = lfi_n->lvl[seg][ref_frame][mode_index];

               if( mb_row != pc->mb_rows-1 )
               {
                   /* Save decoded MB last row data for next-row decoding */
                   memcpy((pbi->mt_yabove_row[mb_row + 1] + 32 + mb_col*16), (xd->dst.y_buffer + 15 * recon_y_stride), 16);
                   memcpy((pbi->mt_uabove_row[mb_row + 1] + 16 + mb_col*8), (xd->dst.u_buffer + 7 * recon_uv_stride), 8);
                   memcpy((pbi->mt_vabove_row[mb_row + 1] + 16 + mb_col*8), (xd->dst.v_buffer + 7 * recon_uv_stride), 8);
               }

               /* save left_col for next MB decoding */
               if(mb_col != pc->mb_cols-1)
               {
                   MODE_INFO *next = xd->mode_info_context +1;

                   if (next->mbmi.ref_frame == INTRA_FRAME)
                   {
                       for (i = 0; i < 16; i++)
                           pbi->mt_yleft_col[mb_row][i] = xd->dst.y_buffer [i* recon_y_stride + 15];
                       for (i = 0; i < 8; i++)
                       {
                           pbi->mt_uleft_col[mb_row][i] = xd->dst.u_buffer [i* recon_uv_stride + 7];
                           pbi->mt_vleft_col[mb_row][i] = xd->dst.v_buffer [i* recon_uv_stride + 7];
                       }
                   }
               }

               /* loopfilter on this macroblock. */
               if (filter_level)
               {
                   if(pc->filter_type == NORMAL_LOOPFILTER)
                   {
                       loop_filter_info lfi;
                       FRAME_TYPE frame_type = pc->frame_type;
                       const int hev_index = lfi_n->hev_thr_lut[frame_type][filter_level];
                       lfi.mblim = lfi_n->mblim[filter_level];
                       lfi.blim = lfi_n->blim[filter_level];
                       lfi.lim = lfi_n->lim[filter_level];
                       lfi.hev_thr = lfi_n->hev_thr[hev_index];

                       if (mb_col > 0)
                           vp8_loop_filter_mbv
                           (xd->dst.y_buffer, xd->dst.u_buffer, xd->dst.v_buffer, recon_y_stride, recon_uv_stride, &lfi);

                       if (!skip_lf)
                           vp8_loop_filter_bv
                           (xd->dst.y_buffer, xd->dst.u_buffer, xd->dst.v_buffer, recon_y_stride, recon_uv_stride, &lfi);

                       /* don't apply across umv border */
                       if (mb_row > 0)
                           vp8_loop_filter_mbh
                           (xd->dst.y_buffer, xd->dst.u_buffer, xd->dst.v_buffer, recon_y_stride, recon_uv_stride, &lfi);

                       if (!skip_lf)
                           vp8_loop_filter_bh
                           (xd->dst.y_buffer, xd->dst.u_buffer, xd->dst.v_buffer,  recon_y_stride, recon_uv_stride, &lfi);
                   }
                   else
                   {
                       if (mb_col > 0)
                           vp8_loop_filter_simple_mbv
                           (xd->dst.y_buffer, recon_y_stride, lfi_n->mblim[filter_level]);

                       if (!skip_lf)
                           vp8_loop_filter_simple_bv
                           (xd->dst.y_buffer, recon_y_stride, lfi_n->blim[filter_level]);

                       /* don't apply across umv border */
                       if (mb_row > 0)
                           vp8_loop_filter_simple_mbh
                           (xd->dst.y_buffer, recon_y_stride, lfi_n->mblim[filter_level]);

                       if (!skip_lf)
                           vp8_loop_filter_simple_bh
                           (xd->dst.y_buffer, recon_y_stride, lfi_n->blim[filter_level]);
                   }
               }

           }

           recon_yoffset += 16;
           recon_uvoffset += 8;

           ++xd->mode_info_context;  /* next mb */

           xd->above_context++;
       }

       /* adjust to the next row of mbs */
       if (pbi->common.filter_level)
       {
           if(mb_row != pc->mb_rows-1)
           {
               int lasty = yv12_fb_lst->y_width + VP8BORDERINPIXELS;
               int lastuv = (yv12_fb_lst->y_width>>1) + (VP8BORDERINPIXELS>>1);

               for (i = 0; i < 4; i++)
               {
                   pbi->mt_yabove_row[mb_row +1][lasty + i] = pbi->mt_yabove_row[mb_row +1][lasty -1];
                   pbi->mt_uabove_row[mb_row +1][lastuv + i] = pbi->mt_uabove_row[mb_row +1][lastuv -1];
                   pbi->mt_vabove_row[mb_row +1][lastuv + i] = pbi->mt_vabove_row[mb_row +1][lastuv -1];
               }
           }
       }
       else
           vp8_extend_mb_row(yv12_fb_new, xd->dst.y_buffer + 16,
                             xd->dst.u_buffer + 8, xd->dst.v_buffer + 8);

       /* last MB of row is ready just after extension is done */
       protected_write(&pbi->pmutex[mb_row], current_mb_col, mb_col + nsync);

       ++xd->mode_info_context;      /* skip prediction column */
       xd->up_available = 1;

       /* since we have multithread */
       xd->mode_info_context += xd->mode_info_stride * pbi->decoding_thread_count;
    }

    /* signal end of frame decoding if this thread processed the last mb_row */
    if (last_mb_row == (pc->mb_rows - 1))
        sem_post(&pbi->h_event_end_decoding);

}


static THREAD_FUNCTION thread_decoding_proc(void *p_data)
{
    int ithread = ((DECODETHREAD_DATA *)p_data)->ithread;
    VP8D_COMP *pbi = (VP8D_COMP *)(((DECODETHREAD_DATA *)p_data)->ptr1);
    MB_ROW_DEC *mbrd = (MB_ROW_DEC *)(((DECODETHREAD_DATA *)p_data)->ptr2);
    ENTROPY_CONTEXT_PLANES mb_row_left_context;

    while (1)
    {
        if (protected_read(&pbi->mt_mutex, &pbi->b_multithreaded_rd) == 0)
            break;

        if (sem_wait(&pbi->h_event_start_decoding[ithread]) == 0)
        {
            if (protected_read(&pbi->mt_mutex, &pbi->b_multithreaded_rd) == 0)
                break;
            else
            {
                MACROBLOCKD *xd = &mbrd->mbd;
                xd->left_context = &mb_row_left_context;

                mt_decode_mb_rows(pbi, xd, ithread+1);
            }
        }
    }

    return 0 ;
}


void vp8_decoder_create_threads(VP8D_COMP *pbi)
{
    int core_count = 0;
    unsigned int ithread;

    pbi->b_multithreaded_rd = 0;
    pbi->allocated_decoding_thread_count = 0;
    pthread_mutex_init(&pbi->mt_mutex, NULL);

    /* limit decoding threads to the max number of token partitions */
    core_count = (pbi->max_threads > 8) ? 8 : pbi->max_threads;

    /* limit decoding threads to the available cores */
    if (core_count > pbi->common.processor_core_count)
        core_count = pbi->common.processor_core_count;

    if (core_count > 1)
    {
        pbi->b_multithreaded_rd = 1;
        pbi->decoding_thread_count = core_count - 1;

        CALLOC_ARRAY(pbi->h_decoding_thread, pbi->decoding_thread_count);
        CALLOC_ARRAY(pbi->h_event_start_decoding, pbi->decoding_thread_count);
        CALLOC_ARRAY_ALIGNED(pbi->mb_row_di, pbi->decoding_thread_count, 32);
        CALLOC_ARRAY(pbi->de_thread_data, pbi->decoding_thread_count);

        for (ithread = 0; ithread < pbi->decoding_thread_count; ithread++)
        {
            sem_init(&pbi->h_event_start_decoding[ithread], 0, 0);

            vp8_setup_block_dptrs(&pbi->mb_row_di[ithread].mbd);

            pbi->de_thread_data[ithread].ithread  = ithread;
            pbi->de_thread_data[ithread].ptr1     = (void *)pbi;
            pbi->de_thread_data[ithread].ptr2     = (void *) &pbi->mb_row_di[ithread];

            pthread_create(&pbi->h_decoding_thread[ithread], 0, thread_decoding_proc, (&pbi->de_thread_data[ithread]));
        }

        sem_init(&pbi->h_event_end_decoding, 0, 0);

        pbi->allocated_decoding_thread_count = pbi->decoding_thread_count;
    }
}


void vp8mt_de_alloc_temp_buffers(VP8D_COMP *pbi, int mb_rows)
{
    int i;

    if (protected_read(&pbi->mt_mutex, &pbi->b_multithreaded_rd))
    {
        /* De-allocate mutex */
        if (pbi->pmutex != NULL) {
            for (i = 0; i < mb_rows; i++) {
                pthread_mutex_destroy(&pbi->pmutex[i]);
            }
            vpx_free(pbi->pmutex);
            pbi->pmutex = NULL;
        }

            vpx_free(pbi->mt_current_mb_col);
            pbi->mt_current_mb_col = NULL ;

        /* Free above_row buffers. */
        if (pbi->mt_yabove_row)
        {
            for (i=0; i< mb_rows; i++)
            {
                    vpx_free(pbi->mt_yabove_row[i]);
                    pbi->mt_yabove_row[i] = NULL ;
            }
            vpx_free(pbi->mt_yabove_row);
            pbi->mt_yabove_row = NULL ;
        }

        if (pbi->mt_uabove_row)
        {
            for (i=0; i< mb_rows; i++)
            {
                    vpx_free(pbi->mt_uabove_row[i]);
                    pbi->mt_uabove_row[i] = NULL ;
            }
            vpx_free(pbi->mt_uabove_row);
            pbi->mt_uabove_row = NULL ;
        }

        if (pbi->mt_vabove_row)
        {
            for (i=0; i< mb_rows; i++)
            {
                    vpx_free(pbi->mt_vabove_row[i]);
                    pbi->mt_vabove_row[i] = NULL ;
            }
            vpx_free(pbi->mt_vabove_row);
            pbi->mt_vabove_row = NULL ;
        }

        /* Free left_col buffers. */
        if (pbi->mt_yleft_col)
        {
            for (i=0; i< mb_rows; i++)
            {
                    vpx_free(pbi->mt_yleft_col[i]);
                    pbi->mt_yleft_col[i] = NULL ;
            }
            vpx_free(pbi->mt_yleft_col);
            pbi->mt_yleft_col = NULL ;
        }

        if (pbi->mt_uleft_col)
        {
            for (i=0; i< mb_rows; i++)
            {
                    vpx_free(pbi->mt_uleft_col[i]);
                    pbi->mt_uleft_col[i] = NULL ;
            }
            vpx_free(pbi->mt_uleft_col);
            pbi->mt_uleft_col = NULL ;
        }

        if (pbi->mt_vleft_col)
        {
            for (i=0; i< mb_rows; i++)
            {
                    vpx_free(pbi->mt_vleft_col[i]);
                    pbi->mt_vleft_col[i] = NULL ;
            }
            vpx_free(pbi->mt_vleft_col);
            pbi->mt_vleft_col = NULL ;
        }
    }
}


void vp8mt_alloc_temp_buffers(VP8D_COMP *pbi, int width, int prev_mb_rows)
{
    VP8_COMMON *const pc = & pbi->common;
    int i;
    int uv_width;

    if (protected_read(&pbi->mt_mutex, &pbi->b_multithreaded_rd))
    {
        vp8mt_de_alloc_temp_buffers(pbi, prev_mb_rows);

        /* our internal buffers are always multiples of 16 */
        if ((width & 0xf) != 0)
            width += 16 - (width & 0xf);

        if (width < 640) pbi->sync_range = 1;
        else if (width <= 1280) pbi->sync_range = 8;
        else if (width <= 2560) pbi->sync_range =16;
        else pbi->sync_range = 32;

        uv_width = width >>1;

        /* Allocate mutex */
        CHECK_MEM_ERROR(pbi->pmutex, vpx_malloc(sizeof(*pbi->pmutex) *
                                                pc->mb_rows));
        if (pbi->pmutex) {
            for (i = 0; i < pc->mb_rows; i++) {
                pthread_mutex_init(&pbi->pmutex[i], NULL);
            }
        }

        /* Allocate an int for each mb row. */
        CALLOC_ARRAY(pbi->mt_current_mb_col, pc->mb_rows);

        /* Allocate memory for above_row buffers. */
        CALLOC_ARRAY(pbi->mt_yabove_row, pc->mb_rows);
        for (i = 0; i < pc->mb_rows; i++)
            CHECK_MEM_ERROR(pbi->mt_yabove_row[i], vpx_memalign(16,sizeof(unsigned char) * (width + (VP8BORDERINPIXELS<<1))));

        CALLOC_ARRAY(pbi->mt_uabove_row, pc->mb_rows);
        for (i = 0; i < pc->mb_rows; i++)
            CHECK_MEM_ERROR(pbi->mt_uabove_row[i], vpx_memalign(16,sizeof(unsigned char) * (uv_width + VP8BORDERINPIXELS)));

        CALLOC_ARRAY(pbi->mt_vabove_row, pc->mb_rows);
        for (i = 0; i < pc->mb_rows; i++)
            CHECK_MEM_ERROR(pbi->mt_vabove_row[i], vpx_memalign(16,sizeof(unsigned char) * (uv_width + VP8BORDERINPIXELS)));

        /* Allocate memory for left_col buffers. */
        CALLOC_ARRAY(pbi->mt_yleft_col, pc->mb_rows);
        for (i = 0; i < pc->mb_rows; i++)
            CHECK_MEM_ERROR(pbi->mt_yleft_col[i], vpx_calloc(sizeof(unsigned char) * 16, 1));

        CALLOC_ARRAY(pbi->mt_uleft_col, pc->mb_rows);
        for (i = 0; i < pc->mb_rows; i++)
            CHECK_MEM_ERROR(pbi->mt_uleft_col[i], vpx_calloc(sizeof(unsigned char) * 8, 1));

        CALLOC_ARRAY(pbi->mt_vleft_col, pc->mb_rows);
        for (i = 0; i < pc->mb_rows; i++)
            CHECK_MEM_ERROR(pbi->mt_vleft_col[i], vpx_calloc(sizeof(unsigned char) * 8, 1));
    }
}


void vp8_decoder_remove_threads(VP8D_COMP *pbi)
{
    /* shutdown MB Decoding thread; */
    if (protected_read(&pbi->mt_mutex, &pbi->b_multithreaded_rd))
    {
        int i;

        protected_write(&pbi->mt_mutex, &pbi->b_multithreaded_rd, 0);

        /* allow all threads to exit */
        for (i = 0; i < pbi->allocated_decoding_thread_count; i++)
        {
            sem_post(&pbi->h_event_start_decoding[i]);
            pthread_join(pbi->h_decoding_thread[i], NULL);
        }

        for (i = 0; i < pbi->allocated_decoding_thread_count; i++)
        {
            sem_destroy(&pbi->h_event_start_decoding[i]);
        }

        sem_destroy(&pbi->h_event_end_decoding);

            vpx_free(pbi->h_decoding_thread);
            pbi->h_decoding_thread = NULL;

            vpx_free(pbi->h_event_start_decoding);
            pbi->h_event_start_decoding = NULL;

            vpx_free(pbi->mb_row_di);
            pbi->mb_row_di = NULL ;

            vpx_free(pbi->de_thread_data);
            pbi->de_thread_data = NULL;
    }
    pthread_mutex_destroy(&pbi->mt_mutex);
}

void vp8mt_decode_mb_rows( VP8D_COMP *pbi, MACROBLOCKD *xd)
{
    VP8_COMMON *pc = &pbi->common;
    unsigned int i;
    int j;

    int filter_level = pc->filter_level;
    YV12_BUFFER_CONFIG *yv12_fb_new = pbi->dec_fb_ref[INTRA_FRAME];

    if (filter_level)
    {
        /* Set above_row buffer to 127 for decoding first MB row */
        memset(pbi->mt_yabove_row[0] + VP8BORDERINPIXELS-1, 127, yv12_fb_new->y_width + 5);
        memset(pbi->mt_uabove_row[0] + (VP8BORDERINPIXELS>>1)-1, 127, (yv12_fb_new->y_width>>1) +5);
        memset(pbi->mt_vabove_row[0] + (VP8BORDERINPIXELS>>1)-1, 127, (yv12_fb_new->y_width>>1) +5);

        for (j=1; j<pc->mb_rows; j++)
        {
            memset(pbi->mt_yabove_row[j] + VP8BORDERINPIXELS-1, (unsigned char)129, 1);
            memset(pbi->mt_uabove_row[j] + (VP8BORDERINPIXELS>>1)-1, (unsigned char)129, 1);
            memset(pbi->mt_vabove_row[j] + (VP8BORDERINPIXELS>>1)-1, (unsigned char)129, 1);
        }

        /* Set left_col to 129 initially */
        for (j=0; j<pc->mb_rows; j++)
        {
            memset(pbi->mt_yleft_col[j], (unsigned char)129, 16);
            memset(pbi->mt_uleft_col[j], (unsigned char)129, 8);
            memset(pbi->mt_vleft_col[j], (unsigned char)129, 8);
        }

        /* Initialize the loop filter for this frame. */
        vp8_loop_filter_frame_init(pc, &pbi->mb, filter_level);
    }
    else
        vp8_setup_intra_recon_top_line(yv12_fb_new);

    setup_decoding_thread_data(pbi, xd, pbi->mb_row_di, pbi->decoding_thread_count);

    for (i = 0; i < pbi->decoding_thread_count; i++)
        sem_post(&pbi->h_event_start_decoding[i]);

    mt_decode_mb_rows(pbi, xd, 0);

    sem_wait(&pbi->h_event_end_decoding);   /* add back for each frame */
}
