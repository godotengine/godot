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
#include "alloccommon.h"
#include "blockd.h"
#include "vpx_mem/vpx_mem.h"
#include "onyxc_int.h"
#include "findnearmv.h"
#include "entropymode.h"
#include "systemdependent.h"

void vp8_de_alloc_frame_buffers(VP8_COMMON *oci)
{
    int i;
    for (i = 0; i < NUM_YV12_BUFFERS; i++)
        vp8_yv12_de_alloc_frame_buffer(&oci->yv12_fb[i]);

    vp8_yv12_de_alloc_frame_buffer(&oci->temp_scale_frame);
#if CONFIG_POSTPROC
    vp8_yv12_de_alloc_frame_buffer(&oci->post_proc_buffer);
    if (oci->post_proc_buffer_int_used)
        vp8_yv12_de_alloc_frame_buffer(&oci->post_proc_buffer_int);

    vpx_free(oci->pp_limits_buffer);
    oci->pp_limits_buffer = NULL;
#endif

    vpx_free(oci->above_context);
    vpx_free(oci->mip);
#if CONFIG_ERROR_CONCEALMENT
    vpx_free(oci->prev_mip);
    oci->prev_mip = NULL;
#endif

    oci->above_context = NULL;
    oci->mip = NULL;
}

int vp8_alloc_frame_buffers(VP8_COMMON *oci, int width, int height)
{
    int i;

    vp8_de_alloc_frame_buffers(oci);

    /* our internal buffers are always multiples of 16 */
    if ((width & 0xf) != 0)
        width += 16 - (width & 0xf);

    if ((height & 0xf) != 0)
        height += 16 - (height & 0xf);


    for (i = 0; i < NUM_YV12_BUFFERS; i++)
    {
        oci->fb_idx_ref_cnt[i] = 0;
        oci->yv12_fb[i].flags = 0;
        if (vp8_yv12_alloc_frame_buffer(&oci->yv12_fb[i], width, height, VP8BORDERINPIXELS) < 0)
            goto allocation_fail;
    }

    oci->new_fb_idx = 0;
    oci->lst_fb_idx = 1;
    oci->gld_fb_idx = 2;
    oci->alt_fb_idx = 3;

    oci->fb_idx_ref_cnt[0] = 1;
    oci->fb_idx_ref_cnt[1] = 1;
    oci->fb_idx_ref_cnt[2] = 1;
    oci->fb_idx_ref_cnt[3] = 1;

    if (vp8_yv12_alloc_frame_buffer(&oci->temp_scale_frame,   width, 16, VP8BORDERINPIXELS) < 0)
        goto allocation_fail;

    oci->mb_rows = height >> 4;
    oci->mb_cols = width >> 4;
    oci->MBs = oci->mb_rows * oci->mb_cols;
    oci->mode_info_stride = oci->mb_cols + 1;
    oci->mip = vpx_calloc((oci->mb_cols + 1) * (oci->mb_rows + 1), sizeof(MODE_INFO));

    if (!oci->mip)
        goto allocation_fail;

    oci->mi = oci->mip + oci->mode_info_stride + 1;

    /* Allocation of previous mode info will be done in vp8_decode_frame()
     * as it is a decoder only data */

    oci->above_context = vpx_calloc(sizeof(ENTROPY_CONTEXT_PLANES) * oci->mb_cols, 1);

    if (!oci->above_context)
        goto allocation_fail;

#if CONFIG_POSTPROC
    if (vp8_yv12_alloc_frame_buffer(&oci->post_proc_buffer, width, height, VP8BORDERINPIXELS) < 0)
        goto allocation_fail;

    oci->post_proc_buffer_int_used = 0;
    memset(&oci->postproc_state, 0, sizeof(oci->postproc_state));
    memset(oci->post_proc_buffer.buffer_alloc, 128,
           oci->post_proc_buffer.frame_size);

    /* Allocate buffer to store post-processing filter coefficients.
     *
     * Note: Round up mb_cols to support SIMD reads
     */
    oci->pp_limits_buffer = vpx_memalign(16, 24 * ((oci->mb_cols + 1) & ~1));
    if (!oci->pp_limits_buffer)
        goto allocation_fail;
#endif

    return 0;

allocation_fail:
    vp8_de_alloc_frame_buffers(oci);
    return 1;
}

void vp8_setup_version(VP8_COMMON *cm)
{
    switch (cm->version)
    {
    case 0:
        cm->no_lpf = 0;
        cm->filter_type = NORMAL_LOOPFILTER;
        cm->use_bilinear_mc_filter = 0;
        cm->full_pixel = 0;
        break;
    case 1:
        cm->no_lpf = 0;
        cm->filter_type = SIMPLE_LOOPFILTER;
        cm->use_bilinear_mc_filter = 1;
        cm->full_pixel = 0;
        break;
    case 2:
        cm->no_lpf = 1;
        cm->filter_type = NORMAL_LOOPFILTER;
        cm->use_bilinear_mc_filter = 1;
        cm->full_pixel = 0;
        break;
    case 3:
        cm->no_lpf = 1;
        cm->filter_type = SIMPLE_LOOPFILTER;
        cm->use_bilinear_mc_filter = 1;
        cm->full_pixel = 1;
        break;
    default:
        /*4,5,6,7 are reserved for future use*/
        cm->no_lpf = 0;
        cm->filter_type = NORMAL_LOOPFILTER;
        cm->use_bilinear_mc_filter = 0;
        cm->full_pixel = 0;
        break;
    }
}
void vp8_create_common(VP8_COMMON *oci)
{
    vp8_machine_specific_config(oci);

    vp8_init_mbmode_probs(oci);
    vp8_default_bmode_probs(oci->fc.bmode_prob);

    oci->mb_no_coeff_skip = 1;
    oci->no_lpf = 0;
    oci->filter_type = NORMAL_LOOPFILTER;
    oci->use_bilinear_mc_filter = 0;
    oci->full_pixel = 0;
    oci->multi_token_partition = ONE_PARTITION;
    oci->clamp_type = RECON_CLAMP_REQUIRED;

    /* Initialize reference frame sign bias structure to defaults */
    memset(oci->ref_frame_sign_bias, 0, sizeof(oci->ref_frame_sign_bias));

    /* Default disable buffer to buffer copying */
    oci->copy_buffer_to_gf = 0;
    oci->copy_buffer_to_arf = 0;
}

void vp8_remove_common(VP8_COMMON *oci)
{
    vp8_de_alloc_frame_buffers(oci);
}
