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
#include <stdlib.h>
#include <string.h>
#include "./vp8_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "./vpx_scale_rtcd.h"
#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "vpx_version.h"
#include "common/alloccommon.h"
#include "common/common.h"
#include "common/onyxd.h"
#include "decoder/onyxd_int.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_mem/vpx_mem.h"
#if CONFIG_ERROR_CONCEALMENT
#include "decoder/error_concealment.h"
#endif
#include "decoder/decoderthreading.h"

#define VP8_CAP_POSTPROC (CONFIG_POSTPROC ? VPX_CODEC_CAP_POSTPROC : 0)
#define VP8_CAP_ERROR_CONCEALMENT (CONFIG_ERROR_CONCEALMENT ? \
                                    VPX_CODEC_CAP_ERROR_CONCEALMENT : 0)

typedef vpx_codec_stream_info_t  vp8_stream_info_t;

/* Structures for handling memory allocations */
typedef enum
{
    VP8_SEG_ALG_PRIV     = 256,
    VP8_SEG_MAX
} mem_seg_id_t;
#define NELEMENTS(x) ((int)(sizeof(x)/sizeof(x[0])))

struct vpx_codec_alg_priv
{
    vpx_codec_priv_t        base;
    vpx_codec_dec_cfg_t     cfg;
    vp8_stream_info_t       si;
    int                     decoder_init;
    int                     postproc_cfg_set;
    vp8_postproc_cfg_t      postproc_cfg;
#if CONFIG_POSTPROC_VISUALIZER
    unsigned int            dbg_postproc_flag;
    int                     dbg_color_ref_frame_flag;
    int                     dbg_color_mb_modes_flag;
    int                     dbg_color_b_modes_flag;
    int                     dbg_display_mv_flag;
#endif
    vpx_decrypt_cb          decrypt_cb;
    void                    *decrypt_state;
    vpx_image_t             img;
    int                     img_setup;
    struct frame_buffers    yv12_frame_buffers;
    void                    *user_priv;
    FRAGMENT_DATA           fragments;
};

static int vp8_init_ctx(vpx_codec_ctx_t *ctx)
{
    vpx_codec_alg_priv_t *priv =
        (vpx_codec_alg_priv_t *)vpx_calloc(1, sizeof(*priv));
    if (!priv) return 1;

    ctx->priv = (vpx_codec_priv_t *)priv;
    ctx->priv->init_flags = ctx->init_flags;

    priv->si.sz = sizeof(priv->si);
    priv->decrypt_cb = NULL;
    priv->decrypt_state = NULL;

    if (ctx->config.dec)
    {
        /* Update the reference to the config structure to an internal copy. */
        priv->cfg = *ctx->config.dec;
        ctx->config.dec = &priv->cfg;
    }

    return 0;
}

static vpx_codec_err_t vp8_init(vpx_codec_ctx_t *ctx,
                                vpx_codec_priv_enc_mr_cfg_t *data)
{
    vpx_codec_err_t res = VPX_CODEC_OK;
    vpx_codec_alg_priv_t *priv = NULL;
    (void) data;

    vp8_rtcd();
    vpx_dsp_rtcd();
    vpx_scale_rtcd();

    /* This function only allocates space for the vpx_codec_alg_priv_t
     * structure. More memory may be required at the time the stream
     * information becomes known.
     */
    if (!ctx->priv) {
      if (vp8_init_ctx(ctx)) return VPX_CODEC_MEM_ERROR;
      priv = (vpx_codec_alg_priv_t *)ctx->priv;

      /* initialize number of fragments to zero */
      priv->fragments.count = 0;
      /* is input fragments enabled? */
      priv->fragments.enabled =
          (priv->base.init_flags & VPX_CODEC_USE_INPUT_FRAGMENTS);

      /*post processing level initialized to do nothing */
    } else {
      priv = (vpx_codec_alg_priv_t *)ctx->priv;
    }

    priv->yv12_frame_buffers.use_frame_threads =
        (ctx->priv->init_flags & VPX_CODEC_USE_FRAME_THREADING);

    /* for now, disable frame threading */
    priv->yv12_frame_buffers.use_frame_threads = 0;

    if (priv->yv12_frame_buffers.use_frame_threads &&
        ((ctx->priv->init_flags & VPX_CODEC_USE_ERROR_CONCEALMENT) ||
         (ctx->priv->init_flags & VPX_CODEC_USE_INPUT_FRAGMENTS))) {
      /* row-based threading, error concealment, and input fragments will
       * not be supported when using frame-based threading */
      res = VPX_CODEC_INVALID_PARAM;
    }

    return res;
}

static vpx_codec_err_t vp8_destroy(vpx_codec_alg_priv_t *ctx)
{
    vp8_remove_decoder_instances(&ctx->yv12_frame_buffers);

    vpx_free(ctx);

    return VPX_CODEC_OK;
}

static vpx_codec_err_t vp8_peek_si_internal(const uint8_t *data,
                                            unsigned int data_sz,
                                            vpx_codec_stream_info_t *si,
                                            vpx_decrypt_cb decrypt_cb,
                                            void *decrypt_state)
{
    vpx_codec_err_t res = VPX_CODEC_OK;

    assert(data != NULL);

    if(data + data_sz <= data)
    {
        res = VPX_CODEC_INVALID_PARAM;
    }
    else
    {
        /* Parse uncompresssed part of key frame header.
         * 3 bytes:- including version, frame type and an offset
         * 3 bytes:- sync code (0x9d, 0x01, 0x2a)
         * 4 bytes:- including image width and height in the lowest 14 bits
         *           of each 2-byte value.
         */
        uint8_t clear_buffer[10];
        const uint8_t *clear = data;
        if (decrypt_cb)
        {
            int n = VPXMIN(sizeof(clear_buffer), data_sz);
            decrypt_cb(decrypt_state, data, clear_buffer, n);
            clear = clear_buffer;
        }
        si->is_kf = 0;

        if (data_sz >= 10 && !(clear[0] & 0x01))  /* I-Frame */
        {
            si->is_kf = 1;

            /* vet via sync code */
            if (clear[3] != 0x9d || clear[4] != 0x01 || clear[5] != 0x2a)
                return VPX_CODEC_UNSUP_BITSTREAM;

            si->w = (clear[6] | (clear[7] << 8)) & 0x3fff;
            si->h = (clear[8] | (clear[9] << 8)) & 0x3fff;

            /*printf("w=%d, h=%d\n", si->w, si->h);*/
            if (!(si->h | si->w))
                res = VPX_CODEC_UNSUP_BITSTREAM;
        }
        else
        {
            res = VPX_CODEC_UNSUP_BITSTREAM;
        }
    }

    return res;
}

static vpx_codec_err_t vp8_peek_si(const uint8_t *data,
                                   unsigned int data_sz,
                                   vpx_codec_stream_info_t *si) {
    return vp8_peek_si_internal(data, data_sz, si, NULL, NULL);
}

static vpx_codec_err_t vp8_get_si(vpx_codec_alg_priv_t    *ctx,
                                  vpx_codec_stream_info_t *si)
{

    unsigned int sz;

    if (si->sz >= sizeof(vp8_stream_info_t))
        sz = sizeof(vp8_stream_info_t);
    else
        sz = sizeof(vpx_codec_stream_info_t);

    memcpy(si, &ctx->si, sz);
    si->sz = sz;

    return VPX_CODEC_OK;
}


static vpx_codec_err_t
update_error_state(vpx_codec_alg_priv_t                 *ctx,
                   const struct vpx_internal_error_info *error)
{
    vpx_codec_err_t res;

    if ((res = error->error_code))
        ctx->base.err_detail = error->has_detail
                               ? error->detail
                               : NULL;

    return res;
}

static void yuvconfig2image(vpx_image_t               *img,
                            const YV12_BUFFER_CONFIG  *yv12,
                            void                      *user_priv)
{
    /** vpx_img_wrap() doesn't allow specifying independent strides for
      * the Y, U, and V planes, nor other alignment adjustments that
      * might be representable by a YV12_BUFFER_CONFIG, so we just
      * initialize all the fields.*/
    img->fmt = VPX_IMG_FMT_I420;
    img->w = yv12->y_stride;
    img->h = (yv12->y_height + 2 * VP8BORDERINPIXELS + 15) & ~15;
    img->d_w = img->r_w = yv12->y_width;
    img->d_h = img->r_h = yv12->y_height;
    img->x_chroma_shift = 1;
    img->y_chroma_shift = 1;
    img->planes[VPX_PLANE_Y] = yv12->y_buffer;
    img->planes[VPX_PLANE_U] = yv12->u_buffer;
    img->planes[VPX_PLANE_V] = yv12->v_buffer;
    img->planes[VPX_PLANE_ALPHA] = NULL;
    img->stride[VPX_PLANE_Y] = yv12->y_stride;
    img->stride[VPX_PLANE_U] = yv12->uv_stride;
    img->stride[VPX_PLANE_V] = yv12->uv_stride;
    img->stride[VPX_PLANE_ALPHA] = yv12->y_stride;
    img->bit_depth = 8;
    img->bps = 12;
    img->user_priv = user_priv;
    img->img_data = yv12->buffer_alloc;
    img->img_data_owner = 0;
    img->self_allocd = 0;
}

static int
update_fragments(vpx_codec_alg_priv_t  *ctx,
                 const uint8_t         *data,
                 unsigned int           data_sz,
                 vpx_codec_err_t       *res)
{
    *res = VPX_CODEC_OK;

    if (ctx->fragments.count == 0)
    {
        /* New frame, reset fragment pointers and sizes */
        memset((void*)ctx->fragments.ptrs, 0, sizeof(ctx->fragments.ptrs));
        memset(ctx->fragments.sizes, 0, sizeof(ctx->fragments.sizes));
    }
    if (ctx->fragments.enabled && !(data == NULL && data_sz == 0))
    {
        /* Store a pointer to this fragment and return. We haven't
         * received the complete frame yet, so we will wait with decoding.
         */
        ctx->fragments.ptrs[ctx->fragments.count] = data;
        ctx->fragments.sizes[ctx->fragments.count] = data_sz;
        ctx->fragments.count++;
        if (ctx->fragments.count > (1 << EIGHT_PARTITION) + 1)
        {
            ctx->fragments.count = 0;
            *res = VPX_CODEC_INVALID_PARAM;
            return -1;
        }
        return 0;
    }

    if (!ctx->fragments.enabled && (data == NULL && data_sz == 0))
    {
        return 0;
    }

    if (!ctx->fragments.enabled)
    {
        ctx->fragments.ptrs[0] = data;
        ctx->fragments.sizes[0] = data_sz;
        ctx->fragments.count = 1;
    }

    return 1;
}

static vpx_codec_err_t vp8_decode(vpx_codec_alg_priv_t  *ctx,
                                  const uint8_t         *data,
                                  unsigned int            data_sz,
                                  void                    *user_priv,
                                  long                    deadline)
{
    vpx_codec_err_t res = VPX_CODEC_OK;
    unsigned int resolution_change = 0;
    unsigned int w, h;

    if (!ctx->fragments.enabled && (data == NULL && data_sz == 0))
    {
        return 0;
    }

    /* Update the input fragment data */
    if(update_fragments(ctx, data, data_sz, &res) <= 0)
        return res;

    /* Determine the stream parameters. Note that we rely on peek_si to
     * validate that we have a buffer that does not wrap around the top
     * of the heap.
     */
    w = ctx->si.w;
    h = ctx->si.h;

    res = vp8_peek_si_internal(ctx->fragments.ptrs[0], ctx->fragments.sizes[0],
                               &ctx->si, ctx->decrypt_cb, ctx->decrypt_state);

    if((res == VPX_CODEC_UNSUP_BITSTREAM) && !ctx->si.is_kf)
    {
        /* the peek function returns an error for non keyframes, however for
         * this case, it is not an error */
        res = VPX_CODEC_OK;
    }

    if(!ctx->decoder_init && !ctx->si.is_kf)
        res = VPX_CODEC_UNSUP_BITSTREAM;

    if ((ctx->si.h != h) || (ctx->si.w != w))
        resolution_change = 1;

    /* Initialize the decoder instance on the first frame*/
    if (!res && !ctx->decoder_init)
    {
      VP8D_CONFIG oxcf;

      oxcf.Width = ctx->si.w;
      oxcf.Height = ctx->si.h;
      oxcf.Version = 9;
      oxcf.postprocess = 0;
      oxcf.max_threads = ctx->cfg.threads;
      oxcf.error_concealment =
          (ctx->base.init_flags & VPX_CODEC_USE_ERROR_CONCEALMENT);

      /* If postprocessing was enabled by the application and a
       * configuration has not been provided, default it.
       */
       if (!ctx->postproc_cfg_set
           && (ctx->base.init_flags & VPX_CODEC_USE_POSTPROC)) {
         ctx->postproc_cfg.post_proc_flag =
             VP8_DEBLOCK | VP8_DEMACROBLOCK | VP8_MFQE;
         ctx->postproc_cfg.deblocking_level = 4;
         ctx->postproc_cfg.noise_level = 0;
       }

       res = vp8_create_decoder_instances(&ctx->yv12_frame_buffers, &oxcf);
       ctx->decoder_init = 1;
    }

    /* Set these even if already initialized.  The caller may have changed the
     * decrypt config between frames.
     */
    if (ctx->decoder_init) {
      ctx->yv12_frame_buffers.pbi[0]->decrypt_cb = ctx->decrypt_cb;
      ctx->yv12_frame_buffers.pbi[0]->decrypt_state = ctx->decrypt_state;
    }

    if (!res)
    {
        VP8D_COMP *pbi = ctx->yv12_frame_buffers.pbi[0];
        if (resolution_change)
        {
            VP8_COMMON *const pc = & pbi->common;
            MACROBLOCKD *const xd  = & pbi->mb;
#if CONFIG_MULTITHREAD
            int i;
#endif
            pc->Width = ctx->si.w;
            pc->Height = ctx->si.h;
            {
                int prev_mb_rows = pc->mb_rows;

                if (setjmp(pbi->common.error.jmp))
                {
                    pbi->common.error.setjmp = 0;
                    vp8_clear_system_state();
                    /* same return value as used in vp8dx_receive_compressed_data */
                    return -1;
                }

                pbi->common.error.setjmp = 1;

                if (pc->Width <= 0)
                {
                    pc->Width = w;
                    vpx_internal_error(&pc->error, VPX_CODEC_CORRUPT_FRAME,
                                       "Invalid frame width");
                }

                if (pc->Height <= 0)
                {
                    pc->Height = h;
                    vpx_internal_error(&pc->error, VPX_CODEC_CORRUPT_FRAME,
                                       "Invalid frame height");
                }

                if (vp8_alloc_frame_buffers(pc, pc->Width, pc->Height))
                    vpx_internal_error(&pc->error, VPX_CODEC_MEM_ERROR,
                                       "Failed to allocate frame buffers");

                xd->pre = pc->yv12_fb[pc->lst_fb_idx];
                xd->dst = pc->yv12_fb[pc->new_fb_idx];

#if CONFIG_MULTITHREAD
                for (i = 0; i < pbi->allocated_decoding_thread_count; i++)
                {
                    pbi->mb_row_di[i].mbd.dst = pc->yv12_fb[pc->new_fb_idx];
                    vp8_build_block_doffsets(&pbi->mb_row_di[i].mbd);
                }
#endif
                vp8_build_block_doffsets(&pbi->mb);

                /* allocate memory for last frame MODE_INFO array */
#if CONFIG_ERROR_CONCEALMENT

                if (pbi->ec_enabled)
                {
                    /* old prev_mip was released by vp8_de_alloc_frame_buffers()
                     * called in vp8_alloc_frame_buffers() */
                    pc->prev_mip = vpx_calloc(
                                       (pc->mb_cols + 1) * (pc->mb_rows + 1),
                                       sizeof(MODE_INFO));

                    if (!pc->prev_mip)
                    {
                        vp8_de_alloc_frame_buffers(pc);
                        vpx_internal_error(&pc->error, VPX_CODEC_MEM_ERROR,
                                           "Failed to allocate"
                                           "last frame MODE_INFO array");
                    }

                    pc->prev_mi = pc->prev_mip + pc->mode_info_stride + 1;

                    if (vp8_alloc_overlap_lists(pbi))
                        vpx_internal_error(&pc->error, VPX_CODEC_MEM_ERROR,
                                           "Failed to allocate overlap lists "
                                           "for error concealment");
                }

#endif

#if CONFIG_MULTITHREAD
                if (pbi->b_multithreaded_rd)
                    vp8mt_alloc_temp_buffers(pbi, pc->Width, prev_mb_rows);
#else
                (void)prev_mb_rows;
#endif
            }

            pbi->common.error.setjmp = 0;

            /* required to get past the first get_free_fb() call */
            pbi->common.fb_idx_ref_cnt[0] = 0;
        }

        /* update the pbi fragment data */
        pbi->fragments = ctx->fragments;

        ctx->user_priv = user_priv;
        if (vp8dx_receive_compressed_data(pbi, data_sz, data, deadline))
        {
            res = update_error_state(ctx, &pbi->common.error);
        }

        /* get ready for the next series of fragments */
        ctx->fragments.count = 0;
    }

    return res;
}

static vpx_image_t *vp8_get_frame(vpx_codec_alg_priv_t  *ctx,
                                  vpx_codec_iter_t      *iter)
{
    vpx_image_t *img = NULL;

    /* iter acts as a flip flop, so an image is only returned on the first
     * call to get_frame.
     */
    if (!(*iter) && ctx->yv12_frame_buffers.pbi[0])
    {
        YV12_BUFFER_CONFIG sd;
        int64_t time_stamp = 0, time_end_stamp = 0;
        vp8_ppflags_t flags;
        vp8_zero(flags);

        if (ctx->base.init_flags & VPX_CODEC_USE_POSTPROC)
        {
            flags.post_proc_flag= ctx->postproc_cfg.post_proc_flag
#if CONFIG_POSTPROC_VISUALIZER

                                | ((ctx->dbg_color_ref_frame_flag != 0) ? VP8D_DEBUG_CLR_FRM_REF_BLKS : 0)
                                | ((ctx->dbg_color_mb_modes_flag != 0) ? VP8D_DEBUG_CLR_BLK_MODES : 0)
                                | ((ctx->dbg_color_b_modes_flag != 0) ? VP8D_DEBUG_CLR_BLK_MODES : 0)
                                | ((ctx->dbg_display_mv_flag != 0) ? VP8D_DEBUG_DRAW_MV : 0)
#endif
                                ;
            flags.deblocking_level      = ctx->postproc_cfg.deblocking_level;
            flags.noise_level           = ctx->postproc_cfg.noise_level;
#if CONFIG_POSTPROC_VISUALIZER
            flags.display_ref_frame_flag= ctx->dbg_color_ref_frame_flag;
            flags.display_mb_modes_flag = ctx->dbg_color_mb_modes_flag;
            flags.display_b_modes_flag  = ctx->dbg_color_b_modes_flag;
            flags.display_mv_flag       = ctx->dbg_display_mv_flag;
#endif
        }

        if (0 == vp8dx_get_raw_frame(ctx->yv12_frame_buffers.pbi[0], &sd,
                                     &time_stamp, &time_end_stamp, &flags))
        {
            yuvconfig2image(&ctx->img, &sd, ctx->user_priv);

            img = &ctx->img;
            *iter = img;
        }
    }

    return img;
}

static vpx_codec_err_t image2yuvconfig(const vpx_image_t   *img,
                                       YV12_BUFFER_CONFIG  *yv12)
{
    const int y_w = img->d_w;
    const int y_h = img->d_h;
    const int uv_w = (img->d_w + 1) / 2;
    const int uv_h = (img->d_h + 1) / 2;
    vpx_codec_err_t        res = VPX_CODEC_OK;
    yv12->y_buffer = img->planes[VPX_PLANE_Y];
    yv12->u_buffer = img->planes[VPX_PLANE_U];
    yv12->v_buffer = img->planes[VPX_PLANE_V];

    yv12->y_crop_width  = y_w;
    yv12->y_crop_height = y_h;
    yv12->y_width  = y_w;
    yv12->y_height = y_h;
    yv12->uv_crop_width = uv_w;
    yv12->uv_crop_height = uv_h;
    yv12->uv_width = uv_w;
    yv12->uv_height = uv_h;

    yv12->y_stride = img->stride[VPX_PLANE_Y];
    yv12->uv_stride = img->stride[VPX_PLANE_U];

    yv12->border  = (img->stride[VPX_PLANE_Y] - img->d_w) / 2;
    return res;
}


static vpx_codec_err_t vp8_set_reference(vpx_codec_alg_priv_t *ctx,
                                         va_list args)
{

    vpx_ref_frame_t *data = va_arg(args, vpx_ref_frame_t *);

    if (data && !ctx->yv12_frame_buffers.use_frame_threads)
    {
        vpx_ref_frame_t *frame = (vpx_ref_frame_t *)data;
        YV12_BUFFER_CONFIG sd;

        image2yuvconfig(&frame->img, &sd);

        return vp8dx_set_reference(ctx->yv12_frame_buffers.pbi[0],
                                   frame->frame_type, &sd);
    }
    else
        return VPX_CODEC_INVALID_PARAM;

}

static vpx_codec_err_t vp8_get_reference(vpx_codec_alg_priv_t *ctx,
                                         va_list args)
{

    vpx_ref_frame_t *data = va_arg(args, vpx_ref_frame_t *);

    if (data && !ctx->yv12_frame_buffers.use_frame_threads)
    {
        vpx_ref_frame_t *frame = (vpx_ref_frame_t *)data;
        YV12_BUFFER_CONFIG sd;

        image2yuvconfig(&frame->img, &sd);

        return vp8dx_get_reference(ctx->yv12_frame_buffers.pbi[0],
                                   frame->frame_type, &sd);
    }
    else
        return VPX_CODEC_INVALID_PARAM;

}

static vpx_codec_err_t vp8_set_postproc(vpx_codec_alg_priv_t *ctx,
                                        va_list args)
{
#if CONFIG_POSTPROC
    vp8_postproc_cfg_t *data = va_arg(args, vp8_postproc_cfg_t *);

    if (data)
    {
        ctx->postproc_cfg_set = 1;
        ctx->postproc_cfg = *((vp8_postproc_cfg_t *)data);
        return VPX_CODEC_OK;
    }
    else
        return VPX_CODEC_INVALID_PARAM;

#else
    (void)ctx;
    (void)args;
    return VPX_CODEC_INCAPABLE;
#endif
}


static vpx_codec_err_t vp8_set_dbg_color_ref_frame(vpx_codec_alg_priv_t *ctx,
                                                   va_list args) {
#if CONFIG_POSTPROC_VISUALIZER && CONFIG_POSTPROC
  ctx->dbg_color_ref_frame_flag = va_arg(args, int);
  return VPX_CODEC_OK;
#else
  (void)ctx;
  (void)args;
  return VPX_CODEC_INCAPABLE;
#endif
}

static vpx_codec_err_t vp8_set_dbg_color_mb_modes(vpx_codec_alg_priv_t *ctx,
                                                  va_list args) {
#if CONFIG_POSTPROC_VISUALIZER && CONFIG_POSTPROC
  ctx->dbg_color_mb_modes_flag = va_arg(args, int);
  return VPX_CODEC_OK;
#else
  (void)ctx;
  (void)args;
  return VPX_CODEC_INCAPABLE;
#endif
}

static vpx_codec_err_t vp8_set_dbg_color_b_modes(vpx_codec_alg_priv_t *ctx,
                                                 va_list args) {
#if CONFIG_POSTPROC_VISUALIZER && CONFIG_POSTPROC
  ctx->dbg_color_b_modes_flag = va_arg(args, int);
  return VPX_CODEC_OK;
#else
  (void)ctx;
  (void)args;
  return VPX_CODEC_INCAPABLE;
#endif
}

static vpx_codec_err_t vp8_set_dbg_display_mv(vpx_codec_alg_priv_t *ctx,
                                              va_list args) {
#if CONFIG_POSTPROC_VISUALIZER && CONFIG_POSTPROC
  ctx->dbg_display_mv_flag = va_arg(args, int);
  return VPX_CODEC_OK;
#else
  (void)ctx;
  (void)args;
  return VPX_CODEC_INCAPABLE;
#endif
}

static vpx_codec_err_t vp8_get_last_ref_updates(vpx_codec_alg_priv_t *ctx,
                                                va_list args)
{
    int *update_info = va_arg(args, int *);

    if (update_info && !ctx->yv12_frame_buffers.use_frame_threads)
    {
        VP8D_COMP *pbi = (VP8D_COMP *)ctx->yv12_frame_buffers.pbi[0];

        *update_info = pbi->common.refresh_alt_ref_frame * (int) VP8_ALTR_FRAME
            + pbi->common.refresh_golden_frame * (int) VP8_GOLD_FRAME
            + pbi->common.refresh_last_frame * (int) VP8_LAST_FRAME;

        return VPX_CODEC_OK;
    }
    else
        return VPX_CODEC_INVALID_PARAM;
}

extern int vp8dx_references_buffer( VP8_COMMON *oci, int ref_frame );
static vpx_codec_err_t vp8_get_last_ref_frame(vpx_codec_alg_priv_t *ctx,
                                              va_list args)
{
    int *ref_info = va_arg(args, int *);

    if (ref_info && !ctx->yv12_frame_buffers.use_frame_threads)
    {
        VP8D_COMP *pbi = (VP8D_COMP *)ctx->yv12_frame_buffers.pbi[0];
        VP8_COMMON *oci = &pbi->common;
        *ref_info =
            (vp8dx_references_buffer( oci, ALTREF_FRAME )?VP8_ALTR_FRAME:0) |
            (vp8dx_references_buffer( oci, GOLDEN_FRAME )?VP8_GOLD_FRAME:0) |
            (vp8dx_references_buffer( oci, LAST_FRAME )?VP8_LAST_FRAME:0);

        return VPX_CODEC_OK;
    }
    else
        return VPX_CODEC_INVALID_PARAM;
}

static vpx_codec_err_t vp8_get_frame_corrupted(vpx_codec_alg_priv_t *ctx,
                                               va_list args)
{

    int *corrupted = va_arg(args, int *);
    VP8D_COMP *pbi = (VP8D_COMP *)ctx->yv12_frame_buffers.pbi[0];

    if (corrupted && pbi)
    {
        const YV12_BUFFER_CONFIG *const frame = pbi->common.frame_to_show;
        if (frame == NULL) return VPX_CODEC_ERROR;
        *corrupted = frame->corrupted;
        return VPX_CODEC_OK;
    }
    else
        return VPX_CODEC_INVALID_PARAM;

}

static vpx_codec_err_t vp8_set_decryptor(vpx_codec_alg_priv_t *ctx,
                                         va_list args)
{
    vpx_decrypt_init *init = va_arg(args, vpx_decrypt_init *);

    if (init)
    {
        ctx->decrypt_cb = init->decrypt_cb;
        ctx->decrypt_state = init->decrypt_state;
    }
    else
    {
        ctx->decrypt_cb = NULL;
        ctx->decrypt_state = NULL;
    }
    return VPX_CODEC_OK;
}

vpx_codec_ctrl_fn_map_t vp8_ctf_maps[] =
{
    {VP8_SET_REFERENCE,             vp8_set_reference},
    {VP8_COPY_REFERENCE,            vp8_get_reference},
    {VP8_SET_POSTPROC,              vp8_set_postproc},
    {VP8_SET_DBG_COLOR_REF_FRAME,   vp8_set_dbg_color_ref_frame},
    {VP8_SET_DBG_COLOR_MB_MODES,    vp8_set_dbg_color_mb_modes},
    {VP8_SET_DBG_COLOR_B_MODES,     vp8_set_dbg_color_b_modes},
    {VP8_SET_DBG_DISPLAY_MV,        vp8_set_dbg_display_mv},
    {VP8D_GET_LAST_REF_UPDATES,     vp8_get_last_ref_updates},
    {VP8D_GET_FRAME_CORRUPTED,      vp8_get_frame_corrupted},
    {VP8D_GET_LAST_REF_USED,        vp8_get_last_ref_frame},
    {VPXD_SET_DECRYPTOR,            vp8_set_decryptor},
    { -1, NULL},
};


#ifndef VERSION_STRING
#define VERSION_STRING
#endif
CODEC_INTERFACE(vpx_codec_vp8_dx) =
{
    "WebM Project VP8 Decoder" VERSION_STRING,
    VPX_CODEC_INTERNAL_ABI_VERSION,
    VPX_CODEC_CAP_DECODER | VP8_CAP_POSTPROC | VP8_CAP_ERROR_CONCEALMENT |
    VPX_CODEC_CAP_INPUT_FRAGMENTS,
    /* vpx_codec_caps_t          caps; */
    vp8_init,         /* vpx_codec_init_fn_t       init; */
    vp8_destroy,      /* vpx_codec_destroy_fn_t    destroy; */
    vp8_ctf_maps,     /* vpx_codec_ctrl_fn_map_t  *ctrl_maps; */
    {
        vp8_peek_si,      /* vpx_codec_peek_si_fn_t    peek_si; */
        vp8_get_si,       /* vpx_codec_get_si_fn_t     get_si; */
        vp8_decode,       /* vpx_codec_decode_fn_t     decode; */
        vp8_get_frame,    /* vpx_codec_frame_get_fn_t  frame_get; */
        NULL,
    },
    { /* encoder functions */
        0,
        NULL,  /* vpx_codec_enc_cfg_map_t */
        NULL,  /* vpx_codec_encode_fn_t */
        NULL,  /* vpx_codec_get_cx_data_fn_t */
        NULL,  /* vpx_codec_enc_config_set_fn_t */
        NULL,  /* vpx_codec_get_global_headers_fn_t */
        NULL,  /* vpx_codec_get_preview_frame_fn_t */
        NULL   /* vpx_codec_enc_mr_get_mem_loc_fn_t */
    }
};
