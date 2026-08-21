/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/*!\file
 * \brief Provides the high level interface to wrap encoder algorithms.
 *
 */
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include "vpx_config.h"
#include "vpx/vpx_encoder.h"
#include "vpx/internal/vpx_codec_internal.h"

#define SAVE_STATUS(ctx, var) ((ctx) ? ((ctx)->err = (var)) : (var))

static vpx_codec_alg_priv_t *get_alg_priv(vpx_codec_ctx_t *ctx) {
  return (vpx_codec_alg_priv_t *)ctx->priv;
}

vpx_codec_err_t vpx_codec_enc_init_ver(vpx_codec_ctx_t *ctx,
                                       vpx_codec_iface_t *iface,
                                       const vpx_codec_enc_cfg_t *cfg,
                                       vpx_codec_flags_t flags, int ver) {
  vpx_codec_err_t res;

  if (ver != VPX_ENCODER_ABI_VERSION)
    res = VPX_CODEC_ABI_MISMATCH;
  else if (!ctx || !iface || !cfg)
    res = VPX_CODEC_INVALID_PARAM;
  else if (iface->abi_version != VPX_CODEC_INTERNAL_ABI_VERSION)
    res = VPX_CODEC_ABI_MISMATCH;
  else if (!(iface->caps & VPX_CODEC_CAP_ENCODER))
    res = VPX_CODEC_INCAPABLE;
  else if ((flags & VPX_CODEC_USE_PSNR) && !(iface->caps & VPX_CODEC_CAP_PSNR))
    res = VPX_CODEC_INCAPABLE;
  else if ((flags & VPX_CODEC_USE_OUTPUT_PARTITION) &&
           !(iface->caps & VPX_CODEC_CAP_OUTPUT_PARTITION))
    res = VPX_CODEC_INCAPABLE;
  else {
    ctx->iface = iface;
    ctx->name = iface->name;
    ctx->priv = NULL;
    ctx->init_flags = flags;
    ctx->config.enc = cfg;
    res = ctx->iface->init(ctx, NULL);

    if (res) {
      // IMPORTANT: ctx->priv->err_detail must be null or point to a string
      // that remains valid after ctx->priv is destroyed, such as a C string
      // literal. This makes it safe to call vpx_codec_error_detail() after
      // vpx_codec_enc_init_ver() failed.
      ctx->err_detail = ctx->priv ? ctx->priv->err_detail : NULL;
      vpx_codec_destroy(ctx);
    }
  }

  return SAVE_STATUS(ctx, res);
}

vpx_codec_err_t vpx_codec_enc_init_multi_ver(
    vpx_codec_ctx_t *ctx, vpx_codec_iface_t *iface,
    const vpx_codec_enc_cfg_t *cfg, int num_enc, vpx_codec_flags_t flags,
    const vpx_rational_t *dsf, int ver) {
  vpx_codec_err_t res = VPX_CODEC_OK;

  if (ver != VPX_ENCODER_ABI_VERSION)
    res = VPX_CODEC_ABI_MISMATCH;
  else if (!ctx || !iface || !cfg || (num_enc > 16 || num_enc < 1) || !dsf)
    res = VPX_CODEC_INVALID_PARAM;
  else if (iface->abi_version != VPX_CODEC_INTERNAL_ABI_VERSION)
    res = VPX_CODEC_ABI_MISMATCH;
  else if (!(iface->caps & VPX_CODEC_CAP_ENCODER))
    res = VPX_CODEC_INCAPABLE;
  else if ((flags & VPX_CODEC_USE_PSNR) && !(iface->caps & VPX_CODEC_CAP_PSNR))
    res = VPX_CODEC_INCAPABLE;
  else if ((flags & VPX_CODEC_USE_OUTPUT_PARTITION) &&
           !(iface->caps & VPX_CODEC_CAP_OUTPUT_PARTITION))
    res = VPX_CODEC_INCAPABLE;
  else {
    int i;
#if CONFIG_MULTI_RES_ENCODING
    int mem_loc_owned = 0;
#endif
    void *mem_loc;

    if (iface->enc.mr_get_mem_loc == NULL) return VPX_CODEC_INCAPABLE;

    if (!(res = iface->enc.mr_get_mem_loc(cfg, &mem_loc))) {
      for (i = 0; i < num_enc; i++) {
        vpx_codec_priv_enc_mr_cfg_t mr_cfg;

        /* Validate down-sampling factor. */
        if (dsf->num < 1 || dsf->num > 4096 || dsf->den < 1 ||
            dsf->den > dsf->num) {
          res = VPX_CODEC_INVALID_PARAM;
        } else {
          mr_cfg.mr_low_res_mode_info = mem_loc;
          mr_cfg.mr_total_resolutions = num_enc;
          mr_cfg.mr_encoder_id = num_enc - 1 - i;
          mr_cfg.mr_down_sampling_factor = *dsf;

          ctx->iface = iface;
          ctx->name = iface->name;
          ctx->priv = NULL;
          ctx->init_flags = flags;
          ctx->config.enc = cfg;
          // ctx takes ownership of mr_cfg.mr_low_res_mode_info if and only if
          // this call succeeds. The first ctx entry in the array is
          // responsible for freeing the memory.
          res = ctx->iface->init(ctx, &mr_cfg);
        }

        if (res) {
          const char *error_detail = ctx->priv ? ctx->priv->err_detail : NULL;
          /* Destroy current ctx */
          ctx->err_detail = error_detail;
          vpx_codec_destroy(ctx);

          /* Destroy already allocated high-level ctx */
          while (i) {
            ctx--;
            ctx->err_detail = error_detail;
            vpx_codec_destroy(ctx);
            i--;
          }
#if CONFIG_MULTI_RES_ENCODING
          if (!mem_loc_owned) {
            assert(mem_loc);
            iface->enc.mr_free_mem_loc(mem_loc);
          }
#endif
          return SAVE_STATUS(ctx, res);
        }
#if CONFIG_MULTI_RES_ENCODING
        mem_loc_owned = 1;
#endif
        ctx++;
        cfg++;
        dsf++;
      }
      ctx--;
    }
  }

  return SAVE_STATUS(ctx, res);
}

vpx_codec_err_t vpx_codec_enc_config_default(vpx_codec_iface_t *iface,
                                             vpx_codec_enc_cfg_t *cfg,
                                             unsigned int usage) {
  vpx_codec_err_t res;

  if (!iface || !cfg || usage != 0)
    res = VPX_CODEC_INVALID_PARAM;
  else if (!(iface->caps & VPX_CODEC_CAP_ENCODER))
    res = VPX_CODEC_INCAPABLE;
  else {
    assert(iface->enc.cfg_map_count == 1);
    *cfg = iface->enc.cfg_maps->cfg;
    res = VPX_CODEC_OK;
  }

  return res;
}

#if VPX_ARCH_X86 || VPX_ARCH_X86_64
/* On X86, disable the x87 unit's internal 80 bit precision for better
 * consistency with the SSE unit's 64 bit precision.
 */
#include "vpx_ports/x86.h"
#define FLOATING_POINT_INIT() \
  do {                        \
  unsigned short x87_orig_mode = x87_set_double_precision()
#define FLOATING_POINT_RESTORE()       \
  x87_set_control_word(x87_orig_mode); \
  }                                    \
  while (0)

#else
static void FLOATING_POINT_INIT(void) {}
static void FLOATING_POINT_RESTORE(void) {}
#endif

vpx_codec_err_t vpx_codec_encode(vpx_codec_ctx_t *ctx, const vpx_image_t *img,
                                 vpx_codec_pts_t pts, unsigned long duration,
                                 vpx_enc_frame_flags_t flags,
                                 vpx_enc_deadline_t deadline) {
  vpx_codec_err_t res = VPX_CODEC_OK;

  if (!ctx || (img && !duration))
    res = VPX_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv)
    res = VPX_CODEC_ERROR;
  else if (!(ctx->iface->caps & VPX_CODEC_CAP_ENCODER))
    res = VPX_CODEC_INCAPABLE;
#if ULONG_MAX > UINT32_MAX
  else if (duration > UINT32_MAX || deadline > UINT32_MAX)
    res = VPX_CODEC_INVALID_PARAM;
#endif
  else {
    unsigned int num_enc = ctx->priv->enc.total_encoders;

    /* Execute in a normalized floating point environment, if the platform
     * requires it.
     */
    FLOATING_POINT_INIT();

    if (num_enc == 1)
      res = ctx->iface->enc.encode(get_alg_priv(ctx), img, pts, duration, flags,
                                   deadline);
    else {
      /* Multi-resolution encoding:
       * Encode multi-levels in reverse order. For example,
       * if mr_total_resolutions = 3, first encode level 2,
       * then encode level 1, and finally encode level 0.
       */
      int i;

      ctx += num_enc - 1;
      if (img) img += num_enc - 1;

      for (i = num_enc - 1; i >= 0; i--) {
        if ((res = ctx->iface->enc.encode(get_alg_priv(ctx), img, pts, duration,
                                          flags, deadline)))
          break;

        ctx--;
        if (img) img--;
      }
      ctx++;
    }

    FLOATING_POINT_RESTORE();
  }

  return SAVE_STATUS(ctx, res);
}

const vpx_codec_cx_pkt_t *vpx_codec_get_cx_data(vpx_codec_ctx_t *ctx,
                                                vpx_codec_iter_t *iter) {
  const vpx_codec_cx_pkt_t *pkt = NULL;

  if (ctx) {
    if (!iter)
      ctx->err = VPX_CODEC_INVALID_PARAM;
    else if (!ctx->iface || !ctx->priv)
      ctx->err = VPX_CODEC_ERROR;
    else if (!(ctx->iface->caps & VPX_CODEC_CAP_ENCODER))
      ctx->err = VPX_CODEC_INCAPABLE;
    else
      pkt = ctx->iface->enc.get_cx_data(get_alg_priv(ctx), iter);
  }

  if (pkt && pkt->kind == VPX_CODEC_CX_FRAME_PKT) {
    // If the application has specified a destination area for the
    // compressed data, and the codec has not placed the data there,
    // and it fits, copy it.
    vpx_codec_priv_t *const priv = ctx->priv;
    char *const dst_buf = (char *)priv->enc.cx_data_dst_buf.buf;

    if (dst_buf && pkt->data.raw.buf != dst_buf &&
        pkt->data.raw.sz + priv->enc.cx_data_pad_before +
                priv->enc.cx_data_pad_after <=
            priv->enc.cx_data_dst_buf.sz) {
      vpx_codec_cx_pkt_t *modified_pkt = &priv->enc.cx_data_pkt;

      memcpy(dst_buf + priv->enc.cx_data_pad_before, pkt->data.raw.buf,
             pkt->data.raw.sz);
      *modified_pkt = *pkt;
      modified_pkt->data.raw.buf = dst_buf;
      modified_pkt->data.raw.sz +=
          priv->enc.cx_data_pad_before + priv->enc.cx_data_pad_after;
      pkt = modified_pkt;
    }

    if (dst_buf == pkt->data.raw.buf) {
      priv->enc.cx_data_dst_buf.buf = dst_buf + pkt->data.raw.sz;
      priv->enc.cx_data_dst_buf.sz -= pkt->data.raw.sz;
    }
  }

  return pkt;
}

vpx_codec_err_t vpx_codec_set_cx_data_buf(vpx_codec_ctx_t *ctx,
                                          const vpx_fixed_buf_t *buf,
                                          unsigned int pad_before,
                                          unsigned int pad_after) {
  if (!ctx || !ctx->priv) return VPX_CODEC_INVALID_PARAM;

  if (buf) {
    ctx->priv->enc.cx_data_dst_buf = *buf;
    ctx->priv->enc.cx_data_pad_before = pad_before;
    ctx->priv->enc.cx_data_pad_after = pad_after;
  } else {
    ctx->priv->enc.cx_data_dst_buf.buf = NULL;
    ctx->priv->enc.cx_data_dst_buf.sz = 0;
    ctx->priv->enc.cx_data_pad_before = 0;
    ctx->priv->enc.cx_data_pad_after = 0;
  }

  return VPX_CODEC_OK;
}

const vpx_image_t *vpx_codec_get_preview_frame(vpx_codec_ctx_t *ctx) {
  vpx_image_t *img = NULL;

  if (ctx) {
    if (!ctx->iface || !ctx->priv)
      ctx->err = VPX_CODEC_ERROR;
    else if (!(ctx->iface->caps & VPX_CODEC_CAP_ENCODER))
      ctx->err = VPX_CODEC_INCAPABLE;
    else if (!ctx->iface->enc.get_preview)
      ctx->err = VPX_CODEC_INCAPABLE;
    else
      img = ctx->iface->enc.get_preview(get_alg_priv(ctx));
  }

  return img;
}

vpx_fixed_buf_t *vpx_codec_get_global_headers(vpx_codec_ctx_t *ctx) {
  vpx_fixed_buf_t *buf = NULL;

  if (ctx) {
    if (!ctx->iface || !ctx->priv)
      ctx->err = VPX_CODEC_ERROR;
    else if (!(ctx->iface->caps & VPX_CODEC_CAP_ENCODER))
      ctx->err = VPX_CODEC_INCAPABLE;
    else if (!ctx->iface->enc.get_glob_hdrs)
      ctx->err = VPX_CODEC_INCAPABLE;
    else
      buf = ctx->iface->enc.get_glob_hdrs(get_alg_priv(ctx));
  }

  return buf;
}

vpx_codec_err_t vpx_codec_enc_config_set(vpx_codec_ctx_t *ctx,
                                         const vpx_codec_enc_cfg_t *cfg) {
  vpx_codec_err_t res;

  if (!ctx || !ctx->iface || !ctx->priv || !cfg)
    res = VPX_CODEC_INVALID_PARAM;
  else if (!(ctx->iface->caps & VPX_CODEC_CAP_ENCODER))
    res = VPX_CODEC_INCAPABLE;
  else
    res = ctx->iface->enc.cfg_set(get_alg_priv(ctx), cfg);

  return SAVE_STATUS(ctx, res);
}

int vpx_codec_pkt_list_add(struct vpx_codec_pkt_list *list,
                           const struct vpx_codec_cx_pkt *pkt) {
  if (list->cnt < list->max) {
    list->pkts[list->cnt++] = *pkt;
    return 0;
  }

  return 1;
}

const vpx_codec_cx_pkt_t *vpx_codec_pkt_list_get(
    struct vpx_codec_pkt_list *list, vpx_codec_iter_t *iter) {
  const vpx_codec_cx_pkt_t *pkt;

  if (!(*iter)) {
    *iter = list->pkts;
  }

  pkt = (const vpx_codec_cx_pkt_t *)*iter;

  if ((size_t)(pkt - list->pkts) < list->cnt)
    *iter = pkt + 1;
  else
    pkt = NULL;

  return pkt;
}
