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
 * \brief Provides the high level interface to wrap decoder algorithms.
 *
 */
#include <stdarg.h>
#include <stdlib.h>
#include "vpx/vpx_integer.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "vpx_version.h"

#define SAVE_STATUS(ctx,var) (ctx?(ctx->err = var):var)

int vpx_codec_version(void) {
  return VERSION_PACKED;
}


const char *vpx_codec_version_str(void) {
  return VERSION_STRING_NOSP;
}


const char *vpx_codec_version_extra_str(void) {
  return VERSION_EXTRA;
}


const char *vpx_codec_iface_name(vpx_codec_iface_t *iface) {
  return iface ? iface->name : "<invalid interface>";
}

const char *vpx_codec_err_to_string(vpx_codec_err_t  err) {
  switch (err) {
    case VPX_CODEC_OK:
      return "Success";
    case VPX_CODEC_ERROR:
      return "Unspecified internal error";
    case VPX_CODEC_MEM_ERROR:
      return "Memory allocation error";
    case VPX_CODEC_ABI_MISMATCH:
      return "ABI version mismatch";
    case VPX_CODEC_INCAPABLE:
      return "Codec does not implement requested capability";
    case VPX_CODEC_UNSUP_BITSTREAM:
      return "Bitstream not supported by this decoder";
    case VPX_CODEC_UNSUP_FEATURE:
      return "Bitstream required feature not supported by this decoder";
    case VPX_CODEC_CORRUPT_FRAME:
      return "Corrupt frame detected";
    case  VPX_CODEC_INVALID_PARAM:
      return "Invalid parameter";
    case VPX_CODEC_LIST_END:
      return "End of iterated list";
  }

  return "Unrecognized error code";
}

const char *vpx_codec_error(vpx_codec_ctx_t  *ctx) {
  return (ctx) ? vpx_codec_err_to_string(ctx->err)
         : vpx_codec_err_to_string(VPX_CODEC_INVALID_PARAM);
}

const char *vpx_codec_error_detail(vpx_codec_ctx_t  *ctx) {
  if (ctx && ctx->err)
    return ctx->priv ? ctx->priv->err_detail : ctx->err_detail;

  return NULL;
}


vpx_codec_err_t vpx_codec_destroy(vpx_codec_ctx_t *ctx) {
  vpx_codec_err_t res;

  if (!ctx)
    res = VPX_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv)
    res = VPX_CODEC_ERROR;
  else {
    ctx->iface->destroy((vpx_codec_alg_priv_t *)ctx->priv);

    ctx->iface = NULL;
    ctx->name = NULL;
    ctx->priv = NULL;
    res = VPX_CODEC_OK;
  }

  return SAVE_STATUS(ctx, res);
}


vpx_codec_caps_t vpx_codec_get_caps(vpx_codec_iface_t *iface) {
  return (iface) ? iface->caps : 0;
}


vpx_codec_err_t vpx_codec_control_(vpx_codec_ctx_t  *ctx,
                                   int               ctrl_id,
                                   ...) {
  vpx_codec_err_t res;

  if (!ctx || !ctrl_id)
    res = VPX_CODEC_INVALID_PARAM;
  else if (!ctx->iface || !ctx->priv || !ctx->iface->ctrl_maps)
    res = VPX_CODEC_ERROR;
  else {
    vpx_codec_ctrl_fn_map_t *entry;

    res = VPX_CODEC_ERROR;

    for (entry = ctx->iface->ctrl_maps; entry && entry->fn; entry++) {
      if (!entry->ctrl_id || entry->ctrl_id == ctrl_id) {
        va_list  ap;

        va_start(ap, ctrl_id);
        res = entry->fn((vpx_codec_alg_priv_t *)ctx->priv, ap);
        va_end(ap);
        break;
      }
    }
  }

  return SAVE_STATUS(ctx, res);
}

void vpx_internal_error(struct vpx_internal_error_info *info,
                        vpx_codec_err_t                 error,
                        const char                     *fmt,
                        ...) {
  va_list ap;

  info->error_code = error;
  info->has_detail = 0;

  if (fmt) {
    size_t  sz = sizeof(info->detail);

    info->has_detail = 1;
    va_start(ap, fmt);
    vsnprintf(info->detail, sz - 1, fmt, ap);
    va_end(ap);
    info->detail[sz - 1] = '\0';
  }

  if (info->setjmp)
    longjmp(info->jmp, info->error_code);
}
