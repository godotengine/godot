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
#include "vpx_dsp_rtcd.h"
#include "vp8_rtcd.h"
#include "vpx_dsp/postproc.h"
#include "vpx_ports/system_state.h"
#include "vpx_scale_rtcd.h"
#include "vpx_scale/yv12config.h"
#include "postproc.h"
#include "common.h"
#include "vpx_scale/vpx_scale.h"
#include "systemdependent.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* clang-format off */
#define RGB_TO_YUV(t)                                     \
  (unsigned char)((0.257 * (float)(t >> 16)) +            \
                  (0.504 * (float)(t >> 8 & 0xff)) +      \
                  (0.098 * (float)(t & 0xff)) + 16),      \
  (unsigned char)(-(0.148 * (float)(t >> 16)) -           \
                  (0.291 * (float)(t >> 8 & 0xff)) +      \
                  (0.439 * (float)(t & 0xff)) + 128),     \
  (unsigned char)((0.439 * (float)(t >> 16)) -            \
                  (0.368 * (float)(t >> 8 & 0xff)) -      \
                  (0.071 * (float)(t & 0xff)) + 128)
/* clang-format on */

extern void vp8_blit_text(const char *msg, unsigned char *address,
                          const int pitch);
extern void vp8_blit_line(int x0, int x1, int y0, int y1, unsigned char *image,
                          const int pitch);
/***********************************************************************************************************
 */
#if CONFIG_POSTPROC
static int q2mbl(int x) {
  if (x < 20) x = 20;

  x = 50 + (x - 50) * 10 / 8;
  return x * x / 3;
}

static void vp8_de_mblock(YV12_BUFFER_CONFIG *post, int q) {
  vpx_mbpost_proc_across_ip(post->y_buffer, post->y_stride, post->y_height,
                            post->y_width, q2mbl(q));
  vpx_mbpost_proc_down(post->y_buffer, post->y_stride, post->y_height,
                       post->y_width, q2mbl(q));
}

void vp8_deblock(VP8_COMMON *cm, YV12_BUFFER_CONFIG *source,
                 YV12_BUFFER_CONFIG *post, int q) {
  double level = 6.0e-05 * q * q * q - .0067 * q * q + .306 * q + .0065;
  int ppl = (int)(level + .5);

  const MODE_INFO *mode_info_context = cm->mi;
  int mbr, mbc;

  /* The pixel thresholds are adjusted according to if or not the macroblock
   * is a skipped block.  */
  unsigned char *ylimits = cm->pp_limits_buffer;
  unsigned char *uvlimits = cm->pp_limits_buffer + 16 * cm->mb_cols;

  if (ppl > 0) {
    for (mbr = 0; mbr < cm->mb_rows; ++mbr) {
      unsigned char *ylptr = ylimits;
      unsigned char *uvlptr = uvlimits;
      for (mbc = 0; mbc < cm->mb_cols; ++mbc) {
        unsigned char mb_ppl;

        if (mode_info_context->mbmi.mb_skip_coeff) {
          mb_ppl = (unsigned char)ppl >> 1;
        } else {
          mb_ppl = (unsigned char)ppl;
        }

        memset(ylptr, mb_ppl, 16);
        memset(uvlptr, mb_ppl, 8);

        ylptr += 16;
        uvlptr += 8;
        mode_info_context++;
      }
      mode_info_context++;

      vpx_post_proc_down_and_across_mb_row(
          source->y_buffer + 16 * mbr * source->y_stride,
          post->y_buffer + 16 * mbr * post->y_stride, source->y_stride,
          post->y_stride, source->y_width, ylimits, 16);

      vpx_post_proc_down_and_across_mb_row(
          source->u_buffer + 8 * mbr * source->uv_stride,
          post->u_buffer + 8 * mbr * post->uv_stride, source->uv_stride,
          post->uv_stride, source->uv_width, uvlimits, 8);
      vpx_post_proc_down_and_across_mb_row(
          source->v_buffer + 8 * mbr * source->uv_stride,
          post->v_buffer + 8 * mbr * post->uv_stride, source->uv_stride,
          post->uv_stride, source->uv_width, uvlimits, 8);
    }
  } else {
    vp8_yv12_copy_frame(source, post);
  }
}

void vp8_de_noise(VP8_COMMON *cm, YV12_BUFFER_CONFIG *source, int q,
                  int uvfilter) {
  int mbr;
  double level = 6.0e-05 * q * q * q - .0067 * q * q + .306 * q + .0065;
  int ppl = (int)(level + .5);
  int mb_rows = cm->mb_rows;
  int mb_cols = cm->mb_cols;
  unsigned char *limits = cm->pp_limits_buffer;

  memset(limits, (unsigned char)ppl, 16 * mb_cols);

  /* TODO: The original code don't filter the 2 outer rows and columns. */
  for (mbr = 0; mbr < mb_rows; ++mbr) {
    vpx_post_proc_down_and_across_mb_row(
        source->y_buffer + 16 * mbr * source->y_stride,
        source->y_buffer + 16 * mbr * source->y_stride, source->y_stride,
        source->y_stride, source->y_width, limits, 16);
    if (uvfilter == 1) {
      vpx_post_proc_down_and_across_mb_row(
          source->u_buffer + 8 * mbr * source->uv_stride,
          source->u_buffer + 8 * mbr * source->uv_stride, source->uv_stride,
          source->uv_stride, source->uv_width, limits, 8);
      vpx_post_proc_down_and_across_mb_row(
          source->v_buffer + 8 * mbr * source->uv_stride,
          source->v_buffer + 8 * mbr * source->uv_stride, source->uv_stride,
          source->uv_stride, source->uv_width, limits, 8);
    }
  }
}
#endif  // CONFIG_POSTPROC

#if CONFIG_POSTPROC
int vp8_post_proc_frame(VP8_COMMON *oci, YV12_BUFFER_CONFIG *dest,
                        vp8_ppflags_t *ppflags) {
  int q = oci->filter_level * 10 / 6;
  int flags = ppflags->post_proc_flag;
  int deblock_level = ppflags->deblocking_level;
  int noise_level = ppflags->noise_level;

  if (!oci->frame_to_show) return -1;

  if (q > 63) q = 63;

  if (!flags) {
    *dest = *oci->frame_to_show;

    /* handle problem with extending borders */
    dest->y_width = oci->Width;
    dest->y_height = oci->Height;
    dest->uv_height = dest->y_height / 2;
    oci->postproc_state.last_base_qindex = oci->base_qindex;
    oci->postproc_state.last_frame_valid = 1;
    return 0;
  }
  if (flags & VP8D_ADDNOISE) {
    if (!oci->postproc_state.generated_noise) {
      oci->postproc_state.generated_noise = vpx_calloc(
          oci->Width + 256, sizeof(*oci->postproc_state.generated_noise));
      if (!oci->postproc_state.generated_noise) return 1;
    }
  }

  /* Allocate post_proc_buffer_int if needed */
  if ((flags & VP8D_MFQE) && !oci->post_proc_buffer_int_used) {
    if ((flags & VP8D_DEBLOCK) || (flags & VP8D_DEMACROBLOCK)) {
      int width = (oci->Width + 15) & ~15;
      int height = (oci->Height + 15) & ~15;

      if (vp8_yv12_alloc_frame_buffer(&oci->post_proc_buffer_int, width, height,
                                      VP8BORDERINPIXELS)) {
        vpx_internal_error(&oci->error, VPX_CODEC_MEM_ERROR,
                           "Failed to allocate MFQE framebuffer");
      }

      oci->post_proc_buffer_int_used = 1;

      /* insure that postproc is set to all 0's so that post proc
       * doesn't pull random data in from edge
       */
      memset((&oci->post_proc_buffer_int)->buffer_alloc, 128,
             (&oci->post_proc_buffer)->frame_size);
    }
  }

  vpx_clear_system_state();

  if ((flags & VP8D_MFQE) && oci->postproc_state.last_frame_valid &&
      oci->current_video_frame > 10 &&
      oci->postproc_state.last_base_qindex < 60 &&
      oci->base_qindex - oci->postproc_state.last_base_qindex >= 20) {
    vp8_multiframe_quality_enhance(oci);
    if (((flags & VP8D_DEBLOCK) || (flags & VP8D_DEMACROBLOCK)) &&
        oci->post_proc_buffer_int_used) {
      vp8_yv12_copy_frame(&oci->post_proc_buffer, &oci->post_proc_buffer_int);
      if (flags & VP8D_DEMACROBLOCK) {
        vp8_deblock(oci, &oci->post_proc_buffer_int, &oci->post_proc_buffer,
                    q + (deblock_level - 5) * 10);
        vp8_de_mblock(&oci->post_proc_buffer, q + (deblock_level - 5) * 10);
      } else if (flags & VP8D_DEBLOCK) {
        vp8_deblock(oci, &oci->post_proc_buffer_int, &oci->post_proc_buffer, q);
      }
    }
    /* Move partially towards the base q of the previous frame */
    oci->postproc_state.last_base_qindex =
        (3 * oci->postproc_state.last_base_qindex + oci->base_qindex) >> 2;
  } else if (flags & VP8D_DEMACROBLOCK) {
    vp8_deblock(oci, oci->frame_to_show, &oci->post_proc_buffer,
                q + (deblock_level - 5) * 10);
    vp8_de_mblock(&oci->post_proc_buffer, q + (deblock_level - 5) * 10);

    oci->postproc_state.last_base_qindex = oci->base_qindex;
  } else if (flags & VP8D_DEBLOCK) {
    vp8_deblock(oci, oci->frame_to_show, &oci->post_proc_buffer, q);
    oci->postproc_state.last_base_qindex = oci->base_qindex;
  } else {
    vp8_yv12_copy_frame(oci->frame_to_show, &oci->post_proc_buffer);
    oci->postproc_state.last_base_qindex = oci->base_qindex;
  }
  oci->postproc_state.last_frame_valid = 1;

  if (flags & VP8D_ADDNOISE) {
    if (oci->postproc_state.last_q != q ||
        oci->postproc_state.last_noise != noise_level) {
      double sigma;
      struct postproc_state *ppstate = &oci->postproc_state;
      vpx_clear_system_state();
      sigma = noise_level + .5 + .6 * q / 63.0;
      ppstate->clamp =
          vpx_setup_noise(sigma, ppstate->generated_noise, oci->Width + 256);
      ppstate->last_q = q;
      ppstate->last_noise = noise_level;
    }

    vpx_plane_add_noise(
        oci->post_proc_buffer.y_buffer, oci->postproc_state.generated_noise,
        oci->postproc_state.clamp, oci->postproc_state.clamp,
        oci->post_proc_buffer.y_width, oci->post_proc_buffer.y_height,
        oci->post_proc_buffer.y_stride);
  }

  *dest = oci->post_proc_buffer;

  /* handle problem with extending borders */
  dest->y_width = oci->Width;
  dest->y_height = oci->Height;
  dest->uv_height = dest->y_height / 2;
  return 0;
}
#endif
