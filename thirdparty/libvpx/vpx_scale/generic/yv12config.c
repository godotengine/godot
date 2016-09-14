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

#include "vpx_scale/yv12config.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

/****************************************************************************
*  Exports
****************************************************************************/

/****************************************************************************
 *
 ****************************************************************************/
#define yv12_align_addr(addr, align) \
    (void*)(((size_t)(addr) + ((align) - 1)) & (size_t)-(align))

int
vp8_yv12_de_alloc_frame_buffer(YV12_BUFFER_CONFIG *ybf) {
  if (ybf) {
    // If libvpx is using frame buffer callbacks then buffer_alloc_sz must
    // not be set.
    if (ybf->buffer_alloc_sz > 0) {
      vpx_free(ybf->buffer_alloc);
    }

    /* buffer_alloc isn't accessed by most functions.  Rather y_buffer,
      u_buffer and v_buffer point to buffer_alloc and are used.  Clear out
      all of this so that a freed pointer isn't inadvertently used */
    memset(ybf, 0, sizeof(YV12_BUFFER_CONFIG));
  } else {
    return -1;
  }

  return 0;
}

int vp8_yv12_realloc_frame_buffer(YV12_BUFFER_CONFIG *ybf,
                                  int width, int height, int border) {
  if (ybf) {
    int aligned_width = (width + 15) & ~15;
    int aligned_height = (height + 15) & ~15;
    int y_stride = ((aligned_width + 2 * border) + 31) & ~31;
    int yplane_size = (aligned_height + 2 * border) * y_stride;
    int uv_width = aligned_width >> 1;
    int uv_height = aligned_height >> 1;
    /** There is currently a bunch of code which assumes
      *  uv_stride == y_stride/2, so enforce this here. */
    int uv_stride = y_stride >> 1;
    int uvplane_size = (uv_height + border) * uv_stride;
    const int frame_size = yplane_size + 2 * uvplane_size;

    if (!ybf->buffer_alloc) {
      ybf->buffer_alloc = (uint8_t *)vpx_memalign(32, frame_size);
      ybf->buffer_alloc_sz = frame_size;
    }

    if (!ybf->buffer_alloc || ybf->buffer_alloc_sz < frame_size)
      return -1;

    /* Only support allocating buffers that have a border that's a multiple
     * of 32. The border restriction is required to get 16-byte alignment of
     * the start of the chroma rows without introducing an arbitrary gap
     * between planes, which would break the semantics of things like
     * vpx_img_set_rect(). */
    if (border & 0x1f)
      return -3;

    ybf->y_crop_width = width;
    ybf->y_crop_height = height;
    ybf->y_width  = aligned_width;
    ybf->y_height = aligned_height;
    ybf->y_stride = y_stride;

    ybf->uv_crop_width = (width + 1) / 2;
    ybf->uv_crop_height = (height + 1) / 2;
    ybf->uv_width = uv_width;
    ybf->uv_height = uv_height;
    ybf->uv_stride = uv_stride;

    ybf->alpha_width = 0;
    ybf->alpha_height = 0;
    ybf->alpha_stride = 0;

    ybf->border = border;
    ybf->frame_size = frame_size;

    ybf->y_buffer = ybf->buffer_alloc + (border * y_stride) + border;
    ybf->u_buffer = ybf->buffer_alloc + yplane_size + (border / 2  * uv_stride) + border / 2;
    ybf->v_buffer = ybf->buffer_alloc + yplane_size + uvplane_size + (border / 2  * uv_stride) + border / 2;
    ybf->alpha_buffer = NULL;

    ybf->corrupted = 0; /* assume not currupted by errors */
    return 0;
  }
  return -2;
}

int vp8_yv12_alloc_frame_buffer(YV12_BUFFER_CONFIG *ybf,
                                int width, int height, int border) {
  if (ybf) {
    vp8_yv12_de_alloc_frame_buffer(ybf);
    return vp8_yv12_realloc_frame_buffer(ybf, width, height, border);
  }
  return -2;
}

#if CONFIG_VP9
// TODO(jkoleszar): Maybe replace this with struct vpx_image

int vpx_free_frame_buffer(YV12_BUFFER_CONFIG *ybf) {
  if (ybf) {
    if (ybf->buffer_alloc_sz > 0) {
      vpx_free(ybf->buffer_alloc);
    }

    /* buffer_alloc isn't accessed by most functions.  Rather y_buffer,
      u_buffer and v_buffer point to buffer_alloc and are used.  Clear out
      all of this so that a freed pointer isn't inadvertently used */
    memset(ybf, 0, sizeof(YV12_BUFFER_CONFIG));
  } else {
    return -1;
  }

  return 0;
}

int vpx_realloc_frame_buffer(YV12_BUFFER_CONFIG *ybf,
                             int width, int height,
                             int ss_x, int ss_y,
#if CONFIG_VP9_HIGHBITDEPTH
                             int use_highbitdepth,
#endif
                             int border,
                             int byte_alignment,
                             vpx_codec_frame_buffer_t *fb,
                             vpx_get_frame_buffer_cb_fn_t cb,
                             void *cb_priv) {
  if (ybf) {
    const int vp9_byte_align = (byte_alignment == 0) ? 1 : byte_alignment;
    const int aligned_width = (width + 7) & ~7;
    const int aligned_height = (height + 7) & ~7;
    const int y_stride = ((aligned_width + 2 * border) + 31) & ~31;
    const uint64_t yplane_size = (aligned_height + 2 * border) *
                                 (uint64_t)y_stride + byte_alignment;
    const int uv_width = aligned_width >> ss_x;
    const int uv_height = aligned_height >> ss_y;
    const int uv_stride = y_stride >> ss_x;
    const int uv_border_w = border >> ss_x;
    const int uv_border_h = border >> ss_y;
    const uint64_t uvplane_size = (uv_height + 2 * uv_border_h) *
                                  (uint64_t)uv_stride + byte_alignment;

#if CONFIG_VP9_HIGHBITDEPTH
    const uint64_t frame_size =
        (1 + use_highbitdepth) * (yplane_size + 2 * uvplane_size);
#else
    const uint64_t frame_size = yplane_size + 2 * uvplane_size;
#endif  // CONFIG_VP9_HIGHBITDEPTH

    uint8_t *buf = NULL;

    if (cb != NULL) {
      const int align_addr_extra_size = 31;
      const uint64_t external_frame_size = frame_size + align_addr_extra_size;

      assert(fb != NULL);

      if (external_frame_size != (size_t)external_frame_size)
        return -1;

      // Allocation to hold larger frame, or first allocation.
      if (cb(cb_priv, (size_t)external_frame_size, fb) < 0)
        return -1;

      if (fb->data == NULL || fb->size < external_frame_size)
        return -1;

      ybf->buffer_alloc = (uint8_t *)yv12_align_addr(fb->data, 32);

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
      // This memset is needed for fixing the issue of using uninitialized
      // value in msan test. It will cause a perf loss, so only do this for
      // msan test.
      memset(ybf->buffer_alloc, 0, (int)frame_size);
#endif
#endif
    } else if (frame_size > (size_t)ybf->buffer_alloc_sz) {
      // Allocation to hold larger frame, or first allocation.
      vpx_free(ybf->buffer_alloc);
      ybf->buffer_alloc = NULL;

      if (frame_size != (size_t)frame_size)
        return -1;

      ybf->buffer_alloc = (uint8_t *)vpx_memalign(32, (size_t)frame_size);
      if (!ybf->buffer_alloc)
        return -1;

      ybf->buffer_alloc_sz = (int)frame_size;

      // This memset is needed for fixing valgrind error from C loop filter
      // due to access uninitialized memory in frame border. It could be
      // removed if border is totally removed.
      memset(ybf->buffer_alloc, 0, ybf->buffer_alloc_sz);
    }

    /* Only support allocating buffers that have a border that's a multiple
     * of 32. The border restriction is required to get 16-byte alignment of
     * the start of the chroma rows without introducing an arbitrary gap
     * between planes, which would break the semantics of things like
     * vpx_img_set_rect(). */
    if (border & 0x1f)
      return -3;

    ybf->y_crop_width = width;
    ybf->y_crop_height = height;
    ybf->y_width  = aligned_width;
    ybf->y_height = aligned_height;
    ybf->y_stride = y_stride;

    ybf->uv_crop_width = (width + ss_x) >> ss_x;
    ybf->uv_crop_height = (height + ss_y) >> ss_y;
    ybf->uv_width = uv_width;
    ybf->uv_height = uv_height;
    ybf->uv_stride = uv_stride;

    ybf->border = border;
    ybf->frame_size = (int)frame_size;
    ybf->subsampling_x = ss_x;
    ybf->subsampling_y = ss_y;

    buf = ybf->buffer_alloc;
#if CONFIG_VP9_HIGHBITDEPTH
    if (use_highbitdepth) {
      // Store uint16 addresses when using 16bit framebuffers
      buf = CONVERT_TO_BYTEPTR(ybf->buffer_alloc);
      ybf->flags = YV12_FLAG_HIGHBITDEPTH;
    } else {
      ybf->flags = 0;
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH

    ybf->y_buffer = (uint8_t *)yv12_align_addr(
        buf + (border * y_stride) + border, vp9_byte_align);
    ybf->u_buffer = (uint8_t *)yv12_align_addr(
        buf + yplane_size + (uv_border_h * uv_stride) + uv_border_w,
        vp9_byte_align);
    ybf->v_buffer = (uint8_t *)yv12_align_addr(
        buf + yplane_size + uvplane_size + (uv_border_h * uv_stride) +
        uv_border_w, vp9_byte_align);

    ybf->corrupted = 0; /* assume not corrupted by errors */
    return 0;
  }
  return -2;
}

int vpx_alloc_frame_buffer(YV12_BUFFER_CONFIG *ybf,
                           int width, int height,
                           int ss_x, int ss_y,
#if CONFIG_VP9_HIGHBITDEPTH
                           int use_highbitdepth,
#endif
                           int border,
                           int byte_alignment) {
  if (ybf) {
    vpx_free_frame_buffer(ybf);
    return vpx_realloc_frame_buffer(ybf, width, height, ss_x, ss_y,
#if CONFIG_VP9_HIGHBITDEPTH
                                    use_highbitdepth,
#endif
                                    border, byte_alignment, NULL, NULL, NULL);
  }
  return -2;
}
#endif
