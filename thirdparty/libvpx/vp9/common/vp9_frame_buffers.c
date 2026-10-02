/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "vp9/common/vp9_frame_buffers.h"
#include "vpx_mem/vpx_mem.h"

int vp9_alloc_internal_frame_buffers(InternalFrameBufferList *list) {
  const int num_buffers = VP9_MAXIMUM_REF_BUFFERS + VPX_MAXIMUM_WORK_BUFFERS;
  assert(list != NULL);
  vp9_free_internal_frame_buffers(list);

  list->int_fb =
      (InternalFrameBuffer *)vpx_calloc(num_buffers, sizeof(*list->int_fb));
  if (list->int_fb) {
    list->num_internal_frame_buffers = num_buffers;
    return 0;
  }
  return -1;
}

void vp9_free_internal_frame_buffers(InternalFrameBufferList *list) {
  int i;

  assert(list != NULL);

  for (i = 0; i < list->num_internal_frame_buffers; ++i) {
    vpx_free(list->int_fb[i].data);
    list->int_fb[i].data = NULL;
  }
  vpx_free(list->int_fb);
  list->int_fb = NULL;
  list->num_internal_frame_buffers = 0;
}

int vp9_get_frame_buffer(void *cb_priv, size_t min_size,
                         vpx_codec_frame_buffer_t *fb) {
  int i;
  InternalFrameBufferList *const int_fb_list =
      (InternalFrameBufferList *)cb_priv;
  if (int_fb_list == NULL) return -1;

  // Find a free frame buffer.
  for (i = 0; i < int_fb_list->num_internal_frame_buffers; ++i) {
    if (!int_fb_list->int_fb[i].in_use) break;
  }

  if (i == int_fb_list->num_internal_frame_buffers) return -1;

  if (int_fb_list->int_fb[i].size < min_size) {
    vpx_free(int_fb_list->int_fb[i].data);
    // The data must be zeroed to fix a valgrind error from the C loop filter
    // due to access uninitialized memory in frame border. It could be
    // skipped if border were totally removed.
    int_fb_list->int_fb[i].data = (uint8_t *)vpx_calloc(1, min_size);
    if (!int_fb_list->int_fb[i].data) return -1;
    int_fb_list->int_fb[i].size = min_size;
  }

  fb->data = int_fb_list->int_fb[i].data;
  fb->size = int_fb_list->int_fb[i].size;
  int_fb_list->int_fb[i].in_use = 1;

  // Set the frame buffer's private data to point at the internal frame buffer.
  fb->priv = &int_fb_list->int_fb[i];
  return 0;
}

int vp9_release_frame_buffer(void *cb_priv, vpx_codec_frame_buffer_t *fb) {
  InternalFrameBuffer *const int_fb = (InternalFrameBuffer *)fb->priv;
  (void)cb_priv;
  if (int_fb) int_fb->in_use = 0;
  return 0;
}
