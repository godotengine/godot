/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "vpx_mem/vpx_mem.h"

#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_entropymv.h"
#include "vp9/common/vp9_onyxc_int.h"

void vp9_set_mi_size(int *mi_rows, int *mi_cols, int *mi_stride, int width,
                     int height) {
  const int aligned_width = ALIGN_POWER_OF_TWO(width, MI_SIZE_LOG2);
  const int aligned_height = ALIGN_POWER_OF_TWO(height, MI_SIZE_LOG2);
  *mi_cols = aligned_width >> MI_SIZE_LOG2;
  *mi_rows = aligned_height >> MI_SIZE_LOG2;
  *mi_stride = calc_mi_size(*mi_cols);
}

void vp9_set_mb_size(int *mb_rows, int *mb_cols, int *mb_num, int mi_rows,
                     int mi_cols) {
  *mb_cols = (mi_cols + 1) >> 1;
  *mb_rows = (mi_rows + 1) >> 1;
  *mb_num = (*mb_rows) * (*mb_cols);
}

void vp9_set_mb_mi(VP9_COMMON *cm, int width, int height) {
  vp9_set_mi_size(&cm->mi_rows, &cm->mi_cols, &cm->mi_stride, width, height);
  vp9_set_mb_size(&cm->mb_rows, &cm->mb_cols, &cm->MBs, cm->mi_rows,
                  cm->mi_cols);
}

static int alloc_seg_map(VP9_COMMON *cm, int seg_map_size) {
  int i;

  for (i = 0; i < NUM_PING_PONG_BUFFERS; ++i) {
    cm->seg_map_array[i] = (uint8_t *)vpx_calloc(seg_map_size, 1);
    if (cm->seg_map_array[i] == NULL) return 1;
  }
  cm->seg_map_alloc_size = seg_map_size;

  // Init the index.
  cm->seg_map_idx = 0;
  cm->prev_seg_map_idx = 1;

  cm->current_frame_seg_map = cm->seg_map_array[cm->seg_map_idx];
  cm->last_frame_seg_map = cm->seg_map_array[cm->prev_seg_map_idx];

  return 0;
}

static void free_seg_map(VP9_COMMON *cm) {
  int i;

  for (i = 0; i < NUM_PING_PONG_BUFFERS; ++i) {
    vpx_free(cm->seg_map_array[i]);
    cm->seg_map_array[i] = NULL;
  }
  cm->seg_map_alloc_size = 0;

  cm->current_frame_seg_map = NULL;
  cm->last_frame_seg_map = NULL;
}

void vp9_free_ref_frame_buffers(BufferPool *pool) {
  int i;

  if (!pool) return;

  for (i = 0; i < FRAME_BUFFERS; ++i) {
    if (!pool->frame_bufs[i].released &&
        pool->frame_bufs[i].raw_frame_buffer.data != NULL) {
      pool->release_fb_cb(pool->cb_priv, &pool->frame_bufs[i].raw_frame_buffer);
      pool->frame_bufs[i].ref_count = 0;
      pool->frame_bufs[i].released = 1;
    }
    vpx_free(pool->frame_bufs[i].mvs);
    pool->frame_bufs[i].mvs = NULL;
    vpx_free_frame_buffer(&pool->frame_bufs[i].buf);
  }
}

void vp9_free_postproc_buffers(VP9_COMMON *cm) {
#if CONFIG_VP9_POSTPROC
  vpx_free_frame_buffer(&cm->post_proc_buffer);
  vpx_free_frame_buffer(&cm->post_proc_buffer_int);
  vpx_free(cm->postproc_state.limits);
  cm->postproc_state.limits = NULL;
  vpx_free(cm->postproc_state.generated_noise);
  cm->postproc_state.generated_noise = NULL;
#else
  (void)cm;
#endif
}

void vp9_free_context_buffers(VP9_COMMON *cm) {
  if (cm->free_mi) cm->free_mi(cm);
  free_seg_map(cm);
  vpx_free(cm->above_context);
  cm->above_context = NULL;
  vpx_free(cm->above_seg_context);
  cm->above_seg_context = NULL;
  cm->above_context_alloc_cols = 0;
  vpx_free(cm->lf.lfm);
  cm->lf.lfm = NULL;
}

int vp9_alloc_loop_filter(VP9_COMMON *cm) {
  vpx_free(cm->lf.lfm);
  // Each lfm holds bit masks for all the 8x8 blocks in a 64x64 region.  The
  // stride and rows are rounded up / truncated to a multiple of 8.
  cm->lf.lfm_stride = (cm->mi_cols + (MI_BLOCK_SIZE - 1)) >> 3;
  cm->lf.lfm = (LOOP_FILTER_MASK *)vpx_calloc(
      ((cm->mi_rows + (MI_BLOCK_SIZE - 1)) >> 3) * cm->lf.lfm_stride,
      sizeof(*cm->lf.lfm));
  if (!cm->lf.lfm) return 1;
  return 0;
}

int vp9_alloc_context_buffers(VP9_COMMON *cm, int width, int height) {
  int new_mi_size;

  vp9_set_mb_mi(cm, width, height);
  new_mi_size = cm->mi_stride * calc_mi_size(cm->mi_rows);
  if (cm->mi_alloc_size < new_mi_size) {
    cm->free_mi(cm);
    if (cm->alloc_mi(cm, new_mi_size)) goto fail;
  }
  if (cm->above_context_alloc_cols < cm->mi_cols) {
    vpx_free(cm->above_context);
    cm->above_context = (ENTROPY_CONTEXT *)vpx_calloc(
        2 * mi_cols_aligned_to_sb(cm->mi_cols) * MAX_MB_PLANE,
        sizeof(*cm->above_context));
    if (!cm->above_context) goto fail;

    vpx_free(cm->above_seg_context);
    cm->above_seg_context = (PARTITION_CONTEXT *)vpx_calloc(
        mi_cols_aligned_to_sb(cm->mi_cols), sizeof(*cm->above_seg_context));
    if (!cm->above_seg_context) goto fail;
    cm->above_context_alloc_cols = cm->mi_cols;
  }

  if (cm->seg_map_alloc_size < cm->mi_rows * cm->mi_cols) {
    // Create the segmentation map structure and set to 0.
    free_seg_map(cm);
    if (alloc_seg_map(cm, cm->mi_rows * cm->mi_cols)) goto fail;
  }

  if (vp9_alloc_loop_filter(cm)) goto fail;

  return 0;

fail:
  // clear the mi_* values to force a realloc on resync
  vp9_set_mb_mi(cm, 0, 0);
  vp9_free_context_buffers(cm);
  return 1;
}

void vp9_remove_common(VP9_COMMON *cm) {
#if CONFIG_VP9_POSTPROC
  vp9_free_postproc_buffers(cm);
#endif
  vp9_free_context_buffers(cm);

  vpx_free(cm->fc);
  cm->fc = NULL;
  vpx_free(cm->frame_contexts);
  cm->frame_contexts = NULL;
}

void vp9_init_context_buffers(VP9_COMMON *cm) {
  cm->setup_mi(cm);
  if (cm->last_frame_seg_map)
    memset(cm->last_frame_seg_map, 0, cm->mi_rows * cm->mi_cols);
}

void vp9_swap_current_and_last_seg_map(VP9_COMMON *cm) {
  // Swap indices.
  const int tmp = cm->seg_map_idx;
  cm->seg_map_idx = cm->prev_seg_map_idx;
  cm->prev_seg_map_idx = tmp;

  cm->current_frame_seg_map = cm->seg_map_array[cm->seg_map_idx];
  cm->last_frame_seg_map = cm->seg_map_array[cm->prev_seg_map_idx];
}
