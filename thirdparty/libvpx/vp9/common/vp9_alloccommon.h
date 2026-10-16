/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_ALLOCCOMMON_H_
#define VPX_VP9_COMMON_VP9_ALLOCCOMMON_H_

#define INVALID_IDX (-1)  // Invalid buffer index.

#ifdef __cplusplus
extern "C" {
#endif

struct VP9Common;
struct BufferPool;

void vp9_remove_common(struct VP9Common *cm);

int vp9_alloc_loop_filter(struct VP9Common *cm);
int vp9_alloc_context_buffers(struct VP9Common *cm, int width, int height);
void vp9_init_context_buffers(struct VP9Common *cm);
void vp9_free_context_buffers(struct VP9Common *cm);

void vp9_free_ref_frame_buffers(struct BufferPool *pool);
void vp9_free_postproc_buffers(struct VP9Common *cm);

int vp9_alloc_state_buffers(struct VP9Common *cm, int width, int height);
void vp9_free_state_buffers(struct VP9Common *cm);

void vp9_set_mi_size(int *mi_rows, int *mi_cols, int *mi_stride, int width,
                     int height);
void vp9_set_mb_size(int *mb_rows, int *mb_cols, int *mb_num, int mi_rows,
                     int mi_cols);

void vp9_set_mb_mi(struct VP9Common *cm, int width, int height);

void vp9_swap_current_and_last_seg_map(struct VP9Common *cm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_ALLOCCOMMON_H_
