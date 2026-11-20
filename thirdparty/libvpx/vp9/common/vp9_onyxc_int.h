/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_ONYXC_INT_H_
#define VPX_VP9_COMMON_VP9_ONYXC_INT_H_

#include "./vpx_config.h"
#include "vpx/internal/vpx_codec_internal.h"
#include "./vp9_rtcd.h"
#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_loopfilter.h"
#include "vp9/common/vp9_entropymv.h"
#include "vp9/common/vp9_entropy.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_frame_buffers.h"
#include "vp9/common/vp9_quant_common.h"
#include "vp9/common/vp9_tile_common.h"

#if CONFIG_VP9_POSTPROC
#include "vp9/common/vp9_postproc.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define REFS_PER_FRAME 3

#define REF_FRAMES_LOG2 3
#define REF_FRAMES (1 << REF_FRAMES_LOG2)

// 1 scratch frame for the new frame, REFS_PER_FRAME for scaled references on
// the encoder.
#define FRAME_BUFFERS (REF_FRAMES + 1 + REFS_PER_FRAME)

#define FRAME_CONTEXTS_LOG2 2
#define FRAME_CONTEXTS (1 << FRAME_CONTEXTS_LOG2)

#define NUM_PING_PONG_BUFFERS 2

extern const struct {
  PARTITION_CONTEXT above;
  PARTITION_CONTEXT left;
} partition_context_lookup[BLOCK_SIZES];

typedef enum {
  SINGLE_REFERENCE = 0,
  COMPOUND_REFERENCE = 1,
  REFERENCE_MODE_SELECT = 2,
  REFERENCE_MODES = 3,
} REFERENCE_MODE;

typedef struct {
  int_mv mv[2];
  MV_REFERENCE_FRAME ref_frame[2];
} MV_REF;

typedef struct {
  int ref_count;
  MV_REF *mvs;
  int mi_rows;
  int mi_cols;
  uint8_t released;

  // Note that frame_index/frame_coding_index are only set by set_frame_index()
  // on the encoder side.

  // TODO(angiebird): Set frame_index/frame_coding_index on the decoder side
  // properly.
  int frame_index;         // Display order in the video, it's equivalent to the
                           // show_idx defined in EncodeFrameInfo.
  int frame_coding_index;  // The coding order (starting from zero) of this
                           // frame.
  vpx_codec_frame_buffer_t raw_frame_buffer;
  YV12_BUFFER_CONFIG buf;
} RefCntBuffer;

typedef struct BufferPool {
  // Private data associated with the frame buffer callbacks.
  void *cb_priv;

  vpx_get_frame_buffer_cb_fn_t get_fb_cb;
  vpx_release_frame_buffer_cb_fn_t release_fb_cb;

  RefCntBuffer frame_bufs[FRAME_BUFFERS];

  // Frame buffers allocated internally by the codec.
  InternalFrameBufferList int_frame_buffers;
} BufferPool;

typedef struct VP9Common {
  struct vpx_internal_error_info error;
  vpx_color_space_t color_space;
  vpx_color_range_t color_range;
  int width;
  int height;
  int render_width;
  int render_height;
  int last_width;
  int last_height;

  // TODO(jkoleszar): this implies chroma ss right now, but could vary per
  // plane. Revisit as part of the future change to YV12_BUFFER_CONFIG to
  // support additional planes.
  int subsampling_x;
  int subsampling_y;

#if CONFIG_VP9_HIGHBITDEPTH
  int use_highbitdepth;  // Marks if we need to use 16bit frame buffers.
#endif

  YV12_BUFFER_CONFIG *frame_to_show;
  RefCntBuffer *prev_frame;

  // TODO(hkuang): Combine this with cur_buf in macroblockd.
  RefCntBuffer *cur_frame;

  int ref_frame_map[REF_FRAMES]; /* maps fb_idx to reference slot */

  // Prepare ref_frame_map for the next frame.
  // Only used in frame parallel decode.
  int next_ref_frame_map[REF_FRAMES];

  // TODO(jkoleszar): could expand active_ref_idx to 4, with 0 as intra, and
  // roll new_fb_idx into it.

  // Each frame can reference REFS_PER_FRAME buffers
  RefBuffer frame_refs[REFS_PER_FRAME];

  int new_fb_idx;

  int cur_show_frame_fb_idx;

#if CONFIG_VP9_POSTPROC
  YV12_BUFFER_CONFIG post_proc_buffer;
  YV12_BUFFER_CONFIG post_proc_buffer_int;
#endif

  FRAME_TYPE last_frame_type; /* last frame's frame type for motion search.*/
  FRAME_TYPE frame_type;

  int show_frame;
  int last_show_frame;
  int show_existing_frame;

  // Flag signaling that the frame is encoded using only INTRA modes.
  uint8_t intra_only;
  uint8_t last_intra_only;

  int allow_high_precision_mv;

  // Flag signaling that the frame context should be reset to default values.
  // 0 or 1 implies don't reset, 2 reset just the context specified in the
  // frame header, 3 reset all contexts.
  int reset_frame_context;

  // MBs, mb_rows/cols is in 16-pixel units; mi_rows/cols is in
  // MODE_INFO (8-pixel) units.
  int MBs;
  int mb_rows, mi_rows;
  int mb_cols, mi_cols;
  int mi_stride;

  /* profile settings */
  TX_MODE tx_mode;

  int base_qindex;
  int y_dc_delta_q;
  int uv_dc_delta_q;
  int uv_ac_delta_q;
  int16_t y_dequant[MAX_SEGMENTS][2];
  int16_t uv_dequant[MAX_SEGMENTS][2];

  /* We allocate a MODE_INFO struct for each macroblock, together with
     an extra row on top and column on the left to simplify prediction. */
  int mi_alloc_size;
  MODE_INFO *mip; /* Base of allocated array */
  MODE_INFO *mi;  /* Corresponds to upper left visible macroblock */

  // TODO(agrange): Move prev_mi into encoder structure.
  // prev_mip and prev_mi will only be allocated in VP9 encoder.
  MODE_INFO *prev_mip; /* MODE_INFO array 'mip' from last decoded frame */
  MODE_INFO *prev_mi;  /* 'mi' from last frame (points into prev_mip) */

  // Separate mi functions between encoder and decoder.
  int (*alloc_mi)(struct VP9Common *cm, int mi_size);
  void (*free_mi)(struct VP9Common *cm);
  void (*setup_mi)(struct VP9Common *cm);

  // Grid of pointers to 8x8 MODE_INFO structs.  Any 8x8 not in the visible
  // area will be NULL.
  MODE_INFO **mi_grid_base;
  MODE_INFO **mi_grid_visible;
  MODE_INFO **prev_mi_grid_base;
  MODE_INFO **prev_mi_grid_visible;

  // Whether to use previous frame's motion vectors for prediction.
  int use_prev_frame_mvs;

  // Persistent mb segment id map used in prediction.
  int seg_map_idx;
  int prev_seg_map_idx;

  uint8_t *seg_map_array[NUM_PING_PONG_BUFFERS];
  uint8_t *last_frame_seg_map;
  uint8_t *current_frame_seg_map;
  int seg_map_alloc_size;

  INTERP_FILTER interp_filter;

  loop_filter_info_n lf_info;

  int refresh_frame_context; /* Two state 0 = NO, 1 = YES */

  int ref_frame_sign_bias[MAX_REF_FRAMES]; /* Two state 0, 1 */

  struct loopfilter lf;
  struct segmentation seg;

  // Context probabilities for reference frame prediction
  MV_REFERENCE_FRAME comp_fixed_ref;
  MV_REFERENCE_FRAME comp_var_ref[2];
  REFERENCE_MODE reference_mode;

  FRAME_CONTEXT *fc;              /* this frame entropy */
  FRAME_CONTEXT *frame_contexts;  // FRAME_CONTEXTS
  unsigned int frame_context_idx; /* Context to use/update */
  FRAME_COUNTS counts;

  // TODO(angiebird): current_video_frame/current_frame_coding_index into a
  // structure
  unsigned int current_video_frame;
  // Each show or no show frame is assigned with a coding index based on its
  // coding order (starting from zero).

  // Current frame's coding index.
  int current_frame_coding_index;
  BITSTREAM_PROFILE profile;

  // VPX_BITS_8 in profile 0 or 1, VPX_BITS_10 or VPX_BITS_12 in profile 2 or 3.
  vpx_bit_depth_t bit_depth;
  vpx_bit_depth_t dequant_bit_depth;  // bit_depth of current dequantizer

#if CONFIG_VP9_POSTPROC
  struct postproc_state postproc_state;
#endif

  int error_resilient_mode;
  int frame_parallel_decoding_mode;

  int log2_tile_cols, log2_tile_rows;
  int byte_alignment;
  int skip_loop_filter;

  // External BufferPool passed from outside.
  BufferPool *buffer_pool;

  PARTITION_CONTEXT *above_seg_context;
  ENTROPY_CONTEXT *above_context;
  int above_context_alloc_cols;

  int lf_row;
} VP9_COMMON;

static INLINE void init_frame_indexes(VP9_COMMON *cm) {
  cm->current_video_frame = 0;
  cm->current_frame_coding_index = 0;
}

static INLINE void update_frame_indexes(VP9_COMMON *cm, int show_frame) {
  if (show_frame) {
    // Don't increment frame counters if this was an altref buffer
    // update not a real frame
    ++cm->current_video_frame;
  }
  ++cm->current_frame_coding_index;
}

typedef struct {
  int frame_width;
  int frame_height;
  int render_frame_width;
  int render_frame_height;
  int mi_rows;
  int mi_cols;
  int mb_rows;
  int mb_cols;
  int num_mbs;
  vpx_bit_depth_t bit_depth;
} FRAME_INFO;

static INLINE void init_frame_info(FRAME_INFO *frame_info,
                                   const VP9_COMMON *cm) {
  frame_info->frame_width = cm->width;
  frame_info->frame_height = cm->height;
  frame_info->render_frame_width = cm->render_width;
  frame_info->render_frame_height = cm->render_height;
  frame_info->mi_cols = cm->mi_cols;
  frame_info->mi_rows = cm->mi_rows;
  frame_info->mb_cols = cm->mb_cols;
  frame_info->mb_rows = cm->mb_rows;
  frame_info->num_mbs = cm->MBs;
  frame_info->bit_depth = cm->bit_depth;
  // TODO(angiebird): Figure out how to get subsampling_x/y here
}

static INLINE YV12_BUFFER_CONFIG *get_buf_frame(VP9_COMMON *cm, int index) {
  if (index < 0 || index >= FRAME_BUFFERS) return NULL;
  if (cm->error.error_code != VPX_CODEC_OK) return NULL;
  return &cm->buffer_pool->frame_bufs[index].buf;
}

static INLINE YV12_BUFFER_CONFIG *get_ref_frame(VP9_COMMON *cm, int index) {
  if (index < 0 || index >= REF_FRAMES) return NULL;
  if (cm->ref_frame_map[index] < 0) return NULL;
  assert(cm->ref_frame_map[index] < FRAME_BUFFERS);
  return &cm->buffer_pool->frame_bufs[cm->ref_frame_map[index]].buf;
}

static INLINE YV12_BUFFER_CONFIG *get_frame_new_buffer(VP9_COMMON *cm) {
  return &cm->buffer_pool->frame_bufs[cm->new_fb_idx].buf;
}

static INLINE int get_free_fb(VP9_COMMON *cm) {
  RefCntBuffer *const frame_bufs = cm->buffer_pool->frame_bufs;
  int i;

  for (i = 0; i < FRAME_BUFFERS; ++i)
    if (frame_bufs[i].ref_count == 0) break;

  if (i != FRAME_BUFFERS) {
    frame_bufs[i].ref_count = 1;
  } else {
    // Reset i to be INVALID_IDX to indicate no free buffer found.
    i = INVALID_IDX;
  }

  return i;
}

static INLINE void ref_cnt_fb(RefCntBuffer *bufs, int *idx, int new_idx) {
  const int ref_index = *idx;

  if (ref_index >= 0 && bufs[ref_index].ref_count > 0)
    bufs[ref_index].ref_count--;

  *idx = new_idx;

  bufs[new_idx].ref_count++;
}

static INLINE int mi_cols_aligned_to_sb(int n_mis) {
  return ALIGN_POWER_OF_TWO(n_mis, MI_BLOCK_SIZE_LOG2);
}

static INLINE int frame_is_intra_only(const VP9_COMMON *const cm) {
  return cm->frame_type == KEY_FRAME || cm->intra_only;
}

static INLINE void set_partition_probs(const VP9_COMMON *const cm,
                                       MACROBLOCKD *const xd) {
  xd->partition_probs =
      frame_is_intra_only(cm)
          ? &vp9_kf_partition_probs[0]
          : (const vpx_prob(*)[PARTITION_TYPES - 1]) cm->fc->partition_prob;
}

static INLINE void vp9_init_macroblockd(VP9_COMMON *cm, MACROBLOCKD *xd,
                                        tran_low_t *dqcoeff) {
  int i;

  for (i = 0; i < MAX_MB_PLANE; ++i) {
    xd->plane[i].dqcoeff = dqcoeff;
    xd->above_context[i] =
        cm->above_context +
        i * sizeof(*cm->above_context) * 2 * mi_cols_aligned_to_sb(cm->mi_cols);

    if (get_plane_type(i) == PLANE_TYPE_Y) {
      memcpy(xd->plane[i].seg_dequant, cm->y_dequant, sizeof(cm->y_dequant));
    } else {
      memcpy(xd->plane[i].seg_dequant, cm->uv_dequant, sizeof(cm->uv_dequant));
    }
    xd->fc = cm->fc;
  }

  xd->above_seg_context = cm->above_seg_context;
  xd->mi_stride = cm->mi_stride;
  xd->error_info = &cm->error;

  set_partition_probs(cm, xd);
}

static INLINE const vpx_prob *get_partition_probs(const MACROBLOCKD *xd,
                                                  int ctx) {
  return xd->partition_probs[ctx];
}

static INLINE void set_skip_context(MACROBLOCKD *xd, int mi_row, int mi_col) {
  const int above_idx = mi_col * 2;
  const int left_idx = (mi_row * 2) & 15;
  int i;
  for (i = 0; i < MAX_MB_PLANE; ++i) {
    struct macroblockd_plane *const pd = &xd->plane[i];
    pd->above_context = &xd->above_context[i][above_idx >> pd->subsampling_x];
    pd->left_context = &xd->left_context[i][left_idx >> pd->subsampling_y];
  }
}

static INLINE int calc_mi_size(int len) {
  // len is in mi units.
  return len + MI_BLOCK_SIZE;
}

static INLINE void set_mi_row_col(MACROBLOCKD *xd, const TileInfo *const tile,
                                  int mi_row, int bh, int mi_col, int bw,
                                  int mi_rows, int mi_cols) {
  xd->mb_to_top_edge = -((mi_row * MI_SIZE) * 8);
  xd->mb_to_bottom_edge = ((mi_rows - bh - mi_row) * MI_SIZE) * 8;
  xd->mb_to_left_edge = -((mi_col * MI_SIZE) * 8);
  xd->mb_to_right_edge = ((mi_cols - bw - mi_col) * MI_SIZE) * 8;

  // Are edges available for intra prediction?
  xd->above_mi = (mi_row != 0) ? xd->mi[-xd->mi_stride] : NULL;
  xd->left_mi = (mi_col > tile->mi_col_start) ? xd->mi[-1] : NULL;
}

static INLINE void update_partition_context(MACROBLOCKD *xd, int mi_row,
                                            int mi_col, BLOCK_SIZE subsize,
                                            BLOCK_SIZE bsize) {
  PARTITION_CONTEXT *const above_ctx = xd->above_seg_context + mi_col;
  PARTITION_CONTEXT *const left_ctx = xd->left_seg_context + (mi_row & MI_MASK);

  // num_4x4_blocks_wide_lookup[bsize] / 2
  const int bs = num_8x8_blocks_wide_lookup[bsize];

  // update the partition context at the end notes. set partition bits
  // of block sizes larger than the current one to be one, and partition
  // bits of smaller block sizes to be zero.
  memset(above_ctx, partition_context_lookup[subsize].above, bs);
  memset(left_ctx, partition_context_lookup[subsize].left, bs);
}

static INLINE int partition_plane_context(const MACROBLOCKD *xd, int mi_row,
                                          int mi_col, BLOCK_SIZE bsize) {
  const PARTITION_CONTEXT *above_ctx = xd->above_seg_context + mi_col;
  const PARTITION_CONTEXT *left_ctx = xd->left_seg_context + (mi_row & MI_MASK);
  const int bsl = mi_width_log2_lookup[bsize];
  int above = (*above_ctx >> bsl) & 1, left = (*left_ctx >> bsl) & 1;

  assert(b_width_log2_lookup[bsize] == b_height_log2_lookup[bsize]);
  assert(bsl >= 0);

  return (left * 2 + above) + bsl * PARTITION_PLOFFSET;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_ONYXC_INT_H_
