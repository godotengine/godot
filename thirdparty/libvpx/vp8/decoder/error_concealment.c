/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>

#include "error_concealment.h"
#include "onyxd_int.h"
#include "decodemv.h"
#include "vpx_mem/vpx_mem.h"
#include "vp8/common/findnearmv.h"
#include "vp8/common/common.h"
#include "vpx_dsp/vpx_dsp_common.h"

#define FLOOR(x, q) ((x) & -(1 << (q)))

#define NUM_NEIGHBORS 20

typedef struct ec_position {
  int row;
  int col;
} EC_POS;

/*
 * Regenerate the table in Matlab with:
 * x = meshgrid((1:4), (1:4));
 * y = meshgrid((1:4), (1:4))';
 * W = round((1./(sqrt(x.^2 + y.^2))*2^7));
 * W(1,1) = 0;
 */
static const int weights_q7[5][5] = { { 0, 128, 64, 43, 32 },
                                      { 128, 91, 57, 40, 31 },
                                      { 64, 57, 45, 36, 29 },
                                      { 43, 40, 36, 30, 26 },
                                      { 32, 31, 29, 26, 23 } };

int vp8_alloc_overlap_lists(VP8D_COMP *pbi) {
  if (pbi->overlaps != NULL) {
    vpx_free(pbi->overlaps);
    pbi->overlaps = NULL;
  }

  pbi->overlaps =
      vpx_calloc(pbi->common.mb_rows * pbi->common.mb_cols, sizeof(MB_OVERLAP));

  if (pbi->overlaps == NULL) return -1;

  return 0;
}

void vp8_de_alloc_overlap_lists(VP8D_COMP *pbi) {
  vpx_free(pbi->overlaps);
  pbi->overlaps = NULL;
}

/* Inserts a new overlap area value to the list of overlaps of a block */
static void assign_overlap(OVERLAP_NODE *overlaps, union b_mode_info *bmi,
                           int overlap) {
  int i;
  if (overlap <= 0) return;
  /* Find and assign to the next empty overlap node in the list of overlaps.
   * Empty is defined as bmi == NULL */
  for (i = 0; i < MAX_OVERLAPS; ++i) {
    if (overlaps[i].bmi == NULL) {
      overlaps[i].bmi = bmi;
      overlaps[i].overlap = overlap;
      break;
    }
  }
}

/* Calculates the overlap area between two 4x4 squares, where the first
 * square has its upper-left corner at (b1_row, b1_col) and the second
 * square has its upper-left corner at (b2_row, b2_col). Doesn't
 * properly handle squares which do not overlap.
 */
static int block_overlap(int b1_row, int b1_col, int b2_row, int b2_col) {
  const int int_top = VPXMAX(b1_row, b2_row);   // top
  const int int_left = VPXMAX(b1_col, b2_col);  // left
  /* Since each block is 4x4 pixels, adding 4 (Q3) to the left/top edge
   * gives us the right/bottom edge.
   */
  const int int_right = VPXMIN(b1_col + (4 << 3), b2_col + (4 << 3));  // right
  const int int_bottom =
      VPXMIN(b1_row + (4 << 3), b2_row + (4 << 3));  // bottom
  return (int_bottom - int_top) * (int_right - int_left);
}

/* Calculates the overlap area for all blocks in a macroblock at position
 * (mb_row, mb_col) in macroblocks, which are being overlapped by a given
 * overlapping block at position (new_row, new_col) (in pixels, Q3). The
 * first block being overlapped in the macroblock has position (first_blk_row,
 * first_blk_col) in blocks relative the upper-left corner of the image.
 */
static void calculate_overlaps_mb(B_OVERLAP *b_overlaps, union b_mode_info *bmi,
                                  int new_row, int new_col, int mb_row,
                                  int mb_col, int first_blk_row,
                                  int first_blk_col) {
  /* Find the blocks within this MB (defined by mb_row, mb_col) which are
   * overlapped by bmi and calculate and assign overlap for each of those
   * blocks. */

  /* Block coordinates relative the upper-left block */
  const int rel_ol_blk_row = first_blk_row - mb_row * 4;
  const int rel_ol_blk_col = first_blk_col - mb_col * 4;
  /* If the block partly overlaps any previous MB, these coordinates
   * can be < 0. We don't want to access blocks in previous MBs.
   */
  const int blk_idx = VPXMAX(rel_ol_blk_row, 0) * 4 + VPXMAX(rel_ol_blk_col, 0);
  /* Upper left overlapping block */
  B_OVERLAP *b_ol_ul = &(b_overlaps[blk_idx]);

  /* Calculate and assign overlaps for all blocks in this MB
   * which the motion compensated block overlaps
   */
  /* Avoid calculating overlaps for blocks in later MBs */
  int end_row = VPXMIN(4 + mb_row * 4 - first_blk_row, 2);
  int end_col = VPXMIN(4 + mb_col * 4 - first_blk_col, 2);
  int row, col;

  /* Check if new_row and new_col are evenly divisible by 4 (Q3),
   * and if so we shouldn't check neighboring blocks
   */
  if (new_row >= 0 && (new_row & 0x1F) == 0) end_row = 1;
  if (new_col >= 0 && (new_col & 0x1F) == 0) end_col = 1;

  /* Check if the overlapping block partly overlaps a previous MB
   * and if so, we're overlapping fewer blocks in this MB.
   */
  if (new_row < (mb_row * 16) << 3) end_row = 1;
  if (new_col < (mb_col * 16) << 3) end_col = 1;

  for (row = 0; row < end_row; ++row) {
    for (col = 0; col < end_col; ++col) {
      /* input in Q3, result in Q6 */
      const int overlap =
          block_overlap(new_row, new_col, (((first_blk_row + row) * 4) << 3),
                        (((first_blk_col + col) * 4) << 3));
      assign_overlap(b_ol_ul[row * 4 + col].overlaps, bmi, overlap);
    }
  }
}

static void calculate_overlaps(MB_OVERLAP *overlap_ul, int mb_rows, int mb_cols,
                               union b_mode_info *bmi, int b_row, int b_col) {
  MB_OVERLAP *mb_overlap;
  int row, col, rel_row, rel_col;
  int new_row, new_col;
  int end_row, end_col;
  int overlap_b_row, overlap_b_col;
  int overlap_mb_row, overlap_mb_col;

  /* mb subpixel position */
  row = (4 * b_row) << 3; /* Q3 */
  col = (4 * b_col) << 3; /* Q3 */

  /* reverse compensate for motion */
  new_row = row - bmi->mv.as_mv.row;
  new_col = col - bmi->mv.as_mv.col;

  if (new_row >= ((16 * mb_rows) << 3) || new_col >= ((16 * mb_cols) << 3)) {
    /* the new block ended up outside the frame */
    return;
  }

  if (new_row <= -32 || new_col <= -32) {
    /* outside the frame */
    return;
  }
  /* overlapping block's position in blocks */
  overlap_b_row = FLOOR(new_row / 4, 3) >> 3;
  overlap_b_col = FLOOR(new_col / 4, 3) >> 3;

  /* overlapping block's MB position in MBs
   * operations are done in Q3
   */
  overlap_mb_row = FLOOR((overlap_b_row << 3) / 4, 3) >> 3;
  overlap_mb_col = FLOOR((overlap_b_col << 3) / 4, 3) >> 3;

  end_row = VPXMIN(mb_rows - overlap_mb_row, 2);
  end_col = VPXMIN(mb_cols - overlap_mb_col, 2);

  /* Don't calculate overlap for MBs we don't overlap */
  /* Check if the new block row starts at the last block row of the MB */
  if (abs(new_row - ((16 * overlap_mb_row) << 3)) < ((3 * 4) << 3)) end_row = 1;
  /* Check if the new block col starts at the last block col of the MB */
  if (abs(new_col - ((16 * overlap_mb_col) << 3)) < ((3 * 4) << 3)) end_col = 1;

  /* find the MB(s) this block is overlapping */
  for (rel_row = 0; rel_row < end_row; ++rel_row) {
    for (rel_col = 0; rel_col < end_col; ++rel_col) {
      if (overlap_mb_row + rel_row < 0 || overlap_mb_col + rel_col < 0)
        continue;
      mb_overlap = overlap_ul + (overlap_mb_row + rel_row) * mb_cols +
                   overlap_mb_col + rel_col;

      calculate_overlaps_mb(mb_overlap->overlaps, bmi, new_row, new_col,
                            overlap_mb_row + rel_row, overlap_mb_col + rel_col,
                            overlap_b_row + rel_row, overlap_b_col + rel_col);
    }
  }
}

/* Estimates a motion vector given the overlapping blocks' motion vectors.
 * Filters out all overlapping blocks which do not refer to the correct
 * reference frame type.
 */
static void estimate_mv(const OVERLAP_NODE *overlaps, union b_mode_info *bmi) {
  int i;
  int overlap_sum = 0;
  int row_acc = 0;
  int col_acc = 0;

  bmi->mv.as_int = 0;
  for (i = 0; i < MAX_OVERLAPS; ++i) {
    if (overlaps[i].bmi == NULL) break;
    col_acc += overlaps[i].overlap * overlaps[i].bmi->mv.as_mv.col;
    row_acc += overlaps[i].overlap * overlaps[i].bmi->mv.as_mv.row;
    overlap_sum += overlaps[i].overlap;
  }
  if (overlap_sum > 0) {
    /* Q9 / Q6 = Q3 */
    bmi->mv.as_mv.col = col_acc / overlap_sum;
    bmi->mv.as_mv.row = row_acc / overlap_sum;
  } else {
    bmi->mv.as_mv.col = 0;
    bmi->mv.as_mv.row = 0;
  }
}

/* Estimates all motion vectors for a macroblock given the lists of
 * overlaps for each block. Decides whether or not the MVs must be clamped.
 */
static void estimate_mb_mvs(const B_OVERLAP *block_overlaps, MODE_INFO *mi,
                            int mb_to_left_edge, int mb_to_right_edge,
                            int mb_to_top_edge, int mb_to_bottom_edge) {
  int row, col;
  int non_zero_count = 0;
  MV *const filtered_mv = &(mi->mbmi.mv.as_mv);
  union b_mode_info *const bmi = mi->bmi;
  filtered_mv->col = 0;
  filtered_mv->row = 0;
  mi->mbmi.need_to_clamp_mvs = 0;
  for (row = 0; row < 4; ++row) {
    int this_b_to_top_edge = mb_to_top_edge + ((row * 4) << 3);
    int this_b_to_bottom_edge = mb_to_bottom_edge - ((row * 4) << 3);
    for (col = 0; col < 4; ++col) {
      int i = row * 4 + col;
      int this_b_to_left_edge = mb_to_left_edge + ((col * 4) << 3);
      int this_b_to_right_edge = mb_to_right_edge - ((col * 4) << 3);
      /* Estimate vectors for all blocks which are overlapped by this */
      /* type. Interpolate/extrapolate the rest of the block's MVs */
      estimate_mv(block_overlaps[i].overlaps, &(bmi[i]));
      mi->mbmi.need_to_clamp_mvs |= vp8_check_mv_bounds(
          &bmi[i].mv, this_b_to_left_edge, this_b_to_right_edge,
          this_b_to_top_edge, this_b_to_bottom_edge);
      if (bmi[i].mv.as_int != 0) {
        ++non_zero_count;
        filtered_mv->col += bmi[i].mv.as_mv.col;
        filtered_mv->row += bmi[i].mv.as_mv.row;
      }
    }
  }
  if (non_zero_count > 0) {
    filtered_mv->col /= non_zero_count;
    filtered_mv->row /= non_zero_count;
  }
}

static void calc_prev_mb_overlaps(MB_OVERLAP *overlaps, MODE_INFO *prev_mi,
                                  int mb_row, int mb_col, int mb_rows,
                                  int mb_cols) {
  int sub_row;
  int sub_col;
  for (sub_row = 0; sub_row < 4; ++sub_row) {
    for (sub_col = 0; sub_col < 4; ++sub_col) {
      calculate_overlaps(overlaps, mb_rows, mb_cols,
                         &(prev_mi->bmi[sub_row * 4 + sub_col]),
                         4 * mb_row + sub_row, 4 * mb_col + sub_col);
    }
  }
}

/* Estimate all missing motion vectors. This function does the same as the one
 * above, but has different input arguments. */
static void estimate_missing_mvs(MB_OVERLAP *overlaps, MODE_INFO *mi,
                                 MODE_INFO *prev_mi, int mb_rows, int mb_cols,
                                 unsigned int first_corrupt) {
  int mb_row, mb_col;
  memset(overlaps, 0, sizeof(MB_OVERLAP) * mb_rows * mb_cols);
  /* First calculate the overlaps for all blocks */
  for (mb_row = 0; mb_row < mb_rows; ++mb_row) {
    for (mb_col = 0; mb_col < mb_cols; ++mb_col) {
      /* We're only able to use blocks referring to the last frame
       * when extrapolating new vectors.
       */
      if (prev_mi->mbmi.ref_frame == LAST_FRAME) {
        calc_prev_mb_overlaps(overlaps, prev_mi, mb_row, mb_col, mb_rows,
                              mb_cols);
      }
      ++prev_mi;
    }
    ++prev_mi;
  }

  mb_row = first_corrupt / mb_cols;
  mb_col = first_corrupt - mb_row * mb_cols;
  mi += mb_row * (mb_cols + 1) + mb_col;
  /* Go through all macroblocks in the current image with missing MVs
   * and calculate new MVs using the overlaps.
   */
  for (; mb_row < mb_rows; ++mb_row) {
    int mb_to_top_edge = -((mb_row * 16)) << 3;
    int mb_to_bottom_edge = ((mb_rows - 1 - mb_row) * 16) << 3;
    for (; mb_col < mb_cols; ++mb_col) {
      int mb_to_left_edge = -((mb_col * 16) << 3);
      int mb_to_right_edge = ((mb_cols - 1 - mb_col) * 16) << 3;
      const B_OVERLAP *block_overlaps =
          overlaps[mb_row * mb_cols + mb_col].overlaps;
      mi->mbmi.ref_frame = LAST_FRAME;
      mi->mbmi.mode = SPLITMV;
      mi->mbmi.uv_mode = DC_PRED;
      mi->mbmi.partitioning = 3;
      mi->mbmi.segment_id = 0;
      estimate_mb_mvs(block_overlaps, mi, mb_to_left_edge, mb_to_right_edge,
                      mb_to_top_edge, mb_to_bottom_edge);
      ++mi;
    }
    mb_col = 0;
    ++mi;
  }
}

void vp8_estimate_missing_mvs(VP8D_COMP *pbi) {
  VP8_COMMON *const pc = &pbi->common;
  estimate_missing_mvs(pbi->overlaps, pc->mi, pc->prev_mi, pc->mb_rows,
                       pc->mb_cols, pbi->mvs_corrupt_from_mb);
}

static void assign_neighbor(EC_BLOCK *neighbor, MODE_INFO *mi, int block_idx) {
  assert(mi->mbmi.ref_frame < MAX_REF_FRAMES);
  neighbor->ref_frame = mi->mbmi.ref_frame;
  neighbor->mv = mi->bmi[block_idx].mv.as_mv;
}

/* Finds the neighboring blocks of a macroblocks. In the general case
 * 20 blocks are found. If a fewer number of blocks are found due to
 * image boundaries, those positions in the EC_BLOCK array are left "empty".
 * The neighbors are enumerated with the upper-left neighbor as the first
 * element, the second element refers to the neighbor to right of the previous
 * neighbor, and so on. The last element refers to the neighbor below the first
 * neighbor.
 */
static void find_neighboring_blocks(MODE_INFO *mi, EC_BLOCK *neighbors,
                                    int mb_row, int mb_col, int mb_rows,
                                    int mb_cols, int mi_stride) {
  int i = 0;
  int j;
  if (mb_row > 0) {
    /* upper left */
    if (mb_col > 0) assign_neighbor(&neighbors[i], mi - mi_stride - 1, 15);
    ++i;
    /* above */
    for (j = 12; j < 16; ++j, ++i)
      assign_neighbor(&neighbors[i], mi - mi_stride, j);
  } else
    i += 5;
  if (mb_col < mb_cols - 1) {
    /* upper right */
    if (mb_row > 0) assign_neighbor(&neighbors[i], mi - mi_stride + 1, 12);
    ++i;
    /* right */
    for (j = 0; j <= 12; j += 4, ++i) assign_neighbor(&neighbors[i], mi + 1, j);
  } else
    i += 5;
  if (mb_row < mb_rows - 1) {
    /* lower right */
    if (mb_col < mb_cols - 1)
      assign_neighbor(&neighbors[i], mi + mi_stride + 1, 0);
    ++i;
    /* below */
    for (j = 0; j < 4; ++j, ++i)
      assign_neighbor(&neighbors[i], mi + mi_stride, j);
  } else
    i += 5;
  if (mb_col > 0) {
    /* lower left */
    if (mb_row < mb_rows - 1)
      assign_neighbor(&neighbors[i], mi + mi_stride - 1, 4);
    ++i;
    /* left */
    for (j = 3; j < 16; j += 4, ++i) {
      assign_neighbor(&neighbors[i], mi - 1, j);
    }
  } else
    i += 5;
  assert(i == 20);
}

/* Interpolates all motion vectors for a macroblock from the neighboring blocks'
 * motion vectors.
 */
static void interpolate_mvs(MACROBLOCKD *mb, EC_BLOCK *neighbors,
                            MV_REFERENCE_FRAME dom_ref_frame) {
  int row, col, i;
  MODE_INFO *const mi = mb->mode_info_context;
  /* Table with the position of the neighboring blocks relative the position
   * of the upper left block of the current MB. Starting with the upper left
   * neighbor and going to the right.
   */
  const EC_POS neigh_pos[NUM_NEIGHBORS] = {
    { -1, -1 }, { -1, 0 }, { -1, 1 }, { -1, 2 }, { -1, 3 }, { -1, 4 }, { 0, 4 },
    { 1, 4 },   { 2, 4 },  { 3, 4 },  { 4, 4 },  { 4, 3 },  { 4, 2 },  { 4, 1 },
    { 4, 0 },   { 4, -1 }, { 3, -1 }, { 2, -1 }, { 1, -1 }, { 0, -1 }
  };
  mi->mbmi.need_to_clamp_mvs = 0;
  for (row = 0; row < 4; ++row) {
    int mb_to_top_edge = mb->mb_to_top_edge + ((row * 4) << 3);
    int mb_to_bottom_edge = mb->mb_to_bottom_edge - ((row * 4) << 3);
    for (col = 0; col < 4; ++col) {
      int mb_to_left_edge = mb->mb_to_left_edge + ((col * 4) << 3);
      int mb_to_right_edge = mb->mb_to_right_edge - ((col * 4) << 3);
      int w_sum = 0;
      int mv_row_sum = 0;
      int mv_col_sum = 0;
      int_mv *const mv = &(mi->bmi[row * 4 + col].mv);
      mv->as_int = 0;
      for (i = 0; i < NUM_NEIGHBORS; ++i) {
        /* Calculate the weighted sum of neighboring MVs referring
         * to the dominant frame type.
         */
        const int w = weights_q7[abs(row - neigh_pos[i].row)]
                                [abs(col - neigh_pos[i].col)];
        if (neighbors[i].ref_frame != dom_ref_frame) continue;
        w_sum += w;
        /* Q7 * Q3 = Q10 */
        mv_row_sum += w * neighbors[i].mv.row;
        mv_col_sum += w * neighbors[i].mv.col;
      }
      if (w_sum > 0) {
        /* Avoid division by zero.
         * Normalize with the sum of the coefficients
         * Q3 = Q10 / Q7
         */
        mv->as_mv.row = mv_row_sum / w_sum;
        mv->as_mv.col = mv_col_sum / w_sum;
        mi->mbmi.need_to_clamp_mvs |=
            vp8_check_mv_bounds(mv, mb_to_left_edge, mb_to_right_edge,
                                mb_to_top_edge, mb_to_bottom_edge);
      }
    }
  }
}

void vp8_interpolate_motion(MACROBLOCKD *mb, int mb_row, int mb_col,
                            int mb_rows, int mb_cols) {
  /* Find relevant neighboring blocks */
  EC_BLOCK neighbors[NUM_NEIGHBORS];
  int i;
  /* Initialize the array. MAX_REF_FRAMES is interpreted as "doesn't exist" */
  for (i = 0; i < NUM_NEIGHBORS; ++i) {
    neighbors[i].ref_frame = MAX_REF_FRAMES;
    neighbors[i].mv.row = neighbors[i].mv.col = 0;
  }
  find_neighboring_blocks(mb->mode_info_context, neighbors, mb_row, mb_col,
                          mb_rows, mb_cols, mb->mode_info_stride);
  /* Interpolate MVs for the missing blocks from the surrounding
   * blocks which refer to the last frame. */
  interpolate_mvs(mb, neighbors, LAST_FRAME);

  mb->mode_info_context->mbmi.ref_frame = LAST_FRAME;
  mb->mode_info_context->mbmi.mode = SPLITMV;
  mb->mode_info_context->mbmi.uv_mode = DC_PRED;
  mb->mode_info_context->mbmi.partitioning = 3;
  mb->mode_info_context->mbmi.segment_id = 0;
}
