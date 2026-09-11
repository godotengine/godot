/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/quantize.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/mem.h"

#if CONFIG_MISMATCH_DEBUG
#include "vpx_util/vpx_debug_util.h"
#endif

#include "vp9/common/vp9_idct.h"
#include "vp9/common/vp9_reconinter.h"
#include "vp9/common/vp9_reconintra.h"
#include "vp9/common/vp9_scan.h"

#include "vp9/encoder/vp9_encodemb.h"
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_rd.h"
#include "vp9/encoder/vp9_tokenize.h"

struct optimize_ctx {
  ENTROPY_CONTEXT ta[MAX_MB_PLANE][16];
  ENTROPY_CONTEXT tl[MAX_MB_PLANE][16];
};

void vp9_subtract_plane(MACROBLOCK *x, BLOCK_SIZE bsize, int plane) {
  struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &x->e_mbd.plane[plane];
  const BLOCK_SIZE plane_bsize = get_plane_block_size(bsize, pd);
  const int bw = 4 * num_4x4_blocks_wide_lookup[plane_bsize];
  const int bh = 4 * num_4x4_blocks_high_lookup[plane_bsize];

#if CONFIG_VP9_HIGHBITDEPTH
  if (x->e_mbd.cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    vpx_highbd_subtract_block(bh, bw, p->src_diff, bw, p->src.buf,
                              p->src.stride, pd->dst.buf, pd->dst.stride,
                              x->e_mbd.bd);
    return;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH
  vpx_subtract_block(bh, bw, p->src_diff, bw, p->src.buf, p->src.stride,
                     pd->dst.buf, pd->dst.stride);
}

static const int plane_rd_mult[REF_TYPES][PLANE_TYPES] = {
  { 10, 6 },
  { 8, 5 },
};

// 'num' can be negative, but 'shift' must be non-negative.
#define RIGHT_SHIFT_POSSIBLY_NEGATIVE(num, shift) \
  (((num) >= 0) ? (num) >> (shift) : -((-(num)) >> (shift)))

int vp9_optimize_b(MACROBLOCK *mb, int plane, int block, TX_SIZE tx_size,
                   int ctx) {
  MACROBLOCKD *const xd = &mb->e_mbd;
  struct macroblock_plane *const p = &mb->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  const int ref = is_inter_block(xd->mi[0]);
  uint8_t token_cache[1024];
  const tran_low_t *const coeff = BLOCK_OFFSET(p->coeff, block);
  tran_low_t *const qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  const int eob = p->eobs[block];
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const int default_eob = 16 << (tx_size << 1);
  const int shift = (tx_size == TX_32X32);
  const int16_t *const dequant_ptr = pd->dequant;
  const uint8_t *const band_translate = get_band_translate(tx_size);
  const ScanOrder *const so = get_scan(xd, tx_size, plane_type, block);
  const int16_t *const scan = so->scan;
  const int16_t *const nb = so->neighbors;
  const MODE_INFO *mbmi = xd->mi[0];
  const int sharpness = mb->sharpness;
  const int64_t rdadj = (int64_t)mb->rdmult * plane_rd_mult[ref][plane_type];
  const int64_t rdmult =
      (sharpness == 0 ? rdadj >> 1
                      : (rdadj * (8 - sharpness + mbmi->segment_id)) >> 4);

  const int64_t rddiv = mb->rddiv;
  int64_t rd_cost0, rd_cost1;
  int64_t rate0, rate1;
  int16_t t0, t1;
  int i, final_eob;
  int count_high_values_after_eob = 0;
#if CONFIG_VP9_HIGHBITDEPTH
  const uint16_t *cat6_high_cost = vp9_get_high_cost_table(xd->bd);
#else
  const uint16_t *cat6_high_cost = vp9_get_high_cost_table(8);
#endif
  unsigned int(*const token_costs)[2][COEFF_CONTEXTS][ENTROPY_TOKENS] =
      mb->token_costs[tx_size][plane_type][ref];
  unsigned int(*token_costs_cur)[2][COEFF_CONTEXTS][ENTROPY_TOKENS];
  int64_t eob_cost0, eob_cost1;
  const int ctx0 = ctx;
  int64_t accu_rate = 0;
  // Initialized to the worst possible error for the largest transform size.
  // This ensures that it never goes negative.
  int64_t accu_error = ((int64_t)1) << 50;
  int64_t best_block_rd_cost = INT64_MAX;
  int x_prev = 1;
  tran_low_t before_best_eob_qc = 0;
  tran_low_t before_best_eob_dqc = 0;

  assert((!plane_type && !plane) || (plane_type && plane));
  assert(eob <= default_eob);

  for (i = 0; i < eob; i++) {
    const int rc = scan[i];
    token_cache[rc] = vp9_pt_energy_class[vp9_get_token(qcoeff[rc])];
  }
  final_eob = 0;

  // Initial RD cost.
  token_costs_cur = token_costs + band_translate[0];
  rate0 = (*token_costs_cur)[0][ctx0][EOB_TOKEN];
  best_block_rd_cost = RDCOST(rdmult, rddiv, rate0, accu_error);

  // For each token, pick one of two choices greedily:
  // (i) First candidate: Keep current quantized value, OR
  // (ii) Second candidate: Reduce quantized value by 1.
  for (i = 0; i < eob; i++) {
    const int rc = scan[i];
    const int x = qcoeff[rc];
    const int band_cur = band_translate[i];
    const int ctx_cur = (i == 0) ? ctx : get_coef_context(nb, token_cache, i);
    const int token_tree_sel_cur = (x_prev == 0);
    token_costs_cur = token_costs + band_cur;
    if (x == 0) {  // No need to search
      const int token = vp9_get_token(x);
      rate0 = (*token_costs_cur)[token_tree_sel_cur][ctx_cur][token];
      accu_rate += rate0;
      x_prev = 0;
      // Note: accu_error does not change.
    } else {
      const int dqv = dequant_ptr[rc != 0];
      // Compute the distortion for quantizing to 0.
      const int diff_for_zero_raw = (0 - coeff[rc]) * (1 << shift);
      const int diff_for_zero =
#if CONFIG_VP9_HIGHBITDEPTH
          (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH)
              ? RIGHT_SHIFT_POSSIBLY_NEGATIVE(diff_for_zero_raw, xd->bd - 8)
              :
#endif
              diff_for_zero_raw;
      const int64_t distortion_for_zero =
          (int64_t)diff_for_zero * diff_for_zero;

      // Compute the distortion for the first candidate
      const int diff0_raw = (dqcoeff[rc] - coeff[rc]) * (1 << shift);
      const int diff0 =
#if CONFIG_VP9_HIGHBITDEPTH
          (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH)
              ? RIGHT_SHIFT_POSSIBLY_NEGATIVE(diff0_raw, xd->bd - 8)
              :
#endif  // CONFIG_VP9_HIGHBITDEPTH
              diff0_raw;
      const int64_t distortion0 = (int64_t)diff0 * diff0;

      // Compute the distortion for the second candidate
      const int sign = -(x < 0);        // -1 if x is negative and 0 otherwise.
      const int x1 = x - 2 * sign - 1;  // abs(x1) = abs(x) - 1.
      int64_t distortion1;
      if (x1 != 0) {
        const int dqv_step =
#if CONFIG_VP9_HIGHBITDEPTH
            (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) ? dqv >> (xd->bd - 8)
                                                          :
#endif  // CONFIG_VP9_HIGHBITDEPTH
                                                          dqv;
        const int diff_step = (dqv_step + sign) ^ sign;
        const int diff1 = diff0 - diff_step;
        assert(dqv > 0);  // We aren't right shifting a negative number above.
        distortion1 = (int64_t)diff1 * diff1;
      } else {
        distortion1 = distortion_for_zero;
      }
      {
        // Calculate RDCost for current coeff for the two candidates.
        const int64_t base_bits0 = vp9_get_token_cost(x, &t0, cat6_high_cost);
        const int64_t base_bits1 = vp9_get_token_cost(x1, &t1, cat6_high_cost);
        rate0 =
            base_bits0 + (*token_costs_cur)[token_tree_sel_cur][ctx_cur][t0];
        rate1 =
            base_bits1 + (*token_costs_cur)[token_tree_sel_cur][ctx_cur][t1];
      }
      {
        int rdcost_better_for_x1, eob_rdcost_better_for_x1;
        int dqc0, dqc1;
        int64_t best_eob_cost_cur;
        int use_x1;

        // Calculate RD Cost effect on the next coeff for the two candidates.
        int64_t next_bits0 = 0;
        int64_t next_bits1 = 0;
        int64_t next_eob_bits0 = 0;
        int64_t next_eob_bits1 = 0;
        if (i < default_eob - 1) {
          int ctx_next, token_tree_sel_next;
          const int band_next = band_translate[i + 1];
          const int token_next =
              (i + 1 != eob) ? vp9_get_token(qcoeff[scan[i + 1]]) : EOB_TOKEN;
          unsigned int(*const token_costs_next)[2][COEFF_CONTEXTS]
                                               [ENTROPY_TOKENS] =
                                                   token_costs + band_next;
          token_cache[rc] = vp9_pt_energy_class[t0];
          ctx_next = get_coef_context(nb, token_cache, i + 1);
          token_tree_sel_next = (x == 0);
          next_bits0 =
              (*token_costs_next)[token_tree_sel_next][ctx_next][token_next];
          next_eob_bits0 =
              (*token_costs_next)[token_tree_sel_next][ctx_next][EOB_TOKEN];
          token_cache[rc] = vp9_pt_energy_class[t1];
          ctx_next = get_coef_context(nb, token_cache, i + 1);
          token_tree_sel_next = (x1 == 0);
          next_bits1 =
              (*token_costs_next)[token_tree_sel_next][ctx_next][token_next];
          if (x1 != 0) {
            next_eob_bits1 =
                (*token_costs_next)[token_tree_sel_next][ctx_next][EOB_TOKEN];
          }
        }

        // Compare the total RD costs for two candidates.
        rd_cost0 = RDCOST(rdmult, rddiv, (rate0 + next_bits0), distortion0);
        rd_cost1 = RDCOST(rdmult, rddiv, (rate1 + next_bits1), distortion1);
        rdcost_better_for_x1 = (rd_cost1 < rd_cost0);
        eob_cost0 = RDCOST(rdmult, rddiv, (accu_rate + rate0 + next_eob_bits0),
                           (accu_error + distortion0 - distortion_for_zero));
        eob_cost1 = eob_cost0;
        if (x1 != 0) {
          eob_cost1 =
              RDCOST(rdmult, rddiv, (accu_rate + rate1 + next_eob_bits1),
                     (accu_error + distortion1 - distortion_for_zero));
          eob_rdcost_better_for_x1 = (eob_cost1 < eob_cost0);
        } else {
          eob_rdcost_better_for_x1 = 0;
        }

        // Calculate the two candidate de-quantized values.
        dqc0 = dqcoeff[rc];
        dqc1 = 0;
        if (rdcost_better_for_x1 + eob_rdcost_better_for_x1) {
          if (x1 != 0) {
            dqc1 = RIGHT_SHIFT_POSSIBLY_NEGATIVE(x1 * dqv, shift);
          } else {
            dqc1 = 0;
          }
        }

        // Pick and record the better quantized and de-quantized values.
        if (rdcost_better_for_x1) {
          qcoeff[rc] = x1;
          dqcoeff[rc] = dqc1;
          accu_rate += rate1;
          accu_error += distortion1 - distortion_for_zero;
          assert(distortion1 <= distortion_for_zero);
          token_cache[rc] = vp9_pt_energy_class[t1];
        } else {
          accu_rate += rate0;
          accu_error += distortion0 - distortion_for_zero;
          assert(distortion0 <= distortion_for_zero);
          token_cache[rc] = vp9_pt_energy_class[t0];
        }
        if (sharpness > 0 && abs(qcoeff[rc]) > 1) count_high_values_after_eob++;
        assert(accu_error >= 0);
        x_prev = qcoeff[rc];  // Update based on selected quantized value.

        use_x1 = (x1 != 0) && eob_rdcost_better_for_x1;
        best_eob_cost_cur = use_x1 ? eob_cost1 : eob_cost0;

        // Determine whether to move the eob position to i+1
        if (best_eob_cost_cur < best_block_rd_cost) {
          best_block_rd_cost = best_eob_cost_cur;
          final_eob = i + 1;
          count_high_values_after_eob = 0;
          if (use_x1) {
            before_best_eob_qc = x1;
            before_best_eob_dqc = dqc1;
          } else {
            before_best_eob_qc = x;
            before_best_eob_dqc = dqc0;
          }
        }
      }
    }
  }
  if (count_high_values_after_eob > 0) {
    final_eob = eob - 1;
    for (; final_eob >= 0; final_eob--) {
      const int rc = scan[final_eob];
      const int x = qcoeff[rc];
      if (x) {
        break;
      }
    }
    final_eob++;
  } else {
    assert(final_eob <= eob);
    if (final_eob > 0) {
      int rc;
      assert(before_best_eob_qc != 0);
      i = final_eob - 1;
      rc = scan[i];
      qcoeff[rc] = before_best_eob_qc;
      dqcoeff[rc] = before_best_eob_dqc;
    }
    for (i = final_eob; i < eob; i++) {
      int rc = scan[i];
      qcoeff[rc] = 0;
      dqcoeff[rc] = 0;
    }
  }
  mb->plane[plane].eobs[block] = final_eob;
  return final_eob;
}
#undef RIGHT_SHIFT_POSSIBLY_NEGATIVE

static INLINE void fdct32x32(int rd_transform, const int16_t *src,
                             tran_low_t *dst, int src_stride) {
  if (rd_transform)
    vpx_fdct32x32_rd(src, dst, src_stride);
  else
    vpx_fdct32x32(src, dst, src_stride);
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void highbd_fdct32x32(int rd_transform, const int16_t *src,
                                    tran_low_t *dst, int src_stride) {
  if (rd_transform)
    vpx_highbd_fdct32x32_rd(src, dst, src_stride);
  else
    vpx_highbd_fdct32x32(src, dst, src_stride);
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

void vp9_xform_quant_fp(MACROBLOCK *x, int plane, int block, int row, int col,
                        BLOCK_SIZE plane_bsize, TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const ScanOrder *const scan_order = &vp9_default_scan_orders[tx_size];
  tran_low_t *const coeff = BLOCK_OFFSET(p->coeff, block);
  tran_low_t *const qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  uint16_t *const eob = &p->eobs[block];
  const int diff_stride = 4 * num_4x4_blocks_wide_lookup[plane_bsize];
  const int16_t *src_diff;
  src_diff = &p->src_diff[4 * (row * diff_stride + col)];
  // skip block condition should be handled before this is called.
  assert(!x->skip_block);

#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    switch (tx_size) {
      case TX_32X32:
        highbd_fdct32x32(x->use_lp32x32fdct, src_diff, coeff, diff_stride);
        vp9_highbd_quantize_fp_32x32(coeff, 1024, p, qcoeff, dqcoeff,
                                     pd->dequant, eob, scan_order);
        break;
      case TX_16X16:
        vpx_highbd_fdct16x16(src_diff, coeff, diff_stride);
        vp9_highbd_quantize_fp(coeff, 256, p, qcoeff, dqcoeff, pd->dequant, eob,
                               scan_order);
        break;
      case TX_8X8:
        vpx_highbd_fdct8x8(src_diff, coeff, diff_stride);
        vp9_highbd_quantize_fp(coeff, 64, p, qcoeff, dqcoeff, pd->dequant, eob,
                               scan_order);
        break;
      default:
        assert(tx_size == TX_4X4);
        x->fwd_txfm4x4(src_diff, coeff, diff_stride);
        vp9_highbd_quantize_fp(coeff, 16, p, qcoeff, dqcoeff, pd->dequant, eob,
                               scan_order);
        break;
    }
    return;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  switch (tx_size) {
    case TX_32X32:
      fdct32x32(x->use_lp32x32fdct, src_diff, coeff, diff_stride);
      vp9_quantize_fp_32x32(coeff, 1024, p, qcoeff, dqcoeff, pd->dequant, eob,
                            scan_order);
      break;
    case TX_16X16:
      vpx_fdct16x16(src_diff, coeff, diff_stride);
      vp9_quantize_fp(coeff, 256, p, qcoeff, dqcoeff, pd->dequant, eob,
                      scan_order);
      break;
    case TX_8X8:
      vpx_fdct8x8(src_diff, coeff, diff_stride);
      vp9_quantize_fp(coeff, 64, p, qcoeff, dqcoeff, pd->dequant, eob,
                      scan_order);

      break;
    default:
      assert(tx_size == TX_4X4);
      x->fwd_txfm4x4(src_diff, coeff, diff_stride);
      vp9_quantize_fp(coeff, 16, p, qcoeff, dqcoeff, pd->dequant, eob,
                      scan_order);
      break;
  }
}

void vp9_xform_quant_dc(MACROBLOCK *x, int plane, int block, int row, int col,
                        BLOCK_SIZE plane_bsize, TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const coeff = BLOCK_OFFSET(p->coeff, block);
  tran_low_t *const qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  uint16_t *const eob = &p->eobs[block];
  const int diff_stride = 4 * num_4x4_blocks_wide_lookup[plane_bsize];
  const int16_t *src_diff;
  src_diff = &p->src_diff[4 * (row * diff_stride + col)];
  // skip block condition should be handled before this is called.
  assert(!x->skip_block);

#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    switch (tx_size) {
      case TX_32X32:
        vpx_highbd_fdct32x32_1(src_diff, coeff, diff_stride);
        vpx_highbd_quantize_dc_32x32(coeff, p->round, p->quant_fp[0], qcoeff,
                                     dqcoeff, pd->dequant[0], eob);
        break;
      case TX_16X16:
        vpx_highbd_fdct16x16_1(src_diff, coeff, diff_stride);
        vpx_highbd_quantize_dc(coeff, 256, p->round, p->quant_fp[0], qcoeff,
                               dqcoeff, pd->dequant[0], eob);
        break;
      case TX_8X8:
        vpx_highbd_fdct8x8_1(src_diff, coeff, diff_stride);
        vpx_highbd_quantize_dc(coeff, 64, p->round, p->quant_fp[0], qcoeff,
                               dqcoeff, pd->dequant[0], eob);
        break;
      default:
        assert(tx_size == TX_4X4);
        x->fwd_txfm4x4(src_diff, coeff, diff_stride);
        vpx_highbd_quantize_dc(coeff, 16, p->round, p->quant_fp[0], qcoeff,
                               dqcoeff, pd->dequant[0], eob);
        break;
    }
    return;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  switch (tx_size) {
    case TX_32X32:
      vpx_fdct32x32_1(src_diff, coeff, diff_stride);
      vpx_quantize_dc_32x32(coeff, p->round, p->quant_fp[0], qcoeff, dqcoeff,
                            pd->dequant[0], eob);
      break;
    case TX_16X16:
      vpx_fdct16x16_1(src_diff, coeff, diff_stride);
      vpx_quantize_dc(coeff, 256, p->round, p->quant_fp[0], qcoeff, dqcoeff,
                      pd->dequant[0], eob);
      break;
    case TX_8X8:
      vpx_fdct8x8_1(src_diff, coeff, diff_stride);
      vpx_quantize_dc(coeff, 64, p->round, p->quant_fp[0], qcoeff, dqcoeff,
                      pd->dequant[0], eob);
      break;
    default:
      assert(tx_size == TX_4X4);
      x->fwd_txfm4x4(src_diff, coeff, diff_stride);
      vpx_quantize_dc(coeff, 16, p->round, p->quant_fp[0], qcoeff, dqcoeff,
                      pd->dequant[0], eob);
      break;
  }
}

void vp9_xform_quant(MACROBLOCK *x, int plane, int block, int row, int col,
                     BLOCK_SIZE plane_bsize, TX_SIZE tx_size) {
  MACROBLOCKD *const xd = &x->e_mbd;
  const struct macroblock_plane *const p = &x->plane[plane];
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const ScanOrder *const scan_order = &vp9_default_scan_orders[tx_size];
  tran_low_t *const coeff = BLOCK_OFFSET(p->coeff, block);
  tran_low_t *const qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  uint16_t *const eob = &p->eobs[block];
  const int diff_stride = 4 * num_4x4_blocks_wide_lookup[plane_bsize];
  const int16_t *src_diff;
  src_diff = &p->src_diff[4 * (row * diff_stride + col)];
  // skip block condition should be handled before this is called.
  assert(!x->skip_block);

#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    switch (tx_size) {
      case TX_32X32:
        highbd_fdct32x32(x->use_lp32x32fdct, src_diff, coeff, diff_stride);
        vpx_highbd_quantize_b_32x32(coeff, p, qcoeff, dqcoeff, pd->dequant, eob,
                                    scan_order);
        break;
      case TX_16X16:
        vpx_highbd_fdct16x16(src_diff, coeff, diff_stride);
        vpx_highbd_quantize_b(coeff, 256, p, qcoeff, dqcoeff, pd->dequant, eob,
                              scan_order);
        break;
      case TX_8X8:
        vpx_highbd_fdct8x8(src_diff, coeff, diff_stride);
        vpx_highbd_quantize_b(coeff, 64, p, qcoeff, dqcoeff, pd->dequant, eob,
                              scan_order);
        break;
      default:
        assert(tx_size == TX_4X4);
        x->fwd_txfm4x4(src_diff, coeff, diff_stride);
        vpx_highbd_quantize_b(coeff, 16, p, qcoeff, dqcoeff, pd->dequant, eob,
                              scan_order);
        break;
    }
    return;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  switch (tx_size) {
    case TX_32X32:
      fdct32x32(x->use_lp32x32fdct, src_diff, coeff, diff_stride);
      vpx_quantize_b_32x32(coeff, p, qcoeff, dqcoeff, pd->dequant, eob,
                           scan_order);
      break;
    case TX_16X16:
      vpx_fdct16x16(src_diff, coeff, diff_stride);
      vpx_quantize_b(coeff, 256, p, qcoeff, dqcoeff, pd->dequant, eob,
                     scan_order);
      break;
    case TX_8X8:
      vpx_fdct8x8(src_diff, coeff, diff_stride);
      vpx_quantize_b(coeff, 64, p, qcoeff, dqcoeff, pd->dequant, eob,
                     scan_order);
      break;
    default:
      assert(tx_size == TX_4X4);
      x->fwd_txfm4x4(src_diff, coeff, diff_stride);
      vpx_quantize_b(coeff, 16, p, qcoeff, dqcoeff, pd->dequant, eob,
                     scan_order);
      break;
  }
}

static void encode_block(int plane, int block, int row, int col,
                         BLOCK_SIZE plane_bsize, TX_SIZE tx_size, void *arg) {
  struct encode_b_args *const args = arg;
#if CONFIG_MISMATCH_DEBUG
  int mi_row = args->mi_row;
  int mi_col = args->mi_col;
  int output_enabled = args->output_enabled;
#endif
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  uint8_t *dst;
  ENTROPY_CONTEXT *a, *l;
  dst = &pd->dst.buf[4 * row * pd->dst.stride + 4 * col];
  a = &args->ta[col];
  l = &args->tl[row];

  // TODO(jingning): per transformed block zero forcing only enabled for
  // luma component. will integrate chroma components as well.
  if (x->zcoeff_blk[tx_size][block] && plane == 0) {
    p->eobs[block] = 0;
    *a = *l = 0;
#if CONFIG_MISMATCH_DEBUG
    goto encode_block_end;
#else
    return;
#endif
  }

  if (!x->skip_recode) {
    if (x->quant_fp) {
      // Encoding process for rtc mode
      if (x->skip_txfm[0] == SKIP_TXFM_AC_DC && plane == 0) {
        // skip forward transform
        p->eobs[block] = 0;
        *a = *l = 0;
#if CONFIG_MISMATCH_DEBUG
        goto encode_block_end;
#else
        return;
#endif
      } else {
        vp9_xform_quant_fp(x, plane, block, row, col, plane_bsize, tx_size);
      }
    } else {
      if (max_txsize_lookup[plane_bsize] == tx_size) {
        int txfm_blk_index = (plane << 2) + (block >> (tx_size << 1));
        if (x->skip_txfm[txfm_blk_index] == SKIP_TXFM_NONE) {
          // full forward transform and quantization
          vp9_xform_quant(x, plane, block, row, col, plane_bsize, tx_size);
        } else if (x->skip_txfm[txfm_blk_index] == SKIP_TXFM_AC_ONLY) {
          // fast path forward transform and quantization
          vp9_xform_quant_dc(x, plane, block, row, col, plane_bsize, tx_size);
        } else {
          // skip forward transform
          p->eobs[block] = 0;
          *a = *l = 0;
#if CONFIG_MISMATCH_DEBUG
          goto encode_block_end;
#else
          return;
#endif
        }
      } else {
        vp9_xform_quant(x, plane, block, row, col, plane_bsize, tx_size);
      }
    }
  }

  if (x->optimize && (!x->skip_recode || !x->skip_optimize)) {
    const int ctx = combine_entropy_contexts(*a, *l);
    *a = *l = vp9_optimize_b(x, plane, block, tx_size, ctx) > 0;
  } else {
    *a = *l = p->eobs[block] > 0;
  }

  if (p->eobs[block]) *(args->skip) = 0;

  if (x->skip_encode || p->eobs[block] == 0) {
#if CONFIG_MISMATCH_DEBUG
    goto encode_block_end;
#else
    return;
#endif
  }
#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    uint16_t *const dst16 = CONVERT_TO_SHORTPTR(dst);
    switch (tx_size) {
      case TX_32X32:
        vp9_highbd_idct32x32_add(dqcoeff, dst16, pd->dst.stride, p->eobs[block],
                                 xd->bd);
        break;
      case TX_16X16:
        vp9_highbd_idct16x16_add(dqcoeff, dst16, pd->dst.stride, p->eobs[block],
                                 xd->bd);
        break;
      case TX_8X8:
        vp9_highbd_idct8x8_add(dqcoeff, dst16, pd->dst.stride, p->eobs[block],
                               xd->bd);
        break;
      default:
        assert(tx_size == TX_4X4);
        // this is like vp9_short_idct4x4 but has a special case around eob<=1
        // which is significant (not just an optimization) for the lossless
        // case.
        x->highbd_inv_txfm_add(dqcoeff, dst16, pd->dst.stride, p->eobs[block],
                               xd->bd);
        break;
    }
#if CONFIG_MISMATCH_DEBUG
    goto encode_block_end;
#else
    return;
#endif
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  switch (tx_size) {
    case TX_32X32:
      vp9_idct32x32_add(dqcoeff, dst, pd->dst.stride, p->eobs[block]);
      break;
    case TX_16X16:
      vp9_idct16x16_add(dqcoeff, dst, pd->dst.stride, p->eobs[block]);
      break;
    case TX_8X8:
      vp9_idct8x8_add(dqcoeff, dst, pd->dst.stride, p->eobs[block]);
      break;
    default:
      assert(tx_size == TX_4X4);
      // this is like vp9_short_idct4x4 but has a special case around eob<=1
      // which is significant (not just an optimization) for the lossless
      // case.
      x->inv_txfm_add(dqcoeff, dst, pd->dst.stride, p->eobs[block]);
      break;
  }
#if CONFIG_MISMATCH_DEBUG
encode_block_end:
  if (output_enabled) {
    int pixel_c, pixel_r;
    int blk_w = 1 << (tx_size + TX_UNIT_SIZE_LOG2);
    int blk_h = 1 << (tx_size + TX_UNIT_SIZE_LOG2);
    mi_to_pixel_loc(&pixel_c, &pixel_r, mi_col, mi_row, col, row,
                    pd->subsampling_x, pd->subsampling_y);
    mismatch_record_block_tx(dst, pd->dst.stride, plane, pixel_c, pixel_r,
                             blk_w, blk_h,
                             xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH);
  }
#endif
}

static void encode_block_pass1(int plane, int block, int row, int col,
                               BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                               void *arg) {
  MACROBLOCK *const x = (MACROBLOCK *)arg;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *const dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  uint8_t *dst;
  dst = &pd->dst.buf[4 * row * pd->dst.stride + 4 * col];

  vp9_xform_quant(x, plane, block, row, col, plane_bsize, tx_size);

  if (p->eobs[block] > 0) {
#if CONFIG_VP9_HIGHBITDEPTH
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      x->highbd_inv_txfm_add(dqcoeff, CONVERT_TO_SHORTPTR(dst), pd->dst.stride,
                             p->eobs[block], xd->bd);
      return;
    }
#endif  // CONFIG_VP9_HIGHBITDEPTH
    x->inv_txfm_add(dqcoeff, dst, pd->dst.stride, p->eobs[block]);
  }
}

void vp9_encode_sby_pass1(MACROBLOCK *x, BLOCK_SIZE bsize) {
  vp9_subtract_plane(x, bsize, 0);
  vp9_foreach_transformed_block_in_plane(&x->e_mbd, bsize, 0,
                                         encode_block_pass1, x);
}

void vp9_encode_sb(MACROBLOCK *x, BLOCK_SIZE bsize, int mi_row, int mi_col,
                   int output_enabled) {
  MACROBLOCKD *const xd = &x->e_mbd;
  struct optimize_ctx ctx;
  MODE_INFO *mi = xd->mi[0];
  int plane;
#if CONFIG_MISMATCH_DEBUG
  struct encode_b_args arg = { x,
                               1,     // enable_trellis_opt
                               0.0,   // trellis_opt_thresh
                               NULL,  // &sse_calc_done
                               NULL,  // &sse
                               NULL,  // above entropy context
                               NULL,  // left entropy context
                               &mi->skip, mi_row, mi_col, output_enabled };
#else
  struct encode_b_args arg = { x,
                               1,     // enable_trellis_opt
                               0.0,   // trellis_opt_thresh
                               NULL,  // &sse_calc_done
                               NULL,  // &sse
                               NULL,  // above entropy context
                               NULL,  // left entropy context
                               &mi->skip };
  (void)mi_row;
  (void)mi_col;
  (void)output_enabled;
#endif

  mi->skip = 1;

  if (x->skip) return;

  for (plane = 0; plane < MAX_MB_PLANE; ++plane) {
    if (!x->skip_recode) vp9_subtract_plane(x, bsize, plane);

    if (x->optimize && (!x->skip_recode || !x->skip_optimize)) {
      const struct macroblockd_plane *const pd = &xd->plane[plane];
      const TX_SIZE tx_size = plane ? get_uv_tx_size(mi, pd) : mi->tx_size;
      vp9_get_entropy_contexts(bsize, tx_size, pd, ctx.ta[plane],
                               ctx.tl[plane]);
      arg.enable_trellis_opt = 1;
    } else {
      arg.enable_trellis_opt = 0;
    }
    arg.ta = ctx.ta[plane];
    arg.tl = ctx.tl[plane];

    vp9_foreach_transformed_block_in_plane(xd, bsize, plane, encode_block,
                                           &arg);
  }
}

void vp9_encode_block_intra(int plane, int block, int row, int col,
                            BLOCK_SIZE plane_bsize, TX_SIZE tx_size,
                            void *arg) {
  struct encode_b_args *const args = arg;
  MACROBLOCK *const x = args->x;
  MACROBLOCKD *const xd = &x->e_mbd;
  MODE_INFO *mi = xd->mi[0];
  struct macroblock_plane *const p = &x->plane[plane];
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *coeff = BLOCK_OFFSET(p->coeff, block);
  tran_low_t *qcoeff = BLOCK_OFFSET(p->qcoeff, block);
  tran_low_t *dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
  const ScanOrder *scan_order;
  TX_TYPE tx_type = DCT_DCT;
  PREDICTION_MODE mode;
  const int bwl = b_width_log2_lookup[plane_bsize];
  const int diff_stride = 4 * (1 << bwl);
  uint8_t *src, *dst;
  int16_t *src_diff;
  uint16_t *eob = &p->eobs[block];
  const int src_stride = p->src.stride;
  const int dst_stride = pd->dst.stride;
  int enable_trellis_opt = !x->skip_recode;
  ENTROPY_CONTEXT *a = NULL;
  ENTROPY_CONTEXT *l = NULL;
  int entropy_ctx = 0;
  dst = &pd->dst.buf[4 * (row * dst_stride + col)];
  src = &p->src.buf[4 * (row * src_stride + col)];
  src_diff = &p->src_diff[4 * (row * diff_stride + col)];

  if (tx_size == TX_4X4) {
    tx_type = get_tx_type_4x4(get_plane_type(plane), xd, block);
    scan_order = &vp9_scan_orders[TX_4X4][tx_type];
    mode = plane == 0 ? get_y_mode(xd->mi[0], block) : mi->uv_mode;
  } else {
    mode = plane == 0 ? mi->mode : mi->uv_mode;
    if (tx_size == TX_32X32) {
      scan_order = &vp9_default_scan_orders[TX_32X32];
    } else {
      tx_type = get_tx_type(get_plane_type(plane), xd);
      scan_order = &vp9_scan_orders[tx_size][tx_type];
    }
  }

  vp9_predict_intra_block(
      xd, bwl, tx_size, mode, (x->skip_encode || x->fp_src_pred) ? src : dst,
      (x->skip_encode || x->fp_src_pred) ? src_stride : dst_stride, dst,
      dst_stride, col, row, plane);

  // skip block condition should be handled before this is called.
  assert(!x->skip_block);

  if (!x->skip_recode) {
    const int tx_size_in_pixels = (1 << tx_size) << 2;
#if CONFIG_VP9_HIGHBITDEPTH
    if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
      vpx_highbd_subtract_block(tx_size_in_pixels, tx_size_in_pixels, src_diff,
                                diff_stride, src, src_stride, dst, dst_stride,
                                xd->bd);
    } else {
      vpx_subtract_block(tx_size_in_pixels, tx_size_in_pixels, src_diff,
                         diff_stride, src, src_stride, dst, dst_stride);
    }
#else
    vpx_subtract_block(tx_size_in_pixels, tx_size_in_pixels, src_diff,
                       diff_stride, src, src_stride, dst, dst_stride);
#endif
    enable_trellis_opt = do_trellis_opt(pd, src_diff, diff_stride, row, col,
                                        plane_bsize, tx_size, args);
  }

  if (enable_trellis_opt) {
    a = &args->ta[col];
    l = &args->tl[row];
    entropy_ctx = combine_entropy_contexts(*a, *l);
  }

#if CONFIG_VP9_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    uint16_t *const dst16 = CONVERT_TO_SHORTPTR(dst);
    switch (tx_size) {
      case TX_32X32:
        if (!x->skip_recode) {
          highbd_fdct32x32(x->use_lp32x32fdct, src_diff, coeff, diff_stride);
          vpx_highbd_quantize_b_32x32(coeff, p, qcoeff, dqcoeff, pd->dequant,
                                      eob, scan_order);
        }
        if (enable_trellis_opt) {
          *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
        }
        if (!x->skip_encode && *eob) {
          vp9_highbd_idct32x32_add(dqcoeff, dst16, dst_stride, *eob, xd->bd);
        }
        break;
      case TX_16X16:
        if (!x->skip_recode) {
          if (tx_type == DCT_DCT)
            vpx_highbd_fdct16x16(src_diff, coeff, diff_stride);
          else
            vp9_highbd_fht16x16(src_diff, coeff, diff_stride, tx_type);
          vpx_highbd_quantize_b(coeff, 256, p, qcoeff, dqcoeff, pd->dequant,
                                eob, scan_order);
        }
        if (enable_trellis_opt) {
          *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
        }
        if (!x->skip_encode && *eob) {
          vp9_highbd_iht16x16_add(tx_type, dqcoeff, dst16, dst_stride, *eob,
                                  xd->bd);
        }
        break;
      case TX_8X8:
        if (!x->skip_recode) {
          if (tx_type == DCT_DCT)
            vpx_highbd_fdct8x8(src_diff, coeff, diff_stride);
          else
            vp9_highbd_fht8x8(src_diff, coeff, diff_stride, tx_type);
          vpx_highbd_quantize_b(coeff, 64, p, qcoeff, dqcoeff, pd->dequant, eob,
                                scan_order);
        }
        if (enable_trellis_opt) {
          *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
        }
        if (!x->skip_encode && *eob) {
          vp9_highbd_iht8x8_add(tx_type, dqcoeff, dst16, dst_stride, *eob,
                                xd->bd);
        }
        break;
      default:
        assert(tx_size == TX_4X4);
        if (!x->skip_recode) {
          if (tx_type != DCT_DCT)
            vp9_highbd_fht4x4(src_diff, coeff, diff_stride, tx_type);
          else
            x->fwd_txfm4x4(src_diff, coeff, diff_stride);
          vpx_highbd_quantize_b(coeff, 16, p, qcoeff, dqcoeff, pd->dequant, eob,
                                scan_order);
        }
        if (enable_trellis_opt) {
          *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
        }
        if (!x->skip_encode && *eob) {
          if (tx_type == DCT_DCT) {
            // this is like vp9_short_idct4x4 but has a special case around
            // eob<=1 which is significant (not just an optimization) for the
            // lossless case.
            x->highbd_inv_txfm_add(dqcoeff, dst16, dst_stride, *eob, xd->bd);
          } else {
            vp9_highbd_iht4x4_16_add(dqcoeff, dst16, dst_stride, tx_type,
                                     xd->bd);
          }
        }
        break;
    }
    if (*eob) *(args->skip) = 0;
    return;
  }
#endif  // CONFIG_VP9_HIGHBITDEPTH

  switch (tx_size) {
    case TX_32X32:
      if (!x->skip_recode) {
        fdct32x32(x->use_lp32x32fdct, src_diff, coeff, diff_stride);
        vpx_quantize_b_32x32(coeff, p, qcoeff, dqcoeff, pd->dequant, eob,
                             scan_order);
      }
      if (enable_trellis_opt) {
        *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
      }
      if (!x->skip_encode && *eob)
        vp9_idct32x32_add(dqcoeff, dst, dst_stride, *eob);
      break;
    case TX_16X16:
      if (!x->skip_recode) {
        vp9_fht16x16(src_diff, coeff, diff_stride, tx_type);
        vpx_quantize_b(coeff, 256, p, qcoeff, dqcoeff, pd->dequant, eob,
                       scan_order);
      }
      if (enable_trellis_opt) {
        *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
      }
      if (!x->skip_encode && *eob)
        vp9_iht16x16_add(tx_type, dqcoeff, dst, dst_stride, *eob);
      break;
    case TX_8X8:
      if (!x->skip_recode) {
        vp9_fht8x8(src_diff, coeff, diff_stride, tx_type);
        vpx_quantize_b(coeff, 64, p, qcoeff, dqcoeff, pd->dequant, eob,
                       scan_order);
      }
      if (enable_trellis_opt) {
        *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
      }
      if (!x->skip_encode && *eob)
        vp9_iht8x8_add(tx_type, dqcoeff, dst, dst_stride, *eob);
      break;
    default:
      assert(tx_size == TX_4X4);
      if (!x->skip_recode) {
        if (tx_type != DCT_DCT)
          vp9_fht4x4(src_diff, coeff, diff_stride, tx_type);
        else
          x->fwd_txfm4x4(src_diff, coeff, diff_stride);
        vpx_quantize_b(coeff, 16, p, qcoeff, dqcoeff, pd->dequant, eob,
                       scan_order);
      }
      if (enable_trellis_opt) {
        *a = *l = vp9_optimize_b(x, plane, block, tx_size, entropy_ctx) > 0;
      }
      if (!x->skip_encode && *eob) {
        if (tx_type == DCT_DCT)
          // this is like vp9_short_idct4x4 but has a special case around eob<=1
          // which is significant (not just an optimization) for the lossless
          // case.
          x->inv_txfm_add(dqcoeff, dst, dst_stride, *eob);
        else
          vp9_iht4x4_16_add(dqcoeff, dst, dst_stride, tx_type);
      }
      break;
  }
  if (*eob) *(args->skip) = 0;
}

void vp9_encode_intra_block_plane(MACROBLOCK *x, BLOCK_SIZE bsize, int plane,
                                  int enable_trellis_opt) {
  const MACROBLOCKD *const xd = &x->e_mbd;
  struct optimize_ctx ctx;
#if CONFIG_MISMATCH_DEBUG
  // TODO(angiebird): make mismatch_debug support intra mode
  struct encode_b_args arg = {
    x,
    enable_trellis_opt,
    0.0,   // trellis_opt_thresh
    NULL,  // &sse_calc_done
    NULL,  // &sse
    ctx.ta[plane],
    ctx.tl[plane],
    &xd->mi[0]->skip,
    0,  // mi_row
    0,  // mi_col
    0   // output_enabled
  };
#else
  struct encode_b_args arg = { x,
                               enable_trellis_opt,
                               0.0,   // trellis_opt_thresh
                               NULL,  // &sse_calc_done
                               NULL,  // &sse
                               ctx.ta[plane],
                               ctx.tl[plane],
                               &xd->mi[0]->skip };
#endif

  if (enable_trellis_opt && x->optimize &&
      (!x->skip_recode || !x->skip_optimize)) {
    const struct macroblockd_plane *const pd = &xd->plane[plane];
    const TX_SIZE tx_size =
        plane ? get_uv_tx_size(xd->mi[0], pd) : xd->mi[0]->tx_size;
    vp9_get_entropy_contexts(bsize, tx_size, pd, ctx.ta[plane], ctx.tl[plane]);
  } else {
    arg.enable_trellis_opt = 0;
  }

  vp9_foreach_transformed_block_in_plane(xd, bsize, plane,
                                         vp9_encode_block_intra, &arg);
}
