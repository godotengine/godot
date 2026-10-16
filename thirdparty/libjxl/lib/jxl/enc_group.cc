// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_group.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/memory_manager_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_group.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quantizer-inl.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/simd_util.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::MaskFromVec;
using hwy::HWY_NAMESPACE::Round;

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeBlockAC(const Quantizer& quantizer, const bool error_diffusion,
                     size_t c, float qm_multiplier, AcStrategyType quant_kind,
                     size_t xsize, size_t ysize, float* thresholds,
                     const float* JXL_RESTRICT block_in, const int32_t* quant,
                     int32_t* JXL_RESTRICT block_out) {
  const float* JXL_RESTRICT qm = quantizer.InvDequantMatrix(quant_kind, c);
  float qac = quantizer.Scale() * (*quant);
  // Not SIMD-ified for now.
  if (c != 1 && xsize * ysize >= 4) {
    for (int i = 0; i < 4; ++i) {
      thresholds[i] -= 0.00744f * xsize * ysize;
      if (thresholds[i] < 0.5) {
        thresholds[i] = 0.5;
      }
    }
  }
  HWY_CAPPED(float, kBlockDim) df;
  HWY_CAPPED(int32_t, kBlockDim) di;
  HWY_CAPPED(uint32_t, kBlockDim) du;
  const auto quantv = Set(df, qac * qm_multiplier);
  for (size_t y = 0; y < ysize * kBlockDim; y++) {
    size_t yfix = static_cast<size_t>(y >= ysize * kBlockDim / 2) * 2;
    const size_t off = y * kBlockDim * xsize;
    for (size_t x = 0; x < xsize * kBlockDim; x += Lanes(df)) {
      auto threshold = Zero(df);
      if (xsize == 1) {
        HWY_ALIGN uint32_t kMask[kBlockDim] = {0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u};
        const auto mask = MaskFromVec(BitCast(df, Load(du, kMask + x)));
        threshold = IfThenElse(mask, Set(df, thresholds[yfix + 1]),
                               Set(df, thresholds[yfix]));
      } else {
        // Same for all lanes in the vector.
        threshold = Set(
            df,
            thresholds[yfix + static_cast<size_t>(x >= xsize * kBlockDim / 2)]);
      }
      const auto q = Mul(Load(df, qm + off + x), quantv);
      const auto in = Load(df, block_in + off + x);
      const auto val = Mul(q, in);
      const auto nzero_mask = Ge(Abs(val), threshold);
      const auto v = ConvertTo(di, IfThenElseZero(nzero_mask, Round(val)));
      Store(v, di, block_out + off + x);
    }
  }
}

void AdjustQuantBlockAC(const Quantizer& quantizer, size_t c,
                        float qm_multiplier, AcStrategyType quant_kind,
                        size_t xsize, size_t ysize, float* thresholds,
                        const float* JXL_RESTRICT block_in, int32_t* quant) {
  // No quantization adjusting for these small blocks.
  // Quantization adjusting attempts to fix some known issues
  // with larger blocks and on the 8x8 dct's emerging 8x8 blockiness
  // when there are not many non-zeros.
  constexpr size_t kPartialBlockKinds =
      (1 << static_cast<size_t>(AcStrategyType::IDENTITY)) |
      (1 << static_cast<size_t>(AcStrategyType::DCT2X2)) |
      (1 << static_cast<size_t>(AcStrategyType::DCT4X4)) |
      (1 << static_cast<size_t>(AcStrategyType::DCT4X8)) |
      (1 << static_cast<size_t>(AcStrategyType::DCT8X4)) |
      (1 << static_cast<size_t>(AcStrategyType::AFV0)) |
      (1 << static_cast<size_t>(AcStrategyType::AFV1)) |
      (1 << static_cast<size_t>(AcStrategyType::AFV2)) |
      (1 << static_cast<size_t>(AcStrategyType::AFV3));
  if ((1 << static_cast<size_t>(quant_kind)) & kPartialBlockKinds) {
    return;
  }

  const float* JXL_RESTRICT qm = quantizer.InvDequantMatrix(quant_kind, c);
  float qac = quantizer.Scale() * (*quant);
  if (xsize > 1 || ysize > 1) {
    for (int i = 0; i < 4; ++i) {
      thresholds[i] -= Clamp1(0.003f * xsize * ysize, 0.f, 0.08f);
      if (thresholds[i] < 0.54) {
        thresholds[i] = 0.54;
      }
    }
  }
  float sum_of_highest_freq_row_and_column = 0;
  float sum_of_error = 0;
  float sum_of_vals = 0;
  float hfNonZeros[4] = {};
  float hfMaxError[4] = {};

  for (size_t y = 0; y < ysize * kBlockDim; y++) {
    for (size_t x = 0; x < xsize * kBlockDim; x++) {
      const size_t pos = y * kBlockDim * xsize + x;
      if (x < xsize && y < ysize) {
        continue;
      }
      const size_t hfix = (static_cast<size_t>(y >= ysize * kBlockDim / 2) * 2 +
                           static_cast<size_t>(x >= xsize * kBlockDim / 2));
      const float val = block_in[pos] * (qm[pos] * qac * qm_multiplier);
      const float v = (std::abs(val) < thresholds[hfix]) ? 0 : rintf(val);
      const float error = std::abs(val - v);
      sum_of_error += error;
      sum_of_vals += std::abs(v);
      if (c == 1 && v == 0) {
        if (hfMaxError[hfix] < error) {
          hfMaxError[hfix] = error;
        }
      }
      if (v != 0.0f) {
        hfNonZeros[hfix] += std::abs(v);
        bool in_corner = y >= 7 * ysize && x >= 7 * xsize;
        bool on_border =
            y == ysize * kBlockDim - 1 || x == xsize * kBlockDim - 1;
        bool in_larger_corner = x >= 4 * xsize && y >= 4 * ysize;
        if (in_corner || (on_border && in_larger_corner)) {
          sum_of_highest_freq_row_and_column += std::abs(val);
        }
      }
    }
  }
  if (c == 1 && sum_of_vals * 8 < xsize * ysize) {
    static const double kLimit[4] = {
        0.46,
        0.46,
        0.46,
        0.46,
    };
    static const double kMul[4] = {
        0.9999,
        0.9999,
        0.9999,
        0.9999,
    };
    const int32_t orig_quant = *quant;
    int32_t new_quant = *quant;
    for (int i = 1; i < 4; ++i) {
      if (hfNonZeros[i] == 0.0 && hfMaxError[i] > kLimit[i]) {
        new_quant = orig_quant + 1;
        break;
      }
    }
    *quant = new_quant;
    if (hfNonZeros[3] == 0.0 && hfMaxError[3] > kLimit[3]) {
      thresholds[3] = kMul[3] * hfMaxError[3] * new_quant / orig_quant;
    } else if ((hfNonZeros[1] == 0.0 && hfMaxError[1] > kLimit[1]) ||
               (hfNonZeros[2] == 0.0 && hfMaxError[2] > kLimit[2])) {
      thresholds[1] = kMul[1] * std::max(hfMaxError[1], hfMaxError[2]) *
                      new_quant / orig_quant;
      thresholds[2] = thresholds[1];
    } else if (hfNonZeros[0] == 0.0 && hfMaxError[0] > kLimit[0]) {
      thresholds[0] = kMul[0] * hfMaxError[0] * new_quant / orig_quant;
    }
  }
  // Heuristic for improving accuracy of high-frequency patterns
  // occurring in an environment with no medium-frequency masking
  // patterns.
  {
    float all =
        hfNonZeros[0] + hfNonZeros[1] + hfNonZeros[2] + hfNonZeros[3] + 1;
    float mul[3] = {70, 30, 60};
    if (mul[c] * sum_of_highest_freq_row_and_column >= all) {
      *quant += mul[c] * sum_of_highest_freq_row_and_column / all;
      if (*quant >= Quantizer::kQuantMax) {
        *quant = Quantizer::kQuantMax - 1;
      }
    }
  }
  if (quant_kind == AcStrategyType::DCT) {
    // If this 8x8 block is too flat, increase the adaptive quantization level
    // a bit to reduce visible block boundaries and requantize the block.
    if (hfNonZeros[0] + hfNonZeros[1] + hfNonZeros[2] + hfNonZeros[3] < 11) {
      *quant += 1;
      if (*quant >= Quantizer::kQuantMax) {
        *quant = Quantizer::kQuantMax - 1;
      }
    }
  }
  {
    static const double kMul1[4][3] = {
        {
            0.22080615753848404,
            0.45797479824262011,
            0.29859235095977965,
        },
        {
            0.70109486510286834,
            0.16185281305512639,
            0.14387691730035473,
        },
        {
            0.114985964456218638,
            0.44656840441027695,
            0.10587658215149048,
        },
        {
            0.46849665264409396,
            0.41239077937781954,
            0.088667407767185444,
        },
    };
    static const double kMul2[4][3] = {
        {
            0.27450281941822197,
            1.1255766549984996,
            0.98950459134128388,
        },
        {
            0.4652168675598285,
            0.40945807983455818,
            0.36581899811751367,
        },
        {
            0.28034972424715715,
            0.9182653201929738,
            1.5581531543057416,
        },
        {
            0.26873118114033728,
            0.68863712390392484,
            1.2082185408666786,
        },
    };
    static const double kQuantNormalizer = 2.2942708343284721;
    sum_of_error *= kQuantNormalizer;
    sum_of_vals *= kQuantNormalizer;
    if (quant_kind >= AcStrategyType::DCT16X16) {
      int ix = 3;
      if (quant_kind == AcStrategyType::DCT32X16 ||
          quant_kind == AcStrategyType::DCT16X32) {
        ix = 1;
      } else if (quant_kind == AcStrategyType::DCT16X16) {
        ix = 0;
      } else if (quant_kind == AcStrategyType::DCT32X32) {
        ix = 2;
      }
      int step =
          sum_of_error / (kMul1[ix][c] * xsize * ysize * kBlockDim * kBlockDim +
                          kMul2[ix][c] * sum_of_vals);
      if (step >= 2) {
        step = 2;
      }
      if (step < 0) {
        step = 0;
      }
      if (sum_of_error > kMul1[ix][c] * xsize * ysize * kBlockDim * kBlockDim +
                             kMul2[ix][c] * sum_of_vals) {
        *quant += step;
        if (*quant >= Quantizer::kQuantMax) {
          *quant = Quantizer::kQuantMax - 1;
        }
      }
    }
  }
  {
    // Reduce quant in highly active areas.
    int32_t div = (xsize * ysize);
    float min_hf_non_zeros =
        std::min({hfNonZeros[0], hfNonZeros[1], hfNonZeros[2], hfNonZeros[3]});
    int32_t activity = 15;
    if (min_hf_non_zeros < 15.0f * div) {
      activity = (static_cast<int32_t>(min_hf_non_zeros) + div / 2) / div;
    }
    int32_t qp = *quant - activity;
    if (c == 1) {
      for (int i = 1; i < 4; ++i) {
        thresholds[i] += 0.01 * activity;
      }
    }
    int32_t orig_qp_limit = std::max(4, *quant / 2);
    if (qp < orig_qp_limit) {
      qp = orig_qp_limit;
    }
    *quant = qp;
  }
}

// NOTE: caller takes care of extracting quant from rect of RawQuantField.
void QuantizeRoundtripYBlockAC(PassesEncoderState* enc_state, const size_t size,
                               const Quantizer& quantizer,
                               const bool error_diffusion,
                               AcStrategyType quant_kind, size_t xsize,
                               size_t ysize, const float* JXL_RESTRICT biases,
                               int32_t* quant, float* JXL_RESTRICT inout,
                               int32_t* JXL_RESTRICT quantized) {
  float thres_y[4] = {0.58f, 0.64f, 0.64f, 0.64f};
  if (enc_state->cparams.speed_tier <= SpeedTier::kHare) {
    int32_t max_quant = 0;
    int quant_orig = *quant;
    float val[3] = {enc_state->x_qm_multiplier, 1.0f,
                    enc_state->b_qm_multiplier};
    for (int c : {1, 0, 2}) {
      float thres[4] = {0.58f, 0.64f, 0.64f, 0.64f};
      *quant = quant_orig;
      AdjustQuantBlockAC(quantizer, c, val[c], quant_kind, xsize, ysize,
                         &thres[0], inout + c * size, quant);
      // Dead zone adjustment
      if (c == 1) {
        for (int k = 0; k < 4; ++k) {
          thres_y[k] = thres[k];
        }
      }
      max_quant = std::max(*quant, max_quant);
    }
    *quant = max_quant;
  } else {
    thres_y[0] = 0.56;
    thres_y[1] = 0.62;
    thres_y[2] = 0.62;
    thres_y[3] = 0.62;
  }

  QuantizeBlockAC(quantizer, error_diffusion, 1, 1.0f, quant_kind, xsize, ysize,
                  &thres_y[0], inout + size, quant, quantized + size);

  const float* JXL_RESTRICT dequant_matrix =
      quantizer.DequantMatrix(quant_kind, 1);

  HWY_CAPPED(float, kDCTBlockSize) df;
  HWY_CAPPED(int32_t, kDCTBlockSize) di;
  const auto inv_qac = Set(df, quantizer.inv_quant_ac(*quant));
  for (size_t k = 0; k < kDCTBlockSize * xsize * ysize; k += Lanes(df)) {
    const auto oquant = Load(di, quantized + size + k);
    const auto adj_quant = AdjustQuantBias(di, 1, oquant, biases);
    const auto dequantm = Load(df, dequant_matrix + k);
    Store(Mul(Mul(adj_quant, dequantm), inv_qac), df, inout + size + k);
  }
}

Status ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                           const Image3F& opsin, const Rect& rect,
                           Image3F* dc) {
  JxlMemoryManager* memory_manager = opsin.memory_manager();
  const Rect block_group_rect =
      enc_state->shared.frame_dim.BlockGroupRect(group_idx);
  const Rect cmap_rect(
      block_group_rect.x0() / kColorTileDimInBlocks,
      block_group_rect.y0() / kColorTileDimInBlocks,
      DivCeil(block_group_rect.xsize(), kColorTileDimInBlocks),
      DivCeil(block_group_rect.ysize(), kColorTileDimInBlocks));
  const Rect group_rect =
      enc_state->shared.frame_dim.GroupRect(group_idx).Translate(rect.x0(),
                                                                 rect.y0());

  const size_t xsize_blocks = block_group_rect.xsize();
  const size_t ysize_blocks = block_group_rect.ysize();

  const size_t dc_stride = static_cast<size_t>(dc->PixelsPerRow());
  const size_t opsin_stride = static_cast<size_t>(opsin.PixelsPerRow());

  ImageI& full_quant_field = enc_state->shared.raw_quant_field;
  const CompressParams& cparams = enc_state->cparams;

  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;

  // TODO(veluca): consider strategies to reduce this memory.
  size_t mem_bytes = 3 * AcStrategy::kMaxCoeffArea * sizeof(int32_t);
  JXL_ASSIGN_OR_RETURN(auto mem,
                       AlignedMemory::Create(memory_manager, mem_bytes));
  size_t fmem_bytes =
      (5 * AcStrategy::kMaxCoeffArea + dct_scratch_size) * sizeof(float);
  JXL_ASSIGN_OR_RETURN(auto fmem,
                       AlignedMemory::Create(memory_manager, fmem_bytes));
  float* JXL_RESTRICT scratch_space =
      fmem.address<float>() + 3 * AcStrategy::kMaxCoeffArea;
  {
    // Only use error diffusion in Squirrel mode or slower.
    const bool error_diffusion = cparams.speed_tier <= SpeedTier::kSquirrel;
    constexpr HWY_CAPPED(float, kDCTBlockSize) d;

    int32_t* JXL_RESTRICT coeffs[3][kMaxNumPasses] = {};
    size_t num_passes = enc_state->progressive_splitter.GetNumPasses();
    JXL_ENSURE(num_passes > 0);
    for (size_t i = 0; i < num_passes; i++) {
      // TODO(veluca): 16-bit quantized coeffs are not implemented yet.
      JXL_ENSURE(enc_state->coeffs[i]->Type() == ACType::k32);
      for (size_t c = 0; c < 3; c++) {
        coeffs[c][i] = enc_state->coeffs[i]->PlaneRow(c, group_idx, 0).ptr32;
      }
    }

    HWY_ALIGN float* coeffs_in = fmem.address<float>();
    HWY_ALIGN int32_t* quantized = mem.address<int32_t>();

    for (size_t by = 0; by < ysize_blocks; ++by) {
      int32_t* JXL_RESTRICT row_quant_ac =
          block_group_rect.Row(&full_quant_field, by);
      size_t ty = by / kColorTileDimInBlocks;
      const int8_t* JXL_RESTRICT row_cmap[3] = {
          cmap_rect.ConstRow(enc_state->shared.cmap.ytox_map, ty),
          nullptr,
          cmap_rect.ConstRow(enc_state->shared.cmap.ytob_map, ty),
      };
      const float* JXL_RESTRICT opsin_rows[3] = {
          group_rect.ConstPlaneRow(opsin, 0, by * kBlockDim),
          group_rect.ConstPlaneRow(opsin, 1, by * kBlockDim),
          group_rect.ConstPlaneRow(opsin, 2, by * kBlockDim),
      };
      float* JXL_RESTRICT dc_rows[3] = {
          block_group_rect.PlaneRow(dc, 0, by),
          block_group_rect.PlaneRow(dc, 1, by),
          block_group_rect.PlaneRow(dc, 2, by),
      };
      AcStrategyRow ac_strategy_row =
          enc_state->shared.ac_strategy.ConstRow(block_group_rect, by);
      for (size_t tx = 0; tx < DivCeil(xsize_blocks, kColorTileDimInBlocks);
           tx++) {
        const auto x_factor =
            Set(d, enc_state->shared.cmap.base().YtoXRatio(row_cmap[0][tx]));
        const auto b_factor =
            Set(d, enc_state->shared.cmap.base().YtoBRatio(row_cmap[2][tx]));
        for (size_t bx = tx * kColorTileDimInBlocks;
             bx < xsize_blocks && bx < (tx + 1) * kColorTileDimInBlocks; ++bx) {
          const AcStrategy acs = ac_strategy_row[bx];
          if (!acs.IsFirstBlock()) continue;

          size_t xblocks = acs.covered_blocks_x();
          size_t yblocks = acs.covered_blocks_y();

          CoefficientLayout(&yblocks, &xblocks);

          size_t size = kDCTBlockSize * xblocks * yblocks;

          // DCT Y channel, roundtrip-quantize it and set DC.
          int32_t quant_ac = row_quant_ac[bx];
          for (size_t c : {0, 1, 2}) {
            TransformFromPixels(acs.Strategy(), opsin_rows[c] + bx * kBlockDim,
                                opsin_stride, coeffs_in + c * size,
                                scratch_space);
          }
          DCFromLowestFrequencies(acs.Strategy(), coeffs_in + size,
                                  dc_rows[1] + bx, dc_stride, scratch_space);

          QuantizeRoundtripYBlockAC(
              enc_state, size, enc_state->shared.quantizer, error_diffusion,
              acs.Strategy(), xblocks, yblocks, kDefaultQuantBias, &quant_ac,
              coeffs_in, quantized);

          // Unapply color correlation
          for (size_t k = 0; k < size; k += Lanes(d)) {
            const auto in_x = Load(d, coeffs_in + k);
            const auto in_y = Load(d, coeffs_in + size + k);
            const auto in_b = Load(d, coeffs_in + 2 * size + k);
            const auto out_x = NegMulAdd(x_factor, in_y, in_x);
            const auto out_b = NegMulAdd(b_factor, in_y, in_b);
            Store(out_x, d, coeffs_in + k);
            Store(out_b, d, coeffs_in + 2 * size + k);
          }

          // Quantize X and B channels and set DC.
          for (size_t c : {0, 2}) {
            float thres[4] = {0.58f, 0.62f, 0.62f, 0.62f};
            QuantizeBlockAC(enc_state->shared.quantizer, error_diffusion, c,
                            c == 0 ? enc_state->x_qm_multiplier
                                   : enc_state->b_qm_multiplier,
                            acs.Strategy(), xblocks, yblocks, &thres[0],
                            coeffs_in + c * size, &quant_ac,
                            quantized + c * size);
            DCFromLowestFrequencies(acs.Strategy(), coeffs_in + c * size,
                                    dc_rows[c] + bx, dc_stride, scratch_space);
          }
          row_quant_ac[bx] = quant_ac;
          for (size_t c = 0; c < 3; c++) {
            enc_state->progressive_splitter.SplitACCoefficients(
                quantized + c * size, acs, bx, by, coeffs[c]);
            for (size_t p = 0; p < num_passes; p++) {
              coeffs[c][p] += size;
            }
          }
        }
      }
    }
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ComputeCoefficients);
Status ComputeCoefficients(size_t group_idx, PassesEncoderState* enc_state,
                           const Image3F& opsin, const Rect& rect,
                           Image3F* dc) {
  return HWY_DYNAMIC_DISPATCH(ComputeCoefficients)(group_idx, enc_state, opsin,
                                                   rect, dc);
}

Status EncodeGroupTokenizedCoefficients(size_t group_idx, size_t pass_idx,
                                        size_t histogram_idx,
                                        const PassesEncoderState& enc_state,
                                        BitWriter* writer, AuxOut* aux_out) {
  // Select which histogram to use among those of the current pass.
  const size_t num_histograms = enc_state.shared.num_histograms;
  // num_histograms is 0 only for lossless.
  JXL_ENSURE(num_histograms == 0 || histogram_idx < num_histograms);
  size_t histo_selector_bits = CeilLog2Nonzero(num_histograms);

  if (histo_selector_bits != 0) {
    JXL_RETURN_IF_ERROR(
        writer->WithMaxBits(histo_selector_bits, LayerType::Ac, aux_out, [&] {
          writer->Write(histo_selector_bits, histogram_idx);
          return true;
        }));
  }
  size_t context_offset =
      histogram_idx * enc_state.shared.block_ctx_map.NumACContexts();
  JXL_RETURN_IF_ERROR(
      WriteTokens(enc_state.passes[pass_idx].ac_tokens[group_idx],
                  enc_state.passes[pass_idx].codes, context_offset, writer,
                  LayerType::AcTokens, aux_out));

  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
