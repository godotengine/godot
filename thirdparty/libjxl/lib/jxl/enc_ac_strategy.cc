// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_ac_strategy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/common.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/quant_weights.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_ac_strategy.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_debug_image.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/simd_util.h"

// Some of the floating point constants in this file and in other
// files in the libjxl project have been obtained using the
// tools/optimizer/simplex_fork.py tool. It is a variation of
// Nelder-Mead optimization, and we generally try to minimize
// BPP * pnorm aggregate as reported by the benchmark_xl tool,
// but occasionally the values are optimized by using additional
// constraints such as maintaining a certain density, or ratio of
// popularity of integral transforms. Jyrki visually reviews all
// such changes and often makes manual changes to maintain good
// visual quality to changes where butteraugli was not sufficiently
// sensitive to some kind of degradation. Unfortunately image quality
// is still more of an art than science.

// Set JXL_DEBUG_AC_STRATEGY to 1 to enable debugging.
#ifndef JXL_DEBUG_AC_STRATEGY
#define JXL_DEBUG_AC_STRATEGY 0
#endif

// This must come before the begin/end_target, but HWY_ONCE is only true
// after that, so use an "include guard".
#ifndef LIB_JXL_ENC_AC_STRATEGY_
#define LIB_JXL_ENC_AC_STRATEGY_
// Parameters of the heuristic are marked with a OPTIMIZE comment.
namespace jxl {
namespace {

// Debugging utilities.

// Returns a linear sRGB color (as bytes) for each AC strategy.
const uint8_t* TypeColor(uint8_t raw_strategy) {
  JXL_DASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 27, "Update colors");
  static constexpr uint8_t kColors[AcStrategy::kNumValidStrategies + 1][3] = {
      {0xFF, 0xFF, 0x00},  // DCT8       | yellow
      {0xFF, 0x80, 0x80},  // HORNUSS    | vivid tangerine
      {0xFF, 0x80, 0x80},  // DCT2x2     | vivid tangerine
      {0xFF, 0x80, 0x80},  // DCT4x4     | vivid tangerine
      {0x80, 0xFF, 0x00},  // DCT16x16   | chartreuse
      {0x00, 0xC0, 0x00},  // DCT32x32   | waystone green
      {0xC0, 0xFF, 0x00},  // DCT16x8    | lime
      {0xC0, 0xFF, 0x00},  // DCT8x16    | lime
      {0x00, 0xFF, 0x00},  // DCT32x8    | green
      {0x00, 0xFF, 0x00},  // DCT8x32    | green
      {0x00, 0xFF, 0x00},  // DCT32x16   | green
      {0x00, 0xFF, 0x00},  // DCT16x32   | green
      {0xFF, 0x80, 0x00},  // DCT4x8     | orange juice
      {0xFF, 0x80, 0x00},  // DCT8x4     | orange juice
      {0xFF, 0xFF, 0x80},  // AFV0       | butter
      {0xFF, 0xFF, 0x80},  // AFV1       | butter
      {0xFF, 0xFF, 0x80},  // AFV2       | butter
      {0xFF, 0xFF, 0x80},  // AFV3       | butter
      {0x00, 0xC0, 0xFF},  // DCT64x64   | capri
      {0x00, 0xFF, 0xFF},  // DCT64x32   | aqua
      {0x00, 0xFF, 0xFF},  // DCT32x64   | aqua
      {0x00, 0x40, 0xFF},  // DCT128x128 | rare blue
      {0x00, 0x80, 0xFF},  // DCT128x64  | magic ink
      {0x00, 0x80, 0xFF},  // DCT64x128  | magic ink
      {0x00, 0x00, 0xC0},  // DCT256x256 | keese blue
      {0x00, 0x00, 0xFF},  // DCT256x128 | blue
      {0x00, 0x00, 0xFF},  // DCT128x256 | blue
      {0x00, 0x00, 0x00}   // invalid    | black
  };
  raw_strategy =
      Clamp1<uint8_t>(raw_strategy, 0, AcStrategy::kNumValidStrategies);
  return kColors[raw_strategy];
}

const uint8_t* TypeMask(uint8_t raw_strategy) {
  JXL_DASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 27, "Update masks");
  // implicitly, first row and column is made dark
  static constexpr uint8_t kMask[AcStrategy::kNumValidStrategies + 1][64] = {
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT8
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // HORNUSS
      {
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
      },                           // 2x2
      {
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
      },                           // 4x4
      {},                          // DCT16x16 (unused)
      {},                          // DCT32x32 (unused)
      {},                          // DCT16x8 (unused)
      {},                          // DCT8x16 (unused)
      {},                          // DCT32x8 (unused)
      {},                          // DCT8x32 (unused)
      {},                          // DCT32x16 (unused)
      {},                          // DCT16x32 (unused)
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT4x8
      {
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
      },                           // DCT8x4
      {
          1, 1, 1, 1, 1, 0, 0, 0,  //
          1, 1, 1, 1, 0, 0, 0, 0,  //
          1, 1, 1, 0, 0, 0, 0, 0,  //
          1, 1, 0, 0, 0, 0, 0, 0,  //
          1, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV0
      {
          0, 0, 0, 0, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV1
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 0, 0, 0, 0,  //
      },                           // AFV2
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
          0, 0, 0, 0, 0, 0, 1, 1,  //
          0, 0, 0, 0, 0, 1, 1, 1,  //
      },                           // AFV3
      {}                           // invalid
  };
  raw_strategy =
      Clamp1<uint8_t>(raw_strategy, 0, AcStrategy::kNumValidStrategies);
  return kMask[raw_strategy];
}

Status DumpAcStrategy(const AcStrategyImage& ac_strategy, size_t xsize,
                      size_t ysize, const char* tag, AuxOut* aux_out,
                      const CompressParams& cparams) {
  JxlMemoryManager* memory_manager = ac_strategy.memory_manager();
  JXL_ASSIGN_OR_RETURN(Image3F color_acs,
                       Image3F::Create(memory_manager, xsize, ysize));
  for (size_t y = 0; y < ysize; y++) {
    float* JXL_RESTRICT rows[3] = {
        color_acs.PlaneRow(0, y),
        color_acs.PlaneRow(1, y),
        color_acs.PlaneRow(2, y),
    };
    const AcStrategyRow acs_row = ac_strategy.ConstRow(y / kBlockDim);
    for (size_t x = 0; x < xsize; x++) {
      AcStrategy acs = acs_row[x / kBlockDim];
      const uint8_t* JXL_RESTRICT color = TypeColor(acs.RawStrategy());
      for (size_t c = 0; c < 3; c++) {
        rows[c][x] = color[c] / 255.f;
      }
    }
  }
  size_t stride = color_acs.PixelsPerRow();
  for (size_t c = 0; c < 3; c++) {
    for (size_t by = 0; by < DivCeil(ysize, kBlockDim); by++) {
      float* JXL_RESTRICT row = color_acs.PlaneRow(c, by * kBlockDim);
      const AcStrategyRow acs_row = ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < DivCeil(xsize, kBlockDim); bx++) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const uint8_t* JXL_RESTRICT color = TypeColor(acs.RawStrategy());
        const uint8_t* JXL_RESTRICT mask = TypeMask(acs.RawStrategy());
        if (acs.covered_blocks_x() == 1 && acs.covered_blocks_y() == 1) {
          for (size_t iy = 0; iy < kBlockDim && by * kBlockDim + iy < ysize;
               iy++) {
            for (size_t ix = 0; ix < kBlockDim && bx * kBlockDim + ix < xsize;
                 ix++) {
              if (mask[iy * kBlockDim + ix]) {
                row[iy * stride + bx * kBlockDim + ix] = color[c] / 800.f;
              }
            }
          }
        }
        // draw block edges
        for (size_t ix = 0; ix < kBlockDim * acs.covered_blocks_x() &&
                            bx * kBlockDim + ix < xsize;
             ix++) {
          row[0 * stride + bx * kBlockDim + ix] = color[c] / 350.f;
        }
        for (size_t iy = 0; iy < kBlockDim * acs.covered_blocks_y() &&
                            by * kBlockDim + iy < ysize;
             iy++) {
          row[iy * stride + bx * kBlockDim + 0] = color[c] / 350.f;
        }
      }
    }
  }
  return DumpImage(cparams, tag, color_acs);
}

}  // namespace
}  // namespace jxl
#endif  // LIB_JXL_ENC_AC_STRATEGY_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::AbsDiff;
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::IfThenZeroElse;
using hwy::HWY_NAMESPACE::Round;
using hwy::HWY_NAMESPACE::Sqrt;

bool MultiBlockTransformCrossesHorizontalBoundary(
    const AcStrategyImage& ac_strategy, size_t start_x, size_t y,
    size_t end_x) {
  if (start_x >= ac_strategy.xsize() || y >= ac_strategy.ysize()) {
    return false;
  }
  if (y % 8 == 0) {
    // Nothing crosses 64x64 boundaries, and the memory on the other side
    // of the 64x64 block may still uninitialized.
    return false;
  }
  end_x = std::min(end_x, ac_strategy.xsize());
  // The first multiblock might be before the start_x, let's adjust it
  // to point to the first IsFirstBlock() == true block we find by backward
  // tracing.
  AcStrategyRow row = ac_strategy.ConstRow(y);
  const size_t start_x_limit = start_x & ~7;
  while (start_x != start_x_limit && !row[start_x].IsFirstBlock()) {
    --start_x;
  }
  for (size_t x = start_x; x < end_x;) {
    if (row[x].IsFirstBlock()) {
      x += row[x].covered_blocks_x();
    } else {
      return true;
    }
  }
  return false;
}

bool MultiBlockTransformCrossesVerticalBoundary(
    const AcStrategyImage& ac_strategy, size_t x, size_t start_y,
    size_t end_y) {
  if (x >= ac_strategy.xsize() || start_y >= ac_strategy.ysize()) {
    return false;
  }
  if (x % 8 == 0) {
    // Nothing crosses 64x64 boundaries, and the memory on the other side
    // of the 64x64 block may still uninitialized.
    return false;
  }
  end_y = std::min(end_y, ac_strategy.ysize());
  // The first multiblock might be before the start_y, let's adjust it
  // to point to the first IsFirstBlock() == true block we find by backward
  // tracing.
  const size_t start_y_limit = start_y & ~7;
  while (start_y != start_y_limit &&
         !ac_strategy.ConstRow(start_y)[x].IsFirstBlock()) {
    --start_y;
  }

  for (size_t y = start_y; y < end_y;) {
    AcStrategyRow row = ac_strategy.ConstRow(y);
    if (row[x].IsFirstBlock()) {
      y += row[x].covered_blocks_y();
    } else {
      return true;
    }
  }
  return false;
}

Status EstimateEntropy(const AcStrategy& acs, float entropy_mul, size_t x,
                       size_t y, const ACSConfig& config,
                       const float* JXL_RESTRICT cmap_factors, float* block,
                       float* full_scratch_space, uint32_t* quantized,
                       float& entropy) {
  entropy = 0.0f;
  float* mem = full_scratch_space;
  float* scratch_space = full_scratch_space + AcStrategy::kMaxCoeffArea;
  const size_t size = (1 << acs.log2_covered_blocks()) * kDCTBlockSize;

  // Apply transform.
  for (size_t c = 0; c < 3; c++) {
    float* JXL_RESTRICT block_c = block + size * c;
    TransformFromPixels(acs.Strategy(), &config.Pixel(c, x, y),
                        config.src_stride, block_c, scratch_space);
  }
  HWY_FULL(float) df;

  const size_t num_blocks = acs.covered_blocks_x() * acs.covered_blocks_y();
  // avoid large blocks when there is a lot going on in red-green.
  float quant_norm16 = 0;
  if (num_blocks == 1) {
    // When it is only one 8x8, we don't need aggregation of values.
    quant_norm16 = config.Quant(x / 8, y / 8);
  } else if (num_blocks == 2) {
    // Taking max instead of 8th norm seems to work
    // better for smallest blocks up to 16x8. Jyrki couldn't get
    // improvements in trying the same for 16x16 blocks.
    if (acs.covered_blocks_y() == 2) {
      quant_norm16 =
          std::max(config.Quant(x / 8, y / 8), config.Quant(x / 8, y / 8 + 1));
    } else {
      quant_norm16 =
          std::max(config.Quant(x / 8, y / 8), config.Quant(x / 8 + 1, y / 8));
    }
  } else {
    // Load QF value, calculate empirical heuristic on masking field
    // for weighting the information loss. Information loss manifests
    // itself as ringing, and masking could hide it.
    for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
      for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
        float qval = config.Quant(x / 8 + ix, y / 8 + iy);
        qval *= qval;
        qval *= qval;
        qval *= qval;
        quant_norm16 += qval * qval;
      }
    }
    quant_norm16 /= num_blocks;
    quant_norm16 = FastPowf(quant_norm16, 1.0f / 16.0f);
  }
  const auto quant = Set(df, quant_norm16);

  // Compute entropy.
  const HWY_CAPPED(float, 8) df8;

  auto loss = Zero(df8);
  for (size_t c = 0; c < 3; c++) {
    const float* inv_matrix = config.dequant->InvMatrix(acs.Strategy(), c);
    const float* matrix = config.dequant->Matrix(acs.Strategy(), c);
    const auto cmap_factor = Set(df, cmap_factors[c]);

    auto entropy_v = Zero(df);
    auto nzeros_v = Zero(df);
    for (size_t i = 0; i < num_blocks * kDCTBlockSize; i += Lanes(df)) {
      const auto in = Load(df, block + c * size + i);
      const auto in_y = Mul(Load(df, block + size + i), cmap_factor);
      const auto im = Load(df, inv_matrix + i);
      const auto val = Mul(Sub(in, in_y), Mul(im, quant));
      const auto rval = Round(val);
      const auto diff = Sub(val, rval);
      const auto m = Load(df, matrix + i);
      Store(Mul(m, diff), df, &mem[i]);
      const auto q = Abs(rval);
      const auto q_is_zero = Eq(q, Zero(df));
      // We used to have q * C here, but that cost model seems to
      // be punishing large values more than necessary. Sqrt tries
      // to avoid large values less aggressively.
      entropy_v = Add(Sqrt(q), entropy_v);
      nzeros_v = Add(nzeros_v, IfThenZeroElse(q_is_zero, Set(df, 1.0f)));
    }

    {
      float masku_lut[3] = {
          12.0,
          0.0,
          4.0,
      };
      auto masku_off = Set(df8, masku_lut[c]);
      auto lossc = Zero(df8);
      TransformToPixels(acs.Strategy(), &mem[0], block,
                        acs.covered_blocks_x() * 8, scratch_space);

      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          for (size_t dy = 0; dy < kBlockDim; ++dy) {
            for (size_t dx = 0; dx < kBlockDim; dx += Lanes(df8)) {
              auto in = Load(df8, block +
                                      (iy * kBlockDim + dy) *
                                          (acs.covered_blocks_x() * kBlockDim) +
                                      ix * kBlockDim + dx);
              if (x + ix * 8 + dx + Lanes(df8) <= config.mask1x1_xsize) {
                auto masku =
                    Add(Load(df8, config.MaskingPtr1x1(x + ix * 8 + dx,
                                                       y + iy * 8 + dy)),
                        masku_off);
                in = Mul(masku, in);
                in = Mul(in, in);
                in = Mul(in, in);
                in = Mul(in, in);
                lossc = Add(lossc, in);
              }
            }
          }
        }
      }
      static const double kChannelMul[3] = {
          pow(8.2, 8.0),
          pow(1.0, 8.0),
          pow(1.03, 8.0),
      };
      lossc = Mul(Set(df8, kChannelMul[c]), lossc);
      loss = Add(loss, lossc);
    }
    entropy += config.cost_delta * GetLane(SumOfLanes(df, entropy_v));
    size_t num_nzeros = GetLane(SumOfLanes(df, nzeros_v));
    // Add #bit of num_nonzeros, as an estimate of the cost for encoding the
    // number of non-zeros of the block.
    size_t nbits = CeilLog2Nonzero(num_nzeros + 1) + 1;
    // Also add #bit of #bit of num_nonzeros, to estimate the ANS cost, with a
    // bias.
    entropy += config.zeros_mul * (CeilLog2Nonzero(nbits + 17) + nbits);
    if (c == 0 && num_blocks >= 2) {
      // It is X channel (red-green) and we often see ringing
      // in the large blocks. Let's punish that more here.
      float w = 1.0 + std::min(3.0, num_blocks / 8.0);
      entropy *= w;
      loss = Mul(loss, Set(df8, w));
    }
  }
  float loss_scalar =
      pow(GetLane(SumOfLanes(df8, loss)) / (num_blocks * kDCTBlockSize),
          1.0f / 8.0f) *
      (num_blocks * kDCTBlockSize) / quant_norm16;
  entropy *= entropy_mul;
  entropy += config.info_loss_multiplier * loss_scalar;
  return true;
}

Status FindBest8x8Transform(size_t x, size_t y, int encoding_speed_tier,
                            float butteraugli_target, const ACSConfig& config,
                            const float* JXL_RESTRICT cmap_factors,
                            AcStrategyImage* JXL_RESTRICT ac_strategy,
                            float* block, float* scratch_space,
                            uint32_t* quantized, float* entropy_out,
                            AcStrategyType& best_tx) {
  struct TransformTry8x8 {
    AcStrategyType type;
    int encoding_speed_tier_max_limit;
    double entropy_mul;
  };
  static const TransformTry8x8 kTransforms8x8[] = {
      {
          AcStrategyType::DCT,
          9,
          0.8,
      },
      {
          AcStrategyType::DCT4X4,
          5,
          1.08,
      },
      {
          AcStrategyType::DCT2X2,
          5,
          0.95,
      },
      {
          AcStrategyType::DCT4X8,
          4,
          0.85931637428340035,
      },
      {
          AcStrategyType::DCT8X4,
          4,
          0.85931637428340035,
      },
      {
          AcStrategyType::IDENTITY,
          5,
          1.0427542510634957,
      },
      {
          AcStrategyType::AFV0,
          4,
          0.81779489591359944,
      },
      {
          AcStrategyType::AFV1,
          4,
          0.81779489591359944,
      },
      {
          AcStrategyType::AFV2,
          4,
          0.81779489591359944,
      },
      {
          AcStrategyType::AFV3,
          4,
          0.81779489591359944,
      },
  };
  double best = 1e30;
  best_tx = kTransforms8x8[0].type;
  for (auto tx : kTransforms8x8) {
    if (tx.encoding_speed_tier_max_limit < encoding_speed_tier) {
      continue;
    }
    AcStrategy acs = AcStrategy::FromRawStrategy(tx.type);
    float entropy_mul = tx.entropy_mul / kTransforms8x8[0].entropy_mul;
    if ((tx.type == AcStrategyType::DCT2X2 ||
         tx.type == AcStrategyType::IDENTITY) &&
        butteraugli_target < 5.0) {
      static const float kFavor2X2AtHighQuality = 0.4;
      float weight = pow((5.0f - butteraugli_target) / 5.0f, 2.0f);
      entropy_mul -= kFavor2X2AtHighQuality * weight;
    }
    if ((tx.type != AcStrategyType::DCT && tx.type != AcStrategyType::DCT2X2 &&
         tx.type != AcStrategyType::IDENTITY) &&
        butteraugli_target > 4.0) {
      static const float kAvoidEntropyOfTransforms = 0.5;
      float mul = 1.0;
      if (butteraugli_target < 12.0) {
        mul *= (12.0 - 4.0) / (butteraugli_target - 4.0);
      }
      entropy_mul += kAvoidEntropyOfTransforms * mul;
    }
    float entropy;
    JXL_RETURN_IF_ERROR(EstimateEntropy(acs, entropy_mul, x, y, config,
                                        cmap_factors, block, scratch_space,
                                        quantized, entropy));
    if (entropy < best) {
      best_tx = tx.type;
      best = entropy;
    }
  }
  *entropy_out = best;
  return true;
}

// bx, by addresses the 64x64 block at 8x8 subresolution
// cx, cy addresses the left, upper 8x8 block position of the candidate
// transform.
Status TryMergeAcs(AcStrategyType acs_raw, size_t bx, size_t by, size_t cx,
                   size_t cy, const ACSConfig& config,
                   const float* JXL_RESTRICT cmap_factors,
                   AcStrategyImage* JXL_RESTRICT ac_strategy,
                   const float entropy_mul, const uint8_t candidate_priority,
                   uint8_t* priority, float* JXL_RESTRICT entropy_estimate,
                   float* block, float* scratch_space, uint32_t* quantized) {
  AcStrategy acs = AcStrategy::FromRawStrategy(acs_raw);
  float entropy_current = 0;
  for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
    for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
      if (priority[(cy + iy) * 8 + (cx + ix)] >= candidate_priority) {
        // Transform would reuse already allocated blocks and
        // lead to invalid overlaps, for example DCT64X32 vs.
        // DCT32X64.
        return true;
      }
      entropy_current += entropy_estimate[(cy + iy) * 8 + (cx + ix)];
    }
  }
  float entropy_candidate;
  JXL_RETURN_IF_ERROR(EstimateEntropy(
      acs, entropy_mul, (bx + cx) * 8, (by + cy) * 8, config, cmap_factors,
      block, scratch_space, quantized, entropy_candidate));
  if (entropy_candidate >= entropy_current) return true;
  // Accept the candidate.
  for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
    for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
      entropy_estimate[(cy + iy) * 8 + cx + ix] = 0;
      priority[(cy + iy) * 8 + cx + ix] = candidate_priority;
    }
  }
  JXL_RETURN_IF_ERROR(ac_strategy->Set(bx + cx, by + cy, acs_raw));
  entropy_estimate[cy * 8 + cx] = entropy_candidate;
  return true;
}

static void SetEntropyForTransform(size_t cx, size_t cy,
                                   const AcStrategyType acs_raw, float entropy,
                                   float* JXL_RESTRICT entropy_estimate) {
  const AcStrategy acs = AcStrategy::FromRawStrategy(acs_raw);
  for (size_t dy = 0; dy < acs.covered_blocks_y(); ++dy) {
    for (size_t dx = 0; dx < acs.covered_blocks_x(); ++dx) {
      entropy_estimate[(cy + dy) * 8 + cx + dx] = 0.0;
    }
  }
  entropy_estimate[cy * 8 + cx] = entropy;
}

AcStrategyType AcsSquare(size_t blocks) {
  if (blocks == 2) {
    return AcStrategyType::DCT16X16;
  } else if (blocks == 4) {
    return AcStrategyType::DCT32X32;
  } else {
    return AcStrategyType::DCT64X64;
  }
}

AcStrategyType AcsVerticalSplit(size_t blocks) {
  if (blocks == 2) {
    return AcStrategyType::DCT16X8;
  } else if (blocks == 4) {
    return AcStrategyType::DCT32X16;
  } else {
    return AcStrategyType::DCT64X32;
  }
}

AcStrategyType AcsHorizontalSplit(size_t blocks) {
  if (blocks == 2) {
    return AcStrategyType::DCT8X16;
  } else if (blocks == 4) {
    return AcStrategyType::DCT16X32;
  } else {
    return AcStrategyType::DCT32X64;
  }
}

// The following function tries to merge smaller transforms into
// squares and the rectangles originating from a single middle division
// (horizontal or vertical) fairly.
//
// This is now generalized to concern about squares
// of blocks X blocks size, where a block is 8x8 pixels.
Status FindBestFirstLevelDivisionForSquare(
    size_t blocks, bool allow_square_transform, size_t bx, size_t by, size_t cx,
    size_t cy, const ACSConfig& config, const float* JXL_RESTRICT cmap_factors,
    AcStrategyImage* JXL_RESTRICT ac_strategy, const float entropy_mul_JXK,
    const float entropy_mul_JXJ, float* JXL_RESTRICT entropy_estimate,
    float* block, float* scratch_space, uint32_t* quantized) {
  // We denote J for the larger dimension here, and K for the smaller.
  // For example, for 32x32 block splitting, J would be 32, K 16.
  const size_t blocks_half = blocks / 2;
  const AcStrategyType acs_rawJXK = AcsVerticalSplit(blocks);
  const AcStrategyType acs_rawKXJ = AcsHorizontalSplit(blocks);
  const AcStrategyType acs_rawJXJ = AcsSquare(blocks);
  const AcStrategy acsJXK = AcStrategy::FromRawStrategy(acs_rawJXK);
  const AcStrategy acsKXJ = AcStrategy::FromRawStrategy(acs_rawKXJ);
  const AcStrategy acsJXJ = AcStrategy::FromRawStrategy(acs_rawJXJ);
  AcStrategyRow row0 = ac_strategy->ConstRow(by + cy + 0);
  AcStrategyRow row1 = ac_strategy->ConstRow(by + cy + blocks_half);
  // Let's check if we can consider a JXJ block here at all.
  // This is not necessary in the basic use of hierarchically merging
  // blocks in the simplest possible way, but is needed when we try other
  // 'floating' options of merging, possibly after a simple hierarchical
  // merge has been explored.
  if (MultiBlockTransformCrossesHorizontalBoundary(*ac_strategy, bx + cx,
                                                   by + cy, bx + cx + blocks) ||
      MultiBlockTransformCrossesHorizontalBoundary(
          *ac_strategy, bx + cx, by + cy + blocks, bx + cx + blocks) ||
      MultiBlockTransformCrossesVerticalBoundary(*ac_strategy, bx + cx, by + cy,
                                                 by + cy + blocks) ||
      MultiBlockTransformCrossesVerticalBoundary(*ac_strategy, bx + cx + blocks,
                                                 by + cy, by + cy + blocks)) {
    return true;  // not suitable for JxJ analysis, some transforms leak out.
  }
  // For floating transforms there may be
  // already blocks selected that make either or both JXK and
  // KXJ not feasible for this location.
  const bool allow_JXK = !MultiBlockTransformCrossesVerticalBoundary(
      *ac_strategy, bx + cx + blocks_half, by + cy, by + cy + blocks);
  const bool allow_KXJ = !MultiBlockTransformCrossesHorizontalBoundary(
      *ac_strategy, bx + cx, by + cy + blocks_half, bx + cx + blocks);
  // Current entropies aggregated on NxN resolution.
  float entropy[2][2] = {};
  for (size_t dy = 0; dy < blocks; ++dy) {
    for (size_t dx = 0; dx < blocks; ++dx) {
      entropy[dy / blocks_half][dx / blocks_half] +=
          entropy_estimate[(cy + dy) * 8 + (cx + dx)];
    }
  }
  float entropy_JXK_left = std::numeric_limits<float>::max();
  float entropy_JXK_right = std::numeric_limits<float>::max();
  float entropy_KXJ_top = std::numeric_limits<float>::max();
  float entropy_KXJ_bottom = std::numeric_limits<float>::max();
  float entropy_JXJ = std::numeric_limits<float>::max();
  if (allow_JXK) {
    if (row0[bx + cx + 0].Strategy() != acs_rawJXK) {
      JXL_RETURN_IF_ERROR(EstimateEntropy(
          acsJXK, entropy_mul_JXK, (bx + cx + 0) * 8, (by + cy + 0) * 8, config,
          cmap_factors, block, scratch_space, quantized, entropy_JXK_left));
    }
    if (row0[bx + cx + blocks_half].Strategy() != acs_rawJXK) {
      JXL_RETURN_IF_ERROR(
          EstimateEntropy(acsJXK, entropy_mul_JXK, (bx + cx + blocks_half) * 8,
                          (by + cy + 0) * 8, config, cmap_factors, block,
                          scratch_space, quantized, entropy_JXK_right));
    }
  }
  if (allow_KXJ) {
    if (row0[bx + cx].Strategy() != acs_rawKXJ) {
      JXL_RETURN_IF_ERROR(EstimateEntropy(
          acsKXJ, entropy_mul_JXK, (bx + cx + 0) * 8, (by + cy + 0) * 8, config,
          cmap_factors, block, scratch_space, quantized, entropy_KXJ_top));
    }
    if (row1[bx + cx].Strategy() != acs_rawKXJ) {
      JXL_RETURN_IF_ERROR(
          EstimateEntropy(acsKXJ, entropy_mul_JXK, (bx + cx + 0) * 8,
                          (by + cy + blocks_half) * 8, config, cmap_factors,
                          block, scratch_space, quantized, entropy_KXJ_bottom));
    }
  }
  if (allow_square_transform) {
    // We control the exploration of the square transform separately so that
    // we can turn it off at high decoding speeds for 32x32, but still allow
    // exploring 16x32 and 32x16.
    JXL_RETURN_IF_ERROR(EstimateEntropy(
        acsJXJ, entropy_mul_JXJ, (bx + cx + 0) * 8, (by + cy + 0) * 8, config,
        cmap_factors, block, scratch_space, quantized, entropy_JXJ));
  }

  // Test if this block should have JXK or KXJ transforms,
  // because it can have only one or the other.
  float costJxN = std::min(entropy_JXK_left, entropy[0][0] + entropy[1][0]) +
                  std::min(entropy_JXK_right, entropy[0][1] + entropy[1][1]);
  float costNxJ = std::min(entropy_KXJ_top, entropy[0][0] + entropy[0][1]) +
                  std::min(entropy_KXJ_bottom, entropy[1][0] + entropy[1][1]);
  if (entropy_JXJ < costJxN && entropy_JXJ < costNxJ) {
    JXL_RETURN_IF_ERROR(ac_strategy->Set(bx + cx, by + cy, acs_rawJXJ));
    SetEntropyForTransform(cx, cy, acs_rawJXJ, entropy_JXJ, entropy_estimate);
  } else if (costJxN < costNxJ) {
    if (entropy_JXK_left < entropy[0][0] + entropy[1][0]) {
      JXL_RETURN_IF_ERROR(ac_strategy->Set(bx + cx, by + cy, acs_rawJXK));
      SetEntropyForTransform(cx, cy, acs_rawJXK, entropy_JXK_left,
                             entropy_estimate);
    }
    if (entropy_JXK_right < entropy[0][1] + entropy[1][1]) {
      JXL_RETURN_IF_ERROR(
          ac_strategy->Set(bx + cx + blocks_half, by + cy, acs_rawJXK));
      SetEntropyForTransform(cx + blocks_half, cy, acs_rawJXK,
                             entropy_JXK_right, entropy_estimate);
    }
  } else {
    if (entropy_KXJ_top < entropy[0][0] + entropy[0][1]) {
      JXL_RETURN_IF_ERROR(ac_strategy->Set(bx + cx, by + cy, acs_rawKXJ));
      SetEntropyForTransform(cx, cy, acs_rawKXJ, entropy_KXJ_top,
                             entropy_estimate);
    }
    if (entropy_KXJ_bottom < entropy[1][0] + entropy[1][1]) {
      JXL_RETURN_IF_ERROR(
          ac_strategy->Set(bx + cx, by + cy + blocks_half, acs_rawKXJ));
      SetEntropyForTransform(cx, cy + blocks_half, acs_rawKXJ,
                             entropy_KXJ_bottom, entropy_estimate);
    }
  }
  return true;
}

Status ProcessRectACS(const CompressParams& cparams, const ACSConfig& config,
                      const Rect& rect, const ColorCorrelationMap& cmap,
                      float* JXL_RESTRICT block,
                      uint32_t* JXL_RESTRICT quantized,
                      AcStrategyImage* ac_strategy) {
  // Main philosophy here:
  // 1. First find best 8x8 transform for each area.
  // 2. Merging them into larger transforms where possibly, but
  // starting from the smallest transforms (16x8 and 8x16).
  // Additional complication: 16x8 and 8x16 are considered
  // simultaneously and fairly against each other.
  // We are looking at 64x64 squares since the Y-to-X and Y-to-B
  // maps happen to be at that resolution, and having
  // integral transforms cross these boundaries leads to
  // additional complications.
  const float butteraugli_target = cparams.butteraugli_distance;
  float* JXL_RESTRICT scratch_space = block + 3 * AcStrategy::kMaxCoeffArea;
  size_t bx = rect.x0();
  size_t by = rect.y0();
  JXL_ENSURE(rect.xsize() <= 8);
  JXL_ENSURE(rect.ysize() <= 8);
  size_t tx = bx / kColorTileDimInBlocks;
  size_t ty = by / kColorTileDimInBlocks;
  const float cmap_factors[3] = {
      cmap.base().YtoXRatio(cmap.ytox_map.ConstRow(ty)[tx]),
      0.0f,
      cmap.base().YtoBRatio(cmap.ytob_map.ConstRow(ty)[tx]),
  };
  if (cparams.speed_tier > SpeedTier::kHare) return true;
  // First compute the best 8x8 transform for each square. Later, we do not
  // experiment with different combinations, but only use the best of the 8x8s
  // when DCT8X8 is specified in the tree search.
  // 8x8 transforms have 10 variants, but every larger transform is just a DCT.
  float entropy_estimate[64] = {};
  // Favor all 8x8 transforms (against 16x8 and larger transforms)) at
  // low butteraugli_target distances.
  static const float k8x8mul1 = -0.4;
  static const float k8x8mul2 = 1.0;
  static const float k8x8base = 1.4;
  const float mul8x8 = k8x8mul2 + k8x8mul1 / (butteraugli_target + k8x8base);
  for (size_t iy = 0; iy < rect.ysize(); iy++) {
    for (size_t ix = 0; ix < rect.xsize(); ix++) {
      float entropy = 0.0;
      AcStrategyType best_of_8x8s;
      JXL_RETURN_IF_ERROR(FindBest8x8Transform(
          8 * (bx + ix), 8 * (by + iy), static_cast<int>(cparams.speed_tier),
          butteraugli_target, config, cmap_factors, ac_strategy, block,
          scratch_space, quantized, &entropy, best_of_8x8s));
      JXL_RETURN_IF_ERROR(ac_strategy->Set(bx + ix, by + iy, best_of_8x8s));
      entropy_estimate[iy * 8 + ix] = entropy * mul8x8;
    }
  }
  // Merge when a larger transform is better than the previously
  // searched best combination of 8x8 transforms.
  struct MergeTry {
    AcStrategyType type;
    uint8_t priority;
    uint8_t decoding_speed_tier_max_limit;
    uint8_t encoding_speed_tier_max_limit;
    float entropy_mul;
  };
  // These numbers need to be figured out manually and looking at
  // ringing next to sky etc. Optimization will find smaller numbers
  // and produce more ringing than is ideal. Larger numbers will
  // help stop ringing.
  const float entropy_mul16X8 = 1.21;
  const float entropy_mul16X16 = 1.34;
  const float entropy_mul16X32 = 1.49;
  const float entropy_mul32X32 = 1.48;
  const float entropy_mul64X32 = 2.25;
  const float entropy_mul64X64 = 2.25;
  // TODO(jyrki): Consider this feedback in further changes:
  // Also effectively when the multipliers for smaller blocks are
  // below 1, this raises the bar for the bigger blocks even higher
  // in that sense these constants are not independent (e.g. changing
  // the constant for DCT16x32 by -5% (making it more likely) also
  // means that DCT32x32 becomes harder to do when starting from
  // two DCT16x32s). It might be better to make them more independent,
  // e.g. by not applying the multiplier when storing the new entropy
  // estimates in TryMergeToACSCandidate().
  const MergeTry kTransformsForMerge[9] = {
      {AcStrategyType::DCT16X8, 2, 4, 5, entropy_mul16X8},
      {AcStrategyType::DCT8X16, 2, 4, 5, entropy_mul16X8},
      // FindBestFirstLevelDivisionForSquare looks for DCT16X16 and its
      // subdivisions. {AcStrategyType::DCT16X16, 3, entropy_mul16X16},
      {AcStrategyType::DCT16X32, 4, 4, 4, entropy_mul16X32},
      {AcStrategyType::DCT32X16, 4, 4, 4, entropy_mul16X32},
      // FindBestFirstLevelDivisionForSquare looks for DCT32X32 and its
      // subdivisions. {AcStrategyType::DCT32X32, 5, 1, 5,
      // 0.9822994906548809f},
      {AcStrategyType::DCT64X32, 6, 1, 3, entropy_mul64X32},
      {AcStrategyType::DCT32X64, 6, 1, 3, entropy_mul64X32},
      // {AcStrategyType::DCT64X64, 8, 1, 3, 2.0846542128012948f},
  };
  /*
  These sizes not yet included in merge heuristic:
  set(AcStrategyType::DCT32X8, 0.0f, 2.261390410971102f);
  set(AcStrategyType::DCT8X32, 0.0f, 2.261390410971102f);
  set(AcStrategyType::DCT128X128, 0.0f, 1.0f);
  set(AcStrategyType::DCT128X64, 0.0f, 0.73f);
  set(AcStrategyType::DCT64X128, 0.0f, 0.73f);
  set(AcStrategyType::DCT256X256, 0.0f, 1.0f);
  set(AcStrategyType::DCT256X128, 0.0f, 0.73f);
  set(AcStrategyType::DCT128X256, 0.0f, 0.73f);
  */

  // Priority is a tricky kludge to avoid collisions so that transforms
  // don't overlap.
  uint8_t priority[64] = {};
  bool enable_32x32 = cparams.decoding_speed_tier < 4;
  for (auto mt : kTransformsForMerge) {
    if (mt.decoding_speed_tier_max_limit < cparams.decoding_speed_tier) {
      continue;
    }
    AcStrategy acs = AcStrategy::FromRawStrategy(mt.type);

    for (size_t cy = 0; cy + acs.covered_blocks_y() - 1 < rect.ysize();
         cy += acs.covered_blocks_y()) {
      for (size_t cx = 0; cx + acs.covered_blocks_x() - 1 < rect.xsize();
           cx += acs.covered_blocks_x()) {
        if (cy + 7 < rect.ysize() && cx + 7 < rect.xsize()) {
          if (cparams.decoding_speed_tier < 4 &&
              mt.type == AcStrategyType::DCT32X64) {
            // We handle both DCT8X16 and DCT16X8 at the same time.
            if ((cy | cx) % 8 == 0) {
              JXL_RETURN_IF_ERROR(FindBestFirstLevelDivisionForSquare(
                  8, true, bx, by, cx, cy, config, cmap_factors, ac_strategy,
                  mt.entropy_mul, entropy_mul64X64, entropy_estimate, block,
                  scratch_space, quantized));
            }
            continue;
          } else if (mt.type == AcStrategyType::DCT32X16) {
            // We handled both DCT8X16 and DCT16X8 at the same time,
            // and that is above. The last column and last row,
            // when the last column or last row is odd numbered,
            // are still handled by TryMergeAcs.
            continue;
          }
        }
        if ((mt.type == AcStrategyType::DCT16X32 && cy % 4 != 0) ||
            (mt.type == AcStrategyType::DCT32X16 && cx % 4 != 0)) {
          // already covered by FindBest32X32
          continue;
        }

        if (cy + 3 < rect.ysize() && cx + 3 < rect.xsize()) {
          if (mt.type == AcStrategyType::DCT16X32) {
            // We handle both DCT8X16 and DCT16X8 at the same time.
            if ((cy | cx) % 4 == 0) {
              JXL_RETURN_IF_ERROR(FindBestFirstLevelDivisionForSquare(
                  4, enable_32x32, bx, by, cx, cy, config, cmap_factors,
                  ac_strategy, mt.entropy_mul, entropy_mul32X32,
                  entropy_estimate, block, scratch_space, quantized));
            }
            continue;
          } else if (mt.type == AcStrategyType::DCT32X16) {
            // We handled both DCT8X16 and DCT16X8 at the same time,
            // and that is above. The last column and last row,
            // when the last column or last row is odd numbered,
            // are still handled by TryMergeAcs.
            continue;
          }
        }
        if ((mt.type == AcStrategyType::DCT16X32 && cy % 4 != 0) ||
            (mt.type == AcStrategyType::DCT32X16 && cx % 4 != 0)) {
          // already covered by FindBest32X32
          continue;
        }
        if (cy + 1 < rect.ysize() && cx + 1 < rect.xsize()) {
          if (mt.type == AcStrategyType::DCT8X16) {
            // We handle both DCT8X16 and DCT16X8 at the same time.
            if ((cy | cx) % 2 == 0) {
              JXL_RETURN_IF_ERROR(FindBestFirstLevelDivisionForSquare(
                  2, true, bx, by, cx, cy, config, cmap_factors, ac_strategy,
                  mt.entropy_mul, entropy_mul16X16, entropy_estimate, block,
                  scratch_space, quantized));
            }
            continue;
          } else if (mt.type == AcStrategyType::DCT16X8) {
            // We handled both DCT8X16 and DCT16X8 at the same time,
            // and that is above. The last column and last row,
            // when the last column or last row is odd numbered,
            // are still handled by TryMergeAcs.
            continue;
          }
        }
        if ((mt.type == AcStrategyType::DCT8X16 && cy % 2 == 1) ||
            (mt.type == AcStrategyType::DCT16X8 && cx % 2 == 1)) {
          // already covered by FindBestFirstLevelDivisionForSquare
          continue;
        }
        // All other merge sizes are handled here.
        // Some of the DCT16X8s and DCT8X16s will still leak through here
        // when there is an odd number of 8x8 blocks, then the last row
        // and column will get their DCT16X8s and DCT8X16s through the
        // normal integral transform merging process.
        JXL_RETURN_IF_ERROR(
            TryMergeAcs(mt.type, bx, by, cx, cy, config, cmap_factors,
                        ac_strategy, mt.entropy_mul, mt.priority, &priority[0],
                        entropy_estimate, block, scratch_space, quantized));
      }
    }
  }
  if (cparams.speed_tier >= SpeedTier::kHare) {
    return true;
  }
  // Here we still try to do some non-aligned matching, find a few more
  // 16X8, 8X16 and 16X16s between the non-2-aligned blocks.
  for (size_t cy = 0; cy + 1 < rect.ysize(); ++cy) {
    for (size_t cx = 0; cx + 1 < rect.xsize(); ++cx) {
      if ((cy | cx) % 2 != 0) {
        JXL_RETURN_IF_ERROR(FindBestFirstLevelDivisionForSquare(
            2, true, bx, by, cx, cy, config, cmap_factors, ac_strategy,
            entropy_mul16X8, entropy_mul16X16, entropy_estimate, block,
            scratch_space, quantized));
      }
    }
  }
  // Non-aligned matching for 32X32, 16X32 and 32X16.
  size_t step = cparams.speed_tier >= SpeedTier::kTortoise ? 2 : 1;
  for (size_t cy = 0; cy + 3 < rect.ysize(); cy += step) {
    for (size_t cx = 0; cx + 3 < rect.xsize(); cx += step) {
      if ((cy | cx) % 4 == 0) {
        continue;  // Already tried with loop above (DCT16X32 case).
      }
      JXL_RETURN_IF_ERROR(FindBestFirstLevelDivisionForSquare(
          4, enable_32x32, bx, by, cx, cy, config, cmap_factors, ac_strategy,
          entropy_mul16X32, entropy_mul32X32, entropy_estimate, block,
          scratch_space, quantized));
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
HWY_EXPORT(ProcessRectACS);

Status AcStrategyHeuristics::Init(const Image3F& src, const Rect& rect_in,
                                  const ImageF& quant_field, const ImageF& mask,
                                  const ImageF& mask1x1,
                                  DequantMatrices* matrices) {
  config.dequant = matrices;

  if (cparams.speed_tier >= SpeedTier::kCheetah) {
    JXL_RETURN_IF_ERROR(
        matrices->EnsureComputed(memory_manager, 1));  // DCT8 only
  } else {
    uint32_t acs_mask = 0;
    // All transforms up to 64x64.
    for (size_t i = 0; i < static_cast<size_t>(AcStrategyType::DCT128X128);
         i++) {
      acs_mask |= (1 << i);
    }
    JXL_RETURN_IF_ERROR(matrices->EnsureComputed(memory_manager, acs_mask));
  }

  // Image row pointers and strides.
  config.quant_field_row = quant_field.Row(0);
  config.quant_field_stride = quant_field.PixelsPerRow();
  if (mask.xsize() > 0 && mask.ysize() > 0) {
    config.masking_field_row = mask.Row(0);
    config.masking_field_stride = mask.PixelsPerRow();
  }
  config.mask1x1_xsize = mask1x1.xsize();
  if (mask1x1.xsize() > 0 && mask1x1.ysize() > 0) {
    config.masking1x1_field_row = mask1x1.Row(0);
    config.masking1x1_field_stride = mask1x1.PixelsPerRow();
  }

  config.src_rows[0] = rect_in.ConstPlaneRow(src, 0, 0);
  config.src_rows[1] = rect_in.ConstPlaneRow(src, 1, 0);
  config.src_rows[2] = rect_in.ConstPlaneRow(src, 2, 0);
  config.src_stride = src.PixelsPerRow();

  // Entropy estimate is composed of two factors:
  //  - estimate of the number of bits that will be used by the block
  //  - information loss due to quantization
  // The following constant controls the relative weights of these components.
  config.info_loss_multiplier = 1.2;
  config.zeros_mul = 9.3089059022677905;
  config.cost_delta = 10.833273317067883;

  static const float kBias = 0.13731742964354549;
  const float ratio = (cparams.butteraugli_distance + kBias) / (1.0f + kBias);

  static const float kPow1 = 0.33677806662454718;
  static const float kPow2 = 0.50990926717963703;
  static const float kPow3 = 0.36702940662370243;
  config.info_loss_multiplier *= std::pow(ratio, kPow1);
  config.zeros_mul *= std::pow(ratio, kPow2);
  config.cost_delta *= std::pow(ratio, kPow3);
  return true;
}

Status AcStrategyHeuristics::PrepareForThreads(std::size_t num_threads) {
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
  mem_per_thread = 6 * AcStrategy::kMaxCoeffArea + dct_scratch_size;
  size_t mem_bytes = num_threads * mem_per_thread * sizeof(float);
  JXL_ASSIGN_OR_RETURN(mem, AlignedMemory::Create(memory_manager, mem_bytes));
  qmem_per_thread = AcStrategy::kMaxCoeffArea;
  size_t qmem_bytes = num_threads * qmem_per_thread * sizeof(uint32_t);
  JXL_ASSIGN_OR_RETURN(qmem, AlignedMemory::Create(memory_manager, qmem_bytes));
  return true;
}

Status AcStrategyHeuristics::ProcessRect(const Rect& rect,
                                         const ColorCorrelationMap& cmap,
                                         AcStrategyImage* ac_strategy,
                                         size_t thread) {
  // In Cheetah mode, use DCT8 everywhere and uniform quantization.
  if (cparams.speed_tier >= SpeedTier::kCheetah) {
    ac_strategy->FillDCT8(rect);
    return true;
  }
  return HWY_DYNAMIC_DISPATCH(ProcessRectACS)(
      cparams, config, rect, cmap,
      mem.address<float>() + thread * mem_per_thread,
      qmem.address<uint32_t>() + thread * qmem_per_thread, ac_strategy);
}

Status AcStrategyHeuristics::Finalize(const FrameDimensions& frame_dim,
                                      const AcStrategyImage& ac_strategy,
                                      AuxOut* aux_out) {
  // Accounting and debug output.
  if (aux_out != nullptr) {
    aux_out->num_small_blocks =
        ac_strategy.CountBlocks(AcStrategyType::IDENTITY) +
        ac_strategy.CountBlocks(AcStrategyType::DCT2X2) +
        ac_strategy.CountBlocks(AcStrategyType::DCT4X4);
    aux_out->num_dct4x8_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT4X8) +
        ac_strategy.CountBlocks(AcStrategyType::DCT8X4);
    aux_out->num_afv_blocks = ac_strategy.CountBlocks(AcStrategyType::AFV0) +
                              ac_strategy.CountBlocks(AcStrategyType::AFV1) +
                              ac_strategy.CountBlocks(AcStrategyType::AFV2) +
                              ac_strategy.CountBlocks(AcStrategyType::AFV3);
    aux_out->num_dct8_blocks = ac_strategy.CountBlocks(AcStrategyType::DCT);
    aux_out->num_dct8x16_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT8X16) +
        ac_strategy.CountBlocks(AcStrategyType::DCT16X8);
    aux_out->num_dct8x32_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT8X32) +
        ac_strategy.CountBlocks(AcStrategyType::DCT32X8);
    aux_out->num_dct16_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT16X16);
    aux_out->num_dct16x32_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT16X32) +
        ac_strategy.CountBlocks(AcStrategyType::DCT32X16);
    aux_out->num_dct32_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT32X32);
    aux_out->num_dct32x64_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT32X64) +
        ac_strategy.CountBlocks(AcStrategyType::DCT64X32);
    aux_out->num_dct64_blocks =
        ac_strategy.CountBlocks(AcStrategyType::DCT64X64);
  }

  if (JXL_DEBUG_AC_STRATEGY && WantDebugOutput(cparams)) {
    JXL_RETURN_IF_ERROR(DumpAcStrategy(ac_strategy, frame_dim.xsize,
                                       frame_dim.ysize, "ac_strategy", aux_out,
                                       cparams));
  }
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
