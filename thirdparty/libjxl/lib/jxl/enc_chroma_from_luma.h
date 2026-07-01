// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_CHROMA_FROM_LUMA_H_
#define LIB_JXL_ENC_CHROMA_FROM_LUMA_H_

// Chroma-from-luma, computed using heuristics to determine the best linear
// model for the X and B channels from the Y channel.

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/simd_util.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;
class Quantizer;

Status ColorCorrelationEncodeDC(const ColorCorrelation& color_correlation,
                                BitWriter* writer, LayerType layer,
                                AuxOut* aux_out);

struct CfLHeuristics {
  explicit CfLHeuristics(JxlMemoryManager* memory_manager)
      : memory_manager(memory_manager) {}

  Status Init(const Rect& rect);

  Status PrepareForThreads(size_t num_threads) {
    size_t mem_bytes = num_threads * ItemsPerThread() * sizeof(float);
    JXL_ASSIGN_OR_RETURN(mem, AlignedMemory::Create(memory_manager, mem_bytes));
    return true;
  }

  Status ComputeTile(const Rect& r, const Image3F& opsin,
                     const Rect& opsin_rect, const DequantMatrices& dequant,
                     const AcStrategyImage* ac_strategy,
                     const ImageI* raw_quant_field, const Quantizer* quantizer,
                     bool fast, size_t thread, ColorCorrelationMap* cmap);

  JxlMemoryManager* memory_manager;
  ImageF dc_values;
  AlignedMemory mem;

  // Working set is too large for stack; allocate dynamically.
  static size_t ItemsPerThread() {
    const size_t dct_scratch_size =
        3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;
    return AcStrategy::kMaxCoeffArea * 3        // Blocks
           + kColorTileDim * kColorTileDim * 4  // AC coeff storage
           + AcStrategy::kMaxCoeffArea * 2      // Scratch space
           + dct_scratch_size;
  }
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_CHROMA_FROM_LUMA_H_
