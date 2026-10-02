// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_CACHE_H_
#define LIB_JXL_ENC_CACHE_H_

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_progressive_split.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"

namespace jxl {

struct AuxOut;

// Contains encoder state.
struct PassesEncoderState {
  explicit PassesEncoderState(JxlMemoryManager* memory_manager)
      : shared(memory_manager) {}

  PassesSharedState shared;

  bool streaming_mode = false;
  bool initialize_global_state = true;
  size_t dc_group_index = 0;

  // Per-pass DCT coefficients for the image. One row per group.
  std::vector<std::unique_ptr<ACImage>> coeffs;

  // Raw data for special (reference+DC) frames.
  std::vector<std::unique_ptr<BitWriter>> special_frames;

  // For splitting into passes.
  ProgressiveSplitter progressive_splitter;

  CompressParams cparams;

  struct PassData {
    std::vector<std::vector<Token>> ac_tokens;
    std::vector<uint8_t> context_map;
    EntropyEncodingData codes;
  };

  std::vector<PassData> passes;
  std::vector<size_t> histogram_idx;

  // Block sizes seen so far.
  uint32_t used_acs = 0;
  // Coefficient orders that are non-default.
  std::vector<uint32_t> used_orders;

  // Multiplier to be applied to the quant matrices of the x channel.
  float x_qm_multiplier = 1.0f;
  float b_qm_multiplier = 1.0f;

  ImageF initial_quant_masking1x1;

  JxlMemoryManager* memory_manager() const { return shared.memory_manager; }
};

// Initialize per-frame information.
class ModularFrameEncoder;
Status InitializePassesEncoder(const FrameHeader& frame_header,
                               const Image3F& opsin, const Rect& rect,
                               const JxlCmsInterface& cms, ThreadPool* pool,
                               PassesEncoderState* passes_enc_state,
                               ModularFrameEncoder* modular_frame_encoder,
                               AuxOut* aux_out);

Status ComputeACMetadata(ThreadPool* pool, PassesEncoderState* enc_state,
                         ModularFrameEncoder* modular_frame_encoder);

}  // namespace jxl

#endif  // LIB_JXL_ENC_CACHE_H_
