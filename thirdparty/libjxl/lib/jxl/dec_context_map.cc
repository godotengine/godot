// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_context_map.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/inverse_mtf-inl.h"

namespace jxl {

namespace {

// Context map uses uint8_t.
constexpr size_t kMaxClusters = 256;

Status VerifyContextMap(const std::vector<uint8_t>& context_map,
                        const size_t num_htrees) {
  std::vector<bool> have_htree(num_htrees);
  size_t num_found = 0;
  for (const uint8_t htree : context_map) {
    if (htree >= num_htrees) {
      return JXL_FAILURE("Invalid histogram index in context map.");
    }
    if (!have_htree[htree]) {
      have_htree[htree] = true;
      ++num_found;
    }
  }
  if (num_found != num_htrees) {
    return JXL_FAILURE("Incomplete context map.");
  }
  return true;
}

}  // namespace

Status DecodeContextMap(JxlMemoryManager* memory_manager,
                        std::vector<uint8_t>* context_map, size_t* num_htrees,
                        BitReader* input) {
  bool is_simple = static_cast<bool>(input->ReadFixedBits<1>());
  if (is_simple) {
    int bits_per_entry = input->ReadFixedBits<2>();
    if (bits_per_entry != 0) {
      for (uint8_t& entry : *context_map) {
        entry = input->ReadBits(bits_per_entry);
      }
    } else {
      std::fill(context_map->begin(), context_map->end(), 0);
    }
  } else {
    bool use_mtf = static_cast<bool>(input->ReadFixedBits<1>());
    ANSCode code;
    std::vector<uint8_t> sink_ctx_map;
    // Usage of LZ77 is disallowed if decoding only two symbols. This doesn't
    // make sense in non-malicious bitstreams, and could cause a stack overflow
    // in malicious bitstreams by making every context map require its own
    // context map.
    JXL_RETURN_IF_ERROR(
        DecodeHistograms(memory_manager, input, 1, &code, &sink_ctx_map,
                         /*disallow_lz77=*/context_map->size() <= 2));
    JXL_ASSIGN_OR_RETURN(ANSSymbolReader reader,
                         ANSSymbolReader::Create(&code, input));
    size_t i = 0;
    uint32_t maxsym = 0;
    while (i < context_map->size()) {
      uint32_t sym = reader.ReadHybridUintInlined</*uses_lz77=*/true>(
          0, input, sink_ctx_map);
      maxsym = sym > maxsym ? sym : maxsym;
      (*context_map)[i] = sym;
      i++;
    }
    if (maxsym >= kMaxClusters) {
      return JXL_FAILURE("Invalid cluster ID");
    }
    if (!reader.CheckANSFinalState()) {
      return JXL_FAILURE("Invalid context map");
    }
    if (use_mtf) {
      InverseMoveToFrontTransform(context_map->data(), context_map->size());
    }
  }
  *num_htrees = *std::max_element(context_map->begin(), context_map->end()) + 1;
  return VerifyContextMap(*context_map, *num_htrees);
}

}  // namespace jxl
