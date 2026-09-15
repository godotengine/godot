// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/coeff_order.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/lehmer_code.h"
#include "lib/jxl/modular/encoding/encoding.h"

namespace jxl {

static_assert(AcStrategy::kNumValidStrategies == kStrategyOrder.size(),
              "Update this array when adding or removing AC strategies.");

uint32_t CoeffOrderContext(uint32_t val) {
  uint32_t token, nbits, bits;
  HybridUintConfig(0, 0, 0).Encode(val, &token, &nbits, &bits);
  return std::min(token, kPermutationContexts - 1);
}

namespace {
Status ReadPermutation(size_t skip, size_t size, coeff_order_t* order,
                       BitReader* br, ANSSymbolReader* reader,
                       const std::vector<uint8_t>& context_map) {
  std::vector<LehmerT> lehmer(size);
  // temp space needs to be as large as the next power of 2, so doubling the
  // allocated size is enough.
  std::vector<uint32_t> temp(size * 2);
  uint32_t end =
      reader->ReadHybridUint(CoeffOrderContext(size), br, context_map) + skip;
  if (end > size) {
    return JXL_FAILURE("Invalid permutation size");
  }
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    lehmer[i] =
        reader->ReadHybridUint(CoeffOrderContext(last), br, context_map);
    last = lehmer[i];
    if (lehmer[i] >= size - i) {
      return JXL_FAILURE("Invalid lehmer code");
    }
  }
  if (order == nullptr) return true;
  JXL_RETURN_IF_ERROR(
      DecodeLehmerCode(lehmer.data(), temp.data(), size, order));
  return true;
}

}  // namespace

Status DecodePermutation(JxlMemoryManager* memory_manager, size_t skip,
                         size_t size, coeff_order_t* order, BitReader* br) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(DecodeHistograms(memory_manager, br, kPermutationContexts,
                                       &code, &context_map));
  JXL_ASSIGN_OR_RETURN(ANSSymbolReader reader,
                       ANSSymbolReader::Create(&code, br));
  JXL_RETURN_IF_ERROR(
      ReadPermutation(skip, size, order, br, &reader, context_map));
  if (!reader.CheckANSFinalState()) {
    return JXL_FAILURE("Invalid ANS stream");
  }
  return true;
}

namespace {

Status DecodeCoeffOrder(AcStrategy acs, coeff_order_t* order, BitReader* br,
                        ANSSymbolReader* reader,
                        std::vector<coeff_order_t>& natural_order,
                        const std::vector<uint8_t>& context_map) {
  const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = kDCTBlockSize * llf;

  JXL_RETURN_IF_ERROR(
      ReadPermutation(llf, size, order, br, reader, context_map));
  if (order == nullptr) return true;
  for (size_t k = 0; k < size; ++k) {
    order[k] = natural_order[order[k]];
  }
  return true;
}

}  // namespace

Status DecodeCoeffOrders(JxlMemoryManager* memory_manager, uint16_t used_orders,
                         uint32_t used_acs, coeff_order_t* order,
                         BitReader* br) {
  uint16_t computed = 0;
  std::vector<uint8_t> context_map;
  ANSCode code;
  ANSSymbolReader reader;
  std::vector<coeff_order_t> natural_order;
  // Bitstream does not have histograms if no coefficient order is used.
  if (used_orders != 0) {
    JXL_RETURN_IF_ERROR(DecodeHistograms(
        memory_manager, br, kPermutationContexts, &code, &context_map));
    JXL_ASSIGN_OR_RETURN(reader, ANSSymbolReader::Create(&code, br));
  }
  uint32_t acs_mask = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    if ((used_acs & (1 << o)) == 0) continue;
    acs_mask |= 1 << kStrategyOrder[o];
  }
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    bool used = (acs_mask & (1 << ord)) != 0;

    const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
    const size_t size = kDCTBlockSize * llf;

    if (used || (used_orders & (1 << ord))) {
      if (natural_order.size() < size) natural_order.resize(size);
      acs.ComputeNaturalCoeffOrder(natural_order.data());
    }

    if ((used_orders & (1 << ord)) == 0) {
      // No need to set the default order if no ACS uses this order.
      if (used) {
        for (size_t c = 0; c < 3; c++) {
          memcpy(&order[CoeffOrderOffset(ord, c)], natural_order.data(),
                 size * sizeof(*order));
        }
      }
    } else {
      for (size_t c = 0; c < 3; c++) {
        coeff_order_t* dest = used ? &order[CoeffOrderOffset(ord, c)] : nullptr;
        JXL_RETURN_IF_ERROR(DecodeCoeffOrder(acs, dest, br, &reader,
                                             natural_order, context_map));
      }
    }
  }
  if (used_orders && !reader.CheckANSFinalState()) {
    return JXL_FAILURE("Invalid ANS stream");
  }
  return true;
}

}  // namespace jxl
