// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/toc.h"

#include <jxl/memory_manager.h>
#include <stdint.h>

#include "lib/jxl/base/common.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/fields.h"

namespace jxl {
size_t MaxBits(const size_t num_sizes) {
  const size_t entry_bits = U32Coder::MaxEncodedBits(kTocDist) * num_sizes;
  // permutation bit (not its tokens!), padding, entries, padding.
  return 1 + kBitsPerByte + entry_bits + kBitsPerByte;
}

Status ReadToc(JxlMemoryManager* memory_manager, size_t toc_entries,
               BitReader* JXL_RESTRICT reader,
               std::vector<uint32_t>* JXL_RESTRICT sizes,
               std::vector<coeff_order_t>* JXL_RESTRICT permutation) {
  if (toc_entries > 65536) {
    // Prevent out of memory if invalid JXL codestream causes a bogus amount
    // of toc_entries such as 2720436919446 to be computed.
    // TODO(lode): verify whether 65536 is a reasonable upper bound
    return JXL_FAILURE("too many toc entries");
  }

  sizes->clear();
  sizes->resize(toc_entries);
  if (reader->TotalBitsConsumed() >= reader->TotalBytes() * kBitsPerByte) {
    return JXL_STATUS(StatusCode::kNotEnoughBytes, "Not enough bytes for TOC");
  }
  const auto check_bit_budget = [&](size_t num_entries) -> Status {
    // U32Coder reads 2 bits to recognize variant and kTocDist cheapest variant
    // is Bits(10), this way at least 12 bits are required per toc-entry.
    size_t minimal_bit_cost = num_entries * (2 + 10);
    size_t bit_budget = reader->TotalBytes() * 8;
    size_t expenses = reader->TotalBitsConsumed();
    if ((expenses <= bit_budget) &&
        (minimal_bit_cost <= bit_budget - expenses)) {
      return true;
    }
    return JXL_STATUS(StatusCode::kNotEnoughBytes, "Not enough bytes for TOC");
  };

  JXL_ENSURE(toc_entries > 0);
  if (reader->ReadFixedBits<1>() == 1) {
    JXL_RETURN_IF_ERROR(check_bit_budget(toc_entries));
    permutation->resize(toc_entries);
    JXL_RETURN_IF_ERROR(DecodePermutation(
        memory_manager, /*skip=*/0, toc_entries, permutation->data(), reader));
  }
  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  JXL_RETURN_IF_ERROR(check_bit_budget(toc_entries));
  for (size_t i = 0; i < toc_entries; ++i) {
    (*sizes)[i] = U32Coder::Read(kTocDist, reader);
  }
  JXL_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  JXL_RETURN_IF_ERROR(check_bit_budget(0));
  return true;
}

Status ReadGroupOffsets(JxlMemoryManager* memory_manager, size_t toc_entries,
                        BitReader* JXL_RESTRICT reader,
                        std::vector<uint64_t>* JXL_RESTRICT offsets,
                        std::vector<uint32_t>* JXL_RESTRICT sizes,
                        uint64_t* total_size) {
  std::vector<coeff_order_t> permutation;
  JXL_RETURN_IF_ERROR(
      ReadToc(memory_manager, toc_entries, reader, sizes, &permutation));

  offsets->clear();
  offsets->resize(toc_entries);

  // Prefix sum starting with 0 and ending with the offset of the last group
  uint64_t offset = 0;
  for (size_t i = 0; i < toc_entries; ++i) {
    if (offset + (*sizes)[i] < offset) {
      return JXL_FAILURE("group offset overflow");
    }
    (*offsets)[i] = offset;
    offset += (*sizes)[i];
  }
  if (total_size) {
    *total_size = offset;
  }

  if (!permutation.empty()) {
    std::vector<uint64_t> permuted_offsets;
    std::vector<uint32_t> permuted_sizes;
    permuted_offsets.reserve(toc_entries);
    permuted_sizes.reserve(toc_entries);
    for (coeff_order_t index : permutation) {
      permuted_offsets.push_back((*offsets)[index]);
      permuted_sizes.push_back((*sizes)[index]);
    }
    std::swap(*offsets, permuted_offsets);
    std::swap(*sizes, permuted_sizes);
  }

  return true;
}
}  // namespace jxl
