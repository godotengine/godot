// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_TOC_H_
#define LIB_JXL_TOC_H_

#include <jxl/memory_manager.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"

namespace jxl {

// (2+bits) = 2,3,4 bytes so encoders can patch TOC after encoding.
// 30 is sufficient for 4K channels of uncompressed 16-bit samples.
constexpr U32Enc kTocDist(Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408),
                          BitsOffset(30, 4211712));

size_t MaxBits(size_t num_sizes);

// TODO(veluca): move these to FrameDimensions.
static JXL_INLINE size_t AcGroupIndex(size_t pass, size_t group,
                                      size_t num_groups, size_t num_dc_groups) {
  return 2 + num_dc_groups + pass * num_groups + group;
}

static JXL_INLINE size_t NumTocEntries(size_t num_groups, size_t num_dc_groups,
                                       size_t num_passes) {
  if (num_groups == 1 && num_passes == 1) return 1;
  return AcGroupIndex(0, 0, num_groups, num_dc_groups) +
         num_groups * num_passes;
}

Status ReadToc(JxlMemoryManager* memory_manager, size_t toc_entries,
               BitReader* JXL_RESTRICT reader,
               std::vector<uint32_t>* JXL_RESTRICT sizes,
               std::vector<coeff_order_t>* JXL_RESTRICT permutation);

Status ReadGroupOffsets(JxlMemoryManager* memory_manager, size_t toc_entries,
                        BitReader* JXL_RESTRICT reader,
                        std::vector<uint64_t>* JXL_RESTRICT offsets,
                        std::vector<uint32_t>* JXL_RESTRICT sizes,
                        uint64_t* total_size);

}  // namespace jxl

#endif  // LIB_JXL_TOC_H_
