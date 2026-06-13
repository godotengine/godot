// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_CONTEXT_MAP_H_
#define LIB_JXL_DEC_CONTEXT_MAP_H_

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"

namespace jxl {

// Reads the context map from the bit stream. On calling this function,
// context_map->size() must be the number of possible context ids.
// Sets *num_htrees to the number of different histogram ids in
// *context_map.
Status DecodeContextMap(JxlMemoryManager* memory_manager,
                        std::vector<uint8_t>* context_map, size_t* num_htrees,
                        BitReader* input);

}  // namespace jxl

#endif  // LIB_JXL_DEC_CONTEXT_MAP_H_
