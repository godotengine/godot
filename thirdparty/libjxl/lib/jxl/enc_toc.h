// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_TOC_H_
#define LIB_JXL_ENC_TOC_H_

#include <memory>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_bit_writer.h"

namespace jxl {

struct AuxOut;

// Writes the group offsets. If the permutation vector is empty, the identity
// permutation will be used.
Status WriteGroupOffsets(
    const std::vector<std::unique_ptr<BitWriter>>& group_codes,
    const std::vector<coeff_order_t>& permutation,
    BitWriter* JXL_RESTRICT writer, AuxOut* aux_out);

}  // namespace jxl

#endif  // LIB_JXL_ENC_TOC_H_
