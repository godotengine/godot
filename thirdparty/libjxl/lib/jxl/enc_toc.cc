// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_toc.h"

#include <cstddef>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_coeff_order.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/toc.h"

namespace jxl {

Status WriteTocPermutation(const std::vector<coeff_order_t>& permutation,
                           BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  return writer->WithMaxBits(
      MaxBits(0), LayerType::Toc, aux_out, [&]() -> Status {
        if (!permutation.empty()) {
          writer->Write(1, 1);  // permutation present
          JXL_RETURN_IF_ERROR(EncodePermutation(
              permutation.data(), /*skip=*/0, permutation.size(), writer,
              LayerType::Header, aux_out));
        } else {
          writer->Write(1, 0);  // no permutation
        }
        writer->ZeroPadToByte();  // before TOC entries
        return true;
      });
}

Status WriteTocSizes(const std::vector<size_t>& group_sizes,
                     BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  return writer->WithMaxBits(
      MaxBits(group_sizes.size()), LayerType::Toc, aux_out, [&]() -> Status {
        for (size_t group_size : group_sizes) {
          JXL_RETURN_IF_ERROR(U32Coder::Write(kTocDist, group_size, writer));
        }
        writer->ZeroPadToByte();  // before first group
        return true;
      });
}

}  // namespace jxl
