// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COEFF_ORDER_FWD_H_
#define LIB_JXL_COEFF_ORDER_FWD_H_

// Breaks circular dependency between ac_strategy and coeff_order.

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

// Needs at least 16 bits. A 32-bit type speeds up DecodeAC by 2% at the cost of
// more memory.
using coeff_order_t = uint32_t;

// Maximum number of orders to be used. Note that this needs to be multiplied by
// the number of channels. One per "size class" (plus one extra for DCT8),
// shared between transforms of size XxY and of size YxX.
constexpr uint8_t kNumOrders = 13;

// DCT coefficients are laid out in such a way that the number of rows of
// coefficients is always the smaller coordinate.
JXL_INLINE constexpr size_t CoefficientRows(size_t rows, size_t columns) {
  return rows < columns ? rows : columns;
}

JXL_INLINE constexpr size_t CoefficientColumns(size_t rows, size_t columns) {
  return rows < columns ? columns : rows;
}

JXL_INLINE void CoefficientLayout(size_t* JXL_RESTRICT rows,
                                  size_t* JXL_RESTRICT columns) {
  size_t r = *rows;
  size_t c = *columns;
  *rows = CoefficientRows(r, c);
  *columns = CoefficientColumns(r, c);
}

}  // namespace jxl

#endif  // LIB_JXL_COEFF_ORDER_FWD_H_
