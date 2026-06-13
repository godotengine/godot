// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COEFF_ORDER_H_
#define LIB_JXL_COEFF_ORDER_H_

#include <jxl/memory_manager.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/frame_dimensions.h"

namespace jxl {

class BitReader;

// Those offsets get multiplied by kDCTBlockSize.

static constexpr size_t kCoeffOrderLimit = 6156;

static constexpr std::array<size_t, 3 * kNumOrders + 1> JXL_MAYBE_UNUSED
    kCoeffOrderOffset = {
        0,    1,    2,    3,    4,    5,    6,    10,   14,   18,
        34,   50,   66,   68,   70,   72,   76,   80,   84,   92,
        100,  108,  172,  236,  300,  332,  364,  396,  652,  908,
        1164, 1292, 1420, 1548, 2572, 3596, 4620, 5132, 5644, kCoeffOrderLimit};

// TODO(eustas): rollback to constexpr once modern C++ becomes required.
#define CoeffOrderOffset(O, C) \
  (kCoeffOrderOffset[3 * (O) + (C)] * kDCTBlockSize)

static JXL_MAYBE_UNUSED constexpr size_t kCoeffOrderMaxSize =
    kCoeffOrderLimit * kDCTBlockSize;

// Mapping from AC strategy to order bucket. Strategies with different natural
// orders must have different buckets.
constexpr std::array<uint8_t, 27> kStrategyOrder = {
    0, 1, 1, 1, 2, 3, 4, 4, 5,  5,  6,  6,  1,  1,
    1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
};

constexpr JXL_MAYBE_UNUSED uint32_t kPermutationContexts = 8;

uint32_t CoeffOrderContext(uint32_t val);

Status DecodeCoeffOrders(JxlMemoryManager* memory_manager, uint16_t used_orders,
                         uint32_t used_acs, coeff_order_t* order,
                         BitReader* br);

Status DecodePermutation(JxlMemoryManager* memory_manager, size_t skip,
                         size_t size, coeff_order_t* order, BitReader* br);

}  // namespace jxl

#endif  // LIB_JXL_COEFF_ORDER_H_
