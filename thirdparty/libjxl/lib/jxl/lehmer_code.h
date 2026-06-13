// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_LEHMER_CODE_H_
#define LIB_JXL_LEHMER_CODE_H_

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

// Permutation <=> factorial base representation (Lehmer code).

using LehmerT = uint32_t;

template <typename T>
constexpr T ValueOfLowest1Bit(T t) {
  return t & -t;
}

// Computes the Lehmer (factorial basis) code of permutation, an array of n
// unique indices in [0..n), and stores it in code[0..len). N*logN time.
// temp must have n + 1 elements but need not be initialized.
template <typename PermutationT>
Status ComputeLehmerCode(const PermutationT* JXL_RESTRICT permutation,
                         uint32_t* JXL_RESTRICT temp, const size_t n,
                         LehmerT* JXL_RESTRICT code) {
  for (size_t idx = 0; idx < n + 1; ++idx) temp[idx] = 0;

  for (size_t idx = 0; idx < n; ++idx) {
    const PermutationT s = permutation[idx];

    // Compute sum in Fenwick tree
    uint32_t penalty = 0;
    uint32_t i = s + 1;
    while (i != 0) {
      penalty += temp[i];
      i &= i - 1;  // clear lowest bit
    }
    JXL_ENSURE(s >= penalty);
    code[idx] = s - penalty;
    i = s + 1;
    // Add operation in Fenwick tree
    while (i < n + 1) {
      temp[i] += 1;
      i += ValueOfLowest1Bit(i);
    }
  }
  return true;
}

// Decodes the Lehmer code in code[0..n) into permutation[0..n).
// temp must have 1 << CeilLog2(n) elements but need not be initialized.
template <typename PermutationT>
Status DecodeLehmerCode(const LehmerT* JXL_RESTRICT code,
                        uint32_t* JXL_RESTRICT temp, size_t n,
                        PermutationT* JXL_RESTRICT permutation) {
  JXL_ENSURE(n != 0);
  const size_t log2n = CeilLog2Nonzero(n);
  const size_t padded_n = 1ull << log2n;

  for (size_t i = 0; i < padded_n; i++) {
    const int32_t i1 = static_cast<int32_t>(i + 1);
    temp[i] = static_cast<uint32_t>(ValueOfLowest1Bit(i1));
  }

  for (size_t i = 0; i < n; i++) {
    JXL_ENSURE(code[i] + i < n);
    uint32_t rank = code[i] + 1;

    // Extract i-th unused element via implicit order-statistics tree.
    size_t bit = padded_n;
    size_t next = 0;
    for (size_t i = 0; i <= log2n; i++) {
      const size_t cand = next + bit;
      JXL_ENSURE(cand >= 1);
      bit >>= 1;
      if (temp[cand - 1] < rank) {
        next = cand;
        rank -= temp[cand - 1];
      }
    }

    permutation[i] = next;

    // Mark as used
    next += 1;
    while (next <= padded_n) {
      temp[next - 1] -= 1;
      next += ValueOfLowest1Bit(next);
    }
  }
  return true;
}

}  // namespace jxl

#endif  // LIB_JXL_LEHMER_CODE_H_
