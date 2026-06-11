// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_CORE_MATH_UTILS_H_
#define DRACO_CORE_MATH_UTILS_H_

#include <inttypes.h>

#include "draco/core/vector_d.h"

namespace draco {

#define DRACO_INCREMENT_MOD(I, M) (((I) == ((M)-1)) ? 0 : ((I) + 1))

// Returns floor(sqrt(x)) where x is an integer number. The main intend of this
// function is to provide a cross platform and deterministic implementation of
// square root for integer numbers. This function is not intended to be a
// replacement for std::sqrt() for general cases. IntSqrt is in fact about 3X
// slower compared to most implementation of std::sqrt().
inline uint64_t IntSqrt(uint64_t number) {
  if (number == 0) {
    return 0;
  }
  // First estimate good initial value of the square root as log2(number).
  uint64_t act_number = number;
  uint64_t square_root = 1;
  while (act_number >= 2) {
    // Double the square root until |square_root * square_root > number|.
    square_root *= 2;
    act_number /= 4;
  }
  // Perform Newton's (or Babylonian) method to find the true floor(sqrt()).
  do {
    // New |square_root| estimate is computed as the average between
    // |square_root| and |number / square_root|.
    square_root = (square_root + number / square_root) / 2;

    // Note that after the first iteration, the estimate is always going to be
    // larger or equal to the true square root value. Therefore to check
    // convergence, we can simply detect condition when the square of the
    // estimated square root is larger than the input.
  } while (square_root * square_root > number);
  return square_root;
}

// Performs the addition in unsigned type to avoid signed integer overflow. Note
// that the result will be the same (for non-overflowing values).
template <
    typename DataTypeT,
    typename std::enable_if<std::is_integral<DataTypeT>::value &&
                            std::is_signed<DataTypeT>::value>::type * = nullptr>
inline DataTypeT AddAsUnsigned(DataTypeT a, DataTypeT b) {
  typedef typename std::make_unsigned<DataTypeT>::type DataTypeUT;
  return static_cast<DataTypeT>(static_cast<DataTypeUT>(a) +
                                static_cast<DataTypeUT>(b));
}

template <typename DataTypeT,
          typename std::enable_if<!std::is_integral<DataTypeT>::value ||
                                  !std::is_signed<DataTypeT>::value>::type * =
              nullptr>
inline DataTypeT AddAsUnsigned(DataTypeT a, DataTypeT b) {
  return a + b;
}

}  // namespace draco

#endif  // DRACO_CORE_MATH_UTILS_H_
