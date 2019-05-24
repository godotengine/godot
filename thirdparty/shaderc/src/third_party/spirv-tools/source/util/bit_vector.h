// Copyright (c) 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_UTIL_BIT_VECTOR_H_
#define SOURCE_UTIL_BIT_VECTOR_H_

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace spvtools {
namespace utils {

// Implements a bit vector class.
//
// All bits default to zero, and the upper bound is 2^32-1.
class BitVector {
 private:
  using BitContainer = uint64_t;
  enum { kBitContainerSize = 64 };
  enum { kInitialNumBits = 1024 };

 public:
  // Creates a bit vector contianing 0s.
  BitVector(uint32_t reserved_size = kInitialNumBits)
      : bits_((reserved_size - 1) / kBitContainerSize + 1, 0) {}

  // Sets the |i|th bit to 1.  Returns the |i|th bit before it was set.
  bool Set(uint32_t i) {
    uint32_t element_index = i / kBitContainerSize;
    uint32_t bit_in_element = i % kBitContainerSize;

    if (element_index >= bits_.size()) {
      bits_.resize(element_index + 1, 0);
    }

    BitContainer original = bits_[element_index];
    BitContainer ith_bit = static_cast<BitContainer>(1) << bit_in_element;

    if ((original & ith_bit) != 0) {
      return true;
    } else {
      bits_[element_index] = original | ith_bit;
      return false;
    }
  }

  // Sets the |i|th bit to 0.  Return the |i|th bit before it was cleared.
  bool Clear(uint32_t i) {
    uint32_t element_index = i / kBitContainerSize;
    uint32_t bit_in_element = i % kBitContainerSize;

    if (element_index >= bits_.size()) {
      return false;
    }

    BitContainer original = bits_[element_index];
    BitContainer ith_bit = static_cast<BitContainer>(1) << bit_in_element;

    if ((original & ith_bit) == 0) {
      return false;
    } else {
      bits_[element_index] = original & (~ith_bit);
      return true;
    }
  }

  // Returns the |i|th bit.
  bool Get(uint32_t i) const {
    uint32_t element_index = i / kBitContainerSize;
    uint32_t bit_in_element = i % kBitContainerSize;

    if (element_index >= bits_.size()) {
      return false;
    }

    return (bits_[element_index] &
            (static_cast<BitContainer>(1) << bit_in_element)) != 0;
  }

  // Returns true if every bit is 0.
  bool Empty() const {
    for (BitContainer b : bits_) {
      if (b != 0) {
        return false;
      }
    }
    return true;
  }

  // Print a report on the densicy of the bit vector, number of 1 bits, number
  // of bytes, and average bytes for 1 bit, to |out|.
  void ReportDensity(std::ostream& out);

  friend std::ostream& operator<<(std::ostream&, const BitVector&);

  // Performs a bitwise-or operation on |this| and |that|, storing the result in
  // |this|.  Return true if |this| changed.
  bool Or(const BitVector& that);

 private:
  std::vector<BitContainer> bits_;
};

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_BIT_VECTOR_H_
