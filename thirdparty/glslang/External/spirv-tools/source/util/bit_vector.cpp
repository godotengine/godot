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

#include "source/util/bit_vector.h"

#include <cassert>
#include <iostream>

namespace spvtools {
namespace utils {

void BitVector::ReportDensity(std::ostream& out) {
  uint32_t count = 0;

  for (BitContainer e : bits_) {
    while (e != 0) {
      if ((e & 1) != 0) {
        ++count;
      }
      e = e >> 1;
    }
  }

  out << "count=" << count
      << ", total size (bytes)=" << bits_.size() * sizeof(BitContainer)
      << ", bytes per element="
      << (double)(bits_.size() * sizeof(BitContainer)) / (double)(count);
}

bool BitVector::Or(const BitVector& other) {
  auto this_it = this->bits_.begin();
  auto other_it = other.bits_.begin();
  bool modified = false;

  while (this_it != this->bits_.end() && other_it != other.bits_.end()) {
    auto temp = *this_it | *other_it;
    if (temp != *this_it) {
      modified = true;
      *this_it = temp;
    }
    ++this_it;
    ++other_it;
  }

  if (other_it != other.bits_.end()) {
    modified = true;
    this->bits_.insert(this->bits_.end(), other_it, other.bits_.end());
  }

  return modified;
}

std::ostream& operator<<(std::ostream& out, const BitVector& bv) {
  out << "{";
  for (uint32_t i = 0; i < bv.bits_.size(); ++i) {
    BitVector::BitContainer b = bv.bits_[i];
    uint32_t j = 0;
    while (b != 0) {
      if (b & 1) {
        out << ' ' << i * BitVector::kBitContainerSize + j;
      }
      ++j;
      b = b >> 1;
    }
  }
  out << "}";
  return out;
}

}  // namespace utils
}  // namespace spvtools
