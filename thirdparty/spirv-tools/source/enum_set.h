// Copyright (c) 2016 Google Inc.
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

#ifndef SOURCE_ENUM_SET_H_
#define SOURCE_ENUM_SET_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <utility>

#include "source/latest_version_spirv_header.h"
#include "source/util/make_unique.h"

namespace spvtools {

// A set of values of a 32-bit enum type.
// It is fast and compact for the common case, where enum values
// are at most 63.  But it can represent enums with larger values,
// as may appear in extensions.
template <typename EnumType>
class EnumSet {
 private:
  // The ForEach method will call the functor on enum values in
  // enum value order (lowest to highest).  To make that easier, use
  // an ordered set for the overflow values.
  using OverflowSetType = std::set<uint32_t>;

 public:
  // Construct an empty set.
  EnumSet() {}
  // Construct an set with just the given enum value.
  explicit EnumSet(EnumType c) { Add(c); }
  // Construct an set from an initializer list of enum values.
  EnumSet(std::initializer_list<EnumType> cs) {
    for (auto c : cs) Add(c);
  }
  EnumSet(uint32_t count, const EnumType* ptr) {
    for (uint32_t i = 0; i < count; ++i) Add(ptr[i]);
  }
  // Copy constructor.
  EnumSet(const EnumSet& other) { *this = other; }
  // Move constructor.  The moved-from set is emptied.
  EnumSet(EnumSet&& other) {
    mask_ = other.mask_;
    overflow_ = std::move(other.overflow_);
    other.mask_ = 0;
    other.overflow_.reset(nullptr);
  }
  // Assignment operator.
  EnumSet& operator=(const EnumSet& other) {
    if (&other != this) {
      mask_ = other.mask_;
      overflow_.reset(other.overflow_ ? new OverflowSetType(*other.overflow_)
                                      : nullptr);
    }
    return *this;
  }

  friend bool operator==(const EnumSet& a, const EnumSet& b) {
    if (a.mask_ != b.mask_) {
      return false;
    }

    if (a.overflow_ == nullptr && b.overflow_ == nullptr) {
      return true;
    }

    if (a.overflow_ == nullptr || b.overflow_ == nullptr) {
      return false;
    }

    return *a.overflow_ == *b.overflow_;
  }

  friend bool operator!=(const EnumSet& a, const EnumSet& b) {
    return !(a == b);
  }

  // Adds the given enum value to the set.  This has no effect if the
  // enum value is already in the set.
  void Add(EnumType c) { AddWord(ToWord(c)); }

  // Removes the given enum value from the set.  This has no effect if the
  // enum value is not in the set.
  void Remove(EnumType c) { RemoveWord(ToWord(c)); }

  // Returns true if this enum value is in the set.
  bool Contains(EnumType c) const { return ContainsWord(ToWord(c)); }

  // Applies f to each enum in the set, in order from smallest enum
  // value to largest.
  void ForEach(std::function<void(EnumType)> f) const {
    for (uint32_t i = 0; i < 64; ++i) {
      if (mask_ & AsMask(i)) f(static_cast<EnumType>(i));
    }
    if (overflow_) {
      for (uint32_t c : *overflow_) f(static_cast<EnumType>(c));
    }
  }

  // Returns true if the set is empty.
  bool IsEmpty() const {
    if (mask_) return false;
    if (overflow_ && !overflow_->empty()) return false;
    return true;
  }

  // Returns true if the set contains ANY of the elements of |in_set|,
  // or if |in_set| is empty.
  bool HasAnyOf(const EnumSet<EnumType>& in_set) const {
    if (in_set.IsEmpty()) return true;

    if (mask_ & in_set.mask_) return true;

    if (!overflow_ || !in_set.overflow_) return false;

    for (uint32_t item : *in_set.overflow_) {
      if (overflow_->find(item) != overflow_->end()) return true;
    }

    return false;
  }

 private:
  // Adds the given enum value (as a 32-bit word) to the set.  This has no
  // effect if the enum value is already in the set.
  void AddWord(uint32_t word) {
    if (auto new_bits = AsMask(word)) {
      mask_ |= new_bits;
    } else {
      Overflow().insert(word);
    }
  }

  // Removes the given enum value (as a 32-bit word) from the set.  This has no
  // effect if the enum value is not in the set.
  void RemoveWord(uint32_t word) {
    if (auto new_bits = AsMask(word)) {
      mask_ &= ~new_bits;
    } else {
      auto itr = Overflow().find(word);
      if (itr != Overflow().end()) Overflow().erase(itr);
    }
  }

  // Returns true if the enum represented as a 32-bit word is in the set.
  bool ContainsWord(uint32_t word) const {
    // We shouldn't call Overflow() since this is a const method.
    if (auto bits = AsMask(word)) {
      return (mask_ & bits) != 0;
    } else if (auto overflow = overflow_.get()) {
      return overflow->find(word) != overflow->end();
    }
    // The word is large, but the set doesn't have large members, so
    // it doesn't have an overflow set.
    return false;
  }

  // Returns the enum value as a uint32_t.
  uint32_t ToWord(EnumType value) const {
    static_assert(sizeof(EnumType) <= sizeof(uint32_t),
                  "EnumType must statically castable to uint32_t");
    return static_cast<uint32_t>(value);
  }

  // Determines whether the given enum value can be represented
  // as a bit in a uint64_t mask. If so, then returns that mask bit.
  // Otherwise, returns 0.
  uint64_t AsMask(uint32_t word) const {
    if (word > 63) return 0;
    return uint64_t(1) << word;
  }

  // Ensures that overflow_set_ references a set.  A new empty set is
  // allocated if one doesn't exist yet.  Returns overflow_set_.
  OverflowSetType& Overflow() {
    if (overflow_.get() == nullptr) {
      overflow_ = MakeUnique<OverflowSetType>();
    }
    return *overflow_;
  }

  // Enums with values up to 63 are stored as bits in this mask.
  uint64_t mask_ = 0;
  // Enums with values larger than 63 are stored in this set.
  // This set should normally be empty or very small.
  std::unique_ptr<OverflowSetType> overflow_ = {};
};

// A set of spv::Capability, optimized for small capability values.
using CapabilitySet = EnumSet<spv::Capability>;

}  // namespace spvtools

#endif  // SOURCE_ENUM_SET_H_
