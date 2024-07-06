// Copyright (c) 2023 Google Inc.
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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#ifndef SOURCE_ENUM_SET_H_
#define SOURCE_ENUM_SET_H_

#include "source/latest_version_spirv_header.h"

namespace spvtools {

// This container is optimized to store and retrieve unsigned enum values.
// The base model for this implementation is an open-addressing hashtable with
// linear probing. For small enums (max index < 64), all operations are O(1).
//
// - Enums are stored in buckets (64 contiguous values max per bucket)
// - Buckets ranges don't overlap, but don't have to be contiguous.
// - Enums are packed into 64-bits buckets, using 1 bit per enum value.
//
// Example:
//  - MyEnum { A = 0, B = 1, C = 64, D = 65 }
//  - 2 buckets are required:
//      - bucket 0, storing values in the range [ 0;  64[
//      - bucket 1, storing values in the range [64; 128[
//
// - Buckets are stored in a sorted vector (sorted by bucket range).
// - Retrieval is done by computing the theoretical bucket index using the enum
// value, and
//   doing a linear scan from this position.
// - Insertion is done by retrieving the bucket and either:
//   - inserting a new bucket in the sorted vector when no buckets has a
//   compatible range.
//   - setting the corresponding bit in the bucket.
//   This means insertion in the middle/beginning can cause a memmove when no
//   bucket is available. In our case, this happens at most 23 times for the
//   largest enum we have (Opcodes).
template <typename T>
class EnumSet {
 private:
  using BucketType = uint64_t;
  using ElementType = std::underlying_type_t<T>;
  static_assert(std::is_enum_v<T>, "EnumSets only works with enums.");
  static_assert(std::is_signed_v<ElementType> == false,
                "EnumSet doesn't supports signed enums.");

  // Each bucket can hold up to `kBucketSize` distinct, contiguous enum values.
  // The first value a bucket can hold must be aligned on `kBucketSize`.
  struct Bucket {
    // bit mask to store `kBucketSize` enums.
    BucketType data;
    // 1st enum this bucket can represent.
    T start;

    friend bool operator==(const Bucket& lhs, const Bucket& rhs) {
      return lhs.start == rhs.start && lhs.data == rhs.data;
    }
  };

  // How many distinct values can a bucket hold? 1 bit per value.
  static constexpr size_t kBucketSize = sizeof(BucketType) * 8ULL;

 public:
  class Iterator {
   public:
    typedef Iterator self_type;
    typedef T value_type;
    typedef T& reference;
    typedef T* pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef size_t difference_type;

    Iterator(const Iterator& other)
        : set_(other.set_),
          bucketIndex_(other.bucketIndex_),
          bucketOffset_(other.bucketOffset_) {}

    Iterator& operator++() {
      do {
        if (bucketIndex_ >= set_->buckets_.size()) {
          bucketIndex_ = set_->buckets_.size();
          bucketOffset_ = 0;
          break;
        }

        if (bucketOffset_ + 1 == kBucketSize) {
          bucketOffset_ = 0;
          ++bucketIndex_;
        } else {
          ++bucketOffset_;
        }

      } while (bucketIndex_ < set_->buckets_.size() &&
               !set_->HasEnumAt(bucketIndex_, bucketOffset_));
      return *this;
    }

    Iterator operator++(int) {
      Iterator old = *this;
      operator++();
      return old;
    }

    T operator*() const {
      assert(set_->HasEnumAt(bucketIndex_, bucketOffset_) &&
             "operator*() called on an invalid iterator.");
      return GetValueFromBucket(set_->buckets_[bucketIndex_], bucketOffset_);
    }

    bool operator!=(const Iterator& other) const {
      return set_ != other.set_ || bucketOffset_ != other.bucketOffset_ ||
             bucketIndex_ != other.bucketIndex_;
    }

    bool operator==(const Iterator& other) const {
      return !(operator!=(other));
    }

    Iterator& operator=(const Iterator& other) {
      set_ = other.set_;
      bucketIndex_ = other.bucketIndex_;
      bucketOffset_ = other.bucketOffset_;
      return *this;
    }

   private:
    Iterator(const EnumSet* set, size_t bucketIndex, ElementType bucketOffset)
        : set_(set), bucketIndex_(bucketIndex), bucketOffset_(bucketOffset) {}

   private:
    const EnumSet* set_ = nullptr;
    // Index of the bucket in the vector.
    size_t bucketIndex_ = 0;
    // Offset in bits in the current bucket.
    ElementType bucketOffset_ = 0;

    friend class EnumSet;
  };

  // Required to allow the use of std::inserter.
  using value_type = T;
  using const_iterator = Iterator;
  using iterator = Iterator;

 public:
  iterator cbegin() const noexcept {
    auto it = iterator(this, /* bucketIndex= */ 0, /* bucketOffset= */ 0);
    if (buckets_.size() == 0) {
      return it;
    }

    // The iterator has the logic to find the next valid bit. If the value 0
    // is not stored, use it to find the next valid bit.
    if (!HasEnumAt(it.bucketIndex_, it.bucketOffset_)) {
      ++it;
    }

    return it;
  }

  iterator begin() const noexcept { return cbegin(); }

  iterator cend() const noexcept {
    return iterator(this, buckets_.size(), /* bucketOffset= */ 0);
  }

  iterator end() const noexcept { return cend(); }

  // Creates an empty set.
  EnumSet() : buckets_(0), size_(0) {}

  // Creates a set and store `value` in it.
  EnumSet(T value) : EnumSet() { insert(value); }

  // Creates a set and stores each `values` in it.
  EnumSet(std::initializer_list<T> values) : EnumSet() {
    for (auto item : values) {
      insert(item);
    }
  }

  // Creates a set, and insert `count` enum values pointed by `array` in it.
  EnumSet(ElementType count, const T* array) : EnumSet() {
    for (ElementType i = 0; i < count; i++) {
      insert(array[i]);
    }
  }

  // Creates a set initialized with the content of the range [begin; end[.
  template <class InputIt>
  EnumSet(InputIt begin, InputIt end) : EnumSet() {
    for (; begin != end; ++begin) {
      insert(*begin);
    }
  }

  // Copies the EnumSet `other` into a new EnumSet.
  EnumSet(const EnumSet& other)
      : buckets_(other.buckets_), size_(other.size_) {}

  // Moves the EnumSet `other` into a new EnumSet.
  EnumSet(EnumSet&& other)
      : buckets_(std::move(other.buckets_)), size_(other.size_) {}

  // Deep-copies the EnumSet `other` into this EnumSet.
  EnumSet& operator=(const EnumSet& other) {
    buckets_ = other.buckets_;
    size_ = other.size_;
    return *this;
  }

  // Matches std::unordered_set::insert behavior.
  std::pair<iterator, bool> insert(const T& value) {
    const size_t index = FindBucketForValue(value);
    const ElementType offset = ComputeBucketOffset(value);

    if (index >= buckets_.size() ||
        buckets_[index].start != ComputeBucketStart(value)) {
      size_ += 1;
      InsertBucketFor(index, value);
      return std::make_pair(Iterator(this, index, offset), true);
    }

    auto& bucket = buckets_[index];
    const auto mask = ComputeMaskForValue(value);
    if (bucket.data & mask) {
      return std::make_pair(Iterator(this, index, offset), false);
    }

    size_ += 1;
    bucket.data |= ComputeMaskForValue(value);
    return std::make_pair(Iterator(this, index, offset), true);
  }

  // Inserts `value` in the set if possible.
  // Similar to `std::unordered_set::insert`, except the hint is ignored.
  // Returns an iterator to the inserted element, or the element preventing
  // insertion.
  iterator insert(const_iterator, const T& value) {
    return insert(value).first;
  }

  // Inserts `value` in the set if possible.
  // Similar to `std::unordered_set::insert`, except the hint is ignored.
  // Returns an iterator to the inserted element, or the element preventing
  // insertion.
  iterator insert(const_iterator, T&& value) { return insert(value).first; }

  // Inserts all the values in the range [`first`; `last[.
  // Similar to `std::unordered_set::insert`.
  template <class InputIt>
  void insert(InputIt first, InputIt last) {
    for (auto it = first; it != last; ++it) {
      insert(*it);
    }
  }

  // Removes the value `value` into the set.
  // Similar to `std::unordered_set::erase`.
  // Returns the number of erased elements.
  size_t erase(const T& value) {
    const size_t index = FindBucketForValue(value);
    if (index >= buckets_.size() ||
        buckets_[index].start != ComputeBucketStart(value)) {
      return 0;
    }

    auto& bucket = buckets_[index];
    const auto mask = ComputeMaskForValue(value);
    if (!(bucket.data & mask)) {
      return 0;
    }

    size_ -= 1;
    bucket.data &= ~mask;
    if (bucket.data == 0) {
      buckets_.erase(buckets_.cbegin() + index);
    }
    return 1;
  }

  // Returns true if `value` is present in the set.
  bool contains(T value) const {
    const size_t index = FindBucketForValue(value);
    if (index >= buckets_.size() ||
        buckets_[index].start != ComputeBucketStart(value)) {
      return false;
    }
    auto& bucket = buckets_[index];
    return bucket.data & ComputeMaskForValue(value);
  }

  // Returns the 1 if `value` is present in the set, `0` otherwise.
  inline size_t count(T value) const { return contains(value) ? 1 : 0; }

  // Returns true if the set is holds no values.
  inline bool empty() const { return size_ == 0; }

  // Returns the number of enums stored in this set.
  size_t size() const { return size_; }

  // Returns true if this set contains at least one value contained in `in_set`.
  // Note: If `in_set` is empty, this function returns true.
  bool HasAnyOf(const EnumSet<T>& in_set) const {
    if (in_set.empty()) {
      return true;
    }

    auto lhs = buckets_.cbegin();
    auto rhs = in_set.buckets_.cbegin();

    while (lhs != buckets_.cend() && rhs != in_set.buckets_.cend()) {
      if (lhs->start == rhs->start) {
        if (lhs->data & rhs->data) {
          // At least 1 bit is shared. Early return.
          return true;
        }

        lhs++;
        rhs++;
        continue;
      }

      // LHS bucket is smaller than the current RHS bucket. Catching up on RHS.
      if (lhs->start < rhs->start) {
        lhs++;
        continue;
      }

      // Otherwise, RHS needs to catch up on LHS.
      rhs++;
    }

    return false;
  }

 private:
  // Returns the index of the last bucket in which `value` could be stored.
  static constexpr inline size_t ComputeLargestPossibleBucketIndexFor(T value) {
    return static_cast<size_t>(value) / kBucketSize;
  }

  // Returns the smallest enum value that could be contained in the same bucket
  // as `value`.
  static constexpr inline T ComputeBucketStart(T value) {
    return static_cast<T>(kBucketSize *
                          ComputeLargestPossibleBucketIndexFor(value));
  }

  //  Returns the index of the bit that corresponds to `value` in the bucket.
  static constexpr inline ElementType ComputeBucketOffset(T value) {
    return static_cast<ElementType>(value) % kBucketSize;
  }

  // Returns the bitmask used to represent the enum `value` in its bucket.
  static constexpr inline BucketType ComputeMaskForValue(T value) {
    return 1ULL << ComputeBucketOffset(value);
  }

  // Returns the `enum` stored in `bucket` at `offset`.
  // `offset` is the bit-offset in the bucket storage.
  static constexpr inline T GetValueFromBucket(const Bucket& bucket,
                                               BucketType offset) {
    return static_cast<T>(static_cast<ElementType>(bucket.start) + offset);
  }

  // For a given enum `value`, finds the bucket index that could contain this
  // value. If no such bucket is found, the index at which the new bucket should
  // be inserted is returned.
  size_t FindBucketForValue(T value) const {
    // Set is empty, insert at 0.
    if (buckets_.size() == 0) {
      return 0;
    }

    const T wanted_start = ComputeBucketStart(value);
    assert(buckets_.size() > 0 &&
           "Size must not be 0 here. Has the code above changed?");
    size_t index = std::min(buckets_.size() - 1,
                            ComputeLargestPossibleBucketIndexFor(value));

    // This loops behaves like std::upper_bound with a reverse iterator.
    // Buckets are sorted. 3 main cases:
    //  - The bucket matches
    //    => returns the bucket index.
    //  - The found bucket is larger
    //    => scans left until it finds the correct bucket, or insertion point.
    //  - The found bucket is smaller
    //    => We are at the end, so we return past-end index for insertion.
    for (; buckets_[index].start >= wanted_start; index--) {
      if (index == 0) {
        return 0;
      }
    }

    return index + 1;
  }

  // Creates a new bucket to store `value` and inserts it at `index`.
  // If the `index` is past the end, the bucket is inserted at the end of the
  // vector.
  void InsertBucketFor(size_t index, T value) {
    const T bucket_start = ComputeBucketStart(value);
    Bucket bucket = {1ULL << ComputeBucketOffset(value), bucket_start};
    auto it = buckets_.emplace(buckets_.begin() + index, std::move(bucket));
#if defined(NDEBUG)
    (void)it;  // Silencing unused variable warning.
#else
    assert(std::next(it) == buckets_.end() ||
           std::next(it)->start > bucket_start);
    assert(it == buckets_.begin() || std::prev(it)->start < bucket_start);
#endif
  }

  // Returns true if the bucket at `bucketIndex/ stores the enum at
  // `bucketOffset`, false otherwise.
  bool HasEnumAt(size_t bucketIndex, BucketType bucketOffset) const {
    assert(bucketIndex < buckets_.size());
    assert(bucketOffset < kBucketSize);
    return buckets_[bucketIndex].data & (1ULL << bucketOffset);
  }

  // Returns true if `lhs` and `rhs` hold the exact same values.
  friend bool operator==(const EnumSet& lhs, const EnumSet& rhs) {
    if (lhs.size_ != rhs.size_) {
      return false;
    }

    if (lhs.buckets_.size() != rhs.buckets_.size()) {
      return false;
    }
    return lhs.buckets_ == rhs.buckets_;
  }

  // Returns true if `lhs` and `rhs` hold at least 1 different value.
  friend bool operator!=(const EnumSet& lhs, const EnumSet& rhs) {
    return !(lhs == rhs);
  }

  // Storage for the buckets.
  std::vector<Bucket> buckets_;
  // How many enums is this set storing.
  size_t size_ = 0;
};

// A set of spv::Capability.
using CapabilitySet = EnumSet<spv::Capability>;

}  // namespace spvtools

#endif  // SOURCE_ENUM_SET_H_
