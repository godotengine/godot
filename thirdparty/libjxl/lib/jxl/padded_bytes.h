// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_PADDED_BYTES_H_
#define LIB_JXL_BASE_PADDED_BYTES_H_

// std::vector replacement with padding to reduce bounds checks in WriteBits

#include <jxl/memory_manager.h>

#include <algorithm>  // max
#include <cstddef>
#include <cstdint>
#include <cstring>  // memcpy
#include <initializer_list>
#include <utility>  // swap

#include "lib/jxl/base/status.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

// Provides a subset of the std::vector interface with some differences:
// - allows BitWriter to write 64 bits at a time without bounds checking;
// - ONLY zero-initializes the first byte (required by BitWriter);
// - ensures cache-line alignment.
class PaddedBytes {
 public:
  // Required for output params.
  explicit PaddedBytes(JxlMemoryManager* memory_manager)
      : memory_manager_(memory_manager), size_(0), capacity_(0) {}

  static StatusOr<PaddedBytes> WithInitialSpace(
      JxlMemoryManager* memory_manager, size_t size) {
    PaddedBytes result(memory_manager);
    JXL_RETURN_IF_ERROR(result.Init(size));
    return result;
  }

  // Deleting copy constructor and copy assignment operator to prevent copying
  PaddedBytes(const PaddedBytes&) = delete;
  PaddedBytes& operator=(const PaddedBytes&) = delete;

  // default is not OK - need to set other.size_ to 0!
  PaddedBytes(PaddedBytes&& other) noexcept
      : memory_manager_(other.memory_manager_),
        size_(other.size_),
        capacity_(other.capacity_),
        data_(std::move(other.data_)) {
    other.size_ = other.capacity_ = 0;
  }
  PaddedBytes& operator=(PaddedBytes&& other) noexcept {
    memory_manager_ = other.memory_manager_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    data_ = std::move(other.data_);

    if (&other != this) {
      other.size_ = other.capacity_ = 0;
    }
    return *this;
  }

  JxlMemoryManager* memory_manager() const { return memory_manager_; }

  void swap(PaddedBytes& other) noexcept {
    std::swap(memory_manager_, other.memory_manager_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
    std::swap(data_, other.data_);
  }

  // If current capacity is greater than requested, then no-op. Otherwise
  // copies existing data to newly allocated "data_".
  // The new capacity will be at least 1.5 times the old capacity. This ensures
  // that we avoid quadratic behaviour.
  Status reserve(size_t capacity) {
    if (capacity <= capacity_) return true;

    size_t new_capacity;
    if (!SafeAdd(capacity_, capacity_ / 2, new_capacity)) {
      return JXL_FAILURE("PaddedBytes reserve: capacity overflow");
    }
    new_capacity = std::max(capacity, new_capacity);

    return change_capacity(new_capacity);
  }

  Status shrink_to_fit() { return change_capacity(size_); }

  // NOTE: unlike vector, this does not initialize the new data!
  // However, we guarantee that write_bits can safely append after
  // the resize, as we zero-initialize the first new byte of data.
  // If size < capacity(), does not invalidate the memory.
  Status resize(size_t size) {
    JXL_RETURN_IF_ERROR(reserve(size));
    size_ = size;
    return true;
  }

  // resize(size) plus explicit initialization of the new data with `value`.
  Status resize(size_t size, uint8_t value) {
    size_t old_size = size_;
    JXL_RETURN_IF_ERROR(resize(size));
    if (size_ > old_size) {
      memset(data() + old_size, value, size_ - old_size);
    }
    return true;
  }

  // Amortized constant complexity due to exponential growth.
  Status push_back(uint8_t x) {
    if (size_ == capacity_) {
      JXL_RETURN_IF_ERROR(reserve(capacity_ + 1));
    }

    data_.address<uint8_t>()[size_++] = x;
    return true;
  }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  uint8_t* data() { return data_.address<uint8_t>(); }
  const uint8_t* data() const { return data_.address<uint8_t>(); }

  // std::vector operations implemented in terms of the public interface above.

  void clear() {
    // Not passing on the Status, because resizing to 0 cannot fail.
    static_cast<void>(resize(0));
  }
  bool empty() const { return size() == 0; }

  Status assign(std::initializer_list<uint8_t> il) {
    JXL_RETURN_IF_ERROR(resize(il.size()));
    memcpy(data(), il.begin(), il.size());
    return true;
  }

  uint8_t* begin() { return data(); }
  const uint8_t* begin() const { return data(); }
  uint8_t* end() { return begin() + size(); }
  const uint8_t* end() const { return begin() + size(); }

  uint8_t& operator[](const size_t i) {
    BoundsCheck(i);
    return data()[i];
  }
  const uint8_t& operator[](const size_t i) const {
    BoundsCheck(i);
    return data()[i];
  }

  template <typename T>
  Status append(const T& other) {
    return append(
        reinterpret_cast<const uint8_t*>(other.data()),
        reinterpret_cast<const uint8_t*>(other.data()) + other.size());
  }

  Status append(const uint8_t* begin, const uint8_t* end) {
    if (end > begin) {
      size_t old_size = size();
      size_t new_size;
      if (!SafeAdd(old_size, static_cast<size_t>(end - begin), new_size)) {
        return JXL_FAILURE("PaddedBytes append: overflow");
      }
      JXL_RETURN_IF_ERROR(resize(new_size));
      memcpy(data() + old_size, begin, end - begin);
    }
    return true;
  }

 private:
  Status Init(size_t size) {
    size_ = size;
    return reserve(size);
  }

  void BoundsCheck(size_t i) const {
    // <= is safe due to padding and required by BitWriter.
    JXL_DASSERT(i <= size());
  }

  Status change_capacity(size_t new_capacity) {
    JXL_DASSERT(new_capacity >= size_);

    new_capacity = std::max<size_t>(64, new_capacity);

    // BitWriter writes up to 7 bytes past the end.
    size_t padded_capacity;
    if (!SafeAdd(new_capacity, static_cast<size_t>(8), padded_capacity)) {
      return JXL_FAILURE("PaddedBytes capacity too large");
    }
    JXL_ASSIGN_OR_RETURN(
        AlignedMemory new_data,
        AlignedMemory::Create(memory_manager_, padded_capacity));

    if (data_.address<void>() == nullptr) {
      // First allocation: ensure first byte is initialized (won't be copied).
      new_data.address<uint8_t>()[0] = 0;
    } else {
      // Subsequent resize: copy existing data to new location.
      memmove(new_data.address<void>(), data_.address<void>(), size_);
      // Ensure that the first new byte is initialized, to allow write_bits to
      // safely append to the newly-resized PaddedBytes.
      new_data.address<uint8_t>()[size_] = 0;
    }

    capacity_ = new_capacity;
    data_ = std::move(new_data);
    return true;
  }

  JxlMemoryManager* memory_manager_;
  size_t size_;
  size_t capacity_;
  AlignedMemory data_;
};

}  // namespace jxl

#endif  // LIB_JXL_BASE_PADDED_BYTES_H_
