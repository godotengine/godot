// Copyright (c) 2011, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// memory_range.h: Define the google_breakpad::MemoryRange class, which
// is a lightweight wrapper with a pointer and a length to encapsulate
// a contiguous range of memory.

#ifndef COMMON_MEMORY_RANGE_H_
#define COMMON_MEMORY_RANGE_H_

#include <stddef.h>

#include "google_breakpad/common/breakpad_types.h"

namespace google_breakpad {

// A lightweight wrapper with a pointer and a length to encapsulate a
// contiguous range of memory. It provides helper methods for checked
// access of a subrange of the memory. Its implemementation does not
// allocate memory or call into libc functions, and is thus safer to use
// in a crashed environment.
class MemoryRange {
 public:
  MemoryRange() : data_(NULL), length_(0) {}

  MemoryRange(const void* data, size_t length) {
    Set(data, length);
  }

  // Returns true if this memory range contains no data.
  bool IsEmpty() const {
    // Set() guarantees that |length_| is zero if |data_| is NULL.
    return length_ == 0;
  }

  // Resets to an empty range.
  void Reset() {
    data_ = NULL;
    length_ = 0;
  }

  // Sets this memory range to point to |data| and its length to |length|.
  void Set(const void* data, size_t length) {
    data_ = reinterpret_cast<const uint8_t*>(data);
    // Always set |length_| to zero if |data_| is NULL.
    length_ = data ? length : 0;
  }

  // Returns true if this range covers a subrange of |sub_length| bytes
  // at |sub_offset| bytes of this memory range, or false otherwise.
  bool Covers(size_t sub_offset, size_t sub_length) const {
    // The following checks verify that:
    // 1. sub_offset is within [ 0 .. length_ - 1 ]
    // 2. sub_offset + sub_length is within
    //    [ sub_offset .. length_ ]
    return sub_offset < length_ &&
           sub_offset + sub_length >= sub_offset &&
           sub_offset + sub_length <= length_;
  }

  // Returns a raw data pointer to a subrange of |sub_length| bytes at
  // |sub_offset| bytes of this memory range, or NULL if the subrange
  // is out of bounds.
  const void* GetData(size_t sub_offset, size_t sub_length) const {
    return Covers(sub_offset, sub_length) ? (data_ + sub_offset) : NULL;
  }

  // Same as the two-argument version of GetData() but uses sizeof(DataType)
  // as the subrange length and returns an |DataType| pointer for convenience.
  template <typename DataType>
  const DataType* GetData(size_t sub_offset) const {
    return reinterpret_cast<const DataType*>(
        GetData(sub_offset, sizeof(DataType)));
  }

  // Returns a raw pointer to the |element_index|-th element of an array
  // of elements of length |element_size| starting at |sub_offset| bytes
  // of this memory range, or NULL if the element is out of bounds.
  const void* GetArrayElement(size_t element_offset,
                              size_t element_size,
                              unsigned element_index) const {
    size_t sub_offset = element_offset + element_index * element_size;
    return GetData(sub_offset, element_size);
  }

  // Same as the three-argument version of GetArrayElement() but deduces
  // the element size using sizeof(ElementType) and returns an |ElementType|
  // pointer for convenience.
  template <typename ElementType>
  const ElementType* GetArrayElement(size_t element_offset,
                                     unsigned element_index) const {
    return reinterpret_cast<const ElementType*>(
        GetArrayElement(element_offset, sizeof(ElementType), element_index));
  }

  // Returns a subrange of |sub_length| bytes at |sub_offset| bytes of
  // this memory range, or an empty range if the subrange is out of bounds.
  MemoryRange Subrange(size_t sub_offset, size_t sub_length) const {
    return Covers(sub_offset, sub_length) ?
        MemoryRange(data_ + sub_offset, sub_length) : MemoryRange();
  }

  // Returns a pointer to the beginning of this memory range.
  const uint8_t* data() const { return data_; }

  // Returns the length, in bytes, of this memory range.
  size_t length() const { return length_; }

 private:
  // Pointer to the beginning of this memory range.
  const uint8_t* data_;

  // Length, in bytes, of this memory range.
  size_t length_;
};

}  // namespace google_breakpad

#endif  // COMMON_MEMORY_RANGE_H_
