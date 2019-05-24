// Copyright 2006 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef UTIL_SPARSE_ARRAY_H_
#define UTIL_SPARSE_ARRAY_H_

// DESCRIPTION
//
// SparseArray<T>(m) is a map from integers in [0, m) to T values.
// It requires (sizeof(T)+sizeof(int))*m memory, but it provides
// fast iteration through the elements in the array and fast clearing
// of the array.  The array has a concept of certain elements being
// uninitialized (having no value).
//
// Insertion and deletion are constant time operations.
//
// Allocating the array is a constant time operation
// when memory allocation is a constant time operation.
//
// Clearing the array is a constant time operation (unusual!).
//
// Iterating through the array is an O(n) operation, where n
// is the number of items in the array (not O(m)).
//
// The array iterator visits entries in the order they were first
// inserted into the array.  It is safe to add items to the array while
// using an iterator: the iterator will visit indices added to the array
// during the iteration, but will not re-visit indices whose values
// change after visiting.  Thus SparseArray can be a convenient
// implementation of a work queue.
//
// The SparseArray implementation is NOT thread-safe.  It is up to the
// caller to make sure only one thread is accessing the array.  (Typically
// these arrays are temporary values and used in situations where speed is
// important.)
//
// The SparseArray interface does not present all the usual STL bells and
// whistles.
//
// Implemented with reference to Briggs & Torczon, An Efficient
// Representation for Sparse Sets, ACM Letters on Programming Languages
// and Systems, Volume 2, Issue 1-4 (March-Dec.  1993), pp.  59-69.
//
// Briggs & Torczon popularized this technique, but it had been known
// long before their paper.  They point out that Aho, Hopcroft, and
// Ullman's 1974 Design and Analysis of Computer Algorithms and Bentley's
// 1986 Programming Pearls both hint at the technique in exercises to the
// reader (in Aho & Hopcroft, exercise 2.12; in Bentley, column 1
// exercise 8).
//
// Briggs & Torczon describe a sparse set implementation.  I have
// trivially generalized it to create a sparse array (actually the original
// target of the AHU and Bentley exercises).

// IMPLEMENTATION
//
// SparseArray is an array dense_ and an array sparse_ of identical size.
// At any point, the number of elements in the sparse array is size_.
//
// The array dense_ contains the size_ elements in the sparse array (with
// their indices),
// in the order that the elements were first inserted.  This array is dense:
// the size_ pairs are dense_[0] through dense_[size_-1].
//
// The array sparse_ maps from indices in [0,m) to indices in [0,size_).
// For indices present in the array, dense_[sparse_[i]].index_ == i.
// For indices not present in the array, sparse_ can contain any value at all,
// perhaps outside the range [0, size_) but perhaps not.
//
// The lax requirement on sparse_ values makes clearing the array very easy:
// set size_ to 0.  Lookups are slightly more complicated.
// An index i has a value in the array if and only if:
//   sparse_[i] is in [0, size_) AND
//   dense_[sparse_[i]].index_ == i.
// If both these properties hold, only then it is safe to refer to
//   dense_[sparse_[i]].value_
// as the value associated with index i.
//
// To insert a new entry, set sparse_[i] to size_,
// initialize dense_[size_], and then increment size_.
//
// To make the sparse array as efficient as possible for non-primitive types,
// elements may or may not be destroyed when they are deleted from the sparse
// array through a call to resize(). They immediately become inaccessible, but
// they are only guaranteed to be destroyed when the SparseArray destructor is
// called.
//
// A moved-from SparseArray will be empty.

// Doing this simplifies the logic below.
#ifndef __has_feature
#define __has_feature(x) 0
#endif

#include <assert.h>
#include <stdint.h>
#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif
#include <algorithm>
#include <memory>
#include <utility>

#include "util/pod_array.h"

namespace re2 {

template<typename Value>
class SparseArray {
 public:
  SparseArray();
  explicit SparseArray(int max_size);
  ~SparseArray();

  // IndexValue pairs: exposed in SparseArray::iterator.
  class IndexValue;

  typedef IndexValue* iterator;
  typedef const IndexValue* const_iterator;

  SparseArray(const SparseArray& src);
  SparseArray(SparseArray&& src);

  SparseArray& operator=(const SparseArray& src);
  SparseArray& operator=(SparseArray&& src);

  // Return the number of entries in the array.
  int size() const {
    return size_;
  }

  // Indicate whether the array is empty.
  int empty() const {
    return size_ == 0;
  }

  // Iterate over the array.
  iterator begin() {
    return dense_.data();
  }
  iterator end() {
    return dense_.data() + size_;
  }

  const_iterator begin() const {
    return dense_.data();
  }
  const_iterator end() const {
    return dense_.data() + size_;
  }

  // Change the maximum size of the array.
  // Invalidates all iterators.
  void resize(int new_max_size);

  // Return the maximum size of the array.
  // Indices can be in the range [0, max_size).
  int max_size() const {
    if (dense_.data() != NULL)
      return dense_.size();
    else
      return 0;
  }

  // Clear the array.
  void clear() {
    size_ = 0;
  }

  // Check whether index i is in the array.
  bool has_index(int i) const;

  // Comparison function for sorting.
  // Can sort the sparse array so that future iterations
  // will visit indices in increasing order using
  // std::sort(arr.begin(), arr.end(), arr.less);
  static bool less(const IndexValue& a, const IndexValue& b);

 public:
  // Set the value at index i to v.
  iterator set(int i, const Value& v) {
    return SetInternal(true, i, v);
  }

  // Set the value at new index i to v.
  // Fast but unsafe: only use if has_index(i) is false.
  iterator set_new(int i, const Value& v) {
    return SetInternal(false, i, v);
  }

  // Set the value at index i to v.
  // Fast but unsafe: only use if has_index(i) is true.
  iterator set_existing(int i, const Value& v) {
    return SetExistingInternal(i, v);
  }

  // Get the value at index i.
  // Fast but unsafe: only use if has_index(i) is true.
  Value& get_existing(int i) {
    assert(has_index(i));
    return dense_[sparse_[i]].value_;
  }
  const Value& get_existing(int i) const {
    assert(has_index(i));
    return dense_[sparse_[i]].value_;
  }

 private:
  iterator SetInternal(bool allow_existing, int i, const Value& v) {
    DebugCheckInvariants();
    if (static_cast<uint32_t>(i) >= static_cast<uint32_t>(max_size())) {
      assert(false && "illegal index");
      // Semantically, end() would be better here, but we already know
      // the user did something stupid, so begin() insulates them from
      // dereferencing an invalid pointer.
      return begin();
    }
    if (!allow_existing) {
      assert(!has_index(i));
      create_index(i);
    } else {
      if (!has_index(i))
        create_index(i);
    }
    return SetExistingInternal(i, v);
  }

  iterator SetExistingInternal(int i, const Value& v) {
    DebugCheckInvariants();
    assert(has_index(i));
    dense_[sparse_[i]].value_ = v;
    DebugCheckInvariants();
    return dense_.data() + sparse_[i];
  }

  // Add the index i to the array.
  // Only use if has_index(i) is known to be false.
  // Since it doesn't set the value associated with i,
  // this function is private, only intended as a helper
  // for other methods.
  void create_index(int i);

  // In debug mode, verify that some invariant properties of the class
  // are being maintained. This is called at the end of the constructor
  // and at the beginning and end of all public non-const member functions.
  void DebugCheckInvariants() const;

  // Initializes memory for elements [min, max).
  void MaybeInitializeMemory(int min, int max) {
#if __has_feature(memory_sanitizer)
    __msan_unpoison(sparse_.data() + min, (max - min) * sizeof sparse_[0]);
#elif defined(RE2_ON_VALGRIND)
    for (int i = min; i < max; i++) {
      sparse_[i] = 0xababababU;
    }
#endif
  }

  int size_ = 0;
  PODArray<int> sparse_;
  PODArray<IndexValue> dense_;
};

template<typename Value>
SparseArray<Value>::SparseArray() = default;

template<typename Value>
SparseArray<Value>::SparseArray(const SparseArray& src)
    : size_(src.size_),
      sparse_(src.max_size()),
      dense_(src.max_size()) {
  std::copy_n(src.sparse_.data(), src.max_size(), sparse_.data());
  std::copy_n(src.dense_.data(), src.max_size(), dense_.data());
}

template<typename Value>
SparseArray<Value>::SparseArray(SparseArray&& src)
    : size_(src.size_),
      sparse_(std::move(src.sparse_)),
      dense_(std::move(src.dense_)) {
  src.size_ = 0;
}

template<typename Value>
SparseArray<Value>& SparseArray<Value>::operator=(const SparseArray& src) {
  // Construct these first for exception safety.
  PODArray<int> a(src.max_size());
  PODArray<IndexValue> b(src.max_size());

  size_ = src.size_;
  sparse_ = std::move(a);
  dense_ = std::move(b);
  std::copy_n(src.sparse_.data(), src.max_size(), sparse_.data());
  std::copy_n(src.dense_.data(), src.max_size(), dense_.data());
  return *this;
}

template<typename Value>
SparseArray<Value>& SparseArray<Value>::operator=(SparseArray&& src) {
  size_ = src.size_;
  sparse_ = std::move(src.sparse_);
  dense_ = std::move(src.dense_);
  src.size_ = 0;
  return *this;
}

// IndexValue pairs: exposed in SparseArray::iterator.
template<typename Value>
class SparseArray<Value>::IndexValue {
 public:
  int index() const { return index_; }
  Value& value() { return value_; }
  const Value& value() const { return value_; }

 private:
  friend class SparseArray;
  int index_;
  Value value_;
};

// Change the maximum size of the array.
// Invalidates all iterators.
template<typename Value>
void SparseArray<Value>::resize(int new_max_size) {
  DebugCheckInvariants();
  if (new_max_size > max_size()) {
    const int old_max_size = max_size();

    // Construct these first for exception safety.
    PODArray<int> a(new_max_size);
    PODArray<IndexValue> b(new_max_size);

    std::copy_n(sparse_.data(), old_max_size, a.data());
    std::copy_n(dense_.data(), old_max_size, b.data());

    sparse_ = std::move(a);
    dense_ = std::move(b);

    MaybeInitializeMemory(old_max_size, new_max_size);
  }
  if (size_ > new_max_size)
    size_ = new_max_size;
  DebugCheckInvariants();
}

// Check whether index i is in the array.
template<typename Value>
bool SparseArray<Value>::has_index(int i) const {
  assert(i >= 0);
  assert(i < max_size());
  if (static_cast<uint32_t>(i) >= static_cast<uint32_t>(max_size())) {
    return false;
  }
  // Unsigned comparison avoids checking sparse_[i] < 0.
  return (uint32_t)sparse_[i] < (uint32_t)size_ &&
         dense_[sparse_[i]].index_ == i;
}

template<typename Value>
void SparseArray<Value>::create_index(int i) {
  assert(!has_index(i));
  assert(size_ < max_size());
  sparse_[i] = size_;
  dense_[size_].index_ = i;
  size_++;
}

template<typename Value> SparseArray<Value>::SparseArray(int max_size) :
    sparse_(max_size), dense_(max_size) {
  MaybeInitializeMemory(size_, max_size);
  DebugCheckInvariants();
}

template<typename Value> SparseArray<Value>::~SparseArray() {
  DebugCheckInvariants();
}

template<typename Value> void SparseArray<Value>::DebugCheckInvariants() const {
  assert(0 <= size_);
  assert(size_ <= max_size());
}

// Comparison function for sorting.
template<typename Value> bool SparseArray<Value>::less(const IndexValue& a,
                                                       const IndexValue& b) {
  return a.index_ < b.index_;
}

}  // namespace re2

#endif  // UTIL_SPARSE_ARRAY_H_
