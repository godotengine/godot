// Copyright 2023 The Manifold Authors.
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

#pragma once

#include "manifold/optional_assert.h"

namespace manifold {

/**
 * View for Vec, can perform offset operation.
 * This will be invalidated when the original vector is dropped or changes
 * length. Roughly equivalent to std::span<T> from c++20
 */
template <typename T>
class VecView {
 public:
  using Iter = T *;
  using IterC = const T *;

  VecView(T *ptr, size_t size) : ptr_(ptr), size_(size) {}

  VecView(const VecView &other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
  }

  VecView &operator=(const VecView &other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    return *this;
  }

  // allows conversion to a const VecView
  operator VecView<const T>() const { return {ptr_, size_}; }

  inline const T &operator[](size_t i) const {
    ASSERT(i < size_, std::out_of_range("Vec out of range"));
    return ptr_[i];
  }

  inline T &operator[](size_t i) {
    ASSERT(i < size_, std::out_of_range("Vec out of range"));
    return ptr_[i];
  }

  IterC cbegin() const { return ptr_; }
  IterC cend() const { return ptr_ + size_; }

  IterC begin() const { return cbegin(); }
  IterC end() const { return cend(); }

  Iter begin() { return ptr_; }
  Iter end() { return ptr_ + size_; }

  const T &front() const {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the front of an empty vector"));
    return ptr_[0];
  }

  const T &back() const {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the back of an empty vector"));
    return ptr_[size_ - 1];
  }

  T &front() {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the front of an empty vector"));
    return ptr_[0];
  }

  T &back() {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the back of an empty vector"));
    return ptr_[size_ - 1];
  }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

#ifdef MANIFOLD_DEBUG
  void Dump() const {
    std::cout << "Vec = " << std::endl;
    for (size_t i = 0; i < size(); ++i) {
      std::cout << i << ", " << ptr_[i] << ", " << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 protected:
  T *ptr_ = nullptr;
  size_t size_ = 0;

  VecView() = default;
};

}  // namespace manifold
