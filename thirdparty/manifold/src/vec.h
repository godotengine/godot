// Copyright 2021 The Manifold Authors.
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
#if TRACY_ENABLE && TRACY_MEMORY_USAGE
#include "tracy/Tracy.hpp"
#else
#define TracyAllocS(ptr, size, n) (void)0
#define TracyFreeS(ptr, n) (void)0
#endif

#if __has_include(<tracy/Tracy.hpp>)
#include <tracy/Tracy.hpp>
#else
#define FrameMarkStart(x)
#define FrameMarkEnd(x)
// putting ZoneScoped in a function will instrument the function execution when
// TRACY_ENABLE is set, which allows the profiler to record more accurate
// timing.
#define ZoneScoped
#define ZoneScopedN(name)
#endif
#include <vector>

#include "manifold/vec_view.h"
#include "parallel.h"

namespace manifold {

#if (MANIFOLD_PAR == 1)
extern tbb::task_arena gc_arena;
#endif

struct Empty {};

template <typename T, bool shared = false>
class Vec;

// Shareable vector that forbids structural update when shared.
// Note that if you do `Vec<T>& vec = sharedVec`, this will perform a copy,
// so please do `SharedVec<T>& vec = shareVec` instead.
//
// The copy constructor and copy assignment operator will share the
// underlying vector.
template <typename T>
using SharedVec = Vec<T, true>;

/*
 * Specialized vector implementation with multithreaded fill and uninitialized
 * memory optimizations.
 * Note that the constructor and resize function will not perform initialization
 * if the parameter val is not set. Also, this implementation is a toy
 * implementation that did not consider things like non-trivial
 * constructor/destructor, please keep T trivial.
 */
template <typename T, bool shared>
class Vec : public VecView<T> {
 public:
  Vec() : VecView<T>() {
    if constexpr (shared) count_ = new std::atomic<int>(1);
  }

  // Note that the vector constructed with this constructor will contain
  // uninitialized memory. Please specify `val` if you need to make sure that
  // the data is initialized.
  inline Vec(size_t size) : VecView<T>() {
    ZoneScopedN("Vec(size)");
    if constexpr (shared) count_ = new std::atomic<int>(1);
    reserve(size);
    this->size_ = size;
  }

  inline Vec(size_t size, T val) : VecView<T>() {
    ZoneScopedN("Vec(size, val)");
    if constexpr (shared) count_ = new std::atomic<int>(1);
    resize(size, val);
  }

  // manually specialized because copy constructor and copy assignment operators
  // cannot be template methods
  inline Vec(const Vec<T, true>& vec) : VecView<T>() {
    ZoneScopedN("Vec(const vec&)");
    // initialized with operator=
    if constexpr (shared) count_ = nullptr;
    *this = Vec(vec.view());
  }

  inline Vec(const Vec<T, false>& vec) : VecView<T>() {
    ZoneScopedN("Vec(const vec&)");
    // initialized with operator=
    if constexpr (shared) count_ = nullptr;
    *this = Vec(vec.view());
  }

  inline Vec(Vec<T, true>&& other) : VecView<T>() {
    if constexpr (shared) {
      this->count_ = other.count_;
      other.count_ = nullptr;
    }
    moveContent(other);
  }

  inline Vec(Vec<T, false>&& other) : VecView<T>() {
    if constexpr (shared) this->count_ = new std::atomic<int>(1);
    moveContent(other);
  }

  inline Vec(const VecView<const T>& vec) : VecView<T>() {
    ZoneScopedN("Vec(const VecView&)");
    if constexpr (shared) this->count_ = new std::atomic<int>(1);
    this->size_ = vec.size();
    this->capacity_ = this->size_;
    auto policy = autoPolicy(this->size_);
    if (this->size_ != 0) {
      this->ptr_ = reinterpret_cast<T*>(malloc(this->size_ * sizeof(T)));
      ASSERT(this->ptr_ != nullptr, std::bad_alloc());
      TracyAllocS(this->ptr_, this->size_ * sizeof(T), 3);
      copy(policy, vec.begin(), vec.end(), this->ptr_);
    }
  }

  inline Vec(const std::vector<T>& vec)
      : Vec<T>(VecView<const T>(vec.data(), vec.size())) {}

  inline operator VecView<T>() { return {this->ptr_, this->size_}; }
  inline operator VecView<T>() const { return {this->ptr_, this->size_}; }

  ~Vec() { dealloc(); }

  Vec& operator=(const Vec& other) {
    ZoneScopedN("Vec operator=(other)");
    if (&other == this) return *this;
    dealloc();
    if constexpr (shared) {
      other.count_->fetch_add(1);
      this->count_ = other.count_;
      this->ptr_ = other.ptr_;
      this->size_ = other.size_;
      this->capacity_ = other.capacity_;
      return *this;
    }
    this->size_ = other.size_;
    this->capacity_ = other.size_;
    if (this->size_ != 0) {
      this->ptr_ = reinterpret_cast<T*>(malloc(this->size_ * sizeof(T)));
      ASSERT(this->ptr_ != nullptr, std::bad_alloc());
      TracyAllocS(this->ptr_, this->size_ * sizeof(T), 3);
      manifold::copy(other.begin(), other.end(), this->ptr_);
    }
    return *this;
  }

  inline Vec<T, shared>& operator=(const Vec<T, !shared>& other) {
    ZoneScopedN("Vec operator=(other)");
    dealloc();
    if constexpr (shared) this->count_ = new std::atomic<int>(1);
    this->size_ = other.size_;
    this->capacity_ = other.size_;
    if (this->size_ != 0) {
      this->ptr_ = reinterpret_cast<T*>(malloc(this->size_ * sizeof(T)));
      ASSERT(this->ptr_ != nullptr, std::bad_alloc());
      TracyAllocS(this->ptr_, this->size_ * sizeof(T), 3);
      manifold::copy(other.begin(), other.end(), this->ptr_);
    }
    return *this;
  }

  inline Vec& operator=(Vec&& other) {
    if (&other == this) return *this;
    dealloc();
    if constexpr (shared) {
      this->count_ = other.count_;
      other.count_ = nullptr;
    }
    moveContent(other);
    return *this;
  }

  inline Vec<T, shared>& operator=(Vec<T, !shared>&& other) {
    dealloc();
    if constexpr (shared) this->count_ = new std::atomic<int>(1);
    moveContent(other);
    return *this;
  }

  template <bool shared2>
  void swap(Vec<T, shared2>& other) {
    AssertUnique();
    if constexpr (shared2) other.AssertUnique();
    std::swap(this->ptr_, other.ptr_);
    std::swap(this->size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

  inline void push_back(const T& val) {
    AssertUnique();
    if (this->size_ >= capacity_) {
      // avoid dangling pointer in case val is a reference of our array
      T val_copy = val;
      reserve(capacity_ == 0 ? 128 : capacity_ * 2);
      this->ptr_[this->size_++] = val_copy;
      return;
    }
    this->ptr_[this->size_++] = val;
  }

  void extend(size_t n) {
    AssertUnique();
    if (this->size_ + n >= capacity_)
      reserve(capacity_ == 0 ? 128 : std::max(capacity_ * 2, this->size_ + n));
    this->size_ += n;
  }

  void reserve(size_t n) {
    AssertUnique();
    if (n > capacity_) {
      T* newBuffer = reinterpret_cast<T*>(malloc(n * sizeof(T)));
      ASSERT(newBuffer != nullptr, std::bad_alloc());
      TracyAllocS(newBuffer, n * sizeof(T), 3);
      if (this->size_ > 0)
        manifold::copy(autoPolicy(this->size_), this->ptr_,
                       this->ptr_ + this->size_, newBuffer);
      if (this->ptr_ != nullptr) {
        free_async(this->ptr_, capacity_);
      }
      this->ptr_ = newBuffer;
      capacity_ = n;
    }
  }

  void resize(size_t newSize, T val = T()) {
    AssertUnique();
    bool shrink = this->size_ > 2 * newSize && this->size_ > 16;
    if (this->size_ < newSize) {
      reserve(newSize);
      fill(autoPolicy(newSize - this->size_), this->ptr_ + this->size_,
           this->ptr_ + newSize, val);
    }
    this->size_ = newSize;
    if (shrink) shrink_to_fit();
  }

  inline void resize_nofill(size_t newSize) {
    AssertUnique();
    bool shrink = this->size_ > 2 * newSize && this->size_ > 16;
    reserve(newSize);
    this->size_ = newSize;
    if (shrink) shrink_to_fit();
  }

  inline void pop_back() {
    AssertUnique();
    resize_nofill(this->size_ - 1);
  }

  void clear(bool shrink = true) {
    AssertUnique();
    this->size_ = 0;
    if (shrink) shrink_to_fit();
  }

  void shrink_to_fit() {
    AssertUnique();
    T* newBuffer = nullptr;
    if (this->size_ > 0) {
      newBuffer = reinterpret_cast<T*>(malloc(this->size_ * sizeof(T)));
      ASSERT(newBuffer != nullptr, std::bad_alloc());
      TracyAllocS(newBuffer, this->size_ * sizeof(T), 3);
      manifold::copy(this->ptr_, this->ptr_ + this->size_, newBuffer);
    }
    if (this->ptr_ != nullptr) {
      free_async(this->ptr_, capacity_);
    }
    this->ptr_ = newBuffer;
    capacity_ = this->size_;
  }

  inline size_t capacity() const { return capacity_; }

  // Makes the SharedVec unique, so mutation operations are allowed.
  // Note that this does nothing if the vector is already unique.
  void MakeUnique() {
    if constexpr (shared) {
      if (count_->load() > 1) {
        *this = Vec<T, true>(this->view());
      }
    }
  }

 private:
  size_t capacity_ = 0;

  // Normally empty structs have size 1 (1 byte) to ensure every field has a
  // unique address, but in c++20 no_unique_address can avoid this and the field
  // takes no space.
#if __cplusplus >= 202002L
  [[no_unique_address]]
#endif
  std::conditional_t<shared, std::atomic<int>*, Empty> count_;

  inline void AssertUnique() const {
    if constexpr (shared) {
      ASSERT(count_->load() == 1, logicErr("can only mutate unique vector"));
    }
  }

  inline void dealloc() {
    if constexpr (shared) {
      if (count_ == nullptr || count_->fetch_sub(1) > 1) return;
      delete count_;
      this->count_ = nullptr;
    }
    if (this->ptr_ != nullptr) {
      Vec<T, false>::free_async(this->ptr_, this->capacity_);
    }
    this->ptr_ = nullptr;
    this->size_ = 0;
    this->capacity_ = 0;
  }

  template <typename V>
  inline void moveContent(V&& other) {
    this->ptr_ = other.ptr_;
    this->size_ = other.size_;
    this->capacity_ = other.capacity_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  static_assert(std::is_trivially_destructible<T>::value);
  // This is required so we can access private fields between shared and
  // non-shared vectors.
  friend Vec<T, !shared>;

  static void free_async(T* ptr, size_t size) {
    // Only do async free if the size is large, because otherwise we may be able
    // to reuse the allocation, and the deallocation probably won't trigger
    // munmap.
    // Currently it is set to 64 pages (4kB page).
    constexpr size_t ASYNC_FREE_THRESHOLD = 1 << 18;
    TracyFreeS(ptr, 3);
#if defined(__has_feature)
#if !__has_feature(address_sanitizer)
#if (MANIFOLD_PAR == 1)
    if (size * sizeof(T) > ASYNC_FREE_THRESHOLD)
      gc_arena.enqueue([ptr]() { free(ptr); });
    else
#endif
#endif
#endif
      free(ptr);
  }
};
}  // namespace manifold
