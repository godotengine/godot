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
#ifdef MANIFOLD_USE_CUDA
#include <cuda.h>
#endif

#include "par.h"
#include "structs.h"

namespace manifold {

// Vector implementation optimized for managed memory, will perform memory
// prefetching to minimize page faults and use parallel/GPU copy/fill depending
// on data size. This will also handle builds without CUDA or builds with CUDA
// but runned without CUDA GPU properly.
//
// Note that the constructor and resize function will not perform initialization
// if the parameter val is not set. Also, this implementation is a toy
// implementation that did not consider things like non-trivial
// constructor/destructor, please keep T trivial.
template <typename T>
class ManagedVec {
 public:
  typedef T *Iter;
  typedef const T *IterC;

  ManagedVec() {
    size_ = 0;
    capacity_ = 0;
    onHost = true;
  }

  // note that we leave the memory uninitialized
  ManagedVec(size_t n) {
    size_ = n;
    capacity_ = n;
    onHost = autoPolicy(n) != ExecutionPolicy::ParUnseq;
    if (n == 0) return;
    mallocManaged(&ptr_, size_ * sizeof(T));
  }

  ManagedVec(size_t n, const T &val) {
    size_ = n;
    capacity_ = n;
    if (n == 0) return;
    auto policy = autoPolicy(n);
    onHost = policy != ExecutionPolicy::ParUnseq;
    mallocManaged(&ptr_, size_ * sizeof(T));
    prefetch(ptr_, size_ * sizeof(T), onHost);
    uninitialized_fill_n(policy, ptr_, n, val);
  }

  ~ManagedVec() {
    if (ptr_ != nullptr) freeManaged(ptr_);
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  ManagedVec(const std::vector<T> &vec) {
    size_ = vec.size();
    capacity_ = size_;
    auto policy = autoPolicy(size_);
    onHost = policy != ExecutionPolicy::ParUnseq;
    if (size_ != 0) {
      mallocManaged(&ptr_, size_ * sizeof(T));
      fastUninitializedCopy(ptr_, vec.data(), size_, policy);
    }
  }

  ManagedVec(const ManagedVec<T> &vec) {
    size_ = vec.size_;
    capacity_ = size_;
    auto policy = autoPolicy(size_);
    onHost = policy != ExecutionPolicy::ParUnseq;
    if (size_ != 0) {
      mallocManaged(&ptr_, size_ * sizeof(T));
      prefetch(ptr_, size_ * sizeof(T), onHost);
      uninitialized_copy(policy, vec.begin(), vec.end(), ptr_);
    }
  }

  ManagedVec(ManagedVec<T> &&vec) {
    ptr_ = vec.ptr_;
    size_ = vec.size_;
    capacity_ = vec.capacity_;
    onHost = vec.onHost;
    vec.ptr_ = nullptr;
    vec.size_ = 0;
    vec.capacity_ = 0;
  }

  ManagedVec &operator=(const ManagedVec<T> &vec) {
    if (&vec == this) return *this;
    if (ptr_ != nullptr) freeManaged(ptr_);
    size_ = vec.size_;
    capacity_ = vec.size_;
    auto policy = autoPolicy(size_);
    onHost = policy != ExecutionPolicy::ParUnseq;
    if (size_ != 0) {
      mallocManaged(&ptr_, size_ * sizeof(T));
      prefetch(ptr_, size_ * sizeof(T), onHost);
      uninitialized_copy(policy, vec.begin(), vec.end(), ptr_);
    }
    return *this;
  }

  ManagedVec &operator=(ManagedVec<T> &&vec) {
    if (&vec == this) return *this;
    if (ptr_ != nullptr) freeManaged(ptr_);
    onHost = vec.onHost;
    size_ = vec.size_;
    capacity_ = vec.capacity_;
    ptr_ = vec.ptr_;
    vec.ptr_ = nullptr;
    vec.size_ = 0;
    vec.capacity_ = 0;
    return *this;
  }

  void resize(size_t n) {
    reserve(n);
    size_ = n;
  }

  void resize(size_t n, const T &val) {
    reserve(n);
    if (size_ < n) {
      uninitialized_fill(autoPolicy(n - size_), ptr_ + size_, ptr_ + n, val);
    }
    size_ = n;
  }

  void reserve(size_t n) {
    if (n > capacity_) {
      T *newBuffer;
      mallocManaged(&newBuffer, n * sizeof(T));
      prefetch(newBuffer, size_ * sizeof(T), onHost);
      if (size_ > 0) {
        uninitialized_copy(autoPolicy(size_), ptr_, ptr_ + size_, newBuffer);
      }
      if (ptr_ != nullptr) freeManaged(ptr_);
      ptr_ = newBuffer;
      capacity_ = n;
    }
  }

  void shrink_to_fit() {
    T *newBuffer = nullptr;
    if (size_ > 0) {
      mallocManaged(&newBuffer, size_ * sizeof(T));
      prefetch(newBuffer, size_ * sizeof(T), onHost);
      uninitialized_copy(autoPolicy(size_), ptr_, ptr_ + size_, newBuffer);
    }
    freeManaged(ptr_);
    ptr_ = newBuffer;
    capacity_ = size_;
  }

  void push_back(const T &val) {
    if (size_ >= capacity_) {
      // avoid dangling pointer in case val is a reference of our array
      T val_copy = val;
      reserve(capacity_ == 0 ? 128 : capacity_ * 2);
      onHost = true;
      ptr_[size_++] = val_copy;
      return;
    }
    ptr_[size_++] = val;
  }

  void swap(ManagedVec<T> &other) {
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
    std::swap(onHost, other.onHost);
  }

  void prefetch_to(bool toHost) const {
    if (toHost != onHost) prefetch(ptr_, size_ * sizeof(T), toHost);
    onHost = toHost;
  }

  IterC cbegin() const { return ptr_; }

  IterC cend() const { return ptr_ + size_; }

  Iter begin() { return ptr_; }

  Iter end() { return ptr_ + size_; }

  IterC begin() const { return cbegin(); }

  IterC end() const { return cend(); }

  T *data() { return ptr_; }

  const T *data() const { return ptr_; }

  size_t size() const { return size_; }

  size_t capacity() const { return capacity_; }

  T &front() { return *ptr_; }

  const T &front() const { return *ptr_; }

  T &back() { return *(ptr_ + size_ - 1); }

  const T &back() const { return *(ptr_ + size_ - 1); }

  T &operator[](size_t i) { return *(ptr_ + i); }

  const T &operator[](size_t i) const { return *(ptr_ + i); }

  bool empty() const { return size_ == 0; }

 private:
  T *ptr_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
  mutable bool onHost = true;

  static constexpr int DEVICE_MAX_BYTES = 1 << 16;

  static void mallocManaged(T **ptr, size_t bytes) {
#ifdef MANIFOLD_USE_CUDA
    if (CUDA_ENABLED == -1) check_cuda_available();
    if (CUDA_ENABLED)
      cudaMallocManaged(reinterpret_cast<void **>(ptr), bytes);
    else
#endif
      *ptr = reinterpret_cast<T *>(malloc(bytes));
  }

  static void freeManaged(T *ptr) {
#ifdef MANIFOLD_USE_CUDA
    if (CUDA_ENABLED)
      cudaFree(ptr);
    else
#endif
      free(ptr);
  }

  static void prefetch(T *ptr, int bytes, bool onHost) {
#ifdef MANIFOLD_USE_CUDA
    if (bytes > 0 && CUDA_ENABLED)
      cudaMemPrefetchAsync(ptr, std::min(bytes, DEVICE_MAX_BYTES),
                           onHost ? cudaCpuDeviceId : 0);
#endif
  }

  // fast routine for memcpy from std::vector to ManagedVec
  static void fastUninitializedCopy(T *dst, const T *src, int n,
                                    ExecutionPolicy policy) {
    prefetch(dst, n * sizeof(T), policy != ExecutionPolicy::ParUnseq);
    switch (policy) {
      case ExecutionPolicy::ParUnseq:
#ifdef MANIFOLD_USE_CUDA
        cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
#endif
      case ExecutionPolicy::Par:
        thrust::uninitialized_copy_n(thrust::MANIFOLD_PAR_NS::par, src, n, dst);
        break;
      case ExecutionPolicy::Seq:
        thrust::uninitialized_copy_n(thrust::cpp::par, src, n, dst);
        break;
    }
  }
};

/*
 * Host and device vector implementation. This uses `thrust::universal_vector`
 * for storage, so data can be moved by the hardware on demand, allows using
 * more memory than the available GPU memory, reduce memory overhead and provide
 * speedup due to less synchronization.
 *
 * Due to https://github.com/NVIDIA/thrust/issues/1690 , `push_back` operations
 * on universal vectors are *VERY* slow, so a `std::vector` is used as a cache.
 * The cache will be created when we perform `push_back` or `reserve` operations
 * on the `VecDH`, and destroyed when we try to access device iterator/pointer.
 * For better performance, please avoid interspersing `push_back` between device
 * memory accesses, as that will cause repeated synchronization and hurts
 * performance.
 * Note that it is *NOT SAFE* to first obtain a host(device) pointer, perform
 * some device(host) modification, and then read the host(device) pointer again
 * (on the same vector). The memory will be inconsistent in that case.
 */
template <typename T>
class VecDH {
 public:
  VecDH() {}

  // Note that the vector constructed with this constructor will contain
  // uninitialized memory. Please specify `val` if you need to make sure that
  // the data is initialized.
  VecDH(int size) { impl_.resize(size); }

  VecDH(int size, T val) { impl_.resize(size, val); }

  VecDH(const std::vector<T> &vec) { impl_ = vec; }

  VecDH(const VecDH<T> &other) { impl_ = other.impl_; }

  VecDH(VecDH<T> &&other) { impl_ = std::move(other.impl_); }

  VecDH<T> &operator=(const VecDH<T> &other) {
    impl_ = other.impl_;
    return *this;
  }

  VecDH<T> &operator=(VecDH<T> &&other) {
    impl_ = std::move(other.impl_);
    return *this;
  }

  int size() const { return impl_.size(); }

  void resize(int newSize, T val = T()) {
    bool shrink = size() > 2 * newSize;
    impl_.resize(newSize, val);
    if (shrink) impl_.shrink_to_fit();
  }

  void swap(VecDH<T> &other) { impl_.swap(other.impl_); }

  using Iter = typename ManagedVec<T>::Iter;
  using IterC = typename ManagedVec<T>::IterC;

  Iter begin() {
    impl_.prefetch_to(autoPolicy(size()) != ExecutionPolicy::ParUnseq);
    return impl_.begin();
  }

  Iter end() { return impl_.end(); }

  IterC cbegin() const {
    impl_.prefetch_to(autoPolicy(size()) != ExecutionPolicy::ParUnseq);
    return impl_.cbegin();
  }

  IterC cend() const { return impl_.cend(); }

  IterC begin() const { return cbegin(); }
  IterC end() const { return cend(); }

  T *ptrD() {
    if (size() == 0) return nullptr;
    impl_.prefetch_to(autoPolicy(size()) != ExecutionPolicy::ParUnseq);
    return impl_.data();
  }

  const T *cptrD() const {
    if (size() == 0) return nullptr;
    impl_.prefetch_to(autoPolicy(size()) != ExecutionPolicy::ParUnseq);
    return impl_.data();
  }

  const T *ptrD() const { return cptrD(); }

  T *ptrH() {
    if (size() == 0) return nullptr;
    impl_.prefetch_to(true);
    return impl_.data();
  }

  const T *cptrH() const {
    if (size() == 0) return nullptr;
    impl_.prefetch_to(true);
    return impl_.data();
  }

  const T *ptrH() const { return cptrH(); }

  T &operator[](int i) {
    impl_.prefetch_to(true);
    return impl_[i];
  }

  const T &operator[](int i) const {
    impl_.prefetch_to(true);
    return impl_[i];
  }

  T &back() { return impl_.back(); }

  const T &back() const { return impl_.back(); }

  void push_back(const T &val) { impl_.push_back(val); }

  void reserve(int n) { impl_.reserve(n); }

#ifdef MANIFOLD_DEBUG
  void Dump() const {
    std::cout << "VecDH = " << std::endl;
    for (int i = 0; i < impl_.size(); ++i) {
      std::cout << i << ", " << impl_[i] << ", " << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 private:
  ManagedVec<T> impl_;
};

template <typename T>
class VecD {
 public:
  VecD(const VecDH<T> &vec) : ptr_(vec.ptrD()), size_(vec.size()) {}

  __host__ __device__ const T &operator[](int i) const { return ptr_[i]; }
  __host__ __device__ int size() const { return size_; }

 private:
  T const *const ptr_;
  const int size_;
};
/** @} */
}  // namespace manifold
