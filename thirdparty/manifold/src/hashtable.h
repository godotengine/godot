// Copyright 2022 The Manifold Authors.
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
#include <stdint.h>

#include <atomic>

#include "utils.h"
#include "vec.h"

namespace {
using hash_fun_t = uint64_t(uint64_t);
inline constexpr uint64_t kOpen = std::numeric_limits<uint64_t>::max();

template <typename T>
T AtomicCAS(T& target, T compare, T val) {
  std::atomic<T>& tar = reinterpret_cast<std::atomic<T>&>(target);
  tar.compare_exchange_strong(compare, val, std::memory_order_acq_rel);
  return compare;
}

template <typename T>
void AtomicStore(T& target, T val) {
  std::atomic<T>& tar = reinterpret_cast<std::atomic<T>&>(target);
  // release is good enough, although not really something general
  tar.store(val, std::memory_order_release);
}

template <typename T>
T AtomicLoad(const T& target) {
  const std::atomic<T>& tar = reinterpret_cast<const std::atomic<T>&>(target);
  // acquire is good enough, although not general
  return tar.load(std::memory_order_acquire);
}

}  // namespace

namespace manifold {

template <typename V, hash_fun_t H = hash64bit>
class HashTableD {
 public:
  HashTableD(Vec<uint64_t>& keys, Vec<V>& values, std::atomic<size_t>& used,
             uint32_t step = 1)
      : step_{step}, keys_{keys}, values_{values}, used_{used} {}

  int Size() const { return keys_.size(); }

  bool Full() const {
    return used_.load(std::memory_order_relaxed) * 2 >
           static_cast<size_t>(Size());
  }

  void Insert(uint64_t key, const V& val) {
    uint32_t idx = H(key) & (Size() - 1);
    while (1) {
      if (Full()) return;
      uint64_t& k = keys_[idx];
      const uint64_t found = AtomicCAS(k, kOpen, key);
      if (found == kOpen) {
        used_.fetch_add(1, std::memory_order_relaxed);
        values_[idx] = val;
        return;
      }
      if (found == key) return;
      idx = (idx + step_) & (Size() - 1);
    }
  }

  V& operator[](uint64_t key) {
    uint32_t idx = H(key) & (Size() - 1);
    while (1) {
      const uint64_t k = AtomicLoad(keys_[idx]);
      if (k == key || k == kOpen) {
        return values_[idx];
      }
      idx = (idx + step_) & (Size() - 1);
    }
  }

  const V& operator[](uint64_t key) const {
    uint32_t idx = H(key) & (Size() - 1);
    while (1) {
      const uint64_t k = AtomicLoad(keys_[idx]);
      if (k == key || k == kOpen) {
        return values_[idx];
      }
      idx = (idx + step_) & (Size() - 1);
    }
  }

  uint64_t KeyAt(int idx) const { return AtomicLoad(keys_[idx]); }
  V& At(int idx) { return values_[idx]; }
  const V& At(int idx) const { return values_[idx]; }

 private:
  uint32_t step_;
  VecView<uint64_t> keys_;
  VecView<V> values_;
  std::atomic<size_t>& used_;
};

template <typename V, hash_fun_t H = hash64bit>
class HashTable {
 public:
  HashTable(size_t size, uint32_t step = 1)
      : keys_{size == 0 ? 0 : 1_uz << (int)ceil(log2(size)), kOpen},
        values_{size == 0 ? 0 : 1_uz << (int)ceil(log2(size)), {}},
        step_(step) {}

  HashTable(const HashTable& other)
      : keys_(other.keys_), values_(other.values_), step_(other.step_) {
    used_.store(other.used_.load());
  }

  HashTable& operator=(const HashTable& other) {
    if (this == &other) return *this;
    keys_ = other.keys_;
    values_ = other.values_;
    used_.store(other.used_.load());
    step_ = other.step_;
    return *this;
  }

  HashTableD<V, H> D() { return {keys_, values_, used_, step_}; }

  int Entries() const { return used_.load(std::memory_order_relaxed); }

  size_t Size() const { return keys_.size(); }

  bool Full() const {
    return used_.load(std::memory_order_relaxed) * 2 > Size();
  }

  double FilledFraction() const {
    return static_cast<double>(used_.load(std::memory_order_relaxed)) / Size();
  }

  Vec<V>& GetValueStore() { return values_; }

  static uint64_t Open() { return kOpen; }

 private:
  Vec<uint64_t> keys_;
  Vec<V> values_;
  std::atomic<size_t> used_ = 0;
  uint32_t step_;
};
}  // namespace manifold
