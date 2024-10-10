// Copyright 2020 The Manifold Authors, Jared Hoberock and Nathan Bell of
// NVIDIA Research
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
#include <atomic>
#include <mutex>
#include <unordered_map>

#ifdef MANIFOLD_DEBUG
#include <chrono>
#include <iostream>
#endif

#include "manifold/parallel.h"
#include "manifold/vec.h"

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

namespace manifold {

/** @defgroup Private
 *  @brief Internal classes of the library; not currently part of the public API
 *  @{
 */
#ifdef MANIFOLD_DEBUG
struct Timer {
  std::chrono::high_resolution_clock::time_point start, end;

  void Start() { start = std::chrono::high_resolution_clock::now(); }

  void Stop() { end = std::chrono::high_resolution_clock::now(); }

  float Elapsed() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
  }
  void Print(std::string message) {
    std::cout << "----------- " << std::round(Elapsed()) << " ms for "
              << message << std::endl;
  }
};
#endif

inline int Next3(int i) {
  constexpr ivec3 next3(1, 2, 0);
  return next3[i];
}

inline int Prev3(int i) {
  constexpr ivec3 prev3(2, 0, 1);
  return prev3[i];
}

template <typename T, typename T1>
void Permute(Vec<T>& inOut, const Vec<T1>& new2Old) {
  Vec<T> tmp(std::move(inOut));
  inOut.resize(new2Old.size());
  gather(new2Old.begin(), new2Old.end(), tmp.begin(), inOut.begin());
}

template <typename T, typename T1>
void Permute(std::vector<T>& inOut, const Vec<T1>& new2Old) {
  std::vector<T> tmp(std::move(inOut));
  inOut.resize(new2Old.size());
  gather(new2Old.begin(), new2Old.end(), tmp.begin(), inOut.begin());
}

template <typename T>
T AtomicAdd(T& target, T add) {
  std::atomic<T>& tar = reinterpret_cast<std::atomic<T>&>(target);
  T old_val = tar.load();
  while (!tar.compare_exchange_weak(old_val, old_val + add,
                                    std::memory_order_seq_cst)) {
  }
  return old_val;
}

template <>
inline int AtomicAdd(int& target, int add) {
  std::atomic<int>& tar = reinterpret_cast<std::atomic<int>&>(target);
  int old_val = tar.fetch_add(add, std::memory_order_seq_cst);
  return old_val;
}

template <typename T>
class ConcurrentSharedPtr {
 public:
  ConcurrentSharedPtr(T value) : impl(std::make_shared<T>(value)) {}
  ConcurrentSharedPtr(const ConcurrentSharedPtr<T>& other)
      : impl(other.impl), mutex(other.mutex) {}
  class SharedPtrGuard {
   public:
    SharedPtrGuard(std::recursive_mutex* mutex, T* content)
        : mutex(mutex), content(content) {
      mutex->lock();
    }
    ~SharedPtrGuard() { mutex->unlock(); }

    T& operator*() { return *content; }
    T* operator->() { return content; }

   private:
    std::recursive_mutex* mutex;
    T* content;
  };
  SharedPtrGuard GetGuard() { return SharedPtrGuard(mutex.get(), impl.get()); };
  unsigned int UseCount() { return impl.use_count(); };

 private:
  std::shared_ptr<T> impl;
  std::shared_ptr<std::recursive_mutex> mutex =
      std::make_shared<std::recursive_mutex>();
};

template <typename I = int, typename R = unsigned char>
struct UnionFind {
  Vec<I> parents;
  // we do union by rank
  // note that we shift rank by 1, rank 0 means it is not connected to anything
  // else
  Vec<R> ranks;

  UnionFind(I numNodes) : parents(numNodes), ranks(numNodes, 0) {
    sequence(parents.begin(), parents.end());
  }

  I find(I x) {
    while (parents[x] != x) {
      parents[x] = parents[parents[x]];
      x = parents[x];
    }
    return x;
  }

  void unionXY(I x, I y) {
    if (x == y) return;
    if (ranks[x] == 0) ranks[x] = 1;
    if (ranks[y] == 0) ranks[y] = 1;
    x = find(x);
    y = find(y);
    if (x == y) return;
    if (ranks[x] < ranks[y]) std::swap(x, y);
    if (ranks[x] == ranks[y]) ranks[x]++;
    parents[y] = x;
  }

  I connectedComponents(std::vector<I>& components) {
    components.resize(parents.size());
    I lonelyNodes = 0;
    std::unordered_map<I, I> toLabel;
    for (size_t i = 0; i < parents.size(); ++i) {
      // we optimize for connected component of size 1
      // no need to put them into the hashmap
      if (ranks[i] == 0) {
        components[i] = static_cast<I>(toLabel.size()) + lonelyNodes++;
        continue;
      }
      parents[i] = find(i);
      auto iter = toLabel.find(parents[i]);
      if (iter == toLabel.end()) {
        I s = static_cast<I>(toLabel.size()) + lonelyNodes;
        toLabel.insert(std::make_pair(parents[i], s));
        components[i] = s;
      } else {
        components[i] = iter->second;
      }
    }
    return toLabel.size() + lonelyNodes;
  }
};

template <typename T>
struct Identity {
  T operator()(T v) const { return v; }
};

template <typename T>
struct Negate {
  T operator()(T v) const { return -v; }
};

/** @} */
}  // namespace manifold
