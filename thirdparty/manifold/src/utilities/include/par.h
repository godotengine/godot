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
#include <../third_party/thrust/thrust/binary_search.h>
#include <../third_party/thrust/thrust/count.h>
#include <../third_party/thrust/thrust/execution_policy.h>
#include <../third_party/thrust/thrust/gather.h>
#include <../third_party/thrust/thrust/logical.h>
#include <../third_party/thrust/thrust/remove.h>
#include <../third_party/thrust/thrust/sequence.h>
#include <../third_party/thrust/thrust/sort.h>
#include <../third_party/thrust/thrust/system/cpp/execution_policy.h>
#include <../third_party/thrust/thrust/uninitialized_copy.h>
#include <../third_party/thrust/thrust/unique.h>

#if MANIFOLD_PAR == 'O'
#include <../third_party/thrust/thrust/system/omp/execution_policy.h>
#define MANIFOLD_PAR_NS omp
#elif MANIFOLD_PAR == 'T'
#include <../third_party/thrust/thrust/system/tbb/execution_policy.h>
#define MANIFOLD_PAR_NS tbb
#else
#define MANIFOLD_PAR_NS cpp
#endif

#ifdef MANIFOLD_USE_CUDA
#include <../third_party/thrust/thrust/system/cuda/execution_policy.h>
#endif

namespace manifold {

void check_cuda_available();
#ifdef MANIFOLD_USE_CUDA
extern int CUDA_ENABLED;
#else
constexpr int CUDA_ENABLED = 0;
#endif

enum class ExecutionPolicy {
  ParUnseq,
  Par,
  Seq,
};

constexpr ExecutionPolicy ParUnseq = ExecutionPolicy::ParUnseq;
constexpr ExecutionPolicy Par = ExecutionPolicy::Par;
constexpr ExecutionPolicy Seq = ExecutionPolicy::Seq;

// ExecutionPolicy:
// - Sequential for small workload,
// - Parallel (CPU) for medium workload,
// - GPU for large workload if available.
inline ExecutionPolicy autoPolicy(int size) {
  // some random numbers
  if (size <= (1 << 12)) {
    return Seq;
  }
  if (size <= (1 << 16) || CUDA_ENABLED != 1) {
    return Par;
  }
  return ParUnseq;
}

#ifdef MANIFOLD_USE_CUDA
#define THRUST_DYNAMIC_BACKEND_VOID(NAME)                    \
  template <typename... Args>                                \
  void NAME(ExecutionPolicy policy, Args... args) {          \
    switch (policy) {                                        \
      case ExecutionPolicy::ParUnseq:                        \
        thrust::NAME(thrust::cuda::par, args...);            \
        break;                                               \
      case ExecutionPolicy::Par:                             \
        thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...); \
        break;                                               \
      case ExecutionPolicy::Seq:                             \
        thrust::NAME(thrust::cpp::par, args...);             \
        break;                                               \
    }                                                        \
  }
#define THRUST_DYNAMIC_BACKEND(NAME, RET)                           \
  template <typename Ret = RET, typename... Args>                   \
  Ret NAME(ExecutionPolicy policy, Args... args) {                  \
    switch (policy) {                                               \
      case ExecutionPolicy::ParUnseq:                               \
        return thrust::NAME(thrust::cuda::par, args...);            \
      case ExecutionPolicy::Par:                                    \
        return thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...); \
      case ExecutionPolicy::Seq:                                    \
        break;                                                      \
    }                                                               \
    return thrust::NAME(thrust::cpp::par, args...);                 \
  }
#else
#define THRUST_DYNAMIC_BACKEND_VOID(NAME)                    \
  template <typename... Args>                                \
  void NAME(ExecutionPolicy policy, Args... args) {          \
    switch (policy) {                                        \
      case ExecutionPolicy::ParUnseq:                        \
      case ExecutionPolicy::Par:                             \
        thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...); \
        break;                                               \
      case ExecutionPolicy::Seq:                             \
        thrust::NAME(thrust::cpp::par, args...);             \
        break;                                               \
    }                                                        \
  }

#define THRUST_DYNAMIC_BACKEND(NAME, RET)                           \
  template <typename Ret = RET, typename... Args>                   \
  Ret NAME(ExecutionPolicy policy, Args... args) {                  \
    switch (policy) {                                               \
      case ExecutionPolicy::ParUnseq:                               \
      case ExecutionPolicy::Par:                                    \
        return thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...); \
      case ExecutionPolicy::Seq:                                    \
        break;                                                      \
    }                                                               \
    return thrust::NAME(thrust::cpp::par, args...);                 \
  }
#endif

THRUST_DYNAMIC_BACKEND_VOID(gather)
THRUST_DYNAMIC_BACKEND_VOID(scatter)
THRUST_DYNAMIC_BACKEND_VOID(for_each)
THRUST_DYNAMIC_BACKEND_VOID(for_each_n)
THRUST_DYNAMIC_BACKEND_VOID(sort)
THRUST_DYNAMIC_BACKEND_VOID(stable_sort)
THRUST_DYNAMIC_BACKEND_VOID(fill)
THRUST_DYNAMIC_BACKEND_VOID(sequence)
THRUST_DYNAMIC_BACKEND_VOID(sort_by_key)
THRUST_DYNAMIC_BACKEND_VOID(stable_sort_by_key)
THRUST_DYNAMIC_BACKEND_VOID(copy)
THRUST_DYNAMIC_BACKEND_VOID(transform)
THRUST_DYNAMIC_BACKEND_VOID(inclusive_scan)
THRUST_DYNAMIC_BACKEND_VOID(exclusive_scan)
THRUST_DYNAMIC_BACKEND_VOID(uninitialized_fill)
THRUST_DYNAMIC_BACKEND_VOID(uninitialized_copy)
THRUST_DYNAMIC_BACKEND_VOID(copy_n)

THRUST_DYNAMIC_BACKEND(all_of, bool)
THRUST_DYNAMIC_BACKEND(is_sorted, bool)
THRUST_DYNAMIC_BACKEND(reduce, void)
THRUST_DYNAMIC_BACKEND(count_if, int)
THRUST_DYNAMIC_BACKEND(binary_search, bool)
// void implies that the user have to specify the return type in the template
// argument, as we are unable to deduce it
THRUST_DYNAMIC_BACKEND(remove, void)
THRUST_DYNAMIC_BACKEND(copy_if, void)
THRUST_DYNAMIC_BACKEND(remove_if, void)
THRUST_DYNAMIC_BACKEND(unique, void)
THRUST_DYNAMIC_BACKEND(find, void)
THRUST_DYNAMIC_BACKEND(find_if, void)
THRUST_DYNAMIC_BACKEND(reduce_by_key, void)
THRUST_DYNAMIC_BACKEND(transform_reduce, void)
THRUST_DYNAMIC_BACKEND(lower_bound, void)
THRUST_DYNAMIC_BACKEND(gather_if, void)

}  // namespace manifold
