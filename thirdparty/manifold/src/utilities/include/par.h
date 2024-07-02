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
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cpp/execution_policy.h>
#include <thrust/uninitialized_copy.h>

#include <algorithm>
#include <numeric>
#if MANIFOLD_PAR == 'T'
#include <thrust/system/tbb/execution_policy.h>

#if MANIFOLD_PAR == 'T' && TBB_INTERFACE_VERSION >= 10000 && \
    __has_include(<pstl/glue_execution_defs.h>)
#include <execution>
#endif

#include "tbb/tbb.h"
#define MANIFOLD_PAR_NS tbb
#else
#define MANIFOLD_PAR_NS cpp
#endif

namespace manifold {

enum class ExecutionPolicy {
  Par,
  Seq,
};

// ExecutionPolicy:
// - Sequential for small workload,
// - Parallel (CPU) for medium workload,
// - GPU for large workload if available.
inline constexpr ExecutionPolicy autoPolicy(size_t size) {
  // some random numbers
  if (size <= (1 << 12)) {
    return ExecutionPolicy::Seq;
  }
  return ExecutionPolicy::Par;
}

#define THRUST_DYNAMIC_BACKEND_VOID(NAME)                    \
  template <typename... Args>                                \
  void NAME(ExecutionPolicy policy, Args... args) {          \
    switch (policy) {                                        \
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
      case ExecutionPolicy::Par:                                    \
        return thrust::NAME(thrust::MANIFOLD_PAR_NS::par, args...); \
      case ExecutionPolicy::Seq:                                    \
        break;                                                      \
    }                                                               \
    return thrust::NAME(thrust::cpp::par, args...);                 \
  }

#if MANIFOLD_PAR != 'T' || \
    (TBB_INTERFACE_VERSION >= 10000 && __has_include(<pstl/glue_execution_defs.h>))
#if MANIFOLD_PAR == 'T'
#define STL_DYNAMIC_BACKEND(NAME, RET)                        \
  template <typename Ret = RET, typename... Args>             \
  Ret NAME(ExecutionPolicy policy, Args... args) {            \
    switch (policy) {                                         \
      case ExecutionPolicy::Par:                              \
        return std::NAME(std::execution::par_unseq, args...); \
      case ExecutionPolicy::Seq:                              \
        break;                                                \
    }                                                         \
    return std::NAME(args...);                                \
  }
#define STL_DYNAMIC_BACKEND_VOID(NAME)                 \
  template <typename... Args>                          \
  void NAME(ExecutionPolicy policy, Args... args) {    \
    switch (policy) {                                  \
      case ExecutionPolicy::Par:                       \
        std::NAME(std::execution::par_unseq, args...); \
        break;                                         \
      case ExecutionPolicy::Seq:                       \
        std::NAME(args...);                            \
        break;                                         \
    }                                                  \
  }
#else
#define STL_DYNAMIC_BACKEND(NAME, RET)             \
  template <typename Ret = RET, typename... Args>  \
  Ret NAME(ExecutionPolicy policy, Args... args) { \
    return std::NAME(args...);                     \
  }
#define STL_DYNAMIC_BACKEND_VOID(NAME)              \
  template <typename... Args>                       \
  void NAME(ExecutionPolicy policy, Args... args) { \
    std::NAME(args...);                             \
  }
#endif

template <typename... Args>
void exclusive_scan(ExecutionPolicy policy, Args... args) {
  // https://github.com/llvm/llvm-project/issues/59810
  std::exclusive_scan(args...);
}
template <typename DerivedPolicy, typename InputIterator1,
          typename InputIterator2, typename OutputIterator, typename Predicate>
OutputIterator copy_if(ExecutionPolicy policy, InputIterator1 first,
                       InputIterator1 last, InputIterator2 stencil,
                       OutputIterator result, Predicate pred) {
  if (policy == ExecutionPolicy::Seq)
    return thrust::copy_if(thrust::cpp::par, first, last, stencil, result,
                           pred);
  else
    // note: this is not a typo, see
    // https://github.com/NVIDIA/thrust/issues/1977
    return thrust::copy_if(first, last, stencil, result, pred);
}
template <typename DerivedPolicy, typename InputIterator1,
          typename OutputIterator, typename Predicate>
OutputIterator copy_if(ExecutionPolicy policy, InputIterator1 first,
                       InputIterator1 last, OutputIterator result,
                       Predicate pred) {
#if MANIFOLD_PAR == 'T'
  if (policy == ExecutionPolicy::Seq)
    return std::copy_if(first, last, result, pred);
  else
    return std::copy_if(std::execution::par_unseq, first, last, result, pred);
#else
  return std::copy_if(first, last, result, pred);
#endif
}

#else
#define STL_DYNAMIC_BACKEND(NAME, RET) THRUST_DYNAMIC_BACKEND(NAME, RET)
#define STL_DYNAMIC_BACKEND_VOID(NAME) THRUST_DYNAMIC_BACKEND_VOID(NAME)

THRUST_DYNAMIC_BACKEND_VOID(exclusive_scan)
THRUST_DYNAMIC_BACKEND(copy_if, void)
#endif

THRUST_DYNAMIC_BACKEND_VOID(gather)
THRUST_DYNAMIC_BACKEND_VOID(scatter)
THRUST_DYNAMIC_BACKEND_VOID(for_each)
THRUST_DYNAMIC_BACKEND_VOID(for_each_n)
THRUST_DYNAMIC_BACKEND_VOID(sequence)
STL_DYNAMIC_BACKEND_VOID(transform)
STL_DYNAMIC_BACKEND_VOID(uninitialized_fill)
STL_DYNAMIC_BACKEND_VOID(uninitialized_copy)
STL_DYNAMIC_BACKEND_VOID(stable_sort)
STL_DYNAMIC_BACKEND_VOID(fill)
STL_DYNAMIC_BACKEND_VOID(copy)
STL_DYNAMIC_BACKEND_VOID(inclusive_scan)
STL_DYNAMIC_BACKEND_VOID(copy_n)

// void implies that the user have to specify the return type in the template
// argument, as we are unable to deduce it
THRUST_DYNAMIC_BACKEND(transform_reduce, void)
THRUST_DYNAMIC_BACKEND(gather_if, void)
THRUST_DYNAMIC_BACKEND(reduce_by_key, void)
STL_DYNAMIC_BACKEND(remove, void)
STL_DYNAMIC_BACKEND(find, void)
STL_DYNAMIC_BACKEND(find_if, void)
STL_DYNAMIC_BACKEND(all_of, bool)
STL_DYNAMIC_BACKEND(is_sorted, bool)
STL_DYNAMIC_BACKEND(reduce, void)
STL_DYNAMIC_BACKEND(count_if, int)
STL_DYNAMIC_BACKEND(remove_if, void)

}  // namespace manifold
