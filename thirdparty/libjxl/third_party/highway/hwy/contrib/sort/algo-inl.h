// Copyright 2021 Google LLC
// SPDX-License-Identifier: Apache-2.0
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

// Normal include guard for target-independent parts
#ifndef HIGHWAY_HWY_CONTRIB_SORT_ALGO_INL_H_
#define HIGHWAY_HWY_CONTRIB_SORT_ALGO_INL_H_

#include <stdint.h>

#include <algorithm>   // std::sort, std::min, std::max
#include <functional>  // std::less, std::greater
#include <vector>

#include "hwy/base.h"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/print.h"

// Third-party algorithms
#define HAVE_AVX2SORT 0
#define HAVE_IPS4O 0
// When enabling, consider changing max_threads (required for Table 1a)
#define HAVE_PARALLEL_IPS4O (HAVE_IPS4O && 1)
#define HAVE_PDQSORT 0
#define HAVE_SORT512 0
#define HAVE_VXSORT 0
#if HWY_ARCH_X86
#define HAVE_INTEL 0
#else
#define HAVE_INTEL 0
#endif

#if HAVE_PARALLEL_IPS4O
#include <thread>  // NOLINT
#endif

#if HAVE_AVX2SORT
HWY_PUSH_ATTRIBUTES("avx2,avx")
#include "avx2sort.h"  //NOLINT
HWY_POP_ATTRIBUTES
#endif
#if HAVE_IPS4O || HAVE_PARALLEL_IPS4O
#include "third_party/ips4o/include/ips4o.hpp"
#include "third_party/ips4o/include/ips4o/thread_pool.hpp"
#endif
#if HAVE_PDQSORT
#include "third_party/boost/allowed/sort/sort.hpp"
#endif
#if HAVE_SORT512
#include "sort512.h"  //NOLINT
#endif

// vxsort is difficult to compile for multiple targets because it also uses
// .cpp files, and we'd also have to #undef its include guards. Instead, compile
// only for AVX2 or AVX3 depending on this macro.
#define VXSORT_AVX3 1
#if HAVE_VXSORT
// inlined from vxsort_targets_enable_avx512 (must close before end of header)
#ifdef __GNUC__
#ifdef __clang__
#if VXSORT_AVX3
#pragma clang attribute push(__attribute__((target("avx512f,avx512dq"))), \
                             apply_to = any(function))
#else
#pragma clang attribute push(__attribute__((target("avx2"))), \
                             apply_to = any(function))
#endif  // VXSORT_AVX3

#else
#pragma GCC push_options
#if VXSORT_AVX3
#pragma GCC target("avx512f,avx512dq")
#else
#pragma GCC target("avx2")
#endif  // VXSORT_AVX3
#endif
#endif

#if VXSORT_AVX3
#include "vxsort/machine_traits.avx512.h"
#else
#include "vxsort/machine_traits.avx2.h"
#endif  // VXSORT_AVX3
#include "vxsort/vxsort.h"
#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif
#endif  // HAVE_VXSORT

namespace hwy {

enum class Dist { kUniform8, kUniform16, kUniform32 };

static inline std::vector<Dist> AllDist() {
  return {/*Dist::kUniform8, Dist::kUniform16,*/ Dist::kUniform32};
}

static inline const char* DistName(Dist dist) {
  switch (dist) {
    case Dist::kUniform8:
      return "uniform8";
    case Dist::kUniform16:
      return "uniform16";
    case Dist::kUniform32:
      return "uniform32";
  }
  return "unreachable";
}

template <typename T>
class InputStats {
 public:
  void Notify(T value) {
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);
    // Converting to integer would truncate floats, multiplying to save digits
    // risks overflow especially when casting, so instead take the sum of the
    // bit representations as the checksum.
    uint64_t bits = 0;
    static_assert(sizeof(T) <= 8, "Expected a built-in type");
    CopyBytes<sizeof(T)>(&value, &bits);  // not same size
    sum_ += bits;
    count_ += 1;
  }

  bool operator==(const InputStats& other) const {
    char type_name[100];
    detail::TypeName(hwy::detail::MakeTypeInfo<T>(), 1, type_name);

    if (count_ != other.count_) {
      HWY_ABORT("Sort %s: count %d vs %d\n", type_name,
                static_cast<int>(count_), static_cast<int>(other.count_));
    }

    if (min_ != other.min_ || max_ != other.max_) {
      HWY_ABORT("Sort %s: minmax %f/%f vs %f/%f\n", type_name,
                static_cast<double>(min_), static_cast<double>(max_),
                static_cast<double>(other.min_),
                static_cast<double>(other.max_));
    }

    // Sum helps detect duplicated/lost values
    if (sum_ != other.sum_) {
      HWY_ABORT("Sort %s: Sum mismatch %g %g; min %g max %g\n", type_name,
                static_cast<double>(sum_), static_cast<double>(other.sum_),
                static_cast<double>(min_), static_cast<double>(max_));
    }

    return true;
  }

 private:
  T min_ = hwy::HighestValue<T>();
  T max_ = hwy::LowestValue<T>();
  uint64_t sum_ = 0;
  size_t count_ = 0;
};

enum class Algo {
#if HAVE_INTEL
  kIntel,
#endif
#if HAVE_AVX2SORT
  kSEA,
#endif
#if HAVE_IPS4O
  kIPS4O,
#endif
#if HAVE_PARALLEL_IPS4O
  kParallelIPS4O,
#endif
#if HAVE_PDQSORT
  kPDQ,
#endif
#if HAVE_SORT512
  kSort512,
#endif
#if HAVE_VXSORT
  kVXSort,
#endif
  kStdSort,
  kStdSelect,
  kStdPartialSort,
  kVQSort,
  kVQPartialSort,
  kVQSelect,
  kHeapSort,
  kHeapPartialSort,
  kHeapSelect,
};

static inline const char* AlgoName(Algo algo) {
  switch (algo) {
#if HAVE_INTEL
    case Algo::kIntel:
      return "intel";
#endif
#if HAVE_AVX2SORT
    case Algo::kSEA:
      return "sea";
#endif
#if HAVE_IPS4O
    case Algo::kIPS4O:
      return "ips4o";
#endif
#if HAVE_PARALLEL_IPS4O
    case Algo::kParallelIPS4O:
      return "par_ips4o";
#endif
#if HAVE_PDQSORT
    case Algo::kPDQ:
      return "pdq";
#endif
#if HAVE_SORT512
    case Algo::kSort512:
      return "sort512";
#endif
#if HAVE_VXSORT
    case Algo::kVXSort:
      return "vxsort";
#endif
    case Algo::kStdSort:
    case Algo::kStdPartialSort:
    case Algo::kStdSelect:
      return "std";
    case Algo::kVQSort:
    case Algo::kVQPartialSort:
    case Algo::kVQSelect:
      return "vq";
    case Algo::kHeapSort:
    case Algo::kHeapPartialSort:
      return "heapsort";
    case Algo::kHeapSelect:
      return "heapselect";
  }
  return "unreachable";
}

}  // namespace hwy
#endif  // HIGHWAY_HWY_CONTRIB_SORT_ALGO_INL_H_

// Per-target
#if defined(HIGHWAY_HWY_CONTRIB_SORT_ALGO_TOGGLE) == defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_SORT_ALGO_TOGGLE
#undef HIGHWAY_HWY_CONTRIB_SORT_ALGO_TOGGLE
#else
#define HIGHWAY_HWY_CONTRIB_SORT_ALGO_TOGGLE
#endif

#include "hwy/aligned_allocator.h"
#include "hwy/contrib/sort/traits-inl.h"
#include "hwy/contrib/sort/traits128-inl.h"
#include "hwy/contrib/sort/vqsort-inl.h"  // HeapSort

HWY_BEFORE_NAMESPACE();

// Requires target pragma set by HWY_BEFORE_NAMESPACE
#if HAVE_INTEL && HWY_TARGET <= HWY_AVX3
// #include "avx512-16bit-qsort.hpp"  // requires vbmi2
#include "avx512-32bit-qsort.hpp"
#include "avx512-64bit-qsort.hpp"
#endif

namespace hwy {
namespace HWY_NAMESPACE {

#if HAVE_INTEL || HAVE_VXSORT  // only supports ascending order
template <typename T>
using OtherOrder = detail::OrderAscending<T>;
#else
template <typename T>
using OtherOrder = detail::OrderDescending<T>;
#endif

class Xorshift128Plus {
  static HWY_INLINE uint64_t SplitMix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

 public:
  // Generates two vectors of 64-bit seeds via SplitMix64 and stores into
  // `seeds`. Generating these afresh in each ChoosePivot is too expensive.
  template <class DU64>
  static void GenerateSeeds(DU64 du64, TFromD<DU64>* HWY_RESTRICT seeds) {
    seeds[0] = SplitMix64(0x9E3779B97F4A7C15ull);
    for (size_t i = 1; i < 2 * Lanes(du64); ++i) {
      seeds[i] = SplitMix64(seeds[i - 1]);
    }
  }

  // Need to pass in the state because vector cannot be class members.
  template <class VU64>
  static VU64 RandomBits(VU64& state0, VU64& state1) {
    VU64 s1 = state0;
    VU64 s0 = state1;
    const VU64 bits = Add(s1, s0);
    state0 = s0;
    s1 = Xor(s1, ShiftLeft<23>(s1));
    state1 = Xor(s1, Xor(s0, Xor(ShiftRight<18>(s1), ShiftRight<5>(s0))));
    return bits;
  }
};

template <class D, class VU64, HWY_IF_NOT_FLOAT_D(D)>
Vec<D> RandomValues(D d, VU64& s0, VU64& s1, const VU64 mask) {
  const VU64 bits = Xorshift128Plus::RandomBits(s0, s1);
  return BitCast(d, And(bits, mask));
}

// It is important to avoid denormals, which are flushed to zero by SIMD but not
// scalar sorts, and NaN, which may be ordered differently in scalar vs. SIMD.
template <class DF, class VU64, HWY_IF_FLOAT_D(DF)>
Vec<DF> RandomValues(DF df, VU64& s0, VU64& s1, const VU64 mask) {
  using TF = TFromD<DF>;
  const RebindToUnsigned<decltype(df)> du;
  using VU = Vec<decltype(du)>;

  const VU64 bits64 = And(Xorshift128Plus::RandomBits(s0, s1), mask);

#if HWY_TARGET == HWY_SCALAR  // Cannot repartition u64 to smaller types
  using TU = MakeUnsigned<TF>;
  const VU bits = Set(du, static_cast<TU>(GetLane(bits64) & LimitsMax<TU>()));
#else
  const VU bits = BitCast(du, bits64);
#endif
  // Avoid NaN/denormal by only generating values in [1, 2), i.e. random
  // mantissas with the exponent taken from the representation of 1.0.
  const VU k1 = BitCast(du, Set(df, TF{1.0}));
  const VU mantissa_mask = Set(du, MantissaMask<TF>());
  const VU representation = OrAnd(k1, bits, mantissa_mask);
  return BitCast(df, representation);
}

template <class DU64>
Vec<DU64> MaskForDist(DU64 du64, const Dist dist, size_t sizeof_t) {
  switch (sizeof_t) {
    case 2:
      return Set(du64, (dist == Dist::kUniform8) ? 0x00FF00FF00FF00FFull
                                                 : 0xFFFFFFFFFFFFFFFFull);
    case 4:
      return Set(du64, (dist == Dist::kUniform8)    ? 0x000000FF000000FFull
                       : (dist == Dist::kUniform16) ? 0x0000FFFF0000FFFFull
                                                    : 0xFFFFFFFFFFFFFFFFull);
    case 8:
      return Set(du64, (dist == Dist::kUniform8)    ? 0x00000000000000FFull
                       : (dist == Dist::kUniform16) ? 0x000000000000FFFFull
                                                    : 0x00000000FFFFFFFFull);
    default:
      HWY_ABORT("Logic error");
      return Zero(du64);
  }
}

template <typename T>
InputStats<T> GenerateInput(const Dist dist, T* v, size_t num) {
  SortTag<uint64_t> du64;
  using VU64 = Vec<decltype(du64)>;
  const size_t N64 = Lanes(du64);
  auto seeds = hwy::AllocateAligned<uint64_t>(2 * N64);
  Xorshift128Plus::GenerateSeeds(du64, seeds.get());
  VU64 s0 = Load(du64, seeds.get());
  VU64 s1 = Load(du64, seeds.get() + N64);

#if HWY_TARGET == HWY_SCALAR
  const Sisd<T> d;
#else
  const Repartition<T, decltype(du64)> d;
#endif
  using V = Vec<decltype(d)>;
  const size_t N = Lanes(d);
  const VU64 mask = MaskForDist(du64, dist, sizeof(T));
  auto buf = hwy::AllocateAligned<T>(N);

  size_t i = 0;
  for (; i + N <= num; i += N) {
    const V values = RandomValues(d, s0, s1, mask);
    StoreU(values, d, v + i);
  }
  if (i < num) {
    const V values = RandomValues(d, s0, s1, mask);
    StoreU(values, d, buf.get());
    CopyBytes(buf.get(), v + i, (num - i) * sizeof(T));
  }

  InputStats<T> input_stats;
  for (size_t i = 0; i < num; ++i) {
    input_stats.Notify(v[i]);
  }
  return input_stats;
}

struct SharedState {
#if HAVE_PARALLEL_IPS4O
  const unsigned max_threads = hwy::LimitsMax<unsigned>();  // 16 for Table 1a
  ips4o::StdThreadPool pool{static_cast<int>(
      HWY_MIN(max_threads, std::thread::hardware_concurrency() / 2))};
#endif
};

// Bridge from keys (passed to Run) to lanes as expected by HeapPartialSort. For
// non-128-bit keys they are the same:
template <class Order, typename KeyType, HWY_IF_NOT_T_SIZE(KeyType, 16)>
void CallHeapPartialSort(KeyType* HWY_RESTRICT keys, const size_t num_keys,
                         const size_t k) {
  using detail::SharedTraits;
  using detail::TraitsLane;
  if (Order().IsAscending()) {
    const SharedTraits<TraitsLane<detail::OrderAscending<KeyType>>> st;
    return detail::HeapPartialSort(st, keys, num_keys, k);
  } else {
    const SharedTraits<TraitsLane<detail::OrderDescending<KeyType>>> st;
    return detail::HeapPartialSort(st, keys, num_keys, k);
  }
}

#if VQSORT_ENABLED
template <class Order>
void CallHeapPartialSort(hwy::uint128_t* HWY_RESTRICT keys,
                         const size_t num_keys, const size_t k) {
  using detail::SharedTraits;
  using detail::Traits128;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys * 2;
  if (Order().IsAscending()) {
    const SharedTraits<Traits128<detail::OrderAscending128>> st;
    return detail::HeapPartialSort(st, lanes, num_lanes, k);
  } else {
    const SharedTraits<Traits128<detail::OrderDescending128>> st;
    return detail::HeapPartialSort(st, lanes, num_lanes, k);
  }
}

template <class Order>
void CallHeapPartialSort(K64V64* HWY_RESTRICT keys, const size_t num_keys,
                         const size_t k) {
  using detail::SharedTraits;
  using detail::Traits128;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys * 2;
  if (Order().IsAscending()) {
    const SharedTraits<Traits128<detail::OrderAscendingKV128>> st;
    return detail::HeapPartialSort(st, lanes, num_lanes, k);
  } else {
    const SharedTraits<Traits128<detail::OrderDescendingKV128>> st;
    return detail::HeapPartialSort(st, lanes, num_lanes, k);
  }
}

template <class Order>
void CallHeapPartialSort(K32V32* HWY_RESTRICT keys, const size_t num_keys,
                         const size_t k) {
  using detail::SharedTraits;
  using detail::TraitsLane;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys;
  if (Order().IsAscending()) {
    const SharedTraits<TraitsLane<detail::OrderAscendingKV64>> st;
    return detail::HeapPartialSort(st, lanes, num_lanes, k);
  } else {
    const SharedTraits<TraitsLane<detail::OrderDescendingKV64>> st;
    return detail::HeapPartialSort(st, lanes, num_lanes, k);
  }
}

#endif  // VQSORT_ENABLED

// Bridge from keys (passed to Run) to lanes as expected by HeapSelect. For
// non-128-bit keys they are the same:
template <class Order, typename KeyType, HWY_IF_NOT_T_SIZE(KeyType, 16)>
void CallHeapSelect(KeyType* HWY_RESTRICT keys, const size_t num_keys,
                    const size_t k) {
  using detail::SharedTraits;
  using detail::TraitsLane;
  if (Order().IsAscending()) {
    const SharedTraits<TraitsLane<detail::OrderAscending<KeyType>>> st;
    return detail::HeapSelect(st, keys, num_keys, k);
  } else {
    const SharedTraits<TraitsLane<detail::OrderDescending<KeyType>>> st;
    return detail::HeapSelect(st, keys, num_keys, k);
  }
}

#if VQSORT_ENABLED
template <class Order>
void CallHeapSelect(hwy::uint128_t* HWY_RESTRICT keys, const size_t num_keys,
                    const size_t k) {
  using detail::SharedTraits;
  using detail::Traits128;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys * 2;
  if (Order().IsAscending()) {
    const SharedTraits<Traits128<detail::OrderAscending128>> st;
    return detail::HeapSelect(st, lanes, num_lanes, k);
  } else {
    const SharedTraits<Traits128<detail::OrderDescending128>> st;
    return detail::HeapSelect(st, lanes, num_lanes, k);
  }
}

template <class Order>
void CallHeapSelect(K64V64* HWY_RESTRICT keys, const size_t num_keys,
                    const size_t k) {
  using detail::SharedTraits;
  using detail::Traits128;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys * 2;
  if (Order().IsAscending()) {
    const SharedTraits<Traits128<detail::OrderAscendingKV128>> st;
    return detail::HeapSelect(st, lanes, num_lanes, k);
  } else {
    const SharedTraits<Traits128<detail::OrderDescendingKV128>> st;
    return detail::HeapSelect(st, lanes, num_lanes, k);
  }
}

template <class Order>
void CallHeapSelect(K32V32* HWY_RESTRICT keys, const size_t num_keys,
                    const size_t k) {
  using detail::SharedTraits;
  using detail::TraitsLane;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys;
  if (Order().IsAscending()) {
    const SharedTraits<TraitsLane<detail::OrderAscendingKV64>> st;
    return detail::HeapSelect(st, lanes, num_lanes, k);
  } else {
    const SharedTraits<TraitsLane<detail::OrderDescendingKV64>> st;
    return detail::HeapSelect(st, lanes, num_lanes, k);
  }
}

#endif  // VQSORT_ENABLED

// Bridge from keys (passed to Run) to lanes as expected by HeapSort. For
// non-128-bit keys they are the same:
template <class Order, typename KeyType, HWY_IF_NOT_T_SIZE(KeyType, 16)>
void CallHeapSort(KeyType* HWY_RESTRICT keys, const size_t num_keys) {
  using detail::SharedTraits;
  using detail::TraitsLane;
  if (Order().IsAscending()) {
    const SharedTraits<TraitsLane<detail::OrderAscending<KeyType>>> st;
    return detail::HeapSort(st, keys, num_keys);
  } else {
    const SharedTraits<TraitsLane<detail::OrderDescending<KeyType>>> st;
    return detail::HeapSort(st, keys, num_keys);
  }
}

#if VQSORT_ENABLED
template <class Order>
void CallHeapSort(hwy::uint128_t* HWY_RESTRICT keys, const size_t num_keys) {
  using detail::SharedTraits;
  using detail::Traits128;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys * 2;
  if (Order().IsAscending()) {
    const SharedTraits<Traits128<detail::OrderAscending128>> st;
    return detail::HeapSort(st, lanes, num_lanes);
  } else {
    const SharedTraits<Traits128<detail::OrderDescending128>> st;
    return detail::HeapSort(st, lanes, num_lanes);
  }
}

template <class Order>
void CallHeapSort(K64V64* HWY_RESTRICT keys, const size_t num_keys) {
  using detail::SharedTraits;
  using detail::Traits128;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys * 2;
  if (Order().IsAscending()) {
    const SharedTraits<Traits128<detail::OrderAscendingKV128>> st;
    return detail::HeapSort(st, lanes, num_lanes);
  } else {
    const SharedTraits<Traits128<detail::OrderDescendingKV128>> st;
    return detail::HeapSort(st, lanes, num_lanes);
  }
}

template <class Order>
void CallHeapSort(K32V32* HWY_RESTRICT keys, const size_t num_keys) {
  using detail::SharedTraits;
  using detail::TraitsLane;
  uint64_t* lanes = reinterpret_cast<uint64_t*>(keys);
  const size_t num_lanes = num_keys;
  if (Order().IsAscending()) {
    const SharedTraits<TraitsLane<detail::OrderAscendingKV64>> st;
    return detail::HeapSort(st, lanes, num_lanes);
  } else {
    const SharedTraits<TraitsLane<detail::OrderDescendingKV64>> st;
    return detail::HeapSort(st, lanes, num_lanes);
  }
}

#endif  // VQSORT_ENABLED

template <class Order, typename KeyType>
void Run(Algo algo, KeyType* HWY_RESTRICT inout, size_t num,
         SharedState& shared, size_t /*thread*/, size_t k = 0) {
  const std::less<KeyType> less;
  const std::greater<KeyType> greater;

#if !HAVE_PARALLEL_IPS4O
  (void)shared;
#endif

  switch (algo) {
#if HAVE_INTEL && HWY_TARGET <= HWY_AVX3
    case Algo::kIntel:
      return avx512_qsort<KeyType>(inout, static_cast<int64_t>(num));
#endif

#if HAVE_AVX2SORT
    case Algo::kSEA:
      return avx2::quicksort(inout, static_cast<int>(num));
#endif

#if HAVE_IPS4O
    case Algo::kIPS4O:
      if (Order().IsAscending()) {
        return ips4o::sort(inout, inout + num, less);
      } else {
        return ips4o::sort(inout, inout + num, greater);
      }
#endif

#if HAVE_PARALLEL_IPS4O
    case Algo::kParallelIPS4O:
      if (Order().IsAscending()) {
        return ips4o::parallel::sort(inout, inout + num, less, shared.pool);
      } else {
        return ips4o::parallel::sort(inout, inout + num, greater, shared.pool);
      }
#endif

#if HAVE_SORT512
    case Algo::kSort512:
      HWY_ABORT("not supported");
      //    return Sort512::Sort(inout, num);
#endif

#if HAVE_PDQSORT
    case Algo::kPDQ:
      if (Order().IsAscending()) {
        return boost::sort::pdqsort_branchless(inout, inout + num, less);
      } else {
        return boost::sort::pdqsort_branchless(inout, inout + num, greater);
      }
#endif

#if HAVE_VXSORT
    case Algo::kVXSort: {
#if (VXSORT_AVX3 && HWY_TARGET != HWY_AVX3) || \
    (!VXSORT_AVX3 && HWY_TARGET != HWY_AVX2)
      fprintf(stderr, "Do not call for target %s\n",
              hwy::TargetName(HWY_TARGET));
      return;
#else
#if VXSORT_AVX3
      vxsort::vxsort<KeyType, vxsort::AVX512> vx;
#else
      vxsort::vxsort<KeyType, vxsort::AVX2> vx;
#endif
      if (Order().IsAscending()) {
        return vx.sort(inout, inout + num - 1);
      } else {
        fprintf(stderr, "Skipping VX - does not support descending order\n");
        return;
      }
#endif  // enabled for this target
    }
#endif  // HAVE_VXSORT

    case Algo::kStdSort:
      if (Order().IsAscending()) {
        return std::sort(inout, inout + num, less);
      } else {
        return std::sort(inout, inout + num, greater);
      }

    case Algo::kStdPartialSort:
      if (Order().IsAscending()) {
        return std::partial_sort(inout, inout + k, inout + num, less);
      } else {
        return std::partial_sort(inout, inout + k, inout + num, greater);
      }

    case Algo::kStdSelect:
      if (Order().IsAscending()) {
        return std::nth_element(inout, inout + k, inout + num, less);
      } else {
        return std::nth_element(inout, inout + k, inout + num, greater);
      }

    case Algo::kVQSort:
      return VQSort(inout, num, Order());

    case Algo::kVQPartialSort:
      return VQPartialSort(inout, num, k, Order());

    case Algo::kVQSelect:
      return VQSelect(inout, num, k, Order());

    case Algo::kHeapSort:
      return CallHeapSort<Order>(inout, num);

    case Algo::kHeapPartialSort:
      return CallHeapPartialSort<Order>(inout, num, k);

    case Algo::kHeapSelect:
      return CallHeapSelect<Order>(inout, num, k);

    default:
      HWY_ABORT("Not implemented");
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SORT_ALGO_TOGGLE
