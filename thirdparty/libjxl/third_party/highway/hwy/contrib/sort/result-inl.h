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

#include "hwy/contrib/sort/algo-inl.h"

// Normal include guard for non-SIMD parts
#ifndef HIGHWAY_HWY_CONTRIB_SORT_RESULT_INL_H_
#define HIGHWAY_HWY_CONTRIB_SORT_RESULT_INL_H_

#include <time.h>

#include <algorithm>  // std::sort
#include <string>

#include "hwy/base.h"
#include "hwy/nanobenchmark.h"
#include "hwy/timer.h"

namespace hwy {

// Returns trimmed mean (we don't want to run an out-of-L3-cache sort often
// enough for the mode to be reliable).
static inline double SummarizeMeasurements(std::vector<double>& seconds) {
  std::sort(seconds.begin(), seconds.end());
  double sum = 0;
  int count = 0;
  const size_t num = seconds.size();
  for (size_t i = num / 4; i < num / 2; ++i) {
    sum += seconds[i];
    count += 1;
  }
  return sum / count;
}

}  // namespace hwy
#endif  // HIGHWAY_HWY_CONTRIB_SORT_RESULT_INL_H_

// Per-target
#if defined(HIGHWAY_HWY_CONTRIB_SORT_RESULT_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_SORT_RESULT_TOGGLE
#undef HIGHWAY_HWY_CONTRIB_SORT_RESULT_TOGGLE
#else
#define HIGHWAY_HWY_CONTRIB_SORT_RESULT_TOGGLE
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

struct Result {
  Result() {}
  Result(const Algo algo, Dist dist, size_t num_keys, size_t num_threads,
         double sec, size_t sizeof_key, const char* key_name)
      : target(HWY_TARGET),
        algo(algo),
        dist(dist),
        num_keys(num_keys),
        num_threads(num_threads),
        sec(sec),
        sizeof_key(sizeof_key),
        key_name(key_name) {}

  void Print() const {
    const double bytes = static_cast<double>(num_keys) *
                         static_cast<double>(num_threads) *
                         static_cast<double>(sizeof_key);
    printf("%10s: %12s: %7s: %9s: %05g %4.0f MB/s (%2zu threads)\n",
           hwy::TargetName(target), AlgoName(algo), key_name.c_str(),
           DistName(dist), static_cast<double>(num_keys), bytes * 1E-6 / sec,
           num_threads);
  }

  int64_t target;
  Algo algo;
  Dist dist;
  size_t num_keys = 0;
  size_t num_threads = 0;
  double sec = 0.0;
  size_t sizeof_key = 0;
  std::string key_name;
};

template <class Traits, typename LaneType>
bool VerifyPartialSort(Traits st, const InputStats<LaneType>& input_stats,
                       const LaneType* out, const size_t num_lanes,
                       const size_t k, const char* caller) {
  constexpr size_t N1 = st.LanesPerKey();
  HWY_ASSERT(num_lanes >= N1);
  HWY_ASSERT(k >= N1 && k < num_lanes);

  InputStats<LaneType> output_stats;
  // Ensure it matches the sort order
  for (size_t i = 0; i < num_lanes - N1; i += N1) {
    output_stats.Notify(out[i]);
    if (N1 == 2) output_stats.Notify(out[i + 1]);

    // Reverse order instead of checking !Compare1 so we accept equal keys.
    if (i < k - N1 && st.Compare1(out + i + N1, out + i)) {
      fprintf(stderr, "%s: i=%d of %d lanes: N1=%d", caller,
              static_cast<int>(i), static_cast<int>(num_lanes),
              static_cast<int>(N1));
      // TODO %5.0f prints unhelpful integers for the float/double tests.
      fprintf(stderr, "%5.0f %5.0f vs. %5.0f %5.0f\n\n",
              static_cast<double>(out[i + 1]), static_cast<double>(out[i + 0]),
              static_cast<double>(out[i + N1 + 1]),
              static_cast<double>(out[i + N1]));
      HWY_ABORT("%d-bit sort is incorrect\n",
                static_cast<int>(sizeof(LaneType) * 8 * N1));
    }
  }
  output_stats.Notify(out[num_lanes - N1]);
  if (N1 == 2) output_stats.Notify(out[num_lanes - N1 + 1]);

  return input_stats == output_stats;
}

template <class Traits, typename LaneType>
bool VerifySort(Traits st, const InputStats<LaneType>& input_stats,
                const LaneType* out, size_t num_lanes, const char* caller) {
  constexpr size_t N1 = st.LanesPerKey();
  HWY_ASSERT(num_lanes >= N1);

  InputStats<LaneType> output_stats;
  // Ensure it matches the sort order
  for (size_t i = 0; i < num_lanes - N1; i += N1) {
    output_stats.Notify(out[i]);
    if (N1 == 2) output_stats.Notify(out[i + 1]);
    // Reverse order instead of checking !Compare1 so we accept equal keys.
    if (st.Compare1(out + i + N1, out + i)) {
      fprintf(stderr, "%s: i=%d of %d lanes: N1=%d", caller,
              static_cast<int>(i), static_cast<int>(num_lanes),
              static_cast<int>(N1));
      // TODO %5.0f prints unhelpful integers for the float/double tests.
      fprintf(stderr, "%5.0f %5.0f vs. %5.0f %5.0f\n\n",
              static_cast<double>(out[i + 1]), static_cast<double>(out[i + 0]),
              static_cast<double>(out[i + N1 + 1]),
              static_cast<double>(out[i + N1]));
      HWY_ABORT("%d-bit sort is incorrect\n",
                static_cast<int>(sizeof(LaneType) * 8 * N1));
    }
  }
  output_stats.Notify(out[num_lanes - N1]);
  if (N1 == 2) output_stats.Notify(out[num_lanes - N1 + 1]);

  return input_stats == output_stats;
}

template <class Traits, typename LaneType>
bool VerifySelect(Traits st, const InputStats<LaneType>& input_stats,
                  const LaneType* out, const size_t num_lanes, const size_t k,
                  const char* caller) {
  constexpr size_t N1 = st.LanesPerKey();
  HWY_ASSERT(num_lanes >= N1);

  InputStats<LaneType> output_stats;
  // Ensure all of the elements below the k_th element are <= the k_th element,
  // and all of the elements above the k_th element are >= the k_th element.
  for (size_t i = 0; i < num_lanes - N1; i += N1) {
    output_stats.Notify(out[i]);
    if (N1 == 2) output_stats.Notify(out[i + 1]);
    // Reverse order instead of checking !Compare1 so we accept equal keys.
    if (i < k ? st.Compare1(out + k, out + i) : st.Compare1(out + i, out + k)) {
      fprintf(stderr, "%s: i=%d of %d lanes: N1=%d k=%d\t", caller,
              static_cast<int>(i), static_cast<int>(num_lanes),
              static_cast<int>(N1), static_cast<int>(k));
      fprintf(stderr, "%5.0f %5.0f vs. %5.0f %5.0f\n\n",
              static_cast<double>(out[i]), static_cast<double>(out[i + 1]),
              static_cast<double>(out[k]), static_cast<double>(out[k + 1]));
      HWY_ABORT("%d-bit select is incorrect\n",
                static_cast<int>(sizeof(LaneType) * 8 * N1));
    }
  }
  output_stats.Notify(out[num_lanes - N1]);
  if (N1 == 2) output_stats.Notify(out[num_lanes - N1 + 1]);

  return input_stats == output_stats;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SORT_RESULT_TOGGLE
