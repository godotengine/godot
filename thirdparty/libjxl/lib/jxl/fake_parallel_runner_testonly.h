// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_FAKE_PARALLEL_RUNNER_TESTONLY_H_
#define LIB_JXL_FAKE_PARALLEL_RUNNER_TESTONLY_H_

#include <jxl/parallel_runner.h>

#include <cstdint>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/random.h"

namespace jxl {

// A parallel runner implementation that runs all the jobs in a single thread
// (the caller thread) but runs them pretending to use multiple threads and
// potentially out of order. This is useful for testing conditions that only
// occur under heavy load where the order of operations is different.
class FakeParallelRunner {
 public:
  FakeParallelRunner(uint32_t order_seed, uint32_t num_threads)
      : order_seed_(order_seed), rng_(order_seed), num_threads_(num_threads) {
    if (num_threads_ < 1) num_threads_ = 1;
  }

  JxlParallelRetCode Run(void* jxl_opaque, JxlParallelRunInit init,
                         JxlParallelRunFunction func, uint32_t start,
                         uint32_t end) {
    JxlParallelRetCode ret = init(jxl_opaque, num_threads_);
    if (ret != 0) return ret;

    if (order_seed_ == 0) {
      for (uint32_t i = start; i < end; i++) {
        func(jxl_opaque, i, i % num_threads_);
      }
    } else {
      std::vector<uint32_t> order(end - start);
      for (uint32_t i = start; i < end; i++) {
        order[i - start] = i;
      }
      rng_.Shuffle(order.data(), order.size());
      for (uint32_t i = start; i < end; i++) {
        func(jxl_opaque, order[i - start], i % num_threads_);
      }
    }
    return ret;
  }

 private:
  // Seed for the RNG for defining the execution order. A value of 0 means
  // sequential order from start to end.
  uint32_t order_seed_;

  // The PRNG object, initialized with the order_seed_. Only used if the seed is
  // not 0.
  Rng rng_;

  // Number of fake threads. All the tasks are run on the same thread, but using
  // different thread_id values based on this num_threads.
  uint32_t num_threads_;
};

}  // namespace jxl

extern "C" {
// Function to pass as the parallel runner.
JXL_INLINE JxlParallelRetCode JxlFakeParallelRunner(
    void* runner_opaque, void* jpegxl_opaque, JxlParallelRunInit init,
    JxlParallelRunFunction func, uint32_t start_range, uint32_t end_range) {
  return static_cast<jxl::FakeParallelRunner*>(runner_opaque)
      ->Run(jpegxl_opaque, init, func, start_range, end_range);
}
}

#endif  // LIB_JXL_FAKE_PARALLEL_RUNNER_TESTONLY_H_
