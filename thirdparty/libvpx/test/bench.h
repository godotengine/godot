/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_TEST_BENCH_H_
#define VPX_TEST_BENCH_H_

// Number of iterations used to compute median run time.
#define VPX_BENCH_ROBUST_ITER 15

class AbstractBench {
 public:
  virtual ~AbstractBench() = default;

  void RunNTimes(int n);
  void PrintMedian(const char *title);

 protected:
  // Implement this method and put the code to benchmark in it.
  virtual void Run() = 0;

 private:
  int times_[VPX_BENCH_ROBUST_ITER];
};

#endif  // VPX_TEST_BENCH_H_
