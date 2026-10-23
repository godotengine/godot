/*
 *  Copyright (c) 2018 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdio.h>
#include <algorithm>
#include <cstdlib>

#include "test/bench.h"
#include "vpx_ports/vpx_timer.h"

void AbstractBench::RunNTimes(int n) {
  for (int r = 0; r < VPX_BENCH_ROBUST_ITER; r++) {
    vpx_usec_timer timer;
    vpx_usec_timer_start(&timer);
    for (int j = 0; j < n; ++j) {
      Run();
    }
    vpx_usec_timer_mark(&timer);
    times_[r] = static_cast<int>(vpx_usec_timer_elapsed(&timer));
  }
}

void AbstractBench::PrintMedian(const char *title) {
  std::sort(times_, times_ + VPX_BENCH_ROBUST_ITER);
  const int med = times_[VPX_BENCH_ROBUST_ITER >> 1];
  int sad = 0;
  for (int t = 0; t < VPX_BENCH_ROBUST_ITER; t++) {
    sad += abs(times_[t] - med);
  }
  printf("[%10s] %s %.1f ms ( ±%.1f ms )\n", "BENCH ", title, med / 1000.0,
         sad / (VPX_BENCH_ROBUST_ITER * 1000.0));
}
