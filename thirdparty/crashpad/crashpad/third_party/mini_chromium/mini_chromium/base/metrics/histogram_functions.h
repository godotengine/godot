// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_METRICS_HISTOGRAM_FUNCTIONS_H_
#define MINI_CHROMIUM_BASE_METRICS_HISTOGRAM_FUNCTIONS_H_

#include <string>

// These are no-op stub versions of a subset of the functions from Chromium's
// base/metrics/histogram_functions.h. This allows us to instrument the Crashpad
// code as necessary, while not affecting out-of-Chromium builds.
namespace base {

void UmaHistogramSparse(const std::string& name, int sample) {}

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_METRICS_HISTOGRAM_FUNCTIONS_H_
