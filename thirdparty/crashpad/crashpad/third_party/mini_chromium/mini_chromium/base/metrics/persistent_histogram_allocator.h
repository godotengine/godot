// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_METRICS_PERSISTENT_HISTOGRAM_ALLOCATOR_H_
#define MINI_CHROMIUM_BASE_METRICS_PERSISTENT_HISTOGRAM_ALLOCATOR_H_

#include <inttypes.h>
#include <sys/types.h>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "base/strings/string_piece.h"

// This file is a non-functional stub of the Chromium base interface to allow
// Crashpad to set up and tear down histogram storage when built against
// Chromium. When Crashpad is built standalone these stubs are used which
// silently do nothing.
namespace base {

class GlobalHistogramAllocator {
 public:
  static bool CreateWithActiveFileInDir(const base::FilePath&,
                                        size_t,
                                        uint64_t,
                                        base::StringPiece sp) {
    return false;
  }

  void CreateTrackingHistograms(base::StringPiece) {}
  void DeletePersistentLocation() {}

  static GlobalHistogramAllocator* Get() { return nullptr; }

 private:
  DISALLOW_COPY_AND_ASSIGN(GlobalHistogramAllocator);
};

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_METRICS_PERSISTENT_HISTOGRAM_ALLOCATOR_H_
