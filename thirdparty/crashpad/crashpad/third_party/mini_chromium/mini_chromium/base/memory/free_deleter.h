// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_MEMORY_FREE_DELETER_H_
#define MINI_CHROMIUM_BASE_MEMORY_FREE_DELETER_H_

#include <stdlib.h>

namespace base {

// Function object which invokes 'free' on its parameter, which must be
// a pointer. Can be used to store malloc-allocated pointers in std::unique_ptr:
//
// std::unique_ptr<int, base::FreeDeleter> foo_ptr(
//     static_cast<int*>(malloc(sizeof(int))));
struct FreeDeleter {
  inline void operator()(void* ptr) const {
    free(ptr);
  }
};

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_MEMORY_FREE_DELETER_H_
