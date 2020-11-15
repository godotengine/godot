// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRASHPAD_UTIL_STDLIB_THREAD_SAFE_VECTOR_H_
#define CRASHPAD_UTIL_STDLIB_THREAD_SAFE_VECTOR_H_

#include <utility>
#include <vector>

#include "base/macros.h"
#include "base/synchronization/lock.h"

namespace crashpad {

//! \brief A wrapper for a `std::vector<>` that can be accessed safely from
//!    multiple threads.
//!
//! This is not a drop-in replacement for `std::vector<>`. Only necessary
//! operations are defined.
template <typename T>
class ThreadSafeVector {
 public:
  ThreadSafeVector() : vector_(), lock_() {}
  ~ThreadSafeVector() {}

  //! \brief Wraps `std::vector<>::%push_back()`.
  void PushBack(const T& element) {
    base::AutoLock lock_owner(lock_);
    vector_.push_back(element);
  }

  //! \brief Atomically clears the underlying vector and returns its previous
  //!     contents.
  std::vector<T> Drain() {
    std::vector<T> contents;
    {
      base::AutoLock lock_owner(lock_);
      std::swap(vector_, contents);
    }
    return contents;
  }

 private:
  std::vector<T> vector_;
  base::Lock lock_;

  DISALLOW_COPY_AND_ASSIGN(ThreadSafeVector);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_STDLIB_THREAD_SAFE_VECTOR_H_
