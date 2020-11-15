// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_THREAD_STOPPABLE_H_
#define CRASHPAD_UTIL_THREAD_STOPPABLE_H_

#include "base/macros.h"

namespace crashpad {

//! \brief An interface for operations that may be Started and Stopped.
class Stoppable {
 public:
  virtual ~Stoppable() = default;

  //! \brief Starts the operation.
  virtual void Start() = 0;

  //! \brief Stops the operation.
  virtual void Stop() = 0;

 protected:
  Stoppable() = default;
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_THREAD_STOPPABLE_H_
