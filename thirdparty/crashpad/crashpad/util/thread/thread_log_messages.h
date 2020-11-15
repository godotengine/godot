// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_THREAD_THREAD_LOG_MESSAGES_H_
#define CRASHPAD_UTIL_THREAD_THREAD_LOG_MESSAGES_H_

#include <string>
#include <vector>

#include "base/macros.h"

namespace crashpad {

//! \brief Captures log messages produced on the current thread during an
//!     object’s lifetime.
//!
//! At most one object of this class type may exist on a single thread at a
//! time. When using this class, no other part of the program may call
//! `logging::SetLogMessageHandler()` at any time.
class ThreadLogMessages {
 public:
  ThreadLogMessages();
  ~ThreadLogMessages();

  //! \return The log messages collected on the thread that this object was
  //!     created on since the time it was created.
  const std::vector<std::string>& log_messages() const { return log_messages_; }

 private:
  std::vector<std::string> log_messages_;

  DISALLOW_COPY_AND_ASSIGN(ThreadLogMessages);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_THREAD_THREAD_LOG_MESSAGES_H_
