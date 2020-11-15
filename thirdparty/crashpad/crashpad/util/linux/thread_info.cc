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

#include "util/linux/thread_info.h"

#include <string.h>

namespace crashpad {

ThreadContext::ThreadContext() {
  memset(this, 0, sizeof(*this));
}

ThreadContext::~ThreadContext() {}

FloatContext::FloatContext() {
  memset(this, 0, sizeof(*this));
}

FloatContext::~FloatContext() {}

ThreadInfo::ThreadInfo()
    : thread_context(), float_context(), thread_specific_data_address(0) {}

ThreadInfo::~ThreadInfo() {}

}  // namespace crashpad
