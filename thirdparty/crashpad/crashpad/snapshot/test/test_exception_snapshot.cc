// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "snapshot/test/test_exception_snapshot.h"

namespace crashpad {
namespace test {

TestExceptionSnapshot::TestExceptionSnapshot()
    : context_union_(),
      context_(),
      thread_id_(0),
      exception_(0),
      exception_info_(0),
      exception_address_(0),
      codes_() {
  context_.x86 = &context_union_.x86;
}

TestExceptionSnapshot::~TestExceptionSnapshot() {
}

const CPUContext* TestExceptionSnapshot::Context() const {
  return &context_;
}

uint64_t TestExceptionSnapshot::ThreadID() const {
  return thread_id_;
}

uint32_t TestExceptionSnapshot::Exception() const {
  return exception_;
}

uint32_t TestExceptionSnapshot::ExceptionInfo() const {
  return exception_info_;
}

uint64_t TestExceptionSnapshot::ExceptionAddress() const {
  return exception_address_;
}

const std::vector<uint64_t>& TestExceptionSnapshot::Codes() const {
  return codes_;
}

std::vector<const MemorySnapshot*> TestExceptionSnapshot::ExtraMemory() const {
  std::vector<const MemorySnapshot*> extra_memory;
  for (const auto& em : extra_memory_) {
    extra_memory.push_back(em.get());
  }
  return extra_memory;
}

}  // namespace test
}  // namespace crashpad
