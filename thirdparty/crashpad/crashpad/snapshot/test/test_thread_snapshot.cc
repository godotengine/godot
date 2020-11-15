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

#include "snapshot/test/test_thread_snapshot.h"

namespace crashpad {
namespace test {

TestThreadSnapshot::TestThreadSnapshot()
    : context_union_(),
      context_(),
      stack_(),
      thread_id_(0),
      suspend_count_(0),
      priority_(0),
      thread_specific_data_address_(0) {
  context_.x86 = &context_union_.x86;
}

TestThreadSnapshot::~TestThreadSnapshot() {
}

const CPUContext* TestThreadSnapshot::Context() const {
  return &context_;
}

const MemorySnapshot* TestThreadSnapshot::Stack() const {
  return stack_.get();
}

uint64_t TestThreadSnapshot::ThreadID() const {
  return thread_id_;
}

int TestThreadSnapshot::SuspendCount() const {
  return suspend_count_;
}

int TestThreadSnapshot::Priority() const {
  return priority_;
}

uint64_t TestThreadSnapshot::ThreadSpecificDataAddress() const {
  return thread_specific_data_address_;
}

std::vector<const MemorySnapshot*> TestThreadSnapshot::ExtraMemory() const {
  std::vector<const MemorySnapshot*> extra_memory;
  for (const auto& em : extra_memory_) {
    extra_memory.push_back(em.get());
  }
  return extra_memory;
}

}  // namespace test
}  // namespace crashpad
