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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_THREAD_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_THREAD_SNAPSHOT_H_

#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "base/macros.h"
#include "snapshot/cpu_context.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/thread_snapshot.h"

namespace crashpad {
namespace test {

//! \brief A test ThreadSnapshot that can carry arbitrary data for testing
//!     purposes.
class TestThreadSnapshot final : public ThreadSnapshot {
 public:
  TestThreadSnapshot();
  ~TestThreadSnapshot();

  //! \brief Obtains a pointer to the underlying mutable CPUContext structure.
  //!
  //! This method is intended to be used by callers to populate the CPUContext
  //! structure.
  //!
  //! \return The same pointer that Context() does, while treating the data as
  //!     mutable.
  //!
  //! \attention This returns a non-`const` pointer to this object’s private
  //!     data so that a caller can populate the context structure directly.
  //!     This is done because providing setter interfaces to each field in the
  //!     context structure would be unwieldy and cumbersome. Care must be taken
  //!     to populate the context structure correctly.
  CPUContext* MutableContext() { return &context_; }

  //! \brief Sets the memory region to be returned by Stack().
  //!
  //! \param[in] stack The memory region that Stack() will return. The
  //!     TestThreadSnapshot object takes ownership of \a stack.
  void SetStack(std::unique_ptr<MemorySnapshot> stack) {
    stack_ = std::move(stack);
  }

  void SetThreadID(uint64_t thread_id) { thread_id_ = thread_id; }
  void SetSuspendCount(int suspend_count) { suspend_count_ = suspend_count; }
  void SetPriority(int priority) { priority_ = priority; }
  void SetThreadSpecificDataAddress(uint64_t thread_specific_data_address) {
    thread_specific_data_address_ = thread_specific_data_address;
  }

  //! \brief Add a memory snapshot to be returned by ExtraMemory().
  //!
  //! \param[in] extra_memory The memory snapshot that will be included in
  //!     ExtraMemory(). The TestThreadSnapshot object takes ownership of \a
  //!     extra_memory.
  void AddExtraMemory(std::unique_ptr<MemorySnapshot> extra_memory) {
    extra_memory_.push_back(std::move(extra_memory));
  }

  // ThreadSnapshot:

  const CPUContext* Context() const override;
  const MemorySnapshot* Stack() const override;
  uint64_t ThreadID() const override;
  int SuspendCount() const override;
  int Priority() const override;
  uint64_t ThreadSpecificDataAddress() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
  union {
    CPUContextX86 x86;
    CPUContextX86_64 x86_64;
  } context_union_;
  CPUContext context_;
  std::unique_ptr<MemorySnapshot> stack_;
  uint64_t thread_id_;
  int suspend_count_;
  int priority_;
  uint64_t thread_specific_data_address_;
  std::vector<std::unique_ptr<MemorySnapshot>> extra_memory_;

  DISALLOW_COPY_AND_ASSIGN(TestThreadSnapshot);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_THREAD_SNAPSHOT_H_
