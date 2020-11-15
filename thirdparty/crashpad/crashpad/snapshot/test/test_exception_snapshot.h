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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_EXCEPTION_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_EXCEPTION_SNAPSHOT_H_

#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "base/macros.h"
#include "snapshot/cpu_context.h"
#include "snapshot/exception_snapshot.h"

namespace crashpad {
namespace test {

//! \brief A test ExceptionSnapshot that can carry arbitrary data for testing
//!     purposes.
class TestExceptionSnapshot final : public ExceptionSnapshot {
 public:
  TestExceptionSnapshot();
  ~TestExceptionSnapshot();

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

  void SetThreadID(uint64_t thread_id) { thread_id_ = thread_id; }
  void SetException(uint32_t exception) { exception_ = exception; }
  void SetExceptionInfo(uint32_t exception_information) {
    exception_info_ = exception_information;
  }
  void SetExceptionAddress(uint64_t exception_address) {
    exception_address_ = exception_address;
  }
  void SetCodes(const std::vector<uint64_t>& codes) { codes_ = codes; }
  void AddExtraMemory(std::unique_ptr<MemorySnapshot> extra_memory) {
    extra_memory_.push_back(std::move(extra_memory));
  }

  // ExceptionSnapshot:

  const CPUContext* Context() const override;
  uint64_t ThreadID() const override;
  uint32_t Exception() const override;
  uint32_t ExceptionInfo() const override;
  uint64_t ExceptionAddress() const override;
  const std::vector<uint64_t>& Codes() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
  union {
    CPUContextX86 x86;
    CPUContextX86_64 x86_64;
  } context_union_;
  CPUContext context_;
  uint64_t thread_id_;
  uint32_t exception_;
  uint32_t exception_info_;
  uint64_t exception_address_;
  std::vector<uint64_t> codes_;
  std::vector<std::unique_ptr<MemorySnapshot>> extra_memory_;

  DISALLOW_COPY_AND_ASSIGN(TestExceptionSnapshot);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_EXCEPTION_SNAPSHOT_H_
