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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_MEMORY_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_MEMORY_SNAPSHOT_H_

#include <stdint.h>
#include <sys/types.h>

#include "base/macros.h"
#include "snapshot/memory_snapshot.h"

namespace crashpad {
namespace test {

//! \brief A test MemorySnapshot that can carry arbitrary data for testing
//!     purposes.
class TestMemorySnapshot final : public MemorySnapshot {
 public:
  TestMemorySnapshot();
  ~TestMemorySnapshot();

  void SetAddress(uint64_t address) { address_ = address; }
  void SetSize(size_t size) { size_ = size; }

  //! \brief Sets the value to fill the test memory region with.
  //!
  //! \param[in] value The value to be written to \a delegate when Read() is
  //!     called. This value will be repeated Size() times.
  void SetValue(char value) { value_ = value; }

  void SetShouldFailRead(bool should_fail) { should_fail_ = true; }

  // MemorySnapshot:

  uint64_t Address() const override;
  size_t Size() const override;
  bool Read(Delegate* delegate) const override;
  const MemorySnapshot* MergeWithOtherSnapshot(
      const MemorySnapshot* other) const override;

 private:
  uint64_t address_;
  size_t size_;
  char value_;
  bool should_fail_;

  DISALLOW_COPY_AND_ASSIGN(TestMemorySnapshot);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_MEMORY_SNAPSHOT_H_
