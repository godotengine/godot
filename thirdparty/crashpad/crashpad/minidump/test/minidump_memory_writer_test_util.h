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

#ifndef CRASHPAD_MINIDUMP_TEST_MINIDUMP_MEMORY_WRITER_TEST_UTIL_H_
#define CRASHPAD_MINIDUMP_TEST_MINIDUMP_MEMORY_WRITER_TEST_UTIL_H_

#include "minidump/minidump_memory_writer.h"

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>

#include <string>

#include "base/macros.h"
#include "snapshot/test/test_memory_snapshot.h"
#include "util/file/file_writer.h"

namespace crashpad {
namespace test {

//! \brief A SnapshotMinidumpMemoryWriter implementation used for testing.
//!
//! TestMinidumpMemoryWriter objects are created with a fixed base address and
//! size, and will write the same byte (\a value) repeatedly, \a size times.
class TestMinidumpMemoryWriter final : public SnapshotMinidumpMemoryWriter {
 public:
  TestMinidumpMemoryWriter(uint64_t base_address, size_t size, uint8_t value);
  ~TestMinidumpMemoryWriter();

  void SetShouldFailRead(bool should_fail);

 private:
  TestMemorySnapshot test_snapshot_;

  DISALLOW_COPY_AND_ASSIGN(TestMinidumpMemoryWriter);
};

//! \brief Verifies, via gtest assertions, that a MINIDUMP_MEMORY_DESCRIPTOR
//!     structure contains expected values.
//!
//! In \a expected and \a observed,
//! MINIDUMP_MEMORY_DESCRIPTOR::StartOfMemoryRange and
//! MINIDUMP_LOCATION_DESCRIPTOR::DataSize are compared and must match. If
//! MINIDUMP_LOCATION_DESCRIPTOR::Rva is nonzero in \a expected, the same field
//! in \a observed must match it, subject to a 16-byte alignment augmentation.
//!
//! \param[in] expected A MINIDUMP_MEMORY_DESCRIPTOR structure containing
//!     expected values.
//! \param[in] observed A MINIDUMP_MEMORY_DESCRIPTOR structure containing
//!     observed values.
void ExpectMinidumpMemoryDescriptor(const MINIDUMP_MEMORY_DESCRIPTOR* expected,
                                    const MINIDUMP_MEMORY_DESCRIPTOR* observed);

//! \brief Verifies, via gtest assertions, that a MINIDUMP_MEMORY_DESCRIPTOR
//!     structure contains expected values, and that the memory region it points
//!     to contains expected values assuming it was written by a
//!     TestMinidumpMemoryWriter object.
//!
//! \a expected and \a observed are compared by
//! ExpectMinidumpMemoryDescriptor().
//!
//! \param[in] expected A MINIDUMP_MEMORY_DESCRIPTOR structure containing
//!     expected values.
//! \param[in] observed A MINIDUMP_MEMORY_DESCRIPTOR structure containing
//!     observed values.
//! \param[in] file_contents The contents of the minidump file in which \a
//!     observed was found. The memory region referenced by \a observed will be
//!     read from this string.
//! \param[in] value The \a value used to create a TestMinidumpMemoryWriter.
//!     Each byte of memory in the region referenced by \a observed must be this
//!     value.
//! \param[in] at_eof If `true`, the region referenced by \a observed must
//!     appear at the end of \a file_contents, without any data following it.
void ExpectMinidumpMemoryDescriptorAndContents(
    const MINIDUMP_MEMORY_DESCRIPTOR* expected,
    const MINIDUMP_MEMORY_DESCRIPTOR* observed,
    const std::string& file_contents,
    uint8_t value,
    bool at_eof);

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_TEST_MINIDUMP_MEMORY_WRITER_TEST_UTIL_H_
