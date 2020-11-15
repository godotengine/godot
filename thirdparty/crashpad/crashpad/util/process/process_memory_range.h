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

#ifndef CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_RANGE_H_
#define CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_RANGE_H_

#include <sys/types.h>

#include <string>

#include "base/macros.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/numeric/checked_vm_address_range.h"
#include "util/process/process_memory.h"

namespace crashpad {

//! \brief Provides range protected access to the memory of another process.
class ProcessMemoryRange {
 public:
  ProcessMemoryRange();
  ~ProcessMemoryRange();

  //! \brief Initializes this object.
  //!
  //! One of the Initialize methods must be successfully called on this object
  //! before calling any other.
  //!
  //! \param[in] memory The memory reader to delegate to.
  //! \param[in] is_64_bit Whether the target process is 64-bit.
  //! \param[in] base The base address of the initial range.
  //! \param[in] size The size of the initial range.
  //! \return `true` on success. `false` on failure with a message logged.
  bool Initialize(const ProcessMemory* memory,
                  bool is_64_bit,
                  VMAddress base,
                  VMSize size);

  //! \brief Initializes this object with the maximum range for the address
  //!     space.
  //!
  //! One of the Initialize methods must be successfully called on this object
  //! before calling any other.
  //!
  //! \param[in] memory The memory reader to delegate to.
  //! \param[in] is_64_bit Whether the target process is 64-bit.
  bool Initialize(const ProcessMemory* memory, bool is_64_bit);

  //! \brief Initializes this object from an existing memory range.
  //!
  //! One of the Initialize methods must be successfully called on this object
  //! before calling any other.
  //!
  //! \param[in] other The memory range object to initialize from.
  //! \return `true` on success. `false` on failure with a message logged.
  bool Initialize(const ProcessMemoryRange& other);

  //! \brief Returns whether the range is part of a 64-bit address space.
  bool Is64Bit() const { return range_.Is64Bit(); }

  //! \brief Returns the base address of the range.
  VMAddress Base() const { return range_.Base(); }

  //! \brief Returns the size of the range.
  VMSize Size() const { return range_.Size(); }

  //! \brief Shrinks the range to the new base and size.
  //!
  //! The new range must be contained within the existing range for this object.
  //!
  //! \param[in] base The new base of the range.
  //! \param[in] size The new size of the range.
  //! \return `true` on success. `false` on failure with a message logged.
  bool RestrictRange(VMAddress base, VMSize size);

  //! \brief Copies memory from the target process into a caller-provided buffer
  //!     in the current process.
  //!
  //! \param[in] address The address, in the target process' address space, of
  //!     the memory region to copy.
  //! \param[in] size The size, in bytes, of the memory region to copy.
  //!     \a buffer must be at least this size.
  //! \param[out] buffer The buffer into which the contents of the other
  //!     process' memory will be copied.
  //!
  //! \return `true` on success, with \a buffer filled appropriately. `false` on
  //!     failure, with a message logged.
  bool Read(VMAddress address, size_t size, void* buffer) const;

  //! \brief Reads a `NUL`-terminated C string from the target process into a
  //!     string in the current process.
  //!
  //! \param[in] address The address, in the target process’s address space, of
  //!     the string to copy.
  //! \param[in] size The maximum number of bytes to read. The string is
  //!     required to be `NUL`-terminated within this many bytes.
  //! \param[out] string The string read from the other process.
  //!
  //! \return `true` on success, with \a string set appropriately. `false` on
  //!     failure, with a message logged. Failures can occur, for example, when
  //!     a `NUL` terminator is not found within \a size bytes, or when
  //!     encountering unmapped or unreadable pages.
  bool ReadCStringSizeLimited(VMAddress address,
                              size_t size,
                              std::string* string) const;

 private:
  const ProcessMemory* memory_;  // weak
  CheckedVMAddressRange range_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessMemoryRange);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_PROCESS_PROCESS_MEMORY_RANGE_H_
