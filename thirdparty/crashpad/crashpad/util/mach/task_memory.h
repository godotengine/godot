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

#ifndef CRASHPAD_UTIL_MACH_TASK_MEMORY_H_
#define CRASHPAD_UTIL_MACH_TASK_MEMORY_H_

#include <mach/mach.h>
#include <sys/types.h>

#include <memory>
#include <string>

#include "base/mac/scoped_mach_vm.h"
#include "base/macros.h"

namespace crashpad {

//! \brief Accesses the memory of another Mach task.
class TaskMemory {
 public:
  //! \brief A memory region mapped from another Mach task.
  //!
  //! The mapping is maintained until this object is destroyed.
  class MappedMemory {
   public:
    ~MappedMemory();

    //! \brief Returns a pointer to the data requested by the user.
    //!
    //! This is the value of the \a vm_address + \a user_offset parameters
    //! passed to the constructor, casted to `const void*`.
    const void* data() const { return data_; }

    //! \brief Reads a `NUL`-terminated C string from the mapped region.
    //!
    //! This method will read contiguous memory until a `NUL` terminator is
    //! found.
    //!
    //! \param[in] offset The offset into data() of the string to be read.
    //! \param[out] string The string, whose contents begin at data() and
    //!     continue up to a `NUL` terminator.
    //!
    //! \return `true` on success, with \a string set appropriately. If \a
    //!     offset is greater than or equal to the \a user_size constructor
    //!     parameter, or if no `NUL` terminator was found in data() after \a
    //!     offset, returns `false` with an appropriate warning logged.
    bool ReadCString(size_t offset, std::string* string) const;

   private:
    //! \brief Creates an object that owns a memory region mapped from another
    //!     Mach task.
    //!
    //! \param[in] vm_address The address in this process’ address space where
    //!     the mapping begins. This must be page-aligned.
    //! \param[in] vm_size The total size of the mapping that begins at \a
    //!     vm_address. This must be page-aligned.
    //! \param[in] user_offset The offset into the mapped region where the data
    //!     requested by the user begins. This accounts for the fact that a
    //!     mapping must be page-aligned but the user data may not be. This
    //!     parameter must be equal to or less than \a vm_size.
    //! \param[in] user_size The size of the data requested by the user. This
    //!     parameter can be used to compute the end address of user data, which
    //!     must be within the mapped region.
    MappedMemory(vm_address_t vm_address,
                 size_t vm_size,
                 size_t user_offset,
                 size_t user_size);

    base::mac::ScopedMachVM vm_;
    const void* data_;
    size_t user_size_;

    // The outer class needs to be able to call this class’ private constructor.
    friend class TaskMemory;

    DISALLOW_COPY_AND_ASSIGN(MappedMemory);
  };

  //! \param[in] task A send right to the target task’s task port. This object
  //!     does not take ownership of the send right.
  explicit TaskMemory(task_t task);

  ~TaskMemory() {}

  //! \brief Copies memory from the target task into a caller-provided buffer in
  //!     the current task.
  //!
  //! \param[in] address The address, in the target task’s address space, of the
  //!     memory region to copy.
  //! \param[in] size The size, in bytes, of the memory region to copy. \a
  //!     buffer must be at least this size.
  //! \param[out] buffer The buffer into which the contents of the other task’s
  //!     memory will be copied.
  //!
  //! \return `true` on success, with \a buffer filled appropriately. `false` on
  //!     failure, with a warning logged. Failures can occur, for example, when
  //!     encountering unmapped or unreadable pages.
  //!
  //! \sa ReadMapped()
  bool Read(mach_vm_address_t address, size_t size, void* buffer);

  //! \brief Maps memory from the target task into the current task.
  //!
  //! This interface is an alternative to Read() that does not require the
  //! caller to provide a buffer to fill. This avoids copying memory, which can
  //! offer a performance improvement.
  //!
  //! \param[in] address The address, in the target task’s address space, of the
  //!     memory region to map.
  //! \param[in] size The size, in bytes, of the memory region to map.
  //!
  //! \return On success, a MappedMemory object that provides access to the data
  //!     requested. On faliure, `nullptr`, with a warning logged. Failures can
  //!     occur, for example, when encountering unmapped or unreadable pages.
  std::unique_ptr<MappedMemory> ReadMapped(mach_vm_address_t address,
                                           size_t size);

  //! \brief Reads a `NUL`-terminated C string from the target task into a
  //!     string in the current task.
  //!
  //! The length of the string need not be known ahead of time. This method will
  //! read contiguous memory until a `NUL` terminator is found.
  //!
  //! \param[in] address The address, in the target task’s address space, of the
  //!     string to copy.
  //! \param[out] string The string read from the other task.
  //!
  //! \return `true` on success, with \a string set appropriately. `false` on
  //!     failure, with a warning logged. Failures can occur, for example, when
  //!     encountering unmapped or unreadable pages.
  //!
  //! \sa MappedMemory::ReadCString()
  bool ReadCString(mach_vm_address_t address, std::string* string);

  //! \brief Reads a `NUL`-terminated C string from the target task into a
  //!     string in the current task.
  //!
  //! \param[in] address The address, in the target task’s address space, of the
  //!     string to copy.
  //! \param[in] size The maximum number of bytes to read. The string is
  //!     required to be `NUL`-terminated within this many bytes.
  //! \param[out] string The string read from the other task.
  //!
  //! \return `true` on success, with \a string set appropriately. `false` on
  //!     failure, with a warning logged. Failures can occur, for example, when
  //!     a `NUL` terminator is not found within \a size bytes, or when
  //!     encountering unmapped or unreadable pages.
  //!
  //! \sa MappedMemory::ReadCString()
  bool ReadCStringSizeLimited(mach_vm_address_t address,
                              mach_vm_size_t size,
                              std::string* string);

 private:
  // The common internal implementation shared by the ReadCString*() methods.
  bool ReadCStringInternal(mach_vm_address_t address,
                           bool has_size,
                           mach_vm_size_t size,
                           std::string* string);

  task_t task_;  // weak

  DISALLOW_COPY_AND_ASSIGN(TaskMemory);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_TASK_MEMORY_H_
