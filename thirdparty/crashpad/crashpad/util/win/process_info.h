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

#ifndef CRASHPAD_UTIL_WIN_PROCESS_INFO_H_
#define CRASHPAD_UTIL_WIN_PROCESS_INFO_H_

#include <windows.h>
#include <sys/types.h>

#include <string>
#include <vector>

#include "base/macros.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/numeric/checked_range.h"
#include "util/stdlib/aligned_allocator.h"
#include "util/win/address_types.h"

namespace crashpad {

//! \brief Gathers information about a process given its `HANDLE`. This consists
//!     primarily of information stored in the Process Environment Block.
class ProcessInfo {
 public:
  //! \brief The return type of MemoryInfo(), for convenience.
  using MemoryBasicInformation64Vector =
      AlignedVector<MEMORY_BASIC_INFORMATION64>;

  //! \brief Contains information about a module loaded into a process.
  struct Module {
    Module();
    ~Module();

    //! \brief The pathname used to load the module from disk.
    std::wstring name;

    //! \brief The base address of the loaded DLL.
    WinVMAddress dll_base;

    //! \brief The size of the module.
    WinVMSize size;

    //! \brief The module's timestamp.
    time_t timestamp;
  };

  struct Handle {
    Handle();
    ~Handle();

    //! \brief A string representation of the handle's type.
    std::wstring type_name;

    //! \brief The handle's value.
    int handle;

    //! \brief The attributes for the handle, e.g. `OBJ_INHERIT`,
    //!     `OBJ_CASE_INSENSITIVE`, etc.
    uint32_t attributes;

    //! \brief The `ACCESS_MASK` for the handle in this process.
    //!
    //! See
    //! https://blogs.msdn.microsoft.com/openspecification/2010/04/01/about-the-access_mask-structure/
    //! for more information.
    uint32_t granted_access;

    //! \brief The number of kernel references to the object that this handle
    //!     refers to.
    uint32_t pointer_count;

    //! \brief The number of open handles to the object that this handle refers
    //!     to.
    uint32_t handle_count;
  };

  ProcessInfo();
  ~ProcessInfo();

  //! \brief Initializes this object with information about the given
  //!     \a process.
  //!
  //! This method must be called successfully prior to calling any other
  //! method in this class. This method may only be called once.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool Initialize(HANDLE process);

  //! \return `true` if the target process is a 64-bit process.
  bool Is64Bit() const;

  //! \return `true` if the target process is running on the Win32-on-Win64
  //!     subsystem.
  bool IsWow64() const;

  //! \return The target process's process ID.
  pid_t ProcessID() const;

  //! \return The target process's parent process ID.
  pid_t ParentProcessID() const;

  //! \return The command line from the target process's Process Environment
  //!     Block.
  bool CommandLine(std::wstring* command_line) const;

  //! \brief Gets the address and size of the process's Process Environment
  //!     Block.
  //!
  //! \param[out] peb_address The address of the Process Environment Block.
  //! \param[out] peb_size The size of the Process Environment Block.
  void Peb(WinVMAddress* peb_address, WinVMSize* peb_size) const;

  //! \brief Retrieves the modules loaded into the target process.
  //!
  //! The modules are enumerated in initialization order as detailed in the
  //!     Process Environment Block. The main executable will always be the
  //!     first element.
  bool Modules(std::vector<Module>* modules) const;

  //! \brief Retrieves information about all pages mapped into the process.
  const MemoryBasicInformation64Vector& MemoryInfo() const;

  //! \brief Given a range to be read from the target process, returns a vector
  //!     of ranges, representing the readable portions of the original range.
  //!
  //! \param[in] range The range being identified.
  //!
  //! \return A vector of ranges corresponding to the portion of \a range that
  //!     is readable based on the memory map.
  std::vector<CheckedRange<WinVMAddress, WinVMSize>> GetReadableRanges(
      const CheckedRange<WinVMAddress, WinVMSize>& range) const;

  //! \brief Given a range in the target process, determines if the entire range
  //!     is readable.
  //!
  //! \param[in] range The range being inspected.
  //!
  //! \return `true` if the range is fully readable, otherwise `false` with a
  //!     message logged.
  bool LoggingRangeIsFullyReadable(
      const CheckedRange<WinVMAddress, WinVMSize>& range) const;

  //! \brief Retrieves information about open handles in the target process.
  const std::vector<Handle>& Handles() const;

 private:
  template <class Traits>
  friend bool GetProcessBasicInformation(HANDLE process,
                                         bool is_wow64,
                                         ProcessInfo* process_info,
                                         WinVMAddress* peb_address,
                                         WinVMSize* peb_size);
  template <class Traits>
  friend bool ReadProcessData(HANDLE process,
                              WinVMAddress peb_address_vmaddr,
                              ProcessInfo* process_info);

  friend bool ReadMemoryInfo(HANDLE process,
                             bool is_64_bit,
                             ProcessInfo* process_info);

  // This function is best-effort under low memory conditions.
  std::vector<Handle> BuildHandleVector(HANDLE process) const;

  pid_t process_id_;
  pid_t inherited_from_process_id_;
  HANDLE process_;
  std::wstring command_line_;
  WinVMAddress peb_address_;
  WinVMSize peb_size_;
  std::vector<Module> modules_;

  // memory_info_ is a MemoryBasicInformation64Vector instead of a
  // std::vector<MEMORY_BASIC_INFORMATION64> because MEMORY_BASIC_INFORMATION64
  // is declared with __declspec(align(16)), but std::vector<> does not maintain
  // this alignment on 32-bit x86. clang-cl (but not MSVC cl) takes advantage of
  // the presumed alignment and emits SSE instructions that require aligned
  // storage. clang-cl should relax (unfortunately), but in the mean time, this
  // provides aligned storage. See https://crbug.com/564691 and
  // https://llvm.org/PR25779.
  //
  // TODO(mark): Remove this workaround when https://llvm.org/PR25779 is fixed
  // and the fix is present in the clang-cl that compiles this code.
  MemoryBasicInformation64Vector memory_info_;

  // Handles() is logically const, but updates this member on first retrieval.
  // See https://crashpad.chromium.org/bug/9.
  mutable std::vector<Handle> handles_;

  bool is_64_bit_;
  bool is_wow64_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessInfo);
};

//! \brief Given a memory map of a process, and a range to be read from the
//!     target process, returns a vector of ranges, representing the readable
//!     portions of the original range.
//!
//! This is a free function for testing, but prefer
//! ProcessInfo::GetReadableRanges().
std::vector<CheckedRange<WinVMAddress, WinVMSize>> GetReadableRangesOfMemoryMap(
    const CheckedRange<WinVMAddress, WinVMSize>& range,
    const ProcessInfo::MemoryBasicInformation64Vector& memory_info);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_PROCESS_INFO_H_
