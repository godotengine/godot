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

#include "util/win/process_info.h"

#include <stddef.h>
#include <winternl.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

#include "base/logging.h"
#include "base/memory/free_deleter.h"
#include "base/process/memory.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "util/misc/from_pointer_cast.h"
#include "util/numeric/safe_assignment.h"
#include "util/win/get_function.h"
#include "util/win/handle.h"
#include "util/win/nt_internals.h"
#include "util/win/ntstatus_logging.h"
#include "util/win/process_structs.h"
#include "util/win/scoped_handle.h"

namespace crashpad {

namespace {

using UniqueMallocPtr = std::unique_ptr<uint8_t[], base::FreeDeleter>;

UniqueMallocPtr UncheckedAllocate(size_t size) {
  void* raw_ptr = nullptr;
  if (!base::UncheckedMalloc(size, &raw_ptr))
    return UniqueMallocPtr();

  return UniqueMallocPtr(new (raw_ptr) uint8_t[size]);
}

NTSTATUS NtQueryInformationProcess(HANDLE process_handle,
                                   PROCESSINFOCLASS process_information_class,
                                   PVOID process_information,
                                   ULONG process_information_length,
                                   PULONG return_length) {
  static const auto nt_query_information_process =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtQueryInformationProcess);
  return nt_query_information_process(process_handle,
                                      process_information_class,
                                      process_information,
                                      process_information_length,
                                      return_length);
}

bool IsProcessWow64(HANDLE process_handle) {
  static const auto is_wow64_process =
      GET_FUNCTION(L"kernel32.dll", ::IsWow64Process);
  if (!is_wow64_process)
    return false;
  BOOL is_wow64;
  if (!is_wow64_process(process_handle, &is_wow64)) {
    PLOG(ERROR) << "IsWow64Process";
    return false;
  }
  return !!is_wow64;
}

template <class T>
bool ReadUnicodeString(HANDLE process,
                       const process_types::UNICODE_STRING<T>& us,
                       std::wstring* result) {
  if (us.Length == 0) {
    result->clear();
    return true;
  }
  DCHECK_EQ(us.Length % sizeof(wchar_t), 0u);
  result->resize(us.Length / sizeof(wchar_t));
  SIZE_T bytes_read;
  if (!ReadProcessMemory(
          process,
          reinterpret_cast<const void*>(static_cast<uintptr_t>(us.Buffer)),
          &result->operator[](0),
          us.Length,
          &bytes_read)) {
    PLOG(ERROR) << "ReadProcessMemory UNICODE_STRING";
    return false;
  }
  if (bytes_read != us.Length) {
    LOG(ERROR) << "ReadProcessMemory UNICODE_STRING incorrect size";
    return false;
  }
  return true;
}

template <class T>
bool ReadStruct(HANDLE process, WinVMAddress at, T* into) {
  SIZE_T bytes_read;
  if (!ReadProcessMemory(process,
                         reinterpret_cast<const void*>(at),
                         into,
                         sizeof(T),
                         &bytes_read)) {
    // We don't have a name for the type we're reading, so include the signature
    // to get the type of T.
    PLOG(ERROR) << "ReadProcessMemory " << __FUNCSIG__;
    return false;
  }
  if (bytes_read != sizeof(T)) {
    LOG(ERROR) << "ReadProcessMemory " << __FUNCSIG__ << " incorrect size";
    return false;
  }
  return true;
}

bool RegionIsAccessible(const MEMORY_BASIC_INFORMATION64& memory_info) {
  return memory_info.State == MEM_COMMIT &&
         (memory_info.Protect & PAGE_NOACCESS) == 0 &&
         (memory_info.Protect & PAGE_GUARD) == 0;
}

MEMORY_BASIC_INFORMATION64 MemoryBasicInformationToMemoryBasicInformation64(
    const MEMORY_BASIC_INFORMATION& mbi) {
  MEMORY_BASIC_INFORMATION64 mbi64 = {0};
  mbi64.BaseAddress = FromPointerCast<ULONGLONG>(mbi.BaseAddress);
  mbi64.AllocationBase = reinterpret_cast<ULONGLONG>(mbi.AllocationBase);
  mbi64.AllocationProtect = mbi.AllocationProtect;
  mbi64.RegionSize = mbi.RegionSize;
  mbi64.State = mbi.State;
  mbi64.Protect = mbi.Protect;
  mbi64.Type = mbi.Type;
  return mbi64;
}

// NtQueryObject with a retry for size mismatch as well as a minimum size to
// retrieve (and expect).
std::unique_ptr<uint8_t[]> QueryObject(
    HANDLE handle,
    OBJECT_INFORMATION_CLASS object_information_class,
    ULONG minimum_size) {
  ULONG size = minimum_size;
  ULONG return_length;
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[size]);
  NTSTATUS status = crashpad::NtQueryObject(
      handle, object_information_class, buffer.get(), size, &return_length);
  if (status == STATUS_INFO_LENGTH_MISMATCH) {
    DCHECK_GT(return_length, size);
    size = return_length;

    // Free the old buffer before attempting to allocate a new one.
    buffer.reset();

    buffer.reset(new uint8_t[size]);
    status = crashpad::NtQueryObject(
        handle, object_information_class, buffer.get(), size, &return_length);
  }

  if (!NT_SUCCESS(status)) {
    NTSTATUS_LOG(ERROR, status) << "NtQueryObject";
    return nullptr;
  }

  DCHECK_LE(return_length, size);
  DCHECK_GE(return_length, minimum_size);
  return buffer;
}

}  // namespace

template <class Traits>
bool GetProcessBasicInformation(HANDLE process,
                                bool is_wow64,
                                ProcessInfo* process_info,
                                WinVMAddress* peb_address,
                                WinVMSize* peb_size) {
  ULONG bytes_returned;
  process_types::PROCESS_BASIC_INFORMATION<Traits> process_basic_information;
  NTSTATUS status =
      crashpad::NtQueryInformationProcess(process,
                                          ProcessBasicInformation,
                                          &process_basic_information,
                                          sizeof(process_basic_information),
                                          &bytes_returned);
  if (!NT_SUCCESS(status)) {
    NTSTATUS_LOG(ERROR, status) << "NtQueryInformationProcess";
    return false;
  }
  if (bytes_returned != sizeof(process_basic_information)) {
    LOG(ERROR) << "NtQueryInformationProcess incorrect size";
    return false;
  }

  // API functions (e.g. OpenProcess) take only a DWORD, so there's no sense in
  // maintaining the top bits.
  process_info->process_id_ =
      static_cast<DWORD>(process_basic_information.UniqueProcessId);
  process_info->inherited_from_process_id_ = static_cast<DWORD>(
      process_basic_information.InheritedFromUniqueProcessId);

  // We now want to read the PEB to gather the rest of our information. The
  // PebBaseAddress as returned above is what we want for 64-on-64 and 32-on-32,
  // but for Wow64, we want to read the 32 bit PEB (a Wow64 process has both).
  // The address of this is found by a second call to NtQueryInformationProcess.
  if (!is_wow64) {
    *peb_address = process_basic_information.PebBaseAddress;
    *peb_size = sizeof(process_types::PEB<Traits>);
  } else {
    ULONG_PTR wow64_peb_address;
    status = crashpad::NtQueryInformationProcess(process,
                                                 ProcessWow64Information,
                                                 &wow64_peb_address,
                                                 sizeof(wow64_peb_address),
                                                 &bytes_returned);
    if (!NT_SUCCESS(status)) {
      NTSTATUS_LOG(ERROR, status), "NtQueryInformationProcess";
      return false;
    }
    if (bytes_returned != sizeof(wow64_peb_address)) {
      LOG(ERROR) << "NtQueryInformationProcess incorrect size";
      return false;
    }
    *peb_address = wow64_peb_address;
    *peb_size = sizeof(process_types::PEB<process_types::internal::Traits32>);
  }

  return true;
}

template <class Traits>
bool ReadProcessData(HANDLE process,
                     WinVMAddress peb_address_vmaddr,
                     ProcessInfo* process_info) {
  typename Traits::Pointer peb_address;
  if (!AssignIfInRange(&peb_address, peb_address_vmaddr)) {
    LOG(ERROR) << base::StringPrintf("peb address 0x%llx out of range",
                                     peb_address_vmaddr);
    return false;
  }

  // Try to read the process environment block.
  process_types::PEB<Traits> peb;
  if (!ReadStruct(process, peb_address, &peb))
    return false;

  process_types::RTL_USER_PROCESS_PARAMETERS<Traits> process_parameters;
  if (!ReadStruct(process, peb.ProcessParameters, &process_parameters))
    return false;

  if (!ReadUnicodeString(process,
                         process_parameters.CommandLine,
                         &process_info->command_line_)) {
    return false;
  }

  process_types::PEB_LDR_DATA<Traits> peb_ldr_data;
  if (!ReadStruct(process, peb.Ldr, &peb_ldr_data))
    return false;

  process_types::LDR_DATA_TABLE_ENTRY<Traits> ldr_data_table_entry;
  ProcessInfo::Module module;

  // Walk the PEB LDR structure (doubly-linked list) to get the list of loaded
  // modules. We use this method rather than EnumProcessModules to get the
  // modules in load order rather than memory order. Notably, this includes the
  // main executable as the first element.
  typename Traits::Pointer last = peb_ldr_data.InLoadOrderModuleList.Blink;
  for (typename Traits::Pointer cur = peb_ldr_data.InLoadOrderModuleList.Flink;;
       cur = ldr_data_table_entry.InLoadOrderLinks.Flink) {
    // |cur| is the pointer to the LIST_ENTRY embedded in the
    // LDR_DATA_TABLE_ENTRY, in the target process's address space. So we need
    // to read from the target, and also offset back to the beginning of the
    // structure.
    if (!ReadStruct(process,
                    static_cast<WinVMAddress>(cur) -
                        offsetof(process_types::LDR_DATA_TABLE_ENTRY<Traits>,
                                 InLoadOrderLinks),
                    &ldr_data_table_entry)) {
      break;
    }
    // TODO(scottmg): Capture Checksum, etc. too?
    if (!ReadUnicodeString(
            process, ldr_data_table_entry.FullDllName, &module.name)) {
      module.name = L"???";
    }
    module.dll_base = ldr_data_table_entry.DllBase;
    module.size = ldr_data_table_entry.SizeOfImage;
    module.timestamp = ldr_data_table_entry.TimeDateStamp;
    process_info->modules_.push_back(module);
    if (cur == last)
      break;
  }

  return true;
}

bool ReadMemoryInfo(HANDLE process, bool is_64_bit, ProcessInfo* process_info) {
  DCHECK(process_info->memory_info_.empty());

  constexpr WinVMAddress min_address = 0;
  // We can't use GetSystemInfo() to get the address space range for another
  // process. VirtualQueryEx() will fail with ERROR_INVALID_PARAMETER if the
  // address is above the highest memory address accessible to the process, so
  // we just probe the entire potential range (2^32 for x86, or 2^64 for x64).
  const WinVMAddress max_address = is_64_bit
                                       ? std::numeric_limits<uint64_t>::max()
                                       : std::numeric_limits<uint32_t>::max();
  MEMORY_BASIC_INFORMATION memory_basic_information;
  for (WinVMAddress address = min_address; address <= max_address;
       address += memory_basic_information.RegionSize) {
    size_t result = VirtualQueryEx(process,
                                   reinterpret_cast<void*>(address),
                                   &memory_basic_information,
                                   sizeof(memory_basic_information));
    if (result == 0) {
      if (GetLastError() == ERROR_INVALID_PARAMETER)
        break;
      PLOG(ERROR) << "VirtualQueryEx";
      return false;
    }

    process_info->memory_info_.push_back(
        MemoryBasicInformationToMemoryBasicInformation64(
            memory_basic_information));

    if (memory_basic_information.RegionSize == 0) {
      LOG(ERROR) << "RegionSize == 0";
      return false;
    }
  }

  return true;
}

std::vector<ProcessInfo::Handle> ProcessInfo::BuildHandleVector(
    HANDLE process) const {
  ULONG buffer_size = 2 * 1024 * 1024;
  // Typically if the buffer were too small, STATUS_INFO_LENGTH_MISMATCH would
  // return the correct size in the final argument, but it does not for
  // SystemExtendedHandleInformation, so we loop and attempt larger sizes.
  NTSTATUS status;
  ULONG returned_length;
  UniqueMallocPtr buffer;
  for (int tries = 0; tries < 5; ++tries) {
    buffer.reset();
    buffer = UncheckedAllocate(buffer_size);
    if (!buffer) {
      LOG(ERROR) << "UncheckedAllocate";
      return std::vector<Handle>();
    }

    status = crashpad::NtQuerySystemInformation(
        static_cast<SYSTEM_INFORMATION_CLASS>(SystemExtendedHandleInformation),
        buffer.get(),
        buffer_size,
        &returned_length);
    if (NT_SUCCESS(status) || status != STATUS_INFO_LENGTH_MISMATCH)
      break;

    buffer_size *= 2;
  }

  if (!NT_SUCCESS(status)) {
    NTSTATUS_LOG(ERROR, status)
        << "NtQuerySystemInformation SystemExtendedHandleInformation";
    return std::vector<Handle>();
  }

  const auto& system_handle_information_ex =
      *reinterpret_cast<process_types::SYSTEM_HANDLE_INFORMATION_EX*>(
          buffer.get());

  DCHECK_LE(offsetof(process_types::SYSTEM_HANDLE_INFORMATION_EX, Handles) +
                system_handle_information_ex.NumberOfHandles *
                    sizeof(system_handle_information_ex.Handles[0]),
            returned_length);

  std::vector<Handle> handles;

  for (size_t i = 0; i < system_handle_information_ex.NumberOfHandles; ++i) {
    const auto& handle = system_handle_information_ex.Handles[i];
    if (handle.UniqueProcessId != process_id_)
      continue;

    Handle result_handle;
    result_handle.handle = HandleToInt(handle.HandleValue);
    result_handle.attributes = handle.HandleAttributes;
    result_handle.granted_access = handle.GrantedAccess;

    // TODO(scottmg): Could special case for self.
    HANDLE dup_handle;
    if (DuplicateHandle(process,
                        handle.HandleValue,
                        GetCurrentProcess(),
                        &dup_handle,
                        0,
                        false,
                        DUPLICATE_SAME_ACCESS)) {
      // Some handles cannot be duplicated, for example, handles of type
      // EtwRegistration. If we fail to duplicate, then we can't gather any more
      // information, but include the information that we do have already.
      ScopedKernelHANDLE scoped_dup_handle(dup_handle);

      std::unique_ptr<uint8_t[]> object_basic_information_buffer =
          QueryObject(dup_handle,
                      ObjectBasicInformation,
                      sizeof(PUBLIC_OBJECT_BASIC_INFORMATION));
      if (object_basic_information_buffer) {
        PUBLIC_OBJECT_BASIC_INFORMATION* object_basic_information =
            reinterpret_cast<PUBLIC_OBJECT_BASIC_INFORMATION*>(
                object_basic_information_buffer.get());
        // The Attributes and GrantedAccess sometimes differ slightly between
        // the data retrieved in SYSTEM_HANDLE_INFORMATION_EX and
        // PUBLIC_OBJECT_TYPE_INFORMATION. We prefer the values in
        // SYSTEM_HANDLE_INFORMATION_EX because they were retrieved from the
        // target process, rather than on the duplicated handle, so don't use
        // them here.

        // Subtract one to account for our DuplicateHandle() and another for
        // NtQueryObject() while the query was being executed.
        DCHECK_GT(object_basic_information->PointerCount, 2u);
        result_handle.pointer_count =
            object_basic_information->PointerCount - 2;

        // Subtract one to account for our DuplicateHandle().
        DCHECK_GT(object_basic_information->HandleCount, 1u);
        result_handle.handle_count = object_basic_information->HandleCount - 1;
      }

      std::unique_ptr<uint8_t[]> object_type_information_buffer =
          QueryObject(dup_handle,
                      ObjectTypeInformation,
                      sizeof(PUBLIC_OBJECT_TYPE_INFORMATION));
      if (object_type_information_buffer) {
        PUBLIC_OBJECT_TYPE_INFORMATION* object_type_information =
            reinterpret_cast<PUBLIC_OBJECT_TYPE_INFORMATION*>(
                object_type_information_buffer.get());

        DCHECK_EQ(object_type_information->TypeName.Length %
                      sizeof(result_handle.type_name[0]),
                  0u);
        result_handle.type_name =
            std::wstring(object_type_information->TypeName.Buffer,
                         object_type_information->TypeName.Length /
                             sizeof(result_handle.type_name[0]));
      }
    }

    handles.push_back(result_handle);
  }
  return handles;
}

ProcessInfo::Module::Module() : name(), dll_base(0), size(0), timestamp() {
}

ProcessInfo::Module::~Module() {
}

ProcessInfo::Handle::Handle()
    : type_name(),
      handle(0),
      attributes(0),
      granted_access(0),
      pointer_count(0),
      handle_count(0) {
}

ProcessInfo::Handle::~Handle() {
}

ProcessInfo::ProcessInfo()
    : process_id_(),
      inherited_from_process_id_(),
      process_(),
      command_line_(),
      peb_address_(0),
      peb_size_(0),
      modules_(),
      memory_info_(),
      handles_(),
      is_64_bit_(false),
      is_wow64_(false),
      initialized_() {
}

ProcessInfo::~ProcessInfo() {
}

bool ProcessInfo::Initialize(HANDLE process) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  process_ = process;

  is_wow64_ = IsProcessWow64(process);

  if (is_wow64_) {
    // If it's WoW64, then it's 32-on-64.
    is_64_bit_ = false;
  } else {
    // Otherwise, it's either 32 on 32, or 64 on 64. Use GetSystemInfo() to
    // distinguish between these two cases.
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    is_64_bit_ =
        system_info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64;
  }

#if defined(ARCH_CPU_32_BITS)
  if (is_64_bit_) {
    LOG(ERROR) << "Reading x64 process from x86 process not supported";
    return false;
  }
#endif  // ARCH_CPU_32_BITS

#if defined(ARCH_CPU_64_BITS)
  bool result = GetProcessBasicInformation<process_types::internal::Traits64>(
      process, is_wow64_, this, &peb_address_, &peb_size_);
#else
  bool result = GetProcessBasicInformation<process_types::internal::Traits32>(
      process, false, this, &peb_address_, &peb_size_);
#endif  // ARCH_CPU_64_BITS

  if (!result) {
    LOG(ERROR) << "GetProcessBasicInformation failed";
    return false;
  }

  result = is_64_bit_ ? ReadProcessData<process_types::internal::Traits64>(
                            process, peb_address_, this)
                      : ReadProcessData<process_types::internal::Traits32>(
                            process, peb_address_, this);
  if (!result) {
    LOG(ERROR) << "ReadProcessData failed";
    return false;
  }

  if (!ReadMemoryInfo(process, is_64_bit_, this)) {
    LOG(ERROR) << "ReadMemoryInfo failed";
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessInfo::Is64Bit() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return is_64_bit_;
}

bool ProcessInfo::IsWow64() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return is_wow64_;
}

pid_t ProcessInfo::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_id_;
}

pid_t ProcessInfo::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return inherited_from_process_id_;
}

bool ProcessInfo::CommandLine(std::wstring* command_line) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *command_line = command_line_;
  return true;
}

void ProcessInfo::Peb(WinVMAddress* peb_address, WinVMSize* peb_size) const {
  *peb_address = peb_address_;
  *peb_size = peb_size_;
}

bool ProcessInfo::Modules(std::vector<Module>* modules) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *modules = modules_;
  return true;
}

const ProcessInfo::MemoryBasicInformation64Vector& ProcessInfo::MemoryInfo()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return memory_info_;
}

std::vector<CheckedRange<WinVMAddress, WinVMSize>>
ProcessInfo::GetReadableRanges(
    const CheckedRange<WinVMAddress, WinVMSize>& range) const {
  return GetReadableRangesOfMemoryMap(range, MemoryInfo());
}

bool ProcessInfo::LoggingRangeIsFullyReadable(
    const CheckedRange<WinVMAddress, WinVMSize>& range) const {
  const auto ranges = GetReadableRanges(range);
  if (ranges.empty()) {
    LOG(ERROR) << base::StringPrintf(
        "range at 0x%llx, size 0x%llx fully unreadable",
        range.base(),
        range.size());
    return false;
  }

  if (ranges.size() != 1 ||
      ranges[0].base() != range.base() || ranges[0].size() != range.size()) {
    LOG(ERROR) << base::StringPrintf(
        "range at 0x%llx, size 0x%llx partially unreadable",
        range.base(),
        range.size());
    return false;
  }

  return true;
}

const std::vector<ProcessInfo::Handle>& ProcessInfo::Handles() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (handles_.empty())
    handles_ = BuildHandleVector(process_);
  return handles_;
}

std::vector<CheckedRange<WinVMAddress, WinVMSize>> GetReadableRangesOfMemoryMap(
    const CheckedRange<WinVMAddress, WinVMSize>& range,
    const ProcessInfo::MemoryBasicInformation64Vector& memory_info) {
  using Range = CheckedRange<WinVMAddress, WinVMSize>;

  // Constructing Ranges and using OverlapsRange() is very, very slow in Debug
  // builds, so do a manual check in this loop. The ranges are still validated
  // by a CheckedRange before being returned.
  WinVMAddress range_base = range.base();
  WinVMAddress range_end = range.end();

  // Find all the ranges that overlap the target range, maintaining their order.
  ProcessInfo::MemoryBasicInformation64Vector overlapping;
  const size_t size = memory_info.size();

  // This loop is written in an ugly fashion to make Debug performance
  // reasonable.
  const MEMORY_BASIC_INFORMATION64* begin = &memory_info[0];
  for (size_t i = 0; i < size; ++i) {
    const MEMORY_BASIC_INFORMATION64& mi = *(begin + i);
    static_assert(std::is_same<decltype(mi.BaseAddress), WinVMAddress>::value,
                  "expected range address to be WinVMAddress");
    static_assert(std::is_same<decltype(mi.RegionSize), WinVMSize>::value,
                  "expected range size to be WinVMSize");
    WinVMAddress mi_end = mi.BaseAddress + mi.RegionSize;
    if (range_base < mi_end && mi.BaseAddress < range_end)
      overlapping.push_back(mi);
  }
  if (overlapping.empty())
    return std::vector<Range>();

  // For the first and last, trim to the boundary of the incoming range.
  MEMORY_BASIC_INFORMATION64& front = overlapping.front();
  WinVMAddress original_front_base_address = front.BaseAddress;
  front.BaseAddress = std::max(front.BaseAddress, range.base());
  front.RegionSize =
      (original_front_base_address + front.RegionSize) - front.BaseAddress;

  MEMORY_BASIC_INFORMATION64& back = overlapping.back();
  WinVMAddress back_end = back.BaseAddress + back.RegionSize;
  back.RegionSize = std::min(range.end(), back_end) - back.BaseAddress;

  // Discard all non-accessible.
  overlapping.erase(std::remove_if(overlapping.begin(),
                                   overlapping.end(),
                                   [](const MEMORY_BASIC_INFORMATION64& mbi) {
                                     return !RegionIsAccessible(mbi);
                                   }),
                    overlapping.end());
  if (overlapping.empty())
    return std::vector<Range>();

  // Convert to return type.
  std::vector<Range> as_ranges;
  for (const auto& mi : overlapping) {
    as_ranges.push_back(Range(mi.BaseAddress, mi.RegionSize));
    DCHECK(as_ranges.back().IsValid());
  }

  // Coalesce remaining regions.
  std::vector<Range> result;
  result.push_back(as_ranges[0]);
  for (size_t i = 1; i < as_ranges.size(); ++i) {
    if (result.back().end() == as_ranges[i].base()) {
      result.back().SetRange(result.back().base(),
                             result.back().size() + as_ranges[i].size());
    } else {
      result.push_back(as_ranges[i]);
    }
    DCHECK(result.back().IsValid());
  }

  return result;
}

}  // namespace crashpad
