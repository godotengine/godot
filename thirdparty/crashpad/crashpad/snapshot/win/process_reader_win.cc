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

#include "snapshot/win/process_reader_win.h"

#include <string.h>
#include <winternl.h>

#include <memory>

#include "base/numerics/safe_conversions.h"
#include "base/strings/stringprintf.h"
#include "util/misc/capture_context.h"
#include "util/misc/time.h"
#include "util/win/nt_internals.h"
#include "util/win/ntstatus_logging.h"
#include "util/win/process_structs.h"
#include "util/win/scoped_handle.h"

namespace crashpad {

namespace {

// Gets a pointer to the process information structure after a given one, or
// null when iteration is complete, assuming they've been retrieved in a block
// via NtQuerySystemInformation().
template <class Traits>
process_types::SYSTEM_PROCESS_INFORMATION<Traits>* NextProcess(
    process_types::SYSTEM_PROCESS_INFORMATION<Traits>* process) {
  ULONG offset = process->NextEntryOffset;
  if (offset == 0)
    return nullptr;
  return reinterpret_cast<process_types::SYSTEM_PROCESS_INFORMATION<Traits>*>(
      reinterpret_cast<uint8_t*>(process) + offset);
}

//! \brief Retrieves the SYSTEM_PROCESS_INFORMATION for a given process.
//!
//! The returned pointer points into the memory block stored by \a buffer.
//! Ownership of \a buffer is transferred to the caller.
//!
//! \return Pointer to the process' data, or nullptr if it was not found or on
//!     error. On error, a message will be logged.
template <class Traits>
process_types::SYSTEM_PROCESS_INFORMATION<Traits>* GetProcessInformation(
    HANDLE process_handle,
    std::unique_ptr<uint8_t[]>* buffer) {
  ULONG buffer_size = 16384;
  ULONG actual_size;
  buffer->reset(new uint8_t[buffer_size]);
  NTSTATUS status;
  // This must be in retry loop, as we're racing with process creation on the
  // system to find a buffer large enough to hold all process information.
  for (int tries = 0; tries < 20; ++tries) {
    status = crashpad::NtQuerySystemInformation(
        SystemProcessInformation,
        reinterpret_cast<void*>(buffer->get()),
        buffer_size,
        &actual_size);
    if (status == STATUS_BUFFER_TOO_SMALL ||
        status == STATUS_INFO_LENGTH_MISMATCH) {
      DCHECK_GT(actual_size, buffer_size);

      // Add a little extra to try to avoid an additional loop iteration. We're
      // racing with system-wide process creation between here and the next call
      // to NtQuerySystemInformation().
      buffer_size = actual_size + 4096;

      // Free the old buffer before attempting to allocate a new one.
      buffer->reset();

      buffer->reset(new uint8_t[buffer_size]);
    } else {
      break;
    }
  }

  if (!NT_SUCCESS(status)) {
    NTSTATUS_LOG(ERROR, status) << "NtQuerySystemInformation";
    return nullptr;
  }

  DCHECK_LE(actual_size, buffer_size);

  process_types::SYSTEM_PROCESS_INFORMATION<Traits>* process =
      reinterpret_cast<process_types::SYSTEM_PROCESS_INFORMATION<Traits>*>(
          buffer->get());
  DWORD process_id = GetProcessId(process_handle);
  for (;;) {
    if (process->UniqueProcessId == process_id)
      return process;
    process = NextProcess(process);
    if (!process)
      break;
  }

  LOG(ERROR) << "process " << process_id << " not found";
  return nullptr;
}

template <class Traits>
HANDLE OpenThread(
    const process_types::SYSTEM_THREAD_INFORMATION<Traits>& thread_info) {
  HANDLE handle;
  ACCESS_MASK query_access =
      THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME | THREAD_QUERY_INFORMATION;
  OBJECT_ATTRIBUTES object_attributes;
  InitializeObjectAttributes(&object_attributes, nullptr, 0, nullptr, nullptr);
  NTSTATUS status = crashpad::NtOpenThread(
      &handle, query_access, &object_attributes, &thread_info.ClientId);
  if (!NT_SUCCESS(status)) {
    NTSTATUS_LOG(ERROR, status) << "NtOpenThread";
    return nullptr;
  }
  return handle;
}

// It's necessary to suspend the thread to grab CONTEXT. SuspendThread has a
// side-effect of returning the SuspendCount of the thread on success, so we
// fill out these two pieces of semi-unrelated data in the same function.
template <class Traits>
bool FillThreadContextAndSuspendCount(HANDLE thread_handle,
                                      ProcessReaderWin::Thread* thread,
                                      ProcessSuspensionState suspension_state,
                                      bool is_64_reading_32) {
  // Don't suspend the thread if it's this thread. This is really only for test
  // binaries, as we won't be walking ourselves, in general.
  bool is_current_thread = thread->id ==
                           reinterpret_cast<process_types::TEB<Traits>*>(
                               NtCurrentTeb())->ClientId.UniqueThread;

  if (is_current_thread) {
    DCHECK(suspension_state == ProcessSuspensionState::kRunning);
    thread->suspend_count = 0;
    DCHECK(!is_64_reading_32);
    CaptureContext(&thread->context.native);
  } else {
    DWORD previous_suspend_count = SuspendThread(thread_handle);
    if (previous_suspend_count == static_cast<DWORD>(-1)) {
      PLOG(ERROR) << "SuspendThread";
      return false;
    }
    DCHECK(previous_suspend_count > 0 ||
           suspension_state == ProcessSuspensionState::kRunning);
    thread->suspend_count =
        previous_suspend_count -
        (suspension_state == ProcessSuspensionState::kSuspended ? 1 : 0);

    memset(&thread->context, 0, sizeof(thread->context));
#if defined(ARCH_CPU_32_BITS)
    const bool is_native = true;
#elif defined(ARCH_CPU_64_BITS)
    const bool is_native = !is_64_reading_32;
    if (is_64_reading_32) {
      thread->context.wow64.ContextFlags = CONTEXT_ALL;
      if (!Wow64GetThreadContext(thread_handle, &thread->context.wow64)) {
        PLOG(ERROR) << "Wow64GetThreadContext";
        return false;
      }
    }
#endif
    if (is_native) {
      thread->context.native.ContextFlags = CONTEXT_ALL;
      if (!GetThreadContext(thread_handle, &thread->context.native)) {
        PLOG(ERROR) << "GetThreadContext";
        return false;
      }
    }

    if (!ResumeThread(thread_handle)) {
      PLOG(ERROR) << "ResumeThread";
      return false;
    }
  }

  return true;
}

}  // namespace

ProcessReaderWin::Thread::Thread()
    : context(),
      id(0),
      teb_address(0),
      teb_size(0),
      stack_region_address(0),
      stack_region_size(0),
      suspend_count(0),
      priority_class(0),
      priority(0) {
}

ProcessReaderWin::ProcessReaderWin()
    : process_(INVALID_HANDLE_VALUE),
      process_info_(),
      threads_(),
      modules_(),
      suspension_state_(),
      initialized_threads_(false),
      initialized_() {
}

ProcessReaderWin::~ProcessReaderWin() {
}

bool ProcessReaderWin::Initialize(HANDLE process,
                                  ProcessSuspensionState suspension_state) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  process_ = process;
  suspension_state_ = suspension_state;
  process_info_.Initialize(process);

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ProcessReaderWin::ReadMemory(WinVMAddress at,
                                  WinVMSize num_bytes,
                                  void* into) const {
  if (num_bytes == 0)
    return 0;

  SIZE_T bytes_read;
  if (!ReadProcessMemory(process_,
                         reinterpret_cast<void*>(at),
                         into,
                         base::checked_cast<SIZE_T>(num_bytes),
                         &bytes_read) ||
      num_bytes != bytes_read) {
    PLOG(ERROR) << "ReadMemory at 0x" << std::hex << at << std::dec << " of "
                << num_bytes << " bytes failed";
    return false;
  }
  return true;
}

WinVMSize ProcessReaderWin::ReadAvailableMemory(WinVMAddress at,
                                                WinVMSize num_bytes,
                                                void* into) const {
  if (num_bytes == 0)
    return 0;

  auto ranges = process_info_.GetReadableRanges(
      CheckedRange<WinVMAddress, WinVMSize>(at, num_bytes));

  // We only read up until the first unavailable byte, so we only read from the
  // first range. If we have no ranges, then no bytes were accessible anywhere
  // in the range.
  if (ranges.empty()) {
    LOG(ERROR) << base::StringPrintf(
        "range at 0x%llx, size 0x%llx completely inaccessible", at, num_bytes);
    return 0;
  }

  // If the start address was adjusted, we couldn't read even the first
  // requested byte.
  if (ranges.front().base() != at) {
    LOG(ERROR) << base::StringPrintf(
        "start of range at 0x%llx, size 0x%llx inaccessible", at, num_bytes);
    return 0;
  }

  DCHECK_LE(ranges.front().size(), num_bytes);

  // If we fail on a normal read, then something went very wrong.
  if (!ReadMemory(ranges.front().base(), ranges.front().size(), into))
    return 0;

  return ranges.front().size();
}

bool ProcessReaderWin::StartTime(timeval* start_time) const {
  FILETIME creation, exit, kernel, user;
  if (!GetProcessTimes(process_, &creation, &exit, &kernel, &user)) {
    PLOG(ERROR) << "GetProcessTimes";
    return false;
  }
  *start_time = FiletimeToTimevalEpoch(creation);
  return true;
}

bool ProcessReaderWin::CPUTimes(timeval* user_time,
                                timeval* system_time) const {
  FILETIME creation, exit, kernel, user;
  if (!GetProcessTimes(process_, &creation, &exit, &kernel, &user)) {
    PLOG(ERROR) << "GetProcessTimes";
    return false;
  }
  *user_time = FiletimeToTimevalInterval(user);
  *system_time = FiletimeToTimevalInterval(kernel);
  return true;
}

const std::vector<ProcessReaderWin::Thread>& ProcessReaderWin::Threads() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (initialized_threads_)
    return threads_;

  initialized_threads_ = true;

#if defined(ARCH_CPU_64_BITS)
  ReadThreadData<process_types::internal::Traits64>(process_info_.IsWow64());
#else
  ReadThreadData<process_types::internal::Traits32>(false);
#endif

  return threads_;
}

const std::vector<ProcessInfo::Module>& ProcessReaderWin::Modules() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (!process_info_.Modules(&modules_)) {
    LOG(ERROR) << "couldn't retrieve modules";
  }

  return modules_;
}

const ProcessInfo& ProcessReaderWin::GetProcessInfo() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_info_;
}

void ProcessReaderWin::DecrementThreadSuspendCounts(uint64_t except_thread_id) {
  Threads();
  for (auto& thread : threads_) {
    if (thread.id != except_thread_id) {
      DCHECK_GT(thread.suspend_count, 0u);
      --thread.suspend_count;
    }
  }
}

template <class Traits>
void ProcessReaderWin::ReadThreadData(bool is_64_reading_32) {
  DCHECK(threads_.empty());

  std::unique_ptr<uint8_t[]> buffer;
  process_types::SYSTEM_PROCESS_INFORMATION<Traits>* process_information =
      GetProcessInformation<Traits>(process_, &buffer);
  if (!process_information)
    return;

  for (unsigned long i = 0; i < process_information->NumberOfThreads; ++i) {
    const process_types::SYSTEM_THREAD_INFORMATION<Traits>& thread_info =
        process_information->Threads[i];
    ProcessReaderWin::Thread thread;
    thread.id = thread_info.ClientId.UniqueThread;

    ScopedKernelHANDLE thread_handle(OpenThread(thread_info));
    if (!thread_handle.is_valid())
      continue;

    if (!FillThreadContextAndSuspendCount<Traits>(thread_handle.get(),
                                                  &thread,
                                                  suspension_state_,
                                                  is_64_reading_32)) {
      continue;
    }

    // TODO(scottmg): I believe we could reverse engineer the PriorityClass from
    // the Priority, BasePriority, and
    // https://msdn.microsoft.com/library/ms685100.aspx. MinidumpThreadWriter
    // doesn't handle it yet in any case, so investigate both of those at the
    // same time if it's useful.
    thread.priority_class = NORMAL_PRIORITY_CLASS;

    thread.priority = thread_info.Priority;

    process_types::THREAD_BASIC_INFORMATION<Traits> thread_basic_info;
    NTSTATUS status = crashpad::NtQueryInformationThread(
        thread_handle.get(),
        static_cast<THREADINFOCLASS>(ThreadBasicInformation),
        &thread_basic_info,
        sizeof(thread_basic_info),
        nullptr);
    if (!NT_SUCCESS(status)) {
      NTSTATUS_LOG(ERROR, status) << "NtQueryInformationThread";
      continue;
    }

    // Read the TIB (Thread Information Block) which is the first element of the
    // TEB, for its stack fields.
    process_types::NT_TIB<Traits> tib;
    thread.teb_address = thread_basic_info.TebBaseAddress;
    thread.teb_size = sizeof(process_types::TEB<Traits>);
    if (ReadMemory(thread.teb_address, sizeof(tib), &tib)) {
      WinVMAddress base = 0;
      WinVMAddress limit = 0;
      // If we're reading a WOW64 process, then the TIB we just retrieved is the
      // x64 one. The first word of the x64 TIB points at the x86 TIB. See
      // https://msdn.microsoft.com/library/dn424783.aspx.
      if (is_64_reading_32) {
        process_types::NT_TIB<process_types::internal::Traits32> tib32;
        thread.teb_address = tib.Wow64Teb;
        thread.teb_size =
            sizeof(process_types::TEB<process_types::internal::Traits32>);
        if (ReadMemory(thread.teb_address, sizeof(tib32), &tib32)) {
          base = tib32.StackBase;
          limit = tib32.StackLimit;
        }
      } else {
        base = tib.StackBase;
        limit = tib.StackLimit;
      }

      // Note, "backwards" because of direction of stack growth.
      thread.stack_region_address = limit;
      if (limit > base) {
        LOG(ERROR) << "invalid stack range: " << base << " - " << limit;
        thread.stack_region_size = 0;
      } else {
        thread.stack_region_size = base - limit;
      }
    }
    threads_.push_back(thread);
  }
}

}  // namespace crashpad
