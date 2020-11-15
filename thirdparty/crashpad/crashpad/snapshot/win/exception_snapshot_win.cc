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

#include "snapshot/win/exception_snapshot_win.h"

#include "client/crashpad_client.h"
#include "snapshot/capture_memory.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/win/cpu_context_win.h"
#include "snapshot/win/capture_memory_delegate_win.h"
#include "snapshot/win/memory_snapshot_win.h"
#include "snapshot/win/process_reader_win.h"
#include "util/win/nt_internals.h"

namespace crashpad {
namespace internal {

namespace {

#if defined(ARCH_CPU_32_BITS)
using Context32 = CONTEXT;
#elif defined(ARCH_CPU_64_BITS)
using Context32 = WOW64_CONTEXT;
#endif

#if defined(ARCH_CPU_64_BITS)
void NativeContextToCPUContext64(const CONTEXT& context_record,
                                 CPUContext* context,
                                 CPUContextUnion* context_union) {
  context->architecture = kCPUArchitectureX86_64;
  context->x86_64 = &context_union->x86_64;
  InitializeX64Context(context_record, context->x86_64);
}
#endif

void NativeContextToCPUContext32(const Context32& context_record,
                                 CPUContext* context,
                                 CPUContextUnion* context_union) {
  context->architecture = kCPUArchitectureX86;
  context->x86 = &context_union->x86;
  InitializeX86Context(context_record, context->x86);
}

}  // namespace

ExceptionSnapshotWin::ExceptionSnapshotWin()
    : ExceptionSnapshot(),
      context_union_(),
      context_(),
      codes_(),
      extra_memory_(),
      thread_id_(0),
      exception_address_(0),
      exception_flags_(0),
      exception_code_(0),
      initialized_() {
}

ExceptionSnapshotWin::~ExceptionSnapshotWin() {
}

bool ExceptionSnapshotWin::Initialize(
    ProcessReaderWin* process_reader,
    DWORD thread_id,
    WinVMAddress exception_pointers_address) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  const ProcessReaderWin::Thread* thread = nullptr;
  for (const auto& loop_thread : process_reader->Threads()) {
    if (thread_id == loop_thread.id) {
      thread = &loop_thread;
      break;
    }
  }

  if (!thread) {
    LOG(ERROR) << "thread ID " << thread_id << " not found in process";
    return false;
  } else {
    thread_id_ = thread_id;
  }

#if defined(ARCH_CPU_32_BITS)
  const bool is_64_bit = false;
#elif defined(ARCH_CPU_64_BITS)
  const bool is_64_bit = process_reader->Is64Bit();
  if (is_64_bit) {
    if (!InitializeFromExceptionPointers<EXCEPTION_RECORD64,
                                         process_types::EXCEPTION_POINTERS64>(
            process_reader,
            exception_pointers_address,
            thread_id,
            &NativeContextToCPUContext64)) {
      return false;
    }
  }
#endif
  if (!is_64_bit) {
    if (!InitializeFromExceptionPointers<EXCEPTION_RECORD32,
                                         process_types::EXCEPTION_POINTERS32>(
            process_reader,
            exception_pointers_address,
            thread_id,
            &NativeContextToCPUContext32)) {
      return false;
    }
  }

  CaptureMemoryDelegateWin capture_memory_delegate(
      process_reader, *thread, &extra_memory_, nullptr);
  CaptureMemory::PointedToByContext(context_, &capture_memory_delegate);

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

const CPUContext* ExceptionSnapshotWin::Context() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &context_;
}

uint64_t ExceptionSnapshotWin::ThreadID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return thread_id_;
}

uint32_t ExceptionSnapshotWin::Exception() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_code_;
}

uint32_t ExceptionSnapshotWin::ExceptionInfo() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_flags_;
}

uint64_t ExceptionSnapshotWin::ExceptionAddress() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_address_;
}

const std::vector<uint64_t>& ExceptionSnapshotWin::Codes() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return codes_;
}

std::vector<const MemorySnapshot*> ExceptionSnapshotWin::ExtraMemory() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const MemorySnapshot*> result;
  result.reserve(extra_memory_.size());
  for (const auto& em : extra_memory_) {
    result.push_back(em.get());
  }
  return result;
}

template <class ExceptionRecordType,
          class ExceptionPointersType,
          class ContextType>
bool ExceptionSnapshotWin::InitializeFromExceptionPointers(
    ProcessReaderWin* process_reader,
    WinVMAddress exception_pointers_address,
    DWORD exception_thread_id,
    void (*native_to_cpu_context)(const ContextType& context_record,
                                  CPUContext* context,
                                  CPUContextUnion* context_union)) {
  ExceptionPointersType exception_pointers;
  if (!process_reader->ReadMemory(exception_pointers_address,
                                  sizeof(exception_pointers),
                                  &exception_pointers)) {
    LOG(ERROR) << "EXCEPTION_POINTERS read failed";
    return false;
  }
  if (!exception_pointers.ExceptionRecord) {
    LOG(ERROR) << "null ExceptionRecord";
    return false;
  }

  ExceptionRecordType first_record;
  if (!process_reader->ReadMemory(
          static_cast<WinVMAddress>(exception_pointers.ExceptionRecord),
          sizeof(first_record),
          &first_record)) {
    LOG(ERROR) << "ExceptionRecord";
    return false;
  }

  const bool triggered_by_client =
      first_record.ExceptionCode == CrashpadClient::kTriggeredExceptionCode &&
      first_record.NumberParameters == 2;
  if (triggered_by_client)
    process_reader->DecrementThreadSuspendCounts(exception_thread_id);

  if (triggered_by_client && first_record.ExceptionInformation[0] != 0) {
    // This special exception code indicates that the target was crashed by
    // another client calling CrashpadClient::DumpAndCrashTargetProcess(). In
    // this case the parameters are a thread id and an exception code which we
    // use to fabricate a new exception record.
    using ArgumentType = decltype(first_record.ExceptionInformation[0]);
    const ArgumentType blame_thread_id = first_record.ExceptionInformation[0];
    exception_code_ = static_cast<DWORD>(first_record.ExceptionInformation[1]);
    exception_flags_ = EXCEPTION_NONCONTINUABLE;
    for (const auto& thread : process_reader->Threads()) {
      if (thread.id == blame_thread_id) {
        thread_id_ = blame_thread_id;
        native_to_cpu_context(
            *reinterpret_cast<const ContextType*>(&thread.context),
            &context_,
            &context_union_);
        exception_address_ = context_.InstructionPointer();
        break;
      }
    }

    if (exception_address_ == 0) {
      LOG(WARNING) << "thread " << blame_thread_id << " not found";
      return false;
    }
  } else {
    // Normal case.
    exception_code_ = first_record.ExceptionCode;
    exception_flags_ = first_record.ExceptionFlags;
    exception_address_ = first_record.ExceptionAddress;
    for (DWORD i = 0; i < first_record.NumberParameters; ++i)
      codes_.push_back(first_record.ExceptionInformation[i]);
    if (first_record.ExceptionRecord) {
      // https://crashpad.chromium.org/bug/43
      LOG(WARNING) << "dropping chained ExceptionRecord";
    }

    ContextType context_record;
    if (!process_reader->ReadMemory(
            static_cast<WinVMAddress>(exception_pointers.ContextRecord),
            sizeof(context_record),
            &context_record)) {
      LOG(ERROR) << "ContextRecord";
      return false;
    }

    native_to_cpu_context(context_record, &context_, &context_union_);
  }

  return true;
}

}  // namespace internal
}  // namespace crashpad
