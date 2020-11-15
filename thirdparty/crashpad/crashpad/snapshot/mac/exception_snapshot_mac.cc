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

#include "snapshot/mac/exception_snapshot_mac.h"

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "snapshot/mac/cpu_context_mac.h"
#include "snapshot/mac/process_reader_mac.h"
#include "util/mach/exception_behaviors.h"
#include "util/mach/exception_types.h"
#include "util/mach/symbolic_constants_mach.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {
namespace internal {

ExceptionSnapshotMac::ExceptionSnapshotMac()
    : ExceptionSnapshot(),
      context_union_(),
      context_(),
      codes_(),
      thread_id_(0),
      exception_address_(0),
      exception_(0),
      exception_code_0_(0),
      initialized_() {
}

ExceptionSnapshotMac::~ExceptionSnapshotMac() {
}

bool ExceptionSnapshotMac::Initialize(ProcessReaderMac* process_reader,
                                      exception_behavior_t behavior,
                                      thread_t exception_thread,
                                      exception_type_t exception,
                                      const mach_exception_data_type_t* code,
                                      mach_msg_type_number_t code_count,
                                      thread_state_flavor_t flavor,
                                      ConstThreadState state,
                                      mach_msg_type_number_t state_count) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  codes_.push_back(exception);
  for (mach_msg_type_number_t code_index = 0;
       code_index < code_count;
       ++code_index) {
    codes_.push_back(code[code_index]);
  }

  exception_ = exception;
  mach_exception_code_t exception_code_0 = code[0];

  if (exception_ == EXC_CRASH) {
    exception_ = ExcCrashRecoverOriginalException(
        exception_code_0, &exception_code_0, nullptr);

    if (!ExcCrashCouldContainException(exception_)) {
      LOG(WARNING) << base::StringPrintf(
          "exception %s invalid in EXC_CRASH",
          ExceptionToString(exception_, kUseFullName | kUnknownIsNumeric)
              .c_str());
    }
  }

  // The operations that follow put exception_code_0 (a mach_exception_code_t,
  // a typedef for int64_t) into exception_code_0_ (a uint32_t). The range
  // checks and bit shifts involved need the same signedness on both sides to
  // work properly.
  const uint64_t unsigned_exception_code_0 = exception_code_0;

  // ExceptionInfo() returns code[0] as a 32-bit value, but exception_code_0 is
  // a 64-bit value. The best treatment for this inconsistency depends on the
  // exception type.
  if (exception_ == EXC_RESOURCE || exception_ == EXC_GUARD) {
    // All 64 bits of code[0] are significant for these exceptions. See
    // <mach/exc_resource.h> for EXC_RESOURCE and 10.10
    // xnu-2782.1.97/bsd/kern/kern_guarded.c fd_guard_ast() for EXC_GUARD.
    // code[0] is structured similarly for these two exceptions.
    //
    // EXC_RESOURCE: see <kern/exc_resource.h>. The resource type and “flavor”
    // together define the resource and are in the highest bits. The resource
    // limit is in the lowest bits.
    //
    // EXC_GUARD: see 10.10 xnu-2782.1.97/osfmk/ipc/mach_port.c
    // mach_port_guard_exception() and xnu-2782.1.97/bsd/kern/kern_guarded.c
    // fd_guard_ast(). The guard type (GUARD_TYPE_MACH_PORT or GUARD_TYPE_FD)
    // and “flavor” (from the mach_port_guard_exception_codes or
    // guard_exception_codes enums) are in the highest bits. The violating Mach
    // port name or file descriptor number is in the lowest bits.

    // If MACH_EXCEPTION_CODES is not set in |behavior|, code[0] will only carry
    // 32 significant bits, and the interesting high bits will have been
    // truncated.
    if (!ExceptionBehaviorHasMachExceptionCodes(behavior)) {
      LOG(WARNING) << base::StringPrintf(
          "behavior %s invalid for exception %s",
          ExceptionBehaviorToString(
              behavior, kUseFullName | kUnknownIsNumeric | kUseOr).c_str(),
          ExceptionToString(exception_, kUseFullName | kUnknownIsNumeric)
              .c_str());
    }

    // Include the more-significant information from the high bits of code[0] in
    // the value to be returned by ExceptionInfo(). The full value of codes[0]
    // including the less-significant lower bits is still available via Codes().
    exception_code_0_ = unsigned_exception_code_0 >> 32;
  } else {
    // For other exceptions, code[0]’s values never exceed 32 bits.
    if (!base::IsValueInRangeForNumericType<decltype(exception_code_0_)>(
            unsigned_exception_code_0)) {
      LOG(WARNING) << base::StringPrintf("exception_code_0 0x%llx out of range",
                                         unsigned_exception_code_0);
    }
    exception_code_0_ = unsigned_exception_code_0;
  }

  const ProcessReaderMac::Thread* thread = nullptr;
  for (const ProcessReaderMac::Thread& loop_thread :
       process_reader->Threads()) {
    if (exception_thread == loop_thread.port) {
      thread = &loop_thread;
      break;
    }
  }

  if (!thread) {
    LOG(ERROR) << "exception_thread not found in task";
    return false;
  }

  thread_id_ = thread->id;

  // Normally, the exception address is present in code[1] for EXC_BAD_ACCESS
  // exceptions, but not for other types of exceptions.
  bool code_1_is_exception_address = exception_ == EXC_BAD_ACCESS;

#if defined(ARCH_CPU_X86_FAMILY)
  if (process_reader->Is64Bit()) {
    context_.architecture = kCPUArchitectureX86_64;
    context_.x86_64 = &context_union_.x86_64;
    InitializeCPUContextX86_64(context_.x86_64,
                               flavor,
                               state,
                               state_count,
                               &thread->thread_context.t64,
                               &thread->float_context.f64,
                               &thread->debug_context.d64);
  } else {
    context_.architecture = kCPUArchitectureX86;
    context_.x86 = &context_union_.x86;
    InitializeCPUContextX86(context_.x86,
                            flavor,
                            state,
                            state_count,
                            &thread->thread_context.t32,
                            &thread->float_context.f32,
                            &thread->debug_context.d32);
  }

  // For x86 and x86_64 EXC_BAD_ACCESS exceptions, some code[0] values indicate
  // that code[1] does not (or may not) carry the exception address:
  // EXC_I386_GPFLT (10.9.5 xnu-2422.115.4/osfmk/i386/trap.c user_trap() for
  // T_GENERAL_PROTECTION) and the oddball (VM_PROT_READ | VM_PROT_EXECUTE)
  // which collides with EXC_I386_BOUNDFLT (10.9.5
  // xnu-2422.115.4/osfmk/i386/fpu.c fpextovrflt()). Other EXC_BAD_ACCESS
  // exceptions come through 10.9.5 xnu-2422.115.4/osfmk/i386/trap.c
  // user_page_fault_continue() and do contain the exception address in code[1].
  if (exception_ == EXC_BAD_ACCESS &&
      (exception_code_0_ == EXC_I386_GPFLT ||
       exception_code_0_ == (VM_PROT_READ | VM_PROT_EXECUTE))) {
    code_1_is_exception_address = false;
  }
#endif

  if (code_1_is_exception_address) {
    if (process_reader->Is64Bit() &&
        !ExceptionBehaviorHasMachExceptionCodes(behavior)) {
      // If code[1] is an address from a 64-bit process, the exception must have
      // been received with MACH_EXCEPTION_CODES or the address will have been
      // truncated.
      LOG(WARNING) << base::StringPrintf(
          "behavior %s invalid for exception %s code %d in 64-bit process",
          ExceptionBehaviorToString(
              behavior, kUseFullName | kUnknownIsNumeric | kUseOr).c_str(),
          ExceptionToString(exception_, kUseFullName | kUnknownIsNumeric)
              .c_str(),
          exception_code_0_);
    }
    exception_address_ = code[1];
  } else {
    exception_address_ = context_.InstructionPointer();
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

const CPUContext* ExceptionSnapshotMac::Context() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &context_;
}

uint64_t ExceptionSnapshotMac::ThreadID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return thread_id_;
}

uint32_t ExceptionSnapshotMac::Exception() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_;
}

uint32_t ExceptionSnapshotMac::ExceptionInfo() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_code_0_;
}

uint64_t ExceptionSnapshotMac::ExceptionAddress() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_address_;
}

const std::vector<uint64_t>& ExceptionSnapshotMac::Codes() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return codes_;
}

std::vector<const MemorySnapshot*> ExceptionSnapshotMac::ExtraMemory() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<const MemorySnapshot*>();
}

}  // namespace internal
}  // namespace crashpad
