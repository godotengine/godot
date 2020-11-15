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

#include "snapshot/linux/exception_snapshot_linux.h"

#include <signal.h>

#include "base/logging.h"
#include "snapshot/linux/cpu_context_linux.h"
#include "snapshot/linux/process_reader_linux.h"
#include "snapshot/linux/signal_context.h"
#include "util/linux/traits.h"
#include "util/misc/reinterpret_bytes.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {
namespace internal {

ExceptionSnapshotLinux::ExceptionSnapshotLinux()
    : ExceptionSnapshot(),
      context_union_(),
      context_(),
      codes_(),
      thread_id_(0),
      exception_address_(0),
      signal_number_(0),
      signal_code_(0),
      initialized_() {}

ExceptionSnapshotLinux::~ExceptionSnapshotLinux() {}

#if defined(ARCH_CPU_X86_FAMILY)

template <>
bool ExceptionSnapshotLinux::ReadContext<ContextTraits32>(
    ProcessReaderLinux* reader,
    LinuxVMAddress context_address) {
  UContext<ContextTraits32> ucontext;
  if (!reader->Memory()->Read(context_address, sizeof(ucontext), &ucontext)) {
    LOG(ERROR) << "Couldn't read ucontext";
    return false;
  }

  context_.architecture = kCPUArchitectureX86;
  context_.x86 = &context_union_.x86;

  if (!ucontext.mcontext.fpptr) {
    InitializeCPUContextX86_NoFloatingPoint(ucontext.mcontext.gprs,
                                            context_.x86);
    return true;
  }

  SignalFloatContext32 fprs;
  if (!reader->Memory()->Read(ucontext.mcontext.fpptr, sizeof(fprs), &fprs)) {
    LOG(ERROR) << "Couldn't read float context";
    return false;
  }

  if (fprs.magic == X86_FXSR_MAGIC) {
    InitializeCPUContextX86_NoFloatingPoint(ucontext.mcontext.gprs,
                                            context_.x86);
    if (!reader->Memory()->Read(
            ucontext.mcontext.fpptr + offsetof(SignalFloatContext32, fxsave),
            sizeof(CPUContextX86::Fxsave),
            &context_.x86->fxsave)) {
      LOG(ERROR) << "Couldn't read fxsave";
      return false;
    }
  } else if (fprs.magic == 0xffff) {
    InitializeCPUContextX86(ucontext.mcontext.gprs, fprs, context_.x86);
  } else {
    LOG(ERROR) << "unexpected magic 0x" << std::hex << fprs.magic;
    return false;
  }

  return true;
}

template <>
bool ExceptionSnapshotLinux::ReadContext<ContextTraits64>(
    ProcessReaderLinux* reader,
    LinuxVMAddress context_address) {
  UContext<ContextTraits64> ucontext;
  if (!reader->Memory()->Read(context_address, sizeof(ucontext), &ucontext)) {
    LOG(ERROR) << "Couldn't read ucontext";
    return false;
  }

  context_.architecture = kCPUArchitectureX86_64;
  context_.x86_64 = &context_union_.x86_64;

  if (!ucontext.mcontext.fpptr) {
    InitializeCPUContextX86_64_NoFloatingPoint(ucontext.mcontext.gprs,
                                               context_.x86_64);
    return true;
  }

  SignalFloatContext64 fprs;
  if (!reader->Memory()->Read(ucontext.mcontext.fpptr, sizeof(fprs), &fprs)) {
    LOG(ERROR) << "Couldn't read float context";
    return false;
  }

  InitializeCPUContextX86_64(ucontext.mcontext.gprs, fprs, context_.x86_64);
  return true;
}

#elif defined(ARCH_CPU_ARM_FAMILY)

template <>
bool ExceptionSnapshotLinux::ReadContext<ContextTraits32>(
    ProcessReaderLinux* reader,
    LinuxVMAddress context_address) {
  context_.architecture = kCPUArchitectureARM;
  context_.arm = &context_union_.arm;

  CPUContextARM* dest_context = context_.arm;
  ProcessMemory* memory = reader->Memory();

  LinuxVMAddress gprs_address =
      context_address + offsetof(UContext<ContextTraits32>, mcontext32) +
      offsetof(ContextTraits32::MContext32, gprs);

  SignalThreadContext32 thread_context;
  if (!memory->Read(gprs_address, sizeof(thread_context), &thread_context)) {
    LOG(ERROR) << "Couldn't read gprs";
    return false;
  }
  InitializeCPUContextARM_NoFloatingPoint(thread_context, dest_context);

  LinuxVMAddress reserved_address =
      context_address + offsetof(UContext<ContextTraits32>, reserved);
  if ((reserved_address & 7) != 0) {
    LOG(ERROR) << "invalid alignment 0x" << std::hex << reserved_address;
    return false;
  }

  constexpr VMSize kMaxContextSpace = 1024;

  ProcessMemoryRange range;
  if (!range.Initialize(memory, false, reserved_address, kMaxContextSpace)) {
    return false;
  }

  do {
    CoprocessorContextHead head;
    if (!range.Read(reserved_address, sizeof(head), &head)) {
      LOG(ERROR) << "missing context terminator";
      return false;
    }
    reserved_address += sizeof(head);

    switch (head.magic) {
      case VFP_MAGIC:
        if (head.size != sizeof(SignalVFPContext) + sizeof(head)) {
          LOG(ERROR) << "unexpected vfp context size " << head.size;
          return false;
        }
        static_assert(
            sizeof(SignalVFPContext::vfp) == sizeof(dest_context->vfp_regs),
            "vfp context size mismatch");
        if (!range.Read(reserved_address + offsetof(SignalVFPContext, vfp),
                        sizeof(dest_context->vfp_regs),
                        &dest_context->vfp_regs)) {
          LOG(ERROR) << "Couldn't read vfp";
          return false;
        }
        dest_context->have_vfp_regs = true;
        return true;

      case CRUNCH_MAGIC:
      case IWMMXT_MAGIC:
      case DUMMY_MAGIC:
        reserved_address += head.size - sizeof(head);
        continue;

      case 0:
        return true;

      default:
        LOG(ERROR) << "invalid magic number 0x" << std::hex << head.magic;
        return false;
    }
  } while (true);
}

template <>
bool ExceptionSnapshotLinux::ReadContext<ContextTraits64>(
    ProcessReaderLinux* reader,
    LinuxVMAddress context_address) {
  context_.architecture = kCPUArchitectureARM64;
  context_.arm64 = &context_union_.arm64;

  CPUContextARM64* dest_context = context_.arm64;
  ProcessMemory* memory = reader->Memory();

  LinuxVMAddress gprs_address =
      context_address + offsetof(UContext<ContextTraits64>, mcontext64) +
      offsetof(ContextTraits64::MContext64, gprs);

  ThreadContext::t64_t thread_context;
  if (!memory->Read(gprs_address, sizeof(thread_context), &thread_context)) {
    LOG(ERROR) << "Couldn't read gprs";
    return false;
  }
  InitializeCPUContextARM64_NoFloatingPoint(thread_context, dest_context);

  LinuxVMAddress reserved_address =
      context_address + offsetof(UContext<ContextTraits64>, reserved);
  if ((reserved_address & 15) != 0) {
    LOG(ERROR) << "invalid alignment 0x" << std::hex << reserved_address;
    return false;
  }

  constexpr VMSize kMaxContextSpace = 4096;

  ProcessMemoryRange range;
  if (!range.Initialize(memory, true, reserved_address, kMaxContextSpace)) {
    return false;
  }

  do {
    CoprocessorContextHead head;
    if (!range.Read(reserved_address, sizeof(head), &head)) {
      LOG(ERROR) << "missing context terminator";
      return false;
    }
    reserved_address += sizeof(head);

    switch (head.magic) {
      case FPSIMD_MAGIC:
        if (head.size != sizeof(SignalFPSIMDContext) + sizeof(head)) {
          LOG(ERROR) << "unexpected fpsimd context size " << head.size;
          return false;
        }
        SignalFPSIMDContext fpsimd;
        if (!range.Read(reserved_address, sizeof(fpsimd), &fpsimd)) {
          LOG(ERROR) << "Couldn't read fpsimd " << head.size;
          return false;
        }
        InitializeCPUContextARM64_OnlyFPSIMD(fpsimd, dest_context);
        return true;

      case ESR_MAGIC:
      case EXTRA_MAGIC:
        reserved_address += head.size - sizeof(head);
        continue;

      case 0:
        LOG(WARNING) << "fpsimd not found";
        return true;

      default:
        LOG(ERROR) << "invalid magic number 0x" << std::hex << head.magic;
        return false;
    }
  } while (true);
}

#elif defined(ARCH_CPU_MIPS_FAMILY)

template <typename Traits>
static bool ReadContext(ProcessReaderLinux* reader,
                        LinuxVMAddress context_address,
                        typename Traits::CPUContext* dest_context) {
  ProcessMemory* memory = reader->Memory();

  LinuxVMAddress gregs_address = context_address +
                                 offsetof(UContext<Traits>, mcontext) +
                                 offsetof(typename Traits::MContext, gregs);

  typename Traits::SignalThreadContext thread_context;
  if (!memory->Read(gregs_address, sizeof(thread_context), &thread_context)) {
    LOG(ERROR) << "Couldn't read gregs";
    return false;
  }

  LinuxVMAddress fpregs_address = context_address +
                                  offsetof(UContext<Traits>, mcontext) +
                                  offsetof(typename Traits::MContext, fpregs);

  typename Traits::SignalFloatContext fp_context;
  if (!memory->Read(fpregs_address, sizeof(fp_context), &fp_context)) {
    LOG(ERROR) << "Couldn't read fpregs";
    return false;
  }

  InitializeCPUContextMIPS<Traits>(thread_context, fp_context, dest_context);

  return true;
}

template <>
bool ExceptionSnapshotLinux::ReadContext<ContextTraits32>(
    ProcessReaderLinux* reader,
    LinuxVMAddress context_address) {
  context_.architecture = kCPUArchitectureMIPSEL;
  context_.mipsel = &context_union_.mipsel;

  return internal::ReadContext<ContextTraits32>(
      reader, context_address, context_.mipsel);
}

template <>
bool ExceptionSnapshotLinux::ReadContext<ContextTraits64>(
    ProcessReaderLinux* reader,
    LinuxVMAddress context_address) {
  context_.architecture = kCPUArchitectureMIPS64EL;
  context_.mips64 = &context_union_.mips64;

  return internal::ReadContext<ContextTraits64>(
      reader, context_address, context_.mips64);
}

#endif  // ARCH_CPU_X86_FAMILY

bool ExceptionSnapshotLinux::Initialize(ProcessReaderLinux* process_reader,
                                        LinuxVMAddress siginfo_address,
                                        LinuxVMAddress context_address,
                                        pid_t thread_id) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  thread_id_ = thread_id;

  if (process_reader->Is64Bit()) {
    if (!ReadContext<ContextTraits64>(process_reader, context_address) ||
        !ReadSiginfo<Traits64>(process_reader, siginfo_address)) {
      return false;
    }
  } else {
    if (!ReadContext<ContextTraits32>(process_reader, context_address) ||
        !ReadSiginfo<Traits32>(process_reader, siginfo_address)) {
      return false;
    }
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

template <typename Traits>
bool ExceptionSnapshotLinux::ReadSiginfo(ProcessReaderLinux* reader,
                                         LinuxVMAddress siginfo_address) {
  Siginfo<Traits> siginfo;
  if (!reader->Memory()->Read(siginfo_address, sizeof(siginfo), &siginfo)) {
    LOG(ERROR) << "Couldn't read siginfo";
    return false;
  }

  signal_number_ = siginfo.signo;
  signal_code_ = siginfo.code;

  uint64_t extra_code;
#define PUSH_CODE(value)                         \
  do {                                           \
    if (!ReinterpretBytes(value, &extra_code)) { \
      LOG(ERROR) << "bad code";                  \
      return false;                              \
    }                                            \
    codes_.push_back(extra_code);                \
  } while (false)

  switch (siginfo.signo) {
    case SIGILL:
    case SIGFPE:
    case SIGSEGV:
    case SIGBUS:
    case SIGTRAP:
      exception_address_ = siginfo.address;
      break;

    case SIGPOLL:  // SIGIO
      PUSH_CODE(siginfo.band);
      PUSH_CODE(siginfo.fd);
      break;

    case SIGSYS:
      exception_address_ = siginfo.call_address;
      PUSH_CODE(siginfo.syscall);
      PUSH_CODE(siginfo.arch);
      break;

    case SIGALRM:
    case SIGVTALRM:
    case SIGPROF:
      PUSH_CODE(siginfo.timerid);
      PUSH_CODE(siginfo.overrun);
      PUSH_CODE(siginfo.sigval.sigval);
      break;

    case SIGABRT:
    case SIGQUIT:
    case SIGXCPU:
    case SIGXFSZ:
    case SIGHUP:
    case SIGINT:
    case SIGPIPE:
    case SIGTERM:
    case SIGUSR1:
    case SIGUSR2:
#if defined(SIGEMT)
    case SIGEMT:
#endif  // SIGEMT
#if defined(SIGPWR)
    case SIGPWR:
#endif  // SIGPWR
#if defined(SIGSTKFLT)
    case SIGSTKFLT:
#endif  // SIGSTKFLT
      PUSH_CODE(siginfo.pid);
      PUSH_CODE(siginfo.uid);
      PUSH_CODE(siginfo.sigval.sigval);
      break;

    default:
      LOG(WARNING) << "Unhandled signal " << siginfo.signo;
  }

  return true;
}

const CPUContext* ExceptionSnapshotLinux::Context() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &context_;
}

uint64_t ExceptionSnapshotLinux::ThreadID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return thread_id_;
}

uint32_t ExceptionSnapshotLinux::Exception() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return signal_number_;
}

uint32_t ExceptionSnapshotLinux::ExceptionInfo() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return signal_code_;
}

uint64_t ExceptionSnapshotLinux::ExceptionAddress() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return exception_address_;
}

const std::vector<uint64_t>& ExceptionSnapshotLinux::Codes() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return codes_;
}

std::vector<const MemorySnapshot*> ExceptionSnapshotLinux::ExtraMemory() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<const MemorySnapshot*>();
}

}  // namespace internal
}  // namespace crashpad
