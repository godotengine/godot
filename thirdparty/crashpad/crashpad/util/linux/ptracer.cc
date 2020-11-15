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

#include "util/linux/ptracer.h"

#include <errno.h>
#include <linux/elf.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/uio.h>

#include "base/logging.h"
#include "build/build_config.h"
#include "util/misc/from_pointer_cast.h"

#if defined(ARCH_CPU_X86_FAMILY)
#include <asm/ldt.h>
#endif

namespace crashpad {

namespace {

#if defined(ARCH_CPU_X86_FAMILY)

template <typename Destination>
bool GetRegisterSet(pid_t tid, int set, Destination* dest, bool can_log) {
  iovec iov;
  iov.iov_base = dest;
  iov.iov_len = sizeof(*dest);
  if (ptrace(PTRACE_GETREGSET, tid, reinterpret_cast<void*>(set), &iov) != 0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  if (iov.iov_len != sizeof(*dest)) {
    LOG_IF(ERROR, can_log) << "Unexpected registers size";
    return false;
  }
  return true;
}

bool GetFloatingPointRegisters32(pid_t tid,
                                 FloatContext* context,
                                 bool can_log) {
  return GetRegisterSet(tid, NT_PRXFPREG, &context->f32.fxsave, can_log);
}

bool GetFloatingPointRegisters64(pid_t tid,
                                 FloatContext* context,
                                 bool can_log) {
  return GetRegisterSet(tid, NT_PRFPREG, &context->f64.fxsave, can_log);
}

bool GetThreadArea32(pid_t tid,
                     const ThreadContext& context,
                     LinuxVMAddress* address,
                     bool can_log) {
  size_t index = (context.t32.xgs & 0xffff) >> 3;
  user_desc desc;
  if (ptrace(
          PTRACE_GET_THREAD_AREA, tid, reinterpret_cast<void*>(index), &desc) !=
      0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }

  *address = desc.base_addr;
  return true;
}

bool GetThreadArea64(pid_t tid,
                     const ThreadContext& context,
                     LinuxVMAddress* address,
                     bool can_log) {
  *address = context.t64.fs_base;
  return true;
}

#elif defined(ARCH_CPU_ARM_FAMILY)

#if defined(ARCH_CPU_ARMEL)
// PTRACE_GETREGSET, introduced in Linux 2.6.34 (2225a122ae26), requires kernel
// support enabled by HAVE_ARCH_TRACEHOOK. This has been set for x86 (including
// x86_64) since Linux 2.6.28 (99bbc4b1e677a), but for ARM only since
// Linux 3.5.0 (0693bf68148c4). Older Linux kernels support PTRACE_GETREGS,
// PTRACE_GETFPREGS, and PTRACE_GETVFPREGS instead, which don't allow checking
// the size of data copied.
//
// Fortunately, 64-bit ARM support only appeared in Linux 3.7.0, so if
// PTRACE_GETREGSET fails on ARM with EIO, indicating that the request is not
// supported, the kernel must be old enough that 64-bit ARM isn’t supported
// either.
//
// TODO(mark): Once helpers to interpret the kernel version are available, add
// a DCHECK to ensure that the kernel is older than 3.5.

bool GetGeneralPurposeRegistersLegacy(pid_t tid,
                                      ThreadContext* context,
                                      bool can_log) {
  if (ptrace(PTRACE_GETREGS, tid, nullptr, &context->t32) != 0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  return true;
}

bool GetFloatingPointRegistersLegacy(pid_t tid,
                                     FloatContext* context,
                                     bool can_log) {
  if (ptrace(PTRACE_GETFPREGS, tid, nullptr, &context->f32.fpregs) != 0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  context->f32.have_fpregs = true;

  if (ptrace(PTRACE_GETVFPREGS, tid, nullptr, &context->f32.vfp) != 0) {
    switch (errno) {
      case EINVAL:
        // These registers are optional on 32-bit ARM cpus
        break;
      default:
        PLOG_IF(ERROR, can_log) << "ptrace";
        return false;
    }
  } else {
    context->f32.have_vfp = true;
  }
  return true;
}
#endif  // ARCH_CPU_ARMEL

// Normally, the Linux kernel will copy out register sets according to the size
// of the struct that contains them. Tracing a 32-bit ARM process running in
// compatibility mode on a 64-bit ARM cpu will only copy data for the number of
// registers times the size of the register, which won't include any possible
// trailing padding in the struct. These are the sizes of the register data, not
// including any possible padding.
constexpr size_t kArmVfpSize = 32 * 8 + 4;

// Target is 32-bit
bool GetFloatingPointRegisters32(pid_t tid,
                                 FloatContext* context,
                                 bool can_log) {
  context->f32.have_fpregs = false;
  context->f32.have_vfp = false;

  iovec iov;
  iov.iov_base = &context->f32.fpregs;
  iov.iov_len = sizeof(context->f32.fpregs);
  if (ptrace(
          PTRACE_GETREGSET, tid, reinterpret_cast<void*>(NT_PRFPREG), &iov) !=
      0) {
    switch (errno) {
#if defined(ARCH_CPU_ARMEL)
      case EIO:
        return GetFloatingPointRegistersLegacy(tid, context, can_log);
#endif  // ARCH_CPU_ARMEL
      case EINVAL:
        // A 32-bit process running on a 64-bit CPU doesn't have this register
        // set. It should have a VFP register set instead.
        break;
      default:
        PLOG_IF(ERROR, can_log) << "ptrace";
        return false;
    }
  } else {
    if (iov.iov_len != sizeof(context->f32.fpregs)) {
      LOG_IF(ERROR, can_log) << "Unexpected registers size";
      return false;
    }
    context->f32.have_fpregs = true;
  }

  iov.iov_base = &context->f32.vfp;
  iov.iov_len = sizeof(context->f32.vfp);
  if (ptrace(
          PTRACE_GETREGSET, tid, reinterpret_cast<void*>(NT_ARM_VFP), &iov) !=
      0) {
    switch (errno) {
      case EINVAL:
        // VFP may not be present on 32-bit ARM cpus.
        break;
      default:
        PLOG_IF(ERROR, can_log) << "ptrace";
        return false;
    }
  } else {
    if (iov.iov_len != kArmVfpSize && iov.iov_len != sizeof(context->f32.vfp)) {
      LOG_IF(ERROR, can_log) << "Unexpected registers size";
      return false;
    }
    context->f32.have_vfp = true;
  }

  if (!(context->f32.have_fpregs || context->f32.have_vfp)) {
    LOG_IF(ERROR, can_log) << "Unable to collect registers";
    return false;
  }
  return true;
}

bool GetFloatingPointRegisters64(pid_t tid,
                                 FloatContext* context,
                                 bool can_log) {
  iovec iov;
  iov.iov_base = context;
  iov.iov_len = sizeof(*context);
  if (ptrace(
          PTRACE_GETREGSET, tid, reinterpret_cast<void*>(NT_PRFPREG), &iov) !=
      0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  if (iov.iov_len != sizeof(context->f64)) {
    LOG_IF(ERROR, can_log) << "Unexpected registers size";
    return false;
  }
  return true;
}

bool GetThreadArea32(pid_t tid,
                     const ThreadContext& context,
                     LinuxVMAddress* address,
                     bool can_log) {
#if defined(ARCH_CPU_ARMEL)
  void* result;
  if (ptrace(PTRACE_GET_THREAD_AREA, tid, nullptr, &result) != 0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  *address = FromPointerCast<LinuxVMAddress>(result);
  return true;
#else
  // TODO(jperaza): it doesn't look like there is a way for a 64-bit ARM process
  // to get the thread area for a 32-bit ARM process with ptrace.
  LOG_IF(WARNING, can_log)
      << "64-bit ARM cannot trace TLS area for a 32-bit process";
  return false;
#endif  // ARCH_CPU_ARMEL
}

bool GetThreadArea64(pid_t tid,
                     const ThreadContext& context,
                     LinuxVMAddress* address,
                     bool can_log) {
  iovec iov;
  iov.iov_base = address;
  iov.iov_len = sizeof(*address);
  if (ptrace(
          PTRACE_GETREGSET, tid, reinterpret_cast<void*>(NT_ARM_TLS), &iov) !=
      0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  if (iov.iov_len != 8) {
    LOG_IF(ERROR, can_log) << "address size mismatch";
    return false;
  }
  return true;
}
#elif defined(ARCH_CPU_MIPS_FAMILY)
// PTRACE_GETREGSET, introduced in Linux 2.6.34 (2225a122ae26), requires kernel
// support enabled by HAVE_ARCH_TRACEHOOK. This has been set for x86 (including
// x86_64) since Linux 2.6.28 (99bbc4b1e677a), but for MIPS only since
// Linux 3.13 (c0ff3c53d4f99). Older Linux kernels support PTRACE_GETREGS,
// and PTRACE_GETFPREGS instead, which don't allow checking the size of data
// copied. Also, PTRACE_GETREGS assumes register size of 64 bits even for 32 bit
// MIPS CPU (contrary to PTRACE_GETREGSET behavior), so we need buffer
// structure here.

bool GetGeneralPurposeRegistersLegacy(pid_t tid,
                                      ThreadContext* context,
                                      bool can_log) {
  ThreadContext context_buffer;
  if (ptrace(PTRACE_GETREGS, tid, nullptr, &context_buffer.t64) != 0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
// Bitness of target process can't be determined through ptrace here, so we
// assume target process has the same as current process, making cross-bit
// ptrace unsupported on MIPS for kernels older than 3.13
#if defined(ARCH_CPU_MIPSEL)
#define THREAD_CONTEXT_FIELD t32
#elif defined(ARCH_CPU_MIPS64EL)
#define THREAD_CONTEXT_FIELD t64
#endif
  for (size_t reg = 0; reg < 32; ++reg) {
    context->THREAD_CONTEXT_FIELD.regs[reg] = context_buffer.t64.regs[reg];
  }
  context->THREAD_CONTEXT_FIELD.lo = context_buffer.t64.lo;
  context->THREAD_CONTEXT_FIELD.hi = context_buffer.t64.hi;
  context->THREAD_CONTEXT_FIELD.cp0_epc = context_buffer.t64.cp0_epc;
  context->THREAD_CONTEXT_FIELD.cp0_badvaddr = context_buffer.t64.cp0_badvaddr;
  context->THREAD_CONTEXT_FIELD.cp0_status = context_buffer.t64.cp0_status;
  context->THREAD_CONTEXT_FIELD.cp0_cause = context_buffer.t64.cp0_cause;
#undef THREAD_CONTEXT_FIELD
  return true;
}

bool GetFloatingPointRegistersLegacy(pid_t tid,
                                     FloatContext* context,
                                     bool can_log) {
  if (ptrace(PTRACE_GETFPREGS, tid, nullptr, &context->f32.fpregs) != 0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  return true;
}

bool GetFloatingPointRegisters32(pid_t tid,
                                 FloatContext* context,
                                 bool can_log) {
  iovec iov;
  iov.iov_base = &context->f32.fpregs;
  iov.iov_len = sizeof(context->f32.fpregs);
  if (ptrace(PTRACE_GETFPREGS, tid, nullptr, &context->f32.fpregs) != 0) {
    switch (errno) {
      case EINVAL:
        // fp may not be present
        break;
      case EIO:
        return GetFloatingPointRegistersLegacy(tid, context, can_log);
      default:
        PLOG_IF(ERROR, can_log) << "ptrace";
        return false;
    }
  }
  return true;
}

bool GetFloatingPointRegisters64(pid_t tid,
                                 FloatContext* context,
                                 bool can_log) {
  iovec iov;
  iov.iov_base = &context->f64.fpregs;
  iov.iov_len = sizeof(context->f64.fpregs);
  if (ptrace(PTRACE_GETFPREGS, tid, nullptr, &context->f64.fpregs) != 0) {
    switch (errno) {
      case EINVAL:
        // fp may not be present
        break;
      case EIO:
        return GetFloatingPointRegistersLegacy(tid, context, can_log);
      default:
        PLOG_IF(ERROR, can_log) << "ptrace";
        return false;
    }
  }
  return true;
}

bool GetThreadArea32(pid_t tid,
                     const ThreadContext& context,
                     LinuxVMAddress* address,
                     bool can_log) {
#if defined(ARCH_CPU_MIPSEL)
  void* result;
  if (ptrace(PTRACE_GET_THREAD_AREA, tid, nullptr, &result) != 0) {
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  *address = FromPointerCast<LinuxVMAddress>(result);
  return true;
#else
  return false;
#endif
}

bool GetThreadArea64(pid_t tid,
                     const ThreadContext& context,
                     LinuxVMAddress* address,
                     bool can_log) {
  void* result;
#if defined(ARCH_CPU_MIPSEL)
  if (ptrace(PTRACE_GET_THREAD_AREA_3264, tid, nullptr, &result) != 0) {
#else
  if (ptrace(PTRACE_GET_THREAD_AREA, tid, nullptr, &result) != 0) {
#endif
    PLOG_IF(ERROR, can_log) << "ptrace";
    return false;
  }
  *address = FromPointerCast<LinuxVMAddress>(result);
  return true;
}

#else
#error Port.
#endif  // ARCH_CPU_X86_FAMILY

size_t GetGeneralPurposeRegistersAndLength(pid_t tid,
                                           ThreadContext* context,
                                           bool can_log) {
  iovec iov;
  iov.iov_base = context;
  iov.iov_len = sizeof(*context);
  if (ptrace(
          PTRACE_GETREGSET, tid, reinterpret_cast<void*>(NT_PRSTATUS), &iov) !=
      0) {
    switch (errno) {
#if defined(ARCH_CPU_ARMEL) || defined(ARCH_CPU_MIPS_FAMILY)
      case EIO:
        return GetGeneralPurposeRegistersLegacy(tid, context, can_log)
                   ? sizeof(context->t32)
                   : 0;
#endif  // ARCH_CPU_ARMEL
      default:
        PLOG_IF(ERROR, can_log) << "ptrace";
        return 0;
    }
  }
  return iov.iov_len;
}

bool GetGeneralPurposeRegisters32(pid_t tid,
                                  ThreadContext* context,
                                  bool can_log) {
  if (GetGeneralPurposeRegistersAndLength(tid, context, can_log) !=
      sizeof(context->t32)) {
    LOG_IF(ERROR, can_log) << "Unexpected registers size";
    return false;
  }
  return true;
}

bool GetGeneralPurposeRegisters64(pid_t tid,
                                  ThreadContext* context,
                                  bool can_log) {
  if (GetGeneralPurposeRegistersAndLength(tid, context, can_log) !=
      sizeof(context->t64)) {
    LOG_IF(ERROR, can_log) << "Unexpected registers size";
    return false;
  }
  return true;
}

}  // namespace

Ptracer::Ptracer(bool can_log)
    : is_64_bit_(false), can_log_(can_log), initialized_() {}

Ptracer::Ptracer(bool is_64_bit, bool can_log)
    : is_64_bit_(is_64_bit), can_log_(can_log), initialized_() {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  INITIALIZATION_STATE_SET_VALID(initialized_);
}

Ptracer::~Ptracer() {}

bool Ptracer::Initialize(pid_t pid) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  ThreadContext context;
  size_t length = GetGeneralPurposeRegistersAndLength(pid, &context, can_log_);
  if (length == sizeof(context.t64)) {
    is_64_bit_ = true;
  } else if (length == sizeof(context.t32)) {
    is_64_bit_ = false;
  } else {
    LOG_IF(ERROR, can_log_) << "Unexpected registers size";
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool Ptracer::Is64Bit() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return is_64_bit_;
}

bool Ptracer::GetThreadInfo(pid_t tid, ThreadInfo* info) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (is_64_bit_) {
    return GetGeneralPurposeRegisters64(tid, &info->thread_context, can_log_) &&
           GetFloatingPointRegisters64(tid, &info->float_context, can_log_) &&
           GetThreadArea64(tid,
                           info->thread_context,
                           &info->thread_specific_data_address,
                           can_log_);
  }

  return GetGeneralPurposeRegisters32(tid, &info->thread_context, can_log_) &&
         GetFloatingPointRegisters32(tid, &info->float_context, can_log_) &&
         GetThreadArea32(tid,
                         info->thread_context,
                         &info->thread_specific_data_address,
                         can_log_);
}

ssize_t Ptracer::ReadUpTo(pid_t pid,
                          LinuxVMAddress address,
                          size_t size,
                          char* buffer) {
  size_t bytes_read = 0;
  while (size > 0) {
    errno = 0;

    if (size >= sizeof(long)) {
      *reinterpret_cast<long*>(buffer) =
          ptrace(PTRACE_PEEKDATA, pid, address, nullptr);

      if (errno == EIO) {
        ssize_t last_bytes = ReadLastBytes(pid, address, size, buffer);
        return last_bytes >= 0 ? bytes_read + last_bytes : -1;
      }

      if (errno != 0) {
        PLOG_IF(ERROR, can_log_) << "ptrace";
        return -1;
      }

      size -= sizeof(long);
      buffer += sizeof(long);
      address += sizeof(long);
      bytes_read += sizeof(long);
    } else {
      long word = ptrace(PTRACE_PEEKDATA, pid, address, nullptr);

      if (errno == 0) {
        memcpy(buffer, reinterpret_cast<char*>(&word), size);
        return bytes_read + size;
      }

      if (errno == EIO) {
        ssize_t last_bytes = ReadLastBytes(pid, address, size, buffer);
        return last_bytes >= 0 ? bytes_read + last_bytes : -1;
      }

      PLOG_IF(ERROR, can_log_);
      return -1;
    }
  }

  return bytes_read;
}

// Handles an EIO by reading at most size bytes from address into buffer if
// address was within a word of a possible page boundary, by aligning to read
// the last word of the page and extracting the desired bytes.
ssize_t Ptracer::ReadLastBytes(pid_t pid,
                               LinuxVMAddress address,
                               size_t size,
                               char* buffer) {
  LinuxVMAddress aligned = ((address + 4095) & ~4095) - sizeof(long);
  if (aligned >= address || aligned == address - sizeof(long)) {
    PLOG_IF(ERROR, can_log_) << "ptrace";
    return -1;
  }
  DCHECK_GT(aligned, address - sizeof(long));

  errno = 0;
  long word = ptrace(PTRACE_PEEKDATA, pid, aligned, nullptr);
  if (errno != 0) {
    PLOG_IF(ERROR, can_log_) << "ptrace";
    return -1;
  }

  size_t bytes_read = address - aligned;
  size_t last_bytes = std::min(sizeof(long) - bytes_read, size);
  memcpy(buffer, reinterpret_cast<char*>(&word) + bytes_read, last_bytes);
  return last_bytes;
}

}  // namespace crashpad
