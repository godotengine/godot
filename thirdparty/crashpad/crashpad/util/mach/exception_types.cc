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

#include "util/mach/exception_types.h"

#include <Availability.h>
#include <AvailabilityMacros.h>
#include <dlfcn.h>
#include <errno.h>
#include <libproc.h>
#include <kern/exc_resource.h>
#include <strings.h>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "util/mac/mac_util.h"
#include "util/mach/mach_extensions.h"
#include "util/numeric/in_range_cast.h"

#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_9

extern "C" {

// proc_get_wakemon_params() is present in the OS X 10.9 SDK, but no declaration
// is provided. This provides a declaration and marks it for weak import if the
// deployment target is below 10.9.
int proc_get_wakemon_params(pid_t pid, int* rate_hz, int* flags)
    __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);

// Redeclare the method without the availability annotation to suppress the
// -Wpartial-availability warning.
int proc_get_wakemon_params(pid_t pid, int* rate_hz, int* flags);

}  // extern "C"

#else

namespace {

using ProcGetWakemonParamsType = int (*)(pid_t, int*, int*);

// The SDK doesn’t have proc_get_wakemon_params() to link against, even with
// weak import. This function returns a function pointer to it if it exists at
// runtime, or nullptr if it doesn’t. proc_get_wakemon_params() is looked up in
// the same module that provides proc_pidinfo().
ProcGetWakemonParamsType GetProcGetWakemonParams() {
  Dl_info dl_info;
  if (!dladdr(reinterpret_cast<void*>(proc_pidinfo), &dl_info)) {
    return nullptr;
  }

  void* dl_handle =
      dlopen(dl_info.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
  if (!dl_handle) {
    return nullptr;
  }

  ProcGetWakemonParamsType proc_get_wakemon_params =
      reinterpret_cast<ProcGetWakemonParamsType>(
          dlsym(dl_handle, "proc_get_wakemon_params"));
  return proc_get_wakemon_params;
}

}  // namespace

#endif

namespace {

// Wraps proc_get_wakemon_params(), calling it if the system provides it. It’s
// present on OS X 10.9 and later. If it’s not available, sets errno to ENOSYS
// and returns -1.
int ProcGetWakemonParams(pid_t pid, int* rate_hz, int* flags) {
#if MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_9
  // proc_get_wakemon_params() isn’t in the SDK. Look it up dynamically.
  static ProcGetWakemonParamsType proc_get_wakemon_params =
      GetProcGetWakemonParams();
#endif

#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_9
  // proc_get_wakemon_params() is definitely available if the deployment target
  // is 10.9 or newer.
  if (!proc_get_wakemon_params) {
    errno = ENOSYS;
    return -1;
  }
#endif

  return proc_get_wakemon_params(pid, rate_hz, flags);
}

}  // namespace

namespace crashpad {

exception_type_t ExcCrashRecoverOriginalException(
    mach_exception_code_t code_0,
    mach_exception_code_t* original_code_0,
    int* signal) {
  // 10.9.4 xnu-2422.110.17/bsd/kern/kern_exit.c proc_prepareexit() sets code[0]
  // based on the signal value, original exception type, and low 20 bits of the
  // original code[0] before calling xnu-2422.110.17/osfmk/kern/exception.c
  // task_exception_notify() to raise an EXC_CRASH.
  //
  // The list of core-generating signals (as used in proc_prepareexit()’s call
  // to hassigprop()) is in 10.9.4 xnu-2422.110.17/bsd/sys/signalvar.h sigprop:
  // entires with SA_CORE are in the set. These signals are SIGQUIT, SIGILL,
  // SIGTRAP, SIGABRT, SIGEMT, SIGFPE, SIGBUS, SIGSEGV, and SIGSYS. Processes
  // killed for code-signing reasons will be killed by SIGKILL and are also
  // eligible for EXC_CRASH handling, but processes killed by SIGKILL for other
  // reasons are not.
  if (signal) {
    *signal = (code_0 >> 24) & 0xff;
  }

  if (original_code_0) {
    *original_code_0 = code_0 & 0xfffff;
  }

  return (code_0 >> 20) & 0xf;
}

bool ExcCrashCouldContainException(exception_type_t exception) {
  // EXC_CRASH should never be wrapped in another EXC_CRASH.
  //
  // EXC_RESOURCE and EXC_GUARD are software exceptions that are never wrapped
  // in EXC_CRASH. The only time EXC_CRASH is generated is for processes exiting
  // due to an unhandled core-generating signal or being killed by SIGKILL for
  // code-signing reasons. Neither of these apply to EXC_RESOURCE or EXC_GUARD.
  // See 10.10 xnu-2782.1.97/bsd/kern/kern_exit.c proc_prepareexit(). Receiving
  // these exception types wrapped in EXC_CRASH would lose information because
  // their code[0] uses all 64 bits (see ExceptionSnapshotMac::Initialize()) and
  // the code[0] recovered from EXC_CRASH only contains 20 significant bits.
  //
  // EXC_CORPSE_NOTIFY may be generated from EXC_CRASH, but the opposite should
  // never occur.
  //
  // kMachExceptionSimulated is a non-fatal Crashpad-specific pseudo-exception
  // that never exists as an exception within the kernel and should thus never
  // be wrapped in EXC_CRASH.
  return exception != EXC_CRASH &&
         exception != EXC_RESOURCE &&
         exception != EXC_GUARD &&
         exception != EXC_CORPSE_NOTIFY &&
         exception != kMachExceptionSimulated;
}

int32_t ExceptionCodeForMetrics(exception_type_t exception,
                                mach_exception_code_t code_0) {
  if (exception == kMachExceptionSimulated) {
    return exception;
  }

  int signal = 0;
  if (exception == EXC_CRASH) {
    const exception_type_t original_exception =
        ExcCrashRecoverOriginalException(code_0, &code_0, &signal);
    if (!ExcCrashCouldContainException(original_exception)) {
      LOG(WARNING) << "EXC_CRASH should not contain exception "
                   << original_exception;
      return InRangeCast<uint16_t>(original_exception, 0xffff) << 16;
    }
    exception = original_exception;
  }

  uint16_t metrics_exception = InRangeCast<uint16_t>(exception, 0xffff);

  uint16_t metrics_code_0;
  switch (exception) {
    case EXC_RESOURCE:
      metrics_code_0 = (EXC_RESOURCE_DECODE_RESOURCE_TYPE(code_0) << 8) |
                       EXC_RESOURCE_DECODE_FLAVOR(code_0);
      break;

    case EXC_GUARD: {
      // This will be GUARD_TYPE_MACH_PORT (1) from <mach/port.h> or
      // GUARD_TYPE_FD (2) from 10.12.2 xnu-3789.31.2/bsd/sys/guarded.h
      const uint8_t guard_type = (code_0) >> 61;

      // These exceptions come through 10.12.2
      // xnu-3789.31.2/osfmk/ipc/mach_port.c mach_port_guard_exception() or
      // xnu-3789.31.2/bsd/kern/kern_guarded.c fp_guard_exception(). In each
      // case, bits 32-60 of code_0 encode the guard type-specific “flavor”. For
      // Mach port guards, these flavor codes come from the
      // mach_port_guard_exception_codes enum in <mach/port.h>. For file
      // descriptor guards, they come from the guard_exception_codes enum in
      // xnu-3789.31.2/bsd/sys/guarded.h. Both of these enums define shifted-bit
      // values (1 << 0, 1 << 1, 1 << 2, etc.) In actual usage as determined by
      // callers to these functions, these “flavor” codes are never ORed with
      // one another. For the purposes of encoding these codes for metrics,
      // convert the flavor codes to their corresponding bit shift values.
      const uint32_t guard_flavor = (code_0 >> 32) & 0x1fffffff;
      const int first_bit = ffs(guard_flavor);
      uint8_t metrics_guard_flavor;
      if (first_bit) {
        metrics_guard_flavor = first_bit - 1;

        const uint32_t test_guard_flavor = 1 << metrics_guard_flavor;
        if (guard_flavor != test_guard_flavor) {
          // Another bit is set.
          DCHECK_EQ(guard_flavor, test_guard_flavor);
          metrics_guard_flavor = 0xff;
        }
      } else {
        metrics_guard_flavor = 0xff;
      }

      metrics_code_0 = (guard_type << 8) | metrics_guard_flavor;
      break;
    }

    case EXC_CORPSE_NOTIFY:
      // code_0 may be a pointer. See 10.12.2 xnu-3789.31.2/osfmk/kern/task.c
      // task_deliver_crash_notification(). Just encode 0 for metrics purposes.
      metrics_code_0 = 0;
      break;

    default:
      metrics_code_0 = InRangeCast<uint16_t>(code_0, 0xffff);
      if (exception == 0 && metrics_code_0 == 0 && signal != 0) {
        // This exception came from a signal that did not originate as another
        // Mach exception. Encode the signal number, using EXC_CRASH as the
        // top-level exception type. This is safe because EXC_CRASH will not
        // otherwise appear as metrics_exception.
        metrics_exception = EXC_CRASH;
        metrics_code_0 = signal;
      }
      break;
  }

  return (metrics_exception << 16) | metrics_code_0;
}

bool IsExceptionNonfatalResource(exception_type_t exception,
                                 mach_exception_code_t code_0,
                                 pid_t pid) {
  if (exception != EXC_RESOURCE) {
    return false;
  }

  const int resource_type = EXC_RESOURCE_DECODE_RESOURCE_TYPE(code_0);
  const int resource_flavor = EXC_RESOURCE_DECODE_FLAVOR(code_0);

  if (resource_type == RESOURCE_TYPE_CPU &&
      (resource_flavor == FLAVOR_CPU_MONITOR ||
       resource_flavor == FLAVOR_CPU_MONITOR_FATAL)) {
    // These exceptions may be fatal. They are not fatal by default at task
    // creation but can be made fatal by calling proc_rlimit_control() with
    // RLIMIT_CPU_USAGE_MONITOR as the second argument and CPUMON_MAKE_FATAL set
    // in the flags.
    if (MacOSXMinorVersion() >= 10) {
      // In OS X 10.10, the exception code indicates whether the exception is
      // fatal. See 10.10 xnu-2782.1.97/osfmk/kern/thread.c
      // THIS_THREAD_IS_CONSUMING_TOO_MUCH_CPU__SENDING_EXC_RESOURCE().
      return resource_flavor == FLAVOR_CPU_MONITOR;
    }

    // In OS X 10.9, there’s no way to determine whether the exception is fatal.
    // Unlike RESOURCE_TYPE_WAKEUPS below, there’s no way to determine this
    // outside the kernel. proc_rlimit_control()’s RLIMIT_CPU_USAGE_MONITOR is
    // the only interface to modify CPUMON_MAKE_FATAL, but it’s only able to set
    // this bit, not obtain its current value.
    //
    // Default to assuming that these exceptions are nonfatal. They are nonfatal
    // by default and no users of proc_rlimit_control() were found on 10.9.5
    // 13F1066 in /System and /usr outside of Metadata.framework and associated
    // tools.
    return true;
  }

  if (resource_type == RESOURCE_TYPE_WAKEUPS &&
      resource_flavor == FLAVOR_WAKEUPS_MONITOR) {
    // These exceptions may be fatal. They are not fatal by default at task
    // creation, but can be made fatal by calling proc_rlimit_control() with
    // RLIMIT_WAKEUPS_MONITOR as the second argument and WAKEMON_MAKE_FATAL set
    // in the flags.
    //
    // proc_get_wakemon_params() (which calls
    // through to proc_rlimit_control() with RLIMIT_WAKEUPS_MONITOR) determines
    // whether these exceptions are fatal. See 10.10
    // xnu-2782.1.97/osfmk/kern/task.c
    // THIS_PROCESS_IS_CAUSING_TOO_MANY_WAKEUPS__SENDING_EXC_RESOURCE().
    //
    // If proc_get_wakemon_params() fails, default to assuming that these
    // exceptions are nonfatal. They are nonfatal by default and no users of
    // proc_rlimit_control() were found on 10.9.5 13F1066 in /System and /usr
    // outside of Metadata.framework and associated tools.
    int wm_rate;
    int wm_flags;
    int rv = ProcGetWakemonParams(pid, &wm_rate, &wm_flags);
    if (rv < 0) {
      PLOG(WARNING) << "ProcGetWakemonParams";
      return true;
    }

    return !(wm_flags & WAKEMON_MAKE_FATAL);
  }

  if (resource_type == RESOURCE_TYPE_MEMORY &&
      resource_flavor == FLAVOR_HIGH_WATERMARK) {
    // These exceptions were never fatal prior to 10.12. See 10.10
    // xnu-2782.1.97/osfmk/kern/task.c
    // THIS_PROCESS_CROSSED_HIGH_WATERMARK__SENDING_EXC_RESOURCE().
    //
    // A superficial examination of 10.12 shows that these exceptions may be
    // fatal, as determined by the P_MEMSTAT_FATAL_MEMLIMIT bit of the
    // kernel-internal struct proc::p_memstat_state. See 10.12.3
    // xnu-3789.41.3/osfmk/kern/task.c task_footprint_exceeded(). This bit is
    // not exposed to user space, which makes it difficult to determine whether
    // the kernel considers a given instance of this exception fatal. However, a
    // close read reveals that it is only possible for this bit to become set
    // when xnu-3789.41.3/bsd/kern/kern_memorystatus.c
    // memorystatus_cmd_set_memlimit_properties() is called, which is only
    // possible when the kernel is built with CONFIG_JETSAM set, or if the
    // kern.memorystatus_highwater_enabled sysctl is used, which is only
    // possible when the kernel is built with DEVELOPMENT or DEBUG set. Although
    // CONFIG_JETSAM is used on iOS, it is not used on macOS. DEVELOPMENT and
    // DEBUG are also not set for production kernels. It therefore remains
    // impossible for these exceptions to be fatal, even on 10.12.
    return true;
  }

  if (resource_type == RESOURCE_TYPE_IO) {
    // These exceptions are never fatal. See 10.12.3
    // xnu-3789.41.3/osfmk/kern/task.c
    // SENDING_NOTIFICATION__THIS_PROCESS_IS_CAUSING_TOO_MUCH_IO().
    return true;
  }

  // Treat unknown exceptions as fatal. This is the conservative approach: it
  // may result in more crash reports being generated, but the type-flavor
  // combinations can be evaluated to determine appropriate handling.
  LOG(WARNING) << "unknown resource type " << resource_type << " flavor "
               << resource_flavor;
  return false;
}

}  // namespace crashpad
