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

#include "util/mach/mach_extensions.h"

#include <AvailabilityMacros.h>
#include <pthread.h>
#include <servers/bootstrap.h>

#include "base/mac/mach_logging.h"
#include "util/mac/mac_util.h"

namespace {

// This forms the internal implementation for BootstrapCheckIn() and
// BootstrapLookUp(), which follow the same logic aside from the routine called
// and the right type returned.

struct BootstrapCheckInTraits {
  using Type = base::mac::ScopedMachReceiveRight;
  static kern_return_t Call(mach_port_t bootstrap_port,
                            const char* service_name,
                            mach_port_t* service_port) {
    return bootstrap_check_in(bootstrap_port, service_name, service_port);
  }
  static constexpr char kName[] = "bootstrap_check_in";
};
constexpr char BootstrapCheckInTraits::kName[];

struct BootstrapLookUpTraits {
  using Type = base::mac::ScopedMachSendRight;
  static kern_return_t Call(mach_port_t bootstrap_port,
                            const char* service_name,
                            mach_port_t* service_port) {
    return bootstrap_look_up(bootstrap_port, service_name, service_port);
  }
  static constexpr char kName[] = "bootstrap_look_up";
};
constexpr char BootstrapLookUpTraits::kName[];

template <typename Traits>
typename Traits::Type BootstrapCheckInOrLookUp(
    const std::string& service_name) {
  // bootstrap_check_in() and bootstrap_look_up() silently truncate service
  // names longer than BOOTSTRAP_MAX_NAME_LEN. This check ensures that the name
  // will not be truncated.
  if (service_name.size() >= BOOTSTRAP_MAX_NAME_LEN) {
    LOG(ERROR) << Traits::kName << " " << service_name << ": name too long";
    return typename Traits::Type(MACH_PORT_NULL);
  }

  mach_port_t service_port;
  kern_return_t kr = Traits::Call(bootstrap_port,
                                  service_name.c_str(),
                                  &service_port);
  if (kr != BOOTSTRAP_SUCCESS) {
    BOOTSTRAP_LOG(ERROR, kr) << Traits::kName << " " << service_name;
    service_port = MACH_PORT_NULL;
  }

  return typename Traits::Type(service_port);
}

}  // namespace

namespace crashpad {

thread_t MachThreadSelf() {
  // The pthreads library keeps its own copy of the thread port. Using it does
  // not increment its reference count.
  return pthread_mach_thread_np(pthread_self());
}

mach_port_t NewMachPort(mach_port_right_t right) {
  mach_port_t port = MACH_PORT_NULL;
  kern_return_t kr = mach_port_allocate(mach_task_self(), right, &port);
  MACH_LOG_IF(ERROR, kr != KERN_SUCCESS, kr) << "mach_port_allocate";
  return port;
}

exception_mask_t ExcMaskAll() {
  // This is necessary because of the way that the kernel validates
  // exception_mask_t arguments to
  // {host,task,thread}_{get,set,swap}_exception_ports(). It is strict,
  // rejecting attempts to operate on any bits that it does not recognize. See
  // 10.9.4 xnu-2422.110.17/osfmk/mach/ipc_host.c and
  // xnu-2422.110.17/osfmk/mach/ipc_tt.c.

#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_9
  const int mac_os_x_minor_version = MacOSXMinorVersion();
#endif

  // See 10.6.8 xnu-1504.15.3/osfmk/mach/exception_types.h. 10.7 uses the same
  // definition as 10.6. See 10.7.5 xnu-1699.32.7/osfmk/mach/exception_types.h
  constexpr exception_mask_t kExcMaskAll_10_6 =
      EXC_MASK_BAD_ACCESS |
      EXC_MASK_BAD_INSTRUCTION |
      EXC_MASK_ARITHMETIC |
      EXC_MASK_EMULATION |
      EXC_MASK_SOFTWARE |
      EXC_MASK_BREAKPOINT |
      EXC_MASK_SYSCALL |
      EXC_MASK_MACH_SYSCALL |
      EXC_MASK_RPC_ALERT |
      EXC_MASK_MACHINE;
#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_8
  if (mac_os_x_minor_version < 8) {
    return kExcMaskAll_10_6;
  }
#endif

  // 10.8 added EXC_MASK_RESOURCE. See 10.8.5
  // xnu-2050.48.11/osfmk/mach/exception_types.h.
  constexpr exception_mask_t kExcMaskAll_10_8 =
      kExcMaskAll_10_6 | EXC_MASK_RESOURCE;
#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_9
  if (mac_os_x_minor_version < 9) {
    return kExcMaskAll_10_8;
  }
#endif

  // 10.9 added EXC_MASK_GUARD. See 10.9.4
  // xnu-2422.110.17/osfmk/mach/exception_types.h.
  constexpr exception_mask_t kExcMaskAll_10_9 =
      kExcMaskAll_10_8 | EXC_MASK_GUARD;
  return kExcMaskAll_10_9;
}

exception_mask_t ExcMaskValid() {
  const exception_mask_t kExcMaskValid_10_6 = ExcMaskAll() | EXC_MASK_CRASH;
#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_11
  if (MacOSXMinorVersion() < 11) {
    return kExcMaskValid_10_6;
  }
#endif

  // 10.11 added EXC_MASK_CORPSE_NOTIFY. See 10.11 <mach/exception_types.h>.
  const exception_mask_t kExcMaskValid_10_11 =
      kExcMaskValid_10_6 | EXC_MASK_CORPSE_NOTIFY;
  return kExcMaskValid_10_11;
}

base::mac::ScopedMachReceiveRight BootstrapCheckIn(
    const std::string& service_name) {
  return BootstrapCheckInOrLookUp<BootstrapCheckInTraits>(service_name);
}

base::mac::ScopedMachSendRight BootstrapLookUp(
    const std::string& service_name) {
  base::mac::ScopedMachSendRight send(
      BootstrapCheckInOrLookUp<BootstrapLookUpTraits>(service_name));

  // It’s possible to race the bootstrap server when the receive right
  // corresponding to the looked-up send right is destroyed immediately before
  // the bootstrap_look_up() call. If the bootstrap server believes that
  // |service_name| is still registered before processing the port-destroyed
  // notification sent to it by the kernel, it will respond to a
  // bootstrap_look_up() request with a send right that has become a dead name,
  // which will be returned to the bootstrap_look_up() caller, translated into
  // the caller’s IPC port name space, as the special MACH_PORT_DEAD port name.
  // Check for that and return MACH_PORT_NULL in its place, as though the
  // bootstrap server had fully processed the port-destroyed notification before
  // responding to bootstrap_look_up().
  if (send.get() == MACH_PORT_DEAD) {
    LOG(ERROR) << "bootstrap_look_up " << service_name << ": service is dead";
    send.reset();
  }

  return send;
}

base::mac::ScopedMachSendRight SystemCrashReporterHandler() {
  return BootstrapLookUp("com.apple.ReportCrash");
}

}  // namespace crashpad
