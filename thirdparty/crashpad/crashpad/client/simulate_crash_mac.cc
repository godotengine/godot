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

#include "client/simulate_crash_mac.h"

#include <string.h>
#include <sys/types.h>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/macros.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "util/mach/exc_client_variants.h"
#include "util/mach/exception_behaviors.h"
#include "util/mach/exception_ports.h"
#include "util/mach/mach_extensions.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {

namespace {

//! \brief Sends an exception message to an exception port in accordance with
//!     the behavior and thread state flavor it’s registered to receive.
//!
//! \param[in] thread, task, exception, code, code_count These parameters will
//!     be passed to the exception handler as appropriate.
//! \param[in] cpu_context The value to use for the thread state, if \a behavior
//!     indicates that the handler should receive a thread state and if the
//!     supplied thread state matches or can be converted to \a flavor. If \a
//!     behavior requires a thread state but this argument cannot be converted
//!     to match \a flavor, `thread_get_state()` will be called to obtain a
//!     suitable thread state value.
//! \param[in] handler The Mach exception handler to deliver the exception to.
//! \param[in] set_state If `true` and \a behavior indicates that the handler
//!     should receive and return a thread state, a new thread state will be set
//!     by `thread_set_state()` upon successful completion of the exception
//!     handler. If `false`, this will be suppressed, even when \a behavior
//!     indicates that the handler receives and returns a thread state.
//!
//! \return `true` if the exception was delivered to the handler and the handler
//!     indicated success. `false` otherwise, with a warning message logged.
bool DeliverException(thread_t thread,
                      task_t task,
                      exception_type_t exception,
                      const mach_exception_data_t code,
                      mach_msg_type_number_t code_count,
                      const NativeCPUContext& cpu_context,
                      const ExceptionPorts::ExceptionHandler& handler,
                      bool set_state) {
  kern_return_t kr;

  bool handler_wants_state = ExceptionBehaviorHasState(handler.behavior);
  if (!handler_wants_state) {
    // Regardless of the passed-in value of |set_state|, if the handler won’t be
    // dealing with any state at all, no state should be set.
    set_state = false;
  }

  // old_state is only used if the context already captured doesn’t match (or
  // can’t be converted to) what’s registered for the handler.
  thread_state_data_t old_state;

  thread_state_flavor_t flavor = handler.flavor;
  ConstThreadState state;
  mach_msg_type_number_t state_count;
  switch (flavor) {
#if defined(ARCH_CPU_X86_FAMILY)
    case x86_THREAD_STATE:
      state = reinterpret_cast<ConstThreadState>(&cpu_context);
      state_count = x86_THREAD_STATE_COUNT;
      break;
#if defined(ARCH_CPU_X86)
    case x86_THREAD_STATE32:
      state = reinterpret_cast<ConstThreadState>(&cpu_context.uts.ts32);
      state_count = cpu_context.tsh.count;
      break;
#elif defined(ARCH_CPU_X86_64)
    case x86_THREAD_STATE64:
      state = reinterpret_cast<ConstThreadState>(&cpu_context.uts.ts64);
      state_count = cpu_context.tsh.count;
      break;
#endif
#else
#error Port to your CPU architecture
#endif

    case THREAD_STATE_NONE:
      // This is only acceptable if the handler doesn’t have one of the “state”
      // behaviors. Otherwise, if the kernel were attempting to send an
      // exception message to this port, it would call thread_getstatus() (known
      // outside the kernel as thread_get_state()) which would fail because
      // THREAD_STATE_NONE is not a valid state to get. See 10.9.5
      // xnu-2422.115.4/osfmk/kern/exception.c exception_deliver() and
      // xnu-2422.115.4/osfmk/i386/pcb.c machine_thread_get_state().
      if (!handler_wants_state) {
        state = nullptr;
        state_count = 0;
        break;
      }

      LOG(WARNING) << "exception handler has unexpected state flavor" << flavor;
      return false;

    default:
      if (!handler_wants_state) {
        // Don’t bother getting any thread state if the handler’s not actually
        // going to use it.
        state = nullptr;
        state_count = 0;
      } else {
        state = old_state;
        state_count = THREAD_STATE_MAX;
        kr = thread_get_state(thread, flavor, old_state, &state_count);
        if (kr != KERN_SUCCESS) {
          MACH_LOG(WARNING, kr) << "thread_get_state";
          return false;
        }
      }
      break;
  }

  // new_state is supposed to be an out parameter only, but in case the handler
  // doesn't touch it, make sure it's initialized to a valid thread state.
  // Otherwise, the thread_set_state() call below would set a garbage thread
  // state.
  thread_state_data_t new_state;
  size_t state_size =
      sizeof(natural_t) *
      std::min(state_count, implicit_cast<unsigned int>(THREAD_STATE_MAX));
  memcpy(new_state, state, state_size);
  mach_msg_type_number_t new_state_count = THREAD_STATE_MAX;

  kr = UniversalExceptionRaise(handler.behavior,
                               handler.port,
                               thread,
                               task,
                               exception,
                               code,
                               code_count,
                               &flavor,
                               state,
                               state_count,
                               new_state,
                               &new_state_count);

  // The kernel treats a return value of MACH_RCV_PORT_DIED as successful,
  // although it will not set a new thread state in that case. See 10.9.5
  // xnu-2422.115.4/osfmk/kern/exception.c exception_deliver(), and the more
  // elaborate comment at util/mach/exc_server_variants.h
  // ExcServerSuccessfulReturnValue(). Duplicate that behavior.
  bool success = kr == KERN_SUCCESS || kr == MACH_RCV_PORT_DIED;
  MACH_LOG_IF(WARNING, !success, kr) << "UniversalExceptionRaise";

  if (kr == KERN_SUCCESS && set_state) {
    kr = thread_set_state(thread, flavor, new_state, new_state_count);
    MACH_LOG_IF(WARNING, kr != KERN_SUCCESS, kr) << "thread_set_state";
  }

  return success;
}

}  // namespace

void SimulateCrash(const NativeCPUContext& cpu_context) {
#if defined(ARCH_CPU_X86)
  DCHECK_EQ(implicit_cast<thread_state_flavor_t>(cpu_context.tsh.flavor),
            implicit_cast<thread_state_flavor_t>(x86_THREAD_STATE32));
  DCHECK_EQ(implicit_cast<mach_msg_type_number_t>(cpu_context.tsh.count),
            x86_THREAD_STATE32_COUNT);
#elif defined(ARCH_CPU_X86_64)
  DCHECK_EQ(implicit_cast<thread_state_flavor_t>(cpu_context.tsh.flavor),
            implicit_cast<thread_state_flavor_t>(x86_THREAD_STATE64));
  DCHECK_EQ(implicit_cast<mach_msg_type_number_t>(cpu_context.tsh.count),
            x86_THREAD_STATE64_COUNT);
#endif

  base::mac::ScopedMachSendRight thread(mach_thread_self());
  exception_type_t exception = kMachExceptionSimulated;
  mach_exception_data_type_t codes[] = {0, 0};
  mach_msg_type_number_t code_count = arraysize(codes);

  // Look up the handler for EXC_CRASH exceptions in the same way that the
  // kernel would: try a thread handler, then a task handler, and finally a host
  // handler. 10.9.5 xnu-2422.115.4/osfmk/kern/exception.c exception_triage().
  static constexpr ExceptionPorts::TargetType kTargetTypes[] = {
      ExceptionPorts::kTargetTypeThread,
      ExceptionPorts::kTargetTypeTask,

      // This is not expected to succeed, because mach_host_self() doesn’t
      // return the host_priv port to non-root users, and this is the port
      // that’s required for host_get_exception_ports().
      //
      // See 10.9.5 xnu-2422.115.4/bsd/kern/kern_prot.c set_security_token(),
      // xnu-2422.115.4/osfmk/kern/task.c host_security_set_task_token(), and
      // xnu-2422.115.4/osfmk/kern/ipc_host.c host_get_exception_ports().
      ExceptionPorts::kTargetTypeHost,
  };

  bool success = false;

  for (size_t target_type_index = 0;
       !success && target_type_index < arraysize(kTargetTypes);
       ++target_type_index) {
    ExceptionPorts::ExceptionHandlerVector handlers;
    ExceptionPorts exception_ports(kTargetTypes[target_type_index],
                                   MACH_PORT_NULL);
    if (exception_ports.GetExceptionPorts(EXC_MASK_CRASH, &handlers)) {
      DCHECK_LE(handlers.size(), 1u);
      if (handlers.size() == 1) {
        DCHECK(handlers[0].mask & EXC_MASK_CRASH);
        success = DeliverException(thread.get(),
                                   mach_task_self(),
                                   exception,
                                   codes,
                                   code_count,
                                   cpu_context,
                                   handlers[0],
                                   false);
      }
    }
  }

  LOG_IF(WARNING, !success)
      << "SimulateCrash did not find an appropriate exception handler";
}

}  // namespace crashpad
