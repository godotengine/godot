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

#include "util/mach/exception_ports.h"

#include "base/logging.h"
#include "base/mac/mach_logging.h"

namespace crashpad {

ExceptionPorts::ExceptionHandlerVector::ExceptionHandlerVector()
    : vector_() {
}

ExceptionPorts::ExceptionHandlerVector::~ExceptionHandlerVector() {
  Deallocate();
}

void ExceptionPorts::ExceptionHandlerVector::clear() {
  Deallocate();
  vector_.clear();
}

void ExceptionPorts::ExceptionHandlerVector::Deallocate() {
  for (ExceptionHandler& exception_handler : vector_) {
    if (exception_handler.port != MACH_PORT_NULL) {
      kern_return_t kr =
          mach_port_deallocate(mach_task_self(), exception_handler.port);
      MACH_LOG_IF(ERROR, kr != KERN_SUCCESS, kr) << "mach_port_deallocate";
    }
  }
}

ExceptionPorts::ExceptionPorts(TargetType target_type, mach_port_t target_port)
    : target_port_(target_port), dealloc_target_port_(false) {
  switch (target_type) {
    case kTargetTypeHost:
      get_exception_ports_ = host_get_exception_ports;
      set_exception_ports_ = host_set_exception_ports;
      target_name_ = "host";
      if (target_port_ == HOST_NULL) {
        target_port_ = mach_host_self();
        dealloc_target_port_ = true;
      }
      break;

    case kTargetTypeTask:
      get_exception_ports_ = task_get_exception_ports;
      set_exception_ports_ = task_set_exception_ports;
      target_name_ = "task";
      if (target_port_ == TASK_NULL) {
        target_port_ = mach_task_self();
        // Don’t deallocate mach_task_self().
      }
      break;

    case kTargetTypeThread:
      get_exception_ports_ = thread_get_exception_ports;
      set_exception_ports_ = thread_set_exception_ports;
      target_name_ = "thread";
      if (target_port_ == THREAD_NULL) {
        target_port_ = mach_thread_self();
        dealloc_target_port_ = true;
      }
      break;

    default:
      NOTREACHED();
      get_exception_ports_ = nullptr;
      set_exception_ports_ = nullptr;
      target_name_ = nullptr;
      target_port_ = MACH_PORT_NULL;
      break;
  }
}

ExceptionPorts::~ExceptionPorts() {
  if (dealloc_target_port_) {
    kern_return_t kr = mach_port_deallocate(mach_task_self(), target_port_);
    MACH_LOG_IF(ERROR, kr != KERN_SUCCESS, kr) << "mach_port_deallocate";
  }
}

bool ExceptionPorts::GetExceptionPorts(exception_mask_t mask,
                                       ExceptionHandlerVector* handlers) const {
  // <mach/mach_types.defs> says that these arrays have room for 32 elements,
  // despite EXC_TYPES_COUNT only being as low as 11 (in the 10.6 SDK), and
  // later operating system versions have defined more exception types. The
  // generated task_get_exception_ports() in taskUser.c expects there to be room
  // for 32.
  constexpr int kMaxPorts = 32;

  // task_get_exception_ports() doesn’t actually use the initial value of
  // handler_count, but 10.9.4
  // xnu-2422.110.17/osfmk/man/task_get_exception_ports.html says it does. Humor
  // the documentation.
  mach_msg_type_number_t handler_count = kMaxPorts;

  exception_mask_t masks[kMaxPorts];
  exception_handler_t ports[kMaxPorts];
  exception_behavior_t behaviors[kMaxPorts];
  thread_state_flavor_t flavors[kMaxPorts];

  kern_return_t kr = get_exception_ports_(
      target_port_, mask, masks, &handler_count, ports, behaviors, flavors);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(ERROR, kr) << TargetTypeName() << "_get_exception_ports";
    return false;
  }

  handlers->clear();
  for (mach_msg_type_number_t index = 0; index < handler_count; ++index) {
    if (ports[index] != MACH_PORT_NULL) {
      ExceptionHandler handler;
      handler.mask = masks[index];
      handler.port = ports[index];
      handler.behavior = behaviors[index];
      handler.flavor = flavors[index];
      handlers->push_back(handler);
    }
  }

  return true;
}

bool ExceptionPorts::SetExceptionPort(exception_mask_t mask,
                                      exception_handler_t port,
                                      exception_behavior_t behavior,
                                      thread_state_flavor_t flavor) const {
  kern_return_t kr =
      set_exception_ports_(target_port_, mask, port, behavior, flavor);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(ERROR, kr) << TargetTypeName() << "_set_exception_ports";
    return false;
  }

  return true;
}

const char* ExceptionPorts::TargetTypeName() const {
  return target_name_;
}

}  // namespace crashpad
