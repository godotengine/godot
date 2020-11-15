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

#include "util/mach/exc_client_variants.h"

#include <sys/types.h>

#include <vector>

#include "base/logging.h"
#include "util/mach/exc.h"
#include "util/mach/mach_exc.h"

namespace crashpad {

kern_return_t UniversalExceptionRaise(exception_behavior_t behavior,
                                      exception_handler_t exception_port,
                                      thread_t thread,
                                      task_t task,
                                      exception_type_t exception,
                                      const mach_exception_data_type_t* code,
                                      mach_msg_type_number_t code_count,
                                      thread_state_flavor_t* flavor,
                                      ConstThreadState old_state,
                                      mach_msg_type_number_t old_state_count,
                                      thread_state_t new_state,
                                      mach_msg_type_number_t* new_state_count) {
  // This function is similar to 10.9.4 xnu-2422.110.17/osfmk/kern/exception.c
  // exception_deliver() as far as the delivery logic is concerned. Unlike
  // exception_deliver(), this function does not get or set thread states for
  // behavior values that require this, as that is left to the caller to do if
  // needed.

  std::vector<exception_data_type_t> small_code_vector;
  exception_data_t small_code = nullptr;
  if ((behavior & MACH_EXCEPTION_CODES) == 0 && code_count) {
    small_code_vector.reserve(code_count);
    for (size_t code_index = 0; code_index < code_count; ++code_index) {
      small_code_vector.push_back(code[code_index]);
    }
    small_code = &small_code_vector[0];
  }

  // The *exception_raise*() family has bad declarations. Their code and
  // old_state arguments aren’t pointers to const data, although they should be.
  // The generated stubs in excUser.c and mach_excUser.c make it clear that the
  // data is never modified, and these parameters could be declared with const
  // appropriately. The uses of const_cast below are thus safe.

  switch (behavior) {
    case EXCEPTION_DEFAULT:
      return exception_raise(
          exception_port, thread, task, exception, small_code, code_count);

    case EXCEPTION_STATE:
      return exception_raise_state(exception_port,
                                   exception,
                                   small_code,
                                   code_count,
                                   flavor,
                                   const_cast<thread_state_t>(old_state),
                                   old_state_count,
                                   new_state,
                                   new_state_count);

    case EXCEPTION_STATE_IDENTITY:
      return exception_raise_state_identity(
          exception_port,
          thread,
          task,
          exception,
          small_code,
          code_count,
          flavor,
          const_cast<thread_state_t>(old_state),
          old_state_count,
          new_state,
          new_state_count);

    case EXCEPTION_DEFAULT | kMachExceptionCodes:
      return mach_exception_raise(exception_port,
                                  thread,
                                  task,
                                  exception,
                                  const_cast<mach_exception_data_type_t*>(code),
                                  code_count);

    case EXCEPTION_STATE | kMachExceptionCodes:
      return mach_exception_raise_state(
          exception_port,
          exception,
          const_cast<mach_exception_data_type_t*>(code),
          code_count,
          flavor,
          const_cast<thread_state_t>(old_state),
          old_state_count,
          new_state,
          new_state_count);

    case EXCEPTION_STATE_IDENTITY | kMachExceptionCodes:
      return mach_exception_raise_state_identity(
          exception_port,
          thread,
          task,
          exception,
          const_cast<mach_exception_data_type_t*>(code),
          code_count,
          flavor,
          const_cast<thread_state_t>(old_state),
          old_state_count,
          new_state,
          new_state_count);

    default:
      NOTREACHED();
      return KERN_INVALID_ARGUMENT;
  }
}

}  // namespace crashpad
