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

#ifndef CRASHPAD_UTIL_MACH_EXCEPTION_PORTS_H_
#define CRASHPAD_UTIL_MACH_EXCEPTION_PORTS_H_

#include <mach/mach.h>

#include <vector>

#include "base/macros.h"

namespace crashpad {

//! \brief A better interface to `*_get_exception_ports()` and
//!     `*_set_exception_ports()`.
//!
//! The same generic interface can be used to operate on host, task, and thread
//! exception ports. The “get” interface is superior to the system’s native
//! interface because it keeps related data about a single exception handler
//! together in one struct, rather than separating it into four parallel arrays.
class ExceptionPorts {
 public:
  //! \brief Various entities which can have their own exception ports set.
  enum TargetType {
    //! \brief The host exception target.
    //!
    //! `host_get_exception_ports()` and `host_set_exception_ports()` will be
    //! used. If no target port is explicitly provided, `mach_host_self()` will
    //! be used as the target port. `mach_host_self()` is the only target port
    //! for this type that is expected to function properly.
    //!
    //! \note Operations on this target type are not expected to succeed as
    //!     non-root, because `mach_host_self()` doesn’t return the privileged
    //!     `host_priv` port to non-root users, and this is the target port
    //!     that’s required for `host_get_exception_ports()` and
    //!     `host_set_exception_ports()`.
    kTargetTypeHost = 0,

    //! \brief A task exception target.
    //!
    //! `task_get_exception_ports()` and `task_set_exception_ports()` will be
    //! used. If no target port is explicitly provided, `mach_task_self()` will
    //! be used as the target port.
    kTargetTypeTask,

    //! \brief A thread exception target.
    //!
    //! `thread_get_exception_ports()` and `thread_set_exception_ports()` will
    //! be used. If no target port is explicitly provided, `mach_thread_self()`
    //! will be used as the target port.
    kTargetTypeThread,
  };

  //! \brief Information about a registered exception handler.
  struct ExceptionHandler {
    //! \brief A mask specifying the exception types to direct to \a port,
    //!     containing `EXC_MASK_*` values.
    exception_mask_t mask;

    //! \brief A send right to a Mach port that will handle exceptions of the
    //!     types indicated in \a mask.
    exception_handler_t port;

    //! \brief The “behavior” that the exception handler at \a port implements:
    //!     `EXCEPTION_DEFAULT`, `EXCEPTION_STATE`, or
    //!     `EXCEPTION_STATE_IDENTITY`, possibly combined with
    //!     `MACH_EXCEPTION_CODES`.
    exception_behavior_t behavior;

    //! \brief The thread state flavor that the exception handler at \a port
    //!     will receive and possibly modify. This member has no effect for \a
    //!     \a behavior values that indicate a “default” behavior.
    thread_state_flavor_t flavor;
  };

  //! \brief Wraps `std::vector<ExceptionHandler>`, providing proper cleanup of
  //!     the send rights contained in each element’s ExceptionHandler::port.
  //!
  //! Upon destruction or clear(), an object of this class will deallocate all
  //! send rights it contains. Otherwise, it is an interface-compatible drop-in
  //! replacement for `std::vector<ExceptionHandler>`. Note that non-`const`
  //! mutators are not provided to avoid accidental Mach right leaks.
  class ExceptionHandlerVector {
   public:
    using VectorType = std::vector<ExceptionHandler>;

    ExceptionHandlerVector();
    ~ExceptionHandlerVector();

    VectorType::const_iterator begin() const { return vector_.begin(); }
    VectorType::const_iterator end() const { return vector_.end(); }
    VectorType::size_type size() const { return vector_.size(); }
    bool empty() const { return vector_.empty(); }
    VectorType::const_reference operator[](VectorType::size_type index) const {
      return vector_[index];
    }
    void push_back(VectorType::value_type& value) { vector_.push_back(value); }
    void clear();

   private:
    void Deallocate();

    VectorType vector_;

    DISALLOW_COPY_AND_ASSIGN(ExceptionHandlerVector);
  };

  //! \brief Constructs an interface object to get or set exception ports on a
  //!     host, task, or thread port.
  //!
  //! \param[in] target_type The type of target on which the exception ports are
  //!     to be get or set: #kTargetTypeHost, #kTargetTypeTask, or or
  //!     #kTargetTypeThread. The correct functions for
  //!     `*_get_exception_ports()` and `*_set_exception_ports()` will be used.
  //! \param[in] target_port The target on which to call
  //!     `*_get_exception_ports()` or `*_set_exception_ports()`. The target
  //!     port must be a send right to a port of the type specified in \a
  //!     target_type. In this case, ownership of \a target_port is not given to
  //!     the new ExceptionPorts object. \a target_port may also be
  //!     `HOST_NULL`, `TASK_NULL`, or `THREAD_NULL`, in which case
  //!     `mach_host_self()`, `mach_task_self()`, or `mach_thread_self()` will
  //!     be used as the target port depending on the value of \a target_type.
  //!     In this case, ownership of the target port will be managed
  //!     appropriately for \a target_type.
  ExceptionPorts(TargetType target_type, mach_port_t target_port);

  ~ExceptionPorts();

  //! \brief Calls `*_get_exception_ports()` on the target.
  //!
  //! \param[in] mask The exception mask, containing the `EXC_MASK_*` values to
  //!     be looked up and returned in \a handlers.
  //! \param[out] handlers The exception handlers registered for \a target_port
  //!     to handle exceptions indicated in \a mask. If no execption port is
  //!     registered for a bit in \a mask, \a handlers will not contain an entry
  //!     corresponding to that bit. This is a departure from the
  //!     `*_get_exception_ports()` functions, which may return a handler whose
  //!     port is set to `EXCEPTION_PORT_NULL` in this case. On failure, this
  //!     argument is untouched.
  //!
  //! \return `true` if `*_get_exception_ports()` returned `KERN_SUCCESS`, with
  //!     \a handlers set appropriately. `false` otherwise, with an appropriate
  //!     message logged.
  bool GetExceptionPorts(exception_mask_t mask,
                         ExceptionHandlerVector* handlers) const;

  //! \brief Calls `*_set_exception_ports()` on the target.
  //!
  //! \param[in] mask A mask specifying the exception types to direct to \a
  //!     port, containing `EXC_MASK_*` values.
  //! \param[in] port A send right to a Mach port that will handle exceptions
  //!     sustained by \a target_port of the types indicated in \a mask. The
  //!     send right is copied, not consumed, by this call.
  //! \param[in] behavior The “behavior” that the exception handler at \a port
  //!     implements: `EXCEPTION_DEFAULT`, `EXCEPTION_STATE`, or
  //!     `EXCEPTION_STATE_IDENTITY`, possibly combined with
  //!     `MACH_EXCEPTION_CODES`.
  //! \param[in] flavor The thread state flavor that the exception handler at \a
  //!     port expects to receive and possibly modify. This argument has no
  //!     effect for \a behavior values that indicate a “default” behavior.
  //!
  //! \return `true` if `*_set_exception_ports()` returned `KERN_SUCCESS`.
  //!     `false` otherwise, with an appropriate message logged.
  bool SetExceptionPort(exception_mask_t mask,
                        exception_handler_t port,
                        exception_behavior_t behavior,
                        thread_state_flavor_t flavor) const;

  //! \brief Returns a string identifying the target type.
  //!
  //! \return `"host"`, `"task"`, or `"thread"`, as appropriate.
  const char* TargetTypeName() const;

 private:
  using GetExceptionPortsType = kern_return_t(*)(mach_port_t,
                                                 exception_mask_t,
                                                 exception_mask_array_t,
                                                 mach_msg_type_number_t*,
                                                 exception_handler_array_t,
                                                 exception_behavior_array_t,
                                                 exception_flavor_array_t);

  using SetExceptionPortsType = kern_return_t(*)(mach_port_t,
                                                 exception_mask_t,
                                                 exception_handler_t,
                                                 exception_behavior_t,
                                                 thread_state_flavor_t);

  GetExceptionPortsType get_exception_ports_;
  SetExceptionPortsType set_exception_ports_;
  const char* target_name_;
  mach_port_t target_port_;

  // If true, target_port_ will be deallocated in the destructor. This will
  // always be false when the user provides a non-null target_port to the
  // constructor. It will also be false when target_type is kTargetTypeTask,
  // even with a TASK_NULL target_port, because it is incorrect to deallocate
  // the result of mach_task_self().
  bool dealloc_target_port_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionPorts);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_EXCEPTION_PORTS_H_
