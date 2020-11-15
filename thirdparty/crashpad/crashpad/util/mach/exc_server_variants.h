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

#ifndef CRASHPAD_UTIL_MACH_EXC_SERVER_VARIANTS_H_
#define CRASHPAD_UTIL_MACH_EXC_SERVER_VARIANTS_H_

#include <mach/mach.h>

#include <memory>
#include <set>

#include "base/macros.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message_server.h"

namespace crashpad {

namespace internal {
class UniversalMachExcServerImpl;
}  // namespace internal

//! \brief A server interface for the `exc` and `mach_exc` Mach subsystems,
//!     unified to handle exceptions delivered to either subsystem, and
//!     simplified to have only a single interface method needing
//!     implementation.
//!
//! The `<mach/exc.defs>` and `<mach/mach_exc.defs>` interfaces are identical,
//! except that the latter allows for 64-bit exception codes, and is requested
//! by setting the MACH_EXCEPTION_CODES behavior bit associated with an
//! exception port.
//!
//! UniversalMachExcServer operates by translating messages received in the
//! `exc` subsystem to a variant that is compatible with the `mach_exc`
//! subsystem. This involves changing the format of \a code, the exception code
//! field, from `exception_data_type_t` to `mach_exception_data_type_t`.
class UniversalMachExcServer final : public MachMessageServer::Interface {
 public:
  //! \brief An interface that the different request messages that are a part of
  //!     the `exc` and `mach_exc` Mach subsystems can be dispatched to.
  class Interface {
   public:
    //! \brief Handles exceptions raised by `exception_raise()`,
    //!     `exception_raise_state()`, `exception_raise_state_identity()`,
    //!     `mach_exception_raise()`, `mach_exception_raise_state()`, and
    //!     `mach_exception_raise_state_identity()`.
    //!
    //! For convenience in implementation, these different “behaviors” of
    //! exception messages are all mapped to a single interface method. The
    //! exception’s original “behavior” is specified in the \a behavior
    //! parameter. Only parameters that were supplied in the request message
    //! are populated, other parameters are set to reasonable default values.
    //!
    //! This behaves equivalently to a `catch_exception_raise_state_identity()`
    //! function used with `exc_server()`, or a
    //! `catch_mach_exception_raise_state_identity()` function used with
    //! `mach_exc_server()`. Except as noted, the parameters and return value
    //! are equivalent to those of these other functions.
    //!
    //! \param[in] behavior `EXCEPTION_DEFAULT`, `EXCEPTION_STATE`,
    //!     or `EXCEPTION_STATE_IDENTITY`, possibly with `MACH_EXCEPTION_CODES`
    //!     ORed in. This identifies which exception request message was
    //!     processed and thus which other parameters are valid.
    //! \param[in] exception_port
    //! \param[in] thread
    //! \param[in] task
    //! \param[in] exception
    //! \param[in] code
    //! \param[in] code_count
    //! \param[in,out] flavor
    //! \param[in] old_state
    //! \param[in] old_state_count
    //! \param[out] new_state
    //! \param[out] new_state_count
    //! \param[in] trailer The trailer received with the request message.
    //! \param[out] destroy_complex_request `true` if the request message is to
    //!     be destroyed even when this method returns success. See
    //!     MachMessageServer::Interface.
    //!
    //! \return A code indicating whether the exception was handled. See
    //!     ExcServerSuccessfulReturnValue() for success codes. On failure,
    //!     a code such as `KERN_FAILURE`.
    virtual kern_return_t CatchMachException(
        exception_behavior_t behavior,
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
        mach_msg_type_number_t* new_state_count,
        const mach_msg_trailer_t* trailer,
        bool* destroy_complex_request) = 0;

   protected:
    ~Interface() {}
  };

  //! \brief Constructs an object of this class.
  //!
  //! \param[in] interface The interface to dispatch requests to. Weak.
  explicit UniversalMachExcServer(Interface* interface);

  ~UniversalMachExcServer();

  // MachMessageServer::Interface:
  bool MachMessageServerFunction(const mach_msg_header_t* in_header,
                                 mach_msg_header_t* out_header,
                                 bool* destroy_complex_request) override;
  std::set<mach_msg_id_t> MachMessageServerRequestIDs() override;
  mach_msg_size_t MachMessageServerRequestSize() override;
  mach_msg_size_t MachMessageServerReplySize() override;

 private:
  std::unique_ptr<internal::UniversalMachExcServerImpl> impl_;

  DISALLOW_COPY_AND_ASSIGN(UniversalMachExcServer);
};

//! \brief Computes an approriate successful return value for an exception
//!     handler function.
//!
//! For exception handlers that respond to state-carrying behaviors, when the
//! handler is called by the kernel (as it is normally), the kernel will attempt
//! to set a new thread state when the exception handler returns successfully.
//! Other code that mimics the kernel’s exception-delivery semantics may
//! implement the same or similar behavior. In some situations, it is
//! undesirable to set a new thread state. If the exception handler were to
//! return unsuccessfully, however, the kernel would continue searching for an
//! exception handler at a wider (task or host) scope. This may also be
//! undesirable.
//!
//! If such exception handlers return `MACH_RCV_PORT_DIED`, the kernel will not
//! set a new thread state and will also not search for another exception
//! handler. See 10.9.4 `xnu-2422.110.17/osfmk/kern/exception.c`.
//! `exception_deliver()` will only set a new thread state if the handler’s
//! return code was `MACH_MSG_SUCCESS` (a synonym for `KERN_SUCCESS`), and
//! subsequently, `exception_triage()` will not search for a new handler if the
//! handler’s return code was `KERN_SUCCESS` or `MACH_RCV_PORT_DIED`.
//!
//! This function allows exception handlers to compute an appropriate return
//! code to influence their caller (the kernel) in the desired way with respect
//! to setting a new thread state while suppressing the caller’s subsequent
//! search for other exception handlers. An exception handler should return the
//! value returned by this function.
//!
//! This function is useful even for `EXC_CRASH` handlers, where returning
//! `KERN_SUCCESS` and allowing the kernel to set a new thread state has been
//! observed to cause a perceptible and unnecessary waste of time. The victim
//! task in an `EXC_CRASH` handler is already being terminated and is no longer
//! schedulable, so there is no point in setting the states of any of its
//! threads.
//!
//! On OS X 10.11, the `MACH_RCV_PORT_DIED` mechanism cannot be used with an
//! `EXC_CRASH` handler without triggering an undesirable `EXC_CORPSE_NOTIFY`
//! exception. In that case, `KERN_SUCCESS` is always returned. Because this
//! function may return `KERN_SUCCESS` for a state-carrying exception, it is
//! important to ensure that the state returned by a state-carrying exception
//! handler is valid, because it will be passed to `thread_set_status()`.
//! ExcServerCopyState() may be used to achieve this.
//!
//! \param[in] exception The exception type passed to the exception handler.
//!     This may be taken directly from the \a exception parameter of
//!     internal::SimplifiedExcServer::Interface::CatchException(), for example.
//! \param[in] behavior The behavior of the exception handler as invoked. This
//!     may be taken directly from the \a behavior parameter of
//!     internal::SimplifiedExcServer::Interface::CatchException(), for example.
//! \param[in] set_thread_state `true` if the handler would like its caller to
//!     set the new thread state using the \a flavor, \a new_state, and \a
//!     new_state_count out parameters. This can only happen when \a behavior is
//!     a state-carrying behavior.
//!
//! \return `KERN_SUCCESS` or `MACH_RCV_PORT_DIED`. `KERN_SUCCESS` is used when
//!     \a behavior is not a state-carrying behavior, or when it is a
//!     state-carrying behavior and \a set_thread_state is `true`, or for
//!     `EXC_CRASH` exceptions on OS X 10.11 and later. Otherwise,
//!     `MACH_RCV_PORT_DIED` is used.
kern_return_t ExcServerSuccessfulReturnValue(exception_type_t exception,
                                             exception_behavior_t behavior,
                                             bool set_thread_state);

//! \brief Copies the old state to the new state for state-carrying exceptions.
//!
//! When the kernel sends a state-carrying exception request and the response is
//! successful (`MACH_MSG_SUCCESS`, a synonym for `KERN_SUCCESS`), it will set
//! a new thread state based on \a new_state and \a new_state_count. To ease
//! initialization of the new state, this function copies \a old_state and
//! \a old_state_count. This is only done if \a behavior indicates a
//! state-carrying exception.
//!
//! \param[in] behavior The behavior of the exception handler as invoked. This
//!     may be taken directly from the \a behavior parameter of
//!     internal::SimplifiedExcServer::Interface::CatchException(), for example.
//! \param[in] old_state The original state value. This may be taken directly
//!     from the \a old_state parameter of
//!     internal::SimplifiedExcServer::Interface::CatchException(), for example.
//! \param[in] old_state_count The number of significant `natural_t` words in \a
//!     old_state. This may be taken directly from the \a old_state_count
//!     parameter of internal::SimplifiedExcServer::Interface::CatchException(),
//!     for example.
//! \param[out] new_state The state value to be set. This may be taken directly
//!     from the \a new_state parameter of
//!     internal::SimplifiedExcServer::Interface::CatchException(), for example.
//!     This parameter is untouched if \a behavior is not state-carrying.
//! \param[in,out] new_state_count On entry, the number of `natural_t` words
//!     available to be written to in \a new_state. On return, the number of
//!     significant `natural_t` words in \a new_state. This may be taken
//!     directly from the \a new_state_count parameter of
//!     internal::SimplifiedExcServer::Interface::CatchException(), for example.
//!     This parameter is untouched if \a behavior is not state-carrying. If \a
//!     \a behavior is state-carrying, this parameter should be at least as
//!     large as \a old_state_count.
void ExcServerCopyState(exception_behavior_t behavior,
                        ConstThreadState old_state,
                        mach_msg_type_number_t old_state_count,
                        thread_state_t new_state,
                        mach_msg_type_number_t* new_state_count);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_EXC_SERVER_VARIANTS_H_
