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

#include "util/mach/exc_server_variants.h"

#include <AvailabilityMacros.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "base/logging.h"
#include "util/mac/mac_util.h"
#include "util/mach/composite_mach_message_server.h"
#include "util/mach/exc.h"
#include "util/mach/excServer.h"
#include "util/mach/exception_behaviors.h"
#include "util/mach/mach_exc.h"
#include "util/mach/mach_excServer.h"
#include "util/mach/mach_message.h"

namespace crashpad {

namespace {

// Traits for ExcServer<> and SimplifiedExcServer<> adapting them for use with
// the exc subsystem.
struct ExcTraits {
  using ExceptionCode = exception_data_type_t;

  using RequestUnion = __RequestUnion__exc_subsystem;
  using ReplyUnion = __ReplyUnion__exc_subsystem;

  using ExceptionRaiseRequest = __Request__exception_raise_t;
  using ExceptionRaiseStateRequest = __Request__exception_raise_state_t;
  using ExceptionRaiseStateIdentityRequest =
      __Request__exception_raise_state_identity_t;

  using ExceptionRaiseReply = __Reply__exception_raise_t;
  using ExceptionRaiseStateReply = __Reply__exception_raise_state_t;
  using ExceptionRaiseStateIdentityReply =
      __Reply__exception_raise_state_identity_t;

  // The MIG-generated __MIG_check__Request__*() functions are not declared as
  // accepting const data, but they could have been because they in fact do not
  // modify the data.

  static kern_return_t MIGCheckRequestExceptionRaise(
      const ExceptionRaiseRequest* in_request) {
    return __MIG_check__Request__exception_raise_t(
        const_cast<ExceptionRaiseRequest*>(in_request));
  }

  static kern_return_t MIGCheckRequestExceptionRaiseState(
      const ExceptionRaiseStateRequest* in_request,
      const ExceptionRaiseStateRequest** in_request_1) {
    return __MIG_check__Request__exception_raise_state_t(
        const_cast<ExceptionRaiseStateRequest*>(in_request),
        const_cast<ExceptionRaiseStateRequest**>(in_request_1));
  }

  static kern_return_t MIGCheckRequestExceptionRaiseStateIdentity(
      const ExceptionRaiseStateIdentityRequest* in_request,
      const ExceptionRaiseStateIdentityRequest** in_request_1) {
    return __MIG_check__Request__exception_raise_state_identity_t(
        const_cast<ExceptionRaiseStateIdentityRequest*>(in_request),
        const_cast<ExceptionRaiseStateIdentityRequest**>(in_request_1));
  }

  // There are no predefined constants for these.
  static const mach_msg_id_t kMachMessageIDExceptionRaise = 2401;
  static const mach_msg_id_t kMachMessageIDExceptionRaiseState = 2402;
  static const mach_msg_id_t kMachMessageIDExceptionRaiseStateIdentity = 2403;

  static const exception_behavior_t kExceptionBehavior = 0;
};

// Traits for ExcServer<> and SimplifiedExcServer<> adapting them for use with
// the mach_exc subsystem.
struct MachExcTraits {
  using ExceptionCode = mach_exception_data_type_t;

  using RequestUnion = __RequestUnion__mach_exc_subsystem;
  using ReplyUnion = __ReplyUnion__mach_exc_subsystem;

  using ExceptionRaiseRequest = __Request__mach_exception_raise_t;
  using ExceptionRaiseStateRequest = __Request__mach_exception_raise_state_t;
  using ExceptionRaiseStateIdentityRequest =
      __Request__mach_exception_raise_state_identity_t;

  using ExceptionRaiseReply = __Reply__mach_exception_raise_t;
  using ExceptionRaiseStateReply = __Reply__mach_exception_raise_state_t;
  using ExceptionRaiseStateIdentityReply =
      __Reply__mach_exception_raise_state_identity_t;

  // The MIG-generated __MIG_check__Request__*() functions are not declared as
  // accepting const data, but they could have been because they in fact do not
  // modify the data.

  static kern_return_t MIGCheckRequestExceptionRaise(
      const ExceptionRaiseRequest* in_request) {
    return __MIG_check__Request__mach_exception_raise_t(
        const_cast<ExceptionRaiseRequest*>(in_request));
  }

  static kern_return_t MIGCheckRequestExceptionRaiseState(
      const ExceptionRaiseStateRequest* in_request,
      const ExceptionRaiseStateRequest** in_request_1) {
    return __MIG_check__Request__mach_exception_raise_state_t(
        const_cast<ExceptionRaiseStateRequest*>(in_request),
        const_cast<ExceptionRaiseStateRequest**>(in_request_1));
  }

  static kern_return_t MIGCheckRequestExceptionRaiseStateIdentity(
      const ExceptionRaiseStateIdentityRequest* in_request,
      const ExceptionRaiseStateIdentityRequest** in_request_1) {
    return __MIG_check__Request__mach_exception_raise_state_identity_t(
        const_cast<ExceptionRaiseStateIdentityRequest*>(in_request),
        const_cast<ExceptionRaiseStateIdentityRequest**>(in_request_1));
  }

  // There are no predefined constants for these.
  static const mach_msg_id_t kMachMessageIDExceptionRaise = 2405;
  static const mach_msg_id_t kMachMessageIDExceptionRaiseState = 2406;
  static const mach_msg_id_t kMachMessageIDExceptionRaiseStateIdentity = 2407;

  static const exception_behavior_t kExceptionBehavior = MACH_EXCEPTION_CODES;
};

//! \brief A server interface for the `exc` or `mach_exc` Mach subsystems.
template <typename Traits>
class ExcServer : public MachMessageServer::Interface {
 public:
  //! \brief An interface that the different request messages that are a part of
  //!     the `exc` or `mach_exc` Mach subsystems can be dispatched to.
  class Interface {
   public:
    //! \brief Handles exceptions raised by `exception_raise()` or
    //!     `mach_exception_raise()`.
    //!
    //! This behaves equivalently to a `catch_exception_raise()` function used
    //! with `exc_server()`, or a `catch_mach_exception_raise()` function used
    //! with `mach_exc_server()`.
    //!
    //! \param[in] trailer The trailer received with the request message.
    //! \param[out] destroy_request `true` if the request message is to be
    //!     destroyed even when this method returns success. See
    //!     MachMessageServer::Interface.
    virtual kern_return_t CatchExceptionRaise(
        exception_handler_t exception_port,
        thread_t thread,
        task_t task,
        exception_type_t exception,
        const typename Traits::ExceptionCode* code,
        mach_msg_type_number_t code_count,
        const mach_msg_trailer_t* trailer,
        bool* destroy_request) = 0;

    //! \brief Handles exceptions raised by `exception_raise_state()` or
    //!     `mach_exception_raise_state()`.
    //!
    //! This behaves equivalently to a `catch_exception_raise_state()` function
    //! used with `exc_server()`, or a `catch_mach_exception_raise_state()`
    //! function used with `mach_exc_server()`.
    //!
    //! There is no \a destroy_request parameter because, unlike
    //! CatchExceptionRaise() and CatchExceptionRaiseStateIdentity(), the
    //! request message is not complex (it does not carry the \a thread or \a
    //! task port rights) and thus there is nothing to destroy.
    //!
    //! \param[in] trailer The trailer received with the request message.
    virtual kern_return_t CatchExceptionRaiseState(
        exception_handler_t exception_port,
        exception_type_t exception,
        const typename Traits::ExceptionCode* code,
        mach_msg_type_number_t code_count,
        thread_state_flavor_t* flavor,
        ConstThreadState old_state,
        mach_msg_type_number_t old_state_count,
        thread_state_t new_state,
        mach_msg_type_number_t* new_state_count,
        const mach_msg_trailer_t* trailer) = 0;

    //! \brief Handles exceptions raised by `exception_raise_state_identity()`
    //!     or `mach_exception_raise_state_identity()`.
    //!
    //! This behaves equivalently to a `catch_exception_raise_state_identity()`
    //! function used with `exc_server()`, or a
    //! `catch_mach_exception_raise_state_identity()` function used with
    //! `mach_exc_server()`.
    //!
    //! \param[in] trailer The trailer received with the request message.
    //! \param[out] destroy_request `true` if the request message is to be
    //!     destroyed even when this method returns success. See
    //!     MachMessageServer::Interface.
    virtual kern_return_t CatchExceptionRaiseStateIdentity(
        exception_handler_t exception_port,
        thread_t thread,
        task_t task,
        exception_type_t exception,
        const typename Traits::ExceptionCode* code,
        mach_msg_type_number_t code_count,
        thread_state_flavor_t* flavor,
        ConstThreadState old_state,
        mach_msg_type_number_t old_state_count,
        thread_state_t new_state,
        mach_msg_type_number_t* new_state_count,
        const mach_msg_trailer_t* trailer,
        bool* destroy_request) = 0;

   protected:
    ~Interface() {}
  };

  //! \brief Constructs an object of this class.
  //!
  //! \param[in] interface The interface to dispatch requests to. Weak.
  explicit ExcServer(Interface* interface)
      : MachMessageServer::Interface(), interface_(interface) {}

  // MachMessageServer::Interface:

  bool MachMessageServerFunction(const mach_msg_header_t* in_header,
                                 mach_msg_header_t* out_header,
                                 bool* destroy_complex_request) override;

  std::set<mach_msg_id_t> MachMessageServerRequestIDs() override {
    constexpr mach_msg_id_t request_ids[] = {
        Traits::kMachMessageIDExceptionRaise,
        Traits::kMachMessageIDExceptionRaiseState,
        Traits::kMachMessageIDExceptionRaiseStateIdentity,
    };
    return std::set<mach_msg_id_t>(&request_ids[0],
                                   &request_ids[arraysize(request_ids)]);
  }

  mach_msg_size_t MachMessageServerRequestSize() override {
    return sizeof(typename Traits::RequestUnion);
  }

  mach_msg_size_t MachMessageServerReplySize() override {
    return sizeof(typename Traits::ReplyUnion);
  }

 private:
  Interface* interface_;  // weak

  DISALLOW_COPY_AND_ASSIGN(ExcServer);
};

template <typename Traits>
bool ExcServer<Traits>::MachMessageServerFunction(
    const mach_msg_header_t* in_header,
    mach_msg_header_t* out_header,
    bool* destroy_complex_request) {
  PrepareMIGReplyFromRequest(in_header, out_header);

  const mach_msg_trailer_t* in_trailer =
      MachMessageTrailerFromHeader(in_header);

  switch (in_header->msgh_id) {
    case Traits::kMachMessageIDExceptionRaise: {
      // exception_raise(), catch_exception_raise(), mach_exception_raise(),
      // catch_mach_exception_raise().
      using Request = typename Traits::ExceptionRaiseRequest;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);
      kern_return_t kr = Traits::MIGCheckRequestExceptionRaise(in_request);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = typename Traits::ExceptionRaiseReply;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->RetCode =
          interface_->CatchExceptionRaise(in_header->msgh_local_port,
                                          in_request->thread.name,
                                          in_request->task.name,
                                          in_request->exception,
                                          in_request->code,
                                          in_request->codeCnt,
                                          in_trailer,
                                          destroy_complex_request);
      if (out_reply->RetCode != KERN_SUCCESS) {
        return true;
      }

      out_header->msgh_size = sizeof(*out_reply);
      return true;
    }

    case Traits::kMachMessageIDExceptionRaiseState: {
      // exception_raise_state(), catch_exception_raise_state(),
      // mach_exception_raise_state(), catch_mach_exception_raise_state().
      using Request = typename Traits::ExceptionRaiseStateRequest;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);

      // in_request_1 is used for the portion of the request after the codes,
      // which in theory can be variable-length. The check function will set it.
      const Request* in_request_1;
      kern_return_t kr =
          Traits::MIGCheckRequestExceptionRaiseState(in_request, &in_request_1);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = typename Traits::ExceptionRaiseStateReply;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->flavor = in_request_1->flavor;
      out_reply->new_stateCnt = arraysize(out_reply->new_state);
      out_reply->RetCode =
          interface_->CatchExceptionRaiseState(in_header->msgh_local_port,
                                               in_request->exception,
                                               in_request->code,
                                               in_request->codeCnt,
                                               &out_reply->flavor,
                                               in_request_1->old_state,
                                               in_request_1->old_stateCnt,
                                               out_reply->new_state,
                                               &out_reply->new_stateCnt,
                                               in_trailer);
      if (out_reply->RetCode != KERN_SUCCESS) {
        return true;
      }

      out_header->msgh_size =
          sizeof(*out_reply) - sizeof(out_reply->new_state) +
          sizeof(out_reply->new_state[0]) * out_reply->new_stateCnt;
      return true;
    }

    case Traits::kMachMessageIDExceptionRaiseStateIdentity: {
      // exception_raise_state_identity(),
      // catch_exception_raise_state_identity(),
      // mach_exception_raise_state_identity(),
      // catch_mach_exception_raise_state_identity().
      using Request = typename Traits::ExceptionRaiseStateIdentityRequest;
      const Request* in_request = reinterpret_cast<const Request*>(in_header);

      // in_request_1 is used for the portion of the request after the codes,
      // which in theory can be variable-length. The check function will set it.
      const Request* in_request_1;
      kern_return_t kr = Traits::MIGCheckRequestExceptionRaiseStateIdentity(
          in_request, &in_request_1);
      if (kr != MACH_MSG_SUCCESS) {
        SetMIGReplyError(out_header, kr);
        return true;
      }

      using Reply = typename Traits::ExceptionRaiseStateIdentityReply;
      Reply* out_reply = reinterpret_cast<Reply*>(out_header);
      out_reply->flavor = in_request_1->flavor;
      out_reply->new_stateCnt = arraysize(out_reply->new_state);
      out_reply->RetCode = interface_->CatchExceptionRaiseStateIdentity(
          in_header->msgh_local_port,
          in_request->thread.name,
          in_request->task.name,
          in_request->exception,
          in_request->code,
          in_request->codeCnt,
          &out_reply->flavor,
          in_request_1->old_state,
          in_request_1->old_stateCnt,
          out_reply->new_state,
          &out_reply->new_stateCnt,
          in_trailer,
          destroy_complex_request);
      if (out_reply->RetCode != KERN_SUCCESS) {
        return true;
      }

      out_header->msgh_size =
          sizeof(*out_reply) - sizeof(out_reply->new_state) +
          sizeof(out_reply->new_state[0]) * out_reply->new_stateCnt;
      return true;
    }

    default: {
      SetMIGReplyError(out_header, MIG_BAD_ID);
      return false;
    }
  }
}

//! \brief A server interface for the `exc` or `mach_exc` Mach subsystems,
//!     simplified to have only a single interface method needing
//!     implementation.
template <typename Traits>
class SimplifiedExcServer final : public ExcServer<Traits>,
                                  public ExcServer<Traits>::Interface {
 public:
  //! \brief An interface that the different request messages that are a part of
  //!     the `exc` or `mach_exc` Mach subsystems can be dispatched to.
  class Interface {
   public:
    //! \brief Handles exceptions raised by `exception_raise()`,
    //!     `exception_raise_state()`, and `exception_raise_state_identity()`;
    //!     or `mach_exception_raise()`, `mach_exception_raise_state()`, and
    //!     `mach_exception_raise_state_identity()`.
    //!
    //! For convenience in implementation, these different “behaviors” of
    //! exception messages are all mapped to a single interface method. The
    //! exception’s original “behavior” is specified in the \a behavior
    //! parameter. Only parameters that were supplied in the request message
    //! are populated, other parameters are set to reasonable default values.
    //!
    //! The meanings of most parameters are identical to that of
    //! ExcServer<>::Interface::CatchExceptionRaiseStateIdentity().
    //!
    //! \param[in] behavior `EXCEPTION_DEFAULT`, `EXCEPTION_STATE`, or
    //!     `EXCEPTION_STATE_IDENTITY`, identifying which exception request
    //!     message was processed and thus which other parameters are valid.
    //!     When used with the `mach_exc` subsystem, `MACH_EXCEPTION_CODES` will
    //!     be ORed in to this parameter.
    virtual kern_return_t CatchException(
        exception_behavior_t behavior,
        exception_handler_t exception_port,
        thread_t thread,
        task_t task,
        exception_type_t exception,
        const typename Traits::ExceptionCode* code,
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
  explicit SimplifiedExcServer(Interface* interface)
      : ExcServer<Traits>(this),
        ExcServer<Traits>::Interface(),
        interface_(interface) {}

  // ExcServer::Interface:

  kern_return_t CatchExceptionRaise(exception_handler_t exception_port,
                                    thread_t thread,
                                    task_t task,
                                    exception_type_t exception,
                                    const typename Traits::ExceptionCode* code,
                                    mach_msg_type_number_t code_count,
                                    const mach_msg_trailer_t* trailer,
                                    bool* destroy_request) override {
    thread_state_flavor_t flavor = THREAD_STATE_NONE;
    mach_msg_type_number_t new_state_count = 0;
    return interface_->CatchException(
        Traits::kExceptionBehavior | EXCEPTION_DEFAULT,
        exception_port,
        thread,
        task,
        exception,
        code_count ? code : nullptr,
        code_count,
        &flavor,
        nullptr,
        0,
        nullptr,
        &new_state_count,
        trailer,
        destroy_request);
  }

  kern_return_t CatchExceptionRaiseState(
      exception_handler_t exception_port,
      exception_type_t exception,
      const typename Traits::ExceptionCode* code,
      mach_msg_type_number_t code_count,
      thread_state_flavor_t* flavor,
      ConstThreadState old_state,
      mach_msg_type_number_t old_state_count,
      thread_state_t new_state,
      mach_msg_type_number_t* new_state_count,
      const mach_msg_trailer_t* trailer) override {
    bool destroy_complex_request = false;
    return interface_->CatchException(
        Traits::kExceptionBehavior | EXCEPTION_STATE,
        exception_port,
        THREAD_NULL,
        TASK_NULL,
        exception,
        code_count ? code : nullptr,
        code_count,
        flavor,
        old_state_count ? old_state : nullptr,
        old_state_count,
        new_state_count ? new_state : nullptr,
        new_state_count,
        trailer,
        &destroy_complex_request);
  }

  kern_return_t CatchExceptionRaiseStateIdentity(
      exception_handler_t exception_port,
      thread_t thread,
      task_t task,
      exception_type_t exception,
      const typename Traits::ExceptionCode* code,
      mach_msg_type_number_t code_count,
      thread_state_flavor_t* flavor,
      ConstThreadState old_state,
      mach_msg_type_number_t old_state_count,
      thread_state_t new_state,
      mach_msg_type_number_t* new_state_count,
      const mach_msg_trailer_t* trailer,
      bool* destroy_request) override {
    return interface_->CatchException(
        Traits::kExceptionBehavior | EXCEPTION_STATE_IDENTITY,
        exception_port,
        thread,
        task,
        exception,
        code_count ? code : nullptr,
        code_count,
        flavor,
        old_state_count ? old_state : nullptr,
        old_state_count,
        new_state_count ? new_state : nullptr,
        new_state_count,
        trailer,
        destroy_request);
  }

 private:
  Interface* interface_;  // weak

  DISALLOW_COPY_AND_ASSIGN(SimplifiedExcServer);
};

}  // namespace

namespace internal {

class UniversalMachExcServerImpl final
    : public CompositeMachMessageServer,
      public SimplifiedExcServer<ExcTraits>::Interface,
      public SimplifiedExcServer<MachExcTraits>::Interface {
 public:
  explicit UniversalMachExcServerImpl(
      UniversalMachExcServer::Interface* interface)
      : CompositeMachMessageServer(),
        SimplifiedExcServer<ExcTraits>::Interface(),
        SimplifiedExcServer<MachExcTraits>::Interface(),
        exc_server_(this),
        mach_exc_server_(this),
        interface_(interface) {
    AddHandler(&exc_server_);
    AddHandler(&mach_exc_server_);
  }

  ~UniversalMachExcServerImpl() {}

  // SimplifiedExcServer<ExcTraits>::Interface:
  kern_return_t CatchException(exception_behavior_t behavior,
                               exception_handler_t exception_port,
                               thread_t thread,
                               task_t task,
                               exception_type_t exception,
                               const exception_data_type_t* code,
                               mach_msg_type_number_t code_count,
                               thread_state_flavor_t* flavor,
                               ConstThreadState old_state,
                               mach_msg_type_number_t old_state_count,
                               thread_state_t new_state,
                               mach_msg_type_number_t* new_state_count,
                               const mach_msg_trailer_t* trailer,
                               bool* destroy_complex_request) {
    std::vector<mach_exception_data_type_t> mach_codes;
    mach_codes.reserve(code_count);
    for (size_t index = 0; index < code_count; ++index) {
      mach_codes.push_back(code[index]);
    }

    return interface_->CatchMachException(behavior,
                                          exception_port,
                                          thread,
                                          task,
                                          exception,
                                          code_count ? &mach_codes[0] : nullptr,
                                          code_count,
                                          flavor,
                                          old_state_count ? old_state : nullptr,
                                          old_state_count,
                                          new_state_count ? new_state : nullptr,
                                          new_state_count,
                                          trailer,
                                          destroy_complex_request);
  }

  // SimplifiedExcServer<MachExcTraits>::Interface:
  kern_return_t CatchException(exception_behavior_t behavior,
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
                               bool* destroy_complex_request) {
    return interface_->CatchMachException(behavior,
                                          exception_port,
                                          thread,
                                          task,
                                          exception,
                                          code_count ? code : nullptr,
                                          code_count,
                                          flavor,
                                          old_state_count ? old_state : nullptr,
                                          old_state_count,
                                          new_state_count ? new_state : nullptr,
                                          new_state_count,
                                          trailer,
                                          destroy_complex_request);
  }

 private:
  SimplifiedExcServer<ExcTraits> exc_server_;
  SimplifiedExcServer<MachExcTraits> mach_exc_server_;
  UniversalMachExcServer::Interface* interface_;  // weak

  DISALLOW_COPY_AND_ASSIGN(UniversalMachExcServerImpl);
};

}  // namespace internal

UniversalMachExcServer::UniversalMachExcServer(
    UniversalMachExcServer::Interface* interface)
    : MachMessageServer::Interface(),
      impl_(new internal::UniversalMachExcServerImpl(interface)) {
}

UniversalMachExcServer::~UniversalMachExcServer() {
}

bool UniversalMachExcServer::MachMessageServerFunction(
    const mach_msg_header_t* in_header,
    mach_msg_header_t* out_header,
    bool* destroy_complex_request) {
  return impl_->MachMessageServerFunction(
      in_header, out_header, destroy_complex_request);
}

std::set<mach_msg_id_t> UniversalMachExcServer::MachMessageServerRequestIDs() {
  return impl_->MachMessageServerRequestIDs();
}

mach_msg_size_t UniversalMachExcServer::MachMessageServerRequestSize() {
  return impl_->MachMessageServerRequestSize();
}

mach_msg_size_t UniversalMachExcServer::MachMessageServerReplySize() {
  return impl_->MachMessageServerReplySize();
}

kern_return_t ExcServerSuccessfulReturnValue(exception_type_t exception,
                                             exception_behavior_t behavior,
                                             bool set_thread_state) {
  if (exception == EXC_CRASH
#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_11
      && MacOSXMinorVersion() >= 11
#endif
     ) {
    return KERN_SUCCESS;
  }

  if (!set_thread_state && ExceptionBehaviorHasState(behavior)) {
    return MACH_RCV_PORT_DIED;
  }

  return KERN_SUCCESS;
}

void ExcServerCopyState(exception_behavior_t behavior,
                        ConstThreadState old_state,
                        mach_msg_type_number_t old_state_count,
                        thread_state_t new_state,
                        mach_msg_type_number_t* new_state_count) {
  if (ExceptionBehaviorHasState(behavior)) {
    *new_state_count = std::min(old_state_count, *new_state_count);
    memcpy(new_state, old_state, *new_state_count * sizeof(old_state[0]));
  }
}

}  // namespace crashpad
