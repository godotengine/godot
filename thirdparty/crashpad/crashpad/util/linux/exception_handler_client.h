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

#ifndef CRASHPAD_UTIL_LINUX_EXCEPTION_HANDLER_CLIENT_H_
#define CRASHPAD_UTIL_LINUX_EXCEPTION_HANDLER_CLIENT_H_

#include <sys/types.h>

#include "base/macros.h"
#include "util/linux/exception_handler_protocol.h"

namespace crashpad {

//! A client for an ExceptionHandlerServer
class ExceptionHandlerClient {
 public:
  //! \brief Constructs this object.
  //!
  //! \param[in] sock A socket connected to an ExceptionHandlerServer.
  explicit ExceptionHandlerClient(int sock);

  ~ExceptionHandlerClient();

  //! \brief Request a crash dump from the ExceptionHandlerServer.
  //!
  //! This method blocks until the crash dump is complete.
  //!
  //! \param[in] info Information about this client.
  //! \return 0 on success or an error code on failure.
  int RequestCrashDump(const ClientInformation& info);

  //! \brief Uses `prctl(PR_SET_PTRACER, ...)` to set the process with
  //!     process ID \a pid as the ptracer for this process.
  //!
  //! \param[in] pid The process ID of the process to be set as this process'
  //!     ptracer.
  //! \return 0 on success or an error code on failure.
  int SetPtracer(pid_t pid);

  //! \brief Enables or disables SetPtracer().
  //! \param[in] can_set_ptracer Whether SetPtracer should be enabled.
  void SetCanSetPtracer(bool can_set_ptracer);

 private:
  int SendCrashDumpRequest(const ClientInformation& info);
  int WaitForCrashDumpComplete();

  int server_sock_;
  pid_t ptracer_;
  bool can_set_ptracer_;

  DISALLOW_COPY_AND_ASSIGN(ExceptionHandlerClient);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_EXCEPTION_HANDLER_CLIENT_H_
