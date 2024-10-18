// Copyright 2010 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "client/mac/crash_generation/crash_generation_client.h"

#include "client/mac/crash_generation/crash_generation_server.h"
#include "common/mac/MachIPC.h"

namespace google_breakpad {

bool CrashGenerationClient::RequestDumpForException(
    int exception_type,
    int exception_code,
    int exception_subcode,
    mach_port_t crashing_thread) {
  // The server will send a message to this port indicating that it
  // has finished its work.
  ReceivePort acknowledge_port;

  MachSendMessage message(kDumpRequestMessage);
  message.AddDescriptor(mach_task_self());            // this task
  message.AddDescriptor(crashing_thread);             // crashing thread
  message.AddDescriptor(mach_thread_self());          // handler thread
  message.AddDescriptor(acknowledge_port.GetPort());  // message receive port

  ExceptionInfo info;
  info.exception_type = exception_type;
  info.exception_code = exception_code;
  info.exception_subcode = exception_subcode;
  message.SetData(&info, sizeof(info));

  const mach_msg_timeout_t kSendTimeoutMs = 2 * 1000;
  kern_return_t result = sender_.SendMessage(message, kSendTimeoutMs);
  if (result != KERN_SUCCESS)
    return false;

  // Give the server slightly longer to reply since it has to
  // inspect this task and write the minidump.
  const mach_msg_timeout_t kReceiveTimeoutMs = 5 * 1000;
  MachReceiveMessage acknowledge_message;
  result = acknowledge_port.WaitForMessage(&acknowledge_message,
					   kReceiveTimeoutMs);
  return result == KERN_SUCCESS;
}

}  // namespace google_breakpad
