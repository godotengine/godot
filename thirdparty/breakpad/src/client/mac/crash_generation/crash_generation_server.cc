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

#include "client/mac/crash_generation/crash_generation_server.h"

#include <pthread.h>

#include "client/mac/crash_generation/client_info.h"
#include "client/mac/handler/minidump_generator.h"
#include "common/mac/scoped_task_suspend-inl.h"

namespace google_breakpad {

CrashGenerationServer::CrashGenerationServer(
    const char* mach_port_name,
    FilterCallback filter,
    void* filter_context,
    OnClientDumpRequestCallback dump_callback,
    void* dump_context,
    OnClientExitingCallback exit_callback,
    void* exit_context,
    bool generate_dumps,
    const std::string& dump_path)
    : filter_(filter),
      filter_context_(filter_context),
      dump_callback_(dump_callback),
      dump_context_(dump_context),
      exit_callback_(exit_callback),
      exit_context_(exit_context),
      generate_dumps_(generate_dumps),
      dump_dir_(dump_path.empty() ? "/tmp" : dump_path),
      started_(false),
      receive_port_(mach_port_name),
      mach_port_name_(mach_port_name) {
}

CrashGenerationServer::~CrashGenerationServer() {
  if (started_)
    Stop();
}

bool CrashGenerationServer::Start() {
  int thread_create_result = pthread_create(&server_thread_, NULL,
                                            &WaitForMessages, this);
  started_ = thread_create_result == 0;
  return started_;
}

bool CrashGenerationServer::Stop() {
  if (!started_)
    return false;

  // Send a quit message to the background thread, and then join it.
  MachPortSender sender(mach_port_name_.c_str());
  MachSendMessage quit_message(kQuitMessage);
  const mach_msg_timeout_t kSendTimeoutMs = 2 * 1000;
  kern_return_t result = sender.SendMessage(quit_message, kSendTimeoutMs);
  if (result == KERN_SUCCESS) {
    int thread_join_result = pthread_join(server_thread_, NULL);
    started_ = thread_join_result != 0;
  }

  return !started_;
}

// static
void* CrashGenerationServer::WaitForMessages(void* server) {
  CrashGenerationServer* self =
      reinterpret_cast<CrashGenerationServer*>(server);
  while (self->WaitForOneMessage()) {}
  return NULL;
}

bool CrashGenerationServer::WaitForOneMessage() {
  MachReceiveMessage message;
  kern_return_t result = receive_port_.WaitForMessage(&message,
                                                      MACH_MSG_TIMEOUT_NONE);
  if (result == KERN_SUCCESS) {
    switch (message.GetMessageID()) {
      case kDumpRequestMessage: {
        ExceptionInfo& info = (ExceptionInfo&)*message.GetData();
      
        mach_port_t remote_task = message.GetTranslatedPort(0);
        mach_port_t crashing_thread = message.GetTranslatedPort(1);
        mach_port_t handler_thread = message.GetTranslatedPort(2);
        mach_port_t ack_port = message.GetTranslatedPort(3);
        pid_t remote_pid = -1;
        pid_for_task(remote_task, &remote_pid);
        ClientInfo client(remote_pid);

        bool result;
        std::string dump_path;
        if (generate_dumps_ && (!filter_ || filter_(filter_context_))) {
          ScopedTaskSuspend suspend(remote_task);

          MinidumpGenerator generator(remote_task, handler_thread);
          dump_path = generator.UniqueNameInDirectory(dump_dir_, NULL);
        
          if (info.exception_type && info.exception_code) {
            generator.SetExceptionInformation(info.exception_type,
                                              info.exception_code,
                                              info.exception_subcode,
                                              crashing_thread);
          }
          result = generator.Write(dump_path.c_str());
        } else {
          result = true;
        }

        if (result && dump_callback_) {
          dump_callback_(dump_context_, client, dump_path);
        }

        // TODO(ted): support a way for the client to send additional data,
        // perhaps with a callback so users of the server can read the data
        // themselves?
      
        if (ack_port != MACH_PORT_DEAD && ack_port != MACH_PORT_NULL) {
          MachPortSender sender(ack_port);
          MachSendMessage ack_message(kAcknowledgementMessage);
          const mach_msg_timeout_t kSendTimeoutMs = 2 * 1000;

          sender.SendMessage(ack_message, kSendTimeoutMs);
        }

        if (exit_callback_) {
          exit_callback_(exit_context_, client);
        }
        break;
      }
      case kQuitMessage:
        return false;
    }
  } else {  // result != KERN_SUCCESS
    return false;
  }
  return true;
}

}  // namespace google_breakpad
