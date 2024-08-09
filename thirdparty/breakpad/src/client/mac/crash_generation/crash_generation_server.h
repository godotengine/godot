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

#ifndef GOOGLE_BREAKPAD_CLIENT_MAC_CRASH_GENERATION_CRASH_GENERATION_SERVER_H_
#define GOOGLE_BREAKPAD_CLIENT_MAC_CRASH_GENERATION_CRASH_GENERATION_SERVER_H_

#include <stdint.h>

#include <string>

#include "common/mac/MachIPC.h"

namespace google_breakpad {

class ClientInfo;

// Messages the server can read via its mach port
enum {
  kDumpRequestMessage     = 1,
  kAcknowledgementMessage = 2,
  kQuitMessage            = 3
};

// Exception details sent by the client when requesting a dump.
struct ExceptionInfo {
  int32_t exception_type;
  int32_t exception_code;
  int32_t exception_subcode;
};

class CrashGenerationServer {
 public:
  // WARNING: callbacks may be invoked on a different thread
  // than that which creates the CrashGenerationServer.  They must
  // be thread safe.
  typedef void (*OnClientDumpRequestCallback)(void* context,
                                              const ClientInfo& client_info,
                                              const std::string& file_path);

  typedef void (*OnClientExitingCallback)(void* context,
                                          const ClientInfo& client_info);
  // If a FilterCallback returns false, the dump will not be written.
  typedef bool (*FilterCallback)(void* context);

  // Create an instance with the given parameters.
  //
  // mach_port_name: Named server port to listen on.
  // filter: Callback for a client to cancel writing a dump.
  // filter_context: Context for the filter callback.
  // dump_callback: Callback for a client crash dump request.
  // dump_context: Context for client crash dump request callback.
  // exit_callback: Callback for client process exit.
  // exit_context: Context for client exit callback.
  // generate_dumps: Whether to automatically generate dumps.
  //     Client code of this class might want to generate dumps explicitly
  //     in the crash dump request callback. In that case, false can be
  //     passed for this parameter.
  // dump_path: Path for generating dumps; required only if true is
  //     passed for generateDumps parameter; NULL can be passed otherwise.
  CrashGenerationServer(const char* mach_port_name,
                        FilterCallback filter,
                        void* filter_context,
                        OnClientDumpRequestCallback dump_callback,
                        void* dump_context,
                        OnClientExitingCallback exit_callback,
                        void* exit_context,
                        bool generate_dumps,
                        const std::string& dump_path);

  ~CrashGenerationServer();

  // Perform initialization steps needed to start listening to clients.
  //
  // Return true if initialization is successful; false otherwise.
  bool Start();

  // Stop the server.
  bool Stop();

 private:
  // Return a unique filename at which a minidump can be written.
  bool MakeMinidumpFilename(std::string& outFilename);

  // Loop reading client messages and responding to them until
  // a quit message is received.
  static void* WaitForMessages(void* server);

  // Wait for a single client message and respond to it. Returns false
  // if a quit message was received or if an error occurred.
  bool WaitForOneMessage();

  FilterCallback filter_;
  void* filter_context_;

  OnClientDumpRequestCallback dump_callback_;
  void* dump_context_;

  OnClientExitingCallback exit_callback_;
  void* exit_context_;

  bool generate_dumps_;

  std::string dump_dir_;

  bool started_;

  // The mach port that receives requests to dump from child processes.
  ReceivePort receive_port_;

  // The name of the mach port. Stored so the Stop method can message
  // the background thread to shut it down.
  std::string mach_port_name_;

  // The thread that waits on the receive port.
  pthread_t server_thread_;

  // Disable copy constructor and operator=.
  CrashGenerationServer(const CrashGenerationServer&);
  CrashGenerationServer& operator=(const CrashGenerationServer&);
};

}  // namespace google_breakpad

#endif  // GOOGLE_BREAKPAD_CLIENT_MAC_CRASH_GENERATION_CRASH_GENERATION_SERVER_H_
