// Copyright (c) 2010 Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

#ifndef CLIENT_LINUX_CRASH_GENERATION_CRASH_GENERATION_SERVER_H_
#define CLIENT_LINUX_CRASH_GENERATION_CRASH_GENERATION_SERVER_H_

#include <pthread.h>

#include <string>

#include "common/using_std_string.h"

namespace google_breakpad {

class ClientInfo;

class CrashGenerationServer {
public:
  // WARNING: callbacks may be invoked on a different thread
  // than that which creates the CrashGenerationServer.  They must
  // be thread safe.
  typedef void (*OnClientDumpRequestCallback)(void* context,
                                              const ClientInfo* client_info,
                                              const string* file_path);

  typedef void (*OnClientExitingCallback)(void* context,
                                          const ClientInfo* client_info);

  // Create an instance with the given parameters.
  //
  // Parameter listen_fd: The server fd created by CreateReportChannel().
  // Parameter dump_callback: Callback for a client crash dump request.
  // Parameter dump_context: Context for client crash dump request callback.
  // Parameter exit_callback: Callback for client process exit.
  // Parameter exit_context: Context for client exit callback.
  // Parameter generate_dumps: Whether to automatically generate dumps.
  //     Client code of this class might want to generate dumps explicitly
  //     in the crash dump request callback. In that case, false can be
  //     passed for this parameter.
  // Parameter dump_path: Path for generating dumps; required only if true is
  //     passed for generateDumps parameter; NULL can be passed otherwise.
  CrashGenerationServer(const int listen_fd,
                        OnClientDumpRequestCallback dump_callback,
                        void* dump_context,
                        OnClientExitingCallback exit_callback,
                        void* exit_context,
                        bool generate_dumps,
                        const string* dump_path);

  ~CrashGenerationServer();

  // Perform initialization steps needed to start listening to clients.
  //
  // Return true if initialization is successful; false otherwise.
  bool Start();

  // Stop the server.
  void Stop();

  // Create a "channel" that can be used by clients to report crashes
  // to a CrashGenerationServer.  |*server_fd| should be passed to
  // this class's constructor, and |*client_fd| should be passed to
  // the ExceptionHandler constructor in the client process.
  static bool CreateReportChannel(int* server_fd, int* client_fd);

private:
  // Run the server's event loop
  void Run();

  // Invoked when an child process (client) event occurs
  // Returning true => "keep running", false => "exit loop"
  bool ClientEvent(short revents);

  // Invoked when the controlling thread (main) event occurs
  // Returning true => "keep running", false => "exit loop"
  bool ControlEvent(short revents);

  // Return a unique filename at which a minidump can be written
  bool MakeMinidumpFilename(string& outFilename);

  // Trampoline to |Run()|
  static void* ThreadMain(void* arg);

  int server_fd_;

  OnClientDumpRequestCallback dump_callback_;
  void* dump_context_;

  OnClientExitingCallback exit_callback_;
  void* exit_context_;

  bool generate_dumps_;

  string dump_dir_;

  bool started_;

  pthread_t thread_;
  int control_pipe_in_;
  int control_pipe_out_;

  // disable these
  CrashGenerationServer(const CrashGenerationServer&);
  CrashGenerationServer& operator=(const CrashGenerationServer&);
};

} // namespace google_breakpad

#endif // CLIENT_LINUX_CRASH_GENERATION_CRASH_GENERATION_SERVER_H_
