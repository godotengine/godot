// Copyright (c) 2008, Google Inc.
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

#ifndef CLIENT_WINDOWS_CRASH_GENERATION_CRASH_GENERATION_SERVER_H__
#define CLIENT_WINDOWS_CRASH_GENERATION_CRASH_GENERATION_SERVER_H__

#include <list>
#include <string>
#include "client/windows/common/ipc_protocol.h"
#include "client/windows/crash_generation/minidump_generator.h"
#include "common/scoped_ptr.h"

namespace google_breakpad {
class ClientInfo;

// Abstraction for server side implementation of out-of-process crash
// generation protocol for Windows platform only. It generates Windows
// minidump files for client processes that request dump generation. When
// the server is requested to start listening for clients (by calling the
// Start method), it creates a named pipe and waits for the clients to
// register. In response, it hands them event handles that the client can
// signal to request dump generation. When the clients request dump
// generation in this way, the server generates Windows minidump files.
class CrashGenerationServer {
 public:
  typedef void (*OnClientConnectedCallback)(void* context,
                                            const ClientInfo* client_info);

  typedef void (*OnClientDumpRequestCallback)(void* context,
                                              const ClientInfo* client_info,
                                              const std::wstring* file_path);

  typedef void (*OnClientExitedCallback)(void* context,
                                         const ClientInfo* client_info);

  typedef void (*OnClientUploadRequestCallback)(void* context,
                                                const DWORD crash_id);

  // Creates an instance with the given parameters.
  //
  // Parameter pipe_name: Name of the Windows named pipe
  // Parameter pipe_sec_attrs Security attributes to set on the pipe. Pass
  //     NULL to use default security on the pipe. By default, the pipe created
  //     allows Local System, Administrators and the Creator full control and
  //     the Everyone group read access on the pipe.
  // Parameter connect_callback: Callback for a new client connection.
  // Parameter connect_context: Context for client connection callback.
  // Parameter crash_callback: Callback for a client crash dump request.
  // Parameter crash_context: Context for client crash dump request callback.
  // Parameter exit_callback: Callback for client process exit.
  // Parameter exit_context: Context for client exit callback.
  // Parameter generate_dumps: Whether to automatically generate dumps.
  // Client code of this class might want to generate dumps explicitly in the
  // crash dump request callback. In that case, false can be passed for this
  // parameter.
  // Parameter dump_path: Path for generating dumps; required only if true is
  // passed for generateDumps parameter; NULL can be passed otherwise.
  CrashGenerationServer(const std::wstring& pipe_name,
                        SECURITY_ATTRIBUTES* pipe_sec_attrs,
                        OnClientConnectedCallback connect_callback,
                        void* connect_context,
                        OnClientDumpRequestCallback dump_callback,
                        void* dump_context,
                        OnClientExitedCallback exit_callback,
                        void* exit_context,
                        OnClientUploadRequestCallback upload_request_callback,
                        void* upload_context,
                        bool generate_dumps,
                        const std::wstring* dump_path);

  ~CrashGenerationServer();

  // Performs initialization steps needed to start listening to clients. Upon
  // successful return clients may connect to this server's pipe.
  //
  // Returns true if initialization is successful; false otherwise.
  bool Start();

  void pre_fetch_custom_info(bool do_pre_fetch) {
    pre_fetch_custom_info_ = do_pre_fetch;
  }

 private:
  // Various states the client can be in during the handshake with
  // the server.
  enum IPCServerState {
    // Server starts in this state.
    IPC_SERVER_STATE_UNINITIALIZED,

    // Server is in error state and it cannot serve any clients.
    IPC_SERVER_STATE_ERROR,

    // Server starts in this state.
    IPC_SERVER_STATE_INITIAL,

    // Server has issued an async connect to the pipe and it is waiting
    // for the connection to be established.
    IPC_SERVER_STATE_CONNECTING,

    // Server is connected successfully.
    IPC_SERVER_STATE_CONNECTED,

    // Server has issued an async read from the pipe and it is waiting for
    // the read to finish.
    IPC_SERVER_STATE_READING,

    // Server is done reading from the pipe.
    IPC_SERVER_STATE_READ_DONE,

    // Server has issued an async write to the pipe and it is waiting for
    // the write to finish.
    IPC_SERVER_STATE_WRITING,

    // Server is done writing to the pipe.
    IPC_SERVER_STATE_WRITE_DONE,

    // Server has issued an async read from the pipe for an ack and it
    // is waiting for the read to finish.
    IPC_SERVER_STATE_READING_ACK,

    // Server is done writing to the pipe and it is now ready to disconnect
    // and reconnect.
    IPC_SERVER_STATE_DISCONNECTING
  };

  //
  // Helper methods to handle various server IPC states.
  //
  void HandleErrorState();
  void HandleInitialState();
  void HandleConnectingState();
  void HandleConnectedState();
  void HandleReadingState();
  void HandleReadDoneState();
  void HandleWritingState();
  void HandleWriteDoneState();
  void HandleReadingAckState();
  void HandleDisconnectingState();

  // Prepares reply for a client from the given parameters.
  bool PrepareReply(const ClientInfo& client_info,
                    ProtocolMessage* reply) const;

  // Duplicates various handles in the ClientInfo object for the client
  // process and stores them in the given ProtocolMessage instance. If
  // creating any handle fails, ProtocolMessage will contain the handles
  // already created successfully, which should be closed by the caller.
  bool CreateClientHandles(const ClientInfo& client_info,
                           ProtocolMessage* reply) const;

  // Response to the given client. Return true if all steps of
  // responding to the client succeed, false otherwise.
  bool RespondToClient(ClientInfo* client_info);

  // Handles a connection request from the client.
  void HandleConnectionRequest();

  // Handles a dump request from the client.
  void HandleDumpRequest(const ClientInfo& client_info);

  // Callback for pipe connected event.
  static void CALLBACK OnPipeConnected(void* context, BOOLEAN timer_or_wait);

  // Callback for a dump request.
  static void CALLBACK OnDumpRequest(void* context, BOOLEAN timer_or_wait);

  // Callback for client process exit event.
  static void CALLBACK OnClientEnd(void* context, BOOLEAN timer_or_wait);

  // Handles client process exit.
  void HandleClientProcessExit(ClientInfo* client_info);

  // Adds the given client to the list of registered clients.
  bool AddClient(ClientInfo* client_info);

  // Generates dump for the given client.
  bool GenerateDump(const ClientInfo& client, std::wstring* dump_path);

  // Puts the server in a permanent error state and sets a signal such that
  // the state will be immediately entered after the current state transition
  // is complete.
  void EnterErrorState();

  // Puts the server in the specified state and sets a signal such that the
  // state is immediately entered after the current state transition is
  // complete.
  void EnterStateImmediately(IPCServerState state);

  // Puts the server in the specified state. No signal will be set, so the state
  // transition will only occur when signaled manually or by completion of an
  // asynchronous IO operation.
  void EnterStateWhenSignaled(IPCServerState state);

  // Sync object for thread-safe access to the shared list of clients.
  CRITICAL_SECTION sync_;

  // List of clients.
  std::list<ClientInfo*> clients_;

  // Pipe name.
  std::wstring pipe_name_;

  // Pipe security attributes
  SECURITY_ATTRIBUTES* pipe_sec_attrs_;

  // Handle to the pipe used for handshake with clients.
  HANDLE pipe_;

  // Pipe wait handle.
  HANDLE pipe_wait_handle_;

  // Handle to server-alive mutex.
  HANDLE server_alive_handle_;

  // Callback for a successful client connection.
  OnClientConnectedCallback connect_callback_;

  // Context for client connected callback.
  void* connect_context_;

  // Callback for a client dump request.
  OnClientDumpRequestCallback dump_callback_;

  // Context for client dump request callback.
  void* dump_context_;

  // Callback for client process exit.
  OnClientExitedCallback exit_callback_;

  // Context for client process exit callback.
  void* exit_context_;

  // Callback for upload request.
  OnClientUploadRequestCallback upload_request_callback_;

  // Context for upload request callback.
  void* upload_context_;

  // Whether to generate dumps.
  bool generate_dumps_;

  // Wether to populate custom information up-front.
  bool pre_fetch_custom_info_;

  // The dump path for the server.
  const std::wstring dump_path_;

  // State of the server in performing the IPC with the client.
  // Note that since we restrict the pipe to one instance, we
  // only need to keep one state of the server. Otherwise, server
  // would have one state per client it is talking to.
  IPCServerState server_state_;

  // Whether the server is shutting down.
  bool shutting_down_;

  // Overlapped instance for async I/O on the pipe.
  OVERLAPPED overlapped_;

  // Message object used in IPC with the client.
  ProtocolMessage msg_;

  // Client Info for the client that's connecting to the server.
  ClientInfo* client_info_;

  // Disable copy ctor and operator=.
  CrashGenerationServer(const CrashGenerationServer& crash_server);
  CrashGenerationServer& operator=(const CrashGenerationServer& crash_server);
};

}  // namespace google_breakpad

#endif  // CLIENT_WINDOWS_CRASH_GENERATION_CRASH_GENERATION_SERVER_H__
