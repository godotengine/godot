// Copyright 2008 Google LLC
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

#include "client/windows/crash_generation/crash_generation_server.h"
#include <windows.h>
#include <cassert>
#include <list>
#include "client/windows/common/auto_critical_section.h"
#include "common/scoped_ptr.h"

#include "client/windows/crash_generation/client_info.h"

namespace google_breakpad {

// Output buffer size.
static const size_t kOutBufferSize = 64;

// Input buffer size.
static const size_t kInBufferSize = 64;

// Access flags for the client on the dump request event.
static const DWORD kDumpRequestEventAccess = EVENT_MODIFY_STATE;

// Access flags for the client on the dump generated event.
static const DWORD kDumpGeneratedEventAccess = EVENT_MODIFY_STATE |
                                               SYNCHRONIZE;

// Access flags for the client on the mutex.
static const DWORD kMutexAccess = SYNCHRONIZE;

// Attribute flags for the pipe.
static const DWORD kPipeAttr = FILE_FLAG_FIRST_PIPE_INSTANCE |
                               PIPE_ACCESS_DUPLEX |
                               FILE_FLAG_OVERLAPPED;

// Mode for the pipe.
static const DWORD kPipeMode = PIPE_TYPE_MESSAGE |
                               PIPE_READMODE_MESSAGE |
                               PIPE_WAIT;

// For pipe I/O, execute the callback in the wait thread itself,
// since the callback does very little work. The callback executes
// the code for one of the states of the server state machine and
// the code for all of the states perform async I/O and hence
// finish very quickly.
static const ULONG kPipeIOThreadFlags = WT_EXECUTEINWAITTHREAD;

// Dump request threads will, most likely, generate dumps. That may
// take some time to finish, so specify WT_EXECUTELONGFUNCTION flag.
static const ULONG kDumpRequestThreadFlags = WT_EXECUTEINWAITTHREAD |
                                             WT_EXECUTELONGFUNCTION;

static bool IsClientRequestValid(const ProtocolMessage& msg) {
  return msg.tag == MESSAGE_TAG_UPLOAD_REQUEST ||
         (msg.tag == MESSAGE_TAG_REGISTRATION_REQUEST &&
          msg.id != 0 &&
          msg.thread_id != NULL &&
          msg.exception_pointers != NULL &&
          msg.assert_info != NULL);
}

#ifndef NDEBUG
static bool CheckForIOIncomplete(bool success) {
  // We should never get an I/O incomplete since we should not execute this
  // unless the operation has finished and the overlapped event is signaled. If
  // we do get INCOMPLETE, we have a bug in our code.
  return success ? false : (GetLastError() == ERROR_IO_INCOMPLETE);
}
#endif

CrashGenerationServer::CrashGenerationServer(
    const std::wstring& pipe_name,
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
    const std::wstring* dump_path)
    : pipe_name_(pipe_name),
      pipe_sec_attrs_(pipe_sec_attrs),
      pipe_(NULL),
      pipe_wait_handle_(NULL),
      server_alive_handle_(NULL),
      connect_callback_(connect_callback),
      connect_context_(connect_context),
      dump_callback_(dump_callback),
      dump_context_(dump_context),
      exit_callback_(exit_callback),
      exit_context_(exit_context),
      upload_request_callback_(upload_request_callback),
      upload_context_(upload_context),
      generate_dumps_(generate_dumps),
      pre_fetch_custom_info_(true),
      dump_path_(dump_path ? *dump_path : L""),
      server_state_(IPC_SERVER_STATE_UNINITIALIZED),
      shutting_down_(false),
      overlapped_(),
      client_info_(NULL) {
  InitializeCriticalSection(&sync_);
}

// This should never be called from the OnPipeConnected callback.
// Otherwise the UnregisterWaitEx call below will cause a deadlock.
CrashGenerationServer::~CrashGenerationServer() {
  // New scope to release the lock automatically.
  {
    // Make sure no clients are added or removed beyond this point.
    // Before adding or removing any clients, the critical section
    // must be entered and the shutting_down_ flag checked. The
    // critical section is then exited only after the clients_ list
    // modifications are done and the list is in a consistent state.
    AutoCriticalSection lock(&sync_);

    // Indicate to existing threads that server is shutting down.
    shutting_down_ = true;
  }
  // No one will modify the clients_ list beyond this point -
  // not even from another thread.

  // Even if there are no current worker threads running, it is possible that
  // an I/O request is pending on the pipe right now but not yet done.
  // In fact, it's very likely this is the case unless we are in an ERROR
  // state. If we don't wait for the pending I/O to be done, then when the I/O
  // completes, it may write to invalid memory. AppVerifier will flag this
  // problem too. So we disconnect from the pipe and then wait for the server
  // to get into error state so that the pending I/O will fail and get
  // cleared.
  DisconnectNamedPipe(pipe_);
  int num_tries = 100;
  while (num_tries-- && server_state_ != IPC_SERVER_STATE_ERROR) {
    Sleep(10);
  }

  // Unregister wait on the pipe.
  if (pipe_wait_handle_) {
    // Wait for already executing callbacks to finish.
    UnregisterWaitEx(pipe_wait_handle_, INVALID_HANDLE_VALUE);
  }

  // Close the pipe to avoid further client connections.
  if (pipe_) {
    CloseHandle(pipe_);
  }

  // Request all ClientInfo objects to unregister all waits.
  // No need to enter the critical section because no one is allowed to modify
  // the clients_ list once the shutting_down_ flag is set.
  std::list<ClientInfo*>::iterator iter;
  for (iter = clients_.begin(); iter != clients_.end(); ++iter) {
    ClientInfo* client_info = *iter;
    // Unregister waits. Wait for already executing callbacks to finish.
    // Unregister the client process exit wait first and only then unregister
    // the dump request wait.  The reason is that the OnClientExit callback
    // also unregisters the dump request wait and such a race (doing the same
    // unregistration from two threads) is undesirable.
    client_info->UnregisterProcessExitWait(true);
    client_info->UnregisterDumpRequestWaitAndBlockUntilNoPending();

    // Destroying the ClientInfo here is safe because all wait operations for
    // this ClientInfo were unregistered and no pending or running callbacks
    // for this ClientInfo can possible exist (block_until_no_pending option
    // was used).
    delete client_info;
  }

  if (server_alive_handle_) {
    // Release the mutex before closing the handle so that clients requesting
    // dumps wait for a long time for the server to generate a dump.
    ReleaseMutex(server_alive_handle_);
    CloseHandle(server_alive_handle_);
  }

  if (overlapped_.hEvent) {
    CloseHandle(overlapped_.hEvent);
  }

  DeleteCriticalSection(&sync_);
}

bool CrashGenerationServer::Start() {
  if (server_state_ != IPC_SERVER_STATE_UNINITIALIZED) {
    return false;
  }

  server_state_ = IPC_SERVER_STATE_INITIAL;

  server_alive_handle_ = CreateMutex(NULL, TRUE, NULL);
  if (!server_alive_handle_) {
    return false;
  }

  // Event to signal the client connection and pipe reads and writes.
  overlapped_.hEvent = CreateEvent(NULL,   // Security descriptor.
                                   TRUE,   // Manual reset.
                                   FALSE,  // Initially nonsignaled.
                                   NULL);  // Name.
  if (!overlapped_.hEvent) {
    return false;
  }

  // Register a callback with the thread pool for the client connection.
  if (!RegisterWaitForSingleObject(&pipe_wait_handle_,
                                   overlapped_.hEvent,
                                   OnPipeConnected,
                                   this,
                                   INFINITE,
                                   kPipeIOThreadFlags)) {
    return false;
  }

  pipe_ = CreateNamedPipe(pipe_name_.c_str(),
                          kPipeAttr,
                          kPipeMode,
                          1,
                          kOutBufferSize,
                          kInBufferSize,
                          0,
                          pipe_sec_attrs_);
  if (pipe_ == INVALID_HANDLE_VALUE) {
    return false;
  }

  // Kick-start the state machine. This will initiate an asynchronous wait
  // for client connections.
  if (!SetEvent(overlapped_.hEvent)) {
    server_state_ = IPC_SERVER_STATE_ERROR;
    return false;
  }

  // If we are in error state, it's because we failed to start listening.
  return true;
}

// If the server thread serving clients ever gets into the
// ERROR state, reset the event, close the pipe and remain
// in the error state forever. Error state means something
// that we didn't account for has happened, and it's dangerous
// to do anything unknowingly.
void CrashGenerationServer::HandleErrorState() {
  assert(server_state_ == IPC_SERVER_STATE_ERROR);

  // If the server is shutting down anyway, don't clean up
  // here since shut down process will clean up.
  if (shutting_down_) {
    return;
  }

  if (pipe_wait_handle_) {
    UnregisterWait(pipe_wait_handle_);
    pipe_wait_handle_ = NULL;
  }

  if (pipe_) {
    CloseHandle(pipe_);
    pipe_ = NULL;
  }

  if (overlapped_.hEvent) {
    CloseHandle(overlapped_.hEvent);
    overlapped_.hEvent = NULL;
  }
}

// When the server thread serving clients is in the INITIAL state,
// try to connect to the pipe asynchronously. If the connection
// finishes synchronously, directly go into the CONNECTED state;
// otherwise go into the CONNECTING state. For any problems, go
// into the ERROR state.
void CrashGenerationServer::HandleInitialState() {
  assert(server_state_ == IPC_SERVER_STATE_INITIAL);

  if (!ResetEvent(overlapped_.hEvent)) {
    EnterErrorState();
    return;
  }

  bool success = ConnectNamedPipe(pipe_, &overlapped_) != FALSE;
  DWORD error_code = success ? ERROR_SUCCESS : GetLastError();

  // From MSDN, it is not clear that when ConnectNamedPipe is used
  // in an overlapped mode, will it ever return non-zero value, and
  // if so, in what cases.
  assert(!success);

  switch (error_code) {
    case ERROR_IO_PENDING:
      EnterStateWhenSignaled(IPC_SERVER_STATE_CONNECTING);
      break;

    case ERROR_PIPE_CONNECTED:
      EnterStateImmediately(IPC_SERVER_STATE_CONNECTED);
      break;

    default:
      EnterErrorState();
      break;
  }
}

// When the server thread serving the clients is in the CONNECTING state,
// try to get the result of the asynchronous connection request using
// the OVERLAPPED object. If the result indicates the connection is done,
// go into the CONNECTED state. If the result indicates I/O is still
// INCOMPLETE, remain in the CONNECTING state. For any problems,
// go into the DISCONNECTING state.
void CrashGenerationServer::HandleConnectingState() {
  assert(server_state_ == IPC_SERVER_STATE_CONNECTING);

  DWORD bytes_count = 0;
  bool success = GetOverlappedResult(pipe_,
                                     &overlapped_,
                                     &bytes_count,
                                     FALSE) != FALSE;
  DWORD error_code = success ? ERROR_SUCCESS : GetLastError();

  if (success) {
    EnterStateImmediately(IPC_SERVER_STATE_CONNECTED);
  } else if (error_code != ERROR_IO_INCOMPLETE) {
    EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
  } else {
    // remain in CONNECTING state
  }
}

// When the server thread serving the clients is in the CONNECTED state,
// try to issue an asynchronous read from the pipe. If read completes
// synchronously or if I/O is pending then go into the READING state.
// For any problems, go into the DISCONNECTING state.
void CrashGenerationServer::HandleConnectedState() {
  assert(server_state_ == IPC_SERVER_STATE_CONNECTED);

  DWORD bytes_count = 0;
  memset(&msg_, 0, sizeof(msg_));
  bool success = ReadFile(pipe_,
                          &msg_,
                          sizeof(msg_),
                          &bytes_count,
                          &overlapped_) != FALSE;
  DWORD error_code = success ? ERROR_SUCCESS : GetLastError();

  // Note that the asynchronous read issued above can finish before the
  // code below executes. But, it is okay to change state after issuing
  // the asynchronous read. This is because even if the asynchronous read
  // is done, the callback for it would not be executed until the current
  // thread finishes its execution.
  if (success || error_code == ERROR_IO_PENDING) {
    EnterStateWhenSignaled(IPC_SERVER_STATE_READING);
  } else {
    EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
  }
}

// When the server thread serving the clients is in the READING state,
// try to get the result of the async read. If async read is done,
// go into the READ_DONE state. For any problems, go into the
// DISCONNECTING state.
void CrashGenerationServer::HandleReadingState() {
  assert(server_state_ == IPC_SERVER_STATE_READING);

  DWORD bytes_count = 0;
  bool success = GetOverlappedResult(pipe_,
                                     &overlapped_,
                                     &bytes_count,
                                     FALSE) != FALSE;
  if (success && bytes_count == sizeof(ProtocolMessage)) {
    EnterStateImmediately(IPC_SERVER_STATE_READ_DONE);
    return;
  }

  assert(!CheckForIOIncomplete(success));
  EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
}

// When the server thread serving the client is in the READ_DONE state,
// validate the client's request message, register the client by
// creating appropriate objects and prepare the response.  Then try to
// write the response to the pipe asynchronously. If that succeeds,
// go into the WRITING state. For any problems, go into the DISCONNECTING
// state.
void CrashGenerationServer::HandleReadDoneState() {
  assert(server_state_ == IPC_SERVER_STATE_READ_DONE);

  if (!IsClientRequestValid(msg_)) {
    EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
    return;
  }

  if (msg_.tag == MESSAGE_TAG_UPLOAD_REQUEST) {
    if (upload_request_callback_)
      upload_request_callback_(upload_context_, msg_.id);
    EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
    return;
  }

  scoped_ptr<ClientInfo> client_info(
      new ClientInfo(this,
                     msg_.id,
                     msg_.dump_type,
                     msg_.thread_id,
                     msg_.exception_pointers,
                     msg_.assert_info,
                     msg_.custom_client_info));

  if (!client_info->Initialize()) {
    EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
    return;
  }

  // Issues an asynchronous WriteFile call if successful.
  // Iff successful, assigns ownership of the client_info pointer to the server
  // instance, in which case we must be sure not to free it in this function.
  if (!RespondToClient(client_info.get())) {
    EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
    return;
  }

  // This is only valid as long as it can be found in the clients_ list
  client_info_ = client_info.release();

  // Note that the asynchronous write issued by RespondToClient function
  // can finish before  the code below executes. But it is okay to change
  // state after issuing the asynchronous write. This is because even if
  // the asynchronous write is done, the callback for it would not be
  // executed until the current thread finishes its execution.
  EnterStateWhenSignaled(IPC_SERVER_STATE_WRITING);
}

// When the server thread serving the clients is in the WRITING state,
// try to get the result of the async write. If the async write is done,
// go into the WRITE_DONE state. For any problems, go into the
// DISONNECTING state.
void CrashGenerationServer::HandleWritingState() {
  assert(server_state_ == IPC_SERVER_STATE_WRITING);

  DWORD bytes_count = 0;
  bool success = GetOverlappedResult(pipe_,
                                     &overlapped_,
                                     &bytes_count,
                                     FALSE) != FALSE;
  if (success) {
    EnterStateImmediately(IPC_SERVER_STATE_WRITE_DONE);
    return;
  }

  assert(!CheckForIOIncomplete(success));
  EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
}

// When the server thread serving the clients is in the WRITE_DONE state,
// try to issue an async read on the pipe. If the read completes synchronously
// or if I/O is still pending then go into the READING_ACK state. For any
// issues, go into the DISCONNECTING state.
void CrashGenerationServer::HandleWriteDoneState() {
  assert(server_state_ == IPC_SERVER_STATE_WRITE_DONE);

  DWORD bytes_count = 0;
  bool success = ReadFile(pipe_,
                           &msg_,
                           sizeof(msg_),
                           &bytes_count,
                           &overlapped_) != FALSE;
  DWORD error_code = success ? ERROR_SUCCESS : GetLastError();

  if (success) {
    EnterStateImmediately(IPC_SERVER_STATE_READING_ACK);
  } else if (error_code == ERROR_IO_PENDING) {
    EnterStateWhenSignaled(IPC_SERVER_STATE_READING_ACK);
  } else {
    EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
  }
}

// When the server thread serving the clients is in the READING_ACK state,
// try to get result of async read. Go into the DISCONNECTING state.
void CrashGenerationServer::HandleReadingAckState() {
  assert(server_state_ == IPC_SERVER_STATE_READING_ACK);

  DWORD bytes_count = 0;
  bool success = GetOverlappedResult(pipe_,
                                     &overlapped_,
                                     &bytes_count,
                                     FALSE) != FALSE;
  if (success) {
    // The connection handshake with the client is now complete; perform
    // the callback.
    if (connect_callback_) {
      // Note that there is only a single copy of the ClientInfo of the
      // currently connected client.  However it is being referenced from
      // two different places:
      //  - the client_info_ member
      //  - the clients_ list
      // The lifetime of this ClientInfo depends on the lifetime of the
      // client process - basically it can go away at any time.
      // However, as long as it is referenced by the clients_ list it
      // is guaranteed to be valid. Enter the critical section and check
      // to see whether the client_info_ can be found in the list.
      // If found, execute the callback and only then leave the critical
      // section.
      AutoCriticalSection lock(&sync_);

      bool client_is_still_alive = false;
      std::list<ClientInfo*>::iterator iter;
      for (iter = clients_.begin(); iter != clients_.end(); ++iter) {
        if (client_info_ == *iter) {
          client_is_still_alive = true;
          break;
        }
      }

      if (client_is_still_alive) {
        connect_callback_(connect_context_, client_info_);
      }
    }
  } else {
    assert(!CheckForIOIncomplete(success));
  }

  EnterStateImmediately(IPC_SERVER_STATE_DISCONNECTING);
}

// When the server thread serving the client is in the DISCONNECTING state,
// disconnect from the pipe and reset the event. If anything fails, go into
// the ERROR state. If it goes well, go into the INITIAL state and set the
// event to start all over again.
void CrashGenerationServer::HandleDisconnectingState() {
  assert(server_state_ == IPC_SERVER_STATE_DISCONNECTING);

  // Done serving the client.
  client_info_ = NULL;

  overlapped_.Internal = NULL;
  overlapped_.InternalHigh = NULL;
  overlapped_.Offset = 0;
  overlapped_.OffsetHigh = 0;
  overlapped_.Pointer = NULL;

  if (!ResetEvent(overlapped_.hEvent)) {
    EnterErrorState();
    return;
  }

  if (!DisconnectNamedPipe(pipe_)) {
    EnterErrorState();
    return;
  }

  // If the server is shutting down do not connect to the
  // next client.
  if (shutting_down_) {
    return;
  }

  EnterStateImmediately(IPC_SERVER_STATE_INITIAL);
}

void CrashGenerationServer::EnterErrorState() {
  SetEvent(overlapped_.hEvent);
  server_state_ = IPC_SERVER_STATE_ERROR;
}

void CrashGenerationServer::EnterStateWhenSignaled(IPCServerState state) {
  server_state_ = state;
}

void CrashGenerationServer::EnterStateImmediately(IPCServerState state) {
  server_state_ = state;

  if (!SetEvent(overlapped_.hEvent)) {
    server_state_ = IPC_SERVER_STATE_ERROR;
  }
}

bool CrashGenerationServer::PrepareReply(const ClientInfo& client_info,
                                         ProtocolMessage* reply) const {
  reply->tag = MESSAGE_TAG_REGISTRATION_RESPONSE;
  reply->id = GetCurrentProcessId();

  if (CreateClientHandles(client_info, reply)) {
    return true;
  }

  // Closing of remote handles (belonging to a different process) can
  // only be done through DuplicateHandle.
  if (reply->dump_request_handle) {
    DuplicateHandle(client_info.process_handle(),  // hSourceProcessHandle
                    reply->dump_request_handle,    // hSourceHandle
                    NULL,                          // hTargetProcessHandle
                    0,                             // lpTargetHandle
                    0,                             // dwDesiredAccess
                    FALSE,                         // bInheritHandle
                    DUPLICATE_CLOSE_SOURCE);       // dwOptions
    reply->dump_request_handle = NULL;
  }

  if (reply->dump_generated_handle) {
    DuplicateHandle(client_info.process_handle(),  // hSourceProcessHandle
                    reply->dump_generated_handle,  // hSourceHandle
                    NULL,                          // hTargetProcessHandle
                    0,                             // lpTargetHandle
                    0,                             // dwDesiredAccess
                    FALSE,                         // bInheritHandle
                    DUPLICATE_CLOSE_SOURCE);       // dwOptions
    reply->dump_generated_handle = NULL;
  }

  if (reply->server_alive_handle) {
    DuplicateHandle(client_info.process_handle(),  // hSourceProcessHandle
                    reply->server_alive_handle,    // hSourceHandle
                    NULL,                          // hTargetProcessHandle
                    0,                             // lpTargetHandle
                    0,                             // dwDesiredAccess
                    FALSE,                         // bInheritHandle
                    DUPLICATE_CLOSE_SOURCE);       // dwOptions
    reply->server_alive_handle = NULL;
  }

  return false;
}

bool CrashGenerationServer::CreateClientHandles(const ClientInfo& client_info,
                                                ProtocolMessage* reply) const {
  HANDLE current_process = GetCurrentProcess();
  if (!DuplicateHandle(current_process,
                       client_info.dump_requested_handle(),
                       client_info.process_handle(),
                       &reply->dump_request_handle,
                       kDumpRequestEventAccess,
                       FALSE,
                       0)) {
    return false;
  }

  if (!DuplicateHandle(current_process,
                       client_info.dump_generated_handle(),
                       client_info.process_handle(),
                       &reply->dump_generated_handle,
                       kDumpGeneratedEventAccess,
                       FALSE,
                       0)) {
    return false;
  }

  if (!DuplicateHandle(current_process,
                       server_alive_handle_,
                       client_info.process_handle(),
                       &reply->server_alive_handle,
                       kMutexAccess,
                       FALSE,
                       0)) {
    return false;
  }

  return true;
}

bool CrashGenerationServer::RespondToClient(ClientInfo* client_info) {
  ProtocolMessage reply;
  if (!PrepareReply(*client_info, &reply)) {
    return false;
  }

  DWORD bytes_count = 0;
  bool success = WriteFile(pipe_,
                            &reply,
                            sizeof(reply),
                            &bytes_count,
                            &overlapped_) != FALSE;
  DWORD error_code = success ? ERROR_SUCCESS : GetLastError();

  if (!success && error_code != ERROR_IO_PENDING) {
    return false;
  }

  // Takes over ownership of client_info. We MUST return true if AddClient
  // succeeds.
  return AddClient(client_info);
}

// The server thread servicing the clients runs this method. The method
// implements the state machine described in ReadMe.txt along with the
// helper methods HandleXXXState.
void CrashGenerationServer::HandleConnectionRequest() {
  // If the server is shutting down, get into ERROR state, reset the event so
  // more workers don't run and return immediately.
  if (shutting_down_) {
    server_state_ = IPC_SERVER_STATE_ERROR;
    ResetEvent(overlapped_.hEvent);
    return;
  }

  switch (server_state_) {
    case IPC_SERVER_STATE_ERROR:
      HandleErrorState();
      break;

    case IPC_SERVER_STATE_INITIAL:
      HandleInitialState();
      break;

    case IPC_SERVER_STATE_CONNECTING:
      HandleConnectingState();
      break;

    case IPC_SERVER_STATE_CONNECTED:
      HandleConnectedState();
      break;

    case IPC_SERVER_STATE_READING:
      HandleReadingState();
      break;

    case IPC_SERVER_STATE_READ_DONE:
      HandleReadDoneState();
      break;

    case IPC_SERVER_STATE_WRITING:
      HandleWritingState();
      break;

    case IPC_SERVER_STATE_WRITE_DONE:
      HandleWriteDoneState();
      break;

    case IPC_SERVER_STATE_READING_ACK:
      HandleReadingAckState();
      break;

    case IPC_SERVER_STATE_DISCONNECTING:
      HandleDisconnectingState();
      break;

    default:
      assert(false);
      // This indicates that we added one more state without
      // adding handling code.
      server_state_ = IPC_SERVER_STATE_ERROR;
      break;
  }
}

bool CrashGenerationServer::AddClient(ClientInfo* client_info) {
  HANDLE request_wait_handle = NULL;
  if (!RegisterWaitForSingleObject(&request_wait_handle,
                                   client_info->dump_requested_handle(),
                                   OnDumpRequest,
                                   client_info,
                                   INFINITE,
                                   kDumpRequestThreadFlags)) {
    return false;
  }

  client_info->set_dump_request_wait_handle(request_wait_handle);

  // OnClientEnd will be called when the client process terminates.
  HANDLE process_wait_handle = NULL;
  if (!RegisterWaitForSingleObject(&process_wait_handle,
                                   client_info->process_handle(),
                                   OnClientEnd,
                                   client_info,
                                   INFINITE,
                                   WT_EXECUTEONLYONCE)) {
    return false;
  }

  client_info->set_process_exit_wait_handle(process_wait_handle);

  // New scope to hold the lock for the shortest time.
  {
    AutoCriticalSection lock(&sync_);
    if (shutting_down_) {
      // If server is shutting down, don't add new clients
      return false;
    }
    clients_.push_back(client_info);
  }

  return true;
}

// static
void CALLBACK CrashGenerationServer::OnPipeConnected(void* context, BOOLEAN) {
  assert(context);

  CrashGenerationServer* obj =
      reinterpret_cast<CrashGenerationServer*>(context);
  obj->HandleConnectionRequest();
}

// static
void CALLBACK CrashGenerationServer::OnDumpRequest(void* context, BOOLEAN) {
  assert(context);
  ClientInfo* client_info = reinterpret_cast<ClientInfo*>(context);

  CrashGenerationServer* crash_server = client_info->crash_server();
  assert(crash_server);
  if (crash_server->pre_fetch_custom_info_) {
    client_info->PopulateCustomInfo();
  }
  crash_server->HandleDumpRequest(*client_info);

  ResetEvent(client_info->dump_requested_handle());
}

// static
void CALLBACK CrashGenerationServer::OnClientEnd(void* context, BOOLEAN) {
  assert(context);
  ClientInfo* client_info = reinterpret_cast<ClientInfo*>(context);

  CrashGenerationServer* crash_server = client_info->crash_server();
  assert(crash_server);

  crash_server->HandleClientProcessExit(client_info);
}

void CrashGenerationServer::HandleClientProcessExit(ClientInfo* client_info) {
  assert(client_info);

  // Must unregister the dump request wait operation and wait for any
  // dump requests that might be pending to finish before proceeding
  // with the client_info cleanup.
  client_info->UnregisterDumpRequestWaitAndBlockUntilNoPending();

  if (exit_callback_) {
    exit_callback_(exit_context_, client_info);
  }

  // Start a new scope to release lock automatically.
  {
    AutoCriticalSection lock(&sync_);
    if (shutting_down_) {
      // The crash generation server is shutting down and as part of the
      // shutdown process it will delete all clients from the clients_ list.
      return;
    }
    clients_.remove(client_info);
  }

  // Explicitly unregister the process exit wait using the non-blocking method.
  // Otherwise, the destructor will attempt to unregister it using the blocking
  // method which will lead to a deadlock because it is being called from the
  // callback of the same wait operation
  client_info->UnregisterProcessExitWait(false);

  delete client_info;
}

void CrashGenerationServer::HandleDumpRequest(const ClientInfo& client_info) {
  bool execute_callback = true;
  // Generate the dump only if it's explicitly requested by the
  // server application; otherwise the server might want to generate
  // dump in the callback.
  std::wstring dump_path;
  if (generate_dumps_) {
    if (!GenerateDump(client_info, &dump_path)) {
      // client proccess terminated or some other error
      execute_callback = false;
    }
  }

  if (dump_callback_ && execute_callback) {
    std::wstring* ptr_dump_path = (dump_path == L"") ? NULL : &dump_path;
    dump_callback_(dump_context_, &client_info, ptr_dump_path);
  }

  SetEvent(client_info.dump_generated_handle());
}

bool CrashGenerationServer::GenerateDump(const ClientInfo& client,
                                         std::wstring* dump_path) {
  assert(client.pid() != 0);
  assert(client.process_handle());

  // We have to get the address of EXCEPTION_INFORMATION from
  // the client process address space.
  EXCEPTION_POINTERS* client_ex_info = NULL;
  if (!client.GetClientExceptionInfo(&client_ex_info)) {
    return false;
  }

  DWORD client_thread_id = 0;
  if (!client.GetClientThreadId(&client_thread_id)) {
    return false;
  }

  MinidumpGenerator dump_generator(dump_path_,
                                   client.process_handle(),
                                   client.pid(),
                                   client_thread_id,
                                   GetCurrentThreadId(),
                                   client_ex_info,
                                   client.assert_info(),
                                   client.dump_type(),
                                   true);

  if (!dump_generator.GenerateDumpFile(dump_path)) {
    return false;
  }

  // If the client requests a full memory dump, we will write a normal mini
  // dump and a full memory dump. Both dump files use the same uuid as file
  // name prefix.
  if (client.dump_type() & MiniDumpWithFullMemory) {
    std::wstring full_dump_path;
    if (!dump_generator.GenerateFullDumpFile(&full_dump_path)) {
      return false;
    }
  }

  return dump_generator.WriteMinidump();
}

}  // namespace google_breakpad
