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

#ifndef CRASHPAD_UTIL_MACH_CHILD_PORT_HANDSHAKE_H_
#define CRASHPAD_UTIL_MACH_CHILD_PORT_HANDSHAKE_H_

#include <mach/mach.h>

#include <string>

#include "base/files/scoped_file.h"
#include "base/macros.h"
#include "util/mach/child_port_types.h"

namespace crashpad {

namespace test {
namespace {
class ChildPortHandshakeTest;
}  // namespace
}  // namespace test

//! \brief Implements a handshake protocol that allows processes to exchange
//!     port rights.
//!
//! Ordinarily, there is no way for parent and child processes to exchange port
//! rights, outside of the rights that children inherit from their parents.
//! These include task-special ports and exception ports, but all of these have
//! system-defined uses, and cannot reliably be replaced: in a multi-threaded
//! parent, it is impossible to temporarily change an inheritable port while
//! maintaining a guarantee that another thread will not attempt to use it, and
//! in children, it difficult to guarantee that nothing will attempt to use an
//! inheritable port before it can be replaced with the correct one. This latter
//! concern is becoming increasingly more pronounced as system libraries perform
//! more operations that rely on an inherited port in module initializers.
//!
//! The protocol implemented by this class involves a server that runs in one
//! process. The server is published with the bootstrap server, which the other
//! process has access to because the bootstrap port is one of the inherited
//! task-special ports. The two processes also share a pipe, which the server
//! can write to and the client can read from. The server will write a random
//! token to this pipe, along with the name under which its service has been
//! registered with the bootstrap server. The client can then obtain a send
//! right to this service with `bootstrap_look_up()`, and send a check-in
//! message containing the token value and the port right of its choice by
//! calling `child_port_check_in()`.
//!
//! The inclusion of the token authenticates the client to the server. This is
//! necessary because the service is published with the bootstrap server, which
//! opens up access to it to more than the intended client. Because the token is
//! passed to the client by a shared pipe, it constitutes a shared secret not
//! known by other processes that may have incidental access to the server. The
//! ChildPortHandshake server considers its randomly-generated token valid until
//! a client checks in with it. This mechanism is used instead of examining the
//! request message’s audit trailer to verify the sender’s process ID because in
//! some process architectures, it may be impossible to verify the client’s
//! process ID.
//!
//! The shared pipe serves another purpose: the server monitors it for an
//! end-of-file (no readers) condition. Once detected, it will stop its blocking
//! wait for a client to check in. This mechanism was also chosen for its
//! ability to function properly in diverse process architectures.
//!
//! This class can be used to allow a child process to provide its parent with a
//! send right to its task port, in cases where it is desirable for the parent
//! to have such access. It can also be used to allow a parent process to
//! transfer a receive right to a child process that implements the server for
//! that right, or for a child process to establish its own server and provide
//! its parent with a send right to that server, for cases where a service is
//! provided and it is undesirable or impossible to provide it via the bootstrap
//! or launchd interfaces.
//!
//! Example parent process, running a client that sends a receive right to its
//! child:
//! \code
//!   ChildPortHandshake child_port_handshake;
//!   base::ScopedFD server_write_fd = child_port_handshake.ServerWriteFD();
//!   std::string server_write_fd_string =
//!       base::StringPrintf("%d", server_write_fd.get());
//!
//!   pid_t pid = fork();
//!   if (pid == 0) {
//!     // Child
//!
//!     // Close all file descriptors above STDERR_FILENO except for
//!     // server_write_fd. Let the child know what file descriptor to use for
//!     // server_write_fd by passing it as argv[1]. Example code for the child
//!     // process is below.
//!     CloseMultipleNowOrOnExec(STDERR_FILENO + 1, server_write_fd.get());
//!     execlp("./child", "child", server_write_fd_string.c_str(), nullptr);
//!   }
//!
//!   // Parent
//!
//!   // Close the child’s end of the pipe.
//!   server_write_fd.reset();
//!
//!   // Make a new Mach receive right.
//!   base::mac::ScopedMachReceiveRight
//!       receive_right(NewMachPort(MACH_PORT_RIGHT_RECEIVE));
//!
//!   // Make a send right corresponding to the receive right.
//!   mach_port_t send_right;
//!   mach_msg_type_name_t send_right_type;
//!   mach_port_extract_right(mach_task_self(),
//!                           receive_right.get(),
//!                           MACH_MSG_TYPE_MAKE_SEND,
//!                           &send_right,
//!                           &send_right_type);
//!   base::mac::ScopedMachSendRight send_right_owner(send_right);
//!
//!   // Send the receive right to the child process, retaining the send right
//!   // for use in the parent process.
//!   if (child_port_handshake.RunClient(receive_right.get(),
//!                                      MACH_MSG_TYPE_MOVE_RECEIVE)) {
//!     ignore_result(receive_right.release());
//!   }
//! \endcode
//!
//! Example child process, running a server that receives a receive right from
//! its parent:
//! \code
//!   int main(int argc, char* argv[]) {
//!     // The parent passed server_write_fd in argv[1].
//!     base::ScopedFD server_write_fd(atoi(argv[1]));
//!
//!     // Obtain a receive right from the parent process.
//!     base::mac::ScopedMachReceiveRight receive_right(
//!         ChildPortHandshake::RunServerForFD(
//!             std::move(server_write_fd),
//!             ChildPortHandshake::PortRightType::kReceiveRight));
//!   }
//! \endcode
class ChildPortHandshake {
 public:
  //! \brief Controls whether a receive or send right is expected to be
  //!     obtained from the client by the server’s call to RunServer().
  enum class PortRightType {
    //! \brief The server expects to receive a receive right.
    kReceiveRight = 0,

    //! \brief The server expects to receive a send or send-once right.
    kSendRight,
  };

  ChildPortHandshake();
  ~ChildPortHandshake();

  //! \brief Obtains the “read” side of the pipe, to be used by the client.
  //!
  //! This file descriptor must be passed to RunClientForFD().
  //!
  //! \return The file descriptor that the client should read from.
  base::ScopedFD ClientReadFD();

  //! \brief Obtains the “write” side of the pipe, to be used by the server.
  //!
  //! This file descriptor must be passed to RunServerForFD().
  //!
  //! \return The file descriptor that the server should write to.
  base::ScopedFD ServerWriteFD();

  //! \brief Runs the server.
  //!
  //! This method closes the “read” side of the pipe in-process, so that the
  //! client process holds the only file descriptor that can read from the pipe.
  //! It then calls RunServerForFD() using the “write” side of the pipe. If
  //! ClientReadFD() has already been called in the server process, the caller
  //! must ensure that the file descriptor returned by ClientReadFD() is closed
  //! prior to calling this method.
  mach_port_t RunServer(PortRightType port_right_type);

  //! \brief Runs the client.
  //!
  //! This method closes the “write” side of the pipe in-process, so that the
  //! server process holds the only file descriptor that can write to the pipe.
  //! It then calls RunClientForFD() using the “read” side of the pipe. If
  //! ServerWriteFD() has already been called in the client process, the caller
  //! must ensure that the file descriptor returned by ServerWriteFD() is closed
  //! prior to calling this method.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool RunClient(mach_port_t port, mach_msg_type_name_t right_type);

  //! \brief Runs the server.
  //!
  //! If a ChildPortHandshake object is available, don’t call this static
  //! function. Instead, call RunServer(), which wraps this function. When using
  //! this function, the caller is responsible for ensuring that the client
  //! “read” side of the pipe is closed in the server process prior to calling
  //! this function.
  //!
  //! This function performs these tasks:
  //!  - Creates a random token and sends it via the pipe.
  //!  - Checks its service in with the bootstrap server, and sends the name
  //!    of its bootstrap service mapping via the pipe.
  //!  - Simultaneously receives messages on its Mach server and monitors the
  //!    pipe for end-of-file. This is a blocking operation.
  //!  - When a Mach message is received, calls HandleChildPortCheckIn() to
  //!    interpret and validate it, and if the message is valid, returns the
  //!    port right extracted from the message. If the message is not valid,
  //!    this method will continue waiting for a valid message. Valid messages
  //!    are properly formatted and have the correct token. The right carried in
  //!    a valid message will be returned. If a message is not valid, this
  //!    method will continue waiting for pipe EOF or a valid message.
  //!  - When notified of pipe EOF, returns `MACH_PORT_NULL`.
  //!  - Regardless of return value, destroys the server’s receive right and
  //!    closes the pipe.
  //!
  //! \param[in] server_write_fd The write side of the pipe shared with the
  //!     client process. This function takes ownership of this file descriptor,
  //!     and will close it prior to returning.
  //! \param[in] port_right_type The port right type expected to be received
  //!     from the client. If the port right received from the client does not
  //!     match the expected type, the received port right will be destroyed,
  //!     and `MACH_PORT_NULL` will be returned.
  //!
  //! \return On success, the port right provided by the client. The caller
  //!     takes ownership of this right. On failure, `MACH_PORT_NULL`,
  //!     indicating that the client did not check in properly before
  //!     terminating, where termination is detected by detecting that the read
  //!     side of the shared pipe has closed. On failure, a message indicating
  //!     the nature of the failure will be logged.
  static mach_port_t RunServerForFD(base::ScopedFD server_write_fd,
                                    PortRightType port_right_type);

  //! \brief Runs the client.
  //!
  //! If a ChildPortHandshake object is available, don’t call this static
  //! function. Instead, call RunClient(), which wraps this function. When using
  //! this function, the caller is responsible for ensuring that the server
  //! “write” side of the pipe is closed in the client process prior to calling
  //! this function.
  //!
  //! This function performs these tasks:
  //!  - Reads the token from the pipe.
  //!  - Reads the bootstrap service name from the pipe.
  //!  - Obtains a send right to the server by calling `bootstrap_look_up()`.
  //!  - Sends a check-in message to the server by calling
  //!    `child_port_check_in()`, providing the token and the user-supplied port
  //!    right.
  //!  - Deallocates the send right to the server, and closes the pipe.
  //!
  //! There is no return value because `child_port_check_in()` is a MIG
  //! `simpleroutine`, and the server does not send a reply. This allows
  //! check-in to occur without blocking to wait for a reply.
  //!
  //! \param[in] client_read_fd The “read” side of the pipe shared with the
  //!     server process. This function takes ownership of this file descriptor,
  //!     and will close it prior to returning.
  //! \param[in] port The port right that will be passed to the server by
  //!     `child_port_check_in()`.
  //! \param[in] right_type The right type to furnish the server with. If \a
  //!     port is a send right, this can be `MACH_MSG_TYPE_COPY_SEND` or
  //!     `MACH_MSG_TYPE_MOVE_SEND`. If \a port is a send-once right, this can
  //!     be `MACH_MSG_TYPE_MOVE_SEND_ONCE`. If \a port is a receive right, this
  //!     can be `MACH_MSG_TYPE_MAKE_SEND`, `MACH_MSG_TYPE_MAKE_SEND_ONCE`, or
  //!     `MACH_MSG_TYPE_MOVE_RECEIVE`.
  //!
  //! \return `true` on success, `false` on failure with a message logged. On
  //!     failure, the port right corresponding to a \a right_type of
  //!     `MACH_MSG_TYPE_MOVE_*` is not consumed, and the caller must dispose of
  //!     the right if necessary.
  static bool RunClientForFD(base::ScopedFD client_read_fd,
                             mach_port_t port,
                             mach_msg_type_name_t right_type);

 private:
  //! \brief Runs the read-from-pipe portion of the client’s side of the
  //!     handshake. This is an implementation detail of RunClient and is only
  //!     exposed for testing purposes.
  //!
  //! When using this function and RunClientInternal_SendCheckIn(), the caller
  //! is responsible for closing \a pipe_read at an appropriate time, normally
  //! after calling RunClientInternal_SendCheckIn().
  //!
  //! \param[in] pipe_read The “read” side of the pipe shared with the server
  //!     process.
  //! \param[out] token The token value read from \a pipe_read.
  //! \param[out] service_name The service name as registered with the bootstrap
  //!     server, read from \a pipe_read.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  static bool RunClientInternal_ReadPipe(int pipe_read,
                                         child_port_token_t* token,
                                         std::string* service_name);

  //! \brief Runs the check-in portion of the client’s side of the handshake.
  //!     This is an implementation detail of RunClient and is only exposed for
  //!     testing purposes.
  //!
  //! When using this RunClientInternal_ReadPipe() and this function, the caller
  //! is responsible for closing the “read” side of the pipe at an appropriate
  //! time, normally after calling this function.
  //!
  //! \param[in] service_name The service name as registered with the bootstrap
  //!     server, to be looked up with `bootstrap_look_up()`.
  //! \param[in] token The token value to provide during check-in.
  //! \param[in] port The port that will be passed to the server by
  //!     `child_port_check_in()`.
  //! \param[in] right_type The right type to furnish the server with.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  static bool RunClientInternal_SendCheckIn(const std::string& service_name,
                                            child_port_token_t token,
                                            mach_port_t port,
                                            mach_msg_type_name_t right_type);

  base::ScopedFD client_read_fd_;
  base::ScopedFD server_write_fd_;

  friend class test::ChildPortHandshakeTest;

  DISALLOW_COPY_AND_ASSIGN(ChildPortHandshake);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MACH_CHILD_PORT_HANDSHAKE_H_
