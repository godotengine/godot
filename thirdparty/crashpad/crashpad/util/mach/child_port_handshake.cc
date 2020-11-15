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

#include "util/mach/child_port_handshake.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/event.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <utility>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "base/mac/scoped_mach_port.h"
#include "base/posix/eintr_wrapper.h"
#include "base/rand_util.h"
#include "base/strings/stringprintf.h"
#include "util/file/file_io.h"
#include "util/mach/child_port.h"
#include "util/mach/child_port_server.h"
#include "util/mach/mach_extensions.h"
#include "util/mach/mach_message.h"
#include "util/mach/mach_message_server.h"
#include "util/misc/implicit_cast.h"
#include "util/misc/random_string.h"

namespace crashpad {
namespace {

class ChildPortHandshakeServer final : public ChildPortServer::Interface {
 public:
  ChildPortHandshakeServer();
  ~ChildPortHandshakeServer();

  mach_port_t RunServer(base::ScopedFD server_write_fd,
                        ChildPortHandshake::PortRightType port_right_type);

 private:
  // ChildPortServer::Interface:
  kern_return_t HandleChildPortCheckIn(child_port_server_t server,
                                       child_port_token_t token,
                                       mach_port_t port,
                                       mach_msg_type_name_t right_type,
                                       const mach_msg_trailer_t* trailer,
                                       bool* destroy_request) override;

  child_port_token_t token_;
  mach_port_t port_;
  mach_msg_type_name_t right_type_;
  bool checked_in_;

  DISALLOW_COPY_AND_ASSIGN(ChildPortHandshakeServer);
};

ChildPortHandshakeServer::ChildPortHandshakeServer()
    : token_(0),
      port_(MACH_PORT_NULL),
      right_type_(MACH_MSG_TYPE_PORT_NONE),
      checked_in_(false) {
}

ChildPortHandshakeServer::~ChildPortHandshakeServer() {
}

mach_port_t ChildPortHandshakeServer::RunServer(
    base::ScopedFD server_write_fd,
    ChildPortHandshake::PortRightType port_right_type) {
  DCHECK_EQ(port_, kMachPortNull);
  DCHECK(!checked_in_);
  DCHECK(server_write_fd.is_valid());

  // Initialize the token and share it with the client via the pipe.
  token_ = base::RandUint64();
  if (!LoggingWriteFile(server_write_fd.get(), &token_, sizeof(token_))) {
    LOG(WARNING) << "no client check-in";
    return MACH_PORT_NULL;
  }

  // Create a unique name for the bootstrap service mapping. Make it unguessable
  // to prevent outsiders from grabbing the name first, which would cause
  // bootstrap_check_in() to fail.
  uint64_t thread_id;
  errno = pthread_threadid_np(pthread_self(), &thread_id);
  PCHECK(errno == 0) << "pthread_threadid_np";
  std::string service_name = base::StringPrintf(
      "org.chromium.crashpad.child_port_handshake.%d.%llu.%s",
      getpid(),
      thread_id,
      RandomString().c_str());

  // Check the new service in with the bootstrap server, obtaining a receive
  // right for it.
  base::mac::ScopedMachReceiveRight server_port(BootstrapCheckIn(service_name));
  CHECK(server_port.is_valid());

  // Share the service name with the client via the pipe.
  uint32_t service_name_length = service_name.size();
  if (!LoggingWriteFile(server_write_fd.get(),
                        &service_name_length,
                        sizeof(service_name_length))) {
    LOG(WARNING) << "no client check-in";
    return MACH_PORT_NULL;
  }

  if (!LoggingWriteFile(
          server_write_fd.get(), service_name.c_str(), service_name_length)) {
    LOG(WARNING) << "no client check-in";
    return MACH_PORT_NULL;
  }

  // A kqueue cannot monitor a raw Mach receive right with EVFILT_MACHPORT. It
  // requires a port set. Create a new port set and add the receive right to it.
  base::mac::ScopedMachPortSet server_port_set(
      NewMachPort(MACH_PORT_RIGHT_PORT_SET));
  CHECK(server_port_set.is_valid());

  kern_return_t kr = mach_port_insert_member(
      mach_task_self(), server_port.get(), server_port_set.get());
  MACH_CHECK(kr == KERN_SUCCESS, kr) << "mach_port_insert_member";

  // Set up a kqueue to monitor both the server’s receive right and the write
  // side of the pipe. Messages from the client will be received via the receive
  // right, and the pipe will show EOF if the client closes its read side
  // prematurely.
  base::ScopedFD kq(kqueue());
  PCHECK(kq != -1) << "kqueue";

  struct kevent changelist[2];
  EV_SET(&changelist[0],
         server_port_set.get(),
         EVFILT_MACHPORT,
         EV_ADD | EV_CLEAR,
         0,
         0,
         nullptr);
  EV_SET(&changelist[1],
         server_write_fd.get(),
         EVFILT_WRITE,
         EV_ADD | EV_CLEAR,
         0,
         0,
         nullptr);
  int rv = HANDLE_EINTR(
      kevent(kq.get(), changelist, arraysize(changelist), nullptr, 0, nullptr));
  PCHECK(rv != -1) << "kevent";

  ChildPortServer child_port_server(this);

  bool blocking = true;
  DCHECK(!checked_in_);
  while (!checked_in_) {
    DCHECK_EQ(port_, kMachPortNull);

    // Get a kevent from the kqueue. Block while waiting for an event unless the
    // write pipe has arrived at EOF, in which case the kevent() should be
    // nonblocking. Although the client sends its check-in message before
    // closing the read side of the pipe, this organization allows the events to
    // be delivered out of order and the check-in message will still be
    // processed.
    struct kevent event;
    constexpr timespec nonblocking_timeout = {};
    const timespec* timeout = blocking ? nullptr : &nonblocking_timeout;
    rv = HANDLE_EINTR(kevent(kq.get(), nullptr, 0, &event, 1, timeout));
    PCHECK(rv != -1) << "kevent";

    if (rv == 0) {
      // Non-blocking kevent() with no events to return.
      DCHECK(!blocking);
      LOG(WARNING) << "no client check-in";
      return MACH_PORT_NULL;
    }

    DCHECK_EQ(rv, 1);

    if (event.flags & EV_ERROR) {
      // kevent() may have put its error here.
      errno = event.data;
      PLOG(FATAL) << "kevent";
    }

    switch (event.filter) {
      case EVFILT_MACHPORT: {
        // There’s something to receive on the port set.
        DCHECK_EQ(event.ident, server_port_set.get());

        // Run the message server in an inner loop instead of using
        // MachMessageServer::kPersistent. This allows the loop to exit as soon
        // as child_port_ is set, even if other messages are queued. This needs
        // to drain all messages, because the use of edge triggering (EV_CLEAR)
        // means that if more than one message is in the queue when kevent()
        // returns, no more notifications will be generated.
        while (!checked_in_) {
          // If a proper message is received from child_port_check_in(),
          // this will call HandleChildPortCheckIn().
          mach_msg_return_t mr =
              MachMessageServer::Run(&child_port_server,
                                     server_port_set.get(),
                                     MACH_MSG_OPTION_NONE,
                                     MachMessageServer::kOneShot,
                                     MachMessageServer::kReceiveLargeIgnore,
                                     kMachMessageTimeoutNonblocking);
          if (mr == MACH_RCV_TIMED_OUT) {
            break;
          } else if (mr != MACH_MSG_SUCCESS) {
            MACH_LOG(ERROR, mr) << "MachMessageServer::Run";
            return MACH_PORT_NULL;
          }
        }
        break;
      }

      case EVFILT_WRITE:
        // The write pipe is ready to be written to, or it’s at EOF. The former
        // case is uninteresting, but a notification for this may be presented
        // because the write pipe will be ready to be written to, at the latest,
        // when the client reads its messages from the read side of the same
        // pipe. Ignore that case. Multiple notifications for that situation
        // will not be generated because edge triggering (EV_CLEAR) is used
        // above.
        DCHECK_EQ(implicit_cast<int>(event.ident), server_write_fd.get());
        if (event.flags & EV_EOF) {
          // There are no readers attached to the write pipe. The client has
          // closed its side of the pipe. There can be one last shot at
          // receiving messages, in case the check-in message is delivered
          // out of order, after the EOF notification.
          blocking = false;
        }
        break;

      default:
        NOTREACHED();
        break;
    }
  }

  if (port_ == MACH_PORT_NULL) {
    return MACH_PORT_NULL;
  }

  bool mismatch = false;
  switch (port_right_type) {
    case ChildPortHandshake::PortRightType::kReceiveRight:
      if (right_type_ != MACH_MSG_TYPE_PORT_RECEIVE) {
        LOG(ERROR) << "expected receive right, observed " << right_type_;
        mismatch = true;
      }
      break;
    case ChildPortHandshake::PortRightType::kSendRight:
      if (right_type_ != MACH_MSG_TYPE_PORT_SEND &&
          right_type_ != MACH_MSG_TYPE_PORT_SEND_ONCE) {
        LOG(ERROR) << "expected send or send-once right, observed "
                   << right_type_;
        mismatch = true;
      }
      break;
  }

  if (mismatch) {
    MachMessageDestroyReceivedPort(port_, right_type_);
    port_ = MACH_PORT_NULL;
    return MACH_PORT_NULL;
  }

  mach_port_t port = MACH_PORT_NULL;
  std::swap(port_, port);
  return port;
}

kern_return_t ChildPortHandshakeServer::HandleChildPortCheckIn(
    child_port_server_t server,
    const child_port_token_t token,
    mach_port_t port,
    mach_msg_type_name_t right_type,
    const mach_msg_trailer_t* trailer,
    bool* destroy_request) {
  DCHECK_EQ(port_, kMachPortNull);
  DCHECK(!checked_in_);

  if (token != token_) {
    // If the token’s not correct, someone’s attempting to spoof the legitimate
    // client.
    LOG(WARNING) << "ignoring incorrect token";
    *destroy_request = true;
  } else {
    checked_in_ = true;

    if (right_type != MACH_MSG_TYPE_PORT_RECEIVE &&
        right_type != MACH_MSG_TYPE_PORT_SEND &&
        right_type != MACH_MSG_TYPE_PORT_SEND_ONCE) {
      // The message needs to carry a receive, send, or send-once right.
      LOG(ERROR) << "invalid right type " << right_type;
      *destroy_request = true;
    } else {
      // Communicate the child port and right type back to the RunServer().
      // *destroy_request is left at false, because RunServer() needs the right
      // to remain intact. It gives ownership of the right to its caller.
      port_ = port;
      right_type_ = right_type;
    }
  }

  // This is a MIG simpleroutine, there is no reply message.
  return MIG_NO_REPLY;
}

}  // namespace

ChildPortHandshake::ChildPortHandshake()
    : client_read_fd_(),
      server_write_fd_() {
  // Use socketpair() instead of pipe(). There is no way to suppress SIGPIPE on
  // pipes in Mac OS X 10.6, because the F_SETNOSIGPIPE fcntl() command was not
  // introduced until 10.7.
  int pipe_fds[2];
  PCHECK(socketpair(AF_UNIX, SOCK_STREAM, PF_UNSPEC, pipe_fds) == 0)
      << "socketpair";

  client_read_fd_.reset(pipe_fds[0]);
  server_write_fd_.reset(pipe_fds[1]);

  // Simulate pipe() semantics by shutting down the “wrong” sides of the socket.
  PCHECK(shutdown(server_write_fd_.get(), SHUT_RD) == 0) << "shutdown SHUT_RD";
  PCHECK(shutdown(client_read_fd_.get(), SHUT_WR) == 0) << "shutdown SHUT_WR";

  // SIGPIPE is undesirable when writing to this pipe. Allow broken-pipe writes
  // to fail with EPIPE instead.
  constexpr int value = 1;
  PCHECK(setsockopt(server_write_fd_.get(),
                    SOL_SOCKET,
                    SO_NOSIGPIPE,
                    &value,
                    sizeof(value)) == 0) << "setsockopt";
}

ChildPortHandshake::~ChildPortHandshake() {
}

base::ScopedFD ChildPortHandshake::ClientReadFD() {
  DCHECK(client_read_fd_.is_valid());
  return std::move(client_read_fd_);
}

base::ScopedFD ChildPortHandshake::ServerWriteFD() {
  DCHECK(server_write_fd_.is_valid());
  return std::move(server_write_fd_);
}

mach_port_t ChildPortHandshake::RunServer(PortRightType port_right_type) {
  client_read_fd_.reset();
  return RunServerForFD(std::move(server_write_fd_), port_right_type);
}

bool ChildPortHandshake::RunClient(mach_port_t port,
                                   mach_msg_type_name_t right_type) {
  server_write_fd_.reset();
  return RunClientForFD(std::move(client_read_fd_), port, right_type);
}

// static
mach_port_t ChildPortHandshake::RunServerForFD(base::ScopedFD server_write_fd,
                                               PortRightType port_right_type) {
  ChildPortHandshakeServer server;
  return server.RunServer(std::move(server_write_fd), port_right_type);
}

// static
bool ChildPortHandshake::RunClientForFD(base::ScopedFD client_read_fd,
                                        mach_port_t port,
                                        mach_msg_type_name_t right_type) {
  DCHECK(client_read_fd.is_valid());

  // Read the token and the service name from the read side of the pipe.
  child_port_token_t token;
  std::string service_name;
  if (!RunClientInternal_ReadPipe(
          client_read_fd.get(), &token, &service_name)) {
    return false;
  }

  // Look up the server and check in with it by providing the token and port.
  return RunClientInternal_SendCheckIn(service_name, token, port, right_type);
}

// static
bool ChildPortHandshake::RunClientInternal_ReadPipe(int client_read_fd,
                                                    child_port_token_t* token,
                                                    std::string* service_name) {
  // Read the token from the pipe.
  if (!LoggingReadFileExactly(client_read_fd, token, sizeof(*token))) {
    return false;
  }

  // Read the service name from the pipe.
  uint32_t service_name_length;
  if (!LoggingReadFileExactly(
          client_read_fd, &service_name_length, sizeof(service_name_length))) {
    return false;
  }

  service_name->resize(service_name_length);
  if (!service_name->empty() &&
      !LoggingReadFileExactly(
          client_read_fd, &(*service_name)[0], service_name_length)) {
    return false;
  }

  return true;
}

// static
bool ChildPortHandshake::RunClientInternal_SendCheckIn(
    const std::string& service_name,
    child_port_token_t token,
    mach_port_t port,
    mach_msg_type_name_t right_type) {
  // Get a send right to the server by looking up the service with the bootstrap
  // server by name.
  base::mac::ScopedMachSendRight server_port(BootstrapLookUp(service_name));
  if (server_port == kMachPortNull) {
    return false;
  }

  // Check in with the server.
  kern_return_t kr = child_port_check_in(
      server_port.get(), token, port, right_type);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(ERROR, kr) << "child_port_check_in";
    return false;
  }

  return true;
}

}  // namespace crashpad
