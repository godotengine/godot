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

#include "client/linux/crash_generation/crash_generation_client.h"

#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <algorithm>

#include "common/linux/eintr_wrapper.h"
#include "common/linux/ignore_ret.h"
#include "third_party/lss/linux_syscall_support.h"

namespace google_breakpad {

namespace {

class CrashGenerationClientImpl : public CrashGenerationClient {
 public:
  explicit CrashGenerationClientImpl(int server_fd) : server_fd_(server_fd) {}
  virtual ~CrashGenerationClientImpl() {}

  virtual bool RequestDump(const void* blob, size_t blob_size) {
    int fds[2];
    if (sys_pipe(fds) < 0)
      return false;
    static const unsigned kControlMsgSize = CMSG_SPACE(sizeof(int));

    struct kernel_iovec iov;
    iov.iov_base = const_cast<void*>(blob);
    iov.iov_len = blob_size;

    struct kernel_msghdr msg = { 0 };
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    char cmsg[kControlMsgSize] = "";
    msg.msg_control = cmsg;
    msg.msg_controllen = sizeof(cmsg);

    struct cmsghdr* hdr = CMSG_FIRSTHDR(&msg);
    hdr->cmsg_level = SOL_SOCKET;
    hdr->cmsg_type = SCM_RIGHTS;
    hdr->cmsg_len = CMSG_LEN(sizeof(int));
    int* p = reinterpret_cast<int*>(CMSG_DATA(hdr));
    *p = fds[1];

    ssize_t ret = HANDLE_EINTR(sys_sendmsg(server_fd_, &msg, 0));
    sys_close(fds[1]);
    if (ret < 0) {
      sys_close(fds[0]);
      return false;
    }

    // Wait for an ACK from the server.
    char b;
    IGNORE_RET(HANDLE_EINTR(sys_read(fds[0], &b, 1)));
    sys_close(fds[0]);

    return true;
  }

 private:
  int server_fd_;

  DISALLOW_COPY_AND_ASSIGN(CrashGenerationClientImpl);
};

}  // namespace

// static
CrashGenerationClient* CrashGenerationClient::TryCreate(int server_fd) {
  if (server_fd < 0)
    return NULL;
  return new CrashGenerationClientImpl(server_fd);
}

}  // namespace google_breakpad
