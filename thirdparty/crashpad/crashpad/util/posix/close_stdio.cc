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

#include "util/posix/close_stdio.h"

#include <fcntl.h>
#include <paths.h>
#include <unistd.h>

#include "base/files/scoped_file.h"
#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"

namespace crashpad {

namespace {

void CloseStdioStream(int desired_fd, int oflag) {
  base::ScopedFD fd(
      HANDLE_EINTR(open(_PATH_DEVNULL, oflag | O_NOCTTY | O_CLOEXEC)));
  if (fd == desired_fd) {
    // Weird, but play along.
    ignore_result(fd.release());
  } else {
    PCHECK(fd.get() >= 0) << "open";
    PCHECK(HANDLE_EINTR(dup2(fd.get(), desired_fd)) != -1) << "dup2";
    fd.reset();
  }
}

}  // namespace

void CloseStdinAndStdout() {
  // Open /dev/null for stdin and stdout separately, so that it can be opened
  // with the correct mode each time. This ensures that attempts to write to
  // stdin or read from stdout fail with EBADF.
  CloseStdioStream(STDIN_FILENO, O_RDONLY);
  CloseStdioStream(STDOUT_FILENO, O_WRONLY);
}

}  // namespace crashpad
