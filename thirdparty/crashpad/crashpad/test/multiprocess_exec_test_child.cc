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

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <sys/types.h>

#include <algorithm>

#include "base/logging.h"
#include "build/build_config.h"

#if defined(OS_POSIX)
#if !defined(OS_FUCHSIA)
#include <sys/resource.h>
#endif  // !OS_FUCHSIA
#include <unistd.h>
#elif defined(OS_WIN)
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#if defined(OS_POSIX)

#if defined(OS_FUCHSIA)
  // getrlimit() is not implemented on Fuchsia. By construction, the child only
  // receieves specific fds that it's given, but check low values as mild
  // verification.
  int last_fd = 1024;
#else
  rlimit rlimit_nofile;
  if (getrlimit(RLIMIT_NOFILE, &rlimit_nofile) != 0) {
    LOG(FATAL) << "getrlimit";
  }
  int last_fd = static_cast<int>(rlimit_nofile.rlim_cur);
#endif  // OS_FUCHSIA

  // Make sure that there’s nothing open at any FD higher than 3. All FDs other
  // than stdin, stdout, and stderr should have been closed prior to or at
  // exec().
  for (int fd = STDERR_FILENO + 1; fd < last_fd; ++fd) {
    if (close(fd) == 0 || errno != EBADF) {
      LOG(FATAL) << "close";
    }
  }

  // Read a byte from stdin, expecting it to be a specific value.
  char c;
  ssize_t rv = read(STDIN_FILENO, &c, 1);
  if (rv != 1 || c != 'z') {
    LOG(FATAL) << "read";
  }

  // Write a byte to stdout.
  c = 'Z';
  rv = write(STDOUT_FILENO, &c, 1);
  if (rv != 1) {
    LOG(FATAL) << "write";
  }
#elif defined(OS_WIN)
  // TODO(scottmg): Verify that only the handles we expect to be open, are.

  // Read a byte from stdin, expecting it to be a specific value.
  char c;
  DWORD bytes_read;
  HANDLE stdin_handle = GetStdHandle(STD_INPUT_HANDLE);
  if (!ReadFile(stdin_handle, &c, 1, &bytes_read, nullptr) ||
      bytes_read != 1 || c != 'z') {
    LOG(FATAL) << "ReadFile";
  }

  // Write a byte to stdout.
  c = 'Z';
  DWORD bytes_written;
  if (!WriteFile(
          GetStdHandle(STD_OUTPUT_HANDLE), &c, 1, &bytes_written, nullptr) ||
      bytes_written != 1) {
    LOG(FATAL) << "WriteFile";
  }
#endif  // OS_POSIX

  return 0;
}
