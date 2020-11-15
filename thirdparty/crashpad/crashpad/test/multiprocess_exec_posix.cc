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

#include "test/multiprocess_exec.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "base/posix/eintr_wrapper.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "util/misc/scoped_forbid_return.h"
#include "util/posix/close_multiple.h"

#if defined(OS_LINUX)
#include <stdio_ext.h>
#endif

namespace crashpad {
namespace test {

MultiprocessExec::MultiprocessExec()
    : Multiprocess(),
      command_(),
      arguments_(),
      argv_() {
}

void MultiprocessExec::SetChildCommand(
    const base::FilePath& command,
    const std::vector<std::string>* arguments) {
  command_ = command;
  if (arguments) {
    arguments_ = *arguments;
  } else {
    arguments_.clear();
  }
}

MultiprocessExec::~MultiprocessExec() {
}

void MultiprocessExec::PreFork() {
  ASSERT_NO_FATAL_FAILURE(Multiprocess::PreFork());

  ASSERT_FALSE(command_.empty());

  // Build up the argv vector. This is done in PreFork() instead of
  // MultiprocessChild() because although the result is only needed in the child
  // process, building it is a hazardous operation in that process.
  ASSERT_TRUE(argv_.empty());

  argv_.push_back(command_.value().c_str());
  for (const std::string& argument : arguments_) {
    argv_.push_back(argument.c_str());
  }
  argv_.push_back(nullptr);
}

void MultiprocessExec::MultiprocessChild() {
  // Make sure that stdin, stdout, and stderr are FDs 0, 1, and 2, respectively.
  // All FDs above this will be closed.
  static_assert(STDIN_FILENO == 0, "stdin must be fd 0");
  static_assert(STDOUT_FILENO == 1, "stdout must be fd 1");
  static_assert(STDERR_FILENO == 2, "stderr must be fd 2");

  // Move the read pipe to stdin.
  FileHandle read_handle = ReadPipeHandle();
  ASSERT_NE(read_handle, STDIN_FILENO);
  ASSERT_NE(read_handle, STDOUT_FILENO);
  ASSERT_EQ(fileno(stdin), STDIN_FILENO);

  int rv;

#if defined(OS_LINUX)
  __fpurge(stdin);
#else
  rv = fpurge(stdin);
  ASSERT_EQ(rv, 0) << ErrnoMessage("fpurge");
#endif

  rv = HANDLE_EINTR(dup2(read_handle, STDIN_FILENO));
  ASSERT_EQ(rv, STDIN_FILENO) << ErrnoMessage("dup2");

  // Move the write pipe to stdout.
  FileHandle write_handle = WritePipeHandle();
  ASSERT_NE(write_handle, STDIN_FILENO);
  ASSERT_NE(write_handle, STDOUT_FILENO);
  ASSERT_EQ(fileno(stdout), STDOUT_FILENO);

  // Make a copy of the original stdout file descriptor so that in case there’s
  // an execv() failure, the original stdout can be restored so that gtest
  // messages directed to stdout go to the right place. Mark it as
  // close-on-exec, so that the child won’t see it after a successful exec(),
  // but it will still be available in this process after an unsuccessful
  // exec().
  int dup_orig_stdout_fd = dup(STDOUT_FILENO);
  ASSERT_GE(dup_orig_stdout_fd, 0) << ErrnoMessage("dup");

  rv = fcntl(dup_orig_stdout_fd, F_SETFD, FD_CLOEXEC);
  ASSERT_NE(rv, -1) << ErrnoMessage("fcntl");

  rv = HANDLE_EINTR(fflush(stdout));
  ASSERT_EQ(rv, 0) << ErrnoMessage("fflush");

  rv = HANDLE_EINTR(dup2(write_handle, STDOUT_FILENO));
  ASSERT_EQ(rv, STDOUT_FILENO) << ErrnoMessage("dup2");

  CloseMultipleNowOrOnExec(STDERR_FILENO + 1, dup_orig_stdout_fd);

  // Start the new program, replacing this one. execv() has a weird declaration
  // where its argv argument is declared as char* const*. In reality, the
  // implementation behaves as if the argument were const char* const*, and this
  // behavior is required by the standard. See
  // http://pubs.opengroup.org/onlinepubs/9699919799/functions/exec.html
  // (search for “constant”).
  execv(argv_[0], const_cast<char* const*>(&argv_[0]));

  // This should not normally be reached. Getting here means that execv()
  // failed.

  // Be sure not to return until FAIL() is reached.
  ScopedForbidReturn forbid_return;

  // Put the original stdout back. Close the copy of the write pipe FD that’s
  // currently on stdout first, so that in case the dup2() that restores the
  // original stdout fails, stdout isn’t left attached to the pipe when the
  // FAIL() statement executes.
  HANDLE_EINTR(fflush(stdout));
  IGNORE_EINTR(close(STDOUT_FILENO));
  HANDLE_EINTR(dup2(dup_orig_stdout_fd, STDOUT_FILENO));
  IGNORE_EINTR(close(dup_orig_stdout_fd));

  forbid_return.Disarm();
  FAIL() << ErrnoMessage("execv") << ": " << argv_[0];
}

ProcessType MultiprocessExec::ChildProcess() {
  return ChildPID();
}

}  // namespace test
}  // namespace crashpad
