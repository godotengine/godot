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

#include "test/multiprocess.h"

#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "base/auto_reset.h"
#include "base/files/scoped_file.h"
#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/errors.h"
#include "util/misc/scoped_forbid_return.h"
#include "util/posix/signals.h"

#if defined(OS_MACOSX)
#include "test/mac/exception_swallower.h"
#endif

namespace crashpad {
namespace test {

namespace internal {

struct MultiprocessInfo {
  MultiprocessInfo()
      : pipe_c2p_read(-1),
        pipe_c2p_write(-1),
        pipe_p2c_read(-1),
        pipe_p2c_write(-1),
        child_pid(0) {}

  base::ScopedFD pipe_c2p_read;  // child to parent
  base::ScopedFD pipe_c2p_write;  // child to parent
  base::ScopedFD pipe_p2c_read;  // parent to child
  base::ScopedFD pipe_p2c_write;  // parent to child
  pid_t child_pid;  // valid only in parent
};

}  // namespace internal

Multiprocess::Multiprocess()
    : info_(nullptr),
      code_(EXIT_SUCCESS),
      reason_(kTerminationNormal) {
}

void Multiprocess::Run() {
  ASSERT_EQ(info_, nullptr);
  std::unique_ptr<internal::MultiprocessInfo> info(
      new internal::MultiprocessInfo);
  base::AutoReset<internal::MultiprocessInfo*> reset_info(&info_, info.get());

  ASSERT_NO_FATAL_FAILURE(PreFork());

#if defined(OS_MACOSX)
  // If the child is expected to crash, set up an exception swallower to swallow
  // the exception instead of allowing it to be seen by the system’s crash
  // reporter.
  std::unique_ptr<ExceptionSwallower> exception_swallower;
  if (reason_ == kTerminationSignal && Signals::IsCrashSignal(code_)) {
    exception_swallower.reset(new ExceptionSwallower());
  }
#endif  // OS_MACOSX

  pid_t pid = fork();
  ASSERT_GE(pid, 0) << ErrnoMessage("fork");

  if (pid > 0) {
    info_->child_pid = pid;

    RunParent();

    // Waiting for the child happens here instead of in RunParent() because even
    // if RunParent() returns early due to a gtest fatal assertion failure, the
    // child should still be reaped.

    // This will make the parent hang up on the child as much as would be
    // visible from the child’s perspective. The child’s side of the pipe will
    // be broken, the child’s remote port will become a dead name, and an
    // attempt by the child to look up the service will fail. If this weren’t
    // done, the child might hang while waiting for a parent that has already
    // triggered a fatal assertion failure to do something.
    info.reset();
    info_ = nullptr;

    int status;
    pid_t wait_pid = HANDLE_EINTR(waitpid(pid, &status, 0));
    ASSERT_EQ(wait_pid, pid) << ErrnoMessage("waitpid");

    TerminationReason reason;
    int code;
    std::string message;
    if (WIFEXITED(status)) {
      reason = kTerminationNormal;
      code = WEXITSTATUS(status);
      message = base::StringPrintf("Child exited with code %d", code);
    } else if (WIFSIGNALED(status)) {
      reason = kTerminationSignal;
      code = WTERMSIG(status);
      message =
          base::StringPrintf("Child terminated by signal %d (%s)%s",
                             code,
                             strsignal(code),
                             WCOREDUMP(status) ? " (core dumped)" : "");
    } else {
      FAIL() << base::StringPrintf("Unknown termination reason 0x%x", status);
    }

    if (reason_ == kTerminationNormal) {
      message += base::StringPrintf(", expected exit with code %d", code_);
    } else if (reason_ == kTerminationSignal) {
      message += base::StringPrintf(", expected termination by signal %d (%s)",
                                    code_,
                                    strsignal(code_));
    }

    if (reason != reason_ || code != code_) {
      ADD_FAILURE() << message;
    }
  } else {
#if defined(OS_MACOSX)
    if (exception_swallower.get()) {
      ExceptionSwallower::SwallowExceptions();
    }
#elif defined(OS_LINUX) || defined(OS_ANDROID)
    if (reason_ == kTerminationSignal && Signals::IsCrashSignal(code_)) {
      Signals::InstallDefaultHandler(code_);
    }
#endif  // OS_MACOSX

    RunChild();
  }
}

void Multiprocess::SetExpectedChildTermination(TerminationReason reason,
                                               int code) {
  EXPECT_EQ(info_, nullptr)
      << "SetExpectedChildTermination() must be called before Run()";
  reason_ = reason;
  code_ = code;
}

void Multiprocess::SetExpectedChildTerminationBuiltinTrap() {
#if defined(ARCH_CPU_ARM64) || defined(ARCH_CPU_MIPS_FAMILY)
  SetExpectedChildTermination(kTerminationSignal, SIGTRAP);
#else
  SetExpectedChildTermination(kTerminationSignal, SIGILL);
#endif
}

Multiprocess::~Multiprocess() {
}

void Multiprocess::PreFork() {
  int pipe_fds_c2p[2];
  int rv = pipe(pipe_fds_c2p);
  ASSERT_EQ(rv, 0) << ErrnoMessage("pipe");

  info_->pipe_c2p_read.reset(pipe_fds_c2p[0]);
  info_->pipe_c2p_write.reset(pipe_fds_c2p[1]);

  int pipe_fds_p2c[2];
  rv = pipe(pipe_fds_p2c);
  ASSERT_EQ(rv, 0) << ErrnoMessage("pipe");

  info_->pipe_p2c_read.reset(pipe_fds_p2c[0]);
  info_->pipe_p2c_write.reset(pipe_fds_p2c[1]);
}

pid_t Multiprocess::ChildPID() const {
  EXPECT_NE(info_->child_pid, 0);
  return info_->child_pid;
}

FileHandle Multiprocess::ReadPipeHandle() const {
  int fd = info_->child_pid ? info_->pipe_c2p_read.get()
                            : info_->pipe_p2c_read.get();
  CHECK_NE(fd, -1);
  return fd;
}

FileHandle Multiprocess::WritePipeHandle() const {
  int fd = info_->child_pid ? info_->pipe_p2c_write.get()
                            : info_->pipe_c2p_write.get();
  CHECK_NE(fd, -1);
  return fd;
}

void Multiprocess::CloseReadPipe() {
  if (info_->child_pid) {
    info_->pipe_c2p_read.reset();
  } else {
    info_->pipe_p2c_read.reset();
  }
}

void Multiprocess::CloseWritePipe() {
  if (info_->child_pid) {
    info_->pipe_p2c_write.reset();
  } else {
    info_->pipe_c2p_write.reset();
  }
}

void Multiprocess::RunParent() {
  // The parent uses the read end of c2p and the write end of p2c.
  info_->pipe_c2p_write.reset();
  info_->pipe_p2c_read.reset();

  MultiprocessParent();

  info_->pipe_c2p_read.reset();
  info_->pipe_p2c_write.reset();
}

void Multiprocess::RunChild() {
  ScopedForbidReturn forbid_return;

  // The child uses the write end of c2p and the read end of p2c.
  info_->pipe_c2p_read.reset();
  info_->pipe_p2c_write.reset();

  MultiprocessChild();

  info_->pipe_c2p_write.reset();
  info_->pipe_p2c_read.reset();

  if (testing::Test::HasFailure()) {
    // Trigger the ScopedForbidReturn destructor.
    return;
  }

  // In a forked child, exit() is unsafe. Use _exit() instead.
  _exit(EXIT_SUCCESS);
}

}  // namespace test
}  // namespace crashpad
