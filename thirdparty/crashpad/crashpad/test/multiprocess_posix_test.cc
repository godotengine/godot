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
#include <unistd.h>

#include "base/macros.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {
namespace {

class TestMultiprocess final : public Multiprocess {
 public:
  TestMultiprocess() : Multiprocess() {}

  ~TestMultiprocess() {}

 private:
  // Multiprocess:

  void MultiprocessParent() override {
    FileHandle read_handle = ReadPipeHandle();
    char c;
    CheckedReadFileExactly(read_handle, &c, 1);
    EXPECT_EQ(c, 'M');

    pid_t pid;
    CheckedReadFileExactly(read_handle, &pid, sizeof(pid));
    EXPECT_EQ(ChildPID(), pid);

    c = 'm';
    CheckedWriteFile(WritePipeHandle(), &c, 1);

    // The child will close its end of the pipe and exit. Make sure that the
    // parent sees EOF.
    CheckedReadFileAtEOF(read_handle);
  }

  void MultiprocessChild() override {
    FileHandle write_handle = WritePipeHandle();

    char c = 'M';
    CheckedWriteFile(write_handle, &c, 1);

    pid_t pid = getpid();
    CheckedWriteFile(write_handle, &pid, sizeof(pid));

    CheckedReadFileExactly(ReadPipeHandle(), &c, 1);
    EXPECT_EQ(c, 'm');
  }

  DISALLOW_COPY_AND_ASSIGN(TestMultiprocess);
};

TEST(Multiprocess, Multiprocess) {
  TestMultiprocess multiprocess;
  multiprocess.Run();
}

class TestMultiprocessUnclean final : public Multiprocess {
 public:
  enum TerminationType {
    kExitSuccess = 0,
    kExitFailure,
    kExit2,
    kAbort,
  };

  explicit TestMultiprocessUnclean(TerminationType type)
      : Multiprocess(),
        type_(type) {
    if (type_ == kAbort) {
      SetExpectedChildTermination(kTerminationSignal, SIGABRT);
    } else {
      SetExpectedChildTermination(kTerminationNormal, ExitCode());
    }
  }

  ~TestMultiprocessUnclean() {}

 private:
  int ExitCode() const {
    return type_;
  }

  // Multiprocess:

  void MultiprocessParent() override {
  }

  void MultiprocessChild() override {
    if (type_ == kAbort) {
      abort();
    } else {
      _exit(ExitCode());
    }
  }

  TerminationType type_;

  DISALLOW_COPY_AND_ASSIGN(TestMultiprocessUnclean);
};

TEST(Multiprocess, SuccessfulExit) {
  TestMultiprocessUnclean multiprocess(TestMultiprocessUnclean::kExitSuccess);
  multiprocess.Run();
}

TEST(Multiprocess, UnsuccessfulExit) {
  TestMultiprocessUnclean multiprocess(TestMultiprocessUnclean::kExitFailure);
  multiprocess.Run();
}

TEST(Multiprocess, Exit2) {
  TestMultiprocessUnclean multiprocess(TestMultiprocessUnclean::kExit2);
  multiprocess.Run();
}

TEST(Multiprocess, AbortSignal) {
  TestMultiprocessUnclean multiprocess(TestMultiprocessUnclean::kAbort);
  multiprocess.Run();
}

class TestMultiprocessClosePipe final : public Multiprocess {
 public:
  enum WhoCloses {
    kParentCloses = 0,
    kChildCloses,
  };
  enum WhatCloses {
    kReadCloses = 0,
    kWriteCloses,
    kReadAndWriteClose,
  };

  TestMultiprocessClosePipe(WhoCloses who_closes, WhatCloses what_closes)
      : Multiprocess(),
        who_closes_(who_closes),
        what_closes_(what_closes) {
  }

  ~TestMultiprocessClosePipe() {}

 private:
  void VerifyInitial() {
    ASSERT_NE(ReadPipeHandle(), -1);
    ASSERT_NE(WritePipeHandle(), -1);
  }

  // Verifies that the partner process did what it was supposed to do. This must
  // only be called when who_closes_ names the partner process, not this
  // process.
  //
  // If the partner was supposed to close its write pipe, the read pipe will be
  // checked to ensure that it shows end-of-file.
  //
  // If the partner was supposed to close its read pipe, the write pipe will be
  // checked to ensure that a checked write causes death. This can only be done
  // if the partner also provides some type of signal when it has closed its
  // read pipe, which is done in the form of it closing its write pipe, causing
  // the read pipe in this process to show end-of-file.
  void VerifyPartner() {
    if (what_closes_ == kWriteCloses) {
      CheckedReadFileAtEOF(ReadPipeHandle());
    } else if (what_closes_ == kReadAndWriteClose) {
      CheckedReadFileAtEOF(ReadPipeHandle());
      char c = '\0';

      // This will raise SIGPIPE. If fatal (the normal case), that will cause
      // process termination. If SIGPIPE is being handled somewhere, the write
      // will still fail and set errno to EPIPE, and CheckedWriteFile() will
      // abort execution. Regardless of how SIGPIPE is handled, the process will
      // be terminated. Because the actual termination mechanism is not known,
      // no regex can be specified.
      EXPECT_DEATH_CHECK(CheckedWriteFile(WritePipeHandle(), &c, 1), "");
    }
  }

  void Close() {
    switch (what_closes_) {
      case kReadCloses:
        CloseReadPipe();
        EXPECT_NE(WritePipeHandle(), -1);
        EXPECT_DEATH_CHECK(ReadPipeHandle(), "fd");
        break;
      case kWriteCloses:
        CloseWritePipe();
        EXPECT_NE(ReadPipeHandle(), -1);
        EXPECT_DEATH_CHECK(WritePipeHandle(), "fd");
        break;
      case kReadAndWriteClose:
        CloseReadPipe();
        CloseWritePipe();
        EXPECT_DEATH_CHECK(ReadPipeHandle(), "fd");
        EXPECT_DEATH_CHECK(WritePipeHandle(), "fd");
        break;
    }
  }

  // Multiprocess:

  void MultiprocessParent() override {
    ASSERT_NO_FATAL_FAILURE(VerifyInitial());

    if (who_closes_ == kParentCloses) {
      Close();
    } else {
      VerifyPartner();
    }
  }

  void MultiprocessChild() override {
    ASSERT_NO_FATAL_FAILURE(VerifyInitial());

    if (who_closes_ == kChildCloses) {
      Close();
    } else {
      VerifyPartner();
    }
  }

  WhoCloses who_closes_;
  WhatCloses what_closes_;

  DISALLOW_COPY_AND_ASSIGN(TestMultiprocessClosePipe);
};

TEST(MultiprocessDeathTest, ParentClosesReadPipe) {
  TestMultiprocessClosePipe multiprocess(
      TestMultiprocessClosePipe::kParentCloses,
      TestMultiprocessClosePipe::kReadCloses);
  multiprocess.Run();
}

TEST(MultiprocessDeathTest, ParentClosesWritePipe) {
  TestMultiprocessClosePipe multiprocess(
      TestMultiprocessClosePipe::kParentCloses,
      TestMultiprocessClosePipe::kWriteCloses);
  multiprocess.Run();
}

TEST(MultiprocessDeathTest, ParentClosesReadAndWritePipe) {
  TestMultiprocessClosePipe multiprocess(
      TestMultiprocessClosePipe::kParentCloses,
      TestMultiprocessClosePipe::kReadAndWriteClose);
  multiprocess.Run();
}

TEST(MultiprocessDeathTest, ChildClosesReadPipe) {
  TestMultiprocessClosePipe multiprocess(
      TestMultiprocessClosePipe::kChildCloses,
      TestMultiprocessClosePipe::kReadCloses);
  multiprocess.Run();
}

TEST(MultiprocessDeathTest, ChildClosesWritePipe) {
  TestMultiprocessClosePipe multiprocess(
      TestMultiprocessClosePipe::kChildCloses,
      TestMultiprocessClosePipe::kWriteCloses);
  multiprocess.Run();
}

TEST(MultiprocessDeathTest, ChildClosesReadAndWritePipe) {
  TestMultiprocessClosePipe multiprocess(
      TestMultiprocessClosePipe::kChildCloses,
      TestMultiprocessClosePipe::kReadAndWriteClose);
  multiprocess.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
