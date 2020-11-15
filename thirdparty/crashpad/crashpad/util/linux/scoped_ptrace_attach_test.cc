// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "util/linux/scoped_ptrace_attach.h"

#include <errno.h>
#include <sys/ptrace.h>
#include <unistd.h>

#include "gtest/gtest.h"
#include "test/errors.h"
#include "test/multiprocess.h"
#include "util/file/file_io.h"
#include "util/linux/scoped_pr_set_ptracer.h"

namespace crashpad {
namespace test {
namespace {

// If Yama is not enabled, the default ptrace restrictions should be
// sufficient for these tests.
//
// If Yama is enabled, then /proc/sys/kernel/yama/ptrace_scope must be 0
// (YAMA_SCOPE_DISABLED) or 1 (YAMA_SCOPE_RELATIONAL) for these tests to
// succeed. If it is 2 (YAMA_SCOPE_CAPABILITY) then the test requires
// CAP_SYS_PTRACE, and if it is 3 (YAMA_SCOPE_NO_ATTACH), these tests will fail.

class AttachTest : public Multiprocess {
 public:
  AttachTest() : Multiprocess() {}
  ~AttachTest() {}

 protected:
  const long kWord = 42;

 private:
  DISALLOW_COPY_AND_ASSIGN(AttachTest);
};

class AttachToChildTest : public AttachTest {
 public:
  AttachToChildTest() : AttachTest() {}
  ~AttachToChildTest() {}

 private:
  void MultiprocessParent() override {
    // Wait for the child to set the parent as its ptracer.
    char c;
    CheckedReadFileExactly(ReadPipeHandle(), &c, sizeof(c));

    pid_t pid = ChildPID();

    ASSERT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), -1);
    EXPECT_EQ(errno, ESRCH) << ErrnoMessage("ptrace");

    ScopedPtraceAttach attachment;
    ASSERT_EQ(attachment.ResetAttach(pid), true);
    EXPECT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), kWord)
        << ErrnoMessage("ptrace");
    attachment.Reset();

    ASSERT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), -1);
    EXPECT_EQ(errno, ESRCH) << ErrnoMessage("ptrace");
  }

  void MultiprocessChild() override {
    ScopedPrSetPtracer set_ptracer(getppid(), /* may_log= */ true);

    char c = '\0';
    CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));

    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  DISALLOW_COPY_AND_ASSIGN(AttachToChildTest);
};

TEST(ScopedPtraceAttach, AttachChild) {
  AttachToChildTest test;
  test.Run();
}

class AttachToParentResetTest : public AttachTest {
 public:
  AttachToParentResetTest() : AttachTest() {}
  ~AttachToParentResetTest() {}

 private:
  void MultiprocessParent() override {
    ScopedPrSetPtracer set_ptracer(ChildPID(), /* may_log= */ true);
    char c = '\0';
    CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));

    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  void MultiprocessChild() override {
    // Wait for the parent to set the child as its ptracer.
    char c;
    CheckedReadFileExactly(ReadPipeHandle(), &c, sizeof(c));

    pid_t pid = getppid();

    ASSERT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), -1);
    EXPECT_EQ(errno, ESRCH) << ErrnoMessage("ptrace");

    ScopedPtraceAttach attachment;
    ASSERT_EQ(attachment.ResetAttach(pid), true);
    EXPECT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), kWord)
        << ErrnoMessage("ptrace");
    attachment.Reset();

    ASSERT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), -1);
    EXPECT_EQ(errno, ESRCH) << ErrnoMessage("ptrace");
  }

  DISALLOW_COPY_AND_ASSIGN(AttachToParentResetTest);
};

TEST(ScopedPtraceAttach, AttachParentReset) {
  AttachToParentResetTest test;
  test.Run();
}

class AttachToParentDestructorTest : public AttachTest {
 public:
  AttachToParentDestructorTest() : AttachTest() {}
  ~AttachToParentDestructorTest() {}

 private:
  void MultiprocessParent() override {
    ScopedPrSetPtracer set_ptracer(ChildPID(), /* may_log= */ true);
    char c = '\0';
    CheckedWriteFile(WritePipeHandle(), &c, sizeof(c));

    CheckedReadFileAtEOF(ReadPipeHandle());
  }

  void MultiprocessChild() override {
    // Wait for the parent to set the child as its ptracer.
    char c;
    CheckedReadFileExactly(ReadPipeHandle(), &c, sizeof(c));

    pid_t pid = getppid();
    ASSERT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), -1);
    EXPECT_EQ(errno, ESRCH) << ErrnoMessage("ptrace");
    {
      ScopedPtraceAttach attachment;
      ASSERT_EQ(attachment.ResetAttach(pid), true);
      EXPECT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), kWord)
          << ErrnoMessage("ptrace");
    }
    ASSERT_EQ(ptrace(PTRACE_PEEKDATA, pid, &kWord, nullptr), -1);
    EXPECT_EQ(errno, ESRCH) << ErrnoMessage("ptrace");
  }

  DISALLOW_COPY_AND_ASSIGN(AttachToParentDestructorTest);
};

TEST(ScopedPtraceAttach, AttachParentDestructor) {
  AttachToParentDestructorTest test;
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
