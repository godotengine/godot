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

#include "base/logging.h"
#include "base/macros.h"
#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/test_paths.h"
#include "util/file/file_io.h"

namespace crashpad {
namespace test {
namespace {

class TestMultiprocessExec final : public MultiprocessExec {
 public:
  TestMultiprocessExec() : MultiprocessExec() {}

  ~TestMultiprocessExec() {}

 private:
  void MultiprocessParent() override {
    // Use Logging*File() instead of Checked*File() so that the test can fail
    // gracefully with a gtest assertion if the child does not execute properly.

    char c = 'z';
    ASSERT_TRUE(LoggingWriteFile(WritePipeHandle(), &c, 1));

    ASSERT_TRUE(LoggingReadFileExactly(ReadPipeHandle(), &c, 1));
    EXPECT_EQ(c, 'Z');
  }

  DISALLOW_COPY_AND_ASSIGN(TestMultiprocessExec);
};

TEST(MultiprocessExec, MultiprocessExec) {
  TestMultiprocessExec multiprocess_exec;
  base::FilePath child_test_executable = TestPaths::BuildArtifact(
      FILE_PATH_LITERAL("test"),
      FILE_PATH_LITERAL("multiprocess_exec_test_child"),
      TestPaths::FileType::kExecutable);
  multiprocess_exec.SetChildCommand(child_test_executable, nullptr);
  multiprocess_exec.Run();
}


CRASHPAD_CHILD_TEST_MAIN(SimpleMultiprocess) {
  char c;
  CheckedReadFileExactly(StdioFileHandle(StdioStream::kStandardInput), &c, 1);
  LOG_IF(FATAL, c != 'z');

  c = 'Z';
  CheckedWriteFile(StdioFileHandle(StdioStream::kStandardOutput), &c, 1);
  return 0;
}

TEST(MultiprocessExec, MultiprocessExecSimpleChild) {
  TestMultiprocessExec exec;
  exec.SetChildTestMainFunction("SimpleMultiprocess");
  exec.Run();
};


CRASHPAD_CHILD_TEST_MAIN(SimpleMultiprocessReturnsNonZero) {
  return 123;
}

class TestMultiprocessExecEmpty final : public MultiprocessExec {
 public:
  TestMultiprocessExecEmpty() = default;
  ~TestMultiprocessExecEmpty() = default;

 private:
  void MultiprocessParent() override {}

  DISALLOW_COPY_AND_ASSIGN(TestMultiprocessExecEmpty);
};

TEST(MultiprocessExec, MultiprocessExecSimpleChildReturnsNonZero) {
  TestMultiprocessExecEmpty exec;
  exec.SetChildTestMainFunction("SimpleMultiprocessReturnsNonZero");
  exec.SetExpectedChildTermination(
      Multiprocess::TerminationReason::kTerminationNormal, 123);
  exec.Run();
};

#if !defined(OS_WIN)

CRASHPAD_CHILD_TEST_MAIN(BuiltinTrapChild) {
  __builtin_trap();
  return EXIT_SUCCESS;
}

class TestBuiltinTrapTermination final : public MultiprocessExec {
 public:
  TestBuiltinTrapTermination() {
    SetChildTestMainFunction("BuiltinTrapChild");
    SetExpectedChildTerminationBuiltinTrap();
  }

  ~TestBuiltinTrapTermination() = default;

 private:
  void MultiprocessParent() override {}

  DISALLOW_COPY_AND_ASSIGN(TestBuiltinTrapTermination);
};

TEST(MultiprocessExec, BuiltinTrapTermination) {
  TestBuiltinTrapTermination test;
  test.Run();
}

#endif  // !OS_WIN

}  // namespace
}  // namespace test
}  // namespace crashpad
