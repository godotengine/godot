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

#ifndef CRASHPAD_TEST_MULTIPROCESS_EXEC_H_
#define CRASHPAD_TEST_MULTIPROCESS_EXEC_H_

#include <string>
#include <vector>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "build/build_config.h"
#include "test/multiprocess.h"
#include "test/process_type.h"

//! \file

namespace crashpad {
namespace test {

namespace internal {

//! \brief Command line argument used to indicate that a child test function
//!     should be run.
constexpr char kChildTestFunction[] = "--child-test-function=";


//! \brief Helper class used by CRASHPAD_CHILD_TEST_MAIN() to insert a child
//!     function into the global mapping.
class AppendMultiprocessTest {
 public:
  AppendMultiprocessTest(const std::string& test_name,
                         int (*main_function_pointer)());
};

//! \brief Used to run a child test function by name, registered by
//!     CRASHPAD_CHILD_TEST_MAIN().
//!
//! \return The exit code of the child process after running the function named
//!     by \a test_name. Aborts with a CHECK() if \a test_name wasn't
//!     registered.
int CheckedInvokeMultiprocessChild(const std::string& test_name);

}  // namespace internal

//! \brief Registers a function that can be invoked as a child process by
//!     MultiprocessExec.
//!
//! Used as:
//!
//! \code
//! CRASHPAD_CHILD_TEST_MAIN(MyChildTestBody) {
//!    ... child body ...
//! }
//! \endcode
//!
//! In the main (parent) test body, this function can be run in a child process
//! via MultiprocessExec::SetChildTestMainFunction().
#define CRASHPAD_CHILD_TEST_MAIN(test_main)                       \
  int test_main();                                                \
  namespace {                                                     \
  ::crashpad::test::internal::AppendMultiprocessTest              \
      AddMultiprocessTest##_##test_main(#test_main, (test_main)); \
  } /* namespace */                                               \
  int test_main()

//! \brief Manages an `exec()`-based multiprocess test.
//!
//! These tests are based on `fork()` and `exec()`. The parent process is able
//! to communicate with the child in the same manner as a base-class
//! Multiprocess parent. The read and write pipes appear in the child process on
//! stdin and stdout, respectively.
//!
//! Subclasses are expected to implement the parent in the same was as a
//! base-class Multiprocess parent. The child must be implemented in an
//! executable to be set by SetChildCommand().
class MultiprocessExec : public Multiprocess {
 public:
  MultiprocessExec();

  //! \brief Sets the command to `exec()` in the child.
  //!
  //! This method must be called before the test can be Run().
  //!
  //! This method is useful when a custom executable is required for the child
  //! binary, however, SetChildTestMainFunction() should generally be preferred.
  //!
  //! \param[in] command The executable’s pathname.
  //! \param[in] arguments The command-line arguments to pass to the child
  //!     process in its `argv[]` vector. This vector must begin at `argv[1]`,
  //!     as \a command is implicitly used as `argv[0]`. This argument may be
  //!     `nullptr` if no command-line arguments are to be passed.
  //!
  //! \sa SetChildTestMainFunction
  void SetChildCommand(const base::FilePath& command,
                       const std::vector<std::string>* arguments);

  //! \brief Calls SetChildCommand() to run a child test main function
  //!     registered with CRASHPAD_CHILD_TEST_MAIN().
  //!
  //! This uses the same launch mechanism as SetChildCommand(), but coordinates
  //! with test/gtest_main.cc to allow for simple registration of a child
  //! processes' entry point via the helper macro, rather than needing to
  //! create a separate build target.
  //!
  //! \param[in] function_name The name of the function as passed to
  //!     CRASHPAD_CHILD_TEST_MAIN().
  void SetChildTestMainFunction(const std::string& function_name);

  //! \brief Returns a ProcessType representing the child process.
  //!
  //! This method is only valid during the body of MultiprocessParent().
  //!
  //! \return A platform-specific type representing the child process.
  ProcessType ChildProcess();

 protected:
  ~MultiprocessExec();

  // Multiprocess:
  void PreFork() override;

 private:
  // Multiprocess:
  void MultiprocessChild() override;

  base::FilePath command_;
  std::vector<std::string> arguments_;
#if defined(OS_POSIX)
  std::vector<const char*> argv_;
#elif defined(OS_WIN)
  std::wstring command_line_;
#endif  // OS_POSIX

  DISALLOW_COPY_AND_ASSIGN(MultiprocessExec);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_MULTIPROCESS_EXEC_H_
