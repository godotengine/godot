// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_TEST_TEST_PATHS_H_
#define CRASHPAD_TEST_TEST_PATHS_H_

#include "base/files/file_path.h"
#include "base/macros.h"
#include "build/build_config.h"

namespace crashpad {
namespace test {

//! \brief Functions to obtain paths from within tests.
class TestPaths {
 public:
  //! \brief The type of file requested of BuildArtifact().
  //!
  //! This is used to establish the file extension used by the returned path.
  enum class FileType {
    //! \brief No file extension is requested.
    kNone = 0,

    //! \brief `.exe` will be used on Windows, and no file extension will be
    //!     used on other platforms.
    kExecutable,

    //! \brief `.dll` will be used on Windows, and `.so` will be used on other
    //!     platforms.
    kLoadableModule,

    //! \brief `.pem` used for all platforms.
    kCertificate,
  };

  //! \brief The architecture of the file requested of BuildArtifact().
  enum class Architecture {
    //! \brief The default architecture is requested. This is usually the same
    //!     architecture as the running process.
    kDefault = 0,

#if (defined(OS_WIN) && defined(ARCH_CPU_64_BITS)) || DOXYGEN
    //! \brief The 32-bit variant is requested.
    //!
    //! On Windows, when running 64-bit code, the 32-bit variant can be
    //! requested. Before doing so, Has32BitBuildArtifacts() must be called and
    //! must return `true`. Otherwise, execution will be aborted.
    k32Bit,
#endif  // OS_WIN && ARCH_CPU_64_BITS
  };

  //! \brief Returns the pathname of the currently-running test executable.
  //!
  //! On failure, aborts execution.
  static base::FilePath Executable();

  //! \brief Returns the expected basename of the currently-running test
  //!     executable.
  //!
  //! In Crashpad’s standalone build, this returns \a name, with the system’s
  //! extension for executables (`.exe`) appended if appropriate.
  //!
  //! When building in Chromium, \a name is ignored, and the name of the
  //! monolithic test executable (`crashpad_tests`) is returned, with the
  //! system’s extension for executables appended if appropriate.
  //!
  //! Only use this function to determine test expectations.
  //!
  //! Do not use this function to obtain the name of the currently running test
  //! executable, use Executable() instead. Do not use this function to locate
  //! other build artifacts, use BuildArtifact() instead.
  static base::FilePath ExpectedExecutableBasename(
      const base::FilePath::StringType& name);

  //! \brief Returns the pathname of the test data root.
  //!
  //! If the `CRASHPAD_TEST_DATA_ROOT` environment variable is set, its value
  //! will be returned. Otherwise, this function will attempt to locate the test
  //! data root relative to the executable path. If this fails, it will fall
  //! back to returning the current working directory.
  //!
  //! At present, the test data root is normally the root of the Crashpad source
  //! tree, although this may not be the case indefinitely. This function may
  //! only be used to locate test data, not for arbitrary access to source
  //! files.
  static base::FilePath TestDataRoot();

  //! \brief Returns the pathname of a build artifact.
  //!
  //! \param[in] module The name of the Crashpad module associated with the
  //!     artifact, such as `"util"` or `"snapshot"`. \a module must correspond
  //!     to the module of the calling code, or execution will be aborted.
  //! \param[in] artifact The name of the specific artifact.
  //! \param[in] file_type The artifact’s type, used to establish the returned
  //!     path’s extension.
  //! \param[in] architecture The artifact’s architecture.
  //!
  //! \return The computed pathname to the build artifact.
  //!
  //! For example, the following snippet will return a path to
  //! `crashpad_snapshot_test_module.so` or `crashpad_snapshot_test_module.dll`
  //! (depending on platform) in the same directory as the currently running
  //! executable:
  //!
  //! \code
  //!    base::FilePath path = TestPaths::BuildArtifact(
  //!        FILE_PATH_LITERAL("snapshot"),
  //!        FILE_PATH_LITERAL("module"),
  //!        TestPaths::FileType::kLoadableModule);
  //! \endcode
  static base::FilePath BuildArtifact(
      const base::FilePath::StringType& module,
      const base::FilePath::StringType& artifact,
      FileType file_type,
      Architecture architecture = Architecture::kDefault);

#if (defined(OS_WIN) && defined(ARCH_CPU_64_BITS)) || DOXYGEN
  //! \return `true` if 32-bit build artifacts are available.
  //!
  //! Tests that require the use of 32-bit build output should call this
  //! function to determine whether that output is available. This function is
  //! only provided to aid 64-bit test code in locating 32-bit output. Only if
  //! this function indicates that 32-bit output is available, 64-bit test code
  //! may call BuildArtifact() with Architecture::k32Bit to obtain a path to the
  //! 32-bit output.
  //!
  //! 32-bit test code may assume the existence of 32-bit build output, which
  //! can be found its own directory, and located by calling BuildArtifact()
  //! with Architecture::kDefault.
  static bool Has32BitBuildArtifacts();
#endif  // OS_WIN && ARCH_CPU_64_BITS

  DISALLOW_IMPLICIT_CONSTRUCTORS(TestPaths);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_TEST_TEST_PATHS_H_
