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

#include "test/test_paths.h"

#include <stdlib.h>
#include <sys/stat.h>

#include "base/logging.h"
#include "build/build_config.h"
#include "util/misc/paths.h"

namespace crashpad {
namespace test {

namespace {

bool IsTestDataRoot(const base::FilePath& candidate) {
  const base::FilePath marker_path =
      candidate.Append(FILE_PATH_LITERAL("test"))
          .Append(FILE_PATH_LITERAL("test_paths_test_data_root.txt"));

#if !defined(OS_WIN)
  struct stat stat_buf;
  int rv = stat(marker_path.value().c_str(), &stat_buf);
#else
  struct _stat stat_buf;
  int rv = _wstat(marker_path.value().c_str(), &stat_buf);
#endif

  return rv == 0;
}

base::FilePath TestDataRootInternal() {
#if defined(OS_FUCHSIA)
  base::FilePath asset_path("/pkg/assets");
#if defined(CRASHPAD_IS_IN_FUCHSIA)
  // Tests are not yet packaged when running in the Fuchsia tree, so assets do
  // not appear as expected at /pkg/assets. Override the default so that tests
  // can find their data for now.
  // https://crashpad.chromium.org/bug/196.
  asset_path = base::FilePath("/system/data/crashpad_tests");
#endif
  if (!IsTestDataRoot(asset_path)) {
    LOG(WARNING) << "Test data root seems invalid, continuing anyway";
  }
  return asset_path;
#else  // defined(OS_FUCHSIA)
#if !defined(OS_WIN)
  const char* environment_value = getenv("CRASHPAD_TEST_DATA_ROOT");
#else  // defined(OS_WIN)
  const wchar_t* environment_value = _wgetenv(L"CRASHPAD_TEST_DATA_ROOT");
#endif

  if (environment_value) {
    // It was specified explicitly, so use it even if it seems incorrect.
    if (!IsTestDataRoot(base::FilePath(environment_value))) {
      LOG(WARNING) << "CRASHPAD_TEST_DATA_ROOT seems invalid, honoring anyway";
    }

    return base::FilePath(environment_value);
  }

  // In a standalone build, the test executable is usually at
  // out/{Debug,Release} relative to the Crashpad root.
  base::FilePath executable_path;
  if (Paths::Executable(&executable_path)) {
    base::FilePath candidate =
        base::FilePath(executable_path.DirName()
                           .Append(base::FilePath::kParentDirectory)
                           .Append(base::FilePath::kParentDirectory));
    if (IsTestDataRoot(candidate)) {
      return candidate;
    }

    // In an in-Chromium build, the test executable is usually at
    // out/{Debug,Release} relative to the Chromium root, and the Crashpad root
    // is at third_party/crashpad/crashpad relative to the Chromium root.
    candidate = candidate.Append(FILE_PATH_LITERAL("third_party"))
                    .Append(FILE_PATH_LITERAL("crashpad"))
                    .Append(FILE_PATH_LITERAL("crashpad"));
    if (IsTestDataRoot(candidate)) {
      return candidate;
    }
  }

  // If nothing else worked, use the current directory, issuing a warning if it
  // doesn’t seem right.
  if (!IsTestDataRoot(base::FilePath(base::FilePath::kCurrentDirectory))) {
    LOG(WARNING) << "could not locate a valid test data root";
  }

  return base::FilePath(base::FilePath::kCurrentDirectory);
#endif  // defined(OS_FUCHSIA)
}

#if defined(OS_WIN) && defined(ARCH_CPU_64_BITS)

// Returns the pathname of a directory containing 32-bit test build output.
//
// It would be better for this to be named 32BitOutputDirectory(), but that’s
// not a legal name.
base::FilePath Output32BitDirectory() {
  const wchar_t* environment_value = _wgetenv(L"CRASHPAD_TEST_32_BIT_OUTPUT");
  if (!environment_value) {
    return base::FilePath();
  }

  return base::FilePath(environment_value);
}

#endif  // defined(OS_WIN) && defined(ARCH_CPU_64_BITS)

}  // namespace

// static
base::FilePath TestPaths::Executable() {
  base::FilePath executable_path;
  CHECK(Paths::Executable(&executable_path));
#if defined(CRASHPAD_IS_IN_FUCHSIA)
  // Tests are not yet packaged when running in the Fuchsia tree, so binaries do
  // not appear as expected at /pkg/bin. Override the default of /pkg/bin/app
  // so that tests can find the correct location for now.
  // https://crashpad.chromium.org/bug/196.
  executable_path = base::FilePath("/system/bin/crashpad_tests/app");
#endif
  return executable_path;
}

// static
base::FilePath TestPaths::ExpectedExecutableBasename(
    const base::FilePath::StringType& name) {
#if defined(OS_FUCHSIA)
  // Apps in Fuchsia packages are always named "app".
  return base::FilePath("app");
#else  // OS_FUCHSIA
#if defined(CRASHPAD_IS_IN_CHROMIUM)
  base::FilePath::StringType executable_name(
      FILE_PATH_LITERAL("crashpad_tests"));
#else  // CRASHPAD_IS_IN_CHROMIUM
  base::FilePath::StringType executable_name(name);
#endif  // CRASHPAD_IS_IN_CHROMIUM

#if defined(OS_WIN)
  executable_name += FILE_PATH_LITERAL(".exe");
#endif  // OS_WIN

  return base::FilePath(executable_name);
#endif  // OS_FUCHSIA
}

// static
base::FilePath TestPaths::TestDataRoot() {
  static base::FilePath* test_data_root =
      new base::FilePath(TestDataRootInternal());
  return *test_data_root;
}

// static
base::FilePath TestPaths::BuildArtifact(
    const base::FilePath::StringType& module,
    const base::FilePath::StringType& artifact,
    FileType file_type,
    Architecture architecture) {
  base::FilePath directory;
  switch (architecture) {
    case Architecture::kDefault:
      directory = Executable().DirName();
      break;

#if defined(OS_WIN) && defined(ARCH_CPU_64_BITS)
    case Architecture::k32Bit:
      directory = Output32BitDirectory();
      CHECK(!directory.empty());
      break;
#endif  // OS_WIN && ARCH_CPU_64_BITS
  }

  base::FilePath::StringType test_name =
      FILE_PATH_LITERAL("crashpad_") + module + FILE_PATH_LITERAL("_test");
#if !defined(CRASHPAD_IS_IN_CHROMIUM) && !defined(OS_FUCHSIA)
  CHECK(Executable().BaseName().RemoveFinalExtension().value() == test_name);
#endif  // !CRASHPAD_IS_IN_CHROMIUM

  base::FilePath::StringType extension;
  switch (file_type) {
    case FileType::kNone:
      break;

    case FileType::kExecutable:
#if defined(OS_WIN)
      extension = FILE_PATH_LITERAL(".exe");
#elif defined(OS_FUCHSIA)
#if defined(CRASHPAD_IS_IN_FUCHSIA)
      // Tests are not yet packaged when running in the Fuchsia tree, so
      // binaries do not appear as expected at /pkg/bin. Override the default of
      // /pkg/bin/app so that tests can find the correct location for now.
      // https://crashpad.chromium.org/bug/196.
      directory =
          base::FilePath(FILE_PATH_LITERAL("/system/bin/crashpad_tests"));
#else
      directory = base::FilePath(FILE_PATH_LITERAL("/pkg/bin"));
#endif
#endif  // OS_WIN
      break;

    case FileType::kLoadableModule:
#if defined(OS_WIN)
      extension = FILE_PATH_LITERAL(".dll");
#else  // OS_WIN
      extension = FILE_PATH_LITERAL(".so");
#endif  // OS_WIN

#if defined(OS_FUCHSIA)
      // TODO(scottmg): .so files are currently deployed into /boot/lib, where
      // they'll be found (without a path) by the loader. Application packaging
      // infrastructure is in progress, so this will likely change again in the
      // future.
      directory = base::FilePath();
#endif
      break;

    case FileType::kCertificate:
#if defined(CRASHPAD_IS_IN_FUCHSIA)
      // When running in the Fuchsia tree, the .pem files are packaged as assets
      // into the test data folder. This will need to be rationalized when
      // things are actually run from a package.
      // https://crashpad.chromium.org/bug/196.
      directory =
          base::FilePath(FILE_PATH_LITERAL("/system/bin/crashpad_tests"));
#endif
      extension = FILE_PATH_LITERAL(".pem");
      break;
  }

  return directory.Append(test_name + FILE_PATH_LITERAL("_") + artifact +
                          extension);
}

#if defined(OS_WIN) && defined(ARCH_CPU_64_BITS)

// static
bool TestPaths::Has32BitBuildArtifacts() {
  return !Output32BitDirectory().empty();
}

#endif  // defined(OS_WIN) && defined(ARCH_CPU_64_BITS)

}  // namespace test
}  // namespace crashpad
