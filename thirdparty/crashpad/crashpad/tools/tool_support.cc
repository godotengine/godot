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

#include "tools/tool_support.h"

#include <stdio.h>

#include <memory>
#include <vector>

#include "base/strings/utf_string_conversions.h"
#include "package.h"

namespace crashpad {

// static
void ToolSupport::Version(const base::FilePath& me) {
  fprintf(stderr,
          "%" PRFilePath " (%s) %s\n%s\n",
          me.value().c_str(),
          PACKAGE_NAME,
          PACKAGE_VERSION,
          PACKAGE_COPYRIGHT);
}

// static
void ToolSupport::UsageTail(const base::FilePath& me) {
  fprintf(stderr,
          "\nReport %" PRFilePath " bugs to\n%s\n%s home page: <%s>\n",
          me.value().c_str(),
          PACKAGE_BUGREPORT,
          PACKAGE_NAME,
          PACKAGE_URL);
}

// static
void ToolSupport::UsageHint(const base::FilePath& me, const char* hint) {
  if (hint) {
    fprintf(stderr, "%" PRFilePath ": %s\n", me.value().c_str(), hint);
  }
  fprintf(stderr,
          "Try '%" PRFilePath " --help' for more information.\n",
          me.value().c_str());
}

#if defined(OS_POSIX)
// static
void ToolSupport::Version(const std::string& me) {
  Version(base::FilePath(me));
}

// static
void ToolSupport::UsageTail(const std::string& me) {
  UsageTail(base::FilePath(me));
}

// static
void ToolSupport::UsageHint(const std::string& me, const char* hint) {
  UsageHint(base::FilePath(me), hint);
}
#endif  // OS_POSIX

#if defined(OS_WIN)

// static
int ToolSupport::Wmain(int argc, wchar_t* argv[], int (*entry)(int, char* [])) {
  std::unique_ptr<char* []> argv_as_utf8(new char*[argc + 1]);
  std::vector<std::string> storage;
  storage.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    storage.push_back(base::UTF16ToUTF8(argv[i]));
    argv_as_utf8[i] = &storage[i][0];
  }
  argv_as_utf8[argc] = nullptr;
  return entry(argc, argv_as_utf8.get());
}

#endif  // OS_WIN

// static
base::FilePath::StringType ToolSupport::CommandLineArgumentToFilePathStringType(
    const base::StringPiece& path) {
#if defined(OS_POSIX)
  return path.as_string();
#elif defined(OS_WIN)
  return base::UTF8ToUTF16(path);
#endif  // OS_POSIX
}

// static
std::string ToolSupport::FilePathToCommandLineArgument(
    const base::FilePath& file_path) {
#if defined(OS_POSIX)
  return file_path.value();
#elif defined(OS_WIN)
  return base::UTF16ToUTF8(file_path.value());
#endif  // OS_POSIX
}

}  // namespace crashpad
