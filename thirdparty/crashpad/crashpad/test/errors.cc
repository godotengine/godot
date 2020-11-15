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

#include "test/errors.h"

#include <errno.h>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "build/build_config.h"

#if defined(OS_POSIX)
#include "base/posix/safe_strerror.h"
#elif defined(OS_WIN)
#include <string.h>
#include <windows.h>
#endif

namespace crashpad {
namespace test {

std::string ErrnoMessage(int err, const std::string& base) {
#if defined(OS_POSIX)
  std::string err_as_string = base::safe_strerror(errno);
  const char* err_string = err_as_string.c_str();
#elif defined(OS_WIN)
  char err_string[256];
  strerror_s(err_string, errno);
#endif
  return base::StringPrintf("%s%s%s (%d)",
                            base.c_str(),
                            base.empty() ? "" : ": ",
                            err_string,
                            err);
}

std::string ErrnoMessage(const std::string& base) {
  return ErrnoMessage(errno, base);
}

#if defined(OS_WIN)
std::string ErrorMessage(const std::string& base) {
  return base::StringPrintf(
      "%s%s%s",
      base.c_str(),
      base.empty() ? "" : ": ",
      logging::SystemErrorCodeToString(GetLastError()).c_str());
}
#endif

}  // namespace test
}  // namespace crashpad
