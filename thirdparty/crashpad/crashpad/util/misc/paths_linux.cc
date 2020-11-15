// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "util/misc/paths.h"

#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <string>

#include "base/logging.h"

namespace crashpad {

// static
bool Paths::Executable(base::FilePath* path) {
  // Linux does not provide a straightforward way to size the buffer before
  // calling readlink(). Normally, the st_size field returned by lstat() could
  // be used, but this is usually zero for things in /proc.
  //
  // The /proc filesystem does not provide any way to read “exe” links for
  // pathnames longer than a page. See linux-4.9.20/fs/proc/base.c
  // do_proc_readlink(), which allocates a single page to receive the path
  // string. Coincidentally, the page size and PATH_MAX are normally the same
  // value, although neither is strictly a limit on the length of a pathname.
  //
  // On Android, the smaller of the page size and PATH_MAX actually does serve
  // as an effective limit on the length of an executable’s pathname. See
  // Android 7.1.1 bionic/linker/linker.cpp get_executable_path(), which aborts
  // via __libc_fatal() if the “exe” link can’t be read into a PATH_MAX-sized
  // buffer.
  std::string exe_path(std::max(getpagesize(), PATH_MAX),
                       std::string::value_type());
  ssize_t exe_path_len =
      readlink("/proc/self/exe", &exe_path[0], exe_path.size());
  if (exe_path_len < 0) {
    PLOG(ERROR) << "readlink";
    return false;
  } else if (static_cast<size_t>(exe_path_len) >= exe_path.size()) {
    LOG(ERROR) << "readlink";
    return false;
  }

  exe_path.resize(exe_path_len);
  *path = base::FilePath(exe_path);
  return true;
}

}  // namespace crashpad
