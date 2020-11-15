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

#include "util/file/filesystem.h"

#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "base/logging.h"
#include "build/build_config.h"

namespace crashpad {

bool FileModificationTime(const base::FilePath& path, timespec* mtime) {
  struct stat st;
  if (lstat(path.value().c_str(), &st) != 0) {
    PLOG(ERROR) << "lstat " << path.value();
    return false;
  }

#if defined(OS_MACOSX)
  *mtime = st.st_mtimespec;
#elif defined(OS_ANDROID)
  // This is needed to compile with traditional NDK headers.
  mtime->tv_sec = st.st_mtime;
  mtime->tv_nsec = st.st_mtime_nsec;
#else
  *mtime = st.st_mtim;
#endif
  return true;
}

bool LoggingCreateDirectory(const base::FilePath& path,
                            FilePermissions permissions,
                            bool may_reuse) {
  if (mkdir(path.value().c_str(),
            permissions == FilePermissions::kWorldReadable ? 0755 : 0700) ==
      0) {
    return true;
  }
  if (may_reuse && errno == EEXIST) {
    if (!IsDirectory(path, true)) {
      LOG(ERROR) << path.value() << " not a directory";
      return false;
    }
    return true;
  }
  PLOG(ERROR) << "mkdir " << path.value();
  return false;
}

bool MoveFileOrDirectory(const base::FilePath& source,
                         const base::FilePath& dest) {
#if defined(OS_FUCHSIA)
  // Fuchsia fails and sets errno to EINVAL if source and dest are the same.
  // Upstream bug is ZX-1729.
  if (!source.empty() && source == dest) {
    return true;
  }
#endif  // OS_FUCHSIA

  if (rename(source.value().c_str(), dest.value().c_str()) != 0) {
    PLOG(ERROR) << "rename " << source.value().c_str() << ", "
                << dest.value().c_str();
    return false;
  }
  return true;
}

bool IsRegularFile(const base::FilePath& path) {
  struct stat st;
  if (lstat(path.value().c_str(), &st) != 0) {
    PLOG(ERROR) << "stat " << path.value();
    return false;
  }
  return S_ISREG(st.st_mode);
}

bool IsDirectory(const base::FilePath& path, bool allow_symlinks) {
  struct stat st;
  if (allow_symlinks) {
    if (stat(path.value().c_str(), &st) != 0) {
      PLOG(ERROR) << "stat " << path.value();
      return false;
    }
  } else if (lstat(path.value().c_str(), &st) != 0) {
    PLOG(ERROR) << "lstat " << path.value();
    return false;
  }
  return S_ISDIR(st.st_mode);
}

bool LoggingRemoveFile(const base::FilePath& path) {
  if (unlink(path.value().c_str()) != 0) {
    PLOG(ERROR) << "unlink " << path.value();
    return false;
  }
  return true;
}

bool LoggingRemoveDirectory(const base::FilePath& path) {
  if (rmdir(path.value().c_str()) != 0) {
    PLOG(ERROR) << "rmdir " << path.value();
    return false;
  }
  return true;
}

}  // namespace crashpad
