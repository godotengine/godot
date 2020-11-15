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

#include "util/file/directory_reader.h"

#include <dirent.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>

#include "base/logging.h"

namespace crashpad {

#define HANDLE_EINTR_IF_EQ(x, val)                             \
  ({                                                           \
    decltype(x) eintr_wrapper_result;                          \
    do {                                                       \
      eintr_wrapper_result = (x);                              \
    } while (eintr_wrapper_result == (val) && errno == EINTR); \
    eintr_wrapper_result;                                      \
  })

DirectoryReader::DirectoryReader() : dir_() {}

DirectoryReader::~DirectoryReader() {}

bool DirectoryReader::Open(const base::FilePath& path) {
  dir_.reset(HANDLE_EINTR_IF_EQ(opendir(path.value().c_str()), nullptr));
  if (!dir_.is_valid()) {
    PLOG(ERROR) << "opendir";
    return false;
  }
  return true;
}

DirectoryReader::Result DirectoryReader::NextFile(base::FilePath* filename) {
  DCHECK(dir_.is_valid());

  errno = 0;
  dirent* entry = HANDLE_EINTR_IF_EQ(readdir(dir_.get()), nullptr);
  if (!entry) {
    if (errno) {
      PLOG(ERROR) << "readdir";
      return Result::kError;
    } else {
      return Result::kNoMoreFiles;
    }
  }

  if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
    return NextFile(filename);
  }

  *filename = base::FilePath(entry->d_name);
  return Result::kSuccess;
}

int DirectoryReader::DirectoryFD() {
  DCHECK(dir_.is_valid());
  int rv = dirfd(dir_.get());
  if (rv < 0) {
    PLOG(ERROR) << "dirfd";
  }
  return rv;
}

}  // namespace crashpad
