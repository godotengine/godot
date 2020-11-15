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

#include <string.h>

#include "base/logging.h"

namespace crashpad {

DirectoryReader::DirectoryReader()
    : find_data_(), handle_(), first_entry_(false) {}

DirectoryReader::~DirectoryReader() {}

bool DirectoryReader::Open(const base::FilePath& path) {
  if (path.empty()) {
    LOG(ERROR) << "Empty directory path";
    return false;
  }

  handle_.reset(
      FindFirstFileEx(path.Append(FILE_PATH_LITERAL("*")).value().c_str(),
                      FindExInfoBasic,
                      &find_data_,
                      FindExSearchNameMatch,
                      nullptr,
                      FIND_FIRST_EX_LARGE_FETCH));

  if (!handle_.is_valid()) {
    PLOG(ERROR) << "FindFirstFile";
    return false;
  }

  first_entry_ = true;
  return true;
}

DirectoryReader::Result DirectoryReader::NextFile(base::FilePath* filename) {
  DCHECK(handle_.is_valid());

  if (!first_entry_) {
    if (!FindNextFile(handle_.get(), &find_data_)) {
      if (GetLastError() != ERROR_NO_MORE_FILES) {
        PLOG(ERROR) << "FindNextFile";
        return Result::kError;
      } else {
        return Result::kNoMoreFiles;
      }
    }
  } else {
    first_entry_ = false;
  }

  if (wcscmp(find_data_.cFileName, L".") == 0 ||
      wcscmp(find_data_.cFileName, L"..") == 0) {
    return NextFile(filename);
  }

  *filename = base::FilePath(find_data_.cFileName);
  return Result::kSuccess;
}

}  // namespace crashpad
