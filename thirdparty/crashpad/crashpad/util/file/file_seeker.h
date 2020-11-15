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

#ifndef CRASHPAD_UTIL_FILE_FILE_SEEKER_H_
#define CRASHPAD_UTIL_FILE_FILE_SEEKER_H_

#include "util/file/file_io.h"

namespace crashpad {

//! \brief An interface to seek in files and other file-like objects with
//!     semantics matching the underlying platform (POSIX or Windows).
class FileSeekerInterface {
 public:
  //! \brief Wraps LoggingSeekFile() or provides an alternate implementation
  //!     with identical semantics.
  //!
  //! \return The return value of LoggingSeekFile(). `-1` on failure,
  //!     with an error message logged.
  virtual FileOffset Seek(FileOffset offset, int whence) = 0;

  //! \brief Wraps Seek(), using `SEEK_CUR` to obtain the file’s current
  //!     position.
  //!
  //! \return The file’s current position on success. `-1` on failure, with an
  //!     error message logged.
  FileOffset SeekGet();

  //! \brief Wraps Seek(), using `SEEK_SET`, ensuring that the seek succeeded
  //!     and the file is positioned as desired.
  //!
  //! \return `true` if the operation succeeded, `false` if it failed, with an
  //!     error message logged. A failure to reposition the file as desired is
  //!     treated as a failure.
  bool SeekSet(FileOffset offset);

 protected:
  ~FileSeekerInterface() {}
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FILE_FILE_SEEKER_H_
