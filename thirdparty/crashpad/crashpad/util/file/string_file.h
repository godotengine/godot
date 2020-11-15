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

#ifndef CRASHPAD_UTIL_FILE_STRING_FILE_H_
#define CRASHPAD_UTIL_FILE_STRING_FILE_H_

#include <sys/types.h>

#include <string>

#include "base/macros.h"
#include "base/numerics/safe_math.h"
#include "util/file/file_io.h"
#include "util/file/file_reader.h"
#include "util/file/file_writer.h"

namespace crashpad {

//! \brief A file reader and writer backed by a virtual file, as opposed to a
//!     file on disk or other operating system file descriptor-based file.
//!
//! The virtual file is a buffer in memory. This class is convenient for use
//! with other code that normally expects to read or write files, when it is
//! impractical or inconvenient to read or write a file. It is expected that
//! tests, in particular, will benefit from using this class.
class StringFile : public FileReaderInterface, public FileWriterInterface {
 public:
  StringFile();
  ~StringFile() override;

  //! \brief Returns a string containing the virtual file’s contents.
  const std::string& string() const { return string_; }

  //! \brief Sets the virtual file’s contents to \a string, and resets its file
  //!     position to `0`.
  void SetString(const std::string& string);

  //! \brief Resets the virtual file’s contents to be empty, and resets its file
  //!     position to `0`.
  void Reset();

  // FileReaderInterface:
  FileOperationResult Read(void* data, size_t size) override;

  // FileWriterInterface:
  bool Write(const void* data, size_t size) override;
  bool WriteIoVec(std::vector<WritableIoVec>* iovecs) override;

  // FileSeekerInterface:
  FileOffset Seek(FileOffset offset, int whence) override;

 private:
  //! \brief The virtual file’s contents.
  std::string string_;

  //! \brief The file offset of the virtual file.
  //!
  //! \note This is stored in a `size_t` to match the characteristics of
  //!     #string_, the `std::string` used to store the virtual file’s contents.
  //!     This type will have different characteristics than the `off_t` used to
  //!     report file offsets. The implementation must take care when converting
  //!     between these distinct types.
  base::CheckedNumeric<size_t> offset_;

  DISALLOW_COPY_AND_ASSIGN(StringFile);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FILE_STRING_FILE_H_
