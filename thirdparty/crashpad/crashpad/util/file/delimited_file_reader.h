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

#ifndef CRASHPAD_UTIL_FILE_DELIMITED_FILE_READER_H_
#define CRASHPAD_UTIL_FILE_DELIMITED_FILE_READER_H_

#include <stdint.h>

#include <string>

#include "base/macros.h"
#include "util/file/file_reader.h"

namespace crashpad {

//! \brief Reads a file one field or line at a time.
//!
//! The file is interpreted as a series of fields separated by delimiter
//! characters. When the delimiter character is the newline character
//! (<code>'\\n'</code>), the file is interpreted as a series of lines.
//!
//! It is safe to mix GetDelim() and GetLine() calls, if appropriate for the
//! format being interpreted.
//!
//! This is a replacement for the standard library’s `getdelim()` and
//! `getline()` functions, adapted to work with FileReaderInterface objects
//! instead of `FILE*` streams.
class DelimitedFileReader {
 public:
  //! \brief The result of a GetDelim() or GetLine() call.
  enum class Result {
    //! \brief An error occurred, and a message was logged.
    kError = -1,

    //! \brief A field or line was read from the file.
    kSuccess,

    //! \brief The end of the file was encountered.
    kEndOfFile,
  };

  explicit DelimitedFileReader(FileReaderInterface* file_reader);
  ~DelimitedFileReader();

  //! \brief Reads a single field from the file.
  //!
  //! \param[in] delimiter The delimiter character that terminates the field.
  //!     It is safe to call this method multiple times while changing the value
  //!     of this parameter, if appropriate for the format being interpreted.
  //! \param[out] field The field read from the file. This parameter will
  //!     include the field’s terminating delimiter character unless the field
  //!     was at the end of the file and was read without such a character.
  //!     This parameter will not be empty.
  //!
  //! \return a #Result value. \a field is only valid when Result::kSuccess is
  //!     returned.
  Result GetDelim(char delimiter, std::string* field);

  //! \brief Reads a single line from the file.
  //!
  //! \param[out] line The line read from the file. This parameter will include
  //!     the line terminating delimiter character unless the line was at the
  //!     end of the file and was read without such a character. This parameter
  //!     will not be empty.
  //!
  //! \return a #Result value. \a line is only valid when Result::kSuccess is
  //!     returned.
  Result GetLine(std::string* line);

 private:
  char buf_[4096];
  FileReaderInterface* file_reader_;  // weak
  uint16_t buf_pos_;  // Index into buf_ of the start of the next field.
  uint16_t buf_len_;  // The size of buf_ that’s been filled.
  bool eof_;  // Caches the EOF signal when detected following a partial field.

  DISALLOW_COPY_AND_ASSIGN(DelimitedFileReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FILE_DELIMITED_FILE_READER_H_
