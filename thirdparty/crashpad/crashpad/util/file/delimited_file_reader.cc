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

#include "util/file/delimited_file_reader.h"

#include <sys/types.h>

#include <algorithm>
#include <limits>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"

namespace crashpad {

DelimitedFileReader::DelimitedFileReader(FileReaderInterface* file_reader)
    : file_reader_(file_reader), buf_pos_(0), buf_len_(0), eof_(false) {
  static_assert(sizeof(buf_) <= std::numeric_limits<decltype(buf_pos_)>::max(),
                "buf_pos_ must cover buf_");
  static_assert(sizeof(buf_) <= std::numeric_limits<decltype(buf_len_)>::max(),
                "buf_len_ must cover buf_");
}

DelimitedFileReader::~DelimitedFileReader() {}

DelimitedFileReader::Result DelimitedFileReader::GetDelim(char delimiter,
                                                          std::string* field) {
  if (eof_) {
    DCHECK_EQ(buf_pos_, buf_len_);

    // Allow subsequent calls to attempt to read more data from the file. If the
    // file is still at EOF in the future, the read will return 0 and cause
    // kEndOfFile to be returned anyway.
    eof_ = false;

    return Result::kEndOfFile;
  }

  std::string local_field;
  while (true) {
    if (buf_pos_ == buf_len_) {
      // buf_ is empty. Refill it.
      FileOperationResult read_result = file_reader_->Read(buf_, sizeof(buf_));
      if (read_result < 0) {
        return Result::kError;
      } else if (read_result == 0) {
        if (!local_field.empty()) {
          // The file ended with a field that wasn’t terminated by a delimiter
          // character.
          //
          // This is EOF, but EOF can’t be returned because there’s a field that
          // needs to be returned to the caller. Cache the detected EOF so it
          // can be returned next time. This is done to support proper semantics
          // for weird “files” like terminal input that can reach EOF and then
          // “grow”, allowing subsequent reads past EOF to block while waiting
          // for more data. Once EOF is detected by a read that returns 0, that
          // EOF signal should propagate to the caller before attempting a new
          // read. Here, it will be returned on the next call to this method
          // without attempting to read more data.
          eof_ = true;
          field->swap(local_field);
          return Result::kSuccess;
        }
        return Result::kEndOfFile;
      }

      DCHECK_LE(static_cast<size_t>(read_result), arraysize(buf_));
      DCHECK(
          base::IsValueInRangeForNumericType<decltype(buf_len_)>(read_result));
      buf_len_ = static_cast<decltype(buf_len_)>(read_result);
      buf_pos_ = 0;
    }

    const char* const start = buf_ + buf_pos_;
    const char* const end = buf_ + buf_len_;
    const char* const found = std::find(start, end, delimiter);

    local_field.append(start, found);
    buf_pos_ = static_cast<decltype(buf_pos_)>(found - buf_);
    DCHECK_LE(buf_pos_, buf_len_);

    if (found != end) {
      // A real delimiter character was found. Append it to the field being
      // built and return it.
      local_field.push_back(delimiter);
      ++buf_pos_;
      DCHECK_LE(buf_pos_, buf_len_);
      field->swap(local_field);
      return Result::kSuccess;
    }
  }
}

DelimitedFileReader::Result DelimitedFileReader::GetLine(std::string* line) {
  return GetDelim('\n', line);
}

}  // namespace crashpad
