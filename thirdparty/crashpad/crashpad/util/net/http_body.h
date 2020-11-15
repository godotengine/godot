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

#ifndef CRASHPAD_UTIL_NET_HTTP_BODY_H_
#define CRASHPAD_UTIL_NET_HTTP_BODY_H_

#include <stdint.h>
#include <sys/types.h>

#include <string>
#include <vector>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "util/file/file_io.h"
#include "util/file/file_reader.h"

namespace crashpad {

//! \brief An interface to a stream that can be used for an HTTP request body.
class HTTPBodyStream {
 public:
  virtual ~HTTPBodyStream() {}

  //! \brief Copies up to \a max_len bytes into the user-supplied buffer.
  //!
  //! \param[out] buffer A user-supplied buffer into which this method will copy
  //!      bytes from the stream.
  //! \param[in] max_len The length (or size) of \a buffer. At most this many
  //!     bytes will be copied.
  //!
  //! \return On success, a positive number indicating the number of bytes
  //!     actually copied to \a buffer. On failure, a negative number. When
  //!     the stream has no more data, returns `0`.
  virtual FileOperationResult GetBytesBuffer(uint8_t* buffer,
                                             size_t max_len) = 0;

 protected:
  HTTPBodyStream() {}
};

//! \brief An implementation of HTTPBodyStream that turns a fixed string into
//!     a stream.
class StringHTTPBodyStream : public HTTPBodyStream {
 public:
  //! \brief Creates a stream with the specified string.
  //!
  //! \param[in] string The string to turn into a stream.
  explicit StringHTTPBodyStream(const std::string& string);

  ~StringHTTPBodyStream() override;

  // HTTPBodyStream:
  FileOperationResult GetBytesBuffer(uint8_t* buffer, size_t max_len) override;

 private:
  std::string string_;
  size_t bytes_read_;

  DISALLOW_COPY_AND_ASSIGN(StringHTTPBodyStream);
};

//! \brief An implementation of HTTPBodyStream that reads from a
//!     FileReaderInterface and provides its contents for an HTTP body.
class FileReaderHTTPBodyStream : public HTTPBodyStream {
 public:
  //! \brief Creates a stream for reading from a FileReaderInterface.
  //!
  //! \param[in] reader A FileReaderInterface from which this HTTPBodyStream
  //!     will read.
  explicit FileReaderHTTPBodyStream(FileReaderInterface* reader);

  ~FileReaderHTTPBodyStream() override;

  // HTTPBodyStream:
  FileOperationResult GetBytesBuffer(uint8_t* buffer, size_t max_len) override;

 private:
  FileReaderInterface* reader_;  // weak
  bool reached_eof_;

  DISALLOW_COPY_AND_ASSIGN(FileReaderHTTPBodyStream);
};

//! \brief An implementation of HTTPBodyStream that combines an array of
//!     several other HTTPBodyStream objects into a single, unified stream.
class CompositeHTTPBodyStream : public HTTPBodyStream {
 public:
  using PartsList = std::vector<HTTPBodyStream*>;

  //! \brief Creates a stream from an array of other stream parts.
  //!
  //! \param[in] parts A vector of HTTPBodyStream objects, of which this object
  //!     takes ownership, that will be represented as a single unified stream.
  //!     Callers should not mutate the stream objects after passing them to
  //!     an instance of this class.
  explicit CompositeHTTPBodyStream(const PartsList& parts);

  ~CompositeHTTPBodyStream() override;

  // HTTPBodyStream:
  FileOperationResult GetBytesBuffer(uint8_t* buffer, size_t max_len) override;

 private:
  PartsList parts_;
  PartsList::iterator current_part_;

  DISALLOW_COPY_AND_ASSIGN(CompositeHTTPBodyStream);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NET_HTTP_BODY_H_
