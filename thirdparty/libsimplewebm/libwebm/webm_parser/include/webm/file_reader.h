// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_FILE_READER_H_
#define INCLUDE_WEBM_FILE_READER_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "./reader.h"
#include "./status.h"

/**
 \file
 A `Reader` implementation that reads from a `FILE*`.
 */

namespace webm {

/**
 A `Reader` implementation that can read from `FILE*` resources.
 */
class FileReader : public Reader {
 public:
  /**
   Constructs a new, empty reader.
   */
  FileReader() = default;

  /**
   Constructs a new reader, using the provided file as the data source.

   Ownership of the file is taken, and `std::fclose()` will be called when the
   object is destroyed.

   \param file The file to use as a data source. Must not be null. The file will
   be closed (via `std::fclose()`) when the `FileReader`'s destructor runs.
   */
  explicit FileReader(FILE* file);

  /**
   Constructs a new reader by moving the provided reader into the new reader.

   \param other The source reader to move. After moving, it will be reset to an
   empty stream.
   */
  FileReader(FileReader&& other);

  /**
   Moves the provided reader into this reader.

   \param other The source reader to move. After moving, it will be reset to an
   empty stream. May be equal to `*this`, in which case this is a no-op.
   \return `*this`.
   */
  FileReader& operator=(FileReader&& other);

  Status Read(std::size_t num_to_read, std::uint8_t* buffer,
              std::uint64_t* num_actually_read) override;

  Status Skip(std::uint64_t num_to_skip,
              std::uint64_t* num_actually_skipped) override;

  /**
   Moves the reader to a new absolute byte position in the file.

   It is required to call DidSeek() on the parser after successfully seeking.
   Seeking will only work on actual files, not stdin or pipes.

   \param seek_position The new absolute byte position in the file.
   \return `Status::kOkCompleted` if reader position is now `seek_position`.
   `Status::kSeekFailed` if the reader was unable to seek to `seek_position`
   such as when the file is stdin.
   */
  Status Seek(std::uint64_t seek_position);

  std::uint64_t Position() const override;

 private:
  struct FileCloseFunctor {
    void operator()(FILE* file) const {
      if (file)
        std::fclose(file);
    }
  };

  std::unique_ptr<FILE, FileCloseFunctor> file_;

  // We can't rely on ftell() for the position (since it only returns long, and
  // doesn't work on things like pipes); we need to manually track the reading
  // position.
  std::uint64_t position_ = 0;
};

}  // namespace webm

#endif  // INCLUDE_WEBM_FILE_READER_H_
