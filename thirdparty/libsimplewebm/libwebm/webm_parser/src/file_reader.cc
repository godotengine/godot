// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "webm/file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

#include "webm/status.h"

#ifdef _MSC_VER
#define FSEEK_(stream, offset, whence) _fseeki64(stream, offset, whence)
#elif defined(_WIN32)
#define FSEEK_(stream, offset, whence) \
  fseeko64(stream, static_cast<off_t>(offset), whence)
#elif _POSIX_C_SOURCE >= 200112L
#define FSEEK_(stream, offset, whence) \
  fseeko(stream, static_cast<off_t>(offset), whence)
#else
#define FSEEK_(stream, offset, whence) \
  std::fseek(stream, static_cast<long>(offset), whence)
#endif

namespace webm {

FileReader::FileReader(FILE* file) : file_(file) { assert(file); }

FileReader::FileReader(FileReader&& other)
    : file_(std::move(other.file_)), position_(other.position_) {
  other.position_ = 0;
}

FileReader& FileReader::operator=(FileReader&& other) {
  if (this != &other) {
    file_ = std::move(other.file_);
    position_ = other.position_;
    other.position_ = 0;
  }
  return *this;
}

Status FileReader::Read(std::size_t num_to_read, std::uint8_t* buffer,
                        std::uint64_t* num_actually_read) {
  assert(num_to_read > 0);
  assert(buffer != nullptr);
  assert(num_actually_read != nullptr);

  if (file_ == nullptr) {
    *num_actually_read = 0;
    return Status(Status::kEndOfFile);
  }

  std::size_t actual =
      std::fread(static_cast<void*>(buffer), 1, num_to_read, file_.get());
  *num_actually_read = static_cast<std::uint64_t>(actual);
  position_ += *num_actually_read;

  if (actual == 0) {
    return Status(Status::kEndOfFile);
  }

  if (actual == num_to_read) {
    return Status(Status::kOkCompleted);
  } else {
    return Status(Status::kOkPartial);
  }
}

Status FileReader::Skip(std::uint64_t num_to_skip,
                        std::uint64_t* num_actually_skipped) {
  assert(num_to_skip > 0);
  assert(num_actually_skipped != nullptr);

  *num_actually_skipped = 0;

  if (file_ == nullptr) {
    return Status(Status::kEndOfFile);
  }

  // Try seeking forward first.
  long seek_offset = std::numeric_limits<long>::max();  // NOLINT
  if (num_to_skip < static_cast<unsigned long>(seek_offset)) {  // NOLINT
    seek_offset = static_cast<long>(num_to_skip);  // NOLINT
  }
  if (!FSEEK_(file_.get(), seek_offset, SEEK_CUR)) {
    *num_actually_skipped = static_cast<std::uint64_t>(seek_offset);
    position_ += static_cast<std::uint64_t>(seek_offset);
    if (static_cast<unsigned long>(seek_offset) == num_to_skip) {  // NOLINT
      return Status(Status::kOkCompleted);
    } else {
      return Status(Status::kOkPartial);
    }
  }
  std::clearerr(file_.get());

  // Seeking doesn't work on things like pipes, so if seeking failed then fall
  // back to reading the data into a junk buffer.
  std::size_t actual = 0;
  do {
    std::uint8_t junk[1024];
    std::size_t num_to_read = sizeof(junk);
    if (num_to_skip < num_to_read) {
      num_to_read = static_cast<std::size_t>(num_to_skip);
    }

    std::size_t actual =
        std::fread(static_cast<void*>(junk), 1, num_to_read, file_.get());
    *num_actually_skipped += static_cast<std::uint64_t>(actual);
    position_ += static_cast<std::uint64_t>(actual);
    num_to_skip -= static_cast<std::uint64_t>(actual);
  } while (actual > 0 && num_to_skip > 0);

  if (*num_actually_skipped == 0) {
    return Status(Status::kEndOfFile);
  }

  if (num_to_skip == 0) {
    return Status(Status::kOkCompleted);
  } else {
    return Status(Status::kOkPartial);
  }
}

Status FileReader::Seek(std::uint64_t seek_position) {
  if (FSEEK_(file_.get(), seek_position, SEEK_SET)) {
    return Status(Status::kSeekFailed);
  }
  position_ = seek_position;
  return Status(Status::kOkCompleted);
}

std::uint64_t FileReader::Position() const { return position_; }

}  // namespace webm
