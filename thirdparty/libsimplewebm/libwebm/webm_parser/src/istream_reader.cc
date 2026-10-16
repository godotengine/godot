// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "webm/istream_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

#include "webm/status.h"

namespace webm {

IstreamReader::IstreamReader(IstreamReader&& other)
    : istream_(std::move(other.istream_)), position_(other.position_) {
  other.position_ = 0;
}

IstreamReader& IstreamReader::operator=(IstreamReader&& other) {
  if (this != &other) {
    istream_ = std::move(other.istream_);
    position_ = other.position_;
    other.position_ = 0;
  }
  return *this;
}

Status IstreamReader::Read(std::size_t num_to_read, std::uint8_t* buffer,
                           std::uint64_t* num_actually_read) {
  assert(num_to_read > 0);
  assert(buffer != nullptr);
  assert(num_actually_read != nullptr);

  if (istream_ == nullptr) {
    *num_actually_read = 0;
    return Status(Status::kEndOfFile);
  }

  using unsigned_streamsize = std::make_unsigned<std::streamsize>::type;
  constexpr std::streamsize streamsize_max =
      std::numeric_limits<std::streamsize>::max();
  std::streamsize limited_num_to_read;
  if (num_to_read > static_cast<unsigned_streamsize>(streamsize_max)) {
    limited_num_to_read = streamsize_max;
  } else {
    limited_num_to_read = static_cast<std::streamsize>(num_to_read);
  }

  istream_->read(reinterpret_cast<char*>(buffer), limited_num_to_read);
  std::streamsize actual = istream_->gcount();
  *num_actually_read = static_cast<std::uint64_t>(actual);
  position_ += *num_actually_read;

  if (actual == 0) {
    return Status(Status::kEndOfFile);
  }

  if (static_cast<std::size_t>(actual) == num_to_read) {
    return Status(Status::kOkCompleted);
  } else {
    return Status(Status::kOkPartial);
  }
}

Status IstreamReader::Skip(std::uint64_t num_to_skip,
                           std::uint64_t* num_actually_skipped) {
  assert(num_to_skip > 0);
  assert(num_actually_skipped != nullptr);

  *num_actually_skipped = 0;
  if (istream_ == nullptr || !istream_->good()) {
    return Status(Status::kEndOfFile);
  }

  // Try seeking forward first.
  using unsigned_streamsize = std::make_unsigned<std::streamsize>::type;
  constexpr std::streamsize streamsize_max =
      std::numeric_limits<std::streamsize>::max();
  std::streamsize seek_offset;
  if (num_to_skip > static_cast<unsigned_streamsize>(streamsize_max)) {
    seek_offset = streamsize_max;
  } else {
    seek_offset = static_cast<std::streamsize>(num_to_skip);
  }
  if (istream_->seekg(seek_offset, std::ios_base::cur)) {
    *num_actually_skipped = static_cast<std::uint64_t>(seek_offset);
    position_ += static_cast<std::uint64_t>(seek_offset);
    if (static_cast<std::uint64_t>(seek_offset) == num_to_skip) {
      return Status(Status::kOkCompleted);
    } else {
      return Status(Status::kOkPartial);
    }
  }
  istream_->clear();

  // Seeking doesn't work on things like pipes, so if seeking failed then fall
  // back to reading the data into a junk buffer.
  std::size_t actual = 0;
  do {
    char junk[1024];
    std::streamsize num_to_read = static_cast<std::streamsize>(sizeof(junk));
    if (num_to_skip < static_cast<std::uint64_t>(num_to_read)) {
      num_to_read = static_cast<std::streamsize>(num_to_skip);
    }

    istream_->read(junk, num_to_read);
    std::streamsize actual = istream_->gcount();
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

std::uint64_t IstreamReader::Position() const { return position_; }

}  // namespace webm
