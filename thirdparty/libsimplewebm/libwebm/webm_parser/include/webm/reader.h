// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_READER_H_
#define INCLUDE_WEBM_READER_H_

#include <cstddef>
#include <cstdint>

#include "./status.h"

/**
 \file
 An interface that acts as a data source for the parser to read from.
 */

namespace webm {

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 A generic interface for reading from a data source.

 Throwing an exception from the member functions is permitted, though if the
 exception will be caught and parsing resumed, then the reader should not
 advance its position before throwing the exception.
 */
class Reader {
 public:
  virtual ~Reader() = default;

  /**
   Reads data from the source and advances the reader's position by the number
   of bytes read.

   Short reads are permitted, as is reading no data.

   \param num_to_read The number of bytes that should be read. Must not be 0.
   \param buffer The buffer to store the read bytes. Must be large enough to
   store at least `num_to_read` bytes. Must not be null.
   \param[out] num_actually_read The number of bytes that were actually read is
   stored in this integer. Must not be null.
   \return `Status::kOkCompleted` if `num_to_read` bytes were read.
   `Status::kOkPartial` if the number of bytes read is > 0 and < `num_to_read`.
   If no bytes are read, then some status other than `Status::kOkCompleted` and
   `Status::kOkPartial` must be returned and `num_actually_read` must be set to
   0.
   */
  virtual Status Read(std::size_t num_to_read, std::uint8_t* buffer,
                      std::uint64_t* num_actually_read) = 0;

  /**
   Skips data from the source and advances the reader's position by the number
   of bytes skipped.

   Short skips are permitted, as is skipping no data. This is similar to the
   `Read()` method, but does not store data in an output buffer.

   \param num_to_skip The number of bytes that should be skipped. Must not be 0.
   \param[out] num_actually_skipped The number of bytes that were actually
   skipped is stored in this integer. Must not be null.
   \return `Status::kOkCompleted` if `num_to_skip` bytes were skipped.
   `Status::kOkPartial` if the number of bytes skipped is > 0 and <
   `num_to_skip`. If no bytes are skipped, then some status other than
   `Status::kOkCompleted` and `Status::kOkPartial` must be returned and
   `num_actually_skipped` must be set to 0.
   */
  virtual Status Skip(std::uint64_t num_to_skip,
                      std::uint64_t* num_actually_skipped) = 0;

  /**
   Gets the Reader's current absolute byte position in the stream.

   Implementations don't necessarily need to start from 0 (which might be the
   case if parsing is starting in the middle of a data source). The value
   `kUnknownElementPosition` must not be returned.
   */
  virtual std::uint64_t Position() const = 0;
};

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_READER_H_
