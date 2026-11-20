// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_ISTREAM_READER_H_
#define INCLUDE_WEBM_ISTREAM_READER_H_

#include <cstdint>
#include <cstdlib>
#include <istream>
#include <memory>
#include <type_traits>

#include "./reader.h"
#include "./status.h"

/**
 \file
 A `Reader` implementation that reads from a `std::istream`.
 */

namespace webm {

/**
 A `Reader` implementation that can read from `std::istream`-based resources.
 */
class IstreamReader : public Reader {
 public:
  /**
   Constructs a new, empty reader.
   */
  IstreamReader() = default;

  /**
   Constructs a new reader, using the provided `std::istream` as the data
   source.

   Ownership of the stream is taken, and it will be moved into a new internal
   instance.

   \param istream The stream to use as a data source. Must be an rvalue
   reference derived from `std::istream`.
   */
  template <typename T>
  explicit IstreamReader(T&& istream) : istream_(new T(std::move(istream))) {}

  /**
   Constructs a new reader by moving the provided reader into the new reader.

   \param other The source reader to move. After moving, it will be reset to an
   empty stream.
   */
  IstreamReader(IstreamReader&& other);

  /**
   Moves the provided reader into this reader.

   \param other The source reader to move. After moving, it will be reset to an
   empty stream. May be equal to `*this`, in which case this is a no-op.
   \return `*this`.
   */
  IstreamReader& operator=(IstreamReader&& other);

  Status Read(std::size_t num_to_read, std::uint8_t* buffer,
              std::uint64_t* num_actually_read) override;

  Status Skip(std::uint64_t num_to_skip,
              std::uint64_t* num_actually_skipped) override;

  std::uint64_t Position() const override;

  /**
   Constructs a new reader and its data source in place.

   `T` must be derived from `std::istream` and constructible from the provided
   arguments.

   \param args Arguments that will be forwarded to the `T`'s constructor.
   \return A new `IstreamReader` backed by the underlying data source equivalent
   to `T(std::forward<Args>(args)...`.
   */
  template <typename T, typename... Args>
  static IstreamReader Emplace(Args&&... args) {
    IstreamReader reader;
    reader.istream_.reset(new T(std::forward<Args>(args)...));
    return reader;
  }

 private:
  std::unique_ptr<std::istream> istream_;
  std::uint64_t position_ = 0;
};

}  // namespace webm

#endif  // INCLUDE_WEBM_ISTREAM_READER_H_
