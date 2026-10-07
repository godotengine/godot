// Copyright (c) 2015 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef LIBWEBM_COMMON_LIBWEBM_UTIL_H_
#define LIBWEBM_COMMON_LIBWEBM_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace libwebm {

const double kNanosecondsPerSecond = 1000000000.0;

// fclose functor for wrapping FILE in std::unique_ptr.
// TODO(tomfinegan): Move this to file_util once c++11 restrictions are
//                   relaxed.
struct FILEDeleter {
  int operator()(std::FILE* f) {
    if (f != nullptr)
      return fclose(f);
    return 0;
  }
};
typedef std::unique_ptr<std::FILE, FILEDeleter> FilePtr;

struct Range {
  Range(std::size_t off, std::size_t len) : offset(off), length(len) {}
  Range() = delete;
  Range(const Range&) = default;
  Range(Range&&) = default;
  ~Range() = default;
  const std::size_t offset;
  const std::size_t length;
};
typedef std::vector<Range> Ranges;

// Converts |nanoseconds| to 90000 Hz clock ticks and vice versa. Each return
// the converted value.
std::int64_t NanosecondsTo90KhzTicks(std::int64_t nanoseconds);
std::int64_t Khz90TicksToNanoseconds(std::int64_t khz90_ticks);

// Returns true and stores frame offsets and lengths in |frame_ranges| when
// |frame| has a valid VP9 super frame index. Sets |error| to true when parsing
// fails for any reason.
bool ParseVP9SuperFrameIndex(const std::uint8_t* frame,
                             std::size_t frame_length, Ranges* frame_ranges,
                             bool* error);

// Writes |val| to |fileptr| and returns true upon success.
bool WriteUint8(std::uint8_t val, std::FILE* fileptr);

// Reads 2 bytes from |buf| and returns them as a uint16_t. Returns 0 when |buf|
// is a nullptr.
std::uint16_t ReadUint16(const std::uint8_t* buf);

}  // namespace libwebm

#endif  // LIBWEBM_COMMON_LIBWEBM_UTIL_H_
