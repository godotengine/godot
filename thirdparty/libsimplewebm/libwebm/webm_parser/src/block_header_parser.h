// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_BLOCK_HEADER_PARSER_H_
#define SRC_BLOCK_HEADER_PARSER_H_

#include <cassert>
#include <cstdint>

#include "src/parser.h"
#include "src/var_int_parser.h"
#include "webm/callback.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

struct BlockHeader {
  std::uint64_t track_number;
  std::int16_t timecode;
  std::uint8_t flags;

  bool operator==(const BlockHeader& other) const {
    return track_number == other.track_number && timecode == other.timecode &&
           flags == other.flags;
  }
};

class BlockHeaderParser : public Parser {
 public:
  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  // Gets the parsed block header information. This must not be called until the
  // parse has been successfully completed.
  const BlockHeader& value() const {
    assert(state_ == State::kDone);
    return value_;
  }

 private:
  BlockHeader value_;

  VarIntParser uint_parser_;

  int timecode_bytes_remaining_ = 2;

  enum class State {
    /* clang-format off */
    // State                 Transitions to state  When
    kReadingTrackNumber,  // kReadingTimecode      track parsed
    kReadingTimecode,     // kReadingFlags         timecode parsed
    kReadingFlags,        // kDone                 flags parsed
    kDone,                // No transitions from here (must call Init)
    /* clang-format on */
  } state_ = State::kReadingTrackNumber;
};

}  // namespace webm

#endif  // SRC_BLOCK_HEADER_PARSER_H_
