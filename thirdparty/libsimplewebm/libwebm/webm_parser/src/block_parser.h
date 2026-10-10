// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_BLOCK_PARSER_H_
#define SRC_BLOCK_PARSER_H_

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "src/block_header_parser.h"
#include "src/element_parser.h"
#include "src/var_int_parser.h"
#include "webm/callback.h"
#include "webm/dom_types.h"
#include "webm/element.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses Block and SimpleBlock elements. It is recommended to use the
// BlockParser and SimpleBlockParser aliases.
// Spec reference:
// http://matroska.org/technical/specs/index.html#Block
// http://matroska.org/technical/specs/index.html#SimpleBlock
// http://www.webmproject.org/docs/container/#SimpleBlock
// http://www.webmproject.org/docs/container/#Block
// http://matroska.org/technical/specs/index.html#block_structure
// http://matroska.org/technical/specs/index.html#simpleblock_structure
template <typename T>
class BasicBlockParser : public ElementParser {
  static_assert(std::is_same<T, Block>::value ||
                    std::is_same<T, SimpleBlock>::value,
                "T must be Block or SimpleBlock");

 public:
  Status Init(const ElementMetadata& metadata, std::uint64_t max_size) override;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  bool WasSkipped() const override;

  // Gets the parsed block header information. The frames are not included. This
  // must not be called until the parse has been successfully completed.
  const T& value() const {
    assert(state_ == State::kDone);
    return value_;
  }

  // Gets the parsed block header information. The frames are not included. This
  // must not be called until the parse has been successfully completed.
  T* mutable_value() {
    assert(state_ == State::kDone);
    return &value_;
  }

 private:
  // The number of header bytes read (header meaning everything before the
  // frames).
  std::uint64_t header_bytes_read_ = 0;

  // The parsed header value for the element.
  T value_{};

  // Metadata for the frame that is currently being read.
  FrameMetadata frame_metadata_;

  // Parser for parsing header metadata that is common between Block and
  // SimpleBlock.
  BlockHeaderParser header_parser_;

  // Parser for parsing unsigned EBML variable-sized integers.
  VarIntParser uint_parser_;

  // The current lace size when parsing Xiph lace sizes.
  std::uint64_t xiph_lace_size_ = 0;

  // Lace (frame) sizes, where each entry represents the size of a frame.
  std::vector<std::uint64_t> lace_sizes_;

  // The current index into lace_sizes_ for the current frame being read.
  std::size_t current_lace_ = 0;

  // Parsing states for the finite-state machine.
  enum class State {
    /* clang-format off */
    // State                        Transitions to state        When
    kReadingHeader,              // kGettingAction              no lacing
                                 // kReadingLaceCount           yes lacing
    kReadingLaceCount,           // kGettingAction              no errors
    kGettingAction,              // kSkipping                   action == skip
                                 // kValidatingSize             no lacing
                                 // kReadingXiphLaceSizes       xiph lacing
                                 // kReadingFirstEbmlLaceSize   ebml lacing
                                 // kCalculatingFixedLaceSizes  fixed lacing
    kReadingXiphLaceSizes,       // kValidatingSize             all sizes read
    kReadingFirstEbmlLaceSize,   // kReadingEbmlLaceSizes       first size read
    kReadingEbmlLaceSizes,       // kValidatingSize             all sizes read
    kCalculatingFixedLaceSizes,  // kReadingFrames              no errors
    kValidatingSize,             // kReadingFrames              no errors
    kSkipping,                   // No transitions from here (must call Init)
    kReadingFrames,              // kDone                       all frames read
    kDone,                       // No transitions from here (must call Init)
    /* clang-format on */
  };

  // The current state of the parser.
  State state_ = State::kReadingHeader;
};

extern template class BasicBlockParser<Block>;
extern template class BasicBlockParser<SimpleBlock>;

using BlockParser = BasicBlockParser<Block>;
using SimpleBlockParser = BasicBlockParser<SimpleBlock>;

}  // namespace webm

#endif  // SRC_BLOCK_PARSER_H_
