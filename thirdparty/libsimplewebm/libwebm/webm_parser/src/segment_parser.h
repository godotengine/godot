// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_SEGMENT_PARSER_H_
#define SRC_SEGMENT_PARSER_H_

#include <cstdint>

#include "src/master_parser.h"
#include "webm/callback.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses Segment elements from a WebM byte stream. This class adheres to the
// ElementParser interface; see element_parser.h for further documentation on
// how it should be used.
// Spec reference:
// http://matroska.org/technical/specs/index.html#Segment
// http://www.webmproject.org/docs/container/#Segment
class SegmentParser : public MasterParser {
 public:
  SegmentParser();

  Status Init(const ElementMetadata& metadata, std::uint64_t max_size) override;

  void InitAfterSeek(const Ancestory& child_ancestory,
                     const ElementMetadata& child_metadata) override;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  bool WasSkipped() const override;

 private:
  // Set to true iff Callback::OnSegmentBegin has completed.
  bool begin_done_;

  // Set to true iff the base class has completed parsing.
  bool parse_completed_;

  // The action requested by Callback::OnSegmentBegin.
  Action action_ = Action::kRead;
};

}  // namespace webm

#endif  // SRC_SEGMENT_PARSER_H_
