// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_VOID_PARSER_H_
#define SRC_VOID_PARSER_H_

#include <cstdint>

#include "src/element_parser.h"
#include "webm/callback.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses a Void element by delegating to Callback::OnVoid.
// Spec reference:
// http://matroska.org/technical/specs/index.html#Void
// http://www.webmproject.org/docs/container/#Void
class VoidParser : public ElementParser {
 public:
  Status Init(const ElementMetadata& metadata, std::uint64_t max_size) override;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

 private:
  // The metadata for this element.
  ElementMetadata metadata_;

  // The number of bytes remaining that have not been read in the element.
  std::uint64_t bytes_remaining_ = 0;
};

}  // namespace webm

#endif  // SRC_VOID_PARSER_H_
