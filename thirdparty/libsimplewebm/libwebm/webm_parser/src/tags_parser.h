// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_TAGS_PARSER_H_
#define SRC_TAGS_PARSER_H_

#include "src/master_parser.h"
#include "src/tag_parser.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Tags
// http://www.webmproject.org/docs/container/#Tags
class TagsParser : public MasterParser {
 public:
  TagsParser() : MasterParser(MakeChild<TagParser>(Id::kTag)) {}
};

}  // namespace webm

#endif  // SRC_TAGS_PARSER_H_
