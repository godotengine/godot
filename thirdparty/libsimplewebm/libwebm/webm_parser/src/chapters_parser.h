// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_CHAPTERS_PARSER_H_
#define SRC_CHAPTERS_PARSER_H_

#include "src/edition_entry_parser.h"
#include "src/master_parser.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Chapters
// http://www.webmproject.org/docs/container/#Chapters
class ChaptersParser : public MasterParser {
 public:
  ChaptersParser()
      : MasterParser(MakeChild<EditionEntryParser>(Id::kEditionEntry)) {}
};

}  // namespace webm

#endif  // SRC_CHAPTERS_PARSER_H_
