// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_EDITION_ENTRY_PARSER_H_
#define SRC_EDITION_ENTRY_PARSER_H_

#include "src/chapter_atom_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#EditionEntry
// http://www.webmproject.org/docs/container/#EditionEntry
class EditionEntryParser : public MasterValueParser<EditionEntry> {
 public:
  EditionEntryParser()
      : MasterValueParser<EditionEntry>(MakeChild<ChapterAtomParser>(
            Id::kChapterAtom, &EditionEntry::atoms)) {}

 protected:
  Status OnParseCompleted(Callback* callback) override {
    return callback->OnEditionEntry(metadata(Id::kEditionEntry), value());
  }
};

}  // namespace webm

#endif  // SRC_EDITION_ENTRY_PARSER_H_
