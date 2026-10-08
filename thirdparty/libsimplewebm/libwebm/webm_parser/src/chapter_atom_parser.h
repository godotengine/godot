// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_CHAPTER_ATOM_PARSER_H_
#define SRC_CHAPTER_ATOM_PARSER_H_

#include "src/byte_parser.h"
#include "src/chapter_display_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#ChapterAtom
// http://www.webmproject.org/docs/container/#ChapterAtom
class ChapterAtomParser : public MasterValueParser<ChapterAtom> {
 public:
  explicit ChapterAtomParser(std::size_t max_recursive_depth = 25)
      : MasterValueParser<ChapterAtom>(
            MakeChild<UnsignedIntParser>(Id::kChapterUid, &ChapterAtom::uid),
            MakeChild<StringParser>(Id::kChapterStringUid,
                                    &ChapterAtom::string_uid),
            MakeChild<UnsignedIntParser>(Id::kChapterTimeStart,
                                         &ChapterAtom::time_start),
            MakeChild<UnsignedIntParser>(Id::kChapterTimeEnd,
                                         &ChapterAtom::time_end),
            MakeChild<ChapterDisplayParser>(Id::kChapterDisplay,
                                            &ChapterAtom::displays),
            MakeChild<ChapterAtomParser>(Id::kChapterAtom, &ChapterAtom::atoms,
                                         max_recursive_depth)) {}
};

}  // namespace webm

#endif  // SRC_CHAPTER_ATOM_PARSER_H_
