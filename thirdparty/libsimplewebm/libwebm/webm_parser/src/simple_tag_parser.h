// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_SIMPLE_TAG_PARSER_H_
#define SRC_SIMPLE_TAG_PARSER_H_

#include "src/bool_parser.h"
#include "src/byte_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#SimpleTag
// http://www.webmproject.org/docs/container/#SimpleTag
class SimpleTagParser : public MasterValueParser<SimpleTag> {
 public:
  SimpleTagParser(std::size_t max_recursive_depth = 25)
      : MasterValueParser<SimpleTag>(
            MakeChild<StringParser>(Id::kTagName, &SimpleTag::name),
            MakeChild<StringParser>(Id::kTagLanguage, &SimpleTag::language),
            MakeChild<BoolParser>(Id::kTagDefault, &SimpleTag::is_default),
            MakeChild<StringParser>(Id::kTagString, &SimpleTag::string),
            MakeChild<BinaryParser>(Id::kTagBinary, &SimpleTag::binary),
            MakeChild<SimpleTagParser>(Id::kSimpleTag, &SimpleTag::tags,
                                       max_recursive_depth)) {}
};

}  // namespace webm

#endif  // SRC_SIMPLE_TAG_PARSER_H_
