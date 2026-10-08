// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_SEEK_PARSER_H_
#define SRC_SEEK_PARSER_H_

#include "src/id_element_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Seek
// http://www.webmproject.org/docs/container/#Seek
class SeekParser : public MasterValueParser<Seek> {
 public:
  SeekParser()
      : MasterValueParser<Seek>(
            MakeChild<IdElementParser>(Id::kSeekId, &Seek::id),
            MakeChild<UnsignedIntParser>(Id::kSeekPosition, &Seek::position)) {}

 protected:
  Status OnParseCompleted(Callback* callback) override {
    return callback->OnSeek(metadata(Id::kSeek), value());
  }
};

}  // namespace webm

#endif  // SRC_SEEK_PARSER_H_
