// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_INFO_PARSER_H_
#define SRC_INFO_PARSER_H_

#include "src/byte_parser.h"
#include "src/date_parser.h"
#include "src/float_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Info
// http://www.webmproject.org/docs/container/#Info
class InfoParser : public MasterValueParser<Info> {
 public:
  InfoParser()
      : MasterValueParser<Info>(
            MakeChild<UnsignedIntParser>(Id::kTimecodeScale,
                                         &Info::timecode_scale),
            MakeChild<FloatParser>(Id::kDuration, &Info::duration),
            MakeChild<DateParser>(Id::kDateUtc, &Info::date_utc),
            MakeChild<StringParser>(Id::kTitle, &Info::title),
            MakeChild<StringParser>(Id::kMuxingApp, &Info::muxing_app),
            MakeChild<StringParser>(Id::kWritingApp, &Info::writing_app)) {}

 protected:
  Status OnParseCompleted(Callback* callback) override {
    return callback->OnInfo(metadata(Id::kInfo), value());
  }
};

}  // namespace webm

#endif  // SRC_INFO_PARSER_H_
