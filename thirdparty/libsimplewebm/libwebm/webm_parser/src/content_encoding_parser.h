// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_CONTENT_ENCODING_PARSER_H_
#define SRC_CONTENT_ENCODING_PARSER_H_

#include "src/content_encryption_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#ContentEncoding
// http://www.webmproject.org/docs/container/#ContentEncoding
class ContentEncodingParser : public MasterValueParser<ContentEncoding> {
 public:
  ContentEncodingParser()
      : MasterValueParser<ContentEncoding>(
            MakeChild<UnsignedIntParser>(Id::kContentEncodingOrder,
                                         &ContentEncoding::order),
            MakeChild<UnsignedIntParser>(Id::kContentEncodingScope,
                                         &ContentEncoding::scope),
            MakeChild<IntParser<ContentEncodingType>>(Id::kContentEncodingType,
                                                      &ContentEncoding::type),
            MakeChild<ContentEncryptionParser>(Id::kContentEncryption,
                                               &ContentEncoding::encryption)) {}
};

}  // namespace webm

#endif  // SRC_CONTENT_ENCODING_PARSER_H_
