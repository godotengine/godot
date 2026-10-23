// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_CONTENT_ENCRYPTION_PARSER_H_
#define SRC_CONTENT_ENCRYPTION_PARSER_H_

#include "src/byte_parser.h"
#include "src/content_enc_aes_settings_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#ContentEncryption
// http://www.webmproject.org/docs/container/#ContentEncryption
class ContentEncryptionParser : public MasterValueParser<ContentEncryption> {
 public:
  ContentEncryptionParser()
      : MasterValueParser<ContentEncryption>(
            MakeChild<IntParser<ContentEncAlgo>>(Id::kContentEncAlgo,
                                                 &ContentEncryption::algorithm),
            MakeChild<BinaryParser>(Id::kContentEncKeyId,
                                    &ContentEncryption::key_id),
            MakeChild<ContentEncAesSettingsParser>(
                Id::kContentEncAesSettings, &ContentEncryption::aes_settings)) {
  }
};

}  // namespace webm

#endif  // SRC_CONTENT_ENCRYPTION_PARSER_H_
