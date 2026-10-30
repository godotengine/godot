// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_CONTENT_ENC_AES_SETTINGS_PARSER_H_
#define SRC_CONTENT_ENC_AES_SETTINGS_PARSER_H_

#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://www.webmproject.org/docs/webm-encryption/#42-new-matroskawebm-elements
class ContentEncAesSettingsParser
    : public MasterValueParser<ContentEncAesSettings> {
 public:
  ContentEncAesSettingsParser()
      : MasterValueParser<ContentEncAesSettings>(
            MakeChild<IntParser<AesSettingsCipherMode>>(
                Id::kAesSettingsCipherMode,
                &ContentEncAesSettings::aes_settings_cipher_mode)) {}
};

}  // namespace webm

#endif  // SRC_CONTENT_ENC_AES_SETTINGS_PARSER_H_
