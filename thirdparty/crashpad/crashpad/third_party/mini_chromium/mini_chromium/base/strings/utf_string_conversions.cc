// Copyright 2009 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/strings/utf_string_conversions.h"

#include <stdint.h>

#include <string>

#include "base/strings/string16.h"
#include "base/strings/utf_string_conversion_utils.h"

namespace {

template<typename SRC_CHAR, typename DEST_STRING>
bool ConvertUnicode(const SRC_CHAR* src,
                    size_t src_len,
                    DEST_STRING* output) {
  bool success = true;
  int32_t src_len32 = static_cast<int32_t>(src_len);
  for (int32_t i = 0; i < src_len32; i++) {
    uint32_t code_point;
    if (base::ReadUnicodeCharacter(src, src_len32, &i, &code_point)) {
      base::WriteUnicodeCharacter(code_point, output);
    } else {
      base::WriteUnicodeCharacter(0xFFFD, output);
      success = false;
    }
  }

  return success;
}

}  // namespace

namespace base {

bool UTF8ToUTF16(const char* src, size_t src_len, string16* output) {
  base::PrepareForUTF16Or32Output(src, src_len, output);
  return ConvertUnicode(src, src_len, output);
}

string16 UTF8ToUTF16(const StringPiece& utf8) {
  string16 ret;
  UTF8ToUTF16(utf8.data(), utf8.length(), &ret);
  return ret;
}

bool UTF16ToUTF8(const char16* src, size_t src_len, std::string* output) {
  base::PrepareForUTF8Output(src, src_len, output);
  return ConvertUnicode(src, src_len, output);
}

std::string UTF16ToUTF8(const StringPiece16& utf16) {
  std::string ret;
  UTF16ToUTF8(utf16.data(), utf16.length(), &ret);
  return ret;
}

}  // namespace
