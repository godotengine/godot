// Copyright 2009 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_STRINGS_UTF_STRING_CONVERSION_UTILS_H_
#define MINI_CHROMIUM_BASE_STRINGS_UTF_STRING_CONVERSION_UTILS_H_

#include "base/strings/string16.h"

namespace base {

inline bool IsValidCodepoint(uint32_t code_point) {
  return code_point < 0xD800u ||
         (code_point >= 0xE000u && code_point <= 0x10FFFFu);
}

bool ReadUnicodeCharacter(const char* src,
                          int32_t src_len,
                          int32_t* char_index,
                          uint32_t* code_point_out);

bool ReadUnicodeCharacter(const char16* src,
                          int32_t src_len,
                          int32_t* char_index,
                          uint32_t* code_point);

size_t WriteUnicodeCharacter(uint32_t code_point, std::string* output);

size_t WriteUnicodeCharacter(uint32_t code_point, string16* output);

template<typename CHAR>
void PrepareForUTF8Output(const CHAR* src, size_t src_len, std::string* output);

template<typename STRING>
void PrepareForUTF16Or32Output(const char* src, size_t src_len, STRING* output);

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_STRINGS_UTF_STRING_CONVERSION_UTILS_H_
