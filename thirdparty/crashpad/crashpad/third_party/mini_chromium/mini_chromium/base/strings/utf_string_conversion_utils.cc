// Copyright 2009 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/strings/utf_string_conversion_utils.h"

#include "base/third_party/icu/icu_utf.h"

namespace base {

bool ReadUnicodeCharacter(const char* src,
                          int32_t src_len,
                          int32_t* char_index,
                          uint32_t* code_point_out) {
  int32_t code_point;
  CBU8_NEXT(src, *char_index, src_len, code_point);
  *code_point_out = static_cast<uint32_t>(code_point);

  (*char_index)--;

  return IsValidCodepoint(code_point);
}

bool ReadUnicodeCharacter(const char16* src,
                          int32_t src_len,
                          int32_t* char_index,
                          uint32_t* code_point) {
  if (CBU16_IS_SURROGATE(src[*char_index])) {
    if (!CBU16_IS_SURROGATE_LEAD(src[*char_index]) ||
        *char_index + 1 >= src_len ||
        !CBU16_IS_TRAIL(src[*char_index + 1])) {
      return false;
    }

    *code_point = CBU16_GET_SUPPLEMENTARY(src[*char_index],
                                          src[*char_index + 1]);
    (*char_index)++;
  } else {
    *code_point = src[*char_index];
  }

  return IsValidCodepoint(*code_point);
}

size_t WriteUnicodeCharacter(uint32_t code_point, std::string* output) {
  if (code_point <= 0x7f) {
    output->push_back(static_cast<char>(code_point));
    return 1;
  }

  size_t char_offset = output->length();
  size_t original_char_offset = char_offset;
  output->resize(char_offset + CBU8_MAX_LENGTH);

  CBU8_APPEND_UNSAFE(&(*output)[0], char_offset, code_point);

  output->resize(char_offset);
  return char_offset - original_char_offset;
}

size_t WriteUnicodeCharacter(uint32_t code_point, string16* output) {
  if (CBU16_LENGTH(code_point) == 1) {
    output->push_back(static_cast<char16>(code_point));
    return 1;
  }
  size_t char_offset = output->length();
  output->resize(char_offset + CBU16_MAX_LENGTH);
  CBU16_APPEND_UNSAFE(&(*output)[0], char_offset, code_point);
  return CBU16_MAX_LENGTH;
}

template<typename CHAR>
void PrepareForUTF8Output(const CHAR* src,
                          size_t src_len,
                          std::string* output) {
  output->clear();
  if (src_len == 0)
    return;
  if (src[0] < 0x80) {
    output->reserve(src_len);
  } else {
    output->reserve(src_len * 3);
  }
}

template void PrepareForUTF8Output(const wchar_t*, size_t, std::string*);
template void PrepareForUTF8Output(const char16*, size_t, std::string*);

template<typename STRING>
void PrepareForUTF16Or32Output(const char* src,
                               size_t src_len,
                               STRING* output) {
  output->clear();
  if (src_len == 0)
    return;
  if (static_cast<unsigned char>(src[0]) < 0x80) {
    output->reserve(src_len);
  } else {
    output->reserve(src_len / 2);
  }
}

template void PrepareForUTF16Or32Output(const char*, size_t, string16*);

}  // namespace base
