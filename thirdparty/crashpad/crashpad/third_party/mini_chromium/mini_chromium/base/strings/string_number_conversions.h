// Copyright 2010 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_STRINGS_STRING_NUMBER_CONVERSIONS_H_
#define MINI_CHROMIUM_BASE_STRINGS_STRING_NUMBER_CONVERSIONS_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "base/strings/string_piece.h"

namespace base {

bool StringToInt(const StringPiece& input, int* output);
bool StringToUint(const StringPiece& input, unsigned int* output);
bool StringToInt64(const StringPiece& input, int64_t* output);
bool StringToUint64(const StringPiece& input, uint64_t* output);
bool StringToSizeT(const StringPiece& input, size_t* output);

bool HexStringToInt(const StringPiece& input, int* output);
bool HexStringToBytes(const std::string& input, std::vector<uint8_t>* output);

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_STRINGS_STRING_NUMBER_CONVERSIONS_H_
