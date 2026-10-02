// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ICC_CODEC_COMMON_H_
#define LIB_JXL_ICC_CODEC_COMMON_H_

// Compressed representation of ICC profiles.

#include <array>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"

namespace jxl {

class PaddedBytes;

static constexpr size_t kICCHeaderSize = 128;

typedef std::array<uint8_t, 4> Tag;

static const Tag kAcspTag = {{'a', 'c', 's', 'p'}};
static const Tag kBkptTag = {{'b', 'k', 'p', 't'}};
static const Tag kBtrcTag = {{'b', 'T', 'R', 'C'}};
static const Tag kBxyzTag = {{'b', 'X', 'Y', 'Z'}};
static const Tag kChadTag = {{'c', 'h', 'a', 'd'}};
static const Tag kChrmTag = {{'c', 'h', 'r', 'm'}};
static const Tag kCprtTag = {{'c', 'p', 'r', 't'}};
static const Tag kCurvTag = {{'c', 'u', 'r', 'v'}};
static const Tag kDescTag = {{'d', 'e', 's', 'c'}};
static const Tag kDmddTag = {{'d', 'm', 'd', 'd'}};
static const Tag kDmndTag = {{'d', 'm', 'n', 'd'}};
static const Tag kGbd_Tag = {{'g', 'b', 'd', ' '}};
static const Tag kGtrcTag = {{'g', 'T', 'R', 'C'}};
static const Tag kGxyzTag = {{'g', 'X', 'Y', 'Z'}};
static const Tag kKtrcTag = {{'k', 'T', 'R', 'C'}};
static const Tag kKxyzTag = {{'k', 'X', 'Y', 'Z'}};
static const Tag kLumiTag = {{'l', 'u', 'm', 'i'}};
static const Tag kMab_Tag = {{'m', 'A', 'B', ' '}};
static const Tag kMba_Tag = {{'m', 'B', 'A', ' '}};
static const Tag kMlucTag = {{'m', 'l', 'u', 'c'}};
static const Tag kMntrTag = {{'m', 'n', 't', 'r'}};
static const Tag kParaTag = {{'p', 'a', 'r', 'a'}};
static const Tag kRgb_Tag = {{'R', 'G', 'B', ' '}};
static const Tag kRtrcTag = {{'r', 'T', 'R', 'C'}};
static const Tag kRxyzTag = {{'r', 'X', 'Y', 'Z'}};
static const Tag kSf32Tag = {{'s', 'f', '3', '2'}};
static const Tag kTextTag = {{'t', 'e', 'x', 't'}};
static const Tag kVcgtTag = {{'v', 'c', 'g', 't'}};
static const Tag kWtptTag = {{'w', 't', 'p', 't'}};
static const Tag kXyz_Tag = {{'X', 'Y', 'Z', ' '}};

// Tag names focused on RGB and GRAY monitor profiles
static constexpr size_t kNumTagStrings = 17;
static constexpr const Tag* kTagStrings[kNumTagStrings] = {
    &kCprtTag, &kWtptTag, &kBkptTag, &kRxyzTag, &kGxyzTag, &kBxyzTag,
    &kKxyzTag, &kRtrcTag, &kGtrcTag, &kBtrcTag, &kKtrcTag, &kChadTag,
    &kDescTag, &kChrmTag, &kDmndTag, &kDmddTag, &kLumiTag};

static constexpr size_t kCommandTagUnknown = 1;
static constexpr size_t kCommandTagTRC = 2;
static constexpr size_t kCommandTagXYZ = 3;
static constexpr size_t kCommandTagStringFirst = 4;

// Tag types focused on RGB and GRAY monitor profiles
static constexpr size_t kNumTypeStrings = 8;
static constexpr const Tag* kTypeStrings[kNumTypeStrings] = {
    &kXyz_Tag, &kDescTag, &kTextTag, &kMlucTag,
    &kParaTag, &kCurvTag, &kSf32Tag, &kGbd_Tag};

static constexpr size_t kCommandInsert = 1;
static constexpr size_t kCommandShuffle2 = 2;
static constexpr size_t kCommandShuffle4 = 3;
static constexpr size_t kCommandPredict = 4;
static constexpr size_t kCommandXYZ = 10;
static constexpr size_t kCommandTypeStartFirst = 16;

static constexpr size_t kFlagBitOffset = 64;
static constexpr size_t kFlagBitSize = 128;

static constexpr size_t kNumICCContexts = 41;

uint32_t DecodeUint32(const uint8_t* data, size_t size, size_t pos);
Status AppendUint32(uint32_t value, PaddedBytes* data);
Tag DecodeKeyword(const uint8_t* data, size_t size, size_t pos);
void EncodeKeyword(const Tag& keyword, uint8_t* data, size_t size, size_t pos);
Status AppendKeyword(const Tag& keyword, PaddedBytes* data);

// Checks if a + b > size, taking possible integer overflow into account.
Status CheckOutOfBounds(uint64_t a, uint64_t b, uint64_t size);
Status CheckIs32Bit(uint64_t v);

std::array<uint8_t, kICCHeaderSize> ICCInitialHeaderPrediction(uint32_t size);
void ICCPredictHeader(const uint8_t* icc, size_t size, uint8_t* header,
                      size_t pos);
uint8_t LinearPredictICCValue(const uint8_t* data, size_t start, size_t i,
                              size_t stride, size_t width, int order);
size_t ICCANSContext(size_t i, size_t b1, size_t b2);

}  // namespace jxl

#endif  // LIB_JXL_ICC_CODEC_COMMON_H_
