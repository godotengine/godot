// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/icc_codec_common.h"

#include <cstdint>
#include <tuple>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/padded_bytes.h"

namespace jxl {
namespace {
uint8_t ByteKind1(uint8_t b) {
  if ('a' <= b && b <= 'z') return 0;
  if ('A' <= b && b <= 'Z') return 0;
  if ('0' <= b && b <= '9') return 1;
  if (b == '.' || b == ',') return 1;
  if (b == 0) return 2;
  if (b == 1) return 3;
  if (b < 16) return 4;
  if (b == 255) return 6;
  if (b > 240) return 5;
  return 7;
}

uint8_t ByteKind2(uint8_t b) {
  if ('a' <= b && b <= 'z') return 0;
  if ('A' <= b && b <= 'Z') return 0;
  if ('0' <= b && b <= '9') return 1;
  if (b == '.' || b == ',') return 1;
  if (b < 16) return 2;
  if (b > 240) return 3;
  return 4;
}

template <typename T>
T PredictValue(T p1, T p2, T p3, int order) {
  if (order == 0) return p1;
  if (order == 1) return 2 * p1 - p2;
  if (order == 2) return 3 * p1 - 3 * p2 + p3;
  return 0;
}
}  // namespace

uint32_t DecodeUint32(const uint8_t* data, size_t size, size_t pos) {
  return pos + 4 > size ? 0 : LoadBE32(data + pos);
}

Status AppendUint32(uint32_t value, PaddedBytes* data) {
  size_t pos = data->size();
  JXL_RETURN_IF_ERROR(data->resize(pos + 4));
  StoreBE32(value, data->data() + pos);
  return true;
}

typedef std::array<uint8_t, 4> Tag;

Tag DecodeKeyword(const uint8_t* data, size_t size, size_t pos) {
  if (pos + 4 > size) return {{' ', ' ', ' ', ' '}};
  return {{data[pos], data[pos + 1], data[pos + 2], data[pos + 3]}};
}

void EncodeKeyword(const Tag& keyword, uint8_t* data, size_t size, size_t pos) {
  if (keyword.size() != 4 || pos + 3 >= size) return;
  for (size_t i = 0; i < 4; ++i) data[pos + i] = keyword[i];
}

Status AppendKeyword(const Tag& keyword, PaddedBytes* data) {
  static_assert(std::tuple_size<Tag>{} == 4);
  return data->append(keyword);
}

// Checks if a + b > size, taking possible integer overflow into account.
Status CheckOutOfBounds(uint64_t a, uint64_t b, uint64_t size) {
  uint64_t pos = a + b;
  if (pos > size) return JXL_FAILURE("Out of bounds");
  if (pos < a) return JXL_FAILURE("Out of bounds");  // overflow happened
  return true;
}

Status CheckIs32Bit(uint64_t v) {
  static constexpr const uint64_t kUpper32 = ~static_cast<uint64_t>(0xFFFFFFFF);
  if ((v & kUpper32) != 0) return JXL_FAILURE("32-bit value expected");
  return true;
}

const std::array<uint8_t, kICCHeaderSize> kIccInitialHeaderPrediction = {
    0,   0,   0,   0,   0,   0,   0,   0,   4, 0, 0, 0, 'm', 'n', 't', 'r',
    'R', 'G', 'B', ' ', 'X', 'Y', 'Z', ' ', 0, 0, 0, 0, 0,   0,   0,   0,
    0,   0,   0,   0,   'a', 'c', 's', 'p', 0, 0, 0, 0, 0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   246, 214, 0, 1, 0, 0, 0,   0,   211, 45,
    0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
};

std::array<uint8_t, kICCHeaderSize> ICCInitialHeaderPrediction(uint32_t size) {
  std::array<uint8_t, kICCHeaderSize> copy(kIccInitialHeaderPrediction);
  StoreBE32(size, copy.data());
  return copy;
}

void ICCPredictHeader(const uint8_t* icc, size_t size, uint8_t* header,
                      size_t pos) {
  if (pos == 8 && size >= 8) {
    header[80] = icc[4];
    header[81] = icc[5];
    header[82] = icc[6];
    header[83] = icc[7];
  }
  if (pos == 41 && size >= 41) {
    if (icc[40] == 'A') {
      header[41] = 'P';
      header[42] = 'P';
      header[43] = 'L';
    }
    if (icc[40] == 'M') {
      header[41] = 'S';
      header[42] = 'F';
      header[43] = 'T';
    }
  }
  if (pos == 42 && size >= 42) {
    if (icc[40] == 'S' && icc[41] == 'G') {
      header[42] = 'I';
      header[43] = ' ';
    }
    if (icc[40] == 'S' && icc[41] == 'U') {
      header[42] = 'N';
      header[43] = 'W';
    }
  }
}

// Predicts a value with linear prediction of given order (0-2), for integers
// with width bytes and given stride in bytes between values.
// The start position is at start + i, and the relevant modulus of i describes
// which byte of the multi-byte integer is being handled.
// The value start + i must be at least stride * 4.
uint8_t LinearPredictICCValue(const uint8_t* data, size_t start, size_t i,
                              size_t stride, size_t width, int order) {
  size_t pos = start + i;
  if (width == 1) {
    uint8_t p1 = data[pos - stride];
    uint8_t p2 = data[pos - stride * 2];
    uint8_t p3 = data[pos - stride * 3];
    return PredictValue(p1, p2, p3, order);
  } else if (width == 2) {
    size_t p = start + (i & ~1);
    uint16_t p1 = (data[p - stride * 1] << 8) + data[p - stride * 1 + 1];
    uint16_t p2 = (data[p - stride * 2] << 8) + data[p - stride * 2 + 1];
    uint16_t p3 = (data[p - stride * 3] << 8) + data[p - stride * 3 + 1];
    uint16_t pred = PredictValue(p1, p2, p3, order);
    return (i & 1) ? (pred & 255) : ((pred >> 8) & 255);
  } else {
    size_t p = start + (i & ~3);
    uint32_t p1 = DecodeUint32(data, pos, p - stride);
    uint32_t p2 = DecodeUint32(data, pos, p - stride * 2);
    uint32_t p3 = DecodeUint32(data, pos, p - stride * 3);
    uint32_t pred = PredictValue(p1, p2, p3, order);
    unsigned shiftbytes = 3 - (i & 3);
    return (pred >> (shiftbytes * 8)) & 255;
  }
}

size_t ICCANSContext(size_t i, size_t b1, size_t b2) {
  if (i <= 128) return 0;
  return 1 + ByteKind1(b1) + ByteKind2(b2) * 8;
}

}  // namespace jxl
