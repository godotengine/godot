// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include "common/webm_endian.h"

#include <stdint.h>

namespace {

// Swaps unsigned 32 bit values to big endian if needed. Returns |value|
// unmodified if architecture is big endian. Returns swapped bytes of |value|
// if architecture is little endian. Returns 0 otherwise.
uint32_t swap32_check_little_endian(uint32_t value) {
  // Check endianness.
  union {
    uint32_t val32;
    uint8_t c[4];
  } check;
  check.val32 = 0x01234567U;

  // Check for big endian.
  if (check.c[3] == 0x67)
    return value;

  // Check for not little endian.
  if (check.c[0] != 0x67)
    return 0;

  return value << 24 | ((value << 8) & 0x0000FF00U) |
         ((value >> 8) & 0x00FF0000U) | value >> 24;
}

// Swaps unsigned 64 bit values to big endian if needed. Returns |value|
// unmodified if architecture is big endian. Returns swapped bytes of |value|
// if architecture is little endian. Returns 0 otherwise.
uint64_t swap64_check_little_endian(uint64_t value) {
  // Check endianness.
  union {
    uint64_t val64;
    uint8_t c[8];
  } check;
  check.val64 = 0x0123456789ABCDEFULL;

  // Check for big endian.
  if (check.c[7] == 0xEF)
    return value;

  // Check for not little endian.
  if (check.c[0] != 0xEF)
    return 0;

  return value << 56 | ((value << 40) & 0x00FF000000000000ULL) |
         ((value << 24) & 0x0000FF0000000000ULL) |
         ((value << 8) & 0x000000FF00000000ULL) |
         ((value >> 8) & 0x00000000FF000000ULL) |
         ((value >> 24) & 0x0000000000FF0000ULL) |
         ((value >> 40) & 0x000000000000FF00ULL) | value >> 56;
}

}  // namespace

namespace libwebm {

uint32_t host_to_bigendian(uint32_t value) {
  return swap32_check_little_endian(value);
}

uint32_t bigendian_to_host(uint32_t value) {
  return swap32_check_little_endian(value);
}

uint64_t host_to_bigendian(uint64_t value) {
  return swap64_check_little_endian(value);
}

uint64_t bigendian_to_host(uint64_t value) {
  return swap64_check_little_endian(value);
}

}  // namespace libwebm
