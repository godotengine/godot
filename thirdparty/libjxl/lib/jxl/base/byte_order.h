// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_BYTE_ORDER_H_
#define LIB_JXL_BASE_BYTE_ORDER_H_

#include <jxl/types.h>
#include <stdint.h>
#include <string.h>  // memcpy

#include "lib/jxl/base/compiler_specific.h"

#if JXL_COMPILER_MSVC
#include <intrin.h>  // _byteswap_*
#endif

#if (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#define JXL_BYTE_ORDER_LITTLE 1
#else
// This means that we don't know that the byte order is little endian, in
// this case we use endian-neutral code that works for both little- and
// big-endian.
#define JXL_BYTE_ORDER_LITTLE 0
#endif

// Returns whether the system is little-endian (least-significant byte first).
#if JXL_BYTE_ORDER_LITTLE
static constexpr bool IsLittleEndian() { return true; }
#else
static inline bool IsLittleEndian() {
  const uint32_t multibyte = 1;
  uint8_t byte;
  memcpy(&byte, &multibyte, 1);
  return byte == 1;
}
#endif

static inline bool SwapEndianness(JxlEndianness endianness) {
  return ((endianness == JXL_BIG_ENDIAN && IsLittleEndian()) ||
          (endianness == JXL_LITTLE_ENDIAN && !IsLittleEndian()));
}

#if JXL_COMPILER_MSVC
#define JXL_BSWAP16(x) _byteswap_ushort(x)
#define JXL_BSWAP32(x) _byteswap_ulong(x)
#define JXL_BSWAP64(x) _byteswap_uint64(x)
#else
#define JXL_BSWAP16(x) __builtin_bswap16(x)
#define JXL_BSWAP32(x) __builtin_bswap32(x)
#define JXL_BSWAP64(x) __builtin_bswap64(x)
#endif

static JXL_INLINE uint32_t LoadBE16(const uint8_t* p) {
  const uint32_t byte1 = p[0];
  const uint32_t byte0 = p[1];
  return (byte1 << 8) | byte0;
}

static JXL_INLINE uint32_t LoadLE16(const uint8_t* p) {
  const uint32_t byte0 = p[0];
  const uint32_t byte1 = p[1];
  return (byte1 << 8) | byte0;
}

static JXL_INLINE uint32_t LoadBE32(const uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  uint32_t big;
  memcpy(&big, p, 4);
  return JXL_BSWAP32(big);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  const uint32_t byte3 = p[0];
  const uint32_t byte2 = p[1];
  const uint32_t byte1 = p[2];
  const uint32_t byte0 = p[3];
  return (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0;
#endif
}

static JXL_INLINE uint64_t LoadBE64(const uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  uint64_t big;
  memcpy(&big, p, 8);
  return JXL_BSWAP64(big);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  const uint64_t byte7 = p[0];
  const uint64_t byte6 = p[1];
  const uint64_t byte5 = p[2];
  const uint64_t byte4 = p[3];
  const uint64_t byte3 = p[4];
  const uint64_t byte2 = p[5];
  const uint64_t byte1 = p[6];
  const uint64_t byte0 = p[7];
  return (byte7 << 56ull) | (byte6 << 48ull) | (byte5 << 40ull) |
         (byte4 << 32ull) | (byte3 << 24ull) | (byte2 << 16ull) |
         (byte1 << 8ull) | byte0;
#endif
}

static JXL_INLINE uint32_t LoadLE32(const uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  uint32_t little;
  memcpy(&little, p, 4);
  return little;
#else
  // Byte-order-independent - can't assume this machine is big endian.
  const uint32_t byte0 = p[0];
  const uint32_t byte1 = p[1];
  const uint32_t byte2 = p[2];
  const uint32_t byte3 = p[3];
  return (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0;
#endif
}

static JXL_INLINE uint64_t LoadLE64(const uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  uint64_t little;
  memcpy(&little, p, 8);
  return little;
#else
  // Byte-order-independent - can't assume this machine is big endian.
  const uint64_t byte0 = p[0];
  const uint64_t byte1 = p[1];
  const uint64_t byte2 = p[2];
  const uint64_t byte3 = p[3];
  const uint64_t byte4 = p[4];
  const uint64_t byte5 = p[5];
  const uint64_t byte6 = p[6];
  const uint64_t byte7 = p[7];
  return (byte7 << 56) | (byte6 << 48) | (byte5 << 40) | (byte4 << 32) |
         (byte3 << 24) | (byte2 << 16) | (byte1 << 8) | byte0;
#endif
}

// Loads a Big-Endian float
static JXL_INLINE float LoadBEFloat(const uint8_t* p) {
  uint32_t u = LoadBE32(p);
  float result;
  memcpy(&result, &u, 4);
  return result;
}

// Loads a Little-Endian float
static JXL_INLINE float LoadLEFloat(const uint8_t* p) {
  uint32_t u = LoadLE32(p);
  float result;
  memcpy(&result, &u, 4);
  return result;
}

static JXL_INLINE void StoreBE16(const uint32_t native, uint8_t* p) {
  p[0] = (native >> 8) & 0xFF;
  p[1] = native & 0xFF;
}

static JXL_INLINE void StoreLE16(const uint32_t native, uint8_t* p) {
  p[1] = (native >> 8) & 0xFF;
  p[0] = native & 0xFF;
}

static JXL_INLINE void StoreBE32(const uint32_t native, uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  const uint32_t big = JXL_BSWAP32(native);
  memcpy(p, &big, 4);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  p[0] = native >> 24;
  p[1] = (native >> 16) & 0xFF;
  p[2] = (native >> 8) & 0xFF;
  p[3] = native & 0xFF;
#endif
}

static JXL_INLINE void StoreBE64(const uint64_t native, uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  const uint64_t big = JXL_BSWAP64(native);
  memcpy(p, &big, 8);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  p[0] = native >> 56ull;
  p[1] = (native >> 48ull) & 0xFF;
  p[2] = (native >> 40ull) & 0xFF;
  p[3] = (native >> 32ull) & 0xFF;
  p[4] = (native >> 24ull) & 0xFF;
  p[5] = (native >> 16ull) & 0xFF;
  p[6] = (native >> 8ull) & 0xFF;
  p[7] = native & 0xFF;
#endif
}

static JXL_INLINE void StoreLE32(const uint32_t native, uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  const uint32_t little = native;
  memcpy(p, &little, 4);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  p[3] = native >> 24;
  p[2] = (native >> 16) & 0xFF;
  p[1] = (native >> 8) & 0xFF;
  p[0] = native & 0xFF;
#endif
}

static JXL_INLINE void StoreLE64(const uint64_t native, uint8_t* p) {
#if JXL_BYTE_ORDER_LITTLE
  const uint64_t little = native;
  memcpy(p, &little, 8);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  p[7] = native >> 56;
  p[6] = (native >> 48) & 0xFF;
  p[5] = (native >> 40) & 0xFF;
  p[4] = (native >> 32) & 0xFF;
  p[3] = (native >> 24) & 0xFF;
  p[2] = (native >> 16) & 0xFF;
  p[1] = (native >> 8) & 0xFF;
  p[0] = native & 0xFF;
#endif
}

static JXL_INLINE float BSwapFloat(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  uint32_t uswap = JXL_BSWAP32(u);
  float xswap;
  memcpy(&xswap, &uswap, 4);
  return xswap;
}

// Big/Little Endian order.
struct OrderBE {};
struct OrderLE {};

// Wrappers for calling from generic code.
static JXL_INLINE void Store16(OrderBE /*tag*/, const uint32_t native,
                               uint8_t* p) {
  StoreBE16(native, p);
}

static JXL_INLINE void Store16(OrderLE /*tag*/, const uint32_t native,
                               uint8_t* p) {
  StoreLE16(native, p);
}

static JXL_INLINE void Store32(OrderBE /*tag*/, const uint32_t native,
                               uint8_t* p) {
  StoreBE32(native, p);
}

static JXL_INLINE void Store32(OrderLE /*tag*/, const uint32_t native,
                               uint8_t* p) {
  StoreLE32(native, p);
}

static JXL_INLINE uint32_t Load16(OrderBE /*tag*/, const uint8_t* p) {
  return LoadBE16(p);
}

static JXL_INLINE uint32_t Load16(OrderLE /*tag*/, const uint8_t* p) {
  return LoadLE16(p);
}

static JXL_INLINE uint32_t Load32(OrderBE /*tag*/, const uint8_t* p) {
  return LoadBE32(p);
}

static JXL_INLINE uint32_t Load32(OrderLE /*tag*/, const uint8_t* p) {
  return LoadLE32(p);
}

#endif  // LIB_JXL_BASE_BYTE_ORDER_H_
