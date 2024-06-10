// -*- mode: c++ -*-

// Copyright 2010 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Original author: Jim Blandy <jim@mozilla.com> <jimb@red-bean.com>

// byteswap.h: Overloaded functions for conveniently byteswapping values.

#ifndef COMMON_MAC_BYTESWAP_H_
#define COMMON_MAC_BYTESWAP_H_

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>

static inline uint16_t ByteSwap(uint16_t v) { return OSSwapInt16(v); }
static inline uint32_t ByteSwap(uint32_t v) { return OSSwapInt32(v); }
static inline uint64_t ByteSwap(uint64_t v) { return OSSwapInt64(v); }
static inline int16_t  ByteSwap(int16_t  v) { return OSSwapInt16(v); }
static inline int32_t  ByteSwap(int32_t  v) { return OSSwapInt32(v); }
static inline int64_t  ByteSwap(int64_t  v) { return OSSwapInt64(v); }

#elif defined(__linux__)
// For NXByteOrder
#include <architecture/byte_order.h>
#include <stdint.h>
#include <endian.h>
#include_next <byteswap.h>

static inline uint16_t ByteSwap(uint16_t v) { return bswap_16(v); }
static inline uint32_t ByteSwap(uint32_t v) { return bswap_32(v); }
static inline uint64_t ByteSwap(uint64_t v) { return bswap_64(v); }
static inline int16_t  ByteSwap(int16_t  v) { return bswap_16(v); }
static inline int32_t  ByteSwap(int32_t  v) { return bswap_32(v); }
static inline int64_t  ByteSwap(int64_t  v) { return bswap_64(v); }

static inline NXByteOrder NXHostByteOrder() {
#ifdef __LITTLE_ENDIAN
  return NX_LittleEndian;
#else
  return NX_BigEndian;
#endif
}

#endif  // __APPLE__

#endif  // COMMON_MAC_BYTESWAP_H_
