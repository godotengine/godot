// Copyright 2007 Google LLC
// Author: liuli@google.com (Liu Li)
#ifndef COMMON_MD5_H__
#define COMMON_MD5_H__

#include <stddef.h>
#include <stdint.h>

namespace google_breakpad {

typedef uint32_t u32;
typedef uint8_t u8;

struct MD5Context {
  u32 buf[4];
  u32 bits[2];
  u8 in[64];
};

void MD5Init(struct MD5Context *ctx);

void MD5Update(struct MD5Context *ctx, unsigned char const *buf, size_t len);

void MD5Final(unsigned char digest[16], struct MD5Context *ctx);

}  // namespace google_breakpad

#endif  // COMMON_MD5_H__
