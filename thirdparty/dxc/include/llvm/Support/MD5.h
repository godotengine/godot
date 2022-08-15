/*
 * This code is derived from (original license follows):
 *
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD5 Message-Digest Algorithm (RFC 1321).
 *
 * Homepage:
 * http://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * Author:
 * Alexander Peslyak, better known as Solar Designer <solar at openwall.com>
 *
 * This software was written by Alexander Peslyak in 2001.  No copyright is
 * claimed, and the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2001 Alexander Peslyak and it is hereby released to the
 * general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * See md5.c for more information.
 */

#ifndef LLVM_SUPPORT_MD5_H
#define LLVM_SUPPORT_MD5_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class MD5 {
  // Any 32-bit or wider unsigned integer data type will do.
  typedef uint32_t MD5_u32plus;

  MD5_u32plus a, b, c, d;
  MD5_u32plus hi, lo;
  uint8_t buffer[64];
  MD5_u32plus block[16];

public:
  typedef uint8_t MD5Result[16];

  MD5();

  /// \brief Updates the hash for the byte stream provided.
  void update(ArrayRef<uint8_t> Data);

  /// \brief Updates the hash for the StringRef provided.
  void update(StringRef Str);

  /// \brief Finishes off the hash and puts the result in result.
  void final(MD5Result &Result);

  /// \brief Translates the bytes in \p Res to a hex string that is
  /// deposited into \p Str. The result will be of length 32.
  static void stringifyResult(MD5Result &Result, SmallString<32> &Str);

private:
  const uint8_t *body(ArrayRef<uint8_t> Data);
};

}

#endif
