//===--- Compression.cpp - Compression implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements compression functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compression.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#if LLVM_ENABLE_ZLIB == 1 && HAVE_ZLIB_H
#include <zlib.h>
#endif

using namespace llvm;

#if LLVM_ENABLE_ZLIB == 1 && HAVE_LIBZ
static int encodeZlibCompressionLevel(zlib::CompressionLevel Level) {
  switch (Level) {
    case zlib::NoCompression: return 0;
    case zlib::BestSpeedCompression: return 1;
    case zlib::DefaultCompression: return Z_DEFAULT_COMPRESSION;
    case zlib::BestSizeCompression: return 9;
  }
  llvm_unreachable("Invalid zlib::CompressionLevel!");
}

static zlib::Status encodeZlibReturnValue(int ReturnValue) {
  switch (ReturnValue) {
    case Z_OK: return zlib::StatusOK;
    case Z_MEM_ERROR: return zlib::StatusOutOfMemory;
    case Z_BUF_ERROR: return zlib::StatusBufferTooShort;
    case Z_STREAM_ERROR: return zlib::StatusInvalidArg;
    case Z_DATA_ERROR: return zlib::StatusInvalidData;
    default: llvm_unreachable("unknown zlib return status!");
  }
}

bool zlib::isAvailable() { return true; }
zlib::Status zlib::compress(StringRef InputBuffer,
                            SmallVectorImpl<char> &CompressedBuffer,
                            CompressionLevel Level) {
  unsigned long CompressedSize = ::compressBound(InputBuffer.size());
  CompressedBuffer.resize(CompressedSize);
  int CLevel = encodeZlibCompressionLevel(Level);
  Status Res = encodeZlibReturnValue(::compress2(
      (Bytef *)CompressedBuffer.data(), &CompressedSize,
      (const Bytef *)InputBuffer.data(), InputBuffer.size(), CLevel));
  // Tell MemorySanitizer that zlib output buffer is fully initialized.
  // This avoids a false report when running LLVM with uninstrumented ZLib.
  __msan_unpoison(CompressedBuffer.data(), CompressedSize);
  CompressedBuffer.resize(CompressedSize);
  return Res;
}

zlib::Status zlib::uncompress(StringRef InputBuffer,
                              SmallVectorImpl<char> &UncompressedBuffer,
                              size_t UncompressedSize) {
  UncompressedBuffer.resize(UncompressedSize);
  Status Res = encodeZlibReturnValue(::uncompress(
      (Bytef *)UncompressedBuffer.data(), (uLongf *)&UncompressedSize,
      (const Bytef *)InputBuffer.data(), InputBuffer.size()));
  // Tell MemorySanitizer that zlib output buffer is fully initialized.
  // This avoids a false report when running LLVM with uninstrumented ZLib.
  __msan_unpoison(UncompressedBuffer.data(), UncompressedSize);
  UncompressedBuffer.resize(UncompressedSize);
  return Res;
}

uint32_t zlib::crc32(StringRef Buffer) {
  return ::crc32(0, (const Bytef *)Buffer.data(), Buffer.size());
}

#else
bool zlib::isAvailable() { return false; }
zlib::Status zlib::compress(StringRef InputBuffer,
                            SmallVectorImpl<char> &CompressedBuffer,
                            CompressionLevel Level) {
  return zlib::StatusUnsupported;
}
zlib::Status zlib::uncompress(StringRef InputBuffer,
                              SmallVectorImpl<char> &UncompressedBuffer,
                              size_t UncompressedSize) {
  return zlib::StatusUnsupported;
}
uint32_t zlib::crc32(StringRef Buffer) {
  llvm_unreachable("zlib::crc32 is unavailable");
}
#endif

