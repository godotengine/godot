//===- StreamingMemoryObject.h - Streamable data interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STREAMINGMEMORYOBJECT_H
#define LLVM_SUPPORT_STREAMINGMEMORYOBJECT_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryObject.h"
#include <memory>
#include <vector>

namespace llvm {

/// Interface to data which is actually streamed from a DataStreamer. In
/// addition to inherited members, it has the dropLeadingBytes and
/// setKnownObjectSize methods which are not applicable to non-streamed objects.
class StreamingMemoryObject : public MemoryObject {
public:
  StreamingMemoryObject(std::unique_ptr<DataStreamer> Streamer);
  uint64_t getExtent() const override;
  uint64_t readBytes(uint8_t *Buf, uint64_t Size,
                     uint64_t Address) const override;
  const uint8_t *getPointer(uint64_t address, uint64_t size) const override {
    // FIXME: This could be fixed by ensuring the bytes are fetched and
    // making a copy, requiring that the bitcode size be known, or
    // otherwise ensuring that the memory doesn't go away/get reallocated,
    // but it's not currently necessary. Users that need the pointer (any
    // that need Blobs) don't stream.
    report_fatal_error("getPointer in streaming memory objects not allowed");
    return nullptr;
  }
  bool isValidAddress(uint64_t address) const override;

  /// Drop s bytes from the front of the stream, pushing the positions of the
  /// remaining bytes down by s. This is used to skip past the bitcode header,
  /// since we don't know a priori if it's present, and we can't put bytes
  /// back into the stream once we've read them.
  bool dropLeadingBytes(size_t s);

  /// If the data object size is known in advance, many of the operations can
  /// be made more efficient, so this method should be called before reading
  /// starts (although it can be called anytime).
  void setKnownObjectSize(size_t size);

private:
  const static uint32_t kChunkSize = 4096 * 4;
  mutable std::vector<unsigned char> Bytes;
  std::unique_ptr<DataStreamer> Streamer;
  mutable size_t BytesRead;   // Bytes read from stream
  size_t BytesSkipped;// Bytes skipped at start of stream (e.g. wrapper/header)
  mutable size_t ObjectSize; // 0 if unknown, set if wrapper seen or EOF reached
  mutable bool EOFReached;

  // Fetch enough bytes such that Pos can be read (i.e. BytesRead >
  // Pos). Returns true if Pos can be read.  Unlike most of the
  // functions in BitcodeReader, returns true on success.  Most of the
  // requests will be small, but we fetch at kChunkSize bytes at a
  // time to avoid making too many potentially expensive GetBytes
  // calls.
  bool fetchToPos(size_t Pos) const {
    while (Pos >= BytesRead) {
      if (EOFReached)
        return false;
      Bytes.resize(BytesRead + BytesSkipped + kChunkSize);
      size_t bytes = Streamer->GetBytes(&Bytes[BytesRead + BytesSkipped],
                                        kChunkSize);
      BytesRead += bytes;
      if (bytes == 0) { // reached EOF/ran out of bytes
        if (ObjectSize == 0)
          ObjectSize = BytesRead;
        EOFReached = true;
      }
    }
    return !ObjectSize || Pos < ObjectSize;
  }

  StreamingMemoryObject(const StreamingMemoryObject&) = delete;
  void operator=(const StreamingMemoryObject&) = delete;
};

MemoryObject *getNonStreamedMemoryObject(
    const unsigned char *Start, const unsigned char *End);

}
#endif  // STREAMINGMEMORYOBJECT_H_
