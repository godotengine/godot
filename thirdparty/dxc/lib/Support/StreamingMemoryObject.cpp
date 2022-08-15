//===- StreamingMemoryObject.cpp - Streamable data interface -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StreamingMemoryObject.h"
#include <cassert>
#include <cstddef>
#include <cstring>
using namespace llvm;

namespace {

class RawMemoryObject : public MemoryObject {
public:
  RawMemoryObject(const unsigned char *Start, const unsigned char *End) :
    FirstChar(Start), LastChar(End) {
    assert(LastChar >= FirstChar && "Invalid start/end range");
  }

  uint64_t getExtent() const override {
    return LastChar - FirstChar;
  }
  uint64_t readBytes(uint8_t *Buf, uint64_t Size,
                     uint64_t Address) const override;
  const uint8_t *getPointer(uint64_t address, uint64_t size) const override;
  bool isValidAddress(uint64_t address) const override {
    return validAddress(address);
  }

private:
  const uint8_t* const FirstChar;
  const uint8_t* const LastChar;

  // These are implemented as inline functions here to avoid multiple virtual
  // calls per public function
  bool validAddress(uint64_t address) const {
    return static_cast<std::ptrdiff_t>(address) < LastChar - FirstChar;
  }

  RawMemoryObject(const RawMemoryObject&) = delete;
  void operator=(const RawMemoryObject&) = delete;
};

uint64_t RawMemoryObject::readBytes(uint8_t *Buf, uint64_t Size,
                                    uint64_t Address) const {
  uint64_t BufferSize = LastChar - FirstChar;
  if (Address >= BufferSize)
    return 0;

  uint64_t End = Address + Size;
  if (End > BufferSize)
    End = BufferSize;

  assert(static_cast<int64_t>(End - Address) >= 0);
  Size = End - Address;
  memcpy(Buf, Address + FirstChar, Size);
  return Size;
}

const uint8_t *RawMemoryObject::getPointer(uint64_t address,
                                           uint64_t size) const {
  return FirstChar + address;
}
} // anonymous namespace

namespace llvm {
// If the bitcode has a header, then its size is known, and we don't have to
// block until we actually want to read it.
bool StreamingMemoryObject::isValidAddress(uint64_t address) const {
  if (ObjectSize && address < ObjectSize) return true;
  return fetchToPos(address);
}

uint64_t StreamingMemoryObject::getExtent() const {
  if (ObjectSize) return ObjectSize;
  size_t pos = BytesRead + kChunkSize;
  // keep fetching until we run out of bytes
  while (fetchToPos(pos)) pos += kChunkSize;
  return ObjectSize;
}

uint64_t StreamingMemoryObject::readBytes(uint8_t *Buf, uint64_t Size,
                                          uint64_t Address) const {
  fetchToPos(Address + Size - 1);
  // Note: For wrapped bitcode files will set ObjectSize after the
  // first call to fetchToPos. In such cases, ObjectSize can be
  // smaller than BytesRead.
  size_t MaxAddress =
      (ObjectSize && ObjectSize < BytesRead) ? ObjectSize : BytesRead;
  if (Address >= MaxAddress)
    return 0;

  uint64_t End = Address + Size;
  if (End > MaxAddress)
    End = MaxAddress;
  assert(End >= Address);
  Size = End - Address;
  memcpy(Buf, &Bytes[Address + BytesSkipped], Size);
  return Size;
}

bool StreamingMemoryObject::dropLeadingBytes(size_t s) {
  if (BytesRead < s) return true;
  BytesSkipped = s;
  BytesRead -= s;
  return false;
}

void StreamingMemoryObject::setKnownObjectSize(size_t size) {
  ObjectSize = size;
  Bytes.reserve(size);
  if (ObjectSize <= BytesRead)
    EOFReached = true;
}

MemoryObject *getNonStreamedMemoryObject(const unsigned char *Start,
                                         const unsigned char *End) {
  return new RawMemoryObject(Start, End);
}

StreamingMemoryObject::StreamingMemoryObject(
    std::unique_ptr<DataStreamer> Streamer)
    : Bytes(kChunkSize), Streamer(std::move(Streamer)), BytesRead(0),
      BytesSkipped(0), ObjectSize(0), EOFReached(false) {
  BytesRead = this->Streamer->GetBytes(&Bytes[0], kChunkSize);
}
}
