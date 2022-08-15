//===- MemoryObject.h - Abstract memory interface ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MEMORYOBJECT_H
#define LLVM_SUPPORT_MEMORYOBJECT_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

/// Interface to data which might be streamed. Streamability has 2 important
/// implications/restrictions. First, the data might not yet exist in memory
/// when the request is made. This just means that readByte/readBytes might have
/// to block or do some work to get it. More significantly, the exact size of
/// the object might not be known until it has all been fetched. This means that
/// to return the right result, getExtent must also wait for all the data to
/// arrive; therefore it should not be called on objects which are actually
/// streamed (this would defeat the purpose of streaming). Instead,
/// isValidAddress can be used to test addresses without knowing the exact size
/// of the stream. Finally, getPointer can be used instead of readBytes to avoid
/// extra copying.
class MemoryObject {
public:
  virtual ~MemoryObject();

  /// Returns the size of the region in bytes.  (The region is contiguous, so
  /// the highest valid address of the region is getExtent() - 1).
  ///
  /// @result         - The size of the region.
  virtual uint64_t getExtent() const = 0;

  /// Tries to read a contiguous range of bytes from the region, up to the end
  /// of the region.
  ///
  /// @param Buf      - A pointer to a buffer to be filled in.  Must be non-NULL
  ///                   and large enough to hold size bytes.
  /// @param Size     - The number of bytes to copy.
  /// @param Address  - The address of the first byte, in the same space as
  ///                   getBase().
  /// @result         - The number of bytes read.
  virtual uint64_t readBytes(uint8_t *Buf, uint64_t Size,
                             uint64_t Address) const = 0;

  /// Ensures that the requested data is in memory, and returns a pointer to it.
  /// More efficient than using readBytes if the data is already in memory. May
  /// block until (address - base + size) bytes have been read
  /// @param address - address of the byte, in the same space as getBase()
  /// @param size    - amount of data that must be available on return
  /// @result        - valid pointer to the requested data
  virtual const uint8_t *getPointer(uint64_t address, uint64_t size) const = 0;

  /// Returns true if the address is within the object (i.e. between base and
  /// base + extent - 1 inclusive). May block until (address - base) bytes have
  /// been read
  /// @param address - address of the byte, in the same space as getBase()
  /// @result        - true if the address may be read with readByte()
  virtual bool isValidAddress(uint64_t address) const = 0;
};

}

#endif
