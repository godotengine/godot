//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMMON_MEMORYBUFFER_H_
#define COMMON_MEMORYBUFFER_H_

#include "common/Optional.h"
#include "common/angleutils.h"
#include "common/debug.h"

#include <stdint.h>
#include <cstddef>

namespace angle
{

class MemoryBuffer final : NonCopyable
{
  public:
    MemoryBuffer() = default;
    MemoryBuffer(size_t size) { resize(size); }
    ~MemoryBuffer();

    MemoryBuffer(MemoryBuffer &&other);
    MemoryBuffer &operator=(MemoryBuffer &&other);

    bool resize(size_t size);
    size_t size() const { return mSize; }
    bool empty() const { return mSize == 0; }

    const uint8_t *data() const { return mData; }
    uint8_t *data()
    {
        ASSERT(mData);
        return mData;
    }

    uint8_t &operator[](size_t pos)
    {
        ASSERT(pos < mSize);
        return mData[pos];
    }
    const uint8_t &operator[](size_t pos) const
    {
        ASSERT(pos < mSize);
        return mData[pos];
    }

    void fill(uint8_t datum);

  private:
    size_t mSize   = 0;
    uint8_t *mData = nullptr;
};

class ScratchBuffer final : NonCopyable
{
  public:
    // If we request a scratch buffer requesting a smaller size this many times, release and
    // recreate the scratch buffer. This ensures we don't have a degenerate case where we are stuck
    // hogging memory.
    ScratchBuffer(uint32_t lifetime);
    ~ScratchBuffer();

    // Returns true with a memory buffer of the requested size, or false on failure.
    bool get(size_t requestedSize, MemoryBuffer **memoryBufferOut);

    // Same as get, but ensures new values are initialized to a fixed constant.
    bool getInitialized(size_t requestedSize, MemoryBuffer **memoryBufferOut, uint8_t initValue);

    // Ticks the release counter for the scratch buffer. Also done implicitly in get().
    void tick();

    void clear();

  private:
    bool getImpl(size_t requestedSize, MemoryBuffer **memoryBufferOut, Optional<uint8_t> initValue);

    const uint32_t mLifetime;
    uint32_t mResetCounter;
    MemoryBuffer mScratchMemory;
};

}  // namespace angle

#endif  // COMMON_MEMORYBUFFER_H_
