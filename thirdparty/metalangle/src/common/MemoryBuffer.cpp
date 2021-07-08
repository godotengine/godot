//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "common/MemoryBuffer.h"

#include <algorithm>
#include <cstdlib>

#include "common/debug.h"

namespace angle
{

// MemoryBuffer implementation.
MemoryBuffer::~MemoryBuffer()
{
    free(mData);
    mData = nullptr;
}

bool MemoryBuffer::resize(size_t size)
{
    if (size == 0)
    {
        free(mData);
        mData = nullptr;
        mSize = 0;
        return true;
    }

    if (size == mSize)
    {
        return true;
    }

    // Only reallocate if the size has changed.
    uint8_t *newMemory = static_cast<uint8_t *>(malloc(sizeof(uint8_t) * size));
    if (newMemory == nullptr)
    {
        return false;
    }

    if (mData)
    {
        // Copy the intersection of the old data and the new data
        std::copy(mData, mData + std::min(mSize, size), newMemory);
        free(mData);
    }

    mData = newMemory;
    mSize = size;

    return true;
}

void MemoryBuffer::fill(uint8_t datum)
{
    if (!empty())
    {
        std::fill(mData, mData + mSize, datum);
    }
}

MemoryBuffer::MemoryBuffer(MemoryBuffer &&other) : MemoryBuffer()
{
    *this = std::move(other);
}

MemoryBuffer &MemoryBuffer::operator=(MemoryBuffer &&other)
{
    std::swap(mSize, other.mSize);
    std::swap(mData, other.mData);
    return *this;
}

// ScratchBuffer implementation.

ScratchBuffer::ScratchBuffer(uint32_t lifetime) : mLifetime(lifetime), mResetCounter(lifetime) {}

ScratchBuffer::~ScratchBuffer() {}

bool ScratchBuffer::get(size_t requestedSize, MemoryBuffer **memoryBufferOut)
{
    return getImpl(requestedSize, memoryBufferOut, Optional<uint8_t>::Invalid());
}

bool ScratchBuffer::getInitialized(size_t requestedSize,
                                   MemoryBuffer **memoryBufferOut,
                                   uint8_t initValue)
{
    return getImpl(requestedSize, memoryBufferOut, Optional<uint8_t>(initValue));
}

bool ScratchBuffer::getImpl(size_t requestedSize,
                            MemoryBuffer **memoryBufferOut,
                            Optional<uint8_t> initValue)
{
    if (mScratchMemory.size() == requestedSize)
    {
        mResetCounter    = mLifetime;
        *memoryBufferOut = &mScratchMemory;
        return true;
    }

    if (mScratchMemory.size() > requestedSize)
    {
        tick();
    }

    if (mResetCounter == 0 || mScratchMemory.size() < requestedSize)
    {
        mScratchMemory.resize(0);
        if (!mScratchMemory.resize(requestedSize))
        {
            return false;
        }
        mResetCounter = mLifetime;
        if (initValue.valid())
        {
            mScratchMemory.fill(initValue.value());
        }
    }

    ASSERT(mScratchMemory.size() >= requestedSize);

    *memoryBufferOut = &mScratchMemory;
    return true;
}

void ScratchBuffer::tick()
{
    if (mResetCounter > 0)
    {
        --mResetCounter;
    }
}

void ScratchBuffer::clear()
{
    mResetCounter = mLifetime;
    mScratchMemory.resize(0);
}

}  // namespace angle
