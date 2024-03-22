/*
 * Copyright 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <memory.h>
#include <stdint.h>

#include "oboe/FifoControllerBase.h"
#include "fifo/FifoController.h"
#include "fifo/FifoControllerIndirect.h"
#include "oboe/FifoBuffer.h"

namespace oboe {

FifoBuffer::FifoBuffer(uint32_t bytesPerFrame, uint32_t capacityInFrames)
        : mBytesPerFrame(bytesPerFrame)
        , mStorage(nullptr)
        , mFramesReadCount(0)
        , mFramesUnderrunCount(0)
{
    mFifo = std::make_unique<FifoController>(capacityInFrames);
    // allocate buffer
    int32_t bytesPerBuffer = bytesPerFrame * capacityInFrames;
    mStorage = new uint8_t[bytesPerBuffer];
    mStorageOwned = true;
}

FifoBuffer::FifoBuffer( uint32_t  bytesPerFrame,
                        uint32_t  capacityInFrames,
                        std::atomic<uint64_t>  *readCounterAddress,
                        std::atomic<uint64_t>  *writeCounterAddress,
                        uint8_t  *dataStorageAddress
                        )
        : mBytesPerFrame(bytesPerFrame)
        , mStorage(dataStorageAddress)
        , mFramesReadCount(0)
        , mFramesUnderrunCount(0)
{
    mFifo = std::make_unique<FifoControllerIndirect>(capacityInFrames,
                                       readCounterAddress,
                                       writeCounterAddress);
    mStorage = dataStorageAddress;
    mStorageOwned = false;
}

FifoBuffer::~FifoBuffer() {
    if (mStorageOwned) {
        delete[] mStorage;
    }
}

int32_t FifoBuffer::convertFramesToBytes(int32_t frames) {
    return frames * mBytesPerFrame;
}

int32_t FifoBuffer::read(void *buffer, int32_t numFrames) {
    if (numFrames <= 0) {
        return 0;
    }
    // safe because numFrames is guaranteed positive
    uint32_t framesToRead = static_cast<uint32_t>(numFrames);
    uint32_t framesAvailable = mFifo->getFullFramesAvailable();
    framesToRead = std::min(framesToRead, framesAvailable);

    uint32_t readIndex = mFifo->getReadIndex(); // ranges 0 to capacity
    uint8_t *destination = reinterpret_cast<uint8_t *>(buffer);
    uint8_t *source = &mStorage[convertFramesToBytes(readIndex)];
    if ((readIndex + framesToRead) > mFifo->getFrameCapacity()) {
        // read in two parts, first part here is at the end of the mStorage buffer
        int32_t frames1 = static_cast<int32_t>(mFifo->getFrameCapacity() - readIndex);
        int32_t numBytes = convertFramesToBytes(frames1);
        if (numBytes < 0) {
            return static_cast<int32_t>(Result::ErrorOutOfRange);
        }
        memcpy(destination, source, static_cast<size_t>(numBytes));
        destination += numBytes;
        // read second part, which is at the beginning of mStorage
        source = &mStorage[0];
        int32_t frames2 = static_cast<uint32_t>(framesToRead - frames1);
        numBytes = convertFramesToBytes(frames2);
        if (numBytes < 0) {
            return static_cast<int32_t>(Result::ErrorOutOfRange);
        }
        memcpy(destination, source, static_cast<size_t>(numBytes));
    } else {
        // just read in one shot
        int32_t numBytes = convertFramesToBytes(framesToRead);
        if (numBytes < 0) {
            return static_cast<int32_t>(Result::ErrorOutOfRange);
        }
        memcpy(destination, source, static_cast<size_t>(numBytes));
    }
    mFifo->advanceReadIndex(framesToRead);

    return framesToRead;
}

int32_t FifoBuffer::write(const void *buffer, int32_t numFrames) {
    if (numFrames <= 0) {
        return 0;
    }
    // Guaranteed positive.
    uint32_t framesToWrite = static_cast<uint32_t>(numFrames);
    uint32_t framesAvailable = mFifo->getEmptyFramesAvailable();
    framesToWrite = std::min(framesToWrite, framesAvailable);

    uint32_t writeIndex = mFifo->getWriteIndex();
    int byteIndex = convertFramesToBytes(writeIndex);
    const uint8_t *source = reinterpret_cast<const uint8_t *>(buffer);
    uint8_t *destination = &mStorage[byteIndex];
    if ((writeIndex + framesToWrite) > mFifo->getFrameCapacity()) {
        // write in two parts, first part here
        int32_t frames1 = static_cast<uint32_t>(mFifo->getFrameCapacity() - writeIndex);
        int32_t numBytes = convertFramesToBytes(frames1);
        if (numBytes < 0) {
            return static_cast<int32_t>(Result::ErrorOutOfRange);
        }
        memcpy(destination, source, static_cast<size_t>(numBytes));
        // read second part
        source += convertFramesToBytes(frames1);
        destination = &mStorage[0];
        int frames2 = static_cast<uint32_t>(framesToWrite - frames1);
        numBytes = convertFramesToBytes(frames2);
        if (numBytes < 0) {
            return static_cast<int32_t>(Result::ErrorOutOfRange);
        }
        memcpy(destination, source, static_cast<size_t>(numBytes));
    } else {
        // just write in one shot
        int32_t numBytes = convertFramesToBytes(framesToWrite);
        if (numBytes < 0) {
            return static_cast<int32_t>(Result::ErrorOutOfRange);
        }
        memcpy(destination, source, static_cast<size_t>(numBytes));
    }
    mFifo->advanceWriteIndex(framesToWrite);

    return framesToWrite;
}

int32_t FifoBuffer::readNow(void *buffer, int32_t numFrames) {
    int32_t framesRead = read(buffer, numFrames);
    if (framesRead < 0) {
        return framesRead;
    }
    int32_t framesLeft = numFrames - framesRead;
    mFramesReadCount += framesRead;
    mFramesUnderrunCount += framesLeft;
    // Zero out any samples we could not set.
    if (framesLeft > 0) {
        uint8_t *destination = reinterpret_cast<uint8_t *>(buffer);
        destination += convertFramesToBytes(framesRead); // point to first byte not set
        int32_t bytesToZero = convertFramesToBytes(framesLeft);
        memset(destination, 0, static_cast<size_t>(bytesToZero));
    }

    return framesRead;
}


uint32_t FifoBuffer::getBufferCapacityInFrames() const {
    return mFifo->getFrameCapacity();
}

} // namespace oboe
