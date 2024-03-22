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

#ifndef OBOE_FIFOPROCESSOR_H
#define OBOE_FIFOPROCESSOR_H

#include <memory>
#include <stdint.h>

#include "oboe/Definitions.h"

#include "oboe/FifoControllerBase.h"

namespace oboe {

class FifoBuffer {
public:
	/**
	 * Construct a `FifoBuffer`.
	 *
	 * @param bytesPerFrame amount of bytes for one frame
	 * @param capacityInFrames the capacity of frames in fifo
	 */
    FifoBuffer(uint32_t bytesPerFrame, uint32_t capacityInFrames);

	/**
	 * Construct a `FifoBuffer`.
	 * To be used if the storage allocation is done outside of FifoBuffer.
	 *
	 * @param bytesPerFrame amount of bytes for one frame
	 * @param capacityInFrames capacity of frames in fifo
	 * @param readCounterAddress address of read counter
	 * @param writeCounterAddress address of write counter
	 * @param dataStorageAddress address of storage
	 */
    FifoBuffer(uint32_t   bytesPerFrame,
               uint32_t   capacityInFrames,
               std::atomic<uint64_t>   *readCounterAddress,
               std::atomic<uint64_t>   *writeCounterAddress,
               uint8_t   *dataStorageAddress);

    ~FifoBuffer();

	/**
	 * Convert a number of frames in bytes.
	 *
	 * @return number of bytes
	 */
    int32_t convertFramesToBytes(int32_t frames);

    /**
     * Read framesToRead or, if not enough, then read as many as are available.
     *
     * @param destination
     * @param framesToRead number of frames requested
     * @return number of frames actually read
     */
    int32_t read(void *destination, int32_t framesToRead);

	/**
	 * Write framesToWrite or, if too enough, then write as many as the fifo are not empty.
	 *
	 * @param destination
	 * @param framesToWrite number of frames requested
	 * @return number of frames actually write
	 */
    int32_t write(const void *source, int32_t framesToWrite);

	/**
	 * Get the buffer capacity in frames.
	 *
	 * @return number of frames
	 */
    uint32_t getBufferCapacityInFrames() const;

    /**
     * Calls read(). If all of the frames cannot be read then the remainder of the buffer
     * is set to zero.
     *
     * @param destination
     * @param framesToRead number of frames requested
     * @return number of frames actually read
     */
    int32_t readNow(void *destination, int32_t numFrames);

	/**
	 * Get the number of frames in the fifo.
	 *
	 * @return number of frames actually in the buffer
	 */
    uint32_t getFullFramesAvailable() {
        return mFifo->getFullFramesAvailable();
    }

	/**
	 * Get the amount of bytes per frame.
	 *
	 * @return number of bytes per frame
	 */
    uint32_t getBytesPerFrame() const {
        return mBytesPerFrame;
    }

	/**
	 * Get the position of read counter.
	 *
	 * @return position of read counter
	 */
    uint64_t getReadCounter() const {
        return mFifo->getReadCounter();
    }

	/**
	 * Set the position of read counter.
	 *
	 * @param n position of read counter
	 */
    void setReadCounter(uint64_t n) {
        mFifo->setReadCounter(n);
    }

	/**
	 * Get the position of write counter.
	 *
	 * @return position of write counter
	 */
    uint64_t getWriteCounter() {
        return mFifo->getWriteCounter();
    }

    /**
	 * Set the position of write counter.
	 *
	 * @param n position of write counter
	 */
    void setWriteCounter(uint64_t n) {
        mFifo->setWriteCounter(n);
    }

private:
    uint32_t mBytesPerFrame;
    uint8_t* mStorage;
    bool     mStorageOwned; // did this object allocate the storage?
    std::unique_ptr<FifoControllerBase> mFifo;
    uint64_t mFramesReadCount;
    uint64_t mFramesUnderrunCount;
};

} // namespace oboe

#endif //OBOE_FIFOPROCESSOR_H
