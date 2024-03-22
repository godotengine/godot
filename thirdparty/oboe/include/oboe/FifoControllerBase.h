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

#ifndef NATIVEOBOE_FIFOCONTROLLERBASE_H
#define NATIVEOBOE_FIFOCONTROLLERBASE_H

#include <stdint.h>

namespace oboe {

/**
 * Manage the read/write indices of a circular buffer.
 *
 * The caller is responsible for reading and writing the actual data.
 * Note that the span of available frames may not be contiguous. They
 * may wrap around from the end to the beginning of the buffer. In that
 * case the data must be read or written in at least two blocks of frames.
 *
 */

class FifoControllerBase {

public:
   /**
	 * Construct a `FifoControllerBase`.
	 *
	 * @param totalFrames capacity of the circular buffer in frames
	 */
    FifoControllerBase(uint32_t totalFrames);

    virtual ~FifoControllerBase() = default;

    /**
     * The frames available to read will be calculated from the read and write counters.
     * The result will be clipped to the capacity of the buffer.
     * If the buffer has underflowed then this will return zero.
     *
     * @return number of valid frames available to read.
     */
    uint32_t getFullFramesAvailable() const;

	/**
     * The index in a circular buffer of the next frame to read.
     *
     * @return read index position
     */
    uint32_t getReadIndex() const;

   /**
	* Advance read index from a number of frames.
	* Equivalent of incrementReadCounter(numFrames).
	*
	* @param numFrames number of frames to advance the read index
	*/
    void advanceReadIndex(uint32_t numFrames);

	/**
	 * Get the number of frame that are not written yet.
	 *
	 * @return maximum number of frames that can be written without exceeding the threshold
	 */
    uint32_t getEmptyFramesAvailable() const;

    /**
	 * The index in a circular buffer of the next frame to write.
	 *
	 * @return index of the next frame to write
	 */
    uint32_t getWriteIndex() const;

	/**
     * Advance write index from a number of frames.
     * Equivalent of incrementWriteCounter(numFrames).
     *
     * @param numFrames number of frames to advance the write index
     */
    void advanceWriteIndex(uint32_t numFrames);

	/**
	 * Get the frame capacity of the fifo.
	 *
	 * @return frame capacity
	 */
    uint32_t getFrameCapacity() const { return mTotalFrames; }

    virtual uint64_t getReadCounter() const = 0;
    virtual void setReadCounter(uint64_t n) = 0;
    virtual void incrementReadCounter(uint64_t n) = 0;
    virtual uint64_t getWriteCounter() const = 0;
    virtual void setWriteCounter(uint64_t n) = 0;
    virtual void incrementWriteCounter(uint64_t n) = 0;

private:
    uint32_t mTotalFrames;
};

} // namespace oboe

#endif //NATIVEOBOE_FIFOCONTROLLERBASE_H
