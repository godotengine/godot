/*
 * Copyright 2017 The Android Open Source Project
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

#ifndef OBOE_LATENCY_TUNER_
#define OBOE_LATENCY_TUNER_

#include <atomic>
#include <cstdint>
#include "oboe/Definitions.h"
#include "oboe/AudioStream.h"

namespace oboe {

/**
 * LatencyTuner can be used to dynamically tune the latency of an output stream.
 * It adjusts the stream's bufferSize by monitoring the number of underruns.
 *
 * This only affects the latency associated with the first level of buffering that is closest
 * to the application. It does not affect low latency in the HAL, or touch latency in the UI.
 *
 * Call tune() right before returning from your data callback function if using callbacks.
 * Call tune() right before calling write() if using blocking writes.
 *
 * If you want to see the ongoing results of this tuning process then call
 * stream->getBufferSize() periodically.
 *
 */
class LatencyTuner {
public:

    /**
     * Construct a new LatencyTuner object which will act on the given audio stream
     *
     * @param stream the stream who's latency will be tuned
     */
    explicit LatencyTuner(AudioStream &stream);

    /**
     * Construct a new LatencyTuner object which will act on the given audio stream.
     *
     * @param stream the stream who's latency will be tuned
     * @param the maximum buffer size which the tune() operation will set the buffer size to
     */
    explicit LatencyTuner(AudioStream &stream, int32_t maximumBufferSize);

    /**
     * Adjust the bufferSizeInFrames to optimize latency.
     * It will start with a low latency and then raise it if an underrun occurs.
     *
     * Latency tuning is only supported for AAudio.
     *
     * @return OK or negative error, ErrorUnimplemented for OpenSL ES
     */
    Result tune();

    /**
     * This may be called from another thread. Then tune() will call reset(),
     * which will lower the latency to the minimum and then allow it to rise back up
     * if there are glitches.
     *
     * This is typically called in response to a user decision to minimize latency. In other words,
     * call this from a button handler.
     */
    void requestReset();

    /**
     * @return true if the audio stream's buffer size is at the maximum value. If no maximum value
     * was specified when constructing the LatencyTuner then the value of
     * stream->getBufferCapacityInFrames is used
     */
    bool isAtMaximumBufferSize();

    /**
     * Set the minimum bufferSize in frames that is used when the tuner is reset.
     * You may wish to call requestReset() after calling this.
     * @param bufferSize
     */
    void setMinimumBufferSize(int32_t bufferSize) {
        mMinimumBufferSize = bufferSize;
    }

    int32_t getMinimumBufferSize() const {
        return mMinimumBufferSize;
    }

    /**
     * Set the amount the bufferSize will be incremented while tuning.
     * By default, this will be one burst.
     *
     * Note that AAudio will quantize the buffer size to a multiple of the burstSize.
     * So the final buffer sizes may not be a multiple of this increment.
     *
     * @param sizeIncrement
     */
    void setBufferSizeIncrement(int32_t sizeIncrement) {
        mBufferSizeIncrement = sizeIncrement;
    }

    int32_t getBufferSizeIncrement() const {
        return mBufferSizeIncrement;
    }

private:

    /**
     * Drop the latency down to the minimum and then let it rise back up.
     * This is useful if a glitch caused the latency to increase and it hasn't gone back down.
     *
     * This should only be called in the same thread as tune().
     */
    void reset();

    enum class State {
        Idle,
        Active,
        AtMax,
        Unsupported
    } ;

    // arbitrary number of calls to wait before bumping up the latency
    static constexpr int32_t kIdleCount = 8;
    static constexpr int32_t kDefaultNumBursts = 2;

    AudioStream           &mStream;
    State                 mState = State::Idle;
    int32_t               mMaxBufferSize = 0;
    int32_t               mPreviousXRuns = 0;
    int32_t               mIdleCountDown = 0;
    int32_t               mMinimumBufferSize;
    int32_t               mBufferSizeIncrement;
    std::atomic<int32_t>  mLatencyTriggerRequests{0}; // TODO user atomic requester from AAudio
    std::atomic<int32_t>  mLatencyTriggerResponses{0};
};

} // namespace oboe

#endif // OBOE_LATENCY_TUNER_
