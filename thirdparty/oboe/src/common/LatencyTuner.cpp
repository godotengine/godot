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

#include "oboe/LatencyTuner.h"

using namespace oboe;

LatencyTuner::LatencyTuner(AudioStream &stream)
        : LatencyTuner(stream, stream.getBufferCapacityInFrames()) {
}

LatencyTuner::LatencyTuner(oboe::AudioStream &stream, int32_t maximumBufferSize)
        : mStream(stream)
        , mMaxBufferSize(maximumBufferSize) {
    int32_t burstSize = stream.getFramesPerBurst();
    setMinimumBufferSize(kDefaultNumBursts * burstSize);
    setBufferSizeIncrement(burstSize);
    reset();
}

Result LatencyTuner::tune() {
    if (mState == State::Unsupported) {
        return Result::ErrorUnimplemented;
    }

    Result result = Result::OK;

    // Process reset requests.
    int32_t numRequests = mLatencyTriggerRequests.load();
    if (numRequests != mLatencyTriggerResponses.load()) {
        mLatencyTriggerResponses.store(numRequests);
        reset();
    }

    // Set state to Active if the idle countdown has reached zero.
    if (mState == State::Idle && --mIdleCountDown <= 0) {
        mState = State::Active;
    }

    // When state is Active attempt to change the buffer size if the number of xRuns has increased.
    if (mState == State::Active) {

        auto xRunCountResult = mStream.getXRunCount();
        if (xRunCountResult == Result::OK) {
            if ((xRunCountResult.value() - mPreviousXRuns) > 0) {
                mPreviousXRuns = xRunCountResult.value();
                int32_t oldBufferSize = mStream.getBufferSizeInFrames();
                int32_t requestedBufferSize = oldBufferSize + getBufferSizeIncrement();

                // Do not request more than the maximum buffer size (which was either user-specified
                // or was from stream->getBufferCapacityInFrames())
                if (requestedBufferSize > mMaxBufferSize) requestedBufferSize = mMaxBufferSize;

                // Note that this will not allocate more memory. It simply determines
                // how much of the existing buffer capacity will be used. The size will be
                // clipped to the bufferCapacity by AAudio.
                auto setBufferResult = mStream.setBufferSizeInFrames(requestedBufferSize);
                if (setBufferResult != Result::OK) {
                    result = setBufferResult;
                    mState = State::Unsupported;
                } else if (setBufferResult.value() == oldBufferSize) {
                    mState = State::AtMax;
                }
            }
        } else {
            mState = State::Unsupported;
        }
    }

    if (mState == State::Unsupported) {
        result = Result::ErrorUnimplemented;
    }

    if (mState == State::AtMax) {
        result = Result::OK;
    }
    return result;
}

void LatencyTuner::requestReset() {
    if (mState != State::Unsupported) {
        mLatencyTriggerRequests++;
    }
}

void LatencyTuner::reset() {
    mState = State::Idle;
    mIdleCountDown = kIdleCount;
    // Set to minimal latency
    mStream.setBufferSizeInFrames(getMinimumBufferSize());
}

bool LatencyTuner::isAtMaximumBufferSize() {
    return mState == State::AtMax;
}
