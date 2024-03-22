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

#include <sys/types.h>
#include <pthread.h>
#include <thread>

#include <oboe/AudioStream.h>
#include "OboeDebug.h"
#include "AudioClock.h"
#include <oboe/Utilities.h>

namespace oboe {

/*
 * AudioStream
 */
AudioStream::AudioStream(const AudioStreamBuilder &builder)
        : AudioStreamBase(builder) {
}

Result AudioStream::close() {
    closePerformanceHint();
    // Update local counters so they can be read after the close.
    updateFramesWritten();
    updateFramesRead();
    return Result::OK;
}

// Call this from fireDataCallback() if you want to monitor CPU scheduler.
void AudioStream::checkScheduler() {
    int scheduler = sched_getscheduler(0) & ~SCHED_RESET_ON_FORK; // for current thread
    if (scheduler != mPreviousScheduler) {
        LOGD("AudioStream::%s() scheduler = %s", __func__,
                ((scheduler == SCHED_FIFO) ? "SCHED_FIFO" :
                ((scheduler == SCHED_OTHER) ? "SCHED_OTHER" :
                ((scheduler == SCHED_RR) ? "SCHED_RR" : "UNKNOWN")))
        );
        mPreviousScheduler = scheduler;
    }
}

DataCallbackResult AudioStream::fireDataCallback(void *audioData, int32_t numFrames) {
    if (!isDataCallbackEnabled()) {
        LOGW("AudioStream::%s() called with data callback disabled!", __func__);
        return DataCallbackResult::Stop; // Should not be getting called
    }

    beginPerformanceHintInCallback();

    // Call the app to do the work.
    DataCallbackResult result;
    if (mDataCallback) {
        result = mDataCallback->onAudioReady(this, audioData, numFrames);
    } else {
        result = onDefaultCallback(audioData, numFrames);
    }
    // On Oreo, we might get called after returning stop.
    // So block that here.
    setDataCallbackEnabled(result == DataCallbackResult::Continue);

    endPerformanceHintInCallback(numFrames);

    return result;
}

Result AudioStream::waitForStateTransition(StreamState startingState,
                                           StreamState endingState,
                                           int64_t timeoutNanoseconds)
{
    StreamState state;
    {
        std::lock_guard<std::mutex> lock(mLock);
        state = getState();
        if (state == StreamState::Closed) {
            return Result::ErrorClosed;
        } else if (state == StreamState::Disconnected) {
            return Result::ErrorDisconnected;
        }
    }

    StreamState nextState = state;
    // TODO Should this be a while()?!
    if (state == startingState && state != endingState) {
        Result result = waitForStateChange(state, &nextState, timeoutNanoseconds);
        if (result != Result::OK) {
            return result;
        }
    }

    if (nextState != endingState) {
        return Result::ErrorInvalidState;
    } else {
        return Result::OK;
    }
}

Result AudioStream::start(int64_t timeoutNanoseconds)
{
    Result result = requestStart();
    if (result != Result::OK) return result;
    if (timeoutNanoseconds <= 0) return result;
    return waitForStateTransition(StreamState::Starting,
                                  StreamState::Started, timeoutNanoseconds);
}

Result AudioStream::pause(int64_t timeoutNanoseconds)
{
    Result result = requestPause();
    if (result != Result::OK) return result;
    if (timeoutNanoseconds <= 0) return result;
    return waitForStateTransition(StreamState::Pausing,
                                  StreamState::Paused, timeoutNanoseconds);
}

Result AudioStream::flush(int64_t timeoutNanoseconds)
{
    Result result = requestFlush();
    if (result != Result::OK) return result;
    if (timeoutNanoseconds <= 0) return result;
    return waitForStateTransition(StreamState::Flushing,
                                  StreamState::Flushed, timeoutNanoseconds);
}

Result AudioStream::stop(int64_t timeoutNanoseconds)
{
    Result result = requestStop();
    if (result != Result::OK) return result;
    if (timeoutNanoseconds <= 0) return result;
    return waitForStateTransition(StreamState::Stopping,
                                  StreamState::Stopped, timeoutNanoseconds);
}

int32_t AudioStream::getBytesPerSample() const {
    return convertFormatToSizeInBytes(mFormat);
}

int64_t AudioStream::getFramesRead() {
    updateFramesRead();
    return mFramesRead;
}

int64_t AudioStream::getFramesWritten() {
    updateFramesWritten();
    return mFramesWritten;
}

ResultWithValue<int32_t> AudioStream::getAvailableFrames() {
    int64_t readCounter = getFramesRead();
    if (readCounter < 0) return ResultWithValue<int32_t>::createBasedOnSign(readCounter);
    int64_t writeCounter = getFramesWritten();
    if (writeCounter < 0) return ResultWithValue<int32_t>::createBasedOnSign(writeCounter);
    int32_t framesAvailable = writeCounter - readCounter;
    return ResultWithValue<int32_t>(framesAvailable);
}

ResultWithValue<int32_t> AudioStream::waitForAvailableFrames(int32_t numFrames,
        int64_t timeoutNanoseconds) {
    if (numFrames == 0) return Result::OK;
    if (numFrames < 0) return Result::ErrorOutOfRange;

    // Make sure we don't try to wait for more frames than the buffer can hold.
    // Subtract framesPerBurst because this is often called from a callback
    // and we don't want to be sleeping if the buffer is close to overflowing.
    const int32_t maxAvailableFrames = getBufferCapacityInFrames() - getFramesPerBurst();
    numFrames = std::min(numFrames, maxAvailableFrames);
    // The capacity should never be less than one burst. But clip to zero just in case.
    numFrames = std::max(0, numFrames);

    int64_t framesAvailable = 0;
    int64_t burstInNanos = getFramesPerBurst() * kNanosPerSecond / getSampleRate();
    bool ready = false;
    int64_t deadline = AudioClock::getNanoseconds() + timeoutNanoseconds;
    do {
        ResultWithValue<int32_t> result = getAvailableFrames();
        if (!result) return result;
        framesAvailable = result.value();
        ready = (framesAvailable >= numFrames);
        if (!ready) {
            int64_t now = AudioClock::getNanoseconds();
            if (now > deadline) break;
            AudioClock::sleepForNanos(burstInNanos);
        }
    } while (!ready);
    return (!ready)
            ? ResultWithValue<int32_t>(Result::ErrorTimeout)
            : ResultWithValue<int32_t>(framesAvailable);
}

ResultWithValue<FrameTimestamp> AudioStream::getTimestamp(clockid_t clockId) {
    FrameTimestamp frame;
    Result result = getTimestamp(clockId, &frame.position, &frame.timestamp);
    if (result == Result::OK){
        return ResultWithValue<FrameTimestamp>(frame);
    } else {
        return ResultWithValue<FrameTimestamp>(static_cast<Result>(result));
    }
}

void AudioStream::calculateDefaultDelayBeforeCloseMillis() {
    // Calculate delay time before close based on burst duration.
    // Start with a burst duration then add 1 msec as a safety margin.
    mDelayBeforeCloseMillis = std::max(kMinDelayBeforeCloseMillis,
                                       1 + ((mFramesPerBurst * 1000) / getSampleRate()));
    LOGD("calculateDefaultDelayBeforeCloseMillis() default = %d",
         static_cast<int>(mDelayBeforeCloseMillis));
}

} // namespace oboe
