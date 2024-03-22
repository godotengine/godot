/*
 * Copyright (C) 2018 The Android Open Source Project
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

#include "oboe/StabilizedCallback.h"
#include "common/AudioClock.h"
#include "common/Trace.h"

constexpr int32_t kLoadGenerationStepSizeNanos = 20000;
constexpr float kPercentageOfCallbackToUse = 0.8;

using namespace oboe;

StabilizedCallback::StabilizedCallback(AudioStreamCallback *callback) : mCallback(callback){
    Trace::initialize();
}

/**
 * An audio callback which attempts to do work for a fixed amount of time.
 *
 * @param oboeStream
 * @param audioData
 * @param numFrames
 * @return
 */
DataCallbackResult
StabilizedCallback::onAudioReady(AudioStream *oboeStream, void *audioData, int32_t numFrames) {

    int64_t startTimeNanos = AudioClock::getNanoseconds();

    if (mFrameCount == 0){
        mEpochTimeNanos = startTimeNanos;
    }

    int64_t durationSinceEpochNanos = startTimeNanos - mEpochTimeNanos;

    // In an ideal world the callback start time will be exactly the same as the duration of the
    // frames already read/written into the stream. In reality the callback can start early
    // or late. By finding the delta we can calculate the target duration for our stabilized
    // callback.
    int64_t idealStartTimeNanos = (mFrameCount * kNanosPerSecond) / oboeStream->getSampleRate();
    int64_t lateStartNanos = durationSinceEpochNanos - idealStartTimeNanos;

    if (lateStartNanos < 0){
        // This was an early start which indicates that our previous epoch was a late callback.
        // Update our epoch to this more accurate time.
        mEpochTimeNanos = startTimeNanos;
        mFrameCount = 0;
    }

    int64_t numFramesAsNanos = (numFrames * kNanosPerSecond) / oboeStream->getSampleRate();
    int64_t targetDurationNanos = static_cast<int64_t>(
            (numFramesAsNanos * kPercentageOfCallbackToUse) - lateStartNanos);

    Trace::beginSection("Actual load");
    DataCallbackResult result = mCallback->onAudioReady(oboeStream, audioData, numFrames);
    Trace::endSection();

    int64_t executionDurationNanos = AudioClock::getNanoseconds() - startTimeNanos;
    int64_t stabilizingLoadDurationNanos = targetDurationNanos - executionDurationNanos;

    Trace::beginSection("Stabilized load for %lldns", stabilizingLoadDurationNanos);
    generateLoad(stabilizingLoadDurationNanos);
    Trace::endSection();

    // Wraparound: At 48000 frames per second mFrameCount wraparound will occur after 6m years,
    // significantly longer than the average lifetime of an Android phone.
    mFrameCount += numFrames;
    return result;
}

void StabilizedCallback::generateLoad(int64_t durationNanos) {

    int64_t currentTimeNanos = AudioClock::getNanoseconds();
    int64_t deadlineTimeNanos = currentTimeNanos + durationNanos;

    // opsPerStep gives us an estimated number of operations which need to be run to fully utilize
    // the CPU for a fixed amount of time (specified by kLoadGenerationStepSizeNanos).
    // After each step the opsPerStep value is re-calculated based on the actual time taken to
    // execute those operations.
    auto opsPerStep = (int)(mOpsPerNano * kLoadGenerationStepSizeNanos);
    int64_t stepDurationNanos = 0;
    int64_t previousTimeNanos = 0;

    while (currentTimeNanos <= deadlineTimeNanos){

        for (int i = 0; i < opsPerStep; i++) cpu_relax();

        previousTimeNanos = currentTimeNanos;
        currentTimeNanos = AudioClock::getNanoseconds();
        stepDurationNanos = currentTimeNanos - previousTimeNanos;

        // Calculate exponential moving average to smooth out values, this acts as a low pass filter.
        // @see https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
        static const float kFilterCoefficient = 0.1;
        auto measuredOpsPerNano = (double) opsPerStep / stepDurationNanos;
        mOpsPerNano = kFilterCoefficient * measuredOpsPerNano + (1.0 - kFilterCoefficient) * mOpsPerNano;
        opsPerStep = (int) (mOpsPerNano * kLoadGenerationStepSizeNanos);
    }
}
