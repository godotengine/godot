/*
 * Copyright 2018 The Android Open Source Project
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

#ifndef OBOE_STABILIZEDCALLBACK_H
#define OBOE_STABILIZEDCALLBACK_H

#include <cstdint>
#include "oboe/AudioStream.h"

namespace oboe {

class StabilizedCallback : public AudioStreamCallback {

public:
    explicit StabilizedCallback(AudioStreamCallback *callback);

    DataCallbackResult
    onAudioReady(AudioStream *oboeStream, void *audioData, int32_t numFrames) override;

    void onErrorBeforeClose(AudioStream *oboeStream, Result error) override {
        return mCallback->onErrorBeforeClose(oboeStream, error);
    }

    void onErrorAfterClose(AudioStream *oboeStream, Result error) override {

        // Reset all fields now that the stream has been closed
        mFrameCount = 0;
        mEpochTimeNanos = 0;
        mOpsPerNano = 1;
        return mCallback->onErrorAfterClose(oboeStream, error);
    }

private:

    AudioStreamCallback *mCallback = nullptr;
    int64_t mFrameCount = 0;
    int64_t mEpochTimeNanos = 0;
    double  mOpsPerNano = 1;

    void generateLoad(int64_t durationNanos);
};

/**
 * cpu_relax is an architecture specific method of telling the CPU that you don't want it to
 * do much work. asm volatile keeps the compiler from optimising these instructions out.
 */
#if defined(__i386__) || defined(__x86_64__)
#define cpu_relax() asm volatile("rep; nop" ::: "memory");

#elif defined(__arm__) || defined(__mips__) || defined(__riscv)
    #define cpu_relax() asm volatile("":::"memory")

#elif defined(__aarch64__)
#define cpu_relax() asm volatile("yield" ::: "memory")

#else
#error "cpu_relax is not defined for this architecture"
#endif

}

#endif //OBOE_STABILIZEDCALLBACK_H
