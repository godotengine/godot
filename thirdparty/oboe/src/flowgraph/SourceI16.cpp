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

#include <algorithm>
#include <unistd.h>

#include "FlowGraphNode.h"
#include "SourceI16.h"

#if FLOWGRAPH_ANDROID_INTERNAL
#include <audio_utils/primitives.h>
#endif

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

SourceI16::SourceI16(int32_t channelCount)
        : FlowGraphSourceBuffered(channelCount) {
}

int32_t SourceI16::onProcess(int32_t numFrames) {
    float *floatData = output.getBuffer();
    int32_t channelCount = output.getSamplesPerFrame();

    int32_t framesLeft = mSizeInFrames - mFrameIndex;
    int32_t framesToProcess = std::min(numFrames, framesLeft);
    int32_t numSamples = framesToProcess * channelCount;

    const int16_t *shortBase = static_cast<const int16_t *>(mData);
    const int16_t *shortData = &shortBase[mFrameIndex * channelCount];

#if FLOWGRAPH_ANDROID_INTERNAL
    memcpy_to_float_from_i16(floatData, shortData, numSamples);
#else
    for (int i = 0; i < numSamples; i++) {
        *floatData++ = *shortData++ * (1.0f / 32768);
    }
#endif

    mFrameIndex += framesToProcess;
    return framesToProcess;
}
