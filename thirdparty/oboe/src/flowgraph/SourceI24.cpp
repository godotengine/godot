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
#include "SourceI24.h"

#if FLOWGRAPH_ANDROID_INTERNAL
#include <audio_utils/primitives.h>
#endif

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

constexpr int kBytesPerI24Packed = 3;

SourceI24::SourceI24(int32_t channelCount)
        : FlowGraphSourceBuffered(channelCount) {
}

int32_t SourceI24::onProcess(int32_t numFrames) {
    float *floatData = output.getBuffer();
    int32_t channelCount = output.getSamplesPerFrame();

    int32_t framesLeft = mSizeInFrames - mFrameIndex;
    int32_t framesToProcess = std::min(numFrames, framesLeft);
    int32_t numSamples = framesToProcess * channelCount;

    const uint8_t *byteBase = (uint8_t *) mData;
    const uint8_t *byteData = &byteBase[mFrameIndex * channelCount * kBytesPerI24Packed];

#if FLOWGRAPH_ANDROID_INTERNAL
    memcpy_to_float_from_p24(floatData, byteData, numSamples);
#else
    static const float scale = 1. / (float)(1UL << 31);
    for (int i = 0; i < numSamples; i++) {
        // Assemble the data assuming Little Endian format.
        int32_t pad = byteData[2];
        pad <<= 8;
        pad |= byteData[1];
        pad <<= 8;
        pad |= byteData[0];
        pad <<= 8; // Shift to 32 bit data so the sign is correct.
        byteData += kBytesPerI24Packed;
        *floatData++ = pad * scale; // scale to range -1.0 to 1.0
    }
#endif

    mFrameIndex += framesToProcess;
    return framesToProcess;
}
