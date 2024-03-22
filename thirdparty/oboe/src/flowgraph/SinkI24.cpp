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
#include "SinkI24.h"

#if FLOWGRAPH_ANDROID_INTERNAL
#include <audio_utils/primitives.h>
#endif

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

SinkI24::SinkI24(int32_t channelCount)
        : FlowGraphSink(channelCount) {}

int32_t SinkI24::read(void *data, int32_t numFrames) {
    uint8_t *byteData = (uint8_t *) data;
    const int32_t channelCount = input.getSamplesPerFrame();

    int32_t framesLeft = numFrames;
    while (framesLeft > 0) {
        // Run the graph and pull data through the input port.
        int32_t framesRead = pullData(framesLeft);
        if (framesRead <= 0) {
            break;
        }
        const float *floatData = input.getBuffer();
        int32_t numSamples = framesRead * channelCount;
#if FLOWGRAPH_ANDROID_INTERNAL
        memcpy_to_p24_from_float(byteData, floatData, numSamples);
        static const int kBytesPerI24Packed = 3;
        byteData += numSamples * kBytesPerI24Packed;
        floatData += numSamples;
#else
        const int32_t kI24PackedMax = 0x007FFFFF;
        const int32_t kI24PackedMin = 0xFF800000;
        for (int i = 0; i < numSamples; i++) {
            int32_t n = (int32_t) (*floatData++ * 0x00800000);
            n = std::min(kI24PackedMax, std::max(kI24PackedMin, n)); // clip
            // Write as a packed 24-bit integer in Little Endian format.
            *byteData++ = (uint8_t) n;
            *byteData++ = (uint8_t) (n >> 8);
            *byteData++ = (uint8_t) (n >> 16);
        }
#endif
        framesLeft -= framesRead;
    }
    return numFrames - framesLeft;
}
