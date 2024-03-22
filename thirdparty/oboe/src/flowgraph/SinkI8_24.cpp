/*
 * Copyright 2023 The Android Open Source Project
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

#include "FlowGraphNode.h"
#include "FlowgraphUtilities.h"
#include "SinkI8_24.h"

#if FLOWGRAPH_ANDROID_INTERNAL
#include <audio_utils/primitives.h>
#endif

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

SinkI8_24::SinkI8_24(int32_t channelCount)
        : FlowGraphSink(channelCount) {}

int32_t SinkI8_24::read(void *data, int32_t numFrames) {
    int32_t *intData = (int32_t *) data;
    const int32_t channelCount = input.getSamplesPerFrame();

    int32_t framesLeft = numFrames;
    while (framesLeft > 0) {
        // Run the graph and pull data through the input port.
        int32_t framesRead = pullData(framesLeft);
        if (framesRead <= 0) {
            break;
        }
        const float *signal = input.getBuffer();
        int32_t numSamples = framesRead * channelCount;
#if FLOWGRAPH_ANDROID_INTERNAL
        memcpy_to_q8_23_from_float_with_clamp(intData, signal, numSamples);
        intData += numSamples;
        signal += numSamples;
#else
        for (int i = 0; i < numSamples; i++) {
            *intData++ = FlowgraphUtilities::clamp24FromFloat(*signal++);
        }
#endif
        framesLeft -= framesRead;
    }
    return numFrames - framesLeft;
}
