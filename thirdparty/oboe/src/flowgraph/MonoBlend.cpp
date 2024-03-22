/*
 * Copyright 2021 The Android Open Source Project
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

#include <unistd.h>

#include "MonoBlend.h"

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

MonoBlend::MonoBlend(int32_t channelCount)
        : FlowGraphFilter(channelCount)
        , mInvChannelCount(1. / channelCount)
{
}

int32_t MonoBlend::onProcess(int32_t numFrames) {
    int32_t channelCount = output.getSamplesPerFrame();
    const float *inputBuffer = input.getBuffer();
    float *outputBuffer = output.getBuffer();

    for (size_t i = 0; i < numFrames; ++i) {
        float accum = 0;
        for (size_t j = 0; j < channelCount; ++j) {
            accum += *inputBuffer++;
        }
        accum *= mInvChannelCount;
        for (size_t j = 0; j < channelCount; ++j) {
            *outputBuffer++ = accum;
        }
    }

    return numFrames;
}
