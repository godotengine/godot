/*
 * Copyright 2022 The Android Open Source Project
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
#include <math.h>
#include <unistd.h>
#include "FlowGraphNode.h"
#include "Limiter.h"

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

Limiter::Limiter(int32_t channelCount)
        : FlowGraphFilter(channelCount) {
}

int32_t Limiter::onProcess(int32_t numFrames) {
    const float *inputBuffer = input.getBuffer();
    float *outputBuffer = output.getBuffer();

    int32_t numSamples = numFrames * output.getSamplesPerFrame();

    // Cache the last valid output to reduce memory read/write
    float lastValidOutput = mLastValidOutput;

    for (int32_t i = 0; i < numSamples; i++) {
        // Use the previous output if the input is NaN
        if (!isnan(*inputBuffer)) {
            lastValidOutput = processFloat(*inputBuffer);
        }
        inputBuffer++;
        *outputBuffer++ = lastValidOutput;
    }
    mLastValidOutput = lastValidOutput;

    return numFrames;
}

float Limiter::processFloat(float in)
{
    float in_abs = fabsf(in);
    if (in_abs <= 1) {
        return in;
    }
    float out;
    if (in_abs < kXWhenYis3Decibels) {
        out = (kPolynomialSplineA * in_abs + kPolynomialSplineB) * in_abs + kPolynomialSplineC;
    } else {
        out = M_SQRT2;
    }
    if (in < 0) {
        out = -out;
    }
    return out;
}
