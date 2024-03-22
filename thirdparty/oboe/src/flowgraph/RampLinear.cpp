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

#include <algorithm>
#include <unistd.h>
#include "FlowGraphNode.h"
#include "RampLinear.h"

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

RampLinear::RampLinear(int32_t channelCount)
        : FlowGraphFilter(channelCount) {
    mTarget.store(1.0f);
}

void RampLinear::setLengthInFrames(int32_t frames) {
    mLengthInFrames = frames;
}

void RampLinear::setTarget(float target) {
    mTarget.store(target);
    // If the ramp has not been used then start immediately at this level.
    if (mLastCallCount == kInitialCallCount) {
        forceCurrent(target);
    }
}

float RampLinear::interpolateCurrent() {
    return mLevelTo - (mRemaining * mScaler);
}

int32_t RampLinear::onProcess(int32_t numFrames) {
    const float *inputBuffer = input.getBuffer();
    float *outputBuffer = output.getBuffer();
    int32_t channelCount = output.getSamplesPerFrame();

    float target = getTarget();
    if (target != mLevelTo) {
        // Start new ramp. Continue from previous level.
        mLevelFrom = interpolateCurrent();
        mLevelTo = target;
        mRemaining = mLengthInFrames;
        mScaler = (mLevelTo - mLevelFrom) / mLengthInFrames; // for interpolation
    }

    int32_t framesLeft = numFrames;

    if (mRemaining > 0) { // Ramping? This doesn't happen very often.
        int32_t framesToRamp = std::min(framesLeft, mRemaining);
        framesLeft -= framesToRamp;
        while (framesToRamp > 0) {
            float currentLevel = interpolateCurrent();
            for (int ch = 0; ch < channelCount; ch++) {
                *outputBuffer++ = *inputBuffer++ * currentLevel;
            }
            mRemaining--;
            framesToRamp--;
        }
    }

    // Process any frames after the ramp.
    int32_t samplesLeft = framesLeft * channelCount;
    for (int i = 0; i < samplesLeft; i++) {
        *outputBuffer++ = *inputBuffer++ * mLevelTo;
    }

    return numFrames;
}
