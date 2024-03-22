/*
 * Copyright 2019 The Android Open Source Project
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

#include "SampleRateConverter.h"

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;
using namespace RESAMPLER_OUTER_NAMESPACE::resampler;

SampleRateConverter::SampleRateConverter(int32_t channelCount,
                                         MultiChannelResampler &resampler)
        : FlowGraphFilter(channelCount)
        , mResampler(resampler) {
    setDataPulledAutomatically(false);
}

void SampleRateConverter::reset() {
    FlowGraphNode::reset();
    mInputCursor = kInitialCallCount;
}

// Return true if there is a sample available.
bool SampleRateConverter::isInputAvailable() {
    // If we have consumed all of the input data then go out and get some more.
    if (mInputCursor >= mNumValidInputFrames) {
        mInputCallCount++;
        mNumValidInputFrames = input.pullData(mInputCallCount, input.getFramesPerBuffer());
        mInputCursor = 0;
    }
    return (mInputCursor < mNumValidInputFrames);
}

const float *SampleRateConverter::getNextInputFrame() {
    const float *inputBuffer = input.getBuffer();
    return &inputBuffer[mInputCursor++ * input.getSamplesPerFrame()];
}

int32_t SampleRateConverter::onProcess(int32_t numFrames) {
    float *outputBuffer = output.getBuffer();
    int32_t channelCount = output.getSamplesPerFrame();
    int framesLeft = numFrames;
    while (framesLeft > 0) {
        // Gather input samples as needed.
        if(mResampler.isWriteNeeded()) {
            if (isInputAvailable()) {
                const float *frame = getNextInputFrame();
                mResampler.writeNextFrame(frame);
            } else {
                break;
            }
        } else {
            // Output frame is interpolated from input samples.
            mResampler.readNextFrame(outputBuffer);
            outputBuffer += channelCount;
            framesLeft--;
        }
    }
    return numFrames - framesLeft;
}
