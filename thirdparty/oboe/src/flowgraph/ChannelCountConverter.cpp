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

#include <unistd.h>
#include "FlowGraphNode.h"
#include "ChannelCountConverter.h"

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

ChannelCountConverter::ChannelCountConverter(
        int32_t inputChannelCount,
        int32_t outputChannelCount)
        : input(*this, inputChannelCount)
        , output(*this, outputChannelCount) {
}

ChannelCountConverter::~ChannelCountConverter() = default;

int32_t ChannelCountConverter::onProcess(int32_t numFrames) {
    const float *inputBuffer = input.getBuffer();
    float *outputBuffer = output.getBuffer();
    int32_t inputChannelCount = input.getSamplesPerFrame();
    int32_t outputChannelCount = output.getSamplesPerFrame();
    for (int i = 0; i < numFrames; i++) {
        int inputChannel = 0;
        for (int outputChannel = 0; outputChannel < outputChannelCount; outputChannel++) {
            // Copy input channels to output channels.
            // Wrap if we run out of inputs.
            // Discard if we run out of outputs.
            outputBuffer[outputChannel] = inputBuffer[inputChannel];
            inputChannel = (inputChannel == inputChannelCount)
                    ? 0 : inputChannel + 1;
        }
        inputBuffer += inputChannelCount;
        outputBuffer += outputChannelCount;
    }
    return numFrames;
}

