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

#include "stdio.h"
#include <algorithm>
#include <sys/types.h>
#include "FlowGraphNode.h"

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

/***************************************************************************/
int32_t FlowGraphNode::pullData(int32_t numFrames, int64_t callCount) {
    int32_t frameCount = numFrames;
    // Prevent recursion and multiple execution of nodes.
    if (callCount > mLastCallCount) {
        mLastCallCount = callCount;
        if (mDataPulledAutomatically) {
            // Pull from all the upstream nodes.
            for (auto &port : mInputPorts) {
                // TODO fix bug of leaving unused data in some ports if using multiple AudioSource
                frameCount = port.get().pullData(callCount, frameCount);
            }
        }
        if (frameCount > 0) {
            frameCount = onProcess(frameCount);
        }
        mLastFrameCount = frameCount;
    } else {
        frameCount = mLastFrameCount;
    }
    return frameCount;
}

void FlowGraphNode::pullReset() {
    if (!mBlockRecursion) {
        mBlockRecursion = true; // for cyclic graphs
        // Pull reset from all the upstream nodes.
        for (auto &port : mInputPorts) {
            port.get().pullReset();
        }
        mBlockRecursion = false;
        reset();
    }
}

void FlowGraphNode::reset() {
    mLastFrameCount = 0;
    mLastCallCount = kInitialCallCount;
}

/***************************************************************************/
FlowGraphPortFloat::FlowGraphPortFloat(FlowGraphNode &parent,
                               int32_t samplesPerFrame,
                               int32_t framesPerBuffer)
        : FlowGraphPort(parent, samplesPerFrame)
        , mFramesPerBuffer(framesPerBuffer)
        , mBuffer(nullptr) {
    size_t numFloats = static_cast<size_t>(framesPerBuffer) * getSamplesPerFrame();
    mBuffer = std::make_unique<float[]>(numFloats);
}

/***************************************************************************/
int32_t FlowGraphPortFloatOutput::pullData(int64_t callCount, int32_t numFrames) {
    numFrames = std::min(getFramesPerBuffer(), numFrames);
    return mContainingNode.pullData(numFrames, callCount);
}

void FlowGraphPortFloatOutput::pullReset() {
    mContainingNode.pullReset();
}

// These need to be in the .cpp file because of forward cross references.
void FlowGraphPortFloatOutput::connect(FlowGraphPortFloatInput *port) {
    port->connect(this);
}

void FlowGraphPortFloatOutput::disconnect(FlowGraphPortFloatInput *port) {
    port->disconnect(this);
}

/***************************************************************************/
int32_t FlowGraphPortFloatInput::pullData(int64_t callCount, int32_t numFrames) {
    return (mConnected == nullptr)
            ? std::min(getFramesPerBuffer(), numFrames)
            : mConnected->pullData(callCount, numFrames);
}
void FlowGraphPortFloatInput::pullReset() {
    if (mConnected != nullptr) mConnected->pullReset();
}

float *FlowGraphPortFloatInput::getBuffer() {
    if (mConnected == nullptr) {
        return FlowGraphPortFloat::getBuffer(); // loaded using setValue()
    } else {
        return mConnected->getBuffer();
    }
}

int32_t FlowGraphSink::pullData(int32_t numFrames) {
    return FlowGraphNode::pullData(numFrames, getLastCallCount() + 1);
}
