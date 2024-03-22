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

#include <algorithm>
#include <unistd.h>
#include "flowgraph/FlowGraphNode.h"
#include "SourceI32Caller.h"

#if FLOWGRAPH_ANDROID_INTERNAL
#include <audio_utils/primitives.h>
#endif

using namespace oboe;
using namespace flowgraph;

int32_t SourceI32Caller::onProcess(int32_t numFrames) {
    int32_t numBytes = mStream->getBytesPerFrame() * numFrames;
    int32_t bytesRead = mBlockReader.read((uint8_t *) mConversionBuffer.get(), numBytes);
    int32_t framesRead = bytesRead / mStream->getBytesPerFrame();

    float *floatData = output.getBuffer();
    const int32_t *intData = mConversionBuffer.get();
    int32_t numSamples = framesRead * output.getSamplesPerFrame();

#if FLOWGRAPH_ANDROID_INTERNAL
    memcpy_to_float_from_i32(floatData, shortData, numSamples);
#else
    for (int i = 0; i < numSamples; i++) {
        *floatData++ = *intData++ * kScale;
    }
#endif

    return framesRead;
}
