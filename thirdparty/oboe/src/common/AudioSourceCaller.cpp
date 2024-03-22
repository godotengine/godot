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

#include "AudioSourceCaller.h"

using namespace oboe;
using namespace flowgraph;

int32_t AudioSourceCaller::onProcessFixedBlock(uint8_t *buffer, int32_t numBytes) {
    AudioStreamDataCallback *callback = mStream->getDataCallback();
    int32_t result = 0;
    int32_t numFrames = numBytes / mStream->getBytesPerFrame();
    if (callback != nullptr) {
        DataCallbackResult callbackResult = callback->onAudioReady(mStream, buffer, numFrames);
        // onAudioReady() does not return the number of bytes processed so we have to assume all.
        result = (callbackResult == DataCallbackResult::Continue)
                ? numBytes
                : -1;
    } else {
        auto readResult = mStream->read(buffer, numFrames, mTimeoutNanos);
        if (!readResult) return (int32_t) readResult.error();
        result = readResult.value() * mStream->getBytesPerFrame();
    }
    return result;
}
