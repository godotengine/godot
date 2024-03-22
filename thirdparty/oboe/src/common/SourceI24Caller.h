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

#ifndef OBOE_SOURCE_I24_CALLER_H
#define OBOE_SOURCE_I24_CALLER_H

#include <unistd.h>
#include <sys/types.h>

#include "flowgraph/FlowGraphNode.h"
#include "AudioSourceCaller.h"
#include "FixedBlockReader.h"

namespace oboe {

/**
 * AudioSource that uses callback to get more data.
 */
class SourceI24Caller : public AudioSourceCaller {
public:
    SourceI24Caller(int32_t channelCount, int32_t framesPerCallback)
    : AudioSourceCaller(channelCount, framesPerCallback, kBytesPerI24Packed) {
        mConversionBuffer = std::make_unique<uint8_t[]>(static_cast<size_t>(kBytesPerI24Packed)
                * static_cast<size_t>(channelCount)
                * static_cast<size_t>(output.getFramesPerBuffer()));
    }

    int32_t onProcess(int32_t numFrames) override;

    const char *getName() override {
        return "SourceI24Caller";
    }

private:
    std::unique_ptr<uint8_t[]>  mConversionBuffer;
    static constexpr int kBytesPerI24Packed = 3;
};

}
#endif //OBOE_SOURCE_I16_CALLER_H
