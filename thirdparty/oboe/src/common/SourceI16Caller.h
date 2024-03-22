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

#ifndef OBOE_SOURCE_I16_CALLER_H
#define OBOE_SOURCE_I16_CALLER_H

#include <unistd.h>
#include <sys/types.h>

#include "flowgraph/FlowGraphNode.h"
#include "AudioSourceCaller.h"
#include "FixedBlockReader.h"

namespace oboe {
/**
 * AudioSource that uses callback to get more data.
 */
class SourceI16Caller : public AudioSourceCaller {
public:
    SourceI16Caller(int32_t channelCount, int32_t framesPerCallback)
    : AudioSourceCaller(channelCount, framesPerCallback, sizeof(int16_t)) {
        mConversionBuffer = std::make_unique<int16_t[]>(static_cast<size_t>(channelCount)
                * static_cast<size_t>(output.getFramesPerBuffer()));
    }

    int32_t onProcess(int32_t numFrames) override;

    const char *getName() override {
        return "SourceI16Caller";
    }
private:
    std::unique_ptr<int16_t[]>  mConversionBuffer;
};

}
#endif //OBOE_SOURCE_I16_CALLER_H
