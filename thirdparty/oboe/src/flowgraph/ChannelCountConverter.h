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

#ifndef FLOWGRAPH_CHANNEL_COUNT_CONVERTER_H
#define FLOWGRAPH_CHANNEL_COUNT_CONVERTER_H

#include <unistd.h>
#include <sys/types.h>

#include "FlowGraphNode.h"

namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph {

/**
 * Change the number of number of channels without mixing.
 * When increasing the channel count, duplicate input channels.
 * When decreasing the channel count, drop input channels.
 */
    class ChannelCountConverter : public FlowGraphNode {
    public:
        explicit ChannelCountConverter(
                int32_t inputChannelCount,
                int32_t outputChannelCount);

        virtual ~ChannelCountConverter();

        int32_t onProcess(int32_t numFrames) override;

        const char *getName() override {
            return "ChannelCountConverter";
        }

        FlowGraphPortFloatInput input;
        FlowGraphPortFloatOutput output;
    };

} /* namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph */

#endif //FLOWGRAPH_CHANNEL_COUNT_CONVERTER_H
