/*
 * Copyright 2023 The Android Open Source Project
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

#ifndef FLOWGRAPH_SINK_I8_24_H
#define FLOWGRAPH_SINK_I8_24_H

#include <stdint.h>

#include "FlowGraphNode.h"

namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph {

    class SinkI8_24 : public FlowGraphSink {
    public:
        explicit SinkI8_24(int32_t channelCount);
        ~SinkI8_24() override = default;

        int32_t read(void *data, int32_t numFrames) override;

        const char *getName() override {
            return "SinkI8_24";
        }
    };

} /* namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph */

#endif //FLOWGRAPH_SINK_I8_24_H
