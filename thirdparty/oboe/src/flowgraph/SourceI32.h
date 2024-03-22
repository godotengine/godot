/*
 * Copyright 2018 The Android Open Source Project
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

#ifndef FLOWGRAPH_SOURCE_I32_H
#define FLOWGRAPH_SOURCE_I32_H

#include <stdint.h>

#include "FlowGraphNode.h"

namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph {

class SourceI32 : public FlowGraphSourceBuffered {
public:
    explicit SourceI32(int32_t channelCount);
    ~SourceI32() override = default;

    int32_t onProcess(int32_t numFrames) override;

    const char *getName() override {
        return "SourceI32";
    }
private:
    static constexpr float kScale = 1.0 / (1UL << 31);
};

} /* namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph */

#endif //FLOWGRAPH_SOURCE_I32_H
