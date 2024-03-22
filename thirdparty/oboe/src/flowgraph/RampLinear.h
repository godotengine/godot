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

#ifndef FLOWGRAPH_RAMP_LINEAR_H
#define FLOWGRAPH_RAMP_LINEAR_H

#include <atomic>
#include <unistd.h>
#include <sys/types.h>

#include "FlowGraphNode.h"

namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph {

/**
 * When the target is modified then the output will ramp smoothly
 * between the original and the new target value.
 * This can be used to smooth out control values and reduce pops.
 *
 * The target may be updated while a ramp is in progress, which will trigger
 * a new ramp from the current value.
 */
class RampLinear : public FlowGraphFilter {
public:
    explicit RampLinear(int32_t channelCount);

    virtual ~RampLinear() = default;

    int32_t onProcess(int32_t numFrames) override;

    /**
     * This is used for the next ramp.
     * Calling this does not affect a ramp that is in progress.
     */
    void setLengthInFrames(int32_t frames);

    int32_t getLengthInFrames() const {
        return mLengthInFrames;
    }

    /**
     * This may be safely called by another thread.
     * @param target
     */
    void setTarget(float target);

    float getTarget() const {
        return mTarget.load();
    }

    /**
     * Force the nextSegment to start from this level.
     *
     * WARNING: this can cause a discontinuity if called while the ramp is being used.
     * Only call this when setting the initial ramp.
     *
     * @param level
     */
    void forceCurrent(float level) {
        mLevelFrom = level;
        mLevelTo = level;
    }

    const char *getName() override {
        return "RampLinear";
    }

private:

    float interpolateCurrent();

    std::atomic<float>  mTarget;

    int32_t             mLengthInFrames  = 48000.0f / 100.0f ; // 10 msec at 48000 Hz;
    int32_t             mRemaining       = 0;
    float               mScaler          = 0.0f;
    float               mLevelFrom       = 0.0f;
    float               mLevelTo         = 0.0f;
};

} /* namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph */

#endif //FLOWGRAPH_RAMP_LINEAR_H
