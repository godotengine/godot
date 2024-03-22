/*
 * Copyright 2022 The Android Open Source Project
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

#ifndef FLOWGRAPH_LIMITER_H
#define FLOWGRAPH_LIMITER_H

#include <atomic>
#include <unistd.h>
#include <sys/types.h>

#include "FlowGraphNode.h"

namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph {

class Limiter : public FlowGraphFilter {
public:
    explicit Limiter(int32_t channelCount);

    int32_t onProcess(int32_t numFrames) override;

    const char *getName() override {
        return "Limiter";
    }

private:
    // These numbers are based on a polynomial spline for a quadratic solution Ax^2 + Bx + C
    // The range is up to 3 dB, (10^(3/20)), to match AudioTrack for float data.
    static constexpr float kPolynomialSplineA = -0.6035533905; // -(1+sqrt(2))/4
    static constexpr float kPolynomialSplineB = 2.2071067811; // (3+sqrt(2))/2
    static constexpr float kPolynomialSplineC = -0.6035533905; // -(1+sqrt(2))/4
    static constexpr float kXWhenYis3Decibels = 1.8284271247; // -1+2sqrt(2)

    /**
     * Process an input based on the following:
     * If between -1 and 1, return the input value.
     * If above kXWhenYis3Decibels, return sqrt(2).
     * If below -kXWhenYis3Decibels, return -sqrt(2).
     * If between 1 and kXWhenYis3Decibels, use a quadratic spline (Ax^2 + Bx + C).
     * If between -kXWhenYis3Decibels and -1, use the absolute value for the spline and flip it.
     * The derivative of the spline is 1 at 1 and 0 at kXWhenYis3Decibels.
     * This way, the graph is both continuous and differentiable.
     */
    float processFloat(float in);

    // Use the previous valid output for NaN inputs
    float mLastValidOutput = 0.0f;
};

} /* namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph */

#endif //FLOWGRAPH_LIMITER_H
