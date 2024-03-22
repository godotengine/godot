/*
 * Copyright 2020 The Android Open Source Project
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

#ifndef FLOWGRAPH_UTILITIES_H
#define FLOWGRAPH_UTILITIES_H

#include <math.h>
#include <unistd.h>

using namespace FLOWGRAPH_OUTER_NAMESPACE::flowgraph;

class FlowgraphUtilities {
public:
// This was copied from audio_utils/primitives.h
/**
 * Convert a single-precision floating point value to a Q0.31 integer value.
 * Rounds to nearest, ties away from 0.
 *
 * Values outside the range [-1.0, 1.0) are properly clamped to -2147483648 and 2147483647,
 * including -Inf and +Inf. NaN values are considered undefined, and behavior may change
 * depending on hardware and future implementation of this function.
 */
static int32_t clamp32FromFloat(float f)
{
    static const float scale = (float)(1UL << 31);
    static const float limpos = 1.;
    static const float limneg = -1.;

    if (f <= limneg) {
        return INT32_MIN;
    } else if (f >= limpos) {
        return INT32_MAX;
    }
    f *= scale;
    /* integer conversion is through truncation (though int to float is not).
     * ensure that we round to nearest, ties away from 0.
     */
    return f > 0 ? f + 0.5 : f - 0.5;
}

/**
 * Convert a single-precision floating point value to a Q0.23 integer value, stored in a
 * 32 bit signed integer (technically stored as Q8.23, but clamped to Q0.23).
 *
 * Values outside the range [-1.0, 1.0) are properly clamped to -8388608 and 8388607,
 * including -Inf and +Inf. NaN values are considered undefined, and behavior may change
 * depending on hardware and future implementation of this function.
 */
static int32_t clamp24FromFloat(float f)
{
    static const float scale = 1 << 23;
    return (int32_t) lroundf(fmaxf(fminf(f * scale, scale - 1.f), -scale));
}

};

#endif // FLOWGRAPH_UTILITIES_H
