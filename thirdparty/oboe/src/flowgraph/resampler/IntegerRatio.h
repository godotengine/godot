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

#ifndef RESAMPLER_INTEGER_RATIO_H
#define RESAMPLER_INTEGER_RATIO_H

#include <sys/types.h>

#include "ResamplerDefinitions.h"

namespace RESAMPLER_OUTER_NAMESPACE::resampler {

/**
 * Represent the ratio of two integers.
 */
class IntegerRatio {
public:
    IntegerRatio(int32_t numerator, int32_t denominator)
            : mNumerator(numerator), mDenominator(denominator) {}

    /**
     * Reduce by removing common prime factors.
     */
    void reduce();

    int32_t getNumerator() {
        return mNumerator;
    }

    int32_t getDenominator() {
        return mDenominator;
    }

private:
    int32_t mNumerator;
    int32_t mDenominator;
};

} /* namespace RESAMPLER_OUTER_NAMESPACE::resampler */

#endif //RESAMPLER_INTEGER_RATIO_H
