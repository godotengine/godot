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

#include <algorithm>   // Do NOT delete. Needed for LLVM. See #1746
#include <cassert>
#include <math.h>
#include "IntegerRatio.h"
#include "PolyphaseResampler.h"

using namespace RESAMPLER_OUTER_NAMESPACE::resampler;

PolyphaseResampler::PolyphaseResampler(const MultiChannelResampler::Builder &builder)
        : MultiChannelResampler(builder)
        {
    assert((getNumTaps() % 4) == 0); // Required for loop unrolling.

    int32_t inputRate = builder.getInputRate();
    int32_t outputRate = builder.getOutputRate();

    int32_t numRows = mDenominator;
    double phaseIncrement = (double) inputRate / (double) outputRate;
    generateCoefficients(inputRate, outputRate,
                         numRows, phaseIncrement,
                         builder.getNormalizedCutoff());
}

void PolyphaseResampler::readFrame(float *frame) {
    // Clear accumulator for mixing.
    std::fill(mSingleFrame.begin(), mSingleFrame.end(), 0.0);

    // Multiply input times windowed sinc function.
    float *coefficients = &mCoefficients[mCoefficientCursor];
    float *xFrame = &mX[static_cast<size_t>(mCursor) * static_cast<size_t>(getChannelCount())];
    for (int i = 0; i < mNumTaps; i++) {
        float coefficient = *coefficients++;
        for (int channel = 0; channel < getChannelCount(); channel++) {
            mSingleFrame[channel] += *xFrame++ * coefficient;
        }
    }

    // Advance and wrap through coefficients.
    mCoefficientCursor = (mCoefficientCursor + mNumTaps) % mCoefficients.size();

    // Copy accumulator to output.
    for (int channel = 0; channel < getChannelCount(); channel++) {
        frame[channel] = mSingleFrame[channel];
    }
}
