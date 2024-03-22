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

#include "SincResamplerStereo.h"

using namespace RESAMPLER_OUTER_NAMESPACE::resampler;

#define STEREO  2

SincResamplerStereo::SincResamplerStereo(const MultiChannelResampler::Builder &builder)
        : SincResampler(builder) {
    assert(builder.getChannelCount() == STEREO);
}

void SincResamplerStereo::writeFrame(const float *frame) {
    // Move cursor before write so that cursor points to last written frame in read.
    if (--mCursor < 0) {
        mCursor = getNumTaps() - 1;
    }
    float *dest = &mX[mCursor * STEREO];
    const int offset = mNumTaps * STEREO;
    // Write each channel twice so we avoid having to wrap when running the FIR.
    const float left =  frame[0];
    const float right = frame[1];
    // Put ordered writes together.
    dest[0] = left;
    dest[1] = right;
    dest[offset] = left;
    dest[1 + offset] = right;
}

// Multiply input times windowed sinc function.
void SincResamplerStereo::readFrame(float *frame) {
    // Clear accumulator for mixing.
    std::fill(mSingleFrame.begin(), mSingleFrame.end(), 0.0);
    std::fill(mSingleFrame2.begin(), mSingleFrame2.end(), 0.0);

    // Determine indices into coefficients table.
    double tablePhase = getIntegerPhase() * mPhaseScaler;
    int index1 = static_cast<int>(floor(tablePhase));
    float *coefficients1 = &mCoefficients[static_cast<size_t>(index1)
            * static_cast<size_t>(getNumTaps())];
    int index2 = (index1 + 1);
    float *coefficients2 = &mCoefficients[static_cast<size_t>(index2)
            * static_cast<size_t>(getNumTaps())];
    float *xFrame = &mX[static_cast<size_t>(mCursor) * static_cast<size_t>(getChannelCount())];
    for (int i = 0; i < mNumTaps; i++) {
        float coefficient1 = *coefficients1++;
        float coefficient2 = *coefficients2++;
        for (int channel = 0; channel < getChannelCount(); channel++) {
            float sample = *xFrame++;
            mSingleFrame[channel] +=  sample * coefficient1;
            mSingleFrame2[channel] += sample * coefficient2;
        }
    }

    // Interpolate and copy to output.
    float fraction = tablePhase - index1;
    for (int channel = 0; channel < getChannelCount(); channel++) {
        float low = mSingleFrame[channel];
        float high = mSingleFrame2[channel];
        frame[channel] = low + (fraction * (high - low));
    }
}
