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

#ifndef RESAMPLER_LINEAR_RESAMPLER_H
#define RESAMPLER_LINEAR_RESAMPLER_H

#include <memory>
#include <sys/types.h>
#include <unistd.h>

#include "MultiChannelResampler.h"
#include "ResamplerDefinitions.h"

namespace RESAMPLER_OUTER_NAMESPACE::resampler {

/**
 * Simple resampler that uses bi-linear interpolation.
 */
class LinearResampler : public MultiChannelResampler {
public:
    explicit LinearResampler(const MultiChannelResampler::Builder &builder);

    void writeFrame(const float *frame) override;

    void readFrame(float *frame) override;

private:
    std::unique_ptr<float[]> mPreviousFrame;
    std::unique_ptr<float[]> mCurrentFrame;
};

} /* namespace RESAMPLER_OUTER_NAMESPACE::resampler */

#endif //RESAMPLER_LINEAR_RESAMPLER_H
