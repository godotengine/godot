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

#ifndef RESAMPLER_POLYPHASE_RESAMPLER_H
#define RESAMPLER_POLYPHASE_RESAMPLER_H

#include <memory>
#include <vector>
#include <sys/types.h>
#include <unistd.h>

#include "MultiChannelResampler.h"
#include "ResamplerDefinitions.h"

namespace RESAMPLER_OUTER_NAMESPACE::resampler {
/**
 * Resampler that is optimized for a reduced ratio of sample rates.
 * All of the coefficients for each possible phase value are pre-calculated.
 */
class PolyphaseResampler : public MultiChannelResampler {
public:
    /**
     *
     * @param builder containing lots of parameters
     */
    explicit PolyphaseResampler(const MultiChannelResampler::Builder &builder);

    virtual ~PolyphaseResampler() = default;

    void readFrame(float *frame) override;

protected:

    int32_t                mCoefficientCursor = 0;

};

} /* namespace RESAMPLER_OUTER_NAMESPACE::resampler */

#endif //RESAMPLER_POLYPHASE_RESAMPLER_H
