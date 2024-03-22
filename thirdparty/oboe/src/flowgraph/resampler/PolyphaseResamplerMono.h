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

#ifndef RESAMPLER_POLYPHASE_RESAMPLER_MONO_H
#define RESAMPLER_POLYPHASE_RESAMPLER_MONO_H

#include <sys/types.h>
#include <unistd.h>

#include "PolyphaseResampler.h"
#include "ResamplerDefinitions.h"

namespace RESAMPLER_OUTER_NAMESPACE::resampler {

class PolyphaseResamplerMono : public PolyphaseResampler {
public:
    explicit PolyphaseResamplerMono(const MultiChannelResampler::Builder &builder);

    virtual ~PolyphaseResamplerMono() = default;

    void writeFrame(const float *frame) override;

    void readFrame(float *frame) override;
};

} /* namespace RESAMPLER_OUTER_NAMESPACE::resampler */

#endif //RESAMPLER_POLYPHASE_RESAMPLER_MONO_H
