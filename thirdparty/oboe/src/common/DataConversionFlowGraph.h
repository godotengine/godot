/*
 * Copyright (C) 2019 The Android Open Source Project
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

#ifndef OBOE_OBOE_FLOW_GRAPH_H
#define OBOE_OBOE_FLOW_GRAPH_H

#include <memory>
#include <stdint.h>
#include <sys/types.h>

#include <flowgraph/ChannelCountConverter.h>
#include <flowgraph/MonoToMultiConverter.h>
#include <flowgraph/MultiToMonoConverter.h>
#include <flowgraph/SampleRateConverter.h>
#include <oboe/Definitions.h>
#include "AudioSourceCaller.h"
#include "FixedBlockWriter.h"

namespace oboe {

class AudioStream;
class AudioSourceCaller;

/**
 * Convert PCM channels, format and sample rate for optimal latency.
 */
class DataConversionFlowGraph : public FixedBlockProcessor {
public:

    DataConversionFlowGraph()
    : mBlockWriter(*this) {}

    void setSource(const void *buffer, int32_t numFrames);

    /** Connect several modules together to convert from source to sink.
     * This should only be called once for each instance.
     *
     * @param sourceFormat
     * @param sourceChannelCount
     * @param sinkFormat
     * @param sinkChannelCount
     * @return
     */
    oboe::Result configure(oboe::AudioStream *sourceStream, oboe::AudioStream *sinkStream);

    int32_t read(void *buffer, int32_t numFrames, int64_t timeoutNanos);

    int32_t write(void *buffer, int32_t numFrames);

    int32_t onProcessFixedBlock(uint8_t *buffer, int32_t numBytes) override;

    DataCallbackResult getDataCallbackResult() {
        return mCallbackResult;
    }

private:
    std::unique_ptr<flowgraph::FlowGraphSourceBuffered>    mSource;
    std::unique_ptr<AudioSourceCaller>                 mSourceCaller;
    std::unique_ptr<flowgraph::MonoToMultiConverter>   mMonoToMultiConverter;
    std::unique_ptr<flowgraph::MultiToMonoConverter>   mMultiToMonoConverter;
    std::unique_ptr<flowgraph::ChannelCountConverter>  mChannelCountConverter;
    std::unique_ptr<resampler::MultiChannelResampler>  mResampler;
    std::unique_ptr<flowgraph::SampleRateConverter>    mRateConverter;
    std::unique_ptr<flowgraph::FlowGraphSink>              mSink;

    FixedBlockWriter                                   mBlockWriter;
    DataCallbackResult                                 mCallbackResult = DataCallbackResult::Continue;
    AudioStream                                       *mFilterStream = nullptr;
    std::unique_ptr<uint8_t[]>                         mAppBuffer;
};

}
#endif //OBOE_OBOE_FLOW_GRAPH_H
