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

#include <memory>

#include "OboeDebug.h"
#include "DataConversionFlowGraph.h"
#include "SourceFloatCaller.h"
#include "SourceI16Caller.h"
#include "SourceI24Caller.h"
#include "SourceI32Caller.h"

#include <flowgraph/MonoToMultiConverter.h>
#include <flowgraph/MultiToMonoConverter.h>
#include <flowgraph/RampLinear.h>
#include <flowgraph/SinkFloat.h>
#include <flowgraph/SinkI16.h>
#include <flowgraph/SinkI24.h>
#include <flowgraph/SinkI32.h>
#include <flowgraph/SourceFloat.h>
#include <flowgraph/SourceI16.h>
#include <flowgraph/SourceI24.h>
#include <flowgraph/SourceI32.h>
#include <flowgraph/SampleRateConverter.h>

using namespace oboe;
using namespace flowgraph;
using namespace resampler;

void DataConversionFlowGraph::setSource(const void *buffer, int32_t numFrames) {
    mSource->setData(buffer, numFrames);
}

static MultiChannelResampler::Quality convertOboeSRQualityToMCR(SampleRateConversionQuality quality) {
    switch (quality) {
        case SampleRateConversionQuality::Fastest:
            return MultiChannelResampler::Quality::Fastest;
        case SampleRateConversionQuality::Low:
            return MultiChannelResampler::Quality::Low;
        default:
        case SampleRateConversionQuality::Medium:
            return MultiChannelResampler::Quality::Medium;
        case SampleRateConversionQuality::High:
            return MultiChannelResampler::Quality::High;
        case SampleRateConversionQuality::Best:
            return MultiChannelResampler::Quality::Best;
    }
}

// Chain together multiple processors.
// Callback Output
//     Use SourceCaller that calls original app callback from the flowgraph.
//     The child callback from FilteredAudioStream read()s from the flowgraph.
// Callback Input
//     Child callback from FilteredAudioStream writes()s to the flowgraph.
//     The output of the flowgraph goes through a BlockWriter to the app callback.
// Blocking Write
//     Write buffer is set on an AudioSource.
//     Data is pulled through the graph and written to the child stream.
// Blocking Read
//     Reads in a loop from the flowgraph Sink to fill the read buffer.
//     A SourceCaller then does a blocking read from the child Stream.
//
Result DataConversionFlowGraph::configure(AudioStream *sourceStream, AudioStream *sinkStream) {

    FlowGraphPortFloatOutput *lastOutput = nullptr;

    bool isOutput = sourceStream->getDirection() == Direction::Output;
    bool isInput = !isOutput;
    mFilterStream = isOutput ? sourceStream : sinkStream;

    AudioFormat sourceFormat = sourceStream->getFormat();
    int32_t sourceChannelCount = sourceStream->getChannelCount();
    int32_t sourceSampleRate = sourceStream->getSampleRate();
    int32_t sourceFramesPerCallback = sourceStream->getFramesPerDataCallback();

    AudioFormat sinkFormat = sinkStream->getFormat();
    int32_t sinkChannelCount = sinkStream->getChannelCount();
    int32_t sinkSampleRate = sinkStream->getSampleRate();
    int32_t sinkFramesPerCallback = sinkStream->getFramesPerDataCallback();

    LOGI("%s() flowgraph converts channels: %d to %d, format: %d to %d"
         ", rate: %d to %d, cbsize: %d to %d, qual = %d",
            __func__,
            sourceChannelCount, sinkChannelCount,
            sourceFormat, sinkFormat,
            sourceSampleRate, sinkSampleRate,
            sourceFramesPerCallback, sinkFramesPerCallback,
            sourceStream->getSampleRateConversionQuality());

    // Source
    // IF OUTPUT and using a callback then call back to the app using a SourceCaller.
    // OR IF INPUT and NOT using a callback then read from the child stream using a SourceCaller.
    bool isDataCallbackSpecified = sourceStream->isDataCallbackSpecified();
    if ((isDataCallbackSpecified && isOutput)
        || (!isDataCallbackSpecified && isInput)) {
        int32_t actualSourceFramesPerCallback = (sourceFramesPerCallback == kUnspecified)
                ? sourceStream->getFramesPerBurst()
                : sourceFramesPerCallback;
        switch (sourceFormat) {
            case AudioFormat::Float:
                mSourceCaller = std::make_unique<SourceFloatCaller>(sourceChannelCount,
                                                                    actualSourceFramesPerCallback);
                break;
            case AudioFormat::I16:
                mSourceCaller = std::make_unique<SourceI16Caller>(sourceChannelCount,
                                                                  actualSourceFramesPerCallback);
                break;
            case AudioFormat::I24:
                mSourceCaller = std::make_unique<SourceI24Caller>(sourceChannelCount,
                                                                  actualSourceFramesPerCallback);
                break;
            case AudioFormat::I32:
                mSourceCaller = std::make_unique<SourceI32Caller>(sourceChannelCount,
                                                                  actualSourceFramesPerCallback);
                break;
            default:
                LOGE("%s() Unsupported source caller format = %d", __func__, sourceFormat);
                return Result::ErrorIllegalArgument;
        }
        mSourceCaller->setStream(sourceStream);
        lastOutput = &mSourceCaller->output;
    } else {
        // IF OUTPUT and NOT using a callback then write to the child stream using a BlockWriter.
        // OR IF INPUT and using a callback then write to the app using a BlockWriter.
        switch (sourceFormat) {
            case AudioFormat::Float:
                mSource = std::make_unique<SourceFloat>(sourceChannelCount);
                break;
            case AudioFormat::I16:
                mSource = std::make_unique<SourceI16>(sourceChannelCount);
                break;
            case AudioFormat::I24:
                mSource = std::make_unique<SourceI24>(sourceChannelCount);
                break;
            case AudioFormat::I32:
                mSource = std::make_unique<SourceI32>(sourceChannelCount);
                break;
            default:
                LOGE("%s() Unsupported source format = %d", __func__, sourceFormat);
                return Result::ErrorIllegalArgument;
        }
        if (isInput) {
            int32_t actualSinkFramesPerCallback = (sinkFramesPerCallback == kUnspecified)
                    ? sinkStream->getFramesPerBurst()
                    : sinkFramesPerCallback;
            // The BlockWriter is after the Sink so use the SinkStream size.
            mBlockWriter.open(actualSinkFramesPerCallback * sinkStream->getBytesPerFrame());
            mAppBuffer = std::make_unique<uint8_t[]>(
                    kDefaultBufferSize * sinkStream->getBytesPerFrame());
        }
        lastOutput = &mSource->output;
    }

    // If we are going to reduce the number of channels then do it before the
    // sample rate converter.
    if (sourceChannelCount > sinkChannelCount) {
        if (sinkChannelCount == 1) {
            mMultiToMonoConverter = std::make_unique<MultiToMonoConverter>(sourceChannelCount);
            lastOutput->connect(&mMultiToMonoConverter->input);
            lastOutput = &mMultiToMonoConverter->output;
        } else {
            mChannelCountConverter = std::make_unique<ChannelCountConverter>(
                    sourceChannelCount,
                    sinkChannelCount);
            lastOutput->connect(&mChannelCountConverter->input);
            lastOutput = &mChannelCountConverter->output;
        }
    }

    // Sample Rate conversion
    if (sourceSampleRate != sinkSampleRate) {
        // Create a resampler to do the math.
        mResampler.reset(MultiChannelResampler::make(lastOutput->getSamplesPerFrame(),
                                                     sourceSampleRate,
                                                     sinkSampleRate,
                                                     convertOboeSRQualityToMCR(
                                                             sourceStream->getSampleRateConversionQuality())));
        // Make a flowgraph node that uses the resampler.
        mRateConverter = std::make_unique<SampleRateConverter>(lastOutput->getSamplesPerFrame(),
                                                               *mResampler.get());
        lastOutput->connect(&mRateConverter->input);
        lastOutput = &mRateConverter->output;
    }

    // Expand the number of channels if required.
    if (sourceChannelCount < sinkChannelCount) {
        if (sourceChannelCount == 1) {
            mMonoToMultiConverter = std::make_unique<MonoToMultiConverter>(sinkChannelCount);
            lastOutput->connect(&mMonoToMultiConverter->input);
            lastOutput = &mMonoToMultiConverter->output;
        } else {
            mChannelCountConverter = std::make_unique<ChannelCountConverter>(
                    sourceChannelCount,
                    sinkChannelCount);
            lastOutput->connect(&mChannelCountConverter->input);
            lastOutput = &mChannelCountConverter->output;
        }
    }

    // Sink
    switch (sinkFormat) {
        case AudioFormat::Float:
            mSink = std::make_unique<SinkFloat>(sinkChannelCount);
            break;
        case AudioFormat::I16:
            mSink = std::make_unique<SinkI16>(sinkChannelCount);
            break;
        case AudioFormat::I24:
            mSink = std::make_unique<SinkI24>(sinkChannelCount);
            break;
        case AudioFormat::I32:
            mSink = std::make_unique<SinkI32>(sinkChannelCount);
            break;
        default:
            LOGE("%s() Unsupported sink format = %d", __func__, sinkFormat);
            return Result::ErrorIllegalArgument;;
    }
    lastOutput->connect(&mSink->input);

    return Result::OK;
}

int32_t DataConversionFlowGraph::read(void *buffer, int32_t numFrames, int64_t timeoutNanos) {
    if (mSourceCaller) {
        mSourceCaller->setTimeoutNanos(timeoutNanos);
    }
    int32_t numRead = mSink->read(buffer, numFrames);
    return numRead;
}

// This is similar to pushing data through the flowgraph.
int32_t DataConversionFlowGraph::write(void *inputBuffer, int32_t numFrames) {
    // Put the data from the input at the head of the flowgraph.
    mSource->setData(inputBuffer, numFrames);
    while (true) {
        // Pull and read some data in app format into a small buffer.
        int32_t framesRead = mSink->read(mAppBuffer.get(), flowgraph::kDefaultBufferSize);
        if (framesRead <= 0) break;
        // Write to a block adapter, which will call the destination whenever it has enough data.
        int32_t bytesRead = mBlockWriter.write(mAppBuffer.get(),
                                               framesRead * mFilterStream->getBytesPerFrame());
        if (bytesRead < 0) return bytesRead; // TODO review
    }
    return numFrames;
}

int32_t DataConversionFlowGraph::onProcessFixedBlock(uint8_t *buffer, int32_t numBytes) {
    int32_t numFrames = numBytes / mFilterStream->getBytesPerFrame();
    mCallbackResult = mFilterStream->getDataCallback()->onAudioReady(mFilterStream, buffer, numFrames);
    // TODO handle STOP from callback, process data remaining in the block adapter
    return numBytes;
}
