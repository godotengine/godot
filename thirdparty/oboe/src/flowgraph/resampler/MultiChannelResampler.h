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

#ifndef RESAMPLER_MULTICHANNEL_RESAMPLER_H
#define RESAMPLER_MULTICHANNEL_RESAMPLER_H

#include <memory>
#include <vector>
#include <sys/types.h>
#include <unistd.h>

#ifndef MCR_USE_KAISER
// It appears from the spectrogram that the HyperbolicCosine window leads to fewer artifacts.
// And it is faster to calculate.
#define MCR_USE_KAISER 0
#endif

#if MCR_USE_KAISER
#include "KaiserWindow.h"
#else
#include "HyperbolicCosineWindow.h"
#endif

#include "ResamplerDefinitions.h"

namespace RESAMPLER_OUTER_NAMESPACE::resampler {

class MultiChannelResampler {

public:

    enum class Quality : int32_t {
        Fastest,
        Low,
        Medium,
        High,
        Best,
    };

    class Builder {
    public:
        /**
         * Construct an optimal resampler based on the specified parameters.
         * @return address of a resampler
         */
        MultiChannelResampler *build();

        /**
         * The number of taps in the resampling filter.
         * More taps gives better quality but uses more CPU time.
         * This typically ranges from 4 to 64. Default is 16.
         *
         * For polyphase filters, numTaps must be a multiple of four for loop unrolling.
         * @param numTaps number of taps for the filter
         * @return address of this builder for chaining calls
         */
        Builder *setNumTaps(int32_t numTaps) {
            mNumTaps = numTaps;
            return this;
        }

        /**
         * Use 1 for mono, 2 for stereo, etc. Default is 1.
         *
         * @param channelCount number of channels
         * @return address of this builder for chaining calls
         */
        Builder *setChannelCount(int32_t channelCount) {
            mChannelCount = channelCount;
            return this;
        }

        /**
         * Default is 48000.
         *
         * @param inputRate sample rate of the input stream
         * @return address of this builder for chaining calls
         */
        Builder *setInputRate(int32_t inputRate) {
            mInputRate = inputRate;
            return this;
        }

        /**
         * Default is 48000.
         *
         * @param outputRate sample rate of the output stream
         * @return address of this builder for chaining calls
         */
        Builder *setOutputRate(int32_t outputRate) {
            mOutputRate = outputRate;
            return this;
        }

        /**
         * Set cutoff frequency relative to the Nyquist rate of the output sample rate.
         * Set to 1.0 to match the Nyquist frequency.
         * Set lower to reduce aliasing.
         * Default is 0.70.
         *
         * Note that this value is ignored when upsampling, which is when
         * the outputRate is higher than the inputRate.
         *
         * @param normalizedCutoff anti-aliasing filter cutoff
         * @return address of this builder for chaining calls
         */
        Builder *setNormalizedCutoff(float normalizedCutoff) {
            mNormalizedCutoff = normalizedCutoff;
            return this;
        }

        int32_t getNumTaps() const {
            return mNumTaps;
        }

        int32_t getChannelCount() const {
            return mChannelCount;
        }

        int32_t getInputRate() const {
            return mInputRate;
        }

        int32_t getOutputRate() const {
            return mOutputRate;
        }

        float getNormalizedCutoff() const {
            return mNormalizedCutoff;
        }

    protected:
        int32_t mChannelCount = 1;
        int32_t mNumTaps = 16;
        int32_t mInputRate = 48000;
        int32_t mOutputRate = 48000;
        float   mNormalizedCutoff = kDefaultNormalizedCutoff;
    };

    virtual ~MultiChannelResampler() = default;

    /**
     * Factory method for making a resampler that is optimal for the given inputs.
     *
     * @param channelCount number of channels, 2 for stereo
     * @param inputRate sample rate of the input stream
     * @param outputRate  sample rate of the output stream
     * @param quality higher quality sounds better but uses more CPU
     * @return an optimal resampler
     */
    static MultiChannelResampler *make(int32_t channelCount,
                                       int32_t inputRate,
                                       int32_t outputRate,
                                       Quality quality);

    bool isWriteNeeded() const {
        return mIntegerPhase >= mDenominator;
    }

    /**
     * Write a frame containing N samples.
     *
     * @param frame pointer to the first sample in a frame
     */
    void writeNextFrame(const float *frame) {
        writeFrame(frame);
        advanceWrite();
    }

    /**
     * Read a frame containing N samples.
     *
     * @param frame pointer to the first sample in a frame
     */
    void readNextFrame(float *frame) {
        readFrame(frame);
        advanceRead();
    }

    int getNumTaps() const {
        return mNumTaps;
    }

    int getChannelCount() const {
        return mChannelCount;
    }

    static float hammingWindow(float radians, float spread);

    static float sinc(float radians);

protected:

    explicit MultiChannelResampler(const MultiChannelResampler::Builder &builder);

    /**
     * Write a frame containing N samples.
     * Call advanceWrite() after calling this.
     * @param frame pointer to the first sample in a frame
     */
    virtual void writeFrame(const float *frame);

    /**
     * Read a frame containing N samples using interpolation.
     * Call advanceRead() after calling this.
     * @param frame pointer to the first sample in a frame
     */
    virtual void readFrame(float *frame) = 0;

    void advanceWrite() {
        mIntegerPhase -= mDenominator;
    }

    void advanceRead() {
        mIntegerPhase += mNumerator;
    }

    /**
     * Generate the filter coefficients in optimal order.
     *
     * Note that normalizedCutoff is ignored when upsampling, which is when
     * the outputRate is higher than the inputRate.
     *
     * @param inputRate sample rate of the input stream
     * @param outputRate  sample rate of the output stream
     * @param numRows number of rows in the array that contain a set of tap coefficients
     * @param phaseIncrement how much to increment the phase between rows
     * @param normalizedCutoff filter cutoff frequency normalized to Nyquist rate of output
     */
    void generateCoefficients(int32_t inputRate,
                              int32_t outputRate,
                              int32_t numRows,
                              double phaseIncrement,
                              float normalizedCutoff);


    int32_t getIntegerPhase() {
        return mIntegerPhase;
    }

    static constexpr int kMaxCoefficients = 8 * 1024;
    std::vector<float>   mCoefficients;

    const int            mNumTaps;
    int                  mCursor = 0;
    std::vector<float>   mX;           // delayed input values for the FIR
    std::vector<float>   mSingleFrame; // one frame for temporary use
    int32_t              mIntegerPhase = 0;
    int32_t              mNumerator = 0;
    int32_t              mDenominator = 0;


private:

#if MCR_USE_KAISER
    KaiserWindow           mKaiserWindow;
#else
    HyperbolicCosineWindow mCoshWindow;
#endif

    static constexpr float kDefaultNormalizedCutoff = 0.70f;

    const int              mChannelCount;
};

} /* namespace RESAMPLER_OUTER_NAMESPACE::resampler */

#endif //RESAMPLER_MULTICHANNEL_RESAMPLER_H
