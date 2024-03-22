/*
 * Copyright 2023 The Android Open Source Project
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

#ifndef OBOE_FULL_DUPLEX_STREAM_
#define OBOE_FULL_DUPLEX_STREAM_

#include <cstdint>
#include "oboe/Definitions.h"
#include "oboe/AudioStream.h"
#include "oboe/AudioStreamCallback.h"

namespace oboe {

/**
 * FullDuplexStream can be used to synchronize an input and output stream.
 *
 * For the builder of the output stream, call setDataCallback() with this object.
 *
 * When both streams are ready, onAudioReady() of the output stream will call onBothStreamsReady().
 * Callers must override onBothStreamsReady().
 *
 * To ensure best results, open an output stream before the input stream.
 * Call inputBuilder.setBufferCapacityInFrames(mOutputStream->getBufferCapacityInFrames() * 2).
 * Also, call inputBuilder.setSampleRate(mOutputStream->getSampleRate()).
 *
 * Callers must call setInputStream() and setOutputStream().
 * Call start() to start both streams and stop() to stop both streams.
 * Caller is responsible for closing both streams.
 *
 * Callers should handle error callbacks with setErrorCallback() for the output stream.
 * When an error callback occurs for the output stream, Oboe will stop and close the output stream.
 * The caller is responsible for stopping and closing the input stream.
 * The caller should also reopen and restart both streams when the error callback is ErrorDisconnected.
 * See the LiveEffect sample as an example of this. 
 *
 */
class FullDuplexStream : public AudioStreamDataCallback {
public:
    FullDuplexStream() {}
    virtual ~FullDuplexStream() = default;

    /**
     * Sets the input stream. Calling this is mandatory.
     *
     * @param stream the output stream
     */
    void setInputStream(AudioStream *stream) {
        mInputStream = stream;
    }

    /**
     * Gets the input stream
     *
     * @return the input stream
     */
    AudioStream *getInputStream() {
        return mInputStream;
    }

    /**
     * Sets the output stream. Calling this is mandatory.
     *
     * @param stream the output stream
     */
    void setOutputStream(AudioStream *stream) {
        mOutputStream = stream;
    }

    /**
     * Gets the output stream
     *
     * @return the output stream
     */
    AudioStream *getOutputStream() {
        return mOutputStream;
    }

    /**
     * Attempts to start both streams. Please call setInputStream() and setOutputStream() before
     * calling this function.
     *
     * @return result of the operation
     */
    virtual Result start() {
        mCountCallbacksToDrain = kNumCallbacksToDrain;
        mCountInputBurstsCushion = mNumInputBurstsCushion;
        mCountCallbacksToDiscard = kNumCallbacksToDiscard;

        // Determine maximum size that could possibly be called.
        int32_t bufferSize = getOutputStream()->getBufferCapacityInFrames()
                             * getOutputStream()->getChannelCount();
        if (bufferSize > mBufferSize) {
            mInputBuffer = std::make_unique<float[]>(bufferSize);
            mBufferSize = bufferSize;
        }

        oboe::Result result = getInputStream()->requestStart();
        if (result != oboe::Result::OK) {
            return result;
        }
        return getOutputStream()->requestStart();
    }

    /**
     * Stops both streams. Returns Result::OK if neither stream had an error during close.
     *
     * @return result of the operation
     */
    virtual Result stop() {
        Result outputResult = Result::OK;
        Result inputResult = Result::OK;
        if (getOutputStream()) {
            outputResult = mOutputStream->requestStop();
        }
        if (getInputStream()) {
            inputResult = mInputStream->requestStop();
        }
        if (outputResult != Result::OK) {
            return outputResult;
        } else {
            return inputResult;
        }
    }

    /**
     * Reads input from the input stream. Callers should not call this directly as this is called
     * in onAudioReady().
     *
     * @param numFrames
     * @return result of the operation
     */
    virtual ResultWithValue<int32_t> readInput(int32_t numFrames) {
        return getInputStream()->read(mInputBuffer.get(), numFrames, 0 /* timeout */);
    }

    /**
     * Called when data is available on both streams.
     * Caller should override this method.
     * numInputFrames and numOutputFrames may be zero.
     *
     * @param inputData buffer containing input data
     * @param numInputFrames number of input frames
     * @param outputData a place to put output data
     * @param numOutputFrames number of output frames
     * @return DataCallbackResult::Continue or DataCallbackResult::Stop
     */
    virtual DataCallbackResult onBothStreamsReady(
            const void *inputData,
            int   numInputFrames,
            void *outputData,
            int   numOutputFrames
            ) = 0;

    /**
     * Called when the output stream is ready to process audio.
     * This in return calls onBothStreamsReady() when data is available on both streams.
     * Callers should call this function when the output stream is ready.
     * Callers must override onBothStreamsReady().
     *
     * @param audioStream pointer to the associated stream
     * @param audioData a place to put output data
     * @param numFrames number of frames to be processed
     * @return DataCallbackResult::Continue or DataCallbackResult::Stop
     *
     */
    DataCallbackResult onAudioReady(
            AudioStream *audioStream,
            void *audioData,
            int numFrames) {
        DataCallbackResult callbackResult = DataCallbackResult::Continue;
        int32_t actualFramesRead = 0;

        // Silence the output.
        int32_t numBytes = numFrames * getOutputStream()->getBytesPerFrame();
        memset(audioData, 0 /* value */, numBytes);

        if (mCountCallbacksToDrain > 0) {
            // Drain the input.
            int32_t totalFramesRead = 0;
            do {
                ResultWithValue<int32_t> result = readInput(numFrames);
                if (!result) {
                    // Ignore errors because input stream may not be started yet.
                    break;
                }
                actualFramesRead = result.value();
                totalFramesRead += actualFramesRead;
            } while (actualFramesRead > 0);
            // Only counts if we actually got some data.
            if (totalFramesRead > 0) {
                mCountCallbacksToDrain--;
            }

        } else if (mCountInputBurstsCushion > 0) {
            // Let the input fill up a bit so we are not so close to the write pointer.
            mCountInputBurstsCushion--;

        } else if (mCountCallbacksToDiscard > 0) {
            mCountCallbacksToDiscard--;
            // Ignore. Allow the input to reach to equilibrium with the output.
            ResultWithValue<int32_t> resultAvailable = getInputStream()->getAvailableFrames();
            if (!resultAvailable) {
                callbackResult = DataCallbackResult::Stop;
            } else {
                int32_t framesAvailable = resultAvailable.value();
                if (framesAvailable >= mMinimumFramesBeforeRead) {
                    ResultWithValue<int32_t> resultRead = readInput(numFrames);
                    if (!resultRead) {
                        callbackResult = DataCallbackResult::Stop;
                    }
                }
            }
        } else {
            int32_t framesRead = 0;
            ResultWithValue<int32_t> resultAvailable = getInputStream()->getAvailableFrames();
            if (!resultAvailable) {
                callbackResult = DataCallbackResult::Stop;
            } else {
                int32_t framesAvailable = resultAvailable.value();
                if (framesAvailable >= mMinimumFramesBeforeRead) {
                    // Read data into input buffer.
                    ResultWithValue<int32_t> resultRead = readInput(numFrames);
                    if (!resultRead) {
                        callbackResult = DataCallbackResult::Stop;
                    } else {
                        framesRead = resultRead.value();
                    }
                }
            }

            if (callbackResult == DataCallbackResult::Continue) {
                callbackResult = onBothStreamsReady(mInputBuffer.get(), framesRead,
                                                    audioData, numFrames);
            }
        }

        if (callbackResult == DataCallbackResult::Stop) {
            getInputStream()->requestStop();
        }

        return callbackResult;
    }

    /**
     *
     * This is a cushion between the DSP and the application processor cursors to prevent collisions.
     * Typically 0 for latency measurements or 1 for glitch tests.
     *
     * @param numBursts number of bursts to leave in the input buffer as a cushion
     */
    void setNumInputBurstsCushion(int32_t numBursts) {
        mNumInputBurstsCushion = numBursts;
    }

    /**
     * Get the number of bursts left in the input buffer as a cushion.
     *
     * @return number of bursts in the input buffer as a cushion
     */
    int32_t getNumInputBurstsCushion() const {
        return mNumInputBurstsCushion;
    }

    /**
     * Minimum number of frames in the input stream buffer before calling readInput().
     *
     * @param numFrames number of bursts in the input buffer as a cushion
     */
    void setMinimumFramesBeforeRead(int32_t numFrames) {
        mMinimumFramesBeforeRead = numFrames;
    }

    /**
     * Gets the minimum number of frames in the input stream buffer before calling readInput().
     *
     * @return minimum number of frames before reading
     */
    int32_t getMinimumFramesBeforeRead() const {
        return mMinimumFramesBeforeRead;
    }

private:

    // TODO add getters and setters
    static constexpr int32_t kNumCallbacksToDrain   = 20;
    static constexpr int32_t kNumCallbacksToDiscard = 30;

    // let input fill back up, usually 0 or 1
    int32_t mNumInputBurstsCushion =  0;
    int32_t mMinimumFramesBeforeRead = 0;

    // We want to reach a state where the input buffer is empty and
    // the output buffer is full.
    // These are used in order.
    // Drain several callback so that input is empty.
    int32_t              mCountCallbacksToDrain = kNumCallbacksToDrain;
    // Let the input fill back up slightly so we don't run dry.
    int32_t              mCountInputBurstsCushion = mNumInputBurstsCushion;
    // Discard some callbacks so the input and output reach equilibrium.
    int32_t              mCountCallbacksToDiscard = kNumCallbacksToDiscard;

    AudioStream   *mInputStream = nullptr;
    AudioStream   *mOutputStream = nullptr;

    int32_t              mBufferSize = 0;
    std::unique_ptr<float[]> mInputBuffer;
};

} // namespace oboe

#endif //OBOE_FULL_DUPLEX_STREAM_
