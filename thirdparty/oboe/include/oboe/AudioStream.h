/*
 * Copyright 2016 The Android Open Source Project
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

#ifndef OBOE_STREAM_H_
#define OBOE_STREAM_H_

#include <atomic>
#include <cstdint>
#include <ctime>
#include <mutex>
#include "oboe/Definitions.h"
#include "oboe/ResultWithValue.h"
#include "oboe/AudioStreamBuilder.h"
#include "oboe/AudioStreamBase.h"

/** WARNING - UNDER CONSTRUCTION - THIS API WILL CHANGE. */

namespace oboe {

/**
 * The default number of nanoseconds to wait for when performing state change operations on the
 * stream, such as `start` and `stop`.
 *
 * @see oboe::AudioStream::start
 */
constexpr int64_t kDefaultTimeoutNanos = (2000 * kNanosPerMillisecond);

/**
 * Base class for Oboe C++ audio stream.
 */
class AudioStream : public AudioStreamBase {
    friend class AudioStreamBuilder; // allow access to setWeakThis() and lockWeakThis()
public:

    AudioStream() {}

    /**
     * Construct an `AudioStream` using the given `AudioStreamBuilder`
     *
     * @param builder containing all the stream's attributes
     */
    explicit AudioStream(const AudioStreamBuilder &builder);

    virtual ~AudioStream() = default;

    /**
     * Open a stream based on the current settings.
     *
     * Note that we do not recommend re-opening a stream that has been closed.
     * TODO Should we prevent re-opening?
     *
     * @return
     */
    virtual Result open() {
        return Result::OK; // Called by subclasses. Might do more in the future.
    }

    /**
     * Free the audio resources associated with a stream created by AAudioStreamBuilder_openStream().
     *
     * AAudioStream_close() should be called at some point after calling this function.
     *
     * After this call, the stream will be in AAUDIO_STREAM_STATE_CLOSING
     *
     * This function is useful if you want to release the audio resources immediately, but still allow
     * queries to the stream to occur from other threads. This often happens if you are monitoring
     * stream progress from a UI thread.
     *
     * NOTE: This function is only fully implemented for MMAP streams, which are low latency streams
     * supported by some devices. On other "Legacy" streams some audio resources will still be in use
     * and some callbacks may still be in process after this call.
     *
     * Available in AAudio since API level 30. Returns Result::ErrorUnimplemented otherwise.
     *
     * * @return either Result::OK or an error.
     */
    virtual Result release() {
        return Result::ErrorUnimplemented;
    }

    /**
     * Close the stream and deallocate any resources from the open() call.
     */
    virtual Result close();

    /**
     * Start the stream. This will block until the stream has been started, an error occurs
     * or `timeoutNanoseconds` has been reached.
     */
    virtual Result start(int64_t timeoutNanoseconds = kDefaultTimeoutNanos);

    /**
     * Pause the stream. This will block until the stream has been paused, an error occurs
     * or `timeoutNanoseconds` has been reached.
     */
    virtual Result pause(int64_t timeoutNanoseconds = kDefaultTimeoutNanos);

    /**
     * Flush the stream. This will block until the stream has been flushed, an error occurs
     * or `timeoutNanoseconds` has been reached.
     */
    virtual Result flush(int64_t timeoutNanoseconds = kDefaultTimeoutNanos);

    /**
     * Stop the stream. This will block until the stream has been stopped, an error occurs
     * or `timeoutNanoseconds` has been reached.
     */
    virtual Result stop(int64_t timeoutNanoseconds = kDefaultTimeoutNanos);

    /* Asynchronous requests.
     * Use waitForStateChange() if you need to wait for completion.
     */

    /**
     * Start the stream asynchronously. Returns immediately (does not block). Equivalent to calling
     * `start(0)`.
     */
    virtual Result requestStart() = 0;

    /**
     * Pause the stream asynchronously. Returns immediately (does not block). Equivalent to calling
     * `pause(0)`.
     */
    virtual Result requestPause() = 0;

    /**
     * Flush the stream asynchronously. Returns immediately (does not block). Equivalent to calling
     * `flush(0)`.
     */
    virtual Result requestFlush() = 0;

    /**
     * Stop the stream asynchronously. Returns immediately (does not block). Equivalent to calling
     * `stop(0)`.
     */
    virtual Result requestStop() = 0;

    /**
     * Query the current state, eg. StreamState::Pausing
     *
     * @return state or a negative error.
     */
    virtual StreamState getState() = 0;

    /**
     * Wait until the stream's current state no longer matches the input state.
     * The input state is passed to avoid race conditions caused by the state
     * changing between calls.
     *
     * Note that generally applications do not need to call this. It is considered
     * an advanced technique and is mostly used for testing.
     *
     * <pre><code>
     * int64_t timeoutNanos = 500 * kNanosPerMillisecond; // arbitrary 1/2 second
     * StreamState currentState = stream->getState();
     * StreamState nextState = StreamState::Unknown;
     * while (result == Result::OK && currentState != StreamState::Paused) {
     *     result = stream->waitForStateChange(
     *                                   currentState, &nextState, timeoutNanos);
     *     currentState = nextState;
     * }
     * </code></pre>
     *
     * If the state does not change within the timeout period then it will
     * return ErrorTimeout. This is true even if timeoutNanoseconds is zero.
     *
     * @param inputState The state we want to change away from.
     * @param nextState Pointer to a variable that will be set to the new state.
     * @param timeoutNanoseconds The maximum time to wait in nanoseconds.
     * @return Result::OK or a Result::Error.
     */
    virtual Result waitForStateChange(StreamState inputState,
                                          StreamState *nextState,
                                          int64_t timeoutNanoseconds) = 0;

    /**
    * This can be used to adjust the latency of the buffer by changing
    * the threshold where blocking will occur.
    * By combining this with getXRunCount(), the latency can be tuned
    * at run-time for each device.
    *
    * This cannot be set higher than getBufferCapacity().
    *
    * This should only be used with Output streams. It will
    * be ignored for Input streams because they are generally kept as empty as possible.
    *
    * For OpenSL ES, this method only has an effect on output stream that do NOT
    * use a callback. The blocking writes goes into a buffer in Oboe and the size of that
    * buffer is controlled by this method.
    *
    * @param requestedFrames requested number of frames that can be filled without blocking
    * @return the resulting buffer size in frames (obtained using value()) or an error (obtained
    * using error())
    */
    virtual ResultWithValue<int32_t> setBufferSizeInFrames(int32_t /* requestedFrames  */) {
        return Result::ErrorUnimplemented;
    }

    /**
     * An XRun is an Underrun or an Overrun.
     * During playing, an underrun will occur if the stream is not written in time
     * and the system runs out of valid data.
     * During recording, an overrun will occur if the stream is not read in time
     * and there is no place to put the incoming data so it is discarded.
     *
     * An underrun or overrun can cause an audible "pop" or "glitch".
     *
     * @return a result which is either Result::OK with the xRun count as the value, or a
     * Result::Error* code
     */
    virtual ResultWithValue<int32_t> getXRunCount() {
        return ResultWithValue<int32_t>(Result::ErrorUnimplemented);
    }

    /**
     * @return true if XRun counts are supported on the stream
     */
    virtual bool isXRunCountSupported() const = 0;

    /**
     * Query the number of frames that are read or written by the endpoint at one time.
     *
     * @return burst size
     */
    int32_t getFramesPerBurst() const {
        return mFramesPerBurst;
    }

    /**
     * Get the number of bytes in each audio frame. This is calculated using the channel count
     * and the sample format. For example, a 2 channel floating point stream will have
     * 2 * 4 = 8 bytes per frame.
     *
     * @return number of bytes in each audio frame.
     */
    int32_t getBytesPerFrame() const { return mChannelCount * getBytesPerSample(); }

    /**
     * Get the number of bytes per sample. This is calculated using the sample format. For example,
     * a stream using 16-bit integer samples will have 2 bytes per sample.
     *
     * @return the number of bytes per sample.
     */
    int32_t getBytesPerSample() const;

    /**
     * The number of audio frames written into the stream.
     * This monotonic counter will never get reset.
     *
     * @return the number of frames written so far
     */
    virtual int64_t getFramesWritten();

    /**
     * The number of audio frames read from the stream.
     * This monotonic counter will never get reset.
     *
     * @return the number of frames read so far
     */
    virtual int64_t getFramesRead();

    /**
     * Calculate the latency of a stream based on getTimestamp().
     *
     * Output latency is the time it takes for a given frame to travel from the
     * app to some type of digital-to-analog converter. If the DAC is external, for example
     * in a USB interface or a TV connected by HDMI, then there may be additional latency
     * that the Android device is unaware of.
     *
     * Input latency is the time it takes to a given frame to travel from an analog-to-digital
     * converter (ADC) to the app.
     *
     * Note that the latency of an OUTPUT stream will increase abruptly when you write data to it
     * and then decrease slowly over time as the data is consumed.
     *
     * The latency of an INPUT stream will decrease abruptly when you read data from it
     * and then increase slowly over time as more data arrives.
     *
     * The latency of an OUTPUT stream is generally higher than the INPUT latency
     * because an app generally tries to keep the OUTPUT buffer full and the INPUT buffer empty.
     *
     * Note that due to issues in Android before R, we recommend NOT calling
     * this method from a data callback. See this tech note for more details.
     * https://github.com/google/oboe/wiki/TechNote_ReleaseBuffer
     *
     * @return a ResultWithValue which has a result of Result::OK and a value containing the latency
     * in milliseconds, or a result of Result::Error*.
     */
    virtual ResultWithValue<double> calculateLatencyMillis() {
        return ResultWithValue<double>(Result::ErrorUnimplemented);
    }

    /**
     * Get the estimated time that the frame at `framePosition` entered or left the audio processing
     * pipeline.
     *
     * This can be used to coordinate events and interactions with the external environment, and to
     * estimate the latency of an audio stream. An example of usage can be found in the hello-oboe
     * sample (search for "calculateCurrentOutputLatencyMillis").
     *
     * The time is based on the implementation's best effort, using whatever knowledge is available
     * to the system, but cannot account for any delay unknown to the implementation.
     *
     * Note that due to issues in Android before R, we recommend NOT calling
     * this method from a data callback. See this tech note for more details.
     * https://github.com/google/oboe/wiki/TechNote_ReleaseBuffer
     *
     * @deprecated since 1.0, use AudioStream::getTimestamp(clockid_t clockId) instead, which
     * returns ResultWithValue
     * @param clockId the type of clock to use e.g. CLOCK_MONOTONIC
     * @param framePosition the frame number to query
     * @param timeNanoseconds an output parameter which will contain the presentation timestamp
     */
    virtual Result getTimestamp(clockid_t /* clockId  */,
                                int64_t* /* framePosition */,
                                int64_t* /* timeNanoseconds */) {
        return Result::ErrorUnimplemented;
    }

    /**
     * Get the estimated time that the frame at `framePosition` entered or left the audio processing
     * pipeline.
     *
     * This can be used to coordinate events and interactions with the external environment, and to
     * estimate the latency of an audio stream. An example of usage can be found in the hello-oboe
     * sample (search for "calculateCurrentOutputLatencyMillis").
     *
     * The time is based on the implementation's best effort, using whatever knowledge is available
     * to the system, but cannot account for any delay unknown to the implementation.
     *
     * Note that due to issues in Android before R, we recommend NOT calling
     * this method from a data callback. See this tech note for more details.
     * https://github.com/google/oboe/wiki/TechNote_ReleaseBuffer
     *
     * See 
     * @param clockId the type of clock to use e.g. CLOCK_MONOTONIC
     * @return a FrameTimestamp containing the position and time at which a particular audio frame
     * entered or left the audio processing pipeline, or an error if the operation failed.
     */
    virtual ResultWithValue<FrameTimestamp> getTimestamp(clockid_t /* clockId */);

    // ============== I/O ===========================
    /**
     * Write data from the supplied buffer into the stream. This method will block until the write
     * is complete or it runs out of time.
     *
     * If `timeoutNanoseconds` is zero then this call will not wait.
     *
     * @param buffer The address of the first sample.
     * @param numFrames Number of frames to write. Only complete frames will be written.
     * @param timeoutNanoseconds Maximum number of nanoseconds to wait for completion.
     * @return a ResultWithValue which has a result of Result::OK and a value containing the number
     * of frames actually written, or result of Result::Error*.
     */
    virtual ResultWithValue<int32_t> write(const void* /* buffer */,
                             int32_t /* numFrames */,
                             int64_t /* timeoutNanoseconds */ ) {
        return ResultWithValue<int32_t>(Result::ErrorUnimplemented);
    }

    /**
     * Read data into the supplied buffer from the stream. This method will block until the read
     * is complete or it runs out of time.
     *
     * If `timeoutNanoseconds` is zero then this call will not wait.
     *
     * @param buffer The address of the first sample.
     * @param numFrames Number of frames to read. Only complete frames will be read.
     * @param timeoutNanoseconds Maximum number of nanoseconds to wait for completion.
     * @return a ResultWithValue which has a result of Result::OK and a value containing the number
     * of frames actually read, or result of Result::Error*.
     */
    virtual ResultWithValue<int32_t> read(void* /* buffer */,
                            int32_t /* numFrames */,
                            int64_t /* timeoutNanoseconds */) {
        return ResultWithValue<int32_t>(Result::ErrorUnimplemented);
    }

    /**
     * Get the underlying audio API which the stream uses.
     *
     * @return the API that this stream uses.
     */
    virtual AudioApi getAudioApi() const = 0;

    /**
     * Returns true if the underlying audio API is AAudio.
     *
     * @return true if this stream is implemented using the AAudio API.
     */
    bool usesAAudio() const {
        return getAudioApi() == AudioApi::AAudio;
    }

    /**
     * Only for debugging. Do not use in production.
     * If you need to call this method something is wrong.
     * If you think you need it for production then please let us know
     * so we can modify Oboe so that you don't need this.
     *
     * @return nullptr or a pointer to a stream from the system API
     */
    virtual void *getUnderlyingStream() const {
        return nullptr;
    }

    /**
     * Update mFramesWritten.
     * For internal use only.
     */
    virtual void updateFramesWritten() = 0;

    /**
     * Update mFramesRead.
     * For internal use only.
     */
    virtual void updateFramesRead() = 0;

    /*
     * Swap old callback for new callback.
     * This not atomic.
     * This should only be used internally.
     * @param dataCallback
     * @return previous dataCallback
     */
    AudioStreamDataCallback *swapDataCallback(AudioStreamDataCallback *dataCallback) {
        AudioStreamDataCallback *previousCallback = mDataCallback;
        mDataCallback = dataCallback;
        return previousCallback;
    }

    /*
     * Swap old callback for new callback.
     * This not atomic.
     * This should only be used internally.
     * @param errorCallback
     * @return previous errorCallback
     */
    AudioStreamErrorCallback *swapErrorCallback(AudioStreamErrorCallback *errorCallback) {
        AudioStreamErrorCallback *previousCallback = mErrorCallback;
        mErrorCallback = errorCallback;
        return previousCallback;
    }

    /**
     * @return number of frames of data currently in the buffer
     */
    ResultWithValue<int32_t> getAvailableFrames();

    /**
     * Wait until the stream has a minimum amount of data available in its buffer.
     * This can be used with an EXCLUSIVE MMAP input stream to avoid reading data too close to
     * the DSP write position, which may cause glitches.
     *
     * Starting with Oboe 1.7.1, the numFrames will be clipped internally against the
     * BufferCapacity minus BurstSize. This is to prevent trying to wait for more frames
     * than could possibly be available. In this case, the return value may be less than numFrames.
     * Note that there may still be glitching if numFrames is too high.
     *
     * @param numFrames requested minimum frames available
     * @param timeoutNanoseconds
     * @return number of frames available, ErrorTimeout
     */
    ResultWithValue<int32_t> waitForAvailableFrames(int32_t numFrames,
                                                    int64_t timeoutNanoseconds);

    /**
     * @return last result passed from an error callback
     */
    virtual oboe::Result getLastErrorCallbackResult() const {
        return mErrorCallbackResult;
    }


    int32_t getDelayBeforeCloseMillis() const {
        return mDelayBeforeCloseMillis;
    }

    /**
     * Set the time to sleep before closing the internal stream.
     *
     * Sometimes a callback can occur shortly after a stream has been stopped and
     * even after a close! If the stream has been closed then the callback
     * might access memory that has been freed, which could cause a crash.
     * This seems to be more likely in Android P or earlier.
     * But it can also occur in later versions. By sleeping, we give time for
     * the callback threads to finish.
     *
     * Note that this only has an effect when OboeGlobals::areWorkaroundsEnabled() is true.
     *
     * @param delayBeforeCloseMillis time to sleep before close.
     */
    void setDelayBeforeCloseMillis(int32_t delayBeforeCloseMillis) {
        mDelayBeforeCloseMillis = delayBeforeCloseMillis;
    }

    /**
     * Enable or disable a device specific CPU performance hint.
     * Runtime benchmarks such as the callback duration may be used to
     * speed up the CPU and improve real-time performance.
     *
     * Note that this feature is device specific and may not be implemented.
     * Also the benefits may vary by device.
     *
     * The flag will be checked in the Oboe data callback. If it transitions from false to true
     * then the PerformanceHint feature will be started.
     * This only needs to be called once.
     *
     * You may want to enable this if you have a dynamically changing workload
     * and you notice that you are getting underruns and glitches when your workload increases.
     * This might happen, for example, if you suddenly go from playing one note to
     * ten notes on a synthesizer.
     *
     * Try the CPU Load test in OboeTester if you would like to experiment with this interactively.
     *
     * On some devices, this may be implemented using the "ADPF" library.
     *
     * @param enabled true if you would like a performance boost
     */
    void setPerformanceHintEnabled(bool enabled) {
        mPerformanceHintEnabled = enabled;
    }

    /**
     * This only tells you if the feature has been requested.
     * It does not tell you if the PerformanceHint feature is implemented or active on the device.
     *
     * @return true if set using setPerformanceHintEnabled().
     */
    bool isPerformanceHintEnabled() {
        return mPerformanceHintEnabled;
    }

protected:

    /**
     * This is used to detect more than one error callback from a stream.
     * These were bugs in some versions of Android that caused multiple error callbacks.
     * Internal bug b/63087953
     *
     * Calling this sets an atomic<bool> true and returns the previous value.
     *
     * @return false on first call, true on subsequent calls
     */
    bool wasErrorCallbackCalled() {
        return mErrorCallbackCalled.exchange(true);
    }

    /**
     * Wait for a transition from one state to another.
     * @return OK if the endingState was observed, or ErrorUnexpectedState
     *   if any state that was not the startingState or endingState was observed
     *   or ErrorTimeout.
     */
    virtual Result waitForStateTransition(StreamState startingState,
                                          StreamState endingState,
                                          int64_t timeoutNanoseconds);

    /**
     * Override this to provide a default for when the application did not specify a callback.
     *
     * @param audioData
     * @param numFrames
     * @return result
     */
    virtual DataCallbackResult onDefaultCallback(void* /* audioData  */, int /* numFrames */) {
        return DataCallbackResult::Stop;
    }

    /**
     * Override this to provide your own behaviour for the audio callback
     *
     * @param audioData container array which audio frames will be written into or read from
     * @param numFrames number of frames which were read/written
     * @return the result of the callback: stop or continue
     *
     */
    DataCallbackResult fireDataCallback(void *audioData, int numFrames);

    /**
     * @return true if callbacks may be called
     */
    bool isDataCallbackEnabled() {
        return mDataCallbackEnabled;
    }

    /**
     * This can be set false internally to prevent callbacks
     * after DataCallbackResult::Stop has been returned.
     */
    void setDataCallbackEnabled(bool enabled) {
        mDataCallbackEnabled = enabled;
    }

    /**
     * This should only be called as a stream is being opened.
     * Otherwise we might override setDelayBeforeCloseMillis().
     */
    void calculateDefaultDelayBeforeCloseMillis();

    /**
     * Try to avoid a race condition when closing.
     */
    void sleepBeforeClose() {
        if (mDelayBeforeCloseMillis > 0) {
            usleep(mDelayBeforeCloseMillis * 1000);
        }
    }

    /**
     * This may be called internally at the beginning of a callback.
     */
    virtual void beginPerformanceHintInCallback() {}

    /**
     * This may be called internally at the end of a callback.
     * @param numFrames passed to the callback
     */
    virtual void endPerformanceHintInCallback(int32_t numFrames) {}

    /**
     * This will be called when the stream is closed just in case performance hints were enabled.
     */
    virtual void closePerformanceHint() {}

    /*
     * Set a weak_ptr to this stream from the shared_ptr so that we can
     * later use a shared_ptr in the error callback.
     */
    void setWeakThis(std::shared_ptr<oboe::AudioStream> &sharedStream) {
        mWeakThis = sharedStream;
    }

    /*
     * Make a shared_ptr that will prevent this stream from being deleted.
     */
    std::shared_ptr<oboe::AudioStream> lockWeakThis() {
        return mWeakThis.lock();
    }

    std::weak_ptr<AudioStream> mWeakThis; // weak pointer to this object

    /**
     * Number of frames which have been written into the stream
     *
     * This is signed integer to match the counters in AAudio.
     * At audio rates, the counter will overflow in about six million years.
     */
    std::atomic<int64_t> mFramesWritten{};

    /**
     * Number of frames which have been read from the stream.
     *
     * This is signed integer to match the counters in AAudio.
     * At audio rates, the counter will overflow in about six million years.
     */
    std::atomic<int64_t> mFramesRead{};

    std::mutex           mLock; // for synchronizing start/stop/close

    oboe::Result         mErrorCallbackResult = oboe::Result::OK;

    /**
     * Number of frames which will be copied to/from the audio device in a single read/write
     * operation
     */
    int32_t              mFramesPerBurst = kUnspecified;

    // Time to sleep in order to prevent a race condition with a callback after a close().
    // Two milliseconds may be enough but 10 msec is even safer.
    static constexpr int kMinDelayBeforeCloseMillis = 10;
    int32_t              mDelayBeforeCloseMillis = kMinDelayBeforeCloseMillis;

private:

    // Log the scheduler if it changes.
    void                 checkScheduler();
    int                  mPreviousScheduler = -1;

    std::atomic<bool>    mDataCallbackEnabled{false};
    std::atomic<bool>    mErrorCallbackCalled{false};

    std::atomic<bool>    mPerformanceHintEnabled{false}; // set only by app
};

/**
 * This struct is a stateless functor which closes an AudioStream prior to its deletion.
 * This means it can be used to safely delete a smart pointer referring to an open stream.
 */
    struct StreamDeleterFunctor {
        void operator()(AudioStream  *audioStream) {
            if (audioStream) {
                audioStream->close();
            }
            delete audioStream;
        }
    };
} // namespace oboe

#endif /* OBOE_STREAM_H_ */
