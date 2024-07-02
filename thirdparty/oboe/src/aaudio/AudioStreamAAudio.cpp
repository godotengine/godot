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

#include <cassert>
#include <stdint.h>
#include <stdlib.h>

#include "aaudio/AAudioLoader.h"
#include "aaudio/AudioStreamAAudio.h"
#include "common/AudioClock.h"
#include "common/OboeDebug.h"
#include "oboe/Utilities.h"
#include "AAudioExtensions.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#include <common/QuirksManager.h>

#endif

#ifndef OBOE_FIX_FORCE_STARTING_TO_STARTED
// Workaround state problems in AAudio
// TODO Which versions does this occur in? Verify fixed in Q.
#define OBOE_FIX_FORCE_STARTING_TO_STARTED 1
#endif // OBOE_FIX_FORCE_STARTING_TO_STARTED

using namespace oboe;
AAudioLoader *AudioStreamAAudio::mLibLoader = nullptr;

// 'C' wrapper for the data callback method
static aaudio_data_callback_result_t oboe_aaudio_data_callback_proc(
        AAudioStream *stream,
        void *userData,
        void *audioData,
        int32_t numFrames) {

    AudioStreamAAudio *oboeStream = reinterpret_cast<AudioStreamAAudio*>(userData);
    if (oboeStream != nullptr) {
        return static_cast<aaudio_data_callback_result_t>(
                oboeStream->callOnAudioReady(stream, audioData, numFrames));

    } else {
        return static_cast<aaudio_data_callback_result_t>(DataCallbackResult::Stop);
    }
}

// This runs in its own thread.
// Only one of these threads will be launched from internalErrorCallback().
// It calls app error callbacks from a static function in case the stream gets deleted.
static void oboe_aaudio_error_thread_proc(AudioStreamAAudio *oboeStream,
                                          Result error) {
    LOGD("%s(,%d) - entering >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", __func__, error);
    AudioStreamErrorCallback *errorCallback = oboeStream->getErrorCallback();
    if (errorCallback == nullptr) return; // should be impossible
    bool isErrorHandled = errorCallback->onError(oboeStream, error);

    if (!isErrorHandled) {
        oboeStream->requestStop();
        errorCallback->onErrorBeforeClose(oboeStream, error);
        oboeStream->close();
        // Warning, oboeStream may get deleted by this callback.
        errorCallback->onErrorAfterClose(oboeStream, error);
    }
    LOGD("%s() - exiting <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", __func__);
}

// This runs in its own thread.
// Only one of these threads will be launched from internalErrorCallback().
// Prevents deletion of the stream if the app is using AudioStreamBuilder::openSharedStream()
static void oboe_aaudio_error_thread_proc_shared(std::shared_ptr<AudioStream> sharedStream,
                                          Result error) {
    AudioStreamAAudio *oboeStream = reinterpret_cast<AudioStreamAAudio*>(sharedStream.get());
    oboe_aaudio_error_thread_proc(oboeStream, error);
}

namespace oboe {

/*
 * Create a stream that uses Oboe Audio API.
 */
AudioStreamAAudio::AudioStreamAAudio(const AudioStreamBuilder &builder)
    : AudioStream(builder)
    , mAAudioStream(nullptr) {
    mCallbackThreadEnabled.store(false);
    mLibLoader = AAudioLoader::getInstance();
}

bool AudioStreamAAudio::isSupported() {
    mLibLoader = AAudioLoader::getInstance();
    int openResult = mLibLoader->open();
    return openResult == 0;
}

// Static method for the error callback.
// We use a method so we can access protected methods on the stream.
// Launch a thread to handle the error.
// That other thread can safely stop, close and delete the stream.
void AudioStreamAAudio::internalErrorCallback(
        AAudioStream *stream,
        void *userData,
        aaudio_result_t error) {
    oboe::Result oboeResult = static_cast<Result>(error);
    AudioStreamAAudio *oboeStream = reinterpret_cast<AudioStreamAAudio*>(userData);

    // Coerce the error code if needed to workaround a regression in RQ1A that caused
    // the wrong code to be passed when headsets plugged in. See b/173928197.
    if (OboeGlobals::areWorkaroundsEnabled()
            && getSdkVersion() == __ANDROID_API_R__
            && oboeResult == oboe::Result::ErrorTimeout) {
        oboeResult = oboe::Result::ErrorDisconnected;
        LOGD("%s() ErrorTimeout changed to ErrorDisconnected to fix b/173928197", __func__);
    }

    oboeStream->mErrorCallbackResult = oboeResult;

    // Prevents deletion of the stream if the app is using AudioStreamBuilder::openStream(shared_ptr)
    std::shared_ptr<AudioStream> sharedStream = oboeStream->lockWeakThis();

    // These checks should be enough because we assume that the stream close()
    // will join() any active callback threads and will not allow new callbacks.
    if (oboeStream->wasErrorCallbackCalled()) { // block extra error callbacks
        LOGE("%s() multiple error callbacks called!", __func__);
    } else if (stream != oboeStream->getUnderlyingStream()) {
        LOGW("%s() stream already closed or closing", __func__); // might happen if there are bugs
    } else if (sharedStream) {
        // Handle error on a separate thread using shared pointer.
        std::thread t(oboe_aaudio_error_thread_proc_shared, sharedStream, oboeResult);
        t.detach();
    } else {
        // Handle error on a separate thread.
        std::thread t(oboe_aaudio_error_thread_proc, oboeStream, oboeResult);
        t.detach();
    }
}

void AudioStreamAAudio::beginPerformanceHintInCallback() {
    if (isPerformanceHintEnabled()) {
        if (!mAdpfOpenAttempted) {
            int64_t targetDurationNanos = (mFramesPerBurst * 1e9) / getSampleRate();
            // This has to be called from the callback thread so we get the right TID.
            int adpfResult = mAdpfWrapper.open(gettid(), targetDurationNanos);
            if (adpfResult < 0) {
                LOGW("WARNING ADPF not supported, %d\n", adpfResult);
            } else {
                LOGD("ADPF is now active\n");
            }
            mAdpfOpenAttempted = true;
        }
        mAdpfWrapper.onBeginCallback();
    } else if (!isPerformanceHintEnabled() && mAdpfOpenAttempted) {
        LOGD("ADPF closed\n");
        mAdpfWrapper.close();
        mAdpfOpenAttempted = false;
    }
}

void AudioStreamAAudio::endPerformanceHintInCallback(int32_t numFrames) {
    if (mAdpfWrapper.isOpen()) {
        // Scale the measured duration based on numFrames so it is normalized to a full burst.
        double durationScaler = static_cast<double>(mFramesPerBurst) / numFrames;
        // Skip this callback if numFrames is very small.
        // This can happen when buffers wrap around, particularly when doing sample rate conversion.
        if (durationScaler < 2.0) {
            mAdpfWrapper.onEndCallback(durationScaler);
        }
    }
}

void AudioStreamAAudio::logUnsupportedAttributes() {
    int sdkVersion = getSdkVersion();

    // These attributes are not supported pre Android "P"
    if (sdkVersion < __ANDROID_API_P__) {
        if (mUsage != Usage::Media) {
            LOGW("Usage [AudioStreamBuilder::setUsage()] "
                 "is not supported on AAudio streams running on pre-Android P versions.");
        }

        if (mContentType != ContentType::Music) {
            LOGW("ContentType [AudioStreamBuilder::setContentType()] "
                 "is not supported on AAudio streams running on pre-Android P versions.");
        }

        if (mSessionId != SessionId::None) {
            LOGW("SessionId [AudioStreamBuilder::setSessionId()] "
                 "is not supported on AAudio streams running on pre-Android P versions.");
        }
    }
}

Result AudioStreamAAudio::open() {
    Result result = Result::OK;

    if (mAAudioStream != nullptr) {
        return Result::ErrorInvalidState;
    }

    result = AudioStream::open();
    if (result != Result::OK) {
        return result;
    }

    AAudioStreamBuilder *aaudioBuilder;
    result = static_cast<Result>(mLibLoader->createStreamBuilder(&aaudioBuilder));
    if (result != Result::OK) {
        return result;
    }

    // Do not set INPUT capacity below 4096 because that prevents us from getting a FAST track
    // when using the Legacy data path.
    // If the app requests > 4096 then we allow it but we are less likely to get LowLatency.
    // See internal bug b/80308183 for more details.
    // Fixed in Q but let's still clip the capacity because high input capacity
    // does not increase latency.
    int32_t capacity = mBufferCapacityInFrames;
    constexpr int kCapacityRequiredForFastLegacyTrack = 4096; // matches value in AudioFinger
    if (OboeGlobals::areWorkaroundsEnabled()
            && mDirection == oboe::Direction::Input
            && capacity != oboe::Unspecified
            && capacity < kCapacityRequiredForFastLegacyTrack
            && mPerformanceMode == oboe::PerformanceMode::LowLatency) {
        capacity = kCapacityRequiredForFastLegacyTrack;
        LOGD("AudioStreamAAudio.open() capacity changed from %d to %d for lower latency",
             static_cast<int>(mBufferCapacityInFrames), capacity);
    }
    mLibLoader->builder_setBufferCapacityInFrames(aaudioBuilder, capacity);

    if (mLibLoader->builder_setSessionId != nullptr) {
        mLibLoader->builder_setSessionId(aaudioBuilder,
                                         static_cast<aaudio_session_id_t>(mSessionId));
        // Output effects do not support PerformanceMode::LowLatency.
        if (OboeGlobals::areWorkaroundsEnabled()
                && mSessionId != SessionId::None
                && mDirection == oboe::Direction::Output
                && mPerformanceMode == PerformanceMode::LowLatency) {
                    mPerformanceMode = PerformanceMode::None;
                    LOGD("AudioStreamAAudio.open() performance mode changed to None when session "
                         "id is requested");
        }
    }

    // Channel mask was added in SC_V2. Given the corresponding channel count of selected channel
    // mask may be different from selected channel count, the last set value will be respected.
    // If channel count is set after channel mask, the previously set channel mask will be cleared.
    // If channel mask is set after channel count, the channel count will be automatically
    // calculated from selected channel mask. In that case, only set channel mask when the API
    // is available and the channel mask is specified.
    if (mLibLoader->builder_setChannelMask != nullptr && mChannelMask != ChannelMask::Unspecified) {
        mLibLoader->builder_setChannelMask(aaudioBuilder,
                                           static_cast<aaudio_channel_mask_t>(mChannelMask));
    } else {
        mLibLoader->builder_setChannelCount(aaudioBuilder, mChannelCount);
    }
    mLibLoader->builder_setDeviceId(aaudioBuilder, mDeviceId);
    mLibLoader->builder_setDirection(aaudioBuilder, static_cast<aaudio_direction_t>(mDirection));
    mLibLoader->builder_setFormat(aaudioBuilder, static_cast<aaudio_format_t>(mFormat));
    mLibLoader->builder_setSampleRate(aaudioBuilder, mSampleRate);
    mLibLoader->builder_setSharingMode(aaudioBuilder,
                                       static_cast<aaudio_sharing_mode_t>(mSharingMode));
    mLibLoader->builder_setPerformanceMode(aaudioBuilder,
                                           static_cast<aaudio_performance_mode_t>(mPerformanceMode));

    // These were added in P so we have to check for the function pointer.
    if (mLibLoader->builder_setUsage != nullptr) {
        mLibLoader->builder_setUsage(aaudioBuilder,
                                     static_cast<aaudio_usage_t>(mUsage));
    }

    if (mLibLoader->builder_setContentType != nullptr) {
        mLibLoader->builder_setContentType(aaudioBuilder,
                                           static_cast<aaudio_content_type_t>(mContentType));
    }

    if (mLibLoader->builder_setInputPreset != nullptr) {
        aaudio_input_preset_t inputPreset = mInputPreset;
        if (getSdkVersion() <= __ANDROID_API_P__ && inputPreset == InputPreset::VoicePerformance) {
            LOGD("InputPreset::VoicePerformance not supported before Q. Using VoiceRecognition.");
            inputPreset = InputPreset::VoiceRecognition; // most similar preset
        }
        mLibLoader->builder_setInputPreset(aaudioBuilder,
                                           static_cast<aaudio_input_preset_t>(inputPreset));
    }

    // These were added in S so we have to check for the function pointer.
    if (mLibLoader->builder_setPackageName != nullptr && !mPackageName.empty()) {
        mLibLoader->builder_setPackageName(aaudioBuilder,
                                           mPackageName.c_str());
    }

    if (mLibLoader->builder_setAttributionTag != nullptr && !mAttributionTag.empty()) {
        mLibLoader->builder_setAttributionTag(aaudioBuilder,
                                           mAttributionTag.c_str());
    }

    // This was added in Q so we have to check for the function pointer.
    if (mLibLoader->builder_setAllowedCapturePolicy != nullptr && mDirection == oboe::Direction::Output) {
        mLibLoader->builder_setAllowedCapturePolicy(aaudioBuilder,
                                           static_cast<aaudio_allowed_capture_policy_t>(mAllowedCapturePolicy));
    }

    if (mLibLoader->builder_setPrivacySensitive != nullptr && mDirection == oboe::Direction::Input
            && mPrivacySensitiveMode != PrivacySensitiveMode::Unspecified) {
        mLibLoader->builder_setPrivacySensitive(aaudioBuilder,
                mPrivacySensitiveMode == PrivacySensitiveMode::Enabled);
    }

    if (mLibLoader->builder_setIsContentSpatialized != nullptr) {
        mLibLoader->builder_setIsContentSpatialized(aaudioBuilder, mIsContentSpatialized);
    }

    if (mLibLoader->builder_setSpatializationBehavior != nullptr) {
        // Override Unspecified as Never to reduce latency.
        if (mSpatializationBehavior == SpatializationBehavior::Unspecified) {
            mSpatializationBehavior = SpatializationBehavior::Never;
        }
        mLibLoader->builder_setSpatializationBehavior(aaudioBuilder,
                static_cast<aaudio_spatialization_behavior_t>(mSpatializationBehavior));
    } else {
        mSpatializationBehavior = SpatializationBehavior::Never;
    }

    if (isDataCallbackSpecified()) {
        mLibLoader->builder_setDataCallback(aaudioBuilder, oboe_aaudio_data_callback_proc, this);
        mLibLoader->builder_setFramesPerDataCallback(aaudioBuilder, getFramesPerDataCallback());

        if (!isErrorCallbackSpecified()) {
            // The app did not specify a callback so we should specify
            // our own so the stream gets closed and stopped.
            mErrorCallback = &mDefaultErrorCallback;
        }
        mLibLoader->builder_setErrorCallback(aaudioBuilder, internalErrorCallback, this);
    }
    // Else if the data callback is not being used then the write method will return an error
    // and the app can stop and close the stream.

    // ============= OPEN THE STREAM ================
    {
        AAudioStream *stream = nullptr;
        result = static_cast<Result>(mLibLoader->builder_openStream(aaudioBuilder, &stream));
        mAAudioStream.store(stream);
    }
    if (result != Result::OK) {
        // Warn developer because ErrorInternal is not very informative.
        if (result == Result::ErrorInternal && mDirection == Direction::Input) {
            LOGW("AudioStreamAAudio.open() may have failed due to lack of "
                 "audio recording permission.");
        }
        goto error2;
    }

    // Query and cache the stream properties
    mDeviceId = mLibLoader->stream_getDeviceId(mAAudioStream);
    mChannelCount = mLibLoader->stream_getChannelCount(mAAudioStream);
    mSampleRate = mLibLoader->stream_getSampleRate(mAAudioStream);
    mFormat = static_cast<AudioFormat>(mLibLoader->stream_getFormat(mAAudioStream));
    mSharingMode = static_cast<SharingMode>(mLibLoader->stream_getSharingMode(mAAudioStream));
    mPerformanceMode = static_cast<PerformanceMode>(
            mLibLoader->stream_getPerformanceMode(mAAudioStream));
    mBufferCapacityInFrames = mLibLoader->stream_getBufferCapacity(mAAudioStream);
    mBufferSizeInFrames = mLibLoader->stream_getBufferSize(mAAudioStream);
    mFramesPerBurst = mLibLoader->stream_getFramesPerBurst(mAAudioStream);

    // These were added in P so we have to check for the function pointer.
    if (mLibLoader->stream_getUsage != nullptr) {
        mUsage = static_cast<Usage>(mLibLoader->stream_getUsage(mAAudioStream));
    }
    if (mLibLoader->stream_getContentType != nullptr) {
        mContentType = static_cast<ContentType>(mLibLoader->stream_getContentType(mAAudioStream));
    }
    if (mLibLoader->stream_getInputPreset != nullptr) {
        mInputPreset = static_cast<InputPreset>(mLibLoader->stream_getInputPreset(mAAudioStream));
    }
    if (mLibLoader->stream_getSessionId != nullptr) {
        mSessionId = static_cast<SessionId>(mLibLoader->stream_getSessionId(mAAudioStream));
    } else {
        mSessionId = SessionId::None;
    }

    // This was added in Q so we have to check for the function pointer.
    if (mLibLoader->stream_getAllowedCapturePolicy != nullptr && mDirection == oboe::Direction::Output) {
        mAllowedCapturePolicy = static_cast<AllowedCapturePolicy>(mLibLoader->stream_getAllowedCapturePolicy(mAAudioStream));
    } else {
        mAllowedCapturePolicy = AllowedCapturePolicy::Unspecified;
    }

    if (mLibLoader->stream_isPrivacySensitive != nullptr && mDirection == oboe::Direction::Input) {
        bool isPrivacySensitive = mLibLoader->stream_isPrivacySensitive(mAAudioStream);
        mPrivacySensitiveMode = isPrivacySensitive ? PrivacySensitiveMode::Enabled :
                PrivacySensitiveMode::Disabled;
    } else {
        mPrivacySensitiveMode = PrivacySensitiveMode::Unspecified;
    }

    if (mLibLoader->stream_getChannelMask != nullptr) {
        mChannelMask = static_cast<ChannelMask>(mLibLoader->stream_getChannelMask(mAAudioStream));
    }

    if (mLibLoader->stream_isContentSpatialized != nullptr) {
        mIsContentSpatialized = mLibLoader->stream_isContentSpatialized(mAAudioStream);
    }

    if (mLibLoader->stream_getSpatializationBehavior != nullptr) {
        mSpatializationBehavior = static_cast<SpatializationBehavior>(
                mLibLoader->stream_getSpatializationBehavior(mAAudioStream));
    }

    if (mLibLoader->stream_getHardwareChannelCount != nullptr) {
        mHardwareChannelCount = mLibLoader->stream_getHardwareChannelCount(mAAudioStream);
    }
    if (mLibLoader->stream_getHardwareSampleRate != nullptr) {
        mHardwareSampleRate = mLibLoader->stream_getHardwareSampleRate(mAAudioStream);
    }
    if (mLibLoader->stream_getHardwareFormat != nullptr) {
        mHardwareFormat = static_cast<AudioFormat>(mLibLoader->stream_getHardwareFormat(mAAudioStream));
    }

    LOGD("AudioStreamAAudio.open() format=%d, sampleRate=%d, capacity = %d",
            static_cast<int>(mFormat), static_cast<int>(mSampleRate),
            static_cast<int>(mBufferCapacityInFrames));

    calculateDefaultDelayBeforeCloseMillis();

error2:
    mLibLoader->builder_delete(aaudioBuilder);
    if (static_cast<int>(result) > 0) {
        // Possibly due to b/267531411
        LOGW("AudioStreamAAudio.open: AAudioStream_Open() returned positive error = %d",
             static_cast<int>(result));
        if (OboeGlobals::areWorkaroundsEnabled()) {
            result = Result::ErrorInternal; // Coerce to negative error.
        }
    } else {
        LOGD("AudioStreamAAudio.open: AAudioStream_Open() returned %s = %d",
             mLibLoader->convertResultToText(static_cast<aaudio_result_t>(result)),
             static_cast<int>(result));
    }
    return result;
}

Result AudioStreamAAudio::release() {
    if (getSdkVersion() < __ANDROID_API_R__) {
        return Result::ErrorUnimplemented;
    }

    // AAudioStream_release() is buggy on Android R.
    if (OboeGlobals::areWorkaroundsEnabled() && getSdkVersion() == __ANDROID_API_R__) {
        LOGW("Skipping release() on Android R");
        return Result::ErrorUnimplemented;
    }

    std::lock_guard<std::mutex> lock(mLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        if (OboeGlobals::areWorkaroundsEnabled()) {
            // Make sure we are really stopped. Do it under mLock
            // so another thread cannot call requestStart() right before the close.
            requestStop_l(stream);
        }
        return static_cast<Result>(mLibLoader->stream_release(stream));
    } else {
        return Result::ErrorClosed;
    }
}

Result AudioStreamAAudio::close() {
    // Prevent two threads from closing the stream at the same time and crashing.
    // This could occur, for example, if an application called close() at the same
    // time that an onError callback was being executed because of a disconnect.
    std::lock_guard<std::mutex> lock(mLock);

    AudioStream::close();

    AAudioStream *stream = nullptr;
    {
        // Wait for any methods using mAAudioStream to finish.
        std::unique_lock<std::shared_mutex> lock2(mAAudioStreamLock);
        // Closing will delete *mAAudioStream so we need to null out the pointer atomically.
        stream = mAAudioStream.exchange(nullptr);
    }
    if (stream != nullptr) {
        if (OboeGlobals::areWorkaroundsEnabled()) {
            // Make sure we are really stopped. Do it under mLock
            // so another thread cannot call requestStart() right before the close.
            requestStop_l(stream);
            sleepBeforeClose();
        }
        return static_cast<Result>(mLibLoader->stream_close(stream));
    } else {
        return Result::ErrorClosed;
    }
}

static void oboe_stop_thread_proc(AudioStream *oboeStream) {
    if (oboeStream != nullptr) {
        oboeStream->requestStop();
    }
}

void AudioStreamAAudio::launchStopThread() {
    // Prevent multiple stop threads from being launched.
    if (mStopThreadAllowed.exchange(false)) {
        // Stop this stream on a separate thread
        std::thread t(oboe_stop_thread_proc, this);
        t.detach();
    }
}

DataCallbackResult AudioStreamAAudio::callOnAudioReady(AAudioStream * /*stream*/,
                                                                 void *audioData,
                                                                 int32_t numFrames) {
    DataCallbackResult result = fireDataCallback(audioData, numFrames);
    if (result == DataCallbackResult::Continue) {
        return result;
    } else {
        if (result == DataCallbackResult::Stop) {
            LOGD("Oboe callback returned DataCallbackResult::Stop");
        } else {
            LOGE("Oboe callback returned unexpected value = %d", result);
        }

        // Returning Stop caused various problems before S. See #1230
        if (OboeGlobals::areWorkaroundsEnabled() && getSdkVersion() <= __ANDROID_API_R__) {
            launchStopThread();
            return DataCallbackResult::Continue;
        } else {
            return DataCallbackResult::Stop; // OK >= API_S
        }
    }
}

Result AudioStreamAAudio::requestStart() {
    std::lock_guard<std::mutex> lock(mLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        // Avoid state machine errors in O_MR1.
        if (getSdkVersion() <= __ANDROID_API_O_MR1__) {
            StreamState state = static_cast<StreamState>(mLibLoader->stream_getState(stream));
            if (state == StreamState::Starting || state == StreamState::Started) {
                // WARNING: On P, AAudio is returning ErrorInvalidState for Output and OK for Input.
                return Result::OK;
            }
        }
        if (isDataCallbackSpecified()) {
            setDataCallbackEnabled(true);
        }
        mStopThreadAllowed = true;
        closePerformanceHint();
        return static_cast<Result>(mLibLoader->stream_requestStart(stream));
    } else {
        return Result::ErrorClosed;
    }
}

Result AudioStreamAAudio::requestPause() {
    std::lock_guard<std::mutex> lock(mLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        // Avoid state machine errors in O_MR1.
        if (getSdkVersion() <= __ANDROID_API_O_MR1__) {
            StreamState state = static_cast<StreamState>(mLibLoader->stream_getState(stream));
            if (state == StreamState::Pausing || state == StreamState::Paused) {
                return Result::OK;
            }
        }
        return static_cast<Result>(mLibLoader->stream_requestPause(stream));
    } else {
        return Result::ErrorClosed;
    }
}

Result AudioStreamAAudio::requestFlush() {
    std::lock_guard<std::mutex> lock(mLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        // Avoid state machine errors in O_MR1.
        if (getSdkVersion() <= __ANDROID_API_O_MR1__) {
            StreamState state = static_cast<StreamState>(mLibLoader->stream_getState(stream));
            if (state == StreamState::Flushing || state == StreamState::Flushed) {
                return Result::OK;
            }
        }
        return static_cast<Result>(mLibLoader->stream_requestFlush(stream));
    } else {
        return Result::ErrorClosed;
    }
}

Result AudioStreamAAudio::requestStop() {
    std::lock_guard<std::mutex> lock(mLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        return requestStop_l(stream);
    } else {
        return Result::ErrorClosed;
    }
}

// Call under mLock
Result AudioStreamAAudio::requestStop_l(AAudioStream *stream) {
    // Avoid state machine errors in O_MR1.
    if (getSdkVersion() <= __ANDROID_API_O_MR1__) {
        StreamState state = static_cast<StreamState>(mLibLoader->stream_getState(stream));
        if (state == StreamState::Stopping || state == StreamState::Stopped) {
            return Result::OK;
        }
    }
    return static_cast<Result>(mLibLoader->stream_requestStop(stream));
}

ResultWithValue<int32_t>   AudioStreamAAudio::write(const void *buffer,
                                     int32_t numFrames,
                                     int64_t timeoutNanoseconds) {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        int32_t result = mLibLoader->stream_write(mAAudioStream, buffer,
                                                  numFrames, timeoutNanoseconds);
        return ResultWithValue<int32_t>::createBasedOnSign(result);
    } else {
        return ResultWithValue<int32_t>(Result::ErrorClosed);
    }
}

ResultWithValue<int32_t>   AudioStreamAAudio::read(void *buffer,
                                 int32_t numFrames,
                                 int64_t timeoutNanoseconds) {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        int32_t result = mLibLoader->stream_read(mAAudioStream, buffer,
                                                 numFrames, timeoutNanoseconds);
        return ResultWithValue<int32_t>::createBasedOnSign(result);
    } else {
        return ResultWithValue<int32_t>(Result::ErrorClosed);
    }
}


// AAudioStream_waitForStateChange() can crash if it is waiting on a stream and that stream
// is closed from another thread.  We do not want to lock the stream for the duration of the call.
// So we call AAudioStream_waitForStateChange() with a timeout of zero so that it will not block.
// Then we can do our own sleep with the lock unlocked.
Result AudioStreamAAudio::waitForStateChange(StreamState currentState,
                                        StreamState *nextState,
                                        int64_t timeoutNanoseconds) {
    Result oboeResult = Result::ErrorTimeout;
    int64_t sleepTimeNanos = 20 * kNanosPerMillisecond; // arbitrary
    aaudio_stream_state_t currentAAudioState = static_cast<aaudio_stream_state_t>(currentState);

    aaudio_result_t result = AAUDIO_OK;
    int64_t timeLeftNanos = timeoutNanoseconds;

    mLock.lock();
    while (true) {
        // Do we still have an AAudio stream? If not then stream must have been closed.
        AAudioStream *stream = mAAudioStream.load();
        if (stream == nullptr) {
            if (nextState != nullptr) {
                *nextState = StreamState::Closed;
            }
            oboeResult = Result::ErrorClosed;
            break;
        }

        // Update and query state change with no blocking.
        aaudio_stream_state_t aaudioNextState;
        result = mLibLoader->stream_waitForStateChange(
                mAAudioStream,
                currentAAudioState,
                &aaudioNextState,
                0); // timeout=0 for non-blocking
        // AAudio will return AAUDIO_ERROR_TIMEOUT if timeout=0 and the state does not change.
        if (result != AAUDIO_OK && result != AAUDIO_ERROR_TIMEOUT) {
            oboeResult = static_cast<Result>(result);
            break;
        }
#if OBOE_FIX_FORCE_STARTING_TO_STARTED
        if (OboeGlobals::areWorkaroundsEnabled()
            && aaudioNextState == static_cast<aaudio_stream_state_t >(StreamState::Starting)) {
            aaudioNextState = static_cast<aaudio_stream_state_t >(StreamState::Started);
        }
#endif // OBOE_FIX_FORCE_STARTING_TO_STARTED
        if (nextState != nullptr) {
            *nextState = static_cast<StreamState>(aaudioNextState);
        }
        if (currentAAudioState != aaudioNextState) { // state changed?
            oboeResult = Result::OK;
            break;
        }

        // Did we timeout or did user ask for non-blocking?
        if (timeLeftNanos <= 0) {
            break;
        }

        // No change yet so sleep.
        mLock.unlock(); // Don't sleep while locked.
        if (sleepTimeNanos > timeLeftNanos) {
            sleepTimeNanos = timeLeftNanos; // last little bit
        }
        AudioClock::sleepForNanos(sleepTimeNanos);
        timeLeftNanos -= sleepTimeNanos;
        mLock.lock();
    }

    mLock.unlock();
    return oboeResult;
}

ResultWithValue<int32_t> AudioStreamAAudio::setBufferSizeInFrames(int32_t requestedFrames) {
    int32_t adjustedFrames = requestedFrames;
    if (adjustedFrames > mBufferCapacityInFrames) {
        adjustedFrames = mBufferCapacityInFrames;
    }
    // This calls getBufferSize() so avoid recursive lock.
    adjustedFrames = QuirksManager::getInstance().clipBufferSize(*this, adjustedFrames);

    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        int32_t newBufferSize = mLibLoader->stream_setBufferSize(mAAudioStream, adjustedFrames);
        // Cache the result if it's valid
        if (newBufferSize > 0) mBufferSizeInFrames = newBufferSize;
        return ResultWithValue<int32_t>::createBasedOnSign(newBufferSize);
    } else {
        return ResultWithValue<int32_t>(Result::ErrorClosed);
    }
}

StreamState AudioStreamAAudio::getState() {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        aaudio_stream_state_t aaudioState = mLibLoader->stream_getState(stream);
#if OBOE_FIX_FORCE_STARTING_TO_STARTED
        if (OboeGlobals::areWorkaroundsEnabled()
            && aaudioState == AAUDIO_STREAM_STATE_STARTING) {
            aaudioState = AAUDIO_STREAM_STATE_STARTED;
        }
#endif // OBOE_FIX_FORCE_STARTING_TO_STARTED
        return static_cast<StreamState>(aaudioState);
    } else {
        return StreamState::Closed;
    }
}

int32_t AudioStreamAAudio::getBufferSizeInFrames() {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        mBufferSizeInFrames = mLibLoader->stream_getBufferSize(stream);
    }
    return mBufferSizeInFrames;
}

void AudioStreamAAudio::updateFramesRead() {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
// Set to 1 for debugging race condition #1180 with mAAudioStream.
// See also DEBUG_CLOSE_RACE in OboeTester.
// This was left in the code so that we could test the fix again easily in the future.
// We could not trigger the race condition without adding these get calls and the sleeps.
#define DEBUG_CLOSE_RACE 0
#if DEBUG_CLOSE_RACE
    // This is used when testing race conditions with close().
    // See DEBUG_CLOSE_RACE in OboeTester
    AudioClock::sleepForNanos(400 * kNanosPerMillisecond);
#endif // DEBUG_CLOSE_RACE
    if (stream != nullptr) {
        mFramesRead = mLibLoader->stream_getFramesRead(stream);
    }
}

void AudioStreamAAudio::updateFramesWritten() {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        mFramesWritten = mLibLoader->stream_getFramesWritten(stream);
    }
}

ResultWithValue<int32_t> AudioStreamAAudio::getXRunCount() {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        return ResultWithValue<int32_t>::createBasedOnSign(mLibLoader->stream_getXRunCount(stream));
    } else {
        return ResultWithValue<int32_t>(Result::ErrorNull);
    }
}

Result AudioStreamAAudio::getTimestamp(clockid_t clockId,
                                   int64_t *framePosition,
                                   int64_t *timeNanoseconds) {
    if (getState() != StreamState::Started) {
        return Result::ErrorInvalidState;
    }
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        return static_cast<Result>(mLibLoader->stream_getTimestamp(stream, clockId,
                                               framePosition, timeNanoseconds));
    } else {
        return Result::ErrorNull;
    }
}

ResultWithValue<double> AudioStreamAAudio::calculateLatencyMillis() {
    // Get the time that a known audio frame was presented.
    int64_t hardwareFrameIndex;
    int64_t hardwareFrameHardwareTime;
    auto result = getTimestamp(CLOCK_MONOTONIC,
                               &hardwareFrameIndex,
                               &hardwareFrameHardwareTime);
    if (result != oboe::Result::OK) {
        return ResultWithValue<double>(static_cast<Result>(result));
    }

    // Get counter closest to the app.
    bool isOutput = (getDirection() == oboe::Direction::Output);
    int64_t appFrameIndex = isOutput ? getFramesWritten() : getFramesRead();

    // Assume that the next frame will be processed at the current time
    using namespace std::chrono;
    int64_t appFrameAppTime =
            duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();

    // Calculate the number of frames between app and hardware
    int64_t frameIndexDelta = appFrameIndex - hardwareFrameIndex;

    // Calculate the time which the next frame will be or was presented
    int64_t frameTimeDelta = (frameIndexDelta * oboe::kNanosPerSecond) / getSampleRate();
    int64_t appFrameHardwareTime = hardwareFrameHardwareTime + frameTimeDelta;

    // The current latency is the difference in time between when the current frame is at
    // the app and when it is at the hardware.
    double latencyNanos = static_cast<double>(isOutput
                          ? (appFrameHardwareTime - appFrameAppTime) // hardware is later
                          : (appFrameAppTime - appFrameHardwareTime)); // hardware is earlier
    double latencyMillis = latencyNanos / kNanosPerMillisecond;

    return ResultWithValue<double>(latencyMillis);
}

bool AudioStreamAAudio::isMMapUsed() {
    std::shared_lock<std::shared_mutex> lock(mAAudioStreamLock);
    AAudioStream *stream = mAAudioStream.load();
    if (stream != nullptr) {
        return AAudioExtensions::getInstance().isMMapUsed(stream);
    } else {
        return false;
    }
}

} // namespace oboe
