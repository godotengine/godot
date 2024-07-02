/*
 * Copyright 2015 The Android Open Source Project
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

#ifndef OBOE_STREAM_BASE_H_
#define OBOE_STREAM_BASE_H_

#include <memory>
#include <string>
#include "oboe/AudioStreamCallback.h"
#include "oboe/Definitions.h"

namespace oboe {

/**
 * Base class containing parameters for audio streams and builders.
 **/
class AudioStreamBase {

public:

    AudioStreamBase() {}

    virtual ~AudioStreamBase() = default;

    // This class only contains primitives so we can use default constructor and copy methods.

    /**
     * Default copy constructor
     */
    AudioStreamBase(const AudioStreamBase&) = default;

    /**
     * Default assignment operator
     */
    AudioStreamBase& operator=(const AudioStreamBase&) = default;

    /**
     * @return number of channels, for example 2 for stereo, or kUnspecified
     */
    int32_t getChannelCount() const { return mChannelCount; }

    /**
     * @return Direction::Input or Direction::Output
     */
    Direction getDirection() const { return mDirection; }

    /**
     * @return sample rate for the stream or kUnspecified
     */
    int32_t getSampleRate() const { return mSampleRate; }

    /**
     * @deprecated use `getFramesPerDataCallback` instead.
     */
    int32_t getFramesPerCallback() const { return getFramesPerDataCallback(); }

    /**
     * @return the number of frames in each data callback or kUnspecified.
     */
    int32_t getFramesPerDataCallback() const { return mFramesPerCallback; }

    /**
     * @return the audio sample format (e.g. Float or I16)
     */
    AudioFormat getFormat() const { return mFormat; }

    /**
     * Query the maximum number of frames that can be filled without blocking.
     * If the stream has been closed the last known value will be returned.
     *
     * @return buffer size
     */
    virtual int32_t getBufferSizeInFrames() { return mBufferSizeInFrames; }

    /**
     * @return capacityInFrames or kUnspecified
     */
    virtual int32_t getBufferCapacityInFrames() const { return mBufferCapacityInFrames; }

    /**
     * @return the sharing mode of the stream.
     */
    SharingMode getSharingMode() const { return mSharingMode; }

    /**
     * @return the performance mode of the stream.
     */
    PerformanceMode getPerformanceMode() const { return mPerformanceMode; }

    /**
     * @return the device ID of the stream.
     */
    int32_t getDeviceId() const { return mDeviceId; }

    /**
     * For internal use only.
     * @return the data callback object for this stream, if set.
     */
    AudioStreamDataCallback *getDataCallback() const {
        return mDataCallback;
    }

    /**
     * For internal use only.
     * @return the error callback object for this stream, if set.
     */
    AudioStreamErrorCallback *getErrorCallback() const {
        return mErrorCallback;
    }

    /**
     * @return true if a data callback was set for this stream
     */
    bool isDataCallbackSpecified() const {
        return mDataCallback != nullptr;
    }

    /**
     * Note that if the app does not set an error callback then a
     * default one may be provided.
     * @return true if an error callback was set for this stream
     */
    bool isErrorCallbackSpecified() const {
        return mErrorCallback != nullptr;
    }

    /**
     * @return the usage for this stream.
     */
    Usage getUsage() const { return mUsage; }

    /**
     * @return the stream's content type.
     */
    ContentType getContentType() const { return mContentType; }

    /**
     * @return the stream's input preset.
     */
    InputPreset getInputPreset() const { return mInputPreset; }

    /**
     * @return the stream's session ID allocation strategy (None or Allocate).
     */
    SessionId getSessionId() const { return mSessionId; }

    /**
     * @return whether the content of the stream is spatialized.
     */
    bool isContentSpatialized() const { return mIsContentSpatialized; }

    /**
     * @return the spatialization behavior for the stream.
     */
    SpatializationBehavior getSpatializationBehavior() const { return mSpatializationBehavior; }

    /**
     * Return the policy that determines whether the audio may or may not be captured
     * by other apps or the system.
     *
     * See AudioStreamBuilder_setAllowedCapturePolicy().
     *
     * Added in API level 29 to AAudio.
     *
     * @return the allowed capture policy, for example AllowedCapturePolicy::All
     */
    AllowedCapturePolicy getAllowedCapturePolicy() const { return mAllowedCapturePolicy; }

    /**
     * Return whether this input stream is marked as privacy sensitive.
     *
     * See AudioStreamBuilder_setPrivacySensitiveMode().
     *
     * Added in API level 30 to AAudio.
     *
     * @return PrivacySensitiveMode::Enabled if privacy sensitive,
     * PrivacySensitiveMode::Disabled if not privacy sensitive, and
     * PrivacySensitiveMode::Unspecified if API is not supported.
     */
    PrivacySensitiveMode getPrivacySensitiveMode() const { return mPrivacySensitiveMode; }

    /**
     * @return true if Oboe can convert channel counts to achieve optimal results.
     */
    bool isChannelConversionAllowed() const {
        return mChannelConversionAllowed;
    }

    /**
     * @return true if  Oboe can convert data formats to achieve optimal results.
     */
    bool  isFormatConversionAllowed() const {
        return mFormatConversionAllowed;
    }

    /**
     * @return whether and how Oboe can convert sample rates to achieve optimal results.
     */
    SampleRateConversionQuality getSampleRateConversionQuality() const {
        return mSampleRateConversionQuality;
    }

    /**
     * @return the stream's channel mask.
     */
    ChannelMask getChannelMask() const {
        return mChannelMask;
    }

    /**
     * @return number of channels for the hardware, for example 2 for stereo, or kUnspecified.
     */
    int32_t getHardwareChannelCount() const { return mHardwareChannelCount; }

    /**
     * @return hardware sample rate for the stream or kUnspecified
     */
    int32_t getHardwareSampleRate() const { return mHardwareSampleRate; }

    /**
     * @return the audio sample format of the hardware (e.g. Float or I16)
     */
    AudioFormat getHardwareFormat() const { return mHardwareFormat; }

protected:
    /** The callback which will be fired when new data is ready to be read/written. **/
    AudioStreamDataCallback        *mDataCallback = nullptr;
    std::shared_ptr<AudioStreamDataCallback> mSharedDataCallback;

    /** The callback which will be fired when an error or a disconnect occurs. **/
    AudioStreamErrorCallback       *mErrorCallback = nullptr;
    std::shared_ptr<AudioStreamErrorCallback> mSharedErrorCallback;

    /** Number of audio frames which will be requested in each callback */
    int32_t                         mFramesPerCallback = kUnspecified;
    /** Stream channel count */
    int32_t                         mChannelCount = kUnspecified;
    /** Stream sample rate */
    int32_t                         mSampleRate = kUnspecified;
    /** Stream audio device ID */
    int32_t                         mDeviceId = kUnspecified;
    /** Stream buffer capacity specified as a number of audio frames */
    int32_t                         mBufferCapacityInFrames = kUnspecified;
    /** Stream buffer size specified as a number of audio frames */
    int32_t                         mBufferSizeInFrames = kUnspecified;
    /** Stream channel mask. Only active on Android 32+ */
    ChannelMask                     mChannelMask = ChannelMask::Unspecified;

    /** Stream sharing mode */
    SharingMode                     mSharingMode = SharingMode::Shared;
    /** Format of audio frames */
    AudioFormat                     mFormat = AudioFormat::Unspecified;
    /** Stream direction */
    Direction                       mDirection = Direction::Output;
    /** Stream performance mode */
    PerformanceMode                 mPerformanceMode = PerformanceMode::None;

    /** Stream usage. Only active on Android 28+ */
    Usage                           mUsage = Usage::Media;
    /** Stream content type. Only active on Android 28+ */
    ContentType                     mContentType = ContentType::Music;
    /** Stream input preset. Only active on Android 28+
     * TODO InputPreset::Unspecified should be considered as a possible default alternative.
    */
    InputPreset                     mInputPreset = InputPreset::VoiceRecognition;
    /** Stream session ID allocation strategy. Only active on Android 28+ */
    SessionId                       mSessionId = SessionId::None;

    /** Allowed Capture Policy. Only active on Android 29+ */
    AllowedCapturePolicy            mAllowedCapturePolicy = AllowedCapturePolicy::Unspecified;

    /** Privacy Sensitive Mode. Only active on Android 30+ */
    PrivacySensitiveMode            mPrivacySensitiveMode = PrivacySensitiveMode::Unspecified;

    /** Control the name of the package creating the stream. Only active on Android 31+ */
    std::string                     mPackageName;
    /** Control the attribution tag of the context creating the stream. Only active on Android 31+ */
    std::string                     mAttributionTag;

    /** Whether the content is already spatialized. Only used on Android 32+ */
    bool                            mIsContentSpatialized = false;
    /** Spatialization Behavior. Only active on Android 32+ */
    SpatializationBehavior          mSpatializationBehavior = SpatializationBehavior::Unspecified;

    /** Hardware channel count. Only specified on Android 34+ AAudio streams */
    int32_t                         mHardwareChannelCount = kUnspecified;
    /** Hardware sample rate. Only specified on Android 34+ AAudio streams */
    int32_t                         mHardwareSampleRate = kUnspecified;
    /** Hardware format. Only specified on Android 34+ AAudio streams */
    AudioFormat                     mHardwareFormat = AudioFormat::Unspecified;

    // Control whether Oboe can convert channel counts to achieve optimal results.
    bool                            mChannelConversionAllowed = false;
    // Control whether Oboe can convert data formats to achieve optimal results.
    bool                            mFormatConversionAllowed = false;
    // Control whether and how Oboe can convert sample rates to achieve optimal results.
    SampleRateConversionQuality     mSampleRateConversionQuality = SampleRateConversionQuality::None;

    /** Validate stream parameters that might not be checked in lower layers */
    virtual Result isValidConfig() {
        switch (mFormat) {
            case AudioFormat::Unspecified:
            case AudioFormat::I16:
            case AudioFormat::Float:
            case AudioFormat::I24:
            case AudioFormat::I32:
            case AudioFormat::IEC61937:
                break;

            default:
                return Result::ErrorInvalidFormat;
        }

        switch (mSampleRateConversionQuality) {
            case SampleRateConversionQuality::None:
            case SampleRateConversionQuality::Fastest:
            case SampleRateConversionQuality::Low:
            case SampleRateConversionQuality::Medium:
            case SampleRateConversionQuality::High:
            case SampleRateConversionQuality::Best:
                return Result::OK;
            default:
                return Result::ErrorIllegalArgument;
        }
    }
};

} // namespace oboe

#endif /* OBOE_STREAM_BASE_H_ */
