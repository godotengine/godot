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

#ifndef OBOE_STREAM_BUILDER_H_
#define OBOE_STREAM_BUILDER_H_

#include "oboe/Definitions.h"
#include "oboe/AudioStreamBase.h"
#include "oboe/Utilities.h"
#include "ResultWithValue.h"

namespace oboe {

    // This depends on AudioStream, so we use forward declaration, it will close and delete the stream
    struct StreamDeleterFunctor;
    using ManagedStream = std::unique_ptr<AudioStream, StreamDeleterFunctor>;

/**
 * Factory class for an audio Stream.
 */
class AudioStreamBuilder : public AudioStreamBase {
public:

    AudioStreamBuilder() : AudioStreamBase() {}

    AudioStreamBuilder(const AudioStreamBase &audioStreamBase): AudioStreamBase(audioStreamBase) {}

    /**
     * Request a specific number of channels.
     *
     * Default is kUnspecified. If the value is unspecified then
     * the application should query for the actual value after the stream is opened.
     *
     * As the channel count here may be different from the corresponding channel count of
     * provided channel mask used in setChannelMask(). The last called will be respected
     * if this function and setChannelMask() are called.
     */
    AudioStreamBuilder *setChannelCount(int channelCount) {
        mChannelCount = channelCount;
        mChannelMask = ChannelMask::Unspecified;
        return this;
    }

    /**
     * Request a specific channel mask.
     *
     * Default is kUnspecified. If the value is unspecified then the application
     * should query for the actual value after the stream is opened.
     *
     * As the corresponding channel count of provided channel mask here may be different
     * from the channel count used in setChannelCount(). The last called will be respected
     * if this function and setChannelCount() are called.
     *
     * As the setChannelMask API is available on Android 32+, this call will only take effects
     * on Android 32+.
     */
     AudioStreamBuilder *setChannelMask(ChannelMask channelMask) {
         mChannelMask = channelMask;
         mChannelCount = getChannelCountFromChannelMask(channelMask);
         return this;
     }

    /**
     * Request the direction for a stream. The default is Direction::Output.
     *
     * @param direction Direction::Output or Direction::Input
     */
    AudioStreamBuilder *setDirection(Direction direction) {
        mDirection = direction;
        return this;
    }

    /**
     * Request a specific sample rate in Hz.
     *
     * Default is kUnspecified. If the value is unspecified then
     * the application should query for the actual value after the stream is opened.
     *
     * Technically, this should be called the "frame rate" or "frames per second",
     * because it refers to the number of complete frames transferred per second.
     * But it is traditionally called "sample rate". Se we use that term.
     *
     */
    AudioStreamBuilder *setSampleRate(int32_t sampleRate) {
        mSampleRate = sampleRate;
        return this;
    }

    /**
     * @deprecated use `setFramesPerDataCallback` instead.
     */
    AudioStreamBuilder *setFramesPerCallback(int framesPerCallback) {
        return setFramesPerDataCallback(framesPerCallback);
    }

    /**
     * Request a specific number of frames for the data callback.
     *
     * Default is kUnspecified. If the value is unspecified then
     * the actual number may vary from callback to callback.
     *
     * If an application can handle a varying number of frames then we recommend
     * leaving this unspecified. This allow the underlying API to optimize
     * the callbacks. But if your application is, for example, doing FFTs or other block
     * oriented operations, then call this function to get the sizes you need.
     *
     * Calling setFramesPerDataCallback() does not guarantee anything about timing.
     * This just collects the data into a the number of frames that your app requires.
     * We encourage leaving this unspecified in most cases.
     *
     * If this number is larger than the burst size, some bursts will not receive a callback.
     * If this number is smaller than the burst size, there may be multiple callbacks in a single
     * burst.
     *
     * @param framesPerCallback
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setFramesPerDataCallback(int framesPerCallback) {
        mFramesPerCallback = framesPerCallback;
        return this;
    }

    /**
     * Request a sample data format, for example Format::Float.
     *
     * Default is Format::Unspecified. If the value is unspecified then
     * the application should query for the actual value after the stream is opened.
     */
    AudioStreamBuilder *setFormat(AudioFormat format) {
        mFormat = format;
        return this;
    }

    /**
     * Set the requested buffer capacity in frames.
     * BufferCapacityInFrames is the maximum possible BufferSizeInFrames.
     *
     * The final stream capacity may differ. For AAudio it should be at least this big.
     * For OpenSL ES, it could be smaller.
     *
     * Default is kUnspecified.
     *
     * @param bufferCapacityInFrames the desired buffer capacity in frames or kUnspecified
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setBufferCapacityInFrames(int32_t bufferCapacityInFrames) {
        mBufferCapacityInFrames = bufferCapacityInFrames;
        return this;
    }

    /**
     * Get the audio API which will be requested when opening the stream. No guarantees that this is
     * the API which will actually be used. Query the stream itself to find out the API which is
     * being used.
     *
     * If you do not specify the API, then AAudio will be used if isAAudioRecommended()
     * returns true. Otherwise OpenSL ES will be used.
     *
     * @return the requested audio API
     */
    AudioApi getAudioApi() const { return mAudioApi; }

    /**
     * If you leave this unspecified then Oboe will choose the best API
     * for the device and SDK version at runtime.
     *
     * This should almost always be left unspecified, except for debugging purposes.
     * Specifying AAudio will force Oboe to use AAudio on 8.0, which is extremely risky.
     * Specifying OpenSLES should mainly be used to test legacy performance/functionality.
     *
     * If the caller requests AAudio and it is supported then AAudio will be used.
     *
     * @param audioApi Must be AudioApi::Unspecified, AudioApi::OpenSLES or AudioApi::AAudio.
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setAudioApi(AudioApi audioApi) {
        mAudioApi = audioApi;
        return this;
    }

    /**
     * Is the AAudio API supported on this device?
     *
     * AAudio was introduced in the Oreo 8.0 release.
     *
     * @return true if supported
     */
    static bool isAAudioSupported();

    /**
     * Is the AAudio API recommended this device?
     *
     * AAudio may be supported but not recommended because of version specific issues.
     * AAudio is not recommended for Android 8.0 or earlier versions.
     *
     * @return true if recommended
     */
    static bool isAAudioRecommended();

    /**
     * Request a mode for sharing the device.
     * The requested sharing mode may not be available.
     * So the application should query for the actual mode after the stream is opened.
     *
     * @param sharingMode SharingMode::Shared or SharingMode::Exclusive
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setSharingMode(SharingMode sharingMode) {
        mSharingMode = sharingMode;
        return this;
    }

    /**
     * Request a performance level for the stream.
     * This will determine the latency, the power consumption, and the level of
     * protection from glitches.
     *
     * @param performanceMode for example, PerformanceMode::LowLatency
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setPerformanceMode(PerformanceMode performanceMode) {
        mPerformanceMode = performanceMode;
        return this;
    }


    /**
     * Set the intended use case for an output stream.
     *
     * The system will use this information to optimize the behavior of the stream.
     * This could, for example, affect how volume and focus is handled for the stream.
     * The usage is ignored for input streams.
     *
     * The default, if you do not call this function, is Usage::Media.
     *
     * Added in API level 28.
     *
     * @param usage the desired usage, eg. Usage::Game
     */
    AudioStreamBuilder *setUsage(Usage usage) {
        mUsage = usage;
        return this;
    }

    /**
     * Set the type of audio data that an output stream will carry.
     *
     * The system will use this information to optimize the behavior of the stream.
     * This could, for example, affect whether a stream is paused when a notification occurs.
     * The contentType is ignored for input streams.
     *
     * The default, if you do not call this function, is ContentType::Music.
     *
     * Added in API level 28.
     *
     * @param contentType the type of audio data, eg. ContentType::Speech
     */
    AudioStreamBuilder *setContentType(ContentType contentType) {
        mContentType = contentType;
        return this;
    }

    /**
     * Set the input (capture) preset for the stream.
     *
     * The system will use this information to optimize the behavior of the stream.
     * This could, for example, affect which microphones are used and how the
     * recorded data is processed.
     *
     * The default, if you do not call this function, is InputPreset::VoiceRecognition.
     * That is because VoiceRecognition is the preset with the lowest latency
     * on many platforms.
     *
     * Added in API level 28.
     *
     * @param inputPreset the desired configuration for recording
     */
    AudioStreamBuilder *setInputPreset(InputPreset inputPreset) {
        mInputPreset = inputPreset;
        return this;
    }

    /** Set the requested session ID.
     *
     * The session ID can be used to associate a stream with effects processors.
     * The effects are controlled using the Android AudioEffect Java API.
     *
     * The default, if you do not call this function, is SessionId::None.
     *
     * If set to SessionId::Allocate then a session ID will be allocated
     * when the stream is opened.
     *
     * The allocated session ID can be obtained by calling AudioStream::getSessionId()
     * and then used with this function when opening another stream.
     * This allows effects to be shared between streams.
     *
     * Session IDs from Oboe can be used the Android Java APIs and vice versa.
     * So a session ID from an Oboe stream can be passed to Java
     * and effects applied using the Java AudioEffect API.
     *
     * Allocated session IDs will always be positive and nonzero.
     *
     * Added in API level 28.
     *
     * @param sessionId an allocated sessionID or SessionId::Allocate
     */
    AudioStreamBuilder *setSessionId(SessionId sessionId) {
        mSessionId = sessionId;
        return this;
    }

    /**
     * Request a stream to a specific audio input/output device given an audio device ID.
     *
     * In most cases, the primary device will be the appropriate device to use, and the
     * deviceId can be left kUnspecified.
     *
     * The ID could be obtained from the Java AudioManager.
     * AudioManager.getDevices() returns an array of AudioDeviceInfo,
     * which contains a getId() method. That ID can be passed to this function.
     *
     * It is possible that you may not get the device that you requested.
     * So if it is important to you, you should call
     * stream->getDeviceId() after the stream is opened to
     * verify the actual ID.
     *
     * Note that when using OpenSL ES, this will be ignored and the created
     * stream will have deviceId kUnspecified.
     *
     * @param deviceId device identifier or kUnspecified
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setDeviceId(int32_t deviceId) {
        mDeviceId = deviceId;
        return this;
    }

    /**
     * Specify whether this stream audio may or may not be captured by other apps or the system.
     *
     * The default is AllowedCapturePolicy::Unspecified which maps to AAUDIO_ALLOW_CAPTURE_BY_ALL.
     *
     * Note that an application can also set its global policy, in which case the most restrictive
     * policy is always applied. See android.media.AudioAttributes.setAllowedCapturePolicy.
     *
     * Added in API level 29 to AAudio.
     *
     * @param inputPreset the desired level of opt-out from being captured.
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setAllowedCapturePolicy(AllowedCapturePolicy allowedCapturePolicy) {
        mAllowedCapturePolicy = allowedCapturePolicy;
        return this;
    }

    /** Indicates whether this input stream must be marked as privacy sensitive or not.
     *
     * When PrivacySensitiveMode::Enabled, this input stream is privacy sensitive and any
     * concurrent capture is not permitted.
     *
     * This is off (PrivacySensitiveMode::Disabled) by default except when the input preset is
     * InputPreset::VoiceRecognition or InputPreset::Camcorder
     *
     * Always takes precedence over default from input preset when set explicitly.
     *
     * Only relevant if the stream direction is Direction::Input and AAudio is used.
     *
     * Added in API level 30 to AAudio.
     *
     * @param privacySensitive PrivacySensitiveMode::Enabled if capture from this stream must be
     * marked as privacy sensitive, PrivacySensitiveMode::Disabled if stream should be marked as
     * not sensitive.
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setPrivacySensitiveMode(PrivacySensitiveMode privacySensitiveMode) {
        mPrivacySensitiveMode = privacySensitiveMode;
        return this;
    }

    /**
     * Specifies whether the audio data of this output stream has already been processed for spatialization.
     *
     * If the stream has been processed for spatialization, setting this to true will prevent issues such as
     * double-processing on platforms that will spatialize audio data.
     *
     * This is false by default.
     *
     * Available since API level 32.
     *
     * @param isContentSpatialized whether the content is already spatialized
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setIsContentSpatialized(bool isContentSpatialized) {
        mIsContentSpatialized = isContentSpatialized;
        return this;
    }

    /**
     * Sets the behavior affecting whether spatialization will be used.
     *
     * The AAudio system will use this information to select whether the stream will go through a
     * spatializer effect or not when the effect is supported and enabled.
     *
     * This is SpatializationBehavior::Never by default.
     *
     * Available since API level 32.
     *
     * @param spatializationBehavior the desired spatialization behavior
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setSpatializationBehavior(SpatializationBehavior spatializationBehavior) {
        mSpatializationBehavior = spatializationBehavior;
        return this;
    }

    /**
     * Specifies an object to handle data related callbacks from the underlying API.
     *
     * <strong>Important: See AudioStreamCallback for restrictions on what may be called
     * from the callback methods.</strong>
     *
     * We pass a shared_ptr so that the sharedDataCallback object cannot be deleted
     * before the stream is deleted.
     *
     * @param sharedDataCallback
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setDataCallback(std::shared_ptr<AudioStreamDataCallback> sharedDataCallback) {
        // Use this raw pointer in the rest of the code to retain backwards compatibility.
        mDataCallback = sharedDataCallback.get();
        // Hold a shared_ptr to protect the raw pointer for the lifetime of the stream.
        mSharedDataCallback = sharedDataCallback;
        return this;
    }

    /**
    * Pass a raw pointer to a data callback. This is not recommended because the dataCallback
    * object might get deleted by the app while it is being used.
    *
    * @deprecated Call setDataCallback(std::shared_ptr<AudioStreamDataCallback>) instead.
    * @param dataCallback
    * @return pointer to the builder so calls can be chained
    */
    AudioStreamBuilder *setDataCallback(AudioStreamDataCallback *dataCallback) {
        mDataCallback = dataCallback;
        mSharedDataCallback = nullptr;
        return this;
    }

    /**
     * Specifies an object to handle error related callbacks from the underlying API.
     * This can occur when a stream is disconnected because a headset is plugged in or unplugged.
     * It can also occur if the audio service fails or if an exclusive stream is stolen by
     * another stream.
     *
     * <strong>Important: See AudioStreamCallback for restrictions on what may be called
     * from the callback methods.</strong>
     *
     * <strong>When an error callback occurs, the associated stream must be stopped and closed
     * in a separate thread.</strong>
     *
     * We pass a shared_ptr so that the errorCallback object cannot be deleted before the stream is deleted.
     * If the stream was created using a shared_ptr then the stream cannot be deleted before the
     * error callback has finished running.
     *
     * @param sharedErrorCallback
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setErrorCallback(std::shared_ptr<AudioStreamErrorCallback> sharedErrorCallback) {
        // Use this raw pointer in the rest of the code to retain backwards compatibility.
        mErrorCallback = sharedErrorCallback.get();
        // Hold a shared_ptr to protect the raw pointer for the lifetime of the stream.
        mSharedErrorCallback = sharedErrorCallback;
        return this;
    }

    /**
    * Pass a raw pointer to an error callback. This is not recommended because the errorCallback
    * object might get deleted by the app while it is being used.
    *
    * @deprecated Call setErrorCallback(std::shared_ptr<AudioStreamErrorCallback>) instead.
    * @param errorCallback
    * @return pointer to the builder so calls can be chained
    */
    AudioStreamBuilder *setErrorCallback(AudioStreamErrorCallback *errorCallback) {
        mErrorCallback = errorCallback;
        mSharedErrorCallback = nullptr;
        return this;
    }

    /**
     * Specifies an object to handle data or error related callbacks from the underlying API.
     *
     * This is the equivalent of calling both setDataCallback() and setErrorCallback().
     *
     * <strong>Important: See AudioStreamCallback for restrictions on what may be called
     * from the callback methods.</strong>
     *
     * @deprecated Call setDataCallback(std::shared_ptr<AudioStreamDataCallback>) and
     *     setErrorCallback(std::shared_ptr<AudioStreamErrorCallback>) instead.
     * @param streamCallback
     * @return pointer to the builder so calls can be chained
     */
    AudioStreamBuilder *setCallback(AudioStreamCallback *streamCallback) {
        // Use the same callback object for both, dual inheritance.
        mDataCallback = streamCallback;
        mErrorCallback = streamCallback;
        return this;
    }

    /**
     * If true then Oboe might convert channel counts to achieve optimal results.
     * On some versions of Android for example, stereo streams could not use a FAST track.
     * So a mono stream might be used instead and duplicated to two channels.
     * On some devices, mono streams might be broken, so a stereo stream might be opened
     * and converted to mono.
     *
     * Default is false.
     */
    AudioStreamBuilder *setChannelConversionAllowed(bool allowed) {
        mChannelConversionAllowed = allowed;
        return this;
    }

    /**
     * If true then Oboe might convert data formats to achieve optimal results.
     * On some versions of Android, for example, a float stream could not get a
     * low latency data path. So an I16 stream might be opened and converted to float.
     *
     * Default is false.
     */
    AudioStreamBuilder *setFormatConversionAllowed(bool allowed) {
        mFormatConversionAllowed = allowed;
        return this;
    }

    /**
     * Specify the quality of the sample rate converter in Oboe.
     *
     * If set to None then Oboe will not do sample rate conversion. But the underlying APIs might
     * still do sample rate conversion if you specify a sample rate.
     * That can prevent you from getting a low latency stream.
     *
     * If you do the conversion in Oboe then you might still get a low latency stream.
     *
     * Default is SampleRateConversionQuality::None
     */
    AudioStreamBuilder *setSampleRateConversionQuality(SampleRateConversionQuality quality) {
        mSampleRateConversionQuality = quality;
        return this;
    }

    /**
    * Declare the name of the package creating the stream.
    *
    * This is usually {@code Context#getPackageName()}.
    *
    * The default, if you do not call this function, is a random package in the calling uid.
    * The vast majority of apps have only one package per calling UID.
    * If an invalid package name is set, input streams may not be given permission to
    * record when started.
    *
    * The package name is usually the applicationId in your app's build.gradle file.
    *
    * Available since API level 31.
    *
    * @param packageName packageName of the calling app.
    */
    AudioStreamBuilder *setPackageName(std::string packageName) {
        mPackageName = packageName;
        return this;
    }

    /**
    * Declare the attribution tag of the context creating the stream.
    *
    * This is usually {@code Context#getAttributionTag()}.
    *
    * The default, if you do not call this function, is null.
    *
    * Available since API level 31.
    *
    * @param attributionTag attributionTag of the calling context.
    */
    AudioStreamBuilder *setAttributionTag(std::string attributionTag) {
        mAttributionTag = attributionTag;
        return this;
    }

    /**
     * @return true if AAudio will be used based on the current settings.
     */
    bool willUseAAudio() const {
        return (mAudioApi == AudioApi::AAudio && isAAudioSupported())
                || (mAudioApi == AudioApi::Unspecified && isAAudioRecommended());
    }

    /**
     * Create and open a stream object based on the current settings.
     *
     * The caller owns the pointer to the AudioStream object
     * and must delete it when finished.
     *
     * @deprecated Use openStream(std::shared_ptr<oboe::AudioStream> &stream) instead.
     * @param stream pointer to a variable to receive the stream address
     * @return OBOE_OK if successful or a negative error code
     */
    Result openStream(AudioStream **stream);

    /**
     * Create and open a stream object based on the current settings.
     *
     * The caller shares the pointer to the AudioStream object.
     * The shared_ptr is used internally by Oboe to prevent the stream from being
     * deleted while it is being used by callbacks.
     *
     * @param stream reference to a shared_ptr to receive the stream address
     * @return OBOE_OK if successful or a negative error code
     */
    Result openStream(std::shared_ptr<oboe::AudioStream> &stream);

    /**
     * Create and open a ManagedStream object based on the current builder state.
     *
     * The caller must create a unique ptr, and pass by reference so it can be
     * modified to point to an opened stream. The caller owns the unique ptr,
     * and it will be automatically closed and deleted when going out of scope.
     *
     * @deprecated Use openStream(std::shared_ptr<oboe::AudioStream> &stream) instead.
     * @param stream Reference to the ManagedStream (uniqueptr) used to keep track of stream
     * @return OBOE_OK if successful or a negative error code.
     */
    Result openManagedStream(ManagedStream &stream);

private:

    /**
     * Use this internally to implement opening with a shared_ptr.
     *
     * @param stream pointer to a variable to receive the stream address
     * @return OBOE_OK if successful or a negative error code.
     */
    Result openStreamInternal(AudioStream **streamPP);

    /**
     * @param other
     * @return true if channels, format and sample rate match
     */
    bool isCompatible(AudioStreamBase &other);

    /**
     * Create an AudioStream object. The AudioStream must be opened before use.
     *
     * The caller owns the pointer.
     *
     * @return pointer to an AudioStream object or nullptr.
     */
    oboe::AudioStream *build();

    AudioApi       mAudioApi = AudioApi::Unspecified;
};

} // namespace oboe

#endif /* OBOE_STREAM_BUILDER_H_ */
