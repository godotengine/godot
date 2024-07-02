/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef OBOE_DEFINITIONS_H
#define OBOE_DEFINITIONS_H

#include <cstdint>
#include <type_traits>

// Oboe needs to be able to build on old NDKs so we use hard coded constants.
// The correctness of these constants is verified in "aaudio/AAudioLoader.cpp".

namespace oboe {

    /**
     * Represents any attribute, property or value which hasn't been specified.
     */
    constexpr int32_t kUnspecified = 0;

    // TODO: Investigate using std::chrono
    /**
     * The number of nanoseconds in a microsecond. 1,000.
     */
    constexpr int64_t kNanosPerMicrosecond =    1000;

    /**
     * The number of nanoseconds in a millisecond. 1,000,000.
     */
    constexpr int64_t kNanosPerMillisecond =    kNanosPerMicrosecond * 1000;

    /**
     * The number of milliseconds in a second. 1,000.
     */
    constexpr int64_t kMillisPerSecond =        1000;

    /**
     * The number of nanoseconds in a second. 1,000,000,000.
     */
    constexpr int64_t kNanosPerSecond =         kNanosPerMillisecond * kMillisPerSecond;

    /**
     * The state of the audio stream.
     */
    enum class StreamState : int32_t { // aaudio_stream_state_t
        Uninitialized = 0, // AAUDIO_STREAM_STATE_UNINITIALIZED,
        Unknown = 1, // AAUDIO_STREAM_STATE_UNKNOWN,
        Open = 2, // AAUDIO_STREAM_STATE_OPEN,
        Starting = 3, // AAUDIO_STREAM_STATE_STARTING,
        Started = 4, // AAUDIO_STREAM_STATE_STARTED,
        Pausing = 5, // AAUDIO_STREAM_STATE_PAUSING,
        Paused = 6, // AAUDIO_STREAM_STATE_PAUSED,
        Flushing = 7, // AAUDIO_STREAM_STATE_FLUSHING,
        Flushed = 8, // AAUDIO_STREAM_STATE_FLUSHED,
        Stopping = 9, // AAUDIO_STREAM_STATE_STOPPING,
        Stopped = 10, // AAUDIO_STREAM_STATE_STOPPED,
        Closing = 11, // AAUDIO_STREAM_STATE_CLOSING,
        Closed = 12, // AAUDIO_STREAM_STATE_CLOSED,
        Disconnected = 13, // AAUDIO_STREAM_STATE_DISCONNECTED,
    };

    /**
     * The direction of the stream.
     */
    enum class Direction : int32_t { // aaudio_direction_t

        /**
         * Used for playback.
         */
        Output = 0, // AAUDIO_DIRECTION_OUTPUT,

        /**
         * Used for recording.
         */
        Input = 1, // AAUDIO_DIRECTION_INPUT,
    };

    /**
     * The format of audio samples.
     */
    enum class AudioFormat : int32_t { // aaudio_format_t
        /**
         * Invalid format.
         */
        Invalid = -1, // AAUDIO_FORMAT_INVALID,

        /**
         * Unspecified format. Format will be decided by Oboe.
         * When calling getHardwareFormat(), this will be returned if
         * the API is not supported.
         */
        Unspecified = 0, // AAUDIO_FORMAT_UNSPECIFIED,

        /**
         * Signed 16-bit integers.
         */
        I16 = 1, // AAUDIO_FORMAT_PCM_I16,

        /**
         * Single precision floating point.
         *
         * This is the recommended format for most applications.
         * But note that the use of Float may prevent the opening of
         * a low-latency input path on OpenSL ES or Legacy AAudio streams.
         */
        Float = 2, // AAUDIO_FORMAT_PCM_FLOAT,

        /**
         * Signed 24-bit integers, packed into 3 bytes.
         *
         * Note that the use of this format does not guarantee that
         * the full precision will be provided.  The underlying device may
         * be using I16 format.
         *
         * Added in API 31 (S).
         */
        I24 = 3, // AAUDIO_FORMAT_PCM_I24_PACKED

        /**
         * Signed 32-bit integers.
         *
         * Note that the use of this format does not guarantee that
         * the full precision will be provided.  The underlying device may
         * be using I16 format.
         *
         * Added in API 31 (S).
         */
        I32 = 4, // AAUDIO_FORMAT_PCM_I32

        /**
        * This format is used for compressed audio wrapped in IEC61937 for HDMI
        * or S/PDIF passthrough.
        *
        * Unlike PCM playback, the Android framework is not able to do format
        * conversion for IEC61937. In that case, when IEC61937 is requested, sampling
        * rate and channel count or channel mask must be specified. Otherwise, it may
        * fail when opening the stream. Apps are able to get the correct configuration
        * for the playback by calling AudioManager#getDevices(int).
        *
        * Available since API 34 (U).
        */
        IEC61937 = 5, // AAUDIO_FORMAT_IEC61937
    };

    /**
     * The result of an audio callback.
     */
    enum class DataCallbackResult : int32_t { // aaudio_data_callback_result_t
        // Indicates to the caller that the callbacks should continue.
        Continue = 0, // AAUDIO_CALLBACK_RESULT_CONTINUE,

        // Indicates to the caller that the callbacks should stop immediately.
        Stop = 1, // AAUDIO_CALLBACK_RESULT_STOP,
    };

    /**
     * The result of an operation. All except the `OK` result indicates that an error occurred.
     * The `Result` can be converted into a human readable string using `convertToText`.
     */
    enum class Result : int32_t { // aaudio_result_t
        OK = 0, // AAUDIO_OK
        ErrorBase = -900, // AAUDIO_ERROR_BASE,
        ErrorDisconnected = -899, // AAUDIO_ERROR_DISCONNECTED,
        ErrorIllegalArgument = -898, // AAUDIO_ERROR_ILLEGAL_ARGUMENT,
        ErrorInternal = -896, // AAUDIO_ERROR_INTERNAL,
        ErrorInvalidState = -895, // AAUDIO_ERROR_INVALID_STATE,
        ErrorInvalidHandle = -892, // AAUDIO_ERROR_INVALID_HANDLE,
        ErrorUnimplemented = -890, // AAUDIO_ERROR_UNIMPLEMENTED,
        ErrorUnavailable = -889, // AAUDIO_ERROR_UNAVAILABLE,
        ErrorNoFreeHandles = -888, // AAUDIO_ERROR_NO_FREE_HANDLES,
        ErrorNoMemory = -887, // AAUDIO_ERROR_NO_MEMORY,
        ErrorNull = -886, // AAUDIO_ERROR_NULL,
        ErrorTimeout = -885, // AAUDIO_ERROR_TIMEOUT,
        ErrorWouldBlock = -884, // AAUDIO_ERROR_WOULD_BLOCK,
        ErrorInvalidFormat = -883, // AAUDIO_ERROR_INVALID_FORMAT,
        ErrorOutOfRange = -882, // AAUDIO_ERROR_OUT_OF_RANGE,
        ErrorNoService = -881, // AAUDIO_ERROR_NO_SERVICE,
        ErrorInvalidRate = -880, // AAUDIO_ERROR_INVALID_RATE,
        // Reserved for future AAudio result types
        Reserved1,
        Reserved2,
        Reserved3,
        Reserved4,
        Reserved5,
        Reserved6,
        Reserved7,
        Reserved8,
        Reserved9,
        Reserved10,
        ErrorClosed = -869,
    };

    /**
     * The sharing mode of the audio stream.
     */
    enum class SharingMode : int32_t { // aaudio_sharing_mode_t

        /**
         * This will be the only stream using a particular source or sink.
         * This mode will provide the lowest possible latency.
         * You should close EXCLUSIVE streams immediately when you are not using them.
         *
         * If you do not need the lowest possible latency then we recommend using Shared,
         * which is the default.
         */
        Exclusive = 0, // AAUDIO_SHARING_MODE_EXCLUSIVE,

        /**
         * Multiple applications can share the same device.
         * The data from output streams will be mixed by the audio service.
         * The data for input streams will be distributed by the audio service.
         *
         * This will have higher latency than the EXCLUSIVE mode.
         */
        Shared = 1, // AAUDIO_SHARING_MODE_SHARED,
    };

    /**
     * The performance mode of the audio stream.
     */
    enum class PerformanceMode : int32_t { // aaudio_performance_mode_t

        /**
         * No particular performance needs. Default.
         */
        None = 10, // AAUDIO_PERFORMANCE_MODE_NONE,

        /**
         * Extending battery life is most important.
         */
        PowerSaving = 11, // AAUDIO_PERFORMANCE_MODE_POWER_SAVING,

        /**
         * Reducing latency is most important.
         */
        LowLatency = 12, // AAUDIO_PERFORMANCE_MODE_LOW_LATENCY
    };

    /**
     * The underlying audio API used by the audio stream.
     */
    enum class AudioApi : int32_t {
        /**
         * Try to use AAudio. If not available then use OpenSL ES.
         */
        Unspecified = kUnspecified,

        /**
         * Use OpenSL ES.
         * Note that OpenSL ES is deprecated in Android 13, API 30 and above.
         */
        OpenSLES,

        /**
         * Try to use AAudio. Fail if unavailable.
         * AAudio was first supported in Android 8, API 26 and above.
         * It is only recommended for API 27 and above.
         */
        AAudio
    };

    /**
     * Specifies the quality of the sample rate conversion performed by Oboe.
     * Higher quality will require more CPU load.
     * Higher quality conversion will probably be implemented using a sinc based resampler.
     */
    enum class SampleRateConversionQuality : int32_t {
        /**
         * No conversion by Oboe. Underlying APIs may still do conversion.
         */
        None,
        /**
         * Fastest conversion but may not sound great.
         * This may be implemented using bilinear interpolation.
         */
        Fastest,
        /**
         * Low quality conversion with 8 taps.
         */
        Low,
        /**
         * Medium quality conversion with 16 taps.
         */
        Medium,
        /**
         * High quality conversion with 32 taps.
         */
        High,
        /**
         * Highest quality conversion, which may be expensive in terms of CPU.
         */
        Best,
    };

    /**
     * The Usage attribute expresses *why* you are playing a sound, what is this sound used for.
     * This information is used by certain platforms or routing policies
     * to make more refined volume or routing decisions.
     *
     * Note that these match the equivalent values in AudioAttributes in the Android Java API.
     *
     * This attribute only has an effect on Android API 28+.
     */
    enum class Usage : int32_t { // aaudio_usage_t
        /**
         * Use this for streaming media, music performance, video, podcasts, etcetera.
         */
        Media =  1, // AAUDIO_USAGE_MEDIA

        /**
         * Use this for voice over IP, telephony, etcetera.
         */
        VoiceCommunication = 2, // AAUDIO_USAGE_VOICE_COMMUNICATION

        /**
         * Use this for sounds associated with telephony such as busy tones, DTMF, etcetera.
         */
        VoiceCommunicationSignalling = 3, // AAUDIO_USAGE_VOICE_COMMUNICATION_SIGNALLING

        /**
         * Use this to demand the users attention.
         */
        Alarm = 4, // AAUDIO_USAGE_ALARM

        /**
         * Use this for notifying the user when a message has arrived or some
         * other background event has occured.
         */
        Notification = 5, // AAUDIO_USAGE_NOTIFICATION

        /**
         * Use this when the phone rings.
         */
        NotificationRingtone = 6, // AAUDIO_USAGE_NOTIFICATION_RINGTONE

        /**
         * Use this to attract the users attention when, for example, the battery is low.
         */
        NotificationEvent = 10, // AAUDIO_USAGE_NOTIFICATION_EVENT

        /**
         * Use this for screen readers, etcetera.
         */
        AssistanceAccessibility = 11, // AAUDIO_USAGE_ASSISTANCE_ACCESSIBILITY

        /**
         * Use this for driving or navigation directions.
         */
        AssistanceNavigationGuidance = 12, // AAUDIO_USAGE_ASSISTANCE_NAVIGATION_GUIDANCE

        /**
         * Use this for user interface sounds, beeps, etcetera.
         */
        AssistanceSonification = 13, // AAUDIO_USAGE_ASSISTANCE_SONIFICATION

        /**
         * Use this for game audio and sound effects.
         */
        Game = 14, // AAUDIO_USAGE_GAME

        /**
         * Use this for audio responses to user queries, audio instructions or help utterances.
         */
        Assistant = 16, // AAUDIO_USAGE_ASSISTANT
    };


    /**
     * The ContentType attribute describes *what* you are playing.
     * It expresses the general category of the content. This information is optional.
     * But in case it is known (for instance {@link Movie} for a
     * movie streaming service or {@link Speech} for
     * an audio book application) this information might be used by the audio framework to
     * enforce audio focus.
     *
     * Note that these match the equivalent values in AudioAttributes in the Android Java API.
     *
     * This attribute only has an effect on Android API 28+.
     */
    enum ContentType : int32_t { // aaudio_content_type_t

        /**
         * Use this for spoken voice, audio books, etcetera.
         */
        Speech = 1, // AAUDIO_CONTENT_TYPE_SPEECH

        /**
         * Use this for pre-recorded or live music.
         */
        Music = 2, // AAUDIO_CONTENT_TYPE_MUSIC

        /**
         * Use this for a movie or video soundtrack.
         */
        Movie = 3, // AAUDIO_CONTENT_TYPE_MOVIE

        /**
         * Use this for sound is designed to accompany a user action,
         * such as a click or beep sound made when the user presses a button.
         */
        Sonification = 4, // AAUDIO_CONTENT_TYPE_SONIFICATION
    };

    /**
     * Defines the audio source.
     * An audio source defines both a default physical source of audio signal, and a recording
     * configuration.
     *
     * Note that these match the equivalent values in MediaRecorder.AudioSource in the Android Java API.
     *
     * This attribute only has an effect on Android API 28+.
     */
    enum InputPreset : int32_t { // aaudio_input_preset_t
        /**
         * Use this preset when other presets do not apply.
         */
        Generic = 1, // AAUDIO_INPUT_PRESET_GENERIC

        /**
         * Use this preset when recording video.
         */
        Camcorder = 5, // AAUDIO_INPUT_PRESET_CAMCORDER

        /**
         * Use this preset when doing speech recognition.
         */
        VoiceRecognition = 6, // AAUDIO_INPUT_PRESET_VOICE_RECOGNITION

        /**
         * Use this preset when doing telephony or voice messaging.
         */
        VoiceCommunication = 7, // AAUDIO_INPUT_PRESET_VOICE_COMMUNICATION

        /**
         * Use this preset to obtain an input with no effects.
         * Note that this input will not have automatic gain control
         * so the recorded volume may be very low.
         */
        Unprocessed = 9, // AAUDIO_INPUT_PRESET_UNPROCESSED

        /**
         * Use this preset for capturing audio meant to be processed in real time
         * and played back for live performance (e.g karaoke).
         * The capture path will minimize latency and coupling with playback path.
         */
         VoicePerformance = 10, // AAUDIO_INPUT_PRESET_VOICE_PERFORMANCE

    };

    /**
     * This attribute can be used to allocate a session ID to the audio stream.
     *
     * This attribute only has an effect on Android API 28+.
     */
    enum SessionId {
        /**
         * Do not allocate a session ID.
         * Effects cannot be used with this stream.
         * Default.
         */
         None = -1, // AAUDIO_SESSION_ID_NONE

        /**
         * Allocate a session ID that can be used to attach and control
         * effects using the Java AudioEffects API.
         * Note that the use of this flag may result in higher latency.
         *
         * Note that this matches the value of AudioManager.AUDIO_SESSION_ID_GENERATE.
         */
         Allocate = 0, // AAUDIO_SESSION_ID_ALLOCATE
    };

    /**
     * The channel count of the audio stream. The underlying type is `int32_t`.
     * Use of this enum is convenient to avoid "magic"
     * numbers when specifying the channel count.
     *
     * For example, you can write
     * `builder.setChannelCount(ChannelCount::Stereo)`
     * rather than `builder.setChannelCount(2)`
     *
     */
    enum ChannelCount : int32_t {
      /**
       * Audio channel count definition, use Mono or Stereo
       */
      Unspecified = kUnspecified,

      /**
       * Use this for mono audio
       */
      Mono = 1,

      /**
       * Use this for stereo audio.
       */
      Stereo = 2,
    };

    /**
     * The channel mask of the audio stream. The underlying type is `uint32_t`.
     * Use of this enum is convenient.
     *
     * ChannelMask::Unspecified means this is not specified.
     * The rest of the enums are channel position masks.
     * Use the combinations of the channel position masks defined below instead of
     * using those values directly.
     *
     * Channel masks are for input only, output only, or both input and output.
     * These channel masks are different than those defined in AudioFormat.java.
     * If an app gets a channel mask from Java API and wants to use it in Oboe,
     * conversion should be done by the app.
     */
    enum class ChannelMask : uint32_t { // aaudio_channel_mask_t
        Unspecified = kUnspecified,
        FrontLeft = 1 << 0,
        FrontRight = 1 << 1,
        FrontCenter = 1 << 2,
        LowFrequency = 1 << 3,
        BackLeft = 1 << 4,
        BackRight = 1 << 5,
        FrontLeftOfCenter = 1 << 6,
        FrontRightOfCenter = 1 << 7,
        BackCenter = 1 << 8,
        SideLeft = 1 << 9,
        SideRight = 1 << 10,
        TopCenter = 1 << 11,
        TopFrontLeft = 1 << 12,
        TopFrontCenter = 1 << 13,
        TopFrontRight = 1 << 14,
        TopBackLeft = 1 << 15,
        TopBackCenter = 1 << 16,
        TopBackRight = 1 << 17,
        TopSideLeft = 1 << 18,
        TopSideRight = 1 << 19,
        BottomFrontLeft = 1 << 20,
        BottomFrontCenter = 1 << 21,
        BottomFrontRight = 1 << 22,
        LowFrequency2 = 1 << 23,
        FrontWideLeft = 1 << 24,
        FrontWideRight = 1 << 25,

        /**
         * Supported for Input and Output
         */
        Mono = FrontLeft,

        /**
         * Supported for Input and Output
         */
        Stereo = FrontLeft |
                 FrontRight,

        /**
         * Supported for only Output
         */
        CM2Point1 = FrontLeft |
                    FrontRight |
                    LowFrequency,

        /**
         * Supported for only Output
         */
        Tri = FrontLeft |
              FrontRight |
              FrontCenter,

        /**
         * Supported for only Output
         */
        TriBack = FrontLeft |
                  FrontRight |
                  BackCenter,

        /**
         * Supported for only Output
         */
        CM3Point1 = FrontLeft |
                    FrontRight |
                    FrontCenter |
                    LowFrequency,

        /**
         * Supported for Input and Output
         */
        CM2Point0Point2 = FrontLeft |
                          FrontRight |
                          TopSideLeft |
                          TopSideRight,

        /**
         * Supported for Input and Output
         */
        CM2Point1Point2 = CM2Point0Point2 |
                          LowFrequency,

        /**
         * Supported for Input and Output
         */
        CM3Point0Point2 = FrontLeft |
                          FrontRight |
                          FrontCenter |
                          TopSideLeft |
                          TopSideRight,

        /**
         * Supported for Input and Output
         */
        CM3Point1Point2 = CM3Point0Point2 |
                          LowFrequency,

        /**
         * Supported for only Output
         */
        Quad = FrontLeft |
               FrontRight |
               BackLeft |
               BackRight,

        /**
         * Supported for only Output
         */
        QuadSide = FrontLeft |
                   FrontRight |
                   SideLeft |
                   SideRight,

        /**
         * Supported for only Output
         */
        Surround = FrontLeft |
                   FrontRight |
                   FrontCenter |
                   BackCenter,

        /**
         * Supported for only Output
         */
        Penta = Quad |
                FrontCenter,

        /**
         * Supported for Input and Output. aka 5Point1Back
         */
        CM5Point1 = FrontLeft |
                    FrontRight |
                    FrontCenter |
                    LowFrequency |
                    BackLeft |
                    BackRight,

        /**
         * Supported for only Output
         */
        CM5Point1Side = FrontLeft |
                        FrontRight |
                        FrontCenter |
                        LowFrequency |
                        SideLeft |
                        SideRight,

        /**
         * Supported for only Output
         */
        CM6Point1 = FrontLeft |
                    FrontRight |
                    FrontCenter |
                    LowFrequency |
                    BackLeft |
                    BackRight |
                    BackCenter,

        /**
         * Supported for only Output
         */
        CM7Point1 = CM5Point1 |
                    SideLeft |
                    SideRight,

        /**
         * Supported for only Output
         */
        CM5Point1Point2 = CM5Point1 |
                          TopSideLeft |
                          TopSideRight,

        /**
         * Supported for only Output
         */
        CM5Point1Point4 = CM5Point1 |
                          TopFrontLeft |
                          TopFrontRight |
                          TopBackLeft |
                          TopBackRight,

        /**
         * Supported for only Output
         */
        CM7Point1Point2 = CM7Point1 |
                          TopSideLeft |
                          TopSideRight,

        /**
         * Supported for only Output
         */
        CM7Point1Point4 = CM7Point1 |
                          TopFrontLeft |
                          TopFrontRight |
                          TopBackLeft |
                          TopBackRight,

        /**
         * Supported for only Output
         */
        CM9Point1Point4 = CM7Point1Point4 |
                          FrontWideLeft |
                          FrontWideRight,

        /**
         * Supported for only Output
         */
        CM9Point1Point6 = CM9Point1Point4 |
                          TopSideLeft |
                          TopSideRight,

        /**
         * Supported for only Input
         */
        FrontBack = FrontCenter |
                    BackCenter,
    };

    /**
     * The spatialization behavior of the audio stream.
     */
    enum class SpatializationBehavior : int32_t {

        /**
         * Constant indicating that the spatialization behavior is not specified.
         */
        Unspecified = kUnspecified,

        /**
         * Constant indicating the audio content associated with these attributes will follow the
         * default platform behavior with regards to which content will be spatialized or not.
         */
        Auto = 1,

        /**
         * Constant indicating the audio content associated with these attributes should never
         * be spatialized.
         */
        Never = 2,
    };

    /**
     * The PrivacySensitiveMode attribute determines whether an input stream can be shared
     * with another privileged app, for example the Assistant.
     *
     * This allows to override the default behavior tied to the audio source (e.g
     * InputPreset::VoiceCommunication is private by default but InputPreset::Unprocessed is not).
     */
    enum class PrivacySensitiveMode : int32_t {

        /**
         * When not explicitly requested, set privacy sensitive mode according to input preset:
         * communication and camcorder captures are considered privacy sensitive by default.
         */
        Unspecified = kUnspecified,

        /**
         * Privacy sensitive mode disabled.
         */
        Disabled = 1,

        /**
         * Privacy sensitive mode enabled.
         */
        Enabled = 2,
    };

    /**
     * Specifies whether audio may or may not be captured by other apps or the system for an
     * output stream.
     *
     * Note that these match the equivalent values in AudioAttributes in the Android Java API.
     *
     * Added in API level 29 for AAudio.
     */
    enum class AllowedCapturePolicy : int32_t {
        /**
         * When not explicitly requested, set privacy sensitive mode according to the Usage.
         * This should behave similarly to setting AllowedCapturePolicy::All.
         */
        Unspecified = kUnspecified,
        /**
         * Indicates that the audio may be captured by any app.
         *
         * For privacy, the following Usages can not be recorded: VoiceCommunication*,
         * Notification*, Assistance* and Assistant.
         *
         * On Android Q, only Usage::Game and Usage::Media may be captured.
         *
         * See ALLOW_CAPTURE_BY_ALL in the AudioAttributes Java API.
         */
        All = 1,
        /**
         * Indicates that the audio may only be captured by system apps.
         *
         * System apps can capture for many purposes like accessibility, user guidance...
         * but have strong restriction. See ALLOW_CAPTURE_BY_SYSTEM in the AudioAttributes Java API
         * for what the system apps can do with the capture audio.
         */
        System = 2,
        /**
         * Indicates that the audio may not be recorded by any app, even if it is a system app.
         *
         * It is encouraged to use AllowedCapturePolicy::System instead of this value as system apps
         * provide significant and useful features for the user (eg. accessibility).
         * See ALLOW_CAPTURE_BY_NONE in the AudioAttributes Java API
         */
        None = 3,
    };

    /**
     * On API 16 to 26 OpenSL ES will be used. When using OpenSL ES the optimal values for sampleRate and
     * framesPerBurst are not known by the native code.
     * On API 17+ these values should be obtained from the AudioManager using this code:
     *
     * <pre><code>
     * // Note that this technique only works for built-in speakers and headphones.
     * AudioManager myAudioMgr = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
     * String sampleRateStr = myAudioMgr.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE);
     * int defaultSampleRate = Integer.parseInt(sampleRateStr);
     * String framesPerBurstStr = myAudioMgr.getProperty(AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER);
     * int defaultFramesPerBurst = Integer.parseInt(framesPerBurstStr);
     * </code></pre>
     *
     * It can then be passed down to Oboe through JNI.
     *
     * AAudio will get the optimal framesPerBurst from the HAL and will ignore this value.
     */
    class DefaultStreamValues {

    public:

        /** The default sample rate to use when opening new audio streams */
        static int32_t SampleRate;
        /** The default frames per burst to use when opening new audio streams */
        static int32_t FramesPerBurst;
        /** The default channel count to use when opening new audio streams */
        static int32_t ChannelCount;

    };

    /**
     * The time at which the frame at `position` was presented
     */
    struct FrameTimestamp {
        int64_t position; // in frames
        int64_t timestamp; // in nanoseconds
    };

    class OboeGlobals {
    public:

        static bool areWorkaroundsEnabled() {
            return mWorkaroundsEnabled;
        }

        /**
         * Disable this when writing tests to reproduce bugs in AAudio or OpenSL ES
         * that have workarounds in Oboe.
         * @param enabled
         */
        static void setWorkaroundsEnabled(bool enabled) {
            mWorkaroundsEnabled = enabled;
        }

    private:
        static bool mWorkaroundsEnabled;
    };
} // namespace oboe

#endif // OBOE_DEFINITIONS_H
