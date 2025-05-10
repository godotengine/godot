/*
  Simple DirectMedia Layer
  Copyright , (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#define SDL_PROC_UNUSED(ret, func, params)

SDL_PROC(const char *, AAudio_convertResultToText, (aaudio_result_t returnCode))
SDL_PROC(const char *, AAudio_convertStreamStateToText, (aaudio_stream_state_t state))
SDL_PROC(aaudio_result_t, AAudio_createStreamBuilder, (AAudioStreamBuilder * *builder))
SDL_PROC(void, AAudioStreamBuilder_setDeviceId, (AAudioStreamBuilder * builder, int32_t deviceId))
SDL_PROC(void, AAudioStreamBuilder_setSampleRate, (AAudioStreamBuilder * builder, int32_t sampleRate))
SDL_PROC(void, AAudioStreamBuilder_setChannelCount, (AAudioStreamBuilder * builder, int32_t channelCount))
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setSamplesPerFrame, (AAudioStreamBuilder * builder, int32_t samplesPerFrame))
SDL_PROC(void, AAudioStreamBuilder_setFormat, (AAudioStreamBuilder * builder, aaudio_format_t format))
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setSharingMode, (AAudioStreamBuilder * builder, aaudio_sharing_mode_t sharingMode))
SDL_PROC(void, AAudioStreamBuilder_setDirection, (AAudioStreamBuilder * builder, aaudio_direction_t direction))
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setBufferCapacityInFrames, (AAudioStreamBuilder * builder, int32_t numFrames))
SDL_PROC(void, AAudioStreamBuilder_setPerformanceMode, (AAudioStreamBuilder * builder, aaudio_performance_mode_t mode))
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setUsage, (AAudioStreamBuilder * builder, aaudio_usage_t usage))                                         // API 28
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setContentType, (AAudioStreamBuilder * builder, aaudio_content_type_t contentType))                      // API 28
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setInputPreset, (AAudioStreamBuilder * builder, aaudio_input_preset_t inputPreset))                      // API 28
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setAllowedCapturePolicy, (AAudioStreamBuilder * builder, aaudio_allowed_capture_policy_t capturePolicy)) // API 29
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setSessionId, (AAudioStreamBuilder * builder, aaudio_session_id_t sessionId))                            // API 28
SDL_PROC_UNUSED(void, AAudioStreamBuilder_setPrivacySensitive, (AAudioStreamBuilder * builder, bool privacySensitive))                             // API 30
SDL_PROC(void, AAudioStreamBuilder_setDataCallback, (AAudioStreamBuilder * builder, AAudioStream_dataCallback callback, void *userData))
SDL_PROC(void, AAudioStreamBuilder_setFramesPerDataCallback, (AAudioStreamBuilder * builder, int32_t numFrames))
SDL_PROC(void, AAudioStreamBuilder_setErrorCallback, (AAudioStreamBuilder * builder, AAudioStream_errorCallback callback, void *userData))
SDL_PROC(aaudio_result_t, AAudioStreamBuilder_openStream, (AAudioStreamBuilder * builder, AAudioStream **stream))
SDL_PROC(aaudio_result_t, AAudioStreamBuilder_delete, (AAudioStreamBuilder * builder))
SDL_PROC_UNUSED(aaudio_result_t, AAudioStream_release, (AAudioStream * stream)) // API 30
SDL_PROC(aaudio_result_t, AAudioStream_close, (AAudioStream * stream))
SDL_PROC(aaudio_result_t, AAudioStream_requestStart, (AAudioStream * stream))
SDL_PROC(aaudio_result_t, AAudioStream_requestPause, (AAudioStream * stream))
SDL_PROC_UNUSED(aaudio_result_t, AAudioStream_requestFlush, (AAudioStream * stream))
SDL_PROC(aaudio_result_t, AAudioStream_requestStop, (AAudioStream * stream))
SDL_PROC(aaudio_stream_state_t, AAudioStream_getState, (AAudioStream * stream))
SDL_PROC_UNUSED(aaudio_result_t, AAudioStream_waitForStateChange, (AAudioStream * stream, aaudio_stream_state_t inputState, aaudio_stream_state_t *nextState, int64_t timeoutNanoseconds))
SDL_PROC_UNUSED(aaudio_result_t, AAudioStream_read, (AAudioStream * stream, void *buffer, int32_t numFrames, int64_t timeoutNanoseconds))
SDL_PROC_UNUSED(aaudio_result_t, AAudioStream_write, (AAudioStream * stream, const void *buffer, int32_t numFrames, int64_t timeoutNanoseconds))
SDL_PROC_UNUSED(aaudio_result_t, AAudioStream_setBufferSizeInFrames, (AAudioStream * stream, int32_t numFrames))
SDL_PROC_UNUSED(int32_t, AAudioStream_getBufferSizeInFrames, (AAudioStream * stream))
SDL_PROC_UNUSED(int32_t, AAudioStream_getFramesPerBurst, (AAudioStream * stream))
SDL_PROC(int32_t, AAudioStream_getBufferCapacityInFrames, (AAudioStream * stream))
SDL_PROC(int32_t, AAudioStream_getFramesPerDataCallback, (AAudioStream * stream))
SDL_PROC_UNUSED(int32_t, AAudioStream_getXRunCount, (AAudioStream * stream))
SDL_PROC(int32_t, AAudioStream_getSampleRate, (AAudioStream * stream))
SDL_PROC(int32_t, AAudioStream_getChannelCount, (AAudioStream * stream))
SDL_PROC_UNUSED(int32_t, AAudioStream_getSamplesPerFrame, (AAudioStream * stream))
SDL_PROC_UNUSED(int32_t, AAudioStream_getDeviceId, (AAudioStream * stream))
SDL_PROC(aaudio_format_t, AAudioStream_getFormat, (AAudioStream * stream))
SDL_PROC_UNUSED(aaudio_sharing_mode_t, AAudioStream_getSharingMode, (AAudioStream * stream))
SDL_PROC_UNUSED(aaudio_performance_mode_t, AAudioStream_getPerformanceMode, (AAudioStream * stream))
SDL_PROC_UNUSED(aaudio_direction_t, AAudioStream_getDirection, (AAudioStream * stream))
SDL_PROC_UNUSED(int64_t, AAudioStream_getFramesWritten, (AAudioStream * stream))
SDL_PROC_UNUSED(int64_t, AAudioStream_getFramesRead, (AAudioStream * stream))
SDL_PROC_UNUSED(aaudio_session_id_t, AAudioStream_getSessionId, (AAudioStream * stream)) // API 28
SDL_PROC(aaudio_result_t, AAudioStream_getTimestamp, (AAudioStream * stream, clockid_t clockid, int64_t *framePosition, int64_t *timeNanoseconds))
SDL_PROC_UNUSED(aaudio_usage_t, AAudioStream_getUsage, (AAudioStream * stream))                                 // API 28
SDL_PROC_UNUSED(aaudio_content_type_t, AAudioStream_getContentType, (AAudioStream * stream))                    // API 28
SDL_PROC_UNUSED(aaudio_input_preset_t, AAudioStream_getInputPreset, (AAudioStream * stream))                    // API 28
SDL_PROC_UNUSED(aaudio_allowed_capture_policy_t, AAudioStream_getAllowedCapturePolicy, (AAudioStream * stream)) // API 29
SDL_PROC_UNUSED(bool, AAudioStream_isPrivacySensitive, (AAudioStream * stream))                                 // API 30

#undef SDL_PROC
#undef SDL_PROC_UNUSED
