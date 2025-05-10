/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

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
#include "SDL_internal.h"

#ifdef SDL_AUDIO_DRIVER_AAUDIO

#include "../SDL_sysaudio.h"
#include "SDL_aaudio.h"

#include "../../core/android/SDL_android.h"
#include <aaudio/AAudio.h>

#if __ANDROID_API__ < 31
#define AAUDIO_FORMAT_PCM_I32 4
#endif

struct SDL_PrivateAudioData
{
    AAudioStream *stream;
    int num_buffers;
    Uint8 *mixbuf;          // Raw mixing buffer
    size_t mixbuf_bytes;    // num_buffers * device->buffer_size
    size_t callback_bytes;
    size_t processed_bytes;
    SDL_Semaphore *semaphore;
    SDL_AtomicInt error_callback_triggered;
};

// Debug
#if 0
#define LOGI(...) SDL_Log(__VA_ARGS__);
#else
#define LOGI(...)
#endif

#define LIB_AAUDIO_SO "libaaudio.so"

typedef struct AAUDIO_Data
{
    SDL_SharedObject *handle;
#define SDL_PROC(ret, func, params) ret (*func) params;
#include "SDL_aaudiofuncs.h"
} AAUDIO_Data;
static AAUDIO_Data ctx;

static bool AAUDIO_LoadFunctions(AAUDIO_Data *data)
{
#define SDL_PROC(ret, func, params)                                                             \
    do {                                                                                        \
        data->func = (ret (*) params)SDL_LoadFunction(data->handle, #func);                                     \
        if (!data->func) {                                                                      \
            return SDL_SetError("Couldn't load AAUDIO function %s: %s", #func, SDL_GetError()); \
        }                                                                                       \
    } while (0);
#include "SDL_aaudiofuncs.h"
    return true;
}


static void AAUDIO_errorCallback(AAudioStream *stream, void *userData, aaudio_result_t error)
{
    LOGI("SDL AAUDIO_errorCallback: %d - %s", error, ctx.AAudio_convertResultToText(error));

    // You MUST NOT close the audio stream from this callback, so we cannot call SDL_AudioDeviceDisconnected here.
    // Just flag the device so we can kill it in PlayDevice instead.
    SDL_AudioDevice *device = (SDL_AudioDevice *) userData;
    SDL_SetAtomicInt(&device->hidden->error_callback_triggered, (int) error);  // AAUDIO_OK is zero, so !triggered means no error.
    SDL_SignalSemaphore(device->hidden->semaphore);  // in case we're blocking in WaitDevice.
}

static aaudio_data_callback_result_t AAUDIO_dataCallback(AAudioStream *stream, void *userData, void *audioData, int32_t numFrames)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *) userData;
    struct SDL_PrivateAudioData *hidden = device->hidden;
    size_t framesize = SDL_AUDIO_FRAMESIZE(device->spec);
    size_t callback_bytes = numFrames * framesize;
    size_t old_buffer_index = hidden->callback_bytes / device->buffer_size;

    if (device->recording) {
        const Uint8 *input = (const Uint8 *)audioData;
        size_t available_bytes = hidden->mixbuf_bytes - (hidden->callback_bytes - hidden->processed_bytes);
        size_t size = SDL_min(available_bytes, callback_bytes);
        size_t offset = hidden->callback_bytes % hidden->mixbuf_bytes;
        size_t end = (offset + size) % hidden->mixbuf_bytes;
        SDL_assert(size <= hidden->mixbuf_bytes);

//LOGI("Recorded %zu frames, %zu available, %zu max (%zu written, %zu read)", callback_bytes / framesize, available_bytes / framesize, hidden->mixbuf_bytes / framesize, hidden->callback_bytes / framesize, hidden->processed_bytes / framesize);

        if (offset <= end) {
            SDL_memcpy(&hidden->mixbuf[offset], input, size);
        } else {
            size_t partial = (hidden->mixbuf_bytes - offset);
            SDL_memcpy(&hidden->mixbuf[offset], &input[0], partial);
            SDL_memcpy(&hidden->mixbuf[0], &input[partial], end);
        }

        SDL_MemoryBarrierRelease();
        hidden->callback_bytes += size;

        if (size < callback_bytes) {
            LOGI("Audio recording overflow, dropped %zu frames", (callback_bytes - size) / framesize);
        }
    } else {
        Uint8 *output = (Uint8 *)audioData;
        size_t available_bytes = (hidden->processed_bytes - hidden->callback_bytes);
        size_t size = SDL_min(available_bytes, callback_bytes);
        size_t offset = hidden->callback_bytes % hidden->mixbuf_bytes;
        size_t end = (offset + size) % hidden->mixbuf_bytes;
        SDL_assert(size <= hidden->mixbuf_bytes);

//LOGI("Playing %zu frames, %zu available, %zu max (%zu written, %zu read)", callback_bytes / framesize, available_bytes / framesize, hidden->mixbuf_bytes / framesize, hidden->processed_bytes / framesize, hidden->callback_bytes / framesize);

        SDL_MemoryBarrierAcquire();
        if (offset <= end) {
            SDL_memcpy(output, &hidden->mixbuf[offset], size);
        } else {
            size_t partial = (hidden->mixbuf_bytes - offset);
            SDL_memcpy(&output[0], &hidden->mixbuf[offset], partial);
            SDL_memcpy(&output[partial], &hidden->mixbuf[0], end);
        }
        hidden->callback_bytes += size;

        if (size < callback_bytes) {
            LOGI("Audio playback underflow, missed %zu frames", (callback_bytes - size) / framesize);
            SDL_memset(&output[size], device->silence_value, (callback_bytes - size));
        }
    }

    size_t new_buffer_index = hidden->callback_bytes / device->buffer_size;
    while (old_buffer_index < new_buffer_index) {
        // Trigger audio processing
        SDL_SignalSemaphore(hidden->semaphore);
        ++old_buffer_index;
    }

    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

static Uint8 *AAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *bufsize)
{
    struct SDL_PrivateAudioData *hidden = device->hidden;
    size_t offset = (hidden->processed_bytes % hidden->mixbuf_bytes);
    return &hidden->mixbuf[offset];
}

static bool AAUDIO_WaitDevice(SDL_AudioDevice *device)
{
    while (!SDL_GetAtomicInt(&device->shutdown)) {
        // this semaphore won't fire when the app is in the background (AAUDIO_PauseDevices was called).
        if (SDL_WaitSemaphoreTimeout(device->hidden->semaphore, 100)) {
            return true;  // semaphore was signaled, let's go!
        }
        // Still waiting on the semaphore (or the system), check other things then wait again.
    }
    return true;
}

static bool BuildAAudioStream(SDL_AudioDevice *device);

static bool RecoverAAudioDevice(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *hidden = device->hidden;

    // attempt to build a new stream, in case there's a new default device.
    ctx.AAudioStream_requestStop(hidden->stream);
    ctx.AAudioStream_close(hidden->stream);
    hidden->stream = NULL;

    SDL_aligned_free(hidden->mixbuf);
    hidden->mixbuf = NULL;

    SDL_DestroySemaphore(hidden->semaphore);
    hidden->semaphore = NULL;

    const int prev_sample_frames = device->sample_frames;
    SDL_AudioSpec prevspec;
    SDL_copyp(&prevspec, &device->spec);

    if (!BuildAAudioStream(device)) {
        return false;  // oh well, we tried.
    }

    // we don't know the new device spec until we open the new device, so we saved off the old one and force it back
    // so SDL_AudioDeviceFormatChanged can set up all the important state if necessary and then set it back to the new spec.
    const int new_sample_frames = device->sample_frames;
    SDL_AudioSpec newspec;
    SDL_copyp(&newspec, &device->spec);

    device->sample_frames = prev_sample_frames;
    SDL_copyp(&device->spec, &prevspec);
    if (!SDL_AudioDeviceFormatChangedAlreadyLocked(device, &newspec, new_sample_frames)) {
        return false;  // ugh
    }
    return true;
}


static bool AAUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    struct SDL_PrivateAudioData *hidden = device->hidden;

    // AAUDIO_dataCallback picks up our work and unblocks AAUDIO_WaitDevice. But make sure we didn't fail here.
    const aaudio_result_t err = (aaudio_result_t) SDL_GetAtomicInt(&hidden->error_callback_triggered);
    if (err) {
        SDL_LogError(SDL_LOG_CATEGORY_AUDIO, "aaudio: Audio device triggered error %d (%s)", (int) err, ctx.AAudio_convertResultToText(err));

        if (!RecoverAAudioDevice(device)) {
            return false;  // oh well, we went down hard.
        }
    } else {
        SDL_MemoryBarrierRelease();
        hidden->processed_bytes += buflen;
    }
    return true;
}

static int AAUDIO_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    struct SDL_PrivateAudioData *hidden = device->hidden;

    // AAUDIO_dataCallback picks up our work and unblocks AAUDIO_WaitDevice. But make sure we didn't fail here.
    if (SDL_GetAtomicInt(&hidden->error_callback_triggered)) {
        SDL_SetAtomicInt(&hidden->error_callback_triggered, 0);
        return -1;
    }

    SDL_assert(buflen == device->buffer_size);  // If this isn't true, we need to change semaphore trigger logic and account for wrapping copies here
    size_t offset = (hidden->processed_bytes % hidden->mixbuf_bytes);
    SDL_MemoryBarrierAcquire();
    SDL_memcpy(buffer, &hidden->mixbuf[offset], buflen);
    hidden->processed_bytes += buflen;
    return buflen;
}

static void AAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *hidden = device->hidden;
    LOGI(__func__);

    if (hidden) {
        if (hidden->stream) {
            ctx.AAudioStream_requestStop(hidden->stream);
            // !!! FIXME: do we have to wait for the state to change to make sure all buffered audio has played, or will close do this (or will the system do this after the close)?
            // !!! FIXME: also, will this definitely wait for a running data callback to finish, and then stop the callback from firing again?
            ctx.AAudioStream_close(hidden->stream);
        }

        if (hidden->semaphore) {
            SDL_DestroySemaphore(hidden->semaphore);
        }

        SDL_aligned_free(hidden->mixbuf);
        SDL_free(hidden);
        device->hidden = NULL;
    }
}

static bool BuildAAudioStream(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *hidden = device->hidden;
    const bool recording = device->recording;
    aaudio_result_t res;

    SDL_SetAtomicInt(&hidden->error_callback_triggered, 0);

    AAudioStreamBuilder *builder = NULL;
    res = ctx.AAudio_createStreamBuilder(&builder);
    if (res != AAUDIO_OK) {
        LOGI("SDL Failed AAudio_createStreamBuilder %d", res);
        return SDL_SetError("SDL Failed AAudio_createStreamBuilder %d", res);
    } else if (!builder) {
        LOGI("SDL Failed AAudio_createStreamBuilder - builder NULL");
        return SDL_SetError("SDL Failed AAudio_createStreamBuilder - builder NULL");
    }

#if ALLOW_MULTIPLE_ANDROID_AUDIO_DEVICES
    const int aaudio_device_id = (int) ((size_t) device->handle);
    LOGI("Opening device id %d", aaudio_device_id);
    ctx.AAudioStreamBuilder_setDeviceId(builder, aaudio_device_id);
#endif

    aaudio_format_t format;
    if ((device->spec.format == SDL_AUDIO_S32) && (SDL_GetAndroidSDKVersion() >= 31)) {
        format = AAUDIO_FORMAT_PCM_I32;
    } else if (device->spec.format == SDL_AUDIO_F32) {
        format = AAUDIO_FORMAT_PCM_FLOAT;
    } else {
        format = AAUDIO_FORMAT_PCM_I16;  // sint16 is a safe bet for everything else.
    }
    ctx.AAudioStreamBuilder_setFormat(builder, format);
    ctx.AAudioStreamBuilder_setSampleRate(builder, device->spec.freq);
    ctx.AAudioStreamBuilder_setChannelCount(builder, device->spec.channels);

    const aaudio_direction_t direction = (recording ? AAUDIO_DIRECTION_INPUT : AAUDIO_DIRECTION_OUTPUT);
    ctx.AAudioStreamBuilder_setDirection(builder, direction);
    ctx.AAudioStreamBuilder_setErrorCallback(builder, AAUDIO_errorCallback, device);
    ctx.AAudioStreamBuilder_setDataCallback(builder, AAUDIO_dataCallback, device);
    // Some devices have flat sounding audio when low latency mode is enabled, but this is a better experience for most people
    if (SDL_GetHintBoolean(SDL_HINT_ANDROID_LOW_LATENCY_AUDIO, true)) {
        SDL_Log("Low latency audio enabled");
        ctx.AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    } else {
        SDL_Log("Low latency audio disabled");
    }

    LOGI("AAudio Try to open %u hz %s %u channels samples %u",
         device->spec.freq, SDL_GetAudioFormatName(device->spec.format),
         device->spec.channels, device->sample_frames);

    res = ctx.AAudioStreamBuilder_openStream(builder, &hidden->stream);
    if (res != AAUDIO_OK) {
        LOGI("SDL Failed AAudioStreamBuilder_openStream %d", res);
        ctx.AAudioStreamBuilder_delete(builder);
        return SDL_SetError("%s : %s", __func__, ctx.AAudio_convertResultToText(res));
    }
    ctx.AAudioStreamBuilder_delete(builder);

    device->sample_frames = (int)ctx.AAudioStream_getFramesPerDataCallback(hidden->stream);
    if (device->sample_frames == AAUDIO_UNSPECIFIED) {
        // We'll get variable frames in the callback, make sure we have at least half a buffer available
        device->sample_frames = (int)ctx.AAudioStream_getBufferCapacityInFrames(hidden->stream) / 2;
    }

    device->spec.freq = ctx.AAudioStream_getSampleRate(hidden->stream);
    device->spec.channels = ctx.AAudioStream_getChannelCount(hidden->stream);

    format = ctx.AAudioStream_getFormat(hidden->stream);
    if (format == AAUDIO_FORMAT_PCM_I16) {
        device->spec.format = SDL_AUDIO_S16;
    } else if (format == AAUDIO_FORMAT_PCM_I32) {
        device->spec.format = SDL_AUDIO_S32;
    } else if (format == AAUDIO_FORMAT_PCM_FLOAT) {
        device->spec.format = SDL_AUDIO_F32;
    } else {
        return SDL_SetError("Got unexpected audio format %d from AAudioStream_getFormat", (int) format);
    }

    SDL_UpdatedAudioDeviceFormat(device);

    // Allocate a triple buffered mixing buffer
    // Two buffers can be in the process of being filled while the third is being read
    hidden->num_buffers = 3;
    hidden->mixbuf_bytes = (hidden->num_buffers * device->buffer_size);
    hidden->mixbuf = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), hidden->mixbuf_bytes);
    if (!hidden->mixbuf) {
        return false;
    }
    hidden->processed_bytes = 0;
    hidden->callback_bytes = 0;

    hidden->semaphore = SDL_CreateSemaphore(recording ? 0 : hidden->num_buffers - 1);
    if (!hidden->semaphore) {
        LOGI("SDL Failed SDL_CreateSemaphore %s recording:%d", SDL_GetError(), recording);
        return false;
    }

    LOGI("AAudio Actually opened %u hz %s %u channels samples %u, buffers %d",
         device->spec.freq, SDL_GetAudioFormatName(device->spec.format),
         device->spec.channels, device->sample_frames, hidden->num_buffers);

    res = ctx.AAudioStream_requestStart(hidden->stream);
    if (res != AAUDIO_OK) {
        LOGI("SDL Failed AAudioStream_requestStart %d recording:%d", res, recording);
        return SDL_SetError("%s : %s", __func__, ctx.AAudio_convertResultToText(res));
    }

    LOGI("SDL AAudioStream_requestStart OK");

    return true;
}

// !!! FIXME: make this non-blocking!
static void SDLCALL RequestAndroidPermissionBlockingCallback(void *userdata, const char *permission, bool granted)
{
    SDL_SetAtomicInt((SDL_AtomicInt *) userdata, granted ? 1 : -1);
}

static bool AAUDIO_OpenDevice(SDL_AudioDevice *device)
{
#if ALLOW_MULTIPLE_ANDROID_AUDIO_DEVICES
    SDL_assert(device->handle);  // AAUDIO_UNSPECIFIED is zero, so legit devices should all be non-zero.
#endif

    LOGI(__func__);

    if (device->recording) {
        // !!! FIXME: make this non-blocking!
        SDL_AtomicInt permission_response;
        SDL_SetAtomicInt(&permission_response, 0);
        if (!SDL_RequestAndroidPermission("android.permission.RECORD_AUDIO", RequestAndroidPermissionBlockingCallback, &permission_response)) {
            return false;
        }

        while (SDL_GetAtomicInt(&permission_response) == 0) {
            SDL_Delay(10);
        }

        if (SDL_GetAtomicInt(&permission_response) < 0) {
            LOGI("This app doesn't have RECORD_AUDIO permission");
            return SDL_SetError("This app doesn't have RECORD_AUDIO permission");
        }
    }

    device->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    return BuildAAudioStream(device);
}

static bool PauseOneDevice(SDL_AudioDevice *device, void *userdata)
{
    struct SDL_PrivateAudioData *hidden = (struct SDL_PrivateAudioData *)device->hidden;
    if (hidden) {
        if (hidden->stream) {
            aaudio_result_t res;

            if (device->recording) {
                // Pause() isn't implemented for recording, use Stop()
                res = ctx.AAudioStream_requestStop(hidden->stream);
            } else {
                res = ctx.AAudioStream_requestPause(hidden->stream);
            }

            if (res != AAUDIO_OK) {
                LOGI("SDL Failed AAudioStream_requestPause %d", res);
                SDL_SetError("%s : %s", __func__, ctx.AAudio_convertResultToText(res));
            }
        }
    }
    return false;  // keep enumerating.
}

// Pause (block) all non already paused audio devices by taking their mixer lock
void AAUDIO_PauseDevices(void)
{
    if (ctx.handle) {  // AAUDIO driver is used?
        (void) SDL_FindPhysicalAudioDeviceByCallback(PauseOneDevice, NULL);
    }
}

// Resume (unblock) all non already paused audio devices by releasing their mixer lock
static bool ResumeOneDevice(SDL_AudioDevice *device, void *userdata)
{
    struct SDL_PrivateAudioData *hidden = device->hidden;
    if (hidden) {
        if (hidden->stream) {
            aaudio_result_t res = ctx.AAudioStream_requestStart(hidden->stream);
            if (res != AAUDIO_OK) {
                LOGI("SDL Failed AAudioStream_requestStart %d", res);
                SDL_SetError("%s : %s", __func__, ctx.AAudio_convertResultToText(res));
            }
        }
    }
    return false;  // keep enumerating.
}

void AAUDIO_ResumeDevices(void)
{
    if (ctx.handle) {  // AAUDIO driver is used?
        (void) SDL_FindPhysicalAudioDeviceByCallback(ResumeOneDevice, NULL);
    }
}

static void AAUDIO_Deinitialize(void)
{
    Android_StopAudioHotplug();

    LOGI(__func__);
    if (ctx.handle) {
        SDL_UnloadObject(ctx.handle);
    }
    SDL_zero(ctx);
    LOGI("End AAUDIO %s", SDL_GetError());
}


static bool AAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    LOGI(__func__);

    /* AAudio was introduced in Android 8.0, but has reference counting crash issues in that release,
     * so don't use it until 8.1.
     *
     * See https://github.com/google/oboe/issues/40 for more information.
     */
    if (SDL_GetAndroidSDKVersion() < 27) {
        return false;
    }

    SDL_zero(ctx);

    ctx.handle = SDL_LoadObject(LIB_AAUDIO_SO);
    if (!ctx.handle) {
        LOGI("SDL couldn't find " LIB_AAUDIO_SO);
        return false;
    }

    if (!AAUDIO_LoadFunctions(&ctx)) {
        SDL_UnloadObject(ctx.handle);
        SDL_zero(ctx);
        return false;
    }

    impl->ThreadInit = Android_AudioThreadInit;
    impl->Deinitialize = AAUDIO_Deinitialize;
    impl->OpenDevice = AAUDIO_OpenDevice;
    impl->CloseDevice = AAUDIO_CloseDevice;
    impl->WaitDevice = AAUDIO_WaitDevice;
    impl->PlayDevice = AAUDIO_PlayDevice;
    impl->GetDeviceBuf = AAUDIO_GetDeviceBuf;
    impl->WaitRecordingDevice = AAUDIO_WaitDevice;
    impl->RecordDevice = AAUDIO_RecordDevice;

    impl->HasRecordingSupport = true;

#if ALLOW_MULTIPLE_ANDROID_AUDIO_DEVICES
    impl->DetectDevices = Android_StartAudioHotplug;
#else
    impl->OnlyHasDefaultPlaybackDevice = true;
    impl->OnlyHasDefaultRecordingDevice = true;
#endif

    LOGI("SDL AAUDIO_Init OK");
    return true;
}

AudioBootStrap AAUDIO_bootstrap = {
    "AAudio", "AAudio audio driver", AAUDIO_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_AAUDIO
