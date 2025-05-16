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

// Output audio to nowhere...

#include "../SDL_sysaudio.h"
#include "SDL_dummyaudio.h"

#if defined(SDL_PLATFORM_EMSCRIPTEN) && !defined(__EMSCRIPTEN_PTHREADS__)
#include <emscripten/emscripten.h>
#endif

static bool DUMMYAUDIO_WaitDevice(SDL_AudioDevice *device)
{
    SDL_Delay(device->hidden->io_delay);
    return true;
}

static bool DUMMYAUDIO_OpenDevice(SDL_AudioDevice *device)
{
    device->hidden = (struct SDL_PrivateAudioData *) SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    if (!device->recording) {
        device->hidden->mixbuf = (Uint8 *) SDL_malloc(device->buffer_size);
        if (!device->hidden->mixbuf) {
            return false;
        }
    }

    device->hidden->io_delay = ((device->sample_frames * 1000) / device->spec.freq);

    const char *hint = SDL_GetHint(SDL_HINT_AUDIO_DUMMY_TIMESCALE);
    if (hint) {
        double scale = SDL_atof(hint);
        if (scale >= 0.0) {
            device->hidden->io_delay = (Uint32)SDL_round(device->hidden->io_delay * scale);
        }
    }

    // on Emscripten without threads, we just fire a repeating timer to consume audio.
    #if defined(SDL_PLATFORM_EMSCRIPTEN) && !defined(__EMSCRIPTEN_PTHREADS__)
    MAIN_THREAD_EM_ASM({
        var a = Module['SDL3'].dummy_audio;
        if (a.timers[$0] !== undefined) { clearInterval(a.timers[$0]); }
        a.timers[$0] = setInterval(function() { dynCall('vi', $3, [$4]); }, ($1 / $2) * 1000);
    }, device->recording ? 1 : 0, device->sample_frames, device->spec.freq, device->recording ? SDL_RecordingAudioThreadIterate : SDL_PlaybackAudioThreadIterate, device);
    #endif

    return true; // we're good; don't change reported device format.
}

static void DUMMYAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        // on Emscripten without threads, we just fire a repeating timer to consume audio.
        #if defined(SDL_PLATFORM_EMSCRIPTEN) && !defined(__EMSCRIPTEN_PTHREADS__)
        MAIN_THREAD_EM_ASM({
            var a = Module['SDL3'].dummy_audio;
            if (a.timers[$0] !== undefined) { clearInterval(a.timers[$0]); }
            a.timers[$0] = undefined;
        }, device->recording ? 1 : 0);
        #endif
        SDL_free(device->hidden->mixbuf);
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

static Uint8 *DUMMYAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    return device->hidden->mixbuf;
}

static int DUMMYAUDIO_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    // always return a full buffer of silence.
    SDL_memset(buffer, device->silence_value, buflen);
    return buflen;
}

static bool DUMMYAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    impl->OpenDevice = DUMMYAUDIO_OpenDevice;
    impl->CloseDevice = DUMMYAUDIO_CloseDevice;
    impl->WaitDevice = DUMMYAUDIO_WaitDevice;
    impl->GetDeviceBuf = DUMMYAUDIO_GetDeviceBuf;
    impl->WaitRecordingDevice = DUMMYAUDIO_WaitDevice;
    impl->RecordDevice = DUMMYAUDIO_RecordDevice;

    impl->OnlyHasDefaultPlaybackDevice = true;
    impl->OnlyHasDefaultRecordingDevice = true;
    impl->HasRecordingSupport = true;

    // on Emscripten without threads, we just fire a repeating timer to consume audio.
    #if defined(SDL_PLATFORM_EMSCRIPTEN) && !defined(__EMSCRIPTEN_PTHREADS__)
    MAIN_THREAD_EM_ASM({
        if (typeof(Module['SDL3']) === 'undefined') {
            Module['SDL3'] = {};
        }
        Module['SDL3'].dummy_audio = {};
        Module['SDL3'].dummy_audio.timers = [];
        Module['SDL3'].dummy_audio.timers[0] = undefined;
        Module['SDL3'].dummy_audio.timers[1] = undefined;
    });
    impl->ProvidesOwnCallbackThread = true;
    #endif

    return true;
}

AudioBootStrap DUMMYAUDIO_bootstrap = {
    "dummy", "SDL dummy audio driver", DUMMYAUDIO_Init, true, false
};
