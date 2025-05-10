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

#ifdef SDL_AUDIO_DRIVER_HAIKU

// Allow access to the audio stream on Haiku

#include <SoundPlayer.h>
#include <signal.h>

#include "../../core/haiku/SDL_BeApp.h"

extern "C"
{

#include "../SDL_sysaudio.h"
#include "SDL_haikuaudio.h"

}

static Uint8 *HAIKUAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    SDL_assert(device->hidden->current_buffer != NULL);
    SDL_assert(device->hidden->current_buffer_len > 0);
    *buffer_size = device->hidden->current_buffer_len;
    return device->hidden->current_buffer;
}

static bool HAIKUAUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buffer_size)
{
    // We already wrote our output right into the BSoundPlayer's callback's stream. Just clean up our stuff.
    SDL_assert(device->hidden->current_buffer != NULL);
    SDL_assert(device->hidden->current_buffer_len > 0);
    device->hidden->current_buffer = NULL;
    device->hidden->current_buffer_len = 0;
    return true;
}

// The Haiku callback for handling the audio buffer
static void FillSound(void *data, void *stream, size_t len, const media_raw_audio_format & format)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)data;
    SDL_assert(device->hidden->current_buffer == NULL);
    SDL_assert(device->hidden->current_buffer_len == 0);
    device->hidden->current_buffer = (Uint8 *) stream;
    device->hidden->current_buffer_len = (int) len;
    SDL_PlaybackAudioThreadIterate(device);
}

static void HAIKUAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->audio_obj) {
            device->hidden->audio_obj->Stop();
            delete device->hidden->audio_obj;
        }
        delete device->hidden;
        device->hidden = NULL;
        SDL_AudioThreadFinalize(device);
    }
}


static const int sig_list[] = {
    SIGHUP, SIGINT, SIGQUIT, SIGPIPE, SIGALRM, SIGTERM, SIGWINCH, 0
};

static inline void MaskSignals(sigset_t * omask)
{
    sigset_t mask;
    int i;

    sigemptyset(&mask);
    for (i = 0; sig_list[i]; ++i) {
        sigaddset(&mask, sig_list[i]);
    }
    sigprocmask(SIG_BLOCK, &mask, omask);
}

static inline void UnmaskSignals(sigset_t * omask)
{
    sigprocmask(SIG_SETMASK, omask, NULL);
}


static bool HAIKUAUDIO_OpenDevice(SDL_AudioDevice *device)
{
    // Initialize all variables that we clean on shutdown
    device->hidden = new SDL_PrivateAudioData;
    if (!device->hidden) {
        return false;
    }
    SDL_zerop(device->hidden);

    // Parse the audio format and fill the Be raw audio format
    media_raw_audio_format format;
    SDL_zero(format);
    format.byte_order = B_MEDIA_LITTLE_ENDIAN;
    format.frame_rate = (float) device->spec.freq;
    format.channel_count = device->spec.channels;        // !!! FIXME: support > 2?

    SDL_AudioFormat test_format;
    const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
        switch (test_format) {
        case SDL_AUDIO_S8:
            format.format = media_raw_audio_format::B_AUDIO_CHAR;
            break;

        case SDL_AUDIO_U8:
            format.format = media_raw_audio_format::B_AUDIO_UCHAR;
            break;

        case SDL_AUDIO_S16LE:
            format.format = media_raw_audio_format::B_AUDIO_SHORT;
            break;

        case SDL_AUDIO_S16BE:
            format.format = media_raw_audio_format::B_AUDIO_SHORT;
            format.byte_order = B_MEDIA_BIG_ENDIAN;
            break;

        case SDL_AUDIO_S32LE:
            format.format = media_raw_audio_format::B_AUDIO_INT;
            break;

        case SDL_AUDIO_S32BE:
            format.format = media_raw_audio_format::B_AUDIO_INT;
            format.byte_order = B_MEDIA_BIG_ENDIAN;
            break;

        case SDL_AUDIO_F32LE:
            format.format = media_raw_audio_format::B_AUDIO_FLOAT;
            break;

        case SDL_AUDIO_F32BE:
            format.format = media_raw_audio_format::B_AUDIO_FLOAT;
            format.byte_order = B_MEDIA_BIG_ENDIAN;
            break;

        default:
            continue;
        }
        break;
    }

    if (!test_format) {      // shouldn't happen, but just in case...
        return SDL_SetError("HAIKU: Unsupported audio format");
    }
    device->spec.format = test_format;

    // Calculate the final parameters for this audio specification
    SDL_UpdatedAudioDeviceFormat(device);

    format.buffer_size = device->buffer_size;

    // Subscribe to the audio stream (creates a new thread)
    sigset_t omask;
    MaskSignals(&omask);
    device->hidden->audio_obj = new BSoundPlayer(&format, "SDL Audio",
                                                FillSound, NULL, device);
    UnmaskSignals(&omask);

    if (device->hidden->audio_obj->Start() == B_NO_ERROR) {
        device->hidden->audio_obj->SetHasData(true);
    } else {
        return SDL_SetError("Unable to start Haiku audio");
    }

    return true;  // We're running!
}

static void HAIKUAUDIO_Deinitialize(void)
{
    SDL_QuitBeApp();
}

static bool HAIKUAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    if (!SDL_InitBeApp()) {
        return false;
    }

    // Set the function pointers
    impl->OpenDevice = HAIKUAUDIO_OpenDevice;
    impl->GetDeviceBuf = HAIKUAUDIO_GetDeviceBuf;
    impl->PlayDevice = HAIKUAUDIO_PlayDevice;
    impl->CloseDevice = HAIKUAUDIO_CloseDevice;
    impl->Deinitialize = HAIKUAUDIO_Deinitialize;
    impl->ProvidesOwnCallbackThread = true;
    impl->OnlyHasDefaultPlaybackDevice = true;

    return true;
}


extern "C" { extern AudioBootStrap HAIKUAUDIO_bootstrap; }

AudioBootStrap HAIKUAUDIO_bootstrap = {
    "haiku", "Haiku BSoundPlayer", HAIKUAUDIO_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_HAIKU
