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

#include "../SDL_sysaudio.h"
#include "SDL_ps2audio.h"

#include <kernel.h>
#include <audsrv.h>
#include <ps2_audio_driver.h>

static bool PS2AUDIO_OpenDevice(SDL_AudioDevice *device)
{
    device->hidden = (struct SDL_PrivateAudioData *) SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    // These are the native supported audio PS2 configs
    switch (device->spec.freq) {
    case 11025:
    case 12000:
    case 22050:
    case 24000:
    case 32000:
    case 44100:
    case 48000:
        break;  // acceptable value, keep it
    default:
        device->spec.freq = 48000;
        break;
    }

    device->sample_frames = 512;
    device->spec.channels = device->spec.channels == 1 ? 1 : 2;
    device->spec.format = device->spec.format == SDL_AUDIO_S8 ? SDL_AUDIO_S8 : SDL_AUDIO_S16;

    struct audsrv_fmt_t format;
    format.bits = device->spec.format == SDL_AUDIO_S8 ? 8 : 16;
    format.freq = device->spec.freq;
    format.channels = device->spec.channels;

    device->hidden->channel = audsrv_set_format(&format);
    audsrv_set_volume(MAX_VOLUME);

    if (device->hidden->channel < 0) {
        return SDL_SetError("Couldn't reserve hardware channel");
    }

    // Update the fragment size as size in bytes.
    SDL_UpdatedAudioDeviceFormat(device);

    /* Allocate the mixing buffer.  Its size and starting address must
       be a multiple of 64 bytes.  Our sample count is already a multiple of
       64, so spec->size should be a multiple of 64 as well. */
    const int mixlen = device->buffer_size * NUM_BUFFERS;
    device->hidden->rawbuf = (Uint8 *)SDL_aligned_alloc(64, mixlen);
    if (!device->hidden->rawbuf) {
        return SDL_SetError("Couldn't allocate mixing buffer");
    }

    SDL_memset(device->hidden->rawbuf, device->silence_value, mixlen);
    for (int i = 0; i < NUM_BUFFERS; i++) {
        device->hidden->mixbufs[i] = &device->hidden->rawbuf[i * device->buffer_size];
    }

    return true;
}

static bool PS2AUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    // this returns number of bytes accepted or a negative error. We assume anything other than buflen is a fatal error.
    return (audsrv_play_audio((char *)buffer, buflen) == buflen);
}

static bool PS2AUDIO_WaitDevice(SDL_AudioDevice *device)
{
    audsrv_wait_audio(device->buffer_size);
    return true;
}

static Uint8 *PS2AUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    Uint8 *buffer = device->hidden->mixbufs[device->hidden->next_buffer];
    device->hidden->next_buffer = (device->hidden->next_buffer + 1) % NUM_BUFFERS;
    return buffer;
}

static void PS2AUDIO_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->channel >= 0) {
            audsrv_stop_audio();
            device->hidden->channel = -1;
        }

        if (device->hidden->rawbuf) {
            SDL_aligned_free(device->hidden->rawbuf);
            device->hidden->rawbuf = NULL;
        }
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

static void PS2AUDIO_ThreadInit(SDL_AudioDevice *device)
{
    /* Increase the priority of this audio thread by 1 to put it
       ahead of other SDL threads. */
    const int32_t thid = GetThreadId();
    ee_thread_status_t status;
    if (ReferThreadStatus(thid, &status) == 0) {
        ChangeThreadPriority(thid, status.current_priority - 1);
    }
}

static void PS2AUDIO_Deinitialize(void)
{
    deinit_audio_driver();
}

static bool PS2AUDIO_Init(SDL_AudioDriverImpl *impl)
{
    if (init_audio_driver() < 0) {
        return false;
    }

    impl->OpenDevice = PS2AUDIO_OpenDevice;
    impl->PlayDevice = PS2AUDIO_PlayDevice;
    impl->WaitDevice = PS2AUDIO_WaitDevice;
    impl->GetDeviceBuf = PS2AUDIO_GetDeviceBuf;
    impl->CloseDevice = PS2AUDIO_CloseDevice;
    impl->ThreadInit = PS2AUDIO_ThreadInit;
    impl->Deinitialize = PS2AUDIO_Deinitialize;
    impl->OnlyHasDefaultPlaybackDevice = true;
    return true; // this audio target is available.
}

AudioBootStrap PS2AUDIO_bootstrap = {
    "ps2", "PS2 audio driver", PS2AUDIO_Init, false, false
};
