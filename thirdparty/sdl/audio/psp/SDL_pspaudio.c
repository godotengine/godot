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

#ifdef SDL_AUDIO_DRIVER_PSP

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../SDL_audiodev_c.h"
#include "../SDL_sysaudio.h"
#include "SDL_pspaudio.h"

#include <pspaudio.h>
#include <pspthreadman.h>

static bool isBasicAudioConfig(const SDL_AudioSpec *spec)
{
    return spec->freq == 44100;
}

static bool PSPAUDIO_OpenDevice(SDL_AudioDevice *device)
{
    device->hidden = (struct SDL_PrivateAudioData *) SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    // device only natively supports S16LSB
    device->spec.format = SDL_AUDIO_S16LE;

    /*  PSP has some limitations with the Audio. It fully supports 44.1KHz (Mono & Stereo),
        however with frequencies different than 44.1KHz, it just supports Stereo,
        so a resampler must be done for these scenarios */
    if (isBasicAudioConfig(&device->spec)) {
        // The sample count must be a multiple of 64.
        device->sample_frames = PSP_AUDIO_SAMPLE_ALIGN(device->sample_frames);
        // The number of channels (1 or 2).
        device->spec.channels = device->spec.channels == 1 ? 1 : 2;
        const int format = (device->spec.channels == 1) ? PSP_AUDIO_FORMAT_MONO : PSP_AUDIO_FORMAT_STEREO;
        device->hidden->channel = sceAudioChReserve(PSP_AUDIO_NEXT_CHANNEL, device->sample_frames, format);
    } else {
        // 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11050, 8000
        switch (device->spec.freq) {
        case 8000:
        case 11025:
        case 12000:
        case 16000:
        case 22050:
        case 24000:
        case 32000:
        case 44100:
        case 48000:
            break;  // acceptable, keep it
        default:
            device->spec.freq = 48000;
            break;
        }
        // The number of samples to output in one output call (min 17, max 4111).
        device->sample_frames = device->sample_frames < 17 ? 17 : (device->sample_frames > 4111 ? 4111 : device->sample_frames);
        device->spec.channels = 2; // we're forcing the hardware to stereo.
        device->hidden->channel = sceAudioSRCChReserve(device->sample_frames, device->spec.freq, 2);
    }

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

static bool PSPAUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    int rc;
    if (!isBasicAudioConfig(&device->spec)) {
        SDL_assert(device->spec.channels == 2);
        rc = sceAudioSRCOutputBlocking(PSP_AUDIO_VOLUME_MAX, (void *) buffer);
    } else {
        rc = sceAudioOutputPannedBlocking(device->hidden->channel, PSP_AUDIO_VOLUME_MAX, PSP_AUDIO_VOLUME_MAX, (void *) buffer);
    }
    return (rc == 0);
}

static bool PSPAUDIO_WaitDevice(SDL_AudioDevice *device)
{
    return true;  // Because we block when sending audio, there's no need for this function to do anything.
}

static Uint8 *PSPAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    Uint8 *buffer = device->hidden->mixbufs[device->hidden->next_buffer];
    device->hidden->next_buffer = (device->hidden->next_buffer + 1) % NUM_BUFFERS;
    return buffer;
}

static void PSPAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->channel >= 0) {
            if (!isBasicAudioConfig(&device->spec)) {
                sceAudioSRCChRelease();
            } else {
                sceAudioChRelease(device->hidden->channel);
            }
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

static void PSPAUDIO_ThreadInit(SDL_AudioDevice *device)
{
    /* Increase the priority of this audio thread by 1 to put it
       ahead of other SDL threads. */
    const SceUID thid = sceKernelGetThreadId();
    SceKernelThreadInfo status;
    status.size = sizeof(SceKernelThreadInfo);
    if (sceKernelReferThreadStatus(thid, &status) == 0) {
        sceKernelChangeThreadPriority(thid, status.currentPriority - 1);
    }
}

static bool PSPAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    impl->OpenDevice = PSPAUDIO_OpenDevice;
    impl->PlayDevice = PSPAUDIO_PlayDevice;
    impl->WaitDevice = PSPAUDIO_WaitDevice;
    impl->GetDeviceBuf = PSPAUDIO_GetDeviceBuf;
    impl->CloseDevice = PSPAUDIO_CloseDevice;
    impl->ThreadInit = PSPAUDIO_ThreadInit;
    impl->OnlyHasDefaultPlaybackDevice = true;
    //impl->HasRecordingSupport = true;
    //impl->OnlyHasDefaultRecordingDevice = true;
    return true;
}

AudioBootStrap PSPAUDIO_bootstrap = {
    "psp", "PSP audio driver", PSPAUDIO_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_PSP
