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

#ifdef SDL_AUDIO_DRIVER_VITA

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../SDL_audiodev_c.h"
#include "../SDL_sysaudio.h"
#include "SDL_vitaaudio.h"

#include <psp2/kernel/threadmgr.h>
#include <psp2/audioout.h>
#include <psp2/audioin.h>

#define SCE_AUDIO_SAMPLE_ALIGN(s) (((s) + 63) & ~63)
#define SCE_AUDIO_MAX_VOLUME      0x8000

static bool VITAAUD_OpenRecordingDevice(SDL_AudioDevice *device)
{
    device->spec.freq = 16000;
    device->spec.channels = 1;
    device->sample_frames = 512;

    SDL_UpdatedAudioDeviceFormat(device);

    device->hidden->port = sceAudioInOpenPort(SCE_AUDIO_IN_PORT_TYPE_VOICE, 512, 16000, SCE_AUDIO_IN_PARAM_FORMAT_S16_MONO);

    if (device->hidden->port < 0) {
        return SDL_SetError("Couldn't open audio in port: %x", device->hidden->port);
    }

    return true;
}

static bool VITAAUD_OpenDevice(SDL_AudioDevice *device)
{
    int format, mixlen, i, port = SCE_AUDIO_OUT_PORT_TYPE_MAIN;
    int vols[2] = { SCE_AUDIO_MAX_VOLUME, SCE_AUDIO_MAX_VOLUME };
    SDL_AudioFormat test_format;
    const SDL_AudioFormat *closefmts;

    device->hidden = (struct SDL_PrivateAudioData *)
        SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
        if (test_format == SDL_AUDIO_S16LE) {
            device->spec.format = test_format;
            break;
        }
    }

    if (!test_format) {
        return SDL_SetError("Unsupported audio format");
    }

    if (device->recording) {
        return VITAAUD_OpenRecordingDevice(device);
    }

    // The sample count must be a multiple of 64.
    device->sample_frames = SCE_AUDIO_SAMPLE_ALIGN(device->sample_frames);

    // Update the fragment size as size in bytes.
    SDL_UpdatedAudioDeviceFormat(device);

    /* Allocate the mixing buffer.  Its size and starting address must
       be a multiple of 64 bytes.  Our sample count is already a multiple of
       64, so spec->size should be a multiple of 64 as well. */
    mixlen = device->buffer_size * NUM_BUFFERS;
    device->hidden->rawbuf = (Uint8 *)SDL_aligned_alloc(64, mixlen);
    if (!device->hidden->rawbuf) {
        return SDL_SetError("Couldn't allocate mixing buffer");
    }

    // Setup the hardware channel.
    if (device->spec.channels == 1) {
        format = SCE_AUDIO_OUT_MODE_MONO;
    } else {
        format = SCE_AUDIO_OUT_MODE_STEREO;
    }

    // the main port requires 48000Hz audio, so this drops to the background music port if necessary
    if (device->spec.freq < 48000) {
        port = SCE_AUDIO_OUT_PORT_TYPE_BGM;
    }

    device->hidden->port = sceAudioOutOpenPort(port, device->sample_frames, device->spec.freq, format);
    if (device->hidden->port < 0) {
        SDL_aligned_free(device->hidden->rawbuf);
        device->hidden->rawbuf = NULL;
        return SDL_SetError("Couldn't open audio out port: %x", device->hidden->port);
    }

    sceAudioOutSetVolume(device->hidden->port, SCE_AUDIO_VOLUME_FLAG_L_CH | SCE_AUDIO_VOLUME_FLAG_R_CH, vols);

    SDL_memset(device->hidden->rawbuf, 0, mixlen);
    for (i = 0; i < NUM_BUFFERS; i++) {
        device->hidden->mixbufs[i] = &device->hidden->rawbuf[i * device->buffer_size];
    }

    device->hidden->next_buffer = 0;
    return true;
}

static bool VITAAUD_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buffer_size)
{
    return (sceAudioOutOutput(device->hidden->port, buffer) == 0);
}

// This function waits until it is possible to write a full sound buffer
static bool VITAAUD_WaitDevice(SDL_AudioDevice *device)
{
    // !!! FIXME: we might just need to sleep roughly as long as playback buffers take to process, based on sample rate, etc.
    while (!SDL_GetAtomicInt(&device->shutdown) && (sceAudioOutGetRestSample(device->hidden->port) >= device->buffer_size)) {
        SDL_Delay(1);
    }
    return true;
}

static Uint8 *VITAAUD_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    Uint8 *result = device->hidden->mixbufs[device->hidden->next_buffer];
    device->hidden->next_buffer = (device->hidden->next_buffer + 1) % NUM_BUFFERS;
    return result;
}

static void VITAAUD_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->port >= 0) {
            if (device->recording) {
                sceAudioInReleasePort(device->hidden->port);
            } else {
                sceAudioOutReleasePort(device->hidden->port);
            }
            device->hidden->port = -1;
        }

        if (!device->recording && device->hidden->rawbuf) {
            SDL_aligned_free(device->hidden->rawbuf); // this uses SDL_aligned_alloc(), not SDL_malloc()
            device->hidden->rawbuf = NULL;
        }
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

static bool VITAAUD_WaitRecordingDevice(SDL_AudioDevice *device)
{
    // there's only a blocking call to obtain more data, so we'll just sleep as
    //  long as a buffer would run.
    const Uint64 endticks = SDL_GetTicks() + ((device->sample_frames * 1000) / device->spec.freq);
    while (!SDL_GetAtomicInt(&device->shutdown) && (SDL_GetTicks() < endticks)) {
        SDL_Delay(1);
    }
    return true;
}

static int VITAAUD_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    int ret;
    SDL_assert(buflen == device->buffer_size);
    ret = sceAudioInInput(device->hidden->port, buffer);
    if (ret < 0) {
        SDL_SetError("Failed to record from device: %x", ret);
        return -1;
    }
    return device->buffer_size;
}

static void VITAAUD_FlushRecording(SDL_AudioDevice *device)
{
    // just grab the latest and dump it.
    sceAudioInInput(device->hidden->port, device->work_buffer);
}

static void VITAAUD_ThreadInit(SDL_AudioDevice *device)
{
    // Increase the priority of this audio thread by 1 to put it ahead of other SDL threads.
    SceUID thid;
    SceKernelThreadInfo info;
    thid = sceKernelGetThreadId();
    info.size = sizeof(SceKernelThreadInfo);
    if (sceKernelGetThreadInfo(thid, &info) == 0) {
        sceKernelChangeThreadPriority(thid, info.currentPriority - 1);
    }
}

static bool VITAAUD_Init(SDL_AudioDriverImpl *impl)
{
    impl->OpenDevice = VITAAUD_OpenDevice;
    impl->PlayDevice = VITAAUD_PlayDevice;
    impl->WaitDevice = VITAAUD_WaitDevice;
    impl->GetDeviceBuf = VITAAUD_GetDeviceBuf;
    impl->CloseDevice = VITAAUD_CloseDevice;
    impl->ThreadInit = VITAAUD_ThreadInit;
    impl->WaitRecordingDevice = VITAAUD_WaitRecordingDevice;
    impl->FlushRecording = VITAAUD_FlushRecording;
    impl->RecordDevice = VITAAUD_RecordDevice;

    impl->HasRecordingSupport = true;
    impl->OnlyHasDefaultPlaybackDevice = true;
    impl->OnlyHasDefaultRecordingDevice = true;

    return true;
}

AudioBootStrap VITAAUD_bootstrap = {
    "vita", "VITA audio driver", VITAAUD_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_VITA
