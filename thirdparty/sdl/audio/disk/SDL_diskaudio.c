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

#ifdef SDL_AUDIO_DRIVER_DISK

// Output raw audio data to a file.

#include "../SDL_sysaudio.h"
#include "SDL_diskaudio.h"

#define DISKDEFAULT_OUTFILE "sdlaudio.raw"
#define DISKDEFAULT_INFILE  "sdlaudio-in.raw"

static bool DISKAUDIO_WaitDevice(SDL_AudioDevice *device)
{
    SDL_Delay(device->hidden->io_delay);
    return true;
}

static bool DISKAUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buffer_size)
{
    const int written = (int)SDL_WriteIO(device->hidden->io, buffer, (size_t)buffer_size);
    if (written != buffer_size) { // If we couldn't write, assume fatal error for now
        return false;
    }
#ifdef DEBUG_AUDIO
    SDL_Log("DISKAUDIO: Wrote %d bytes of audio data", (int) written);
#endif
    return true;
}

static Uint8 *DISKAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    return device->hidden->mixbuf;
}

static int DISKAUDIO_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    const int origbuflen = buflen;

    if (h->io) {
        const int br = (int)SDL_ReadIO(h->io, buffer, (size_t)buflen);
        buflen -= br;
        buffer = ((Uint8 *)buffer) + br;
        if (buflen > 0) { // EOF (or error, but whatever).
            SDL_CloseIO(h->io);
            h->io = NULL;
        }
    }

    // if we ran out of file, just write silence.
    SDL_memset(buffer, device->silence_value, buflen);

    return origbuflen;
}

static void DISKAUDIO_FlushRecording(SDL_AudioDevice *device)
{
    // no op...we don't advance the file pointer or anything.
}

static void DISKAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->io) {
            SDL_CloseIO(device->hidden->io);
        }
        SDL_free(device->hidden->mixbuf);
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

static const char *get_filename(const bool recording)
{
    const char *devname = SDL_GetHint(recording ? SDL_HINT_AUDIO_DISK_INPUT_FILE : SDL_HINT_AUDIO_DISK_OUTPUT_FILE);
    if (!devname) {
        devname = recording ? DISKDEFAULT_INFILE : DISKDEFAULT_OUTFILE;
    }
    return devname;
}

static const char *AudioFormatString(SDL_AudioFormat fmt)
{
    const char *str = SDL_GetAudioFormatName(fmt);
    SDL_assert(str);
    if (SDL_strncmp(str, "SDL_AUDIO_", 10) == 0) {
        str += 10;  // so we return "S8" instead of "SDL_AUDIO_S8", etc.
    }
    return str;
}

static bool DISKAUDIO_OpenDevice(SDL_AudioDevice *device)
{
    bool recording = device->recording;
    const char *fname = get_filename(recording);

    device->hidden = (struct SDL_PrivateAudioData *) SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    device->hidden->io_delay = ((device->sample_frames * 1000) / device->spec.freq);

    const char *hint = SDL_GetHint(SDL_HINT_AUDIO_DISK_TIMESCALE);
    if (hint) {
        double scale = SDL_atof(hint);
        if (scale >= 0.0) {
            device->hidden->io_delay = (Uint32)SDL_round(device->hidden->io_delay * scale);
        }
    }

    // Open the "audio device"
    device->hidden->io = SDL_IOFromFile(fname, recording ? "rb" : "wb");
    if (!device->hidden->io) {
        return false;
    }

    // Allocate mixing buffer
    if (!recording) {
        device->hidden->mixbuf = (Uint8 *)SDL_malloc(device->buffer_size);
        if (!device->hidden->mixbuf) {
            return false;
        }
        SDL_memset(device->hidden->mixbuf, device->silence_value, device->buffer_size);
    }

    SDL_LogCritical(SDL_LOG_CATEGORY_AUDIO, "You are using the SDL disk i/o audio driver!");
    SDL_LogCritical(SDL_LOG_CATEGORY_AUDIO, " %s file [%s], format=%s channels=%d freq=%d.",
                    recording ? "Reading from" : "Writing to", fname,
                    AudioFormatString(device->spec.format), device->spec.channels, device->spec.freq);

    return true;  // We're ready to rock and roll. :-)
}

static void DISKAUDIO_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    *default_playback = SDL_AddAudioDevice(false, DEFAULT_PLAYBACK_DEVNAME, NULL, (void *)0x1);
    *default_recording = SDL_AddAudioDevice(true, DEFAULT_RECORDING_DEVNAME, NULL, (void *)0x2);
}

static bool DISKAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    impl->OpenDevice = DISKAUDIO_OpenDevice;
    impl->WaitDevice = DISKAUDIO_WaitDevice;
    impl->WaitRecordingDevice = DISKAUDIO_WaitDevice;
    impl->PlayDevice = DISKAUDIO_PlayDevice;
    impl->GetDeviceBuf = DISKAUDIO_GetDeviceBuf;
    impl->RecordDevice = DISKAUDIO_RecordDevice;
    impl->FlushRecording = DISKAUDIO_FlushRecording;
    impl->CloseDevice = DISKAUDIO_CloseDevice;
    impl->DetectDevices = DISKAUDIO_DetectDevices;

    impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap DISKAUDIO_bootstrap = {
    "disk", "direct-to-disk audio", DISKAUDIO_Init, true, false
};

#endif // SDL_AUDIO_DRIVER_DISK
