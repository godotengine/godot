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

// !!! FIXME: clean out perror and fprintf calls in here.

#ifdef SDL_AUDIO_DRIVER_OSS

#include <stdio.h>  // For perror()
#include <string.h> // For strerror()
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#include <sys/soundcard.h>

#include "../SDL_audiodev_c.h"
#include "SDL_dspaudio.h"

static void DSP_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    SDL_EnumUnixAudioDevices(false, NULL);
}

static void DSP_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->audio_fd >= 0) {
            close(device->hidden->audio_fd);
        }
        SDL_free(device->hidden->mixbuf);
        SDL_free(device->hidden);
    }
}

static bool DSP_OpenDevice(SDL_AudioDevice *device)
{
    // Make sure fragment size stays a power of 2, or OSS fails.
    // (I don't know which of these are actually legal values, though...)
    if (device->spec.channels > 8) {
        device->spec.channels = 8;
    } else if (device->spec.channels > 4) {
        device->spec.channels = 4;
    } else if (device->spec.channels > 2) {
        device->spec.channels = 2;
    }

    // Initialize all variables that we clean on shutdown
    device->hidden = (struct SDL_PrivateAudioData *) SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    // Open the audio device; we hardcode the device path in `device->name` for lack of better info, so use that.
    const int flags = ((device->recording) ? OPEN_FLAGS_INPUT : OPEN_FLAGS_OUTPUT);
    device->hidden->audio_fd = open(device->name, flags | O_CLOEXEC, 0);
    if (device->hidden->audio_fd < 0) {
        return SDL_SetError("Couldn't open %s: %s", device->name, strerror(errno));
    }

    // Make the file descriptor use blocking i/o with fcntl()
    {
        const long ctlflags = fcntl(device->hidden->audio_fd, F_GETFL) & ~O_NONBLOCK;
        if (fcntl(device->hidden->audio_fd, F_SETFL, ctlflags) < 0) {
            return SDL_SetError("Couldn't set audio blocking mode");
        }
    }

    // Get a list of supported hardware formats
    int value;
    if (ioctl(device->hidden->audio_fd, SNDCTL_DSP_GETFMTS, &value) < 0) {
        perror("SNDCTL_DSP_GETFMTS");
        return SDL_SetError("Couldn't get audio format list");
    }

    // Try for a closest match on audio format
    int format = 0;
    SDL_AudioFormat test_format;
    const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
#ifdef DEBUG_AUDIO
        fprintf(stderr, "Trying format 0x%4.4x\n", test_format);
#endif
        switch (test_format) {
        case SDL_AUDIO_U8:
            if (value & AFMT_U8) {
                format = AFMT_U8;
            }
            break;
        case SDL_AUDIO_S16LE:
            if (value & AFMT_S16_LE) {
                format = AFMT_S16_LE;
            }
            break;
        case SDL_AUDIO_S16BE:
            if (value & AFMT_S16_BE) {
                format = AFMT_S16_BE;
            }
            break;

        default:
            continue;
        }
        break;
    }
    if (format == 0) {
        return SDL_SetError("Couldn't find any hardware audio formats");
    }
    device->spec.format = test_format;

    // Set the audio format
    value = format;
    if ((ioctl(device->hidden->audio_fd, SNDCTL_DSP_SETFMT, &value) < 0) ||
        (value != format)) {
        perror("SNDCTL_DSP_SETFMT");
        return SDL_SetError("Couldn't set audio format");
    }

    // Set the number of channels of output
    value = device->spec.channels;
    if (ioctl(device->hidden->audio_fd, SNDCTL_DSP_CHANNELS, &value) < 0) {
        perror("SNDCTL_DSP_CHANNELS");
        return SDL_SetError("Cannot set the number of channels");
    }
    device->spec.channels = value;

    // Set the DSP frequency
    value = device->spec.freq;
    if (ioctl(device->hidden->audio_fd, SNDCTL_DSP_SPEED, &value) < 0) {
        perror("SNDCTL_DSP_SPEED");
        return SDL_SetError("Couldn't set audio frequency");
    }
    device->spec.freq = value;

    // Calculate the final parameters for this audio specification
    SDL_UpdatedAudioDeviceFormat(device);

    /* Determine the power of two of the fragment size
       Since apps don't control this in SDL3, and this driver only accepts 8, 16
       bit formats and 1, 2, 4, 8 channels, this should always be a power of 2 already. */
    SDL_assert(SDL_powerof2(device->buffer_size) == device->buffer_size);

    int frag_spec = 0;
    while ((0x01U << frag_spec) < device->buffer_size) {
        frag_spec++;
    }
    frag_spec |= 0x00020000; // two fragments, for low latency

    // Set the audio buffering parameters
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Requesting %d fragments of size %d\n",
            (frag_spec >> 16), 1 << (frag_spec & 0xFFFF));
#endif
    if (ioctl(device->hidden->audio_fd, SNDCTL_DSP_SETFRAGMENT, &frag_spec) < 0) {
        perror("SNDCTL_DSP_SETFRAGMENT");
    }
#ifdef DEBUG_AUDIO
    {
        audio_buf_info info;
        ioctl(device->hidden->audio_fd, SNDCTL_DSP_GETOSPACE, &info);
        fprintf(stderr, "fragments = %d\n", info.fragments);
        fprintf(stderr, "fragstotal = %d\n", info.fragstotal);
        fprintf(stderr, "fragsize = %d\n", info.fragsize);
        fprintf(stderr, "bytes = %d\n", info.bytes);
    }
#endif

    // Allocate mixing buffer
    if (!device->recording) {
        device->hidden->mixbuf = (Uint8 *)SDL_malloc(device->buffer_size);
        if (!device->hidden->mixbuf) {
            return false;
        }
        SDL_memset(device->hidden->mixbuf, device->silence_value, device->buffer_size);
    }

    return true;  // We're ready to rock and roll. :-)
}

static bool DSP_WaitDevice(SDL_AudioDevice *device)
{
    const unsigned long ioctlreq = device->recording ? SNDCTL_DSP_GETISPACE : SNDCTL_DSP_GETOSPACE;
    struct SDL_PrivateAudioData *h = device->hidden;

    while (!SDL_GetAtomicInt(&device->shutdown)) {
        audio_buf_info info;
        const int rc = ioctl(h->audio_fd, ioctlreq, &info);
        if (rc < 0) {
            if (errno == EAGAIN) {
                continue;
            }
            // Hmm, not much we can do - abort
            fprintf(stderr, "dsp WaitDevice ioctl failed (unrecoverable): %s\n", strerror(errno));
            return false;
        } else if (info.bytes < device->buffer_size) {
            SDL_Delay(10);
        } else {
            break; // ready to go!
        }
    }

    return true;
}

static bool DSP_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    if (write(h->audio_fd, buffer, buflen) == -1) {
        perror("Audio write");
        return false;
    }
#ifdef DEBUG_AUDIO
    fprintf(stderr, "Wrote %d bytes of audio data\n", h->mixlen);
#endif
    return true;
}

static Uint8 *DSP_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    return device->hidden->mixbuf;
}

static int DSP_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    return (int)read(device->hidden->audio_fd, buffer, buflen);
}

static void DSP_FlushRecording(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    audio_buf_info info;
    if (ioctl(h->audio_fd, SNDCTL_DSP_GETISPACE, &info) == 0) {
        while (info.bytes > 0) {
            char buf[512];
            const size_t len = SDL_min(sizeof(buf), info.bytes);
            const ssize_t br = read(h->audio_fd, buf, len);
            if (br <= 0) {
                break;
            }
            info.bytes -= br;
        }
    }
}

static bool InitTimeDevicesExist = false;
static bool look_for_devices_test(int fd)
{
    InitTimeDevicesExist = true; // note that _something_ exists.
    // Don't add to the device list, we're just seeing if any devices exist.
    return false;
}

static bool DSP_Init(SDL_AudioDriverImpl *impl)
{
    InitTimeDevicesExist = false;
    SDL_EnumUnixAudioDevices(false, look_for_devices_test);
    if (!InitTimeDevicesExist) {
        SDL_SetError("dsp: No such audio device");
        return false; // maybe try a different backend.
    }

    impl->DetectDevices = DSP_DetectDevices;
    impl->OpenDevice = DSP_OpenDevice;
    impl->WaitDevice = DSP_WaitDevice;
    impl->PlayDevice = DSP_PlayDevice;
    impl->GetDeviceBuf = DSP_GetDeviceBuf;
    impl->CloseDevice = DSP_CloseDevice;
    impl->WaitRecordingDevice = DSP_WaitDevice;
    impl->RecordDevice = DSP_RecordDevice;
    impl->FlushRecording = DSP_FlushRecording;

    impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap DSP_bootstrap = {
    "dsp", "Open Sound System (/dev/dsp)", DSP_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_OSS
