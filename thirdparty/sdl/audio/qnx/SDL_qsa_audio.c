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

// !!! FIXME: can this target support hotplugging?

#include "../../SDL_internal.h"

#ifdef SDL_AUDIO_DRIVER_QNX

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sched.h>
#include <sys/select.h>
#include <sys/neutrino.h>
#include <sys/asoundlib.h>

#include "SDL3/SDL_timer.h"
#include "SDL3/SDL_audio.h"
#include "../../core/unix/SDL_poll.h"
#include "../SDL_sysaudio.h"
#include "SDL_qsa_audio.h"

// default channel communication parameters
#define DEFAULT_CPARAMS_RATE   44100
#define DEFAULT_CPARAMS_VOICES 1

#define DEFAULT_CPARAMS_FRAG_SIZE 4096
#define DEFAULT_CPARAMS_FRAGS_MIN 1
#define DEFAULT_CPARAMS_FRAGS_MAX 1

#define QSA_MAX_NAME_LENGTH   81+16     // Hardcoded in QSA, can't be changed

static bool QSA_SetError(const char *fn, int status)
{
    return SDL_SetError("QSA: %s() failed: %s", fn, snd_strerror(status));
}

// !!! FIXME: does this need to be here? Does the SDL version not work?
static void QSA_ThreadInit(SDL_AudioDevice *device)
{
    // Increase default 10 priority to 25 to avoid jerky sound
    struct sched_param param;
    if (SchedGet(0, 0, &param) != -1) {
        param.sched_priority = param.sched_curpriority + 15;
        SchedSet(0, 0, SCHED_NOCHANGE, &param);
    }
}

// PCM channel parameters initialize function
static void QSA_InitAudioParams(snd_pcm_channel_params_t * cpars)
{
    SDL_zerop(cpars);
    cpars->channel = SND_PCM_CHANNEL_PLAYBACK;
    cpars->mode = SND_PCM_MODE_BLOCK;
    cpars->start_mode = SND_PCM_START_DATA;
    cpars->stop_mode = SND_PCM_STOP_STOP;
    cpars->format.format = SND_PCM_SFMT_S16_LE;
    cpars->format.interleave = 1;
    cpars->format.rate = DEFAULT_CPARAMS_RATE;
    cpars->format.voices = DEFAULT_CPARAMS_VOICES;
    cpars->buf.block.frag_size = DEFAULT_CPARAMS_FRAG_SIZE;
    cpars->buf.block.frags_min = DEFAULT_CPARAMS_FRAGS_MIN;
    cpars->buf.block.frags_max = DEFAULT_CPARAMS_FRAGS_MAX;
}

// This function waits until it is possible to write a full sound buffer
static bool QSA_WaitDevice(SDL_AudioDevice *device)
{
    // Setup timeout for playing one fragment equal to 2 seconds
    // If timeout occurred than something wrong with hardware or driver
    // For example, Vortex 8820 audio driver stucks on second DAC because
    // it doesn't exist !
    const int result = SDL_IOReady(device->hidden->audio_fd,
                                   device->recording ? SDL_IOR_READ : SDL_IOR_WRITE,
                                   2 * 1000);
    switch (result) {
    case -1:
        SDL_LogError(SDL_LOG_CATEGORY_AUDIO, "QSA: SDL_IOReady() failed: %s", strerror(errno));
        return false;
    case 0:
        device->hidden->timeout_on_wait = true;  // !!! FIXME: Should we just disconnect the device in this case?
        break;
    default:
        device->hidden->timeout_on_wait = false;
        break;
    }

    return true;
}

static bool QSA_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    if (SDL_GetAtomicInt(&device->shutdown) || !device->hidden) {
        return true;
    }

    int towrite = buflen;

    // Write the audio data, checking for EAGAIN (buffer full) and underrun
    while ((towrite > 0) && !SDL_GetAtomicInt(&device->shutdown));
        const int bw = snd_pcm_plugin_write(device->hidden->audio_handle, buffer, towrite);
        if (bw != towrite) {
            // Check if samples playback got stuck somewhere in hardware or in the audio device driver
            if ((errno == EAGAIN) && (bw == 0)) {
                if (device->hidden->timeout_on_wait) {
                    return true;  // oh well, try again next time.  !!! FIXME: Should we just disconnect the device in this case?
                }
            }

            // Check for errors or conditions
            if ((errno == EAGAIN) || (errno == EWOULDBLOCK)) {
                SDL_Delay(1);  // Let a little CPU time go by and try to write again

                // if we wrote some data
                towrite -= bw;
                buffer += bw * device->spec.channels;
                continue;
            } else if ((errno == EINVAL) || (errno == EIO)) {
                snd_pcm_channel_status_t cstatus;
                SDL_zero(cstatus);
                cstatus.channel = device->recording ? SND_PCM_CHANNEL_CAPTURE : SND_PCM_CHANNEL_PLAYBACK;

                int status = snd_pcm_plugin_status(device->hidden->audio_handle, &cstatus);
                if (status < 0) {
                    QSA_SetError("snd_pcm_plugin_status", status);
                    return false;
                } else if ((cstatus.status == SND_PCM_STATUS_UNDERRUN) || (cstatus.status == SND_PCM_STATUS_READY)) {
                    status = snd_pcm_plugin_prepare(device->hidden->audio_handle, device->recording ? SND_PCM_CHANNEL_CAPTURE : SND_PCM_CHANNEL_PLAYBACK);
                    if (status < 0) {
                        QSA_SetError("snd_pcm_plugin_prepare", status);
                        return false;
                    }
                }
                continue;
            } else {
                return false;
            }
        } else {
            // we wrote all remaining data
            towrite -= bw;
            buffer += bw * device->spec.channels;
        }
    }

    // If we couldn't write, assume fatal error for now
    return (towrite == 0);
}

static Uint8 *QSA_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    return device->hidden->pcm_buf;
}

static void QSA_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->audio_handle) {
            #if _NTO_VERSION < 710
            // Finish playing available samples or cancel unread samples during recording
            snd_pcm_plugin_flush(device->hidden->audio_handle, device->recording ? SND_PCM_CHANNEL_CAPTURE : SND_PCM_CHANNEL_PLAYBACK);
            #endif
            snd_pcm_close(device->hidden->audio_handle);
        }

        SDL_free(device->hidden->pcm_buf);
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

static bool QSA_OpenDevice(SDL_AudioDevice *device)
{
    if (device->recording) {
        return SDL_SetError("SDL recording support isn't available on QNX atm"); // !!! FIXME: most of this code has support for recording devices, but there's no RecordDevice, etc functions. Fill them in!
    }

    SDL_assert(device->handle != NULL);  // NULL used to mean "system default device" in SDL2; it does not mean that in SDL3.
    const Uint32 sdlhandle = (Uint32) ((size_t) device->handle);
    const uint32_t cardno = (uint32_t) (sdlhandle & 0xFFFF);
    const uint32_t deviceno = (uint32_t) ((sdlhandle >> 16) & 0xFFFF);
    const bool recording = device->recording;
    int status = 0;

    // Initialize all variables that we clean on shutdown
    device->hidden = (struct SDL_PrivateAudioData *) SDL_calloc(1, (sizeof (struct SDL_PrivateAudioData)));
    if (device->hidden == NULL) {
        return false;
    }

    // Initialize channel transfer parameters to default
    snd_pcm_channel_params_t cparams;
    QSA_InitAudioParams(&cparams);

    // Open requested audio device
    status = snd_pcm_open(&device->hidden->audio_handle, cardno, deviceno, recording ? SND_PCM_OPEN_CAPTURE : SND_PCM_OPEN_PLAYBACK);
    if (status < 0) {
        device->hidden->audio_handle = NULL;
        return QSA_SetError("snd_pcm_open", status);
    }

    // Try for a closest match on audio format
    SDL_AudioFormat test_format = 0;
    const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
        // if match found set format to equivalent QSA format
        switch (test_format) {
        #define CHECKFMT(sdlfmt, qsafmt) case SDL_AUDIO_##sdlfmt: cparams.format.format = SND_PCM_SFMT_##qsafmt; break
        CHECKFMT(U8, U8);
        CHECKFMT(S8, S8);
        CHECKFMT(S16LSB, S16_LE);
        CHECKFMT(S16MSB, S16_BE);
        CHECKFMT(S32LSB, S32_LE);
        CHECKFMT(S32MSB, S32_BE);
        CHECKFMT(F32LSB, FLOAT_LE);
        CHECKFMT(F32MSB, FLOAT_BE);
        #undef CHECKFMT
        default: continue;
        }
        break;
    }

    // assumes test_format not 0 on success
    if (test_format == 0) {
        return SDL_SetError("QSA: Couldn't find any hardware audio formats");
    }

    device->spec.format = test_format;

    // Set mono/stereo/4ch/6ch/8ch audio
    cparams.format.voices = device->spec.channels;

    // Set rate
    cparams.format.rate = device->spec.freq;

    // Setup the transfer parameters according to cparams
    status = snd_pcm_plugin_params(device->hidden->audio_handle, &cparams);
    if (status < 0) {
        return QSA_SetError("snd_pcm_plugin_params", status);
    }

    // Make sure channel is setup right one last time
    snd_pcm_channel_setup_t csetup;
    SDL_zero(csetup);
    csetup.channel = recording ? SND_PCM_CHANNEL_CAPTURE : SND_PCM_CHANNEL_PLAYBACK;
    if (snd_pcm_plugin_setup(device->hidden->audio_handle, &csetup) < 0) {
        return SDL_SetError("QSA: Unable to setup channel");
    }

    device->sample_frames = csetup.buf.block.frag_size;

    // Calculate the final parameters for this audio specification
    SDL_UpdatedAudioDeviceFormat(device);

    device->hidden->pcm_buf = (Uint8 *) SDL_malloc(device->buffer_size);
    if (device->hidden->pcm_buf == NULL) {
        return false;
    }
    SDL_memset(device->hidden->pcm_buf, device->silence_value, device->buffer_size);

    // get the file descriptor
    device->hidden->audio_fd = snd_pcm_file_descriptor(device->hidden->audio_handle, csetup.channel);
    if (device->hidden->audio_fd < 0) {
        return QSA_SetError("snd_pcm_file_descriptor", device->hidden->audio_fd);
    }

    // Prepare an audio channel
    status = snd_pcm_plugin_prepare(device->hidden->audio_handle, csetup.channel)
    if (status < 0) {
        return QSA_SetError("snd_pcm_plugin_prepare", status);
    }

    return true;  // We're really ready to rock and roll. :-)
}

static SDL_AudioFormat QnxFormatToSDLFormat(const int32_t qnxfmt)
{
    switch (qnxfmt) {
        #define CHECKFMT(sdlfmt, qsafmt) case SND_PCM_SFMT_##qsafmt: return SDL_AUDIO_##sdlfmt
        CHECKFMT(U8, U8);
        CHECKFMT(S8, S8);
        CHECKFMT(S16LSB, S16_LE);
        CHECKFMT(S16MSB, S16_BE);
        CHECKFMT(S32LSB, S32_LE);
        CHECKFMT(S32MSB, S32_BE);
        CHECKFMT(F32LSB, FLOAT_LE);
        CHECKFMT(F32MSB, FLOAT_BE);
        #undef CHECKFMT
        default: break;
    }
    return SDL_AUDIO_S16;  // oh well.
}

static void QSA_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    // Detect amount of available devices
    // this value can be changed in the runtime
    int num_cards = 0;
    (void) snd_cards_list(NULL, 0, &alloc_num_cards);
    bool isstack = false;
    int *cards = SDL_small_alloc(int, num_cards, &isstack);
    if (!cards) {
        return;  // we're in trouble.
    }
    int overflow_cards = 0;
    const int total_num_cards = snd_cards_list(cards, num_cards, &overflow_cards);
    // if overflow_cards > 0 or total_num_cards > num_cards, it changed at the last moment; oh well, we lost some.
    num_cards = SDL_min(num_cards, total_num_cards);  // ...but make sure it didn't _shrink_.

    // If io-audio manager is not running we will get 0 as number of available audio devices
    if (num_cards == 0) {   // not any available audio devices?
        SDL_small_free(cards, isstack);
        return;
    }

    // Find requested devices by type
    for (int it = 0; it < num_cards; it++) {
        const int card = cards[it];
        for (uint32_t deviceno = 0; ; deviceno++) {
            int32_t status;
            char name[QSA_MAX_NAME_LENGTH];

            status = snd_card_get_longname(card, name, sizeof (name));
            if (status == EOK) {
                snd_pcm_t *handle;

                // Add device number to device name
                char fullname[QSA_MAX_NAME_LENGTH + 32];
                SDL_snprintf(fullname, sizeof (fullname), "%s d%d", name, (int) deviceno);

                // Check if this device id could play anything
                bool recording = false;
                status = snd_pcm_open(&handle, card, deviceno, SND_PCM_OPEN_PLAYBACK);
                if (status != EOK) {  // no? See if it's a recording device instead.
                    #if 0  // !!! FIXME: most of this code has support for recording devices, but there's no RecordDevice, etc functions. Fill them in!
                    status = snd_pcm_open(&handle, card, deviceno, SND_PCM_OPEN_CAPTURE);
                    if (status == EOK) {
                        recording = true;
                    }
                    #endif
                }

                if (status == EOK) {
                    SDL_AudioSpec spec;
                    SDL_zero(spec);
                    SDL_AudioSpec *pspec = &spec;
                    snd_pcm_channel_setup_t csetup;
                    SDL_zero(csetup);
                    csetup.channel = recording ? SND_PCM_CHANNEL_CAPTURE : SND_PCM_CHANNEL_PLAYBACK;

                    if (snd_pcm_plugin_setup(device->hidden->audio_handle, &csetup) < 0) {
                        pspec = NULL;  // go on without spec info.
                    } else {
                        spec.format = QnxFormatToSDLFormat(csetup.format.format);
                        spec.channels = csetup.format.channels;
                        spec.freq = csetup.format.rate;
                    }

                    status = snd_pcm_close(handle);
                    if (status == EOK) {
                        // !!! FIXME: I'm assuming each of these values are way less than 0xFFFF. Fix this if not.
                        SDL_assert(card <= 0xFFFF);
                        SDL_assert(deviceno <= 0xFFFF);
                        const Uint32 sdlhandle = ((Uint32) card) | (((Uint32) deviceno) << 16);
                        SDL_AddAudioDevice(recording, fullname, pspec, (void *) ((size_t) sdlhandle));
                    }
                } else {
                    // Check if we got end of devices list
                    if (status == -ENOENT) {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    SDL_small_free(cards, isstack);

    // Try to open the "preferred" devices, which will tell us the card/device pairs for the default devices.
    snd_pcm_t handle;
    int cardno, deviceno;
    if (snd_pcm_open_preferred(&handle, &cardno, &deviceno, SND_PCM_OPEN_PLAYBACK) == 0) {
        snd_pcm_close(handle);
        // !!! FIXME: I'm assuming each of these values are way less than 0xFFFF. Fix this if not.
        SDL_assert(cardno <= 0xFFFF);
        SDL_assert(deviceno <= 0xFFFF);
        const Uint32 sdlhandle = ((Uint32) card) | (((Uint32) deviceno) << 16);
        *default_playback = SDL_FindPhysicalAudioDeviceByHandle((void *) ((size_t) sdlhandle));
    }

    if (snd_pcm_open_preferred(&handle, &cardno, &deviceno, SND_PCM_OPEN_CAPTURE) == 0) {
        snd_pcm_close(handle);
        // !!! FIXME: I'm assuming each of these values are way less than 0xFFFF. Fix this if not.
        SDL_assert(cardno <= 0xFFFF);
        SDL_assert(deviceno <= 0xFFFF);
        const Uint32 sdlhandle = ((Uint32) card) | (((Uint32) deviceno) << 16);
        *default_recording = SDL_FindPhysicalAudioDeviceByHandle((void *) ((size_t) sdlhandle));
    }
}

static void QSA_Deinitialize(void)
{
    // nothing to do here atm.
}

static bool QSA_Init(SDL_AudioDriverImpl * impl)
{
    impl->DetectDevices = QSA_DetectDevices;
    impl->OpenDevice = QSA_OpenDevice;
    impl->ThreadInit = QSA_ThreadInit;
    impl->WaitDevice = QSA_WaitDevice;
    impl->PlayDevice = QSA_PlayDevice;
    impl->GetDeviceBuf = QSA_GetDeviceBuf;
    impl->CloseDevice = QSA_CloseDevice;
    impl->Deinitialize = QSA_Deinitialize;

    // !!! FIXME: most of this code has support for recording devices, but there's no RecordDevice, etc functions. Fill them in!
    //impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap QSAAUDIO_bootstrap = {
    "qsa", "QNX QSA Audio", QSA_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_QNX

