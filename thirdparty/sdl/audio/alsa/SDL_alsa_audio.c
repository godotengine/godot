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

#ifdef SDL_AUDIO_DRIVER_ALSA

#ifndef SDL_ALSA_NON_BLOCKING
#define SDL_ALSA_NON_BLOCKING 0
#endif

// without the thread, you will detect devices on startup, but will not get further hotplug events. But that might be okay.
#ifndef SDL_ALSA_HOTPLUG_THREAD
#define SDL_ALSA_HOTPLUG_THREAD 1
#endif

// this turns off debug logging completely (but by default this goes to the bitbucket).
#ifndef SDL_ALSA_DEBUG
#define SDL_ALSA_DEBUG 1
#endif

#include "../SDL_sysaudio.h"
#include "SDL_alsa_audio.h"

#if SDL_ALSA_DEBUG
#define LOGDEBUG(...) SDL_LogDebug(SDL_LOG_CATEGORY_AUDIO, "ALSA: " __VA_ARGS__)
#else
#define LOGDEBUG(...)
#endif

//TODO: cleanup once the code settled down

static int (*ALSA_snd_pcm_open)(snd_pcm_t **, const char *, snd_pcm_stream_t, int);
static int (*ALSA_snd_pcm_close)(snd_pcm_t *pcm);
static int (*ALSA_snd_pcm_start)(snd_pcm_t *pcm);
static snd_pcm_sframes_t (*ALSA_snd_pcm_writei)(snd_pcm_t *, const void *, snd_pcm_uframes_t);
static snd_pcm_sframes_t (*ALSA_snd_pcm_readi)(snd_pcm_t *, void *, snd_pcm_uframes_t);
static int (*ALSA_snd_pcm_recover)(snd_pcm_t *, int, int);
static int (*ALSA_snd_pcm_prepare)(snd_pcm_t *);
static int (*ALSA_snd_pcm_drain)(snd_pcm_t *);
static const char *(*ALSA_snd_strerror)(int);
static size_t (*ALSA_snd_pcm_hw_params_sizeof)(void);
static size_t (*ALSA_snd_pcm_sw_params_sizeof)(void);
static void (*ALSA_snd_pcm_hw_params_copy)(snd_pcm_hw_params_t *, const snd_pcm_hw_params_t *);
static int (*ALSA_snd_pcm_hw_params_any)(snd_pcm_t *, snd_pcm_hw_params_t *);
static int (*ALSA_snd_pcm_hw_params_set_access)(snd_pcm_t *, snd_pcm_hw_params_t *, snd_pcm_access_t);
static int (*ALSA_snd_pcm_hw_params_set_format)(snd_pcm_t *, snd_pcm_hw_params_t *, snd_pcm_format_t);
static int (*ALSA_snd_pcm_hw_params_set_channels)(snd_pcm_t *, snd_pcm_hw_params_t *, unsigned int);
static int (*ALSA_snd_pcm_hw_params_get_channels)(const snd_pcm_hw_params_t *, unsigned int *);
static int (*ALSA_snd_pcm_hw_params_set_rate_near)(snd_pcm_t *, snd_pcm_hw_params_t *, unsigned int *, int *);
static int (*ALSA_snd_pcm_hw_params_set_period_size_near)(snd_pcm_t *, snd_pcm_hw_params_t *, snd_pcm_uframes_t *, int *);
static int (*ALSA_snd_pcm_hw_params_get_period_size)(const snd_pcm_hw_params_t *, snd_pcm_uframes_t *, int *);
static int (*ALSA_snd_pcm_hw_params_set_periods_min)(snd_pcm_t *, snd_pcm_hw_params_t *, unsigned int *, int *);
static int (*ALSA_snd_pcm_hw_params_set_periods_first)(snd_pcm_t *, snd_pcm_hw_params_t *, unsigned int *, int *);
static int (*ALSA_snd_pcm_hw_params_get_periods)(const snd_pcm_hw_params_t *, unsigned int *, int *);
static int (*ALSA_snd_pcm_hw_params_set_buffer_size_near)(snd_pcm_t *pcm, snd_pcm_hw_params_t *, snd_pcm_uframes_t *);
static int (*ALSA_snd_pcm_hw_params_get_buffer_size)(const snd_pcm_hw_params_t *, snd_pcm_uframes_t *);
static int (*ALSA_snd_pcm_hw_params)(snd_pcm_t *, snd_pcm_hw_params_t *);
static int (*ALSA_snd_pcm_sw_params_current)(snd_pcm_t *,
                                             snd_pcm_sw_params_t *);
static int (*ALSA_snd_pcm_sw_params_set_start_threshold)(snd_pcm_t *, snd_pcm_sw_params_t *, snd_pcm_uframes_t);
static int (*ALSA_snd_pcm_sw_params)(snd_pcm_t *, snd_pcm_sw_params_t *);
static int (*ALSA_snd_pcm_nonblock)(snd_pcm_t *, int);
static int (*ALSA_snd_pcm_wait)(snd_pcm_t *, int);
static int (*ALSA_snd_pcm_sw_params_set_avail_min)(snd_pcm_t *, snd_pcm_sw_params_t *, snd_pcm_uframes_t);
static int (*ALSA_snd_pcm_reset)(snd_pcm_t *);
static int (*ALSA_snd_device_name_hint)(int, const char *, void ***);
static char *(*ALSA_snd_device_name_get_hint)(const void *, const char *);
static int (*ALSA_snd_device_name_free_hint)(void **);
static snd_pcm_sframes_t (*ALSA_snd_pcm_avail)(snd_pcm_t *);
static size_t (*ALSA_snd_ctl_card_info_sizeof)(void);
static size_t (*ALSA_snd_pcm_info_sizeof)(void);
static int (*ALSA_snd_card_next)(int*);
static int (*ALSA_snd_ctl_open)(snd_ctl_t **,const char *,int);
static int (*ALSA_snd_ctl_close)(snd_ctl_t *);
static int (*ALSA_snd_ctl_card_info)(snd_ctl_t *, snd_ctl_card_info_t *);
static int (*ALSA_snd_ctl_pcm_next_device)(snd_ctl_t *, int *);
static unsigned int (*ALSA_snd_pcm_info_get_subdevices_count)(const snd_pcm_info_t *);
static void (*ALSA_snd_pcm_info_set_device)(snd_pcm_info_t *, unsigned int);
static void (*ALSA_snd_pcm_info_set_subdevice)(snd_pcm_info_t *, unsigned int);
static void (*ALSA_snd_pcm_info_set_stream)(snd_pcm_info_t *, snd_pcm_stream_t);
static int (*ALSA_snd_ctl_pcm_info)(snd_ctl_t *, snd_pcm_info_t *);
static unsigned int (*ALSA_snd_pcm_info_get_subdevices_count)(const snd_pcm_info_t *);
static const char *(*ALSA_snd_ctl_card_info_get_id)(const snd_ctl_card_info_t *);
static const char *(*ALSA_snd_pcm_info_get_name)(const snd_pcm_info_t *);
static const char *(*ALSA_snd_pcm_info_get_subdevice_name)(const snd_pcm_info_t *);
static const char *(*ALSA_snd_ctl_card_info_get_name)(const snd_ctl_card_info_t *);
static void (*ALSA_snd_ctl_card_info_clear)(snd_ctl_card_info_t *);
static int (*ALSA_snd_pcm_hw_free)(snd_pcm_t *);
static int (*ALSA_snd_pcm_hw_params_set_channels_near)(snd_pcm_t *, snd_pcm_hw_params_t *, unsigned int *);
static snd_pcm_chmap_query_t **(*ALSA_snd_pcm_query_chmaps)(snd_pcm_t *pcm);
static void (*ALSA_snd_pcm_free_chmaps)(snd_pcm_chmap_query_t **maps);
static int (*ALSA_snd_pcm_set_chmap)(snd_pcm_t *, const snd_pcm_chmap_t *);
static int (*ALSA_snd_pcm_chmap_print)(const snd_pcm_chmap_t *, size_t, char *);

#ifdef SDL_AUDIO_DRIVER_ALSA_DYNAMIC
#define snd_pcm_hw_params_sizeof ALSA_snd_pcm_hw_params_sizeof
#define snd_pcm_sw_params_sizeof ALSA_snd_pcm_sw_params_sizeof

static const char *alsa_library = SDL_AUDIO_DRIVER_ALSA_DYNAMIC;
static SDL_SharedObject *alsa_handle = NULL;

static bool load_alsa_sym(const char *fn, void **addr)
{
    *addr = SDL_LoadFunction(alsa_handle, fn);
    if (!*addr) {
        // Don't call SDL_SetError(): SDL_LoadFunction already did.
        return false;
    }

    return true;
}

// cast funcs to char* first, to please GCC's strict aliasing rules.
#define SDL_ALSA_SYM(x)                                 \
    if (!load_alsa_sym(#x, (void **)(char *)&ALSA_##x)) \
        return false
#else
#define SDL_ALSA_SYM(x) ALSA_##x = x
#endif

static bool load_alsa_syms(void)
{
    SDL_ALSA_SYM(snd_pcm_open);
    SDL_ALSA_SYM(snd_pcm_close);
    SDL_ALSA_SYM(snd_pcm_start);
    SDL_ALSA_SYM(snd_pcm_writei);
    SDL_ALSA_SYM(snd_pcm_readi);
    SDL_ALSA_SYM(snd_pcm_recover);
    SDL_ALSA_SYM(snd_pcm_prepare);
    SDL_ALSA_SYM(snd_pcm_drain);
    SDL_ALSA_SYM(snd_strerror);
    SDL_ALSA_SYM(snd_pcm_hw_params_sizeof);
    SDL_ALSA_SYM(snd_pcm_sw_params_sizeof);
    SDL_ALSA_SYM(snd_pcm_hw_params_copy);
    SDL_ALSA_SYM(snd_pcm_hw_params_any);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_access);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_format);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_channels);
    SDL_ALSA_SYM(snd_pcm_hw_params_get_channels);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_rate_near);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_period_size_near);
    SDL_ALSA_SYM(snd_pcm_hw_params_get_period_size);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_periods_min);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_periods_first);
    SDL_ALSA_SYM(snd_pcm_hw_params_get_periods);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_buffer_size_near);
    SDL_ALSA_SYM(snd_pcm_hw_params_get_buffer_size);
    SDL_ALSA_SYM(snd_pcm_hw_params);
    SDL_ALSA_SYM(snd_pcm_sw_params_current);
    SDL_ALSA_SYM(snd_pcm_sw_params_set_start_threshold);
    SDL_ALSA_SYM(snd_pcm_sw_params);
    SDL_ALSA_SYM(snd_pcm_nonblock);
    SDL_ALSA_SYM(snd_pcm_wait);
    SDL_ALSA_SYM(snd_pcm_sw_params_set_avail_min);
    SDL_ALSA_SYM(snd_pcm_reset);
    SDL_ALSA_SYM(snd_device_name_hint);
    SDL_ALSA_SYM(snd_device_name_get_hint);
    SDL_ALSA_SYM(snd_device_name_free_hint);
    SDL_ALSA_SYM(snd_pcm_avail);
    SDL_ALSA_SYM(snd_ctl_card_info_sizeof);
    SDL_ALSA_SYM(snd_pcm_info_sizeof);
    SDL_ALSA_SYM(snd_card_next);
    SDL_ALSA_SYM(snd_ctl_open);
    SDL_ALSA_SYM(snd_ctl_close);
    SDL_ALSA_SYM(snd_ctl_card_info);
    SDL_ALSA_SYM(snd_ctl_pcm_next_device);
    SDL_ALSA_SYM(snd_pcm_info_get_subdevices_count);
    SDL_ALSA_SYM(snd_pcm_info_set_device);
    SDL_ALSA_SYM(snd_pcm_info_set_subdevice);
    SDL_ALSA_SYM(snd_pcm_info_set_stream);
    SDL_ALSA_SYM(snd_ctl_pcm_info);
    SDL_ALSA_SYM(snd_pcm_info_get_subdevices_count);
    SDL_ALSA_SYM(snd_ctl_card_info_get_id);
    SDL_ALSA_SYM(snd_pcm_info_get_name);
    SDL_ALSA_SYM(snd_pcm_info_get_subdevice_name);
    SDL_ALSA_SYM(snd_ctl_card_info_get_name);
    SDL_ALSA_SYM(snd_ctl_card_info_clear);
    SDL_ALSA_SYM(snd_pcm_hw_free);
    SDL_ALSA_SYM(snd_pcm_hw_params_set_channels_near);
    SDL_ALSA_SYM(snd_pcm_query_chmaps);
    SDL_ALSA_SYM(snd_pcm_free_chmaps);
    SDL_ALSA_SYM(snd_pcm_set_chmap);
    SDL_ALSA_SYM(snd_pcm_chmap_print);

    return true;
}

#undef SDL_ALSA_SYM

#ifdef SDL_AUDIO_DRIVER_ALSA_DYNAMIC

static void UnloadALSALibrary(void)
{
    if (alsa_handle) {
        SDL_UnloadObject(alsa_handle);
        alsa_handle = NULL;
    }
}

static bool LoadALSALibrary(void)
{
    bool retval = true;
    if (!alsa_handle) {
        alsa_handle = SDL_LoadObject(alsa_library);
        if (!alsa_handle) {
            retval = false;
            // Don't call SDL_SetError(): SDL_LoadObject already did.
        } else {
            retval = load_alsa_syms();
            if (!retval) {
                UnloadALSALibrary();
            }
        }
    }
    return retval;
}

#else

static void UnloadALSALibrary(void)
{
}

static bool LoadALSALibrary(void)
{
    load_alsa_syms();
    return true;
}

#endif // SDL_AUDIO_DRIVER_ALSA_DYNAMIC

static const char *ALSA_device_prefix = NULL;
static void ALSA_guess_device_prefix(void)
{
    if (ALSA_device_prefix) {
        return;  // already calculated.
    }

    // Apparently there are several different ways that ALSA lists
    //  actual hardware. It could be prefixed with "hw:" or "default:"
    //  or "sysdefault:" and maybe others. Go through the list and see
    //  if we can find a preferred prefix for the system.

    static const char *const prefixes[] = {
        "hw:", "sysdefault:", "default:"
    };

    void **hints = NULL;
    if (ALSA_snd_device_name_hint(-1, "pcm", &hints) == 0) {
        for (int i = 0; hints[i]; i++) {
            char *name = ALSA_snd_device_name_get_hint(hints[i], "NAME");
            if (name) {
                for (int j = 0; j < SDL_arraysize(prefixes); j++) {
                    const char *prefix = prefixes[j];
                    const size_t prefixlen = SDL_strlen(prefix);
                    if (SDL_strncmp(name, prefix, prefixlen) == 0) {
                        ALSA_device_prefix = prefix;
                        break;
                    }
                }
                free(name); // This should NOT be SDL_free()

                if (ALSA_device_prefix) {
                    break;
                }
            }
        }
    }

    if (!ALSA_device_prefix) {
        ALSA_device_prefix = prefixes[0];  // oh well.
    }

    LOGDEBUG("device prefix is probably '%s'", ALSA_device_prefix);
}

typedef struct ALSA_Device
{
    // the unicity key is the couple (id,recording)
    char *id; // empty means canonical default
    char *name;
    bool recording;
    struct ALSA_Device *next;
} ALSA_Device;

static const ALSA_Device default_playback_handle = {
    "",
    "default",
    false,
    NULL
};

static const ALSA_Device default_recording_handle = {
    "",
    "default",
    true,
    NULL
};

// TODO: Figure out the "right"(TM) way. For the moment we presume that if a system is using a
// software mixer for application audio sharing which is not the linux native alsa[dmix], for
// instance jack/pulseaudio2[pipewire]/pulseaudio1/esound/etc, we expect the system integrators did
// configure the canonical default to the right alsa PCM plugin for their software mixer.
//
// All the above may be completely wrong.
static char *get_pcm_str(void *handle)
{
    SDL_assert(handle != NULL);  // SDL2 used NULL to mean "default" but that's not true in SDL3.
    ALSA_Device *dev = (ALSA_Device *)handle;
    char *pcm_str = NULL;

    if (SDL_strlen(dev->id) == 0) {
        // If the user does not want to go thru the default PCM or the canonical default, the
        // the configuration space being _massive_, give the user the ability to specify
        // its own PCMs using environment variables. It will have to fit SDL constraints though.
        const char *devname = SDL_GetHint(dev->recording ? SDL_HINT_AUDIO_ALSA_DEFAULT_RECORDING_DEVICE : SDL_HINT_AUDIO_ALSA_DEFAULT_PLAYBACK_DEVICE);
        if (!devname) {
            devname = SDL_GetHint(SDL_HINT_AUDIO_ALSA_DEFAULT_DEVICE);
            if (!devname) {
                devname = "default";
            }
        }
        pcm_str = SDL_strdup(devname);
    } else {
        SDL_asprintf(&pcm_str, "%sCARD=%s", ALSA_device_prefix, dev->id);
    }
    return pcm_str;
}

// This function waits until it is possible to write a full sound buffer
static bool ALSA_WaitDevice(SDL_AudioDevice *device)
{
    const int fulldelay = (int) ((((Uint64) device->sample_frames) * 1000) / device->spec.freq);
    const int delay = SDL_max(fulldelay, 10);

    while (!SDL_GetAtomicInt(&device->shutdown)) {
        const int rc = ALSA_snd_pcm_wait(device->hidden->pcm, delay);
        if (rc < 0 && (rc != -EAGAIN)) {
            const int status = ALSA_snd_pcm_recover(device->hidden->pcm, rc, 0);
            if (status < 0) {
                // Hmm, not much we can do - abort
                SDL_LogError(SDL_LOG_CATEGORY_AUDIO, "ALSA: snd_pcm_wait failed (unrecoverable): %s", ALSA_snd_strerror(rc));
                return false;
            }
            continue;
        }

        if (rc > 0) {
            break;  // ready to go!
        }

        // Timed out! Make sure we aren't shutting down and then wait again.
    }

    return true;
}

static bool ALSA_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    SDL_assert(buffer == device->hidden->mixbuf);
    Uint8 *sample_buf = (Uint8 *) buffer;  // !!! FIXME: deal with this without casting away constness
    const int frame_size = SDL_AUDIO_FRAMESIZE(device->spec);
    snd_pcm_uframes_t frames_left = (snd_pcm_uframes_t) (buflen / frame_size);

    while ((frames_left > 0) && !SDL_GetAtomicInt(&device->shutdown)) {
        const int rc = ALSA_snd_pcm_writei(device->hidden->pcm, sample_buf, frames_left);
        //SDL_LogInfo(SDL_LOG_CATEGORY_AUDIO, "ALSA PLAYDEVICE: WROTE %d of %d bytes", (rc >= 0) ? ((int) (rc * frame_size)) : rc, (int) (frames_left * frame_size));
        SDL_assert(rc != 0);  // assuming this can't happen if we used snd_pcm_wait and queried for available space.
        if (rc < 0) {
            SDL_assert(rc != -EAGAIN);  // assuming this can't happen if we used snd_pcm_wait and queried for available space. snd_pcm_recover won't handle it!
            const int status = ALSA_snd_pcm_recover(device->hidden->pcm, rc, 0);
            if (status < 0) {
                // Hmm, not much we can do - abort
                SDL_LogError(SDL_LOG_CATEGORY_AUDIO, "ALSA write failed (unrecoverable): %s", ALSA_snd_strerror(rc));
                return false;
            }
            continue;
        }

        sample_buf += rc * frame_size;
        frames_left -= rc;
    }

    return true;
}

static Uint8 *ALSA_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    snd_pcm_sframes_t rc = ALSA_snd_pcm_avail(device->hidden->pcm);
    if (rc <= 0) {
        // Wait a bit and try again, maybe the hardware isn't quite ready yet?
        SDL_Delay(1);

        rc = ALSA_snd_pcm_avail(device->hidden->pcm);
        if (rc <= 0) {
            // We'll catch it next time
            *buffer_size = 0;
            return NULL;
        }
    }

    const int requested_frames = SDL_min(device->sample_frames, rc);
    const int requested_bytes = requested_frames * SDL_AUDIO_FRAMESIZE(device->spec);
    SDL_assert(requested_bytes <= *buffer_size);
    //SDL_LogInfo(SDL_LOG_CATEGORY_AUDIO, "ALSA GETDEVICEBUF: NEED %d BYTES", requested_bytes);
    *buffer_size = requested_bytes;
    return device->hidden->mixbuf;
}

static int ALSA_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    const int frame_size = SDL_AUDIO_FRAMESIZE(device->spec);
    SDL_assert((buflen % frame_size) == 0);

    const snd_pcm_sframes_t total_available = ALSA_snd_pcm_avail(device->hidden->pcm);
    const int total_frames = SDL_min(buflen / frame_size, total_available);

    const int rc = ALSA_snd_pcm_readi(device->hidden->pcm, buffer, total_frames);

    SDL_assert(rc != -EAGAIN);  // assuming this can't happen if we used snd_pcm_wait and queried for available space. snd_pcm_recover won't handle it!

    if (rc < 0) {
        const int status = ALSA_snd_pcm_recover(device->hidden->pcm, rc, 0);
        if (status < 0) {
            // Hmm, not much we can do - abort
            SDL_LogError(SDL_LOG_CATEGORY_AUDIO, "ALSA read failed (unrecoverable): %s", ALSA_snd_strerror(rc));
            return -1;
        }
        return 0;  // go back to WaitDevice and try again.
    }

    //SDL_LogInfo(SDL_LOG_CATEGORY_AUDIO, "ALSA: recorded %d bytes", rc * frame_size);

    return rc * frame_size;
}

static void ALSA_FlushRecording(SDL_AudioDevice *device)
{
    ALSA_snd_pcm_reset(device->hidden->pcm);
}

static void ALSA_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->pcm) {
            // Wait for the submitted audio to drain. ALSA_snd_pcm_drop() can hang, so don't use that.
            SDL_Delay(((device->sample_frames * 1000) / device->spec.freq) * 2);
            ALSA_snd_pcm_close(device->hidden->pcm);
        }
        SDL_free(device->hidden->mixbuf);
        SDL_free(device->hidden);
    }
}


// To make easier to track parameters during the whole alsa pcm configuration:
struct ALSA_pcm_cfg_ctx {
    SDL_AudioDevice *device;

    snd_pcm_hw_params_t *hwparams;
    snd_pcm_sw_params_t *swparams;

    SDL_AudioFormat matched_sdl_format;
    unsigned int chans_n;
    unsigned int target_chans_n;
    unsigned int rate;
    snd_pcm_uframes_t persize; // alsa period size, SDL audio device sample_frames
    snd_pcm_chmap_query_t **chmap_queries;
    unsigned int sdl_chmap[SDL_AUDIO_ALSA__CHMAP_CHANS_N_MAX];
    unsigned int alsa_chmap_installed[SDL_AUDIO_ALSA__CHMAP_CHANS_N_MAX];

    unsigned int periods;
};
// The following are SDL channel maps with alsa position values, from 0 channels to 8 channels.
// See SDL3/SDL_audio.h
// Strictly speaking those are "parameters" of channel maps, like alsa hwparams and swparams, they
// have to be "reduced/refined" until an exact channel map. Only the 6 channels map requires such
// "reduction/refine".
static enum snd_pcm_chmap_position sdl_channel_maps[SDL_AUDIO_ALSA__SDL_CHMAPS_N][SDL_AUDIO_ALSA__CHMAP_CHANS_N_MAX] = {
    // 0 channels
    {
        0
    },
    // 1 channel
    {
        SND_CHMAP_MONO,
    },
    // 2 channels
    {
        SND_CHMAP_FL,
        SND_CHMAP_FR,
    },
    // 3 channels
    {
        SND_CHMAP_FL,
        SND_CHMAP_FR,
        SND_CHMAP_LFE,
    },
    // 4 channels
    {
        SND_CHMAP_FL,
        SND_CHMAP_FR,
        SND_CHMAP_RL,
        SND_CHMAP_RR,
    },
    // 5 channels
    {
        SND_CHMAP_FL,
        SND_CHMAP_FR,
        SND_CHMAP_LFE,
        SND_CHMAP_RL,
        SND_CHMAP_RR,
    },
    // 6 channels
    // XXX: here we encode not a uniq channel map but a set of channel maps. We will reduce it each
    // time we are going to work with an alsa 6 channels map.
    {
        SND_CHMAP_FL,
        SND_CHMAP_FR,
        SND_CHMAP_FC,
        SND_CHMAP_LFE,
        // The 2 following channel positions are (SND_CHMAP_SL,SND_CHMAP_SR) or
        // (SND_CHMAP_RL,SND_CHMAP_RR)
        SND_CHMAP_UNKNOWN,
        SND_CHMAP_UNKNOWN,
    },
    // 7 channels
    {
        SND_CHMAP_FL,
        SND_CHMAP_FR,
        SND_CHMAP_FC,
        SND_CHMAP_LFE,
        SND_CHMAP_RC,
        SND_CHMAP_SL,
        SND_CHMAP_SR,
    },
    // 8 channels
    {
        SND_CHMAP_FL,
        SND_CHMAP_FR,
        SND_CHMAP_FC,
        SND_CHMAP_LFE,
        SND_CHMAP_RL,
        SND_CHMAP_RR,
        SND_CHMAP_SL,
        SND_CHMAP_SR,
    },
};

// Helper for the function right below.
static bool has_pos(const unsigned int *chmap, unsigned int pos)
{
    for (unsigned int chan_idx = 0; ; chan_idx++) {
        if (chan_idx == 6) {
            return false;
        }
        if (chmap[chan_idx] == pos) {
            return true;
        }
    }
    SDL_assert(!"Shouldn't hit this code.");
    return false;
}

// XXX: Each time we are going to work on an alsa 6 channels map, we must reduce the set of channel
// maps which is encoded in sdl_channel_maps[6] to a uniq one.
#define HAVE_NONE 0
#define HAVE_REAR 1
#define HAVE_SIDE 2
#define HAVE_BOTH 3
static void sdl_6chans_set_rear_or_side_channels_from_alsa_6chans(unsigned int *sdl_6chans, const unsigned int *alsa_6chans)
{
    // For alsa channel maps with 6 channels and with SND_CHMAP_FL,SND_CHMAP_FR,SND_CHMAP_FC,
    // SND_CHMAP_LFE, reduce our 6 channels maps to a uniq one.
    if ( !has_pos(alsa_6chans, SND_CHMAP_FL) ||
         !has_pos(alsa_6chans, SND_CHMAP_FR) ||
         !has_pos(alsa_6chans, SND_CHMAP_FC) ||
         !has_pos(alsa_6chans, SND_CHMAP_LFE)) {
        sdl_6chans[4] = SND_CHMAP_UNKNOWN;
        sdl_6chans[5] = SND_CHMAP_UNKNOWN;
        LOGDEBUG("6channels:unsupported channel map");
        return;
    }

    unsigned int state = HAVE_NONE;
    for (unsigned int chan_idx = 0; chan_idx < 6; chan_idx++) {
        if ((alsa_6chans[chan_idx] == SND_CHMAP_SL) || (alsa_6chans[chan_idx] == SND_CHMAP_SR)) {
            if (state == HAVE_NONE) {
                state = HAVE_SIDE;
            } else if (state == HAVE_REAR) {
                state = HAVE_BOTH;
                break;
            }
        } else if ((alsa_6chans[chan_idx] == SND_CHMAP_RL) || (alsa_6chans[chan_idx] == SND_CHMAP_RR)) {
            if (state == HAVE_NONE) {
                state = HAVE_REAR;
            } else if (state == HAVE_SIDE) {
                state = HAVE_BOTH;
                break;
            }
        }
    }

    if ((state == HAVE_BOTH) || (state == HAVE_NONE)) {
        sdl_6chans[4] = SND_CHMAP_UNKNOWN;
        sdl_6chans[5] = SND_CHMAP_UNKNOWN;
        LOGDEBUG("6channels:unsupported channel map");
    } else if (state == HAVE_REAR) {
        sdl_6chans[4] = SND_CHMAP_RL;
        sdl_6chans[5] = SND_CHMAP_RR;
        LOGDEBUG("6channels:sdl map set to rear");
    } else { // state == HAVE_SIDE
        sdl_6chans[4] = SND_CHMAP_SL;
        sdl_6chans[5] = SND_CHMAP_SR;
        LOGDEBUG("6channels:sdl map set to side");
    }
}
#undef HAVE_NONE
#undef HAVE_REAR
#undef HAVE_SIDE
#undef HAVE_BOTH

static void swizzle_map_compute_alsa_subscan(const struct ALSA_pcm_cfg_ctx *ctx, int *swizzle_map, unsigned int sdl_pos_idx)
{
    swizzle_map[sdl_pos_idx] = -1;
    for (unsigned int alsa_pos_idx = 0; ; alsa_pos_idx++) {
        SDL_assert(alsa_pos_idx != ctx->chans_n);  // no 0 channels or not found matching position should happen here (actually enforce playback/recording symmetry).
        if (ctx->alsa_chmap_installed[alsa_pos_idx] == ctx->sdl_chmap[sdl_pos_idx]) {
            LOGDEBUG("swizzle SDL %u <-> alsa %u", sdl_pos_idx,alsa_pos_idx);
            swizzle_map[sdl_pos_idx] = (int) alsa_pos_idx;
            return;
        }
    }
}

// XXX: this must stay playback/recording symetric.
static void swizzle_map_compute(const struct ALSA_pcm_cfg_ctx *ctx, int *swizzle_map, bool *needs_swizzle)
{
    *needs_swizzle = false;
    for (unsigned int sdl_pos_idx = 0; sdl_pos_idx != ctx->chans_n; sdl_pos_idx++) {
        swizzle_map_compute_alsa_subscan(ctx, swizzle_map, sdl_pos_idx);
        if (swizzle_map[sdl_pos_idx] != sdl_pos_idx) {
            *needs_swizzle = true;
        }
    }
}

#define CHMAP_INSTALLED 0
#define CHANS_N_NEXT            1
#define CHMAP_NOT_FOUND 2
// Should always be a queried alsa channel map unless the queried alsa channel map was of type VAR,
// namely we can program the channel positions directly from the SDL channel map.
static int alsa_chmap_install(struct ALSA_pcm_cfg_ctx *ctx, const unsigned int *chmap)
{
    bool isstack;
    snd_pcm_chmap_t *chmap_to_install = (snd_pcm_chmap_t*)SDL_small_alloc(unsigned int, 1 + ctx->chans_n, &isstack);
    if (!chmap_to_install) {
        return -1;
    }

    chmap_to_install->channels = ctx->chans_n;
    SDL_memcpy(chmap_to_install->pos, chmap, sizeof (unsigned int) * ctx->chans_n);

    #if SDL_ALSA_DEBUG
    char logdebug_chmap_str[128];
    ALSA_snd_pcm_chmap_print(chmap_to_install,sizeof(logdebug_chmap_str),logdebug_chmap_str);
    LOGDEBUG("channel map to install:%s",logdebug_chmap_str);
    #endif

    int status = ALSA_snd_pcm_set_chmap(ctx->device->hidden->pcm, chmap_to_install);
    if (status < 0) {
        SDL_SetError("ALSA: failed to install channel map: %s", ALSA_snd_strerror(status));
        return -1;
    }
    SDL_memcpy(ctx->alsa_chmap_installed, chmap, ctx->chans_n * sizeof (unsigned int));

    SDL_small_free(chmap_to_install, isstack);
    return CHMAP_INSTALLED;
}

// We restrict the alsa channel maps because in the unordered matches we do only simple accounting.
// In the end, this will handle mostly alsa channel maps with more than one SND_CHMAP_NA position fillers.
static bool alsa_chmap_has_duplicate_position(const struct ALSA_pcm_cfg_ctx *ctx, const unsigned int *pos)
{
    if (ctx->chans_n < 2) {// we need at least 2 positions
        LOGDEBUG("channel map:no duplicate");
        return false;
    }

    for (unsigned int chan_idx = 1; chan_idx != ctx->chans_n; chan_idx++) {
        for (unsigned int seen_idx = 0; seen_idx != chan_idx; seen_idx++) {
            if (pos[seen_idx] == pos[chan_idx]) {
                LOGDEBUG("channel map:have duplicate");
                return true;
            }
        }
    }

    LOGDEBUG("channel map:no duplicate");
    return false;
}

static int alsa_chmap_cfg_ordered_fixed_or_paired(struct ALSA_pcm_cfg_ctx *ctx)
{
    for (snd_pcm_chmap_query_t **chmap_query = ctx->chmap_queries; *chmap_query; chmap_query++) {
        if ( ((*chmap_query)->map.channels != ctx->chans_n) ||
             (((*chmap_query)->type != SND_CHMAP_TYPE_FIXED) && ((*chmap_query)->type != SND_CHMAP_TYPE_PAIRED)) ) {
            continue;
        }

        #if SDL_ALSA_DEBUG
        char logdebug_chmap_str[128];
        ALSA_snd_pcm_chmap_print(&(*chmap_query)->map,sizeof(logdebug_chmap_str),logdebug_chmap_str);
        LOGDEBUG("channel map:ordered:fixed|paired:%s",logdebug_chmap_str);
        #endif

        for (int i = 0; i < ctx->chans_n; i++) {
            ctx->sdl_chmap[i] = (unsigned int) sdl_channel_maps[ctx->chans_n][i];
        }

        unsigned int *alsa_chmap = (*chmap_query)->map.pos;
        if (ctx->chans_n == 6) {
            sdl_6chans_set_rear_or_side_channels_from_alsa_6chans(ctx->sdl_chmap, alsa_chmap);
        }
        if (alsa_chmap_has_duplicate_position(ctx, alsa_chmap)) {
            continue;
        }

        for (unsigned int chan_idx = 0; ctx->sdl_chmap[chan_idx] == alsa_chmap[chan_idx]; chan_idx++) {
            if (chan_idx == ctx->chans_n) {
                return alsa_chmap_install(ctx, alsa_chmap);
            }
        }
    }
    return CHMAP_NOT_FOUND;
}

// Here, the alsa channel positions can be programmed in the alsa frame (cf HDMI).
// If the alsa channel map is VAR, we only check we have the unordered set of channel positions we
// are looking for.
static int alsa_chmap_cfg_ordered_var(struct ALSA_pcm_cfg_ctx *ctx)
{
    for (snd_pcm_chmap_query_t **chmap_query = ctx->chmap_queries; *chmap_query; chmap_query++) {
        if (((*chmap_query)->map.channels != ctx->chans_n) || ((*chmap_query)->type != SND_CHMAP_TYPE_VAR)) {
            continue;
        }

        #if SDL_ALSA_DEBUG
        char logdebug_chmap_str[128];
        ALSA_snd_pcm_chmap_print(&(*chmap_query)->map,sizeof(logdebug_chmap_str),logdebug_chmap_str);
        LOGDEBUG("channel map:ordered:var:%s",logdebug_chmap_str);
        #endif

        for (int i = 0; i < ctx->chans_n; i++) {
            ctx->sdl_chmap[i] = (unsigned int) sdl_channel_maps[ctx->chans_n][i];
        }

        unsigned int *alsa_chmap = (*chmap_query)->map.pos;
        if (ctx->chans_n == 6) {
            sdl_6chans_set_rear_or_side_channels_from_alsa_6chans(ctx->sdl_chmap, alsa_chmap);
        }
        if (alsa_chmap_has_duplicate_position(ctx, alsa_chmap)) {
            continue;
        }

        unsigned int pos_matches_n = 0;
        for (unsigned int chan_idx = 0; chan_idx != ctx->chans_n; chan_idx++) {
            for (unsigned int subscan_chan_idx = 0; subscan_chan_idx != ctx->chans_n; subscan_chan_idx++) {
                if (ctx->sdl_chmap[chan_idx] == alsa_chmap[subscan_chan_idx]) {
                    pos_matches_n++;
                    break;
                }
            }
        }

        if (pos_matches_n == ctx->chans_n) {
            return alsa_chmap_install(ctx, ctx->sdl_chmap); // XXX: we program the SDL chmap here
        }
    }

    return CHMAP_NOT_FOUND;
}

static int alsa_chmap_cfg_ordered(struct ALSA_pcm_cfg_ctx *ctx)
{
    const int status = alsa_chmap_cfg_ordered_fixed_or_paired(ctx);
    return (status != CHMAP_NOT_FOUND) ? status : alsa_chmap_cfg_ordered_var(ctx);
}

// In the unordered case, we are just interested to get the same unordered set of alsa channel
// positions than in the SDL channel map since we will swizzle (no duplicate channel position).
static int alsa_chmap_cfg_unordered(struct ALSA_pcm_cfg_ctx *ctx)
{
    for (snd_pcm_chmap_query_t **chmap_query = ctx->chmap_queries; *chmap_query; chmap_query++) {
        if ( ((*chmap_query)->map.channels != ctx->chans_n) ||
             (((*chmap_query)->type != SND_CHMAP_TYPE_FIXED) && ((*chmap_query)->type != SND_CHMAP_TYPE_PAIRED)) ) {
            continue;
        }

        #if SDL_ALSA_DEBUG
        char logdebug_chmap_str[128];
        ALSA_snd_pcm_chmap_print(&(*chmap_query)->map,sizeof(logdebug_chmap_str),logdebug_chmap_str);
        LOGDEBUG("channel map:unordered:fixed|paired:%s",logdebug_chmap_str);
        #endif

        for (int i = 0; i < ctx->chans_n; i++) {
            ctx->sdl_chmap[i] = (unsigned int) sdl_channel_maps[ctx->chans_n][i];
        }

        unsigned int *alsa_chmap = (*chmap_query)->map.pos;
        if (ctx->chans_n == 6) {
            sdl_6chans_set_rear_or_side_channels_from_alsa_6chans(ctx->sdl_chmap, alsa_chmap);
        }

        if (alsa_chmap_has_duplicate_position(ctx, alsa_chmap)) {
            continue;
        }

        unsigned int pos_matches_n = 0;
        for (unsigned int chan_idx = 0; chan_idx != ctx->chans_n; chan_idx++) {
            for (unsigned int subscan_chan_idx = 0; subscan_chan_idx != ctx->chans_n; subscan_chan_idx++) {
                if (ctx->sdl_chmap[chan_idx] == alsa_chmap[subscan_chan_idx]) {
                    pos_matches_n++;
                    break;
                }
            }
        }

        if (pos_matches_n == ctx->chans_n) {
            return alsa_chmap_install(ctx, alsa_chmap);
        }
    }

    return CHMAP_NOT_FOUND;
}

static int alsa_chmap_cfg(struct ALSA_pcm_cfg_ctx *ctx)
{
    int status;

    ctx->chmap_queries = ALSA_snd_pcm_query_chmaps(ctx->device->hidden->pcm);
    if (ctx->chmap_queries == NULL) {
        // We couldn't query the channel map, assume no swizzle necessary
        LOGDEBUG("couldn't query channel map, swizzling off");
        return CHMAP_INSTALLED;
    }

    //----------------------------------------------------------------------------------------------
    status = alsa_chmap_cfg_ordered(ctx); // we prefer first channel maps we don't need to swizzle
    if (status == CHMAP_INSTALLED) {
        LOGDEBUG("swizzling off");
        return status;
    } else if (status != CHMAP_NOT_FOUND) {
        return status; // < 0 error code
    }

    // Fall-thru
    //----------------------------------------------------------------------------------------------
    status = alsa_chmap_cfg_unordered(ctx); // those we will have to swizzle
    if (status == CHMAP_INSTALLED) {
        LOGDEBUG("swizzling on");

        bool isstack;
        int *swizzle_map = SDL_small_alloc(int, ctx->chans_n, &isstack);
        if (!swizzle_map) {
            status = -1;
        } else {
            bool needs_swizzle;
            swizzle_map_compute(ctx, swizzle_map, &needs_swizzle); // fine grained swizzle configuration
            if (needs_swizzle) {
                // let SDL's swizzler handle this one.
                ctx->device->chmap = SDL_ChannelMapDup(swizzle_map, ctx->chans_n);
                if (!ctx->device->chmap) {
                    status = -1;
                }
            }
            SDL_small_free(swizzle_map, isstack);
        }
    }

    if (status == CHMAP_NOT_FOUND) {
        return CHANS_N_NEXT;
    }

    return status; // < 0 error code
}

#define CHANS_N_SCAN_MODE__EQUAL_OR_ABOVE_REQUESTED_CHANS_N 0  // target more hardware pressure
#define CHANS_N_SCAN_MODE__BELOW_REQUESTED_CHANS_N          1  // target less hardware pressure
#define CHANS_N_CONFIGURED      0
#define CHANS_N_NOT_CONFIGURED  1
static int ALSA_pcm_cfg_hw_chans_n_scan(struct ALSA_pcm_cfg_ctx *ctx, unsigned int mode)
{
    unsigned int target_chans_n = ctx->device->spec.channels; // we start at what was specified
    if (mode == CHANS_N_SCAN_MODE__BELOW_REQUESTED_CHANS_N) {
        target_chans_n--;
    }
    while (true) {
        if (mode == CHANS_N_SCAN_MODE__EQUAL_OR_ABOVE_REQUESTED_CHANS_N) {
            if (target_chans_n > SDL_AUDIO_ALSA__CHMAP_CHANS_N_MAX) {
                return CHANS_N_NOT_CONFIGURED;
            }
            // else: CHANS_N_SCAN_MODE__BELOW_REQUESTED_CHANS_N
        } else if (target_chans_n == 0) {
            return CHANS_N_NOT_CONFIGURED;
        }

        LOGDEBUG("target chans_n is %u", target_chans_n);

        int status = ALSA_snd_pcm_hw_params_any(ctx->device->hidden->pcm, ctx->hwparams);
        if (status < 0) {
            SDL_SetError("ALSA: Couldn't get hardware config: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // SDL only uses interleaved sample output
        status = ALSA_snd_pcm_hw_params_set_access(ctx->device->hidden->pcm, ctx->hwparams, SND_PCM_ACCESS_RW_INTERLEAVED);
        if (status < 0) {
            SDL_SetError("ALSA: Couldn't set interleaved access: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // Try for a closest match on audio format
        snd_pcm_format_t alsa_format = 0;
        const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(ctx->device->spec.format);
        ctx->matched_sdl_format = 0;
        while ((ctx->matched_sdl_format = *(closefmts++)) != 0) {
            // XXX: we are forcing the same endianness, namely we won't need byte swapping upon
            // writing/reading to/from the SDL audio buffer.
            switch (ctx->matched_sdl_format) {
            case SDL_AUDIO_U8:
                alsa_format = SND_PCM_FORMAT_U8;
                break;
            case SDL_AUDIO_S8:
                alsa_format = SND_PCM_FORMAT_S8;
                break;
            case SDL_AUDIO_S16LE:
                alsa_format = SND_PCM_FORMAT_S16_LE;
                break;
            case SDL_AUDIO_S16BE:
                alsa_format = SND_PCM_FORMAT_S16_BE;
                break;
            case SDL_AUDIO_S32LE:
                alsa_format = SND_PCM_FORMAT_S32_LE;
                break;
            case SDL_AUDIO_S32BE:
                alsa_format = SND_PCM_FORMAT_S32_BE;
                break;
            case SDL_AUDIO_F32LE:
                alsa_format = SND_PCM_FORMAT_FLOAT_LE;
                break;
            case SDL_AUDIO_F32BE:
                alsa_format = SND_PCM_FORMAT_FLOAT_BE;
                break;
            default:
                continue;
            }
            if (ALSA_snd_pcm_hw_params_set_format(ctx->device->hidden->pcm, ctx->hwparams, alsa_format) >= 0) {
                break;
            }
        }
        if (ctx->matched_sdl_format == 0) {
            SDL_SetError("ALSA: Unsupported audio format: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // let alsa approximate the number of channels
        ctx->chans_n = target_chans_n;
        status = ALSA_snd_pcm_hw_params_set_channels_near(ctx->device->hidden->pcm, ctx->hwparams, &(ctx->chans_n));
        if (status < 0) {
            SDL_SetError("ALSA: Couldn't set audio channels: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // let alsa approximate the audio rate
        ctx->rate = ctx->device->spec.freq;
        status = ALSA_snd_pcm_hw_params_set_rate_near(ctx->device->hidden->pcm, ctx->hwparams, &(ctx->rate), NULL);
        if (status < 0) {
            SDL_SetError("ALSA: Couldn't set audio frequency: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // let approximate the period size to the requested buffer size
        ctx->persize = ctx->device->sample_frames;
        status = ALSA_snd_pcm_hw_params_set_period_size_near(ctx->device->hidden->pcm, ctx->hwparams, &(ctx->persize), NULL);
        if (status < 0) {
            SDL_SetError("ALSA: Couldn't set the period size: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // let approximate the minimun number of periods per buffer (we target a double buffer)
        ctx->periods = 2;
        status = ALSA_snd_pcm_hw_params_set_periods_min(ctx->device->hidden->pcm, ctx->hwparams, &(ctx->periods), NULL);
        if (status < 0) {
            SDL_SetError("ALSA: Couldn't set the minimum number of periods per buffer: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // restrict the number of periods per buffer to an approximation of the approximated minimum
        // number of periods per buffer done right above
        status = ALSA_snd_pcm_hw_params_set_periods_first(ctx->device->hidden->pcm, ctx->hwparams, &(ctx->periods), NULL);
        if (status < 0) {
            SDL_SetError("ALSA: Couldn't set the number of periods per buffer: %s", ALSA_snd_strerror(status));
            return -1;
        }
        // install the hw parameters
        status = ALSA_snd_pcm_hw_params(ctx->device->hidden->pcm, ctx->hwparams);
        if (status < 0) {
            SDL_SetError("ALSA: installation of hardware parameter failed: %s", ALSA_snd_strerror(status));
            return -1;
        }
        //==========================================================================================
        // Here the alsa pcm is in SND_PCM_STATE_PREPARED state, let's figure out a good fit for
        // SDL channel map, it may request to change the target number of channels though.
        status = alsa_chmap_cfg(ctx);
        if (status < 0) {
            return status; // we forward the SDL error
        } else if (status == CHMAP_INSTALLED) {
            return CHANS_N_CONFIGURED; // we are finished here
        }

        // status == CHANS_N_NEXT
        ALSA_snd_pcm_free_chmaps(ctx->chmap_queries);
        ALSA_snd_pcm_hw_free(ctx->device->hidden->pcm); // uninstall those hw params

        if (mode == CHANS_N_SCAN_MODE__EQUAL_OR_ABOVE_REQUESTED_CHANS_N) {
            target_chans_n++;
        } else {  // CHANS_N_SCAN_MODE__BELOW_REQUESTED_CHANS_N
            target_chans_n--;
        }
    }

    SDL_assert(!"Shouldn't reach this code.");
    return CHANS_N_NOT_CONFIGURED;
}
#undef CHMAP_INSTALLED
#undef CHANS_N_NEXT
#undef CHMAP_NOT_FOUND

static bool ALSA_pcm_cfg_hw(struct ALSA_pcm_cfg_ctx *ctx)
{
    LOGDEBUG("target chans_n, equal or above requested chans_n mode");
    int status = ALSA_pcm_cfg_hw_chans_n_scan(ctx, CHANS_N_SCAN_MODE__EQUAL_OR_ABOVE_REQUESTED_CHANS_N);
    if (status < 0) {  // something went too wrong
        return false;
    } else if (status == CHANS_N_CONFIGURED) {
        return true;
    }

    // Here, status == CHANS_N_NOT_CONFIGURED
    LOGDEBUG("target chans_n, below requested chans_n mode");
    status = ALSA_pcm_cfg_hw_chans_n_scan(ctx, CHANS_N_SCAN_MODE__BELOW_REQUESTED_CHANS_N);
    if (status < 0) { // something went too wrong
        return false;
    } else if (status == CHANS_N_CONFIGURED) {
        return true;
    }

    // Here, status == CHANS_N_NOT_CONFIGURED
    return SDL_SetError("ALSA: Coudn't configure targetting any SDL supported channel number");
}
#undef CHANS_N_SCAN_MODE__EQUAL_OR_ABOVE_REQUESTED_CHANS_N
#undef CHANS_N_SCAN_MODE__BELOW_REQUESTED_CHANS_N
#undef CHANS_N_CONFIGURED
#undef CHANS_N_NOT_CONFIGURED


static bool ALSA_pcm_cfg_sw(struct ALSA_pcm_cfg_ctx *ctx)
{
    int status;

    status = ALSA_snd_pcm_sw_params_current(ctx->device->hidden->pcm, ctx->swparams);
    if (status < 0) {
        return SDL_SetError("ALSA: Couldn't get software config: %s", ALSA_snd_strerror(status));
    }

    status = ALSA_snd_pcm_sw_params_set_avail_min(ctx->device->hidden->pcm, ctx->swparams, ctx->persize); // will become device->sample_frames if the alsa pcm configuration is successful
    if (status < 0) {
        return SDL_SetError("Couldn't set minimum available samples: %s", ALSA_snd_strerror(status));
    }

    status = ALSA_snd_pcm_sw_params_set_start_threshold(ctx->device->hidden->pcm, ctx->swparams, 1);
    if (status < 0) {
        return SDL_SetError("ALSA: Couldn't set start threshold: %s", ALSA_snd_strerror(status));
    }
    status = ALSA_snd_pcm_sw_params(ctx->device->hidden->pcm, ctx->swparams);
    if (status < 0) {
        return SDL_SetError("Couldn't set software audio parameters: %s", ALSA_snd_strerror(status));
    }
    return true;
}


static bool ALSA_OpenDevice(SDL_AudioDevice *device)
{
    const bool recording = device->recording;
    struct ALSA_pcm_cfg_ctx cfg_ctx; // used to track everything here
    char *pcm_str;
    int status = 0;

    //device->spec.channels = 8;
    //SDL_SetLogPriority(SDL_LOG_CATEGORY_AUDIO, SDL_LOG_PRIORITY_VERBOSE);
    LOGDEBUG("channels requested %u",device->spec.channels);
    // XXX: We do not use the SDL internal swizzler yet.
    device->chmap = NULL;

    SDL_zero(cfg_ctx);
    cfg_ctx.device = device;

    // Initialize all variables that we clean on shutdown
    cfg_ctx.device->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*cfg_ctx.device->hidden));
    if (!cfg_ctx.device->hidden) {
        return false;
    }

    // Open the audio device
    pcm_str = get_pcm_str(cfg_ctx.device->handle);
    if (pcm_str == NULL) {
        goto err_free_device_hidden;
    }
    LOGDEBUG("PCM open '%s'", pcm_str);
    status = ALSA_snd_pcm_open(&cfg_ctx.device->hidden->pcm,
                               pcm_str,
                               recording ? SND_PCM_STREAM_CAPTURE : SND_PCM_STREAM_PLAYBACK,
                               SND_PCM_NONBLOCK);
    SDL_free(pcm_str);
    if (status < 0) {
        SDL_SetError("ALSA: Couldn't open audio device: %s", ALSA_snd_strerror(status));
        goto err_free_device_hidden;
    }

    // Now we need to configure the opened pcm as close as possible from the requested parameters we
    // can reasonably deal with (and that could change)
    snd_pcm_hw_params_alloca(&(cfg_ctx.hwparams));
    snd_pcm_sw_params_alloca(&(cfg_ctx.swparams));

    if (!ALSA_pcm_cfg_hw(&cfg_ctx)) { // alsa pcm "hardware" part of the pcm
        goto err_close_pcm;
    }

    // from here, we get only the alsa chmap queries in cfg_ctx to explicitely clean, hwparams is
    // uninstalled upon pcm closing

    // This is useful for debugging
    #if SDL_ALSA_DEBUG
    snd_pcm_uframes_t bufsize;
    ALSA_snd_pcm_hw_params_get_buffer_size(cfg_ctx.hwparams, &bufsize);
    SDL_LogError(SDL_LOG_CATEGORY_AUDIO,
                     "ALSA: period size = %ld, periods = %u, buffer size = %lu",
                     cfg_ctx.persize, cfg_ctx.periods, bufsize);
    #endif

    if (!ALSA_pcm_cfg_sw(&cfg_ctx)) { // alsa pcm "software" part of the pcm
        goto err_cleanup_ctx;
    }

    // Now we can update the following parameters in the spec:
    cfg_ctx.device->spec.format = cfg_ctx.matched_sdl_format;
    cfg_ctx.device->spec.channels = cfg_ctx.chans_n;
    cfg_ctx.device->spec.freq = cfg_ctx.rate;
    cfg_ctx.device->sample_frames = cfg_ctx.persize;
    // Calculate the final parameters for this audio specification
    SDL_UpdatedAudioDeviceFormat(cfg_ctx.device);

    // Allocate mixing buffer
    if (!recording) {
        cfg_ctx.device->hidden->mixbuf = (Uint8 *)SDL_malloc(cfg_ctx.device->buffer_size);
        if (cfg_ctx.device->hidden->mixbuf == NULL) {
            goto err_cleanup_ctx;
        }
        SDL_memset(cfg_ctx.device->hidden->mixbuf, cfg_ctx.device->silence_value, cfg_ctx.device->buffer_size);
    }

#if !SDL_ALSA_NON_BLOCKING
    if (!recording) {
        ALSA_snd_pcm_nonblock(cfg_ctx.device->hidden->pcm, 0);
    }
#endif
    return true;  // We're ready to rock and roll. :-)

err_cleanup_ctx:
    ALSA_snd_pcm_free_chmaps(cfg_ctx.chmap_queries);
err_close_pcm:
    ALSA_snd_pcm_close(cfg_ctx.device->hidden->pcm);
err_free_device_hidden:
    SDL_free(cfg_ctx.device->hidden);
    cfg_ctx.device->hidden = NULL;
    return false;
}

static void ALSA_ThreadInit(SDL_AudioDevice *device)
{
    SDL_SetCurrentThreadPriority(device->recording ? SDL_THREAD_PRIORITY_HIGH : SDL_THREAD_PRIORITY_TIME_CRITICAL);
    // do snd_pcm_start as close to the first time we PlayDevice as possible to prevent an underrun at startup.
    ALSA_snd_pcm_start(device->hidden->pcm);
}

static ALSA_Device *hotplug_devices = NULL;

static int hotplug_device_process(snd_ctl_t *ctl, snd_ctl_card_info_t *ctl_card_info, int dev_idx,
                                  snd_pcm_stream_t direction, ALSA_Device **unseen, ALSA_Device **seen)
{
    unsigned int subdevs_n = 1; // we have at least one subdevice (substream since the direction is a stream in alsa terminology)
    unsigned int subdev_idx = 0;
    const bool recording = direction == SND_PCM_STREAM_CAPTURE ? true : false; // used for the unicity of the device
    bool isstack;
    snd_pcm_info_t *pcm_info = (snd_pcm_info_t*)SDL_small_alloc(Uint8, ALSA_snd_pcm_info_sizeof(), &isstack);
    SDL_memset(pcm_info, 0, ALSA_snd_pcm_info_sizeof());

    while (true) {
        ALSA_snd_pcm_info_set_stream(pcm_info, direction);
        ALSA_snd_pcm_info_set_device(pcm_info, dev_idx);
        ALSA_snd_pcm_info_set_subdevice(pcm_info, subdev_idx); // we have at least one subdevice (substream) of index 0

        const int r = ALSA_snd_ctl_pcm_info(ctl, pcm_info);
        if (r < 0) {
            SDL_small_free(pcm_info, isstack);
            // first call to ALSA_snd_ctl_pcm_info
            if (subdev_idx == 0 && r == -ENOENT) { // no such direction/stream for this device
                return 0;
            }
            return -1;
        }

        if (subdev_idx == 0) {
            subdevs_n = ALSA_snd_pcm_info_get_subdevices_count(pcm_info);
        }

        // building the unseen list scanning the list of hotplug devices, if it is already there
        // using the id, move it to the seen list.
        ALSA_Device *unseen_prev_adev = NULL;
        ALSA_Device *adev;
        for (adev = *unseen; adev; adev = adev->next) {
            // the unicity key is the couple (id,recording)
            if ((SDL_strcmp(adev->id, ALSA_snd_ctl_card_info_get_id(ctl_card_info)) == 0) && (adev->recording == recording)) {
                // unchain from unseen
                if (*unseen == adev) { // head
                    *unseen = adev->next;
                } else {
                    unseen_prev_adev->next = adev->next;
                }
                // chain to seen
                adev->next = *seen;
                *seen = adev;
                break;
            }
            unseen_prev_adev = adev;
        }

        if (adev == NULL) { // newly seen device
            adev = SDL_calloc(1, sizeof(*adev));
            if (adev == NULL) {
                SDL_small_free(pcm_info, isstack);
                return -1;
            }

            adev->id = SDL_strdup(ALSA_snd_ctl_card_info_get_id(ctl_card_info));
            if (adev->id == NULL) {
                SDL_small_free(pcm_info, isstack);
                SDL_free(adev);
                return -1;
            }

            if (SDL_asprintf(&adev->name, "%s:%s", ALSA_snd_ctl_card_info_get_name(ctl_card_info), ALSA_snd_pcm_info_get_name(pcm_info)) == -1) {
                SDL_small_free(pcm_info, isstack);
                SDL_free(adev->id);
                SDL_free(adev);
                return -1;
            }

            if (direction == SND_PCM_STREAM_CAPTURE) {
                adev->recording = true;
            } else {
                adev->recording = false;
            }

            if (SDL_AddAudioDevice(recording, adev->name, NULL, adev) == NULL) {
                SDL_small_free(pcm_info, isstack);
                SDL_free(adev->id);
                SDL_free(adev->name);
                SDL_free(adev);
                return -1;
            }

            adev->next = *seen;
            *seen = adev;
        }

        subdev_idx++;
        if (subdev_idx == subdevs_n) {
            SDL_small_free(pcm_info, isstack);
            return 0;
        }

        SDL_memset(pcm_info, 0, ALSA_snd_pcm_info_sizeof());
    }

    SDL_small_free(pcm_info, isstack);
    SDL_assert(!"Shouldn't reach this code");
    return -1;
}

static void ALSA_HotplugIteration(bool *has_default_output, bool *has_default_recording)
{
    if (has_default_output != NULL) {
        *has_default_output = true;
    }

    if (has_default_recording != NULL) {
        *has_default_recording = true;
    }

    bool isstack;
    snd_ctl_card_info_t *ctl_card_info = (snd_ctl_card_info_t *) SDL_small_alloc(Uint8, ALSA_snd_ctl_card_info_sizeof(), &isstack);
    if (!ctl_card_info) {
        return;  // oh well.
    }

    SDL_memset(ctl_card_info, 0, ALSA_snd_ctl_card_info_sizeof());

    snd_ctl_t *ctl = NULL;
    ALSA_Device *unseen = hotplug_devices;
    ALSA_Device *seen = NULL;
    int card_idx = -1;
    while (true) {
        int r = ALSA_snd_card_next(&card_idx);
        if (r < 0) {
            goto failed;
        } else if (card_idx == -1) {
            break;
        }

        char ctl_name[64];
        SDL_snprintf(ctl_name, sizeof (ctl_name), "%s%d", ALSA_device_prefix, card_idx); // card_idx >= 0
        LOGDEBUG("hotplug ctl_name = '%s'", ctl_name);

        r = ALSA_snd_ctl_open(&ctl, ctl_name, 0);
        if (r < 0) {
            continue;
        }

        r = ALSA_snd_ctl_card_info(ctl, ctl_card_info);
        if (r < 0) {
            goto failed;
        }

        int dev_idx = -1;
        while (true) {
            r = ALSA_snd_ctl_pcm_next_device(ctl, &dev_idx);
            if (r < 0) {
                goto failed;
            } else if (dev_idx == -1) {
                break;
            }

            r = hotplug_device_process(ctl, ctl_card_info, dev_idx, SND_PCM_STREAM_PLAYBACK, &unseen, &seen);
            if (r < 0) {
                goto failed;
            }

            r = hotplug_device_process(ctl, ctl_card_info, dev_idx, SND_PCM_STREAM_CAPTURE, &unseen, &seen);
            if (r < 0) {
                goto failed;
            }
        }
        ALSA_snd_ctl_close(ctl);
        ALSA_snd_ctl_card_info_clear(ctl_card_info);
    }

    // remove only the unseen devices
    while (unseen) {
        SDL_AudioDeviceDisconnected(SDL_FindPhysicalAudioDeviceByHandle(unseen));
        SDL_free(unseen->name);
        SDL_free(unseen->id);
        ALSA_Device *next = unseen->next;
        SDL_free(unseen);
        unseen = next;
    }

    // update hotplug devices to be the seen devices
    hotplug_devices = seen;
    SDL_small_free(ctl_card_info, isstack);
    return;

failed:
    if (ctl) {
        ALSA_snd_ctl_close(ctl);
    }

    // remove the unseen
    while (unseen) {
        SDL_AudioDeviceDisconnected(SDL_FindPhysicalAudioDeviceByHandle(unseen));
        SDL_free(unseen->name);
        SDL_free(unseen->id);
        ALSA_Device *next = unseen->next;
        SDL_free(unseen);
        unseen = next;
    }

    // remove the seen
    while (seen) {
        SDL_AudioDeviceDisconnected(SDL_FindPhysicalAudioDeviceByHandle(seen));
        SDL_free(seen->name);
        SDL_free(seen->id);
        ALSA_Device *next = seen->next;
        SDL_free(seen);
        seen = next;
    }

    hotplug_devices = NULL;
    SDL_small_free(ctl_card_info, isstack);
}


#if SDL_ALSA_HOTPLUG_THREAD
static SDL_AtomicInt ALSA_hotplug_shutdown;
static SDL_Thread *ALSA_hotplug_thread;

static int SDLCALL ALSA_HotplugThread(void *arg)
{
    SDL_SetCurrentThreadPriority(SDL_THREAD_PRIORITY_LOW);

    while (!SDL_GetAtomicInt(&ALSA_hotplug_shutdown)) {
        // Block awhile before checking again, unless we're told to stop.
        const Uint64 ticks = SDL_GetTicks() + 5000;
        while (!SDL_GetAtomicInt(&ALSA_hotplug_shutdown) && (SDL_GetTicks() < ticks)) {
            SDL_Delay(100);
        }

        ALSA_HotplugIteration(NULL, NULL); // run the check.
    }

    return 0;
}
#endif

static void ALSA_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    ALSA_guess_device_prefix();

    // ALSA doesn't have a concept of a changeable default device, afaik, so we expose a generic default
    // device here. It's the best we can do at this level.
    bool has_default_playback = false, has_default_recording = false;
    ALSA_HotplugIteration(&has_default_playback, &has_default_recording); // run once now before a thread continues to check.
    if (has_default_playback) {
        *default_playback = SDL_AddAudioDevice(/*recording=*/false, "ALSA default playback device", NULL, (void*)&default_playback_handle);
    }
    if (has_default_recording) {
        *default_recording = SDL_AddAudioDevice(/*recording=*/true, "ALSA default recording device", NULL, (void*)&default_recording_handle);
    }

#if SDL_ALSA_HOTPLUG_THREAD
    SDL_SetAtomicInt(&ALSA_hotplug_shutdown, 0);
    ALSA_hotplug_thread = SDL_CreateThread(ALSA_HotplugThread, "SDLHotplugALSA", NULL);
    // if the thread doesn't spin, oh well, you just don't get further hotplug events.
#endif
}

static void ALSA_DeinitializeStart(void)
{
    ALSA_Device *dev;
    ALSA_Device *next;

#if SDL_ALSA_HOTPLUG_THREAD
    if (ALSA_hotplug_thread) {
        SDL_SetAtomicInt(&ALSA_hotplug_shutdown, 1);
        SDL_WaitThread(ALSA_hotplug_thread, NULL);
        ALSA_hotplug_thread = NULL;
    }
#endif

    // Shutting down! Clean up any data we've gathered.
    for (dev = hotplug_devices; dev; dev = next) {
        //SDL_LogInfo(SDL_LOG_CATEGORY_AUDIO, "ALSA: at shutdown, removing %s device '%s'", dev->recording ? "recording" : "playback", dev->name);
        next = dev->next;
        SDL_free(dev->name);
        SDL_free(dev);
    }
    hotplug_devices = NULL;
}

static void ALSA_Deinitialize(void)
{
    UnloadALSALibrary();
}

static bool ALSA_Init(SDL_AudioDriverImpl *impl)
{
    if (!LoadALSALibrary()) {
        return false;
    }

    impl->DetectDevices = ALSA_DetectDevices;
    impl->OpenDevice = ALSA_OpenDevice;
    impl->ThreadInit = ALSA_ThreadInit;
    impl->WaitDevice = ALSA_WaitDevice;
    impl->GetDeviceBuf = ALSA_GetDeviceBuf;
    impl->PlayDevice = ALSA_PlayDevice;
    impl->CloseDevice = ALSA_CloseDevice;
    impl->DeinitializeStart = ALSA_DeinitializeStart;
    impl->Deinitialize = ALSA_Deinitialize;
    impl->WaitRecordingDevice = ALSA_WaitDevice;
    impl->RecordDevice = ALSA_RecordDevice;
    impl->FlushRecording = ALSA_FlushRecording;

    impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap ALSA_bootstrap = {
    "alsa", "ALSA PCM audio", ALSA_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_ALSA
