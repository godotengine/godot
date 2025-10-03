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

#include "SDL_audio_c.h"
#include "SDL_sysaudio.h"
#include "../thread/SDL_systhread.h"

// Available audio drivers
static const AudioBootStrap *const bootstrap[] = {
#ifdef SDL_AUDIO_DRIVER_PRIVATE
    &PRIVATEAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_PULSEAUDIO
#ifdef SDL_AUDIO_DRIVER_PIPEWIRE
    &PIPEWIRE_PREFERRED_bootstrap,
#endif
    &PULSEAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_PIPEWIRE
    &PIPEWIRE_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_ALSA
    &ALSA_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_SNDIO
    &SNDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_NETBSD
    &NETBSDAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_WASAPI
    &WASAPI_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_DSOUND
    &DSOUND_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_HAIKU
    &HAIKUAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_COREAUDIO
    &COREAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_AAUDIO
    &AAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_OPENSLES
    &OPENSLES_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_PS2
    &PS2AUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_PSP
    &PSPAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_VITA
    &VITAAUD_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_N3DS
    &N3DSAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_EMSCRIPTEN
    &EMSCRIPTENAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_JACK
    &JACK_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_OSS
    &DSP_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_QNX
    &QSAAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_DISK
    &DISKAUDIO_bootstrap,
#endif
#ifdef SDL_AUDIO_DRIVER_DUMMY
    &DUMMYAUDIO_bootstrap,
#endif
    NULL
};

static SDL_AudioDriver current_audio;

// Deduplicated list of audio bootstrap drivers.
static const AudioBootStrap *deduped_bootstrap[SDL_arraysize(bootstrap) - 1];

int SDL_GetNumAudioDrivers(void)
{
    static int num_drivers = -1;

    if (num_drivers >= 0) {
        return num_drivers;
    }

    num_drivers = 0;

    // Build a list of unique audio drivers.
    for (int i = 0; bootstrap[i] != NULL; ++i) {
        bool duplicate = false;
        for (int j = 0; j < i; ++j) {
            if (SDL_strcmp(bootstrap[i]->name, bootstrap[j]->name) == 0) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            deduped_bootstrap[num_drivers++] = bootstrap[i];
        }
    }

    return num_drivers;
}

const char *SDL_GetAudioDriver(int index)
{
    if (index >= 0 && index < SDL_GetNumAudioDrivers()) {
        return deduped_bootstrap[index]->name;
    }
    SDL_InvalidParamError("index");
    return NULL;
}

const char *SDL_GetCurrentAudioDriver(void)
{
    return current_audio.name;
}

int SDL_GetDefaultSampleFramesFromFreq(const int freq)
{
    const char *hint = SDL_GetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES);
    if (hint) {
        const int val = SDL_atoi(hint);
        if (val > 0) {
            return val;
        }
    }

    if (freq <= 22050) {
        return 512;
    } else if (freq <= 48000) {
        return 1024;
    } else if (freq <= 96000) {
        return 2048;
    } else {
        return 4096;
    }
}

int *SDL_ChannelMapDup(const int *origchmap, int channels)
{
    const size_t chmaplen = sizeof (*origchmap) * channels;
    int *chmap = (int *)SDL_malloc(chmaplen);
    if (chmap) {
        SDL_memcpy(chmap, origchmap, chmaplen);
    }
    return chmap;
}

void OnAudioStreamCreated(SDL_AudioStream *stream)
{
    SDL_assert(stream != NULL);

    // NOTE that you can create an audio stream without initializing the audio subsystem,
    //  but it will not be automatically destroyed during a later call to SDL_Quit!
    //  You must explicitly destroy it yourself!
    if (current_audio.device_hash_lock) {
        // this isn't really part of the "device list" but it's a convenient lock to use here.
        SDL_LockRWLockForWriting(current_audio.device_hash_lock);
        if (current_audio.existing_streams) {
            current_audio.existing_streams->prev = stream;
        }
        stream->prev = NULL;
        stream->next = current_audio.existing_streams;
        current_audio.existing_streams = stream;
        SDL_UnlockRWLock(current_audio.device_hash_lock);
    }
}

void OnAudioStreamDestroy(SDL_AudioStream *stream)
{
    SDL_assert(stream != NULL);

    // NOTE that you can create an audio stream without initializing the audio subsystem,
    //  but it will not be automatically destroyed during a later call to SDL_Quit!
    //  You must explicitly destroy it yourself!
    if (current_audio.device_hash_lock) {
        // this isn't really part of the "device list" but it's a convenient lock to use here.
        SDL_LockRWLockForWriting(current_audio.device_hash_lock);
        if (stream->prev) {
            stream->prev->next = stream->next;
        }
        if (stream->next) {
            stream->next->prev = stream->prev;
        }
        if (stream == current_audio.existing_streams) {
            current_audio.existing_streams = stream->next;
        }
        SDL_UnlockRWLock(current_audio.device_hash_lock);
    }
}

// device should be locked when calling this.
static bool AudioDeviceCanUseSimpleCopy(SDL_AudioDevice *device)
{
    SDL_assert(device != NULL);
    return (
        device->logical_devices &&  // there's a logical device
        !device->logical_devices->next &&  // there's only _ONE_ logical device
        !device->logical_devices->postmix && // there isn't a postmix callback
        device->logical_devices->bound_streams &&  // there's a bound stream
        !device->logical_devices->bound_streams->next_binding  // there's only _ONE_ bound stream.
    );
}

// should hold device->lock before calling.
static void UpdateAudioStreamFormatsPhysical(SDL_AudioDevice *device)
{
    if (!device) {
        return;
    }

    const bool recording = device->recording;
    SDL_AudioSpec spec;
    SDL_copyp(&spec, &device->spec);

    const SDL_AudioFormat devformat = spec.format;

    if (!recording) {
        const bool simple_copy = AudioDeviceCanUseSimpleCopy(device);
        device->simple_copy = simple_copy;
        if (!simple_copy) {
            spec.format = SDL_AUDIO_F32;  // mixing and postbuf operates in float32 format.
        }
    }

    for (SDL_LogicalAudioDevice *logdev = device->logical_devices; logdev; logdev = logdev->next) {
        if (recording) {
            const bool need_float32 = (logdev->postmix || logdev->gain != 1.0f);
            spec.format = need_float32 ? SDL_AUDIO_F32 : devformat;
        }

        for (SDL_AudioStream *stream = logdev->bound_streams; stream; stream = stream->next_binding) {
            // set the proper end of the stream to the device's format.
            // SDL_SetAudioStreamFormat does a ton of validation just to memcpy an audiospec.
            SDL_AudioSpec *streamspec = recording ? &stream->src_spec : &stream->dst_spec;
            int **streamchmap = recording ? &stream->src_chmap : &stream->dst_chmap;
            SDL_LockMutex(stream->lock);
            SDL_copyp(streamspec, &spec);
            SetAudioStreamChannelMap(stream, streamspec, streamchmap, device->chmap, device->spec.channels, -1);  // this should be fast for normal cases, though!
            SDL_UnlockMutex(stream->lock);
        }
    }
}

bool SDL_AudioSpecsEqual(const SDL_AudioSpec *a, const SDL_AudioSpec *b, const int *channel_map_a, const int *channel_map_b)
{
    if ((a->format != b->format) || (a->channels != b->channels) || (a->freq != b->freq) || ((channel_map_a != NULL) != (channel_map_b != NULL))) {
        return false;
    } else if (channel_map_a && (SDL_memcmp(channel_map_a, channel_map_b, sizeof (*channel_map_a) * a->channels) != 0)) {
        return false;
    }
    return true;
}

bool SDL_AudioChannelMapsEqual(int channels, const int *channel_map_a, const int *channel_map_b)
{
    if (channel_map_a == channel_map_b) {
        return true;
    } else if ((channel_map_a != NULL) != (channel_map_b != NULL)) {
        return false;
    } else if (channel_map_a && (SDL_memcmp(channel_map_a, channel_map_b, sizeof (*channel_map_a) * channels) != 0)) {
        return false;
    }
    return true;
}


// Zombie device implementation...

// These get used when a device is disconnected or fails, so audiostreams don't overflow with data that isn't being
// consumed and apps relying on audio callbacks don't stop making progress.
static bool ZombieWaitDevice(SDL_AudioDevice *device)
{
    if (!SDL_GetAtomicInt(&device->shutdown)) {
        const int frames = device->buffer_size / SDL_AUDIO_FRAMESIZE(device->spec);
        SDL_Delay((frames * 1000) / device->spec.freq);
    }
    return true;
}

static bool ZombiePlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    return true;  // no-op, just throw the audio away.
}

static Uint8 *ZombieGetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    return device->work_buffer;
}

static int ZombieRecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    // return a full buffer of silence every time.
    SDL_memset(buffer, device->silence_value, buflen);
    return buflen;
}

static void ZombieFlushRecording(SDL_AudioDevice *device)
{
    // no-op, this is all imaginary.
}



// device management and hotplug...


/* SDL_AudioDevice, in SDL3, represents a piece of physical hardware, whether it is in use or not, so these objects exist as long as
   the system-level device is available.

   Physical devices get destroyed for three reasons:
    - They were lost to the system (a USB cable is kicked out, etc).
    - They failed for some other unlikely reason at the API level (which is _also_ probably a USB cable being kicked out).
    - We are shutting down, so all allocated resources are being freed.

   They are _not_ destroyed because we are done using them (when we "close" a playing device).
*/
static void ClosePhysicalAudioDevice(SDL_AudioDevice *device);


SDL_COMPILE_TIME_ASSERT(check_lowest_audio_default_value, SDL_AUDIO_DEVICE_DEFAULT_RECORDING < SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK);

static SDL_AtomicInt last_device_instance_id;  // increments on each device add to provide unique instance IDs
static SDL_AudioDeviceID AssignAudioDeviceInstanceId(bool recording, bool islogical)
{
    /* Assign an instance id! Start at 2, in case there are things from the SDL2 era that still think 1 is a special value.
       Also, make sure we don't assign SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, etc. */

    // The bottom two bits of the instance id tells you if it's an playback device (1<<0), and if it's a physical device (1<<1).
    const SDL_AudioDeviceID flags = (recording ? 0 : (1<<0)) | (islogical ? 0 : (1<<1));

    const SDL_AudioDeviceID instance_id = (((SDL_AudioDeviceID) (SDL_AtomicIncRef(&last_device_instance_id) + 1)) << 2) | flags;
    SDL_assert( (instance_id >= 2) && (instance_id < SDL_AUDIO_DEVICE_DEFAULT_RECORDING) );
    return instance_id;
}

bool SDL_IsAudioDevicePhysical(SDL_AudioDeviceID devid)
{
    return (devid & (1 << 1)) != 0;
}

bool SDL_IsAudioDevicePlayback(SDL_AudioDeviceID devid)
{
    return (devid & (1 << 0)) != 0;
}

static void ObtainPhysicalAudioDeviceObj(SDL_AudioDevice *device) SDL_NO_THREAD_SAFETY_ANALYSIS  // !!! FIXMEL SDL_ACQUIRE
{
    if (device) {
        RefPhysicalAudioDevice(device);
        SDL_LockMutex(device->lock);
    }
}

static void ReleaseAudioDevice(SDL_AudioDevice *device) SDL_NO_THREAD_SAFETY_ANALYSIS  // !!! FIXME: SDL_RELEASE
{
    if (device) {
        SDL_UnlockMutex(device->lock);
        UnrefPhysicalAudioDevice(device);
    }
}

// If found, this locks _the physical device_ this logical device is associated with, before returning.
static SDL_LogicalAudioDevice *ObtainLogicalAudioDevice(SDL_AudioDeviceID devid, SDL_AudioDevice **_device) SDL_NO_THREAD_SAFETY_ANALYSIS    // !!! FIXME: SDL_ACQUIRE
{
    SDL_assert(_device != NULL);

    if (!SDL_GetCurrentAudioDriver()) {
        SDL_SetError("Audio subsystem is not initialized");
        *_device = NULL;
        return NULL;
    }

    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = NULL;

    // bit #1 of devid is set for physical devices and unset for logical.
    const bool islogical = !(devid & (1<<1));
    if (islogical) {  // don't bother looking if it's not a logical device id value.
        SDL_LockRWLockForReading(current_audio.device_hash_lock);
        SDL_FindInHashTable(current_audio.device_hash, (const void *) (uintptr_t) devid, (const void **) &logdev);
        if (logdev) {
            SDL_assert(logdev->instance_id == devid);
            device = logdev->physical_device;
            SDL_assert(device != NULL);
            RefPhysicalAudioDevice(device);  // reference it, in case the logical device migrates to a new default.
        }
        SDL_UnlockRWLock(current_audio.device_hash_lock);

        if (logdev) {
            // we have to release the device_hash_lock before we take the device lock, to avoid deadlocks, so do a loop
            //  to make sure the correct physical device gets locked, in case we're in a race with the default changing.
            while (true) {
                SDL_LockMutex(device->lock);
                SDL_AudioDevice *recheck_device = (SDL_AudioDevice *) SDL_GetAtomicPointer((void **) &logdev->physical_device);
                if (device == recheck_device) {
                    break;
                }

                // default changed from under us! Try again!
                RefPhysicalAudioDevice(recheck_device);
                SDL_UnlockMutex(device->lock);
                UnrefPhysicalAudioDevice(device);
                device = recheck_device;
            }
        }
    }

    if (!logdev) {
        SDL_SetError("Invalid audio device instance ID");
    }

    *_device = device;
    return logdev;
}


/* this finds the physical device associated with `devid` and locks it for use.
   Note that a logical device instance id will return its associated physical device! */
static SDL_AudioDevice *ObtainPhysicalAudioDevice(SDL_AudioDeviceID devid)  // !!! FIXME: SDL_ACQUIRE
{
    SDL_AudioDevice *device = NULL;

    // bit #1 of devid is set for physical devices and unset for logical.
    const bool islogical = !(devid & (1<<1));
    if (islogical) {
        ObtainLogicalAudioDevice(devid, &device);
    } else if (!SDL_GetCurrentAudioDriver()) {  // (the `islogical` path, above, checks this in ObtainLogicalAudioDevice.)
        SDL_SetError("Audio subsystem is not initialized");
    } else {
        SDL_LockRWLockForReading(current_audio.device_hash_lock);
        SDL_FindInHashTable(current_audio.device_hash, (const void *) (uintptr_t) devid, (const void **) &device);
        SDL_assert(device->instance_id == devid);
        SDL_UnlockRWLock(current_audio.device_hash_lock);

        if (!device) {
            SDL_SetError("Invalid audio device instance ID");
        } else {
            ObtainPhysicalAudioDeviceObj(device);
        }
    }

    return device;
}

static SDL_AudioDevice *ObtainPhysicalAudioDeviceDefaultAllowed(SDL_AudioDeviceID devid)  // !!! FIXME: SDL_ACQUIRE
{
    const bool wants_default = ((devid == SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK) || (devid == SDL_AUDIO_DEVICE_DEFAULT_RECORDING));
    if (!wants_default) {
        return ObtainPhysicalAudioDevice(devid);
    }

    const SDL_AudioDeviceID orig_devid = devid;

    while (true) {
        SDL_LockRWLockForReading(current_audio.device_hash_lock);
        if (orig_devid == SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK) {
            devid = current_audio.default_playback_device_id;
        } else if (orig_devid == SDL_AUDIO_DEVICE_DEFAULT_RECORDING) {
            devid = current_audio.default_recording_device_id;
        }
        SDL_UnlockRWLock(current_audio.device_hash_lock);

        if (devid == 0) {
            SDL_SetError("No default audio device available");
            break;
        }

        SDL_AudioDevice *device = ObtainPhysicalAudioDevice(devid);
        if (!device) {
            break;
        }

        // make sure the default didn't change while we were waiting for the lock...
        bool got_it = false;
        SDL_LockRWLockForReading(current_audio.device_hash_lock);
        if ((orig_devid == SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK) && (devid == current_audio.default_playback_device_id)) {
            got_it = true;
        } else if ((orig_devid == SDL_AUDIO_DEVICE_DEFAULT_RECORDING) && (devid == current_audio.default_recording_device_id)) {
            got_it = true;
        }
        SDL_UnlockRWLock(current_audio.device_hash_lock);

        if (got_it) {
            return device;
        }

        ReleaseAudioDevice(device);  // let it go and try again.
    }

    return NULL;
}

// this assumes you hold the _physical_ device lock for this logical device! This will not unlock the lock or close the physical device!
//  It also will not unref the physical device, since we might be shutting down; SDL_CloseAudioDevice handles the unref.
static void DestroyLogicalAudioDevice(SDL_LogicalAudioDevice *logdev)
{
    // Remove ourselves from the device_hash hashtable.
    if (current_audio.device_hash) {  // will be NULL while shutting down.
        SDL_LockRWLockForWriting(current_audio.device_hash_lock);
        SDL_RemoveFromHashTable(current_audio.device_hash, (const void *) (uintptr_t) logdev->instance_id);
        SDL_UnlockRWLock(current_audio.device_hash_lock);
    }

    // remove ourselves from the physical device's list of logical devices.
    if (logdev->next) {
        logdev->next->prev = logdev->prev;
    }
    if (logdev->prev) {
        logdev->prev->next = logdev->next;
    }
    if (logdev->physical_device->logical_devices == logdev) {
        logdev->physical_device->logical_devices = logdev->next;
    }

    // unbind any still-bound streams...
    SDL_AudioStream *next;
    for (SDL_AudioStream *stream = logdev->bound_streams; stream; stream = next) {
        SDL_LockMutex(stream->lock);
        next = stream->next_binding;
        stream->next_binding = NULL;
        stream->prev_binding = NULL;
        stream->bound_device = NULL;
        SDL_UnlockMutex(stream->lock);
    }

    UpdateAudioStreamFormatsPhysical(logdev->physical_device);
    SDL_free(logdev);
}

// this must not be called while `device` is still in a device list, or while a device's audio thread is still running.
static void DestroyPhysicalAudioDevice(SDL_AudioDevice *device)
{
    if (!device) {
        return;
    }

    // Destroy any logical devices that still exist...
    SDL_LockMutex(device->lock);  // don't use ObtainPhysicalAudioDeviceObj because we don't want to change refcounts while destroying.
    while (device->logical_devices) {
        DestroyLogicalAudioDevice(device->logical_devices);
    }

    ClosePhysicalAudioDevice(device);

    current_audio.impl.FreeDeviceHandle(device);

    SDL_UnlockMutex(device->lock);  // don't use ReleaseAudioDevice because we don't want to change refcounts while destroying.

    SDL_DestroyMutex(device->lock);
    SDL_DestroyCondition(device->close_cond);
    SDL_free(device->work_buffer);
    SDL_free(device->chmap);
    SDL_free(device->name);
    SDL_free(device);
}

// Don't hold the device lock when calling this, as we may destroy the device!
void UnrefPhysicalAudioDevice(SDL_AudioDevice *device)
{
    if (SDL_AtomicDecRef(&device->refcount)) {
        // take it out of the device list.
        SDL_LockRWLockForWriting(current_audio.device_hash_lock);
        if (SDL_RemoveFromHashTable(current_audio.device_hash, (const void *) (uintptr_t) device->instance_id)) {
            SDL_AddAtomicInt(device->recording ? &current_audio.recording_device_count : &current_audio.playback_device_count, -1);
        }
        SDL_UnlockRWLock(current_audio.device_hash_lock);
        DestroyPhysicalAudioDevice(device);  // ...and nuke it.
    }
}

void RefPhysicalAudioDevice(SDL_AudioDevice *device)
{
    SDL_AtomicIncRef(&device->refcount);
}

static SDL_AudioDevice *CreatePhysicalAudioDevice(const char *name, bool recording, const SDL_AudioSpec *spec, void *handle, SDL_AtomicInt *device_count)
{
    SDL_assert(name != NULL);

    SDL_LockRWLockForReading(current_audio.device_hash_lock);
    const int shutting_down = SDL_GetAtomicInt(&current_audio.shutting_down);
    SDL_UnlockRWLock(current_audio.device_hash_lock);
    if (shutting_down) {
        return NULL;  // we're shutting down, don't add any devices that are hotplugged at the last possible moment.
    }

    SDL_AudioDevice *device = (SDL_AudioDevice *)SDL_calloc(1, sizeof(SDL_AudioDevice));
    if (!device) {
        return NULL;
    }

    device->name = SDL_strdup(name);
    if (!device->name) {
        SDL_free(device);
        return NULL;
    }

    device->lock = SDL_CreateMutex();
    if (!device->lock) {
        SDL_free(device->name);
        SDL_free(device);
        return NULL;
    }

    device->close_cond = SDL_CreateCondition();
    if (!device->close_cond) {
        SDL_DestroyMutex(device->lock);
        SDL_free(device->name);
        SDL_free(device);
        return NULL;
    }

    SDL_SetAtomicInt(&device->shutdown, 0);
    SDL_SetAtomicInt(&device->zombie, 0);
    device->recording = recording;
    SDL_copyp(&device->spec, spec);
    SDL_copyp(&device->default_spec, spec);
    device->sample_frames = SDL_GetDefaultSampleFramesFromFreq(device->spec.freq);
    device->silence_value = SDL_GetSilenceValueForFormat(device->spec.format);
    device->handle = handle;

    device->instance_id = AssignAudioDeviceInstanceId(recording, /*islogical=*/false);

    SDL_LockRWLockForWriting(current_audio.device_hash_lock);
    if (SDL_InsertIntoHashTable(current_audio.device_hash, (const void *) (uintptr_t) device->instance_id, device, false)) {
        SDL_AddAtomicInt(device_count, 1);
    } else {
        SDL_DestroyCondition(device->close_cond);
        SDL_DestroyMutex(device->lock);
        SDL_free(device->name);
        SDL_free(device);
        device = NULL;
    }
    SDL_UnlockRWLock(current_audio.device_hash_lock);

    RefPhysicalAudioDevice(device);  // unref'd on device disconnect.
    return device;
}

static SDL_AudioDevice *CreateAudioRecordingDevice(const char *name, const SDL_AudioSpec *spec, void *handle)
{
    SDL_assert(current_audio.impl.HasRecordingSupport);
    return CreatePhysicalAudioDevice(name, true, spec, handle, &current_audio.recording_device_count);
}

static SDL_AudioDevice *CreateAudioPlaybackDevice(const char *name, const SDL_AudioSpec *spec, void *handle)
{
    return CreatePhysicalAudioDevice(name, false, spec, handle, &current_audio.playback_device_count);
}

// The audio backends call this when a new device is plugged in.
SDL_AudioDevice *SDL_AddAudioDevice(bool recording, const char *name, const SDL_AudioSpec *inspec, void *handle)
{
    // device handles MUST be unique! If the target reuses the same handle for hardware with both recording and playback interfaces, wrap it in a pointer you SDL_malloc'd!
    SDL_assert(SDL_FindPhysicalAudioDeviceByHandle(handle) == NULL);

    const SDL_AudioFormat default_format = recording ? DEFAULT_AUDIO_RECORDING_FORMAT : DEFAULT_AUDIO_PLAYBACK_FORMAT;
    const int default_channels = recording ? DEFAULT_AUDIO_RECORDING_CHANNELS : DEFAULT_AUDIO_PLAYBACK_CHANNELS;
    const int default_freq = recording ? DEFAULT_AUDIO_RECORDING_FREQUENCY : DEFAULT_AUDIO_PLAYBACK_FREQUENCY;

    SDL_AudioSpec spec;
    SDL_zero(spec);
    if (!inspec) {
        spec.format = default_format;
        spec.channels = default_channels;
        spec.freq = default_freq;
    } else {
        spec.format = (inspec->format != 0) ? inspec->format : default_format;
        spec.channels = (inspec->channels != 0) ? inspec->channels : default_channels;
        spec.freq = (inspec->freq != 0) ? inspec->freq : default_freq;
    }

    SDL_AudioDevice *device = recording ? CreateAudioRecordingDevice(name, &spec, handle) : CreateAudioPlaybackDevice(name, &spec, handle);

    // Add a device add event to the pending list, to be pushed when the event queue is pumped (away from any of our internal threads).
    if (device) {
        SDL_PendingAudioDeviceEvent *p = (SDL_PendingAudioDeviceEvent *) SDL_malloc(sizeof (SDL_PendingAudioDeviceEvent));
        if (p) {  // if allocation fails, you won't get an event, but we can't help that.
            p->type = SDL_EVENT_AUDIO_DEVICE_ADDED;
            p->devid = device->instance_id;
            p->next = NULL;
            SDL_LockRWLockForWriting(current_audio.device_hash_lock);
            SDL_assert(current_audio.pending_events_tail != NULL);
            SDL_assert(current_audio.pending_events_tail->next == NULL);
            current_audio.pending_events_tail->next = p;
            current_audio.pending_events_tail = p;
            SDL_UnlockRWLock(current_audio.device_hash_lock);
        }
    }

    return device;
}

// Called when a device is removed from the system, or it fails unexpectedly, from any thread, possibly even the audio device's thread.
void SDL_AudioDeviceDisconnected(SDL_AudioDevice *device)
{
    if (!device) {
        return;
    }

    // Save off removal info in a list so we can send events for each, next
    //  time the event queue pumps, in case something tries to close a device
    //  from an event filter, as this would risk deadlocks and other disasters
    //  if done from the device thread.
    SDL_PendingAudioDeviceEvent pending;
    pending.next = NULL;
    SDL_PendingAudioDeviceEvent *pending_tail = &pending;

    ObtainPhysicalAudioDeviceObj(device);

    SDL_LockRWLockForReading(current_audio.device_hash_lock);
    const SDL_AudioDeviceID devid = device->instance_id;
    const bool is_default_device = ((devid == current_audio.default_playback_device_id) || (devid == current_audio.default_recording_device_id));
    SDL_UnlockRWLock(current_audio.device_hash_lock);

    const bool first_disconnect = SDL_CompareAndSwapAtomicInt(&device->zombie, 0, 1);
    if (first_disconnect) {   // if already disconnected this device, don't do it twice.
        // Swap in "Zombie" versions of the usual platform interfaces, so the device will keep
        // making progress until the app closes it. Otherwise, streams might continue to
        // accumulate waste data that never drains, apps that depend on audio callbacks to
        // progress will freeze, etc.
        device->WaitDevice = ZombieWaitDevice;
        device->GetDeviceBuf = ZombieGetDeviceBuf;
        device->PlayDevice = ZombiePlayDevice;
        device->WaitRecordingDevice = ZombieWaitDevice;
        device->RecordDevice = ZombieRecordDevice;
        device->FlushRecording = ZombieFlushRecording;

        // on default devices, dump any logical devices that explicitly opened this device. Things that opened the system default can stay.
        // on non-default devices, dump everything.
        // (by "dump" we mean send a REMOVED event; the zombie will keep consuming audio data for these logical devices until explicitly closed.)
        for (SDL_LogicalAudioDevice *logdev = device->logical_devices; logdev; logdev = logdev->next) {
            if (!is_default_device || !logdev->opened_as_default) {  // if opened as a default, leave it on the zombie device for later migration.
                SDL_PendingAudioDeviceEvent *p = (SDL_PendingAudioDeviceEvent *) SDL_malloc(sizeof (SDL_PendingAudioDeviceEvent));
                if (p) {  // if this failed, no event for you, but you have deeper problems anyhow.
                    p->type = SDL_EVENT_AUDIO_DEVICE_REMOVED;
                    p->devid = logdev->instance_id;
                    p->next = NULL;
                    pending_tail->next = p;
                    pending_tail = p;
                }
            }
        }

        SDL_PendingAudioDeviceEvent *p = (SDL_PendingAudioDeviceEvent *) SDL_malloc(sizeof (SDL_PendingAudioDeviceEvent));
        if (p) {  // if this failed, no event for you, but you have deeper problems anyhow.
            p->type = SDL_EVENT_AUDIO_DEVICE_REMOVED;
            p->devid = device->instance_id;
            p->next = NULL;
            pending_tail->next = p;
            pending_tail = p;
        }
    }

    ReleaseAudioDevice(device);

    if (first_disconnect) {
        if (pending.next) {  // NULL if event is disabled or disaster struck.
            SDL_LockRWLockForWriting(current_audio.device_hash_lock);
            SDL_assert(current_audio.pending_events_tail != NULL);
            SDL_assert(current_audio.pending_events_tail->next == NULL);
            current_audio.pending_events_tail->next = pending.next;
            current_audio.pending_events_tail = pending_tail;
            SDL_UnlockRWLock(current_audio.device_hash_lock);
        }

        UnrefPhysicalAudioDevice(device);
    }
}


// stubs for audio drivers that don't need a specific entry point...

static void SDL_AudioThreadDeinit_Default(SDL_AudioDevice *device) { /* no-op. */ }
static bool SDL_AudioWaitDevice_Default(SDL_AudioDevice *device) { return true; /* no-op. */ }
static bool SDL_AudioPlayDevice_Default(SDL_AudioDevice *device, const Uint8 *buffer, int buffer_size) { return true; /* no-op. */ }
static bool SDL_AudioWaitRecordingDevice_Default(SDL_AudioDevice *device) { return true; /* no-op. */ }
static void SDL_AudioFlushRecording_Default(SDL_AudioDevice *device) { /* no-op. */ }
static void SDL_AudioCloseDevice_Default(SDL_AudioDevice *device) { /* no-op. */ }
static void SDL_AudioDeinitializeStart_Default(void) { /* no-op. */ }
static void SDL_AudioDeinitialize_Default(void) { /* no-op. */ }
static void SDL_AudioFreeDeviceHandle_Default(SDL_AudioDevice *device) { /* no-op. */ }

static void SDL_AudioThreadInit_Default(SDL_AudioDevice *device)
{
    SDL_SetCurrentThreadPriority(device->recording ? SDL_THREAD_PRIORITY_HIGH : SDL_THREAD_PRIORITY_TIME_CRITICAL);
}

static void SDL_AudioDetectDevices_Default(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    // you have to write your own implementation if these assertions fail.
    SDL_assert(current_audio.impl.OnlyHasDefaultPlaybackDevice);
    SDL_assert(current_audio.impl.OnlyHasDefaultRecordingDevice || !current_audio.impl.HasRecordingSupport);

    *default_playback = SDL_AddAudioDevice(false, DEFAULT_PLAYBACK_DEVNAME, NULL, (void *)((size_t)0x1));
    if (current_audio.impl.HasRecordingSupport) {
        *default_recording = SDL_AddAudioDevice(true, DEFAULT_RECORDING_DEVNAME, NULL, (void *)((size_t)0x2));
    }
}

static Uint8 *SDL_AudioGetDeviceBuf_Default(SDL_AudioDevice *device, int *buffer_size)
{
    *buffer_size = 0;
    return NULL;
}

static int SDL_AudioRecordDevice_Default(SDL_AudioDevice *device, void *buffer, int buflen)
{
    SDL_Unsupported();
    return -1;
}

static bool SDL_AudioOpenDevice_Default(SDL_AudioDevice *device)
{
    return SDL_Unsupported();
}

// Fill in stub functions for unused driver entry points. This lets us blindly call them without having to check for validity first.
static void CompleteAudioEntryPoints(void)
{
    #define FILL_STUB(x) if (!current_audio.impl.x) { current_audio.impl.x = SDL_Audio##x##_Default; }
    FILL_STUB(DetectDevices);
    FILL_STUB(OpenDevice);
    FILL_STUB(ThreadInit);
    FILL_STUB(ThreadDeinit);
    FILL_STUB(WaitDevice);
    FILL_STUB(PlayDevice);
    FILL_STUB(GetDeviceBuf);
    FILL_STUB(WaitRecordingDevice);
    FILL_STUB(RecordDevice);
    FILL_STUB(FlushRecording);
    FILL_STUB(CloseDevice);
    FILL_STUB(FreeDeviceHandle);
    FILL_STUB(DeinitializeStart);
    FILL_STUB(Deinitialize);
    #undef FILL_STUB
}

typedef struct FindLowestDeviceIDData
{
    const bool recording;
    SDL_AudioDeviceID highest;
    SDL_AudioDevice *result;
} FindLowestDeviceIDData;

static bool SDLCALL FindLowestDeviceID(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    FindLowestDeviceIDData *data = (FindLowestDeviceIDData *) userdata;
    const SDL_AudioDeviceID devid = (SDL_AudioDeviceID) (uintptr_t) key;
    // bit #0 of devid is set for playback devices and unset for recording.
    // bit #1 of devid is set for physical devices and unset for logical.
    const bool devid_recording = !(devid & (1 << 0));
    const bool isphysical = !!(devid & (1 << 1));
    if (isphysical && (devid_recording == data->recording) && (devid < data->highest)) {
        data->highest = devid;
        data->result = (SDL_AudioDevice *) value;
        SDL_assert(data->result->instance_id == devid);
    }
    return true;  // keep iterating.
}

static SDL_AudioDevice *GetFirstAddedAudioDevice(const bool recording)
{
    const SDL_AudioDeviceID highest = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;  // According to AssignAudioDeviceInstanceId, nothing can have a value this large.

    // (Device IDs increase as new devices are added, so the first device added has the lowest SDL_AudioDeviceID value.)
    FindLowestDeviceIDData data = { recording, highest, NULL };
    SDL_LockRWLockForReading(current_audio.device_hash_lock);
    SDL_IterateHashTable(current_audio.device_hash, FindLowestDeviceID, &data);
    SDL_UnlockRWLock(current_audio.device_hash_lock);
    return data.result;
}

static Uint32 SDLCALL HashAudioDeviceID(void *userdata, const void *key)
{
    // shift right 2, to dump the first two bits, since these are flags
    //  (recording vs playback, logical vs physical) and the rest are unique incrementing integers.
    return ((Uint32) ((uintptr_t) key)) >> 2;
}

// !!! FIXME: the video subsystem does SDL_VideoInit, not SDL_InitVideo. Make this match.
bool SDL_InitAudio(const char *driver_name)
{
    if (SDL_GetCurrentAudioDriver()) {
        SDL_QuitAudio(); // shutdown driver if already running.
    }

    // make sure device IDs start at 2 (because of SDL2 legacy interface), but don't reset the counter on each init, in case the app is holding an old device ID somewhere.
    SDL_CompareAndSwapAtomicInt(&last_device_instance_id, 0, 2);

    SDL_ChooseAudioConverters();
    SDL_SetupAudioResampler();

    SDL_RWLock *device_hash_lock = SDL_CreateRWLock();  // create this early, so if it fails we don't have to tear down the whole audio subsystem.
    if (!device_hash_lock) {
        return false;
    }

    SDL_HashTable *device_hash = SDL_CreateHashTable(0, false, HashAudioDeviceID, SDL_KeyMatchID, NULL, NULL);
    if (!device_hash) {
        SDL_DestroyRWLock(device_hash_lock);
        return false;
    }

    // Select the proper audio driver
    if (!driver_name) {
        driver_name = SDL_GetHint(SDL_HINT_AUDIO_DRIVER);
    }

    bool initialized = false;
    bool tried_to_init = false;

    if (driver_name && *driver_name != 0) {
        char *driver_name_copy = SDL_strdup(driver_name);
        const char *driver_attempt = driver_name_copy;

        if (!driver_name_copy) {
            SDL_DestroyRWLock(device_hash_lock);
            SDL_DestroyHashTable(device_hash);
            return false;
        }

        while (driver_attempt && *driver_attempt != 0 && !initialized) {
            char *driver_attempt_end = SDL_strchr(driver_attempt, ',');
            if (driver_attempt_end) {
                *driver_attempt_end = '\0';
            }

            // SDL 1.2 uses the name "dsound", so we'll support both.
            if (SDL_strcmp(driver_attempt, "dsound") == 0) {
                driver_attempt = "directsound";
            } else if (SDL_strcmp(driver_attempt, "pulse") == 0) {  // likewise, "pulse" was renamed to "pulseaudio"
                driver_attempt = "pulseaudio";
            }

            for (int i = 0; bootstrap[i]; ++i) {
                if (!bootstrap[i]->is_preferred && SDL_strcasecmp(bootstrap[i]->name, driver_attempt) == 0) {
                    tried_to_init = true;
                    SDL_zero(current_audio);
                    current_audio.pending_events_tail = &current_audio.pending_events;
                    current_audio.device_hash_lock = device_hash_lock;
                    current_audio.device_hash = device_hash;
                    if (bootstrap[i]->init(&current_audio.impl)) {
                        current_audio.name = bootstrap[i]->name;
                        current_audio.desc = bootstrap[i]->desc;
                        initialized = true;
                        break;
                    }
                }
            }

            driver_attempt = (driver_attempt_end) ? (driver_attempt_end + 1) : NULL;
        }

        SDL_free(driver_name_copy);
    } else {
        for (int i = 0; (!initialized) && (bootstrap[i]); ++i) {
            if (bootstrap[i]->demand_only) {
                continue;
            }

            tried_to_init = true;
            SDL_zero(current_audio);
            current_audio.pending_events_tail = &current_audio.pending_events;
            current_audio.device_hash_lock = device_hash_lock;
            current_audio.device_hash = device_hash;
            if (bootstrap[i]->init(&current_audio.impl)) {
                current_audio.name = bootstrap[i]->name;
                current_audio.desc = bootstrap[i]->desc;
                initialized = true;
            }
        }
    }

    if (!initialized) {
        // specific drivers will set the error message if they fail, but otherwise we do it here.
        if (!tried_to_init) {
            if (driver_name) {
                SDL_SetError("Audio target '%s' not available", driver_name);
            } else {
                SDL_SetError("No available audio device");
            }
        }

        SDL_DestroyRWLock(device_hash_lock);
        SDL_DestroyHashTable(device_hash);
        SDL_zero(current_audio);
        return false;  // No driver was available, so fail.
    }

    CompleteAudioEntryPoints();

    // Make sure we have a list of devices available at startup...
    SDL_AudioDevice *default_playback = NULL;
    SDL_AudioDevice *default_recording = NULL;
    current_audio.impl.DetectDevices(&default_playback, &default_recording);

    // If no default was _ever_ specified, just take the first device we see, if any.
    if (!default_playback) {
        default_playback = GetFirstAddedAudioDevice(/*recording=*/false);
    }

    if (!default_recording) {
        default_recording = GetFirstAddedAudioDevice(/*recording=*/true);
    }

    if (default_playback) {
        current_audio.default_playback_device_id = default_playback->instance_id;
        RefPhysicalAudioDevice(default_playback);  // extra ref on default devices.
    }

    if (default_recording) {
        current_audio.default_recording_device_id = default_recording->instance_id;
        RefPhysicalAudioDevice(default_recording);  // extra ref on default devices.
    }

    return true;
}

static bool SDLCALL DestroyOnePhysicalAudioDevice(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    // bit #1 of devid is set for physical devices and unset for logical.
    const SDL_AudioDeviceID devid = (SDL_AudioDeviceID) (uintptr_t) key;
    const bool isphysical = !!(devid & (1<<1));
    if (isphysical) {
        SDL_AudioDevice *dev = (SDL_AudioDevice *) value;

        SDL_assert(dev->instance_id == devid);
        DestroyPhysicalAudioDevice(dev);
    }
    return true;  // keep iterating.
}

void SDL_QuitAudio(void)
{
    if (!current_audio.name) {  // not initialized?!
        return;
    }

    current_audio.impl.DeinitializeStart();

    // Destroy any audio streams that still exist...
    while (current_audio.existing_streams) {
        SDL_DestroyAudioStream(current_audio.existing_streams);
    }

    SDL_LockRWLockForWriting(current_audio.device_hash_lock);
    SDL_SetAtomicInt(&current_audio.shutting_down, 1);
    SDL_HashTable *device_hash = current_audio.device_hash;
    current_audio.device_hash = NULL;
    SDL_PendingAudioDeviceEvent *pending_events = current_audio.pending_events.next;
    current_audio.pending_events.next = NULL;
    SDL_SetAtomicInt(&current_audio.playback_device_count, 0);
    SDL_SetAtomicInt(&current_audio.recording_device_count, 0);
    SDL_UnlockRWLock(current_audio.device_hash_lock);

    SDL_PendingAudioDeviceEvent *pending_next = NULL;
    for (SDL_PendingAudioDeviceEvent *i = pending_events; i; i = pending_next) {
        pending_next = i->next;
        SDL_free(i);
    }

    SDL_IterateHashTable(device_hash, DestroyOnePhysicalAudioDevice, NULL);

    // Free the driver data
    current_audio.impl.Deinitialize();

    SDL_DestroyRWLock(current_audio.device_hash_lock);
    SDL_DestroyHashTable(device_hash);

    SDL_zero(current_audio);
}


void SDL_AudioThreadFinalize(SDL_AudioDevice *device)
{
}

static void MixFloat32Audio(float *dst, const float *src, const int buffer_size)
{
    if (!SDL_MixAudio((Uint8 *) dst, (const Uint8 *) src, SDL_AUDIO_F32, buffer_size, 1.0f)) {
        SDL_assert(!"This shouldn't happen.");
    }
}


// Playback device thread. This is split into chunks, so backends that need to control this directly can use the pieces they need without duplicating effort.

void SDL_PlaybackAudioThreadSetup(SDL_AudioDevice *device)
{
    SDL_assert(!device->recording);
    current_audio.impl.ThreadInit(device);
}

bool SDL_PlaybackAudioThreadIterate(SDL_AudioDevice *device)
{
    SDL_assert(!device->recording);

    SDL_LockMutex(device->lock);

    if (SDL_GetAtomicInt(&device->shutdown)) {
        SDL_UnlockMutex(device->lock);
        return false;  // we're done, shut it down.
    }

    bool failed = false;
    int buffer_size = device->buffer_size;
    Uint8 *device_buffer = device->GetDeviceBuf(device, &buffer_size);
    if (buffer_size == 0) {
        // WASAPI (maybe others, later) does this to say "just abandon this iteration and try again next time."
    } else if (!device_buffer) {
        failed = true;
    } else {
        SDL_assert(buffer_size <= device->buffer_size);  // you can ask for less, but not more.
        SDL_assert(AudioDeviceCanUseSimpleCopy(device) == device->simple_copy);  // make sure this hasn't gotten out of sync.

        // can we do a basic copy without silencing/mixing the buffer? This is an extremely likely scenario, so we special-case it.
        if (device->simple_copy) {
            SDL_LogicalAudioDevice *logdev = device->logical_devices;
            SDL_AudioStream *stream = logdev->bound_streams;

            // We should have updated this elsewhere if the format changed!
            SDL_assert(SDL_AudioSpecsEqual(&stream->dst_spec, &device->spec, NULL, NULL));

            const int br = SDL_GetAtomicInt(&logdev->paused) ? 0 : SDL_GetAudioStreamDataAdjustGain(stream, device_buffer, buffer_size, logdev->gain);
            if (br < 0) {  // Probably OOM. Kill the audio device; the whole thing is likely dying soon anyhow.
                failed = true;
                SDL_memset(device_buffer, device->silence_value, buffer_size);  // just supply silence to the device before we die.
            } else if (br < buffer_size) {
                SDL_memset(device_buffer + br, device->silence_value, buffer_size - br);  // silence whatever we didn't write to.
            }

            // generally channel maps will line up, but if the audio stream's chmap has been explicitly changed, do a final swizzle to device layout.
            if ((br > 0) && (!SDL_AudioChannelMapsEqual(device->spec.channels, stream->dst_chmap, device->chmap))) {
                ConvertAudio(br / SDL_AUDIO_FRAMESIZE(device->spec), device_buffer, device->spec.format, device->spec.channels, NULL,
                             device_buffer, device->spec.format, device->spec.channels, device->chmap, NULL, 1.0f);
            }
        } else {  // need to actually mix (or silence the buffer)
            float *final_mix_buffer = (float *) ((device->spec.format == SDL_AUDIO_F32) ? device_buffer : device->mix_buffer);
            const int needed_samples = buffer_size / SDL_AUDIO_BYTESIZE(device->spec.format);
            const int work_buffer_size = needed_samples * sizeof (float);
            SDL_AudioSpec outspec;

            SDL_assert(work_buffer_size <= device->work_buffer_size);

            SDL_copyp(&outspec, &device->spec);
            outspec.format = SDL_AUDIO_F32;

            SDL_memset(final_mix_buffer, '\0', work_buffer_size);  // start with silence.

            for (SDL_LogicalAudioDevice *logdev = device->logical_devices; logdev; logdev = logdev->next) {
                if (SDL_GetAtomicInt(&logdev->paused)) {
                    continue;  // paused? Skip this logical device.
                }

                const SDL_AudioPostmixCallback postmix = logdev->postmix;
                float *mix_buffer = final_mix_buffer;
                if (postmix) {
                    mix_buffer = device->postmix_buffer;
                    SDL_memset(mix_buffer, '\0', work_buffer_size);  // start with silence.
                }

                for (SDL_AudioStream *stream = logdev->bound_streams; stream; stream = stream->next_binding) {
                    // We should have updated this elsewhere if the format changed!
                    SDL_assert(SDL_AudioSpecsEqual(&stream->dst_spec, &outspec, NULL, NULL));

                    /* this will hold a lock on `stream` while getting. We don't explicitly lock the streams
                       for iterating here because the binding linked list can only change while the device lock is held.
                       (we _do_ lock the stream during binding/unbinding to make sure that two threads can't try to bind
                       the same stream to different devices at the same time, though.) */
                    const int br = SDL_GetAudioStreamDataAdjustGain(stream, device->work_buffer, work_buffer_size, logdev->gain);
                    if (br < 0) {  // Probably OOM. Kill the audio device; the whole thing is likely dying soon anyhow.
                        failed = true;
                        break;
                    } else if (br > 0) {  // it's okay if we get less than requested, we mix what we have.
                        // generally channel maps will line up, but if the audio stream's chmap has been explicitly changed, do a final swizzle to device layout.
                        if (!SDL_AudioChannelMapsEqual(device->spec.channels, stream->dst_chmap, device->chmap)) {
                            ConvertAudio(br / SDL_AUDIO_FRAMESIZE(device->spec), device->work_buffer, device->spec.format, device->spec.channels, NULL,
                                         device->work_buffer, device->spec.format, device->spec.channels, device->chmap, NULL, 1.0f);
                        }
                        MixFloat32Audio(mix_buffer, (float *) device->work_buffer, br);
                    }
                }

                if (postmix) {
                    SDL_assert(mix_buffer == device->postmix_buffer);
                    postmix(logdev->postmix_userdata, &outspec, mix_buffer, work_buffer_size);
                    MixFloat32Audio(final_mix_buffer, mix_buffer, work_buffer_size);
                }
            }

            if (((Uint8 *) final_mix_buffer) != device_buffer) {
                // !!! FIXME: we can't promise the device buf is aligned/padded for SIMD.
                //ConvertAudio(needed_samples / device->spec.channels, final_mix_buffer, SDL_AUDIO_F32, device->spec.channels, NULL, device_buffer, device->spec.format, device->spec.channels, NULL, NULL, 1.0f);
                ConvertAudio(needed_samples / device->spec.channels, final_mix_buffer, SDL_AUDIO_F32, device->spec.channels, NULL, device->work_buffer, device->spec.format, device->spec.channels, NULL, NULL, 1.0f);
                SDL_memcpy(device_buffer, device->work_buffer, buffer_size);
            }
        }

        // PlayDevice SHOULD NOT BLOCK, as we are holding a lock right now. Block in WaitDevice instead!
        if (!device->PlayDevice(device, device_buffer, buffer_size)) {
            failed = true;
        }
    }

    SDL_UnlockMutex(device->lock);

    if (failed) {
        SDL_AudioDeviceDisconnected(device);  // doh.
    }

    return true;  // always go on if not shutting down, even if device failed.
}

void SDL_PlaybackAudioThreadShutdown(SDL_AudioDevice *device)
{
    SDL_assert(!device->recording);
    const int frames = device->buffer_size / SDL_AUDIO_FRAMESIZE(device->spec);
    // Wait for the audio to drain if device didn't die.
    if (!SDL_GetAtomicInt(&device->zombie)) {
        SDL_Delay(((frames * 1000) / device->spec.freq) * 2);
    }
    current_audio.impl.ThreadDeinit(device);
    SDL_AudioThreadFinalize(device);
}

static int SDLCALL PlaybackAudioThread(void *devicep)  // thread entry point
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)devicep;
    SDL_assert(device != NULL);
    SDL_assert(!device->recording);
    SDL_PlaybackAudioThreadSetup(device);

    while (SDL_PlaybackAudioThreadIterate(device)) {
        if (!device->WaitDevice(device)) {
            SDL_AudioDeviceDisconnected(device);  // doh. (but don't break out of the loop, just be a zombie for now!)
        }
    }

    SDL_PlaybackAudioThreadShutdown(device);
    return 0;
}



// Recording device thread. This is split into chunks, so backends that need to control this directly can use the pieces they need without duplicating effort.

void SDL_RecordingAudioThreadSetup(SDL_AudioDevice *device)
{
    SDL_assert(device->recording);
    current_audio.impl.ThreadInit(device);
}

bool SDL_RecordingAudioThreadIterate(SDL_AudioDevice *device)
{
    SDL_assert(device->recording);

    SDL_LockMutex(device->lock);

    if (SDL_GetAtomicInt(&device->shutdown)) {
        SDL_UnlockMutex(device->lock);
        return false;  // we're done, shut it down.
    }

    bool failed = false;

    if (!device->logical_devices) {
        device->FlushRecording(device); // nothing wants data, dump anything pending.
    } else {
        // this SHOULD NOT BLOCK, as we are holding a lock right now. Block in WaitRecordingDevice!
        int br = device->RecordDevice(device, device->work_buffer, device->buffer_size);
        if (br < 0) {  // uhoh, device failed for some reason!
            failed = true;
        } else if (br > 0) {  // queue the new data to each bound stream.
            for (SDL_LogicalAudioDevice *logdev = device->logical_devices; logdev; logdev = logdev->next) {
                if (SDL_GetAtomicInt(&logdev->paused)) {
                    continue;  // paused? Skip this logical device.
                }

                void *output_buffer = device->work_buffer;

                // I don't know why someone would want a postmix on a recording device, but we offer it for API consistency.
                if (logdev->postmix || (logdev->gain != 1.0f)) {
                    // move to float format.
                    SDL_AudioSpec outspec;
                    SDL_copyp(&outspec, &device->spec);
                    outspec.format = SDL_AUDIO_F32;
                    output_buffer = device->postmix_buffer;
                    const int frames = br / SDL_AUDIO_FRAMESIZE(device->spec);
                    br = frames * SDL_AUDIO_FRAMESIZE(outspec);
                    ConvertAudio(frames, device->work_buffer, device->spec.format, outspec.channels, NULL, device->postmix_buffer, SDL_AUDIO_F32, outspec.channels, NULL, NULL, logdev->gain);
                    if (logdev->postmix) {
                        logdev->postmix(logdev->postmix_userdata, &outspec, device->postmix_buffer, br);
                    }
                }

                for (SDL_AudioStream *stream = logdev->bound_streams; stream; stream = stream->next_binding) {
                    // We should have updated this elsewhere if the format changed!
                    SDL_assert(stream->src_spec.format == ((logdev->postmix || (logdev->gain != 1.0f)) ? SDL_AUDIO_F32 : device->spec.format));
                    SDL_assert(stream->src_spec.channels == device->spec.channels);
                    SDL_assert(stream->src_spec.freq == device->spec.freq);

                    void *final_buf = output_buffer;

                    // generally channel maps will line up, but if the audio stream's chmap has been explicitly changed, do a final swizzle to stream layout.
                    if (!SDL_AudioChannelMapsEqual(device->spec.channels, stream->src_chmap, device->chmap)) {
                        final_buf = device->mix_buffer;  // this is otherwise unused on recording devices, so it makes convenient scratch space here.
                        ConvertAudio(br / SDL_AUDIO_FRAMESIZE(device->spec), output_buffer, device->spec.format, device->spec.channels, NULL,
                                     final_buf, device->spec.format, device->spec.channels, stream->src_chmap, NULL, 1.0f);
                    }

                    /* this will hold a lock on `stream` while putting. We don't explicitly lock the streams
                       for iterating here because the binding linked list can only change while the device lock is held.
                       (we _do_ lock the stream during binding/unbinding to make sure that two threads can't try to bind
                       the same stream to different devices at the same time, though.) */
                    if (!SDL_PutAudioStreamData(stream, final_buf, br)) {
                        // oh crud, we probably ran out of memory. This is possibly an overreaction to kill the audio device, but it's likely the whole thing is going down in a moment anyhow.
                        failed = true;
                        break;
                    }
                }
            }
        }
    }

    SDL_UnlockMutex(device->lock);

    if (failed) {
        SDL_AudioDeviceDisconnected(device);  // doh.
    }

    return true;  // always go on if not shutting down, even if device failed.
}

void SDL_RecordingAudioThreadShutdown(SDL_AudioDevice *device)
{
    SDL_assert(device->recording);
    device->FlushRecording(device);
    current_audio.impl.ThreadDeinit(device);
    SDL_AudioThreadFinalize(device);
}

static int SDLCALL RecordingAudioThread(void *devicep)  // thread entry point
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)devicep;
    SDL_assert(device != NULL);
    SDL_assert(device->recording);
    SDL_RecordingAudioThreadSetup(device);

    do {
        if (!device->WaitRecordingDevice(device)) {
            SDL_AudioDeviceDisconnected(device);  // doh. (but don't break out of the loop, just be a zombie for now!)
        }
    } while (SDL_RecordingAudioThreadIterate(device));

    SDL_RecordingAudioThreadShutdown(device);
    return 0;
}

typedef struct CountAudioDevicesData
{
    int devs_seen;
    const int num_devices;
    SDL_AudioDeviceID *result;
    const bool recording;
} CountAudioDevicesData;

static bool SDLCALL CountAudioDevices(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    CountAudioDevicesData *data = (CountAudioDevicesData *) userdata;
    const SDL_AudioDeviceID devid = (SDL_AudioDeviceID) (uintptr_t) key;
    // bit #0 of devid is set for playback devices and unset for recording.
    // bit #1 of devid is set for physical devices and unset for logical.
    const bool devid_recording = !(devid & (1<<0));
    const bool isphysical = !!(devid & (1<<1));
    if (isphysical && (devid_recording == data->recording)) {
        SDL_assert(data->devs_seen < data->num_devices);
        data->result[data->devs_seen++] = devid;
    }
    return true;  // keep iterating.
}

static SDL_AudioDeviceID *GetAudioDevices(int *count, bool recording)
{
    SDL_AudioDeviceID *result = NULL;
    int num_devices = 0;

    if (SDL_GetCurrentAudioDriver()) {
        SDL_LockRWLockForReading(current_audio.device_hash_lock);
        {
            num_devices = SDL_GetAtomicInt(recording ? &current_audio.recording_device_count : &current_audio.playback_device_count);
            result = (SDL_AudioDeviceID *) SDL_malloc((num_devices + 1) * sizeof (SDL_AudioDeviceID));
            if (result) {
                CountAudioDevicesData data = { 0, num_devices, result, recording };
                SDL_IterateHashTable(current_audio.device_hash, CountAudioDevices, &data);
                SDL_assert(data.devs_seen == num_devices);
                result[data.devs_seen] = 0;  // null-terminated.
            }
        }
        SDL_UnlockRWLock(current_audio.device_hash_lock);
    } else {
        SDL_SetError("Audio subsystem is not initialized");
    }

    if (count) {
        if (result) {
            *count = num_devices;
        } else {
            *count = 0;
        }
    }
    return result;
}

SDL_AudioDeviceID *SDL_GetAudioPlaybackDevices(int *count)
{
    return GetAudioDevices(count, false);
}

SDL_AudioDeviceID *SDL_GetAudioRecordingDevices(int *count)
{
    return GetAudioDevices(count, true);
}

typedef struct FindAudioDeviceByCallbackData
{
    bool (*callback)(SDL_AudioDevice *device, void *userdata);
    void *userdata;
    SDL_AudioDevice *retval;
} FindAudioDeviceByCallbackData;

static bool SDLCALL FindAudioDeviceByCallback(void *userdata, const SDL_HashTable *table, const void *key, const void *value)
{
    FindAudioDeviceByCallbackData *data = (FindAudioDeviceByCallbackData *) userdata;
    const SDL_AudioDeviceID devid = (SDL_AudioDeviceID) (uintptr_t) key;
    // bit #1 of devid is set for physical devices and unset for logical.
    const bool isphysical = !!(devid & (1<<1));
    if (isphysical) {
        SDL_AudioDevice *device = (SDL_AudioDevice *) value;
        if (data->callback(device, data->userdata)) {  // found it?
            data->retval = device;
            SDL_assert(data->retval->instance_id == devid);
            return false;  // stop iterating, we found it.
        }
    }
    return true;  // keep iterating.
}

// !!! FIXME: SDL convention is for userdata to come first in the callback's params. Fix this at some point.
SDL_AudioDevice *SDL_FindPhysicalAudioDeviceByCallback(bool (*callback)(SDL_AudioDevice *device, void *userdata), void *userdata)
{
    if (!SDL_GetCurrentAudioDriver()) {
        SDL_SetError("Audio subsystem is not initialized");
        return NULL;
    }

    FindAudioDeviceByCallbackData data = { callback, userdata, NULL };
    SDL_LockRWLockForReading(current_audio.device_hash_lock);
    SDL_IterateHashTable(current_audio.device_hash, FindAudioDeviceByCallback, &data);
    SDL_UnlockRWLock(current_audio.device_hash_lock);

    if (!data.retval) {
        SDL_SetError("Device not found");
    }

    return data.retval;
}

static bool TestDeviceHandleCallback(SDL_AudioDevice *device, void *handle)
{
    return device->handle == handle;
}

SDL_AudioDevice *SDL_FindPhysicalAudioDeviceByHandle(void *handle)
{
    return SDL_FindPhysicalAudioDeviceByCallback(TestDeviceHandleCallback, handle);
}

const char *SDL_GetAudioDeviceName(SDL_AudioDeviceID devid)
{
    // bit #1 of devid is set for physical devices and unset for logical.
    const bool islogical = !(devid & (1<<1));
    const char *result = NULL;
    const void *vdev = NULL;

    if (!SDL_GetCurrentAudioDriver()) {
        SDL_SetError("Audio subsystem is not initialized");
    } else {
        // This does not call ObtainPhysicalAudioDevice() because the device's name never changes, so
        // it doesn't have to lock the whole device. However, just to make sure the device pointer itself
        // remains valid (in case the device is unplugged at the wrong moment), we hold the
        // device_hash_lock while we copy the string.
        SDL_LockRWLockForReading(current_audio.device_hash_lock);
        SDL_FindInHashTable(current_audio.device_hash, (const void *) (uintptr_t) devid, &vdev);
        if (!vdev) {
            SDL_SetError("Invalid audio device instance ID");
        } else if (islogical) {
            const SDL_LogicalAudioDevice *logdev = (const SDL_LogicalAudioDevice *) vdev;
            SDL_assert(logdev->instance_id == devid);
            result = SDL_GetPersistentString(logdev->physical_device->name);
        } else {
            const SDL_AudioDevice *device = (const SDL_AudioDevice *) vdev;
            SDL_assert(device->instance_id == devid);
            result = SDL_GetPersistentString(device->name);
        }
        SDL_UnlockRWLock(current_audio.device_hash_lock);
    }

    return result;
}

bool SDL_GetAudioDeviceFormat(SDL_AudioDeviceID devid, SDL_AudioSpec *spec, int *sample_frames)
{
    if (!spec) {
        return SDL_InvalidParamError("spec");
    }

    bool result = false;
    SDL_AudioDevice *device = ObtainPhysicalAudioDeviceDefaultAllowed(devid);
    if (device) {
        SDL_copyp(spec, &device->spec);
        if (sample_frames) {
            *sample_frames = device->sample_frames;
        }
        result = true;
    }
    ReleaseAudioDevice(device);

    return result;
}

int *SDL_GetAudioDeviceChannelMap(SDL_AudioDeviceID devid, int *count)
{
    int *result = NULL;
    int channels = 0;
    SDL_AudioDevice *device = ObtainPhysicalAudioDeviceDefaultAllowed(devid);
    if (device) {
        channels = device->spec.channels;
        result = SDL_ChannelMapDup(device->chmap, channels);
    }
    ReleaseAudioDevice(device);

    if (count) {
        *count = channels;
    }

    return result;
}


// this is awkward, but this makes sure we can release the device lock
//  so the device thread can terminate but also not have two things
//  race to close or open the device while the lock is unprotected.
// you hold the lock when calling this, it will release the lock and
//  wait while the shutdown flag is set.
// BE CAREFUL WITH THIS.
static void SerializePhysicalDeviceClose(SDL_AudioDevice *device)
{
    while (SDL_GetAtomicInt(&device->shutdown)) {
        SDL_WaitCondition(device->close_cond, device->lock);
    }
}

// this expects the device lock to be held.
static void ClosePhysicalAudioDevice(SDL_AudioDevice *device)
{
    SerializePhysicalDeviceClose(device);

    SDL_SetAtomicInt(&device->shutdown, 1);

    // YOU MUST PROTECT KEY POINTS WITH SerializePhysicalDeviceClose() WHILE THE THREAD JOINS
    SDL_UnlockMutex(device->lock);

    if (device->thread) {
        SDL_WaitThread(device->thread, NULL);
        device->thread = NULL;
    }

    if (device->currently_opened) {
        current_audio.impl.CloseDevice(device);  // if ProvidesOwnCallbackThread, this must join on any existing device thread before returning!
        device->currently_opened = false;
        device->hidden = NULL;  // just in case.
    }

    SDL_LockMutex(device->lock);
    SDL_SetAtomicInt(&device->shutdown, 0);  // ready to go again.
    SDL_BroadcastCondition(device->close_cond);  // release anyone waiting in SerializePhysicalDeviceClose; they'll still block until we release device->lock, though.

    SDL_aligned_free(device->work_buffer);
    device->work_buffer = NULL;

    SDL_aligned_free(device->mix_buffer);
    device->mix_buffer = NULL;

    SDL_aligned_free(device->postmix_buffer);
    device->postmix_buffer = NULL;

    SDL_copyp(&device->spec, &device->default_spec);
    device->sample_frames = 0;
    device->silence_value = SDL_GetSilenceValueForFormat(device->spec.format);
}

void SDL_CloseAudioDevice(SDL_AudioDeviceID devid)
{
    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(devid, &device);
    if (logdev) {
        DestroyLogicalAudioDevice(logdev);
    }

    if (device) {
        if (!device->logical_devices) {  // no more logical devices? Close the physical device, too.
            ClosePhysicalAudioDevice(device);
        }
        UnrefPhysicalAudioDevice(device);  // one reference for each logical device.
    }

    ReleaseAudioDevice(device);
}


static SDL_AudioFormat ParseAudioFormatString(const char *string)
{
    if (string) {
        #define CHECK_FMT_STRING(x) if (SDL_strcmp(string, #x) == 0) { return SDL_AUDIO_##x; }
        CHECK_FMT_STRING(U8);
        CHECK_FMT_STRING(S8);
        CHECK_FMT_STRING(S16LE);
        CHECK_FMT_STRING(S16BE);
        CHECK_FMT_STRING(S16);
        CHECK_FMT_STRING(S32LE);
        CHECK_FMT_STRING(S32BE);
        CHECK_FMT_STRING(S32);
        CHECK_FMT_STRING(F32LE);
        CHECK_FMT_STRING(F32BE);
        CHECK_FMT_STRING(F32);
        #undef CHECK_FMT_STRING
    }
    return SDL_AUDIO_UNKNOWN;
}

static void PrepareAudioFormat(bool recording, SDL_AudioSpec *spec)
{
    if (spec->freq == 0) {
        spec->freq = recording ? DEFAULT_AUDIO_RECORDING_FREQUENCY : DEFAULT_AUDIO_PLAYBACK_FREQUENCY;

        const char *hint = SDL_GetHint(SDL_HINT_AUDIO_FREQUENCY);
        if (hint) {
            const int val = SDL_atoi(hint);
            if (val > 0) {
                spec->freq = val;
            }
        }
    }

    if (spec->channels == 0) {
        spec->channels = recording ? DEFAULT_AUDIO_RECORDING_CHANNELS : DEFAULT_AUDIO_PLAYBACK_CHANNELS;

        const char *hint = SDL_GetHint(SDL_HINT_AUDIO_CHANNELS);
        if (hint) {
            const int val = SDL_atoi(hint);
            if (val > 0) {
                spec->channels = val;
            }
        }
    }

    if (spec->format == 0) {
        const SDL_AudioFormat val = ParseAudioFormatString(SDL_GetHint(SDL_HINT_AUDIO_FORMAT));
        spec->format = (val != SDL_AUDIO_UNKNOWN) ? val : (recording ? DEFAULT_AUDIO_RECORDING_FORMAT : DEFAULT_AUDIO_PLAYBACK_FORMAT);
    }
}

void SDL_UpdatedAudioDeviceFormat(SDL_AudioDevice *device)
{
    device->silence_value = SDL_GetSilenceValueForFormat(device->spec.format);
    device->buffer_size = device->sample_frames * SDL_AUDIO_FRAMESIZE(device->spec);
    device->work_buffer_size = device->sample_frames * sizeof (float) * device->spec.channels;
    device->work_buffer_size = SDL_max(device->buffer_size, device->work_buffer_size);  // just in case we end up with a 64-bit audio format at some point.
}

char *SDL_GetAudioThreadName(SDL_AudioDevice *device, char *buf, size_t buflen)
{
    (void)SDL_snprintf(buf, buflen, "SDLAudio%c%d", (device->recording) ? 'C' : 'P', (int) device->instance_id);
    return buf;
}


// this expects the device lock to be held.
static bool OpenPhysicalAudioDevice(SDL_AudioDevice *device, const SDL_AudioSpec *inspec)
{
    SerializePhysicalDeviceClose(device);  // make sure another thread that's closing didn't release the lock to let the device thread join...

    if (device->currently_opened) {
        return true;  // we're already good.
    }

    // Just pretend to open a zombie device. It can still collect logical devices on a default device under the assumption they will all migrate when the default device is officially changed.
    if (SDL_GetAtomicInt(&device->zombie)) {
        return true;  // Braaaaaaaaains.
    }

    // These start with the backend's implementation, but we might swap them out with zombie versions later.
    device->WaitDevice = current_audio.impl.WaitDevice;
    device->PlayDevice = current_audio.impl.PlayDevice;
    device->GetDeviceBuf = current_audio.impl.GetDeviceBuf;
    device->WaitRecordingDevice = current_audio.impl.WaitRecordingDevice;
    device->RecordDevice = current_audio.impl.RecordDevice;
    device->FlushRecording = current_audio.impl.FlushRecording;

    SDL_AudioSpec spec;
    SDL_copyp(&spec, inspec ? inspec : &device->default_spec);
    PrepareAudioFormat(device->recording, &spec);

    /* We impose a simple minimum on device formats. This prevents something low quality, like an old game using S8/8000Hz audio,
       from ruining a music thing playing at CD quality that tries to open later, or some VoIP library that opens for mono output
       ruining your surround-sound game because it got there first.
       These are just requests! The backend may change any of these values during OpenDevice method! */

    const SDL_AudioFormat minimum_format = device->recording ? DEFAULT_AUDIO_RECORDING_FORMAT : DEFAULT_AUDIO_PLAYBACK_FORMAT;
    const int minimum_channels = device->recording ? DEFAULT_AUDIO_RECORDING_CHANNELS : DEFAULT_AUDIO_PLAYBACK_CHANNELS;
    const int minimum_freq = device->recording ? DEFAULT_AUDIO_RECORDING_FREQUENCY : DEFAULT_AUDIO_PLAYBACK_FREQUENCY;

    device->spec.format = (SDL_AUDIO_BITSIZE(minimum_format) >= SDL_AUDIO_BITSIZE(spec.format)) ? minimum_format : spec.format;
    device->spec.channels = SDL_max(minimum_channels, spec.channels);
    device->spec.freq = SDL_max(minimum_freq, spec.freq);
    device->sample_frames = SDL_GetDefaultSampleFramesFromFreq(device->spec.freq);
    SDL_UpdatedAudioDeviceFormat(device);  // start this off sane.

    device->currently_opened = true;  // mark this true even if impl.OpenDevice fails, so we know to clean up.
    if (!current_audio.impl.OpenDevice(device)) {
        ClosePhysicalAudioDevice(device);  // clean up anything the backend left half-initialized.
        return false;
    }

    SDL_UpdatedAudioDeviceFormat(device);  // in case the backend changed things and forgot to call this.

    // Allocate a scratch audio buffer
    device->work_buffer = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
    if (!device->work_buffer) {
        ClosePhysicalAudioDevice(device);
        return false;
    }

    if (device->spec.format != SDL_AUDIO_F32) {
        device->mix_buffer = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
        if (!device->mix_buffer) {
            ClosePhysicalAudioDevice(device);
            return false;
        }
    }

    // Start the audio thread if necessary
    if (!current_audio.impl.ProvidesOwnCallbackThread) {
        char threadname[64];
        SDL_GetAudioThreadName(device, threadname, sizeof (threadname));
        device->thread = SDL_CreateThread(device->recording ? RecordingAudioThread : PlaybackAudioThread, threadname, device);

        if (!device->thread) {
            ClosePhysicalAudioDevice(device);
            return SDL_SetError("Couldn't create audio thread");
        }
    }

    return true;
}

SDL_AudioDeviceID SDL_OpenAudioDevice(SDL_AudioDeviceID devid, const SDL_AudioSpec *spec)
{
    if (!SDL_GetCurrentAudioDriver()) {
        SDL_SetError("Audio subsystem is not initialized");
        return 0;
    }

    bool wants_default = ((devid == SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK) || (devid == SDL_AUDIO_DEVICE_DEFAULT_RECORDING));

    // this will let you use a logical device to make a new logical device on the parent physical device. Could be useful?
    SDL_AudioDevice *device = NULL;
    const bool islogical = (!wants_default && !(devid & (1<<1)));
    if (!islogical) {
        device = ObtainPhysicalAudioDeviceDefaultAllowed(devid);
    } else {
        SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(devid, &device);
        if (logdev) {
            wants_default = logdev->opened_as_default;  // was the original logical device meant to be a default? Make this one, too.
        }
    }

    SDL_AudioDeviceID result = 0;

    if (device) {
        SDL_LogicalAudioDevice *logdev = NULL;
        if (!wants_default && SDL_GetAtomicInt(&device->zombie)) {
            // uhoh, this device is undead, and just waiting to be cleaned up. Refuse explicit opens.
            SDL_SetError("Device was already lost and can't accept new opens");
        } else if ((logdev = (SDL_LogicalAudioDevice *) SDL_calloc(1, sizeof (SDL_LogicalAudioDevice))) == NULL) {
            // SDL_calloc already called SDL_OutOfMemory
        } else if (!OpenPhysicalAudioDevice(device, spec)) {  // if this is the first thing using this physical device, open at the OS level if necessary...
            SDL_free(logdev);
        } else {
            RefPhysicalAudioDevice(device);  // unref'd on successful SDL_CloseAudioDevice
            SDL_SetAtomicInt(&logdev->paused, 0);
            result = logdev->instance_id = AssignAudioDeviceInstanceId(device->recording, /*islogical=*/true);
            logdev->physical_device = device;
            logdev->gain = 1.0f;
            logdev->opened_as_default = wants_default;
            logdev->next = device->logical_devices;
            if (device->logical_devices) {
                device->logical_devices->prev = logdev;
            }
            device->logical_devices = logdev;
            UpdateAudioStreamFormatsPhysical(device);
        }
        ReleaseAudioDevice(device);

        if (result) {
            SDL_LockRWLockForWriting(current_audio.device_hash_lock);
            const bool inserted = SDL_InsertIntoHashTable(current_audio.device_hash, (const void *) (uintptr_t) result, logdev, false);
            SDL_UnlockRWLock(current_audio.device_hash_lock);
            if (!inserted) {
                SDL_CloseAudioDevice(result);
                result = 0;
            }
        }
    }

    return result;
}

static bool SetLogicalAudioDevicePauseState(SDL_AudioDeviceID devid, int value)
{
    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(devid, &device);
    if (logdev) {
        SDL_SetAtomicInt(&logdev->paused, value);
    }
    ReleaseAudioDevice(device);
    return logdev ? true : false;  // ObtainLogicalAudioDevice will have set an error.
}

bool SDL_PauseAudioDevice(SDL_AudioDeviceID devid)
{
    return SetLogicalAudioDevicePauseState(devid, 1);
}

bool SDLCALL SDL_ResumeAudioDevice(SDL_AudioDeviceID devid)
{
    return SetLogicalAudioDevicePauseState(devid, 0);
}

bool SDL_AudioDevicePaused(SDL_AudioDeviceID devid)
{
    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(devid, &device);
    bool result = false;
    if (logdev && SDL_GetAtomicInt(&logdev->paused)) {
        result = true;
    }
    ReleaseAudioDevice(device);
    return result;
}

float SDL_GetAudioDeviceGain(SDL_AudioDeviceID devid)
{
    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(devid, &device);
    const float result = logdev ? logdev->gain : -1.0f;
    ReleaseAudioDevice(device);
    return result;
}

bool SDL_SetAudioDeviceGain(SDL_AudioDeviceID devid, float gain)
{
    if (gain < 0.0f) {
        return SDL_InvalidParamError("gain");
    }

    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(devid, &device);
    bool result = false;
    if (logdev) {
        logdev->gain = gain;
        UpdateAudioStreamFormatsPhysical(device);
        result = true;
    }
    ReleaseAudioDevice(device);
    return result;
}

bool SDL_SetAudioPostmixCallback(SDL_AudioDeviceID devid, SDL_AudioPostmixCallback callback, void *userdata)
{
    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(devid, &device);
    bool result = true;
    if (logdev) {
        if (callback && !device->postmix_buffer) {
            device->postmix_buffer = (float *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
            if (!device->postmix_buffer) {
                result = false;
            }
        }

        if (result) {
            logdev->postmix = callback;
            logdev->postmix_userdata = userdata;
        }

        UpdateAudioStreamFormatsPhysical(device);
    }
    ReleaseAudioDevice(device);
    return result;
}

bool SDL_BindAudioStreams(SDL_AudioDeviceID devid, SDL_AudioStream * const *streams, int num_streams)
{
    const bool islogical = !(devid & (1<<1));
    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = NULL;
    bool result = true;

    if (num_streams == 0) {
        return true;  // nothing to do
    } else if (num_streams < 0) {
        return SDL_InvalidParamError("num_streams");
    } else if (!streams) {
        return SDL_InvalidParamError("streams");
    } else if (!islogical) {
        return SDL_SetError("Audio streams are bound to device ids from SDL_OpenAudioDevice, not raw physical devices");
    }

    logdev = ObtainLogicalAudioDevice(devid, &device);
    if (!logdev) {
        result = false;  // ObtainLogicalAudioDevice set the error string.
    } else if (logdev->simplified) {
        result = SDL_SetError("Cannot change stream bindings on device opened with SDL_OpenAudioDeviceStream");
    } else {

        // !!! FIXME: We'll set the device's side's format below, but maybe we should refuse to bind a stream if the app's side doesn't have a format set yet.
        // !!! FIXME: Actually, why do we allow there to be an invalid format, again?

        // make sure start of list is sane.
        SDL_assert(!logdev->bound_streams || (logdev->bound_streams->prev_binding == NULL));

        // lock all the streams upfront, so we can verify they aren't bound elsewhere and add them all in one block, as this is intended to add everything or nothing.
        for (int i = 0; i < num_streams; i++) {
            SDL_AudioStream *stream = streams[i];
            if (!stream) {
                SDL_SetError("Stream #%d is NULL", i);
                result = false;  // to pacify the static analyzer, that doesn't realize SDL_SetError() always returns false.
            } else {
                SDL_LockMutex(stream->lock);
                SDL_assert((stream->bound_device == NULL) == ((stream->prev_binding == NULL) || (stream->next_binding == NULL)));
                if (stream->bound_device) {
                    result = SDL_SetError("Stream #%d is already bound to a device", i);
                } else if (stream->simplified) {  // You can get here if you closed the device instead of destroying the stream.
                    result = SDL_SetError("Cannot change binding on a stream created with SDL_OpenAudioDeviceStream");
                }
            }

            if (!result) {
                int j;
                for (j = 0; j < i; j++) {
                    SDL_UnlockMutex(streams[j]->lock);
                }
                if (stream) {
                    SDL_UnlockMutex(stream->lock);
                }
                break;
            }
        }
    }

    if (result) {
        // Now that everything is verified, chain everything together.
        for (int i = 0; i < num_streams; i++) {
            SDL_AudioStream *stream = streams[i];
            if (stream) {  // shouldn't be NULL, but just in case...
                stream->bound_device = logdev;
                stream->prev_binding = NULL;
                stream->next_binding = logdev->bound_streams;
                if (logdev->bound_streams) {
                    logdev->bound_streams->prev_binding = stream;
                }
                logdev->bound_streams = stream;
                SDL_UnlockMutex(stream->lock);
            }
        }
    }

    UpdateAudioStreamFormatsPhysical(device);

    ReleaseAudioDevice(device);

    return result;
}

bool SDL_BindAudioStream(SDL_AudioDeviceID devid, SDL_AudioStream *stream)
{
    return SDL_BindAudioStreams(devid, &stream, 1);
}

// !!! FIXME: this and BindAudioStreams are mutex nightmares.  :/
void SDL_UnbindAudioStreams(SDL_AudioStream * const *streams, int num_streams)
{
    if (num_streams <= 0 || !streams) {
        return; // nothing to do
    }

    /* to prevent deadlock when holding both locks, we _must_ lock the device first, and the stream second, as that is the order the audio thread will do it.
       But this means we have an unlikely, pathological case where a stream could change its binding between when we lookup its bound device and when we lock everything,
       so we double-check here. */
    for (int i = 0; i < num_streams; i++) {
        SDL_AudioStream *stream = streams[i];
        if (!stream) {
            continue;  // nothing to do, it's a NULL stream.
        }

        while (true) {
            SDL_LockMutex(stream->lock);   // lock to check this and then release it, in case the device isn't locked yet.
            SDL_LogicalAudioDevice *bounddev = stream->bound_device;
            SDL_UnlockMutex(stream->lock);

            // lock in correct order.
            if (bounddev) {
                SDL_LockMutex(bounddev->physical_device->lock);  // this requires recursive mutexes, since we're likely locking the same device multiple times.
            }
            SDL_LockMutex(stream->lock);

            if (bounddev == stream->bound_device) {
                break;  // the binding didn't change in the small window where it could, so we're good.
            } else {
                SDL_UnlockMutex(stream->lock);  // it changed bindings! Try again.
                if (bounddev) {
                    SDL_UnlockMutex(bounddev->physical_device->lock);
                }
            }
        }
    }

    // everything is locked, start unbinding streams.
    for (int i = 0; i < num_streams; i++) {
        SDL_AudioStream *stream = streams[i];
        // don't allow unbinding from "simplified" devices (opened with SDL_OpenAudioDeviceStream). Just ignore them.
        if (stream && stream->bound_device && !stream->bound_device->simplified) {
            if (stream->bound_device->bound_streams == stream) {
                SDL_assert(!stream->prev_binding);
                stream->bound_device->bound_streams = stream->next_binding;
            }
            if (stream->prev_binding) {
                stream->prev_binding->next_binding = stream->next_binding;
            }
            if (stream->next_binding) {
                stream->next_binding->prev_binding = stream->prev_binding;
            }
            stream->prev_binding = stream->next_binding = NULL;
        }
    }

    // Finalize and unlock everything.
    for (int i = 0; i < num_streams; i++) {
        SDL_AudioStream *stream = streams[i];
        if (stream) {
            SDL_LogicalAudioDevice *logdev = stream->bound_device;
            stream->bound_device = NULL;
            SDL_UnlockMutex(stream->lock);
            if (logdev) {
                UpdateAudioStreamFormatsPhysical(logdev->physical_device);
                SDL_UnlockMutex(logdev->physical_device->lock);
            }
        }
    }
}

void SDL_UnbindAudioStream(SDL_AudioStream *stream)
{
    SDL_UnbindAudioStreams(&stream, 1);
}

SDL_AudioDeviceID SDL_GetAudioStreamDevice(SDL_AudioStream *stream)
{
    SDL_AudioDeviceID result = 0;

    if (!stream) {
        SDL_InvalidParamError("stream");
        return 0;
    }

    SDL_LockMutex(stream->lock);
    if (stream->bound_device) {
        result = stream->bound_device->instance_id;
    } else {
        SDL_SetError("Audio stream not bound to an audio device");
    }
    SDL_UnlockMutex(stream->lock);

    return result;
}

SDL_AudioStream *SDL_OpenAudioDeviceStream(SDL_AudioDeviceID devid, const SDL_AudioSpec *spec, SDL_AudioStreamCallback callback, void *userdata)
{
    SDL_AudioDeviceID logdevid = SDL_OpenAudioDevice(devid, spec);
    if (!logdevid) {
        return NULL;  // error string should already be set.
    }

    bool failed = false;
    SDL_AudioStream *stream = NULL;
    SDL_AudioDevice *device = NULL;
    SDL_LogicalAudioDevice *logdev = ObtainLogicalAudioDevice(logdevid, &device);
    if (!logdev) { // this shouldn't happen, but just in case.
        failed = true;
    } else {
        SDL_SetAtomicInt(&logdev->paused, 1);   // start the device paused, to match SDL2.

        SDL_assert(device != NULL);
        const bool recording = device->recording;

        // if the app didn't request a format _at all_, just make a stream that does no conversion; they can query for it later.
        SDL_AudioSpec tmpspec;
        if (!spec) {
            SDL_copyp(&tmpspec, &device->spec);
            spec = &tmpspec;
        }

        if (recording) {
            stream = SDL_CreateAudioStream(&device->spec, spec);
        } else {
            stream = SDL_CreateAudioStream(spec, &device->spec);
        }

        if (!stream) {
            failed = true;
        } else {
            // don't do all the complicated validation and locking of SDL_BindAudioStream just to set a few fields here.
            logdev->bound_streams = stream;
            logdev->simplified = true;  // forbid further binding changes on this logical device.

            stream->bound_device = logdev;
            stream->simplified = true;  // so we know to close the audio device when this is destroyed.

            UpdateAudioStreamFormatsPhysical(device);

            if (callback) {
                bool rc;
                if (recording) {
                    rc = SDL_SetAudioStreamPutCallback(stream, callback, userdata);
                } else {
                    rc = SDL_SetAudioStreamGetCallback(stream, callback, userdata);
                }
                SDL_assert(rc);  // should only fail if stream==NULL atm.
            }
        }
    }

    ReleaseAudioDevice(device);

    if (failed) {
        SDL_DestroyAudioStream(stream);
        SDL_CloseAudioDevice(logdevid);
        stream = NULL;
    }

    return stream;
}

bool SDL_PauseAudioStreamDevice(SDL_AudioStream *stream)
{
    SDL_AudioDeviceID devid = SDL_GetAudioStreamDevice(stream);
    if (!devid) {
        return false;
    }

    return SDL_PauseAudioDevice(devid);
}

bool SDL_ResumeAudioStreamDevice(SDL_AudioStream *stream)
{
    SDL_AudioDeviceID devid = SDL_GetAudioStreamDevice(stream);
    if (!devid) {
        return false;
    }

    return SDL_ResumeAudioDevice(devid);
}

bool SDL_AudioStreamDevicePaused(SDL_AudioStream *stream)
{
    SDL_AudioDeviceID devid = SDL_GetAudioStreamDevice(stream);
    if (!devid) {
        return false;
    }

    return SDL_AudioDevicePaused(devid);
}

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
#define NATIVE(type) SDL_AUDIO_##type##LE
#define SWAPPED(type) SDL_AUDIO_##type##BE
#else
#define NATIVE(type) SDL_AUDIO_##type##BE
#define SWAPPED(type) SDL_AUDIO_##type##LE
#endif

#define NUM_FORMATS 8
// always favor Float32 in native byte order, since we're probably going to convert to that for processing anyhow.
static const SDL_AudioFormat format_list[NUM_FORMATS][NUM_FORMATS + 1] = {
    { SDL_AUDIO_U8, NATIVE(F32),  SWAPPED(F32), SDL_AUDIO_S8, NATIVE(S16),  SWAPPED(S16), NATIVE(S32),  SWAPPED(S32), SDL_AUDIO_UNKNOWN },
    { SDL_AUDIO_S8, NATIVE(F32),  SWAPPED(F32), SDL_AUDIO_U8, NATIVE(S16),  SWAPPED(S16), NATIVE(S32),  SWAPPED(S32), SDL_AUDIO_UNKNOWN },
    { NATIVE(S16),  NATIVE(F32),  SWAPPED(F32), SWAPPED(S16), NATIVE(S32),  SWAPPED(S32), SDL_AUDIO_U8, SDL_AUDIO_S8, SDL_AUDIO_UNKNOWN },
    { SWAPPED(S16), NATIVE(F32),  SWAPPED(F32), NATIVE(S16),  SWAPPED(S32), NATIVE(S32),  SDL_AUDIO_U8, SDL_AUDIO_S8, SDL_AUDIO_UNKNOWN },
    { NATIVE(S32),  NATIVE(F32),  SWAPPED(F32), SWAPPED(S32), NATIVE(S16),  SWAPPED(S16), SDL_AUDIO_U8, SDL_AUDIO_S8, SDL_AUDIO_UNKNOWN },
    { SWAPPED(S32), NATIVE(F32),  SWAPPED(F32), NATIVE(S32),  SWAPPED(S16), NATIVE(S16),  SDL_AUDIO_U8, SDL_AUDIO_S8, SDL_AUDIO_UNKNOWN },
    { NATIVE(F32),  SWAPPED(F32), NATIVE(S32),  SWAPPED(S32), NATIVE(S16),  SWAPPED(S16), SDL_AUDIO_U8, SDL_AUDIO_S8, SDL_AUDIO_UNKNOWN },
    { SWAPPED(F32), NATIVE(F32),  SWAPPED(S32), NATIVE(S32),  SWAPPED(S16), NATIVE(S16),  SDL_AUDIO_U8, SDL_AUDIO_S8, SDL_AUDIO_UNKNOWN },
};

#undef NATIVE
#undef SWAPPED

const SDL_AudioFormat *SDL_ClosestAudioFormats(SDL_AudioFormat format)
{
    for (int i = 0; i < NUM_FORMATS; i++) {
        if (format_list[i][0] == format) {
            return &format_list[i][0];
        }
    }
    return &format_list[0][NUM_FORMATS]; // not found; return what looks like a list with only a zero in it.
}

const char *SDL_GetAudioFormatName(SDL_AudioFormat format)
{
    switch (format) {
#define CASE(X) \
    case X: return #X;
    CASE(SDL_AUDIO_U8)
    CASE(SDL_AUDIO_S8)
    CASE(SDL_AUDIO_S16LE)
    CASE(SDL_AUDIO_S16BE)
    CASE(SDL_AUDIO_S32LE)
    CASE(SDL_AUDIO_S32BE)
    CASE(SDL_AUDIO_F32LE)
    CASE(SDL_AUDIO_F32BE)
#undef CASE
    default:
        return "SDL_AUDIO_UNKNOWN";
    }
}

int SDL_GetSilenceValueForFormat(SDL_AudioFormat format)
{
    return (format == SDL_AUDIO_U8) ? 0x80 : 0x00;
}

// called internally by backends when the system default device changes.
void SDL_DefaultAudioDeviceChanged(SDL_AudioDevice *new_default_device)
{
    if (!new_default_device) {  // !!! FIXME: what should we do in this case? Maybe all devices are lost, so there _isn't_ a default?
        return;  // uhoh.
    }

    const bool recording = new_default_device->recording;

    // change the official default over right away, so new opens will go to the new device.
    SDL_LockRWLockForWriting(current_audio.device_hash_lock);
    const SDL_AudioDeviceID current_devid = recording ? current_audio.default_recording_device_id : current_audio.default_playback_device_id;
    const bool is_already_default = (new_default_device->instance_id == current_devid);
    if (!is_already_default) {
        if (recording) {
            current_audio.default_recording_device_id = new_default_device->instance_id;
        } else {
            current_audio.default_playback_device_id = new_default_device->instance_id;
        }
    }
    SDL_UnlockRWLock(current_audio.device_hash_lock);

    if (is_already_default) {
        return;  // this is already the default.
    }

    // Queue up events to push to the queue next time it pumps (presumably
    //  in a safer thread).
    // !!! FIXME: this duplicates some code we could probably refactor.
    SDL_PendingAudioDeviceEvent pending;
    pending.next = NULL;
    SDL_PendingAudioDeviceEvent *pending_tail = &pending;

    // Default device gets an extra ref, so it lives until a new default replaces it, even if disconnected.
    RefPhysicalAudioDevice(new_default_device);

    ObtainPhysicalAudioDeviceObj(new_default_device);

    SDL_AudioDevice *current_default_device = ObtainPhysicalAudioDevice(current_devid);

    if (current_default_device) {
        // migrate any logical devices that were opened as a default to the new physical device...

        SDL_assert(current_default_device->recording == recording);

        // See if we have to open the new physical device, and if so, find the best audiospec for it.
        SDL_AudioSpec spec;
        bool needs_migration = false;
        SDL_zero(spec);

        for (SDL_LogicalAudioDevice *logdev = current_default_device->logical_devices; logdev; logdev = logdev->next) {
            if (logdev->opened_as_default) {
                needs_migration = true;
                for (SDL_AudioStream *stream = logdev->bound_streams; stream; stream = stream->next_binding) {
                    const SDL_AudioSpec *streamspec = recording ? &stream->dst_spec : &stream->src_spec;
                    if (SDL_AUDIO_BITSIZE(streamspec->format) > SDL_AUDIO_BITSIZE(spec.format)) {
                        spec.format = streamspec->format;
                    }
                    if (streamspec->channels > spec.channels) {
                        spec.channels = streamspec->channels;
                    }
                    if (streamspec->freq > spec.freq) {
                        spec.freq = streamspec->freq;
                    }
                }
            }
        }

        if (needs_migration) {
            // New default physical device not been opened yet? Open at the OS level...
            if (!OpenPhysicalAudioDevice(new_default_device, &spec)) {
                needs_migration = false;  // uhoh, just leave everything on the old default, nothing to be done.
            }
        }

        if (needs_migration) {
            // we don't currently report channel map changes, so we'll leave them as NULL for now.
            const bool spec_changed = !SDL_AudioSpecsEqual(&current_default_device->spec, &new_default_device->spec, NULL, NULL);
            SDL_LogicalAudioDevice *next = NULL;
            for (SDL_LogicalAudioDevice *logdev = current_default_device->logical_devices; logdev; logdev = next) {
                next = logdev->next;

                if (!logdev->opened_as_default) {
                    continue;  // not opened as a default, leave it on the current physical device.
                }

                // now migrate the logical device. Hold device_hash_lock so ObtainLogicalAudioDevice doesn't get a device in the middle of transition.
                SDL_LockRWLockForWriting(current_audio.device_hash_lock);
                if (logdev->next) {
                    logdev->next->prev = logdev->prev;
                }
                if (logdev->prev) {
                    logdev->prev->next = logdev->next;
                }
                if (current_default_device->logical_devices == logdev) {
                    current_default_device->logical_devices = logdev->next;
                }

                logdev->physical_device = new_default_device;
                logdev->prev = NULL;
                logdev->next = new_default_device->logical_devices;
                new_default_device->logical_devices = logdev;
                SDL_UnlockRWLock(current_audio.device_hash_lock);

                SDL_assert(SDL_GetAtomicInt(&current_default_device->refcount) > 1);  // we should hold at least one extra reference to this device, beyond logical devices, during this phase...
                RefPhysicalAudioDevice(new_default_device);
                UnrefPhysicalAudioDevice(current_default_device);

                SDL_SetAudioPostmixCallback(logdev->instance_id, logdev->postmix, logdev->postmix_userdata);

                SDL_PendingAudioDeviceEvent *p;

                // Queue an event for each logical device we moved.
                if (spec_changed) {
                    p = (SDL_PendingAudioDeviceEvent *)SDL_malloc(sizeof(SDL_PendingAudioDeviceEvent));
                    if (p) { // if this failed, no event for you, but you have deeper problems anyhow.
                        p->type = SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED;
                        p->devid = logdev->instance_id;
                        p->next = NULL;
                        pending_tail->next = p;
                        pending_tail = p;
                    }
                }
            }

            UpdateAudioStreamFormatsPhysical(current_default_device);
            UpdateAudioStreamFormatsPhysical(new_default_device);

            if (!current_default_device->logical_devices) {   // nothing left on the current physical device, close it.
                ClosePhysicalAudioDevice(current_default_device);
            }
        }

        ReleaseAudioDevice(current_default_device);
    }

    ReleaseAudioDevice(new_default_device);

    // Default device gets an extra ref, so it lives until a new default replaces it, even if disconnected.
    if (current_default_device) {  // (despite the name, it's no longer current at this point)
        UnrefPhysicalAudioDevice(current_default_device);
    }

    if (pending.next) {
        SDL_LockRWLockForWriting(current_audio.device_hash_lock);
        SDL_assert(current_audio.pending_events_tail != NULL);
        SDL_assert(current_audio.pending_events_tail->next == NULL);
        current_audio.pending_events_tail->next = pending.next;
        current_audio.pending_events_tail = pending_tail;
        SDL_UnlockRWLock(current_audio.device_hash_lock);
    }
}

bool SDL_AudioDeviceFormatChangedAlreadyLocked(SDL_AudioDevice *device, const SDL_AudioSpec *newspec, int new_sample_frames)
{
    const int orig_work_buffer_size = device->work_buffer_size;

    // we don't currently have any place where channel maps change from under you, but we can check that if necessary later.
    if (SDL_AudioSpecsEqual(&device->spec, newspec, NULL, NULL) && (new_sample_frames == device->sample_frames)) {
        return true;  // we're already in that format.
    }

    SDL_copyp(&device->spec, newspec);
    UpdateAudioStreamFormatsPhysical(device);

    bool kill_device = false;

    device->sample_frames = new_sample_frames;
    SDL_UpdatedAudioDeviceFormat(device);
    if (device->work_buffer && (device->work_buffer_size > orig_work_buffer_size)) {
        SDL_aligned_free(device->work_buffer);
        device->work_buffer = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
        if (!device->work_buffer) {
            kill_device = true;
        }

        if (device->postmix_buffer) {
            SDL_aligned_free(device->postmix_buffer);
            device->postmix_buffer = (float *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
            if (!device->postmix_buffer) {
                kill_device = true;
            }
        }

        SDL_aligned_free(device->mix_buffer);
        device->mix_buffer = NULL;
        if (device->spec.format != SDL_AUDIO_F32) {
            device->mix_buffer = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
            if (!device->mix_buffer) {
                kill_device = true;
            }
        }
    }

    // Post an event for the physical device, and each logical device on this physical device.
    if (!kill_device) {
        // Queue up events to push to the queue next time it pumps (presumably
        //  in a safer thread).
        // !!! FIXME: this duplicates some code we could probably refactor.
        SDL_PendingAudioDeviceEvent pending;
        pending.next = NULL;
        SDL_PendingAudioDeviceEvent *pending_tail = &pending;

        SDL_PendingAudioDeviceEvent *p;

        p = (SDL_PendingAudioDeviceEvent *)SDL_malloc(sizeof(SDL_PendingAudioDeviceEvent));
        if (p) { // if this failed, no event for you, but you have deeper problems anyhow.
            p->type = SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED;
            p->devid = device->instance_id;
            p->next = NULL;
            pending_tail->next = p;
            pending_tail = p;
        }

        for (SDL_LogicalAudioDevice *logdev = device->logical_devices; logdev; logdev = logdev->next) {
            p = (SDL_PendingAudioDeviceEvent *)SDL_malloc(sizeof(SDL_PendingAudioDeviceEvent));
            if (p) { // if this failed, no event for you, but you have deeper problems anyhow.
                p->type = SDL_EVENT_AUDIO_DEVICE_FORMAT_CHANGED;
                p->devid = logdev->instance_id;
                p->next = NULL;
                pending_tail->next = p;
                pending_tail = p;
            }
        }

        if (pending.next) {
            SDL_LockRWLockForWriting(current_audio.device_hash_lock);
            SDL_assert(current_audio.pending_events_tail != NULL);
            SDL_assert(current_audio.pending_events_tail->next == NULL);
            current_audio.pending_events_tail->next = pending.next;
            current_audio.pending_events_tail = pending_tail;
            SDL_UnlockRWLock(current_audio.device_hash_lock);
        }
    }

    if (kill_device) {
        return false;
    }
    return true;
}

bool SDL_AudioDeviceFormatChanged(SDL_AudioDevice *device, const SDL_AudioSpec *newspec, int new_sample_frames)
{
    ObtainPhysicalAudioDeviceObj(device);
    const bool result = SDL_AudioDeviceFormatChangedAlreadyLocked(device, newspec, new_sample_frames);
    ReleaseAudioDevice(device);
    return result;
}

// This is an internal function, so SDL_PumpEvents() can check for pending audio device events.
// ("UpdateSubsystem" is the same naming that the other things that hook into PumpEvents use.)
void SDL_UpdateAudio(void)
{
    SDL_LockRWLockForReading(current_audio.device_hash_lock);
    SDL_PendingAudioDeviceEvent *pending_events = current_audio.pending_events.next;
    SDL_UnlockRWLock(current_audio.device_hash_lock);

    if (!pending_events) {
        return;  // nothing to do, check next time.
    }

    // okay, let's take this whole list of events so we can dump the lock, and new ones can queue up for a later update.
    SDL_LockRWLockForWriting(current_audio.device_hash_lock);
    pending_events = current_audio.pending_events.next;  // in case this changed...
    current_audio.pending_events.next = NULL;
    current_audio.pending_events_tail = &current_audio.pending_events;
    SDL_UnlockRWLock(current_audio.device_hash_lock);

    SDL_PendingAudioDeviceEvent *pending_next = NULL;
    for (SDL_PendingAudioDeviceEvent *i = pending_events; i; i = pending_next) {
        pending_next = i->next;
        if (SDL_EventEnabled(i->type)) {
            SDL_Event event;
            SDL_zero(event);
            event.type = i->type;
            event.adevice.which = (Uint32) i->devid;
            event.adevice.recording = ((i->devid & (1<<0)) == 0);  // bit #0 of devid is set for playback devices and unset for recording.
            SDL_PushEvent(&event);
        }
        SDL_free(i);
    }
}

