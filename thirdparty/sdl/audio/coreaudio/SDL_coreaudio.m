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

#ifdef SDL_AUDIO_DRIVER_COREAUDIO

#include "../SDL_sysaudio.h"
#include "SDL_coreaudio.h"
#include "../../thread/SDL_systhread.h"

#define DEBUG_COREAUDIO 0

#if DEBUG_COREAUDIO
    #define CHECK_RESULT(msg) \
        if (result != noErr) { \
            SDL_Log("COREAUDIO: Got error %d from '%s'!", (int)result, msg); \
            return SDL_SetError("CoreAudio error (%s): %d", msg, (int)result); \
        }
#else
    #define CHECK_RESULT(msg) \
        if (result != noErr) { \
            return SDL_SetError("CoreAudio error (%s): %d", msg, (int)result); \
        }
#endif

#ifdef MACOSX_COREAUDIO
// Apparently AudioDeviceID values might not be unique, so we wrap it in an SDL_malloc()'d pointer
//  to make it so. Use FindCoreAudioDeviceByHandle to deal with this redirection, if you need to
//  map from an AudioDeviceID to a SDL handle.
typedef struct SDLCoreAudioHandle
{
    AudioDeviceID devid;
    bool recording;
} SDLCoreAudioHandle;

static bool TestCoreAudioDeviceHandleCallback(SDL_AudioDevice *device, void *handle)
{
    const SDLCoreAudioHandle *a = (const SDLCoreAudioHandle *) device->handle;
    const SDLCoreAudioHandle *b = (const SDLCoreAudioHandle *) handle;
    return (a->devid == b->devid) && (!!a->recording == !!b->recording);
}

static SDL_AudioDevice *FindCoreAudioDeviceByHandle(const AudioDeviceID devid, const bool recording)
{
    SDLCoreAudioHandle handle = { devid, recording };
    return SDL_FindPhysicalAudioDeviceByCallback(TestCoreAudioDeviceHandleCallback, &handle);
}

static const AudioObjectPropertyAddress devlist_address = {
    kAudioHardwarePropertyDevices,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
};

static const AudioObjectPropertyAddress default_playback_device_address = {
    kAudioHardwarePropertyDefaultOutputDevice,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
};

static const AudioObjectPropertyAddress default_recording_device_address = {
    kAudioHardwarePropertyDefaultInputDevice,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
};

static const AudioObjectPropertyAddress alive_address = {
    kAudioDevicePropertyDeviceIsAlive,
    kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyElementMain
};


static OSStatus DeviceAliveNotification(AudioObjectID devid, UInt32 num_addr, const AudioObjectPropertyAddress *addrs, void *data)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)data;
    SDL_assert(((const SDLCoreAudioHandle *) device->handle)->devid == devid);

    UInt32 alive = 1;
    UInt32 size = sizeof(alive);
    const OSStatus error = AudioObjectGetPropertyData(devid, addrs, 0, NULL, &size, &alive);

    bool dead = false;
    if (error == kAudioHardwareBadDeviceError) {
        dead = true; // device was unplugged.
    } else if ((error == kAudioHardwareNoError) && (!alive)) {
        dead = true; // device died in some other way.
    }

    if (dead) {
        #if DEBUG_COREAUDIO
        SDL_Log("COREAUDIO: device '%s' is lost!", device->name);
        #endif
        SDL_AudioDeviceDisconnected(device);
    }

    return noErr;
}

static void COREAUDIO_FreeDeviceHandle(SDL_AudioDevice *device)
{
    SDLCoreAudioHandle *handle = (SDLCoreAudioHandle *) device->handle;
    AudioObjectRemovePropertyListener(handle->devid, &alive_address, DeviceAliveNotification, device);
    SDL_free(handle);
}

// This only _adds_ new devices. Removal is handled by devices triggering kAudioDevicePropertyDeviceIsAlive property changes.
static void RefreshPhysicalDevices(void)
{
    UInt32 size = 0;
    AudioDeviceID *devs = NULL;
    bool isstack;

    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &devlist_address, 0, NULL, &size) != kAudioHardwareNoError) {
        return;
    } else if ((devs = (AudioDeviceID *) SDL_small_alloc(Uint8, size, &isstack)) == NULL) {
        return;
    } else if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &devlist_address, 0, NULL, &size, devs) != kAudioHardwareNoError) {
        SDL_small_free(devs, isstack);
        return;
    }

    const UInt32 total_devices = (UInt32) (size / sizeof(AudioDeviceID));
    for (UInt32 i = 0; i < total_devices; i++) {
        if (FindCoreAudioDeviceByHandle(devs[i], true) || FindCoreAudioDeviceByHandle(devs[i], false)) {
            devs[i] = 0;  // The system and SDL both agree it's already here, don't check it again.
        }
    }

    // any non-zero items remaining in `devs` are new devices to be added.
    for (int recording = 0; recording < 2; recording++) {
        const AudioObjectPropertyAddress addr = {
            kAudioDevicePropertyStreamConfiguration,
            recording ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput,
            kAudioObjectPropertyElementMain
        };
        const AudioObjectPropertyAddress nameaddr = {
            kAudioObjectPropertyName,
            recording ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput,
            kAudioObjectPropertyElementMain
        };
        const AudioObjectPropertyAddress freqaddr = {
            kAudioDevicePropertyNominalSampleRate,
            recording ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput,
            kAudioObjectPropertyElementMain
        };

        for (UInt32 i = 0; i < total_devices; i++) {
            const AudioDeviceID dev = devs[i];
            if (!dev) {
                continue;  // already added.
            }

            AudioBufferList *buflist = NULL;
            double sampleRate = 0;

            if (AudioObjectGetPropertyDataSize(dev, &addr, 0, NULL, &size) != noErr) {
                continue;
            } else if ((buflist = (AudioBufferList *)SDL_malloc(size)) == NULL) {
                continue;
            }

            OSStatus result = AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, buflist);

            SDL_AudioSpec spec;
            SDL_zero(spec);
            if (result == noErr) {
                for (Uint32 j = 0; j < buflist->mNumberBuffers; j++) {
                    spec.channels += buflist->mBuffers[j].mNumberChannels;
                }
            }

            SDL_free(buflist);

            if (spec.channels == 0) {
                continue;
            }

            size = sizeof(sampleRate);
            if (AudioObjectGetPropertyData(dev, &freqaddr, 0, NULL, &size, &sampleRate) == noErr) {
                spec.freq = (int)sampleRate;
            }

            CFStringRef cfstr = NULL;
            size = sizeof(CFStringRef);
            if (AudioObjectGetPropertyData(dev, &nameaddr, 0, NULL, &size, &cfstr) != kAudioHardwareNoError) {
                continue;
            }

            CFIndex len = CFStringGetMaximumSizeForEncoding(CFStringGetLength(cfstr), kCFStringEncodingUTF8);
            char *name = (char *)SDL_malloc(len + 1);
            bool usable = ((name != NULL) && (CFStringGetCString(cfstr, name, len + 1, kCFStringEncodingUTF8)));

            CFRelease(cfstr);

            if (usable) {
                // Some devices have whitespace at the end...trim it.
                len = (CFIndex) SDL_strlen(name);
                while ((len > 0) && (name[len - 1] == ' ')) {
                    len--;
                }
                usable = (len > 0);
            }

            if (usable) {
                name[len] = '\0';

                #if DEBUG_COREAUDIO
                SDL_Log("COREAUDIO: Found %s device #%d: '%s' (devid %d)", ((recording) ? "recording" : "playback"), (int)i, name, (int)dev);
                #endif
                SDLCoreAudioHandle *newhandle = (SDLCoreAudioHandle *) SDL_calloc(1, sizeof (*newhandle));
                if (newhandle) {
                    newhandle->devid = dev;
                    newhandle->recording = recording ? true : false;
                    SDL_AudioDevice *device = SDL_AddAudioDevice(newhandle->recording, name, &spec, newhandle);
                    if (device) {
                        AudioObjectAddPropertyListener(dev, &alive_address, DeviceAliveNotification, device);
                    } else {
                        SDL_free(newhandle);
                    }
                }
            }
            SDL_free(name); // SDL_AddAudioDevice() would have copied the string.
        }
    }

    SDL_small_free(devs, isstack);
}

// this is called when the system's list of available audio devices changes.
static OSStatus DeviceListChangedNotification(AudioObjectID systemObj, UInt32 num_addr, const AudioObjectPropertyAddress *addrs, void *data)
{
    RefreshPhysicalDevices();
    return noErr;
}

static OSStatus DefaultAudioDeviceChangedNotification(const bool recording, AudioObjectID inObjectID, const AudioObjectPropertyAddress *addr)
{
    AudioDeviceID devid;
    UInt32 size = sizeof(devid);
    if (AudioObjectGetPropertyData(inObjectID, addr, 0, NULL, &size, &devid) == noErr) {
        SDL_DefaultAudioDeviceChanged(FindCoreAudioDeviceByHandle(devid, recording));
    }
    return noErr;
}

static OSStatus DefaultPlaybackDeviceChangedNotification(AudioObjectID inObjectID, UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses, void *inUserData)
{
    #if DEBUG_COREAUDIO
    SDL_Log("COREAUDIO: default playback device changed!");
    #endif
    SDL_assert(inNumberAddresses == 1);
    return DefaultAudioDeviceChangedNotification(false, inObjectID, inAddresses);
}

static OSStatus DefaultRecordingDeviceChangedNotification(AudioObjectID inObjectID, UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses, void *inUserData)
{
    #if DEBUG_COREAUDIO
    SDL_Log("COREAUDIO: default recording device changed!");
    #endif
    SDL_assert(inNumberAddresses == 1);
    return DefaultAudioDeviceChangedNotification(true, inObjectID, inAddresses);
}

static void COREAUDIO_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    RefreshPhysicalDevices();

    AudioObjectAddPropertyListener(kAudioObjectSystemObject, &devlist_address, DeviceListChangedNotification, NULL);

    // Get the Device ID
    UInt32 size;
    AudioDeviceID devid;

    size = sizeof(AudioDeviceID);
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &default_playback_device_address, 0, NULL, &size, &devid) == noErr) {
        SDL_AudioDevice *device = FindCoreAudioDeviceByHandle(devid, false);
        if (device) {
            *default_playback = device;
        }
    }
    AudioObjectAddPropertyListener(kAudioObjectSystemObject, &default_playback_device_address, DefaultPlaybackDeviceChangedNotification, NULL);

    size = sizeof(AudioDeviceID);
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &default_recording_device_address, 0, NULL, &size, &devid) == noErr) {
        SDL_AudioDevice *device = FindCoreAudioDeviceByHandle(devid, true);
        if (device) {
            *default_recording = device;
        }
    }
    AudioObjectAddPropertyListener(kAudioObjectSystemObject, &default_recording_device_address, DefaultRecordingDeviceChangedNotification, NULL);
}

#else  // iOS-specific section follows.

static bool session_active = false;

static bool PauseOneAudioDevice(SDL_AudioDevice *device, void *userdata)
{
    if (device->hidden && device->hidden->audioQueue && !device->hidden->interrupted) {
        AudioQueuePause(device->hidden->audioQueue);
    }
    return false;  // keep enumerating devices until we've paused them all.
}

static void PauseAudioDevices(void)
{
    (void) SDL_FindPhysicalAudioDeviceByCallback(PauseOneAudioDevice, NULL);
}

static bool ResumeOneAudioDevice(SDL_AudioDevice *device, void *userdata)
{
    if (device->hidden && device->hidden->audioQueue && !device->hidden->interrupted) {
        AudioQueueStart(device->hidden->audioQueue, NULL);
    }
    return false;  // keep enumerating devices until we've resumed them all.
}

static void ResumeAudioDevices(void)
{
    (void) SDL_FindPhysicalAudioDeviceByCallback(ResumeOneAudioDevice, NULL);
}

static void InterruptionBegin(SDL_AudioDevice *device)
{
    if (device != NULL && device->hidden->audioQueue != NULL) {
        device->hidden->interrupted = true;
        AudioQueuePause(device->hidden->audioQueue);
    }
}

static void InterruptionEnd(SDL_AudioDevice *device)
{
    if (device != NULL && device->hidden != NULL && device->hidden->audioQueue != NULL && device->hidden->interrupted && AudioQueueStart(device->hidden->audioQueue, NULL) == AVAudioSessionErrorCodeNone) {
        device->hidden->interrupted = false;
    }
}

@interface SDLInterruptionListener : NSObject

@property(nonatomic, assign) SDL_AudioDevice *device;

@end

@implementation SDLInterruptionListener

- (void)audioSessionInterruption:(NSNotification *)note
{
    @synchronized(self) {
        NSNumber *type = note.userInfo[AVAudioSessionInterruptionTypeKey];
        if (type.unsignedIntegerValue == AVAudioSessionInterruptionTypeBegan) {
            InterruptionBegin(self.device);
        } else {
            InterruptionEnd(self.device);
        }
    }
}

- (void)applicationBecameActive:(NSNotification *)note
{
    @synchronized(self) {
        InterruptionEnd(self.device);
    }
}

@end

typedef struct
{
    int playback;
    int recording;
} CountOpenAudioDevicesData;

static bool CountOpenAudioDevices(SDL_AudioDevice *device, void *userdata)
{
    CountOpenAudioDevicesData *data = (CountOpenAudioDevicesData *) userdata;
    if (device->hidden != NULL) {  // assume it's open if hidden != NULL
        if (device->recording) {
            data->recording++;
        } else {
            data->playback++;
        }
    }
    return false;  // keep enumerating until all devices have been checked.
}

static bool UpdateAudioSession(SDL_AudioDevice *device, bool open, bool allow_playandrecord)
{
    @autoreleasepool {
        AVAudioSession *session = [AVAudioSession sharedInstance];
        NSNotificationCenter *center = [NSNotificationCenter defaultCenter];

        NSString *category = AVAudioSessionCategoryPlayback;
        NSString *mode = AVAudioSessionModeDefault;
        NSUInteger options = AVAudioSessionCategoryOptionMixWithOthers;
        NSError *err = nil;
        const char *hint;

        CountOpenAudioDevicesData data;
        SDL_zero(data);
        (void) SDL_FindPhysicalAudioDeviceByCallback(CountOpenAudioDevices, &data);

        hint = SDL_GetHint(SDL_HINT_AUDIO_CATEGORY);
        if (hint) {
            if (SDL_strcasecmp(hint, "AVAudioSessionCategoryAmbient") == 0) {
                category = AVAudioSessionCategoryAmbient;
            } else if (SDL_strcasecmp(hint, "AVAudioSessionCategorySoloAmbient") == 0) {
                category = AVAudioSessionCategorySoloAmbient;
                options &= ~AVAudioSessionCategoryOptionMixWithOthers;
            } else if (SDL_strcasecmp(hint, "AVAudioSessionCategoryPlayback") == 0 ||
                       SDL_strcasecmp(hint, "playback") == 0) {
                category = AVAudioSessionCategoryPlayback;
                options &= ~AVAudioSessionCategoryOptionMixWithOthers;
            } else if (SDL_strcasecmp(hint, "AVAudioSessionCategoryPlayAndRecord") == 0 ||
                       SDL_strcasecmp(hint, "playandrecord") == 0) {
                if (allow_playandrecord) {
                    category = AVAudioSessionCategoryPlayAndRecord;
                }
            }
        } else if (data.playback && data.recording) {
            if (allow_playandrecord) {
                category = AVAudioSessionCategoryPlayAndRecord;
            } else {
                // We already failed play and record with AVAudioSessionErrorCodeResourceNotAvailable
                return false;
            }
        } else if (data.recording) {
            category = AVAudioSessionCategoryRecord;
        }

        #ifndef SDL_PLATFORM_TVOS
        if (category == AVAudioSessionCategoryPlayAndRecord) {
            options |= AVAudioSessionCategoryOptionDefaultToSpeaker;
        }
        #endif
        if (category == AVAudioSessionCategoryRecord ||
            category == AVAudioSessionCategoryPlayAndRecord) {
            /* AVAudioSessionCategoryOptionAllowBluetooth isn't available in the SDK for
               Apple TV but is still needed in order to output to Bluetooth devices.
             */
            options |= 0x4; // AVAudioSessionCategoryOptionAllowBluetooth;
        }
        if (category == AVAudioSessionCategoryPlayAndRecord) {
            options |= AVAudioSessionCategoryOptionAllowBluetoothA2DP |
                       AVAudioSessionCategoryOptionAllowAirPlay;
        }
        if (category == AVAudioSessionCategoryPlayback ||
            category == AVAudioSessionCategoryPlayAndRecord) {
            options |= AVAudioSessionCategoryOptionDuckOthers;
        }

        if (![session.category isEqualToString:category] || session.categoryOptions != options) {
            // Stop the current session so we don't interrupt other application audio
            PauseAudioDevices();
            [session setActive:NO error:nil];
            session_active = false;

            if (![session setCategory:category mode:mode options:options error:&err]) {
                NSString *desc = err.description;
                SDL_SetError("Could not set Audio Session category: %s", desc.UTF8String);
                return false;
            }
        }

        if ((data.playback || data.recording) && !session_active) {
            if (![session setActive:YES error:&err]) {
                if ([err code] == AVAudioSessionErrorCodeResourceNotAvailable &&
                    category == AVAudioSessionCategoryPlayAndRecord) {
                    if (UpdateAudioSession(device, open, false)) {
                        return true;
                    } else {
                        return SDL_SetError("Could not activate Audio Session: Resource not available");
                    }
                }

                NSString *desc = err.description;
                return SDL_SetError("Could not activate Audio Session: %s", desc.UTF8String);
            }
            session_active = true;
            ResumeAudioDevices();
        } else if (!data.playback && !data.recording && session_active) {
            PauseAudioDevices();
            [session setActive:NO error:nil];
            session_active = false;
        }

        if (open) {
            SDLInterruptionListener *listener = [SDLInterruptionListener new];
            listener.device = device;

            [center addObserver:listener
                       selector:@selector(audioSessionInterruption:)
                           name:AVAudioSessionInterruptionNotification
                         object:session];

            /* An interruption end notification is not guaranteed to be sent if
             we were previously interrupted... resuming if needed when the app
             becomes active seems to be the way to go. */
            // Note: object: below needs to be nil, as otherwise it filters by the object, and session doesn't send foreground / active notifications.
            [center addObserver:listener
                       selector:@selector(applicationBecameActive:)
                           name:UIApplicationDidBecomeActiveNotification
                         object:nil];

            [center addObserver:listener
                       selector:@selector(applicationBecameActive:)
                           name:UIApplicationWillEnterForegroundNotification
                         object:nil];

            device->hidden->interruption_listener = CFBridgingRetain(listener);
        } else {
            SDLInterruptionListener *listener = nil;
            listener = (SDLInterruptionListener *)CFBridgingRelease(device->hidden->interruption_listener);
            [center removeObserver:listener];
            @synchronized(listener) {
                listener.device = NULL;
            }
        }
    }

    return true;
}
#endif


static bool COREAUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buffer_size)
{
    AudioQueueBufferRef current_buffer = device->hidden->current_buffer;
    SDL_assert(current_buffer != NULL);  // should have been called from PlaybackBufferReadyCallback
    SDL_assert(buffer == (Uint8 *) current_buffer->mAudioData);
    current_buffer->mAudioDataByteSize = current_buffer->mAudioDataBytesCapacity;
    device->hidden->current_buffer = NULL;
    AudioQueueEnqueueBuffer(device->hidden->audioQueue, current_buffer, 0, NULL);
    return true;
}

static Uint8 *COREAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    AudioQueueBufferRef current_buffer = device->hidden->current_buffer;
    SDL_assert(current_buffer != NULL);  // should have been called from PlaybackBufferReadyCallback
    SDL_assert(current_buffer->mAudioData != NULL);
    *buffer_size = (int) current_buffer->mAudioDataBytesCapacity;
    return (Uint8 *) current_buffer->mAudioData;
}

static void PlaybackBufferReadyCallback(void *inUserData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)inUserData;
    SDL_assert(inBuffer != NULL);  // ...right?
    SDL_assert(device->hidden->current_buffer == NULL);  // shouldn't have anything pending
    device->hidden->current_buffer = inBuffer;
    const bool okay = SDL_PlaybackAudioThreadIterate(device);
    SDL_assert((device->hidden->current_buffer == NULL) || !okay);  // PlayDevice should have enqueued and cleaned it out, unless we failed or shutdown.

    // buffer is unexpectedly here? We're probably dying, but try to requeue this buffer with silence.
    if (device->hidden->current_buffer) {
        AudioQueueBufferRef current_buffer = device->hidden->current_buffer;
        device->hidden->current_buffer = NULL;
        SDL_memset(current_buffer->mAudioData, device->silence_value, (size_t) current_buffer->mAudioDataBytesCapacity);
        AudioQueueEnqueueBuffer(device->hidden->audioQueue, current_buffer, 0, NULL);
    }
}

static int COREAUDIO_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    AudioQueueBufferRef current_buffer = device->hidden->current_buffer;
    SDL_assert(current_buffer != NULL);  // should have been called from RecordingBufferReadyCallback
    SDL_assert(current_buffer->mAudioData != NULL);
    SDL_assert(buflen >= (int) current_buffer->mAudioDataByteSize);  // `cpy` makes sure this won't overflow a buffer, but we _will_ drop samples if this assertion fails!
    const int cpy = SDL_min(buflen, (int) current_buffer->mAudioDataByteSize);
    SDL_memcpy(buffer, current_buffer->mAudioData, cpy);
    device->hidden->current_buffer = NULL;
    AudioQueueEnqueueBuffer(device->hidden->audioQueue, current_buffer, 0, NULL);  // requeue for capturing more data later.
    return cpy;
}

static void COREAUDIO_FlushRecording(SDL_AudioDevice *device)
{
    AudioQueueBufferRef current_buffer = device->hidden->current_buffer;
    if (current_buffer != NULL) {  // also gets called at shutdown, when no buffer is available.
        // just requeue the current buffer without reading from it, so it can be refilled with new data later.
        device->hidden->current_buffer = NULL;
        AudioQueueEnqueueBuffer(device->hidden->audioQueue, current_buffer, 0, NULL);
    }
}

static void RecordingBufferReadyCallback(void *inUserData, AudioQueueRef inAQ, AudioQueueBufferRef inBuffer,
                          const AudioTimeStamp *inStartTime, UInt32 inNumberPacketDescriptions,
                          const AudioStreamPacketDescription *inPacketDescs)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)inUserData;
    SDL_assert(inAQ == device->hidden->audioQueue);
    SDL_assert(inBuffer != NULL);  // ...right?
    SDL_assert(device->hidden->current_buffer == NULL);  // shouldn't have anything pending
    device->hidden->current_buffer = inBuffer;
    SDL_RecordingAudioThreadIterate(device);

    // buffer is unexpectedly here? We're probably dying, but try to requeue this buffer anyhow.
    if (device->hidden->current_buffer != NULL) {
        SDL_assert(SDL_GetAtomicInt(&device->shutdown) != 0);
        COREAUDIO_FlushRecording(device);  // just flush it manually, which will requeue it.
    }
}

static void COREAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    if (!device->hidden) {
        return;
    }

    // dispose of the audio queue before waiting on the thread, or it might stall for a long time!
    if (device->hidden->audioQueue) {
        AudioQueueFlush(device->hidden->audioQueue);
        AudioQueueStop(device->hidden->audioQueue, 0);
        AudioQueueDispose(device->hidden->audioQueue, 0);
    }

    if (device->hidden->thread) {
        SDL_assert(SDL_GetAtomicInt(&device->shutdown) != 0); // should have been set by SDL_audio.c
        SDL_WaitThread(device->hidden->thread, NULL);
    }

    #ifndef MACOSX_COREAUDIO
    UpdateAudioSession(device, false, true);
    #endif

    if (device->hidden->ready_semaphore) {
        SDL_DestroySemaphore(device->hidden->ready_semaphore);
    }

    // AudioQueueDispose() frees the actual buffer objects.
    SDL_free(device->hidden->audioBuffer);
    SDL_free(device->hidden->thread_error);
    SDL_free(device->hidden);
}

#ifdef MACOSX_COREAUDIO
static bool PrepareDevice(SDL_AudioDevice *device)
{
    SDL_assert(device->handle != NULL);  // this meant "system default" in SDL2, but doesn't anymore

    const SDLCoreAudioHandle *handle = (const SDLCoreAudioHandle *) device->handle;
    const AudioDeviceID devid = handle->devid;
    OSStatus result = noErr;
    UInt32 size = 0;

    AudioObjectPropertyAddress addr = {
        0,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    UInt32 alive = 0;
    size = sizeof(alive);
    addr.mSelector = kAudioDevicePropertyDeviceIsAlive;
    addr.mScope = device->recording ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput;
    result = AudioObjectGetPropertyData(devid, &addr, 0, NULL, &size, &alive);
    CHECK_RESULT("AudioDeviceGetProperty (kAudioDevicePropertyDeviceIsAlive)");
    if (!alive) {
        return SDL_SetError("CoreAudio: requested device exists, but isn't alive.");
    }

    // some devices don't support this property, so errors are fine here.
    pid_t pid = 0;
    size = sizeof(pid);
    addr.mSelector = kAudioDevicePropertyHogMode;
    result = AudioObjectGetPropertyData(devid, &addr, 0, NULL, &size, &pid);
    if ((result == noErr) && (pid != -1)) {
        return SDL_SetError("CoreAudio: requested device is being hogged.");
    }

    device->hidden->deviceID = devid;

    return true;
}

static bool AssignDeviceToAudioQueue(SDL_AudioDevice *device)
{
    const AudioObjectPropertyAddress prop = {
        kAudioDevicePropertyDeviceUID,
        device->recording ? kAudioDevicePropertyScopeInput : kAudioDevicePropertyScopeOutput,
        kAudioObjectPropertyElementMain
    };

    OSStatus result;
    CFStringRef devuid;
    UInt32 devuidsize = sizeof(devuid);
    result = AudioObjectGetPropertyData(device->hidden->deviceID, &prop, 0, NULL, &devuidsize, &devuid);
    CHECK_RESULT("AudioObjectGetPropertyData (kAudioDevicePropertyDeviceUID)");
    result = AudioQueueSetProperty(device->hidden->audioQueue, kAudioQueueProperty_CurrentDevice, &devuid, devuidsize);
    CFRelease(devuid);  // Release devuid; we're done with it and AudioQueueSetProperty should have retained if it wants to keep it.
    CHECK_RESULT("AudioQueueSetProperty (kAudioQueueProperty_CurrentDevice)");
    return true;
}
#endif

static bool PrepareAudioQueue(SDL_AudioDevice *device)
{
    const AudioStreamBasicDescription *strdesc = &device->hidden->strdesc;
    const bool recording = device->recording;
    OSStatus result;

    SDL_assert(CFRunLoopGetCurrent() != NULL);

    if (recording) {
        result = AudioQueueNewInput(strdesc, RecordingBufferReadyCallback, device, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode, 0, &device->hidden->audioQueue);
        CHECK_RESULT("AudioQueueNewInput");
    } else {
        result = AudioQueueNewOutput(strdesc, PlaybackBufferReadyCallback, device, CFRunLoopGetCurrent(), kCFRunLoopDefaultMode, 0, &device->hidden->audioQueue);
        CHECK_RESULT("AudioQueueNewOutput");
    }

    #ifdef MACOSX_COREAUDIO
    if (!AssignDeviceToAudioQueue(device)) {
        return false;
    }
    #endif

    SDL_UpdatedAudioDeviceFormat(device);  // make sure this is correct.

    // Set the channel layout for the audio queue
    AudioChannelLayout layout;
    SDL_zero(layout);
    switch (device->spec.channels) {
    case 1:
        // a standard mono stream
        layout.mChannelLayoutTag = kAudioChannelLayoutTag_Mono;
        break;
    case 2:
        // a standard stereo stream (L R) - implied playback
        layout.mChannelLayoutTag = kAudioChannelLayoutTag_Stereo;
        break;
    case 3:
        // L R LFE
        layout.mChannelLayoutTag = kAudioChannelLayoutTag_DVD_4;
        break;
    case 4:
        // front left, front right, back left, back right
        layout.mChannelLayoutTag = kAudioChannelLayoutTag_Quadraphonic;
        break;
    case 5:
        // L R LFE Ls Rs
        layout.mChannelLayoutTag = kAudioChannelLayoutTag_DVD_6;
        break;
    case 6:
        //  L R C LFE Ls Rs
        layout.mChannelLayoutTag = kAudioChannelLayoutTag_DVD_12;
        break;
    case 7:
        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            // L R C LFE Cs Ls Rs
            layout.mChannelLayoutTag = kAudioChannelLayoutTag_WAVE_6_1;
        } else {
            // L R C LFE Ls Rs Cs
            layout.mChannelLayoutTag = kAudioChannelLayoutTag_MPEG_6_1_A;

            // Convert from SDL channel layout to kAudioChannelLayoutTag_MPEG_6_1_A
            static const int swizzle_map[7] = {
                0, 1, 2, 3, 6, 4, 5
            };
            device->chmap = SDL_ChannelMapDup(swizzle_map, SDL_arraysize(swizzle_map));
            if (!device->chmap) {
                return false;
            }
        }
        break;
    case 8:
        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            // L R C LFE Rls Rrs Ls Rs
            layout.mChannelLayoutTag = kAudioChannelLayoutTag_WAVE_7_1;
        } else {
            // L R C LFE Ls Rs Rls Rrs
            layout.mChannelLayoutTag = kAudioChannelLayoutTag_MPEG_7_1_C;

            // Convert from SDL channel layout to kAudioChannelLayoutTag_MPEG_7_1_C
            static const int swizzle_map[8] = {
                0, 1, 2, 3, 6, 7, 4, 5
            };
            device->chmap = SDL_ChannelMapDup(swizzle_map, SDL_arraysize(swizzle_map));
            if (!device->chmap) {
                return false;
            }
        }
        break;
    default:
        return SDL_SetError("Unsupported audio channels");
    }
    if (layout.mChannelLayoutTag != 0) {
        result = AudioQueueSetProperty(device->hidden->audioQueue, kAudioQueueProperty_ChannelLayout, &layout, sizeof(layout));
        CHECK_RESULT("AudioQueueSetProperty(kAudioQueueProperty_ChannelLayout)");
    }

    // Make sure we can feed the device a minimum amount of time
    double MINIMUM_AUDIO_BUFFER_TIME_MS = 15.0;
    #ifdef SDL_PLATFORM_IOS
    if (SDL_floor(NSFoundationVersionNumber) <= NSFoundationVersionNumber_iOS_7_1) {
        // Older iOS hardware, use 40 ms as a minimum time
        MINIMUM_AUDIO_BUFFER_TIME_MS = 40.0;
    }
    #endif

    // we use THREE audio buffers by default, unlike most things that would
    // choose two alternating buffers, because it helps with issues on
    // Bluetooth headsets when recording and playing at the same time.
    // See conversation in #8192 for details.
    int numAudioBuffers = 3;
    const double msecs = (device->sample_frames / ((double)device->spec.freq)) * 1000.0;
    if (msecs < MINIMUM_AUDIO_BUFFER_TIME_MS) { // use more buffers if we have a VERY small sample set.
        numAudioBuffers = ((int)SDL_ceil(MINIMUM_AUDIO_BUFFER_TIME_MS / msecs) * 2);
    }

    device->hidden->numAudioBuffers = numAudioBuffers;
    device->hidden->audioBuffer = SDL_calloc(numAudioBuffers, sizeof(AudioQueueBufferRef));
    if (device->hidden->audioBuffer == NULL) {
        return false;
    }

    #if DEBUG_COREAUDIO
    SDL_Log("COREAUDIO: numAudioBuffers == %d", numAudioBuffers);
    #endif

    for (int i = 0; i < numAudioBuffers; i++) {
        result = AudioQueueAllocateBuffer(device->hidden->audioQueue, device->buffer_size, &device->hidden->audioBuffer[i]);
        CHECK_RESULT("AudioQueueAllocateBuffer");
        SDL_memset(device->hidden->audioBuffer[i]->mAudioData, device->silence_value, device->hidden->audioBuffer[i]->mAudioDataBytesCapacity);
        device->hidden->audioBuffer[i]->mAudioDataByteSize = device->hidden->audioBuffer[i]->mAudioDataBytesCapacity;
        // !!! FIXME: should we use AudioQueueEnqueueBufferWithParameters and specify all frames be "trimmed" so these are immediately ready to refill with SDL callback data?
        result = AudioQueueEnqueueBuffer(device->hidden->audioQueue, device->hidden->audioBuffer[i], 0, NULL);
        CHECK_RESULT("AudioQueueEnqueueBuffer");
    }

    result = AudioQueueStart(device->hidden->audioQueue, NULL);
    CHECK_RESULT("AudioQueueStart");

    return true;  // We're running!
}

static int AudioQueueThreadEntry(void *arg)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)arg;

    if (device->recording) {
        SDL_RecordingAudioThreadSetup(device);
    } else {
        SDL_PlaybackAudioThreadSetup(device);
    }

    if (!PrepareAudioQueue(device)) {
        device->hidden->thread_error = SDL_strdup(SDL_GetError());
        SDL_SignalSemaphore(device->hidden->ready_semaphore);
        return 0;
    }

    // init was successful, alert parent thread and start running...
    SDL_SignalSemaphore(device->hidden->ready_semaphore);

    // This would be WaitDevice/WaitRecordingDevice in the normal SDL audio thread, but we get *BufferReadyCallback calls here to know when to iterate.
    while (!SDL_GetAtomicInt(&device->shutdown)) {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.10, 1);
    }

    if (device->recording) {
        SDL_RecordingAudioThreadShutdown(device);
    } else {
        // Drain off any pending playback.
        const CFTimeInterval secs = (((CFTimeInterval)device->sample_frames) / ((CFTimeInterval)device->spec.freq)) * 2.0;
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, secs, 0);
        SDL_PlaybackAudioThreadShutdown(device);
    }

    return 0;
}

static bool COREAUDIO_OpenDevice(SDL_AudioDevice *device)
{
    // Initialize all variables that we clean on shutdown
    device->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*device->hidden));
    if (device->hidden == NULL) {
        return false;
    }

    #ifndef MACOSX_COREAUDIO
    if (!UpdateAudioSession(device, true, true)) {
        return false;
    }

    // Stop CoreAudio from doing expensive audio rate conversion
    @autoreleasepool {
        AVAudioSession *session = [AVAudioSession sharedInstance];
        [session setPreferredSampleRate:device->spec.freq error:nil];
        device->spec.freq = (int)session.sampleRate;
        #ifdef SDL_PLATFORM_TVOS
        if (device->recording) {
            [session setPreferredInputNumberOfChannels:device->spec.channels error:nil];
            device->spec.channels = (int)session.preferredInputNumberOfChannels;
        } else {
            [session setPreferredOutputNumberOfChannels:device->spec.channels error:nil];
            device->spec.channels = (int)session.preferredOutputNumberOfChannels;
        }
        #else
        // Calling setPreferredOutputNumberOfChannels seems to break audio output on iOS
        #endif // SDL_PLATFORM_TVOS
    }
    #endif

    // Setup a AudioStreamBasicDescription with the requested format
    AudioStreamBasicDescription *strdesc = &device->hidden->strdesc;
    strdesc->mFormatID = kAudioFormatLinearPCM;
    strdesc->mFormatFlags = kLinearPCMFormatFlagIsPacked;
    strdesc->mChannelsPerFrame = device->spec.channels;
    strdesc->mSampleRate = device->spec.freq;
    strdesc->mFramesPerPacket = 1;

    const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(device->spec.format);
    SDL_AudioFormat test_format;
    while ((test_format = *(closefmts++)) != 0) {
        // CoreAudio handles most of SDL's formats natively.
        switch (test_format) {
        case SDL_AUDIO_U8:
        case SDL_AUDIO_S8:
        case SDL_AUDIO_S16LE:
        case SDL_AUDIO_S16BE:
        case SDL_AUDIO_S32LE:
        case SDL_AUDIO_S32BE:
        case SDL_AUDIO_F32LE:
        case SDL_AUDIO_F32BE:
            break;

        default:
            continue;
        }
        break;
    }

    if (!test_format) { // shouldn't happen, but just in case...
        return SDL_SetError("%s: Unsupported audio format", "coreaudio");
    }
    device->spec.format = test_format;
    strdesc->mBitsPerChannel = SDL_AUDIO_BITSIZE(test_format);
    if (SDL_AUDIO_ISBIGENDIAN(test_format)) {
        strdesc->mFormatFlags |= kLinearPCMFormatFlagIsBigEndian;
    }

    if (SDL_AUDIO_ISFLOAT(test_format)) {
        strdesc->mFormatFlags |= kLinearPCMFormatFlagIsFloat;
    } else if (SDL_AUDIO_ISSIGNED(test_format)) {
        strdesc->mFormatFlags |= kLinearPCMFormatFlagIsSignedInteger;
    }

    strdesc->mBytesPerFrame = strdesc->mChannelsPerFrame * strdesc->mBitsPerChannel / 8;
    strdesc->mBytesPerPacket = strdesc->mBytesPerFrame * strdesc->mFramesPerPacket;

#ifdef MACOSX_COREAUDIO
    if (!PrepareDevice(device)) {
        return false;
    }
#endif

    // This has to init in a new thread so it can get its own CFRunLoop. :/
    device->hidden->ready_semaphore = SDL_CreateSemaphore(0);
    if (!device->hidden->ready_semaphore) {
        return false; // oh well.
    }

    char threadname[64];
    SDL_GetAudioThreadName(device, threadname, sizeof(threadname));
    device->hidden->thread = SDL_CreateThread(AudioQueueThreadEntry, threadname, device);
    if (!device->hidden->thread) {
        return false;
    }

    SDL_WaitSemaphore(device->hidden->ready_semaphore);
    SDL_DestroySemaphore(device->hidden->ready_semaphore);
    device->hidden->ready_semaphore = NULL;

    if ((device->hidden->thread != NULL) && (device->hidden->thread_error != NULL)) {
        SDL_WaitThread(device->hidden->thread, NULL);
        device->hidden->thread = NULL;
        return SDL_SetError("%s", device->hidden->thread_error);
    }

    return (device->hidden->thread != NULL);
}

static void COREAUDIO_DeinitializeStart(void)
{
#ifdef MACOSX_COREAUDIO
    AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &devlist_address, DeviceListChangedNotification, NULL);
    AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &default_playback_device_address, DefaultPlaybackDeviceChangedNotification, NULL);
    AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &default_recording_device_address, DefaultRecordingDeviceChangedNotification, NULL);
#endif
}

static bool COREAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    impl->OpenDevice = COREAUDIO_OpenDevice;
    impl->PlayDevice = COREAUDIO_PlayDevice;
    impl->GetDeviceBuf = COREAUDIO_GetDeviceBuf;
    impl->RecordDevice = COREAUDIO_RecordDevice;
    impl->FlushRecording = COREAUDIO_FlushRecording;
    impl->CloseDevice = COREAUDIO_CloseDevice;
    impl->DeinitializeStart = COREAUDIO_DeinitializeStart;

#ifdef MACOSX_COREAUDIO
    impl->DetectDevices = COREAUDIO_DetectDevices;
    impl->FreeDeviceHandle = COREAUDIO_FreeDeviceHandle;
#else
    impl->OnlyHasDefaultPlaybackDevice = true;
    impl->OnlyHasDefaultRecordingDevice = true;
#endif

    impl->ProvidesOwnCallbackThread = true;
    impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap COREAUDIO_bootstrap = {
    "coreaudio", "CoreAudio", COREAUDIO_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_COREAUDIO
