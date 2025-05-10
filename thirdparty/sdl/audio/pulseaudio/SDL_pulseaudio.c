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

#ifdef SDL_AUDIO_DRIVER_PULSEAUDIO

// Allow access to a raw mixing buffer

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
#include <unistd.h>
#include <sys/types.h>

#include "../SDL_sysaudio.h"
#include "SDL_pulseaudio.h"
#include "../../thread/SDL_systhread.h"

#if (PA_PROTOCOL_VERSION < 28)
typedef void (*pa_operation_notify_cb_t) (pa_operation *o, void *userdata);
#endif

typedef struct PulseDeviceHandle
{
    char *device_path;
    uint32_t device_index;
} PulseDeviceHandle;

// should we include monitors in the device list? Set at SDL_Init time
static bool include_monitors = false;

static pa_threaded_mainloop *pulseaudio_threaded_mainloop = NULL;
static pa_context *pulseaudio_context = NULL;
static SDL_Thread *pulseaudio_hotplug_thread = NULL;
static SDL_AtomicInt pulseaudio_hotplug_thread_active;

// These are the OS identifiers (i.e. ALSA strings)...these are allocated in a callback
// when the default changes, and noticed by the hotplug thread when it alerts SDL
// to the change.
static char *default_sink_path = NULL;
static char *default_source_path = NULL;
static bool default_sink_changed = false;
static bool default_source_changed = false;


static const char *(*PULSEAUDIO_pa_get_library_version)(void);
static pa_channel_map *(*PULSEAUDIO_pa_channel_map_init_auto)(
    pa_channel_map *, unsigned, pa_channel_map_def_t);
static const char *(*PULSEAUDIO_pa_strerror)(int);
static pa_proplist *(*PULSEAUDIO_pa_proplist_new)(void);
static void (*PULSEAUDIO_pa_proplist_free)(pa_proplist *);
static int (*PULSEAUDIO_pa_proplist_sets)(pa_proplist *, const char *, const char *);

static pa_threaded_mainloop *(*PULSEAUDIO_pa_threaded_mainloop_new)(void);
static void (*PULSEAUDIO_pa_threaded_mainloop_set_name)(pa_threaded_mainloop *, const char *);
static pa_mainloop_api *(*PULSEAUDIO_pa_threaded_mainloop_get_api)(pa_threaded_mainloop *);
static int (*PULSEAUDIO_pa_threaded_mainloop_start)(pa_threaded_mainloop *);
static void (*PULSEAUDIO_pa_threaded_mainloop_stop)(pa_threaded_mainloop *);
static void (*PULSEAUDIO_pa_threaded_mainloop_lock)(pa_threaded_mainloop *);
static void (*PULSEAUDIO_pa_threaded_mainloop_unlock)(pa_threaded_mainloop *);
static void (*PULSEAUDIO_pa_threaded_mainloop_wait)(pa_threaded_mainloop *);
static void (*PULSEAUDIO_pa_threaded_mainloop_signal)(pa_threaded_mainloop *, int);
static void (*PULSEAUDIO_pa_threaded_mainloop_free)(pa_threaded_mainloop *);

static pa_operation_state_t (*PULSEAUDIO_pa_operation_get_state)(
    const pa_operation *);
static void (*PULSEAUDIO_pa_operation_set_state_callback)(pa_operation *, pa_operation_notify_cb_t, void *);
static void (*PULSEAUDIO_pa_operation_cancel)(pa_operation *);
static void (*PULSEAUDIO_pa_operation_unref)(pa_operation *);

static pa_context *(*PULSEAUDIO_pa_context_new_with_proplist)(pa_mainloop_api *,
                                                const char *,
                                                const pa_proplist *);
static void (*PULSEAUDIO_pa_context_set_state_callback)(pa_context *, pa_context_notify_cb_t, void *);
static int (*PULSEAUDIO_pa_context_connect)(pa_context *, const char *,
                                            pa_context_flags_t, const pa_spawn_api *);
static pa_operation *(*PULSEAUDIO_pa_context_get_sink_info_list)(pa_context *, pa_sink_info_cb_t, void *);
static pa_operation *(*PULSEAUDIO_pa_context_get_source_info_list)(pa_context *, pa_source_info_cb_t, void *);
static pa_operation *(*PULSEAUDIO_pa_context_get_sink_info_by_index)(pa_context *, uint32_t, pa_sink_info_cb_t, void *);
static pa_operation *(*PULSEAUDIO_pa_context_get_source_info_by_index)(pa_context *, uint32_t, pa_source_info_cb_t, void *);
static pa_context_state_t (*PULSEAUDIO_pa_context_get_state)(const pa_context *);
static pa_operation *(*PULSEAUDIO_pa_context_subscribe)(pa_context *, pa_subscription_mask_t, pa_context_success_cb_t, void *);
static void (*PULSEAUDIO_pa_context_set_subscribe_callback)(pa_context *, pa_context_subscribe_cb_t, void *);
static void (*PULSEAUDIO_pa_context_disconnect)(pa_context *);
static void (*PULSEAUDIO_pa_context_unref)(pa_context *);

static pa_stream *(*PULSEAUDIO_pa_stream_new)(pa_context *, const char *,
                                              const pa_sample_spec *, const pa_channel_map *);
static void (*PULSEAUDIO_pa_stream_set_state_callback)(pa_stream *, pa_stream_notify_cb_t, void *);
static int (*PULSEAUDIO_pa_stream_connect_playback)(pa_stream *, const char *,
                                                    const pa_buffer_attr *, pa_stream_flags_t, const pa_cvolume *, pa_stream *);
static int (*PULSEAUDIO_pa_stream_connect_record)(pa_stream *, const char *,
                                                  const pa_buffer_attr *, pa_stream_flags_t);
static const pa_buffer_attr *(*PULSEAUDIO_pa_stream_get_buffer_attr)(pa_stream *);
static pa_stream_state_t (*PULSEAUDIO_pa_stream_get_state)(const pa_stream *);
static size_t (*PULSEAUDIO_pa_stream_writable_size)(const pa_stream *);
static size_t (*PULSEAUDIO_pa_stream_readable_size)(const pa_stream *);
static int (*PULSEAUDIO_pa_stream_write)(pa_stream *, const void *, size_t,
                                         pa_free_cb_t, int64_t, pa_seek_mode_t);
static int (*PULSEAUDIO_pa_stream_begin_write)(pa_stream *, void **, size_t *);
static pa_operation *(*PULSEAUDIO_pa_stream_drain)(pa_stream *,
                                                   pa_stream_success_cb_t, void *);
static int (*PULSEAUDIO_pa_stream_peek)(pa_stream *, const void **, size_t *);
static int (*PULSEAUDIO_pa_stream_drop)(pa_stream *);
static pa_operation *(*PULSEAUDIO_pa_stream_flush)(pa_stream *,
                                                   pa_stream_success_cb_t, void *);
static int (*PULSEAUDIO_pa_stream_disconnect)(pa_stream *);
static void (*PULSEAUDIO_pa_stream_unref)(pa_stream *);
static void (*PULSEAUDIO_pa_stream_set_write_callback)(pa_stream *, pa_stream_request_cb_t, void *);
static void (*PULSEAUDIO_pa_stream_set_read_callback)(pa_stream *, pa_stream_request_cb_t, void *);
static pa_operation *(*PULSEAUDIO_pa_context_get_server_info)(pa_context *, pa_server_info_cb_t, void *);

static bool load_pulseaudio_syms(void);

#ifdef SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC

static const char *pulseaudio_library = SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC;
static SDL_SharedObject *pulseaudio_handle = NULL;

static bool load_pulseaudio_sym(const char *fn, void **addr)
{
    *addr = SDL_LoadFunction(pulseaudio_handle, fn);
    if (!*addr) {
        // Don't call SDL_SetError(): SDL_LoadFunction already did.
        return false;
    }

    return true;
}

// cast funcs to char* first, to please GCC's strict aliasing rules.
#define SDL_PULSEAUDIO_SYM(x)                                       \
    if (!load_pulseaudio_sym(#x, (void **)(char *)&PULSEAUDIO_##x)) \
        return false

static void UnloadPulseAudioLibrary(void)
{
    if (pulseaudio_handle) {
        SDL_UnloadObject(pulseaudio_handle);
        pulseaudio_handle = NULL;
    }
}

static bool LoadPulseAudioLibrary(void)
{
    bool result = true;
    if (!pulseaudio_handle) {
        pulseaudio_handle = SDL_LoadObject(pulseaudio_library);
        if (!pulseaudio_handle) {
            result = false;
            // Don't call SDL_SetError(): SDL_LoadObject already did.
        } else {
            result = load_pulseaudio_syms();
            if (!result) {
                UnloadPulseAudioLibrary();
            }
        }
    }
    return result;
}

#else

#define SDL_PULSEAUDIO_SYM(x) PULSEAUDIO_##x = x

static void UnloadPulseAudioLibrary(void)
{
}

static bool LoadPulseAudioLibrary(void)
{
    load_pulseaudio_syms();
    return true;
}

#endif // SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC

static bool load_pulseaudio_syms(void)
{
    SDL_PULSEAUDIO_SYM(pa_get_library_version);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_new);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_get_api);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_start);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_stop);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_lock);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_unlock);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_wait);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_signal);
    SDL_PULSEAUDIO_SYM(pa_threaded_mainloop_free);
    SDL_PULSEAUDIO_SYM(pa_operation_get_state);
    SDL_PULSEAUDIO_SYM(pa_operation_cancel);
    SDL_PULSEAUDIO_SYM(pa_operation_unref);
    SDL_PULSEAUDIO_SYM(pa_context_new_with_proplist);
    SDL_PULSEAUDIO_SYM(pa_context_set_state_callback);
    SDL_PULSEAUDIO_SYM(pa_context_connect);
    SDL_PULSEAUDIO_SYM(pa_context_get_sink_info_list);
    SDL_PULSEAUDIO_SYM(pa_context_get_source_info_list);
    SDL_PULSEAUDIO_SYM(pa_context_get_sink_info_by_index);
    SDL_PULSEAUDIO_SYM(pa_context_get_source_info_by_index);
    SDL_PULSEAUDIO_SYM(pa_context_get_state);
    SDL_PULSEAUDIO_SYM(pa_context_subscribe);
    SDL_PULSEAUDIO_SYM(pa_context_set_subscribe_callback);
    SDL_PULSEAUDIO_SYM(pa_context_disconnect);
    SDL_PULSEAUDIO_SYM(pa_context_unref);
    SDL_PULSEAUDIO_SYM(pa_stream_new);
    SDL_PULSEAUDIO_SYM(pa_stream_set_state_callback);
    SDL_PULSEAUDIO_SYM(pa_stream_connect_playback);
    SDL_PULSEAUDIO_SYM(pa_stream_connect_record);
    SDL_PULSEAUDIO_SYM(pa_stream_get_buffer_attr);
    SDL_PULSEAUDIO_SYM(pa_stream_get_state);
    SDL_PULSEAUDIO_SYM(pa_stream_writable_size);
    SDL_PULSEAUDIO_SYM(pa_stream_readable_size);
    SDL_PULSEAUDIO_SYM(pa_stream_begin_write);
    SDL_PULSEAUDIO_SYM(pa_stream_write);
    SDL_PULSEAUDIO_SYM(pa_stream_drain);
    SDL_PULSEAUDIO_SYM(pa_stream_disconnect);
    SDL_PULSEAUDIO_SYM(pa_stream_peek);
    SDL_PULSEAUDIO_SYM(pa_stream_drop);
    SDL_PULSEAUDIO_SYM(pa_stream_flush);
    SDL_PULSEAUDIO_SYM(pa_stream_unref);
    SDL_PULSEAUDIO_SYM(pa_channel_map_init_auto);
    SDL_PULSEAUDIO_SYM(pa_strerror);
    SDL_PULSEAUDIO_SYM(pa_stream_set_write_callback);
    SDL_PULSEAUDIO_SYM(pa_stream_set_read_callback);
    SDL_PULSEAUDIO_SYM(pa_context_get_server_info);
    SDL_PULSEAUDIO_SYM(pa_proplist_new);
    SDL_PULSEAUDIO_SYM(pa_proplist_free);
    SDL_PULSEAUDIO_SYM(pa_proplist_sets);

    // optional
#ifdef SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC
    load_pulseaudio_sym("pa_operation_set_state_callback", (void **)(char *)&PULSEAUDIO_pa_operation_set_state_callback);  // needs pulseaudio 4.0
    load_pulseaudio_sym("pa_threaded_mainloop_set_name", (void **)(char *)&PULSEAUDIO_pa_threaded_mainloop_set_name);  // needs pulseaudio 5.0
#elif (PA_PROTOCOL_VERSION >= 29)
    PULSEAUDIO_pa_operation_set_state_callback = pa_operation_set_state_callback;
    PULSEAUDIO_pa_threaded_mainloop_set_name = pa_threaded_mainloop_set_name;
#elif (PA_PROTOCOL_VERSION >= 28)
    PULSEAUDIO_pa_operation_set_state_callback = pa_operation_set_state_callback;
    PULSEAUDIO_pa_threaded_mainloop_set_name = NULL;
#else
    PULSEAUDIO_pa_operation_set_state_callback = NULL;
    PULSEAUDIO_pa_threaded_mainloop_set_name = NULL;
#endif

    return true;
}

static const char *getAppName(void)
{
    return SDL_GetAppMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING);
}

static void ThreadedMainloopSignal(void)
{
    PULSEAUDIO_pa_threaded_mainloop_signal(pulseaudio_threaded_mainloop, 0);  // alert waiting threads to unblock.

    // we need to kill any SDL_SetError state; we didn't create this thread
    //  so its SDL TLS slot will leak otherwise, so we do this every time
    //  we're (presumably) losing control of the thread.
    SDL_CleanupTLS();
}

static void OperationStateChangeCallback(pa_operation *o, void *userdata)
{
    ThreadedMainloopSignal();  // just signal any waiting code, it can look up the details.
}

/* This function assume you are holding `mainloop`'s lock. The operation is unref'd in here, assuming
   you did the work in the callback and just want to know it's done, though. */
static void WaitForPulseOperation(pa_operation *o)
{
    // This checks for NO errors currently. Either fix that, check results elsewhere, or do things you don't care about.
    SDL_assert(pulseaudio_threaded_mainloop != NULL);
    if (o) {
        // note that if PULSEAUDIO_pa_operation_set_state_callback == NULL, then `o` must have a callback that will signal pulseaudio_threaded_mainloop.
        // If not, on really old (earlier PulseAudio 4.0, from the year 2013!) installs, this call will block forever.
        // On more modern installs, we won't ever block forever, and maybe be more efficient, thanks to pa_operation_set_state_callback.
        // WARNING: at the time of this writing: the Steam Runtime is still on PulseAudio 1.1!
        if (PULSEAUDIO_pa_operation_set_state_callback) {
            PULSEAUDIO_pa_operation_set_state_callback(o, OperationStateChangeCallback, NULL);
        }
        while (PULSEAUDIO_pa_operation_get_state(o) == PA_OPERATION_RUNNING) {
            PULSEAUDIO_pa_threaded_mainloop_wait(pulseaudio_threaded_mainloop);  // this releases the lock and blocks on an internal condition variable.
        }
        PULSEAUDIO_pa_operation_unref(o);
    }
}

static void DisconnectFromPulseServer(void)
{
    if (pulseaudio_threaded_mainloop) {
        PULSEAUDIO_pa_threaded_mainloop_stop(pulseaudio_threaded_mainloop);
    }
    if (pulseaudio_context) {
        PULSEAUDIO_pa_context_disconnect(pulseaudio_context);
        PULSEAUDIO_pa_context_unref(pulseaudio_context);
        pulseaudio_context = NULL;
    }
    if (pulseaudio_threaded_mainloop) {
        PULSEAUDIO_pa_threaded_mainloop_free(pulseaudio_threaded_mainloop);
        pulseaudio_threaded_mainloop = NULL;
    }
}

static void PulseContextStateChangeCallback(pa_context *context, void *userdata)
{
    ThreadedMainloopSignal();  // just signal any waiting code, it can look up the details.
}

static bool ConnectToPulseServer(void)
{
    pa_mainloop_api *mainloop_api = NULL;
    pa_proplist *proplist = NULL;
    const char *icon_name;
    int state = 0;

    SDL_assert(pulseaudio_threaded_mainloop == NULL);
    SDL_assert(pulseaudio_context == NULL);

    // Set up a new main loop
    pulseaudio_threaded_mainloop = PULSEAUDIO_pa_threaded_mainloop_new();
    if (!pulseaudio_threaded_mainloop) {
        return SDL_SetError("pa_threaded_mainloop_new() failed");
    }

    if (PULSEAUDIO_pa_threaded_mainloop_set_name) {
        PULSEAUDIO_pa_threaded_mainloop_set_name(pulseaudio_threaded_mainloop, "PulseMainloop");
    }

    if (PULSEAUDIO_pa_threaded_mainloop_start(pulseaudio_threaded_mainloop) < 0) {
        PULSEAUDIO_pa_threaded_mainloop_free(pulseaudio_threaded_mainloop);
        pulseaudio_threaded_mainloop = NULL;
        return SDL_SetError("pa_threaded_mainloop_start() failed");
    }

    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);

    mainloop_api = PULSEAUDIO_pa_threaded_mainloop_get_api(pulseaudio_threaded_mainloop);
    SDL_assert(mainloop_api != NULL); // this never fails, right?

    proplist = PULSEAUDIO_pa_proplist_new();
    if (!proplist) {
        SDL_SetError("pa_proplist_new() failed");
        goto failed;
    }

    icon_name = SDL_GetHint(SDL_HINT_AUDIO_DEVICE_APP_ICON_NAME);
    if (!icon_name || *icon_name == '\0') {
        icon_name = "applications-games";
    }
    PULSEAUDIO_pa_proplist_sets(proplist, PA_PROP_APPLICATION_ICON_NAME, icon_name);

    pulseaudio_context = PULSEAUDIO_pa_context_new_with_proplist(mainloop_api, getAppName(), proplist);
    if (!pulseaudio_context) {
        SDL_SetError("pa_context_new_with_proplist() failed");
        goto failed;
    }
    PULSEAUDIO_pa_proplist_free(proplist);

    PULSEAUDIO_pa_context_set_state_callback(pulseaudio_context, PulseContextStateChangeCallback, NULL);

    // Connect to the PulseAudio server
    if (PULSEAUDIO_pa_context_connect(pulseaudio_context, NULL, 0, NULL) < 0) {
        SDL_SetError("Could not setup connection to PulseAudio");
        goto failed;
    }

    state = PULSEAUDIO_pa_context_get_state(pulseaudio_context);
    while (PA_CONTEXT_IS_GOOD(state) && (state != PA_CONTEXT_READY)) {
        PULSEAUDIO_pa_threaded_mainloop_wait(pulseaudio_threaded_mainloop);
        state = PULSEAUDIO_pa_context_get_state(pulseaudio_context);
    }

    if (state != PA_CONTEXT_READY) {
        SDL_SetError("Could not connect to PulseAudio");
        goto failed;
    }

    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);

    return true; // connected and ready!

failed:
    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);
    DisconnectFromPulseServer();
    return false;
}

static void WriteCallback(pa_stream *p, size_t nbytes, void *userdata)
{
    struct SDL_PrivateAudioData *h = (struct SDL_PrivateAudioData *)userdata;
    //SDL_Log("PULSEAUDIO WRITE CALLBACK! nbytes=%u", (unsigned int) nbytes);
    h->bytes_requested += nbytes;
    ThreadedMainloopSignal();
}

// This function waits until it is possible to write a full sound buffer
static bool PULSEAUDIO_WaitDevice(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    bool result = true;

    //SDL_Log("PULSEAUDIO WAITDEVICE START! mixlen=%d", available);

    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);

    while (!SDL_GetAtomicInt(&device->shutdown) && (h->bytes_requested == 0)) {
        //SDL_Log("PULSEAUDIO WAIT IN WAITDEVICE!");
        PULSEAUDIO_pa_threaded_mainloop_wait(pulseaudio_threaded_mainloop);

        if ((PULSEAUDIO_pa_context_get_state(pulseaudio_context) != PA_CONTEXT_READY) || (PULSEAUDIO_pa_stream_get_state(h->stream) != PA_STREAM_READY)) {
            //SDL_Log("PULSEAUDIO DEVICE FAILURE IN WAITDEVICE!");
            result = false;
            break;
        }
    }

    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);

    return result;
}

static bool PULSEAUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buffer_size)
{
    struct SDL_PrivateAudioData *h = device->hidden;

    //SDL_Log("PULSEAUDIO PLAYDEVICE START! mixlen=%d", available);

    SDL_assert(h->bytes_requested >= buffer_size);

    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);
    const int rc = PULSEAUDIO_pa_stream_write(h->stream, buffer, buffer_size, NULL, 0LL, PA_SEEK_RELATIVE);
    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);

    if (rc < 0) {
        return false;
    }

    //SDL_Log("PULSEAUDIO FEED! nbytes=%d", buffer_size);
    h->bytes_requested -= buffer_size;

    //SDL_Log("PULSEAUDIO PLAYDEVICE END! written=%d", written);
    return true;
}

static Uint8 *PULSEAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    const size_t reqsize = (size_t) SDL_min(*buffer_size, h->bytes_requested);
    size_t nbytes = reqsize;
    void *data = NULL;
    if (PULSEAUDIO_pa_stream_begin_write(h->stream, &data, &nbytes) == 0) {
        *buffer_size = (int) nbytes;
        return (Uint8 *) data;
    }

    // don't know why this would fail, but we'll fall back just in case.
    *buffer_size = (int) reqsize;
    return device->hidden->mixbuf;
}

static void ReadCallback(pa_stream *p, size_t nbytes, void *userdata)
{
    //SDL_Log("PULSEAUDIO READ CALLBACK! nbytes=%u", (unsigned int) nbytes);
    ThreadedMainloopSignal();  // the recording code queries what it needs, we just need to signal to end any wait
}

static bool PULSEAUDIO_WaitRecordingDevice(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *h = device->hidden;

    if (h->recordingbuf) {
        return true;  // there's still data available to read.
    }

    bool result = true;

    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);

    while (!SDL_GetAtomicInt(&device->shutdown)) {
        PULSEAUDIO_pa_threaded_mainloop_wait(pulseaudio_threaded_mainloop);
        if ((PULSEAUDIO_pa_context_get_state(pulseaudio_context) != PA_CONTEXT_READY) || (PULSEAUDIO_pa_stream_get_state(h->stream) != PA_STREAM_READY)) {
            //SDL_Log("PULSEAUDIO DEVICE FAILURE IN WAITRECORDINGDEVICE!");
            result = false;
            break;
        } else if (PULSEAUDIO_pa_stream_readable_size(h->stream) > 0) {
            // a new fragment is available!
            const void *data = NULL;
            size_t nbytes = 0;
            PULSEAUDIO_pa_stream_peek(h->stream, &data, &nbytes);
            SDL_assert(nbytes > 0);
            if (!data) {  // If NULL, then the buffer had a hole, ignore that
                PULSEAUDIO_pa_stream_drop(h->stream);  // drop this fragment.
            } else {
                // store this fragment's data for use with RecordDevice
                //SDL_Log("PULSEAUDIO: recorded %d new bytes", (int) nbytes);
                h->recordingbuf = (const Uint8 *)data;
                h->recordinglen = nbytes;
                break;
            }
        }
    }

    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);

    return result;
}

static int PULSEAUDIO_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    struct SDL_PrivateAudioData *h = device->hidden;

    if (h->recordingbuf) {
        const int cpy = SDL_min(buflen, h->recordinglen);
        if (cpy > 0) {
            //SDL_Log("PULSEAUDIO: fed %d recorded bytes", cpy);
            SDL_memcpy(buffer, h->recordingbuf, cpy);
            h->recordingbuf += cpy;
            h->recordinglen -= cpy;
        }
        if (h->recordinglen == 0) {
            h->recordingbuf = NULL;
            PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);  // don't know if you _have_ to lock for this, but just in case.
            PULSEAUDIO_pa_stream_drop(h->stream); // done with this fragment.
            PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);
        }
        return cpy; // new data, return it.
    }

    return 0;
}

static void PULSEAUDIO_FlushRecording(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    const void *data = NULL;
    size_t nbytes = 0, buflen = 0;

    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);

    if (h->recordingbuf) {
        PULSEAUDIO_pa_stream_drop(h->stream);
        h->recordingbuf = NULL;
        h->recordinglen = 0;
    }

    buflen = PULSEAUDIO_pa_stream_readable_size(h->stream);
    while (!SDL_GetAtomicInt(&device->shutdown) && (buflen > 0)) {
        PULSEAUDIO_pa_threaded_mainloop_wait(pulseaudio_threaded_mainloop);
        if ((PULSEAUDIO_pa_context_get_state(pulseaudio_context) != PA_CONTEXT_READY) || (PULSEAUDIO_pa_stream_get_state(h->stream) != PA_STREAM_READY)) {
            //SDL_Log("PULSEAUDIO DEVICE FAILURE IN FLUSHRECORDING!");
            SDL_AudioDeviceDisconnected(device);
            break;
        }

        // a fragment of audio present before FlushCapture was call is
        // still available! Just drop it.
        PULSEAUDIO_pa_stream_peek(h->stream, &data, &nbytes);
        PULSEAUDIO_pa_stream_drop(h->stream);
        buflen -= nbytes;
    }

    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);
}

static void PULSEAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);

    if (device->hidden->stream) {
        if (device->hidden->recordingbuf) {
            PULSEAUDIO_pa_stream_drop(device->hidden->stream);
        }
        PULSEAUDIO_pa_stream_disconnect(device->hidden->stream);
        PULSEAUDIO_pa_stream_unref(device->hidden->stream);
    }
    PULSEAUDIO_pa_threaded_mainloop_signal(pulseaudio_threaded_mainloop, 0);  // in case the device thread is waiting somewhere, this will unblock it.
    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);

    SDL_free(device->hidden->mixbuf);
    SDL_free(device->hidden);
}

static void PulseStreamStateChangeCallback(pa_stream *stream, void *userdata)
{
    ThreadedMainloopSignal();  // just signal any waiting code, it can look up the details.
}

static bool PULSEAUDIO_OpenDevice(SDL_AudioDevice *device)
{
    const bool recording = device->recording;
    struct SDL_PrivateAudioData *h = NULL;
    SDL_AudioFormat test_format;
    const SDL_AudioFormat *closefmts;
    pa_sample_spec paspec;
    pa_buffer_attr paattr;
    pa_channel_map pacmap;
    pa_stream_flags_t flags = 0;
    int format = PA_SAMPLE_INVALID;
    bool result = true;

    SDL_assert(pulseaudio_threaded_mainloop != NULL);
    SDL_assert(pulseaudio_context != NULL);

    // Initialize all variables that we clean on shutdown
    h = device->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    // Try for a closest match on audio format
    closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
#ifdef DEBUG_AUDIO
        SDL_Log("pulseaudio: Trying format 0x%4.4x", test_format);
#endif
        switch (test_format) {
        case SDL_AUDIO_U8:
            format = PA_SAMPLE_U8;
            break;
        case SDL_AUDIO_S16LE:
            format = PA_SAMPLE_S16LE;
            break;
        case SDL_AUDIO_S16BE:
            format = PA_SAMPLE_S16BE;
            break;
        case SDL_AUDIO_S32LE:
            format = PA_SAMPLE_S32LE;
            break;
        case SDL_AUDIO_S32BE:
            format = PA_SAMPLE_S32BE;
            break;
        case SDL_AUDIO_F32LE:
            format = PA_SAMPLE_FLOAT32LE;
            break;
        case SDL_AUDIO_F32BE:
            format = PA_SAMPLE_FLOAT32BE;
            break;
        default:
            continue;
        }
        break;
    }
    if (!test_format) {
        return SDL_SetError("pulseaudio: Unsupported audio format");
    }
    device->spec.format = test_format;
    paspec.format = format;

    // Calculate the final parameters for this audio specification
    SDL_UpdatedAudioDeviceFormat(device);

    // Allocate mixing buffer
    if (!recording) {
        h->mixbuf = (Uint8 *)SDL_malloc(device->buffer_size);
        if (!h->mixbuf) {
            return false;
        }
        SDL_memset(h->mixbuf, device->silence_value, device->buffer_size);
    }

    paspec.channels = device->spec.channels;
    paspec.rate = device->spec.freq;

    // Reduced prebuffering compared to the defaults.
    paattr.fragsize = device->buffer_size;   // despite the name, this is only used for recording devices, according to PulseAudio docs!
    paattr.tlength = device->buffer_size;
    paattr.prebuf = -1;
    paattr.maxlength = -1;
    paattr.minreq = -1;
    flags |= PA_STREAM_ADJUST_LATENCY;

    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);

    const char *name = SDL_GetHint(SDL_HINT_AUDIO_DEVICE_STREAM_NAME);
    // The SDL ALSA output hints us that we use Windows' channel mapping
    // https://bugzilla.libsdl.org/show_bug.cgi?id=110
    PULSEAUDIO_pa_channel_map_init_auto(&pacmap, device->spec.channels, PA_CHANNEL_MAP_WAVEEX);

    h->stream = PULSEAUDIO_pa_stream_new(
        pulseaudio_context,
        (name && *name) ? name : "Audio Stream", // stream description
        &paspec,                                 // sample format spec
        &pacmap                                  // channel map
    );

    if (!h->stream) {
        result = SDL_SetError("Could not set up PulseAudio stream");
    } else {
        int rc;

        PULSEAUDIO_pa_stream_set_state_callback(h->stream, PulseStreamStateChangeCallback, NULL);

        // SDL manages device moves if the default changes, so don't ever let Pulse automatically migrate this stream.
        // UPDATE: This prevents users from moving the audio to a new sink (device) using standard tools. This is slightly in conflict
        //  with how SDL wants to manage audio devices, but if people want to do it, we should let them, so this is commented out
        //  for now. We might revisit later.
        //flags |= PA_STREAM_DONT_MOVE;

        const char *device_path = ((PulseDeviceHandle *) device->handle)->device_path;
        if (recording) {
            PULSEAUDIO_pa_stream_set_read_callback(h->stream, ReadCallback, h);
            rc = PULSEAUDIO_pa_stream_connect_record(h->stream, device_path, &paattr, flags);
        } else {
            PULSEAUDIO_pa_stream_set_write_callback(h->stream, WriteCallback, h);
            rc = PULSEAUDIO_pa_stream_connect_playback(h->stream, device_path, &paattr, flags, NULL, NULL);
        }

        if (rc < 0) {
            result = SDL_SetError("Could not connect PulseAudio stream");
        } else {
            int state = PULSEAUDIO_pa_stream_get_state(h->stream);
            while (PA_STREAM_IS_GOOD(state) && (state != PA_STREAM_READY)) {
                PULSEAUDIO_pa_threaded_mainloop_wait(pulseaudio_threaded_mainloop);
                state = PULSEAUDIO_pa_stream_get_state(h->stream);
            }

            if (!PA_STREAM_IS_GOOD(state)) {
                result = SDL_SetError("Could not connect PulseAudio stream");
            } else {
                const pa_buffer_attr *actual_bufattr = PULSEAUDIO_pa_stream_get_buffer_attr(h->stream);
                if (!actual_bufattr) {
                    result = SDL_SetError("Could not determine connected PulseAudio stream's buffer attributes");
                } else {
                    device->buffer_size = (int) recording ? actual_bufattr->tlength : actual_bufattr->fragsize;
                    device->sample_frames = device->buffer_size / SDL_AUDIO_FRAMESIZE(device->spec);
                }
            }
        }
    }

    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);

    // We're (hopefully) ready to rock and roll. :-)
    return result;
}

// device handles are device index + 1, cast to void*, so we never pass a NULL.

static SDL_AudioFormat PulseFormatToSDLFormat(pa_sample_format_t format)
{
    switch (format) {
    case PA_SAMPLE_U8:
        return SDL_AUDIO_U8;
    case PA_SAMPLE_S16LE:
        return SDL_AUDIO_S16LE;
    case PA_SAMPLE_S16BE:
        return SDL_AUDIO_S16BE;
    case PA_SAMPLE_S32LE:
        return SDL_AUDIO_S32LE;
    case PA_SAMPLE_S32BE:
        return SDL_AUDIO_S32BE;
    case PA_SAMPLE_FLOAT32LE:
        return SDL_AUDIO_F32LE;
    case PA_SAMPLE_FLOAT32BE:
        return SDL_AUDIO_F32BE;
    default:
        return 0;
    }
}

static void AddPulseAudioDevice(const bool recording, const char *description, const char *name, const uint32_t index, const pa_sample_spec *sample_spec)
{
    SDL_AudioSpec spec;
    SDL_zero(spec);
    spec.format = PulseFormatToSDLFormat(sample_spec->format);
    spec.channels = sample_spec->channels;
    spec.freq = sample_spec->rate;
    PulseDeviceHandle *handle = (PulseDeviceHandle *) SDL_malloc(sizeof (PulseDeviceHandle));
    if (handle) {
        handle->device_path = SDL_strdup(name);
        if (!handle->device_path) {
            SDL_free(handle);
        } else {
            handle->device_index = index;
            SDL_AddAudioDevice(recording, description, &spec, handle);
        }
    }
}

// This is called when PulseAudio adds an playback ("sink") device.
static void SinkInfoCallback(pa_context *c, const pa_sink_info *i, int is_last, void *data)
{
    if (i) {
        AddPulseAudioDevice(false, i->description, i->name, i->index, &i->sample_spec);
    }
    ThreadedMainloopSignal();
}

// This is called when PulseAudio adds a recording ("source") device.
static void SourceInfoCallback(pa_context *c, const pa_source_info *i, int is_last, void *data)
{
    // Maybe skip "monitor" sources. These are just output from other sinks.
    if (i && (include_monitors || (i->monitor_of_sink == PA_INVALID_INDEX))) {
        AddPulseAudioDevice(true, i->description, i->name, i->index, &i->sample_spec);
    }
    ThreadedMainloopSignal();
}

static void ServerInfoCallback(pa_context *c, const pa_server_info *i, void *data)
{
    //SDL_Log("PULSEAUDIO ServerInfoCallback!");

    if (!default_sink_path || (SDL_strcmp(default_sink_path, i->default_sink_name) != 0)) {
        char *str = SDL_strdup(i->default_sink_name);
        if (str) {
            SDL_free(default_sink_path);
            default_sink_path = str;
            default_sink_changed = true;
        }
    }

    if (!default_source_path || (SDL_strcmp(default_source_path, i->default_source_name) != 0)) {
        char *str = SDL_strdup(i->default_source_name);
        if (str) {
            SDL_free(default_source_path);
            default_source_path = str;
            default_source_changed = true;
        }
    }

    ThreadedMainloopSignal();
}

static bool FindAudioDeviceByIndex(SDL_AudioDevice *device, void *userdata)
{
    const uint32_t idx = (uint32_t) (uintptr_t) userdata;
    const PulseDeviceHandle *handle = (const PulseDeviceHandle *) device->handle;
    return (handle->device_index == idx);
}

static bool FindAudioDeviceByPath(SDL_AudioDevice *device, void *userdata)
{
    const char *path = (const char *) userdata;
    const PulseDeviceHandle *handle = (const PulseDeviceHandle *) device->handle;
    return (SDL_strcmp(handle->device_path, path) == 0);
}

// This is called when PulseAudio has a device connected/removed/changed.
static void HotplugCallback(pa_context *c, pa_subscription_event_type_t t, uint32_t idx, void *data)
{
    const bool added = ((t & PA_SUBSCRIPTION_EVENT_TYPE_MASK) == PA_SUBSCRIPTION_EVENT_NEW);
    const bool removed = ((t & PA_SUBSCRIPTION_EVENT_TYPE_MASK) == PA_SUBSCRIPTION_EVENT_REMOVE);
    const bool changed = ((t & PA_SUBSCRIPTION_EVENT_TYPE_MASK) == PA_SUBSCRIPTION_EVENT_CHANGE);

    if (added || removed || changed) { // we only care about add/remove events.
        const bool sink = ((t & PA_SUBSCRIPTION_EVENT_FACILITY_MASK) == PA_SUBSCRIPTION_EVENT_SINK);
        const bool source = ((t & PA_SUBSCRIPTION_EVENT_FACILITY_MASK) == PA_SUBSCRIPTION_EVENT_SOURCE);

        if (changed) {
            PULSEAUDIO_pa_operation_unref(PULSEAUDIO_pa_context_get_server_info(pulseaudio_context, ServerInfoCallback, NULL));
        }

        /* adds need sink details from the PulseAudio server. Another callback...
           (just unref all these operations right away, because we aren't going to wait on them
           and their callbacks will handle any work, so they can free as soon as that happens.) */
        if (added && sink) {
            PULSEAUDIO_pa_operation_unref(PULSEAUDIO_pa_context_get_sink_info_by_index(pulseaudio_context, idx, SinkInfoCallback, NULL));
        } else if (added && source) {
            PULSEAUDIO_pa_operation_unref(PULSEAUDIO_pa_context_get_source_info_by_index(pulseaudio_context, idx, SourceInfoCallback, NULL));
        } else if (removed && (sink || source)) {
            // removes we can handle just with the device index.
            SDL_AudioDeviceDisconnected(SDL_FindPhysicalAudioDeviceByCallback(FindAudioDeviceByIndex, (void *)(uintptr_t)idx));
        }
    }
    ThreadedMainloopSignal();
}

static bool CheckDefaultDevice(const bool changed, char *device_path)
{
    if (!changed) {
        return false;  // nothing's happening, leave the flag marked as unchanged.
    } else if (!device_path) {
        return true;  // check again later, we don't have a device name...
    }

    SDL_AudioDevice *device = SDL_FindPhysicalAudioDeviceByCallback(FindAudioDeviceByPath, device_path);
    if (device) {  // if NULL, we might still be waiting for a SinkInfoCallback or something, we'll try later.
        SDL_DefaultAudioDeviceChanged(device);
        return false;  // changing complete, set flag to unchanged for future tests.
    }
    return true;  // couldn't find the changed device, leave it marked as changed to try again later.
}

// this runs as a thread while the Pulse target is initialized to catch hotplug events.
static int SDLCALL HotplugThread(void *data)
{
    pa_operation *op;

    SDL_SetCurrentThreadPriority(SDL_THREAD_PRIORITY_LOW);
    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);
    PULSEAUDIO_pa_context_set_subscribe_callback(pulseaudio_context, HotplugCallback, NULL);

    // don't WaitForPulseOperation on the subscription; when it's done we'll be able to get hotplug events, but waiting doesn't changing anything.
    op = PULSEAUDIO_pa_context_subscribe(pulseaudio_context, PA_SUBSCRIPTION_MASK_SINK | PA_SUBSCRIPTION_MASK_SOURCE | PA_SUBSCRIPTION_MASK_SERVER, NULL, NULL);

    SDL_SignalSemaphore((SDL_Semaphore *) data);

    while (SDL_GetAtomicInt(&pulseaudio_hotplug_thread_active)) {
        PULSEAUDIO_pa_threaded_mainloop_wait(pulseaudio_threaded_mainloop);
        if (op && PULSEAUDIO_pa_operation_get_state(op) != PA_OPERATION_RUNNING) {
            PULSEAUDIO_pa_operation_unref(op);
            op = NULL;
        }

        // Update default devices; don't hold the pulse lock during this, since it could deadlock vs a playing device that we're about to lock here.
        bool check_default_sink = default_sink_changed;
        bool check_default_source = default_source_changed;
        char *current_default_sink = check_default_sink ? SDL_strdup(default_sink_path) : NULL;
        char *current_default_source = check_default_source ? SDL_strdup(default_source_path) : NULL;
        default_sink_changed = default_source_changed = false;
        PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);
        check_default_sink = CheckDefaultDevice(check_default_sink, current_default_sink);
        check_default_source = CheckDefaultDevice(check_default_source, current_default_source);
        PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);

        // free our copies (which will be NULL if nothing changed)
        SDL_free(current_default_sink);
        SDL_free(current_default_source);

        // set these to true if we didn't handle the change OR there was _another_ change while we were working unlocked.
        default_sink_changed = (default_sink_changed || check_default_sink);
        default_source_changed = (default_source_changed || check_default_source);
    }

    if (op) {
        PULSEAUDIO_pa_operation_unref(op);
    }

    PULSEAUDIO_pa_context_set_subscribe_callback(pulseaudio_context, NULL, NULL);
    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);
    return 0;
}

static void PULSEAUDIO_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    SDL_Semaphore *ready_sem = SDL_CreateSemaphore(0);

    PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);
    WaitForPulseOperation(PULSEAUDIO_pa_context_get_server_info(pulseaudio_context, ServerInfoCallback, NULL));
    WaitForPulseOperation(PULSEAUDIO_pa_context_get_sink_info_list(pulseaudio_context, SinkInfoCallback, NULL));
    WaitForPulseOperation(PULSEAUDIO_pa_context_get_source_info_list(pulseaudio_context, SourceInfoCallback, NULL));
    PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);

    if (default_sink_path) {
        *default_playback = SDL_FindPhysicalAudioDeviceByCallback(FindAudioDeviceByPath, default_sink_path);
    }

    if (default_source_path) {
        *default_recording = SDL_FindPhysicalAudioDeviceByCallback(FindAudioDeviceByPath, default_source_path);
    }

    // ok, we have a sane list, let's set up hotplug notifications now...
    SDL_SetAtomicInt(&pulseaudio_hotplug_thread_active, 1);
    pulseaudio_hotplug_thread = SDL_CreateThread(HotplugThread, "PulseHotplug", ready_sem);
    if (pulseaudio_hotplug_thread) {
        SDL_WaitSemaphore(ready_sem);  // wait until the thread hits it's main loop.
    } else {
        SDL_SetAtomicInt(&pulseaudio_hotplug_thread_active, 0);  // thread failed to start, we'll go on without hotplug.
    }

    SDL_DestroySemaphore(ready_sem);
}

static void PULSEAUDIO_FreeDeviceHandle(SDL_AudioDevice *device)
{
    PulseDeviceHandle *handle = (PulseDeviceHandle *) device->handle;
    SDL_free(handle->device_path);
    SDL_free(handle);
}

static void PULSEAUDIO_DeinitializeStart(void)
{
    if (pulseaudio_hotplug_thread) {
        PULSEAUDIO_pa_threaded_mainloop_lock(pulseaudio_threaded_mainloop);
        SDL_SetAtomicInt(&pulseaudio_hotplug_thread_active, 0);
        PULSEAUDIO_pa_threaded_mainloop_signal(pulseaudio_threaded_mainloop, 0);
        PULSEAUDIO_pa_threaded_mainloop_unlock(pulseaudio_threaded_mainloop);
        SDL_WaitThread(pulseaudio_hotplug_thread, NULL);
        pulseaudio_hotplug_thread = NULL;
    }
}

static void PULSEAUDIO_Deinitialize(void)
{
    DisconnectFromPulseServer();

    SDL_free(default_sink_path);
    default_sink_path = NULL;
    default_sink_changed = false;
    SDL_free(default_source_path);
    default_source_path = NULL;
    default_source_changed = false;

    UnloadPulseAudioLibrary();
}

static bool PULSEAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    if (!LoadPulseAudioLibrary()) {
        return false;
    } else if (!ConnectToPulseServer()) {
        UnloadPulseAudioLibrary();
        return false;
    }

    include_monitors = SDL_GetHintBoolean(SDL_HINT_AUDIO_INCLUDE_MONITORS, false);

    impl->DetectDevices = PULSEAUDIO_DetectDevices;
    impl->OpenDevice = PULSEAUDIO_OpenDevice;
    impl->PlayDevice = PULSEAUDIO_PlayDevice;
    impl->WaitDevice = PULSEAUDIO_WaitDevice;
    impl->GetDeviceBuf = PULSEAUDIO_GetDeviceBuf;
    impl->CloseDevice = PULSEAUDIO_CloseDevice;
    impl->DeinitializeStart = PULSEAUDIO_DeinitializeStart;
    impl->Deinitialize = PULSEAUDIO_Deinitialize;
    impl->WaitRecordingDevice = PULSEAUDIO_WaitRecordingDevice;
    impl->RecordDevice = PULSEAUDIO_RecordDevice;
    impl->FlushRecording = PULSEAUDIO_FlushRecording;
    impl->FreeDeviceHandle = PULSEAUDIO_FreeDeviceHandle;

    impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap PULSEAUDIO_bootstrap = {
    "pulseaudio", "PulseAudio", PULSEAUDIO_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_PULSEAUDIO
