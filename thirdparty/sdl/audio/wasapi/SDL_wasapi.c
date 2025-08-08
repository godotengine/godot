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

#ifdef SDL_AUDIO_DRIVER_WASAPI

#include "../../core/windows/SDL_windows.h"
#include "../../core/windows/SDL_immdevice.h"
#include "../../thread/SDL_systhread.h"
#include "../SDL_sysaudio.h"

#define COBJMACROS
#include <audioclient.h>

#include "SDL_wasapi.h"

// These constants aren't available in older SDKs
#ifndef AUDCLNT_STREAMFLAGS_RATEADJUST
#define AUDCLNT_STREAMFLAGS_RATEADJUST 0x00100000
#endif
#ifndef AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
#define AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY 0x08000000
#endif
#ifndef AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
#define AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM 0x80000000
#endif

// handle to Avrt.dll--Vista and later!--for flagging the callback thread as "Pro Audio" (low latency).
static HMODULE libavrt = NULL;
typedef HANDLE(WINAPI *pfnAvSetMmThreadCharacteristicsW)(LPCWSTR, LPDWORD);
typedef BOOL(WINAPI *pfnAvRevertMmThreadCharacteristics)(HANDLE);
static pfnAvSetMmThreadCharacteristicsW pAvSetMmThreadCharacteristicsW = NULL;
static pfnAvRevertMmThreadCharacteristics pAvRevertMmThreadCharacteristics = NULL;

// Some GUIDs we need to know without linking to libraries that aren't available before Vista.
static const IID SDL_IID_IAudioRenderClient = { 0xf294acfc, 0x3146, 0x4483, { 0xa7, 0xbf, 0xad, 0xdc, 0xa7, 0xc2, 0x60, 0xe2 } };
static const IID SDL_IID_IAudioCaptureClient = { 0xc8adbd64, 0xe71e, 0x48a0, { 0xa4, 0xde, 0x18, 0x5c, 0x39, 0x5c, 0xd3, 0x17 } };
static const IID SDL_IID_IAudioClient = { 0x1cb9ad4c, 0xdbfa, 0x4c32, { 0xb1, 0x78, 0xc2, 0xf5, 0x68, 0xa7, 0x03, 0xb2 } };
#ifdef __IAudioClient3_INTERFACE_DEFINED__
static const IID SDL_IID_IAudioClient3 = { 0x7ed4ee07, 0x8e67, 0x4cd4, { 0x8c, 0x1a, 0x2b, 0x7a, 0x59, 0x87, 0xad, 0x42 } };
#endif //

static bool immdevice_initialized = false;

// WASAPI is _really_ particular about various things happening on the same thread, for COM and such,
//  so we proxy various stuff to a single background thread to manage.

typedef struct ManagementThreadPendingTask
{
    ManagementThreadTask fn;
    void *userdata;
    bool result;
    SDL_Semaphore *task_complete_sem;
    char *errorstr;
    struct ManagementThreadPendingTask *next;
} ManagementThreadPendingTask;

static SDL_Thread *ManagementThread = NULL;
static ManagementThreadPendingTask *ManagementThreadPendingTasks = NULL;
static SDL_Mutex *ManagementThreadLock = NULL;
static SDL_Condition *ManagementThreadCondition = NULL;
static SDL_AtomicInt ManagementThreadShutdown;

static void ManagementThreadMainloop(void)
{
    SDL_LockMutex(ManagementThreadLock);
    ManagementThreadPendingTask *task;
    while (((task = (ManagementThreadPendingTask *)SDL_GetAtomicPointer((void **)&ManagementThreadPendingTasks)) != NULL) || !SDL_GetAtomicInt(&ManagementThreadShutdown)) {
        if (!task) {
            SDL_WaitCondition(ManagementThreadCondition, ManagementThreadLock); // block until there's something to do.
        } else {
            SDL_SetAtomicPointer((void **) &ManagementThreadPendingTasks, task->next); // take task off the pending list.
            SDL_UnlockMutex(ManagementThreadLock);                       // let other things add to the list while we chew on this task.
            task->result = task->fn(task->userdata);                     // run this task.
            if (task->task_complete_sem) {                               // something waiting on result?
                task->errorstr = SDL_strdup(SDL_GetError());
                SDL_SignalSemaphore(task->task_complete_sem);
            } else { // nothing waiting, we're done, free it.
                SDL_free(task);
            }
            SDL_LockMutex(ManagementThreadLock); // regrab the lock so we can get the next task; if nothing to do, we'll release the lock in SDL_WaitCondition.
        }
    }
    SDL_UnlockMutex(ManagementThreadLock); // told to shut down and out of tasks, let go of the lock and return.
}

bool WASAPI_ProxyToManagementThread(ManagementThreadTask task, void *userdata, bool *wait_on_result)
{
    // We want to block for a result, but we are already running from the management thread! Just run the task now so we don't deadlock.
    if ((wait_on_result) && (SDL_GetCurrentThreadID() == SDL_GetThreadID(ManagementThread))) {
        *wait_on_result = task(userdata);
        return true;  // completed!
    }

    if (SDL_GetAtomicInt(&ManagementThreadShutdown)) {
        return SDL_SetError("Can't add task, we're shutting down");
    }

    ManagementThreadPendingTask *pending = (ManagementThreadPendingTask *)SDL_calloc(1, sizeof(ManagementThreadPendingTask));
    if (!pending) {
        return false;
    }

    pending->fn = task;
    pending->userdata = userdata;

    if (wait_on_result) {
        pending->task_complete_sem = SDL_CreateSemaphore(0);
        if (!pending->task_complete_sem) {
            SDL_free(pending);
            return false;
        }
    }

    pending->next = NULL;

    SDL_LockMutex(ManagementThreadLock);

    // add to end of task list.
    ManagementThreadPendingTask *prev = NULL;
    for (ManagementThreadPendingTask *i = (ManagementThreadPendingTask *)SDL_GetAtomicPointer((void **)&ManagementThreadPendingTasks); i; i = i->next) {
        prev = i;
    }

    if (prev) {
        prev->next = pending;
    } else {
        SDL_SetAtomicPointer((void **) &ManagementThreadPendingTasks, pending);
    }

    // task is added to the end of the pending list, let management thread rip!
    SDL_SignalCondition(ManagementThreadCondition);
    SDL_UnlockMutex(ManagementThreadLock);

    if (wait_on_result) {
        SDL_WaitSemaphore(pending->task_complete_sem);
        SDL_DestroySemaphore(pending->task_complete_sem);
        *wait_on_result = pending->result;
        if (pending->errorstr) {
            SDL_SetError("%s", pending->errorstr);
            SDL_free(pending->errorstr);
        }
        SDL_free(pending);
    }

    return true; // successfully added (and possibly executed)!
}

static bool mgmtthrtask_AudioDeviceDisconnected(void *userdata)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *) userdata;
    SDL_AudioDeviceDisconnected(device);
    UnrefPhysicalAudioDevice(device);  // make sure this lived until the task completes.
    return true;
}

static void AudioDeviceDisconnected(SDL_AudioDevice *device)
{
    // don't wait on this, IMMDevice's own thread needs to return or everything will deadlock.
    if (device) {
        RefPhysicalAudioDevice(device);  // make sure this lives until the task completes.
        WASAPI_ProxyToManagementThread(mgmtthrtask_AudioDeviceDisconnected, device, NULL);
    }
}

static bool mgmtthrtask_DefaultAudioDeviceChanged(void *userdata)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *) userdata;
    SDL_DefaultAudioDeviceChanged(device);
    UnrefPhysicalAudioDevice(device);  // make sure this lived until the task completes.
    return true;
}

static void DefaultAudioDeviceChanged(SDL_AudioDevice *new_default_device)
{
    // don't wait on this, IMMDevice's own thread needs to return or everything will deadlock.
    if (new_default_device) {
        RefPhysicalAudioDevice(new_default_device);  // make sure this lives until the task completes.
        WASAPI_ProxyToManagementThread(mgmtthrtask_DefaultAudioDeviceChanged, new_default_device, NULL);
    }
}

static void StopWasapiHotplug(void)
{
    if (immdevice_initialized) {
        SDL_IMMDevice_Quit();
        immdevice_initialized = false;
    }
}

static void Deinit(void)
{
    if (libavrt) {
        FreeLibrary(libavrt);
        libavrt = NULL;
    }

    pAvSetMmThreadCharacteristicsW = NULL;
    pAvRevertMmThreadCharacteristics = NULL;

    StopWasapiHotplug();

    WIN_CoUninitialize();
}

static bool ManagementThreadPrepare(void)
{
    const SDL_IMMDevice_callbacks callbacks = { AudioDeviceDisconnected, DefaultAudioDeviceChanged };
    if (FAILED(WIN_CoInitialize())) {
        return SDL_SetError("CoInitialize() failed");
    } else if (!SDL_IMMDevice_Init(&callbacks)) {
        return false; // Error string is set by SDL_IMMDevice_Init
    }

    immdevice_initialized = true;

    libavrt = LoadLibrary(TEXT("avrt.dll")); // this library is available in Vista and later. No WinXP, so have to LoadLibrary to use it for now!
    if (libavrt) {
        pAvSetMmThreadCharacteristicsW = (pfnAvSetMmThreadCharacteristicsW)GetProcAddress(libavrt, "AvSetMmThreadCharacteristicsW");
        pAvRevertMmThreadCharacteristics = (pfnAvRevertMmThreadCharacteristics)GetProcAddress(libavrt, "AvRevertMmThreadCharacteristics");
    }

    ManagementThreadLock = SDL_CreateMutex();
    if (!ManagementThreadLock) {
        Deinit();
        return false;
    }

    ManagementThreadCondition = SDL_CreateCondition();
    if (!ManagementThreadCondition) {
        SDL_DestroyMutex(ManagementThreadLock);
        ManagementThreadLock = NULL;
        Deinit();
        return false;
    }

    return true;
}

typedef struct
{
    char *errorstr;
    SDL_Semaphore *ready_sem;
} ManagementThreadEntryData;

static int ManagementThreadEntry(void *userdata)
{
    ManagementThreadEntryData *data = (ManagementThreadEntryData *)userdata;

    if (!ManagementThreadPrepare()) {
        data->errorstr = SDL_strdup(SDL_GetError());
        SDL_SignalSemaphore(data->ready_sem); // unblock calling thread.
        return 0;
    }

    SDL_SignalSemaphore(data->ready_sem); // unblock calling thread.
    ManagementThreadMainloop();

    Deinit();
    return 0;
}

static bool InitManagementThread(void)
{
    ManagementThreadEntryData mgmtdata;
    SDL_zero(mgmtdata);
    mgmtdata.ready_sem = SDL_CreateSemaphore(0);
    if (!mgmtdata.ready_sem) {
        return false;
    }

    SDL_SetAtomicPointer((void **) &ManagementThreadPendingTasks, NULL);
    SDL_SetAtomicInt(&ManagementThreadShutdown, 0);
    ManagementThread = SDL_CreateThreadWithStackSize(ManagementThreadEntry, "SDLWASAPIMgmt", 256 * 1024, &mgmtdata); // !!! FIXME: maybe even smaller stack size?
    if (!ManagementThread) {
        return false;
    }

    SDL_WaitSemaphore(mgmtdata.ready_sem);
    SDL_DestroySemaphore(mgmtdata.ready_sem);

    if (mgmtdata.errorstr) {
        SDL_WaitThread(ManagementThread, NULL);
        ManagementThread = NULL;
        SDL_SetError("%s", mgmtdata.errorstr);
        SDL_free(mgmtdata.errorstr);
        return false;
    }

    return true;
}

static void DeinitManagementThread(void)
{
    if (ManagementThread) {
        SDL_SetAtomicInt(&ManagementThreadShutdown, 1);
        SDL_LockMutex(ManagementThreadLock);
        SDL_SignalCondition(ManagementThreadCondition);
        SDL_UnlockMutex(ManagementThreadLock);
        SDL_WaitThread(ManagementThread, NULL);
        ManagementThread = NULL;
    }

    SDL_assert(SDL_GetAtomicPointer((void **) &ManagementThreadPendingTasks) == NULL);

    SDL_DestroyCondition(ManagementThreadCondition);
    SDL_DestroyMutex(ManagementThreadLock);
    ManagementThreadCondition = NULL;
    ManagementThreadLock = NULL;
    SDL_SetAtomicInt(&ManagementThreadShutdown, 0);
}

typedef struct
{
    SDL_AudioDevice **default_playback;
    SDL_AudioDevice **default_recording;
} mgmtthrtask_DetectDevicesData;

static bool mgmtthrtask_DetectDevices(void *userdata)
{
    mgmtthrtask_DetectDevicesData *data = (mgmtthrtask_DetectDevicesData *)userdata;
    SDL_IMMDevice_EnumerateEndpoints(data->default_playback, data->default_recording);
    return true;
}

static void WASAPI_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    bool rc;
    // this blocks because it needs to finish before the audio subsystem inits
    mgmtthrtask_DetectDevicesData data;
    data.default_playback = default_playback;
    data.default_recording = default_recording;
    WASAPI_ProxyToManagementThread(mgmtthrtask_DetectDevices, &data, &rc);
}

static bool mgmtthrtask_DisconnectDevice(void *userdata)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *) userdata;
    SDL_AudioDeviceDisconnected(device);
    UnrefPhysicalAudioDevice(device);
    return true;
}

void WASAPI_DisconnectDevice(SDL_AudioDevice *device)
{
    if (SDL_CompareAndSwapAtomicInt(&device->hidden->device_disconnecting, 0, 1)) {
        RefPhysicalAudioDevice(device); // will unref when the task ends.
        WASAPI_ProxyToManagementThread(mgmtthrtask_DisconnectDevice, device, NULL);
    }
}

static bool WasapiFailed(SDL_AudioDevice *device, const HRESULT err)
{
    if (err == S_OK) {
        return false;
    } else if (err == AUDCLNT_E_DEVICE_INVALIDATED) {
        device->hidden->device_lost = true;
    } else {
        device->hidden->device_dead = true;
    }

    return true;
}

static bool mgmtthrtask_StopAndReleaseClient(void *userdata)
{
    IAudioClient *client = (IAudioClient *) userdata;
    IAudioClient_Stop(client);
    IAudioClient_Release(client);
    return true;
}

static bool mgmtthrtask_ReleaseCaptureClient(void *userdata)
{
    IAudioCaptureClient_Release((IAudioCaptureClient *)userdata);
    return true;
}

static bool mgmtthrtask_ReleaseRenderClient(void *userdata)
{
    IAudioRenderClient_Release((IAudioRenderClient *)userdata);
    return true;
}

static bool mgmtthrtask_CoTaskMemFree(void *userdata)
{
    CoTaskMemFree(userdata);
    return true;
}

static bool mgmtthrtask_CloseHandle(void *userdata)
{
    CloseHandle((HANDLE) userdata);
    return true;
}

static void ResetWasapiDevice(SDL_AudioDevice *device)
{
    if (!device || !device->hidden) {
        return;
    }

    // just queue up all the tasks in the management thread and don't block.
    // We don't care when any of these actually get free'd.

    if (device->hidden->client) {
        IAudioClient *client = device->hidden->client;
        device->hidden->client = NULL;
        WASAPI_ProxyToManagementThread(mgmtthrtask_StopAndReleaseClient, client, NULL);
    }

    if (device->hidden->render) {
        IAudioRenderClient *render = device->hidden->render;
        device->hidden->render = NULL;
        WASAPI_ProxyToManagementThread(mgmtthrtask_ReleaseRenderClient, render, NULL);
    }

    if (device->hidden->capture) {
        IAudioCaptureClient *capture = device->hidden->capture;
        device->hidden->capture = NULL;
        WASAPI_ProxyToManagementThread(mgmtthrtask_ReleaseCaptureClient, capture, NULL);
    }

    if (device->hidden->waveformat) {
        void *ptr = device->hidden->waveformat;
        device->hidden->waveformat = NULL;
        WASAPI_ProxyToManagementThread(mgmtthrtask_CoTaskMemFree, ptr, NULL);
    }

    if (device->hidden->event) {
        HANDLE event = device->hidden->event;
        device->hidden->event = NULL;
        WASAPI_ProxyToManagementThread(mgmtthrtask_CloseHandle, (void *) event, NULL);
    }
}

static bool mgmtthrtask_ActivateDevice(void *userdata)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *) userdata;

    IMMDevice *immdevice = NULL;
    if (!SDL_IMMDevice_Get(device, &immdevice, device->recording)) {
        device->hidden->client = NULL;
        return false; // This is already set by SDL_IMMDevice_Get
    }

    // this is _not_ async in standard win32, yay!
    HRESULT ret = IMMDevice_Activate(immdevice, &SDL_IID_IAudioClient, CLSCTX_ALL, NULL, (void **)&device->hidden->client);
    IMMDevice_Release(immdevice);

    if (FAILED(ret)) {
        SDL_assert(device->hidden->client == NULL);
        return WIN_SetErrorFromHRESULT("WASAPI can't activate audio endpoint", ret);
    }

    SDL_assert(device->hidden->client != NULL);
    if (!WASAPI_PrepDevice(device)) { // not async, fire it right away.
        return false;
    }

    return true; // good to go.
}

static bool ActivateWasapiDevice(SDL_AudioDevice *device)
{
    // this blocks because we're either being notified from a background thread or we're running during device open,
    //  both of which won't deadlock vs the device thread.
    bool rc = false;
    return (WASAPI_ProxyToManagementThread(mgmtthrtask_ActivateDevice, device, &rc) && rc);
}

// do not call when holding the device lock!
static bool RecoverWasapiDevice(SDL_AudioDevice *device)
{
    ResetWasapiDevice(device); // dump the lost device's handles.

    // This handles a non-default device that simply had its format changed in the Windows Control Panel.
    if (!ActivateWasapiDevice(device)) {
        WASAPI_DisconnectDevice(device);
        return false;
    }

    device->hidden->device_lost = false;

    return true; // okay, carry on with new device details!
}

// do not call when holding the device lock!
static bool RecoverWasapiIfLost(SDL_AudioDevice *device)
{
    if (SDL_GetAtomicInt(&device->shutdown)) {
        return false;                         // closing, stop trying.
    } else if (SDL_GetAtomicInt(&device->hidden->device_disconnecting)) {
        return false; // failing via the WASAPI management thread, stop trying.
    } else if (device->hidden->device_dead) { // had a fatal error elsewhere, clean up and quit
        IAudioClient_Stop(device->hidden->client);
        WASAPI_DisconnectDevice(device);
        SDL_assert(SDL_GetAtomicInt(&device->shutdown));  // so we don't come back through here.
        return false; // already failed.
    } else if (SDL_GetAtomicInt(&device->zombie)) {
        return false;  // we're already dead, so just leave and let the Zombie implementations take over.
    } else if (!device->hidden->client) {
        return true; // still waiting for activation.
    }

    return device->hidden->device_lost ? RecoverWasapiDevice(device) : true;
}

static Uint8 *WASAPI_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    // get an endpoint buffer from WASAPI.
    BYTE *buffer = NULL;

    if (device->hidden->render) {
        const HRESULT ret = IAudioRenderClient_GetBuffer(device->hidden->render, device->sample_frames, &buffer);
        if (ret == AUDCLNT_E_BUFFER_TOO_LARGE) {
            SDL_assert(buffer == NULL);
            *buffer_size = 0;  // just go back to WaitDevice and try again after the hardware has consumed some more data.
        } else if (WasapiFailed(device, ret)) {
            SDL_assert(buffer == NULL);
            if (device->hidden->device_lost) {  // just use an available buffer, we won't be playing it anyhow.
                *buffer_size = 0;  // we'll recover during WaitDevice and try again.
            }
        }
    }

    return (Uint8 *)buffer;
}

static bool WASAPI_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    if (device->hidden->render && !SDL_GetAtomicInt(&device->hidden->device_disconnecting)) { // definitely activated?
        // WasapiFailed() will mark the device for reacquisition or removal elsewhere.
        WasapiFailed(device, IAudioRenderClient_ReleaseBuffer(device->hidden->render, device->sample_frames, 0));
    }
    return true;
}

static bool WASAPI_WaitDevice(SDL_AudioDevice *device)
{
    // WaitDevice does not hold the device lock, so check for recovery/disconnect details here.
    while (RecoverWasapiIfLost(device) && device->hidden->client && device->hidden->event) {
        if (device->recording) {
            // Recording devices should return immediately if there is any data available
            UINT32 padding = 0;
            if (!WasapiFailed(device, IAudioClient_GetCurrentPadding(device->hidden->client, &padding))) {
                //SDL_Log("WASAPI EVENT! padding=%u maxpadding=%u", (unsigned int)padding, (unsigned int)maxpadding);
                if (padding > 0) {
                    break;
                }
            }

            switch (WaitForSingleObjectEx(device->hidden->event, 200, FALSE)) {
            case WAIT_OBJECT_0:
            case WAIT_TIMEOUT:
                break;

            default:
                //SDL_Log("WASAPI FAILED EVENT!");
                IAudioClient_Stop(device->hidden->client);
                return false;
            }
        } else {
            DWORD waitResult = WaitForSingleObjectEx(device->hidden->event, 200, FALSE);
            if (waitResult == WAIT_OBJECT_0) {
                UINT32 padding = 0;
                if (!WasapiFailed(device, IAudioClient_GetCurrentPadding(device->hidden->client, &padding))) {
                    //SDL_Log("WASAPI EVENT! padding=%u maxpadding=%u", (unsigned int)padding, (unsigned int)maxpadding);
                    if (padding <= (UINT32)device->sample_frames) {
                        break;
                    }
                }
            } else if (waitResult != WAIT_TIMEOUT) {
                //SDL_Log("WASAPI FAILED EVENT!");*/
                IAudioClient_Stop(device->hidden->client);
                return false;
            }
        }
    }

    return true;
}

static int WASAPI_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    BYTE *ptr = NULL;
    UINT32 frames = 0;
    DWORD flags = 0;

    while (device->hidden->capture) {
        const HRESULT ret = IAudioCaptureClient_GetBuffer(device->hidden->capture, &ptr, &frames, &flags, NULL, NULL);
        if (ret == AUDCLNT_S_BUFFER_EMPTY) {
            return 0;  // in theory we should have waited until there was data, but oh well, we'll go back to waiting. Returning 0 is safe in SDL3.
        }

        WasapiFailed(device, ret); // mark device lost/failed if necessary.

        if (ret == S_OK) {
            const int total = ((int)frames) * device->hidden->framesize;
            const int cpy = SDL_min(buflen, total);
            const int leftover = total - cpy;
            const bool silent = (flags & AUDCLNT_BUFFERFLAGS_SILENT) ? true : false;

            SDL_assert(leftover == 0);  // according to MSDN, this isn't everything available, just one "packet" of data per-GetBuffer call.

            if (silent) {
                SDL_memset(buffer, device->silence_value, cpy);
            } else {
                SDL_memcpy(buffer, ptr, cpy);
            }

            WasapiFailed(device, IAudioCaptureClient_ReleaseBuffer(device->hidden->capture, frames));

            return cpy;
        }
    }

    return -1; // unrecoverable error.
}

static void WASAPI_FlushRecording(SDL_AudioDevice *device)
{
    BYTE *ptr = NULL;
    UINT32 frames = 0;
    DWORD flags = 0;

    // just read until we stop getting packets, throwing them away.
    while (!SDL_GetAtomicInt(&device->shutdown) && device->hidden->capture) {
        const HRESULT ret = IAudioCaptureClient_GetBuffer(device->hidden->capture, &ptr, &frames, &flags, NULL, NULL);
        if (ret == AUDCLNT_S_BUFFER_EMPTY) {
            break; // no more buffered data; we're done.
        } else if (WasapiFailed(device, ret)) {
            break; // failed for some other reason, abort.
        } else if (WasapiFailed(device, IAudioCaptureClient_ReleaseBuffer(device->hidden->capture, frames))) {
            break; // something broke.
        }
    }
}

static void WASAPI_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        ResetWasapiDevice(device);
        SDL_free(device->hidden->devid);
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

static bool mgmtthrtask_PrepDevice(void *userdata)
{
    SDL_AudioDevice *device = (SDL_AudioDevice *)userdata;

    /* !!! FIXME: we could request an exclusive mode stream, which is lower latency;
       !!!  it will write into the kernel's audio buffer directly instead of
       !!!  shared memory that a user-mode mixer then writes to the kernel with
       !!!  everything else. Doing this means any other sound using this device will
       !!!  stop playing, including the user's MP3 player and system notification
       !!!  sounds. You'd probably need to release the device when the app isn't in
       !!!  the foreground, to be a good citizen of the system. It's doable, but it's
       !!!  more work and causes some annoyances, and I don't know what the latency
       !!!  wins actually look like. Maybe add a hint to force exclusive mode at
       !!!  some point. To be sure, defaulting to shared mode is the right thing to
       !!!  do in any case. */
    const AUDCLNT_SHAREMODE sharemode = AUDCLNT_SHAREMODE_SHARED;

    IAudioClient *client = device->hidden->client;
    SDL_assert(client != NULL);

    device->hidden->event = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!device->hidden->event) {
        return WIN_SetError("WASAPI can't create an event handle");
    }

    HRESULT ret;

    WAVEFORMATEX *waveformat = NULL;
    ret = IAudioClient_GetMixFormat(client, &waveformat);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't determine mix format", ret);
    }
    SDL_assert(waveformat != NULL);
    device->hidden->waveformat = waveformat;

    SDL_AudioSpec newspec;
    newspec.channels = (Uint8)waveformat->nChannels;

    // Make sure we have a valid format that we can convert to whatever WASAPI wants.
    const SDL_AudioFormat wasapi_format = SDL_WaveFormatExToSDLFormat(waveformat);

    SDL_AudioFormat test_format;
    const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
        if (test_format == wasapi_format) {
            newspec.format = test_format;
            break;
        }
    }

    if (!test_format) {
        return SDL_SetError("%s: Unsupported audio format", "wasapi");
    }

    REFERENCE_TIME default_period = 0;
    ret = IAudioClient_GetDevicePeriod(client, &default_period, NULL);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't determine minimum device period", ret);
    }

    DWORD streamflags = 0;

    /* we've gotten reports that WASAPI's resampler introduces distortions, but in the short term
       it fixes some other WASAPI-specific quirks we haven't quite tracked down.
       Refer to bug #6326 for the immediate concern. */
#if 1
    // favor WASAPI's resampler over our own
    if ((DWORD)device->spec.freq != waveformat->nSamplesPerSec) {
        streamflags |= (AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY);
        waveformat->nSamplesPerSec = device->spec.freq;
        waveformat->nAvgBytesPerSec = waveformat->nSamplesPerSec * waveformat->nChannels * (waveformat->wBitsPerSample / 8);
    }
#endif

    newspec.freq = waveformat->nSamplesPerSec;

    streamflags |= AUDCLNT_STREAMFLAGS_EVENTCALLBACK;

    int new_sample_frames = 0;
    bool iaudioclient3_initialized = false;

#ifdef __IAudioClient3_INTERFACE_DEFINED__
    // Try querying IAudioClient3 if sharemode is AUDCLNT_SHAREMODE_SHARED
    if (sharemode == AUDCLNT_SHAREMODE_SHARED) {
        IAudioClient3 *client3 = NULL;
        ret = IAudioClient_QueryInterface(client, &SDL_IID_IAudioClient3, (void**)&client3);
        if (SUCCEEDED(ret)) {
            UINT32 default_period_in_frames = 0;
            UINT32 fundamental_period_in_frames = 0;
            UINT32 min_period_in_frames = 0;
            UINT32 max_period_in_frames = 0;
            ret = IAudioClient3_GetSharedModeEnginePeriod(client3, waveformat,
                                                          &default_period_in_frames, &fundamental_period_in_frames, &min_period_in_frames, &max_period_in_frames);
            if (SUCCEEDED(ret)) {
                // IAudioClient3_InitializeSharedAudioStream only accepts the integral multiple of fundamental_period_in_frames
                UINT32 period_in_frames = fundamental_period_in_frames * (UINT32)SDL_round((double)device->sample_frames / fundamental_period_in_frames);
                period_in_frames = SDL_clamp(period_in_frames, min_period_in_frames, max_period_in_frames);

                ret = IAudioClient3_InitializeSharedAudioStream(client3, streamflags, period_in_frames, waveformat, NULL);
                if (SUCCEEDED(ret)) {
                    new_sample_frames = (int)period_in_frames;
                    iaudioclient3_initialized = true;
                }
            }

            IAudioClient3_Release(client3);
        }
    }
#endif

    if (!iaudioclient3_initialized)
        ret = IAudioClient_Initialize(client, sharemode, streamflags, 0, 0, waveformat, NULL);

    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't initialize audio client", ret);
    }

    ret = IAudioClient_SetEventHandle(client, device->hidden->event);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't set event handle", ret);
    }

    UINT32 bufsize = 0; // this is in sample frames, not samples, not bytes.
    ret = IAudioClient_GetBufferSize(client, &bufsize);
    if (FAILED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't determine buffer size", ret);
    }

    // Match the callback size to the period size to cut down on the number of
    // interrupts waited for in each call to WaitDevice
    if (new_sample_frames <= 0) {
        const float period_millis = default_period / 10000.0f;
        const float period_frames = period_millis * newspec.freq / 1000.0f;
        new_sample_frames = (int) SDL_ceilf(period_frames);
    }

    // regardless of what we calculated for the period size, clamp it to the expected hardware buffer size.
    if (new_sample_frames > (int) bufsize) {
        new_sample_frames = (int) bufsize;
    }

    // Update the fragment size as size in bytes
    if (!SDL_AudioDeviceFormatChangedAlreadyLocked(device, &newspec, new_sample_frames)) {
        return false;
    }

    device->hidden->framesize = SDL_AUDIO_FRAMESIZE(device->spec);

    if (device->recording) {
        IAudioCaptureClient *capture = NULL;
        ret = IAudioClient_GetService(client, &SDL_IID_IAudioCaptureClient, (void **)&capture);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't get capture client service", ret);
        }

        SDL_assert(capture != NULL);
        device->hidden->capture = capture;
        ret = IAudioClient_Start(client);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't start capture", ret);
        }

        WASAPI_FlushRecording(device); // MSDN says you should flush the recording endpoint right after startup.
    } else {
        IAudioRenderClient *render = NULL;
        ret = IAudioClient_GetService(client, &SDL_IID_IAudioRenderClient, (void **)&render);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't get render client service", ret);
        }

        SDL_assert(render != NULL);
        device->hidden->render = render;
        ret = IAudioClient_Start(client);
        if (FAILED(ret)) {
            return WIN_SetErrorFromHRESULT("WASAPI can't start playback", ret);
        }
    }

    return true; // good to go.
}

// This is called once a device is activated, possibly asynchronously.
bool WASAPI_PrepDevice(SDL_AudioDevice *device)
{
    bool rc = true;
    return (WASAPI_ProxyToManagementThread(mgmtthrtask_PrepDevice, device, &rc) && rc);
}

static bool WASAPI_OpenDevice(SDL_AudioDevice *device)
{
    // Initialize all variables that we clean on shutdown
    device->hidden = (struct SDL_PrivateAudioData *) SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    } else if (!ActivateWasapiDevice(device)) {
        return false; // already set error.
    }

    /* Ready, but possibly waiting for async device activation.
       Until activation is successful, we will report silence from recording
       devices and ignore data on playback devices. Upon activation, we'll make
       sure any bound audio streams are adjusted for the final device format. */

    return true;
}

static void WASAPI_ThreadInit(SDL_AudioDevice *device)
{
    // this thread uses COM.
    if (SUCCEEDED(WIN_CoInitialize())) { // can't report errors, hope it worked!
        device->hidden->coinitialized = true;
    }

    // Set this thread to very high "Pro Audio" priority.
    if (pAvSetMmThreadCharacteristicsW) {
        DWORD idx = 0;
        device->hidden->task = pAvSetMmThreadCharacteristicsW(L"Pro Audio", &idx);
    } else {
        SDL_SetCurrentThreadPriority(device->recording ? SDL_THREAD_PRIORITY_HIGH : SDL_THREAD_PRIORITY_TIME_CRITICAL);
    }
}

static void WASAPI_ThreadDeinit(SDL_AudioDevice *device)
{
    // Set this thread back to normal priority.
    if (device->hidden->task && pAvRevertMmThreadCharacteristics) {
        pAvRevertMmThreadCharacteristics(device->hidden->task);
        device->hidden->task = NULL;
    }

    if (device->hidden->coinitialized) {
        WIN_CoUninitialize();
        device->hidden->coinitialized = false;
    }
}

static bool mgmtthrtask_FreeDeviceHandle(void *userdata)
{
    SDL_IMMDevice_FreeDeviceHandle((SDL_AudioDevice *) userdata);
    return true;
}

static void WASAPI_FreeDeviceHandle(SDL_AudioDevice *device)
{
    bool rc;
    WASAPI_ProxyToManagementThread(mgmtthrtask_FreeDeviceHandle, device, &rc);
}

static bool mgmtthrtask_DeinitializeStart(void *userdata)
{
    StopWasapiHotplug();
    return true;
}

static void WASAPI_DeinitializeStart(void)
{
    bool rc;
    WASAPI_ProxyToManagementThread(mgmtthrtask_DeinitializeStart, NULL, &rc);
}

static void WASAPI_Deinitialize(void)
{
    DeinitManagementThread();
}

static bool WASAPI_Init(SDL_AudioDriverImpl *impl)
{
    if (!InitManagementThread()) {
        return false;
    }

    impl->DetectDevices = WASAPI_DetectDevices;
    impl->ThreadInit = WASAPI_ThreadInit;
    impl->ThreadDeinit = WASAPI_ThreadDeinit;
    impl->OpenDevice = WASAPI_OpenDevice;
    impl->PlayDevice = WASAPI_PlayDevice;
    impl->WaitDevice = WASAPI_WaitDevice;
    impl->GetDeviceBuf = WASAPI_GetDeviceBuf;
    impl->WaitRecordingDevice = WASAPI_WaitDevice;
    impl->RecordDevice = WASAPI_RecordDevice;
    impl->FlushRecording = WASAPI_FlushRecording;
    impl->CloseDevice = WASAPI_CloseDevice;
    impl->DeinitializeStart = WASAPI_DeinitializeStart;
    impl->Deinitialize = WASAPI_Deinitialize;
    impl->FreeDeviceHandle = WASAPI_FreeDeviceHandle;

    impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap WASAPI_bootstrap = {
    "wasapi", "WASAPI", WASAPI_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_WASAPI
