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

#if defined(SDL_PLATFORM_WINDOWS) && defined(HAVE_MMDEVICEAPI_H)

#include "SDL_windows.h"
#include "SDL_immdevice.h"
#include "../../audio/SDL_sysaudio.h"
#include <objbase.h> // For CLSIDFromString

typedef struct SDL_IMMDevice_HandleData
{
    LPWSTR immdevice_id;
    GUID directsound_guid;
} SDL_IMMDevice_HandleData;

static const ERole SDL_IMMDevice_role = eConsole; // !!! FIXME: should this be eMultimedia? Should be a hint?

// This is global to the WASAPI target, to handle hotplug and default device lookup.
static IMMDeviceEnumerator *enumerator = NULL;
static SDL_IMMDevice_callbacks immcallbacks;

// PropVariantInit() is an inline function/macro in PropIdl.h that calls the C runtime's memset() directly. Use ours instead, to avoid dependency.
#ifdef PropVariantInit
#undef PropVariantInit
#endif
#define PropVariantInit(p) SDL_zerop(p)

// Some GUIDs we need to know without linking to libraries that aren't available before Vista.
/* *INDENT-OFF* */ // clang-format off
static const CLSID SDL_CLSID_MMDeviceEnumerator = { 0xbcde0395, 0xe52f, 0x467c,{ 0x8e, 0x3d, 0xc4, 0x57, 0x92, 0x91, 0x69, 0x2e } };
static const IID SDL_IID_IMMDeviceEnumerator = { 0xa95664d2, 0x9614, 0x4f35,{ 0xa7, 0x46, 0xde, 0x8d, 0xb6, 0x36, 0x17, 0xe6 } };
static const IID SDL_IID_IMMNotificationClient = { 0x7991eec9, 0x7e89, 0x4d85,{ 0x83, 0x90, 0x6c, 0x70, 0x3c, 0xec, 0x60, 0xc0 } };
static const IID SDL_IID_IMMEndpoint = { 0x1be09788, 0x6894, 0x4089,{ 0x85, 0x86, 0x9a, 0x2a, 0x6c, 0x26, 0x5a, 0xc5 } };
static const PROPERTYKEY SDL_PKEY_Device_FriendlyName = { { 0xa45c254e, 0xdf1c, 0x4efd,{ 0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0, } }, 14 };
static const PROPERTYKEY SDL_PKEY_AudioEngine_DeviceFormat = { { 0xf19f064d, 0x82c, 0x4e27,{ 0xbc, 0x73, 0x68, 0x82, 0xa1, 0xbb, 0x8e, 0x4c, } }, 0 };
static const PROPERTYKEY SDL_PKEY_AudioEndpoint_GUID = { { 0x1da5d803, 0xd492, 0x4edd,{ 0x8c, 0x23, 0xe0, 0xc0, 0xff, 0xee, 0x7f, 0x0e, } }, 4 };
/* *INDENT-ON* */ // clang-format on

static bool FindByDevIDCallback(SDL_AudioDevice *device, void *userdata)
{
    LPCWSTR devid = (LPCWSTR)userdata;
    if (devid && device && device->handle) {
        const SDL_IMMDevice_HandleData *handle = (const SDL_IMMDevice_HandleData *)device->handle;
        if (handle->immdevice_id && SDL_wcscmp(handle->immdevice_id, devid) == 0) {
            return true;
        }
    }
    return false;
}

static SDL_AudioDevice *SDL_IMMDevice_FindByDevID(LPCWSTR devid)
{
    return SDL_FindPhysicalAudioDeviceByCallback(FindByDevIDCallback, (void *) devid);
}

LPGUID SDL_IMMDevice_GetDirectSoundGUID(SDL_AudioDevice *device)
{
    return (device && device->handle) ? &(((SDL_IMMDevice_HandleData *) device->handle)->directsound_guid) : NULL;
}

LPCWSTR SDL_IMMDevice_GetDevID(SDL_AudioDevice *device)
{
    return (device && device->handle) ? ((const SDL_IMMDevice_HandleData *) device->handle)->immdevice_id : NULL;
}

static void GetMMDeviceInfo(IMMDevice *device, char **utf8dev, WAVEFORMATEXTENSIBLE *fmt, GUID *guid)
{
    /* PKEY_Device_FriendlyName gives you "Speakers (SoundBlaster Pro)" which drives me nuts. I'd rather it be
       "SoundBlaster Pro (Speakers)" but I guess that's developers vs users. Windows uses the FriendlyName in
       its own UIs, like Volume Control, etc. */
    IPropertyStore *props = NULL;
    *utf8dev = NULL;
    SDL_zerop(fmt);
    if (SUCCEEDED(IMMDevice_OpenPropertyStore(device, STGM_READ, &props))) {
        PROPVARIANT var;
        PropVariantInit(&var);
        if (SUCCEEDED(IPropertyStore_GetValue(props, &SDL_PKEY_Device_FriendlyName, &var))) {
            *utf8dev = WIN_StringToUTF8W(var.pwszVal);
        }
        PropVariantClear(&var);
        if (SUCCEEDED(IPropertyStore_GetValue(props, &SDL_PKEY_AudioEngine_DeviceFormat, &var))) {
            SDL_memcpy(fmt, var.blob.pBlobData, SDL_min(var.blob.cbSize, sizeof(WAVEFORMATEXTENSIBLE)));
        }
        PropVariantClear(&var);
        if (SUCCEEDED(IPropertyStore_GetValue(props, &SDL_PKEY_AudioEndpoint_GUID, &var))) {
            (void)CLSIDFromString(var.pwszVal, guid);
        }
        PropVariantClear(&var);
        IPropertyStore_Release(props);
    }
}

void SDL_IMMDevice_FreeDeviceHandle(SDL_AudioDevice *device)
{
    if (device && device->handle) {
        SDL_IMMDevice_HandleData *handle = (SDL_IMMDevice_HandleData *) device->handle;
        SDL_free(handle->immdevice_id);
        SDL_free(handle);
        device->handle = NULL;
    }
}

static SDL_AudioDevice *SDL_IMMDevice_Add(const bool recording, const char *devname, WAVEFORMATEXTENSIBLE *fmt, LPCWSTR devid, GUID *dsoundguid)
{
    /* You can have multiple endpoints on a device that are mutually exclusive ("Speakers" vs "Line Out" or whatever).
       In a perfect world, things that are unplugged won't be in this collection. The only gotcha is probably for
       phones and tablets, where you might have an internal speaker and a headphone jack and expect both to be
       available and switch automatically. (!!! FIXME...?) */

    if (!devname) {
        return NULL;
    }

    // see if we already have this one first.
    SDL_AudioDevice *device = SDL_IMMDevice_FindByDevID(devid);
    if (device) {
        if (SDL_GetAtomicInt(&device->zombie)) {
            // whoa, it came back! This can happen if you unplug and replug USB headphones while we're still keeping the SDL object alive.
            // Kill this device's IMMDevice id; the device will go away when the app closes it, or maybe a new default device is chosen
            // (possibly this reconnected device), so we just want to make sure IMMDevice doesn't try to find the old device by the existing ID string.
            SDL_IMMDevice_HandleData *handle = (SDL_IMMDevice_HandleData *) device->handle;
            SDL_free(handle->immdevice_id);
            handle->immdevice_id = NULL;
            device = NULL;  // add a new device, below.
        }
    }

    if (!device) {
        // handle is freed by SDL_IMMDevice_FreeDeviceHandle!
        SDL_IMMDevice_HandleData *handle = (SDL_IMMDevice_HandleData *)SDL_malloc(sizeof(SDL_IMMDevice_HandleData));
        if (!handle) {
            return NULL;
        }
        handle->immdevice_id = SDL_wcsdup(devid);
        if (!handle->immdevice_id) {
            SDL_free(handle);
            return NULL;
        }
        SDL_memcpy(&handle->directsound_guid, dsoundguid, sizeof(GUID));

        SDL_AudioSpec spec;
        SDL_zero(spec);
        spec.channels = (Uint8)fmt->Format.nChannels;
        spec.freq = fmt->Format.nSamplesPerSec;
        spec.format = SDL_WaveFormatExToSDLFormat((WAVEFORMATEX *)fmt);

        device = SDL_AddAudioDevice(recording, devname, &spec, handle);
        if (!device) {
            SDL_free(handle->immdevice_id);
            SDL_free(handle);
        }
    }

    return device;
}

/* We need a COM subclass of IMMNotificationClient for hotplug support, which is
   easy in C++, but we have to tapdance more to make work in C.
   Thanks to this page for coaching on how to make this work:
     https://www.codeproject.com/Articles/13601/COM-in-plain-C */

typedef struct SDLMMNotificationClient
{
    const IMMNotificationClientVtbl *lpVtbl;
    SDL_AtomicInt refcount;
} SDLMMNotificationClient;

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_QueryInterface(IMMNotificationClient *client, REFIID iid, void **ppv)
{
    if ((WIN_IsEqualIID(iid, &IID_IUnknown)) || (WIN_IsEqualIID(iid, &SDL_IID_IMMNotificationClient))) {
        *ppv = client;
        client->lpVtbl->AddRef(client);
        return S_OK;
    }

    *ppv = NULL;
    return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE SDLMMNotificationClient_AddRef(IMMNotificationClient *iclient)
{
    SDLMMNotificationClient *client = (SDLMMNotificationClient *)iclient;
    return (ULONG)(SDL_AtomicIncRef(&client->refcount) + 1);
}

static ULONG STDMETHODCALLTYPE SDLMMNotificationClient_Release(IMMNotificationClient *iclient)
{
    // client is a static object; we don't ever free it.
    SDLMMNotificationClient *client = (SDLMMNotificationClient *)iclient;
    const ULONG rc = SDL_AtomicDecRef(&client->refcount);
    if (rc == 0) {
        SDL_SetAtomicInt(&client->refcount, 0); // uhh...
        return 0;
    }
    return rc - 1;
}

// These are the entry points called when WASAPI device endpoints change.
static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDefaultDeviceChanged(IMMNotificationClient *iclient, EDataFlow flow, ERole role, LPCWSTR pwstrDeviceId)
{
    if (role == SDL_IMMDevice_role) {
        immcallbacks.default_audio_device_changed(SDL_IMMDevice_FindByDevID(pwstrDeviceId));
    }
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDeviceAdded(IMMNotificationClient *iclient, LPCWSTR pwstrDeviceId)
{
    /* we ignore this; devices added here then progress to ACTIVE, if appropriate, in
       OnDeviceStateChange, making that a better place to deal with device adds. More
       importantly: the first time you plug in a USB audio device, this callback will
       fire, but when you unplug it, it isn't removed (it's state changes to NOTPRESENT).
       Plugging it back in won't fire this callback again. */
    return S_OK;
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDeviceRemoved(IMMNotificationClient *iclient, LPCWSTR pwstrDeviceId)
{
    return S_OK;  // See notes in OnDeviceAdded handler about why we ignore this.
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnDeviceStateChanged(IMMNotificationClient *iclient, LPCWSTR pwstrDeviceId, DWORD dwNewState)
{
    IMMDevice *device = NULL;

    if (SUCCEEDED(IMMDeviceEnumerator_GetDevice(enumerator, pwstrDeviceId, &device))) {
        IMMEndpoint *endpoint = NULL;
        if (SUCCEEDED(IMMDevice_QueryInterface(device, &SDL_IID_IMMEndpoint, (void **)&endpoint))) {
            EDataFlow flow;
            if (SUCCEEDED(IMMEndpoint_GetDataFlow(endpoint, &flow))) {
                const bool recording = (flow == eCapture);
                if (dwNewState == DEVICE_STATE_ACTIVE) {
                    char *utf8dev;
                    WAVEFORMATEXTENSIBLE fmt;
                    GUID dsoundguid;
                    GetMMDeviceInfo(device, &utf8dev, &fmt, &dsoundguid);
                    if (utf8dev) {
                        SDL_IMMDevice_Add(recording, utf8dev, &fmt, pwstrDeviceId, &dsoundguid);
                        SDL_free(utf8dev);
                    }
                } else {
                    immcallbacks.audio_device_disconnected(SDL_IMMDevice_FindByDevID(pwstrDeviceId));
                }
            }
            IMMEndpoint_Release(endpoint);
        }
        IMMDevice_Release(device);
    }

    return S_OK;
}

static HRESULT STDMETHODCALLTYPE SDLMMNotificationClient_OnPropertyValueChanged(IMMNotificationClient *client, LPCWSTR pwstrDeviceId, const PROPERTYKEY key)
{
    return S_OK; // we don't care about these.
}

static const IMMNotificationClientVtbl notification_client_vtbl = {
    SDLMMNotificationClient_QueryInterface,
    SDLMMNotificationClient_AddRef,
    SDLMMNotificationClient_Release,
    SDLMMNotificationClient_OnDeviceStateChanged,
    SDLMMNotificationClient_OnDeviceAdded,
    SDLMMNotificationClient_OnDeviceRemoved,
    SDLMMNotificationClient_OnDefaultDeviceChanged,
    SDLMMNotificationClient_OnPropertyValueChanged
};

static SDLMMNotificationClient notification_client = { &notification_client_vtbl, { 1 } };

bool SDL_IMMDevice_Init(const SDL_IMMDevice_callbacks *callbacks)
{
    HRESULT ret;

    // just skip the discussion with COM here.
    if (!WIN_IsWindowsVistaOrGreater()) {
        return SDL_SetError("IMMDevice support requires Windows Vista or later");
    }

    if (FAILED(WIN_CoInitialize())) {
        return SDL_SetError("IMMDevice: CoInitialize() failed");
    }

    ret = CoCreateInstance(&SDL_CLSID_MMDeviceEnumerator, NULL, CLSCTX_INPROC_SERVER, &SDL_IID_IMMDeviceEnumerator, (LPVOID *)&enumerator);
    if (FAILED(ret)) {
        WIN_CoUninitialize();
        return WIN_SetErrorFromHRESULT("IMMDevice CoCreateInstance(MMDeviceEnumerator)", ret);
    }

    if (callbacks) {
        SDL_copyp(&immcallbacks, callbacks);
    } else {
        SDL_zero(immcallbacks);
    }

    if (!immcallbacks.audio_device_disconnected) {
        immcallbacks.audio_device_disconnected = SDL_AudioDeviceDisconnected;
    }
    if (!immcallbacks.default_audio_device_changed) {
        immcallbacks.default_audio_device_changed = SDL_DefaultAudioDeviceChanged;
    }

    return true;
}

void SDL_IMMDevice_Quit(void)
{
    if (enumerator) {
        IMMDeviceEnumerator_UnregisterEndpointNotificationCallback(enumerator, (IMMNotificationClient *)&notification_client);
        IMMDeviceEnumerator_Release(enumerator);
        enumerator = NULL;
    }

    SDL_zero(immcallbacks);

    WIN_CoUninitialize();
}

bool SDL_IMMDevice_Get(SDL_AudioDevice *device, IMMDevice **immdevice, bool recording)
{
    const Uint64 timeout = SDL_GetTicks() + 8000;  // intel's audio drivers can fail for up to EIGHT SECONDS after a device is connected or we wake from sleep.

    SDL_assert(device != NULL);
    SDL_assert(immdevice != NULL);

    LPCWSTR devid = SDL_IMMDevice_GetDevID(device);
    SDL_assert(devid != NULL);

    HRESULT ret;
    while ((ret = IMMDeviceEnumerator_GetDevice(enumerator, devid, immdevice)) == E_NOTFOUND) {
        const Uint64 now = SDL_GetTicks();
        if (timeout > now) {
            const Uint64 ticksleft = timeout - now;
            SDL_Delay((Uint32)SDL_min(ticksleft, 300));   // wait awhile and try again.
            continue;
        }
        break;
    }

    if (!SUCCEEDED(ret)) {
        return WIN_SetErrorFromHRESULT("WASAPI can't find requested audio endpoint", ret);
    }
    return true;
}

static void EnumerateEndpointsForFlow(const bool recording, SDL_AudioDevice **default_device)
{
    /* Note that WASAPI separates "adapter devices" from "audio endpoint devices"
       ...one adapter device ("SoundBlaster Pro") might have multiple endpoint devices ("Speakers", "Line-Out"). */

    IMMDeviceCollection *collection = NULL;
    if (FAILED(IMMDeviceEnumerator_EnumAudioEndpoints(enumerator, recording ? eCapture : eRender, DEVICE_STATE_ACTIVE, &collection))) {
        return;
    }

    UINT total = 0;
    if (FAILED(IMMDeviceCollection_GetCount(collection, &total))) {
        IMMDeviceCollection_Release(collection);
        return;
    }

    LPWSTR default_devid = NULL;
    if (default_device) {
        IMMDevice *default_immdevice = NULL;
        const EDataFlow dataflow = recording ? eCapture : eRender;
        if (SUCCEEDED(IMMDeviceEnumerator_GetDefaultAudioEndpoint(enumerator, dataflow, SDL_IMMDevice_role, &default_immdevice))) {
            LPWSTR devid = NULL;
            if (SUCCEEDED(IMMDevice_GetId(default_immdevice, &devid))) {
                default_devid = SDL_wcsdup(devid);  // if this fails, oh well.
                CoTaskMemFree(devid);
            }
            IMMDevice_Release(default_immdevice);
        }
    }

    for (UINT i = 0; i < total; i++) {
        IMMDevice *immdevice = NULL;
        if (SUCCEEDED(IMMDeviceCollection_Item(collection, i, &immdevice))) {
            LPWSTR devid = NULL;
            if (SUCCEEDED(IMMDevice_GetId(immdevice, &devid))) {
                char *devname = NULL;
                WAVEFORMATEXTENSIBLE fmt;
                GUID dsoundguid;
                SDL_zero(fmt);
                SDL_zero(dsoundguid);
                GetMMDeviceInfo(immdevice, &devname, &fmt, &dsoundguid);
                if (devname) {
                    SDL_AudioDevice *sdldevice = SDL_IMMDevice_Add(recording, devname, &fmt, devid, &dsoundguid);
                    if (default_device && default_devid && SDL_wcscmp(default_devid, devid) == 0) {
                        *default_device = sdldevice;
                    }
                    SDL_free(devname);
                }
                CoTaskMemFree(devid);
            }
            IMMDevice_Release(immdevice);
        }
    }

    SDL_free(default_devid);

    IMMDeviceCollection_Release(collection);
}

void SDL_IMMDevice_EnumerateEndpoints(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
    EnumerateEndpointsForFlow(false, default_playback);
    EnumerateEndpointsForFlow(true, default_recording);

    // if this fails, we just won't get hotplug events. Carry on anyhow.
    IMMDeviceEnumerator_RegisterEndpointNotificationCallback(enumerator, (IMMNotificationClient *)&notification_client);
}

#endif // defined(SDL_PLATFORM_WINDOWS) && defined(HAVE_MMDEVICEAPI_H)
