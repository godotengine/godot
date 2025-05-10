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

#ifdef SDL_AUDIO_DRIVER_DSOUND

#include "../SDL_sysaudio.h"
#include "SDL_directsound.h"
#include <mmreg.h>
#ifdef HAVE_MMDEVICEAPI_H
#include "../../core/windows/SDL_immdevice.h"
#endif

#ifndef WAVE_FORMAT_IEEE_FLOAT
#define WAVE_FORMAT_IEEE_FLOAT 0x0003
#endif

// For Vista+, we can enumerate DSound devices with IMMDevice
#ifdef HAVE_MMDEVICEAPI_H
static bool SupportsIMMDevice = false;
#endif

// DirectX function pointers for audio
static SDL_SharedObject *DSoundDLL = NULL;
typedef HRESULT(WINAPI *fnDirectSoundCreate8)(LPGUID, LPDIRECTSOUND *, LPUNKNOWN);
typedef HRESULT(WINAPI *fnDirectSoundEnumerateW)(LPDSENUMCALLBACKW, LPVOID);
typedef HRESULT(WINAPI *fnDirectSoundCaptureCreate8)(LPCGUID, LPDIRECTSOUNDCAPTURE8 *, LPUNKNOWN);
typedef HRESULT(WINAPI *fnDirectSoundCaptureEnumerateW)(LPDSENUMCALLBACKW, LPVOID);
typedef HRESULT(WINAPI *fnGetDeviceID)(LPCGUID, LPGUID);
static fnDirectSoundCreate8 pDirectSoundCreate8 = NULL;
static fnDirectSoundEnumerateW pDirectSoundEnumerateW = NULL;
static fnDirectSoundCaptureCreate8 pDirectSoundCaptureCreate8 = NULL;
static fnDirectSoundCaptureEnumerateW pDirectSoundCaptureEnumerateW = NULL;
static fnGetDeviceID pGetDeviceID = NULL;

#include <initguid.h>
DEFINE_GUID(SDL_DSDEVID_DefaultPlayback, 0xdef00000, 0x9c6d, 0x47ed, 0xaa, 0xf1, 0x4d, 0xda, 0x8f, 0x2b, 0x5c, 0x03);
DEFINE_GUID(SDL_DSDEVID_DefaultCapture, 0xdef00001, 0x9c6d, 0x47ed, 0xaa, 0xf1, 0x4d, 0xda, 0x8f, 0x2b, 0x5c, 0x03);

static const GUID SDL_KSDATAFORMAT_SUBTYPE_PCM = { 0x00000001, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };
static const GUID SDL_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = { 0x00000003, 0x0000, 0x0010, { 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71 } };

static void DSOUND_Unload(void)
{
    pDirectSoundCreate8 = NULL;
    pDirectSoundEnumerateW = NULL;
    pDirectSoundCaptureCreate8 = NULL;
    pDirectSoundCaptureEnumerateW = NULL;
    pGetDeviceID = NULL;

    if (DSoundDLL) {
        SDL_UnloadObject(DSoundDLL);
        DSoundDLL = NULL;
    }
}

static bool DSOUND_Load(void)
{
    bool loaded = false;

    DSOUND_Unload();

    DSoundDLL = SDL_LoadObject("DSOUND.DLL");
    if (!DSoundDLL) {
        SDL_SetError("DirectSound: failed to load DSOUND.DLL");
    } else {
// Now make sure we have DirectX 8 or better...
#define DSOUNDLOAD(f)                                  \
    {                                                  \
        p##f = (fn##f)SDL_LoadFunction(DSoundDLL, #f); \
        if (!p##f)                                     \
            loaded = false;                                \
    }
        loaded = true; // will reset if necessary.
        DSOUNDLOAD(DirectSoundCreate8);
        DSOUNDLOAD(DirectSoundEnumerateW);
        DSOUNDLOAD(DirectSoundCaptureCreate8);
        DSOUNDLOAD(DirectSoundCaptureEnumerateW);
        DSOUNDLOAD(GetDeviceID);
#undef DSOUNDLOAD

        if (!loaded) {
            SDL_SetError("DirectSound: System doesn't appear to have DX8.");
        }
    }

    if (!loaded) {
        DSOUND_Unload();
    }

    return loaded;
}

static bool SetDSerror(const char *function, int code)
{
    const char *error;

    switch (code) {
    case E_NOINTERFACE:
        error = "Unsupported interface -- Is DirectX 8.0 or later installed?";
        break;
    case DSERR_ALLOCATED:
        error = "Audio device in use";
        break;
    case DSERR_BADFORMAT:
        error = "Unsupported audio format";
        break;
    case DSERR_BUFFERLOST:
        error = "Mixing buffer was lost";
        break;
    case DSERR_CONTROLUNAVAIL:
        error = "Control requested is not available";
        break;
    case DSERR_INVALIDCALL:
        error = "Invalid call for the current state";
        break;
    case DSERR_INVALIDPARAM:
        error = "Invalid parameter";
        break;
    case DSERR_NODRIVER:
        error = "No audio device found";
        break;
    case DSERR_OUTOFMEMORY:
        error = "Out of memory";
        break;
    case DSERR_PRIOLEVELNEEDED:
        error = "Caller doesn't have priority";
        break;
    case DSERR_UNSUPPORTED:
        error = "Function not supported";
        break;
    default:
        error = "Unknown DirectSound error";
        break;
    }

    return SDL_SetError("%s: %s (0x%x)", function, error, code);
}

static void DSOUND_FreeDeviceHandle(SDL_AudioDevice *device)
{
#ifdef HAVE_MMDEVICEAPI_H
    if (SupportsIMMDevice) {
        SDL_IMMDevice_FreeDeviceHandle(device);
    } else
#endif
    {
        SDL_free(device->handle);
    }
}

// FindAllDevs is presumably only used on WinXP; Vista and later can use IMMDevice for better results.
typedef struct FindAllDevsData
{
    bool recording;
    SDL_AudioDevice **default_device;
    LPCGUID default_device_guid;
} FindAllDevsData;

static BOOL CALLBACK FindAllDevs(LPGUID guid, LPCWSTR desc, LPCWSTR module, LPVOID userdata)
{
    FindAllDevsData *data = (FindAllDevsData *) userdata;
    if (guid != NULL) { // skip default device
        char *str = WIN_LookupAudioDeviceName(desc, guid);
        if (str) {
            LPGUID cpyguid = (LPGUID)SDL_malloc(sizeof(GUID));
            if (cpyguid) {
                SDL_copyp(cpyguid, guid);

                /* Note that spec is NULL, because we are required to connect to the
                 * device before getting the channel mask and output format, making
                 * this information inaccessible at enumeration time
                 */
                SDL_AudioDevice *device = SDL_AddAudioDevice(data->recording, str, NULL, cpyguid);
                if (device && data->default_device && data->default_device_guid) {
                    if (SDL_memcmp(cpyguid, data->default_device_guid, sizeof (GUID)) == 0) {
                        *data->default_device = device;
                    }
                }
            }
            SDL_free(str); // SDL_AddAudioDevice() makes a copy of this string.
        }
    }
    return TRUE; // keep enumerating.
}

static void DSOUND_DetectDevices(SDL_AudioDevice **default_playback, SDL_AudioDevice **default_recording)
{
#ifdef HAVE_MMDEVICEAPI_H
    if (SupportsIMMDevice) {
        SDL_IMMDevice_EnumerateEndpoints(default_playback, default_recording);
    } else
#endif
    {
        // Without IMMDevice, you can enumerate devices and figure out the default devices,
        //  but you won't get device hotplug or default device change notifications. But this is
        //  only for WinXP; Windows Vista and later should be using IMMDevice.
        FindAllDevsData data;
        GUID guid;

        data.recording = true;
        data.default_device = default_recording;
        data.default_device_guid = (pGetDeviceID(&SDL_DSDEVID_DefaultCapture, &guid) == DS_OK) ? &guid : NULL;
        pDirectSoundCaptureEnumerateW(FindAllDevs, &data);

        data.recording = false;
        data.default_device = default_playback;
        data.default_device_guid = (pGetDeviceID(&SDL_DSDEVID_DefaultPlayback, &guid) == DS_OK) ? &guid : NULL;
        pDirectSoundEnumerateW(FindAllDevs, &data);
    }

}

static bool DSOUND_WaitDevice(SDL_AudioDevice *device)
{
    /* Semi-busy wait, since we have no way of getting play notification
       on a primary mixing buffer located in hardware (DirectX 5.0)
     */
    while (!SDL_GetAtomicInt(&device->shutdown)) {
        DWORD status = 0;
        DWORD cursor = 0;
        DWORD junk = 0;
        HRESULT result = DS_OK;

        // Try to restore a lost sound buffer
        IDirectSoundBuffer_GetStatus(device->hidden->mixbuf, &status);
        if (status & DSBSTATUS_BUFFERLOST) {
            IDirectSoundBuffer_Restore(device->hidden->mixbuf);
        } else if (!(status & DSBSTATUS_PLAYING)) {
            result = IDirectSoundBuffer_Play(device->hidden->mixbuf, 0, 0, DSBPLAY_LOOPING);
        } else {
            // Find out where we are playing
            result = IDirectSoundBuffer_GetCurrentPosition(device->hidden->mixbuf, &junk, &cursor);
            if ((result == DS_OK) && ((cursor / device->buffer_size) != device->hidden->lastchunk)) {
                break;  // ready for next chunk!
            }
        }

        if ((result != DS_OK) && (result != DSERR_BUFFERLOST)) {
            return false;
        }

        SDL_Delay(1);  // not ready yet; sleep a bit.
    }

    return true;
}

static bool DSOUND_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    // Unlock the buffer, allowing it to play
    SDL_assert(buflen == device->buffer_size);
    if (IDirectSoundBuffer_Unlock(device->hidden->mixbuf, (LPVOID) buffer, buflen, NULL, 0) != DS_OK) {
        return false;
    }
    return true;
}

static Uint8 *DSOUND_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    DWORD cursor = 0;
    DWORD junk = 0;
    HRESULT result = DS_OK;

    SDL_assert(*buffer_size == device->buffer_size);

    // Figure out which blocks to fill next
    device->hidden->locked_buf = NULL;
    result = IDirectSoundBuffer_GetCurrentPosition(device->hidden->mixbuf,
                                                   &junk, &cursor);
    if (result == DSERR_BUFFERLOST) {
        IDirectSoundBuffer_Restore(device->hidden->mixbuf);
        result = IDirectSoundBuffer_GetCurrentPosition(device->hidden->mixbuf,
                                                       &junk, &cursor);
    }
    if (result != DS_OK) {
        SetDSerror("DirectSound GetCurrentPosition", result);
        return NULL;
    }
    cursor /= device->buffer_size;
#ifdef DEBUG_SOUND
    // Detect audio dropouts
    {
        DWORD spot = cursor;
        if (spot < device->hidden->lastchunk) {
            spot += device->hidden->num_buffers;
        }
        if (spot > device->hidden->lastchunk + 1) {
            fprintf(stderr, "Audio dropout, missed %d fragments\n",
                    (spot - (device->hidden->lastchunk + 1)));
        }
    }
#endif
    device->hidden->lastchunk = cursor;
    cursor = (cursor + 1) % device->hidden->num_buffers;
    cursor *= device->buffer_size;

    // Lock the audio buffer
    DWORD rawlen = 0;
    result = IDirectSoundBuffer_Lock(device->hidden->mixbuf, cursor,
                                     device->buffer_size,
                                     (LPVOID *)&device->hidden->locked_buf,
                                     &rawlen, NULL, &junk, 0);
    if (result == DSERR_BUFFERLOST) {
        IDirectSoundBuffer_Restore(device->hidden->mixbuf);
        result = IDirectSoundBuffer_Lock(device->hidden->mixbuf, cursor,
                                         device->buffer_size,
                                         (LPVOID *)&device->hidden->locked_buf, &rawlen, NULL,
                                         &junk, 0);
    }
    if (result != DS_OK) {
        SetDSerror("DirectSound Lock", result);
        return NULL;
    }
    return device->hidden->locked_buf;
}

static bool DSOUND_WaitRecordingDevice(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    while (!SDL_GetAtomicInt(&device->shutdown)) {
        DWORD junk, cursor;
        if (IDirectSoundCaptureBuffer_GetCurrentPosition(h->capturebuf, &junk, &cursor) != DS_OK) {
            return false;
        } else if ((cursor / device->buffer_size) != h->lastchunk) {
            break;
        }
        SDL_Delay(1);
    }

    return true;
}

static int DSOUND_RecordDevice(SDL_AudioDevice *device, void *buffer, int buflen)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    DWORD ptr1len, ptr2len;
    VOID *ptr1, *ptr2;

    SDL_assert(buflen == device->buffer_size);

    if (IDirectSoundCaptureBuffer_Lock(h->capturebuf, h->lastchunk * buflen, buflen, &ptr1, &ptr1len, &ptr2, &ptr2len, 0) != DS_OK) {
        return -1;
    }

    SDL_assert(ptr1len == (DWORD)buflen);
    SDL_assert(ptr2 == NULL);
    SDL_assert(ptr2len == 0);

    SDL_memcpy(buffer, ptr1, ptr1len);

    if (IDirectSoundCaptureBuffer_Unlock(h->capturebuf, ptr1, ptr1len, ptr2, ptr2len) != DS_OK) {
        return -1;
    }

    h->lastchunk = (h->lastchunk + 1) % h->num_buffers;

    return (int) ptr1len;
}

static void DSOUND_FlushRecording(SDL_AudioDevice *device)
{
    struct SDL_PrivateAudioData *h = device->hidden;
    DWORD junk, cursor;
    if (IDirectSoundCaptureBuffer_GetCurrentPosition(h->capturebuf, &junk, &cursor) == DS_OK) {
        h->lastchunk = cursor / device->buffer_size;
    }
}

static void DSOUND_CloseDevice(SDL_AudioDevice *device)
{
    if (device->hidden) {
        if (device->hidden->mixbuf) {
            IDirectSoundBuffer_Stop(device->hidden->mixbuf);
            IDirectSoundBuffer_Release(device->hidden->mixbuf);
        }
        if (device->hidden->sound) {
            IDirectSound_Release(device->hidden->sound);
        }
        if (device->hidden->capturebuf) {
            IDirectSoundCaptureBuffer_Stop(device->hidden->capturebuf);
            IDirectSoundCaptureBuffer_Release(device->hidden->capturebuf);
        }
        if (device->hidden->capture) {
            IDirectSoundCapture_Release(device->hidden->capture);
        }
        SDL_free(device->hidden);
        device->hidden = NULL;
    }
}

/* This function tries to create a secondary audio buffer, and returns the
   number of audio chunks available in the created buffer. This is for
   playback devices, not recording.
*/
static bool CreateSecondary(SDL_AudioDevice *device, const DWORD bufsize, WAVEFORMATEX *wfmt)
{
    LPDIRECTSOUND sndObj = device->hidden->sound;
    LPDIRECTSOUNDBUFFER *sndbuf = &device->hidden->mixbuf;
    HRESULT result = DS_OK;
    DSBUFFERDESC format;
    LPVOID pvAudioPtr1, pvAudioPtr2;
    DWORD dwAudioBytes1, dwAudioBytes2;

    // Try to create the secondary buffer
    SDL_zero(format);
    format.dwSize = sizeof(format);
    format.dwFlags = DSBCAPS_GETCURRENTPOSITION2;
    format.dwFlags |= DSBCAPS_GLOBALFOCUS;
    format.dwBufferBytes = bufsize;
    format.lpwfxFormat = wfmt;
    result = IDirectSound_CreateSoundBuffer(sndObj, &format, sndbuf, NULL);
    if (result != DS_OK) {
        return SetDSerror("DirectSound CreateSoundBuffer", result);
    }
    IDirectSoundBuffer_SetFormat(*sndbuf, wfmt);

    // Silence the initial audio buffer
    result = IDirectSoundBuffer_Lock(*sndbuf, 0, format.dwBufferBytes,
                                     (LPVOID *)&pvAudioPtr1, &dwAudioBytes1,
                                     (LPVOID *)&pvAudioPtr2, &dwAudioBytes2,
                                     DSBLOCK_ENTIREBUFFER);
    if (result == DS_OK) {
        SDL_memset(pvAudioPtr1, device->silence_value, dwAudioBytes1);
        IDirectSoundBuffer_Unlock(*sndbuf,
                                  (LPVOID)pvAudioPtr1, dwAudioBytes1,
                                  (LPVOID)pvAudioPtr2, dwAudioBytes2);
    }

    return true;  // We're ready to go
}

/* This function tries to create a capture buffer, and returns the
   number of audio chunks available in the created buffer. This is for
   recording devices, not playback.
*/
static bool CreateCaptureBuffer(SDL_AudioDevice *device, const DWORD bufsize, WAVEFORMATEX *wfmt)
{
    LPDIRECTSOUNDCAPTURE capture = device->hidden->capture;
    LPDIRECTSOUNDCAPTUREBUFFER *capturebuf = &device->hidden->capturebuf;
    DSCBUFFERDESC format;
    HRESULT result;

    SDL_zero(format);
    format.dwSize = sizeof(format);
    format.dwFlags = DSCBCAPS_WAVEMAPPED;
    format.dwBufferBytes = bufsize;
    format.lpwfxFormat = wfmt;

    result = IDirectSoundCapture_CreateCaptureBuffer(capture, &format, capturebuf, NULL);
    if (result != DS_OK) {
        return SetDSerror("DirectSound CreateCaptureBuffer", result);
    }

    result = IDirectSoundCaptureBuffer_Start(*capturebuf, DSCBSTART_LOOPING);
    if (result != DS_OK) {
        IDirectSoundCaptureBuffer_Release(*capturebuf);
        return SetDSerror("DirectSound Start", result);
    }

#if 0
    // presumably this starts at zero, but just in case...
    result = IDirectSoundCaptureBuffer_GetCurrentPosition(*capturebuf, &junk, &cursor);
    if (result != DS_OK) {
        IDirectSoundCaptureBuffer_Stop(*capturebuf);
        IDirectSoundCaptureBuffer_Release(*capturebuf);
        return SetDSerror("DirectSound GetCurrentPosition", result);
    }

    device->hidden->lastchunk = cursor / device->buffer_size;
#endif

    return true;
}

static bool DSOUND_OpenDevice(SDL_AudioDevice *device)
{
    // Initialize all variables that we clean on shutdown
    device->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    // Open the audio device
    LPGUID guid;
#ifdef HAVE_MMDEVICEAPI_H
    if (SupportsIMMDevice) {
        guid = SDL_IMMDevice_GetDirectSoundGUID(device);
    } else
#endif
    {
        guid = (LPGUID) device->handle;
    }

    SDL_assert(guid != NULL);

    HRESULT result;
    if (device->recording) {
        result = pDirectSoundCaptureCreate8(guid, &device->hidden->capture, NULL);
        if (result != DS_OK) {
            return SetDSerror("DirectSoundCaptureCreate8", result);
        }
    } else {
        result = pDirectSoundCreate8(guid, &device->hidden->sound, NULL);
        if (result != DS_OK) {
            return SetDSerror("DirectSoundCreate8", result);
        }
        result = IDirectSound_SetCooperativeLevel(device->hidden->sound,
                                                  GetDesktopWindow(),
                                                  DSSCL_NORMAL);
        if (result != DS_OK) {
            return SetDSerror("DirectSound SetCooperativeLevel", result);
        }
    }

    const DWORD numchunks = 8;
    DWORD bufsize;
    bool tried_format = false;
    SDL_AudioFormat test_format;
    const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
        switch (test_format) {
        case SDL_AUDIO_U8:
        case SDL_AUDIO_S16:
        case SDL_AUDIO_S32:
        case SDL_AUDIO_F32:
            tried_format = true;

            device->spec.format = test_format;

            // Update the fragment size as size in bytes
            SDL_UpdatedAudioDeviceFormat(device);

            bufsize = numchunks * device->buffer_size;
            if ((bufsize < DSBSIZE_MIN) || (bufsize > DSBSIZE_MAX)) {
                SDL_SetError("Sound buffer size must be between %d and %d",
                             (int)((DSBSIZE_MIN < numchunks) ? 1 : DSBSIZE_MIN / numchunks),
                             (int)(DSBSIZE_MAX / numchunks));
            } else {
                WAVEFORMATEXTENSIBLE wfmt;
                SDL_zero(wfmt);
                if (device->spec.channels > 2) {
                    wfmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
                    wfmt.Format.cbSize = sizeof(wfmt) - sizeof(WAVEFORMATEX);

                    if (SDL_AUDIO_ISFLOAT(device->spec.format)) {
                        SDL_memcpy(&wfmt.SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, sizeof(GUID));
                    } else {
                        SDL_memcpy(&wfmt.SubFormat, &SDL_KSDATAFORMAT_SUBTYPE_PCM, sizeof(GUID));
                    }
                    wfmt.Samples.wValidBitsPerSample = SDL_AUDIO_BITSIZE(device->spec.format);

                    switch (device->spec.channels) {
                    case 3: // 3.0 (or 2.1)
                        wfmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER;
                        break;
                    case 4: // 4.0
                        wfmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;
                        break;
                    case 5: // 5.0 (or 4.1)
                        wfmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;
                        break;
                    case 6: // 5.1
                        wfmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;
                        break;
                    case 7: // 6.1
                        wfmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_BACK_CENTER;
                        break;
                    case 8: // 7.1
                        wfmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT | SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT;
                        break;
                    default:
                        SDL_assert(!"Unsupported channel count!");
                        break;
                    }
                } else if (SDL_AUDIO_ISFLOAT(device->spec.format)) {
                    wfmt.Format.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
                } else {
                    wfmt.Format.wFormatTag = WAVE_FORMAT_PCM;
                }

                wfmt.Format.wBitsPerSample = SDL_AUDIO_BITSIZE(device->spec.format);
                wfmt.Format.nChannels = (WORD)device->spec.channels;
                wfmt.Format.nSamplesPerSec = device->spec.freq;
                wfmt.Format.nBlockAlign = wfmt.Format.nChannels * (wfmt.Format.wBitsPerSample / 8);
                wfmt.Format.nAvgBytesPerSec = wfmt.Format.nSamplesPerSec * wfmt.Format.nBlockAlign;

                const bool rc = device->recording ? CreateCaptureBuffer(device, bufsize, (WAVEFORMATEX *)&wfmt) : CreateSecondary(device, bufsize, (WAVEFORMATEX *)&wfmt);
                if (rc) {
                    device->hidden->num_buffers = numchunks;
                    break;
                }
            }
            continue;
        default:
            continue;
        }
        break;
    }

    if (!test_format) {
        if (tried_format) {
            return false; // CreateSecondary() should have called SDL_SetError().
        }
        return SDL_SetError("%s: Unsupported audio format", "directsound");
    }

    // Playback buffers will auto-start playing in DSOUND_WaitDevice()

    return true; // good to go.
}

static void DSOUND_DeinitializeStart(void)
{
#ifdef HAVE_MMDEVICEAPI_H
    if (SupportsIMMDevice) {
        SDL_IMMDevice_Quit();
    }
#endif
}

static void DSOUND_Deinitialize(void)
{
    DSOUND_Unload();
#ifdef HAVE_MMDEVICEAPI_H
    SupportsIMMDevice = false;
#endif
}

static bool DSOUND_Init(SDL_AudioDriverImpl *impl)
{
    if (!DSOUND_Load()) {
        return false;
    }

#ifdef HAVE_MMDEVICEAPI_H
    SupportsIMMDevice = SDL_IMMDevice_Init(NULL);
#endif

    impl->DetectDevices = DSOUND_DetectDevices;
    impl->OpenDevice = DSOUND_OpenDevice;
    impl->PlayDevice = DSOUND_PlayDevice;
    impl->WaitDevice = DSOUND_WaitDevice;
    impl->GetDeviceBuf = DSOUND_GetDeviceBuf;
    impl->WaitRecordingDevice = DSOUND_WaitRecordingDevice;
    impl->RecordDevice = DSOUND_RecordDevice;
    impl->FlushRecording = DSOUND_FlushRecording;
    impl->CloseDevice = DSOUND_CloseDevice;
    impl->FreeDeviceHandle = DSOUND_FreeDeviceHandle;
    impl->DeinitializeStart = DSOUND_DeinitializeStart;
    impl->Deinitialize = DSOUND_Deinitialize;

    impl->HasRecordingSupport = true;

    return true;
}

AudioBootStrap DSOUND_bootstrap = {
    "directsound", "DirectSound", DSOUND_Init, false, false
};

#endif // SDL_AUDIO_DRIVER_DSOUND
