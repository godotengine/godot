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

#ifdef SDL_AUDIO_DRIVER_N3DS

// N3DS Audio driver

#include "../SDL_sysaudio.h"
#include "SDL_n3dsaudio.h"

#define N3DSAUDIO_DRIVER_NAME "n3ds"

static dspHookCookie dsp_hook;
static SDL_AudioDevice *audio_device;

// fully local functions related to the wavebufs / DSP, not the same as the `device->lock` SDL_Mutex!
static SDL_INLINE void contextLock(SDL_AudioDevice *device)
{
    LightLock_Lock(&device->hidden->lock);
}

static SDL_INLINE void contextUnlock(SDL_AudioDevice *device)
{
    LightLock_Unlock(&device->hidden->lock);
}

static void N3DSAUD_DspHook(DSP_HookType hook)
{
    if (hook == DSPHOOK_ONCANCEL) {
        contextLock(audio_device);
        audio_device->hidden->isCancelled = true;
        SDL_AudioDeviceDisconnected(audio_device);
        CondVar_Broadcast(&audio_device->hidden->cv);
        contextUnlock(audio_device);
    }
}

static void AudioFrameFinished(void *vdevice)
{
    bool shouldBroadcast = false;
    unsigned i;
    SDL_AudioDevice *device = (SDL_AudioDevice *)vdevice;

    contextLock(device);

    for (i = 0; i < NUM_BUFFERS; i++) {
        if (device->hidden->waveBuf[i].status == NDSP_WBUF_DONE) {
            device->hidden->waveBuf[i].status = NDSP_WBUF_FREE;
            shouldBroadcast = true;
        }
    }

    if (shouldBroadcast) {
        CondVar_Broadcast(&device->hidden->cv);
    }

    contextUnlock(device);
}

static bool N3DSAUDIO_OpenDevice(SDL_AudioDevice *device)
{
    Result ndsp_init_res;
    Uint8 *data_vaddr;
    float mix[12];

    device->hidden = (struct SDL_PrivateAudioData *)SDL_calloc(1, sizeof(*device->hidden));
    if (!device->hidden) {
        return false;
    }

    // Initialise the DSP service
    ndsp_init_res = ndspInit();
    if (R_FAILED(ndsp_init_res)) {
        if ((R_SUMMARY(ndsp_init_res) == RS_NOTFOUND) && (R_MODULE(ndsp_init_res) == RM_DSP)) {
            return SDL_SetError("DSP init failed: dspfirm.cdc missing!");
        } else {
            return SDL_SetError("DSP init failed. Error code: 0x%lX", ndsp_init_res);
        }
    }

    // Initialise internal state
    LightLock_Init(&device->hidden->lock);
    CondVar_Init(&device->hidden->cv);

    if (device->spec.channels > 2) {
        device->spec.channels = 2;
    }

    Uint32 format = 0;
    SDL_AudioFormat test_format;
    const SDL_AudioFormat *closefmts = SDL_ClosestAudioFormats(device->spec.format);
    while ((test_format = *(closefmts++)) != 0) {
        if (test_format == SDL_AUDIO_S8) {  // Signed 8-bit audio supported
            format = (device->spec.channels == 2) ? NDSP_FORMAT_STEREO_PCM8 : NDSP_FORMAT_MONO_PCM8;
            break;
        } else if (test_format == SDL_AUDIO_S16) {  // Signed 16-bit audio supported
            format = (device->spec.channels == 2) ? NDSP_FORMAT_STEREO_PCM16 : NDSP_FORMAT_MONO_PCM16;
            break;
        }
    }

    if (!test_format) {      // shouldn't happen, but just in case...
        return SDL_SetError("No supported audio format found.");
    }

    device->spec.format = test_format;

    // Update the fragment size as size in bytes
    SDL_UpdatedAudioDeviceFormat(device);

    // Allocate mixing buffer
    if (device->buffer_size >= SDL_MAX_UINT32 / 2) {
        return SDL_SetError("Mixing buffer is too large.");
    }

    device->hidden->mixbuf = (Uint8 *)SDL_malloc(device->buffer_size);
    if (!device->hidden->mixbuf) {
        return false;
    }

    SDL_memset(device->hidden->mixbuf, device->silence_value, device->buffer_size);

    data_vaddr = (Uint8 *)linearAlloc(device->buffer_size * NUM_BUFFERS);
    if (!data_vaddr) {
        return SDL_OutOfMemory();
    }

    SDL_memset(data_vaddr, 0, device->buffer_size * NUM_BUFFERS);
    DSP_FlushDataCache(data_vaddr, device->buffer_size * NUM_BUFFERS);

    device->hidden->nextbuf = 0;

    ndspChnReset(0);

    ndspChnSetInterp(0, NDSP_INTERP_LINEAR);
    ndspChnSetRate(0, device->spec.freq);
    ndspChnSetFormat(0, format);

    SDL_zeroa(mix);
    mix[0] = mix[1] = 1.0f;
    ndspChnSetMix(0, mix);

    SDL_memset(device->hidden->waveBuf, 0, sizeof(ndspWaveBuf) * NUM_BUFFERS);

    const int sample_frame_size = SDL_AUDIO_FRAMESIZE(device->spec);
    for (unsigned i = 0; i < NUM_BUFFERS; i++) {
        device->hidden->waveBuf[i].data_vaddr = data_vaddr;
        device->hidden->waveBuf[i].nsamples = device->buffer_size / sample_frame_size;
        data_vaddr += device->buffer_size;
    }

    // Setup callback
    audio_device = device;
    ndspSetCallback(AudioFrameFinished, device);
    dspHook(&dsp_hook, N3DSAUD_DspHook);

    return true;
}

static bool N3DSAUDIO_PlayDevice(SDL_AudioDevice *device, const Uint8 *buffer, int buflen)
{
    contextLock(device);

    const size_t nextbuf = device->hidden->nextbuf;

    if (device->hidden->isCancelled ||
        device->hidden->waveBuf[nextbuf].status != NDSP_WBUF_FREE) {
        contextUnlock(device);
        return true;  // !!! FIXME: is this a fatal error? If so, this should return false.
    }

    device->hidden->nextbuf = (nextbuf + 1) % NUM_BUFFERS;

    contextUnlock(device);

    SDL_memcpy((void *)device->hidden->waveBuf[nextbuf].data_vaddr, buffer, buflen);
    DSP_FlushDataCache(device->hidden->waveBuf[nextbuf].data_vaddr, buflen);

    ndspChnWaveBufAdd(0, &device->hidden->waveBuf[nextbuf]);

    return true;
}

static bool N3DSAUDIO_WaitDevice(SDL_AudioDevice *device)
{
    contextLock(device);
    while (!device->hidden->isCancelled && !SDL_GetAtomicInt(&device->shutdown) &&
           device->hidden->waveBuf[device->hidden->nextbuf].status != NDSP_WBUF_FREE) {
        CondVar_Wait(&device->hidden->cv, &device->hidden->lock);
    }
    contextUnlock(device);
    return true;
}

static Uint8 *N3DSAUDIO_GetDeviceBuf(SDL_AudioDevice *device, int *buffer_size)
{
    return device->hidden->mixbuf;
}

static void N3DSAUDIO_CloseDevice(SDL_AudioDevice *device)
{
    if (!device->hidden) {
        return;
    }

    contextLock(device);

    dspUnhook(&dsp_hook);
    ndspSetCallback(NULL, NULL);

    if (!device->hidden->isCancelled) {
        ndspChnReset(0);
        SDL_memset(device->hidden->waveBuf, 0, sizeof(ndspWaveBuf) * NUM_BUFFERS);
        CondVar_Broadcast(&device->hidden->cv);
    }

    contextUnlock(device);

    ndspExit();

    if (device->hidden->waveBuf[0].data_vaddr) {
        linearFree((void *)device->hidden->waveBuf[0].data_vaddr);
    }

    if (device->hidden->mixbuf) {
        SDL_free(device->hidden->mixbuf);
        device->hidden->mixbuf = NULL;
    }

    SDL_free(device->hidden);
    device->hidden = NULL;
}

static void N3DSAUDIO_ThreadInit(SDL_AudioDevice *device)
{
    s32 current_priority = 0x30;
    svcGetThreadPriority(&current_priority, CUR_THREAD_HANDLE);
    current_priority--;
    // 0x18 is reserved for video, 0x30 is the default for main thread
    current_priority = SDL_clamp(current_priority, 0x19, 0x2F);
    svcSetThreadPriority(CUR_THREAD_HANDLE, current_priority);
}

static bool N3DSAUDIO_Init(SDL_AudioDriverImpl *impl)
{
    impl->OpenDevice = N3DSAUDIO_OpenDevice;
    impl->PlayDevice = N3DSAUDIO_PlayDevice;
    impl->WaitDevice = N3DSAUDIO_WaitDevice;
    impl->GetDeviceBuf = N3DSAUDIO_GetDeviceBuf;
    impl->CloseDevice = N3DSAUDIO_CloseDevice;
    impl->ThreadInit = N3DSAUDIO_ThreadInit;
    impl->OnlyHasDefaultPlaybackDevice = true;

    // Should be possible, but micInit would fail
    impl->HasRecordingSupport = false;

    return true;
}

AudioBootStrap N3DSAUDIO_bootstrap = {
    N3DSAUDIO_DRIVER_NAME,
    "SDL N3DS audio driver",
    N3DSAUDIO_Init,
    false,
    false
};

#endif // SDL_AUDIO_DRIVER_N3DS
