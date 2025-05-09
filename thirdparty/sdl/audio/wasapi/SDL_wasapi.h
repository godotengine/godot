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

#ifndef SDL_wasapi_h_
#define SDL_wasapi_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "../SDL_sysaudio.h"

struct SDL_PrivateAudioData
{
    WCHAR *devid;
    WAVEFORMATEX *waveformat;
    IAudioClient *client;
    IAudioRenderClient *render;
    IAudioCaptureClient *capture;
    HANDLE event;
    HANDLE task;
    bool coinitialized;
    int framesize;
    SDL_AtomicInt device_disconnecting;
    bool device_lost;
    bool device_dead;
};

// win32 implementation calls into these.
bool WASAPI_PrepDevice(SDL_AudioDevice *device);
void WASAPI_DisconnectDevice(SDL_AudioDevice *device);  // don't hold the device lock when calling this!


// BE CAREFUL: if you are holding the device lock and proxy to the management thread with wait_until_complete, and grab the lock again, you will deadlock.
typedef bool (*ManagementThreadTask)(void *userdata);
bool WASAPI_ProxyToManagementThread(ManagementThreadTask task, void *userdata, bool *wait_until_complete);

#ifdef __cplusplus
}
#endif

#endif // SDL_wasapi_h_
