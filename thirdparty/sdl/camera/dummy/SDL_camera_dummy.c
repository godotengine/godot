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

#ifdef SDL_CAMERA_DRIVER_DUMMY

#include "../SDL_syscamera.h"

static bool DUMMYCAMERA_OpenDevice(SDL_Camera *device, const SDL_CameraSpec *spec)
{
    return SDL_Unsupported();
}

static void DUMMYCAMERA_CloseDevice(SDL_Camera *device)
{
}

static bool DUMMYCAMERA_WaitDevice(SDL_Camera *device)
{
    return SDL_Unsupported();
}

static SDL_CameraFrameResult DUMMYCAMERA_AcquireFrame(SDL_Camera *device, SDL_Surface *frame, Uint64 *timestampNS)
{
    SDL_Unsupported();
    return SDL_CAMERA_FRAME_ERROR;
}

static void DUMMYCAMERA_ReleaseFrame(SDL_Camera *device, SDL_Surface *frame)
{
}

static void DUMMYCAMERA_DetectDevices(void)
{
}

static void DUMMYCAMERA_FreeDeviceHandle(SDL_Camera *device)
{
}

static void DUMMYCAMERA_Deinitialize(void)
{
}

static bool DUMMYCAMERA_Init(SDL_CameraDriverImpl *impl)
{
    impl->DetectDevices = DUMMYCAMERA_DetectDevices;
    impl->OpenDevice = DUMMYCAMERA_OpenDevice;
    impl->CloseDevice = DUMMYCAMERA_CloseDevice;
    impl->WaitDevice = DUMMYCAMERA_WaitDevice;
    impl->AcquireFrame = DUMMYCAMERA_AcquireFrame;
    impl->ReleaseFrame = DUMMYCAMERA_ReleaseFrame;
    impl->FreeDeviceHandle = DUMMYCAMERA_FreeDeviceHandle;
    impl->Deinitialize = DUMMYCAMERA_Deinitialize;

    return true;
}

CameraBootStrap DUMMYCAMERA_bootstrap = {
    "dummy", "SDL dummy camera driver", DUMMYCAMERA_Init, true
};

#endif  // SDL_CAMERA_DRIVER_DUMMY
