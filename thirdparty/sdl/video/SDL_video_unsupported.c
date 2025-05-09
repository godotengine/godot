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

#ifndef SDL_VIDEO_DRIVER_WINDOWS

#if defined(SDL_PLATFORM_WINDOWS)

bool SDL_RegisterApp(const char *name, Uint32 style, void *hInst)
{
    (void)name;
    (void)style;
    (void)hInst;
    return true;
}

void SDL_UnregisterApp(void)
{
}

void SDL_SetWindowsMessageHook(SDL_WindowsMessageHook callback, void *userdata)
{
}

#endif // defined(SDL_PLATFORM_WINDOWS)

SDL_DECLSPEC bool SDLCALL SDL_GetDXGIOutputInfo(SDL_DisplayID displayID, int *adapterIndex, int *outputIndex);
bool SDL_GetDXGIOutputInfo(SDL_DisplayID displayID, int *adapterIndex, int *outputIndex)
{
    (void)displayID;
    (void)adapterIndex;
    (void)outputIndex;
    return SDL_Unsupported();
}

SDL_DECLSPEC int SDLCALL SDL_GetDirect3D9AdapterIndex(SDL_DisplayID displayID);
int SDL_GetDirect3D9AdapterIndex(SDL_DisplayID displayID)
{
    (void)displayID;
    SDL_Unsupported();
    return -1;
}

#elif defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)

SDL_DECLSPEC int SDLCALL SDL_GetDirect3D9AdapterIndex(SDL_DisplayID displayID);
int SDL_GetDirect3D9AdapterIndex(SDL_DisplayID displayID)
{
    (void)displayID;
    SDL_Unsupported();
    return -1;
}

#endif // !SDL_VIDEO_DRIVER_WINDOWS

#ifndef SDL_PLATFORM_GDK

SDL_DECLSPEC bool SDLCALL SDL_GetGDKTaskQueue(void *outTaskQueue);
bool SDL_GetGDKTaskQueue(void *outTaskQueue)
{
    (void)outTaskQueue;
    return SDL_Unsupported();
}

#endif

#ifndef SDL_VIDEO_DRIVER_UIKIT

SDL_DECLSPEC void SDLCALL SDL_OnApplicationDidChangeStatusBarOrientation(void);
void SDL_OnApplicationDidChangeStatusBarOrientation(void)
{
    SDL_Unsupported();
}

#endif

#ifndef SDL_VIDEO_DRIVER_UIKIT

typedef void (SDLCALL *SDL_iOSAnimationCallback)(void *userdata);
SDL_DECLSPEC bool SDLCALL SDL_SetiOSAnimationCallback(SDL_Window *window, int interval, SDL_iOSAnimationCallback callback, void *callbackParam);
bool SDL_SetiOSAnimationCallback(SDL_Window *window, int interval, SDL_iOSAnimationCallback callback, void *callbackParam)
{
    (void)window;
    (void)interval;
    (void)callback;
    (void)callbackParam;
    return SDL_Unsupported();
}

SDL_DECLSPEC void SDLCALL SDL_SetiOSEventPump(bool enabled);
void SDL_SetiOSEventPump(bool enabled)
{
    (void)enabled;
    SDL_Unsupported();
}
#endif

