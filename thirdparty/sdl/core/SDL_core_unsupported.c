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

#ifndef SDL_VIDEO_DRIVER_X11

SDL_DECLSPEC void SDLCALL SDL_SetX11EventHook(SDL_X11EventHook callback, void *userdata);
void SDL_SetX11EventHook(SDL_X11EventHook callback, void *userdata)
{
}

#endif

#ifndef SDL_PLATFORM_LINUX

SDL_DECLSPEC bool SDLCALL SDL_SetLinuxThreadPriority(Sint64 threadID, int priority);
bool SDL_SetLinuxThreadPriority(Sint64 threadID, int priority)
{
    (void)threadID;
    (void)priority;
    return SDL_Unsupported();
}

SDL_DECLSPEC bool SDLCALL SDL_SetLinuxThreadPriorityAndPolicy(Sint64 threadID, int sdlPriority, int schedPolicy);
bool SDL_SetLinuxThreadPriorityAndPolicy(Sint64 threadID, int sdlPriority, int schedPolicy)
{
    (void)threadID;
    (void)sdlPriority;
    (void)schedPolicy;
    return SDL_Unsupported();
}

#endif

#ifndef SDL_PLATFORM_GDK

SDL_DECLSPEC void SDLCALL SDL_GDKSuspendComplete(void);
void SDL_GDKSuspendComplete(void)
{
    SDL_Unsupported();
}

SDL_DECLSPEC bool SDLCALL SDL_GetGDKDefaultUser(void *outUserHandle); /* XUserHandle *outUserHandle */
bool SDL_GetGDKDefaultUser(void *outUserHandle)
{
    return SDL_Unsupported();
}

SDL_DECLSPEC void SDLCALL SDL_GDKSuspendGPU(SDL_GPUDevice *device);
void SDL_GDKSuspendGPU(SDL_GPUDevice *device)
{
}

SDL_DECLSPEC void SDLCALL SDL_GDKResumeGPU(SDL_GPUDevice *device);
void SDL_GDKResumeGPU(SDL_GPUDevice *device)
{
}

#endif

#if !defined(SDL_PLATFORM_WINDOWS)

SDL_DECLSPEC bool SDLCALL SDL_RegisterApp(const char *name, Uint32 style, void *hInst);
bool SDL_RegisterApp(const char *name, Uint32 style, void *hInst)
{
    (void)name;
    (void)style;
    (void)hInst;
    return SDL_Unsupported();
}

SDL_DECLSPEC void SDLCALL SDL_SetWindowsMessageHook(void *callback, void *userdata); // SDL_WindowsMessageHook callback
void SDL_SetWindowsMessageHook(void *callback, void *userdata)
{
    (void)callback;
    (void)userdata;
    SDL_Unsupported();
}

SDL_DECLSPEC void SDLCALL SDL_UnregisterApp(void);
void SDL_UnregisterApp(void)
{
    SDL_Unsupported();
}

#endif

#ifndef SDL_PLATFORM_ANDROID

SDL_DECLSPEC void SDLCALL SDL_SendAndroidBackButton(void);
void SDL_SendAndroidBackButton(void)
{
    SDL_Unsupported();
}

SDL_DECLSPEC void * SDLCALL SDL_GetAndroidActivity(void);
void *SDL_GetAndroidActivity(void)
{
    SDL_Unsupported();
    return NULL;
}

SDL_DECLSPEC const char * SDLCALL SDL_GetAndroidCachePath(void);
const char* SDL_GetAndroidCachePath(void)
{
    SDL_Unsupported();
    return NULL;
}


SDL_DECLSPEC const char * SDLCALL SDL_GetAndroidExternalStoragePath(void);
const char* SDL_GetAndroidExternalStoragePath(void)
{
    SDL_Unsupported();
    return NULL;
}

SDL_DECLSPEC Uint32 SDLCALL SDL_GetAndroidExternalStorageState(void);
Uint32 SDL_GetAndroidExternalStorageState(void)
{
    SDL_Unsupported();
    return 0;
}
SDL_DECLSPEC const char * SDLCALL SDL_GetAndroidInternalStoragePath(void);
const char *SDL_GetAndroidInternalStoragePath(void)
{
    SDL_Unsupported();
    return NULL;
}

SDL_DECLSPEC void * SDLCALL SDL_GetAndroidJNIEnv(void);
void *SDL_GetAndroidJNIEnv(void)
{
    SDL_Unsupported();
    return NULL;
}

typedef void (SDLCALL *SDL_RequestAndroidPermissionCallback)(void *userdata, const char *permission, bool granted);
SDL_DECLSPEC bool SDLCALL SDL_RequestAndroidPermission(const char *permission, SDL_RequestAndroidPermissionCallback cb, void *userdata);
bool SDL_RequestAndroidPermission(const char *permission, SDL_RequestAndroidPermissionCallback cb, void *userdata)
{
    (void)permission;
    (void)cb;
    (void)userdata;
    return SDL_Unsupported();
}

SDL_DECLSPEC bool SDLCALL SDL_SendAndroidMessage(Uint32 command, int param);
bool SDL_SendAndroidMessage(Uint32 command, int param)
{
    (void)command;
    (void)param;
    return SDL_Unsupported();
}

SDL_DECLSPEC bool SDLCALL SDL_ShowAndroidToast(const char *message, int duration, int gravity, int xoffset, int yoffset);
bool SDL_ShowAndroidToast(const char* message, int duration, int gravity, int xoffset, int yoffset)
{
    (void)message;
    (void)duration;
    (void)gravity;
    (void)xoffset;
    (void)yoffset;
    return SDL_Unsupported();
}

SDL_DECLSPEC int SDLCALL SDL_GetAndroidSDKVersion(void);
int SDL_GetAndroidSDKVersion(void)
{
    return SDL_Unsupported();
}

SDL_DECLSPEC bool SDLCALL SDL_IsChromebook(void);
bool SDL_IsChromebook(void)
{
    SDL_Unsupported();
    return false;
}

SDL_DECLSPEC bool SDLCALL SDL_IsDeXMode(void);
bool SDL_IsDeXMode(void)
{
    SDL_Unsupported();
    return false;
}

SDL_DECLSPEC Sint32 SDLCALL JNI_OnLoad(void *vm, void *reserved);
Sint32 JNI_OnLoad(void *vm, void *reserved)
{
    (void)vm;
    (void)reserved;
    SDL_Unsupported();
    return -1; // JNI_ERR
}
#endif
