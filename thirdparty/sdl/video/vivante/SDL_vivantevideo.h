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

#ifndef SDL_vivantevideo_h_
#define SDL_vivantevideo_h_

#include "../SDL_sysvideo.h"

// Set up definitions for Vivante EGL
#include <SDL3/SDL_egl.h>

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
#include <gc_vdk.h>
#else
#include <EGL/egl.h>
#endif

struct SDL_VideoData
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    vdkPrivate vdk_private;
#else
    SDL_SharedObject *egl_handle; // EGL shared library handle
    EGLNativeDisplayType(EGLAPIENTRY *fbGetDisplay)(void *context);
    EGLNativeDisplayType(EGLAPIENTRY *fbGetDisplayByIndex)(int DisplayIndex);
    void(EGLAPIENTRY *fbGetDisplayGeometry)(EGLNativeDisplayType Display, int *Width, int *Height);
    void(EGLAPIENTRY *fbGetDisplayInfo)(EGLNativeDisplayType Display, int *Width, int *Height, unsigned long *Physical, int *Stride, int *bits_per_pixel);
    void(EGLAPIENTRY *fbDestroyDisplay)(EGLNativeDisplayType Display);
    EGLNativeWindowType(EGLAPIENTRY *fbCreateWindow)(EGLNativeDisplayType Display, int X, int Y, int Width, int Height);
    void(EGLAPIENTRY *fbGetWindowGeometry)(EGLNativeWindowType Window, int *X, int *Y, int *Width, int *Height);
    void(EGLAPIENTRY *fbGetWindowInfo)(EGLNativeWindowType Window, int *X, int *Y, int *Width, int *Height, int *bits_per_pixel, unsigned int *Offset);
    void(EGLAPIENTRY *fbDestroyWindow)(EGLNativeWindowType Window);
#endif
};

struct SDL_DisplayData
{
    EGLNativeDisplayType native_display;
};

struct SDL_WindowData
{
    EGLNativeWindowType native_window;
    EGLSurface egl_surface;
};

/****************************************************************************/
// SDL_VideoDevice functions declaration
/****************************************************************************/

// Display and window functions
bool VIVANTE_VideoInit(SDL_VideoDevice *_this);
void VIVANTE_VideoQuit(SDL_VideoDevice *_this);
bool VIVANTE_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display);
bool VIVANTE_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
bool VIVANTE_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
void VIVANTE_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window);
bool VIVANTE_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window);
void VIVANTE_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window);
void VIVANTE_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window);
void VIVANTE_HideWindow(SDL_VideoDevice *_this, SDL_Window *window);
void VIVANTE_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);

// Event functions
void VIVANTE_PumpEvents(SDL_VideoDevice *_this);

#endif // SDL_vivantevideo_h_
