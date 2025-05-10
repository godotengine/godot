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

#ifdef SDL_VIDEO_DRIVER_VIVANTE

// SDL internals
#include "../SDL_sysvideo.h"
#include "../../events/SDL_events_c.h"

#ifdef SDL_INPUT_LINUXEV
#include "../../core/linux/SDL_evdev.h"
#endif

#include "SDL_vivantevideo.h"
#include "SDL_vivanteplatform.h"
#include "SDL_vivanteopengles.h"
#include "SDL_vivantevulkan.h"

static void VIVANTE_Destroy(SDL_VideoDevice *device)
{
    SDL_free(device->internal);
    SDL_free(device);
}

static SDL_VideoDevice *VIVANTE_Create(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *data;

    // Initialize SDL_VideoDevice structure
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }

    // Initialize internal data
    data = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!data) {
        SDL_free(device);
        return NULL;
    }

    device->internal = data;

    // Setup amount of available displays
    device->num_displays = 0;

    // Set device free function
    device->free = VIVANTE_Destroy;

    // Setup all functions which we can handle
    device->VideoInit = VIVANTE_VideoInit;
    device->VideoQuit = VIVANTE_VideoQuit;
    device->CreateSDLWindow = VIVANTE_CreateWindow;
    device->SetWindowTitle = VIVANTE_SetWindowTitle;
    device->SetWindowPosition = VIVANTE_SetWindowPosition;
    device->SetWindowSize = VIVANTE_SetWindowSize;
    device->ShowWindow = VIVANTE_ShowWindow;
    device->HideWindow = VIVANTE_HideWindow;
    device->DestroyWindow = VIVANTE_DestroyWindow;

#ifdef SDL_VIDEO_OPENGL_EGL
    device->GL_LoadLibrary = VIVANTE_GLES_LoadLibrary;
    device->GL_GetProcAddress = VIVANTE_GLES_GetProcAddress;
    device->GL_UnloadLibrary = VIVANTE_GLES_UnloadLibrary;
    device->GL_CreateContext = VIVANTE_GLES_CreateContext;
    device->GL_MakeCurrent = VIVANTE_GLES_MakeCurrent;
    device->GL_SetSwapInterval = VIVANTE_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = VIVANTE_GLES_GetSwapInterval;
    device->GL_SwapWindow = VIVANTE_GLES_SwapWindow;
    device->GL_DestroyContext = VIVANTE_GLES_DestroyContext;
#endif

#ifdef SDL_VIDEO_VULKAN
    device->Vulkan_LoadLibrary = VIVANTE_Vulkan_LoadLibrary;
    device->Vulkan_UnloadLibrary = VIVANTE_Vulkan_UnloadLibrary;
    device->Vulkan_GetInstanceExtensions = VIVANTE_Vulkan_GetInstanceExtensions;
    device->Vulkan_CreateSurface = VIVANTE_Vulkan_CreateSurface;
    device->Vulkan_DestroySurface = VIVANTE_Vulkan_DestroySurface;
#endif

    device->PumpEvents = VIVANTE_PumpEvents;

    return device;
}

VideoBootStrap VIVANTE_bootstrap = {
    "vivante",
    "Vivante EGL Video Driver",
    VIVANTE_Create,
    NULL, // no ShowMessageBox implementation
    false
};

/*****************************************************************************/
// SDL Video and Display initialization/handling functions
/*****************************************************************************/

static bool VIVANTE_AddVideoDisplays(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_VideoDisplay display;
    SDL_DisplayMode mode;
    SDL_DisplayData *data;
    int pitch = 0, bpp = 0;
    unsigned long pixels = 0;

    data = (SDL_DisplayData *)SDL_calloc(1, sizeof(SDL_DisplayData));
    if (!data) {
        return false;
    }

    SDL_zero(mode);
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    data->native_display = vdkGetDisplay(videodata->vdk_private);

    vdkGetDisplayInfo(data->native_display, &mode.w, &mode.h, &pixels, &pitch,
                      &bpp);
#else
    data->native_display = videodata->fbGetDisplayByIndex(0);

    videodata->fbGetDisplayInfo(data->native_display, &mode.w, &mode.h,
                                &pixels, &pitch, &bpp);
#endif // SDL_VIDEO_DRIVER_VIVANTE_VDK

    switch (bpp) {
    default: // Is another format used?
    case 32:
        mode.format = SDL_PIXELFORMAT_ARGB8888;
        break;
    case 16:
        mode.format = SDL_PIXELFORMAT_RGB565;
        break;
    }
    // FIXME: How do we query refresh rate?
    mode.refresh_rate = 60.0f;

    SDL_zero(display);
    display.name = VIVANTE_GetDisplayName(_this);
    display.desktop_mode = mode;
    display.internal = data;
    if (SDL_AddVideoDisplay(&display, false) == 0) {
        return false;
    }
    return true;
}

bool VIVANTE_VideoInit(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = _this->internal;

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    videodata->vdk_private = vdkInitialize();
    if (!videodata->vdk_private) {
        return SDL_SetError("vdkInitialize() failed");
    }
#else
    videodata->egl_handle = SDL_LoadObject("libEGL.so.1");
    if (!videodata->egl_handle) {
        videodata->egl_handle = SDL_LoadObject("libEGL.so");
        if (!videodata->egl_handle) {
            return false;
        }
    }
#define LOAD_FUNC(TYPE, NAME)                                               \
    videodata->NAME = (TYPE)SDL_LoadFunction(videodata->egl_handle, #NAME); \
    if (!videodata->NAME)                                                   \
        return false;

    LOAD_FUNC(EGLNativeDisplayType (EGLAPIENTRY *)(void *), fbGetDisplay);
    LOAD_FUNC(EGLNativeDisplayType (EGLAPIENTRY *)(int), fbGetDisplayByIndex);
    LOAD_FUNC(void (EGLAPIENTRY *)(EGLNativeDisplayType, int *, int *), fbGetDisplayGeometry);
    LOAD_FUNC(void (EGLAPIENTRY *)(EGLNativeDisplayType, int *, int *, unsigned long *, int *, int *), fbGetDisplayInfo);
    LOAD_FUNC(void (EGLAPIENTRY *)(EGLNativeDisplayType), fbDestroyDisplay);
    LOAD_FUNC(EGLNativeWindowType (EGLAPIENTRY *)(EGLNativeDisplayType, int, int, int, int), fbCreateWindow);
    LOAD_FUNC(void (EGLAPIENTRY *)(EGLNativeWindowType, int *, int *, int *, int *), fbGetWindowGeometry);
    LOAD_FUNC(void (EGLAPIENTRY *)(EGLNativeWindowType, int *, int *, int *, int *, int *, unsigned int *), fbGetWindowInfo);
    LOAD_FUNC(void (EGLAPIENTRY *)(EGLNativeWindowType), fbDestroyWindow);
#endif

    if (!VIVANTE_SetupPlatform(_this)) {
        return false;
    }

    if (!VIVANTE_AddVideoDisplays(_this)) {
        return false;
    }

    VIVANTE_UpdateDisplayScale(_this);

#ifdef SDL_INPUT_LINUXEV
    if (!SDL_EVDEV_Init()) {
        return false;
    }
#endif

    return true;
}

void VIVANTE_VideoQuit(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = _this->internal;

#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Quit();
#endif

    VIVANTE_CleanupPlatform(_this);

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    if (videodata->vdk_private) {
        vdkExit(videodata->vdk_private);
        videodata->vdk_private = NULL;
    }
#else
    if (videodata->egl_handle) {
        SDL_UnloadObject(videodata->egl_handle);
        videodata->egl_handle = NULL;
    }
#endif
}

bool VIVANTE_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_DisplayData *displaydata;
    SDL_WindowData *data;

    displaydata = SDL_GetDisplayDriverData(SDL_GetPrimaryDisplay());

    // Allocate window internal data
    data = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!data) {
        return false;
    }

    // Setup driver data for this window
    window->internal = data;

    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_VIVANTE_DISPLAY_POINTER, displaydata->native_display);

#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    data->native_window = vdkCreateWindow(displaydata->native_display, window->x, window->y, window->w, window->h);
#else
    data->native_window = videodata->fbCreateWindow(displaydata->native_display, window->x, window->y, window->w, window->h);
#endif
    if (!data->native_window) {
        return SDL_SetError("VIVANTE: Can't create native window");
    }
    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_VIVANTE_WINDOW_POINTER, data->native_window);

#ifdef SDL_VIDEO_OPENGL_EGL
    if (window->flags & SDL_WINDOW_OPENGL) {
        data->egl_surface = SDL_EGL_CreateSurface(_this, window, data->native_window);
        if (data->egl_surface == EGL_NO_SURFACE) {
            return SDL_SetError("VIVANTE: Can't create EGL surface");
        }
    } else {
        data->egl_surface = EGL_NO_SURFACE;
    }
    SDL_SetPointerProperty(props, SDL_PROP_WINDOW_VIVANTE_SURFACE_POINTER, data->egl_surface);
#endif

    // Window has been successfully created
    return true;
}

void VIVANTE_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_WindowData *data;

    data = window->internal;
    if (data) {
#ifdef SDL_VIDEO_OPENGL_EGL
        if (data->egl_surface != EGL_NO_SURFACE) {
            SDL_EGL_DestroySurface(_this, data->egl_surface);
        }
#endif

        if (data->native_window) {
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
            vdkDestroyWindow(data->native_window);
#else
            videodata->fbDestroyWindow(data->native_window);
#endif
        }

        SDL_free(data);
    }
    window->internal = NULL;
}

void VIVANTE_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    SDL_WindowData *data = window->internal;
    vdkSetWindowTitle(data->native_window, window->title);
#endif
}

bool VIVANTE_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window)
{
    // FIXME
    return SDL_Unsupported();
}

void VIVANTE_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    // FIXME
}

void VIVANTE_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    SDL_WindowData *data = window->internal;
    vdkShowWindow(data->native_window);
#endif
    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);
}

void VIVANTE_HideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
#ifdef SDL_VIDEO_DRIVER_VIVANTE_VDK
    SDL_WindowData *data = window->internal;
    vdkHideWindow(data->native_window);
#endif
    SDL_SetMouseFocus(NULL);
    SDL_SetKeyboardFocus(NULL);
}

/*****************************************************************************/
// SDL event functions
/*****************************************************************************/
void VIVANTE_PumpEvents(SDL_VideoDevice *_this)
{
#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Poll();
#endif
}

#endif // SDL_VIDEO_DRIVER_VIVANTE
