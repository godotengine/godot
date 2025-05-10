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

#ifdef SDL_VIDEO_DRIVER_RPI

/* References
 * http://elinux.org/RPi_VideoCore_APIs
 * https://github.com/raspberrypi/firmware/blob/master/opt/vc/src/hello_pi/hello_triangle/triangle.c
 * http://cgit.freedesktop.org/wayland/weston/tree/src/rpi-renderer.c
 * http://cgit.freedesktop.org/wayland/weston/tree/src/compositor-rpi.c
 */

// SDL internals
#include "../SDL_sysvideo.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_keyboard_c.h"

#ifdef SDL_INPUT_LINUXEV
#include "../../core/linux/SDL_evdev.h"
#endif

// RPI declarations
#include "SDL_rpivideo.h"
#include "SDL_rpievents_c.h"
#include "SDL_rpiopengles.h"
#include "SDL_rpimouse.h"

static void RPI_Destroy(SDL_VideoDevice *device)
{
    SDL_free(device->internal);
    SDL_free(device);
}

static void RPI_GetRefreshRate(int *numerator, int *denominator)
{
    TV_DISPLAY_STATE_T tvstate;
    if (vc_tv_get_display_state(&tvstate) == 0) {
        // The width/height parameters are in the same position in the union
        // for HDMI and SDTV
        HDMI_PROPERTY_PARAM_T property;
        property.property = HDMI_PROPERTY_PIXEL_CLOCK_TYPE;
        vc_tv_hdmi_get_property(&property);
        if (property.param1 == HDMI_PIXEL_CLOCK_TYPE_NTSC) {
            *numerator = tvstate.display.hdmi.frame_rate * 1000;
            *denominator = 1001;
        } else {
            *numerator = tvstate.display.hdmi.frame_rate;
            *denominator = 1;
        }
        return;
    }

    // Failed to get display state, default to 60
    *numerator = 60;
    *denominator = 1;
}

static SDL_VideoDevice *RPI_Create(void)
{
    SDL_VideoDevice *device;
    SDL_VideoData *phdata;

    // Initialize SDL_VideoDevice structure
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }

    // Initialize internal data
    phdata = (SDL_VideoData *)SDL_calloc(1, sizeof(SDL_VideoData));
    if (!phdata) {
        SDL_free(device);
        return NULL;
    }

    device->internal = phdata;

    // Setup amount of available displays
    device->num_displays = 0;

    // Set device free function
    device->free = RPI_Destroy;

    // Setup all functions which we can handle
    device->VideoInit = RPI_VideoInit;
    device->VideoQuit = RPI_VideoQuit;
    device->CreateSDLWindow = RPI_CreateWindow;
    device->SetWindowTitle = RPI_SetWindowTitle;
    device->SetWindowPosition = RPI_SetWindowPosition;
    device->SetWindowSize = RPI_SetWindowSize;
    device->ShowWindow = RPI_ShowWindow;
    device->HideWindow = RPI_HideWindow;
    device->RaiseWindow = RPI_RaiseWindow;
    device->MaximizeWindow = RPI_MaximizeWindow;
    device->MinimizeWindow = RPI_MinimizeWindow;
    device->RestoreWindow = RPI_RestoreWindow;
    device->DestroyWindow = RPI_DestroyWindow;
    device->GL_LoadLibrary = RPI_GLES_LoadLibrary;
    device->GL_GetProcAddress = RPI_GLES_GetProcAddress;
    device->GL_UnloadLibrary = RPI_GLES_UnloadLibrary;
    device->GL_CreateContext = RPI_GLES_CreateContext;
    device->GL_MakeCurrent = RPI_GLES_MakeCurrent;
    device->GL_SetSwapInterval = RPI_GLES_SetSwapInterval;
    device->GL_GetSwapInterval = RPI_GLES_GetSwapInterval;
    device->GL_SwapWindow = RPI_GLES_SwapWindow;
    device->GL_DestroyContext = RPI_GLES_DestroyContext;
    device->GL_DefaultProfileConfig = RPI_GLES_DefaultProfileConfig;

    device->PumpEvents = RPI_PumpEvents;

    return device;
}

VideoBootStrap RPI_bootstrap = {
    "rpi",
    "RPI Video Driver",
    RPI_Create,
    NULL, // no ShowMessageBox implementation
    false
};

/*****************************************************************************/
// SDL Video and Display initialization/handling functions
/*****************************************************************************/

static void AddDispManXDisplay(const int display_id)
{
    DISPMANX_MODEINFO_T modeinfo;
    DISPMANX_DISPLAY_HANDLE_T handle;
    SDL_VideoDisplay display;
    SDL_DisplayMode mode;
    SDL_DisplayData *data;

    handle = vc_dispmanx_display_open(display_id);
    if (!handle) {
        return; // this display isn't available
    }

    if (vc_dispmanx_display_get_info(handle, &modeinfo) < 0) {
        vc_dispmanx_display_close(handle);
        return;
    }

    // RPI_GetRefreshRate() doesn't distinguish between displays. I'm not sure the hardware distinguishes either
    SDL_zero(mode);
    mode.w = modeinfo.width;
    mode.h = modeinfo.height;
    RPI_GetRefreshRate(&mode.refresh_rate_numerator, &mode.refresh_rate_denominator);

    // 32 bpp for default
    mode.format = SDL_PIXELFORMAT_ABGR8888;

    SDL_zero(display);
    display.desktop_mode = mode;

    // Allocate display internal data
    data = (SDL_DisplayData *)SDL_calloc(1, sizeof(SDL_DisplayData));
    if (!data) {
        vc_dispmanx_display_close(handle);
        return; // oh well
    }

    data->dispman_display = handle;

    display.internal = data;

    SDL_AddVideoDisplay(&display, false);
}

bool RPI_VideoInit(SDL_VideoDevice *_this)
{
    // Initialize BCM Host
    bcm_host_init();

    AddDispManXDisplay(DISPMANX_ID_MAIN_LCD);    // your default display
    AddDispManXDisplay(DISPMANX_ID_FORCE_OTHER); // an "other" display...maybe DSI-connected screen while HDMI is your main

#ifdef SDL_INPUT_LINUXEV
    if (!SDL_EVDEV_Init()) {
        return false;
    }
#endif

    RPI_InitMouse(_this);

    return true;
}

void RPI_VideoQuit(SDL_VideoDevice *_this)
{
#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Quit();
#endif
}

static void RPI_vsync_callback(DISPMANX_UPDATE_HANDLE_T u, void *data)
{
    SDL_WindowData *wdata = (SDL_WindowData *)data;

    SDL_LockMutex(wdata->vsync_cond_mutex);
    SDL_SignalCondition(wdata->vsync_cond);
    SDL_UnlockMutex(wdata->vsync_cond_mutex);
}

bool RPI_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    SDL_WindowData *wdata;
    SDL_VideoDisplay *display;
    SDL_DisplayData *displaydata;
    VC_RECT_T dst_rect;
    VC_RECT_T src_rect;
    VC_DISPMANX_ALPHA_T dispman_alpha;
    DISPMANX_UPDATE_HANDLE_T dispman_update;
    uint32_t layer = SDL_RPI_VIDEOLAYER;
    const char *env;

    // Disable alpha, otherwise the app looks composed with whatever dispman is showing (X11, console,etc)
    dispman_alpha.flags = DISPMANX_FLAGS_ALPHA_FIXED_ALL_PIXELS;
    dispman_alpha.opacity = 0xFF;
    dispman_alpha.mask = 0;

    // Allocate window internal data
    wdata = (SDL_WindowData *)SDL_calloc(1, sizeof(SDL_WindowData));
    if (!wdata) {
        return false;
    }
    display = SDL_GetVideoDisplayForWindow(window);
    displaydata = display->internal;

    // Windows have one size for now
    window->w = display->desktop_mode.w;
    window->h = display->desktop_mode.h;

    // OpenGL ES is the law here, buddy
    window->flags |= SDL_WINDOW_OPENGL;

    // Create a dispman element and associate a window to it
    dst_rect.x = 0;
    dst_rect.y = 0;
    dst_rect.width = window->w;
    dst_rect.height = window->h;

    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.width = window->w << 16;
    src_rect.height = window->h << 16;

    env = SDL_GetHint(SDL_HINT_RPI_VIDEO_LAYER);
    if (env) {
        layer = SDL_atoi(env);
    }

    dispman_update = vc_dispmanx_update_start(0);
    wdata->dispman_window.element = vc_dispmanx_element_add(dispman_update,
                                                            displaydata->dispman_display,
                                                            layer /* layer */,
                                                            &dst_rect,
                                                            0 /*src*/,
                                                            &src_rect,
                                                            DISPMANX_PROTECTION_NONE,
                                                            &dispman_alpha /*alpha*/,
                                                            0 /*clamp*/,
                                                            0 /*transform*/);
    wdata->dispman_window.width = window->w;
    wdata->dispman_window.height = window->h;
    vc_dispmanx_update_submit_sync(dispman_update);

    if (!_this->egl_data) {
        if (!SDL_GL_LoadLibrary(NULL)) {
            return false;
        }
    }
    wdata->egl_surface = SDL_EGL_CreateSurface(_this, window, (NativeWindowType)&wdata->dispman_window);

    if (wdata->egl_surface == EGL_NO_SURFACE) {
        return SDL_SetError("Could not create GLES window surface");
    }

    // Start generating vsync callbacks if necessary
    wdata->double_buffer = false;
    if (SDL_GetHintBoolean(SDL_HINT_VIDEO_DOUBLE_BUFFER, false)) {
        wdata->vsync_cond = SDL_CreateCondition();
        wdata->vsync_cond_mutex = SDL_CreateMutex();
        wdata->double_buffer = true;
        vc_dispmanx_vsync_callback(displaydata->dispman_display, RPI_vsync_callback, (void *)wdata);
    }

    // Setup driver data for this window
    window->internal = wdata;

    // One window, it always has focus
    SDL_SetMouseFocus(window);
    SDL_SetKeyboardFocus(window);

    // Window has been successfully created
    return true;
}

void RPI_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    SDL_DisplayData *displaydata = SDL_GetDisplayDriverDataForWindow(window);

    if (data) {
        if (data->double_buffer) {
            // Wait for vsync, and then stop vsync callbacks and destroy related stuff, if needed
            SDL_LockMutex(data->vsync_cond_mutex);
            SDL_WaitCondition(data->vsync_cond, data->vsync_cond_mutex);
            SDL_UnlockMutex(data->vsync_cond_mutex);

            vc_dispmanx_vsync_callback(displaydata->dispman_display, NULL, NULL);

            SDL_DestroyCondition(data->vsync_cond);
            SDL_DestroyMutex(data->vsync_cond_mutex);
        }

#ifdef SDL_VIDEO_OPENGL_EGL
        if (data->egl_surface != EGL_NO_SURFACE) {
            SDL_EGL_DestroySurface(_this, data->egl_surface);
        }
#endif
        SDL_free(data);
        window->internal = NULL;
    }
}

void RPI_SetWindowTitle(SDL_VideoDevice *_this, SDL_Window *window)
{
}
bool RPI_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window)
{
    return SDL_Unsupported();
}
void RPI_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void RPI_ShowWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void RPI_HideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void RPI_RaiseWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void RPI_MaximizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void RPI_MinimizeWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}
void RPI_RestoreWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
}

#endif // SDL_VIDEO_DRIVER_RPI
