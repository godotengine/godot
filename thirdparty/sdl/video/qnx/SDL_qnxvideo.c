/*
  Simple DirectMedia Layer
  Copyright (C) 2017 BlackBerry Limited

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
#include "../../SDL_internal.h"
#include "../SDL_sysvideo.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_mouse_c.h"
#include "SDL_qnx.h"

static screen_context_t context;
static screen_event_t   event;

/**
 * Initializes the QNX video plugin.
 * Creates the Screen context and event handles used for all window operations
 * by the plugin.
 * @param   SDL_VideoDevice *_this
 * @return  0 if successful, -1 on error
 */
static bool videoInit(SDL_VideoDevice *_this)
{
    SDL_VideoDisplay display;

    if (screen_create_context(&context, 0) < 0) {
        return false;
    }

    if (screen_create_event(&event) < 0) {
        return false;
    }

    SDL_zero(display);

    if (SDL_AddVideoDisplay(&display, false) == 0) {
        return false;
    }

    // Assume we have a mouse and keyboard
    SDL_AddKeyboard(SDL_DEFAULT_KEYBOARD_ID, NULL, false);
    SDL_AddMouse(SDL_DEFAULT_MOUSE_ID, NULL, false);

    return true;
}

static void videoQuit(SDL_VideoDevice *_this)
{
}

/**
 * Creates a new native Screen window and associates it with the given SDL
 * window.
 * @param   SDL_VideoDevice *_this
 * @param   window  SDL window to initialize
 * @return  0 if successful, -1 on error
 */
static bool createWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    window_impl_t   *impl;
    int             size[2];
    int             numbufs;
    int             format;
    int             usage;

    impl = SDL_calloc(1, sizeof(*impl));
    if (!impl) {
        return false;
    }

    // Create a native window.
    if (screen_create_window(&impl->window, context) < 0) {
        goto fail;
    }

    // Set the native window's size to match the SDL window.
    size[0] = window->w;
    size[1] = window->h;

    if (screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_SIZE,
                                      size) < 0) {
        goto fail;
    }

    if (screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_SOURCE_SIZE,
                                      size) < 0) {
        goto fail;
    }

    // Create window buffer(s).
    if (window->flags & SDL_WINDOW_OPENGL) {
        if (glGetConfig(&impl->conf, &format) < 0) {
            goto fail;
        }
        numbufs = 2;

        usage = SCREEN_USAGE_OPENGL_ES2;
        if (screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_USAGE,
                                          &usage) < 0) {
            return false;
        }
    } else {
        format = SCREEN_FORMAT_RGBX8888;
        numbufs = 1;
    }

    // Set pixel format.
    if (screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_FORMAT,
                                      &format) < 0) {
        goto fail;
    }

    // Create buffer(s).
    if (screen_create_window_buffers(impl->window, numbufs) < 0) {
        goto fail;
    }

    window->internal = impl;
    return true;

fail:
    if (impl->window) {
        screen_destroy_window(impl->window);
    }

    SDL_free(impl);
    return false;
}

/**
 * Gets a pointer to the Screen buffer associated with the given window. Note
 * that the buffer is actually created in createWindow().
 * @param       SDL_VideoDevice *_this
 * @param       window  SDL window to get the buffer for
 * @param[out]  pixles  Holds a pointer to the window's buffer
 * @param[out]  format  Holds the pixel format for the buffer
 * @param[out]  pitch   Holds the number of bytes per line
 * @return  0 if successful, -1 on error
 */
static bool createWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window * window, SDL_PixelFormat * format,
                        void ** pixels, int *pitch)
{
    window_impl_t   *impl = (window_impl_t *)window->internal;
    screen_buffer_t buffer;

    // Get a pointer to the buffer's memory.
    if (screen_get_window_property_pv(impl->window, SCREEN_PROPERTY_BUFFERS,
                                      (void **)&buffer) < 0) {
        return false;
    }

    if (screen_get_buffer_property_pv(buffer, SCREEN_PROPERTY_POINTER,
                                      pixels) < 0) {
        return false;
    }

    // Set format and pitch.
    if (screen_get_buffer_property_iv(buffer, SCREEN_PROPERTY_STRIDE,
                                      pitch) < 0) {
        return false;
    }

    *format = SDL_PIXELFORMAT_XRGB8888;
    return true;
}

/**
 * Informs the window manager that the window needs to be updated.
 * @param   SDL_VideoDevice *_this
 * @param   window      The window to update
 * @param   rects       An array of reectangular areas to update
 * @param   numrects    Rect array length
 * @return  0 if successful, -1 on error
 */
static bool updateWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window, const SDL_Rect *rects,
                        int numrects)
{
    window_impl_t   *impl = (window_impl_t *)window->internal;
    screen_buffer_t buffer;

    if (screen_get_window_property_pv(impl->window, SCREEN_PROPERTY_BUFFERS,
                                      (void **)&buffer) < 0) {
        return false;
    }

    screen_post_window(impl->window, buffer, numrects, (int *)rects, 0);
    screen_flush_context(context, 0);
    return true;
}

/**
 * Runs the main event loop.
 * @param   SDL_VideoDevice *_this
 */
static void pumpEvents(SDL_VideoDevice *_this)
{
    int             type;

    for (;;) {
        if (screen_get_event(context, event, 0) < 0) {
            break;
        }

        if (screen_get_event_property_iv(event, SCREEN_PROPERTY_TYPE, &type)
            < 0) {
            break;
        }

        if (type == SCREEN_EVENT_NONE) {
            break;
        }

        switch (type) {
        case SCREEN_EVENT_KEYBOARD:
            handleKeyboardEvent(event);
            break;

        default:
            break;
        }
    }
}

/**
 * Updates the size of the native window using the geometry of the SDL window.
 * @param   SDL_VideoDevice *_this
 * @param   window  SDL window to update
 */
static void setWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    window_impl_t   *impl = (window_impl_t *)window->internal;
    int             size[2];

    size[0] = window->pending.w;
    size[1] = window->pending.h;

    screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_SIZE, size);
    screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_SOURCE_SIZE, size);
}

/**
 * Makes the native window associated with the given SDL window visible.
 * @param   SDL_VideoDevice *_this
 * @param   window  SDL window to update
 */
static void showWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    window_impl_t   *impl = (window_impl_t *)window->internal;
    const int       visible = 1;

    screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_VISIBLE,
                                  &visible);
}

/**
 * Makes the native window associated with the given SDL window invisible.
 * @param   SDL_VideoDevice *_this
 * @param   window  SDL window to update
 */
static void hideWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    window_impl_t   *impl = (window_impl_t *)window->internal;
    const int       visible = 0;

    screen_set_window_property_iv(impl->window, SCREEN_PROPERTY_VISIBLE,
        &visible);
}

/**
 * Destroys the native window associated with the given SDL window.
 * @param   SDL_VideoDevice *_this
 * @param   window  SDL window that is being destroyed
 */
static void destroyWindow(SDL_VideoDevice *_this, SDL_Window *window)
{
    window_impl_t   *impl = (window_impl_t *)window->internal;

    if (impl) {
        screen_destroy_window(impl->window);
        window->internal = NULL;
    }
}

/**
 * Frees the plugin object created by createDevice().
 * @param   device  Plugin object to free
 */
static void deleteDevice(SDL_VideoDevice *device)
{
    SDL_free(device);
}

/**
 * Creates the QNX video plugin used by SDL.
 * @return  Initialized device if successful, NULL otherwise
 */
static SDL_VideoDevice *createDevice(void)
{
    SDL_VideoDevice *device;

    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }

    device->internal = NULL;
    device->VideoInit = videoInit;
    device->VideoQuit = videoQuit;
    device->CreateSDLWindow = createWindow;
    device->CreateWindowFramebuffer = createWindowFramebuffer;
    device->UpdateWindowFramebuffer = updateWindowFramebuffer;
    device->SetWindowSize = setWindowSize;
    device->ShowWindow = showWindow;
    device->HideWindow = hideWindow;
    device->PumpEvents = pumpEvents;
    device->DestroyWindow = destroyWindow;

    device->GL_LoadLibrary = glLoadLibrary;
    device->GL_GetProcAddress = glGetProcAddress;
    device->GL_CreateContext = glCreateContext;
    device->GL_SetSwapInterval = glSetSwapInterval;
    device->GL_SwapWindow = glSwapWindow;
    device->GL_MakeCurrent = glMakeCurrent;
    device->GL_DestroyContext = glDeleteContext;
    device->GL_UnloadLibrary = glUnloadLibrary;

    device->free = deleteDevice;
    return device;
}

VideoBootStrap QNX_bootstrap = {
    "qnx", "QNX Screen",
    createDevice,
    NULL, // no ShowMessageBox implementation
    false
};
