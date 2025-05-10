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

#ifdef SDL_VIDEO_DRIVER_DUMMY

/* Dummy SDL video driver implementation; this is just enough to make an
 *  SDL-based application THINK it's got a working video driver, for
 *  applications that call SDL_Init(SDL_INIT_VIDEO) when they don't need it,
 *  and also for use as a collection of stubs when porting SDL to a new
 *  platform for which you haven't yet written a valid video driver.
 *
 * This is also a great way to determine bottlenecks: if you think that SDL
 *  is a performance problem for a given platform, enable this driver, and
 *  then see if your application runs faster without video overhead.
 *
 * Initial work by Ryan C. Gordon (icculus@icculus.org). A good portion
 *  of this was cut-and-pasted from Stephane Peter's work in the AAlib
 *  SDL video driver.  Renamed to "DUMMY" by Sam Lantinga.
 */

#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"
#ifdef SDL_INPUT_LINUXEV
#include "../../core/linux/SDL_evdev.h"
#endif

#include "SDL_nullvideo.h"
#include "SDL_nullevents_c.h"
#include "SDL_nullframebuffer_c.h"

#define DUMMYVID_DRIVER_NAME       "dummy"
#define DUMMYVID_DRIVER_EVDEV_NAME "evdev"

// Initialization/Query functions
static bool DUMMY_VideoInit(SDL_VideoDevice *_this);
static void DUMMY_VideoQuit(SDL_VideoDevice *_this);

static bool DUMMY_SetWindowPosition(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_MOVED, window->pending.x, window->pending.y);
    return true;
}

static void DUMMY_SetWindowSize(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESIZED, window->pending.w, window->pending.h);
}

// DUMMY driver bootstrap functions

static bool DUMMY_Available(const char *enable_hint)
{
    const char *hint = SDL_GetHint(SDL_HINT_VIDEO_DRIVER);
    if (hint) {
        if (SDL_strcmp(hint, enable_hint) == 0) {
            return true;
        }
    }
    return false;
}

static void DUMMY_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_free(device);
}

static SDL_VideoDevice *DUMMY_InternalCreateDevice(const char *enable_hint)
{
    SDL_VideoDevice *device;

    if (!DUMMY_Available(enable_hint)) {
        return NULL;
    }

    // Initialize all variables that we clean on shutdown
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }
    device->is_dummy = true;

    // Set the function pointers
    device->VideoInit = DUMMY_VideoInit;
    device->VideoQuit = DUMMY_VideoQuit;
    device->PumpEvents = DUMMY_PumpEvents;
    device->SetWindowSize = DUMMY_SetWindowSize;
    device->SetWindowPosition = DUMMY_SetWindowPosition;
    device->CreateWindowFramebuffer = SDL_DUMMY_CreateWindowFramebuffer;
    device->UpdateWindowFramebuffer = SDL_DUMMY_UpdateWindowFramebuffer;
    device->DestroyWindowFramebuffer = SDL_DUMMY_DestroyWindowFramebuffer;
    device->free = DUMMY_DeleteDevice;

    return device;
}

static SDL_VideoDevice *DUMMY_CreateDevice(void)
{
    return DUMMY_InternalCreateDevice(DUMMYVID_DRIVER_NAME);
}

VideoBootStrap DUMMY_bootstrap = {
    DUMMYVID_DRIVER_NAME, "SDL dummy video driver",
    DUMMY_CreateDevice,
    NULL, // no ShowMessageBox implementation
    false
};

#ifdef SDL_INPUT_LINUXEV

static void DUMMY_EVDEV_Poll(SDL_VideoDevice *_this)
{
    (void)_this;
    SDL_EVDEV_Poll();
}

static SDL_VideoDevice *DUMMY_EVDEV_CreateDevice(void)
{
    SDL_VideoDevice *device = DUMMY_InternalCreateDevice(DUMMYVID_DRIVER_EVDEV_NAME);
    if (device) {
        device->PumpEvents = DUMMY_EVDEV_Poll;
    }
    return device;
}

VideoBootStrap DUMMY_evdev_bootstrap = {
    DUMMYVID_DRIVER_EVDEV_NAME, "SDL dummy video driver with evdev",
    DUMMY_EVDEV_CreateDevice,
    NULL, // no ShowMessageBox implementation
    false
};

#endif // SDL_INPUT_LINUXEV

bool DUMMY_VideoInit(SDL_VideoDevice *_this)
{
    SDL_DisplayMode mode;

    // Use a fake 32-bpp desktop mode
    SDL_zero(mode);
    mode.format = SDL_PIXELFORMAT_XRGB8888;
    mode.w = 1024;
    mode.h = 768;
    if (SDL_AddBasicVideoDisplay(&mode) == 0) {
        return false;
    }

#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Init();
#endif

    // We're done!
    return true;
}

void DUMMY_VideoQuit(SDL_VideoDevice *_this)
{
#ifdef SDL_INPUT_LINUXEV
    SDL_EVDEV_Quit();
#endif
}

#endif // SDL_VIDEO_DRIVER_DUMMY
