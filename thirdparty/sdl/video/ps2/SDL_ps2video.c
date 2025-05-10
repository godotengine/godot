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

#ifdef SDL_VIDEO_DRIVER_PS2

/* PS2 SDL video driver implementation; this is just enough to make an
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
 *  SDL video driver.  Renamed to "PS2" by Sam Lantinga.
 */

#include "../SDL_sysvideo.h"
#include "../SDL_pixels_c.h"
#include "../../events/SDL_events_c.h"

#include "SDL_ps2video.h"

// PS2 driver bootstrap functions

static bool PS2_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode)
{
    return true;
}

static void PS2_DeleteDevice(SDL_VideoDevice *device)
{
    SDL_free(device);
}

static bool PS2_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props)
{
    SDL_SetKeyboardFocus(window);

    // Window has been successfully created
    return true;
}

static bool PS2_VideoInit(SDL_VideoDevice *_this)
{
    SDL_DisplayMode mode;

    SDL_zero(mode);
    mode.w = 640;
    mode.h = 480;
    mode.refresh_rate = 60.0f;

    // 32 bpp for default
    mode.format = SDL_PIXELFORMAT_ABGR8888;

    SDL_AddBasicVideoDisplay(&mode);

    return true;
}

static void PS2_VideoQuit(SDL_VideoDevice *_this)
{
}

static void PS2_PumpEvents(SDL_VideoDevice *_this)
{
    // do nothing.
}

static SDL_VideoDevice *PS2_CreateDevice(void)
{
    SDL_VideoDevice *device;

    // Initialize all variables that we clean on shutdown
    device = (SDL_VideoDevice *)SDL_calloc(1, sizeof(SDL_VideoDevice));
    if (!device) {
        return NULL;
    }

    // Set the function pointers
    device->VideoInit = PS2_VideoInit;
    device->VideoQuit = PS2_VideoQuit;
    device->SetDisplayMode = PS2_SetDisplayMode;
    device->CreateSDLWindow = PS2_CreateWindow;
    device->PumpEvents = PS2_PumpEvents;
    device->free = PS2_DeleteDevice;

    return device;
}

VideoBootStrap PS2_bootstrap = {
    "ps2",
    "PS2 Video Driver",
    PS2_CreateDevice,
    NULL, // no ShowMessageBox implementation
    false
};

#endif // SDL_VIDEO_DRIVER_PS2
