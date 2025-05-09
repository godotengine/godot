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

#if defined(SDL_VIDEO_DRIVER_X11)

#include "SDL_x11video.h"
#include "SDL_x11xtest.h"

static bool xtest_initialized = false;

void X11_InitXTest(SDL_VideoDevice *_this)
{
// This is currently disabled since it doesn't appear to work on XWayland
#if 0//def SDL_VIDEO_DRIVER_X11_XTEST
    Display *display = _this->internal->display;
    int event, error;
    int opcode;

    if (!SDL_X11_HAVE_XTEST ||
        !X11_XQueryExtension(display, "XTEST", &opcode, &event, &error)) {
        return;
    }

    xtest_initialized = true;
#endif
}

bool X11_XTestIsInitialized(void)
{
    return xtest_initialized;
}

bool X11_WarpMouseXTest(SDL_VideoDevice *_this, SDL_Window *window, float x, float y)
{
#ifdef SDL_VIDEO_DRIVER_X11_XTEST
    if (!X11_XTestIsInitialized()) {
        return false;
    }

    Display *display = _this->internal->display;
    SDL_DisplayData *displaydata = window ? SDL_GetDisplayDriverDataForWindow(window) : SDL_GetDisplayDriverData(SDL_GetPrimaryDisplay());
    if (!displaydata) {
        return false;
    }

    int motion_x = (int)SDL_roundf(x);
    int motion_y = (int)SDL_roundf(y);
    if (window) {
        motion_x += window->x;
        motion_y += window->y;
    }

    if (!X11_XTestFakeMotionEvent(display, displaydata->screen, motion_x, motion_y, CurrentTime)) {
        return false;
    }
    X11_XSync(display, False);

    return true;
#else
    return false;
#endif
}

#endif // SDL_VIDEO_DRIVER_X11
