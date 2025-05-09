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
#include "../../SDL_internal.h"

#ifndef SDL_x11pen_h_
#define SDL_x11pen_h_

// Pressure-sensitive pen support for X11.

#include "SDL_x11video.h"
#include "../../events/SDL_pen_c.h"

// Prep pen support (never fails; pens simply won't be added if there's a problem).
extern void X11_InitPen(SDL_VideoDevice *_this);

// Clean up pen support.
extern void X11_QuitPen(SDL_VideoDevice *_this);

#ifdef SDL_VIDEO_DRIVER_X11_XINPUT2

// Forward definition for SDL_x11video.h
struct SDL_VideoData;

#define SDL_X11_PEN_AXIS_VALUATOR_MISSING -1

typedef struct X11_PenHandle
{
    SDL_PenID pen;
    bool is_eraser;
    int x11_deviceid;
    int valuator_for_axis[SDL_PEN_AXIS_COUNT];
    float slider_bias;      // shift value to add to PEN_AXIS_SLIDER (before normalisation)
    float rotation_bias;    // rotation to add to PEN_AXIS_ROTATION  (after normalisation)
    float axis_min[SDL_PEN_AXIS_COUNT];
    float axis_max[SDL_PEN_AXIS_COUNT];
} X11_PenHandle;

// Converts XINPUT2 valuators into pen axis information, including normalisation.
extern void X11_PenAxesFromValuators(const X11_PenHandle *pen,
                                     const double *input_values, const unsigned char *mask, int mask_len,
                                     float axis_values[SDL_PEN_AXIS_COUNT]);

// Add a pen (if this function's further checks validate it).
extern X11_PenHandle *X11_MaybeAddPenByDeviceID(SDL_VideoDevice *_this, int deviceid);

// Remove a pen. It's okay if deviceid is bogus or not a pen, we'll check it.
extern void X11_RemovePenByDeviceID(int deviceid);

// Map X11 device ID to pen ID.
extern X11_PenHandle *X11_FindPenByDeviceID(int deviceid);

#endif // SDL_VIDEO_DRIVER_X11_XINPUT2

#endif // SDL_x11pen_h_
