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

#ifndef SDL_x11mouse_h_
#define SDL_x11mouse_h_

typedef struct SDL_XInput2DeviceInfo
{
    int device_id;
    bool relative[2];
    double minval[2];
    double maxval[2];
    double prev_coords[2];
    struct SDL_XInput2DeviceInfo *next;
} SDL_XInput2DeviceInfo;

extern void X11_InitMouse(SDL_VideoDevice *_this);
extern void X11_QuitMouse(SDL_VideoDevice *_this);
extern void X11_SetHitTestCursor(SDL_HitTestResult rc);

#endif // SDL_x11mouse_h_
