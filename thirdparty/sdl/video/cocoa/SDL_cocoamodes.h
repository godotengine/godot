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

#ifndef SDL_cocoamodes_h_
#define SDL_cocoamodes_h_

struct SDL_DisplayData
{
    CGDirectDisplayID display;
};

struct SDL_DisplayModeData
{
    CFMutableArrayRef modes;
};

extern void Cocoa_InitModes(SDL_VideoDevice *_this);
extern void Cocoa_UpdateDisplays(SDL_VideoDevice *_this);
extern bool Cocoa_GetDisplayBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect);
extern bool Cocoa_GetDisplayUsableBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect);
extern bool Cocoa_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display);
extern bool Cocoa_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode);
extern void Cocoa_QuitModes(SDL_VideoDevice *_this);
extern SDL_VideoDisplay *Cocoa_FindSDLDisplayByCGDirectDisplayID(SDL_VideoDevice *_this, CGDirectDisplayID displayid);

#endif // SDL_cocoamodes_h_
