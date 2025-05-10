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

#ifndef SDL_BMODES_H
#define SDL_BMODES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "../SDL_sysvideo.h"

extern SDL_PixelFormat HAIKU_ColorSpaceToSDLPxFormat(uint32 colorspace);

extern bool HAIKU_InitModes(SDL_VideoDevice *_this);
extern void HAIKU_QuitModes(SDL_VideoDevice *_this);
extern bool HAIKU_GetDisplayBounds(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_Rect *rect);
extern bool HAIKU_GetDisplayModes(SDL_VideoDevice *_this, SDL_VideoDisplay *display);
extern bool HAIKU_SetDisplayMode(SDL_VideoDevice *_this, SDL_VideoDisplay *display, SDL_DisplayMode *mode);

#ifdef __cplusplus
}
#endif

#endif
