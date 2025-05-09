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

#ifndef SDL_RPI_mouse_h_
#define SDL_RPI_mouse_h_

#include "../SDL_sysvideo.h"

struct SDL_CursorData
{
    DISPMANX_RESOURCE_HANDLE_T resource;
    DISPMANX_ELEMENT_HANDLE_T element;
    int hot_x, hot_y;
    int w, h;
};

#define SDL_RPI_CURSORDATA(curs) RPI_CursorData *curdata = (RPI_CursorData *)((curs) ? (curs)->internal : NULL)

extern void RPI_InitMouse(SDL_VideoDevice *_this);
extern void RPI_QuitMouse(SDL_VideoDevice *_this);

#endif // SDL_RPI_mouse_h_
