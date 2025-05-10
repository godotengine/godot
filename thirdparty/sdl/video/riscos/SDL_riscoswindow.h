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

#ifndef SDL_riscoswindow_h_
#define SDL_riscoswindow_h_

#include "SDL_riscosdefs.h"

struct SDL_WindowData
{
    SDL_Window *window;
    sprite_area *fb_area;
    sprite_header *fb_sprite;
};

extern bool RISCOS_CreateWindow(SDL_VideoDevice *_this, SDL_Window *window, SDL_PropertiesID create_props);
extern void RISCOS_DestroyWindow(SDL_VideoDevice *_this, SDL_Window *window);

#endif // SDL_riscoswindow_h_
