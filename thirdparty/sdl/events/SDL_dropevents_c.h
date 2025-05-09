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

#ifndef SDL_dropevents_c_h_
#define SDL_dropevents_c_h_

extern bool SDL_SendDropFile(SDL_Window *window, const char *source, const char *file);
extern bool SDL_SendDropPosition(SDL_Window *window, float x, float y);
extern bool SDL_SendDropText(SDL_Window *window, const char *text);
extern bool SDL_SendDropComplete(SDL_Window *window);

#endif // SDL_dropevents_c_h_
