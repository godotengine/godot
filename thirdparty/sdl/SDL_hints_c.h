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

// This file defines useful function for working with SDL hints

#ifndef SDL_hints_c_h_
#define SDL_hints_c_h_

extern void SDL_InitHints(void);
extern bool SDL_GetStringBoolean(const char *value, bool default_value);
extern int SDL_GetStringInteger(const char *value, int default_value);
extern void SDL_QuitHints(void);

#endif // SDL_hints_c_h_
