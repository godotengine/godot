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

#ifdef SDL_JOYSTICK_EMSCRIPTEN
#include "../SDL_sysjoystick.h"

#include <emscripten/html5.h>

// A linked list of available joysticks
typedef struct SDL_joylist_item
{
    int index;
    char *name;
    char *mapping;
    SDL_JoystickID device_instance;
    SDL_Joystick *joystick;
    int nbuttons;
    int naxes;
    double timestamp;
    double axis[64];
    double analogButton[64];
    EM_BOOL digitalButton[64];

    struct SDL_joylist_item *next;
} SDL_joylist_item;

typedef SDL_joylist_item joystick_hwdata;

#endif // SDL_JOYSTICK_EMSCRIPTEN
