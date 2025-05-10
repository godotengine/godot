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

#ifndef SDL_categories_c_h_
#define SDL_categories_c_h_

typedef enum SDL_EventCategory
{
    SDL_EVENTCATEGORY_UNKNOWN,
    SDL_EVENTCATEGORY_SYSTEM,
    SDL_EVENTCATEGORY_DISPLAY,
    SDL_EVENTCATEGORY_WINDOW,
    SDL_EVENTCATEGORY_KDEVICE,
    SDL_EVENTCATEGORY_KEY,
    SDL_EVENTCATEGORY_EDIT,
    SDL_EVENTCATEGORY_EDIT_CANDIDATES,
    SDL_EVENTCATEGORY_TEXT,
    SDL_EVENTCATEGORY_MDEVICE,
    SDL_EVENTCATEGORY_MOTION,
    SDL_EVENTCATEGORY_BUTTON,
    SDL_EVENTCATEGORY_WHEEL,
    SDL_EVENTCATEGORY_JDEVICE,
    SDL_EVENTCATEGORY_JAXIS,
    SDL_EVENTCATEGORY_JBALL,
    SDL_EVENTCATEGORY_JHAT,
    SDL_EVENTCATEGORY_JBUTTON,
    SDL_EVENTCATEGORY_JBATTERY,
    SDL_EVENTCATEGORY_GDEVICE,
    SDL_EVENTCATEGORY_GAXIS,
    SDL_EVENTCATEGORY_GBUTTON,
    SDL_EVENTCATEGORY_GTOUCHPAD,
    SDL_EVENTCATEGORY_GSENSOR,
    SDL_EVENTCATEGORY_ADEVICE,
    SDL_EVENTCATEGORY_CDEVICE,
    SDL_EVENTCATEGORY_SENSOR,
    SDL_EVENTCATEGORY_QUIT,
    SDL_EVENTCATEGORY_USER,
    SDL_EVENTCATEGORY_TFINGER,
    SDL_EVENTCATEGORY_PPROXIMITY,
    SDL_EVENTCATEGORY_PTOUCH,
    SDL_EVENTCATEGORY_PMOTION,
    SDL_EVENTCATEGORY_PBUTTON,
    SDL_EVENTCATEGORY_PAXIS,
    SDL_EVENTCATEGORY_DROP,
    SDL_EVENTCATEGORY_CLIPBOARD,
    SDL_EVENTCATEGORY_RENDER,
} SDL_EventCategory;

extern SDL_EventCategory SDL_GetEventCategory(Uint32 type);

#endif // SDL_categories_c_h_
