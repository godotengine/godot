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

#ifdef SDL_TIME_PS2

#include "../SDL_time_c.h"

// PS2 epoch is Jan 1 2000 JST (UTC +9)
#define UNIX_EPOCH_OFFSET_SEC 946717200

// TODO: Implement this...
void SDL_GetSystemTimeLocalePreferences(SDL_DateFormat *df, SDL_TimeFormat *tf)
{
}

bool SDL_GetCurrentTime(SDL_Time *ticks)
{
    if (!ticks) {
        return SDL_InvalidParamError("ticks");
    }

    *ticks = 0;

    return true;
}

bool SDL_TimeToDateTime(SDL_Time ticks, SDL_DateTime *dt, bool localTime)
{
    if (!dt) {
        return SDL_InvalidParamError("dt");
    }

    dt->year = 1970;
    dt->month = 1;
    dt->day = 1;
    dt->hour = 0;
    dt->minute = 0;
    dt->second = 0;
    dt->nanosecond = 0;
    dt->day_of_week = 4;
    dt->utc_offset = 0;

    return true;
}

#endif // SDL_TIME_PS2
