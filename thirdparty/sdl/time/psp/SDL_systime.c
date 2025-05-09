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

#ifdef SDL_TIME_PSP

#include <psptypes.h>
#include <psprtc.h>
#include <psputility_sysparam.h>

#include "../SDL_time_c.h"

// Sony seems to use 0001-01-01T00:00:00 as an epoch.
#define DELTA_EPOCH_0001_OFFSET 62135596800ULL

void SDL_GetSystemTimeLocalePreferences(SDL_DateFormat *df, SDL_TimeFormat *tf)
{
    int val;

    if (df && sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_DATE_FORMAT, &val) == 0) {
        switch (val) {
        case PSP_SYSTEMPARAM_DATE_FORMAT_YYYYMMDD:
            *df = SDL_DATE_FORMAT_YYYYMMDD;
            break;
        case PSP_SYSTEMPARAM_DATE_FORMAT_MMDDYYYY:
            *df = SDL_DATE_FORMAT_MMDDYYYY;
            break;
        case PSP_SYSTEMPARAM_DATE_FORMAT_DDMMYYYY:
            *df = SDL_DATE_FORMAT_DDMMYYYY;
            break;
        default:
            break;
        }
    }

    if (tf && sceUtilityGetSystemParamInt(PSP_SYSTEMPARAM_ID_INT_TIME_FORMAT, &val) == 0) {
        switch (val) {
        case PSP_SYSTEMPARAM_TIME_FORMAT_24HR:
            *tf = SDL_TIME_FORMAT_24HR;
            break;
        case PSP_SYSTEMPARAM_TIME_FORMAT_12HR:
            *tf = SDL_TIME_FORMAT_12HR;
            break;
        default:
            break;
        }
    }
}

bool SDL_GetCurrentTime(SDL_Time *ticks)
{
    u64 sceTicks;

    if (!ticks) {
        return SDL_InvalidParamError("ticks");
    }

    const int ret = sceRtcGetCurrentTick(&sceTicks);
    if (!ret) {
        const u32 res = sceRtcGetTickResolution();
        const u32 div = SDL_NS_PER_SECOND / res;
        const Uint64 epoch_offset = DELTA_EPOCH_0001_OFFSET * res;

        const Uint64 scetime_min = (Uint64)((SDL_MIN_TIME / div) + epoch_offset);
        const Uint64 scetime_max = (Uint64)((SDL_MAX_TIME / div) + epoch_offset);

        // Clamp to the valid SDL_Time range.
        sceTicks = SDL_clamp(sceTicks, scetime_min, scetime_max);

        *ticks = (SDL_Time)(sceTicks - epoch_offset) * div;

        return true;
    }

    return SDL_SetError("Failed to retrieve system time (%i)", ret);
}

bool SDL_TimeToDateTime(SDL_Time ticks, SDL_DateTime *dt, bool localTime)
{
    ScePspDateTime t;
    u64 local;
    int ret = 0;

    if (!dt) {
        return SDL_InvalidParamError("dt");
    }

    const u32 res = sceRtcGetTickResolution();
    const u32 div = (SDL_NS_PER_SECOND / res);
    const u64 sceTicks = (u64)((ticks / div) + (DELTA_EPOCH_0001_OFFSET * div));

    if (localTime) {
        ret = sceRtcConvertUtcToLocalTime(&sceTicks, &local);
    } else {
        local = sceTicks;
    }

    if (!ret) {
        ret = sceRtcSetTick(&t, &local);
        if (!ret) {
            dt->year = t.year;
            dt->month = t.month;
            dt->day = t.day;
            dt->hour = t.hour;
            dt->minute = t.minute;
            dt->second = t.second;
            dt->nanosecond = ticks % SDL_NS_PER_SECOND;
            dt->utc_offset = (int)(((Sint64)local - (Sint64)sceTicks) / (Sint64)res);

            SDL_CivilToDays(dt->year, dt->month, dt->day, &dt->day_of_week, NULL);

            return true;
        }
    }

    return SDL_SetError("Local time conversion failed (%i)", ret);
}

#endif // SDL_TIME_PSP
