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

#ifdef SDL_TIME_N3DS

#include "../SDL_time_c.h"
#include <3ds.h>

/*
 * The 3DS clock is essentially a simple digital watch and provides
 * no timezone or DST functionality.
 */

// 3DS epoch is Jan 1 1900
#define DELTA_EPOCH_1900_OFFSET_MS 2208988800000LL

/* Returns year/month/day triple in civil calendar
 * Preconditions:  z is number of days since 1970-01-01 and is in the range:
 *                 [INT_MIN, INT_MAX-719468].
 *
 * http://howardhinnant.github.io/date_algorithms.html#civil_from_days
 */
static void civil_from_days(int days, int *year, int *month, int *day)
{
    days += 719468;
    const int era = (days >= 0 ? days : days - 146096) / 146097;
    const unsigned doe = (unsigned)(days - era * 146097);                       // [0, 146096]
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    const int y = (int)(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    const unsigned mp = (5 * doy + 2) / 153;                      // [0, 11]
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;              // [1, 31]
    const unsigned m = mp < 10 ? mp + 3 : mp - 9;                 // [1, 12]

    *year = y + (m <= 2);
    *month = (int)m;
    *day = (int)d;
}

void SDL_GetSystemTimeLocalePreferences(SDL_DateFormat *df, SDL_TimeFormat *tf)
{
    // The 3DS only has 12 supported languages, so take the standard for each
    static const SDL_DateFormat LANG_TO_DATE_FORMAT[] = {
        SDL_DATE_FORMAT_YYYYMMDD, // JP
        SDL_DATE_FORMAT_DDMMYYYY, // EN, assume non-american format
        SDL_DATE_FORMAT_DDMMYYYY, // FR
        SDL_DATE_FORMAT_DDMMYYYY, // DE
        SDL_DATE_FORMAT_DDMMYYYY, // IT
        SDL_DATE_FORMAT_DDMMYYYY, // ES
        SDL_DATE_FORMAT_YYYYMMDD, // ZH (CN)
        SDL_DATE_FORMAT_YYYYMMDD, // KR
        SDL_DATE_FORMAT_DDMMYYYY, // NL
        SDL_DATE_FORMAT_DDMMYYYY, // PT
        SDL_DATE_FORMAT_DDMMYYYY, // RU
        SDL_DATE_FORMAT_YYYYMMDD  // ZH (TW)
    };
    u8 system_language, is_north_america;
    Result result, has_region;

    if (R_FAILED(cfguInit())) {
        return;
    }
    result = CFGU_GetSystemLanguage(&system_language);
    has_region = CFGU_GetRegionCanadaUSA(&is_north_america);
    cfguExit();
    if (R_FAILED(result)) {
        return;
    }

    if (df) {
        *df = LANG_TO_DATE_FORMAT[system_language];
    }
    if (tf) {
        *tf = SDL_TIME_FORMAT_24HR;
    }

    /* Only American English (en_US) uses MM/DD/YYYY and 12hr system, this gets
       the formats wrong for canadians though (en_CA) */
    if (system_language == CFG_LANGUAGE_EN &&
        R_SUCCEEDED(has_region) && is_north_america) {
        if (df) {
            *df = SDL_DATE_FORMAT_MMDDYYYY;
        }
        if (tf) {
            *tf = SDL_TIME_FORMAT_12HR;
        }
    }
}

bool SDL_GetCurrentTime(SDL_Time *ticks)
{
    if (!ticks) {
        return SDL_InvalidParamError("ticks");
    }

    // Returns milliseconds since the epoch.
    const Uint64 ndsTicksMax = (SDL_MAX_TIME / SDL_NS_PER_MS) + DELTA_EPOCH_1900_OFFSET_MS;
    const Uint64 ndsTicks = SDL_min(osGetTime(), ndsTicksMax);

    *ticks = SDL_MS_TO_NS(ndsTicks - DELTA_EPOCH_1900_OFFSET_MS);

    return true;
}

bool SDL_TimeToDateTime(SDL_Time ticks, SDL_DateTime *dt, bool localTime)
{
    if (!dt) {
        return SDL_InvalidParamError("dt");
    }

    const int days = (int)(SDL_NS_TO_SECONDS(ticks) / SDL_SECONDS_PER_DAY);
    civil_from_days(days, &dt->year, &dt->month, &dt->day);

    int rem = (int)(SDL_NS_TO_SECONDS(ticks) - (days * SDL_SECONDS_PER_DAY));
    dt->hour = rem / (60 * 60);
    rem -= dt->hour * 60 * 60;
    dt->minute = rem / 60;
    rem -= dt->minute * 60;
    dt->second = rem;
    dt->nanosecond = ticks % SDL_NS_PER_SECOND;
    dt->utc_offset = 0; // Unknown

    SDL_CivilToDays(dt->year, dt->month, dt->day, &dt->day_of_week, NULL);

    return true;
}

#endif // SDL_TIME_N3DS
