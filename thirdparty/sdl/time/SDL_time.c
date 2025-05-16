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

#include "SDL_time_c.h"

/* The following algorithms are based on those of Howard Hinnant and are in the public domain.
 *
 * http://howardhinnant.github.io/date_algorithms.html
 */

/* Given a calendar date, returns days since Jan 1 1970, and optionally
 * the day of the week [0-6, 0 is Sunday] and day of the year [0-365].
 */
Sint64 SDL_CivilToDays(int year, int month, int day, int *day_of_week, int *day_of_year)
{

    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yoe = (unsigned)(year - era * 400);                                  // [0, 399]
    const unsigned doy = (153 * (month > 2 ? month - 3 : month + 9) + 2) / 5 + day - 1; // [0, 365]
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;                         // [0, 146096]
    const Sint64 z = (Sint64)(era) * 146097 + (Sint64)(doe)-719468;

    if (day_of_week) {
        *day_of_week = (int)(z >= -4 ? (z + 4) % 7 : (z + 5) % 7 + 6);
    }
    if (day_of_year) {
        // This algorithm considers March 1 to be the first day of the year, so offset by Jan + Feb.
        if (doy > 305) {
            // Day 0 is the first day of the year.
            *day_of_year = doy - 306;
        } else {
            const int doy_offset = 59 + (!(year % 4) && ((year % 100) || !(year % 400)));
            *day_of_year = doy + doy_offset;
        }
    }

    return z;
}

bool SDL_GetDateTimeLocalePreferences(SDL_DateFormat *dateFormat, SDL_TimeFormat *timeFormat)
{
    // Default to ISO 8061 date format, as it is unambiguous, and 24 hour time.
    if (dateFormat) {
        *dateFormat = SDL_DATE_FORMAT_YYYYMMDD;
    }
    if (timeFormat) {
        *timeFormat = SDL_TIME_FORMAT_24HR;
    }

    SDL_GetSystemTimeLocalePreferences(dateFormat, timeFormat);

    return true;
}

int SDL_GetDaysInMonth(int year, int month)
{
    static const int DAYS_IN_MONTH[] = {
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    };

    if (month < 1 || month > 12) {
        SDL_SetError("Month out of range [1-12], requested: %i", month);
        return -1;
    }

    int days = DAYS_IN_MONTH[month - 1];

    /* A leap year occurs every 4 years...
     * but not every 100 years...
     * except for every 400 years.
     */
    if (month == 2 && (!(year % 4) && ((year % 100) || !(year % 400)))) {
        ++days;
    }

    return days;
}

int SDL_GetDayOfYear(int year, int month, int day)
{
    int dayOfYear;

    if (month < 1 || month > 12) {
        SDL_SetError("Month out of range [1-12], requested: %i", month);
        return -1;
    }
    if (day < 1 || day > SDL_GetDaysInMonth(year, month)) {
        SDL_SetError("Day out of range [1-%i], requested: %i", SDL_GetDaysInMonth(year, month), month);
        return -1;
    }

    SDL_CivilToDays(year, month, day, NULL, &dayOfYear);
    return dayOfYear;
}

int SDL_GetDayOfWeek(int year, int month, int day)
{
    int dayOfWeek;

    if (month < 1 || month > 12) {
        SDL_SetError("Month out of range [1-12], requested: %i", month);
        return -1;
    }
    if (day < 1 || day > SDL_GetDaysInMonth(year, month)) {
        SDL_SetError("Day out of range [1-%i], requested: %i", SDL_GetDaysInMonth(year, month), month);
        return -1;
    }

    SDL_CivilToDays(year, month, day, &dayOfWeek, NULL);
    return dayOfWeek;
}

static bool SDL_DateTimeIsValid(const SDL_DateTime *dt)
{
    if (dt->month < 1 || dt->month > 12) {
        SDL_SetError("Malformed SDL_DateTime: month out of range [1-12], current: %i", dt->month);
        return false;
    }

    const int daysInMonth = SDL_GetDaysInMonth(dt->year, dt->month);
    if (dt->day < 1 || dt->day > daysInMonth) {
        SDL_SetError("Malformed SDL_DateTime: day of month out of range [1-%i], current: %i", daysInMonth, dt->month);
        return false;
    }
    if (dt->hour < 0 || dt->hour > 23) {
        SDL_SetError("Malformed SDL_DateTime: hour out of range [0-23], current: %i", dt->hour);
        return false;
    }
    if (dt->minute < 0 || dt->minute > 59) {
        SDL_SetError("Malformed SDL_DateTime: minute out of range [0-59], current: %i", dt->minute);
        return false;
    }
    if (dt->second < 0 || dt->second > 60) {
        SDL_SetError("Malformed SDL_DateTime: second out of range [0-60], current: %i", dt->second);
        return false; // 60 accounts for a possible leap second.
    }
    if (dt->nanosecond < 0 || dt->nanosecond >= SDL_NS_PER_SECOND) {
        SDL_SetError("Malformed SDL_DateTime: nanosecond out of range [0-999999999], current: %i", dt->nanosecond);
        return false;
    }

    return true;
}

bool SDL_DateTimeToTime(const SDL_DateTime *dt, SDL_Time *ticks)
{
    static const Sint64 max_seconds = SDL_NS_TO_SECONDS(SDL_MAX_TIME) - 1;
    static const Sint64 min_seconds = SDL_NS_TO_SECONDS(SDL_MIN_TIME) + 1;
    bool result = true;

    if (!dt) {
        return SDL_InvalidParamError("dt");
    }
    if (!ticks) {
        return SDL_InvalidParamError("ticks");
    }
    if (!SDL_DateTimeIsValid(dt)) {
        // The validation function sets the error string.
        return false;
    }

    *ticks = SDL_CivilToDays(dt->year, dt->month, dt->day, NULL, NULL) * SDL_SECONDS_PER_DAY;
    *ticks += (((dt->hour * 60) + dt->minute) * 60) + dt->second - dt->utc_offset;
    if (*ticks > max_seconds || *ticks < min_seconds) {
        *ticks = SDL_clamp(*ticks, min_seconds, max_seconds);
        result = SDL_SetError("Date out of range for SDL_Time representation; SDL_Time value clamped");
    }
    *ticks = SDL_SECONDS_TO_NS(*ticks) + dt->nanosecond;

    return result;
}

#define DELTA_EPOCH_1601_100NS (11644473600ll * 10000000ll) // [100 ns] (100 ns units between 1601-01-01 and 1970-01-01, 11644473600 seconds)

void SDL_TimeToWindows(SDL_Time ticks, Uint32 *dwLowDateTime, Uint32 *dwHighDateTime)
{
    /* Convert nanoseconds to Win32 ticks.
     * SDL_Time has a range of roughly 292 years, so even SDL_MIN_TIME can't underflow the Win32 epoch.
     */
    const Uint64 wtime = (Uint64)((ticks / 100) + DELTA_EPOCH_1601_100NS);

    if (dwLowDateTime) {
        *dwLowDateTime = (Uint32)wtime;
    }

    if (dwHighDateTime) {
        *dwHighDateTime = (Uint32)(wtime >> 32);
    }
}

SDL_Time SDL_TimeFromWindows(Uint32 dwLowDateTime, Uint32 dwHighDateTime)
{
    static const Uint64 wintime_min = (Uint64)((SDL_MIN_TIME / 100) + DELTA_EPOCH_1601_100NS);
    static const Uint64 wintime_max = (Uint64)((SDL_MAX_TIME / 100) + DELTA_EPOCH_1601_100NS);

    Uint64 wtime = (((Uint64)dwHighDateTime << 32) | dwLowDateTime);

    // Clamp the windows time range to the SDL_Time min/max
    wtime = SDL_clamp(wtime, wintime_min, wintime_max);

    return (SDL_Time)(wtime - DELTA_EPOCH_1601_100NS) * 100;
}
