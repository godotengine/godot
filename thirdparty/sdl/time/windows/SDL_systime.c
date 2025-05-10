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

#ifdef SDL_TIME_WINDOWS

#include "../../core/windows/SDL_windows.h"

#include "../SDL_time_c.h"

#define NS_PER_WINDOWS_TICK   100ULL
#define WINDOWS_TICK          10000000ULL
#define UNIX_EPOCH_OFFSET_SEC 11644473600ULL

typedef void(WINAPI *pfnGetSystemTimePreciseAsFileTime)(FILETIME *);

void SDL_GetSystemTimeLocalePreferences(SDL_DateFormat *df, SDL_TimeFormat *tf)
{
    WCHAR str[80]; // Per the docs, the time and short date format strings can be a max of 80 characters.

    if (df && GetLocaleInfoW(LOCALE_USER_DEFAULT, LOCALE_SSHORTDATE, str, sizeof(str) / sizeof(WCHAR))) {
        LPWSTR s = str;
        while (*s) {
            switch (*s++) {
            case L'y':
                *df = SDL_DATE_FORMAT_YYYYMMDD;
                goto found_date;
            case L'd':
                *df = SDL_DATE_FORMAT_DDMMYYYY;
                goto found_date;
            case L'M':
                *df = SDL_DATE_FORMAT_MMDDYYYY;
                goto found_date;
            default:
                break;
            }
        }
    }

found_date:

    // Figure out the preferred system date format.
    if (tf && GetLocaleInfoW(LOCALE_USER_DEFAULT, LOCALE_STIMEFORMAT, str, sizeof(str) / sizeof(WCHAR))) {
        LPWSTR s = str;
        while (*s) {
            switch (*s++) {
            case L'H':
                *tf = SDL_TIME_FORMAT_24HR;
                return;
            case L'h':
                *tf = SDL_TIME_FORMAT_12HR;
                return;
            default:
                break;
            }
        }
    }
}

bool SDL_GetCurrentTime(SDL_Time *ticks)
{
    FILETIME ft;

    if (!ticks) {
        return SDL_InvalidParamError("ticks");
    }

    SDL_zero(ft);

    static pfnGetSystemTimePreciseAsFileTime pGetSystemTimePreciseAsFileTime = NULL;
    static bool load_attempted = false;

    // Only available in Win8/Server 2012 or higher.
    if (!pGetSystemTimePreciseAsFileTime && !load_attempted) {
        HMODULE kernel32 = GetModuleHandle(TEXT("kernel32.dll"));
        if (kernel32) {
            pGetSystemTimePreciseAsFileTime = (pfnGetSystemTimePreciseAsFileTime)GetProcAddress(kernel32, "GetSystemTimePreciseAsFileTime");
        }
        load_attempted = true;
    }

    if (pGetSystemTimePreciseAsFileTime) {
        pGetSystemTimePreciseAsFileTime(&ft);
    } else {
        GetSystemTimeAsFileTime(&ft);
    }

    *ticks = SDL_TimeFromWindows(ft.dwLowDateTime, ft.dwHighDateTime);

    return true;
}

bool SDL_TimeToDateTime(SDL_Time ticks, SDL_DateTime *dt, bool localTime)
{
    FILETIME ft, local_ft;
    SYSTEMTIME utc_st, local_st;
    SYSTEMTIME *st = NULL;
    Uint32 low, high;

    if (!dt) {
        return SDL_InvalidParamError("dt");
    }

    SDL_TimeToWindows(ticks, &low, &high);
    ft.dwLowDateTime = (DWORD)low;
    ft.dwHighDateTime = (DWORD)high;

    if (FileTimeToSystemTime(&ft, &utc_st)) {
        if (localTime) {
            if (SystemTimeToTzSpecificLocalTime(NULL, &utc_st, &local_st)) {
                // Calculate the difference for the UTC offset.
                SystemTimeToFileTime(&local_st, &local_ft);
                const SDL_Time local_ticks = SDL_TimeFromWindows(local_ft.dwLowDateTime, local_ft.dwHighDateTime);
                dt->utc_offset = (int)SDL_NS_TO_SECONDS(local_ticks - ticks);
                st = &local_st;
            }
        } else {
            dt->utc_offset = 0;
            st = &utc_st;
        }

        if (st) {
            dt->year = st->wYear;
            dt->month = st->wMonth;
            dt->day = st->wDay;
            dt->hour = st->wHour;
            dt->minute = st->wMinute;
            dt->second = st->wSecond;
            dt->nanosecond = ticks % SDL_NS_PER_SECOND;
            dt->day_of_week = st->wDayOfWeek;

            return true;
        }
    }

    return SDL_SetError("SDL_DateTime conversion failed (%lu)", GetLastError());
}

#endif // SDL_TIME_WINDOWS
