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

#ifdef SDL_TIME_UNIX

#include "../SDL_time_c.h"
#include <errno.h>
#include <langinfo.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#if !defined(HAVE_CLOCK_GETTIME) && defined(SDL_PLATFORM_APPLE)
#include <mach/clock.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

void SDL_GetSystemTimeLocalePreferences(SDL_DateFormat *df, SDL_TimeFormat *tf)
{
    /* This *should* be well-supported aside from very old legacy systems, but apparently
     * Android didn't add this until SDK version 26, so a check is needed...
     */
#ifdef HAVE_NL_LANGINFO
    if (df) {
        const char *s = nl_langinfo(D_FMT);

        // Figure out the preferred system date format from the first format character.
        if (s) {
            while (*s) {
                switch (*s++) {
                case 'Y':
                case 'y':
                case 'F':
                case 'C':
                    *df = SDL_DATE_FORMAT_YYYYMMDD;
                    goto found_date;
                case 'd':
                case 'e':
                    *df = SDL_DATE_FORMAT_DDMMYYYY;
                    goto found_date;
                case 'b':
                case 'D':
                case 'h':
                case 'm':
                    *df = SDL_DATE_FORMAT_MMDDYYYY;
                    goto found_date;
                default:
                    break;
                }
            }
        }
    }

found_date:

    if (tf) {
        const char *s = nl_langinfo(T_FMT);

        // Figure out the preferred system date format.
        if (s) {
            while (*s) {
                switch (*s++) {
                case 'H':
                case 'k':
                case 'T':
                    *tf = SDL_TIME_FORMAT_24HR;
                    return;
                case 'I':
                case 'l':
                case 'r':
                    *tf = SDL_TIME_FORMAT_12HR;
                    return;
                default:
                    break;
                }
            }
        }
    }
#endif
}

bool SDL_GetCurrentTime(SDL_Time *ticks)
{
    if (!ticks) {
        return SDL_InvalidParamError("ticks");
    }
#ifdef HAVE_CLOCK_GETTIME
    struct timespec tp;

    if (clock_gettime(CLOCK_REALTIME, &tp) == 0) {
        //tp.tv_sec = SDL_min(tp.tv_sec, SDL_NS_TO_SECONDS(SDL_MAX_TIME) - 1);
        *ticks = SDL_SECONDS_TO_NS(tp.tv_sec) + tp.tv_nsec;
        return true;
    }

    SDL_SetError("Failed to retrieve system time (%i)", errno);

#elif defined(SDL_PLATFORM_APPLE)
    clock_serv_t cclock;
    int ret = host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    if (ret == 0) {
        mach_timespec_t mts;

        SDL_zero(mts);
        ret = clock_get_time(cclock, &mts);
        if (ret == 0) {
            // mach_timespec_t tv_sec is 32-bit, so no overflow possible
            *ticks = SDL_SECONDS_TO_NS(mts.tv_sec) + mts.tv_nsec;
        }
        mach_port_deallocate(mach_task_self(), cclock);

        if (!ret) {
            return true;
        }
    }

    SDL_SetError("Failed to retrieve system time (%i)", ret);

#else
    struct timeval tv;
    SDL_zero(tv);
    if (gettimeofday(&tv, NULL) == 0) {
        tv.tv_sec = SDL_min(tv.tv_sec, SDL_NS_TO_SECONDS(SDL_MAX_TIME) - 1);
        *ticks = SDL_SECONDS_TO_NS(tv.tv_sec) + SDL_US_TO_NS(tv.tv_usec);
        return true;
    }

    SDL_SetError("Failed to retrieve system time (%i)", errno);
#endif

    return false;
}

bool SDL_TimeToDateTime(SDL_Time ticks, SDL_DateTime *dt, bool localTime)
{
#if defined (HAVE_GMTIME_R) || defined(HAVE_LOCALTIME_R)
    struct tm tm_storage;
#endif
    struct tm *tm = NULL;

    if (!dt) {
        return SDL_InvalidParamError("dt");
    }

    const time_t tval = (time_t)SDL_NS_TO_SECONDS(ticks);

    if (localTime) {
#ifdef HAVE_LOCALTIME_R
        tm = localtime_r(&tval, &tm_storage);
#else
        tm = localtime(&tval);
#endif
    } else {
#ifdef HAVE_GMTIME_R
        tm = gmtime_r(&tval, &tm_storage);
#else
        tm = gmtime(&tval);
#endif
    }

    if (tm) {
        dt->year = tm->tm_year + 1900;
        dt->month = tm->tm_mon + 1;
        dt->day = tm->tm_mday;
        dt->hour = tm->tm_hour;
        dt->minute = tm->tm_min;
        dt->second = tm->tm_sec;
        dt->nanosecond = ticks % SDL_NS_PER_SECOND;
        dt->day_of_week = tm->tm_wday;

        /* tm_gmtoff wasn't formally standardized until POSIX.1-2024, but practically it has been available on desktop
         * *nix platforms such as Linux/glibc, FreeBSD, OpenBSD, NetBSD, OSX/macOS, and others since the 1990s.
         *
         * The notable exception is Solaris, where the timezone offset must still be retrieved in the strictly POSIX.1-2008
         * compliant way.
         */
#if (_POSIX_VERSION >= 202405L) || (!defined(sun) && !defined(__sun))
        dt->utc_offset = (int)tm->tm_gmtoff;
#else
        if (localTime) {
            tzset();
            dt->utc_offset = (int)timezone;
        } else {
            dt->utc_offset = 0;
        }
#endif

        return true;
    }

    return SDL_SetError("SDL_DateTime conversion failed (%i)", errno);
}

#endif // SDL_TIME_UNIX
