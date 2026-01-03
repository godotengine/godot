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

#ifdef SDL_TIMER_UNIX

#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#include "../SDL_timer_c.h"

#ifdef SDL_PLATFORM_EMSCRIPTEN
#include <emscripten.h>
#endif

/* The clock_gettime provides monotonous time, so we should use it if
   it's available. The clock_gettime function is behind ifdef
   for __USE_POSIX199309
   Tommi Kyntola (tommi.kyntola@ray.fi) 27/09/2005
*/
/* Reworked monotonic clock to not assume the current system has one
   as not all linux kernels provide a monotonic clock (yeah recent ones
   probably do)
   Also added macOS Monotonic clock support
   Based on work in https://github.com/ThomasHabets/monotonic_clock
 */
#if defined(HAVE_NANOSLEEP) || defined(HAVE_CLOCK_GETTIME)
#include <time.h>
#endif
#ifdef SDL_PLATFORM_APPLE
#include <mach/mach_time.h>
#endif

// Use CLOCK_MONOTONIC_RAW, if available, which is not subject to adjustment by NTP
#ifdef HAVE_CLOCK_GETTIME
#ifdef CLOCK_MONOTONIC_RAW
#define SDL_MONOTONIC_CLOCK CLOCK_MONOTONIC_RAW
#else
#define SDL_MONOTONIC_CLOCK CLOCK_MONOTONIC
#endif
#endif

// The first ticks value of the application
#if !defined(HAVE_CLOCK_GETTIME) && defined(SDL_PLATFORM_APPLE)
mach_timebase_info_data_t mach_base_info;
#endif
static bool checked_monotonic_time = false;
static bool has_monotonic_time = false;

static void CheckMonotonicTime(void)
{
#ifdef HAVE_CLOCK_GETTIME
    struct timespec value;
    if (clock_gettime(SDL_MONOTONIC_CLOCK, &value) == 0) {
        has_monotonic_time = true;
    }
#elif defined(SDL_PLATFORM_APPLE)
    if (mach_timebase_info(&mach_base_info) == 0) {
        has_monotonic_time = true;
    }
#endif
    checked_monotonic_time = true;
}

Uint64 SDL_GetPerformanceCounter(void)
{
    Uint64 ticks;

    if (!checked_monotonic_time) {
        CheckMonotonicTime();
    }

    if (has_monotonic_time) {
#ifdef HAVE_CLOCK_GETTIME
        struct timespec now;

        clock_gettime(SDL_MONOTONIC_CLOCK, &now);
        ticks = now.tv_sec;
        ticks *= SDL_NS_PER_SECOND;
        ticks += now.tv_nsec;
#elif defined(SDL_PLATFORM_APPLE)
        ticks = mach_absolute_time();
#else
        SDL_assert(false);
        ticks = 0;
#endif
    } else {
        struct timeval now;

        gettimeofday(&now, NULL);
        ticks = now.tv_sec;
        ticks *= SDL_US_PER_SECOND;
        ticks += now.tv_usec;
    }
    return ticks;
}

Uint64 SDL_GetPerformanceFrequency(void)
{
    if (!checked_monotonic_time) {
        CheckMonotonicTime();
    }

    if (has_monotonic_time) {
#ifdef HAVE_CLOCK_GETTIME
        return SDL_NS_PER_SECOND;
#elif defined(SDL_PLATFORM_APPLE)
        Uint64 freq = mach_base_info.denom;
        freq *= SDL_NS_PER_SECOND;
        freq /= mach_base_info.numer;
        return freq;
#endif
    }

    return SDL_US_PER_SECOND;
}

void SDL_SYS_DelayNS(Uint64 ns)
{
    int was_error;

#ifdef HAVE_NANOSLEEP
    struct timespec tv, remaining;
#else
    struct timeval tv;
    Uint64 then, now, elapsed;
#endif

#ifdef SDL_PLATFORM_EMSCRIPTEN
    if (emscripten_has_asyncify() && SDL_GetHintBoolean(SDL_HINT_EMSCRIPTEN_ASYNCIFY, true)) {
        // pseudo-synchronous pause, used directly or through e.g. SDL_WaitEvent
        emscripten_sleep(ns / SDL_NS_PER_MS);
        return;
    }
#endif

    // Set the timeout interval
#ifdef HAVE_NANOSLEEP
    remaining.tv_sec = (time_t)(ns / SDL_NS_PER_SECOND);
    remaining.tv_nsec = (long)(ns % SDL_NS_PER_SECOND);
#else
    then = SDL_GetTicksNS();
#endif
    do {
        errno = 0;

#ifdef HAVE_NANOSLEEP
        tv.tv_sec = remaining.tv_sec;
        tv.tv_nsec = remaining.tv_nsec;
        was_error = nanosleep(&tv, &remaining);
#else
        // Calculate the time interval left (in case of interrupt)
        now = SDL_GetTicksNS();
        elapsed = (now - then);
        then = now;
        if (elapsed >= ns) {
            break;
        }
        ns -= elapsed;
        tv.tv_sec = (ns / SDL_NS_PER_SECOND);
        tv.tv_usec = SDL_NS_TO_US(ns % SDL_NS_PER_SECOND);

        was_error = select(0, NULL, NULL, NULL, &tv);
#endif // HAVE_NANOSLEEP
    } while (was_error && (errno == EINTR));
}

#endif // SDL_TIMER_UNIX
