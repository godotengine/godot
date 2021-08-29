/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/
#include "tool_setup.h"

#include "tool_util.h"

#include "memdebug.h" /* keep this as LAST include */

#if defined(WIN32) && !defined(MSDOS)

/* set in win32_init() */
extern LARGE_INTEGER tool_freq;
extern bool tool_isVistaOrGreater;

/* In case of bug fix this function has a counterpart in timeval.c */
struct timeval tvnow(void)
{
  struct timeval now;
  if(tool_isVistaOrGreater) { /* QPC timer might have issues pre-Vista */
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    now.tv_sec = (long)(count.QuadPart / tool_freq.QuadPart);
    now.tv_usec = (long)((count.QuadPart % tool_freq.QuadPart) * 1000000 /
                         tool_freq.QuadPart);
  }
  else {
    /* Disable /analyze warning that GetTickCount64 is preferred  */
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:28159)
#endif
    DWORD milliseconds = GetTickCount();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

    now.tv_sec = (long)(milliseconds / 1000);
    now.tv_usec = (long)((milliseconds % 1000) * 1000);
  }
  return now;
}

#elif defined(HAVE_CLOCK_GETTIME_MONOTONIC)

struct timeval tvnow(void)
{
  /*
  ** clock_gettime() is granted to be increased monotonically when the
  ** monotonic clock is queried. Time starting point is unspecified, it
  ** could be the system start-up time, the Epoch, or something else,
  ** in any case the time starting point does not change once that the
  ** system has started up.
  */
  struct timeval now;
  struct timespec tsnow;
  if(0 == clock_gettime(CLOCK_MONOTONIC, &tsnow)) {
    now.tv_sec = tsnow.tv_sec;
    now.tv_usec = (int)(tsnow.tv_nsec / 1000);
  }
  /*
  ** Even when the configure process has truly detected monotonic clock
  ** availability, it might happen that it is not actually available at
  ** run-time. When this occurs simply fallback to other time source.
  */
#ifdef HAVE_GETTIMEOFDAY
  else
    (void)gettimeofday(&now, NULL);
#else
  else {
    now.tv_sec = time(NULL);
    now.tv_usec = 0;
  }
#endif
  return now;
}

#elif defined(HAVE_GETTIMEOFDAY)

struct timeval tvnow(void)
{
  /*
  ** gettimeofday() is not granted to be increased monotonically, due to
  ** clock drifting and external source time synchronization it can jump
  ** forward or backward in time.
  */
  struct timeval now;
  (void)gettimeofday(&now, NULL);
  return now;
}

#else

struct timeval tvnow(void)
{
  /*
  ** time() returns the value of time in seconds since the Epoch.
  */
  struct timeval now;
  now.tv_sec = time(NULL);
  now.tv_usec = 0;
  return now;
}

#endif

/*
 * Make sure that the first argument is the more recent time, as otherwise
 * we'll get a weird negative time-diff back...
 *
 * Returns: the time difference in number of milliseconds.
 */
long tvdiff(struct timeval newer, struct timeval older)
{
  return (long)(newer.tv_sec-older.tv_sec)*1000+
    (long)(newer.tv_usec-older.tv_usec)/1000;
}
