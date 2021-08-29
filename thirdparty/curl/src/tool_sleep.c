/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2020, Daniel Stenberg, <daniel@haxx.se>, et al.
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

#ifdef HAVE_SYS_SELECT_H
#  include <sys/select.h>
#elif defined(HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#ifdef HAVE_POLL_H
#  include <poll.h>
#elif defined(HAVE_SYS_POLL_H)
#  include <sys/poll.h>
#endif

#ifdef MSDOS
#  include <dos.h>
#endif

#include "tool_sleep.h"

#include "memdebug.h" /* keep this as LAST include */

void tool_go_sleep(long ms)
{
#if defined(MSDOS)
  delay(ms);
#elif defined(WIN32)
  Sleep(ms);
#elif defined(HAVE_POLL_FINE)
  (void)poll((void *)0, 0, (int)ms);
#else
  struct timeval timeout;
  timeout.tv_sec = ms / 1000L;
  ms = ms % 1000L;
  timeout.tv_usec = (int)ms * 1000;
  select(0, NULL,  NULL, NULL, &timeout);
#endif
}
