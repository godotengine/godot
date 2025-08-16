#ifndef foortclockfoo
#define foortclockfoo

/***
  This file is part of PulseAudio.

  Copyright 2004-2009 Lennart Poettering

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation; either version 2.1 of the
  License, or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <pulse/cdecl.h>
#include <pulse/sample.h>

/** \file
 *  Monotonic clock utilities. */

PA_C_DECL_BEGIN

/** Return the current monotonic system time in usec, if such a clock
 * is available.  If it is not available this will return the
 * wallclock time instead.  \since 0.9.16 */
pa_usec_t pa_rtclock_now(void);

PA_C_DECL_END

#endif
