#ifndef fooutf8hfoo
#define fooutf8hfoo

/***
  This file is part of PulseAudio.

  Copyright 2006 Lennart Poettering
  Copyright 2006 Pierre Ossman <ossman@cendio.se> for Cendio AB

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
#include <pulse/gccmacro.h>
#include <pulse/version.h>

/** \file
 * UTF-8 validation functions
 */

PA_C_DECL_BEGIN

/** Test if the specified strings qualifies as valid UTF8. Return the string if so, otherwise NULL */
char *pa_utf8_valid(const char *str) PA_GCC_PURE;

/** Test if the specified strings qualifies as valid 7-bit ASCII. Return the string if so, otherwise NULL. \since 0.9.15 */
char *pa_ascii_valid(const char *str) PA_GCC_PURE;

/** Filter all invalid UTF8 characters from the specified string, returning a new fully UTF8 valid string. Don't forget to free the returned string with pa_xfree() */
char *pa_utf8_filter(const char *str);

/** Filter all invalid ASCII characters from the specified string, returning a new fully ASCII valid string. Don't forget to free the returned string with pa_xfree(). \since 0.9.15 */
char *pa_ascii_filter(const char *str);

/** Convert a UTF-8 string to the current locale. Free the string using pa_xfree(). */
char* pa_utf8_to_locale (const char *str);

/** Convert a string in the current locale to UTF-8. Free the string using pa_xfree(). */
char* pa_locale_to_utf8 (const char *str);

PA_C_DECL_END

#endif
