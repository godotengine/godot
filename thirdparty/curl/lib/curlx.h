#ifndef HEADER_CURL_CURLX_H
#define HEADER_CURL_CURLX_H
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

/*
 * Defines protos and includes all header files that provide the curlx_*
 * functions. The curlx_* functions are not part of the libcurl API, but are
 * stand-alone functions whose sources can be built and linked by apps if need
 * be.
 */

#include <curl/mprintf.h>
/* this is still a public header file that provides the curl_mprintf()
   functions while they still are offered publicly. They will be made library-
   private one day */

#include "strcase.h"
/* "strcase.h" provides the strcasecompare protos */

#include "strtoofft.h"
/* "strtoofft.h" provides this function: curlx_strtoofft(), returns a
   curl_off_t number from a given string.
*/

#include "nonblock.h"
/* "nonblock.h" provides curlx_nonblock() */

#include "warnless.h"
/* "warnless.h" provides functions:

  curlx_ultous()
  curlx_ultouc()
  curlx_uztosi()
*/

#include "curl_multibyte.h"
/* "curl_multibyte.h" provides these functions and macros:

  curlx_convert_UTF8_to_wchar()
  curlx_convert_wchar_to_UTF8()
  curlx_convert_UTF8_to_tchar()
  curlx_convert_tchar_to_UTF8()
  curlx_unicodefree()
*/

#include "version_win32.h"
/* "version_win32.h" provides curlx_verify_windows_version() */

/* Now setup curlx_ * names for the functions that are to become curlx_ and
   be removed from a future libcurl official API:
   curlx_getenv
   curlx_mprintf (and its variations)
   curlx_strcasecompare
   curlx_strncasecompare

*/

#define curlx_getenv curl_getenv
#define curlx_mvsnprintf curl_mvsnprintf
#define curlx_msnprintf curl_msnprintf
#define curlx_maprintf curl_maprintf
#define curlx_mvaprintf curl_mvaprintf
#define curlx_msprintf curl_msprintf
#define curlx_mprintf curl_mprintf
#define curlx_mfprintf curl_mfprintf
#define curlx_mvsprintf curl_mvsprintf
#define curlx_mvprintf curl_mvprintf
#define curlx_mvfprintf curl_mvfprintf

#ifdef ENABLE_CURLX_PRINTF
/* If this define is set, we define all "standard" printf() functions to use
   the curlx_* version instead. It makes the source code transparent and
   easier to understand/patch. Undefine them first. */
# undef printf
# undef fprintf
# undef sprintf
# undef msnprintf
# undef vprintf
# undef vfprintf
# undef vsprintf
# undef mvsnprintf
# undef aprintf
# undef vaprintf

# define printf curlx_mprintf
# define fprintf curlx_mfprintf
# define sprintf curlx_msprintf
# define msnprintf curlx_msnprintf
# define vprintf curlx_mvprintf
# define vfprintf curlx_mvfprintf
# define mvsnprintf curlx_mvsnprintf
# define aprintf curlx_maprintf
# define vaprintf curlx_mvaprintf
#endif /* ENABLE_CURLX_PRINTF */

#endif /* HEADER_CURL_CURLX_H */
