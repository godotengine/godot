#ifndef HEADER_CURL_TOOL_SETUP_H
#define HEADER_CURL_TOOL_SETUP_H
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

#define CURL_NO_OLDIES

/*
 * curl_setup.h may define preprocessor macros such as _FILE_OFFSET_BITS and
 * _LARGE_FILES in order to support files larger than 2 GB. On platforms
 * where this happens it is mandatory that these macros are defined before
 * any system header file is included, otherwise file handling function
 * prototypes will be misdeclared and curl tool may not build properly;
 * therefore we must include curl_setup.h before curl.h when building curl.
 */

#include "curl_setup.h" /* from the lib directory */

/*
 * curl tool certainly uses libcurl's external interface.
 */

#include <curl/curl.h> /* external interface */

/*
 * Platform specific stuff.
 */

#if defined(macintosh) && defined(__MRC__)
#  define main(x,y) curl_main(x,y)
#endif

#ifdef TPF
#  undef select
   /* change which select is used for the curl command line tool */
#  define select(a,b,c,d,e) tpf_select_bsd(a,b,c,d,e)
   /* and turn off the progress meter */
#  define CONF_DEFAULT (0|CONF_NOPROGRESS)
#endif

#ifndef OS
#  define OS "unknown"
#endif

#ifndef UNPRINTABLE_CHAR
   /* define what to use for unprintable characters */
#  define UNPRINTABLE_CHAR '.'
#endif

#ifndef HAVE_STRDUP
#  include "tool_strdup.h"
#endif

#endif /* HEADER_CURL_TOOL_SETUP_H */
