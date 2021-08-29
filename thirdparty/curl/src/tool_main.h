#ifndef HEADER_CURL_TOOL_MAIN_H
#define HEADER_CURL_TOOL_MAIN_H
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

#define DEFAULT_MAXREDIRS  50L

#define RETRY_SLEEP_DEFAULT 1000L   /* ms */
#define RETRY_SLEEP_MAX     600000L /* ms == 10 minutes */

#define MAX_PARALLEL 300 /* conservative */
#define PARALLEL_DEFAULT 50

#ifndef STDIN_FILENO
#  define STDIN_FILENO  fileno(stdin)
#endif

#ifndef STDOUT_FILENO
#  define STDOUT_FILENO  fileno(stdout)
#endif

#ifndef STDERR_FILENO
#  define STDERR_FILENO  fileno(stderr)
#endif

#endif /* HEADER_CURL_TOOL_MAIN_H */
