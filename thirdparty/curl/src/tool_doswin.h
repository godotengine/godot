#ifndef HEADER_CURL_TOOL_DOSWIN_H
#define HEADER_CURL_TOOL_DOSWIN_H
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

#if defined(MSDOS) || defined(WIN32)

#define SANITIZE_ALLOW_COLONS    (1<<0)  /* Allow colons */
#define SANITIZE_ALLOW_PATH      (1<<1)  /* Allow path separators and colons */
#define SANITIZE_ALLOW_RESERVED  (1<<2)  /* Allow reserved device names */
#define SANITIZE_ALLOW_TRUNCATE  (1<<3)  /* Allow truncating a long filename */

typedef enum {
  SANITIZE_ERR_OK = 0,           /* 0 - OK */
  SANITIZE_ERR_INVALID_PATH,     /* 1 - the path is invalid */
  SANITIZE_ERR_BAD_ARGUMENT,     /* 2 - bad function parameter */
  SANITIZE_ERR_OUT_OF_MEMORY,    /* 3 - out of memory */
  SANITIZE_ERR_LAST /* never use! */
} SANITIZEcode;

SANITIZEcode sanitize_file_name(char **const sanitized, const char *file_name,
                                int flags);
#ifdef UNITTESTS
SANITIZEcode truncate_dryrun(const char *path, const size_t truncate_pos);
SANITIZEcode msdosify(char **const sanitized, const char *file_name,
                      int flags);
SANITIZEcode rename_if_reserved_dos_device_name(char **const sanitized,
                                                const char *file_name,
                                                int flags);
#endif /* UNITTESTS */

#if defined(MSDOS) && (defined(__DJGPP__) || defined(__GO32__))

char **__crt0_glob_function(char *arg);

#endif /* MSDOS && (__DJGPP__ || __GO32__) */

#ifdef WIN32

CURLcode FindWin32CACert(struct OperationConfig *config,
                         curl_sslbackend backend,
                         const TCHAR *bundle_file);
struct curl_slist *GetLoadedModulePaths(void);
CURLcode win32_init(void);

#endif /* WIN32 */

#endif /* MSDOS || WIN32 */

#endif /* HEADER_CURL_TOOL_DOSWIN_H */
