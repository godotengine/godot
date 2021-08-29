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

#include "curl_setup.h"

#include "curl_gethostname.h"

/*
 * Curl_gethostname() is a wrapper around gethostname() which allows
 * overriding the host name that the function would normally return.
 * This capability is used by the test suite to verify exact matching
 * of NTLM authentication, which exercises libcurl's MD4 and DES code
 * as well as by the SMTP module when a hostname is not provided.
 *
 * For libcurl debug enabled builds host name overriding takes place
 * when environment variable CURL_GETHOSTNAME is set, using the value
 * held by the variable to override returned host name.
 *
 * Note: The function always returns the un-qualified hostname rather
 * than being provider dependent.
 *
 * For libcurl shared library release builds the test suite preloads
 * another shared library named libhostname using the LD_PRELOAD
 * mechanism which intercepts, and might override, the gethostname()
 * function call. In this case a given platform must support the
 * LD_PRELOAD mechanism and additionally have environment variable
 * CURL_GETHOSTNAME set in order to override the returned host name.
 *
 * For libcurl static library release builds no overriding takes place.
 */

int Curl_gethostname(char * const name, GETHOSTNAME_TYPE_ARG2 namelen)
{
#ifndef HAVE_GETHOSTNAME

  /* Allow compilation and return failure when unavailable */
  (void) name;
  (void) namelen;
  return -1;

#else
  int err;
  char *dot;

#ifdef DEBUGBUILD

  /* Override host name when environment variable CURL_GETHOSTNAME is set */
  const char *force_hostname = getenv("CURL_GETHOSTNAME");
  if(force_hostname) {
    strncpy(name, force_hostname, namelen);
    err = 0;
  }
  else {
    name[0] = '\0';
    err = gethostname(name, namelen);
  }

#else /* DEBUGBUILD */

  /* The call to system's gethostname() might get intercepted by the
     libhostname library when libcurl is built as a non-debug shared
     library when running the test suite. */
  name[0] = '\0';
  err = gethostname(name, namelen);

#endif

  name[namelen - 1] = '\0';

  if(err)
    return err;

  /* Truncate domain, leave only machine name */
  dot = strchr(name, '.');
  if(dot)
    *dot = '\0';

  return 0;
#endif

}
