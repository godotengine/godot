#ifndef HEADER_CURL_NETRC_H
#define HEADER_CURL_NETRC_H
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
#ifndef CURL_DISABLE_NETRC

/* returns -1 on failure, 0 if the host is found, 1 is the host isn't found */
int Curl_parsenetrc(const char *host,
                    char **loginp,
                    char **passwordp,
                    bool *login_changed,
                    bool *password_changed,
                    char *filename);
  /* Assume: (*passwordp)[0]=0, host[0] != 0.
   * If (*loginp)[0] = 0, search for login and password within a machine
   * section in the netrc.
   * If (*loginp)[0] != 0, search for password within machine and login.
   */
#else
/* disabled */
#define Curl_parsenetrc(a,b,c,d,e,f) 1
#endif

#endif /* HEADER_CURL_NETRC_H */
